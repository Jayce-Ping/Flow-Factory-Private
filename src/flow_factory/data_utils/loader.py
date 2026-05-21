# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/data_utils/loader.py
import json
import os
import shutil
from typing import Dict, Literal, Optional, Tuple, Union

from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..data_utils.dataset import PreprocessCallable
from ..hparams import Arguments
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger
from .dataset import GeneralDataset
from .sampler_loader import get_data_sampler

logger = setup_logger(__name__, rank_zero_only=False)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _get_local_process_info(accelerator: Accelerator):
    """
    Get local_rank and local_world_size within the current node.
    Prefers environment variables set by torchrun / accelerate launch,
    falls back to accelerator attributes.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    # If LOCAL_WORLD_SIZE is not set but we have multiple processes, try to infer
    if local_world_size == 1 and accelerator.num_processes > 1:
        num_machines = int(os.environ.get("NUM_MACHINES", os.environ.get("NNODES", 1)))
        local_world_size = accelerator.num_processes // num_machines
    return local_rank, local_world_size


def _create_or_load_dataset(
    split: str,
    accelerator: Accelerator,
    base_kwargs: dict,
    enable_distributed: bool,
    preprocess_parallelism: Literal["global", "local"] = "global",
) -> GeneralDataset:
    """Create or load preprocessed dataset with optional distributed sharding.

    Each rank writes its preprocessed Arrow shard exactly once via
    ``Dataset.map(cache_file_name=...)`` straight into the final cache directory.
    The consolidator (``local_main`` for ``"local"``, global rank 0 for ``"global"``,
    the lone process for single-process) then writes ``state.json`` and
    ``dataset_info.json`` referencing those per-rank Arrow files and atomically
    renames the build directory ``{merged_cache_path}.tmp`` to ``merged_cache_path``.
    No shard data is re-copied.

    Args:
        split: Dataset split (``"train"``, ``"test"``, ...).
        accelerator: Accelerator used for cross-rank synchronization.
        base_kwargs: Base arguments forwarded to ``GeneralDataset``.
        enable_distributed: ``True`` when more than one process needs to share work.
        preprocess_parallelism: ``"global"`` for cross-node parallelism (shared FS
            required); ``"local"`` for per-node parallelism (no shared FS required).

    Returns:
        Fully preprocessed ``GeneralDataset`` ready for training.
    """
    kwargs = base_kwargs.copy()

    if not kwargs.get("enable_preprocess", True):
        logger.info(
            f"Loading {split} dataset without preprocessing (enable_preprocess=False); "
            f"skipping consolidate pipeline"
        )
        return GeneralDataset(split=split, **kwargs)

    if enable_distributed:
        if preprocess_parallelism == "local":
            local_rank, local_world_size = _get_local_process_info(accelerator)
            kwargs["num_shards"] = local_world_size
            kwargs["shard_index"] = local_rank
        else:
            kwargs["num_shards"] = accelerator.num_processes
            kwargs["shard_index"] = accelerator.process_index
    else:
        kwargs["num_shards"] = 1
        kwargs["shard_index"] = 0

    merged_cache_path = GeneralDataset.compute_cache_path(
        dataset_dir=kwargs["dataset_dir"],
        split=split,
        cache_dir=kwargs["cache_dir"],
        max_dataset_size=kwargs.get("max_dataset_size"),
        preprocess_func=kwargs.get("preprocess_func"),
        preprocess_kwargs=kwargs.get("preprocess_kwargs"),
        extra_hash_strs=kwargs.get("extra_hash_strs", []),
    )

    if os.path.exists(merged_cache_path) and not base_kwargs.get("force_reprocess", False):
        if accelerator.is_local_main_process:
            logger.info(f"Loading {split} dataset from merged cache: {merged_cache_path}")
        return GeneralDataset.load_merged(merged_cache_path)

    shard_idx = kwargs["shard_index"]
    num_shards = kwargs["num_shards"]

    build_dir = merged_cache_path + ".tmp"
    sentinel = os.path.join(build_dir, "_build_meta.json")

    def _meta_matches() -> bool:
        if not os.path.isfile(sentinel):
            return False
        try:
            with open(sentinel) as f:
                return json.load(f).get("num_shards") == num_shards
        except (json.JSONDecodeError, OSError):
            # Sentinel was corrupted (e.g., previous run crashed mid-write).
            # Treat as stale so the orchestrator wipes and recreates the build dir,
            # matching the existing "missing -> return False -> wipe" semantics.
            return False

    # Pick the single owner of build-dir prep + final consolidation per case:
    #   - non-distributed:  the lone process.
    #   - "global" mode:    rank-0 globally. Required when the cache_dir lives
    #                       on a shared FS visible to every node — a single
    #                       orchestrator eliminates the cross-node race on
    #                       shutil.rmtree and sentinel writes.
    #   - "local"  mode:    per-node local main. ASSUMES cache_dir is on
    #                       node-local storage (each node has its own copy of
    #                       the build dir). Pointing "local" mode at a shared
    #                       FS WILL race across node-local mains and corrupt
    #                       the build dir; that configuration is unsupported.
    if not enable_distributed:
        is_orchestrator = True
    elif preprocess_parallelism == "local":
        is_orchestrator = accelerator.is_local_main_process
    else:
        is_orchestrator = accelerator.is_main_process

    # 1. Orchestrator prepares (or wipes-then-prepares) the build dir. The wipe only
    #    fires when num_shards changed since the last attempt; otherwise per-rank
    #    Arrow files written before a previous crash are reused via HF's
    #    load_from_cache_file path.
    if is_orchestrator:
        if os.path.exists(build_dir) and not _meta_matches():
            logger.warning(f"Wiping stale build dir {build_dir} (num_shards changed)")
            shutil.rmtree(build_dir)
        os.makedirs(build_dir, exist_ok=True)
        if not os.path.isfile(sentinel):
            with open(sentinel, "w") as f:
                json.dump({"num_shards": num_shards}, f)
    if enable_distributed:
        accelerator.wait_for_everyone()

    # 2. Per-rank Arrow file. Basename is byte-equivalent to today's HF auto-cache
    #    name; the rank_*_of_N subdir prevents cross-config collisions if a stale
    #    .tmp directory survives a launch-config change between runs. Layout is
    #    owned by GeneralDataset so the writer and the consolidator cannot drift.
    part_arrow_path = GeneralDataset.build_part_arrow_path(merged_cache_path, shard_idx, num_shards)
    kwargs["target_arrow_path"] = part_arrow_path

    logger.info(
        f"Preprocessing {split} dataset shard {shard_idx:04d}/{num_shards - 1:04d} "
        f"-> {part_arrow_path}"
    )
    _ = GeneralDataset(split=split, **kwargs)

    if enable_distributed:
        accelerator.wait_for_everyone()

    # 3. Consolidate: write top-level state.json + dataset_info.json (no row data
    #    copied) and atomically rename .tmp -> merged_cache_path. A single call;
    #    consolidate_parts iterates the per-rank layout itself via
    #    GeneralDataset.build_part_arrow_path.
    if is_orchestrator:
        GeneralDataset.consolidate_parts(merged_cache_path, num_shards, split=split)
        mode_label = preprocess_parallelism if enable_distributed else "single"
        logger.info(
            f"[{mode_label}] Consolidated {num_shards} part(s) for {split} split "
            f"-> {merged_cache_path}"
        )

    if enable_distributed:
        accelerator.wait_for_everyone()
    return GeneralDataset.load_merged(merged_cache_path)


def _load_and_concat_datasets(
    dataset_dirs: list,
    accelerator: "Accelerator",
    base_kwargs: dict,
    train_preprocess_kwargs: dict,
    enable_distributed: bool,
    preprocess_parallelism: str,
    max_dataset_size: Optional[int] = None,
) -> "GeneralDataset":
    """Load multiple datasets, inject ``__source__`` column, and concatenate.

    Each dataset is loaded and preprocessed independently. A ``__source__``
    column is added to each (= basename of its directory path) so downstream
    components (reward models, teacher routing) can identify sample origin.
    The resulting datasets are concatenated into a single ``GeneralDataset``.

    Args:
        dataset_dirs: List of dataset directory paths.
        accelerator: Accelerator for distributed sync.
        base_kwargs: Base kwargs for ``_create_or_load_dataset``.
        train_preprocess_kwargs: Preprocess kwargs for training split.
        enable_distributed: Whether distributed preprocessing is active.
        preprocess_parallelism: ``"global"`` or ``"local"``.
        max_dataset_size: If set, truncate the concatenated dataset.

    Returns:
        A single ``GeneralDataset`` containing all samples with ``__source__``.
    """
    from datasets import concatenate_datasets as hf_concat

    all_hf_datasets = []
    for dataset_dir in dataset_dirs:
        source_name = os.path.basename(os.path.expanduser(dataset_dir).rstrip("/"))
        logger.info(f"Multi-dataset: loading '{source_name}' from {dataset_dir}")

        per_dir_kwargs = {**base_kwargs}
        per_dir_kwargs["dataset_dir"] = dataset_dir
        # Add source to extra_hash_strs so caches are distinct per source
        extra = list(base_kwargs.get("extra_hash_strs", []))
        extra.append(f"source:{source_name}")
        per_dir_kwargs["extra_hash_strs"] = extra

        ds = _create_or_load_dataset(
            split="train",
            accelerator=accelerator,
            base_kwargs={**per_dir_kwargs, "preprocess_kwargs": train_preprocess_kwargs},
            enable_distributed=enable_distributed,
            preprocess_parallelism=preprocess_parallelism,
        )

        # Inject __source__ column
        hf_ds = ds.processed_dataset
        hf_ds = hf_ds.add_column("__source__", [source_name] * len(hf_ds))
        all_hf_datasets.append(hf_ds)

    # Align schemas: keep only columns common to ALL datasets to avoid
    # concatenation failures when datasets have incompatible feature types
    # (e.g., different `metadata` schemas across geneval vs ocr).
    common_columns = set(all_hf_datasets[0].column_names)
    for ds in all_hf_datasets[1:]:
        common_columns &= set(ds.column_names)

    # Further filter: remove columns whose feature types are incompatible
    # across datasets (structural mismatch even if column name is shared).
    safe_columns = sorted(common_columns)
    aligned_datasets = []
    for ds in all_hf_datasets:
        aligned_datasets.append(ds.select_columns(safe_columns))

    # Concatenate all HuggingFace datasets
    try:
        combined = hf_concat(aligned_datasets)
    except ValueError as e:
        # If schema alignment still fails, try removing the problematic
        # 'metadata' column which is the most common offender.
        if "metadata" in str(e) and "metadata" in safe_columns:
            safe_columns = [c for c in safe_columns if c != "metadata"]
            logger.warning(
                f"Multi-dataset: 'metadata' column has incompatible schemas across "
                f"datasets, removing it. Remaining columns: {safe_columns}"
            )
            aligned_datasets = [ds.select_columns(safe_columns) for ds in all_hf_datasets]
            combined = hf_concat(aligned_datasets)
        else:
            raise

    # Apply max_dataset_size to the concatenated result
    if max_dataset_size is not None and len(combined) > max_dataset_size:
        combined = combined.select(range(max_dataset_size))
        logger.info(f"Multi-dataset: truncated concatenated dataset to {max_dataset_size} samples.")

    logger.info(
        f"Multi-dataset: concatenated {len(dataset_dirs)} datasets, "
        f"total {len(combined)} samples."
    )

    # Wrap back into a GeneralDataset shell
    result = GeneralDataset.__new__(GeneralDataset)
    result.processed_dataset = combined
    try:
        result.processed_dataset.set_format(
            type="torch", columns=result.processed_dataset.column_names
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Multi-dataset: could not set torch format: {e}")

    return result


def _load_per_source_dataloaders(
    dataset_dirs: list,
    accelerator: "Accelerator",
    base_kwargs: dict,
    train_preprocess_kwargs: dict,
    enable_distributed: bool,
    preprocess_parallelism: str,
    max_dataset_size: Optional[int],
    config: "Arguments",
    data_args,
) -> Dict[str, DataLoader]:
    """Load each dataset independently and create per-source DataLoaders.

    Each dataset is loaded, preprocessed, and wrapped in its own DataLoader.
    The source name is the basename of the directory path. Metadata columns
    are preserved (no schema alignment needed since datasets are never merged).

    Returns:
        Dict mapping source name to its DataLoader.
    """
    from .sampler_loader import get_data_sampler

    result: Dict[str, DataLoader] = {}
    for dataset_dir in dataset_dirs:
        source_name = os.path.basename(os.path.expanduser(dataset_dir).rstrip("/"))
        logger.info(f"Multi-dataset: loading source '{source_name}' from {dataset_dir}")

        per_dir_kwargs = {**base_kwargs}
        per_dir_kwargs["dataset_dir"] = dataset_dir
        extra = list(base_kwargs.get("extra_hash_strs", []))
        extra.append(f"source:{source_name}")
        per_dir_kwargs["extra_hash_strs"] = extra

        dataset = _create_or_load_dataset(
            split="train",
            accelerator=accelerator,
            base_kwargs={**per_dir_kwargs, "preprocess_kwargs": train_preprocess_kwargs},
            enable_distributed=enable_distributed,
            preprocess_parallelism=preprocess_parallelism,
        )

        if max_dataset_size is not None and len(dataset) > max_dataset_size:
            dataset.processed_dataset = dataset.processed_dataset.select(
                range(max_dataset_size)
            )

        sampler = get_data_sampler(
            dataset=dataset,
            config=config,
            accelerator=accelerator,
        )

        dl = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=data_args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=GeneralDataset.collate_fn,
            persistent_workers=data_args.dataloader_num_workers > 0,
        )
        result[source_name] = dl
        logger.info(
            f"Multi-dataset: source '{source_name}' → {len(dataset)} samples"
        )

    return result


def get_dataloader(
    config: Arguments,
    accelerator: Accelerator,
    preprocess_func: Optional[PreprocessCallable] = None,
    **kwargs,
) -> Tuple[Optional[DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Factory to create DDP/FSDP compatible DataLoader with distributed preprocessing.

    Returns:
        Tuple of (train_dataloader, train_dataloaders_by_source, test_dataloaders).
        - train_dataloader: Single DataLoader (from dataset_dir) or None.
        - train_dataloaders_by_source: Dict mapping source name to DataLoader
          when dataset_dirs is set; empty dict otherwise.
        - test_dataloaders: Dict mapping test set name to DataLoader.
    """
    data_args = config.data_args
    training_args = config.training_args
    eval_args = config.eval_args

    # Determine if distributed preprocessing is needed
    enable_distributed = accelerator.num_processes > 1 and data_args.enable_preprocess
    preprocess_parallelism = getattr(data_args, "preprocess_parallelism", "local")

    # Common dataset kwargs
    base_kwargs = {
        "preprocess_func": preprocess_func,
        "preprocess_kwargs": (
            filter_kwargs(preprocess_func, **data_args) if preprocess_func else None
        ),  # Preprocess kwargs
        "extra_hash_strs": [
            config.model_args.model_type,
            config.model_args.model_name_or_path,
        ],  # Use model info to differentiate caches
    }
    base_kwargs.update(filter_kwargs(GeneralDataset.__init__, **data_args))
    base_kwargs["force_reprocess"] = data_args.force_reprocess

    dataloader: Optional[DataLoader] = None
    train_dataloaders_by_source: Dict[str, DataLoader] = {}
    if training_args.skips_train_dataloader:
        logger.info("ensemble-eval: skipping train split preprocessing and train DataLoader")
    else:
        # === CREATE/LOAD TRAIN DATASET ===
        train_preprocess_kwargs = base_kwargs.get("preprocess_kwargs", {}).copy()
        train_preprocess_kwargs.update(
            {
                "is_train": True,
                **training_args,
            }
        )
        # Use algorithm-aware guidance scale for preprocessing — ensures negative
        # prompts are encoded when any optimizer-time CFG scale needs them
        # (e.g., DGPO kl_cfg > 1.0 with training guidance_scale = 1.0).
        train_preprocess_kwargs["guidance_scale"] = training_args.get_preprocess_guidance_scale()
        train_preprocess_kwargs = filter_kwargs(preprocess_func, **train_preprocess_kwargs)

        if data_args.dataset_dirs:
            # Multi-dataset mode: create per-source DataLoaders (no concat)
            train_dataloaders_by_source = _load_per_source_dataloaders(
                dataset_dirs=data_args.dataset_dirs,
                accelerator=accelerator,
                base_kwargs=base_kwargs,
                train_preprocess_kwargs=train_preprocess_kwargs,
                enable_distributed=enable_distributed,
                preprocess_parallelism=preprocess_parallelism,
                max_dataset_size=data_args.max_dataset_size,
                config=config,
                data_args=data_args,
            )
        else:
            dataset = _create_or_load_dataset(
                split="train",
                accelerator=accelerator,
                base_kwargs={**base_kwargs, "preprocess_kwargs": train_preprocess_kwargs},
                enable_distributed=enable_distributed,
                preprocess_parallelism=preprocess_parallelism,
            )

            # === CREATE TRAIN DATALOADER ===
            sampler = get_data_sampler(
                dataset=dataset,
                config=config,
                accelerator=accelerator,
            )

            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=data_args.dataloader_num_workers,
                pin_memory=True,
                collate_fn=GeneralDataset.collate_fn,
                persistent_workers=data_args.dataloader_num_workers > 0,
            )

    # === CREATE/LOAD TEST DATASET(S) ===
    test_dataloaders: Dict[str, DataLoader] = {}

    if eval_args.test_sets is None:
        if GeneralDataset.check_exists(data_args.dataset, "test"):
            test_preprocess_kwargs = base_kwargs.get("preprocess_kwargs", {}).copy()
            test_preprocess_kwargs.update(
                {
                    "is_train": False,
                    **eval_args,
                }
            )
            test_preprocess_kwargs = filter_kwargs(preprocess_func, **test_preprocess_kwargs)
            test_dataset = _create_or_load_dataset(
                split="test",
                accelerator=accelerator,
                base_kwargs={**base_kwargs, "preprocess_kwargs": test_preprocess_kwargs},
                enable_distributed=enable_distributed,
                preprocess_parallelism=preprocess_parallelism,
            )

            test_dataloaders["test"] = DataLoader(
                test_dataset,
                batch_size=eval_args.per_device_batch_size,
                shuffle=False,
                num_workers=data_args.dataloader_num_workers,
                collate_fn=GeneralDataset.collate_fn,
                persistent_workers=data_args.dataloader_num_workers > 0,
            )
    elif len(eval_args.test_sets) > 0:
        for ts in eval_args.test_sets:
            resolved_dir = ts.dataset_dir if ts.dataset_dir is not None else data_args.dataset
            if not GeneralDataset.check_exists(resolved_dir, ts.split):
                raise FileNotFoundError(
                    f"eval.test_sets entry {ts.name!r}: no {ts.split}.jsonl or {ts.split}.txt "
                    f"under {resolved_dir}"
                )
            merged_eval = eval_args.merged_eval_args_for_test_set(ts)
            test_preprocess_kwargs = base_kwargs.get("preprocess_kwargs", {}).copy()
            test_preprocess_kwargs.update(
                {
                    "is_train": False,
                    **merged_eval,
                }
            )
            test_preprocess_kwargs = filter_kwargs(preprocess_func, **test_preprocess_kwargs)

            test_base = {**base_kwargs}
            test_base["dataset_dir"] = resolved_dir
            extra = list(base_kwargs.get("extra_hash_strs", []))
            extra.append(f"testset:{ts.name}")
            test_base["extra_hash_strs"] = extra

            test_dataset = _create_or_load_dataset(
                split=ts.split,
                accelerator=accelerator,
                base_kwargs={**test_base, "preprocess_kwargs": test_preprocess_kwargs},
                enable_distributed=enable_distributed,
                preprocess_parallelism=preprocess_parallelism,
            )

            test_dataloaders[ts.name] = DataLoader(
                test_dataset,
                batch_size=merged_eval.per_device_batch_size,
                shuffle=False,
                num_workers=data_args.dataloader_num_workers,
                collate_fn=GeneralDataset.collate_fn,
                persistent_workers=data_args.dataloader_num_workers > 0,
            )

    return dataloader, train_dataloaders_by_source, test_dataloaders
