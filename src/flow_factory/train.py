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

# src/flow_factory/train.py
import os
import argparse
import logging
from .hparams import Arguments
from .trainers import load_trainer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("flow_factory.train")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Flow-Factory Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    return parser.parse_known_args()


def _enable_offline_local_files_only():
    """When HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE is set, force diffusers/transformers
    ``from_pretrained`` to default ``local_files_only=True``.

    Why: diffusers' sharded-checkpoint loader (`_get_checkpoint_shard_files`) issues
    a network ``model_info`` call for repo-id paths whenever ``local_files_only`` is
    falsy, and HF offline mode turns that into a hard ``OfflineModeIsEnabled`` error
    instead of reading the already-cached ``*.index.json``. On worker nodes without
    huggingface.co reachability this aborts the run even though every shard is cached.
    Coercing ``local_files_only=True`` at the offline boundary takes the local fast
    path. No-op when not in offline mode.
    """
    if os.environ.get("HF_HUB_OFFLINE", "0") not in ("1", "true", "True") and \
       os.environ.get("TRANSFORMERS_OFFLINE", "0") not in ("1", "true", "True"):
        return
    import functools

    def _patch(cls):
        if cls is None or not hasattr(cls, "from_pretrained"):
            return
        orig = cls.from_pretrained.__func__ if hasattr(cls.from_pretrained, "__func__") else cls.from_pretrained

        @classmethod
        @functools.wraps(orig)
        def _wrapped(c, *a, **kw):
            kw.setdefault("local_files_only", True)
            return orig(c, *a, **kw)

        try:
            cls.from_pretrained = _wrapped
        except (AttributeError, TypeError):
            pass

    try:
        from diffusers import DiffusionPipeline, ModelMixin
        from diffusers.models.modeling_utils import ModelMixin as _MM
        for c in {DiffusionPipeline, ModelMixin, _MM}:
            _patch(c)
    except Exception:
        pass
    try:
        from transformers import PreTrainedModel
        _patch(PreTrainedModel)
    except Exception:
        pass


def main():
    args, unknown = parse_args()

    # Honor HF offline mode for cached multi-node loads (see helper docstring).
    _enable_offline_local_files_only()

    # Load configuration
    config = Arguments.load_from_yaml(args.config)
    
    # Log distributed setup info (only from rank 0)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if local_rank == 0:
        logger.info("=" * 100)
        logger.info("Flow-Factory Training Initialized")
        logger.info(f"World Size: {world_size}")
        logger.info("=" * 100)
        logger.info(f"Config: {args.config}")
        logger.info(f"\n{config}")
        logger.info("=" * 100)
    
    # Launch trainer
    trainer = None
    try:
        trainer = load_trainer(config)
        trainer.start()
        if local_rank == 0:
            logger.info("Training completed successfully")
    except KeyboardInterrupt:
        if local_rank == 0:
            logger.info("Training interrupted. Cleaning up...")
        try:
            if trainer is not None:
                trainer.cleanup()
        finally:
            os._exit(0)


if __name__ == "__main__":
    main()
