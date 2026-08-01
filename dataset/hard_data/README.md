# Flow-Factory Hard Data

Reproducible construction and reconstruction tools for three OPD curricula:

- `t2i`: long, compositional and reasoning-heavy text-to-image prompts;
- `edit`: single/multi-turn image editing with required-change and preservation checks;
- `multiref`: multi-reference generation with an explicit role for every input image.

The data product is deliberately separate from model-specific preprocessing caches. A release
contains normalized JSONL, manifests, annotations and redistributable assets. Flow-Factory rebuilds
Arrow caches locally because their contents depend on model path, text encoder, CFG and adapter
implementation.

## Repository layout

```text
dataset/hard_data/
  README.md
  download_and_build.sh
  reconstruct.py
  build_t2i.py
  build_edit.py
  build_multiref.py
  serve_qwen3vl_235b.sh
  lib/
    schemas.py
    qwen_client.py
  manifests/
    benchmarks.yaml
    sources.yaml
  fixtures/
    t2i.jsonl
    edit.jsonl
    multiref.jsonl
    images/
```

Generated datasets use the native Flow-Factory contract:

```text
dataset/<name>/
  train.jsonl
  test.jsonl
  images/
```

- T2I record: `{"prompt": "...", ...metadata}`
- single-image edit: `{"prompt": "...", "image": "source.png", ...metadata}`
- multi-reference: `{"prompt": "...", "images": ["a.png", "b.png"], ...metadata}`

All extra JSONL fields are preserved by `GeneralDataset` as sample metadata.

## Qwen constructor service

The single constructor is `Qwen/Qwen3-VL-235B-A22B-Instruct`. It handles pure text rewriting and
image-grounded edit/reference instructions. It is served over an OpenAI-compatible API on two
8-GPU H20 nodes using the CUDA-12.9 vLLM environment documented at:

```text
/apdcephfs_zwfy8/share_305110755/hunyuan/bowenping/envs/vllm-judge/README.md
```

Install the packaged environment on both service nodes if needed:

```bash
bash /apdcephfs_zwfy8/share_305110755/hunyuan/bowenping/envs/vllm-judge/install_vllm_judge.sh
```

Start/status/stop:

```bash
bash dataset/hard_data/serve_qwen3vl_235b.sh start
bash dataset/hard_data/serve_qwen3vl_235b.sh status
bash dataset/hard_data/serve_qwen3vl_235b.sh stop
```

Default endpoint: `http://28.7.185.156:8000/v1`, served model name
`qwen3-vl-235b-a22b-instruct`. The server accepts up to ten images per request. Data builders cache
every request by a deterministic hash, so interrupted runs resume without spending inference again.

## Construction workflow

1. Freeze official benchmark revisions from `manifests/benchmarks.yaml`; they are evaluation-only.
2. Download independent training sources from `manifests/sources.yaml`.
3. Run a lane builder against the Qwen endpoint.
4. Validate schema/feasibility and remove benchmark contamination.
5. Generate matched-seed 9B/4B outputs and retain a teacher-winning stratum plus neutral coverage.
6. Publish allowed artifacts to a private Hugging Face dataset revision.
7. Reconstruct from that immutable revision in a clean directory.

Builder CLIs consume JSONL and emit canonical JSONL:

```bash
python dataset/hard_data/build_t2i.py \
  --input raw_captions.jsonl --output candidates/t2i.jsonl

python dataset/hard_data/build_edit.py \
  --input source_edits.jsonl --output candidates/edit.jsonl

python dataset/hard_data/build_multiref.py \
  --input source_references.jsonl --output candidates/multiref.jsonl
```

Use `--limit 32` for the first smoke test. Inputs and generated records are append-safe and
identified by deterministic IDs.

## Fast local/shared release

Large edit and multi-reference images may stay on a shared filesystem such as fsgm3. Keep the
canonical JSONL, QC reports, checksums and asset-root mapping under
`dataset/hard_data/releases/<version>/`; do not commit model-specific Arrow caches. This makes the
metadata self-contained in the repository while avoiding a second copy of large source images.

Package validated lanes into deterministic train/test splits:

```bash
PYTHONPATH=dataset/hard_data python dataset/hard_data/package_local_release.py \
  --version pilot-v1 \
  --output dataset/hard_data/releases/pilot_v1 \
  --contamination-index /shared/benchmarks/contamination.sqlite \
  --benchmark-lock /shared/benchmarks/benchmark_lock.json \
  --t2i-input /shared/candidates/t2i_valid.jsonl \
  --t2i-report /shared/reports/t2i.json \
  --edit-input /shared/candidates/edit_valid.jsonl \
  --edit-report /shared/reports/edit.json \
  --edit-asset-root /shared/sources/edit/images \
  --multiref-input /shared/candidates/multiref_valid.jsonl \
  --multiref-report /shared/reports/multiref.json \
  --multiref-asset-root /shared/sources/multiref/images
```

Materialize a Flow-Factory-native dataset. `train.jsonl` and `test.jsonl` are copied; `images/`
is a verified symlink to the pinned shared asset root:

```bash
python dataset/hard_data/reconstruct.py \
  --lane edit \
  --local-release dataset/hard_data/releases/pilot_v1 \
  --output dataset/hard_edit_pilot_v1

python dataset/hard_data/reconstruct.py \
  --lane edit \
  --local-release dataset/hard_data/releases/pilot_v1 \
  --output dataset/hard_edit_pilot_v1 \
  --verify-only
```

Use `--metadata-only` when images are not needed. Use `--asset-root` to override the manifest's
shared root in a different cluster mount. Local releases are checksum-verified before any files or
symlinks are materialized.

## Reconstruct a released dataset

Private-first repository defaults:

```text
Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-T2I-v1
Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-Edit-v1
Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-MultiRef-v1
```

Always pin an immutable revision:

```bash
bash dataset/hard_data/download_and_build.sh \
  --lane t2i \
  --revision <40-character-hf-commit> \
  --output dataset/hard_t2i_v1
```

Useful modes:

```bash
# Download/verify metadata without image shards.
python dataset/hard_data/reconstruct.py --lane multiref --revision <sha> --metadata-only

# Verify a previously reconstructed directory.
python dataset/hard_data/reconstruct.py --lane edit --revision <sha> \
  --output dataset/hard_edit_v1 --verify-only
```

The reconstruction process is idempotent, verifies `SHA256SUMS`, rejects unsafe tar members and
never stores absolute local asset paths in JSONL.

## Evaluation and leakage policy

Official benchmark rows, checklist questions and reference images are immutable holdouts and must
never enter a training repository. Before a candidate enters training:

- exact/n-gram and text-embedding checks run against benchmark prompts/questions;
- SHA256 and perceptual image hashes run against benchmark images;
- 9B and 4B are generated with identical sampler, guidance and seeds;
- 9B must exceed 4B by a calibrated checklist margin and clear a visual-quality floor;
- a neutral/random valid stratum remains in the curriculum to avoid selection collapse.

Report faithfulness, preservation/reference fidelity and visual quality separately. Multi-reference
evaluation must report per-reference fidelity, minimum/p10 fidelity, reference coverage and
reference-count scaling curves; an average alone can hide ignored references.

## Licensing

`manifests/*.yaml` are the source of truth. Generated prompts, checklists, hashes and annotations
may be published when their source license permits. Restricted source images are not mirrored:
private HF repos contain source IDs/indexes and `reconstruct.py` downloads from the official
upstream source. Public visibility is a separate explicit approval after a manual license audit.
