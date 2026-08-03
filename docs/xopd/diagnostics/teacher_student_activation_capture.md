# 4B rollout with 9B/32B teacher activation capture

This diagnostic runs the 4B base student once, then queries the 9B and 32B
teachers on the exact student-visited `(x_t, t, prompt)` states. It does not
train or update any model.

## Corpus

`build_activation_probe_manifest.py` deterministically selects 128 prompts:

- 32 GenEval
- 32 OCR
- 32 PickScore
- 32 Hard-T2I

Four prompts per source (16 total) receive full activation capture. The 16
prompts are spread over 16 distinct ranks under 32-way data parallelism.

```bash
python scripts/xopd_analysis/build_activation_probe_manifest.py \
  --output-dir \
  /apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/diagnostics/teacher_gap_v1
```

## Capture tiers

Every prompt, model, and denoising step stores:

- student state `x_t` (and the student `x_{t+1}` used by every teacher query);
- model output, transition mean, timestep, `dt`, and transition standard
  deviation;
- block scalar statistics, per-token RMS, per-channel mean/std;
- deterministic 64-dimensional CountSketch projections.

For the 16 full-capture prompts, every double/single block output is stored
losslessly as raw BF16 bits. The first input to the double stream and the first
input to the single stream are also stored. Other block inputs are exactly the
previous block outputs and need not be duplicated.

At denoising steps `0, 9, 18, 27`, the first/middle/last blocks additionally
store full attention projections and feed-forward inputs/outputs. Single-stream
FLUX.2 blocks fuse q/k/v and MLP projections; their
`to_qkv_mlp_proj` tensor is stored as the architecture's native representation.

## Storage layout

```text
teacher_gap_v1/capture/
  run_manifest.json
  run_manifest.started.json
  ranks/rank_000_done.json
  summary/
    student_4b/rank_000.h5
    teacher_9b/rank_000.h5
    teacher_32b/rank_000.h5
  full/
    student_4b/sample_000000.h5
    teacher_9b/sample_000000.h5
    teacher_32b/sample_000000.h5
```

Each rank writes only its own HDF5 files; model tensors are never gathered.
Files are written as `.inprogress`, flushed, atomically renamed, and paired
with a SHA256 sidecar. BF16 tensors are represented as `uint16` with
`storage_encoding=bfloat16_uint16`, avoiding FP16 overflow or precision loss.
The reader transparently decodes them to FP32.

## Run

The launcher handles the four nodes, disables keepalive immediately before
model work, and restores it on exit:

```bash
MASTER_PORT=29740 \
bash scripts/xopd_analysis/run_4node_activation_capture.sh
```

Useful smoke modes:

```bash
# One prompt, two steps, summary/projection only.
CUDA_VISIBLE_DEVICES=0 python \
  scripts/xopd_analysis/capture_teacher_student_activations.py \
  --max-samples 1 --num-steps 2 --projection-dim 8 \
  --internal-steps 0,1 --summary-only \
  --output-root /path/to/smoke

# Four-node partitioning smoke, one prompt per rank and summary only.
bash scripts/xopd_analysis/run_4node_activation_capture.sh \
  --max-samples 32 --num-steps 2 --projection-dim 8 \
  --internal-steps 0,1 --summary-only \
  --output-root /path/to/smoke_4node
```

Before model loading, the script estimates output bytes from the actual model
architectures and refuses to run unless free space exceeds the estimate plus
25% headroom.

## Offline loading

```python
from pathlib import Path
from scripts.xopd_analysis.load_activation_capture import ActivationCaptureReader

reader = ActivationCaptureReader(
    Path("/shared/xopd/diagnostics/teacher_gap_v1/capture"),
    verify_checksums=True,
)

# Full BF16 block output, decoded to FP32.
h = reader.load(
    sample_index=0,
    model="teacher_32b",
    step=9,
    tensor_key="blocks/single/47/output_joint",
    representation="full",
)

# Small projected representations aligned across all three models.
aligned = reader.load_aligned(
    sample_index=5,
    step=18,
    tensor_key="blocks/double/00/output_image",
    representation="projection",
)

# One scalar without loading a large tensor.
rms = reader.load(
    sample_index=5,
    model="teacher_9b",
    step=18,
    tensor_key="trajectory/model_output",
    representation="summary",
    summary_field="rms",
)
```

CLI integrity check:

```bash
python scripts/xopd_analysis/load_activation_capture.py \
  --root /shared/xopd/diagnostics/teacher_gap_v1/capture \
  --verify-checksums \
  --sample-index 0 --model teacher_32b --step 9 \
  --tensor-key blocks/single/47/output_joint --representation full
```
