# HY→4B blur: velocity-field and decoded-x0 analysis

## Protocol

- HY checkpoint: `/root/iter_0002400`, copied locally on all four nodes.
- Runtime: `torch-base`, DeepEP `1.2.1+R03C03`, EP=16, world size 32.
- Inputs: the same 128 prompts and the same 28-step 4B student trajectories used
  for the 9B/32B analysis.
- Resolution/guidance: 512px, guidance scale 1.
- Shared flow convention: `x0 = x_t - sigma * velocity`.
- HY uses the same FLUX.2 latent layout and the shared FLUX.2-dev VAE.

HY receives BF16 states. The exact FP16 state from the FLUX capture and the BF16
query state are both stored. Their RMS quantization difference is only `0.001376`.

## Capture

```text
/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/
 diagnostics/teacher_gap_v1/capture_hy
```

The corpus contains 128 prompts × 28 steps, 16 full-capture prompts, 48 HY
layers, 32 DP ranks and EP=16. The 157,626,101,442 bytes across 48 shards were
fully checksum-verified.

## Field result

Across 3,584 prompt-step observations, with cluster-bootstrap confidence
intervals over 128 prompts:

- `RMS(vHY - v4) = 0.1614`, CI `[0.1582, 0.1646]`.
- `RMS(vHY - v9) = 0.1496`, CI `[0.1464, 0.1527]`.
- `RMS(vHY - v32) = 0.5521`, CI `[0.5379, 0.5660]`.
- `cos(ΔHY, Δ9) = 0.4640`, CI `[0.4531, 0.4748]`.
- `cos(ΔHY, Δ32) = 0.2034`, CI `[0.1899, 0.2165]`.

HY is substantially more 9B-like than 32B-like in correction direction.

The detail-direction metrics are also more negative than 9B:

- gradient cosine: HY `-0.2121`, 9B `-0.1681`, 32B `+0.1911`;
- Laplacian cosine: HY `-0.2189`, 9B `-0.1864`, 32B `+0.1614`;
- HY x0 total-variation shift: `-0.00182`, CI
  `[-0.00271, -0.00094]`.

This reproduces the blur signature: HY points against the spatial gradients and
curvature already present in the 4B clean prediction.

## FP32 VAE decoded x0

The 16 full-capture prompts were decoded at steps
`0,4,8,12,16,20,24,27` with the shared VAE in FP32.

Global 4B/HY distance over 128 prompt-step pairs:

- LPIPS `0.0904`, CI `[0.0786, 0.1040]`;
- DINO cosine `0.8990`, CI `[0.8828, 0.9140]`;
- CLIP cosine `0.9498`, CI `[0.9358, 0.9614]`.

The aggregate decoded derivative-energy shifts are negative but their
all-step CIs cross zero because the sign varies at high noise. The steps most
heavily weighted by the direct transition objective are clearer:

- step 16: gradient energy `-1.96e-4`, CI
  `[-3.31e-4, -7.68e-5]`; Laplacian variance `-1.19e-3`;
- step 24: gradient energy `-8.52e-5`, CI
  `[-1.40e-4, -3.55e-5]`;
- step 27: gradient energy `-4.09e-5`, CI
  `[-9.05e-5, -4.82e-6]`.

Thus the decoded images agree with the latent diagnosis specifically in the
middle/late denoising region where `|dt| * |Δv|` gives direct-OPD substantial
gradient weight.

## Implication for selective OPD

HY and 9B both support a detached, per-sample-step directional gate:

```text
c = cos(grad(x0_student), grad(x0_teacher - x0_student))
weight = stop_gradient(1[c >= 0])
loss = mean(weight * per_sample_pathwise_loss)
```

Samples must remain in the microbatch; only their loss is zeroed. This preserves
the number of backward calls, gradient accumulation and one optimizer update per
epoch. A random mask with the same keep rate is required to separate directional
selection from a lower effective learning rate.

## Artifacts

```text
.../analysis/hy/hy_teacher_field_analysis.json
.../analysis/hy/hy_teacher_field_summary.json
.../analysis/hy/hy_teacher_field_analysis.pdf
.../analysis/hy/full_x0/summary.json
.../analysis/hy/full_x0/grids/
```

Reproduce:

```bash
python scripts/xopd_analysis/analyze_hy_teacher_fields.py
bash scripts/xopd_analysis/run_hy_full_capture_x0.sh
```
