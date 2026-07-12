# 2026-07-12 · XOPD experiment plan + auto-relay update

## Experiment matrix (32B FLUX.2-dev -> 4B FLUX.2-klein, XOPD pure-L1, 512px, ZeRO-2, 1000 ep)

Capacity study (three 4B students, one shared teacher ceiling from the OCR run):
| run | data | config | wandb |
|---|---|---|---|
| OCR specialist | ocr | `flux2_klein_32b_to_4b_l1_ocr_1kep.yaml` (teacher baseline ON) | h5j2xknk (done) |
| geneval_enh specialist | geneval_enhanced | `flux2_klein_32b_to_4b_l1_geneval_enh_1kep.yaml` | running (this cluster) |
| mixed 1:1 | geneval_enh + ocr | `flux2_klein_32b_to_4b_l1_geneval_enh_ocr_mixed_1kep.yaml` | 4cnkluwk (SEPARATE cluster, parallel) |

Loss-space / timestep ablations vs the OCR specialist (all OCR data, otherwise identical):
| variant | config | knob |
|---|---|---|
| full-timestep v-MSE | `flux2_klein_32b_to_4b_l1_ocr_1kep.yaml` | default (= OCR specialist) |
| late-timestep v-MSE | `flux2_klein_32b_to_4b_l1_ocr_selective_teacher_1kep.yaml` | xopd_train_steps=[21..27], num_xopd_steps=1, xopd_resample_steps_per_batch |
| full-timestep x-MSE | `flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml` | xopd_dk_space="x" (clean-latent; see x_space_distillation_loss.md) |

## Auto-relay on THIS cluster ([run_three_experts.sh](../../scripts/xopd_cluster/run_three_experts.sh))
Sequence: OCR (done) -> geneval_enhanced (running) -> **OCR x-MSE** -> **OCR selective late-v**.
The MIXED run is dropped from this chain (it trains on a separate cluster in parallel).

Live hand-off (2026-07-12): the original orchestrator (chain ending in mixed) was replaced while
geneval_enhanced was mid-run. The new orchestrator uses `WAIT_FOR=<geneval config>` to block
(without killing geneval or touching keepalive) until geneval finishes, then `START_AT=2` runs the
x-MSE and selective ablations. `WAIT_FOR` requires the awaited run to end with the completion
marker (else abort). Each ablation reuses the OCR run's teacher ceiling (eval_teacher_at_start=false).
