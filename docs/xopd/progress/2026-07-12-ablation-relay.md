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
| full-timestep xt-MSE (default) | `flux2_klein_32b_to_4b_l1_ocr_1kep.yaml` | `xopd_dk_space="xt"` |
| late-timestep xt-MSE | `flux2_klein_32b_to_4b_l1_ocr_selective_teacher_1kep.yaml` | xopd_train_steps=[21..27], num_xopd_steps=1, xopd_resample_steps_per_batch |
| full-timestep x0-MSE | `flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml` | xopd_dk_space="x0" |
| full-timestep v-MSE | `flux2_klein_32b_to_4b_l1_ocr_vmse_1kep.yaml` | xopd_dk_space="v" |

Theory: [`../per_timestep_loss_dominance_theory.tex`](../per_timestep_loss_dominance_theory.tex).

## Auto-relay on THIS cluster ([run_three_experts.sh](../../scripts/xopd_cluster/run_three_experts.sh))
**Superseded 2026-07-16** — see [`2026-07-16-loss-space-ablation-plan.md`](2026-07-16-loss-space-ablation-plan.md).

Revised order: OCR (done) -> geneval_enhanced (done) -> **OCR v-MSE** (this cluster) in parallel with
**OCR x0-MSE** (idle / other cluster). **Selective is deferred** until v vs x0 picks a winner
(do not auto-chain selective on xt after vmse).

Historical (2026-07-12): chain was OCR -> geneval_enhanced -> x0-MSE -> selective late-xt -> v-MSE;
that selective-on-xt step is cancelled.
