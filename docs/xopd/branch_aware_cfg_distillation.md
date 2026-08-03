# XOPD 的 Branch-Aware CFG 蒸馏

> 状态：已实现（PDM v1）  
> 范围：`trainer_type: xopd` 的 L0、direct L1 和 A4 / marginal-CFM  
> 参考：[Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation](https://arxiv.org/abs/2607.24731v2)

## 1. 目标与非目标

当前 XOPD 在一次 teacher/student forward 后只保留一个 CFG 合成结果：

$$
\widetilde v_M(\gamma_M)
=v_M^-+\gamma_M(v_M^+-v_M^-),
\qquad M\in\{T,S\}.
$$

训练因此只约束一个合成场；`teacher_guidance_scale=1` 或
`student_guidance_scale=1` 时，对应 forward 甚至不会执行 negative branch。这和
“同时监督 positive/negative 两个分支”是两个不同概念。

本文档定义一个只作用于 XOPD 算法族的 branch-aware feature。目标是：

1. 保持 rollout 的 CFG 行为不变，仍由现有 teacher/student guidance scale 决定轨迹；
2. 在同一个已访问状态上暴露 teacher/student 的 positive 和 negative 原始预测；
3. 使用可辨识的 branch-aware objective，而不是只匹配某个 guidance scale 下的合成值；
4. 默认完全保持旧行为和旧配置兼容；
5. 不改变 `opd`、`diffusion-opd`、MoF、GRPO 等 trainer。

这里的“相关变体”特指同一个 `XOPDTrainer` 内的 target mode：
`direct`、`p_opd` 和 `marginal_cfm`。虽然 `xpdm`、`xdmd`、`xopd_dm`
在代码上复用了部分 XOPD 基类设施，但它们不是这里讨论的 OPD 局部速度匹配目标，
第一版不自动改变它们的 loss。

## 2. 论文结论：不是简单地加两个独立 MSE

定义 branch error

$$
e_+=v_T^+-v_S^+,\qquad e_-=v_T^--v_S^-.
$$

如果 teacher/student 使用同一个训练 guidance scale $\gamma$，只匹配 CFG 合成结果得到

$$
\ell_{\mathrm{composed}}
=\|\gamma e_+ +(1-\gamma)e_-\|_2^2.
$$

它只约束 branch error 的一个线性组合。任意满足

$$
e_+=\frac{\gamma-1}{\gamma}e_-
$$

的非零误差都可以使合成误差为零。两个分支可以互相抵消，训练 guidance scale
之外的推理 scale 会暴露这种退化。论文把 privileged negative conditioning 下
positive error 下降、negative error 上升的对抗性动力学称为
Negative Branch Asymmetry（NBA）。

论文的主方法是 Positive--Direction Matching（PDM），不是简单的
“positive MSE + negative MSE”。令

$$
d_M=v_M^+-v_M^-,
$$

则

$$
\ell_{\mathrm{PDM}}
=\|v_T^+-v_S^+\|_2^2
+\lambda\|d_T-d_S\|_2^2
=\|e_+\|_2^2+\lambda\|e_+-e_-\|_2^2,
\qquad \lambda>0.
$$

第一项锚定 positive prediction，第二项保留 CFG conditional direction。零损失要求
$e_+=e_-=0$，因此可以在任意 inference guidance scale 下重新合成。

论文也给出 Independent Branch Matching（IBM）作为对照：

$$
\ell_{\mathrm{IBM}}
=\gamma^2\|e_+\|_2^2+(\gamma-1)^2\|e_-\|_2^2.
$$

IBM 同样可辨识，但它的主要价值是 ablation。XOPD PDM v1 不实现 IBM；若后续需要
论文对照，可在不改变 branch API 的前提下增加一个纯函数 loss helper。

## 3. 当前 XOPD 的行为与缺口

### 3.1 两个 guidance scale 只选择合成场

`XOPDTrainingArguments` 当前有：

```yaml
train:
  teacher_guidance_scale: 4.0
  student_guidance_scale: 1.0
```

它们控制 teacher query 和 student rollout/forward 的合成 scale。它们不是两个
branch loss 的权重。尤其是 `guidance_scale <= 1` 时，
`Flux2KleinAdapter._predict_velocity` 只运行 positive pass，negative embeddings
也不会被要求存在。

当 $\gamma_T\ne\gamma_S$ 时，当前 direct loss 实际比较

$$
\widetilde v_T(\gamma_T)-\widetilde v_S(\gamma_S),
$$

它既混合了模型差异，也混合了 CFG scale 差异，仍不能识别各 branch。

### 3.2 Adapter 丢弃了原始分支

`Flux2KleinAdapter._predict_velocity` 已经顺序计算 positive 和 negative forward，
但在返回前立刻执行

```text
v_composed = v_negative + guidance_scale * (v_positive - v_negative)
```

并只返回 `v_composed`。因此最简洁的实现不是在 trainer 再做两次重复 forward，
而是在 adapter 内部一次性返回

```text
positive, negative, composed
```

并让旧调用默认仍只看到 `composed`。

### 3.3 预处理缓存由 scale 隐式决定

当前 negative prompt embeddings 是否进入 preprocessing cache 取决于
teacher/student guidance scale 是否大于 1。Branch-aware loss 即使在 rollout
scale 为 1 时也必须获得 negative branch，所以 cache key 和预处理条件必须显式包含
“训练 loss 是否需要 branches”，不能再只从 rollout scale 推断。

### 3.4 A4 callback 只保存合成速度

`marginal_cfm` rollout 当前只保存 `noise_pred`。Branch-aware A4 需要在相同 callback
位置额外保存 source model 的 `positive_noise_pred` 和 `negative_noise_pred`。
old/teacher 的 trajectory-level Bernoulli routing、状态、`latent_index_map` 和
`callback_index_map` 都不应改变。

## 4. 建议的配置语义

使用 objective enum，而不是含义模糊的 `dual_branch: true`：

```yaml
train:
  # Backward-compatible default.
  xopd_cfg_objective: composed  # composed | pdm

  # Only used by pdm; must be finite and > 0.
  xopd_pdm_lambda: 1.0

  # Existing knobs remain rollout/composition knobs, not branch-loss weights.
  teacher_guidance_scale: 4.0
  student_guidance_scale: 1.0
```

约束：

- `composed`：现有行为逐 bit 保持，不能增加额外 forward；
- `pdm`：强制 teacher/student 都提供 positive 和 negative conditioning；
- `xopd_pdm_lambda` 在非 `pdm` 模式下忽略；
- branch-aware 第一版只支持 same-architecture、same-VAE、identity transport；
- 与 P-OPD、cross-VAE transport、pixel-space loss、detail mask 的组合 fail fast，
  不应静默退回 composed loss。

`teacher_guidance_scale` 和 `student_guidance_scale` 继续只决定 rollout 或需要的
composed diagnostics。PDM 本身与训练 scale 无关。

## 5. 统一的数据结构和 loss contract

建议增加一个不可变的 branch prediction value object：

```python
CFGVelocityPrediction(
    positive=v_pos,
    negative=v_neg,
    composed=v_neg + guidance_scale * (v_pos - v_neg),
)
```

必须验证：

- 三个 tensor shape、dtype、device 一致；
- branch-aware 模式下 `negative` 不允许为 `None`；
- 所有 tensor 必须 finite；
- batch/event shape 必须与当前 latent 一致。

统一的纯函数 loss 接口返回 per-sample tensor `(B,)`：

```python
compute_xopd_pdm_loss(
    student_positive: Tensor,
    student_negative: Tensor,
    teacher_positive: Tensor,
    teacher_negative: Tensor,
    pdm_lambda: float,
    teacher_guidance_scale: float,
    student_guidance_scale: float,
) -> PDMVelocityLoss
```

`CFGDistillationLoss` 至少包含：

- `loss`: 最终 per-sample objective；
- `positive_mse`;
- `negative_mse`;
- `direction_mse`;
- `composed_mse`;
- `positive_error_rms`;
- `negative_error_rms`;
- `direction_error_rms`.

所有 reduction 使用 float32 event mean；teacher targets 和 diagnostics detach，
student tensors 保留梯度。Trainer 只负责 target-mode routing、mask/gate、backward
和 logging，不应在多个分支内重复 PDM 公式。

## 6. 各 XOPD target mode 的接入

### 6.1 L0 warmup

保持 teacher-generated `z_0`、随机连续 $t$ 和 weighting $w(t)$ 不变。在同一个
$z_t$ 上各做一次 teacher/student branch-aware prediction：

$$
L_0=w(t)\,\ell_{\mathrm{PDM}}.
$$

这与论文的 pointwise branch-aware velocity matching 直接一致。

### 6.2 Direct L1

保持 student on-policy rollout 和训练 timestep 选择不变。Teacher pre-pass 缓存
branch target，gradient pass 得到 student branches，然后调用统一 loss helper。
第一版在 velocity space 计算 PDM；旧 `composed` 路径仍使用现有
transition-mean `D_k`，保证兼容。

这会把“rollout 使用哪个 CFG scale”和“loss 监督哪些 branch”彻底解耦：
轨迹仍可由 GS=1 或 GS=4 的 composed field 生成，但每个访问状态同时监督两个 branch。

### 6.3 P-OPD

P-OPD 的 probability-mixture responsibility 仍由行为 transition、teacher 的
**composed rollout field** 和共享协方差计算，不能把 PDM 的两个 MSE 当成新的概率密度。
一种可能的 branch-aware 扩展会用同一个 responsibility gate PDM：

$$
L_{\mathrm{P\text{-}OPD+PDM}}
=\operatorname{sg}(\Gamma_T)\,\ell_{\mathrm{PDM}}.
$$

它可以保留 P-OPD 的 teacher-vs-old proximity gate，但不再是原始
“Gaussian transition-mixture KL 的精确一阶 surrogate”；日志和文档必须称为
branch-aware gated surrogate，不能继续宣称完全相同的概率解释。

PDM v1 因此明确拒绝 `xopd_target_mode: p_opd`，而不是静默改变概率语义。

### 6.4 A4 / marginal-CFM

source routing 和 source trajectory 不变。每个 selected source forward 保存：

```text
positive_noise_pred
negative_noise_pred
noise_pred  # 原 composed scheduler input，继续真正驱动轨迹
```

训练在 routed source states 上匹配 positive prediction 和 conditional direction。
由于重合时 student 可以在任意 $\gamma$ 下重建 source composed field，原 A4 的
mixture trajectory 目标仍由 `noise_pred` 所定义；branch tensors 是对该目标的
可辨识分解，而不是另画一次 source branch。

必须保持各 rank 的 old-then-teacher collective call order 不变。新增 callback
不能引入按 rank 不同的 forward 分支。

## 7. 性能与显存

一次 CFG branch-aware prediction 的 transformer 成本是两个 branch forward：

- 旧 GS>1 composed 模式本来就是两次 forward，因此暴露 branches 几乎没有额外算力，
  只增加被选择 timestep 的 target/callback 存储；
- 旧 GS=1 模式只有一次 forward，切换 PDM 后计算量接近 2 倍，这是 feature 本身不可避免；
- 不允许 trainer 为了取 branches 在现有 composed forward 之外再重复两次模型调用；
- teacher branch cache 应使用现有 latent storage dtype/offload 策略；
- A4 额外保存两个 callback，峰值存储约从一个 velocity tensor 增至三个。若显存敏感，
  `composed` 可由 `positive/negative + rollout scale` 重建，训练样本中只需持久化两个
  branches；scheduler 当步仍使用即时 composed tensor。

## 8. 日志与验收

新增 `train/cfg_branch/...`：

- `positive_mse`;
- `negative_mse`;
- `direction_mse`;
- `composed_mse_at_teacher_gs`;
- `composed_mse_at_student_gs`;
- `positive_error_rms`;
- `negative_error_rms`;
- `nba_gap = negative_error_rms - positive_error_rms`.

评估至少覆盖 guidance scale `{1, train_scale, 4}`。仅看某个 inference scale 的绝对
退化不能证明 NBA；应比较 teacher、composed-XOPD 和 PDM-XOPD 的 scale sensitivity。

验收条件：

1. `composed` 模式的现有单测、配置和数值行为不变；
2. 合成 toy case 中，composed loss 可以在非零 branch errors 下为零，而 PDM 不为零；
3. PDM 为零当且仅当两个 branch errors 都为零；
4. GS=1 rollout + PDM 仍实际执行并监督 negative branch；
5. 四节点 A4 保持 rank-synchronous source call order；
6. 日志能分别观察 positive/negative error，不能只记录最终 PDM 标量。

## 9. 推荐的首轮实验

固定 9B $\rightarrow$ 4B、Geneval+OCR、A4 alpha、seed、batch/timestep 选择和优化器，
只比较：

| run | rollout GS | objective | $\lambda$ |
|---|---:|---|---:|
| baseline | 1 | composed | -- |
| PDM-1 | 1 | pdm | 1.0 |
| baseline-CFG | 4 | composed | -- |
| PDM-CFG | 4 | pdm | 1.0 |

每个 checkpoint 用 GS=1/2/4 做 matched-seed eval，并画
`positive_error_rms`、`negative_error_rms`、`direction_error_rms` 和 reward。
先使用静态 $\lambda=1$；只有确认两个 branch 的尺度明显失衡后再讨论动态权重。
