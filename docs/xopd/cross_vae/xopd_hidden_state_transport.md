# M8：Hidden-State-Conditioned Transport (HSCT) —— 用 student transformer 的 hidden_states 补全跨 VAE 逆映射

> 配套理论：`docs/xopd/cross_vae/xopd_vae_space_align.tex`（§"M3 深入：维度缺口、L1 逆映射瓶颈与 VAE 对齐"）、`docs/xopd/cross_vae/vae_align_stage1_formulation.md`（Stage-1 四行形式化）。
> 配套实现锚点：`src/flow_factory/trainers/xopd/transport.py`、`src/flow_factory/trainers/xopd/trainer.py`、`src/flow_factory/models/stable_diffusion/sd3_5.py`、`scripts/vae_align/train_align.py`。
>
> 本文是一份**架构设计 + 理论分析方案**（不写生产代码，只定架构、接口、训练目标、理论性质、落地路径与风险）。

---

## 0. TL;DR（核心改写）

当前所有传输（M1 pixel / M2 linear / M3 AlignedTransport）的逆映射都是 **只读 $z_S$** 的函数：
$$
Q:\ z_S\in\mathbb{R}^{16\times64\times64}\ \longmapsto\ z_T\in\mathbb{R}^{32\times64\times64}.
$$
这是一个 **欠定扩张**（$d_S{=}16 < d_T{=}32$）。由 Prop.~inverse-deficit（理论文档）：任何**冻结**的 $Q$ 都必须"编造"缺失的 16 维信息，在 on-policy 的**带噪**状态上这种偏置逐步放大 → L1 崩溃。我们刚做的三个 Stage-1 实验（对抗 / 感知 / U-Net 增容）只是让"编造"**更像真的**，并没有补回**丢失的信息**——这正是你的判断："单纯通过 $z_S$ 的信息不足以传输"。

**M8 的改写**：逆映射不再只从 $z_S$ 扩张，而是 **条件于 student transformer 的 hidden_states** $h_S$：
$$
\boxed{\ Q:\ (z_S,\ h_S,\ \sigma)\ \longmapsto\ z_T\ }
$$
其中 $h_S$ 是 student 去噪网络在该状态 $(x_t,t,c)$ 下的内部表征（维度 **1536** $\gg$ 32），蕴含 *噪声 latent + 文本条件 + 学到的图像先验*。它把"**不可约的信息缺口**"换成"**从一个高信息表征里读出缺失信息**"。并且：在 XOPD L1 的 pre-pass 中 student transformer 的前向**已经算过了**（`trainer.py:1867`），$h_S$ 是 **零额外算力**的副产品。

---

## 1. 为什么 $z_S$-only 不够（问题复述 + 经验天花板）

### 1.1 维度缺口（理论）
- SD3.5 student VAE：raw latent $z_S\in\mathbb{R}^{16\times64\times64}$。
- FLUX.2 teacher VAE：raw latent $z_T\in\mathbb{R}^{32\times64\times64}$。
- 逆映射 $z_S\to z_T$ 是 $16\to32$ 扩张：固定 $Q$ 的输出被锁在一个 $\le 16$ 维的子流形里，无法覆盖 teacher 流形 $\mathcal{M}_T$ 的真实 $\sim32$ 维结构。残差有一个**与 $(A,b)$/容量/正则无关的楼层**（Prop.~inverse-deficit）。

### 1.2 经验天花板（Stage-1 A/B/C 实测）
在**干净** latent 上，三种增强都改善了 row-4 的*观感*，但 $\mathcal{L}_{\text{inv-px}}$ 的像素层下界（以及 $\mathcal{L}_{\text{inv-lat}}$ 高居 $\sim$1.5–2.9 不降）说明 $Q(z_S)$ 与真 $z_T$ 之间存在结构性差距：

| 配置 | 机制 | row-4 结果 | 仍存在的根因 |
|---|---|---|---|
| GAN (baseline) | 对抗锐化 | 锐利但"编造"细节 | 信息来自先验，非来自输入 |
| +感知 (A) | teacher-decoder 特征匹配 | 平滑区略好 | 同上 |
| 低 inv_lat (B) | 松绑 $Q{=}z_T$ 约束 | 更自由的流形点 | 仍只读 $z_S$ |
| Q→U-Net (C) | 更大空间感受野 | 略好 | 感受野≠新信息 |

**结论**：在 $z_S$-only 框架内，三者都在"如何更好地猜"上打转。要突破，必须**引入新的信息源**。

---

## 2. 信号：student transformer 的 hidden_states $h_S$

### 2.1 它是什么
SD3.5-medium 的 `SD3Transformer2DModel`（`transformer/config.json`）：

| 配置 | 值 | 含义 |
|---|---|---|
| `num_layers` | 24 | MMDiT 双流 block 数 |
| `num_attention_heads`×`attention_head_dim` | 24×64 = **1536** | hidden 维度 $D$ |
| `patch_size` | 2 | latent→token 的 2×2 patchify |
| `caption_projection_dim` | 1536 | 文本流维度 |

第 $l$ 个 block 的 **图像流 hidden states**：$h_S^{(l)}\in\mathbb{R}^{B\times N_{\text{img}}\times D}$。对 $64\times64$ 的 student latent，patch=2 ⇒ token 网格 $32\times32$，$N_{\text{img}}=1024$，$D=1536$。reshape 回空间即
$$
h_S^{(l)}\ \in\ \mathbb{R}^{B\times 1536\times 32\times 32}.
$$

### 2.2 信息论论证（为什么 $h_S$ 正是缺的那块）
- $z_S$（16ch）是 $x$ 的有损压缩；从它**单独**恢复 $z_T$ 的额外 16 维是病态的。
- $h_S^{(l)}$（1536ch）是 student 去噪网络在 $(x_t,t,c)$ 下的内部状态，是 student **自己预测速度的充分统计量**：它编码了语义内容、纹理、以及**文本条件 $c$**。
- 关键：$h_S$ **条件于 prompt $c$**，这正好消解逆映射的歧义——"这团灰色是毛发还是布料？"由 prompt 决定。$z_S$-only 的 $Q$ 没有这个信息，只能取均值（→糊）。
- 形式上：逆映射误差的不可约下界来自 $I(z_T; \cdot)$ 的信息量。把输入从 $z_S$ 换成 $(z_S,h_S)$ 后
$$
I\big(z_T;\,(z_S,h_S)\big)\ \ge\ I(z_T;\,z_S),
$$
且因为 $h_S$ 携带 $c$ 与高维图像先验，通常 **$\gg$**。"维度缺口"不再以 $d_S{=}16$ 为底，而是以 $16{+}1536$ 的有效输入为底——**缺口被信息填上**。

### 2.3 空间对齐（设计上极干净的巧合）
| 张量 | 通道 | 空间网格 |
|---|---|---|
| student VAE latent $z_S$ | 16 | 64×64 |
| **student transformer 图像 token $h_S$** | **1536** | **32×32** |
| **teacher packed/transformer latent** | **128** | **32×32** |
| teacher raw VAE latent $z_T$ | 32 | 64×64 |

student transformer token 网格（32×32）与 teacher packed latent 网格（32×32）**天然重合**（都是某个 64×64 基的 2×2 patchify）。所以条件逆映射**就在 32×32 网格上做**，且 teacher 查询本来就在 packed latent 上进行（`AlignedTransport._raw_to_packed`）：

$$
\underbrace{\text{patchify}(z_S)}_{64\text{ch}@32^2}\ \Vert\ \underbrace{h_S}_{1536\text{ch}@32^2}
\ \xrightarrow{\ Q\ }\ \underbrace{\widehat{z}_T^{\text{packed}}}_{128\text{ch}@32^2}
\ \xrightarrow{\text{unpatchify}}\ \underbrace{\widehat z_T}_{32\text{ch}@64^2}.
$$

**无 bilinear、无 patchify 错位**（对照早期 un-shuffle 棋盘伪影）。

---

## 3. 架构：M8 / HSCT

### 3.1 前向 $P$（teacher→student）：**保持不变**
仍是 **线性** $1{\times}1$（$32\to16$）。前向是 $32\to16$ 的**过定**压缩，无缺口、无需 hidden_states。线性是为了 L1 转移均值推前精确：$\mathbb{E}[Pz]=P\,\mathbb{E}[z]$（Prop.~仿射）。**这一性质是 L1 正确性的基石，绝不破坏。**

### 3.2 逆向 $Q$（student→teacher）：**条件化**
$$
Q(z_S,h_S,\sigma)\;=\;\underbrace{\text{base}(z_S)}_{\text{do-no-harm 线性基}}\;+\;\underbrace{g_\theta\big(\text{patchify}(z_S),\,\pi(h_S),\,\sigma\big)}_{\text{hidden-state 补全残差}}
$$
- **base**：沿用现有"线性 base + zero-init"逆映射（只读 $z_S$），保证起步 = 现有 AlignedTransport（do-no-harm，$\ge$ baseline from epoch 0）。
- **$\pi$**：hidden 投影 $1{\times}1$ 卷积 $1536\to d_h$（如 $d_h{=}256$，降维省算力）。
- **$g_\theta$**（融合网络，packed 网格 32×32 上）：concat$[\text{patchify}(z_S)\,(64),\ \pi(h_S)\,(d_h)]$ → 若干 $3{\times}3$ conv（可加残差块）→ 输出 128ch。**最后一层 zero-init**（残差从 0 注入）。
- **$\sigma$-条件**：因为 rollout 上 $Q$ 作用于**带噪** $x_t$，映射依赖噪声水平 $\sigma{=}t/T$。用 AdaLN-Zero 式调制注入 $\sigma$（见 `AdaLNTransport._modulation` 的 sinusoidal+MLP，zero-init）。

**融合方案备选**（doc 给出，先用 (a)）：
- (a) **Concat-conv**（推荐首版）：简单、稳、足够。
- (b) **FiLM**：用 $h_S$ 回归逐通道 $(\gamma,\beta)$ 调制 base。
- (c) **Cross-attention**：latent 网格 query、$h_S$ token 作 key/value。最强但最重，留作 ablation。

### 3.3 用哪一层 / 哪几层的 $h_S$？
- 太早：偏低层纹理、语义弱；太晚：偏预测残差、可能丢细节。
- **首版**：取中后层单块（如第 18–20 / 24 块）。
- **进阶**：对 $\{$早, 中, 晚$\}$ 三块做**可学习加权和**（一个 softmax 权重），兼顾语义与纹理。设为 config。

### 3.4 CFG 分支的选择
adapter 在 CFG 时会把 cond/uncond 拼 batch（`sd3_5.py:608-619`）。$h_S$ 要取**条件（text）分支**（或 CFG 合成后的等效表示），与"$h_S$ 携带 prompt 信息"的论证一致。设为 config，默认取 cond 分支。

---

## 4. 训练/推理同分布：clean vs noisy（最关键的研究点）

现有 Stage-1 在**干净** $z_S{=}\mathcal E_S(x)$ 上训 $Q$；但 L1 rollout 上 $Q$ 作用于**带噪** $x_t$ 且需 $h_S(x_t,t,c)$。必须把 Stage-1 **扩到带噪域**，并把 student transformer（冻结）放进训练回路产 $h_S$。

### 4.1 带噪样本构造
对每张语料图 $x$：
$$
z_{S,0}=\mathcal E_S(x),\quad z_{T,0}=\mathcal E_T(x);\qquad
\sigma\sim\mathcal U(0,1),\ \ \varepsilon_S\sim\mathcal N(0,I),
$$
$$
x_t=(1-\sigma)\,z_{S,0}+\sigma\,\varepsilon_S\quad(\text{student flow-matching 路径}),
$$
冻结 student transformer 前向 $\Rightarrow h_S=h_S^{(l)}(x_t,\sigma,c)$。

### 4.2 $Q$ 的回归目标（给出选项 + 推荐）
L1 需要在 student 状态 $x_t$ 处拿 teacher 的**转移均值** $\mu_T$（在 teacher 对应状态 $x_{T,t}$ 上）。于是 $Q$ 要把 $(x_t,h_S,\sigma)$ 映到 teacher 路径上 level-$\sigma$ 的合法点：

- **目标 (i) latent 一致**：$x_{T,t}=(1-\sigma)\,z_{T,0}+\sigma\,\varepsilon_T$（$\varepsilon_T$ 固定采样，定义良好的回归靶），损失 $\lVert Q(x_t,h_S,\sigma)-x_{T,t}\rVert^2$。
- **目标 (ii) teacher 路径 on-manifold**（带噪版 $\mathcal L_{\text{inv-px}}$）：由 teacher 自身在 level-$\sigma$ 的去噪估计 $\widehat z_{T,0}=\text{denoise}_T(Q(\cdot),\sigma)$ 应解回 $x$：$\lVert \mathcal D_T(\widehat z_{T,0})-x\rVert_1$。把 $Q$ 钉到 teacher **路径流形**而非仅干净流形。
- **目标 (iii) 速度一致（最强、可选）**：teacher 在 $Q(\cdot)$ 处的速度/转移均值与"真 $x_{T,t}$ 处"一致——直接对齐 L1 真正要用的量。

**推荐**：(i)+(ii) 为主（定义清晰、稳），(iii) 作为后续增强。$\sigma$-条件必开。可叠加我们已验证有效的 **GAN/感知** 项（带噪域同样适用）。

> 备注：跨两个 VAE 的"同一噪声"无良定义，故用"**共享干净内容** $z_{T,0}$（同一 $x$）+ teacher 自身 level-$\sigma$ 噪声"来定义对应关系；L1 只需 teacher 转移均值落在合理区域，噪声实现的精确性次要。

---

## 5. 理论性质

### 5.1 推前精确性：保持
前向 $P$ 线性 ⇒ L1 转移均值推前 $\mu_S=P\mu_T$ 仍**精确**。$Q$ 只产出 L1 的**查询点** $x_T$（不参与任何期望/推前），故 $Q$ 任意非线性且**任意条件化**都不破坏推前性质（同 stage1 doc §3.1 论证）。**条件于 $h_S$ 是"免费的"理论午餐。**

### 5.2 无偏界：缺口被信息替换
推广理论文档 Prop.~"M3 后 L1 无偏"。设条件逆映射的 on-manifold 残差
$$
\eta(x_t)=\big\lVert \mathcal D_{T,\sigma}\!\big(Q(x_t,h_S,\sigma)\big)-\text{(target)}\big\rVert,
$$
则 L1 转移均值传输偏差
$$
\text{bias}\ \le\ \mathrm{Lip}(v_T)\cdot \eta\ +\ \delta_{\text{align}}.
$$
与 M3 不同的是：$\eta$ 的下界由 $I\big(z_T;(z_S,h_S)\big)$ 决定，而非 $I(z_T;z_S)$。因 $h_S$ 信息量 $\gg z_S$，$\eta$ **不再以 $d_S{<}d_T$ 缺口为底**，可训练压到远低于 $z_S$-only 的水平。这就是把"不可约缺口"换成"可压低残差"的精确陈述。

### 5.3 移动靶问题（重要风险，单列）
$h_S$ 来自**正在被训练的 student**。XOPD 进行中 student 在变 ⇒ $h_S$ 分布漂移 ⇒ 用初始 student 训的 $Q$ 会过期。缓解：
1. **跨状态鲁棒训练**：Stage-1 在多样 $(\sigma,c,\text{student-state})$ 上训 $Q$，提升对 $h_S$ 漂移的鲁棒性。
2. **在线刷新**：复用 trainer 已有的 `update_online`（`transport.py`）在 XOPD warm-up 周期性微调 $Q$（$h_S$ 由当前 student 现采）。
3. **on-policy 语料**：Stage-1 后期用当前 student 的 rollout 状态而非纯合成 $x_t$。
4. **低敏感设计**：base 只读 $z_S$ 保证即便 $h_S$ 漂移，仍有合理兜底（do-no-harm 起点不退化）。

---

## 6. 代码落地（设计级，附锚点）

| 改动面 | 文件:锚点 | 内容 |
|---|---|---|
| **暴露 $h_S$** | `models/stable_diffusion/sd3_5.py:622` | 在 `transformer(...)` 上挂 forward hook 抓第 $l$ 块图像流 hidden（或自定义 forward 返回中间层）；经 `return_kwargs="hidden_states"` 暴露到 forward 输出 dataclass。 |
| **L1 传 $h_S$** | `trainer.py:1867`（pre-pass）→`:1874`/`:1783` | `student_out` 已含 $h_S$；把它透传进 `transition_mean_to_student(x_S, query_teacher_mean, sigma=sigma, student_hidden=h_S)`。两赪结构不变（逆映射在 no-grad pre-pass，正合适）。 |
| **条件传输类** | `transport.py` 新增 `HiddenStateAlignedTransport` | 继承/复用 `AlignedTransport`：`P` 不变；`transition_mean_to_student` 从 `ctx["student_hidden"]` 取 $h_S$ 喂条件 $Q$；packed↔raw 用现有无损互转。 |
| **Stage-1 扩展** | 新脚本 `scripts/vae_align/train_align_hsct.py`（基于 `train_align.py`） | 带噪域 + 冻结 student transformer 产 $h_S$ + $\sigma$-条件 $Q$ + §4.2 损失（可叠加已验证的 GAN/感知）。 |
| **配置** | `hparams/training_args.py` | `vae_transport="hsct"`；hidden 层选择/层数、融合类型 (a/b/c)、$d_h$、$\sigma$-条件开关、CFG 分支、在线刷新周期。 |
| **工厂** | `transport.py:build_transport` | 注册 `"hsct"`。 |

> 注：`AlignedTransport` 的 packed↔raw 无损互转、teacher 查询管线已就绪（`transport.py:1397-1427`），M8 主要是把"逆映射的输入"从 $z_S$ 扩成 $(z_S,h_S,\sigma)$，并新增带噪 Stage-1 训练脚本。

---

## 7. 训练与验证计划

### 7.1 Stage-1（带噪域，离线）
1. 语料复用现有 11.6k teacher 图（`/root/vae_align_corpus`）。
2. 每步：采 $\sigma,\varepsilon$ 造 $x_t$ → 冻结 student transformer 取 $h_S$ → 训条件 $Q$（+ $P$ + student decoder 仍可微调，复用 stage1 框架）。
3. 损失：$\mathcal L_{\text{inv-lat}}$(i) + $\mathcal L_{\text{inv-px}}$(ii，带噪版) + $\sigma$-条件 +（可选 GAN/感知）+ 前向/AE/cyc 沿用。
4. viz 仍四行，但 row-4 = $Q(x_t,h_S,\sigma)$；**关键对比**：同 $\sigma$ 下 $Q(z_S)$ vs $Q(z_S,h_S)$ 的 inv-recon。

### 7.2 Ablation
- 输入：$Q(z_S)$（=旧）vs $Q(z_S,h_S)$ vs $Q(x_t,h_S,\sigma)$。
- 层：单块（早/中/晚）vs 多块加权。
- 融合：(a) concat-conv / (b) FiLM / (c) cross-attn。
- $d_h$（hidden 投影维度）扫描。

### 7.3 Stage-2（真正判据）
接入 XOPD L1：**geneval 是否稳定不崩**（对照 $z_S$-only 的崩溃曲线）。这是 M8 成功与否的最终标准。

---

## 8. 成本 / 风险一览

| 项 | 评估 |
|---|---|
| L1 推理算力 | $h_S$ 免费（pre-pass 已算）；条件 $Q$ 轻量。**几乎零增量。** |
| Stage-1 算力 | 新增冻结 student transformer 前向（仅训练期）。 |
| 移动靶（§5.3） | 主要风险；用鲁棒训练 + 在线刷新缓解。 |
| 分布匹配 | $h_S$ 须在真实/接近真实 rollout 状态上训（on-policy 语料）。 |
| 实现复杂度 | hook 抓 hidden + 透传 ctx + 带噪 Stage-1；改动可控、接口已就位。 |

---

## 9. 增量里程碑（建议落地顺序）

1. **M8.0 探针**：给 sd3_5 adapter 加 hidden hook + return_kwarg，离线 dump $h_S$ 形状/统计，确认 32×32×1536 与 teacher packed 网格对齐。
2. **M8.1 干净域 sanity**：先在**干净** $z_S$ + $h_S(z_{S,0},\sigma{=}0,c)$ 上训条件 $Q$，验证 inv-recon 明显低于 §1.2 的 A/B/C 天花板（证明"信息确实补上了"）。
3. **M8.2 带噪域**：上 §4 完整带噪 + $\sigma$-条件训练。
4. **M8.3 Stage-2**：接 XOPD L1，看 geneval 不崩。
5. **M8.4 在线刷新**：开 `update_online` 应对 student 漂移。

---

## 10. 一句话总结

把逆映射从"**从 16 维硬扩到 32 维**"（必然编造、必然有不可约缺口）改写为"**从 student 去噪网络的 1536 维、带 prompt 条件的内部表征里读出缺失信息**"——在 L1 pre-pass 中这份表征已被免费算出，且其 token 网格与 teacher packed latent 天然对齐。前向 $P$ 保持线性以维持 L1 推前精确性；逆向 $Q$ 条件化不破坏任何理论性质，却把"不可约信息缺口"替换为"可训练压低的对齐残差"。
