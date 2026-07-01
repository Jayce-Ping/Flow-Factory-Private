# VAE 隐空间对齐 (M3 档A) —— 四条重建路径的形式化

> 配套实现：`scripts/vae_align/train_align.py`（Stage 1b 训练）、`scripts/vae_align/gen_corpus.py`（Stage 1a 语料）。
> 理论背景：`docs/mof/xopd_vae_space_align.tex` §"M3 深入：维度缺口、L1 逆映射瓶颈与 VAE 对齐"。
>
> 本文形式化对齐训练可视化网格（`/root/vae_align_ckpt/viz_step*.png`）中的**四行**——即四条重建/映射路径（"四个实验"），给出每条的公式、作用、对应 loss 与验证目标。

---

## 0. 设定与记号

跨 VAE 蒸馏：teacher = FLUX.2-klein（latent 32ch@64×64），student = SD3.5-medium（latent 16ch@64×64），两者 8× 下采样、空间分辨率相同。

| 符号 | 含义 | 状态 |
|---|---|---|
| $x$ | 训练样本图（teacher 生成语料），归一化到 $[-1,1]$ | 数据 |
| $\mathcal{E}_T,\ \mathcal{D}_T$ | teacher VAE 的编码器 / 解码器 | **冻结** |
| $\mathcal{E}_S$ | student VAE 的编码器 | **冻结** |
| $\mathcal{D}_S'$ | student VAE 的解码器（+ `post_quant_conv`） | **微调** |
| $P:\ \mathbb{R}^{32}\!\to\!\mathbb{R}^{16}$ | 前向传输 teacher→student，$1{\times}1$ 卷积（**线性**） | **训练** |
| $Q:\ \mathbb{R}^{16}\!\to\!\mathbb{R}^{32}$ | 逆向传输 student→teacher，线性 base + zero-init 卷积残差（**非线性**） | **训练** |

记 raw VAE latent：
$$
z_T \;=\; \mathcal{E}_T(x)\ \in\ \mathbb{R}^{32\times64\times64},
\qquad
z_S \;=\; \mathcal{E}_S(x)\ \in\ \mathbb{R}^{16\times64\times64}.
$$

可训练参数集中在一个 `AlignModule`（含 $P,Q,\mathcal{D}_S'$）以保证 DDP 梯度同步正确。

---

## 1. 四条重建路径（viz 的四行 / "四个实验"）

| 行 | 名称 | 公式 | encode | 传输 | decode |
|---|---|---|---|---|---|
| **1** | 目标 Target | $x$ | — | — | — |
| **2** | student 自重建 (AE) | $\hat{x}_{\text{ae}}=\mathcal{D}_S'(\mathcal{E}_S(x))$ | $\mathcal{E}_S$ | 无 | $\mathcal{D}_S'$ |
| **3** | 前向 P (tea→stu) | $\hat{x}_{\text{fwd}}=\mathcal{D}_S'\!\big(P\,\mathcal{E}_T(x)\big)$ | $\mathcal{E}_T$ | $P$ | $\mathcal{D}_S'$ |
| **4** | 逆向 Q (stu→tea) | $\hat{x}_{\text{inv}}=\mathcal{D}_T\!\big(Q\,\mathcal{E}_S(x)\big)$ | $\mathcal{E}_S$ | $Q$ | $\mathcal{D}_T$ |

### 行 1 — 目标 $x$
- **形式化**：$x$（语料图，监督参照，无 VAE 往返）。
- **作用**：所有 loss 的 ground-truth。

### 行 2 — student 自重建 $\hat{x}_{\text{ae}}=\mathcal{D}_S'(z_S)$
- **含义**：冻结 student 编码器 + 微调 student 解码器，重建 $x$ 自身的 latent。
- **对应 loss**：$\displaystyle \mathcal{L}_{\text{ae}}=\big\lVert \mathcal{D}_S'(\mathcal{E}_S(x))-x\big\rVert_1$。
- **验证（行 1 vs 2）**：**do-no-harm** —— 微调 $\mathcal{D}_S'$ 去解 $P$ 搬来的 teacher latent 时，不破坏它解自己 latent 的能力。

### 行 3 — 前向 $P$ $\hat{x}_{\text{fwd}}=\mathcal{D}_S'(P\,z_T)$
- **含义**：teacher latent 经**线性** $P$ 搬到 student 空间，再用微调 student 解码器解出。
- **对应 loss**：像素 $\displaystyle \mathcal{L}_{\text{fwd-px}}=\big\lVert \mathcal{D}_S'(P\,\mathcal{E}_T(x))-x\big\rVert_1$；隐空间 $\displaystyle \mathcal{L}_{\text{fwd-lat}}=\big\lVert P\,z_T-z_S\big\rVert_2^2$。
- **验证（行 1 vs 3）**：**前向高保真** —— $P$(线性) + 解码器适应 共同保证 teacher 信号搬进 student 空间后能解出正确图。$P$ 线性是为了让 L1 转移均值推前精确（Prop. 仿射）。

### 行 4 — 逆向 $Q$ $\hat{x}_{\text{inv}}=\mathcal{D}_T(Q\,z_S)$
- **含义**：student latent 经**非线性** $Q$ 搬回 teacher 空间，再用**冻结** teacher 解码器解出。
- **对应 loss（关键）**：像素 $\displaystyle \mathcal{L}_{\text{inv-px}}=\big\lVert \mathcal{D}_T(Q\,\mathcal{E}_S(x))-x\big\rVert_1$；隐空间 $\displaystyle \mathcal{L}_{\text{inv-lat}}=\big\lVert Q\,z_S-z_T\big\rVert_2^2$。
- **验证（行 2 vs 4）**：**逆映射 on-manifold** —— 同一 student latent $z_S$，行 2 直接 student 解码、行 4 经 $Q$ 后 **teacher** 解码。因为只有落在 teacher 数据流形 $\mathcal{M}_T$ 上的 latent 才能被冻结的 $\mathcal{D}_T$ 解成合法图，故 $\mathcal{L}_{\text{inv-px}}$ 把 $Q(z_S)$ **钉到流形**。这是修复跨 VAE on-policy L1 崩溃（$d_S<d_T$ 逆映射 off-manifold，见理论文档 Prop. inverse-deficit）的核心。

---

## 2. 完整训练目标

还有循环一致性：
$$
\mathcal{L}_{\text{cyc}}
=\big\lVert Q(P\,z_T)-z_T\big\rVert_2^2
+\big\lVert P(Q\,z_S)-z_S\big\rVert_2^2 .
$$

总损失（默认权重 $w_{\text{ae}}{=}w_{\text{fwd-px}}{=}w_{\text{fwd-lat}}{=}w_{\text{inv-px}}{=}w_{\text{inv-lat}}{=}1,\ w_{\text{cyc}}{=}0.5$）：
$$
\boxed{\;
\mathcal{L}
= w_{\text{ae}}\mathcal{L}_{\text{ae}}
+ w_{\text{fwd-px}}\mathcal{L}_{\text{fwd-px}} + w_{\text{fwd-lat}}\mathcal{L}_{\text{fwd-lat}}
+ w_{\text{inv-px}}\mathcal{L}_{\text{inv-px}} + w_{\text{inv-lat}}\mathcal{L}_{\text{inv-lat}}
+ w_{\text{cyc}}\mathcal{L}_{\text{cyc}}
\;}
$$

仅更新 $\{P,\ Q,\ \mathcal{D}_S'(\text{+post\_quant})\}$；$\mathcal{E}_T,\mathcal{D}_T,\mathcal{E}_S$ 冻结。
注意 $\mathcal{L}_{\text{inv-px}}$ 中 $\mathcal{D}_T$ **冻结但在计算图中**（梯度回传到 $Q$），故 teacher 解码器只提供"流形先验"、不更新。

---

## 3. 设计理由（为何这样形式化）

1. **$P$ 线性、$Q$ 非线性（不对称）**
   - $P$ 用于 L1 把 teacher **转移均值** $\mu_T$ 推前到 student：$\mu_S=P\,\mu_T$。线性 ⇒ $\mathbb{E}[Pz]=P\,\mathbb{E}[z]$，推前**精确**、无逐点 Jacobian（Prop. 仿射）。
   - $Q$ 只产出 L1 的**查询点** $x_T=Q(x_S)$（不经它做期望/推前），故非线性无害；且 $16\!\to\!32$ 是欠定扩张，必须靠非线性 + $\mathcal{L}_{\text{inv-px}}$ 钉到流形。

2. **微调 $\mathcal{D}_S'$（CV-VAE / M3 精髓）**：线性 $P$ 无法把 FLUX latent 完美映成 SD3.5 latent（$\mathcal{L}_{\text{fwd-lat}}$ 有残差），让 student 解码器吸收该残差 ⇒ 行 3 仍锐利；$\mathcal{L}_{\text{ae}}$ 保证 do-no-harm。

3. **raw VAE latent 层对齐**：两 VAE 同在 $64{\times}64$，$P/Q$ 为逐位置通道映射，**无 bilinear、无 patchify 怪象**（对照昨晚 un-shuffle 的棋盘伪影）。

---

## 4. 在 XOPD L1 中的使用（Stage 2，`AlignedTransport`）

on-policy：student rollout 出状态 $x_S$，需 teacher 在该处的意见：
$$
x_S \xrightarrow{\;Q\;(\text{stu}\to\text{tea})\;} x_T\in\mathcal{M}_T
\;\xrightarrow{\ \text{teacher transition}\ } \mu_T
\;\xrightarrow{\;P\;(\text{tea}\to\text{stu})\;} \mu_S,
$$
student 匹配 $\mu_S$。FLUX 的 packed/transformer latent 与 raw 32ch 之间用其自带的 `_patchify/_unpatchify` + BatchNorm **无损**互转（teacher 查询仍在真 packed latent 上）。

理论命题（见 `xopd_vae_space_align.tex`）：若 $P$ 线性 + $Q$ on-manifold（$\mathcal{D}_T(Q\,x_S)\!\approx\!x$，误差 $\le\eta$），则 L1 转移均值传输偏差 $\le \mathrm{Lip}(v_T)\,\eta+\delta_{\text{align}}$，**与 $d_S<d_T$ 无关**——把不可约信息缺口替换为可训练压低的对齐残差。

---

## 5. 验证速查（看哪一对、验证什么）

| 对比 | 看什么 | 期望 |
|---|---|---|
| 行 1 vs 2 | $\mathcal{D}_S'(\mathcal{E}_S(x))$ vs $x$ | do-no-harm：几乎一致 |
| 行 1 vs 3 | $\mathcal{D}_S'(P\,\mathcal{E}_T(x))$ vs $x$ | 前向 $P$ 保真：锐利、无伪影 |
| 行 2 vs 4 | 同 $z_S$，student vs (Q 后) teacher 解码 | 逆向 $Q$ on-manifold：行 4 解出正确内容 |

定量：$\mathcal{L}_{\text{inv-px}}$（关键）显著下降 ⇒ $Q$ 已落流形；$\mathcal{L}_{\text{fwd-px}}$ 下降 ⇒ 前向高保真。
