# DanceOPD 算法原理与 Flow-Factory 方法对比

> **论文**: [DanceOPD: On-Policy Generative Field Distillation](https://arxiv.org/abs/2606.27377) (arXiv:2606.27377, 2026-06-25)  
> **项目页**: https://danceopd.github.io/  
> **作者**: Wei Zhou et al. (ByteDance Seed, NUS, UMD, HKUST)

---

## 1. 背景与动机

现代图像生成模型需要在一个部署单元内同时支持多种能力：文生图 (T2I)、局部编辑、全局编辑、风格化等。这些能力往往**互相冲突**——编辑能力会侵蚀 T2I 质量，局部编辑与全局编辑对"保留 vs 变换"的偏好相反。

现有组合范式各有局限：

| 范式 | 问题 |
|------|------|
| 数据混合 / 联合训练 | 能力特异性监督被稀释，多任务梯度冲突 |
| 参数空间合并 (weight merge) | 折中解，难以保留各能力峰值 |
| 推理时 score 组合 (CFG 等) | 组合留在推理侧，未内化到 student |

DanceOPD 将问题重新表述为 **on-policy generative field distillation**：每个冻结的能力源定义共享 flow state space 上的一个 velocity field，student 在自己的 rollout 状态上查询并匹配这些 field，从而组合多种能力。

---

## 2. 核心算法

### 2.1 问题形式化

给定 $M \geq 2$ 个冻结能力源 $\{v_m\}_{m=1}^M$，每个源在共享 state space 上定义 velocity field：

$$v_m(z_t, t, c), \quad m \in \{1, \ldots, M\}$$

其中 $z_t$ 为 flow state，$c$ 为条件（文本、源图、编辑指令等）。Student $v_\theta$ 通过 field alignment 学习组合这些能力。

训练涉及三个耦合的设计选择：

1. **哪个 field 监督当前样本**（target field）
2. **在 state space 的哪个位置查询 field**（query state）
3. **从 student rollout 取多少个状态做监督**（trajectory supervision density）

### 2.2 三个设计原则

DanceOPD 针对三个 alignment 挑战给出对应设计：

| 挑战 | 诊断 | DanceOPD 设计 |
|------|------|---------------|
| **Target-field ambiguity** | 同一样本内混合多个 field 的目标不再对应任何明确定义的能力查询 | **Hard-routed sample-wise matching**：$m \sim \pi(m)$，$(x,c) \sim \mathcal{D}_m$，每样本只查一个 field |
| **State-distribution mismatch** | 在 data state 或 teacher trajectory 上查询，student 部署时访问的状态分布不同 | **On-policy querying**：在 student rollout 的 stop-gradient 状态 $\bar{z}_t = \mathrm{sg}(z_t^\theta)$ 上查询 |
| **Trajectory-query correlation** | 同一条 rollout 上多个状态高度相关（共享 noise seed、prompt、path history），密集监督信号不独立 | **Semantic-side single query**：$K=1$，从偏向低噪声端的分布 $q_{\mathrm{sem}}(s)$ 采样一个语义侧状态 |

### 2.3 训练步骤（Algorithm 1）

```
A. Route:     m ~ π(m),  (x, c) ~ D_m
B. Rollout:   z_{0:T}^θ ← Rollout(v_θ; z_T, c),  z_T ~ p_T
              s ~ q_sem(s),  t ← t(s),  z̄_t ← sg(z_t^θ)
              u ← v_m(z̄_t, t, c)          # frozen routed field
C. Update:    L = ||v_θ(z̄_t, t, c) - u||²
              θ ← OptStep(θ, ∇_θ L)
```

**关键细节**：

- **Hard routing**：$\pi$ 默认在活跃能力桶上均匀（两桶 1:1，三桶 1:1:1），不做数据比例调参
- **Stop-gradient query**：梯度只通过 $v_\theta(\bar{z}_t, t, c)$ 回传，不穿过整个 rollout solver
- **Semantic-side low-$t$ query**：$s \sim \mathrm{Beta}(5, 2)$（低噪声端），能力特异性信息（风格、编辑属性）集中在轨迹末端
- **Plain velocity MSE**：目标为确定性 velocity field，MSE 是局部 Gaussian transition KL 的自然形式（附录 7.1 推导）

### 2.4 目标函数

$$\mathcal{L}_{\mathrm{DanceOPD}} = \mathbb{E}_{m \sim \pi,\, (x,c) \sim \mathcal{D}_m,\, z_T \sim p_T,\, s \sim q_{\mathrm{sem}}} \left[ \left\| v_\theta(\bar{z}_t, t, c) - v_m(\bar{z}_t, t, c) \right\|_2^2 \right], \quad t = t(s)$$

在共享协方差 $\sigma_t^2 I$ 的局部 Gaussian transition 视角下，KL 匹配退化为 timestep-weighted velocity MSE：

$$D_{\mathrm{KL}}(p_m \| p_\theta) = \frac{\Delta t^2}{2\sigma_t^2} \|v_\theta - v_m\|_2^2$$

论文实验表明 plain MSE 比 KL-$\bar{\sigma}^2$ weighting、Min-SNR weighting、consistency、DMD 等变体更稳定。

### 2.5 扩展：Operator-defined Fields

**CFG absorption**：将 CFG 视为额外 capability field

$$v_\alpha(z_t, t, c) = v_\emptyset(z_t, t) + \alpha \bigl(v_{\mathrm{cond}}(z_t, t, c) - v_\emptyset(z_t, t)\bigr)$$

用相同 MSE 目标吸收进 student。注意 train-time 吸收 scale $\alpha$ 与 inference-time CFG scale $\beta$ 会**乘法复合**（有效 scale $\approx \alpha\beta$）。

**Realism-field absorption**：将 realism-oriented teacher 视为 quality field，在保持 anchor T2I 能力的同时吸收视觉统计。

### 2.6 计算复杂度

每样本仅需 **1 次 student rollout + 1 次 teacher forward + 1 次 student forward**，远低于密集 per-timestep 监督。论文 Fig. 1 右图以 marker size 标注 per-step training cost，DanceOPD 显著低于 DiffusionOPD / Flow-OPD 等密集基线。

---

## 3. 实验结论摘要

### 3.1 主结果

| 设置 | DanceOPD vs 最佳基线 |
|------|---------------------|
| T2I + Edit 组合 | GEditBench +8.1%（vs 最佳 OPD 基线），GenEval 略超 T2I source |
| Local + Global Edit 组合 | GEditBench +16.1%（vs 最佳组合基线） |
| Realism-field absorption | Realism reward +9.9%（vs off-policy distill），T2I 保持 within 0.1% |
| CFG absorption | GEditBench +7.6%（vs train-only absorption） |

### 3.2 关键消融

| 消融维度 | 结论 |
|----------|------|
| Hard routing vs soft all-teacher mixing | Hard routing +15.2%（MSE）/ +10.6%（KL） |
| Low-$t$ vs median/high-$t$ query | Low-$t$ +23.7% / +19.5% |
| Single query ($K{=}1$) vs dense ($K{=}2,4,8,16$) | Single query 全面优于 dense |
| Same-step multi-teacher accumulation ($G{=}3$) | 平均 -4.6%；加 dense 后 -22.8% |
| SDE decorrelation on dense stress case | 部分恢复 +18.4%，但仍 -8.6% vs single-query default |
| Plain MSE vs alternatives | MSE 最优，+2.8% ~ +4.5% over weighted variants |
| Rollout steps | 16-step 足够，不必与 eval sampler 步数完全匹配（ODE solver） |

---

## 4. 与 Flow-Factory 方法的对比

### 4.1 方法全景

| 维度 | **DanceOPD** | **OPDTrainer** | **MOF → Distill** | **XOPD** | **DiffusionOPD** |
|------|-------------|----------------|-------------------|----------|------------------|
| **Registry key** | — (外部) | `opd` | `mof` + distill stage | `xopd` | `diffusion_opd` |
| **Teacher 形态** | 完整冻结模型（不同能力） | 同架构 LoRA snapshots | 同架构 LoRA + learned router | 跨模型 frozen transformer | 同架构 LoRA |
| **Teacher 路由** | Hard sample-wise ($m \sim \pi$) | `route_by_source` 或 soft aggregation | Stage 1 RL 学连续 $\lambda_k(t,s,c)$ | 单一 cross-model teacher | Per-teacher isolation |
| **Query 状态** | 1 个 semantic-side low-$t$ 状态 | 全部 $T$ 个 timestep | Stage 2: 全部 $T$ | L0: 随机 $t$；L1: 全部 $T$ | 全部 $T$（interleaved） |
| **On-policy** | 是（sg student rollout state） | 是（epoch 初 rollout 存轨迹） | 是 | L0: 否（teacher rollout）；L1: 是 | 是（interleaved rollout） |
| **目标函数** | Plain velocity MSE | Per-step KL $D_k$ + optional REINFORCE | MSE to MoF-mixed $v^{\mathrm{MoF}^*}$ | L0: weighted MSE；L1: KL + REINFORCE | KL or MSE per step |
| **动力学** | ODE | SDE / ODE | ODE 为主 | SDE / ODE | ODE |
| **CFG 处理** | CFG 作为 operator field 吸收 | `teacher_guidance_scale` | 通过 MoF router | Dual CFG (teacher/student) | 同 OPD |
| **多能力组合** | 核心场景 | Multi-teacher LoRA 组合 | 核心场景（两阶段） | 跨模型压缩 | Multi-teacher |
| **每样本计算** | 1 rollout + 2 forward | 1 rollout + $T \times$ (pre-pass + main-pass) | 同 OPD 量级 | L0 + L1 两阶段 | $T \times$ interleaved |

### 4.2 与 OPDTrainer 的详细对比

**相同点**：

- 均在 student 自己产生的状态上监督（on-policy）
- `route_by_source=true` 时与 DanceOPD 的 hard routing 哲学一致：每个 data source 对应一个 in-domain teacher，保持样本级语义身份
- 局部 Gaussian transition 视角下，pathwise KL $D_k$ 与 velocity MSE 在 ODE / $\sigma^2{=}1$ 约定下等价

**不同点**：

| 方面 | OPDTrainer | DanceOPD |
|------|-----------|----------|
| 监督密度 | 密集：每个 training timestep 都算 $D_k$ | 稀疏：每 rollout 仅 1 个 low-$t$ query |
| REINFORCE | 支持 $R_{\bar{k+1}} \cdot \log p_\theta$，用未来 $D_j$ 作 dense reward | 无 policy gradient，纯 regression |
| Teacher aggregation | 5 种模式（round_robin, average, sum, pcgrad, v_pcgrad） | 仅 hard routing；论文证明 soft mixing 劣于 hard routing |
| SDE 支持 | 完整 Flow-SDE + log_prob | 仅 ODE 为主；SDE 仅作 dense-query 去相关诊断 |
| 计算成本 | 高（pre-pass + main-pass × $T$ timesteps × inner epochs） | 低（1 query / sample） |
| `normalize_d_k` | 可配置 $D_k / (2\sigma_{\bar{k}}^2)$ | 默认 unweighted MSE（等价于 `normalize_d_k=false`） |

**理论联系**（参见 [`mof_vs_direct_opd_gap.tex`](mof_vs_direct_opd_gap.tex)）：

- OPD `route_by_source` = 本文 Paradigm B（直接 multi-teacher OPD）
- DanceOPD 的 hard routing 与 Paradigm B 在 target 构造上一致：$\bar{v} - v_y = \sum_{m \neq y} w_m (v_m - v_y)$ 的 bias 在 hard routing 下为零
- OPD 的 `average` / `sum` aggregation 更接近 DanceOPD 批评的 soft all-teacher mixing

### 4.3 与 MOF → Distill 的详细对比

**Paradigm A**（[`mof_vs_direct_opd_gap.tex`](mof_vs_direct_opd_gap.tex)）：

$$\text{Stage 1: } \lambda_k^* = \arg\max_\psi \mathbb{E}[R(\text{rollout}(v^{\mathrm{MoF}}(\psi)))]$$
$$\text{Stage 2: } v^{\mathrm{MoF}^*}(x,t,c) = \sum_k \lambda_k^*(t,s,c)\, v^{(k)}(x,t,c) \;\Rightarrow\; \text{MSE distill to } v^\theta$$

| 方面 | MOF → Distill | DanceOPD |
|------|--------------|----------|
| Teacher 组合 | **Soft**：连续可学习权重 $\lambda_k(t,s,c)$ | **Hard**：离散 per-sample routing |
| 搜索空间 | $\mathcal{T}_A \supsetneq \mathcal{T}_B$（A 严格泛化 B） | 固定 routing，无 learned router |
| 时间维度 | $\lambda_k$ 可随 $t$ 变化（时间互补性） | 固定 $K{=}1$，$t$ 从 $q_{\mathrm{sem}}$ 采样 |
| 训练阶段 | 两阶段（RL + distill） | 单阶段 |
| 与 DanceOPD 诊断的关系 | Stage 1 soft mix 类似 DanceOPD 批评的 target-field ambiguity；Stage 2 dense distill 类似 trajectory-query correlation 风险 | 明确反对 soft mix 和 dense query |

**启示**：MOF 的 Paradigm A 在理论上 target 空间更大，但 DanceOPD 的实验表明 soft mixing 和 dense supervision 在实践中引入显著退化。若 MOF Stage 2 采用 DanceOPD 风格的 sparse single-query + hard routing，可能降低计算成本并改善稳定性——这是值得探索的交叉点。

### 4.4 与 XOPD 的详细对比

XOPD（[`docs/opd/cross_opd_xopd.md`](../opd/cross_opd_xopd.md)）面向**跨模型压缩**（9B → 4B），与 DanceOPD 的**多能力组合**场景不同，但 L0/L1 两阶段设计与 DanceOPD 有结构性对比：

| 方面 | XOPD | DanceOPD |
|------|------|----------|
| 场景 | 大模型 → 小模型（同 family，共享 VAE/encoder） | 多能力 field 组合（同 backbone family） |
| L0 warmup | Off-policy：teacher rollout 生成 $z_0$，随机 $t$ 上 velocity MSE | 无 off-policy warmup；始终 on-policy |
| L1 | 密集 per-step KL + REINFORCE（同 OPD） | 稀疏 single-query MSE |
| Teacher 数量 | 1 个 cross-model teacher | $M$ 个 capability fields |
| CFG | Dual CFG（teacher_gs / student_gs） | CFG 作为 operator field 吸收 |

XOPD L0 的 off-policy velocity regression 在 DanceOPD 框架下属于 "state-distribution mismatch" 风险点，但 XOPD 用它解决跨模型初始化问题（student 初始 velocity field 与 teacher 差距大），是合理的 warmup 策略而非最终训练范式。

### 4.5 与 DiffusionOPD 的详细对比

DiffusionOPD 是 DanceOPD 论文中最接近的 Flow-Factory 基线：

| 方面 | DiffusionOPD | DanceOPD |
|------|-------------|----------|
| Rollout | On-policy interleaved（optimize 中在线 rollout） | On-policy，先 rollout 再 query |
| 监督密度 | Dense per-timestep | Single semantic-side query |
| 目标 | KL or MSE per step | Plain MSE at 1 state |
| 论文实验 | DanceOPD 在 T2I+Edit 组合上 GEditBench 全面优于 DiffusionOPD | — |

两者核心差异是 **query density**：DiffusionOPD 与我们的 OPDTrainer 一样做密集监督，DanceOPD 认为这引入 trajectory-query correlation 且计算昂贵。

---

## 5. 设计空间对照图

```
                    Teacher 组合方式
                    ─────────────────────────────────────────
                    Hard route          Soft mix / learned λ
                    (per-sample)        (per-step aggregation)
                         │                      │
    Query 密度           │                      │
    ──────────           │                      │
    K=1 single query  ───┼── DanceOPD ────────┼── (DanceOPD 反对)
    K=T dense query   ───┼── DiffusionOPD ────┼── OPDTrainer
                         │   OPDTrainer         │   MOF Stage 2
                         │   XOPD L1            │
                         │                      │
    K=random t        ───┼── XOPD L0            │
    (off-policy)         │   (warmup only)      │
```

---

## 6. 对 Flow-Factory 的启示与潜在改进方向

### 6.1 已验证的设计选择

1. **`route_by_source=true`** 与 DanceOPD hard routing 一致，是其多能力组合成功的关键之一。应继续作为 multi-teacher OPD 的默认推荐。
2. **`normalize_d_k=false`** 与 DanceOPD plain MSE 默认一致。论文消融支持 unweighted MSE 更稳定。
3. **On-policy 监督** 是 OPD / XOPD L1 / MOF Stage 2 的共同基础，DanceOPD 从理论上进一步论证了 on-policy query 的必要性（Lipschitz mismatch bound, 附录 7.2）。

### 6.2 值得实验探索的方向

| 方向 | 描述 | 预期收益 |
|------|------|----------|
| **Sparse-query OPD** | 在 OPDTrainer 中增加 `query_mode: single_semantic` 选项，每 rollout 只在 low-$t$ 状态算一次 $D_k$ | 大幅降低计算成本（~$T\times$），DanceOPD 表明性能可能不降反升 |
| **DanceOPD-style L1 for XOPD** | XOPD L1 从密集 KL 改为 single-query MSE | 简化跨模型蒸馏，降低 9B teacher forward 次数 |
| **CFG field absorption** | 用 DanceOPD 方式将 `teacher_guidance_scale > 1` 的 CFG field 吸收进 student | 已有 `teacher_guidance_scale` 基础设施，可系统化评估 |
| **MOF Stage 2 sparse distill** | Stage 2 从 dense per-$t$ MSE 改为 $K{=}1$ semantic-side query | 降低 Stage 2 成本，避免 trajectory correlation |
| **Same-step accumulation 警告** | 当 OPD `teacher_aggregation=sum` 且 multi-source batch 时，等效于 DanceOPD 批评的 same-step multi-teacher accumulation | 文档化风险，考虑强制 step alternation |

### 6.3 我们方法的独特优势（DanceOPD 未覆盖）

1. **REINFORCE 路径**：OPD 的 $R_{\bar{k+1}} \cdot \log p_\theta$ 提供 trajectory-level credit assignment，DanceOPD 纯 regression 无此机制。在 SDE 动力学下，REINFORCE 可补偿 sparse query 的信息损失。
2. **SDE 动力学**：Flow-SDE 支持随机 rollout 和 log_prob，DanceOPD 以 ODE 为主。
3. **Teacher aggregation 灵活性**：PCGrad / v_pcgrad 等梯度手术方法处理 field conflict，DanceOPD 用 hard routing 回避而非解决冲突。
4. **跨模型蒸馏 (XOPD)**：DanceOPD 假设同 backbone family 的兼容 field；XOPD 处理不同参数量模型的压缩。
5. **Learned routing (MOF)**：DanceOPD 用预定义 bucket routing；MOF Stage 1 可学习 time-varying $\lambda_k$，在 time-complementarity 场景下理论上更优（参见 `mof_vs_direct_opd_gap.tex` 边界情形分析）。

---

## 7. 参考文献对照

DanceOPD 论文 Table 1 将相关工作分类如下（摘录与我们的映射）：

| 方法 | Domain | Teacher Signal | Objective | FM-OPD | Multi-Cap | Func. Absorp. | Flow-Factory 对应 |
|------|--------|---------------|-----------|--------|-----------|---------------|-------------------|
| DiffusionOPD [53] | Flow | velocity field | KL/MSE | ✓ | ○ | — | `diffusion_opd` trainer |
| Flow-OPD [24] | Flow | dense scalar reward | PPO clip-min | ✓ | task-routed | — | 部分思想在 OPD REINFORCE 中 |
| **DanceOPD** | Flow | routed velocity field | MSE | ✓ | ✓ | ✓ (CFG, realism) | 本文档 |
| — | Flow | LoRA snapshots | KL + REINFORCE | ✓ | ✓ (route/agg) | CFG via guidance_scale | `opd` trainer |
| — | Flow | MoF-mixed field | MSE | ✓ | ✓ (2-stage) | — | `mof` trainer |
| — | Flow | cross-model transformer | L0 MSE + L1 KL | ✓ | — | dual CFG | `xopd` trainer |

---

## 8. 总结

DanceOPD 将多能力图像生成重新表述为 **on-policy generative field distillation**，通过三个简洁设计（hard routing、student-induced query state、single semantic-side low-$t$ query）+ plain velocity MSE，在多种能力组合和 field absorption 场景上优于密集监督的 OPD 基线。

与 Flow-Factory 的关系：

- **OPD `route_by_source`** 与 DanceOPD hard routing 高度一致，是我们的优势对齐点
- **OPD 密集 per-timestep 监督** 是 DanceOPD 明确反对的设计，带来计算成本和 correlation 风险
- **MOF soft mixing** 对应 DanceOPD 批评的 target-field ambiguity，但 MOF 的 learned time-varying weights 提供了 DanceOPD 未探索的搜索维度
- **XOPD** 解决不同问题（跨模型压缩），其 L0 off-policy warmup 是 DanceOPD 框架外的合理工程选择
- 最值得借鉴的是 **sparse single-query** 范式，可作为 OPD / XOPD / MOF Stage 2 的低成本替代模式
