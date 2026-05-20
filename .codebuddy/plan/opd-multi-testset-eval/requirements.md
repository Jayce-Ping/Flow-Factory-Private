# 需求文档

## 引言

### 背景调研结论（重要）

多 test_set 评估能力**已在 `BaseTrainer` 中通用实现**，非 `ensemble-eval` trainer 独占：

1. `BaseTrainer.evaluate()`（[abc.py](src/flow_factory/trainers/abc.py) L479）的实现是
   `for test_set_name in sorted(self.test_dataloaders.keys()): self._evaluate_test_set(...)`，
   并且 `_evaluate_test_set` / `_run_eval_inference_batches` /
   `_eval_log_prefix` / `_log_eval_reward_metrics` 全部在基类完成。
2. `data_utils/loader.py` 在看到 `eval.test_sets` 配置（`Optional[List[TestSetArguments]]`）时
   会按列表逐条构建 `test_dataloaders[ts.name]`；只要 yaml 写了，**任何 trainer**
   （`opd` / `opd-ode` / `grpo` / `ensemble-eval` …）都自动多次评估。
3. `EvaluationArguments.merged_eval_args_for_test_set()` 已支持 per-test-set 覆盖
   `resolution / per_device_batch_size / guidance_scale / num_inference_steps / seed` 等字段；
   `TestSetArguments.eval_reward_names` 已支持「该 test_set 只跑哪些 reward」的子集投影。

**现状缺口（来自代码事实，非推测）：**

- `opd_configs/` 共 6 份 yaml（`group_pathwise_only` / `group_reinforce_only` /
  `group_reinforce_pathwise` / `pathwise_only` / `pathwise_reinforce` /
  `reinforce_only`）和 `opd-ode_configs/` 共 3 份 yaml
  （`pathwise_full` / `solver_checkpointing` / `tbptt_steps_3`）
  **均未声明 `eval.test_sets`**；当前 evaluation 隐式只跑
  `data.dataset_dir/test.{jsonl,txt}`（绝大多数为 `dataset/ocr`）。
- 这 9 份 yaml 的 `eval_rewards` 列表里**都缺 `geneval` reward**。
- 三套数据集本身已就位：`dataset/{geneval,ocr,pickscore}` 的 `test.{jsonl,txt}` 文件均存在，
  可直接被加载。
- 已被验证可工作的范本是 [ensemble-eval/lora/sd3_5/default.yaml](ensemble-eval/lora/sd3_5/default.yaml)，
  其 `eval.test_sets` + `eval_rewards`（含 `geneval` 且 `dtype: float32`）的写法是
  本次跨配置移植的标准模板。

### 目标

把上述 9 份 OPD/OPD-ODE 配置升级为「单次训练 → 三 test_set 自动评估
（geneval / ocr / pickscore）」的统一评估姿态，**无需改任何 Python 代码**。

## 需求

### 需求 1（已有能力的直接确认与采纳）

**用户故事：** 作为 Flow-Factory 的训练用户，我希望知道当前框架是否已具备多 test_set 评估能力，
以便决定要不要写新代码。

#### 验收标准

1. WHEN 用户阅读本需求文档 THEN 文档 SHALL 明确指出
   `BaseTrainer.evaluate()` 已遍历所有 `test_dataloaders.keys()`，
   多 test_set 评估为基类通用能力。
2. WHEN 任意 trainer（含 `opd` / `opd-ode`）的 yaml 提供了
   `eval.test_sets: [...]` 列表 THEN 系统 SHALL 自动按列表中的
   每个 entry 构建独立的测试 dataloader 并依次跑评估，
   wandb 指标分别落到 `eval/{ts.name}/...` 前缀下。
3. IF 用户的需求只是"让 OPD trainer 也多 test_set 评估" THEN
   方案 SHALL 仅修改 yaml 配置文件，**不引入任何 Python 代码改动**。

### 需求 2（OPD / OPD-ODE 配置补全多 test_set 评估）

**用户故事：** 作为 OPD 算法的训练者，我希望训练过程中能在 GenEval、OCR、PickScore
三个测试集上同时输出指标，以便横向比较算法在不同任务族上的表现。

#### 验收标准

1. WHEN 用户启动 `opd_configs/` 下任一 yaml 训练 THEN 该 yaml 的 `eval:`
   段 SHALL 包含一个 `test_sets:` 列表，依序声明 `ocr` / `pickscore` /
   `geneval` 三个 entry（顺序与 [ensemble-eval/lora/sd3_5/default.yaml](ensemble-eval/lora/sd3_5/default.yaml)
   保持一致以便对比）。
2. WHEN 用户启动 `opd-ode_configs/` 下任一 yaml 训练 THEN 同上规则同样适用。
3. WHEN 任一 test_set entry 被定义 THEN 它 SHALL 包含 `name` /
   `dataset_dir` / `split: test` / `eval_reward_names`，且 `eval_reward_names`
   仅引用本 yaml `eval_rewards` 列表中真实存在的 name。
4. IF test_set 是 `ocr` THEN `eval_reward_names` SHALL 为
   `[ocr, pick_score]`（与现有 ensemble-eval 模板一致）。
5. IF test_set 是 `pickscore` THEN `eval_reward_names` SHALL 为 `[pick_score]`。
6. IF test_set 是 `geneval` THEN `eval_reward_names` SHALL 为
   `[geneval, pick_score]`，**且** yaml 的 `eval_rewards` 列表 SHALL
   包含一个 `name: geneval` 的条目（reward_model: `geneval`，
   `dtype: float32`，附 fp32 强制说明注释，与 ensemble-eval default.yaml 一致）。

### 需求 3（向后兼容 / 不破坏现有训练 reward 配置）

**用户故事：** 作为已经在跑 OPD 训练的用户，我不希望本次配置升级影响训练侧的
loss 计算与现有 reward logging。

#### 验收标准

1. WHEN 配置升级完成 THEN 各 yaml 的训练侧 `train:` 段 SHALL 保持原样
   （teacher_paths / pathwise_coef / reinforce_coef / kl_beta / 学习率 / 调度
   等所有训练超参不动）。
2. WHEN 配置升级完成 THEN 各 yaml 的 `data.dataset_dir` SHALL 保持不变
   （训练采样仍走原数据集；多 test_set 仅影响 evaluation 阶段）。
3. WHEN 配置升级完成 THEN 各 yaml 的 `rewards:` 列表（训练时使用的 reward）
   SHALL 保持不变，**只在 `eval_rewards:` 中补 `geneval` 一项**。
4. WHEN 用户 grep 各 yaml 的 `trainer_type` THEN 其值 SHALL 与升级前完全一致
   （`opd` / `opd-ode` / `ensemble-eval` 不交叉污染）。

### 需求 4（文档与可发现性）

**用户故事：** 作为接手该仓库的新成员，我希望能从配置文件本身一眼看出
"为什么 geneval reward 的 dtype 是 float32"等关键信息，避免重复踩坑。

#### 验收标准

1. WHEN 任一 yaml 新增 `geneval` 的 `eval_rewards` 条目 THEN 该条目上方
   SHALL 附带与 ensemble-eval default.yaml 相同语义的注释，说明
   GenEval reward 内部强制 fp32（autocast disabled + model.float()），
   yaml 里的 `dtype` 仅为占位/防误会。
2. WHEN 任一 yaml 新增 `eval.test_sets` THEN 文件头注释 SHALL（在原有
   配置注释段末尾）追加一句话说明"evaluation 现在覆盖三个 test_set"，
   或在 `eval:` 段上方加一行行内注释，便于浏览者第一眼定位。

### 需求 5（健壮性 / 边界情况）

**用户故事：** 作为评估管线的运行者，我希望在某个测试集临时缺失或某个 reward
配置异常时，能立即得到清晰报错，而不是默默吃掉一个 test_set。

#### 验收标准

1. WHEN 任一 test_set 的 `dataset_dir` 在磁盘上不存在 `{split}.jsonl`/
   `{split}.txt` THEN 系统 SHALL 在 `data_utils/loader.py` 既有逻辑下
   抛出 `FileNotFoundError` 终止启动（已在 L320-323 实现，无需新增代码）。
2. WHEN 任一 test_set 的 `eval_reward_names` 引用了 `eval_rewards` 中不存在
   的 name THEN 系统 SHALL 在 `BaseTrainer._validate_eval_reward_names_for_test_sets`
   中抛错（已在 abc.py 实现，无需新增代码），本次需求只需在写 yaml 时
   做到"name 必须真实存在"即可。
3. IF 后续要新增第 4 个测试集（例如 `t2is`）THEN 用户 SHALL 仅需追加一个
   `test_sets[i]` 条目和必要的 `eval_rewards`，而无需改任何 trainer 代码。

### 需求 6（暂不做的事 / 范围澄清）

**用户故事：** 作为审稿人，我希望本次改动范围足够小、足够正交，便于回滚。

#### 验收标准

1. WHEN 本工作流结束 THEN 它 SHALL 不修改 `src/flow_factory/` 下任何 Python 文件。
2. WHEN 本工作流结束 THEN 它 SHALL 不引入新的 yaml 文件，
   **仅就地修改既有 9 份 yaml**。
3. WHEN 本工作流结束 THEN 它 SHALL 不调整训练阶段的 reward 列表
   （`rewards:`），不会影响 OPD/OPD-ODE 的 loss 数值。
