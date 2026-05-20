# 实施计划

> 本计划仅涉及 **yaml 配置就地修改**，不触碰 `src/flow_factory/` 下任何 Python 代码。
> 模板基线为 [ensemble-eval/lora/sd3_5/default.yaml](ensemble-eval/lora/sd3_5/default.yaml)
> 中已被验证可工作的 `eval.test_sets` + `eval_rewards`（含 `geneval` fp32）写法。

- [ ] 1. 抽取并固化"三 test_set + geneval reward"yaml 片段模板
   - 从 `ensemble-eval/lora/sd3_5/default.yaml` 摘取以下两段作为本任务的"标准片段"：
     (a) `eval.test_sets:` 列表 —— 含 `ocr` / `pickscore` / `geneval` 三个 entry，
         每个 entry 写明 `name` / `dataset_dir` / `split: test` / `eval_reward_names`；
     (b) `eval_rewards:` 列表中的 `geneval` 条目 —— `reward_model: geneval`，
         `dtype: float32`，并保留原文中关于"GenEval reward 内部强制 fp32
         （autocast disabled + model.float()），yaml 里 dtype 仅为占位/防误会"的注释
   - 校验 `eval_reward_names` 对应关系：`ocr → [ocr, pick_score]`、
     `pickscore → [pick_score]`、`geneval → [geneval, pick_score]`
   - _需求：2.3、2.4、2.5、2.6、4.1_

- [ ] 2. 升级 `opd_configs/group_pathwise_only.yaml`
   - 在 `eval:` 段尾追加任务 1 中的 `test_sets:` 片段
   - 在 `eval_rewards:` 列表追加 `geneval` 条目（含 fp32 注释），保持原有 `ocr` / `pick_score` 条目顺序与字段不动
   - 文件头注释末尾追加一行说明："evaluation 现覆盖 ocr / pickscore / geneval 三个 test_set"
   - 不改 `train:` / `data.dataset_dir` / `rewards:` / `trainer_type`
   - _需求：2.1、2.3、2.4、2.5、2.6、3.1、3.2、3.3、3.4、4.1、4.2_

- [ ] 3. 升级 `opd_configs/group_reinforce_only.yaml` 与 `opd_configs/group_reinforce_pathwise.yaml`
   - 对这 2 份 yaml 重复任务 2 的全部步骤（追加 `test_sets`、追加 `geneval` eval_reward、文件头注释、不动训练侧）
   - 注意 `group_reinforce_pathwise.yaml` 同时含 pathwise + REINFORCE，仍只动 eval 段
   - _需求：2.1、2.3、2.4、2.5、2.6、3.1、3.2、3.3、3.4、4.1、4.2_

- [ ] 4. 升级 `opd_configs/pathwise_only.yaml`、`opd_configs/pathwise_reinforce.yaml`、`opd_configs/reinforce_only.yaml`
   - 对这 3 份较长的 yaml 重复任务 2 的全部步骤
   - 这三份 yaml 的 `eval:` 段位于较靠后的行号（127 / 130 / 137），追加片段时需保持现有字段（`resolution` / `per_device_batch_size` / `guidance_scale` / `num_inference_steps` / `eval_freq` / `seed`）不动
   - _需求：2.1、2.3、2.4、2.5、2.6、3.1、3.2、3.3、3.4、4.1、4.2_

- [ ] 5. 升级 `opd-ode_configs/pathwise_full.yaml`
   - 重复任务 2 的全部步骤；该文件 `trainer_type` 为 `opd-ode`，验证它在新增 `eval.test_sets` 后**仍保持 `opd-ode`** 不被改动（需求 3.4）
   - _需求：2.2、2.3、2.4、2.5、2.6、3.1、3.2、3.3、3.4、4.1、4.2_

- [ ] 6. 升级 `opd-ode_configs/solver_checkpointing.yaml` 与 `opd-ode_configs/tbptt_steps_3.yaml`
   - 对这 2 份 OPD-ODE 消融配置重复任务 2 的全部步骤
   - _需求：2.2、2.3、2.4、2.5、2.6、3.1、3.2、3.3、3.4、4.1、4.2_

- [ ] 7. 配置健全性自检（静态校验，无需启动训练）
   - 对 9 份 yaml 逐一执行：
     (a) `python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"` 通过 —— YAML 语法合法；
     (b) grep 校验：每份 yaml 都恰好出现一次 `test_sets:`、三处 `eval_reward_names:`、
         `eval_rewards:` 列表中恰好出现一次 `name: "geneval"`；
     (c) grep 校验：每份 yaml 的 `trainer_type` 与升级前完全一致（用 `git diff` 确认 train 段零改动）；
     (d) 确认 `dataset/{geneval,ocr,pickscore}/test.{jsonl,txt}` 存在（命中需求 5.1 的前置条件）
   - _需求：3.1、3.2、3.3、3.4、5.1、5.2_

- [ ] 8. 端到端冒烟（任选一份 OPD yaml，最小步数运行 evaluate）
   - 选 `opd_configs/group_pathwise_only.yaml`（最短、训练超参最轻）作为冒烟入口，
     用极小 `eval.per_device_batch_size` + 单步评估方式触发一次 `evaluate()` 路径
   - 校验 `train.log` 出现以下 wandb key 前缀：
     `eval/ocr/...`、`eval/pickscore/...`、`eval/geneval/...`，
     且 `eval/geneval/reward_geneval/{tag}_mean` 正常（沿用上一工作流的 per-tag scoping 修复）
   - 校验**未**出现 `FileNotFoundError` 与 `eval_reward_names` 校验报错
   - _需求：1.2、2.1、2.6、5.1、5.2_

- [ ] 9. 提交并 push
   - `git add` 仅限 9 份 yaml + 本规划目录；不要把任何 `src/flow_factory/` 文件混入
   - 提交信息使用约定式格式，例：
     `feat(opd): enable multi test_set eval (geneval/ocr/pickscore) in opd & opd-ode configs`
   - push 到当前工作分支
   - _需求：6.1、6.2、6.3_
