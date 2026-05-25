# GenEval 数据集与 Reward 适配问题总结

本文档记录在 MoF/GRPO 训练中适配 GenEval 数据集和 reward model 过程中遇到的所有问题及解决方案。

---

## 问题 1: Arrow 序列化失败（异构 struct 字段）

### 现象

数据预处理时报错：
```
ArrowInvalid: cannot mix struct and non-struct, non-null values
```

### 根因

`dataset/geneval/train.jsonl` 中 `include` 字段是一个嵌套 struct 列表，但不同行的 struct schema 不一致：

- 部分行有 `position` 字段（如 `["above", 0]`），部分行没有
- 部分行有 `color` 字段，部分行为 null
- `exclude` 字段有时为空列表 `[]`，有时缺失

Arrow 在推断 nested struct 的 schema 时要求所有行结构一致，否则报错。

### 解决方案

**方案 A（最终采用）**：将 `include` 和 `exclude` 字段存储为 JSON 字符串（`json.dumps()`）。Arrow 只看到 `string` 类型，避免嵌套 struct 推断问题。GenEval reward model 在使用时再 `json.loads()` 解析。

**方案 B（早期尝试）**：补全所有 struct 使其一致（添加 null 的 `color`/`position`/`exclude`）。有效但脆弱——新增字段仍可能触发。

### 相关 commits

- `917adfd`: 修复异构 struct schema
- `30be5f3`: 补全缺失字段

---

## 问题 2: 数据集重复样本导致 group_size 溢出

### 现象

GRPO sampler 在按 `unique_id`（prompt hash）分组时，某些组超出配置的 `group_size=18`，触发 assertion error。

### 根因

原始 `train.jsonl` 有 50,000 行但仅 33,199 个 unique prompts（16,801 条重复）。相同 prompt 被分到同一组，导致组大小超出预期。

### 解决方案

对 JSONL 文件进行去重（基于 prompt 字段），保留 33,199 条唯一记录。

### 相关 commit

- `f26d7dd`: 去重数据集

---

## 问题 3: GRPO/MoF 的 reward model 要求 `include` 元数据，但 sample 中没有

### 现象

GenEval reward model 在评估生成的图片时报错：
```
GenEval reward requires 'include' metadata in sample but got None
```

### 根因

GRPO trainer 的 `sample()` 方法调用 `adapter.inference()` 生成图片后，生成的 `BaseSample` 对象中不包含原始 batch 的 metadata（如 `include`、`tag`、`__source__`）。Reward model 无法获取 GenEval 所需的评估配置。

### 解决方案

添加 `stitch_batch_metadata(batch, sample_batch)` 调用，将 dataloader batch 中的 metadata 字段注入到生成的 sample 的 `extra_kwargs` 中。

**函数位置**：`src/flow_factory/utils/base.py`

**工作原理**：
```python
def stitch_batch_metadata(batch: Dict, samples: List[BaseSample]):
    """将 batch 中的 metadata 字段注入到 samples 的 extra_kwargs 中。"""
    metadata_keys = ['include', 'exclude', 'tag', '__source__']
    for key in metadata_keys:
        if key in batch:
            for i, sample in enumerate(samples):
                sample.extra_kwargs[key] = batch[key][i]
```

**应用位置**：
- `trainers/grpo.py` — GRPOTrainer.sample() 和 GRPOGuardTrainer.sample()
- `trainers/mof/common.py` — MoFTrainerBase (inherited by NFT/GRPO)
- `trainers/abc.py` — BaseTrainer eval inference

---

## 问题 4: Eval 时 GenEval reward 返回 NaN

### 现象

训练时 GenEval reward 正常，但在 evaluation 阶段（test set）所有 GenEval reward 返回 NaN。

### 根因

Test dataloader 的 batch 缺少 `__source__` metadata。Reward model 的 `applicable_sources` 过滤机制检查 sample 的 `__source__` 字段，如果为 None 则跳过该 reward，返回 NaN。

### 解决方案

在 MoF trainer 的 `_run_eval_inference_batches()` 方法中，显式为 eval batch 添加 `__source__` tag：
```python
batch["__source__"] = [test_set_name] * batch_size
```

这确保 eval samples 带有正确的 source 标签，使 reward 的 `applicable_sources` 过滤能正确匹配。

---

## 问题 5: `applicable_sources` 机制与多数据集训练

### 背景

MoF 训练使用多个数据集（geneval, pickscore, ocr），每个 reward model 只适用于特定数据源：
- GenEval reward → 仅适用于 geneval 样本（需要 `include` 元数据）
- OCR reward → 仅适用于 ocr 样本（需要文字内容）
- PickScore reward → 适用于所有样本

### 配置方式

在 config 中为每个 reward 指定 `applicable_sources`：
```yaml
rewards:
  - name: "geneval"
    reward_model: "geneval"
    applicable_sources: [geneval]        # 只评估 geneval 样本
  - name: "pick_score"
    reward_model: "PickScore"
    applicable_sources: [geneval, pickscore, ocr]  # 评估所有样本
  - name: "ocr"
    reward_model: "ocr"
    applicable_sources: [ocr]            # 只评估 ocr 样本
```

### 实现

`reward_processor.py` 中，对于不匹配 `applicable_sources` 的 sample，reward 值设为 NaN。下游的 advantage computation 自动跳过 NaN 值。

---

## 问题 6: GenEval Reward Model 的 mmdet 配置兼容性

### 现象

GenEval reward 初始化时加载 object detection 模型失败。

### 根因

GenEval reward 使用 mmdetection 进行物体检测（验证生成图像中的物体数量和类别）。mmdet 3.x 版本更新了配置文件命名规范（如 `8xb2-lsj-50e`）。

### 解决方案

更新 GenEval reward 中 mmdet 模型配置路径和加载方式以兼容 3.x API。

---

## 总结：完整的适配链路

```
Dataset (JSONL)
  │  ① Arrow 序列化：include/exclude 存为 JSON string
  │  ② 去重：确保 unique prompts
  ▼
Dataloader (batch with metadata)
  │  ③ stitch_batch_metadata：注入 include/tag/__source__ 到 samples
  ▼
Reward Evaluation
  │  ④ applicable_sources 过滤：每个 reward 只评估对应源的 samples
  │  ⑤ GenEval reward：json.loads(include) → 物体检测 + 属性验证
  ▼
Advantage Computation
     ⑥ NaN-aware normalization：跳过不适用的 reward 值
```

每一步的正确性都依赖前一步的输出格式，形成一条严格的依赖链。任何环节断裂都会导致 reward 为 NaN 或训练报错。
