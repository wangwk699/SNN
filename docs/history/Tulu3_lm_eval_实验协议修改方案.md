# Tulu-3 / lm-eval 实验协议与结果路径修改方案

> 目标仓库：`/home/wangwenkang/SNN`  
> GitHub：`https://github.com/wangwk699/SNN`  
> 基准分支：当前 `main`  
> 本文档用于部署在服务器上的 Codex 在**没有任何先前对话上下文**的情况下完成本轮代码修改。  
> **以当前仓库实际代码为准实施；不要只修改 YAML 表面字段。所有配置、数据采样、实际执行、artifact path、metadata/provenance、验证脚本、测试和 `实验执行总结.md` 必须同步。**

---

# 1. 本轮修改目标

本轮只针对 **Tulu-3 / lm-eval 实验协议**做系统性修改，同时保持 TL;DR 现有实验语义和历史输出路径不变。

核心目标：

1. Tulu-3 ANN fine-tuning 不再默认使用几乎完整的 Tulu-3 train split，而是像 TL;DR 一样支持固定训练子集：
   - `training.train_samples`
   - `training.train_seed`
   - 默认 `train_samples: 10000`
   - 默认 `train_seed: 42`
   - 从 Tulu-3 可训练池中按 `train_seed` **随机、无放回**采样。
2. Tulu-3：
   - `gradient_accumulation_steps: 16`
   - `num_train_epochs: 2`
   - 其余未明确要求改变的训练超参数保持当前值，包括当前 `lr_scheduler_type: cosine`、`warmup_ratio: 0.0`。
3. lm-eval 六个 benchmark 增加：
   - `enabled`
   - `test_samples`
   - `test_seed`
4. `cot` 不尝试传入 lm-eval，因为固定的 lm-eval 0.4.8 `simple_evaluate()` 没有通用 `cot` 参数。
   - `cot` 改为**项目侧对固定 lm-eval task 真实 CoT 语义的强校验字段**。
   - 配置值和固定 lm-eval task 实际语义不一致时，必须在评估开始前直接报错。
5. lm-eval 每个 benchmark 独立保存结果。
6. evaluation 路径中**删除 `lm_harness/` 这一层**。
7. 每个 benchmark 的 evaluation 子目录必须显式包含：
   - `num_fewshot`
   - `cot`
   - `test_samples`
   - `test_seed`
8. Tulu-3 的训练 run path 在 TL;DR 现有路径设计基础上扩展，同时显式体现：
   - `lr_scheduler_type`
   - `warmup_ratio`
9. `实验执行总结.md` 第一部分所有 Tulu-3 命令及路径说明必须同步为新协议。

---

# 2. 当前代码的重要现状

## 2.1 当前 Tulu-3 数据逻辑

当前 `snn2/data.py::_manifest_split_selection()` 对 `task == "tulu3"` 的逻辑大致是：

1. 从 `data.train_split` 取得源 train split；
2. 用 `experiment.seed` 创建 permutation；
3. 当前 `validation_size: 1000`；
4. 当前 `data.train_size: null` 时：
   - permutation 前 1000 条作为 validation；
   - 其余所有行作为 train。

因此当前 Tulu-3 实际 ANN training 近似使用完整 Tulu-3 数据，而不是固定 10k 子集。

本轮需要删除 `data.train_size` 作为 Tulu-3 训练规模控制入口，改为 `training.train_samples/train_seed`。

## 2.2 当前 TL;DR 已有训练子采样逻辑

当前 `snn2/data.py` 已有 `_tldr_train_selection()`：

- `training.tldr_train_samples`
- `training.tldr_train_seed`
- `random.Random(seed).sample(...)`
- 无放回
- 对选择后的 indices 排序，以保证执行顺序稳定
- `null` 表示 full split

Tulu-3 新逻辑应尽量复用这一设计原则，但不能机械复制，因为 Tulu-3 需要先保留固定 validation holdout。

## 2.3 当前 lm-eval 调用

当前 `scripts/evaluate_lm_harness.py` 会遍历：

```python
task_specs = cfg["evaluation"]["lm_eval_task_specs"]
```

并对每个 spec 调用类似：

```python
simple_evaluate(
    model=harness_model,
    tasks=[name],
    num_fewshot=int(spec["num_fewshot"]),
    batch_size=batch_size,
    limit=cfg["evaluation"].get("limit"),
    random_seed=...,
    numpy_random_seed=...,
    torch_random_seed=...,
    fewshot_random_seed=...,
    apply_chat_template=...,
)
```

当前存在以下问题：

- 没有 `enabled`；
- 没有 per-task `test_samples/test_seed`；
- `cot` 完全没有被执行代码读取；
- 当前 `evaluation.limit` 是全局字段，而且 lm-eval `limit` 不是 seeded random sampling；
- 所有 benchmark 最终写入同一个 `results.json`；
- 输出路径中存在 `evaluation/lm_harness/...`。

这些都需要本轮一起修正。

---

# 3. 最终配置协议

## 3.1 Tulu-3 `training`

在 `configs/experiment_matrix.yaml` 的 Tulu-3 experiment 中：

### 删除

```yaml
data:
  train_size: null
```

Tulu-3 不再允许 `data.train_size` 和 `training.train_samples` 两套规模控制语义并存。

### 新增 / 修改

```yaml
training:
  train_samples: 10000
  train_seed: 42

  num_train_epochs: 2
  gradient_accumulation_steps: 16

  # 以下当前值继续保留，不因本轮修改而改变：
  lr_scheduler_type: cosine
  warmup_ratio: 0.0
```

### 语义

- `training.train_samples: 10000`
  - 表示最终 ANN training 使用 10000 条 Tulu-3 样本。
- `training.train_seed: 42`
  - 只负责从 Tulu-3 可训练池中选择 ANN training 子集。
- 建议继续支持：
  - `train_samples: null` → 使用完整可训练池；
  - 正整数 → seeded random without replacement。
- 非法：
  - `0`
  - 负数
  - bool
  - 超过可训练池大小。

TL;DR **不重命名**现有：
- `tldr_train_samples`
- `tldr_train_seed`

不要借本轮修改顺手统一字段名，否则会无必要扩大改动范围并破坏现有 TL;DR 配置/路径。

---

# 4. Tulu-3 数据切分与训练采样的精确定义

这是本轮最重要的数据语义之一。

## 4.1 必须先产生 validation，再产生 training subset

Tulu-3 源数据只有一个 train split。目标逻辑必须为：

### Stage 1：固定 validation holdout

继续保留：

```yaml
data:
  validation_size: 1000
```

用当前已有的 validation seed 语义，即 `experiment.seed`：

```python
validation_rng = random.Random(int(cfg["experiment"]["seed"]))
permutation = list(range(len(raw_train)))
validation_rng.shuffle(permutation)

validation_indices = permutation[:validation_size]
training_pool_indices = permutation[validation_size:]
```

即：

- validation 使用当前项目原有的 `experiment.seed`；
- validation 大小固定为 1000；
- training pool 是排除 validation 后的剩余数据。

### Stage 2：training 子采样

然后在 `training_pool_indices` 内按：

```python
training.train_seed
training.train_samples
```

无放回随机采样。

建议实现独立 helper，例如：

```python
_tulu3_train_selection(
    training_pool_indices,
    cfg,
) -> tuple[list[int], str]
```

逻辑：

```text
train_samples == null
    -> 使用整个 training_pool
    -> sampling = "full_training_pool"

train_samples == len(training_pool)
    -> 使用整个 training_pool
    -> sampling = "full_training_pool"

0 < train_samples < len(training_pool)
    -> Random(train_seed).sample(training_pool, k=train_samples)
    -> 对最终 selected indices 排序
    -> sampling = "seeded_random_without_replacement"
```

### 必须保证

```text
train_indices ∩ validation_indices == ∅
```

并写单元测试。

---

# 5. Tulu-3 training manifest / provenance

修改 `snn2/data.py` 时同步更新 train manifest。

Tulu-3 train manifest 至少应记录：

```json
{
  "split": "train",
  "sampling": "seeded_random_without_replacement",
  "train_samples": 10000,
  "train_seed": 42,
  "validation_size": 1000,
  "indices": [...],
  "record_ids": [...],
  "selection_scope": "current_ann_training_config"
}
```

命名可根据项目现有 schema 调整，但必须能从 manifest 唯一知道：

- Tulu-3 数据集 revision；
- validation 如何产生；
- training 使用多少条；
- training seed；
- sampling 是否无放回；
- 实际 indices / record IDs。

## 5.1 Calibration 与 training subset 的关系

Stage-A calibration 继续从**最终 selected training set**中选择：

```text
Tulu source train
  -> remove validation
  -> training.train_samples/train_seed
  -> selected ANN train
  -> calibration.num_samples/calibration.seed
```

即 calibration examples 必须属于最终 10k ANN training subset，并继续：

```json
"retained_in_training": true
```

不要让 calibration 又退回整个 training pool。

## 5.2 canonical preprocessing calibration

除非当前实现因为上述改动必须适配，否则**不要改变**已有 canonical preprocessing calibration 的独立协议。该部分不是本轮实验设计目标。

---

# 6. Tulu-3 实际 trainer 必须使用选中的 10k

检查：

- `snn2/training.py`
- `load_selected_raw(...)`
- `DatasetBundle.train`
- SFT Trainer 实际传入的数据

当前 TL;DR 有 `use_configured_train_subset=True` 的特殊分支；Tulu-3 本轮修改后必须确保 trainer 真实拿到 manifest/当前配置对应的 10000 条，而不是仍然把完整 Tulu training pool 交给 trainer。

验收必须检查：

```python
len(bundle.train) == 10000
```

在默认 Tulu-3 配置下成立。

训练日志/metadata 中的 `train_samples` 也必须报告实际选择数。

---

# 7. lm-eval task specs 最终格式

Tulu-3 的 `evaluation.lm_eval_task_specs` 改成以下协议。

默认全部启用：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8

  lm_eval_revision: 6d2abda4fd171e68a8789330c4149e37c1ca0bda
  apply_chat_template: true

  lm_eval_task_specs:
  - name: truthfulqa_mc1
    enabled: true
    num_fewshot: 0
    metric: acc_mc1
    cot: false
    test_samples: null
    test_seed: 42

  - name: mmlu_pro
    enabled: true
    num_fewshot: 5
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: bbh
    enabled: true
    num_fewshot: 3
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: agieval
    enabled: true
    num_fewshot: 0
    metric: acc
    cot: false
    test_samples: null
    test_seed: 42

  - name: gsm8k_cot
    enabled: true
    num_fewshot: 4
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: minerva_math
    enabled: true
    num_fewshot: 4
    metric: exact_match
    cot: false
    test_samples: null
    test_seed: 42
```

注意：

- GSM8K-CoT 从当前 8-shot 改为 **4-shot**。
- 其余 `num_fewshot/cot` 按本轮最终确认值设置。
- `test_samples: null` = 完整 evaluation split。
- `test_seed: 42` = 当未来设置整数 `test_samples` 时控制随机测试子集选择。
- `enabled: false` 时该 task 完全跳过，不创建对应 benchmark 结果目录。

---

# 8. `cot` 的最终语义：只做强校验，不传给 lm-eval

## 8.1 原因

固定 lm-eval commit：

```text
6d2abda4fd171e68a8789330c4149e37c1ca0bda
```

对应 0.4.8。

该版本 `simple_evaluate()` 没有通用：

```python
cot=True/False
```

参数。

因此绝对不要实现：

```python
simple_evaluate(..., cot=spec["cot"])
```

这会报错，也不是 lm-eval 的官方语义。

## 8.2 项目侧固定 CoT semantic map

在项目代码中建立唯一、明确、可测试的映射，例如放在适当模块：

```python
LM_EVAL_0_4_8_TASK_COT = {
    "truthfulqa_mc1": False,
    "mmlu_pro": True,
    "bbh": True,
    "agieval": False,
    "gsm8k_cot": True,
    "minerva_math": False,
}
```

该 mapping 明确绑定当前固定 revision：

```python
LM_EVAL_PINNED_REVISION = "6d2abda4fd171e68a8789330c4149e37c1ca0bda"
```

## 8.3 校验规则

每个 spec：

```python
configured_cot = spec["cot"]
expected_cot = LM_EVAL_0_4_8_TASK_COT[spec["name"]]
```

若：

```python
configured_cot != expected_cot
```

必须在启动实际 evaluation 前报错，例如：

```text
ValueError:
Configured cot=false conflicts with pinned lm-eval task 'bbh',
whose audited CoT semantic is true at revision 6d2abda...
```

不要：

- 静默忽略；
- 自动切换 task variant；
- 自动改配置；
- 只记录 warning 后继续。

### 未知 task

如果以后增加一个不在 audited map 中的 task：

- 不允许仅凭 task name 猜 CoT；
- 应报错，要求先扩展 audited semantic map；
- 这样 `cot` 才真正是强 provenance 字段。

---

# 9. `num_fewshot` 的语义

`num_fewshot` 与 `cot` 不同：

- `num_fewshot` **继续真实传给 lm-eval**；
- 允许以后修改/实验；
- 不要求等于 lm-eval YAML 中原始默认值。

因此 GSM8K：

```yaml
name: gsm8k_cot
num_fewshot: 4
cot: true
```

含义是：

- task 本身仍然是 CoT variant；
- 通过 `num_fewshot=4` 覆盖官方 task 默认 8-shot；
- `cot: true` 只是验证该 task 的真实 prompt variant 确实属于 CoT。

不要把 `cot` 和 `num_fewshot` 合并成一个逻辑。

---

# 10. `enabled` 的语义

每个 task spec 新增：

```yaml
enabled: true
```

执行时：

```python
enabled_specs = [
    spec for spec in task_specs
    if bool(spec["enabled"])
]
```

规则：

- `enabled: true` → 执行；
- `enabled: false` → 完全跳过；
- false task：
  - 不调用 lm-eval；
  - 不创建 benchmark output dir；
  - 不进入“本次实际执行任务”列表；
- 如果全部 task 都是 false：
  - 在加载模型进行昂贵评估前尽早 `ValueError`；
  - 错误信息明确写“no enabled lm-eval tasks”。

`results.json` 中应保存当前 task 的完整 resolved spec。

---

# 11. `test_samples` / `test_seed` 的精确定义

## 11.1 基本规则

每个 task：

```yaml
test_samples: null
test_seed: 42
```

### `test_samples: null`

- 使用该 benchmark 的完整 evaluation population；
- 不进行随机子采样；
- path 中写：

```text
test_samples_full
```

### `test_samples: N`

必须：

```text
seeded random without replacement
```

并满足：

```text
0 < N <= benchmark evaluation population size
```

超过总样本数直接 ValueError。

## 11.2 禁止使用 `simple_evaluate(limit=N)` 冒充随机采样

lm-eval 的：

```python
limit=N
```

只代表截断 evaluation docs，不实现：

```text
Random(test_seed) + without replacement
```

因此不能把：

```yaml
test_samples: 1000
test_seed: 42
```

简单翻译成：

```python
simple_evaluate(limit=1000)
```

否则 `test_seed` 没有真正作用。

本轮完成后建议移除 Tulu-3 对全局：

```yaml
evaluation.limit
```

的依赖。

如果为了兼容其他旧逻辑需要保留字段，也不要让 Tulu-3 新 per-task sampling 再使用它。

---

# 12. 对 group benchmark 的 `test_samples` 语义

特别注意：

- `bbh`
- `agieval`
- `minerva_math`

在 lm-eval 中属于 group / 多 leaf-task benchmark。

不能把：

```yaml
bbh:
  test_samples: 100
```

解释为“每个 BBH 子任务取 100 条”，否则实际可能变成 23×100，而 path 仍写 100，语义错误。

本项目定义：

> 对一个顶层 `lm_eval_task_specs` entry，`test_samples=N` 表示该**顶层 benchmark 总共评估 N 个 logical evaluation documents**。

建议稳定实现：

1. 用 pinned lm-eval `TaskManager` 解析顶层 task/group；
2. 展开所有 leaf tasks；
3. 按 leaf task name 的稳定排序建立 flattened population：

```text
(leaf_task_name, local_eval_doc_index)
```

4. 用：

```python
random.Random(test_seed).sample(...)
```

从 flattened population 中全局无放回选择 N 个；
5. 再按 leaf task 把选中的 local indices 分配回各 leaf；
6. 保持 lm-eval 原有 group metric aggregation 语义。

这样：

```text
test_samples_100
```

就严格代表该顶层 benchmark 本次总共选了 100 个 evaluation docs。

## 12.1 不要污染 few-shot pool

实现随机 test subset 时，不要直接原地缩小某个 HF dataset split，尤其当该 split 还可能被 few-shot sampler 使用。

不要为了减少 eval docs，粗暴执行：

```python
task.dataset["validation"] = task.dataset["validation"].select(...)
```

如果 validation 同时被 few-shot 使用，会错误改变 prompt。

推荐：

- 通过 task wrapper；
- 或对 evaluation `doc_iterator` 做只读 subset view；
- 或使用 lm-eval 的 task-object / lower-level API 构建仅改变 evaluation docs 的包装；

但必须保证：

```text
test subset selection
```

只改变 evaluation population，不改变：

```text
few-shot pool
task prompt
metric
generation config
dataset preprocessing
```

Codex 实现前应检查 pinned 0.4.8 的 `TaskManager`、`ConfigurableTask`、`doc_iterator`、`evaluate/simple_evaluate` API，选最小侵入实现。

---

# 13. `test_seed` 与 few-shot seed 必须分离

`test_seed` 只控制：

```text
benchmark evaluation subset
```

不要让它顺手改变：

```text
fewshot_random_seed
```

当前项目将 `fewshot_random_seed` 等绑定 `experiment.seed`；本轮没有要求改变这部分。

因此保持：

```text
few-shot sampling randomness = 原项目既有 seed 语义
test subset randomness        = spec.test_seed
```

两者 metadata 中要区分。

---

# 14. 每个 benchmark 的 test selection provenance

为了保证结果完全可复现，建议每个 benchmark evaluation output dir 除 `results.json` 外增加：

```text
test_selection.json
```

至少记录：

```json
{
  "task": "bbh",
  "test_samples": 100,
  "test_seed": 42,
  "sampling": "seeded_random_without_replacement",
  "total_population_size": 6511,
  "selected_count": 100,
  "selected_leaf_docs": [
    {
      "leaf_task": "...",
      "local_index": 123
    }
  ]
}
```

当 `test_samples: null` 时可以：

```json
{
  "task": "...",
  "test_samples": null,
  "test_seed": 42,
  "sampling": "full_evaluation_population",
  "total_population_size": ...,
  "selected_count": ...
}
```

如果项目更倾向于只写一个 `results.json`，则这些字段至少必须进入 `results.json` 的 metadata；但优先建议独立 selection manifest，便于 verify。

---

# 15. 配置 validation

修改 `snn2/config.py::validate_config()`。

## 15.1 Tulu-3 training

仅当：

```python
cfg["experiment"]["task"] == "tulu3"
```

校验：

```text
training.train_samples:
  null 或 positive int
  bool 非法

training.train_seed:
  int
  bool 非法
```

并且建议拒绝旧字段：

```yaml
data.train_size
```

若 materialized config 仍出现它，可直接报：

```text
Tulu-3 data.train_size is deprecated; use training.train_samples
```

避免旧配置悄悄继续生效。

## 15.2 lm-eval specs

对 Tulu-3 每个 spec 校验：

- `name`: non-empty str
- task names 唯一
- `enabled`: bool
- `num_fewshot`: int >= 0，bool 非法
- `metric`: non-empty str
- `cot`: bool
- `test_samples`: null 或 positive int，bool 非法
- `test_seed`: int，bool 非法
- `name` 必须能在 audited CoT map 中找到
- `cot` 必须等于 audited CoT
- `evaluation.lm_eval_revision` 必须仍是固定 revision；如果不是该 revision，不应继续假装当前 audited CoT map 有效

建议至少一个 task `enabled: true`。

---

# 16. Tulu-3 run path：加入 train samples

当前 `snn2/artifacts.py::ArtifactLayout` 对 TL;DR 有：

```text
lr<LR>_train_samples_<N>
```

Tulu-3 应采用同样的 train-sample identity。

默认：

```text
lr1e-06_train_samples_10000
```

若：

```yaml
train_samples: null
```

则：

```text
lr1e-06_train_samples_full
```

注意：

- `train_seed` 本轮用户没有要求写入 run path；
- 必须写入 resolved config / manifest / provenance；
- 不要擅自把 path 改成 `train_seed_42`，以免偏离已确认路径协议。

---

# 17. Tulu-3 run path：加入 scheduler + warmup

用户明确要求：

> Tulu-3 参考 TL;DR 实验路径时，原本只体现 `warmup_ratio` 的位置，Tulu-3 改为同时体现 `lr_scheduler_type` 与 `warmup_ratio`。

TL;DR 现有历史路径保持不变。

## 17.1 设计原则

仅对：

```python
task == "tulu3"
```

在 run identity 中体现：

```text
lr_scheduler_type_<TYPE>_warmup_ratio_<VALUE>
```

当前默认：

```text
lr_scheduler_type_cosine_warmup_ratio_0.0
```

## 17.2 vanilla / unaware

应包含：

```text
.../<mode>/
  lr1e-06_train_samples_10000/
  <run_variant>/
  lr_scheduler_type_cosine_warmup_ratio_0.0/
  seed42/
```

具体 `prefix_enabled_*` 等继续服从当前 mode protocol，不要硬编码。

## 17.3 phase-aware

保留当前 aware run identity：

```text
num_samples_<N>_lr<LR>_train_samples_<N>_calibration_group_size_<G>
```

并在 phase configuration layer 中形成：

```text
phase_T_<P>_mtn_T_<M>_surrogate_slope_<S>_lr_scheduler_type_<TYPE>_warmup_ratio_<W>
```

示例：

```text
.../phase_aware/
  num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/
  prefix_enabled_ture_common_clip_enabled_true/
  phase_T_4_mtn_T_4_surrogate_slope_1.0_lr_scheduler_type_cosine_warmup_ratio_0.0/
  seed42/
```

## 17.4 gif-aware

```text
.../gif_aware/
  num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/
  prefix_enabled_ture_common_clip_enabled_true/
  phase_T_4_mtn_T_4_lr_scheduler_type_cosine_warmup_ratio_0.0/
  seed42/
```

## 17.5 TL;DR 路径不得变化

特别写测试确认：

- TL;DR 原有 `ArtifactLayout` path snapshot 完全不变；
- 不要因为新增 helper 就让 TL;DR 也出现 `lr_scheduler_type_cosine`。

---

# 18. lm-eval output path：删除 `lm_harness/`

当前路径类似：

```text
.../evaluation/lm_harness/...
```

本轮改成：

```text
.../evaluation/...
```

即**彻底删除 `lm_harness` 这一级**。

---

# 19. lm-eval output path：每个 benchmark 独立目录

每个 enabled spec：

```text
<model_output_dir>/
  evaluation/
    [prefix_enabled_.../]
      <task_name>/
        num_fewshot_<N>_cot_<bool>_test_samples_<N/full>_test_seed_<seed>/
          results.json
          test_selection.json
```

## 19.1 spec dirname helper

建议新增统一 helper，例如：

```python
def lm_eval_spec_dirname(spec):
    ...
```

输出严格形如：

```text
num_fewshot_4_cot_true_test_samples_full_test_seed_42
```

规则：

- bool：小写 `true/false`
- null samples：`full`
- int：十进制字符串
- task name 使用现有 `safe_name()`

不要在多个脚本重复字符串拼接。

---

# 20. Base evaluation 路径

Base 不使用 Prefix，继续遵守当前 Base 特殊语义。

目标示例：

```text
.../<model>/base/seed42/
  evaluation/
    truthfulqa_mc1/
      num_fewshot_0_cot_false_test_samples_full_test_seed_42/
        results.json
```

Base 路径中不要因为本轮改动新增无意义的 Prefix layer。

---

# 21. Rotated-pre-finetuning evaluation 路径

继续沿用当前：

```text
_shared/.../rotated_pre_finetuning/evaluation/
```

Prefix 是否启用由当前：

```yaml
rotated_pre_finetuning.prefix_enabled
```

决定。

其后新增：

```text
prefix_enabled_.../
  <task>/
    num_fewshot_..._cot_..._test_samples_..._test_seed_.../
```

不要改变 rotated-pre-finetuning 的模型/Prefix 来源。

---

# 22. Final ANN evaluation 路径

目标：

```text
<run>/seed42/ann/
  evaluation/
    prefix_enabled_<...>/
      <task>/
        num_fewshot_<...>_cot_<...>_test_samples_<...>_test_seed_<...>/
          results.json
```

继续保持当前 mode-aware Prefix 行为：

- vanilla final ANN：实际不加载 Prefix；
- unaware final ANN：现有 Post-finetuning Prefix 规则；
- aware final ANN：现有 Pre-finetuning Prefix 规则。

本轮只改变 lm-eval benchmark path，不改变 Prefix source semantics。

---

# 23. SNN evaluation 路径

继续使用当前：

```text
snn/use_post_finetuning_artifacts_<bool>/...
```

和：

- phase runtime T
- mtn runtime T/K
- aware/non-aware calibration grouping
- conversion descriptor
- Prefix selector

等既有路径。

在每个 neuron 的：

```text
evaluation/
```

下新增：

```text
prefix_enabled_<...>/
  <task>/
    num_fewshot_<N>_cot_<bool>_test_samples_<N/full>_test_seed_<seed>/
      results.json
```

不要因为多 benchmark 重构 SNN source/layout。

---

# 24. `scripts/evaluate_lm_harness.py` 的目标结构

建议把当前大循环拆成几个可测试 helper，而不是继续把所有逻辑塞进 `main()`。

至少应有等价职责：

```python
validate_lm_eval_task_spec(...)
enabled_lm_eval_task_specs(...)
resolve_and_validate_cot_semantics(...)
build_test_selection(...)
lm_eval_spec_dirname(...)
evaluate_one_lm_eval_spec(...)
```

具体文件位置可根据仓库风格决定；如果 helper 对 tests/verify 都有用，优先放 `snn2/` 模块而不是 script 私有函数。

主循环应近似：

```python
task_specs = ...
enabled_specs = ...

for spec in enabled_specs:
    validate pinned cot semantic

    build deterministic evaluation selection
    # full split or seeded random subset

    task_result = evaluate exactly this spec

    output_dir = (
        model_output_dir
        / "evaluation"
        / [prefix_dir_if_applicable]
        / safe_task_name
        / lm_eval_spec_dirname(spec)
    )

    write results
    write selection provenance
```

不要再把所有 benchmark 累积后只写一个总 `results.json`。

每个 benchmark 应拥有自己的完整 results 文件。

---

# 25. 每个 task 的 results metadata

每个 `results.json` 的项目 metadata 至少增加/保留：

```json
{
  "lm_eval_task_spec": {
    "name": "...",
    "enabled": true,
    "num_fewshot": 4,
    "metric": "exact_match",
    "cot": true,
    "test_samples": null,
    "test_seed": 42
  },
  "lm_eval_revision": "6d2abda...",
  "cot_semantic_source": "project_audited_pinned_lm_eval_0_4_8",
  "test_sampling": "full_evaluation_population",
  "actual_test_samples": 1319,
  "fewshot_random_seed": 42,
  "test_seed": 42
}
```

并继续保留现有：

- model_variant
- model_source
- neuron
- temporal metadata
- Prefix metadata
- calibration metadata
- execution counter
- site/operator metadata
- common clip metadata
- deployment policy metadata

不要因为改成 per-task result 而丢失当前 `snn2_metadata`。

---

# 26. metric 校验

保留 spec 中：

```yaml
metric: ...
```

本轮不改变 metric 定义。

建议评估完成后检查 `task_result` 中确实包含用户声明的 metric；如果 lm-eval group 返回的 key 具有后缀/聚合结构，按当前 harness 真实返回格式实现稳健校验。

如果当前项目已有 metric extraction/validation，复用它；不要重复实现。

---

# 27. `evaluation.limit`

Tulu-3 新协议中 per-task：

```yaml
test_samples
test_seed
```

已经承担测试规模控制。

因此：

- `scripts/evaluate_lm_harness.py` 不应再使用全局 `evaluation.limit` 控制正式 Tulu-3 评估；
- 可以从 Tulu-3 matrix 删除 `limit`；
- 或保留旧字段仅做兼容，但如果非 null 则拒绝与新协议冲突。

推荐更严格：

```text
Tulu-3 + lm_eval_task_specs 使用 per-task test_samples；
evaluation.limit 必须为 null 或不存在。
```

避免出现两套相互冲突的测试规模入口。

---

# 28. 更新 `snn2/artifacts.py`

至少需要实现/调整：

1. Tulu-3 `train_samples` 加入 learning-rate run identity；
2. Tulu-3 scheduler/warmup path；
3. TL;DR 路径保持完全不变；
4. 新增 lm-eval task/spec dirname helper（若放此模块合适）；
5. 删除脚本侧 `lm_harness` path；
6. benchmark/spec path 统一由 helper 产生，避免字符串漂移。

如果 `phase_training_dirname()` / `gif_training_dirname()` 当前只接收 `warmup_ratio`：

- 不要直接无条件改变它们导致 TL;DR path 变化；
- 可以新增 task-aware helper；
- 或让 Tulu-3 走独立参数分支。

必须有 path snapshot tests。

---

# 29. 更新 `scripts/verify_artifacts.py`

当前 verify 主要围绕：

- training provenance
- Prefix
- calibration
- conversion
- evaluation

本轮 Tulu-3 evaluation 结果变为 per-benchmark path，因此 verify 要能：

1. 只遍历 `enabled: true` 的 task；
2. 按新 helper 找到每个 benchmark output；
3. 验证目录名和 resolved spec 一致：
   - num_fewshot
   - cot
   - test_samples
   - test_seed
4. 检查 `results.json` 存在；
5. 检查 results metadata 与 config 一致；
6. 若有 `test_selection.json`：
   - test seed 一致；
   - selected_count 正确；
   - 无重复；
   - selected_count 不超过 population；
7. `enabled: false`：
   - 不要求结果存在；
8. 验证 `cot` 与 audited map 一致；
9. 验证 pinned lm_eval revision。

不要破坏 TL;DR verify。

---

# 30. 更新 `scripts/materialize_configs.py`

`configs/experiment_matrix.yaml` 是 source of truth，generated config 不进 Git。

materializer 必须确保：

- 新字段原样进入 4 个 Tulu mode generated configs；
- mode resolution 不覆盖：
  - `training.train_samples`
  - `training.train_seed`
  - task specs 的 enabled/test_samples/test_seed/cot；
- 不给 TL;DR 自动添加 Tulu-only `train_samples`；
- 不改变 TL;DR evaluation 行为。

生成后检查：

```bash
python scripts/materialize_configs.py   --matrix configs/experiment_matrix.yaml   --output-dir configs/generated
```

并逐一检查四个：

```text
exp2_llama3_8b_tulu3__vanilla.yaml
exp2_llama3_8b_tulu3__unaware.yaml
exp2_llama3_8b_tulu3__phase_aware.yaml
exp2_llama3_8b_tulu3__gif_aware.yaml
```

---

# 31. 更新 `实验执行总结.md`

这是当前项目唯一的实验执行顺序说明，因此必须同步。

## 31.1 Step 1

把旧说明：

```text
Tulu 3 设置 data.train_size: null
lm-eval 设置 evaluation.limit: null
```

改成新协议：

```text
Tulu-3:
training.train_samples: 10000
training.train_seed: 42

lm-eval:
每个 task 有 enabled / num_fewshot / cot / test_samples / test_seed
test_samples: null 表示完整 benchmark
```

说明：

- Tulu training sample 为 seeded random without replacement；
- validation 仍固定 1000；
- calibration 从 selected training subset 取样。

## 31.2 Step 2

原文中：

```text
Tulu 3 的 data.train_size:null 表示...
```

必须删除。

改成：

```text
training.train_samples:null -> 完整 training pool
training.train_samples:N -> 从排除 1000 validation 后的 training pool 中，
按 training.train_seed 随机无放回选择 N 条
```

## 31.3 Step 6

明确 Tulu：

```text
train_samples=10000
epochs=2
gradient_accumulation_steps=16
```

训练步数由真实 selected sample count 计算。

## 31.4 Step 3 / 4 / 9 / 10 的 Tulu lm-eval 路径说明

所有原来含：

```text
evaluation/lm_harness/
```

的说明删除 `lm_harness/`。

增加 per-benchmark path 示例。

## 31.5 lm-eval task 表

加入最终表：

| task | enabled | few-shot | CoT | test_samples | test_seed |
|---|---:|---:|---:|---:|---:|
| truthfulqa_mc1 | true | 0 | false | null/full | 42 |
| mmlu_pro | true | 5 | true | null/full | 42 |
| bbh | true | 3 | true | null/full | 42 |
| agieval | true | 0 | false | null/full | 42 |
| gsm8k_cot | true | 4 | true | null/full | 42 |
| minerva_math | true | 4 | false | null/full | 42 |

同时明确：

> `cot` 不传入 `simple_evaluate()`；它是项目对 pinned lm-eval task prompt variant 的强一致性校验。

---

# 32. 推荐的测试覆盖

不能只跑现有 tests；本轮必须新增针对新协议的单元测试。

可以更新已有相关测试，也可以新增：

```text
tests/test_tulu3_lm_eval_protocol.py
```

名称不强制，但覆盖项必须完整。

## 32.1 Tulu train sampling tests

### Test A：默认 10k

模拟足够大的 source split：

```text
validation_size=1000
train_samples=10000
train_seed=42
```

断言：

- validation = 1000
- train = 10000
- 无交集
- train 无重复
- 相同 seed 两次完全一致

### Test B：train_seed 改变

只改变 `train_seed`：

- validation indices 不变；
- training subset 改变。

证明 `train_seed` 没有污染 validation seed。

### Test C：null

`train_samples=null`：

- train 使用 validation 之外所有行。

### Test D：非法

覆盖：

- 0
- -1
- bool
- 大于 training pool

都必须报错。

### Test E：calibration 来源

断言 calibration indices：

```text
subset of selected train indices
```

且与 validation 无交集。

---

# 33. lm-eval config validation tests

覆盖：

1. `enabled` 非 bool → error
2. `num_fewshot < 0` → error
3. `cot` 非 bool → error
4. `test_samples=0/-1/bool` → error
5. `test_seed=bool` → error
6. 重复 task name → error
7. 未知 task → audited CoT error
8. pinned revision 改变 → audited semantic invalid/error
9. `bbh + cot=false` → error
10. `mmlu_pro + cot=false` → error
11. `gsm8k_cot + cot=false` → error
12. 正确六个组合 → pass
13. 全部 enabled=false → evaluation 入口 fail fast

---

# 34. test subset sampling tests

用 fake task/group，不依赖下载真实 benchmark。

## 34.1 单 task

population=100：

```text
test_samples=20
test_seed=42
```

断言：

- exactly 20
- unique
- deterministic
- seed 改变 selection 改变。

## 34.2 group

例如 3 leaf：

```text
A=10
B=20
C=30
```

顶层：

```text
test_samples=15
```

断言实际总 selected：

```text
15
```

而不是每 leaf 15。

## 34.3 full

`test_samples=null`：

- 所有 leaf 全量进入 evaluation；
- metadata 为 full。

## 34.4 few-shot pool 不受影响

构造 validation 同时可作为 few-shot pool 的 fake task：

- test subset 改变；
- few-shot candidate pool 保持原样。

---

# 35. path tests

必须给 TL;DR 和 Tulu-3 都写 snapshot/字符串断言。

## 35.1 TL;DR regression

本轮修改前后，给同一 TL;DR config：

```text
ArtifactLayout.root
ann_dir
snn_dir
```

保持原路径。

特别断言 TL;DR **没有**新增：

```text
lr_scheduler_type_
```

## 35.2 Tulu vanilla/unaware

断言包含：

```text
lr1e-06_train_samples_10000
lr_scheduler_type_cosine_warmup_ratio_0.0
```

## 35.3 Tulu aware

断言：

```text
num_samples_..._lr..._train_samples_..._calibration_group_size_...
```

以及 phase/gif layer 中 scheduler/warmup。

## 35.4 evaluation

断言：

```text
/evaluation/<task>/num_fewshot_..._cot_..._test_samples_..._test_seed_...
```

或 non-base：

```text
/evaluation/prefix_enabled_.../<task>/...
```

并断言：

```text
"lm_harness" not in output path parts
```

---

# 36. enabled task output tests

给两个 task：

```yaml
A enabled: true
B enabled: false
```

断言：

- A 执行一次；
- B 不调用 evaluator；
- A path 创建；
- B path 不创建；
- run log 中“executed tasks”只有 A。

---

# 37. 结果 metadata / provenance tests

用 fake evaluator result 测试：

- resolved task spec 写入 results；
- cot audited value 与 config 一致；
- actual_test_samples 正确；
- test_seed 正确；
- fewshot seed 和 test seed 是两个独立字段；
- lm_eval revision 正确；
- 原有 SNN metadata 不丢失。

---

# 38. 不应修改的内容

本轮不要借机改变以下语义：

1. TL;DR：
   - `tldr_train_samples`
   - `tldr_train_seed`
   - `tldr_test_samples`
   - `tldr_test_seed`
   - ROUGE evaluation
   - 现有 TL;DR artifact paths
2. Rotation 算法；
3. Prefix discovery/source protocol；
4. Phase/GIF/MTN neuron 计算逻辑；
5. calibration A/B 数学规则；
6. common Clip 逻辑；
7. SNN conversion artifact selector；
8. `phase.T/mtn.T/mtn.K` runtime override；
9. 当前 Tulu `lr_scheduler_type` 数值本身：
   - 仍为 `cosine`
10. 当前 Tulu `warmup_ratio` 数值本身：
   - 仍为 `0.0`
11. 模型仍保持当前：
   - `meta-llama/Meta-Llama-3-8B`
   - 本轮没有要求切换到 Llama 3.1。

---

# 39. 推荐实施顺序

Codex 按以下顺序实施，降低路径/配置漂移风险。

## Phase 1：配置 schema

1. 修改 `configs/experiment_matrix.yaml`
2. 修改 `snn2/config.py`
3. 修改/检查 `scripts/materialize_configs.py`
4. materialize 临时 generated configs，检查新字段

## Phase 2：Tulu 数据

5. 修改 `snn2/data.py`
6. 增加 Tulu train selection tests
7. 确认 calibration 基于 selected 10k
8. 确认 trainer 实际拿 10k

## Phase 3：Artifact path

9. 修改 `snn2/artifacts.py`
10. 加 TL;DR path regression tests
11. 加 Tulu path tests

## Phase 4：lm-eval protocol

12. 加 audited CoT map / validation helper
13. 实现 enabled specs
14. 实现 test_samples/test_seed deterministic subset
15. 特别实现 group benchmark 全局 N 语义
16. 修改 `scripts/evaluate_lm_harness.py`
17. 删除 `lm_harness/`
18. 改成 per-task result

## Phase 5：provenance / verify

19. 更新 evaluation metadata
20. 更新 `scripts/verify_artifacts.py`
21. 加 selection/metadata tests

## Phase 6：文档

22. 更新 `实验执行总结.md`

## Phase 7：全量测试

23. `pytest -q`
24. materialize configs
25. 运行轻量 fake/smoke test
26. 不实际启动大规模 Llama 训练即可完成代码验收。

---

# 40. 推荐默认 Tulu-3 配置最终片段

Codex 修改后，Tulu-3 experiment 至少应呈现以下关键结构：

```yaml
data:
  max_seq_length: 2048
  truncation: true
  truncation_side: right
  packing: false
  dataset_name: allenai/tulu-3-sft-mixture
  dataset_revision: 3c09be1fe3882f2b1cb5a42ab26ceeab3dc9c6ea
  train_split: train
  validation_size: 1000
  chat_template_override: true
  chat_template: ...

training:
  num_train_epochs: 2
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-06
  weight_decay: 0.0
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-08
  lr_scheduler_type: cosine
  warmup_ratio: 0.0
  bf16: true
  fp16: false
  dtype: bfloat16
  max_grad_norm: 1.0
  gradient_checkpointing: false
  attn_implementation: eager
  flash_attention: false
  eval_strategy: "no"
  save_strategy: "no"
  load_best_model_at_end: false
  logging_steps: 10
  deepspeed_config: configs/deepspeed_zero3.json
  resume_from_checkpoint: null

  train_samples: 10000
  train_seed: 42

evaluation:
  prefix_enabled: true
  batch_size: 8
  lm_eval_revision: 6d2abda4fd171e68a8789330c4149e37c1ca0bda
  apply_chat_template: true

  lm_eval_task_specs:
  - name: truthfulqa_mc1
    enabled: true
    num_fewshot: 0
    metric: acc_mc1
    cot: false
    test_samples: null
    test_seed: 42

  - name: mmlu_pro
    enabled: true
    num_fewshot: 5
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: bbh
    enabled: true
    num_fewshot: 3
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: agieval
    enabled: true
    num_fewshot: 0
    metric: acc
    cot: false
    test_samples: null
    test_seed: 42

  - name: gsm8k_cot
    enabled: true
    num_fewshot: 4
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: minerva_math
    enabled: true
    num_fewshot: 4
    metric: exact_match
    cot: false
    test_samples: null
    test_seed: 42
```

注意：旧：

```yaml
data:
  train_size: null
```

必须移除。

Tulu 正式评估不再依赖全局：

```yaml
evaluation:
  limit: ...
```

---

# 41. 新 evaluation 目录示例

以 Final ANN、Prefix enabled、GSM8K-CoT 为例：

```text
.../ann/
└── evaluation/
    └── prefix_enabled_ture/
        └── gsm8k_cot/
            └── num_fewshot_4_cot_true_test_samples_full_test_seed_42/
                ├── results.json
                └── test_selection.json
```

注意明确**没有**：

```text
lm_harness/
```

MMLU-Pro：

```text
.../ann/evaluation/prefix_enabled_ture/
└── mmlu_pro/
    └── num_fewshot_5_cot_true_test_samples_full_test_seed_42/
        └── results.json
```

未来快速测试：

```yaml
test_samples: 1000
test_seed: 123
```

则：

```text
.../<task>/
└── num_fewshot_<N>_cot_<bool>_test_samples_1000_test_seed_123/
```

并且真实 evaluation selection 必须由 seed=123 无放回产生。

---

# 42. 最终验收 checklist

Codex 完成修改后逐项确认。

## Configuration

- [ ] Tulu `data.train_size` 已删除
- [ ] `training.train_samples: 10000`
- [ ] `training.train_seed: 42`
- [ ] `num_train_epochs: 2`
- [ ] `gradient_accumulation_steps: 16`
- [ ] scheduler 仍为 cosine
- [ ] warmup ratio 仍为 0.0
- [ ] 六 task 都有 enabled/test_samples/test_seed
- [ ] GSM8K-CoT = 4-shot
- [ ] CoT map 为 0.4.8 已确认组合

## Data

- [ ] validation 1000
- [ ] train 10000
- [ ] train/validation disjoint
- [ ] training sampling seeded without replacement
- [ ] calibration 从 selected train 采样
- [ ] trainer 真实使用 10000

## CoT

- [ ] `cot` 没有传给 `simple_evaluate`
- [ ] `cot` 被实际读取
- [ ] mismatch 会 fail fast
- [ ] unknown task 不会静默猜 CoT

## Test subset

- [ ] null = full
- [ ] integer = seeded random without replacement
- [ ] test_seed 真正影响选择
- [ ] 没有使用 `limit=N` 假装 random sampling
- [ ] group 的 N 是顶层 benchmark 总 N
- [ ] few-shot pool 未被 test subsetting 污染

## Paths

- [ ] Tulu run root 包含 train_samples
- [ ] Tulu run identity 包含 lr_scheduler_type + warmup_ratio
- [ ] TL;DR path 完全不变
- [ ] evaluation path 删除 lm_harness
- [ ] 每 benchmark 独立目录
- [ ] spec dirname 包含 fewshot/cot/test_samples/test_seed

## Outputs

- [ ] 每 enabled benchmark 有独立 results.json
- [ ] disabled benchmark 不执行
- [ ] results 保留全部原有 snn2 metadata
- [ ] test selection provenance 可恢复
- [ ] verify 支持新路径

## Documentation / tests

- [ ] `实验执行总结.md` 已同步
- [ ] 新增/更新单测
- [ ] `pytest -q` 全部通过
- [ ] generated configs materialize 成功
- [ ] 不修改/提交 `configs/generated/`（继续按仓库现有 Git ignore 规则）

---

# 43. 最重要的实现原则

本轮修改完成后应满足以下一句话：

> **Tulu-3 ANN training 的训练样本规模、随机性、scheduler identity，以及 lm-eval 每个 benchmark 的启用状态、few-shot、真实 CoT 语义、测试样本规模和测试随机性，都必须从配置一路贯穿到实际执行、artifact path、results metadata 和 verify；任何只写在 YAML 但没有改变/校验真实计算路径的字段都视为实现失败。**
