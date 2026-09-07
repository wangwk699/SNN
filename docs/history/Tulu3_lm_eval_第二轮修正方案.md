# Tulu-3 lm-eval 第二轮修正方案

> 目标仓库：`/home/wangwenkang/SNN`
> GitHub：`https://github.com/wangwk699/SNN`
> 基准分支：当前 `main`
> 本文档用于部署在服务器上的 Codex 在**没有此前对话上下文**的情况下完成本轮修正。

## 1. 两条最高优先级硬约束

### 1.1 TL;DR summarization 必须零影响

本轮所有新增修正只针对 `experiment.task == "tulu3"`。

TL;DR summarization 的数据采样、shared artifacts、ANN run path、SNN path、evaluation path、Prefix path、calibration path、`warmup_ratio` path、`train_samples` path、`prefix_enabled_ture/false` 以及 `evaluate_tldr.py` 均必须保持当前 `main` **逐字符不变**。

不得因为抽象公共 helper、统一路径函数或重构 `ArtifactLayout` 间接改变任何 TL;DR artifact/save path。

必须新增/保留 regression tests，证明 TL;DR 四种 mode 的完整路径字符串在本轮修改前后完全一致。

### 1.2 Tulu-3 `training.train_samples` 与 `_shared` 是同级体系

参考 TL;DR 当前 model-root 组织方式，Tulu-3 也保持：

```text
artifacts/snn2_main_v1/tulu3/meta-llama_Meta-Llama-3-8B/
├── _shared/
├── vanilla/
├── unaware/
├── phase_aware/
└── gif_aware/
```

`training.train_samples` 只属于具体 ANN run identity，不进入 `_shared`。

禁止：

```text
train_samples_10000/_shared/...
_shared/train_samples_10000/...
```

允许：

```text
unaware/lr1e-06_train_samples_10000/...
unaware/lr1e-06_train_samples_100000/...
```

两者继续共享：

```text
_shared/seed42/...
```

shared calibration / Pre-finetuning Prefix / shared Stage A 路径中都不能加入 `training.train_samples` 或 `training.train_seed`。

---

## 2. 本轮需要修正的 6 个问题

1. group benchmark 在有限 `test_samples` 下存在 zero-leaf 崩溃风险。
2. Tulu unaware Final ANN 的 `verify_artifacts.py` 路径与 evaluator 实际保存路径不一致。
3. Tulu vanilla / unaware run path 还没有加入 `lr_scheduler_type_<...>_warmup_ratio_<...>`。
4. 有限 `test_samples` 时，lm-eval 自带 `n-samples.effective` 可能仍记录 full size，与真实随机子集不一致。
5. Tulu lm-eval provenance verify 不够严格。
6. Tulu shared calibration selection 当前仍依赖最终 ANN `training.train_samples` subset，需要与具体 ANN train subset 解耦。

---

## 3. Tulu shared training pool 与 ANN train subset 解耦

当前错误关系：

```text
Tulu source train
    ↓
固定 1000 validation
    ↓
training.train_samples / train_seed
    ↓
selected ANN training subset
    ↓
calibration.num_samples / calibration.seed
    ↓
shared calibration manifest
    ↓
Pre-finetuning Prefix / shared Stage A
```

必须改成：

```text
Tulu source train
    ↓
固定 1000 validation
    ↓
shared training pool
    ├──────────────────────────────┐
    │                              │
    ↓                              ↓
shared calibration            ANN train selection
calibration.seed              training.train_seed
calibration.num_samples       training.train_samples
    ↓                              ↓
calibration manifest          ANN training subset
    ↓
Pre-finetuning Prefix
    ↓
shared Stage A
```

也就是说：Tulu-3 的 shared calibration 和具体 ANN training subset 都从同一个 **validation-excluded shared training pool** 独立采样。

---

## 4. Tulu shared training pool 的精确定义

继续使用：

```yaml
data:
  validation_size: 1000
```

validation 保持当前既有 seed 语义：

```python
seed = int(cfg["experiment"]["seed"])
rng = random.Random(seed)

permutation = list(range(len(raw_train)))
rng.shuffle(permutation)

validation_indices = permutation[:validation_size]
shared_training_pool_indices = permutation[validation_size:]
```

三种随机性必须分离：

```text
validation randomness       = experiment.seed
ANN train subset randomness = training.train_seed
shared calibration randomness = calibration.seed
```

不要用 `training.train_seed` 重新决定 validation。

---

## 5. ANN training subset

ANN training 继续：

```yaml
training:
  train_samples: 10000
  train_seed: 42
```

并从 `shared_training_pool_indices` 中随机无放回选择：

```text
train_samples == null
    -> 使用整个 shared training pool

train_samples == N
    -> random.Random(train_seed).sample(shared_training_pool, k=N)
```

最终 selected indices 建议继续排序，以保持执行顺序稳定。

该逻辑只影响 concrete ANN training run，不影响 shared calibration。

---

## 6. Shared calibration selection

Tulu-3 的 `layout.calibration_data_manifest_path` 必须直接基于 `shared_training_pool_indices` 选择 calibration，而不是基于 selected ANN training subset：

```python
calibration_positions, calibration_indices = _calibration_selection(
    shared_training_pool_indices,
    seed=int(cfg["calibration"]["seed"]),
    num_samples=int(cfg["calibration"]["num_samples"]),
    with_replacement=False,
)
```

因此以下两组配置，只要 dataset/validation/calibration 配置相同：

```yaml
training:
  train_samples: 10000
  train_seed: 42
calibration:
  num_samples: 128
  seed: 42
```

和：

```yaml
training:
  train_samples: 100000
  train_seed: 123
calibration:
  num_samples: 128
  seed: 42
```

必须得到完全相同的 shared calibration indices、calibration manifest、Pre-finetuning Prefix 和 shared Stage A。

---

## 7. Calibration manifest 中 `retained_in_training` 的修正

新语义下，shared calibration 只保证来自 shared training pool，不保证属于某个具体 ANN `training.train_samples` subset。

因此 Tulu-3 shared calibration manifest 不应再无条件写：

```json
"retained_in_training": true
```

推荐改成：

```json
{
  "selection_pool": "validation_excluded_shared_training_pool",
  "retained_in_shared_training_pool": true,
  "retained_in_ann_training_subset": null
}
```

如果不希望扩大 schema，也可以只对 Tulu-3 删除 `retained_in_training`。

**TL;DR manifest 保持现有字段和语义完全不变。**

---

## 8. Shared artifact path 不加入 `train_samples`

以下 Tulu shared 路径必须继续与 `training.train_samples` 无关：

```text
_shared/seed42/data/calibration/num_samples_<N>/
_shared/seed42/rotated_prefix/pre_finetuning_prefix/
_shared/.../ann_training_calibration/
```

以及所有由 shared calibration 派生的：

- Pre-finetuning Prefix；
- ANN-training shared Stage A；
- selector=false 时复用的 shared SNN calibration source。

不得添加 `train_samples_<N>` 或 `train_seed_<N>`。

---

## 9. Concrete ANN run path 继续包含 `train_samples`

具体 ANN run 必须继续包含：

```text
lr<LR>_train_samples_<N/full>
```

例如：

```text
vanilla/lr1e-06_train_samples_10000/...
unaware/lr1e-06_train_samples_10000/...
phase_aware/num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/...
gif_aware/num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/...
```

不要移除 `train_samples`。

---

## 10. Tulu vanilla / unaware 补 scheduler path

只对 `experiment.task == "tulu3"` 增加：

```text
lr_scheduler_type_<TYPE>_warmup_ratio_<W>
```

目标：

```text
.../vanilla/
  lr1e-06_train_samples_10000/
  prefix_enabled_false/
  lr_scheduler_type_cosine_warmup_ratio_0.0/
  seed42/
```

```text
.../unaware/
  lr1e-06_train_samples_10000/
  prefix_enabled_ture/
  lr_scheduler_type_cosine_warmup_ratio_0.0/
  seed42/
```

---

## 11. Tulu aware path

Phase-aware：

```text
.../phase_aware/
  num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/
  prefix_enabled_ture_common_clip_enabled_true/
  phase_T_4_mtn_T_4_surrogate_slope_1.0_lr_scheduler_type_cosine_warmup_ratio_0.0/
  seed42/
```

GIF-aware：

```text
.../gif_aware/
  num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/
  prefix_enabled_ture_common_clip_enabled_true/
  phase_T_4_mtn_T_4_lr_scheduler_type_cosine_warmup_ratio_0.0/
  seed42/
```

已有 aware 路径若已正确，不要无必要重构。

---

## 12. TL;DR ArtifactLayout 硬回归

修改 `snn2/artifacts.py` 时必须保证：

- TL;DR vanilla 路径逐字符不变；
- TL;DR unaware 路径逐字符不变；
- TL;DR phase-aware 路径逐字符不变；
- TL;DR gif-aware 路径逐字符不变；
- TL;DR 不出现 `lr_scheduler_type_`。

建议为四种 mode 写明确 expected full string，而不是只检查部分 substring。

---

## 13. Group benchmark 有限 `test_samples` 的 zero-leaf bug

当前如果：

```python
selected = selection_by_leaf(selection)
for leaf_name, task in leaves:
    indices = sorted(selected[leaf_name])
```

而某个 leaf 没抽到任何 document，会直接 `KeyError`。

即使改成：

```python
selected.get(leaf_name, set())
```

仍不够，因为 lm-eval 0.4.8 后续对 0-instance leaf 可能访问：

```python
task.instances[0]
```

从而产生 `IndexError`。

---

## 14. Group benchmark 正确处理方式

对于顶层 group：

```text
BBH / AGIEval / Minerva Math / 其他 group
```

`test_samples=N` 继续定义为：

> 整个顶层 benchmark 总共评估 N 个 logical evaluation documents，而不是每个 leaf 各 N。

流程：

1. 展开所有 leaf；
2. 建 flattened population：
   ```text
   (leaf_task_name, local_index)
   ```
3. `random.Random(test_seed).sample(...)` 全局抽 N；
4. 得到 `leaf -> selected indices`；
5. 没有抽中任何 document 的 leaf 不能以 0-instance task 留在本次 task tree；
6. 必须从本次 evaluation task tree 中安全剔除该 leaf；
7. 递归删除因此变空的中间 group；
8. group aggregation 只聚合本次实际参与 evaluation 的 leaf；
9. 不改变 prompt、few-shot pool、generation config、metric config。

推荐实现独立 helper：

```python
prune_empty_selected_leaves(task_tree, selected_by_leaf)
```

不要原地污染全局 lm-eval registry。

---

## 15. 单 task 有限 `test_samples`

非 group task 继续可以通过 evaluation-only `doc_iterator` wrapper 实现。

必须保证：

- selected doc ids 唯一；
- exact selected count；
- few-shot pool 不变；
- underlying dataset split 不被原地裁剪。

---

## 16. `test_samples: null`

`test_samples: null`：

- 使用完整 evaluation population；
- 不做随机子采样；
- 不做 leaf pruning；
- path 继续为 `test_samples_full`；
- `test_seed` 仍进入 path/provenance；
- few-shot seed 保持原项目 `experiment.seed` 语义。

---

## 17. 修正 lm-eval `n-samples.effective`

当前 `simple_evaluate(..., limit=None)` 时，task 的 `eval_docs` 仍是 full dataset，因此 lm-eval 自带：

```json
"n-samples": {
  "...": {
    "original": full,
    "effective": full
  }
}
```

可能和实际 `test_samples=N` 不一致。

目标：对每个 leaf：

```text
original  = 原 task full eval population size
effective = 实际 selected count
```

例如：

```json
"n-samples": {
  "bbh_boolean_expressions": {
    "original": 250,
    "effective": 4
  }
}
```

项目自己的：

```json
"snn2_metadata": {
  "actual_test_samples": 100
}
```

继续代表顶层 spec 的总 selected count。

推荐在 `simple_evaluate()` 返回后，根据 `test_selection` 修正 `task_result["n-samples"][leaf]["effective"]`。

只改 `effective`，不改 `original`，不改 metric。

---

## 18. unaware Final ANN verify 路径 bug

evaluator 对 Tulu unaware Final ANN + Prefix enabled 会通过：

```python
append_evaluation_num_samples_if_needed(...)
```

加入：

```text
num_samples_<calibration.num_samples>
```

但当前 Tulu `verify_artifacts.py` 构造 lm-eval path 时没有同步这层。

必须让 verify 和 evaluator 共用同一 helper/条件。

目标示例：

```text
ann/evaluation/
prefix_enabled_ture/
num_samples_128/
mmlu_pro/
num_fewshot_5_cot_true_test_samples_full_test_seed_42/
results.json
```

Base 不应错误出现 `num_samples_<N>`。

---

## 19. 强化 Tulu lm-eval provenance verify

当前 verify 不能只检查文件存在。

必须逐 enabled task 验证：

```text
results.json
test_selection.json
```

### `results.json` 至少验证

```text
snn2_metadata.lm_eval_task_spec
snn2_metadata.lm_eval_revision
snn2_metadata.cot_semantic_source
snn2_metadata.test_sampling
snn2_metadata.actual_test_samples
snn2_metadata.fewshot_random_seed
snn2_metadata.test_seed
```

`lm_eval_task_spec` 逐字段比对：

```text
name
enabled
num_fewshot
metric
cot
test_samples
test_seed
```

### `test_selection.json` 至少验证

1. `task == spec.name`
2. `test_samples == spec.test_samples`
3. `test_seed == spec.test_seed`
4. `selected_count == len(selected_leaf_docs)`
5. `(leaf_task, local_index)` 无重复
6. `selected_count <= total_population_size`
7. full：
   ```text
   selected_count == total_population_size
   sampling == full_evaluation_population
   ```
8. finite：
   ```text
   selected_count == N
   sampling == seeded_random_without_replacement
   ```
9. `actual_test_samples == selected_count`

---

## 20. CoT 与 enabled 语义保持不变

`cot` 继续只作为固定 lm-eval 0.4.8 task 的真实 CoT 语义强校验字段。

不要传：

```python
cot=...
```

给 `simple_evaluate()`。

Mismatch 必须 `ValueError`。

`enabled:false`：

- 不执行 task；
- 不要求结果存在；
- 不删除历史旧结果；
- verify 只遍历 enabled task。

---

## 21. Evaluation 路径保持上一轮协议

不要重新引入：

```text
lm_harness/
```

继续：

```text
.../evaluation/
[prefix layer]/
[optional num_samples layer]/
<task>/
num_fewshot_<N>_cot_<bool>_test_samples_<N/full>_test_seed_<seed>/
```

---

## 22. 推荐修改文件

至少检查并修改：

```text
snn2/data.py
snn2/artifacts.py
snn2/lm_eval_protocol.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
tests/test_tulu3_lm_eval_protocol.py
tests/test_evaluation_paths.py
实验执行总结.md
```

必要时补：

```text
tests/test_generated_configs.py
```

不要修改与本轮无关的 neuron、rotation、clip、conversion 数学实现。

---

## 23. `snn2/data.py` 推荐结构

建议显式拆成：

```python
_tulu3_shared_split_selection(...)
_tulu3_train_selection(...)
```

其中：

```python
_tulu3_shared_split_selection(raw_train, cfg)
```

负责：

```text
validation_indices
shared_training_pool_indices
```

而：

```python
_tulu3_train_selection(shared_training_pool_indices, cfg)
```

只负责：

```text
training.train_samples
training.train_seed
```

shared calibration 直接调用：

```python
_calibration_selection(
    shared_training_pool_indices,
    seed=calibration.seed,
    ...
)
```

不要让 shared calibration helper 接受 selected ANN train list。

---

## 24. Shared artifact compatibility

如果当前 shared Prefix / Stage A compatibility validation 比较了整个 resolved config 或 config hash，导致仅修改：

```text
training.train_samples
training.train_seed
learning_rate
epochs
gradient accumulation
scheduler
warmup
```

就把 shared artifacts 判为 stale/incompatible，则需要只对 Tulu-3 修正。

Tulu shared artifact compatibility identity 应只包含真正影响 shared artifact 内容的字段，例如：

```text
dataset_name
dataset_revision
experiment.seed
data.validation_size
calibration.seed
calibration.num_samples
calibration.group_size
model identity
rotation state
Prefix state
```

不应把 concrete ANN training hyperparameters 当作 shared artifact identity。

---

## 25. 关键测试：shared calibration 与 train_samples 解耦

同一 fake Tulu dataset：

```yaml
experiment.seed: 7
validation_size: 5
calibration.seed: 42
calibration.num_samples: 4
```

分别：

```yaml
train_samples: 10
train_seed: 42
```

和：

```yaml
train_samples: 20
train_seed: 99
```

断言：

```text
validation indices 完全相同
shared calibration indices 完全相同
ANN train indices 不同
```

---

## 26. 关键测试：calibration 不要求属于 ANN subset

构造很小的 `train_samples`。

断言：

- calibration 全部属于 shared training pool；
- calibration 与 validation 无交集；
- 不要求 calibration ⊆ ANN train subset。

---

## 27. 关键测试：Tulu `_shared` 与 train_samples 无关

分别构造：

```text
train_samples=10000
train_samples=100000
```

两个 `ArtifactLayout`。

断言：

```python
layout_a.shared_task_root == layout_b.shared_task_root
layout_a.shared_model_root == layout_b.shared_model_root
layout_a.calibration_data_manifest_path == layout_b.calibration_data_manifest_path
layout_a.ann_training_prefix_dir == layout_b.ann_training_prefix_dir
layout_a.ann_training_site_dir == layout_b.ann_training_site_dir
```

同时：

```python
layout_a.root != layout_b.root
```

---

## 28. 关键测试：Tulu vanilla/unaware scheduler path

断言 vanilla/unaware 都包含：

```text
lr_scheduler_type_cosine_warmup_ratio_0.0
```

再把：

```text
cosine -> linear
```

必须得到不同 Tulu run root。

TL;DR 对应 path 不得变化。

---

## 29. 关键测试：TL;DR path 逐字符不变

针对四种 TL;DR mode 构造固定 config：

```text
vanilla
unaware
phase_aware
gif_aware
```

写完整 expected path string。

额外断言：

```python
"lr_scheduler_type_" not in str(tldr_layout.root)
```

---

## 30. 关键测试：group zero leaf

不要只测试 `build_test_selection()`。

必须测试真实 task-tree wrapper / pruning helper。

例如：

```text
A=10
B=20
C=30
test_samples=1
```

断言：

- selected_count == 1；
- 只有实际 selected leaf 被保留；
- 0-doc leaf 被删；
- evaluator 不 KeyError；
- evaluator 不因 `task.instances[0]` IndexError；
- group aggregation 正常。

---

## 31. 关键测试：group global N

例如：

```text
A=10
B=20
C=30
test_samples=15
```

断言总 selected：

```text
15
```

而不是每 leaf 15。

---

## 32. 关键测试：few-shot pool 不变

随机 test subset 前后：

```text
few-shot candidate population
```

必须完全不变。

---

## 33. 关键测试：`n-samples`

有限 subset：

```text
original=100
selected=20
```

必须：

```json
"original": 100,
"effective": 20
```

full：

```json
"original": 100,
"effective": 100
```

group 要逐 leaf 验证。

---

## 34. 关键测试：unaware verify path

Tulu unaware：

```text
evaluation.prefix_enabled=true
calibration.num_samples=128
```

evaluator 与 verify 必须共同定位：

```text
ann/evaluation/
prefix_enabled_ture/
num_samples_128/
<task>/
<spec>/
```

---

## 35. Verify provenance tests

至少覆盖：

- wrong `num_fewshot` -> fail
- wrong `cot` -> fail
- wrong `test_seed` -> fail
- wrong `selected_count` -> fail
- duplicate selected doc -> fail
- wrong `lm_eval_revision` -> fail
- disabled task missing result -> pass
- enabled task missing result -> fail

---

## 36. 更新 `实验执行总结.md`

只改 Tulu 相关说明。

必须明确：

```text
Tulu-3 source train
    ↓
fixed 1000 validation
    ↓
shared training pool
    ├── shared calibration / Pre-finetuning Prefix / shared Stage A
    └── training.train_samples-specific ANN subset
```

删除或修正任何类似：

```text
calibration 从最终 10k ANN train subset 中取样
```

的旧表述。

改成：

> Tulu-3 先固定剔除 1000 validation，剩余部分形成 shared training pool。shared calibration 从该 pool 按 calibration seed 独立抽样；ANN training 再按 training.train_seed/train_samples 从同一 pool 独立抽样。不同 `training.train_samples` 共享同一 Pre-finetuning Prefix / calibration manifest / shared Stage A。

路径文档中明确：

```text
model root
├── _shared
├── vanilla
├── unaware
├── phase_aware
└── gif_aware
```

---

## 37. 本轮不要修改的内容

不要修改：

- TL;DR 路径；
- TL;DR data semantics；
- `evaluate_tldr.py`；
- Rotation 数学实现；
- Prefix discovery 算法；
- Phase/GIF/MTN neuron 数学实现；
- GIF saliency；
- common Clip；
- calibration group 数学规则；
- conversion selector；
- Final ANN forward semantics；
- SNN temporal execution；
- lm-eval 六个 task 名称；
- current CoT mapping；
- GSM8K 4-shot；
- Tulu `max_seq_length=2048`；
- Tulu model identity；
- Tulu scheduler 当前值 `cosine`；
- Tulu warmup 当前值 `0.0`。

---

## 38. 推荐实施顺序

### Phase A：数据语义

1. 重构 Tulu fixed validation / shared training pool helper；
2. ANN train subset 从 shared pool 独立采样；
3. shared calibration 从 shared pool 独立采样；
4. 更新 Tulu manifest provenance；
5. 检查 shared artifact compatibility；
6. 补数据单测。

### Phase B：路径

7. Tulu vanilla/unaware 加 scheduler path；
8. 确认 aware path；
9. 加 Tulu `_shared` independence tests；
10. 加 TL;DR exact path regression tests。

### Phase C：lm-eval subset

11. 修 group zero-leaf；
12. prune empty leaves；
13. 保持 few-shot pool 不变；
14. 修 `n-samples.effective`；
15. 补 fake task/group tests。

### Phase D：verify

16. 修 unaware Final ANN path；
17. 逐 enabled task 验证 results/test_selection；
18. 校验 spec/CoT/revision/provenance；
19. 补 verify tests。

### Phase E：文档与验收

20. 更新 `实验执行总结.md`；
21. materialize configs；
22. `pytest -q`；
23. 轻量 lm-eval fake/smoke test。

---

## 39. 最终验收 checklist

### TL;DR

- [ ] 所有 TL;DR path 逐字符不变
- [ ] TL;DR data/calibration/evaluation 语义不变
- [ ] TL;DR 不出现 `lr_scheduler_type_`
- [ ] TL;DR tests 全部通过

### Tulu shared pool

- [ ] validation 固定 1000
- [ ] shared training pool = source train - validation
- [ ] shared calibration 从 shared pool 采样
- [ ] ANN train subset 从 shared pool 独立采样
- [ ] 改 `train_samples/train_seed` 不改变 shared calibration
- [ ] shared paths 不含 `train_samples`
- [ ] concrete ANN run path 含 `train_samples`

### Scheduler path

- [ ] vanilla 含 scheduler + warmup
- [ ] unaware 含 scheduler + warmup
- [ ] phase aware 含 scheduler + warmup
- [ ] gif aware 含 scheduler + warmup
- [ ] scheduler 改值会改变 Tulu run path

### lm-eval subset

- [ ] full benchmark 正常
- [ ] single task finite subset 正常
- [ ] group finite subset 正常
- [ ] zero-selected leaf 不进入 evaluator
- [ ] group selected 总数严格等于 N
- [ ] few-shot pool 不变
- [ ] `n-samples.effective` 正确

### verify

- [ ] unaware Final ANN 路径正确
- [ ] 每个 enabled task 被验证
- [ ] disabled task 不被要求
- [ ] spec metadata 一致
- [ ] test selection 一致
- [ ] duplicate selection 会 fail
- [ ] lm_eval revision 一致
- [ ] CoT audited semantic 一致

### General

- [ ] `lm_harness/` 没有重新出现
- [ ] `cot` 没有传给 `simple_evaluate`
- [ ] GSM8K 仍是 4-shot CoT
- [ ] `pytest -q` 全通过
- [ ] generated configs 可正常 materialize

---

## 40. 最终实现原则

> **Tulu-3 的 shared calibration / Pre-finetuning Prefix / shared Stage A 只依赖 validation-excluded shared training pool 和真正相关的 calibration/model/rotation 配置，不依赖具体 ANN `training.train_samples`；具体 ANN run 仍通过 `train_samples` 区分。**

同时：

> **所有本轮新修正必须严格限定在 Tulu-3，TL;DR summarization 的现有 artifact/save path 必须逐字符保持不变。**
