# Tulu-3 lm-eval 第四轮修正方案

> 目标仓库：`/home/wangwenkang/SNN`  
> GitHub：`https://github.com/wangwk699/SNN`  
> 基准：当前 `main`，提交 `3ce13f28567f1d1fc4c0cc2e7c23caace7191483`  
> 本文档用于服务器上的 Codex 在**没有此前对话上下文**的情况下完成本轮修正。

---

# 1. 本轮修改范围

前三轮修改后，以下内容已经正确，本轮不要重复重构：

- Tulu-3 fixed validation -> shared training pool；
- shared calibration 与 concrete ANN `training.train_samples` 解耦；
- `_shared` 路径不包含 `training.train_samples/train_seed`；
- concrete ANN run path 保留 `train_samples`；
- Tulu vanilla/unaware/phase-aware/gif-aware scheduler path；
- group benchmark `test_samples=N` 为顶层 benchmark 全局 N；
- zero-selected leaf pruning；
- `n-samples.effective` 修正；
- unaware Final ANN verify path；
- per-task execution counter；
- ANN / Phase / GIF / MTN 逐 benchmark provenance verification；
- `cot` 作为 pinned lm-eval task semantic 的强校验字段，且不传给 `simple_evaluate()`。

本轮只处理以下内容：

1. **修正 TruthfulQA `metric: acc_mc1` -> `metric: acc`。**
2. **像 `cot` 一样，将六个 benchmark 的 `metric` 与 pinned lm-eval revision 强绑定。**
3. **加强 `test_selection.json` provenance，使 verify 能验证具体抽中的 document，而不仅是数量。**
4. **补齐此前仍缺失的 path / provenance / integration regression tests。**

---

# 2. 最高优先级硬约束：TL;DR 零改动

本轮所有新逻辑必须严格限定在：

```python
cfg["experiment"]["task"] == "tulu3"
```

或 Tulu 专用 lm-eval helper 中。

TL;DR summarization 的：

- data semantics；
- calibration；
- Prefix；
- ANN/SNN forward；
- run path；
- evaluation path；
- verify path；
- result format；
- selection format；

全部保持当前 `main` **逐字符、逐语义不变**。

不要修改：

```text
scripts/evaluate_tldr.py
```

也不要因为抽象新的 metric/test-selection helper 而间接改变 TL;DR。

---

# 3. 问题 1：TruthfulQA metric 当前写错

当前 Tulu 配置：

```yaml
- name: truthfulqa_mc1
  enabled: true
  num_fewshot: 0
  metric: acc_mc1
  cot: false
  test_samples: null
  test_seed: 42
```

但固定的 lm-eval revision：

```text
6d2abda4fd171e68a8789330c4149e37c1ca0bda
```

中，`truthfulqa_mc1` 的实际 `metric_list` 是：

```yaml
metric_list:
  - metric: acc
```

因此配置必须改为：

```yaml
- name: truthfulqa_mc1
  enabled: true
  num_fewshot: 0
  metric: acc
  cot: false
  test_samples: null
  test_seed: 42
```

---

# 4. 六个 benchmark 的 canonical metric

新增 pinned metric mapping，例如放在：

```text
snn2/lm_eval_protocol.py
```

推荐：

```python
LM_EVAL_0_4_8_TASK_METRIC = {
    "truthfulqa_mc1": "acc",
    "mmlu_pro": "exact_match",
    "bbh": "exact_match",
    "agieval": "acc",
    "gsm8k_cot": "exact_match",
    "minerva_math": "exact_match",
}
```

与当前：

```python
LM_EVAL_0_4_8_TASK_COT
```

并列维护。

---

# 5. Metric 必须成为强协议字段

当前 `validate_lm_eval_task_specs()` 只检查：

```python
isinstance(spec["metric"], str)
```

这不够。

改成：

```python
expected_metric = LM_EVAL_0_4_8_TASK_METRIC[name]

if spec["metric"] != expected_metric:
    raise ValueError(
        f"Configured metric={spec['metric']!r} conflicts with pinned "
        f"lm-eval task {name!r}, whose audited metric is "
        f"{expected_metric!r} at revision {LM_EVAL_PINNED_REVISION}"
    )
```

因此：

```text
cot
metric
```

都属于：

> project-side audited pinned lm-eval semantic fields

配置与固定 revision 的实际任务定义不一致时，应在加载模型前 fail。

---

# 6. 不要把 `metric` 作为参数传给 `simple_evaluate()`

本轮不要做类似：

```python
simple_evaluate(..., metric=spec["metric"])
```

lm-eval 不需要该参数。

`metric` 的作用与 `cot` 类似：

> 校验当前项目配置声明的协议是否与 pinned task 定义一致。

实际 metric 仍由 pinned lm-eval task YAML / Python task 定义执行。

---

# 7. verify 要检查 metric provenance

当前：

```python
_validate_tulu_lm_eval_result(...)
```

已经验证：

```text
lm_eval_task_spec
lm_eval_revision
cot_semantic_source
fewshot seed
test seed
sampling
```

继续保留。

由于：

```python
lm_eval_task_spec == current spec
```

会间接验证 `metric`，但建议再补一层实际 result validation：

> 配置声明的 canonical metric 必须真的存在于 lm-eval 返回结果中。

---

# 8. Metric result validation

对于 single task / group task，lm-eval 的返回结构通常位于：

```python
task_result["results"]
```

或 group aggregate result 中。

不要假设所有 benchmark 都使用完全相同的 filter suffix。

推荐写 helper：

```python
def result_contains_metric(task_result, metric_name):
    ...
```

匹配规则：

```text
metric_name
metric_name,<filter>
metric_name_stderr,<filter>  # 不能单独以 stderr 当 metric 存在证明
```

至少要找到一个非 stderr 的：

```text
metric_name
```

结果项。

例如：

```text
TruthfulQA:
  acc,none

MMLU-Pro:
  exact_match,custom-extract

BBH:
  exact_match,get-answer

AGIEval:
  acc,...

GSM8K:
  exact_match,strict-match
  exact_match,flexible-extract

Minerva Math:
  exact_match,...
```

不要把 filter name 写死进项目协议，除非当前任务真的要求。

---

# 9. Group benchmark metric 的验证

对于：

```text
mmlu_pro
bbh
agieval
minerva_math
```

优先验证顶层 group aggregate result 是否包含 config 中声明 metric。

如果 pinned lm-eval 在某个 task 下只在 leaf results 中存 metric，则允许 fallback 到 leaf results。

核心要求：

> `spec["metric"]` 不能只是 metadata 字符串，必须能够在真实 lm-eval 返回 payload 中找到对应 metric。

---

# 10. 问题 2：当前 test-selection verify 只能验证数量，不能验证具体抽样结果

当前 verify 会检查：

```text
task
test_samples
test_seed
selected_count
total_population_size
sampling
无重复
n-samples.effective
```

但如果有人把：

```json
selected_leaf_docs
```

中的若干：

```text
(leaf_task, local_index)
```

替换成其它合法 index，只要：

- 数量不变；
- 没有重复；
- selected_count 正确；

当前 verify 可能仍然通过。

这不满足项目的强 provenance 目标。

---

# 11. 正确目标：verify 能重新计算 exact selection

对于 finite：

```yaml
test_samples: N
test_seed: 42
```

verify 应能够根据 evaluation population 重新运行：

```python
build_test_selection(...)
```

并确认：

```text
expected selected_leaf_docs
==
recorded selected_leaf_docs
```

逐项完全一致。

---

# 12. 推荐方式：在 `test_selection.json` 记录 `leaf_population`

在 `build_test_selection()` 输出中新增：

```json
"leaf_population": {
  "leaf_a": 250,
  "leaf_b": 250,
  "leaf_c": 250
}
```

推荐完整结构：

```json
{
  "task": "bbh",
  "test_samples": 100,
  "test_seed": 42,
  "sampling": "seeded_random_without_replacement",
  "total_population_size": 6750,
  "leaf_population": {
    "bbh_cot_fewshot_boolean_expressions": 250,
    "...": 250
  },
  "selected_count": 100,
  "selected_leaf_docs": [
    {
      "leaf_task": "...",
      "local_index": 1
    }
  ]
}
```

---

# 13. `leaf_population` 的生成

当前 `_selected_lm_eval_tasks()` 已经有：

```python
leaves = list(_leaf_tasks(task_tree))
```

并调用：

```python
build_test_selection(
    {leaf_name: len(task.eval_docs) for leaf_name, task in leaves},
    ...
)
```

因此可以直接让：

```python
build_test_selection()
```

把传入的：

```python
leaf_population
```

原样规范化保存。

建议：

```python
normalized_population = {
    str(name): int(size)
    for name, size in sorted(leaf_population.items())
}
```

---

# 14. `build_test_selection()` 输出必须 deterministic

继续保证：

```python
population = [
    (leaf, index)
    for leaf in sorted(leaf_population)
    for index in range(leaf_population[leaf])
]
```

finite：

```python
random.Random(test_seed).sample(...)
```

然后：

```python
selected.sort()
```

因此只要：

```text
leaf_population
test_samples
test_seed
```

一致，selection 必须逐项一致。

---

# 15. verify exact selection

在：

```python
_validate_tulu_lm_eval_result(...)
```

中新增：

```python
leaf_population = selection.get("leaf_population")
```

强校验：

- 必须是 mapping；
- key 为非空 string；
- value 为非负 integer；
- sum(values) == total_population_size。

然后：

```python
expected_selection = build_test_selection(
    leaf_population,
    task=spec["name"],
    test_samples=spec["test_samples"],
    test_seed=int(spec["test_seed"]),
)
```

比较：

```python
selection["selected_leaf_docs"]
==
expected_selection["selected_leaf_docs"]
```

以及：

```text
sampling
selected_count
total_population_size
```

全部一致。

---

# 16. full evaluation 也要 exact verify

`test_samples: null` 时：

```python
build_test_selection(...)
```

应返回完整 population。

verify 同样逐项比较。

因此 full evaluation 也能证明：

```text
没有漏 leaf
没有漏 doc
没有多 doc
```

---

# 17. `leaf_population` 与真实 lm-eval task 的关系

本轮至少做到：

```text
recorded leaf_population
→ deterministic replay
→ exact selected docs
```

如果希望更严格，可进一步在 verify 时重新加载 pinned TaskManager，重新得到真实：

```python
len(task.eval_docs)
```

再与 recorded `leaf_population` 比较。

但这会让 `verify_artifacts.py` 重新依赖 dataset loading，运行更重。

本轮推荐优先使用：

```text
recorded leaf_population + exact replay
```

作为 artifact integrity/provenance。

---

# 18. 可选增强：selection SHA256

可以额外把：

```text
test_selection.json
```

的 SHA256 写入：

```json
results.json -> snn2_metadata
```

例如：

```json
"test_selection_sha256": "..."
```

这样：

```text
results.json
test_selection.json
```

之间形成直接绑定。

如果实现，注意写文件顺序：

1. 先准备 canonical selection payload；
2. 对 canonical JSON bytes 算 hash，或先写 selection 再 `sha256_file()`；
3. 再写 results metadata。

但这不是本轮必须项，exact replay 已经足够解决主要问题。

---

# 19. ANN 与 SNN selection 一致性

同一 config 下：

```text
Final ANN
Phase SNN
GIF SNN
MTN SNN
```

对于同一个 benchmark 应使用完全相同：

```text
test_seed
test_samples
leaf_population
selected_leaf_docs
```

因为 test subset 应只由 benchmark spec 决定，与 neuron 无关。

建议 `verify_artifacts.py` 在逐 task 校验完成后，再交叉比较四种 evaluation artifact：

```python
ANN selection
Phase selection
GIF selection
MTN selection
```

要求完全一致。

这能避免某一次 SNN evaluation 因任务版本或 accidental state 变化使用不同 subset。

---

# 20. 问题 3：缺失 regression tests

当前已有：

```text
shared calibration independence
group global selection
toy zero-leaf pruning
n-samples alias protection
execution counter delta
CoT conflict
```

但仍缺少几个关键测试。

本轮补齐。

---

# 21. Test：TruthfulQA metric 强校验

新增：

```python
def test_truthfulqa_metric_matches_pinned_protocol():
    ...
```

正确：

```text
metric=acc
```

通过。

错误：

```text
metric=acc_mc1
```

必须：

```python
ValueError
```

---

# 22. Test：六个 benchmark canonical metric

parameterized test：

```python
[
    ("truthfulqa_mc1", "acc"),
    ("mmlu_pro", "exact_match"),
    ("bbh", "exact_match"),
    ("agieval", "acc"),
    ("gsm8k_cot", "exact_match"),
    ("minerva_math", "exact_match"),
]
```

确保 mapping 与 config 一致。

---

# 23. Test：metric result validation

构造 fake lm-eval result：

```python
{"results": {"truthfulqa_mc1": {"acc,none": 0.5}}}
```

应通过：

```text
metric=acc
```

不应通过：

```text
metric=acc_mc1
```

同时测试：

```text
exact_match,custom-extract
exact_match,get-answer
```

等 filter suffix 不影响 canonical metric detection。

---

# 24. Test：exact finite selection replay

例如：

```python
leaf_population = {
    "a": 10,
    "b": 20,
    "c": 30,
}
```

：

```text
test_samples=15
test_seed=42
```

生成 selection。

verify/replay 后必须逐项相等。

然后篡改一条：

```text
("b", 3) -> ("b", 4)
```

数量保持 15、无重复。

验证必须 fail。

---

# 25. Test：full selection replay

：

```text
test_samples=None
```

断言：

```text
selected_count == total_population_size
selected_leaf_docs == full flattened population
```

故意删一个 doc，必须 fail。

---

# 26. Test：leaf_population consistency

故意让：

```json
"total_population_size": 60
```

但：

```json
"leaf_population": {
  "a": 10,
  "b": 20,
  "c": 29
}
```

总计 59。

必须 fail。

---

# 27. Test：ANN/SNN selection equality

构造同一 task 的：

```text
ANN
Phase
GIF
MTN
```

四份 selection。

完全相同 -> pass。

Phase 中改一个 local_index -> fail。

---

# 28. Test：Tulu unaware verify path

虽然当前代码已经修对，但本轮必须补 regression test，避免以后再次破坏。

Tulu unaware：

```yaml
evaluation:
  prefix_enabled: true

calibration:
  num_samples: 128
```

断言：

```text
ann/evaluation/
prefix_enabled_ture/
num_samples_128/
<task>/<spec>/
```

---

# 29. Test：Tulu negative path cases

parameterized：

```text
vanilla Final ANN
phase_aware Final ANN
gif_aware Final ANN
Phase SNN
GIF SNN
MTN SNN
```

都不能在不该出现的位置错误追加：

```text
num_samples_<N>
```

---

# 30. Test：TL;DR 四种 mode path 逐字符 regression

必须新增明确 expected full path：

```text
vanilla
unaware
phase_aware
gif_aware
```

断言和当前 `main` 已有 TL;DR path 完全一致。

额外：

```python
assert "lr_scheduler_type_" not in str(tldr_layout.root)
```

这项是长期硬回归，不只是本轮临时测试。

---

# 31. Test：SNN 第二个 benchmark provenance corruption

当前代码已经逐 task verify，但必须有 test 防止以后退化成只检查第一个。

例如两个 enabled tasks：

```text
truthfulqa_mc1
mmlu_pro
```

第一个完全正确。

第二个：

```text
test_seed
```

故意改错。

ANN / Phase / GIF / MTN parameterized，必须 fail。

---

# 32. Test：真实 selected-task zero-leaf integration

现有：

```python
test_prune_empty_group_leaves()
```

只覆盖纯 dict。

如方便，补一个更贴近：

```python
_selected_lm_eval_tasks()
```

的 integration/helper test：

```text
3 leaves
test_samples=1
```

确认：

- 0-selected leaf 不进入 final tree；
- selected leaf doc_iterator 只返回 selected docs；
- no empty-task path；
- few-shot pool 不被修改。

如果直接加载 lm-eval TaskManager 使 unit test 太慢，可拆出可注入 fake task-tree 的 lower-level helper。

---

# 33. 文件修改范围

建议重点修改：

```text
configs/experiment_matrix.yaml
snn2/lm_eval_protocol.py
scripts/verify_artifacts.py
tests/test_tulu3_lm_eval_protocol.py
tests/test_evaluation_paths.py
实验执行总结.md
```

如果需要实际 result metric validation，也可能修改：

```text
scripts/evaluate_lm_harness.py
```

但不要无必要改 evaluator 的核心 execution path。

不要修改：

```text
snn2/data.py
snn2/artifacts.py
snn2/training.py
scripts/evaluate_tldr.py
```

除非新增 test helper 必须 import；不要重构这些已经正确的部分。

---

# 34. 配置修改

Tulu base config：

```yaml
- name: truthfulqa_mc1
  enabled: true
  num_fewshot: 0
  metric: acc
  cot: false
  test_samples: null
  test_seed: 42
```

其余五个 metric 保持：

```yaml
mmlu_pro:
  metric: exact_match

bbh:
  metric: exact_match

agieval:
  metric: acc

gsm8k_cot:
  metric: exact_match

minerva_math:
  metric: exact_match
```

---

# 35. `实验执行总结.md` 更新

只更新 Tulu lm-eval 部分。

明确六个 canonical metric：

```text
TruthfulQA MC1 -> acc
MMLU-Pro       -> exact_match
BBH            -> exact_match
AGIEval        -> acc
GSM8K-CoT      -> exact_match
Minerva MATH   -> exact_match
```

并说明：

> `metric` 与 `cot` 一样，均是对固定 lm-eval 0.4.8 revision 的 project-side audited semantic validation；二者不作为自定义参数改变 task，而是用来防止配置与 pinned task 定义漂移。

同时说明：

> `test_selection.json` 保存 leaf population，并可由 `test_seed/test_samples` deterministic replay；verify 会逐项检查 exact selected documents。

TL;DR 文档部分不改。

---

# 36. 关于 TruthfulQA “0-shot”的说明

不要修改 pinned TruthfulQA task prompt。

当前：

```text
num_fewshot=0
```

仍按 lm-eval 定义称为：

```text
0-shot
```

但该 pinned task 的 `doc_to_text` 本身包含固定示例 prompt。

这不是代码错误。

论文/文档若需要严谨，可表述：

> TruthfulQA MC1 uses lm-eval's pinned 0-shot configuration (`num_fewshot=0`).

不要表述为：

> prompt 中没有 demonstration

除非后续另行定义自定义 TruthfulQA task。

---

# 37. 推荐实施顺序

## Phase A：metric protocol

1. 修 `truthfulqa_mc1.metric: acc`；
2. 新增 `LM_EVAL_0_4_8_TASK_METRIC`；
3. `validate_lm_eval_task_specs()` 强校验 metric；
4. 补 metric result existence validation；
5. 补 metric tests。

## Phase B：selection provenance

6. `build_test_selection()` 保存 `leaf_population`；
7. verify 校验 leaf population；
8. deterministic replay exact selection；
9. 可选 ANN/SNN cross-selection equality；
10. 补 corruption tests。

## Phase C：regression coverage

11. unaware verify path test；
12. vanilla/aware/SNN negative path tests；
13. TL;DR four-mode exact path tests；
14. SNN second-task corruption tests；
15. zero-leaf integration test。

## Phase D：文档和验收

16. 更新 `实验执行总结.md`；
17. materialize configs；
18. `pytest -q`；
19. 轻量 lm-eval smoke test。

---

# 38. 最终验收 checklist

## Hard constraints

- [ ] 仅 Tulu-3 protocol/verify/tests 有变化
- [ ] TL;DR path 逐字符不变
- [ ] TL;DR computation semantics 不变
- [ ] shared pool / calibration / Prefix 不被改动

## Metric

- [ ] TruthfulQA metric = `acc`
- [ ] MMLU-Pro = `exact_match`
- [ ] BBH = `exact_match`
- [ ] AGIEval = `acc`
- [ ] GSM8K-CoT = `exact_match`
- [ ] Minerva MATH = `exact_match`
- [ ] metric mismatch 在 model load 前 fail
- [ ] real result payload 中存在对应 canonical metric

## Selection provenance

- [ ] `test_selection.json` 记录 `leaf_population`
- [ ] `sum(leaf_population.values()) == total_population_size`
- [ ] finite selection 可 deterministic replay
- [ ] full selection 可 deterministic replay
- [ ] exact `selected_leaf_docs` 被验证
- [ ] 替换一个合法 local_index 也会 fail
- [ ] ANN / Phase / GIF / MTN 使用同一 task subset（若实现 cross-check）

## Existing third-round behavior

- [ ] unaware verify path 正确
- [ ] per-task execution counter 正确
- [ ] ANN / Phase / GIF / MTN 每个 enabled task 均逐项 verify
- [ ] `n-samples.effective` 正确
- [ ] group zero-leaf pruning 正确

## Regression tests

- [ ] metric conflict test
- [ ] six-task metric mapping test
- [ ] metric result detection test
- [ ] finite exact-selection corruption test
- [ ] full exact-selection corruption test
- [ ] leaf_population consistency test
- [ ] Tulu unaware path regression
- [ ] Tulu vanilla/aware/SNN negative path regression
- [ ] TL;DR four-mode exact path regression
- [ ] SNN non-first-task corruption regression
- [ ] zero-leaf integration coverage
- [ ] `pytest -q` 全部通过

---

# 39. 本轮不要修改的内容

不要修改：

- TL;DR 路径；
- TL;DR 数据；
- TL;DR evaluation；
- Tulu shared training pool；
- Tulu ANN train subset；
- Tulu shared calibration；
- Prefix discovery；
- rotation；
- common Clip；
- Phase/GIF/MTN neuron；
- ANN/SNN forward semantics；
- scheduler / warmup；
- six benchmark task names；
- num_fewshot；
- CoT mapping；
- group `test_samples=N` 全局 N 的定义；
- zero-leaf pruning；
- per-task execution counter。

---

# 40. 最终原则

本轮修改后，Tulu lm-eval protocol 应满足：

> **task name、num_fewshot、CoT semantic、metric semantic、test subset 和 lm-eval revision 都能够被明确审计，且配置不能与 pinned lm-eval 0.4.8 真实任务定义漂移。**

同时：

> **`test_seed` 不再只是 metadata；verify 能 deterministic replay 并确认 exact selected documents。**

并继续保持：

> **TL;DR summarization 的所有现有路径与计算逻辑完全不变。**
