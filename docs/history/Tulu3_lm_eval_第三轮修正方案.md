# Tulu-3 lm-eval 第三轮修正方案

> 目标仓库：`/home/wangwenkang/SNN`  
> GitHub：`https://github.com/wangwk699/SNN`  
> 基准：当前 `main`，提交 `4971925f161abb536bc77b814d61579785ba484a`  
> 本文档用于部署在服务器上的 Codex 在**没有此前对话上下文**的情况下完成本轮修正。

---

# 1. 本轮修改范围

上一轮第二轮修正后，以下核心逻辑已经正确，不要重复重构：

- Tulu-3 fixed validation -> shared training pool；
- shared calibration 与 concrete ANN `training.train_samples` 解耦；
- `_shared` 路径不含 `training.train_samples/train_seed`；
- concrete ANN run path 保留 `train_samples`；
- Tulu vanilla/unaware 已加入 scheduler + warmup path；
- group benchmark `test_samples=N` 使用顶层 benchmark 全局 N；
- zero-selected leaf 已通过 task-tree pruning 处理；
- lm-eval `n-samples.effective` 已加入项目侧修正；
- `cot` 仍只作为 pinned lm-eval task 语义校验，不传给 `simple_evaluate()`。

本轮只修以下 3 个问题：

1. **Tulu unaware Final ANN 的 `verify_artifacts.py` required-path 仍然走旧路径。**
2. **每个 benchmark 的 execution counter 目前错误地记录为全部 enabled benchmarks 的累计总数。**
3. **Tulu SNN 的每个 benchmark 还没有逐 task 做完整 lm-eval provenance validation。**

同时补齐缺失的 regression tests。

---

# 2. 最高优先级硬约束：TL;DR 零改动

本轮所有修改必须严格限定在：

```python
cfg["experiment"]["task"] == "tulu3"
```

或 Tulu 专用 lm-eval helper。

TL;DR summarization 的：

- data semantics；
- `evaluate_tldr.py`；
- ANN/SNN forward；
- calibration；
- Prefix；
- run path；
- evaluation path；
- verify path；
- result metadata；
- selection metadata；

全部保持当前 `main` **逐字符/逐语义不变**。

尤其：

> 不允许因为统一 `evaluation_paths()`、execution-counter helper 或 provenance helper 而改变 TL;DR 的保存路径。

---

# 3. 问题 1：Tulu unaware Final ANN verify required-path 仍错误

## 3.1 当前问题

当前 evaluator：

```python
output_root = model_output_dir / "evaluation"

if not args.base:
    output_root = output_root / prefix_enabled_dirname(active_prefix_enabled)

output_root = append_evaluation_num_samples_if_needed(
    output_root,
    cfg,
    base=args.base,
    rotated_pre_finetuning=args.rotated_pre_finetuning,
    neuron=args.neuron,
)
```

因此 Tulu unaware Final ANN + Prefix enabled 时，实际结果路径为：

```text
.../ann/evaluation/
prefix_enabled_ture/
num_samples_128/
<task>/
<num_fewshot...>/
results.json
```

但是当前 `scripts/verify_artifacts.py` 中 `evaluation_paths()` 的 Tulu 分支仍然：

```python
directory = root / "evaluation"
directory = directory / prefix_enabled_dirname(enabled)

return [
    directory
    / safe_name(spec["name"])
    / lm_eval_spec_dirname(spec)
    / filename
    ...
]
```

这里遗漏：

```python
append_evaluation_num_samples_if_needed(...)
```

所以 verify 的第一个 existence check 会先去错误目录：

```text
.../ann/evaluation/
prefix_enabled_ture/
mmlu_pro/...
```

并在后面的正确 provenance validation 运行前直接：

```python
raise FileNotFoundError(...)
```

---

# 4. 问题 1 的正确修法

不要在 verify 里维护两套 Tulu evaluation path 逻辑。

直接修：

```python
evaluation_paths(...)
```

的 Tulu 分支，使其与 evaluator 使用同一个 helper。

推荐结构：

```python
def evaluation_paths(root, *, neuron="ann"):
    directory = root / "evaluation"

    if task == "tldr":
        # 现有 TL;DR 逻辑完全不动
        ...
        return ...

    enabled = (
        final_ann_evaluation_prefix_enabled(cfg)
        if neuron == "ann"
        else evaluation_prefix_enabled(cfg)
    )

    directory = directory / prefix_enabled_dirname(enabled)

    directory = append_evaluation_num_samples_if_needed(
        directory,
        cfg,
        neuron=neuron,
    )

    return [
        directory
        / safe_name(spec["name"])
        / lm_eval_spec_dirname(spec)
        / filename
        for spec in enabled_lm_eval_task_specs(cfg)
        for filename in evaluation_files
    ]
```

注意：

- 对当前 `verify_artifacts.py` 的正常 Final ANN / SNN verification，不需要 `base=True` 或 `rotated_pre_finetuning=True`。
- SNN `neuron != "ann"` 时，`append_evaluation_num_samples_if_needed()` 本身应返回原 directory，不额外添加 num_samples。
- unaware Final ANN 会正确添加：
  ```text
  num_samples_<N>
  ```
- aware Final ANN 不应错误添加这层。
- vanilla Final ANN 不应添加这层。

---

# 5. 删除 verify 中重复构造的第二套 Tulu ANN eval root

当前后面还有：

```python
if task == "tulu3":
    enabled = final_ann_evaluation_prefix_enabled(cfg)
    eval_root = layout.ann_dir / "evaluation" / prefix_enabled_dirname(enabled)
    eval_root = append_evaluation_num_samples_if_needed(
        eval_root,
        cfg,
        neuron="ann",
    )
    ...
```

修完 `evaluation_paths()` 后，不建议继续保留这套重复路径构造。

应改为统一使用一个 helper，例如新增：

```python
def tulu_eval_task_root(root, spec, *, neuron="ann"):
    ...
```

或者让 `evaluation_paths()` 支持返回 task root。

目标：

> evaluator 和 verifier 的 Tulu path decision 只能有一套 canonical 条件判断。

避免以后再次发生：

```text
evaluator path != verify path
```

---

# 6. 问题 2：per-task execution counter 当前是错误的累计总数

当前 `scripts/evaluate_lm_harness.py`：

```python
for spec in task_specs:
    task_result = simple_evaluate(...)
    task_results[spec["name"]] = ...

execution_counter = dict(proxy.execution_counter)

common_snn2_metadata = {
    ...
    "execution_counter": execution_counter,
    ...
}
```

然后：

```python
for spec in task_specs:
    result = {
        ...
        "snn2_metadata": {
            **common_snn2_metadata,
            ...
        }
    }
```

这意味着：

```text
truthfulqa_mc1/results.json
mmlu_pro/results.json
bbh/results.json
agieval/results.json
gsm8k_cot/results.json
minerva_math/results.json
```

都得到同一份：

```text
所有 enabled benchmark 累计 execution_counter
```

而不是各 benchmark 自己的计数。

---

# 7. execution counter 的正确语义

每个独立 benchmark 的 `results.json` 应只记录**该 benchmark 本身**产生的：

```text
model_forward_calls
temporal_model_step_forwards
sample_forward_equivalents
temporal_sample_step_forwards
batched_sample_slots
batched_temporal_sample_slots
activation_site_temporal_operator_calls
batched_activation_site_temporal_slots
```

例如：

```text
mmlu_pro/results.json
```

不能包含：

```text
truthfulqa_mc1 + mmlu_pro + bbh + ...
```

的累计值。

---

# 8. 推荐实现：before/after delta

在每个 task evaluate 前：

```python
before_counter = dict(proxy.execution_counter)
```

执行：

```python
task_result = simple_evaluate(...)
```

后：

```python
after_counter = dict(proxy.execution_counter)
```

然后计算：

```python
def execution_counter_delta(before, after):
    keys = set(before) | set(after)
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
    }
```

得到：

```python
task_counter
```

将：

```python
task_results[spec["name"]]
```

保存为：

```python
(
    task_result,
    test_selection,
    task_counter,
)
```

而不是只保存：

```python
(task_result, test_selection)
```

---

# 9. per-task operator metadata

对每个 benchmark 根据：

```python
task_counter
```

计算：

```python
temporal_sample_step_forwards = task_counter.get(
    "temporal_sample_step_forwards", 0
)

batched_temporal_sample_slots = task_counter.get(
    "batched_temporal_sample_slots", 0
)

activation_site_temporal_operator_calls = (
    temporal_sample_step_forwards * per_forward_operators
)

batched_activation_site_temporal_slots = (
    batched_temporal_sample_slots * per_forward_operators
)
```

这些字段必须放到该 benchmark 自己的 metadata。

---

# 10. common metadata 与 task-specific metadata 分离

建议将当前：

```python
common_snn2_metadata
```

拆成两部分：

## 10.1 真正 common 的 metadata

例如：

```text
model_variant
model_source
model_revision
neuron
full_temporal_steps
site_count
site_topology_version
per_temporal_forward_activation_neuron_operators
global_final_norm...
ann_training_common_clip...
deployment policy
batch_size
prefix...
calibration...
evaluation forward...
```

这些可以共用。

## 10.2 每个 task 独立 metadata

每个 task 单独加入：

```text
execution_counter
activation_site_temporal_operator_calls
batched_activation_site_temporal_slots

lm_eval_task_spec
lm_eval_revision
cot_semantic_source
test_sampling
actual_test_samples
fewshot_random_seed
test_seed
```

避免未来把 task-specific 统计误放到 common metadata。

---

# 11. 是否需要保存全局累计 counter

不是必须。

如果希望保留整次 evaluate script 的总计，可以另存一个：

```text
run summary
```

或日志 event，例如：

```python
run.event(
    "lm_eval_all_tasks_complete",
    execution_counter=dict(proxy.execution_counter),
)
```

但不要把总计伪装成每个 benchmark 的独立 metadata。

本轮最重要的是：

> 每个 `results.json` 内的 execution counter 必须是 task-local delta。

---

# 12. 问题 3：SNN 每个 benchmark provenance 仍未完整验证

当前 Final ANN Tulu 有：

```python
_validate_tulu_lm_eval_result(
    results_path,
    selection_path,
    spec,
    experiment_seed=...
)
```

逐 benchmark 检查。

但 Phase/GIF/MTN SNN 当前主要只是：

```python
required.extend(evaluation_paths(...))
```

确认文件存在。

后续每个 neuron 又只使用：

```python
evaluation_paths(...)[0]
```

即只读取第一个 benchmark 的：

```text
results.json
```

做 forward metadata / temporal policy 验证。

所以其它 benchmark 即使：

- `test_selection.json` 错；
- test seed 错；
- selected_count 错；
- results metadata 是旧版本；
- `lm_eval_revision` 错；

只要文件存在，verify 仍可能通过。

---

# 13. 正确目标：ANN + 三类 SNN 都逐 task 验证

Tulu-3 应对：

```text
Final ANN
Phase SNN
GIF SNN
MTN SNN
```

每个 enabled benchmark 都执行：

```python
_validate_tulu_lm_eval_result(...)
```

即总的验证结构：

```python
evaluation_roots = [
    ("ann", layout.ann_dir),
    ("phase", layout.snn_dir("phase")),
    ("gif", layout.snn_dir("gif")),
    ("mtn", layout.snn_dir("mtn")),
]

for neuron, root in evaluation_roots:
    for spec in enabled_lm_eval_task_specs(cfg):
        result_root = ...
        _validate_tulu_lm_eval_result(...)
```

---

# 14. 推荐新增 canonical task-root helper

为了避免：

```text
required files
provenance validation
forward metadata validation
```

分别重写路径逻辑，建议新增 Tulu-only helper：

```python
def _tulu_lm_eval_task_root(
    cfg,
    root,
    spec,
    *,
    neuron,
):
    enabled = (
        final_ann_evaluation_prefix_enabled(cfg)
        if neuron == "ann"
        else evaluation_prefix_enabled(cfg)
    )

    directory = root / "evaluation"
    directory = directory / prefix_enabled_dirname(enabled)

    directory = append_evaluation_num_samples_if_needed(
        directory,
        cfg,
        neuron=neuron,
    )

    return (
        directory
        / safe_name(spec["name"])
        / lm_eval_spec_dirname(spec)
    )
```

然后统一用于：

```text
results.json existence
test_selection.json existence
provenance validation
forward metadata validation
```

只用于 Tulu。

TL;DR 继续使用当前原逻辑，不迁移到这个 helper。

---

# 15. Forward metadata 验证仍可只选一个 task，但要明确语义

同一：

```text
model variant + neuron
```

下，每个 benchmark 的：

```text
evaluation_forward_kind
controller_mode
temporal_execution
replacement state
deployment policy
```

理论上相同。

因此：

```python
_verify_final_ann_forward_metadata(...)
_validate_snn_forward_metadata(...)
_validate_snn_source_metadata(...)
```

可以继续只选择第一个 enabled benchmark 做**模型执行图级验证**。

但：

> lm-eval task provenance / selection provenance 必须逐 benchmark 全部验证。

也就是说区分：

```text
model-forward invariant metadata
```

和：

```text
task-specific provenance
```

不要混为一谈。

---

# 16. 加强 `n-samples.effective` provenance verification

当前 evaluator 已调用：

```python
correct_effective_sample_counts(
    task_result,
    test_selection,
)
```

但 verify 还没有确认：

```text
n-samples.effective
```

真的与 `test_selection.json` 一致。

建议补验证。

---

# 17. `n-samples` 验证规则

从：

```json
test_selection.json
```

构造：

```python
selected_counts_by_leaf = Counter(
    item["leaf_task"]
    for item in selected_leaf_docs
)
```

然后解析 `results.json` 中当前 task 的 lm-eval raw result：

当前保存结构类似：

```json
{
  "tasks": {
    "<top_task_name>": {
      "n-samples": {...},
      ...
    }
  },
  "snn2_metadata": {...}
}
```

按真实结构检查。

对每个实际参与 leaf：

```text
n-samples[leaf].effective == selected_counts_by_leaf[leaf]
```

对 full evaluation：

```text
effective == original
```

对 finite group subset：

0-selected leaf 已被 prune，不应要求它出现在 `n-samples`。

---

# 18. `correct_effective_sample_counts()` 本身再核一个边界

当前：

```python
for leaf_name, count in counts.items():
    effective = len(selected.get(leaf_name, set()))
    ...
```

需要实际确认 pinned lm-eval 返回的：

```text
task_result["n-samples"]
```

key 与：

```text
selection["selected_leaf_docs"][].leaf_task
```

完全一致。

如果 group result 里还有：

```text
top-level group alias
```

或非 leaf key，不要把它错误改成：

```text
effective = 0
```

建议改成：

```python
if leaf_name not in selected:
    continue
```

而不是无条件：

```python
effective = 0
```

除非已经确认 `n-samples` 只包含实际 leaf task。

这个属于本轮必须核实的 implementation detail。

---

# 19. `prune_empty_selected_leaves()` 也要加真实 integration test

当前 toy test：

```python
tree = {
    "group": {
        "a": object(),
        "b": object(),
        "c": object(),
    }
}
```

只证明纯 dict pruning。

还需要测试：

```python
_selected_lm_eval_tasks(...)
```

或拆出一个可注入 fake TaskManager 的 lower-level helper。

必须证明：

```text
test_samples=1
```

时：

- 0-selected leaf 不进入 final task tree；
- selected leaf 的 `doc_iterator` 正确；
- no KeyError；
- no empty instances；
- few-shot docs 没有被修改。

---

# 20. 本轮必须补的 regression tests

至少补以下测试。

## 20.1 unaware verify path

构造 Tulu unaware：

```yaml
evaluation:
  prefix_enabled: true

calibration:
  num_samples: 128
```

断言：

```text
evaluator expected root
==
verify expected root
```

且包含：

```text
prefix_enabled_ture/num_samples_128/
```

---

## 20.2 vanilla/aware path negative cases

确保：

```text
vanilla Final ANN
```

不会错误加：

```text
num_samples_128
```

确保：

```text
phase_aware Final ANN
gif_aware Final ANN
```

不会因为本轮 helper 重构错误加这层。

---

## 20.3 SNN path negative case

Phase/GIF/MTN SNN evaluation 不因 unaware Prefix dependency 而错误追加：

```text
num_samples_<N>
```

到 evaluation result root。

SNN 自己的 calibration variant 已在既有 SNN run path 中处理，不在 evaluation 层重复追加。

---

## 20.4 per-task execution counter delta

fake proxy counter：

Task A 前：

```python
{}
```

Task A 后：

```python
{
    "model_forward_calls": 10,
    "temporal_sample_step_forwards": 20,
}
```

Task B 后：

```python
{
    "model_forward_calls": 17,
    "temporal_sample_step_forwards": 35,
}
```

断言：

```text
Task A counter = 10 / 20
Task B counter = 7 / 15
```

而不是两个 task 都保存：

```text
17 / 35
```

---

## 20.5 operator metadata

根据 per-task counter 验证：

```text
activation_site_temporal_operator_calls
batched_activation_site_temporal_slots
```

使用 task-local counter，而不是 global counter。

---

## 20.6 ANN per-task provenance

至少两个 enabled benchmark：

- task A 正确；
- task B `test_seed` 故意错误；

verify 必须 fail。

---

## 20.7 SNN per-task provenance

对 Phase/GIF/MTN 分别至少做一个 parameterized test：

- 第一个 task 正确；
- 第二个 task metadata 故意错误；

verify 必须 fail。

重点证明：

> verify 不再只检查 `[0]` 第一个 benchmark。

---

## 20.8 `n-samples.effective`

single task：

```text
original=100
selected=20
effective=20
```

group：

```text
leaf A selected 3
leaf B selected 7
```

必须：

```text
A.effective=3
B.effective=7
```

---

## 20.9 TL;DR path regression

上一轮虽然要求过，但当前测试覆盖仍不足。

必须补完整四 mode TL;DR expected root regression：

```text
vanilla
unaware
phase_aware
gif_aware
```

并明确断言：

```python
"lr_scheduler_type_" not in str(layout.root)
```

本轮任何 helper 重构不能改变这些字符串。

---

# 21. 当前 toy pruning test 保留，但不能作为唯一 coverage

保留：

```python
test_prune_empty_group_leaves()
```

作为 unit test。

但必须再补：

```text
真实 selected-task construction
```

层级的 integration test。

---

# 22. 测试 shared training pool 不需要再重复改算法

上一轮已经正确实现并已有测试：

```text
validation indices 相同
shared calibration indices 相同
ANN train indices 不同
```

本轮不要再修改该算法。

只需确保新的代码改动没有破坏既有 tests。

---

# 23. metadata 字段名清理：非阻断可顺手修

当前 Tulu shared calibration manifest 已从 shared pool 采样，但字段仍叫：

```text
positions_in_selected_train
```

对 Tulu 语义已经不准确。

可选修正为 Tulu-only：

```text
positions_in_selection_pool
```

或者同时保留兼容字段：

```json
{
  "positions_in_selection_pool": [...],
  "selection_pool": "validation_excluded_shared_training_pool"
}
```

但：

- 不要改变 TL;DR 字段；
- 不要因此改变已有 calibration calculation；
- 这不是正式运行前的阻断项。

如果改动会扩大兼容风险，本轮可以不改。

---

# 24. `实验执行总结.md`

本轮只需要小幅补充 Tulu lm-eval verify/metadata 说明。

建议写明：

> Tulu-3 每个 enabled benchmark 独立保存 results/test_selection，并独立记录该 benchmark 的 execution counter。`verify_artifacts.py` 会逐 benchmark 验证 task spec、lm-eval revision、CoT semantic、test seed、sampling selection 与 selected count；ANN 与 Phase/GIF/MTN SNN 均执行同一套 task-specific provenance validation。

TL;DR 文档部分不要改。

---

# 25. 推荐修改文件

重点：

```text
scripts/verify_artifacts.py
scripts/evaluate_lm_harness.py
snn2/lm_eval_protocol.py
tests/test_tulu3_lm_eval_protocol.py
tests/test_evaluation_paths.py
实验执行总结.md
```

如果能通过现有 helper 完成，不需要修改：

```text
snn2/data.py
snn2/artifacts.py
snn2/training.py
```

特别是：

> 不要为了这轮问题再次重构已经正确的 Tulu shared pool 和 ArtifactLayout。

---

# 26. 推荐实施顺序

## Phase A：verify path

1. 修 Tulu `evaluation_paths()`；
2. 统一 task-root path helper；
3. 删除/减少重复路径构造；
4. 补 unaware/vanilla/aware/SNN path tests。

## Phase B：execution counter

5. 增加 counter delta helper；
6. 每个 benchmark 前后 snapshot；
7. task-local counter 写入对应 results；
8. task-local operator metadata；
9. 补 counter tests。

## Phase C：SNN provenance

10. ANN + phase + gif + mtn 全部逐 enabled task 调 `_validate_tulu_lm_eval_result()`；
11. 保留第一个 task 做 model-forward invariant validation；
12. 增加 `n-samples.effective` 与 selection 一致性验证；
13. 补多 task corruption tests。

## Phase D：integration regression

14. 补真实 zero-leaf selected-task construction test；
15. 补 TL;DR exact-path regression；
16. 更新 Tulu 文档；
17. `pytest -q`。

---

# 27. 最终验收 checklist

## Hard constraints

- [ ] 本轮只影响 Tulu-3
- [ ] TL;DR path 逐字符不变
- [ ] TL;DR computation semantics 不变
- [ ] Tulu shared pool/calibration 逻辑不被再次改坏

## Verify path

- [ ] unaware Final ANN verify 正确包含 `num_samples_<N>`
- [ ] vanilla Final ANN 不含该层
- [ ] aware Final ANN 不错误添加该层
- [ ] SNN evaluation 不错误添加该层
- [ ] required-file check 与 provenance check 使用同一 canonical task root

## Per-task execution metadata

- [ ] 每个 benchmark 保存独立 counter delta
- [ ] 不再保存所有 benchmarks 的累计 counter
- [ ] operator-call metadata 使用 task-local counter
- [ ] global cumulative counter 如保留，只进入 run-level log/summary

## Per-task provenance

- [ ] Final ANN 每个 enabled task 验证
- [ ] Phase SNN 每个 enabled task 验证
- [ ] GIF SNN 每个 enabled task 验证
- [ ] MTN SNN 每个 enabled task 验证
- [ ] disabled task 不要求结果存在
- [ ] wrong spec/test_seed/revision/selection 会 fail

## `n-samples`

- [ ] single task effective count 正确
- [ ] group leaf effective count 正确
- [ ] full evaluation effective == original
- [ ] zero-selected pruned leaf 不制造虚假 `effective=0` metadata

## Tests

- [ ] unaware verify path regression
- [ ] vanilla/aware/SNN negative path regression
- [ ] counter delta test
- [ ] operator metadata test
- [ ] ANN multi-task provenance corruption test
- [ ] SNN multi-task provenance corruption test
- [ ] real zero-leaf integration test
- [ ] TL;DR four-mode exact path regression
- [ ] `pytest -q` 全部通过

---

# 28. 本轮不要修改的内容

不要修改：

- TL;DR path；
- TL;DR data；
- `evaluate_tldr.py`；
- Tulu shared training pool 采样算法；
- Tulu shared calibration 采样算法；
- Prefix discovery；
- rotation；
- calibration 数学；
- common Clip；
- Phase/GIF/MTN neuron；
- ANN/SNN forward semantics；
- lm-eval task 列表；
- CoT mapping；
- num_fewshot 配置；
- GSM8K 4-shot；
- `test_samples=N` 顶层 group 全局 N 的定义；
- zero-leaf pruning 的既有正确语义；
- scheduler 当前值；
- warmup 当前值；
- Tulu model identity。

---

# 29. 最终原则

本轮修正的目标不是再改实验协议，而是把已经确定的协议完整落实到：

```text
path verification
per-task runtime metadata
per-task provenance verification
```

最终必须满足：

> **Tulu-3 每个 enabled benchmark 都是一个独立、可审计的 evaluation artifact：路径与 evaluator 一致，selection/provenance 可验证，execution counter 只代表该 benchmark 本身。**

同时继续满足：

> **TL;DR summarization 的现有路径和计算逻辑完全不变。**
