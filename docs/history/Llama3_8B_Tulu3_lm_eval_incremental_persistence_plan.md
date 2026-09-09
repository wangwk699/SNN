# Llama3-8B / Tulu3 lm-eval 逐任务落盘与 `log_samples=False` 修改方案

## 1. 修改目标

本轮仅优化 **Llama3-8B / Tulu3 的 lm-eval 评估结果生命周期与内存占用**，不修改任何 SNN 数学逻辑、任务定义、数据采样、metric 计算、Prefix、conversion、Temporal Phase/GIF/MTN forward 或 distributed document data parallel 逻辑。

当前 `scripts/evaluate_lm_harness.py` 的行为是：

1. 对每个 enabled task 依次调用 `simple_evaluate(...)`；
2. 每个 task 完成后，将完整 `task_result` 暂存在：
   ```python
   task_results[spec["name"]] = (
       task_result,
       test_selection,
       task_counter,
       timing,
   )
   ```
3. 等所有任务都完成后，才统一将每个 task 的：
   - `results.json`
   - `test_selection.json`
   写到本地；
4. 最后写：
   - `evaluation_summary.json`
5. 当前没有显式传 `log_samples=False`，因此 pinned lm-eval 默认：
   ```python
   log_samples=True
   ```
   会在 `task_result["samples"]` 中保留逐样本 `doc / target / response / metrics / hashes` 等内容。

本轮目标改成：

```text
Task 1
  ↓
simple_evaluate(log_samples=False)
  ↓
立即校验 metric
  ↓
立即写 results.json
  ↓
立即写 test_selection.json
  ↓
更新 task_metrics / task_times
  ↓
原子更新 evaluation_summary.json
  ↓
释放 task_result
  ↓
barrier

Task 2
  ↓
...
```

最终要求：

- 不再保存逐样本结果；
- 不再把所有 task 的完整 `task_result` 长期积压在 rank 0 CPU 内存中；
- 每个 task 完成后立即写盘；
- 即使后续 task 失败，前面已完成 task 的结果仍然保留；
- `evaluation_summary.json` 必须继续存在；
- `evaluation_summary.json` 继续记录：
  ```json
  {
    "task_times": {...},
    "task_metrics": {...}
  }
  ```
- 在每个 task 完成后，就更新一次 `evaluation_summary.json`；
- 所有 task 完成后，再写一次最终 summary；
- 不调用 `torch.cuda.empty_cache()`。

---

## 2. 修改范围

### 必须修改

```text
scripts/evaluate_lm_harness.py
```

### 建议增加/修改测试

优先考虑：

```text
tests/test_lm_eval_distributed.py
```

如果现有测试结构不适合测试脚本级结果生命周期，可新增：

```text
tests/test_lm_eval_incremental_persistence.py
```

名称可调整，但测试应保持：

- CPU-only；
- 不下载真实 8B 模型；
- 不实际运行完整 lm-eval benchmark；
- 使用 monkeypatch / fake result / tmp_path 验证结果持久化逻辑。

### 不应修改

除非测试明确暴露现有 bug，否则不要修改：

```text
snn2/model_integration.py
snn2/evaluation.py
snn2/controller.py
snn2/neurons.py
snn2/conversion.py
snn2/lm_eval_distributed.py
snn2/lm_eval_protocol.py
snn2/modeling.py
configs/experiment_matrix.yaml
configs/generated/*
```

也不要修改：

- Phase / GIF / MTN neuron 定义；
- temporal layout；
- Prefix KV；
- task sampling；
- num_fewshot；
- test_seed；
- `correct_effective_sample_counts(...)`；
- `extract_metric_value(...)`；
- distributed counter aggregation；
- output path schema。

---

# 3. 当前代码中需要解决的两个问题

## 3.1 `log_samples=True` 导致逐样本结果进入 `task_result`

当前调用类似：

```python
task_result = simple_evaluate(
    model=harness_model,
    tasks=[spec["name"]],
    task_manager=task_manager,
    num_fewshot=int(spec["num_fewshot"]),
    batch_size=batch_size,
    limit=None,
    random_seed=int(cfg["experiment"]["seed"]),
    numpy_random_seed=int(cfg["experiment"]["seed"]),
    torch_random_seed=int(cfg["experiment"]["seed"]),
    fewshot_random_seed=int(cfg["experiment"]["seed"]),
    apply_chat_template=bool(
        cfg["evaluation"].get("apply_chat_template", True)
    ),
)
```

由于没有传 `log_samples`，pinned lm-eval 默认：

```python
log_samples=True
```

这会导致：

```python
task_result["samples"]
```

包含逐样本详细结果。

本轮要求显式设置：

```python
log_samples=False
```

### 作用范围

本轮只要求 Tulu3 lm-eval 使用该行为。

推荐写法：

```python
log_samples=False if is_tulu_lm_eval else True
```

这样可以尽量不改变脚本对未来其他非 Tulu3 harness 任务的历史行为。

如果确认 `evaluate_lm_harness.py` 实际只用于当前 Tulu3 协议，也可直接：

```python
log_samples=False
```

但优先推荐前一种“只收敛 Tulu3 行为”的写法。

当前 Tulu3 enabled tasks 为：

```text
truthfulqa_mc1
agieval
arc_challenge
piqa
winogrande
boolq
```

其配置 metric 为：

```text
acc
acc_norm
```

不存在当前已启用的 `bypass` metric task，因此可以安全运行 `log_samples=False`。

注意：

pinned lm-eval 自身对于要求 sample logging 的 `bypass` task 会在 `log_samples=False` 时主动报错，因此不要绕过该保护。

---

# 4. 删除长期 `task_results` 缓存

当前代码：

```python
task_results: dict[str, tuple] = {}
```

以及：

```python
task_results[spec["name"]] = (
    task_result,
    test_selection,
    task_counter,
    timing,
)
```

全部删除。

不再保留：

```text
task 1 完整 task_result
task 2 完整 task_result
task 3 完整 task_result
...
```

直到所有任务结束。

### 新的长期状态只保留

对于 Tulu3：

```python
task_times: dict[str, str] = {}
task_metrics: dict[str, object] = {}
```

这两个 dict 只保存非常小的 aggregate 信息。

例如：

```python
task_times = {
    "truthfulqa_mc1": "00:12:34",
    "agieval": "00:08:51",
}
```

```python
task_metrics = {
    "truthfulqa_mc1": 0.37,
    "agieval": 0.41,
}
```

不要在这两个 dict 中放：

- samples；
- prompts；
- responses；
- logits；
- task_result 全对象；
- test_selection 全对象。

---

# 5. 为了逐任务立即写盘，需要重排当前代码顺序

这是本轮最重要的结构调整。

当前代码是：

```text
A. 初始化模型 / harness
B. 跑完所有 task，保存 task_results
C. 计算 common metadata
D. 计算 output_root
E. 遍历 task_results 统一写盘
F. 写 evaluation_summary
```

要改成：

```text
A. 初始化模型 / harness
B. 提前计算 common metadata
C. 提前计算 output_root
D. 初始化 task_times / task_metrics
E. 对每个 task：
      1. evaluate
      2. validate
      3. build result payload
      4. write task files
      5. update task_times / task_metrics
      6. atomic rewrite evaluation_summary
      7. release task_result
      8. barrier
F. 全部 task 完成后再写最终 evaluation_summary
```

也就是说，以下当前位于 task loop 之后的公共计算需要移动到 task loop 之前：

```python
layers = int(
    getattr(
        model.config,
        "num_hidden_layers",
    )
)

per_forward_operators = (
    activation_neuron_operators_per_temporal_forward(
        num_hidden_layers=layers,
        neuron=args.neuron,
    )
)
```

以及：

```python
common_snn2_metadata = {...}
```

以及 output root 计算：

```python
if args.base:
    model_output_dir = layout.base_dir
elif args.rotated_pre_finetuning:
    model_output_dir = layout.rotated_pre_finetuning_dir
elif args.neuron == "ann":
    model_output_dir = layout.ann_dir
else:
    model_output_dir = layout.snn_dir(args.neuron)

output_root = model_output_dir / "evaluation"

if not args.base:
    output_root = (
        output_root
        / prefix_enabled_dirname(active_prefix_enabled)
    )

output_root = append_evaluation_num_samples_if_needed(
    output_root,
    cfg,
    base=args.base,
    rotated_pre_finetuning=args.rotated_pre_finetuning,
    neuron=args.neuron,
)
```

以及：

```python
task_results_root = (
    output_root / "task_results"
    if is_tulu_lm_eval
    else output_root
)
```

这些值都不依赖某个 task 的实际输出，因此可以安全提前计算。

---

# 6. 新的 Tulu3 task loop 结构

建议将现有循环重构为如下语义。

注意：下面是结构示意，Codex 应结合当前文件实际变量名和已有 metadata 字段完整保留，不要机械覆盖。

```python
task_specs = enabled_lm_eval_task_specs(cfg)
is_tulu_lm_eval = cfg["experiment"]["task"] == "tulu3"

task_times = {}
task_metrics = {}

for spec in task_specs:
    task_started = (
        time.perf_counter()
        if is_tulu_lm_eval
        else None
    )

    with _suppress_zero_shot_lm_eval_warnings(
        enabled=int(spec["num_fewshot"]) == 0
    ):
        task_manager, test_selection = (
            _selected_lm_eval_tasks(
                spec["name"],
                spec,
            )
        )

        setup_seconds = (
            time.perf_counter() - task_started
            if task_started is not None
            else None
        )

        before_counter = dict(
            proxy.execution_counter
        )

        evaluation_started = (
            time.perf_counter()
            if is_tulu_lm_eval
            else None
        )

        task_result = simple_evaluate(
            model=harness_model,
            tasks=[spec["name"]],
            task_manager=task_manager,
            num_fewshot=int(
                spec["num_fewshot"]
            ),
            batch_size=batch_size,
            limit=None,
            random_seed=int(
                cfg["experiment"]["seed"]
            ),
            numpy_random_seed=int(
                cfg["experiment"]["seed"]
            ),
            torch_random_seed=int(
                cfg["experiment"]["seed"]
            ),
            fewshot_random_seed=int(
                cfg["experiment"]["seed"]
            ),
            apply_chat_template=bool(
                cfg["evaluation"].get(
                    "apply_chat_template",
                    True,
                )
            ),
            log_samples=(
                False
                if is_tulu_lm_eval
                else True
            ),
        )

    ...
```

之后保持现有：

```python
local_timing
after_counter
local_counter
task_counter
timing
```

计算不变。

---

# 7. main rank 在每个 task 结束后立即写结果

当前 main rank 只把结果塞入：

```python
task_results[name]
```

要改为直接完成以下工作。

## 7.1 保留现有结果合法性检查

必须保留：

```python
if task_result is None:
    raise RuntimeError(
        f"lm-eval returned no result on main rank "
        f"for {spec['name']!r}"
    )
```

保留：

```python
correct_effective_sample_counts(
    task_result,
    test_selection,
)
```

保留：

```python
if not result_contains_metric(
    task_result,
    spec["metric"],
):
    raise ValueError(...)
```

顺序建议仍为：

```text
task_result is not None
↓
correct sample counts
↓
metric existence validation
↓
写盘
```

---

# 8. 逐任务构造原有 `result` payload

当前统一写盘阶段中的：

```python
actual = int(
    selection["selected_count"]
)

task_temporal_forwards = task_counter.get(
    "temporal_sample_step_forwards",
    0,
)

task_temporal_slots = task_counter.get(
    "batched_temporal_sample_slots",
    0,
)
```

移动到当前 task 的 main-rank 分支。

继续构造完全相同语义的：

```python
result = {
    "tasks": {
        name: task_result,
    },
    "snn2_metadata": {
        **common_snn2_metadata,
        ...
    },
}
```

必须保留当前所有 `snn2_metadata` 字段，包括：

```text
execution_counter
evaluation_timing
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

不要因为重构删除任何现有 metadata。

---

# 9. 每个 task 立即写原有路径

继续使用现有：

```python
output_dir = (
    task_results_root
    / safe_name(name)
    / lm_eval_spec_dirname(spec)
)
```

然后立即：

```python
write_json(
    output_dir / "results.json",
    result,
)

write_json(
    output_dir / "test_selection.json",
    test_selection,
)
```

路径必须完全保持现状。

不要新增一套新的 task 输出目录。

不要改变：

```text
task_results/<task>/<spec>/
```

层级。

---

# 10. 每个 task 完成后立即更新 `task_times` 和 `task_metrics`

对于 Tulu3：

```python
task_times[name] = seconds_to_hms(
    timing["lm_eval_seconds"]
)
```

以及：

```python
task_metrics[name] = extract_metric_value(
    task_result,
    spec["metric"],
    task_name=name,
)
```

必须在释放 `task_result` 之前完成。

---

# 11. `evaluation_summary.json` 必须逐任务更新

用户明确要求继续保留：

```text
evaluation_summary.json
```

并且必须继续包含：

```json
{
  "task_times": {...},
  "task_metrics": {...}
}
```

## 11.1 每完成一个 task 就写一次

例如 Task 1 完成：

```json
{
  "task_times": {
    "truthfulqa_mc1": "00:12:34"
  },
  "task_metrics": {
    "truthfulqa_mc1": 0.37
  }
}
```

Task 2 完成后覆盖为：

```json
{
  "task_times": {
    "truthfulqa_mc1": "00:12:34",
    "agieval": "00:08:51"
  },
  "task_metrics": {
    "truthfulqa_mc1": 0.37,
    "agieval": 0.41
  }
}
```

直到最终完整 summary。

---

# 12. `evaluation_summary.json` 应采用原子替换

当前公共：

```python
write_json(...)
```

是直接：

```python
open(path, "w")
json.dump(...)
```

它不是“写临时文件再 rename”的原子替换。

本轮不要全局修改 `snn2.artifacts.write_json()`，避免影响整个项目所有 artifact 写入行为。

推荐在：

```text
scripts/evaluate_lm_harness.py
```

局部增加 helper。

需要额外：

```python
from pathlib import Path
```

如果当前脚本尚未 import `Path`。

示意：

```python
def _write_json_atomic(path, payload):
    path = Path(path)

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    write_json(
        temporary,
        payload,
    )

    os.replace(
        temporary,
        path,
    )

    return path
```

因为：

```python
os.replace(...)
```

在同一文件系统中提供原子 rename/replace 语义。

本脚本已经 import `os`，因此只需保证 `Path` 可用。

### 为什么只对 summary 用 atomic

Task-specific：

```text
results.json
test_selection.json
```

是第一次创建，且每个 task 完成后只写一次。

而：

```text
evaluation_summary.json
```

会被连续重写多次，因此更值得使用：

```text
temp file
→ os.replace()
```

避免进程中断时留下半写入 JSON。

---

# 13. 推荐增加 helper，避免 task loop 过长

为了减少重复和降低重构风险，可增加两个局部 helper。

## 13.1 `_write_json_atomic`

如上。

## 13.2 可选 `_write_tulu_summary`

例如：

```python
def _write_tulu_summary(
    output_root,
    *,
    task_times,
    task_metrics,
):
    summary = {
        "task_times": dict(task_times),
        "task_metrics": dict(task_metrics),
    }

    _write_json_atomic(
        output_root
        / "evaluation_summary.json",
        summary,
    )

    return summary
```

这样 task loop 中只需要：

```python
summary = _write_tulu_summary(
    output_root,
    task_times=task_times,
    task_metrics=task_metrics,
)
```

不要让 helper 改变 summary 的字段名和结构。

---

# 14. 每个 task 写完后的 `run.event(...)` 保持原语义

现有：

```python
run.event(
    "evaluation_saved",
    output_dir=str(output_dir),
    model_variant=...,
    tasks=[name],
    ...
)
```

应从原来的“所有任务结束后统一循环”移动到：

```text
当前 task results.json 写完后
```

立即执行。

字段保持不变：

```text
output_dir
model_variant
tasks
lm_eval_seconds
total_seconds
selected_documents
```

不要改变 event 名：

```text
evaluation_saved
```

---

# 15. `lm_eval_summary_saved` event 只在最终完成时保留一次

虽然 `evaluation_summary.json` 每个 task 都更新，但建议不要每个 task 都发：

```text
lm_eval_summary_saved
```

以免改变现有 log semantics。

推荐：

```text
每个 task：
    原子更新 evaluation_summary.json
    不发 lm_eval_summary_saved

所有 task 全部完成：
    再原子写一次最终 summary
    发一次原有 lm_eval_summary_saved event
```

最终：

```python
if is_tulu_lm_eval:
    summary = {
        "task_times": task_times,
        "task_metrics": task_metrics,
    }

    _write_json_atomic(
        output_root
        / "evaluation_summary.json",
        summary,
    )

    run.event(
        "lm_eval_summary_saved",
        output_dir=str(output_root),
        **summary,
    )
```

这样保持现有 event 的“最终完成”含义。

---

# 16. Worker rank 行为保持不变

非 main rank 当前要求：

```python
elif task_result is not None:
    raise RuntimeError(...)
```

继续保留。

Worker 不应：

- 写 `results.json`；
- 写 `test_selection.json`；
- 写 `evaluation_summary.json`；
- 保存 `task_metrics` 最终文件；
- 保存 `task_times` 最终文件。

所有文件仍然只由：

```python
accelerator.is_main_process
```

写入。

---

# 17. Barrier 顺序

推荐每个 task 的顺序：

```text
所有 rank 完成 simple_evaluate
↓
distributed counter / timing gather
↓
main rank 校验并写 task result
↓
main rank 更新 summary
↓
main rank 释放结果引用
↓
accelerator.wait_for_everyone()
↓
下一 task
```

也就是说现有：

```python
accelerator.wait_for_everyone()
```

继续保留在每个 task 的末尾。

这样保证：

- main rank 写盘完成；
- worker 不提前进入下一任务造成生命周期混乱；
- 当前 distributed lm-eval 任务边界保持明确。

---

# 18. 释放每个 task 的 Python 结果对象

task 写盘完成并提取 metric 后，可以显式：

```python
del task_result
```

必要时也可：

```python
del task_manager
```

不要为了“释放显存”调用：

```python
torch.cuda.empty_cache()
```

本轮主要释放的是：

```text
Python result dict
task sample metadata
dataset/task manager references
```

对应的是 CPU RAM / Python heap。

CUDA 内存本身不应依赖前一个 task 的 `task_result`。

显式 `del` 的主要目的：

- 缩短引用生命周期；
- 明确代码意图；
- 让 Python GC 能够回收前一个 task 的对象。

不需要强制：

```python
gc.collect()
```

除非实际 profiling 证明存在持续 CPU memory buildup。

默认不加。

---

# 19. 不要修改 `task_result["samples"]` 手动删除

不要采用：

```python
task_result.pop("samples", None)
```

作为主要方案。

正确方式是从 lm-eval 源头：

```python
log_samples=False
```

理由：

1. 避免 lm-eval 运行期间构造逐样本 logged samples；
2. 多 GPU 时避免 sample object gather 到 rank 0；
3. 减少 CPU RAM；
4. 减少 Python object serialization；
5. 减少最终 `results.json` 体积。

`pop("samples")` 只能减少最终保存内容，不能避免前面已经发生的 sample 收集和 gather。

---

# 20. 不能修改的评估语义

本轮必须保证以下行为完全不变。

## 20.1 Task list

当前 enabled task：

```text
truthfulqa_mc1
agieval
arc_challenge
piqa
winogrande
boolq
```

保持不变。

## 20.2 Few-shot

保持各 task 当前：

```text
num_fewshot
```

不变。

## 20.3 Sampling

继续使用：

```python
_selected_lm_eval_tasks(...)
build_test_selection(...)
selection_by_leaf(...)
indices_for_rank(...)
```

不变。

## 20.4 Seed

保持：

```python
random_seed
numpy_random_seed
torch_random_seed
fewshot_random_seed
test_seed
```

不变。

## 20.5 Metric

继续：

```python
result_contains_metric(...)
extract_metric_value(...)
```

不变。

## 20.6 Effective sample count

继续：

```python
correct_effective_sample_counts(...)
```

不变。

## 20.7 Distributed aggregation

继续：

```python
gather_sum_execution_counter(...)
distributed_max_seconds(...)
```

不变。

## 20.8 SNN execution path

不要修改：

```text
build_evaluation_controller
install_model_integration
EvaluationModelProxy
DistributedPreinitializedHFLM
Prefix KV
Temporal Phase
Temporal GIF
Temporal MTN
R1/R2/R3/R4
replacement sites
final norm neuron
```

---

# 21. 建议的最终代码结构

重构后 `main()` 评估部分建议概念结构如下：

```python
# model / tokenizer / controller / proxy / harness
...

task_specs = enabled_lm_eval_task_specs(cfg)
is_tulu_lm_eval = (
    cfg["experiment"]["task"] == "tulu3"
)

# ----------------------------------------
# 提前构造整个 run 不随 task 改变的 metadata
# ----------------------------------------

layers = int(
    getattr(
        model.config,
        "num_hidden_layers",
    )
)

per_forward_operators = (
    activation_neuron_operators_per_temporal_forward(
        num_hidden_layers=layers,
        neuron=args.neuron,
    )
)

common_snn2_metadata = {
    ...
}

# ----------------------------------------
# 提前构造 output root
# ----------------------------------------

...
task_results_root = (
    output_root / "task_results"
    if is_tulu_lm_eval
    else output_root
)

task_times = {}
task_metrics = {}

# ----------------------------------------
# 逐 task evaluate + 立即持久化
# ----------------------------------------

for spec in task_specs:

    ...
    task_result = simple_evaluate(
        ...,
        log_samples=(
            False
            if is_tulu_lm_eval
            else True
        ),
    )

    ...
    task_counter = ...
    timing = ...

    if accelerator.is_main_process:

        validate task_result
        correct sample count
        validate metric

        name = spec["name"]

        result = {
            ...
        }

        output_dir = (
            task_results_root
            / safe_name(name)
            / lm_eval_spec_dirname(spec)
        )

        write_json(
            output_dir / "results.json",
            result,
        )

        write_json(
            output_dir / "test_selection.json",
            test_selection,
        )

        if is_tulu_lm_eval:
            task_times[name] = ...
            task_metrics[name] = ...

            summary = {
                "task_times": task_times,
                "task_metrics": task_metrics,
            }

            _write_json_atomic(
                output_root
                / "evaluation_summary.json",
                summary,
            )

        run.event(
            "evaluation_saved",
            ...
        )

        del task_result

    elif task_result is not None:
        raise RuntimeError(...)

    accelerator.wait_for_everyone()

# ----------------------------------------
# 最终 summary + 保留原 event
# ----------------------------------------

if (
    accelerator.is_main_process
    and is_tulu_lm_eval
):
    summary = {
        "task_times": task_times,
        "task_metrics": task_metrics,
    }

    _write_json_atomic(
        output_root
        / "evaluation_summary.json",
        summary,
    )

    run.event(
        "lm_eval_summary_saved",
        output_dir=str(output_root),
        **summary,
    )
```

---

# 22. 关于部分失败时的预期行为

例如当前 enabled tasks 顺序是：

```text
1 truthfulqa_mc1
2 agieval
3 arc_challenge
4 piqa
5 winogrande
6 boolq
```

假设：

```text
truthfulqa_mc1 成功
agieval 成功
arc_challenge 成功
piqa 运行时进程失败
```

期望本地已经存在：

```text
task_results/truthfulqa_mc1/.../results.json
task_results/truthfulqa_mc1/.../test_selection.json

task_results/agieval/.../results.json
task_results/agieval/.../test_selection.json

task_results/arc_challenge/.../results.json
task_results/arc_challenge/.../test_selection.json
```

并且：

```text
evaluation_summary.json
```

至少包含：

```json
{
  "task_times": {
    "truthfulqa_mc1": "...",
    "agieval": "...",
    "arc_challenge": "..."
  },
  "task_metrics": {
    "truthfulqa_mc1": "...",
    "agieval": "...",
    "arc_challenge": "..."
  }
}
```

这就是本轮最核心的容错收益。

注意：

本轮**不要求实现自动 resume / skip completed tasks**。

如果脚本重新从头执行，仍可以按原 task 顺序重新评估并覆盖对应文件。

不要擅自实现：

```text
发现 results.json 已存在 → 自动跳过 task
```

因为这会引入新的 artifact freshness / config compatibility / partial-run consistency 问题。

---

# 23. 结果文件内容变化预期

由于：

```python
log_samples=False
```

新的 `task_result` 不应再包含：

```json
"samples": {
  ...
}
```

所以：

```text
results.json
```

会明显变小。

但是下面内容必须仍存在：

```text
results
groups
group_subtasks
configs
versions
n-shot
higher_is_better
n-samples
config
git_hash
date
snn2_metadata
```

具体以 pinned lm-eval 返回内容为准。

不要手工构造或删减其它 lm-eval aggregate 字段。

---

# 24. 测试要求

## 24.1 `log_samples=False` 参数回归

测试需验证 Tulu3 分支调用：

```python
simple_evaluate(
    ...,
    log_samples=False,
)
```

不要只检查源代码字符串。

优先用 monkeypatch fake `simple_evaluate` 记录 kwargs。

至少 assert：

```python
assert call_kwargs["log_samples"] is False
```

如果保留非 Tulu3 历史行为，则另测：

```python
log_samples is True
```

用于非 Tulu3 分支。

---

## 24.2 task result 不再跨任务累积

推荐把“单 task 保存”逻辑提取成小 helper，便于测试。

例如可选：

```python
def _persist_task_result(...):
    ...
```

测试可传入两个 fake task_result：

```text
task_a
task_b
```

确认：

```text
task_a results.json
```

在 task_b 执行前已经存在。

测试重点不是 Python GC 本身，而是：

```text
逐任务 immediate persistence
```

---

## 24.3 `evaluation_summary.json` 增量行为

tmp_path 下模拟：

### Task 1

写：

```python
task_times = {
    "task_a": "00:01:00",
}

task_metrics = {
    "task_a": 0.5,
}
```

读回：

```text
evaluation_summary.json
```

必须等于：

```json
{
  "task_times": {
    "task_a": "00:01:00"
  },
  "task_metrics": {
    "task_a": 0.5
  }
}
```

### Task 2

更新：

```python
task_times["task_b"] = "00:02:00"
task_metrics["task_b"] = 0.6
```

再次读回，应同时包含 A+B。

---

## 24.4 原子 summary helper 测试

测试 `_write_json_atomic(...)`：

1. 写入第一版；
2. 写入第二版；
3. 最终目标文件是完整第二版 JSON；
4. 正常完成后临时文件不存在。

例如检查：

```python
assert not (
    tmp_path / ".evaluation_summary.json.tmp"
).exists()
```

如果 helper 临时文件命名不同，按实际实现检查。

---

## 24.5 保留 summary schema

测试必须明确：

```python
set(summary) == {
    "task_times",
    "task_metrics",
}
```

除非当前仓库未来有明确新增字段。

本轮不要擅自改成：

```json
{
  "completed_tasks": ...,
  "status": ...,
  "task_times": ...,
  "task_metrics": ...
}
```

用户要求保留现有 summary 结构。

---

## 24.6 不再依赖 `samples`

测试 fake `task_result` 可以完全没有：

```python
"samples"
```

并确认：

```python
extract_metric_value(...)
result_contains_metric(...)
write_json(...)
```

整个保存流程仍然正常。

---

# 25. 运行测试

至少执行：

```bash
pytest -q tests/test_lm_eval_distributed.py
```

如果新增专门测试，例如：

```text
tests/test_lm_eval_incremental_persistence.py
```

则执行：

```bash
pytest -q tests/test_lm_eval_incremental_persistence.py
```

然后：

```bash
pytest -q
```

必须全部通过。

---

# 26. 轻量 integration smoke test

不要直接先跑完整 6 个任务。

建议临时使用一个小测试配置：

```text
truthfulqa_mc1
agieval
```

并将：

```text
test_samples
```

缩到很小，仅用于验证文件生命周期。

注意：

正式配置不要永久改小。

smoke test 应确认：

## Task 1 结束后，Task 2 还在运行时

本地已经能看到：

```text
task_results/truthfulqa_mc1/.../results.json
task_results/truthfulqa_mc1/.../test_selection.json
evaluation_summary.json
```

并确认：

```text
evaluation_summary.json
```

此时已经有 truthfulqa 的：

```text
task_times
task_metrics
```

## 所有任务结束后

确认：

```text
evaluation_summary.json
```

包含全部任务。

---

# 27. 验证 `results.json` 不含逐样本结果

运行 smoke test 后：

```bash
python - <<'PY'
import json
from pathlib import Path

for path in Path("artifacts").rglob("results.json"):
    data = json.loads(path.read_text())
    tasks = data.get("tasks", {})
    for name, result in tasks.items():
        if isinstance(result, dict) and "samples" in result:
            print("HAS SAMPLES:", path, name)
PY
```

对于本轮新生成的 Tulu3 lm-eval 结果：

```text
不应出现 HAS SAMPLES
```

注意不要用旧 artifact 误判，因为旧结果可能是在 `log_samples=True` 时生成的。

---

# 28. CPU RAM 验证

本轮优化的主要收益是：

```text
rank 0 CPU RAM
```

不是 GPU VRAM。

可以在真实 6-task evaluation 中观察：

```bash
htop
```

或：

```bash
ps -o pid,rss,vsz,cmd -p <PID>
```

期望：

```text
完成 task 1
→ task 1 完整 sample result 不再长期驻留

完成 task 2
→ 不会继续累计 task 1 + task 2 的逐样本结果
```

但 Hugging Face datasets / Python allocator / OS page cache 可能不会立即把 RSS 完全降回原值，因此不要要求 RSS 严格逐 task 下降。

重点是：

```text
不再因为 task_results + samples 线性保存所有历史任务详细结果
```

---

# 29. GPU 显存验证

本轮不要添加：

```python
torch.cuda.empty_cache()
```

如果 GPU `nvidia-smi` 显存没有明显下降，这是正常的，因为本轮优化的目标并不是 CUDA activation。

SNN evaluation GPU peak 主要仍来自：

```text
model weights
current batch
current sequence length
temporal activation
attention
MLP
Prefix KV
```

若未来观察到 GPU 显存随 task 数量持续上涨，应单独排查：

```text
CUDA allocator
controller lazy-loaded modules
temporal tensor references
lm-eval HFLM buffers
KV cache
```

不要把本轮 CPU result persistence 改动与 GPU leak 混为一谈。

---

# 30. 兼容性要求

本轮完成后，以下命令接口不能改变：

```bash
accelerate launch \
  scripts/evaluate_lm_harness.py \
  --config <CFG> \
  --neuron phase
```

以及：

```text
--neuron gif
--neuron mtn
--neuron ann
--base
--rotated-pre-finetuning
```

都不应因为本轮改动出现 CLI 行为变化。

---

# 31. 不要新增 YAML 配置开关

本轮不要增加：

```yaml
evaluation:
  log_samples: false
```

也不要增加：

```yaml
evaluation:
  incremental_save: true
```

当前用户已经明确要求 Tulu3 固定行为：

```text
log_samples=False
逐 task 立即落盘
保留 evaluation_summary.json
```

无需再增加新的实验维度和路径维度。

这样也避免：

```text
同一 output path
却可能由不同 log_samples / persistence 配置生成
```

造成 artifact 语义分叉。

---

# 32. 完成后的最终预期

正式 Llama3-8B / Tulu3 SNN lm-eval：

```text
truthfulqa_mc1
  ↓
evaluate
  ↓
results.json
test_selection.json
evaluation_summary.json
  ↓
释放完整 task_result

agieval
  ↓
evaluate
  ↓
results.json
test_selection.json
evaluation_summary.json
  ↓
释放完整 task_result

arc_challenge
  ↓
...

piqa
  ↓
...

winogrande
  ↓
...

boolq
  ↓
...
```

最终：

```text
evaluation_summary.json
```

仍是：

```json
{
  "task_metrics": {
    "truthfulqa_mc1": ...,
    "agieval": ...,
    "arc_challenge": ...,
    "piqa": ...,
    "winogrande": ...,
    "boolq": ...
  },
  "task_times": {
    "truthfulqa_mc1": "...",
    "agieval": "...",
    "arc_challenge": "...",
    "piqa": "...",
    "winogrande": "...",
    "boolq": "..."
  }
}
```

同时每个 task 的：

```text
results.json
```

不再包含逐样本：

```text
samples
```

---

# 33. 验收标准

本轮修改只有在以下全部满足时才算完成。

- [ ] Tulu3 `simple_evaluate()` 显式 `log_samples=False`
- [ ] 当前 enabled 6 tasks 可以正常运行
- [ ] 删除长期 `task_results` 完整结果缓存
- [ ] 每个 task 完成后立即写 `results.json`
- [ ] 每个 task 完成后立即写 `test_selection.json`
- [ ] 每个 task 完成后立即更新 `task_metrics`
- [ ] 每个 task 完成后立即更新 `task_times`
- [ ] 每个 task 完成后原子重写 `evaluation_summary.json`
- [ ] `evaluation_summary.json` schema 仍只保留现有 `task_metrics` / `task_times`
- [ ] 所有 task 完成后再次写最终 summary
- [ ] 最终保留现有 `lm_eval_summary_saved` event
- [ ] `evaluation_saved` event 改为对应 task 真正写盘后立即记录
- [ ] Worker rank 不写 artifact
- [ ] 每个 task 后保留 distributed barrier
- [ ] 不调用 `torch.cuda.empty_cache()`
- [ ] 不实现自动 resume / skip completed tasks
- [ ] 不修改 task list / metric / fewshot / sampling / seed
- [ ] 不修改 SNN conversion / forward / Prefix / temporal implementation
- [ ] 新生成 Tulu3 `results.json` 不含 `samples`
- [ ] 相关单元测试通过
- [ ] `pytest -q` 全部通过
- [ ] 两任务 smoke test 中，Task 1 结束后即可在 Task 2 运行期间看到 Task 1 文件及部分 `evaluation_summary.json`

---

## 最终原则

本轮不是改变评估结果，而是改变：

```text
“结果在内存中保存多久”
```

以及：

```text
“什么时候写到磁盘”
```

数学评估语义应保持：

```text
same model
same SNN forward
same selected documents
same few-shot
same seeds
same lm-eval revision
same metrics
```

仅将：

```text
所有 task 结束后统一保存
```

改成：

```text
每个 task 完成后立即保存
```

并将：

```text
log_samples=True
```

改成：

```text
Tulu3: log_samples=False
```

以降低 rank 0 CPU RAM 占用、减少 distributed Python object gather、缩小结果文件，并提高长时间 SNN evaluation 的容错性。
