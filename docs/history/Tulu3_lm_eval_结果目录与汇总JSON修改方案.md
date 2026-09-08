# Tulu-3 / lm-eval 评估结果目录与汇总 JSON 修改方案

## 0. 修改目标

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

本轮只修改 **Tulu-3 / lm-eval 的评估结果组织与汇总输出**。

当前 Tulu-3 / lm-eval 在某个 evaluation root 下直接保存：

```text
<evaluation_root>/
├── truthfulqa_mc1/
├── agieval/
├── arc_challenge/
├── piqa/
├── winogrande/
├── boolq/
└── evaluation_timing.json
```

本轮修改为：

```text
<evaluation_root>/
├── task_results/
│   ├── truthfulqa_mc1/
│   ├── agieval/
│   ├── arc_challenge/
│   ├── piqa/
│   ├── winogrande/
│   └── boolq/
└── evaluation_summary.json
```

其中：

- `task_results/` 下保存所有 task 原有的 `results.json` 与 `test_selection.json`；
- `evaluation_summary.json` 同时汇总：
  - 每个 enabled task 的实际运行时间；
  - 每个 enabled task 的论文主指标；
- 不再生成或保留 `evaluation_timing.json`；
- 不计算、不保存 `overall_metric`。

---

# 1. 硬性约束

本轮不要修改：

- `scripts/evaluate_tldr.py`
- TL;DR summarization 的任何路径或行为
- Tulu benchmark task selection
- `test_samples`
- `test_seed`
- `num_fewshot`
- `metric`
- `cot`
- lm-eval pinned revision
- 多卡数据并行语义
- rank/world_size 分片
- Base / Final ANN / Phase / GIF / MTN forward
- Prefix
- rotation
- calibration
- conversion
- execution counter
- metric 计算本身
- training

特别说明：

```yaml
agieval:
  test_samples: 1000
```

是当前正式实验设定，**必须保持不变**。

---

# 2. “evaluation root”的定义

不要重新设计 Prefix / calibration selector 层级。

本方案中的：

```text
<evaluation_root>
```

指当前代码中**原本直接存放六个 task 目录和 `evaluation_timing.json` 的目录**。

即保留现有：

```text
Base / ANN / SNN
Prefix enabled/disabled
num_samples selector
```

等路径层级，只在最末端的 task 容器位置增加一层：

```text
task_results/
```

例如如果当前 Final ANN 路径是：

```text
.../ann/evaluation/
└── prefix_enabled_ture/
    ├── truthfulqa_mc1/
    ├── agieval/
    └── evaluation_timing.json
```

则改为：

```text
.../ann/evaluation/
└── prefix_enabled_ture/
    ├── task_results/
    │   ├── truthfulqa_mc1/
    │   └── agieval/
    └── evaluation_summary.json
```

Base 没有 Prefix selector 时则例如：

```text
.../base/seed42/evaluation/
├── task_results/
│   ├── truthfulqa_mc1/
│   ├── agieval/
│   ├── arc_challenge/
│   ├── piqa/
│   ├── winogrande/
│   └── boolq/
└── evaluation_summary.json
```

---

# 3. task_results/ 内部结构

Task 自身原有目录结构不变，只整体下移到：

```text
task_results/
```

例如：

```text
<evaluation_root>/
└── task_results/
    ├── truthfulqa_mc1/
    │   └── num_fewshot_0_cot_false_test_samples_full_test_seed_42/
    │       ├── results.json
    │       └── test_selection.json
    ├── agieval/
    │   └── num_fewshot_0_cot_false_test_samples_1000_test_seed_42/
    │       ├── results.json
    │       └── test_selection.json
    ├── arc_challenge/
    ├── piqa/
    ├── winogrande/
    └── boolq/
```

不要修改：

```text
lm_eval_spec_dirname(spec)
```

的命名规则。

---

# 4. evaluation_summary.json 格式

最终只需要两个顶层字段：

```json
{
  "task_times": {
    "truthfulqa_mc1": "00:02:31",
    "agieval": "00:08:47",
    "arc_challenge": "00:01:42",
    "piqa": "00:01:35",
    "winogrande": "00:01:08",
    "boolq": "00:03:21"
  },
  "task_metrics": {
    "truthfulqa_mc1": 0.4213456789012345,
    "agieval": 0.4827123456789012,
    "arc_challenge": 0.5461234567890123,
    "piqa": 0.7204567890123456,
    "winogrande": 0.6132345678901234,
    "boolq": 0.6708123456789012
  }
}
```

不要增加：

```text
overall_metric
overall_score
average
mean
macro_average
```

等总分字段。

---

# 5. task_metrics 的定义

`task_metrics` 对每个 enabled task 只保存该 task 当前 config 中指定的 canonical metric。

当前六个 enabled benchmark：

```text
truthfulqa_mc1 -> acc
agieval        -> acc
arc_challenge  -> acc_norm
piqa           -> acc_norm
winogrande     -> acc
boolq          -> acc
```

因此：

```json
"task_metrics": {
  "truthfulqa_mc1": <TruthfulQA acc>,
  "agieval": <AGIEval acc>,
  "arc_challenge": <ARC-Challenge acc_norm>,
  "piqa": <PIQA acc_norm>,
  "winogrande": <WinoGrande acc>,
  "boolq": <BoolQ acc>
}
```

---

# 6. task_metrics 数值格式

必须保存 lm-eval 原始 metric 数值。

例如 lm-eval 返回：

```text
0.5461234567890123
```

则 JSON 直接保存：

```json
0.5461234567890123
```

禁止：

```text
×100
round(...)
格式化成百分数
人为截断小数位
```

所以不要输出：

```text
54.61
0.5461
54.6123%
```

应尽量保持 `results.json` 中 canonical metric 对应的原始 Python float 值。

---

# 7. metric key 的读取规则

现有项目已经支持带 filter suffix 的 lm-eval metric key，例如：

```text
acc,none
acc_norm,none
exact_match,get-answer
```

本轮不要自己写脆弱的：

```python
task_result[spec["metric"]]
```

如果现有代码只有 `result_contains_metric()` 用于存在性校验，建议新增一个与其语义一致的 helper，例如：

```python
def extract_metric_value(task_result: dict, metric: str) -> float:
    ...
```

其规则：

1. 接受精确 key，例如 `acc`；
2. 接受 `acc,<filter>`；
3. 不接受 stderr，例如 `acc_stderr`、`acc_stderr,none`；
4. 找不到 canonical metric 时立即报错；
5. 出现多个无法唯一判定的 canonical metric value 时立即报错；
6. 返回值必须是数值类型；
7. `bool` 不视为合法 metric number。

建议把 helper 放在：

```text
snn2/lm_eval_protocol.py
```

因为这是 lm-eval result semantic，不属于路径或 distributed helper。

---

# 8. AGIEval group metric

AGIEval 是 lm-eval group。

`task_metrics["agieval"]` 必须取：

```text
agieval group 的 canonical acc
```

即论文实际报告的 AGIEval 总指标。

不要：

- 取某一个 AGIEval leaf；
- 自己重新对 leaf metric 做平均；
- 从 `n-samples` 推导；
- 自己根据 sample-level result 重新计算。

应直接从 pinned lm-eval 已完成 group aggregation 后的 `task_result` 中读取 canonical `acc`。

---

# 9. task_times 的定义

每个：

```text
task_times[task_name]
```

使用当前已经存在的：

```text
timing["lm_eval_seconds"]
```

即真正执行 `simple_evaluate()` 的 wall-clock 时间。

不要使用：

```text
selection_setup_seconds
```

也不要使用：

```text
total_seconds
```

作为论文里的 task runtime。

---

# 10. 多卡 task time 语义保持不变

当前多卡实现已经将：

```text
lm_eval_seconds
```

定义为各 rank 耗时的最大值：

```text
T_task = max(T_rank0, T_rank1, ..., T_rankN)
```

这是实际多卡 wall-clock 时间。

本轮不要改成 rank 时间求和或求平均。

---

# 11. 时间格式

`evaluation_summary.json` 中时间统一保存为：

```text
HH:MM:SS
```

例如：

```text
37 秒      -> "00:00:37"
5 分 9 秒  -> "00:05:09"
1 小时     -> "01:00:00"
27:03:08   -> "27:03:08"
```

超过 24 小时时不要按 datetime clock 回卷。

统一先将秒数 round 到最近整数秒，再转换为 `HH:MM:SS`。

建议 helper：

```python
def seconds_to_hms(seconds: float) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError(...)
    if seconds < 0:
        raise ValueError(...)

    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
```

---

# 12. 修改 scripts/evaluate_lm_harness.py

当前每个 task 直接写到 `output_root` 下。

改为：

```python
task_results_root = output_root / "task_results"
```

每个 task：

```python
output_dir = (
    task_results_root
    / safe_name(name)
    / lm_eval_spec_dirname(spec)
)
```

然后仍然保存：

```python
write_json(output_dir / "results.json", result)
write_json(output_dir / "test_selection.json", selection)
```

---

# 13. 删除 evaluation_timing.json

完全删除：

```python
write_json(
    output_root / "evaluation_timing.json",
    ...
)
```

以及仅用于生成旧 `evaluation_timing.json` 的旧 summary payload。

最终不得再产生：

```text
evaluation_timing.json
```

---

# 14. 构建 evaluation_summary.json

只由：

```python
accelerator.is_main_process
```

构建。

伪代码：

```python
task_times = {}
task_metrics = {}

for spec in task_specs:
    name = spec["name"]
    task_result, selection, task_counter, timing = task_results[name]

    task_times[name] = seconds_to_hms(
        timing["lm_eval_seconds"]
    )

    task_metrics[name] = extract_metric_value(
        task_result,
        spec["metric"],
    )

summary = {
    "task_times": task_times,
    "task_metrics": task_metrics,
}

write_json(
    output_root / "evaluation_summary.json",
    summary,
)
```

---

# 15. evaluation_summary.json 只包含 enabled tasks

不要把：

```text
mmlu_pro
bbh
gsm8k_cot
minerva_math
```

写进 `evaluation_summary.json`，因为当前它们 `enabled: false`。

summary 中只包含：

```text
truthfulqa_mc1
agieval
arc_challenge
piqa
winogrande
boolq
```

顺序与 `enabled_lm_eval_task_specs(cfg)` 保持一致。

---

# 16. results.json 内部 evaluation_timing 的处理

当前每个 task 的 `results.json` 中已有：

```json
"snn2_metadata": {
  "evaluation_timing": {
    ...
  }
}
```

本轮建议 **保留该 per-task provenance**。

用户要求删除的是独立的：

```text
evaluation_timing.json
```

而不是删除 task-level results metadata。

---

# 17. evaluation_summary.json 不复制其他 provenance

不要把下面内容塞进 summary：

- model_variant
- world_size
- batch_size
- Prefix
- calibration
- lm_eval revision
- seeds
- sample count
- execution counter
- forward semantics

这些仍保留在每个 task 的 `results.json` / `test_selection.json` 中。

`evaluation_summary.json` 只保留：

```json
{
  "task_times": {...},
  "task_metrics": {...}
}
```

---

# 18. 修改 scripts/verify_artifacts.py

当前 Tulu lm-eval task root 改为加入：

```text
task_results/
```

即类似：

```python
return (
    directory
    / "task_results"
    / safe_name(spec["name"])
    / lm_eval_spec_dirname(spec)
)
```

只修改 Tulu lm-eval path helper，不影响 TL;DR verifier。

---

# 19. verifier 增加 evaluation_summary.json 检查

对于每一个实际 Tulu lm-eval evaluation root（Base / Final ANN / Phase / GIF / MTN），都验证：

```text
evaluation_summary.json
```

存在。

并要求：

```python
set(summary) == {
    "task_times",
    "task_metrics",
}
```

---

# 20. verifier 检查 task_times

要求：

```python
set(summary["task_times"]) == enabled_task_names
```

每个 value 必须匹配：

```python
r"^\d{2,}:[0-5]\d:[0-5]\d$"
```

小时允许超过 23。

然后从对应 task `results.json` 的：

```text
snn2_metadata.evaluation_timing.lm_eval_seconds
```

重新执行 `seconds_to_hms()`，要求与 summary 完全一致。

---

# 21. verifier 检查 task_metrics

要求：

```python
set(summary["task_metrics"]) == enabled_task_names
```

每个 metric value 必须为数字，且 `bool` 不合法。

然后 verifier 重新读取对应 task `results.json`，调用与 evaluator 相同的：

```python
extract_metric_value(
    task_result,
    spec["metric"],
)
```

要求：

```python
summary["task_metrics"][name] == extracted_value
```

不要对数值做 `round`、乘 100 或格式化。

---

# 22. verifier 禁止旧 evaluation_timing.json

如果看到：

```text
<evaluation_root>/evaluation_timing.json
```

建议直接报错，提示存在 stale Tulu lm-eval timing artifact。

---

# 23. 旧 task 目录防残留

因为路径从：

```text
<evaluation_root>/<task>/
```

迁移成：

```text
<evaluation_root>/task_results/<task>/
```

服务器若已跑过旧版评估，旧 task 目录可能残留。

正式重新评估某个 model variant 前，清理对应旧 evaluation 输出，避免同时存在新旧结构。

代码不要自动删除整个 artifact root。

---

# 24. tests

至少增加或修改以下测试。

## 24.1 seconds_to_hms

```python
assert seconds_to_hms(0) == "00:00:00"
assert seconds_to_hms(37) == "00:00:37"
assert seconds_to_hms(309) == "00:05:09"
assert seconds_to_hms(3600) == "01:00:00"
assert seconds_to_hms(27 * 3600 + 3 * 60 + 8) == "27:03:08"
```

并测试小数秒采用 round。

## 24.2 extract_metric_value

覆盖：

```text
acc
acc,none
acc_norm,none
```

并确保：

```text
acc_stderr
acc_stderr,none
```

不会被当成 canonical metric。

还要测试 missing / ambiguous / bool value 都失败。

## 24.3 summary schema

验证只有：

```text
task_times
task_metrics
```

且 `overall_metric` 不存在。

## 24.4 task_results path

Tulu path 必须包含：

```text
task_results/<task>/<spec>/
```

## 24.5 TL;DR 回归

必须确认 TL;DR evaluation path 完全不出现：

```text
task_results
evaluation_summary.json
```

---

# 25. 实验执行总结.md

只更新 Tulu-3 / lm-eval artifact 说明。

旧结构：

```text
evaluation/
├── <task>/
└── evaluation_timing.json
```

改成：

```text
evaluation/
├── task_results/
│   └── <task>/
└── evaluation_summary.json
```

如果还有 Prefix / num_samples selector 中间层，原样保留。

---

# 26. 不修改 configs/experiment_matrix.yaml 的 benchmark 参数

本轮不需要修改 task 参数。

特别保持：

```yaml
truthfulqa_mc1:
  test_samples: null

agieval:
  test_samples: 1000

arc_challenge:
  test_samples: null

piqa:
  test_samples: null

winogrande:
  test_samples: null

boolq:
  test_samples: null
```

以及现有 metric / cot / num_fewshot / test_seed 全部不变。

---

# 27. 最终示例

```text
evaluation/
├── task_results/
│   ├── truthfulqa_mc1/
│   │   └── num_fewshot_0_cot_false_test_samples_full_test_seed_42/
│   │       ├── results.json
│   │       └── test_selection.json
│   ├── agieval/
│   │   └── num_fewshot_0_cot_false_test_samples_1000_test_seed_42/
│   │       ├── results.json
│   │       └── test_selection.json
│   ├── arc_challenge/
│   ├── piqa/
│   ├── winogrande/
│   └── boolq/
└── evaluation_summary.json
```

`evaluation_summary.json`：

```json
{
  "task_times": {
    "truthfulqa_mc1": "00:02:31",
    "agieval": "00:08:47",
    "arc_challenge": "00:01:42",
    "piqa": "00:01:35",
    "winogrande": "00:01:08",
    "boolq": "00:03:21"
  },
  "task_metrics": {
    "truthfulqa_mc1": 0.4213456789012345,
    "agieval": 0.4827123456789012,
    "arc_challenge": 0.5461234567890123,
    "piqa": 0.7204567890123456,
    "winogrande": 0.6132345678901234,
    "boolq": 0.6708123456789012
  }
}
```

---

# 28. 完成后验证

运行：

```bash
cd /home/wangwenkang/SNN
pytest -q
```

必须全部通过。

然后用一个 Tulu Base 真实评估确认：

1. `task_results/` 被创建；
2. 所有 enabled task 的 `results.json` 和 `test_selection.json` 都位于其下；
3. `evaluation_summary.json` 位于 `<evaluation_root>/`；
4. 不再生成 `evaluation_timing.json`；
5. `task_times` 是 `HH:MM:SS`；
6. `task_metrics` 与每个 task `results.json` 的 canonical metric 完全一致；
7. 没有 `overall_metric`；
8. metric 没有乘 100；
9. metric 没有人为 round / truncate；
10. 多卡结果与原有 distributed evaluation 数学语义完全一致。

---

# 29. 最终验收条件

1. TL;DR summarization 代码和路径未修改。
2. AGIEval 继续 `test_samples: 1000`。
3. Tulu lm-eval evaluation root 下新增 `task_results/`。
4. 六个 enabled task 的原有结果整体移动到 `task_results/` 下。
5. task 自身 `<task>/<spec>/results.json` 与 `test_selection.json` 层级不变。
6. `<evaluation_root>/evaluation_summary.json` 被生成。
7. `evaluation_summary.json` 只有 `task_times` 和 `task_metrics` 两个顶层字段。
8. 不存在 `overall_metric`。
9. `task_times` 使用 `lm_eval_seconds`。
10. 多卡 time 继续使用各 rank 最大 wall time。
11. 时间统一输出为 `HH:MM:SS`。
12. 超过 24 小时时小时数不回卷。
13. `task_metrics` 使用每个 task config 指定的 canonical metric。
14. ARC-Challenge / PIQA 使用 `acc_norm`。
15. TruthfulQA / AGIEval / WinoGrande / BoolQ 使用 `acc`。
16. AGIEval 使用 lm-eval group aggregate `acc`。
17. metric 保存原始 `[0,1]` 数值。
18. metric 不乘 100。
19. metric 不人为截断或 round。
20. 不再生成 `evaluation_timing.json`。
21. per-task `results.json` 中原有 detailed timing provenance 可以继续保留。
22. verifier 已同步新路径与 summary。
23. tests 覆盖新 summary/path/helper 逻辑。
24. `pytest -q` 全部通过。
