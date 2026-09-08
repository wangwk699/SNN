# Tulu-3 / lm-eval 汇总校验与 AGIEval 文档修正方案

## 0. 修改范围

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

本轮只修改以下两点：

1. 修正 `evaluation_summary.json` 的 verifier，避免因为 `write_json(..., sort_keys=True)` 对 JSON key 排序而误报 task 顺序错误。
2. 修正 `实验执行总结.md` 中 AGIEval 的 `test_samples` 描述，从 `null/full` 改为正式实验设定 `1000`。

## 明确不修改

不要修改：

- `scripts/evaluate_lm_harness.py` 的结果写入逻辑
- `evaluation_summary.json` 的生成逻辑
- `task_results/` 目录结构
- `evaluation_timing.json` 的旧 artifact 清理逻辑
- 不要自动删除旧 artifact
- `configs/experiment_matrix.yaml`
- AGIEval 当前正式配置
- Tulu-3 多卡评估逻辑
- Base / ANN / Phase / GIF / MTN forward
- Prefix
- calibration
- rotation
- conversion
- TL;DR summarization
- TL;DR 路径
- 其他 benchmark 参数

特别确认：

```yaml
agieval:
  test_samples: 1000
```

是当前正式实验要求，必须保持不变。

---

# 1. 修正 `scripts/verify_artifacts.py`

文件：

```text
scripts/verify_artifacts.py
```

当前 `_validate_tulu_evaluation_summary()` 中存在类似逻辑：

```python
specs = enabled_lm_eval_task_specs(cfg)
names = [spec["name"] for spec in specs]

if list(summary["task_times"]) != names or list(summary["task_metrics"]) != names:
    raise ValueError(
        f"Tulu evaluation summary task order mismatch: {summary_path}"
    )
```

这一校验是错误的。

原因是项目统一的：

```python
write_json(...)
```

使用：

```python
json.dump(
    ...,
    sort_keys=True,
)
```

因此 `evaluation_summary.json` 写入磁盘时：

```json
"task_times": {...}
"task_metrics": {...}
```

内部 key 会按字母排序。

即使 evaluator 原本按配置顺序构建：

```text
truthfulqa_mc1
agieval
arc_challenge
piqa
winogrande
boolq
```

实际 JSON 读回来后 key 顺序也可能变成：

```text
agieval
arc_challenge
boolq
piqa
truthfulqa_mc1
winogrande
```

JSON object 的 key 顺序本身也不应该作为协议语义。

---

# 2. 正确的 verifier 规则

不要验证顺序。

只验证：

> `task_times` 和 `task_metrics` 的 task 名集合，必须严格等于当前 enabled task 集合。

推荐改为：

```python
specs = enabled_lm_eval_task_specs(cfg)
names = [spec["name"] for spec in specs]
expected_names = set(names)

task_times = summary.get("task_times")
task_metrics = summary.get("task_metrics")

if not isinstance(task_times, dict):
    raise ValueError(
        f"Invalid Tulu evaluation summary task_times: {summary_path}"
    )

if not isinstance(task_metrics, dict):
    raise ValueError(
        f"Invalid Tulu evaluation summary task_metrics: {summary_path}"
    )

if set(task_times) != expected_names:
    raise ValueError(
        f"Tulu evaluation summary task_times mismatch: {summary_path}"
    )

if set(task_metrics) != expected_names:
    raise ValueError(
        f"Tulu evaluation summary task_metrics mismatch: {summary_path}"
    )
```

也可以把错误信息做得更明确：

```python
missing_times = expected_names - set(task_times)
extra_times = set(task_times) - expected_names

missing_metrics = expected_names - set(task_metrics)
extra_metrics = set(task_metrics) - expected_names

if missing_times or extra_times:
    raise ValueError(
        "Tulu evaluation summary task_times mismatch at "
        f"{summary_path}: missing={sorted(missing_times)}, "
        f"extra={sorted(extra_times)}"
    )

if missing_metrics or extra_metrics:
    raise ValueError(
        "Tulu evaluation summary task_metrics mismatch at "
        f"{summary_path}: missing={sorted(missing_metrics)}, "
        f"extra={sorted(extra_metrics)}"
    )
```

推荐使用后一种，便于后续定位 artifact 错误。

---

# 3. 保留其他 summary validation

除了取消“顺序相等”这一要求，其余现有 validation 都应继续保留。

包括：

```text
summary 顶层必须只有：
- task_times
- task_metrics
```

继续保留：

```python
if set(summary) != {"task_times", "task_metrics"}:
    raise ValueError(...)
```

继续验证：

- `task_times` 每个 value 是 `HH:MM:SS`
- `task_metrics` 每个 value 是数值
- `bool` 不能作为 metric
- summary 中每个 task 的时间必须和对应 `results.json` 中：
  ```text
  snn2_metadata.evaluation_timing.lm_eval_seconds
  ```
  经 `seconds_to_hms()` 转换后的值一致
- summary 中每个 task metric 必须和对应 `results.json` 的 canonical metric 一致
- `evaluation_summary.json` 必须存在
- task results 路径继续是：
  ```text
  task_results/<task>/<spec>/
  ```

不要放宽这些校验。

---

# 4. 不修改旧 `evaluation_timing.json` 的处理

用户明确决定：

> 不让代码负责迁移或删除旧 artifact。

因此本轮不要在：

```text
scripts/evaluate_lm_harness.py
```

中添加：

```python
unlink()
rmtree()
```

等清理逻辑。

也不要自动删除：

```text
evaluation_timing.json
```

用户会在正式重跑之前手工清理旧：

```text
evaluation/
```

结果目录。

当前 verifier 中如果发现旧：

```text
evaluation_timing.json
```

则报 stale artifact 的逻辑可以继续保留：

```python
stale_timing = summary_path.parent / "evaluation_timing.json"

if stale_timing.exists():
    raise ValueError(...)
```

这是合理的，因为正式新实验目录中不应该再存在旧格式 artifact。

---

# 5. 修正 `实验执行总结.md`

文件：

```text
实验执行总结.md
```

当前 Tulu-3 benchmark 表格中 AGIEval 仍然写成：

```text
| agieval | true | 0 | acc | false | null/full | 42 |
```

这是过时描述。

改成：

```text
| agieval | true | 0 | acc | false | 1000 | 42 |
```

---

# 6. 当前六个 enabled task 的正式配置

文档中最终应与实际 matrix 对齐：

| task | enabled | few-shot | metric | CoT | test_samples | test_seed |
|---|---:|---:|---:|---:|---:|---:|
| truthfulqa_mc1 | true | 0 | acc | false | null/full | 42 |
| agieval | true | 0 | acc | false | 1000 | 42 |
| arc_challenge | true | 0 | acc_norm | false | null/full | 42 |
| piqa | true | 0 | acc_norm | false | null/full | 42 |
| winogrande | true | 0 | acc | false | null/full | 42 |
| boolq | true | 0 | acc | false | null/full | 42 |

旧四个 disabled task 保持现状，不需要在本轮修改。

---

# 7. 不修改 `configs/experiment_matrix.yaml`

本轮不改：

```text
configs/experiment_matrix.yaml
```

当前：

```yaml
- name: agieval
  enabled: true
  num_fewshot: 0
  metric: acc
  cot: false
  test_samples: 1000
  test_seed: 42
```

已经正确。

不要把：

```yaml
test_samples: 1000
```

改回：

```yaml
test_samples: null
```

---

# 8. 建议补充测试

建议在现有 verifier / protocol tests 中增加一个针对 JSON key 排序的 regression test。

核心目标：

> summary 即使 task key 顺序与 config 顺序不同，只要集合完全相同，也必须验证通过。

例如构造：

```python
summary = {
    "task_times": {
        "agieval": "00:01:00",
        "arc_challenge": "00:01:00",
        "boolq": "00:01:00",
        "piqa": "00:01:00",
        "truthfulqa_mc1": "00:01:00",
        "winogrande": "00:01:00",
    },
    "task_metrics": {
        "agieval": 0.1,
        "arc_challenge": 0.2,
        "boolq": 0.3,
        "piqa": 0.4,
        "truthfulqa_mc1": 0.5,
        "winogrande": 0.6,
    },
}
```

即字母排序顺序。

只要 enabled task set 正确，不能因为顺序不同报：

```text
task order mismatch
```

---

# 9. 增加 missing / extra task regression test

还应验证集合校验仍然严格。

## 缺少 task

例如删掉：

```text
boolq
```

必须 fail。

## 多出 task

例如加入：

```text
mmlu_pro
```

即使它存在于 config 中但：

```yaml
enabled: false
```

summary 也必须 fail。

因为 `evaluation_summary.json` 只允许当前 enabled tasks：

```text
truthfulqa_mc1
agieval
arc_challenge
piqa
winogrande
boolq
```

---

# 10. 不需要修改 evaluator

当前：

```text
scripts/evaluate_lm_harness.py
```

已经正确构造：

```python
summary = {
    "task_times": task_times,
    "task_metrics": task_metrics,
}
```

并写到：

```text
<evaluation_root>/evaluation_summary.json
```

不需要为了本轮 verifier 修复去改 evaluator。

尤其不要关闭全局：

```python
sort_keys=True
```

因为这是整个项目 `write_json()` 的统一行为。

正确修法是：

> verifier 不应该依赖 JSON object key 顺序。

---

# 11. TL;DR 完全不动

确认不要修改：

```text
scripts/evaluate_tldr.py
```

不要修改：

```text
resolve_tldr_evaluation_layout
TL;DR evaluation paths
metrics.json
selection.json
ROUGE
```

本轮所有修改都只属于：

```text
Tulu-3 / lm-eval verifier + experiment documentation
```

---

# 12. 完成后测试

修改后执行：

```bash
cd /home/wangwenkang/SNN

pytest -q
```

必须全部通过。

然后正式评估前，由用户自行清理旧 evaluation artifact。

例如仅作为操作原则：

```text
删除对应旧 evaluation 输出
→ 使用新版代码重跑
→ 新目录只产生 task_results/ + evaluation_summary.json
→ 再运行 verify_artifacts.py
```

不要把这一步自动化进 evaluator。

---

# 13. 最终验收条件

本轮修改完成后必须满足：

1. `evaluation_summary.json` verifier 不再依赖 task key 顺序。
2. `task_times` 的 key 集合严格等于 enabled task 集合。
3. `task_metrics` 的 key 集合严格等于 enabled task 集合。
4. 缺少 enabled task 必须 fail。
5. 多出 disabled / unknown task 必须 fail。
6. `task_times` 的 `HH:MM:SS` validation 保留。
7. `task_metrics` 与 per-task `results.json` canonical metric 一致性 validation 保留。
8. task runtime 与 `results.json` timing provenance 一致性 validation 保留。
9. 旧 `evaluation_timing.json` stale 检查保留。
10. evaluator 不新增自动删除旧 artifact 的逻辑。
11. `scripts/evaluate_lm_harness.py` 的 summary 写入逻辑不需要改。
12. `write_json(sort_keys=True)` 不改。
13. `实验执行总结.md` 中：
    ```text
    agieval test_samples
    ```
    改为：
    ```text
    1000
    ```
14. `configs/experiment_matrix.yaml` 中 AGIEval 保持 `test_samples: 1000`。
15. TL;DR summarization 完全不变。
16. `pytest -q` 全部通过。
