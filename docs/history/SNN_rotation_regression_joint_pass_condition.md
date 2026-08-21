# Rotation Regression：修改 Passed 联合判定条件

## 目标

将当前 Rotation regression 的通过条件改为同时满足：

```text
relative_l2_error <= 0.05
top1_agreement > 0.95
```

即：

```python
passed = (
    relative_l2_error <= 0.05
    and top1_agreement > 0.95
)
```

注意 Top-1 条件是**严格大于 0.95**，不是 `>= 0.95`。

---

## 1. 配置

在 `configs/experiment_matrix.yaml` 的 `rotation` 下修改 / 增加：

```yaml
rotation:
  regression_relative_l2_threshold: 0.05
  regression_top1_agreement_threshold: 0.95
```

如果 generated configs 会从 `experiment_matrix.yaml` materialize，则修改后重新生成 generated configs，并确认其中也包含：

```yaml
regression_relative_l2_threshold: 0.05
regression_top1_agreement_threshold: 0.95
```

---

## 2. `snn2/rotation.py`

修改 `enforce_rotation_regression()`，让它同时接收：

```python
relative_l2_threshold
top1_agreement_threshold
```

逻辑改为：

```python
relative_l2_threshold = float(relative_l2_threshold)
top1_agreement_threshold = float(top1_agreement_threshold)

relative_l2_passed = (
    float(result["relative_l2_error"])
    <= relative_l2_threshold
)

top1_passed = (
    float(result["top1_agreement"])
    > top1_agreement_threshold
)

checked = {
    **result,
    "threshold": {
        "relative_l2_error": relative_l2_threshold,
        "top1_agreement": top1_agreement_threshold,
    },
    "passed": (
        relative_l2_passed
        and top1_passed
    ),
}
```

如果 `passed=False`，继续抛 `RotationRegressionError`。

同时修改 `RotationRegressionError` 的报错信息，使其同时显示：

```text
relative_l2_error / threshold
top1_agreement / threshold
```

例如：

```text
Rotation logits regression failed:
relative_l2_error=... (required <= 0.05),
top1_agreement=... (required > 0.95)
```

---

## 3. `validate_rotation_logits()`

调用 `enforce_rotation_regression()` 时改为：

```python
return enforce_rotation_regression(
    result,
    float(
        cfg["rotation"].get(
            "regression_relative_l2_threshold",
            0.05,
        )
    ),
    float(
        cfg["rotation"].get(
            "regression_top1_agreement_threshold",
            0.95,
        )
    ),
)
```

默认值同步改为：

```text
L2   = 0.05
Top1 = 0.95
```

---

## 4. `rotation_regression.json`

最终 threshold 字段改为：

```json
"threshold": {
  "relative_l2_error": 0.05,
  "top1_agreement": 0.95
}
```

`passed` 只有在：

```text
relative_l2_error <= 0.05
AND
top1_agreement > 0.95
```

时才为 `true`。

不需要因为这次只修改判定规则而再次升级 `format_version`；保持当前版本即可。

---

## 5. `scripts/verify_artifacts.py`

更新 Rotation regression 的一致性检查。

必须检查：

```python
threshold = regression["threshold"]

relative_l2_threshold = float(
    threshold["relative_l2_error"]
)

top1_threshold = float(
    threshold["top1_agreement"]
)

expected_passed = (
    float(regression["relative_l2_error"])
    <= relative_l2_threshold
    and
    float(regression["top1_agreement"])
    > top1_threshold
)

if bool(regression["passed"]) != expected_passed:
    raise ValueError(
        "Rotation regression passed flag "
        "contradicts its hard gates"
    )
```

同时要求：

```text
threshold.relative_l2_error
threshold.top1_agreement
```

两个字段都存在。

---

## 6. Tests

更新 `tests/test_rotation_regression.py`。

至少覆盖以下情况：

### Case 1：两个条件都满足 → PASS

```text
relative_l2_error = 0.04
top1_agreement    = 0.96
```

结果：

```text
passed = True
```

### Case 2：L2 超阈值 → FAIL

```text
relative_l2_error = 0.06
top1_agreement    = 0.99
```

结果：

```text
passed = False
```

### Case 3：Top-1 低于阈值 → FAIL

```text
relative_l2_error = 0.01
top1_agreement    = 0.94
```

结果：

```text
passed = False
```

### Case 4：Top-1 恰好等于 0.95 → FAIL

```text
relative_l2_error = 0.01
top1_agreement    = 0.95
```

结果：

```text
passed = False
```

因为要求：

```text
top1_agreement > 0.95
```

### Case 5：L2 恰好等于 0.05，Top-1 > 0.95 → PASS

```text
relative_l2_error = 0.05
top1_agreement    = 0.96
```

结果：

```text
passed = True
```

---

## 7. 文档同步

把项目中关于 Rotation regression hard gate 的旧描述：

```text
relative_l2_error <= 0.01
```

统一改为：

```text
relative_l2_error <= 0.05
AND
top1_agreement > 0.95
```

重点检查：

```text
实验执行总结.md
```

以及其他明确写死旧 `0.01` 判定标准的 Markdown。

---

## 8. 修改完成后验证

运行：

```bash
python -m compileall -q snn2 scripts tests

python -m pytest -q \
  tests/test_rotation_regression.py

python -m pytest -q
```

然后重新运行三个模型的：

```bash
python scripts/prepare_rotation.py \
  --config "$ROT_CFG"
```

按当前已有结果，新的判定规则下三个模型预期都应通过：

```text
Qwen3-1.7B:
L2   = 0.007705 < 0.05
Top1 = 0.962217 > 0.95
=> PASS

Qwen3-8B:
L2   = 0.018208 < 0.05
Top1 = 0.967143 > 0.95
=> PASS

Llama3-8B:
L2   = 0.033239 < 0.05
Top1 = 0.976498 > 0.95
=> PASS
```
