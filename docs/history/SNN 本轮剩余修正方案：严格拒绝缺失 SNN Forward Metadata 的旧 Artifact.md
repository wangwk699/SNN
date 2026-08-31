# SNN 本轮剩余修正方案：严格拒绝缺失 SNN Forward Metadata 的旧 Artifact

目标仓库：`https://github.com/wangwk699/SNN`，基于当前 `main`。

本轮只修改 `scripts/verify_artifacts.py` 及对应测试，不改任何前向传播、calibration、operator count 或 Final RMSNorm topology。

## 1. 修正 `verify_artifacts.py`

当前 SNN evaluation validation 中存在：

```python
if "evaluation_forward_kind" in policy_source:
    ...
```

该逻辑会导致旧 evaluation artifact 如果缺失：

```text
evaluation_forward_kind
controller_mode
temporal_execution
evaluation_common_clip_applied
global_final_norm_replacement
global_final_norm_clip_applied
```

则整段 topology validation 被跳过。

删除该外层条件，改为 **无条件要求这些字段全部存在且完全匹配**：

```python
expected_forward = {
    "evaluation_forward_kind": f"temporal_{neuron}_snn",
    "controller_mode": f"deploy_{neuron}",
    "temporal_execution": True,
    "evaluation_common_clip_applied": False,
    "global_final_norm_replacement": {
        "phase": "temporal_phase",
        "mtn": "temporal_mtn",
        "gif": "identity",
    }[neuron],
    "global_final_norm_clip_applied": False,
}
```

建议显式区分：

```python
missing = [
    key
    for key in expected_forward
    if key not in policy_source
]

mismatched = {
    key: {
        "expected": expected,
        "actual": policy_source.get(key),
    }
    for key, expected in expected_forward.items()
    if key in policy_source
    and policy_source[key] != expected
}

if missing or mismatched:
    raise ValueError(
        f"SNN metrics have incompatible temporal forward metadata: "
        f"missing={missing}, mismatched={mismatched}"
    )
```

禁止 legacy fallback 或字段缺失时跳过验证。

## 2. 补测试

修改：

```text
tests/test_verify_artifacts.py
```

至少增加两个 negative tests：

### 缺失 `evaluation_forward_kind`

从合法 SNN evaluation metadata 中删除：

```python
del metadata["evaluation_forward_kind"]
```

调用 verifier，必须抛出 `ValueError`。

### 缺失 `global_final_norm_replacement`

删除：

```python
del metadata["global_final_norm_replacement"]
```

调用 verifier，必须抛出 `ValueError`。

建议再补一个 mismatch test，例如 MTN：

```python
metadata["global_final_norm_replacement"] = "identity"
```

应被拒绝，因为 MTN 必须是：

```text
temporal_mtn
```

## 3. 不要修改

本轮不要改：

```text
Phase / MTN calibration
Final RMSNorm forward
operator count
evaluation metadata 生成逻辑
Embedding / Prefix temporal policy
10-site topology
schema/version
```

## 4. 完成标准

- [ ] 删除 `if "evaluation_forward_kind" in policy_source:` 兼容分支。
- [ ] SNN forward metadata 缺任意必需字段都直接失败。
- [ ] Phase / GIF / MTN topology metadata 不匹配都直接失败。
- [ ] 补 missing-field negative tests。
- [ ] `pytest -q` 全部通过。