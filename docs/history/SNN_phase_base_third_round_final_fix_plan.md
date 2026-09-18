# SNN `phase.base` 第三轮收尾修补方案

> 目标仓库：`https://github.com/wangwk699/SNN`  
> 目标分支：`main`  
> 本轮检查基准提交：`ca6e97de8dc40500abe919b3712caa02ee327870`
>
> 本文档仅处理当前剩余的收尾问题。上一轮 `phase.base` 的核心数学、路径隔离、deployment override、source/deployment provenance 分离均已基本正确，本轮不要重新改动这些稳定逻辑。

---

## 1. 本轮不改的核心逻辑

保持：

\[
a_t = 	au b^{-(t+1)}
\]

\[
v_0 = rac12	au b^{-T}
\]

\[
B_{\mathrm{phase}}
=
	aurac{1-b^{-T}}{b-1}
\]

以及现有 `tau` calibration。

当前代码已经正确支持：

```text
source ANN training phase.base = 2.0
deployment Phase SNN phase.base = 1.5
```

并且 `--phase-base` 不应改变 source ANN checkpoint。

本轮只修两个必改点：

1. `phase_previous_layers_snn=true` 时，sequential Phase calibration guard 目前存在 fail-open 分支；
2. matching sequential calibration/deployment runtime 的通过测试目前 monkeypatch 了 `_source_bundle()`，不是真正 end-to-end。

另有一个可选 provenance 增强：为 ANN checkpoint 增加 `snn2_mtn_T` 并做交叉验证，但本轮不强制。

---

## 2. 必改一：sequential Phase guard 改为 fail-closed

文件：

```text
snn2/conversion.py
```

当前 `_validate_phase_deployment_calibration_runtime()` 的逻辑中存在类似：

```python
if neuron != "phase" or not previous_layers_snn_enabled(cfg, "phase"):
    return

manifest_path = layout.conversion_site_dir / "calibration_state_manifest.json"
if not manifest_path.exists():
    return

trajectory = (
    read_json(manifest_path)
    .get("calibration_trajectory", {})
    .get("phase", {})
)

if (
    not isinstance(trajectory, dict)
    or not trajectory.get("previous_layers_snn")
):
    return
```

这里不够严格。

当配置明确：

```yaml
calibration:
  phase_previous_layers_snn: true
```

时，selected calibration artifact 必须是 sequential Phase trajectory。因此以下情况都必须报错，而不能 `return`：

```text
manifest 缺失
calibration_trajectory 缺失
calibration_trajectory.phase 缺失
previous_layers_snn != true
phase_T 缺失
phase_base 缺失
phase_T 不匹配
phase_base 不匹配
```

推荐重写为：

```python
def _validate_phase_deployment_calibration_runtime(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    neuron: str | None,
) -> None:
    if neuron != "phase":
        return

    if not previous_layers_snn_enabled(cfg, "phase"):
        return

    manifest_path = (
        layout.conversion_site_dir
        / "calibration_state_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Sequential Phase deployment requires calibration manifest: "
            f"{manifest_path}"
        )

    manifest = read_json(manifest_path)

    calibration_trajectory = manifest.get("calibration_trajectory")
    if not isinstance(calibration_trajectory, dict):
        raise ValueError(
            "Phase deployment requires sequential Phase calibration "
            "provenance when phase_previous_layers_snn=true"
        )

    trajectory = calibration_trajectory.get("phase")
    if (
        not isinstance(trajectory, dict)
        or trajectory.get("previous_layers_snn") is not True
    ):
        raise ValueError(
            "Phase deployment requires sequential Phase calibration "
            "provenance when phase_previous_layers_snn=true"
        )

    expected_T = int(cfg["phase"]["T"])
    expected_base = validate_phase_base(cfg["phase"]["base"])

    actual_T = trajectory.get("phase_T")
    actual_base = trajectory.get("phase_base")

    matches = (
        isinstance(actual_T, int)
        and not isinstance(actual_T, bool)
        and actual_T == expected_T
        and isinstance(actual_base, (int, float))
        and not isinstance(actual_base, bool)
        and math.isclose(
            float(actual_base),
            expected_base,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )

    if not matches:
        raise ValueError(
            "Phase deployment runtime disagrees with sequential "
            "Stage-A calibration "
            f"(calibrated phase_T={actual_T}, "
            f"phase_base={actual_base}; "
            f"deployment phase_T={expected_T}, "
            f"phase_base={expected_base}). "
            "Re-run calibration for the selected deployment parameters."
        )
```

---

## 3. 为什么必须 fail-closed

当：

```yaml
phase_previous_layers_snn: true
```

时，后续 block 的 calibration activation trajectory 已经受到前层 Temporal Phase Neuron 影响。

因此：

```text
phase.T
phase.base
```

都属于 sequential calibration trajectory identity。

如果 artifact 没有完整记录：

```text
previous_layers_snn = true
phase_T
phase_base
```

就不能证明该 artifact 属于当前 deployment runtime。

因此应采用：

```text
missing provenance -> reject
```

而不是：

```text
missing provenance -> silently accept
```

---

## 4. common trajectory 仍然允许 deployment sweep

以下逻辑必须保持不变：

```yaml
phase_previous_layers_snn: false
```

此时 Stage-A `tau` statistics 对：

```text
phase.T
phase.base
```

独立。

因此：

```text
ANN training base = 2.0
deployment base = 1.5
```

仍然合法。

也就是说，guard 的前两层必须保留：

```python
if neuron != "phase":
    return

if not previous_layers_snn_enabled(cfg, "phase"):
    return
```

---

## 5. 不要比较 source ANN training base 与 deployment base

这里比较的是：

```text
selected sequential calibration runtime
vs
deployment runtime
```

不是：

```text
source ANN training runtime
vs
deployment runtime
```

以下必须合法：

```text
source ANN training base = 2.0
selected calibration = common trajectory
deployment base = 1.5
```

以下才必须拒绝：

```text
phase_previous_layers_snn = true
selected calibration base = 2.0
deployment base = 1.5
```

---

## 6. 必改二：补真正 end-to-end sequential Phase conversion test

当前：

```text
test_sequential_phase_calibration_allows_matching_deployment_base
```

使用：

```python
monkeypatch.setattr(
    "snn2.conversion._source_bundle",
    ...
)
```

所以它没有真正验证：

```text
ArtifactLayout
Stage-A manifest
effective_previous_layers_snn
state_statistics_source
phase_state provenance
final RMSNorm Phase state provenance
_source_bundle()
validate_site_state_bundle()
validate_calibration_trajectory_provenance()
create_conversion()
```

能否完整一起通过。

必须增加一个不 monkeypatch `_source_bundle()` 的真实 end-to-end 测试。

---

## 7. End-to-end 测试目标

构造：

```text
phase_previous_layers_snn = true
calibration phase_T = 4
calibration phase_base = 1.5
deployment phase_T = 4
deployment phase_base = 1.5
```

然后直接：

```python
metadata = create_conversion(
    cfg,
    layout,
    "phase",
)
```

必须成功。

---

## 8. Fixture 必须完整一致

不能只改：

```python
manifest["calibration_trajectory"]["phase"]
```

至少要保证：

```text
requested_previous_layers_snn.phase = true
effective_previous_layers_snn.phase = true
```

manifest 中：

```json
{
  "calibration_trajectory": {
    "phase": {
      "previous_layers_snn": true,
      "source": "sequential_temporal_phase",
      "phase_T": 4,
      "phase_base": 1.5
    }
  }
}
```

每个 Phase state：

```text
previous_layers_snn = true
calibration_phase_T = 4
calibration_phase_base = 1.5
```

Final RMSNorm Phase state 同样必须记录：

```text
previous_layers_snn = true
calibration_phase_T = 4
calibration_phase_base = 1.5
```

并且 Phase statistics source provenance 必须与真实 sequential calibration 一致。

如果项目已有可复用的：

```python
materialize_calibration_states(...)
```

以及 sequential statistics fixture，优先用真实 materialization helper，不要手写不完整 state。

---

## 9. 建议形成 5 类 sequential coverage

至少保留/新增：

### A. common trajectory override 合法

```text
phase_previous_layers_snn=false
source ANN base=2.0
deployment base=1.5
```

结果：通过。

### B. sequential trajectory provenance 缺失

```text
phase_previous_layers_snn=true
calibration_trajectory.phase 缺失
```

结果：失败。

### C. sequential flag 不一致

```text
phase_previous_layers_snn=true
trajectory.previous_layers_snn=false
```

结果：失败。

### D. sequential runtime mismatch

```text
calibration base=2.0
deployment base=1.5
```

结果：失败。

### E. sequential runtime match

```text
calibration base=1.5, T=4
deployment base=1.5, T=4
```

结果：真实 end-to-end 通过。

---

## 10. 对 `validate_conversion_metadata()` 也要覆盖

`validate_conversion_metadata()` 也会重新调用 sequential runtime guard。

建议至少增加一个测试：

1. 先生成合法 `conversion_metadata.json`；
2. 再篡改 calibration manifest，使 sequential Phase provenance 缺失或 base 不匹配；
3. 调用：
   ```python
   validate_conversion_metadata(...)
   ```
4. 必须失败。

这样能保证：

```text
生成 conversion 时安全
+
后续 evaluation 前重新验证时仍安全
```

---

## 11. verifier 不要复制一套 sequential guard

当前：

```text
scripts/verify_artifacts.py
```

会调用：

```python
validate_conversion_metadata(
    cfg,
    layout,
    neuron,
)
```

因此只要 `validate_conversion_metadata()` 内部的 guard 正确，verifier 就自动获得保护。

不要在 `verify_artifacts.py` 再复制一套：

```text
phase_T
phase_base
previous_layers_snn
```

比较逻辑。

避免未来两套 validator 漂移。

---

## 12. 可选增强：`snn2_mtn_T`

这一项不是必改。

当前 `training_result.json` 已记录：

```text
ann_training_mtn_T
```

但 ANN checkpoint `config.json` 只保存：

```text
snn2_phase_T
snn2_phase_base
```

没有：

```text
snn2_mtn_T
```

因为 aware ANN common Clip 也可能依赖 `mtn.T`，从 provenance 对称性上可以增加：

```python
model.config.snn2_mtn_T = int(
    cfg["mtn"]["T"]
)
```

然后 `_load_source_ann_training_runtime_provenance()` 交叉验证：

```text
training_result["ann_training_mtn_T"]
==
checkpoint config["snn2_mtn_T"]
```

但这会使旧 ANN checkpoint 缺少该字段。

因此本轮推荐：

> 不把 `snn2_mtn_T` 设为强制要求，只记录为后续 provenance enhancement。

---

## 13. 本轮必改文件

主要：

```text
snn2/conversion.py
tests/test_conversion_metadata.py
```

如果为了构造真实 sequential fixture，也可能修改：

```text
tests fixture/helper utilities
```

不需要重新修改：

```text
snn2/phase_math.py
snn2/neurons.py
snn2/artifacts.py
scripts/_common.py
```

除非测试暴露新的直接问题。

---

## 14. 不要修改的稳定逻辑

本轮不要重新修改：

```text
Phase amplitude
v0
Stage-B PhaseBound
tau calibration
ArtifactLayout source/deployment split
--phase-base CLI
phase_snn_dirname
ANN training path
calibration result path
clip profile path
source_ann_training_phase_base / deployment_phase_base 允许不同
```

---

## 15. 建议测试名

新增：

```python
def test_sequential_phase_calibration_missing_trajectory_fails():
    ...
```

```python
def test_sequential_phase_calibration_false_trajectory_fails():
    ...
```

```python
def test_sequential_phase_calibration_matching_runtime_end_to_end():
    ...
```

以及可选：

```python
def test_validate_conversion_metadata_rejects_stale_sequential_phase_runtime():
    ...
```

保留已有：

```python
test_common_phase_calibration_allows_deployment_base_override
```

```python
test_sequential_phase_calibration_rejects_deployment_base_override
```

---

## 16. 行为矩阵

最终必须满足：

| Config | Calibration artifact | Deployment | 结果 |
|---|---|---|---|
| `phase_previous_layers_snn=false` | common | base 1.5 | 允许 |
| `phase_previous_layers_snn=true` | trajectory 缺失 | base 1.5 | 拒绝 |
| `phase_previous_layers_snn=true` | `previous_layers_snn=false` | base 1.5 | 拒绝 |
| `phase_previous_layers_snn=true` | base 2.0 | base 1.5 | 拒绝 |
| `phase_previous_layers_snn=true` | base 1.5, T=4 | base 1.5, T=4 | 允许 |
| `phase_previous_layers_snn=true` | base 1.5, T=4 | base 1.5, T=6 | 拒绝 |

---

## 17. 建议 grep

修改后执行：

```bash
grep -R "_validate_phase_deployment_calibration_runtime" -n snn2 tests
grep -R "phase_previous_layers_snn" -n tests/test_conversion_metadata.py
grep -R "calibration_phase_base" -n snn2 tests
grep -R "calibration_trajectory" -n tests/test_conversion_metadata.py
```

确认 sequential Phase 路径没有新的 fail-open。

---

## 18. 必须运行

```bash
pytest -q
```

必须全部通过。

建议额外：

```bash
pytest -q tests/test_conversion_metadata.py
pytest -q tests/test_calibration_profiles.py
pytest -q tests/test_post_finetuning_protocol.py
```

---

## 19. 最终验收标准

全部满足才算完成：

1. `phase_previous_layers_snn=true` 时，缺失 sequential Phase provenance 不再静默通过。
2. `calibration_trajectory.phase.previous_layers_snn` 必须显式为 `true`。
3. sequential calibration 的 `phase_T`、`phase_base` 必须与 deployment 一致。
4. common trajectory (`phase_previous_layers_snn=false`) 仍允许 deployment-only base sweep。
5. 至少有一个不 monkeypatch `_source_bundle()` 的真实 matching-runtime sequential conversion test。
6. `create_conversion()` 和 `validate_conversion_metadata()` 都受同一 guard 保护。
7. 不新增 `deployment base == source ANN training base` 的错误限制。

---

## 20. 最终结论

本轮不再修改 `phase.base` 的核心实现，只把最后一个 provenance 边界从：

```text
missing sequential provenance -> silently return
```

改成：

```text
missing sequential provenance -> fail fast
```

并用真实 end-to-end test 证明：

```text
完整一致的 sequential Phase calibration
+
matching deployment phase.base / phase.T
```

可以正常完成 conversion。

完成这两点后，本轮 `phase.base` 可调参化与 deployment sweep 修改即可正式收尾。
