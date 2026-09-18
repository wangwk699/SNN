# SNN `phase.base` 第二轮修补方案

> 仓库：`https://github.com/wangwk699/SNN`  
> 分支：`main`  
> 检查基准提交：`af9fcb1e6e234fe59614ad86678ec8caa6dbe5d1`
>
> 本文档用于在上一轮 `phase.base` 可调参化已经完成的基础上，修补 provenance、runtime 参数显式性、regression 覆盖和 sequential calibration deployment 边界问题。Codex 应仅依据本文档完成修改。

---

## 1. 本轮不改的核心数学

保持：

\[
a_t=\tau b^{-(t+1)}
\]

\[
v_0=\frac12\tau b^{-T}
\]

\[
B_{\mathrm{phase}}=\tau\frac{1-b^{-T}}{b-1}
\]

以及：

```text
phase.base: finite float and > 1.0
```

现有 `tau` calibration 统计规则完全不变。

本轮只修补：

1. post-finetuning conversion 下 source ANN base/T provenance 丢失；
2. `PhaseSurrogate` / `SiteController` 仍有隐式 `base=2` fallback；
3. `stage_a_parameter_independence`、regression metadata 和 micro regression 漏掉 `phase.base`；
4. `phase_previous_layers_snn=true` 时 deployment base override 的安全边界；
5. 两个小问题：Stage-A 顶层 forbidden runtime field 漏 `phase_base`、identity ANN evaluation 不应记录未参与 forward 的 `phase_base`。

---

# 2. 修复 source ANN training runtime provenance

## 2.1 当前问题

当前 `snn2/conversion.py::_source_bundle()` 只有在：

```python
reused and is_aware_ann_mode(cfg)
```

时才执行：

```python
_validate_aware_training_provenance(...)
```

因此当默认：

```yaml
conversion:
  use_post_finetuning_artifacts: true
```

时：

```python
training_provenance = {}
```

从而 conversion metadata 中：

```json
{
  "source_ann_training_phase_T": null,
  "source_ann_training_phase_base": null,
  "source_ann_training_mtn_T": null
}
```

这是不正确的。

即使使用 post-finetuning calibration，source ANN checkpoint 仍然由确定的 ANN training runtime config 产生。

---

## 2.2 新增 source ANN runtime provenance helper

在：

```text
snn2/conversion.py
```

新增类似：

```python
def _load_source_ann_training_runtime_provenance(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
) -> dict[str, Any]:
    ...
```

仅对：

```text
phase_aware
gif_aware
```

启用。

读取：

```text
layout.ann_dir / "training_result.json"
layout.ann_checkpoint_dir / "config.json"
```

必须得到：

```text
ann_training_phase_T
ann_training_phase_base
ann_training_mtn_T
```

并检查 checkpoint config：

```text
snn2_phase_T
snn2_phase_base
```

与 `training_result.json` 一致。

建议：

```python
if int(ann_config["snn2_phase_T"]) != int(result["ann_training_phase_T"]):
    raise ValueError("source ANN runtime provenance mismatch")
```

以及：

```python
if not math.isclose(
    float(ann_config["snn2_phase_base"]),
    float(result["ann_training_phase_base"]),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise ValueError("source ANN runtime provenance mismatch")
```

返回：

```python
{
    "ann_training_phase_T": int(...),
    "ann_training_phase_base": float(...),
    "ann_training_mtn_T": int(...),
}
```

---

## 2.3 `_source_bundle()` 逻辑

对于 aware ANN，无论：

```text
use_post_finetuning_artifacts = true / false
```

都先读取：

```python
source_runtime_provenance = (
    _load_source_ann_training_runtime_provenance(cfg, layout)
    if is_aware_ann_mode(cfg)
    else {}
)
```

然后：

### pre-finetuning reuse

继续执行现有：

```python
_validate_aware_training_provenance(...)
```

验证 ANN-training Prefix / Stage-A / Stage-B 等 frozen artifacts。

### post-finetuning conversion

不要求 post-finetuning calibration 等于 ANN-training calibration，但仍保留：

```text
source_runtime_provenance
```

最终返回的 training provenance 至少应包含 source runtime fields。

---

## 2.4 Conversion metadata

保留字段：

```text
source_ann_training_phase_T
source_ann_training_phase_base
source_ann_training_mtn_T
deployment_phase_T
deployment_phase_base
```

对于 aware ANN，无论 conversion selector 如何：

```text
source_ann_training_phase_base
```

都必须是真实训练值。

合法示例：

```json
{
  "source_ann_training_phase_T": 4,
  "source_ann_training_phase_base": 2.0,
  "source_ann_training_mtn_T": 4,
  "deployment_phase_T": 4,
  "deployment_phase_base": 1.5
}
```

**不得要求 training base == deployment base。**

---

# 3. 更新 post-finetuning conversion tests

修改：

```text
tests/test_conversion_metadata.py
```

当前 post-bundle aware test 不能只检查：

```text
reused_ann_training_artifacts == false
```

还必须检查：

```python
assert metadata["source_ann_training_phase_T"] == 4
assert metadata["source_ann_training_phase_base"] == 2.0
assert metadata["source_ann_training_mtn_T"] == 4
```

同时仍应保持：

```python
assert metadata["use_post_finetuning_artifacts"] is True
assert metadata["calibration_source_stage"] == "post_finetuning"
assert metadata["reused_ann_training_artifacts"] is False
assert metadata["post_finetuning_recalibration"] is True
```

---

# 4. 移除 `PhaseSurrogate` 的隐式 base=2

当前：

```python
class PhaseSurrogate(nn.Module):
    def __init__(
        ...,
        T: int,
        base: float = 2.0,
        ...
    ):
```

改为：

```python
class PhaseSurrogate(nn.Module):
    def __init__(
        self,
        state: dict[str, Any],
        *,
        T: int,
        base: float,
        surrogate_slope: float | None = None,
    ):
```

即：

```text
base 必填
```

不要保留 default。

然后全仓库所有：

```python
PhaseSurrogate(
```

调用点必须显式传：

```python
base=...
```

production 代码使用当前 cfg/controller base。

tests 若测试旧 base=2 行为，则显式：

```python
base=2.0
```

---

# 5. 移除 `SiteController` 的隐式 base=2

当前：

```python
self.phase_base = validate_phase_base(
    2.0 if phase_base is None else phase_base
)
```

改为：

```python
self.phase_base = (
    None
    if phase_base is None
    else validate_phase_base(phase_base)
)
```

这样：

```text
phase_base=None
```

真正表示“此 Controller 未配置 Phase runtime”。

---

## 5.1 Phase-aware ANN

保留严格检查：

```python
if (
    self.mode == "phase"
    and (
        self.phase_surrogate_slope is None
        or self.phase_T is None
        or self.phase_base is None
    )
):
    raise ValueError(
        "Phase ANN replacement requires explicit phase_T, "
        "phase_base and phase_surrogate_slope"
    )
```

---

## 5.2 Phase deployment

保留：

```python
if neuron == "phase":
    if self.phase_T is None or self.phase_base is None:
        raise ValueError(
            "Phase deployment requires phase_T and phase_base"
        )
```

现在这个检查才真正有意义。

---

## 5.3 Sequential Phase calibration

在：

```python
begin_sequential_calibration(...)
```

中建议提前增加：

```python
if neuron == "phase":
    if self.phase_T is None or self.phase_base is None:
        raise ValueError(
            "Sequential Phase calibration requires phase_T and phase_base"
        )
```

避免后续 `_load()` 才报错。

---

# 6. 更新测试中的 PhaseSurrogate 调用

例如当前：

```python
PhaseSurrogate(_phase_state(), T=4)
```

改为：

```python
PhaseSurrogate(
    _phase_state(),
    T=4,
    base=2.0,
)
```

包括：

```text
tests/test_neurons.py
tests/test_phase_conversion_regression.py
其他 grep 命中的测试
```

新增：

```python
def test_phase_surrogate_requires_explicit_base():
    with pytest.raises(TypeError):
        PhaseSurrogate(_phase_state(), T=4)
```

以及：

```python
def test_identity_controller_does_not_default_phase_base():
    controller = SiteController(mode="identity")
    assert controller.phase_base is None
```

和：

```python
def test_phase_controller_requires_explicit_phase_base():
    with pytest.raises(ValueError, match="phase_base"):
        SiteController(
            mode="phase",
            phase_T=4,
            phase_surrogate_slope=1.0,
        )
```

---

# 7. 修复 `stage_a_parameter_independence`

文件：

```text
snn2/calibration.py
```

当前 independence list 里只有：

```python
("phase.T", flags["phase"])
```

必须增加：

```python
("phase.base", flags["phase"])
```

最终类似：

```python
"stage_a_parameter_independence": [
    name
    for name, active in (
        ("phase.T", flags["phase"]),
        ("phase.base", flags["phase"]),
        ("mtn.T", flags["mtn"]),
        ("mtn.K", flags["mtn"]),
        ("mtn.threshold_factor", flags["mtn"]),
    )
    if not active
]
```

语义：

### 当

```yaml
phase_previous_layers_snn: false
```

必须包含：

```text
phase.T
phase.base
```

### 当

```yaml
phase_previous_layers_snn: true
```

二者都不能出现在 independence list。

补对应 tests。

---

# 8. 修复 Phase conversion micro regression

文件：

```text
scripts/regress_phase_conversion.py
```

当前：

```python
micro = run_phase_neuron_micro_regression(
    layout.ann_training_site_dir,
    num_layers,
    phase_T=int(cfg["phase"]["T"]),
)
```

改为：

```python
micro = run_phase_neuron_micro_regression(
    layout.ann_training_site_dir,
    num_layers,
    phase_T=int(cfg["phase"]["T"]),
    phase_base=float(cfg["phase"]["base"]),
)
```

否则：

```bash
--phase-base 1.5
```

时 full graph 用 1.5，但 micro regression 仍默认使用 2.0。

---

# 9. `run_phase_neuron_micro_regression()` 取消 default base

当前：

```python
def run_phase_neuron_micro_regression(
    ...,
    phase_T: int,
    phase_base: float = 2.0,
    ...
):
```

改为：

```python
def run_phase_neuron_micro_regression(
    ...,
    phase_T: int,
    phase_base: float,
    ...
):
```

所有 caller/tests 显式传 base。

---

# 10. Regression metadata 加 source/deployment base

在：

```text
scripts/regress_phase_conversion.py
```

override 前保存：

```python
source_phase_T = int(cfg["phase"]["T"])
source_phase_base = float(cfg["phase"]["base"])
source_mtn_T = int(cfg["mtn"]["T"])
```

然后：

```python
cfg = apply_deployment_overrides(args, cfg)
```

metadata 增加：

```python
"source_phase_T": source_phase_T,
"source_phase_base": source_phase_base,
"deployment_phase_T": int(cfg["phase"]["T"]),
"deployment_phase_base": float(cfg["phase"]["base"]),
```

示例：

```json
{
  "source_phase_T": 4,
  "source_phase_base": 2.0,
  "deployment_phase_T": 4,
  "deployment_phase_base": 1.5
}
```

---

# 11. Regression output path 按 deployment base/T 隔离

当前：

```python
output_dir = (
    layout.root
    / "analysis"
    / "phase_conversion_regression"
)
```

会导致同一个 ANN checkpoint 的多个 base sweep 互相覆盖。

改为：

```python
from snn2.artifacts import phase_snn_dirname
```

然后：

```python
output_dir = (
    layout.root
    / "analysis"
    / "phase_conversion_regression"
    / phase_snn_dirname(
        cfg["phase"]["base"],
        cfg["phase"]["T"],
    )
)
```

例如：

```text
analysis/
phase_conversion_regression/
phase_base_1.5_T_4/
```

---

# 12. Stage-A 顶层 forbidden runtime field 增加 `phase_base`

文件：

```text
snn2/state_validation.py
```

当前：

```python
forbidden_paths = _forbidden_manifest_paths(
    runtime_free_manifest,
    {
        "phase_T",
        "mtn_T",
        "mtn_K",
        "max_spikes",
        "v0",
    },
)
```

增加：

```text
phase_base
```

即：

```python
{
    "phase_T",
    "phase_base",
    "mtn_T",
    "mtn_K",
    "max_spikes",
    "v0",
}
```

注意继续保留：

```python
runtime_free_manifest.pop("calibration_trajectory", None)
```

因为：

```text
calibration_trajectory.phase.phase_base
```

属于合法 trajectory provenance。

新增测试：

```python
manifest["phase_base"] = 1.5
```

必须触发：

```text
Stage A manifest contains runtime-dependent fields
```

---

# 13. Identity ANN evaluation 不应记录 phase_base

文件：

```text
snn2/evaluation.py
```

当前：

```python
"phase_base":
    float(cfg["phase"]["base"])
    if neuron in {"ann", "phase"}
    else None
```

这会让：

```text
vanilla ANN
unaware ANN
```

也记录 base，虽然实际 forward 是 identity。

改为：

```python
"phase_base": (
    float(cfg["phase"]["base"])
    if (
        neuron == "phase"
        or (
            neuron == "ann"
            and controller.mode == "phase"
        )
    )
    else None
),
```

如果 metadata 中还有类似 `phase_T` 字段，也按同样原则处理。

预期：

### phase-aware ANN

```json
{
  "evaluation_forward_kind": "phase_surrogate_ann",
  "phase_base": 2.0
}
```

### Phase SNN

```json
{
  "evaluation_forward_kind": "temporal_phase_snn",
  "phase_base": 1.5
}
```

### vanilla/unaware ANN

```json
{
  "evaluation_forward_kind": "identity_ann",
  "phase_base": null
}
```

---

# 14. `phase_previous_layers_snn=true` 的 deployment override 规则

这是本轮必须明确的边界。

## 14.1 common trajectory

当：

```yaml
phase_previous_layers_snn: false
```

Stage-A `tau` state 对：

```text
phase.T
phase.base
```

独立。

所以允许：

```text
ANN training base = 2.0
deployment base = 1.5
```

直接使用同一个 common Stage-A state。

---

## 14.2 sequential Phase trajectory

当：

```yaml
phase_previous_layers_snn: true
```

Stage-A activation trajectory 已依赖：

```text
phase.T
phase.base
```

因此：

```text
calibration base = 2.0
deployment base = 1.5
```

必须拒绝。

当前 state provenance 已能在后续发现 mismatch，但建议增加更明确的入口 guard。

---

# 15. 本轮不要给 `calibrate_sites.py` 新增 `--phase-base`

不要为了 sequential calibration 再加一套 calibration override CLI。

本轮推荐：

> 如果 `phase_previous_layers_snn=true` 且 deployment base 与已存在 sequential calibration base 不一致，则直接报错。

以后如果确实需要：

```text
same ANN checkpoint
+
different sequential calibration base
```

再单独设计 calibration override protocol。

---

# 16. 建议新增 deployment/calibration compatibility guard

可在：

```text
snn2/conversion.py
```

新增：

```python
def _validate_phase_deployment_calibration_runtime(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    *,
    neuron: str,
) -> None:
    ...
```

仅：

```python
if neuron == "phase"
```

时执行。

如果：

```python
previous_layers_snn_enabled(cfg, "phase")
```

为 true，则检查 selected conversion calibration state/manifest 中：

```text
calibration_phase_T
calibration_phase_base
```

与当前 deployment：

```text
cfg["phase"]["T"]
cfg["phase"]["base"]
```

一致。

不一致时明确报：

```text
Phase deployment override is incompatible with sequential Phase calibration.
Regenerate sequential calibration for the requested phase.T/phase.base.
```

---

# 17. 注意：不要和 source ANN training base 比较

这里必须比较：

```text
deployment runtime
vs
selected sequential calibration runtime
```

不是比较：

```text
deployment base
vs
source ANN training base
```

因此以下是合法的：

```text
source ANN base = 2.0
selected calibration = common ANN trajectory
deployment base = 1.5
```

只有 sequential calibration 才要求 base 一致。

---

# 18. 对应 tests

至少新增：

## 18.1 common trajectory 允许 override

```text
source ANN base = 2.0
phase_previous_layers_snn = false
deployment base = 1.5
```

应允许 conversion。

并得到：

```python
metadata["source_ann_training_phase_base"] == 2.0
metadata["deployment_phase_base"] == 1.5
```

---

## 18.2 sequential trajectory mismatch 拒绝

```text
phase_previous_layers_snn = true
calibration_phase_base = 2.0
deployment_phase_base = 1.5
```

必须失败。

---

## 18.3 sequential trajectory match 通过

```text
phase_previous_layers_snn = true
calibration_phase_base = 1.5
deployment_phase_base = 1.5
```

应通过。

---

# 19. `scripts/verify_artifacts.py` 加强

对于：

```text
phase_aware
gif_aware
```

读取：

```text
training_result.json
ann/final/config.json
```

检查：

```text
ann_training_phase_T
ann_training_phase_base
```

与：

```text
snn2_phase_T
snn2_phase_base
```

一致。

如果存在 conversion metadata，则：

```text
source_ann_training_phase_base
```

必须等于 training result 中的 base。

但绝对不要检查：

```text
deployment_phase_base == source_ann_training_phase_base
```

因为 deployment sweep 允许不同。

---

# 20. 重点文件

至少检查并按需修改：

```text
snn2/conversion.py
snn2/neurons.py
snn2/controller.py
snn2/calibration.py
snn2/state_validation.py
snn2/evaluation.py
snn2/phase_conversion_regression.py
scripts/regress_phase_conversion.py
scripts/verify_artifacts.py
tests/test_conversion_metadata.py
tests/test_neurons.py
tests/test_calibration_profiles.py
tests/test_post_finetuning_protocol.py
tests/test_phase_conversion_regression.py
tests/test_evaluation_paths.py
```

---

# 21. 全仓库扫描

修改后执行：

```bash
grep -R "PhaseSurrogate(" -n snn2 scripts tests
grep -R "SiteController(" -n snn2 scripts tests
grep -R "phase_base: float = 2.0" -n snn2 scripts tests
grep -R "2.0 if phase_base is None" -n snn2 scripts tests
grep -R "run_phase_neuron_micro_regression" -n .
grep -R "source_ann_training_phase_base" -n snn2 scripts tests
grep -R "stage_a_parameter_independence" -n snn2 tests
```

目标：

- 所有 `PhaseSurrogate` caller 显式传 base；
- 不再存在 Phase runtime 默认 base=2；
- regression micro caller 显式传 deployment base；
- source ANN base provenance 在 pre/post conversion 都完整。

---

# 22. 不要误删合法的显式 base=2

以下仍是合法的：

```python
PhaseSurrogate(
    state,
    T=4,
    base=2.0,
)
```

本轮禁止的是：

```text
implicit fallback
```

而不是禁止：

```text
explicit base=2
```

---

# 23. 最终 API 语义

## identity

允许：

```python
SiteController(
    mode="identity",
    site_root=...,
)
```

此时：

```python
controller.phase_base is None
```

---

## phase-aware ANN

必须：

```python
SiteController(
    mode="phase",
    site_root=...,
    phase_T=4,
    phase_base=1.5,
    phase_surrogate_slope=1.0,
)
```

---

## Phase SNN

必须：

```python
controller = SiteController(
    mode="identity",
    site_root=...,
    phase_T=4,
    phase_base=1.5,
)
controller.set_deployment(
    "phase",
    clip_bundle_policy="forbid_all",
)
```

---

## GIF / MTN

不应因为 `SiteController` 的构造而被迫提供 Phase base。

---

# 24. 推荐测试命令

必须：

```bash
pytest -q
```

并建议额外：

```bash
pytest -q tests/test_neurons.py
pytest -q tests/test_conversion_metadata.py
pytest -q tests/test_calibration_profiles.py
pytest -q tests/test_post_finetuning_protocol.py
pytest -q tests/test_phase_conversion_regression.py
pytest -q tests/test_evaluation_paths.py
```

---

# 25. 建议人工 smoke test

如果已有对应 artifacts：

```bash
python scripts/convert_snn.py \
  --config "$CFG" \
  --neuron phase \
  --phase-base 1.5
```

确认：

```text
source ANN checkpoint
```

仍然是 YAML training base 对应 checkpoint。

然后：

```bash
python scripts/regress_phase_conversion.py \
  --config "$CFG" \
  --phase-base 1.5 \
  --skip-locked-decode
```

确认：

```json
{
  "source_phase_base": 2.0,
  "deployment_phase_base": 1.5
}
```

并确认 micro regression 使用 1.5。

---

# 26. 最终验收标准

以下全部满足才算完成：

1. `PhaseSurrogate` 不再有：
   ```python
   base: float = 2.0
   ```

2. `SiteController` 不再有：
   ```python
   2.0 if phase_base is None else ...
   ```

3. 所有真实 Phase runtime caller 显式传入 base。

4. `stage_a_parameter_independence` 在 common trajectory 下同时包含：
   ```text
   phase.T
   phase.base
   ```

5. Stage-A manifest 顶层 `phase_base` 被拒绝。

6. `regress_phase_conversion.py --phase-base 1.5`：
   - full graph 使用 1.5；
   - micro regression 也使用 1.5。

7. Regression metadata 同时记录：
   ```text
   source_phase_base
   deployment_phase_base
   ```

8. 不同 deployment base 的 regression 输出目录不互相覆盖。

9. aware ANN 即使：
   ```text
   use_post_finetuning_artifacts=true
   ```
   conversion metadata 仍正确记录：
   ```text
   source_ann_training_phase_base
   ```

10. 合法支持：
    ```text
    source_ann_training_phase_base = 2.0
    deployment_phase_base = 1.5
    ```

11. vanilla/unaware ANN evaluation：
    ```text
    phase_base = null
    ```

12. `phase_previous_layers_snn=true` 时：
    ```text
    sequential calibration base != deployment base
    ```
    必须被拒绝。

13. `phase_previous_layers_snn=false` 时：
    ```text
    training base != deployment base
    ```
    允许进行 deployment sweep。

---

# 27. 核心设计总结

最终必须明确区分三种 runtime identity：

```text
1. source ANN training runtime
2. selected calibration trajectory runtime
3. Phase SNN deployment runtime
```

关系：

```text
source ANN training base
    可以 != deployment base
```

但：

```text
sequential calibration base
    必须 == deployment base
```

而普通 common Stage-A calibration：

```text
phase_previous_layers_snn=false
```

对：

```text
phase.T
phase.base
```

独立，因此可以安全服务多个 deployment base。

这就是本轮所有 provenance、validation 与 regression 修补的核心。
