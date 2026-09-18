# SNN 项目 `phase.base` 可调参化、路径隔离与 Deployment Override 完整修改方案

> 目标仓库：`https://github.com/wangwk699/SNN`  
> 基准分支：`main`  
> 本方案基于检查时的 `main` tree SHA：`8f85cb19010e613769681ff2b64e787a8274ed34`  
> 适用实验：
>
> - Qwen3 Base 系列上的 TL;DR summarization
> - Llama 3 Base 上的 Tulu 3 / lm-eval
>
> 本文档应作为本轮代码修改的唯一实现依据。实现者不应依赖此前聊天上下文。

---

## 1. 修改目标

当前配置文件中已经存在：

```yaml
phase:
  T: 4
  base: 2.0
  surrogate_slope: 1.0
  timestep_indexing: t_0_to_T_minus_1
  threshold_rule: "tau * base ** (-(t + 1))"
```

但当前实际 Phase Neuron 运行逻辑仍然把 `base=2` 硬编码在若干关键位置。因此目前 `phase.base` 不是一个真正可调的运行参数。

本轮修改需要完成以下目标：

1. 将 `phase.base` 从“配置字段”升级为真正控制 Phase ANN/SNN 行为的运行时参数。
2. 允许对 `phase.base` 进行实验 sweep。
3. `phase.base` 必须体现在相关 ANN Training、Calibration、Stage-B Clip、Phase SNN conversion/evaluation 的保存路径中，避免不同 base 之间 artifact 覆盖或误复用。
4. Stage-B common clipping 中 Phase 的可表示范围必须从固定 base=2 推广到任意 `base > 1`。
5. 保持现有 `tau` calibration 统计方法完全不变。
6. 新增 CLI：
   ```bash
   --phase-base 1.5
   ```
   使用户可以在**同一个已经训练完成的 ANN checkpoint** 上单独 sweep Phase SNN deployment 的 base，语义与当前 `--phase-T` deployment override 保持一致。
7. 保持 base=2 时与当前代码的数学行为完全兼容。
8. 增加充分的 provenance 和 validation，确保不同 base 的 artifact 不会被静默混用。

---

# 2. 已确认且不可再更改的设计决策

## 2.1 Phase threshold / amplitude 定义

Phase 第 `t` 个 timestep 的幅值继续采用：

\[
a_t = \tau b^{-(t+1)},
\qquad t=0,\ldots,T-1,
\]

其中：

- \(b = \texttt{phase.base}\)
- \(T = \texttt{phase.T}\)
- \(\tau\) 来自现有 calibration

代码语义应对应：

```python
amplitude = tau * base ** (-(timestep + 1))
```

不得继续出现：

```python
tau * 2.0 ** (-(timestep + 1))
```

作为 Phase-specific runtime 公式。

---

## 2.2 `v0` 的推广规则

保留当前“**最小 threshold / amplitude 的一半**”这一原始语义。

最小 amplitude 为：

\[
a_{T-1}=\tau b^{-T}.
\]

因此：

\[
\boxed{
v_0 = \frac{1}{2}\tau b^{-T}
}
\]

代码应对应：

```python
v0 = 0.5 * tau * base ** (-T)
```

注意：

```python
tau * base ** (-(T + 1))
```

**不是**本项目采用的定义。

它只在 `base=2` 时与旧公式数值偶然一致，但 base 改变后不再等价。

---

## 2.3 `phase.base` 的合法范围

合法值必须满足：

```text
finite float
base > 1.0
```

允许示例：

```text
1.1
1.25
1.5
2.0
2.5
3.0
4.0
```

禁止：

```text
base <= 1.0
NaN
+inf
-inf
无法转换为 float 的字符串
bool
```

本轮不支持 `base == 1.0`。

因此不需要为等比数列公式增加 `base == 1` 的特殊分支。

---

## 2.4 `tau` calibration 完全保持不变

当前 Phase `tau` 的统计方法保持原样。

不要因为 base 变化而修改：

- EMA 统计
- abs-max 规则
- channel/group reduction
- clamp 规则
- `phase_state.pt` 中的 `tau`

也不要按照 base 对 `tau` 做重新归一化。

因此改变 base 后，Phase 的总可表示范围会真实发生变化。

这正是本轮 sweep 想要研究的内容。

---

# 3. 任意 base 下的 Phase 可表示范围

当前代码把 PhaseBound 写死为：

\[
\tau(1-2^{-T}).
\]

这是 `base=2` 时的特例。

一般情况下：

\[
\sum_{t=0}^{T-1}\tau b^{-(t+1)}
=
\tau\sum_{k=1}^{T}b^{-k}.
\]

利用等比数列求和：

\[
\sum_{k=1}^{T}b^{-k}
=
\frac{1-b^{-T}}{b-1}.
\]

所以 Phase 最大绝对可表示幅值必须改为：

\[
\boxed{
B_{\mathrm{phase}}
=
\tau\frac{1-b^{-T}}{b-1}
}
\]

对应 PhaseBound：

\[
\boxed{
\left[
-\tau\frac{1-b^{-T}}{b-1},
\quad
\tau\frac{1-b^{-T}}{b-1}
\right]
}
\]

当：

\[
b=2
\]

时：

\[
\frac{1-2^{-T}}{2-1}
=
1-2^{-T},
\]

因此精确退化回当前实现。

---

# 4. 建议新增统一 Phase 数学 helper

为了避免：

- `neurons.py`
- `calibration.py`
- `state_validation.py`
- regression tests

分别维护不同版本的 Phase 数学公式，建议新增一个非常轻量的共享 helper。

可以放在：

```text
snn2/neurons.py
```

或者新建：

```text
snn2/phase_math.py
```

如果新建模块，建议至少提供：

```python
def validate_phase_base(base: Any) -> float:
    ...
```

以及：

```python
def phase_amplitude_sum_factor(base: float, T: int) -> float:
    # sum_{k=1}^T base^{-k}
    # = (1 - base^{-T}) / (base - 1)
    ...
```

和/或：

```python
def phase_representable_bound(
    tau: torch.Tensor,
    *,
    T: int,
    base: float,
) -> torch.Tensor:
    ...
```

建议逻辑：

```python
factor = (1.0 - base ** (-T)) / (base - 1.0)
return tau * factor
```

所有 Stage-B Clip 生成与独立 validation 必须遵循同一个数学定义。

但要注意：

> validation 不能简单读取保存好的 PhaseBound 当作“真值”。

仍然应该根据：

```text
tau
T
base
```

独立重算，以便真正发现错误 artifact。

---

# 5. `snn2/config.py` 修改

## 5.1 删除当前固定 base=2 的限制

当前存在类似：

```python
if float(cfg["phase"].get("base", float("nan"))) != 2.0:
    raise ValueError("phase.base is fixed and must equal 2.0")
```

必须删除。

改为：

```python
try:
    phase_base = float(cfg["phase"]["base"])
except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(
        "phase.base must be a finite number greater than 1.0"
    ) from exc

if (
    not math.isfinite(phase_base)
    or phase_base <= 1.0
):
    raise ValueError(
        "phase.base must be a finite number greater than 1.0"
    )
```

如果做共享 `validate_phase_base()` helper，则这里直接调用 helper。

---

## 5.2 `calibration_trajectory_config()`

当前当：

```yaml
calibration:
  phase_previous_layers_snn: true
```

时 trajectory 只包含：

```text
phase_T
```

必须增加：

```text
phase_base
```

例如：

```python
if active["phase"]:
    result["phase_T"] = int(cfg["phase"]["T"])
    result["phase_base"] = float(cfg["phase"]["base"])
```

原因：

当 `phase_previous_layers_snn=true` 时，后续 block 的 calibration activation trajectory 已经经过前层 Temporal Phase Neuron。

Temporal Phase 的输出依赖：

```text
T
base
```

所以：

```text
base=1.5
```

和：

```text
base=2.0
```

的 Stage-A sequential calibration 不能视为同一个 trajectory。

---

# 6. `snn2/neurons.py` 修改

这是本轮最核心的修改之一。

---

## 6.1 `PhaseSurrogate.__init__()` 增加 `base`

当前接口类似：

```python
PhaseSurrogate(
    state,
    *,
    T,
    surrogate_slope=None,
)
```

修改为：

```python
PhaseSurrogate(
    state,
    *,
    T,
    base,
    surrogate_slope=None,
)
```

保存：

```python
self.T = int(T)
self.base = validate_phase_base(base)
```

---

## 6.2 `v0` 修改

当前：

```python
self.register_buffer(
    "v0",
    (0.5 * self.tau * 2.0 ** (-self.T)).float(),
)
```

改成：

```python
self.register_buffer(
    "v0",
    (
        0.5
        * self.tau
        * self.base ** (-self.T)
    ).float(),
)
```

语义必须始终是：

```text
half of the smallest Phase amplitude
```

---

## 6.3 Phase forward amplitude 修改

以下两个路径都必须改：

- `encode()`
- `_forward_ann_streaming()`

当前：

```python
amplitude = tau * 2.0 ** (-(timestep + 1))
```

改为：

```python
amplitude = (
    tau
    * self.base ** (-(timestep + 1))
)
```

ANN PhaseSurrogate 与 Temporal Phase 必须使用同一套 base。

---

## 6.4 sequential calibration provenance

当前 Phase state 在：

```text
previous_layers_snn=true
```

时只记录：

```text
calibration_phase_T
```

必须额外记录：

```text
calibration_phase_base
```

例如：

```python
state["calibration_phase_T"] = int(cfg["phase"]["T"])
state["calibration_phase_base"] = float(cfg["phase"]["base"])
```

---

## 6.5 `validate_phase_state_schema()`

当：

```python
state["previous_layers_snn"] is True
```

时除了验证：

```text
calibration_phase_T
```

还应验证：

```text
calibration_phase_base
```

要求：

```text
finite
> 1.0
```

注意：

普通 Stage-A common Phase state 仍然不能把 runtime base 写进 `phase_state.pt`。

也就是说：

```text
T
base
surrogate_slope
```

仍然不属于普通 Phase neuron parameter state。

只有当：

```text
previous_layers_snn=true
```

时，允许以：

```text
calibration_phase_T
calibration_phase_base
```

的形式保存 **trajectory provenance**。

不要重新把：

```text
base
```

字段直接塞回 legacy-style Phase state。

---

## 6.6 `PhaseSurrogate` sequential provenance mismatch 检查

当前：

```python
if state.get("previous_layers_snn") is True:
    recorded = state.get("calibration_phase_T")
    ...
```

改为同时比较：

```text
T
base
```

建议：

```python
recorded_T = state.get("calibration_phase_T")
recorded_base = state.get("calibration_phase_base")
```

要求：

```python
recorded_T == self.T
```

并且：

```python
math.isclose(
    float(recorded_base),
    self.base,
    rel_tol=0.0,
    abs_tol=1e-12,
)
```

否则：

```text
Phase sequential-calibration runtime provenance mismatch
```

---

# 7. `snn2/controller.py` 修改

## 7.1 `SiteController.__init__()` 增加参数

新增：

```python
phase_base: float | None = None
```

保存：

```python
self.phase_base = (
    None
    if phase_base is None
    else validate_phase_base(phase_base)
)
```

---

## 7.2 Phase ANN replacement 参数要求

当前 Phase ANN replacement 要求：

```text
phase_T
phase_surrogate_slope
```

修改为同时要求：

```text
phase_T
phase_base
phase_surrogate_slope
```

---

## 7.3 Phase deployment 参数要求

`set_deployment("phase")` 时要求：

```text
phase_T
phase_base
```

均存在且合法。

---

## 7.4 所有 `PhaseSurrogate(...)` 构造点传入 base

包括但不限于：

### 普通 Site Phase module

```python
PhaseSurrogate(
    state,
    T=int(self.phase_T),
    base=float(self.phase_base),
    surrogate_slope=...,
)
```

### Final RMSNorm ANN Phase

```python
PhaseSurrogate(
    state,
    T=int(self.phase_T),
    base=float(self.phase_base),
    surrogate_slope=self.phase_surrogate_slope,
)
```

### Final RMSNorm SNN Phase

```python
PhaseSurrogate(
    state,
    T=int(self.phase_T),
    base=float(self.phase_base),
)
```

---

## 7.5 block-wise sequential Phase calibration

`SiteController.begin_sequential_calibration("phase", ...)` 最终会通过 `_load()` 创建 Phase neuron。

因此必须确保：

```text
phase_base
```

已经保存在 Controller 中并传给 `PhaseSurrogate`。

这是：

```yaml
phase_previous_layers_snn: true
```

时 trajectory 正确性的必要条件。

---

# 8. `snn2/calibration.py` 修改

## 8.1 `stage_a_trajectory_metadata()`

当前 Phase trajectory details：

```text
phase_T
```

必须增加：

```text
phase_base
```

例如：

```python
if flags["phase"]:
    details["phase"]["phase_T"] = trajectory["phase_T"]
    details["phase"]["phase_base"] = trajectory["phase_base"]
```

---

## 8.2 `stage_a_parameter_independence`

当前当：

```text
phase_previous_layers_snn=false
```

时 Stage A 参数对：

```text
phase.T
```

独立。

现在应明确加入：

```text
phase.base
```

即：

当 `phase_previous_layers_snn=false` 时：

```text
phase.T
phase.base
```

都不参与 `tau` 的统计。

但即使数学上独立，本轮按用户要求仍允许 Calibration **结果目录** 中显式体现 configured base，见后面的 Artifact Layout 章节。

不要修改 `tau` 数值计算。

---

## 8.3 `_annotate_trajectory_state()`

当前：

```python
if enabled and neuron == "phase":
    state["calibration_phase_T"] = int(cfg["phase"]["T"])
```

改为：

```python
if enabled and neuron == "phase":
    state["calibration_phase_T"] = int(cfg["phase"]["T"])
    state["calibration_phase_base"] = float(cfg["phase"]["base"])
```

---

# 9. Stage-B Clip 数学修改

## 9.1 `build_clip_state()`

当前接口：

```python
build_clip_state(
    phase_state,
    gif_state,
    mtn_state,
    *,
    phase_T,
    mtn_T,
)
```

改成：

```python
build_clip_state(
    phase_state,
    gif_state,
    mtn_state,
    *,
    phase_T,
    phase_base,
    mtn_T,
)
```

---

## 9.2 PhaseBound

当前：

```python
phase_bound = (
    phase_state["tau"].double()
    * (1.0 - 2.0 ** (-phase_T))
)
```

必须改为一般形式：

```python
phase_factor = (
    1.0
    - phase_base ** (-phase_T)
) / (
    phase_base - 1.0
)

phase_bound = (
    phase_state["tau"].double()
    * phase_factor
)
```

更推荐调用统一 helper。

---

## 9.3 Clip state metadata

Stage-B `clip_state.pt` 的 common metadata 增加：

```text
phase_base
```

例如：

```python
"phase_T": phase_T,
"phase_base": phase_base,
"mtn_T": mtn_T,
```

---

# 10. `materialize_clip_profile()`

当前：

```python
phase_T, mtn_T = ...
```

修改为：

```python
phase_T = int(cfg["phase"]["T"])
phase_base = float(cfg["phase"]["base"])
mtn_T = int(cfg["mtn"]["T"])
```

调用：

```python
build_clip_state(
    phase_state,
    gif_state,
    mtn_state,
    phase_T=phase_T,
    phase_base=phase_base,
    mtn_T=mtn_T,
)
```

---

## 10.1 当前硬编码必须删除

当前 manifest 中存在：

```python
"phase_base": 2.0
```

必须改为：

```python
"phase_base": phase_base
```

---

## 10.2 每个 site summary

建议每个 Stage-B site summary 也增加：

```text
phase_base
```

和已有：

```text
phase_T
mtn_T
```

保持对称。

---

# 11. `snn2/state_validation.py` 修改

这一部分必须与 Stage-B 生成逻辑同步，但必须保持独立重算。

---

## 11.1 `_expected_clip_intervals()`

当前：

```python
def _expected_clip_intervals(
    phase,
    gif,
    mtn,
    *,
    phase_T,
    mtn_T,
):
```

改成：

```python
def _expected_clip_intervals(
    phase,
    gif,
    mtn,
    *,
    phase_T,
    phase_base,
    mtn_T,
):
```

当前：

```python
phase_bound = (
    phase["tau"].double()
    * (1.0 - 2.0 ** (-int(phase_T)))
)
```

改为：

\[
\tau\frac{1-b^{-T}}{b-1}.
\]

---

## 11.2 `_validate_clip_semantics()`

增加：

```text
phase_base
```

参数，并传给 `_expected_clip_intervals()`。

---

## 11.3 `validate_clip_profile()`

当前签名：

```python
validate_clip_profile(
    site_root,
    clip_root,
    *,
    phase_T,
    mtn_T,
    group_size,
    num_samples,
)
```

改为：

```python
validate_clip_profile(
    site_root,
    clip_root,
    *,
    phase_T,
    phase_base,
    mtn_T,
    group_size,
    num_samples,
)
```

expected metadata 中：

当前：

```python
"phase_base": 2.0
```

改为：

```python
"phase_base": float(phase_base)
```

---

## 11.4 Clip state runtime provenance

当前 site clip state 只检查：

```text
phase_T
mtn_T
```

需要增加：

```text
phase_base
```

例如：

```python
if (
    state.get("phase_T") != int(phase_T)
    or not math.isclose(
        float(state.get("phase_base")),
        float(phase_base),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    or state.get("mtn_T") != int(mtn_T)
):
    raise ValueError(...)
```

---

# 12. Artifact Path 总体设计

必须遵循一个原则：

> source ANN run identity 与 deployment override identity 必须分离。

也就是说：

```bash
--phase-base 1.5
```

用于 deployment sweep 时：

- 不允许改变 source ANN checkpoint
- 必须改变目标 Phase SNN 的 conversion/evaluation output path

这正是当前 `--phase-T` 的设计理念。

---

# 13. 建议新增统一 base path formatter

在：

```text
snn2/artifacts.py
```

增加类似：

```python
def format_phase_base(base: Any) -> str:
    value = validate_phase_base(base)
    return format(value, ".12g")
```

这样：

```text
2.0 -> "2"
1.5 -> "1.5"
1.25 -> "1.25"
```

避免 Python 浮点字符串出现不稳定长尾。

如果项目更希望保留：

```text
2.0
```

也可以用统一的稳定 formatter。

重点是：

> 全项目只能有一种 canonical path string 规则。

不要有的目录叫：

```text
phase_base_2
```

另一些叫：

```text
phase_base_2.0
```

---

# 14. ANN Training 路径修改

当前 Phase-aware 路径包含：

```text
phase_T_4_mtn_T_4_...
```

修改成：

```text
phase_base_<BASE>_T_4_mtn_T_4_...
```

例如：

```text
phase_base_1.5_T_4_mtn_T_4_surrogate_slope_1.0_...
```

---

## 14.1 `phase_training_dirname()`

当前：

```python
phase_training_dirname(
    *,
    phase_T,
    mtn_T,
    surrogate_slope,
    warmup_ratio,
)
```

增加：

```text
phase_base
```

输出：

```text
phase_base_<BASE>_T_<T>_mtn_T_<T>_...
```

---

# 15. GIF-aware ANN Training 路径也必须带 Phase base

不能只改 `phase_aware`。

原因：

GIF-aware ANN training 在：

```yaml
replacement:
  common_clip_enabled: true
```

时 common clipping 是：

\[
\mathrm{PhaseBound}
\cap
\mathrm{MTNBound}
\cap
\mathrm{GIFBound}.
\]

PhaseBound 依赖 `phase.base`。

因此不同 Phase base 的 GIF-aware ANN Training 可能得到不同训练轨迹。

修改：

```python
gif_training_dirname(...)
```

增加：

```text
phase_base
```

当前：

```text
phase_T_4_mtn_T_4_...
```

改为：

```text
phase_base_<BASE>_T_4_mtn_T_4_...
```

即使：

```text
common_clip_enabled=false
```

也建议保持相同路径 schema，避免同一 mode 内路径规则依配置开关发生变化。

---

# 16. Calibration 结果路径

用户明确要求：

> calibration dataset 上生成的结果保存路径也应体现 `base_*`。

因此 Calibration **artifact/result directories** 应加入：

```text
phase_base_<BASE>
```

但原始 calibration sample selection 本身不需要复制。

---

## 16.1 以下 raw data selection 路径保持不变

不要修改：

```text
.../_shared/.../data/calibration/num_samples_<N>/
```

以及：

```text
calibration_manifest.json
```

原因：

选择了哪些 calibration samples 与 Phase base 无关。

同一个 sample subset 应继续复用。

---

## 16.2 以下 Calibration 结果目录增加 base

至少覆盖：

```text
ann_training_calibration
post_finetuning/conversion_calibration
```

建议也对：

```text
vanilla_analysis_calibration
```

使用统一 schema，以减少 ArtifactLayout 特例。

例如：

```text
.../
ann_training_calibration/
prefix_enabled_ture/
calibration_group_size_128_num_samples_128/
phase_base_1.5/
<trajectory_signature>/
sites/
```

或者将 base 合并进 calibration variant dirname：

```text
calibration_group_size_128_num_samples_128_phase_base_1.5
```

二者任选其一，但必须全项目一致。

推荐单独目录：

```text
phase_base_<BASE>/
```

可读性更好。

---

# 17. Calibration trajectory path

当前：

```python
calibration_trajectory_dirname(cfg)
```

当：

```yaml
phase_previous_layers_snn: true
```

时只有：

```text
phase_T_4
```

必须变成：

```text
phase_base_<BASE>_T_4
```

例如：

```text
phase_previous_layers_snn_true_...
phase_base_1.5_T_4
...
```

这样 sequential Phase calibration 不会跨 base 误复用。

---

# 18. Stage-B Clip profile 路径

当前：

```text
clip_profiles/
phase_T_4_mtn_T_4/
```

必须改为：

```text
clip_profiles/
phase_base_<BASE>_T_4_mtn_T_4/
```

对应修改：

```python
clip_profile_dirname(...)
```

从：

```python
clip_profile_dirname(phase_T, mtn_T)
```

改为：

```python
clip_profile_dirname(
    phase_base,
    phase_T,
    mtn_T,
)
```

---

# 19. `materialize_clip_profile()` 的目录校验

当前：

```python
expected_name = (
    f"phase_T_{phase_T}_mtn_T_{mtn_T}"
)
```

改为：

```python
expected_name = (
    f"phase_base_{formatted_base}"
    f"_T_{phase_T}"
    f"_mtn_T_{mtn_T}"
)
```

必须使用和 ArtifactLayout 完全相同的 formatter。

不要手写第二套 base string formatter。

---

# 20. Phase SNN 路径

当前：

```python
phase_snn_dirname(phase_T)
```

输出：

```text
phase_T_4
```

必须改为：

```python
phase_snn_dirname(
    phase_base,
    phase_T,
)
```

输出：

```text
phase_base_<BASE>_T_<T>
```

例如：

```text
phase_base_1.5_T_4
```

这是 deployment sweep 能否安全工作的关键。

---

# 21. `--phase-base` CLI

文件：

```text
scripts/_common.py
```

---

## 21.1 parser 增加参数

当前在：

```python
if neuron or deployment_overrides:
```

下面已有：

```text
--phase-T
--mtn-T
--mtn-K
```

增加：

```python
result.add_argument(
    "--phase-base",
    type=float,
    default=None,
)
```

---

# 22. `apply_deployment_overrides()`

读取：

```python
phase_base = getattr(
    args,
    "phase_base",
    None,
)
```

---

## 22.1 合法性

如果给出：

```bash
--phase-base X
```

必须满足：

```text
finite
X > 1.0
```

---

## 22.2 neuron 限制

沿用 `--phase-T` 语义。

### `--neuron phase`

允许：

```text
--phase-T
--phase-base
```

禁止：

```text
--mtn-T
--mtn-K
```

### `--neuron mtn`

允许：

```text
--mtn-T
--mtn-K
```

禁止：

```text
--phase-T
--phase-base
```

### `--neuron gif`

禁止所有：

```text
--phase-T
--phase-base
--mtn-T
--mtn-K
```

### `--neuron ann`

同样禁止所有 deployment override。

---

## 22.3 修改 config

如果：

```python
phase_base is not None
```

执行：

```python
cfg["phase"]["base"] = float(phase_base)
```

---

# 23. Deployment override 最重要的路径语义

当前调用顺序是：

```python
cfg, layout = setup(args.config)
apply_deployment_overrides(args, cfg)
```

这一点应继续保留。

不要改成：

```python
apply override
# then ArtifactLayout(cfg)
```

原因：

用户要求：

> 在同一个 ANN checkpoint 上 sweep deployment Phase base。

因此：

```text
ArtifactLayout.root
```

必须首先根据 YAML 中的**训练时 base**固定 source ANN run。

然后：

```text
cfg["phase"]["base"]
```

才被 CLI override 为 deployment base。

例如 YAML：

```yaml
phase:
  T: 4
  base: 2.0
```

已有 ANN：

```text
.../
phase_base_2_T_4_mtn_T_4/
.../
ann/final/
```

运行：

```bash
python scripts/convert_snn.py \
  --config ... \
  --neuron phase \
  --phase-base 1.5
```

必须：

### source ANN

仍然读取：

```text
phase_base_2_T_4...
/ann/final/
```

### target SNN

写入：

```text
.../
snn/
.../
phase/
phase_base_1.5_T_4/
...
```

不能切换 source ANN checkpoint。

---

# 24. ArtifactLayout 的 mutable cfg 注意事项

当前：

```python
self._cfg = cfg
```

保存的是同一个 mutable dict。

因此：

```python
apply_deployment_overrides(args, cfg)
```

之后：

```python
layout._cfg
```

也会看到新的 deployment base。

这是当前 `--phase-T` 能动态影响：

```python
layout.snn_dir(...)
```

的基础。

但：

```python
self.root
```

已经在 `ArtifactLayout.__init__()` 时根据 source training config 固定，因此 source ANN path 不会被 deployment override 改掉。

必须增加专门测试锁定这一行为。

---

# 25. `scripts/calibrate_sites.py`

Stage-B calibration 不属于 deployment override。

因此：

```bash
python scripts/calibrate_sites.py \
  --config ... \
  --stage ann_training \
  --calibration-phase B
```

继续从 YAML：

```yaml
phase:
  base: ...
```

读取 training-time Phase base。

不建议给 `calibrate_sites.py` 增加 deployment-only `--phase-base` override。

如果要为 ANN Training sweep base：

> 修改 config / generated config，再重新运行 Stage-B。

Deployment-only CLI sweep 仅用于：

```text
convert_snn
evaluate_tldr
evaluate_lm_harness
以及其他 neuron-aware deployment/evaluation scripts
```

---

# 26. `scripts/calibrate_sites.py` Controller

当前构造：

```python
SiteController(
    mode="collect",
    site_root=site_root,
    phase_T=int(cfg["phase"]["T"]),
    ...
)
```

必须增加：

```python
phase_base=float(cfg["phase"]["base"])
```

否则：

```text
phase_previous_layers_snn=true
```

时 block-wise Phase calibration 仍然会丢失 base。

---

# 27. `snn2/training.py`

## 27.1 ANN Training Controller

当前：

```python
SiteController(
    ...
    phase_T=int(cfg["phase"]["T"]),
    ...
)
```

增加：

```python
phase_base=float(cfg["phase"]["base"])
```

---

## 27.2 `validate_clip_profile()`

所有调用增加：

```python
phase_base=float(cfg["phase"]["base"])
```

---

## 27.3 training provenance

当前保存：

```text
ann_training_phase_T
ann_training_mtn_T
```

增加：

```text
ann_training_phase_base
```

例如：

```python
"ann_training_phase_base":
    float(cfg["phase"]["base"]),
```

---

## 27.4 Final ANN Hugging Face config

保存 ANN checkpoint 前，建议加入：

```python
model.config.snn2_phase_T = int(
    cfg["phase"]["T"]
)

model.config.snn2_phase_base = float(
    cfg["phase"]["base"]
)
```

对于 aware modes 建议都记录。

原因：

`gif_aware` + common Clip 也可能依赖 Phase base。

也可增加：

```text
snn2_phase_surrogate_slope
```

如果项目已有类似 metadata 规范则遵循现有形式。

---

# 28. `snn2/evaluation.py`

## 28.1 Final ANN evaluation Controller

增加：

```python
phase_base=float(cfg["phase"]["base"])
```

---

## 28.2 SNN evaluation Controller

增加：

```python
phase_base=float(cfg["phase"]["base"])
```

注意：

当用户执行：

```bash
--phase-base 1.5
```

时这里拿到的必须是 override 后的 `1.5`。

---

## 28.3 `validate_clip_profile()`

ANN evaluation 的 training-time Clip validation 增加：

```python
phase_base=float(cfg["phase"]["base"])
```

对于 `neuron=ann` 不允许 CLI override，因此这里始终是 ANN training config 中的 base。

---

## 28.4 Evaluation metadata

建议 `evaluation_forward_metadata()` 至少增加：

```text
phase_base
```

对于：

### Phase-aware ANN

记录：

```text
phase_base = training base
```

### Phase SNN

记录：

```text
phase_base = deployment base
```

### 其他 neuron

可以：

```text
phase_base = null
```

或明确使用：

```text
deployment_phase_base
```

更推荐避免歧义：

```text
configured_phase_base
deployment_phase_base
```

其中 Phase SNN 的 deployment metadata 必须能直接看到 CLI override 后的值。

---

# 29. `snn2/conversion.py`

## 29.1 Conversion Controller

当前：

```python
SiteController(
    site_root=...,
    phase_T=int(cfg["phase"]["T"]),
    ...
)
```

增加：

```python
phase_base=float(cfg["phase"]["base"])
```

---

## 29.2 aware ANN training provenance validation

当前 `_validate_aware_training_provenance()` 检查：

```text
ann_training_phase_T
ann_training_mtn_T
```

增加：

```text
ann_training_phase_base
```

---

## 29.3 Conversion metadata

至少新增：

```text
source_ann_training_phase_base
deployment_phase_base
```

语义：

### `source_ann_training_phase_base`

ANN checkpoint 训练时使用的 base。

来自：

```text
training_result.json
```

例如：

```text
2.0
```

### `deployment_phase_base`

仅当：

```text
deployment_neuron == "phase"
```

时设置为当前：

```python
float(cfg["phase"]["base"])
```

包括 CLI override。

例如：

```text
1.5
```

因此一次合法 sweep 可以产生：

```json
{
  "source_ann_training_phase_base": 2.0,
  "deployment_phase_base": 1.5
}
```

这是合法且预期的。

---

## 29.4 `validate_conversion_metadata()`

expected metadata 同步加入：

```text
source_ann_training_phase_base
deployment_phase_base
```

不能要求二者相等。

因为本轮明确允许：

```text
training base != deployment base
```

---

# 30. `snn2/phase_conversion_regression.py`

所有直接构造：

```python
PhaseSurrogate(
    state,
    T=phase_T,
)
```

都增加：

```python
base=phase_base
```

---

## 30.1 `run_phase_neuron_micro_regression()`

建议签名改为：

```python
run_phase_neuron_micro_regression(
    site_root,
    num_layers,
    *,
    phase_T,
    phase_base,
    seed=42,
)
```

---

## 30.2 回归测试必须覆盖非 2 base

至少新增：

```text
base = 1.5
base = 3.0
```

验证：

```text
ANN PhaseSurrogate.forward(x)
```

与：

```text
Temporal Phase temporal(incoming).sum(0)
```

仍然一致。

---

# 31. sequential calibration provenance

当：

```yaml
calibration:
  phase_previous_layers_snn: true
```

必须完整绑定：

```text
phase_T
phase_base
```

---

## 31.1 State metadata

保存：

```text
calibration_phase_T
calibration_phase_base
```

---

## 31.2 Manifest trajectory

保存：

```json
{
  "phase": {
    "previous_layers_snn": true,
    "source": "sequential_temporal_phase",
    "phase_T": 4,
    "phase_base": 1.5
  }
}
```

---

## 31.3 `calibration_trajectory_config()`

signature 中包含：

```text
phase_base
```

---

## 31.4 `calibration_trajectory_dirname()`

path 中包含：

```text
phase_base_<BASE>_T_<T>
```

---

## 31.5 `_validate_state_runtime_provenance()`

当前检查：

```text
calibration_phase_T
```

增加：

```text
calibration_phase_base
```

与 cfg 比较。

如果不同：

```text
Phase sequential-calibration runtime provenance mismatch
```

---

# 32. Stage-A state 本身不要绑定普通 runtime base

必须区分两个概念：

### Calibration-derived parameter

```text
tau
```

### Runtime coding parameter

```text
T
base
surrogate_slope
```

普通：

```text
phase_state.pt
```

仍然只保存 calibration-derived parameter。

因此不要重新添加：

```python
state["base"] = ...
```

也不要：

```python
state["T"] = ...
```

这会破坏当前 A/B calibration architecture。

只有 sequential trajectory provenance 可以记录：

```text
calibration_phase_T
calibration_phase_base
```

---

# 33. `phase.base` 对不同实验 mode 的语义

## vanilla

ANN Training：

```text
不使用 Phase Neuron
```

因此 vanilla ANN checkpoint path 不需要按 `phase.base` 拆。

但之后：

```text
vanilla ANN -> Phase SNN
```

时 Phase SNN deployment path 必须带 deployment base。

---

## unaware

ANN Training：

```text
不使用 Phase Neuron
```

因此 unaware ANN checkpoint path 不需要按 `phase.base` 拆。

但：

```text
unaware ANN -> Phase SNN
```

时 Phase SNN path 必须带 deployment base。

---

## phase_aware

ANN Training 使用：

```text
PhaseSurrogate
```

因此 ANN Training path 必须带：

```text
phase_base
```

---

## gif_aware

GIF surrogate 本身不使用 Phase base。

但是如果：

```text
common_clip_enabled=true
```

Clip 上界包含 PhaseBound。

所以 GIF-aware ANN Training path 也必须带：

```text
phase_base
```

为保持 schema 稳定，即使 common Clip 关闭，也继续保留这个目录字段。

---

# 34. 保存路径示例

假设：

```yaml
phase:
  base: 1.5
  T: 4

mtn:
  T: 4
```

---

## Phase-aware ANN Training

应出现：

```text
.../
phase_aware/
.../
phase_base_1.5_T_4_mtn_T_4_surrogate_slope_1.0_...
/
seed42/
ann/
final/
```

---

## GIF-aware ANN Training

应出现：

```text
.../
gif_aware/
.../
phase_base_1.5_T_4_mtn_T_4_round_gradient_estimator_STE_...
/
seed42/
ann/
final/
```

---

## ANN Training Calibration

建议：

```text
.../
_shared/
seed42/
rotated_prefix/
ann_training_calibration/
prefix_enabled_ture/
calibration_group_size_128_num_samples_128/
phase_base_1.5/
...
```

---

## Stage-B Clip profile

```text
.../
clip_profiles/
phase_base_1.5_T_4_mtn_T_4/
```

---

## Phase SNN

```text
.../
snn/
use_post_finetuning_artifacts_false/
phase/
phase_base_1.5_T_4/
conversion/
...
```

---

# 35. CLI 使用示例

假设 ANN 是按：

```yaml
phase:
  base: 2.0
  T: 4
```

训练完成。

---

## 转换 base=1.5 的 Phase SNN

```bash
python scripts/convert_snn.py \
  --config "$CFG" \
  --neuron phase \
  --phase-base 1.5
```

---

## TL;DR 评估

```bash
accelerate launch --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$CFG" \
  --neuron phase \
  --phase-base 1.5
```

---

## Tulu3 / lm-eval

```bash
accelerate launch --num_processes 1 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG" \
  --neuron phase \
  --phase-base 1.5
```

如果多卡继续沿用当前项目原有调用方式，只增加：

```text
--phase-base 1.5
```

即可。

---

# 36. Deployment base sweep 示例

同一个 ANN checkpoint：

```bash
for BASE in 1.25 1.5 2.0 2.5 3.0; do
  python scripts/convert_snn.py \
    --config "$CFG" \
    --neuron phase \
    --phase-base "$BASE"

  accelerate launch --num_processes 1 \
    scripts/evaluate_tldr.py \
    --config "$CFG" \
    --neuron phase \
    --phase-base "$BASE"
done
```

每个 base 必须写入不同的：

```text
phase_base_<BASE>_T_<T>
```

目录。

---

# 37. 与 `--phase-T` 联合 sweep

必须允许：

```bash
--phase-T 6 \
--phase-base 1.5
```

对于：

```text
--neuron phase
```

这是合法组合。

结果目录：

```text
phase_base_1.5_T_6
```

source ANN checkpoint 仍然保持训练时：

```text
base
T
```

对应的原始 ANN run。

---

# 38. 一个重要边界：sequential Phase calibration

如果：

```yaml
phase_previous_layers_snn: false
```

那么 Phase Stage-A `tau` statistics 不依赖 deployment base。

所以同一个 Phase `tau` state 可以在数学上服务不同 deployment base。

---

如果：

```yaml
phase_previous_layers_snn: true
```

则后层 calibration activation trajectory 会依赖 Phase base。

因此：

```text
base=1.5
```

与：

```text
base=2.0
```

必须使用不同的 sequential calibration artifacts。

代码必须通过：

```text
trajectory dirname
manifest provenance
state provenance
```

三层机制阻止误用。

---

# 39. Deployment override 与 sequential calibration 的规则

如果 conversion 所选 Stage-A artifact：

```text
phase_previous_layers_snn=true
```

那么 deployment Phase base 必须与这个 sequential calibration artifact 的：

```text
calibration_phase_base
```

一致。

否则应明确报错，而不是继续运行。

也就是说：

## Common ANN trajectory calibration

```text
phase_previous_layers_snn=false
```

允许：

```text
training base 2.0
deployment base 1.5
```

直接 sweep。

## Phase-conditioned sequential calibration

```text
phase_previous_layers_snn=true
```

则：

```text
deployment base
```

必须有对应 base 的 calibration trajectory。

不得拿 base=2 的 sequential calibration state 部署 base=1.5。

---

# 40. Tests：配置验证

修改：

```text
tests/test_generated_configs.py
```

或者合适的新测试文件。

---

## 40.1 合法 base

至少：

```python
@pytest.mark.parametrize(
    "base",
    [1.01, 1.25, 1.5, 2.0, 3.0],
)
```

全部通过。

---

## 40.2 非法 base

至少：

```text
1.0
0.0
-1.0
inf
-inf
nan
"invalid"
True
```

必须拒绝。

---

# 41. Tests：Phase 数学

在：

```text
tests/test_neurons.py
```

增加非 2 base。

---

## 41.1 amplitude

例如：

```text
tau = 8
T = 3
base = 4
```

理论 amplitude：

\[
2,\ 0.5,\ 0.125.
\]

---

## 41.2 v0

同样参数：

\[
v_0
=
0.5\times 8\times 4^{-3}
=
0.0625.
\]

必须精确匹配。

---

## 41.3 base=2 回归

确认 base=2 时新代码与旧公式一致：

\[
0.5\tau 2^{-T}
\]

以及：

\[
\tau 2^{-(t+1)}.
\]

---

# 42. Tests：PhaseBound

至少测试：

```text
base=1.5
base=2.0
base=3.0
```

理论：

\[
B
=
\tau
\frac{1-b^{-T}}
{b-1}.
\]

检查：

```text
build_clip_state()
```

实际上下界与理论一致。

---

# 43. Tests：Stage-B validation

需要验证：

1. 正确 base 能通过。
2. manifest base 不同会失败。
3. `clip_state.pt` 中 base 不同会失败。
4. 目录名 base 不同会失败。
5. 独立 recomputation 使用一般等比数列公式。

---

# 44. Tests：路径

至少测试：

```text
phase_training_dirname
gif_training_dirname
clip_profile_dirname
phase_snn_dirname
calibration result path
```

例如：

```text
base=1.5
T=4
```

应包含：

```text
phase_base_1.5_T_4
```

---

# 45. Tests：deployment override 不改变 source ANN

这是本轮必须新增的关键回归测试。

流程：

1. cfg：
   ```yaml
   phase:
     base: 2.0
     T: 4
   ```
2. 创建：
   ```python
   layout = ArtifactLayout(cfg)
   ```
3. 保存：
   ```python
   original_ann = layout.ann_checkpoint_dir
   ```
4. 执行：
   ```python
   apply_deployment_overrides(
       --neuron phase
       --phase-base 1.5
   )
   ```
5. 验证：
   ```python
   layout.ann_checkpoint_dir == original_ann
   ```
6. 同时验证：
   ```python
   "phase_base_1.5_T_4"
   in str(layout.snn_dir("phase"))
   ```

这样锁定：

```text
same ANN checkpoint
+
different deployment SNN
```

这一核心需求。

---

# 46. Tests：training vs deployment provenance

构造：

```text
source_ann_training_phase_base = 2.0
deployment_phase_base = 1.5
```

Phase deployment conversion metadata 必须允许这种组合。

不要错误写成：

```python
source_base == deployment_base
```

的强制检查。

---

# 47. Tests：sequential Phase calibration

当：

```text
phase_previous_layers_snn=true
```

分别构造：

```text
calibration base = 2.0
runtime base = 2.0
```

应通过。

然后：

```text
calibration base = 2.0
runtime base = 1.5
```

必须失败。

---

# 48. Tests：ANN vs Temporal Phase

现有 Phase conversion micro regression 必须增加：

```text
base != 2
```

案例。

验证：

```python
module.forward(x)
```

与：

```python
module.temporal(
    sum-preserving temporal input
).sum(0)
```

仍满足现有误差阈值。

建议至少：

```text
base=1.5
base=3.0
```

---

# 49. 需要检查并更新的主要文件

至少检查：

```text
configs/experiment_matrix.yaml
snn2/config.py
snn2/artifacts.py
snn2/neurons.py
snn2/controller.py
snn2/calibration.py
snn2/blockwise_calibration.py
snn2/state_validation.py
snn2/training.py
snn2/conversion.py
snn2/evaluation.py
snn2/phase_conversion_regression.py
scripts/_common.py
scripts/calibrate_sites.py
scripts/convert_snn.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
tests/test_generated_configs.py
tests/test_neurons.py
tests/test_calibration_profiles.py
tests/test_blockwise_calibration_config.py
tests/test_blockwise_calibration_runtime.py
tests/test_conversion_metadata.py
tests/test_evaluation_paths.py
tests/test_phase_conversion_regression.py
tests/test_training.py
tests/test_post_finetuning_protocol.py
tests/test_verify_artifacts.py
```

不要求每个文件一定修改。

但必须检查所有相关调用点。

---

# 50. Config 文件

当前：

```yaml
phase:
  base: 2.0
```

默认值可以继续保持：

```yaml
base: 2.0
```

本轮的目的不是改变默认实验值，而是：

```text
允许合法 sweep
+
让代码真正使用它
```

因此 `configs/experiment_matrix.yaml` 中现有默认：

```text
2.0
```

可以继续保留。

---

# 51. Generated config

执行：

```bash
python scripts/materialize_configs.py
```

后生成配置必须继续包含：

```yaml
phase:
  base: 2.0
```

手工或 sweep script 修改为：

```yaml
phase:
  base: 1.5
```

后：

```python
load_config()
```

必须正常通过。

---

# 52. Verify Artifacts

`scripts/verify_artifacts.py` 必须检查所有受影响的：

```text
Stage-B Clip profile
conversion metadata
training provenance
trajectory provenance
```

不要让 verifier 仍然隐含：

```text
phase_base = 2.0
```

建议至少验证：

```text
clip_profile_manifest.phase_base
clip_state.phase_base
training_result.ann_training_phase_base
conversion_metadata.source_ann_training_phase_base
conversion_metadata.deployment_phase_base
sequential Phase calibration base provenance
```

---

# 53. 不应该修改的内容

本轮不要修改以下算法语义：

- `tau` calibration
- Phase EMA factor
- Phase tau clamp
- calibration group policy
- MTN 统计方式
- GIF quantization
- GIF salient selection
- Prefix
- Rotation
- TL;DR dataset protocol
- Tulu3 dataset protocol
- lm-eval task protocol
- common Clip 的 intersection 结构
- Phase surrogate gradient 公式
- `surrogate_slope`
- Temporal Phase 的 time-major execution 结构

本轮只把：

```text
Phase geometric base
```

从固定 2 推广为真正可配置参数。

---

# 54. Base=2 backward compatibility

在：

```text
base=2
```

时必须保证：

### amplitude

\[
\tau 2^{-(t+1)}
\]

不变。

### v0

\[
0.5\tau 2^{-T}
\]

不变。

### PhaseBound

\[
\tau
\frac{1-2^{-T}}
{2-1}
=
\tau(1-2^{-T})
\]

不变。

因此旧实验在数学意义上应完全复现。

保存路径会因为新增 base 字段而变化，这是本轮预期行为。

---

# 55. 全仓库硬编码扫描

完成修改后必须执行类似：

```bash
grep -R "phase.base is fixed" -n .
grep -R '"phase_base": 2.0' -n .
grep -R "1.0 - 2.0 \*\*" -n snn2 scripts tests
grep -R "2.0 \*\* (-(timestep + 1))" -n snn2 scripts tests
grep -R "PhaseSurrogate(" -n snn2 scripts tests
grep -R "SiteController(" -n snn2 scripts tests
grep -R "validate_clip_profile(" -n snn2 scripts tests
grep -R "phase_T_" -n snn2 scripts tests
grep -R "calibration_phase_T" -n snn2 scripts tests
```

逐项检查。

注意：

不是所有：

```text
2.0 ** ...
```

都一定与 Phase base 相关。

不要机械替换 MTN 或其他算法中的 base=2 逻辑。

只修改 Phase coding 的 base。

---

# 56. `2` 不能被误改的地方

例如 MTN 当前使用：

```python
torch.pow(2.0, levels)
```

这是 MTN threshold level 定义。

本轮：

```text
不要改
```

GIF bit decomposition 中的 2-step 逻辑也不要改。

因此不能做：

```text
全仓库 2.0 -> phase.base
```

这样的替换。

---

# 57. 推荐实现顺序

建议 Codex 按以下顺序实施：

1. 增加 Phase base validation/helper。
2. 修改 `PhaseSurrogate`。
3. 修改 `SiteController`。
4. 修改所有 Controller/PhaseSurrogate 调用点。
5. 修改 sequential calibration provenance。
6. 修改 Stage-B PhaseBound。
7. 修改 Stage-B independent validation。
8. 修改 ArtifactLayout 路径。
9. 修改 training provenance。
10. 修改 conversion provenance。
11. 修改 evaluation metadata。
12. 增加 CLI `--phase-base`。
13. 增加 deployment override source/target 路径测试。
14. 更新 verifier。
15. 更新全部 tests。
16. 执行全仓库 grep。
17. 运行完整测试。

---

# 58. 最终必须运行的测试

至少：

```bash
pytest -q
```

必须全部通过。

如果完整测试时间允许，还建议：

```bash
python scripts/materialize_configs.py
pytest -q
```

确认 generated configs 与测试同步。

---

# 59. 建议增加一个 Phase 数学单元测试

可以直接用简单人工数值。

例如：

```text
tau = 8
T = 3
base = 4
```

threshold/amplitude：

```text
t0 = 2
t1 = 0.5
t2 = 0.125
```

总范围：

```text
2.625
```

公式：

\[
8
\frac{1-4^{-3}}
{4-1}
=
2.625.
\]

`v0`：

```text
0.0625
```

这样可以同时锁定：

```text
amplitude
v0
PhaseBound
```

三种数学语义。

---

# 60. 最终验收标准

本轮修改只有在以下全部成立时才算完成。

## A. Config

```yaml
phase:
  base: 1.5
```

合法。

```yaml
base: 1.0
```

非法。

---

## B. ANN PhaseSurrogate

真实使用：

```text
base=1.5
```

而不是继续使用硬编码 2。

---

## C. Temporal Phase SNN

真实使用：

```text
base=1.5
```

并与 ANN PhaseSurrogate 共享同一编码数学。

---

## D. v0

满足：

\[
0.5\tau b^{-T}.
\]

---

## E. tau

现有 calibration 方法完全不变。

---

## F. Stage-B PhaseBound

满足：

\[
\tau
\frac{1-b^{-T}}
{b-1}.
\]

---

## G. Clip validation

独立使用相同数学重新计算，并检测错误 base。

---

## H. ANN path

Phase-aware/GIF-aware ANN run 显式包含：

```text
phase_base_<BASE>_T_<T>
```

---

## I. Calibration result path

显式包含：

```text
phase_base_<BASE>
```

但 raw calibration sample selection 继续共享。

---

## J. Phase SNN path

显式包含：

```text
phase_base_<BASE>_T_<T>
```

---

## K. Deployment CLI

支持：

```bash
--phase-base 1.5
```

---

## L. Same ANN checkpoint sweep

YAML training base：

```text
2.0
```

CLI deployment base：

```text
1.5
```

时：

```text
source ANN checkpoint 不变
target Phase SNN path 改变
```

---

## M. Metadata

能够同时记录：

```text
source_ann_training_phase_base = 2.0
deployment_phase_base = 1.5
```

并把这种情况视为合法。

---

## N. Sequential calibration

当：

```text
phase_previous_layers_snn=true
```

时：

```text
phase base mismatch
```

必须被拒绝。

---

## O. Regression

base=2 行为保持原数学结果。

base!=2 的：

```text
ANN Phase
Temporal Phase
Stage-B Clip
```

全部通过测试。

---

# 61. 最终实现原则总结

本轮最重要的不是简单地给路径加一个：

```text
base_*
```

而是彻底建立以下闭环：

```text
YAML phase.base
        ↓
config validation
        ↓
ANN PhaseSurrogate
        ↓
Temporal Phase SNN
        ↓
block-wise Phase calibration trajectory
        ↓
Stage-B geometric PhaseBound
        ↓
Clip profile
        ↓
ANN Training provenance
        ↓
Conversion provenance
        ↓
Evaluation runtime
        ↓
Phase SNN path
        ↓
--phase-base deployment sweep
```

同时保持：

```text
tau calibration unchanged
```

并严格区分：

```text
ANN training-time Phase base
```

与：

```text
SNN deployment-time Phase base
```

这两个概念。

这一区分是本轮修改正确与否的核心。
