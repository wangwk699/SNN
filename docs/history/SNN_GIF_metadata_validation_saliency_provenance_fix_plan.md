# SNN 项目补充修正方案：GIF-aware Evaluation Metadata、Site 2 All-Low Validation、Saliency Provenance

> 目标仓库：`https://github.com/wangwk699/SNN`
>
> 目标分支：`main`
>
> 本文档面向部署在服务器上的 Codex。**假设 Codex 没有本次对话上下文，必须仅凭本文档完成代码修改。**
>
> 本轮只修正三个已经确认的问题：
>
> 1. GIF-aware ANN evaluation metadata 描述已经过时；
> 2. Site 2 all-low GIF state validation 不够严格；
> 3. calibration manifest 没有显式记录 saliency rule / precision / mask-role provenance。
>
> **不要修改上一轮已经完成的核心算法、Site topology、quantization policy、calibration protocol 或 ANN/SNN forward 语义。**

---

# 1. 当前代码背景

上一轮已经完成以下核心修改，本轮必须保持不变：

- Site 1：GIF 使用 `q/k/v` 三个独立 role mask；
- Site 2：GIF 为 `all-low static 4-bit`；
- Site 3/4：在 `repeat_kv()` 之后 replacement，实际 runtime tensor 为 merged `[B,L,HD]`，但参数仍是 logical per-head + group-size；
- Site 5：GIF exact identity；
- Site 6：head merge 后 replacement，参数为普通 last-dim + group-size，不再 per-head；
- Site 7：GIF 使用 `gate/up` 两个独立 role mask；
- Site 8/9：GIF exact identity；
- Site 10：普通 single-mask salient GIF；
- salient Linear score：FP32；
- K/V matmul saliency：FP64；
- mask selection：SpikeLLM 官方 `<= threshold` tie 规则；
- GIF 仍是 static；
- `high_qmax=30` 保持；
- SNN 仍使用 `time_major_flattened_TB`。

本轮只做 metadata / validation / provenance 加强。

---

# 2. 修正 1：GIF-aware ANN evaluation metadata 描述已经过时

## 2.1 当前问题

文件：

```text
snn2/evaluation.py
```

函数：

```python
evaluation_forward_metadata(...)
```

当前 GIF-aware ANN 分支中，metadata 仍大致写成：

```python
implementation = "StaticGIF/SoftmaxIdentityGIF.forward"
```

这是旧描述。

当前真实 GIF runtime 已经包含四类 module：

```text
StaticGIF
AllLowStaticGIF
IdentityGIF
SoftmaxIdentityGIF
```

并且 Site 1 / Site 7 的 `StaticGIF` 是 role-aware：

```text
Site 1: q / k / v
Site 7: gate / up
```

因此当前 metadata 已不能准确描述真实 ANN execution graph。

---

## 2.2 修改要求

在：

```text
snn2/evaluation.py
```

更新 `evaluation_forward_metadata()` 的 GIF ANN metadata。

### 最低要求

将旧字符串：

```text
StaticGIF/SoftmaxIdentityGIF.forward
```

替换为能覆盖真实实现的明确描述，例如：

```text
StaticGIF/AllLowStaticGIF/IdentityGIF/SoftmaxIdentityGIF.forward
```

但仅修改字符串还不够，推荐同时写入结构化 GIF policy metadata。

---

## 2.3 推荐新增的 evaluation metadata 字段

对于：

```text
controller.mode == "gif"
```

在 `evaluation_forward_metadata()` 返回值中新增以下字段：

```python
"gif_salient_site_ids": [1, 3, 4, 6, 7, 10],
"gif_all_low_site_ids": [2],
"gif_identity_site_ids": [5, 8, 9],
"gif_multi_mask_roles": {
    "1": ["q", "k", "v"],
    "7": ["gate", "up"],
},
```

不要手写这些常量。

从：

```text
snn2/sites.py
```

导入：

```python
GIF_SALIENT_SITE_IDS
GIF_ALL_LOW_SITE_IDS
GIF_IDENTITY_SITE_IDS
GIF_MULTI_MASK_ROLES
```

然后动态生成：

```python
"gif_salient_site_ids": sorted(GIF_SALIENT_SITE_IDS),
"gif_all_low_site_ids": sorted(GIF_ALL_LOW_SITE_IDS),
"gif_identity_site_ids": sorted(GIF_IDENTITY_SITE_IDS),
"gif_multi_mask_roles": {
    str(site): list(roles)
    for site, roles in sorted(GIF_MULTI_MASK_ROLES.items())
},
```

### 非 GIF ANN / SNN evaluation

为了 schema 稳定，建议这些字段始终存在。

非 GIF ANN / 非 GIF SNN 时可仍记录全局 GIF topology policy，因为这些字段描述的是当前项目实现，而不是“当前此次 forward 是否启用”。

推荐始终返回上述字段。

这样所有 evaluation result metadata 都能明确记录当前代码所采用的 GIF topology。

---

## 2.4 `static_replacement_impl`

对于 GIF-aware ANN：

```python
static_replacement_impl
```

改成：

```text
StaticGIF/AllLowStaticGIF/IdentityGIF/SoftmaxIdentityGIF.forward
```

Phase-aware ANN 不改：

```text
PhaseSurrogate.forward
```

Vanilla / unaware identity 不改。

SNN deployment metadata 不要因此改变 execution semantics。

---

## 2.5 建议同步文档字符串

如果：

```text
README.md
代码结构总结.md
实验执行总结.md
```

中还存在：

```text
GIF ANN = StaticGIF + SoftmaxIdentityGIF
```

这种旧描述，则同步更新为：

```text
Site 1/7 role-aware StaticGIF
Site 2 AllLowStaticGIF
Site 5/8/9 IdentityGIF
Site 3/4/6/10 ordinary StaticGIF
```

只修正文档描述，不修改算法。

---

# 3. 修正 2：加强 Site 2 all-low GIF state validation

## 3.1 当前问题

Site 2 当前 state policy 是：

```text
GIF_ALL_LOW_POLICY = "all_low_static_qmax15"
```

当前 `build_site_states()` 生成的 Site 2 state 已经正确包含：

```text
low_qmin = 0
low_qmax = 15
temporal_steps = 2
per_step_qmin = 0
per_step_qmax = 15
quantization_path = low_only
quantization_applied = True
saliency_enabled = False
temporal_policy = low_at_t0_zero_at_t1
```

同时没有：

```text
mask_low
mask_low_by_role
high_scale
high_zero
high_qmax
```

但是目前 validation 主要只检查：

```text
policy
saliency_enabled
high/mask fields 不存在
```

对 `low_qmax / temporal_steps / per_step_qmax` 等关键 numerical policy 检查不够严格。

这意味着一个损坏或手工修改的 Site 2 artifact 可能没有第一时间 fail-fast。

---

# 4. `AllLowStaticGIF` 构造函数必须严格验证 policy

文件：

```text
snn2/neurons.py
```

类：

```python
class AllLowStaticGIF(...)
```

当前构造函数需要加强。

---

## 4.1 必须检查的字段

要求严格验证：

```python
expected = {
    "gif_policy": GIF_ALL_LOW_POLICY,
    "base_bits": GIF_BASE_BITS,
    "add_bits": GIF_ADD_BITS,
    "low_qmin": 0,
    "low_qmax": GIF_LOW_QMAX,
    "temporal_steps": GIF_LOCAL_STEPS,
    "per_step_qmin": 0,
    "per_step_qmax": GIF_STEP_QMAX,
    "quantization_path": "low_only",
    "quantization_applied": True,
    "saliency_enabled": False,
    "temporal_policy": "low_at_t0_zero_at_t1",
}
```

如果任一字段：

- 缺失；
- 数值不同；
- 类型不符合现有 state contract；

必须：

```python
raise ValueError(...)
```

错误信息需明确表明：

```text
Invalid all-low GIF state
```

或：

```text
Incompatible all-low GIF policy
```

---

## 4.2 必须禁止的字段

Site 2 all-low state 不允许包含任何 high branch / saliency mask 信息。

至少禁止：

```python
forbidden = {
    "mask_low",
    "mask_low_by_role",
    "mask_roles",
    "saliency_score",
    "saliency_score_by_role",
    "high_scale",
    "high_zero",
    "high_qmin",
    "high_qmax",
    "integer_decomposition",
}
```

说明：

Site 2 的 temporal policy 是：

```text
low quantized value at t0
0 at t1
```

它不执行 ordinary salient GIF 的 high integer decomposition。

因此不要让 `integer_decomposition` 进入 Site 2 state。

如果当前 `build_site_states()` 的 all-low state 没有这些字段，就继续保持。

---

## 4.3 qparam shape 仍严格验证

继续使用现有：

```python
_state_layout(...)
_require_parameter_shape(...)
```

严格验证：

```text
low_scale
low_zero
```

Site 2 仍然是：

```text
attention_head_grouped
```

参数 shape：

```text
[H, groups_per_head]
```

不要改 Site 2 grouping policy。

---

# 5. `state_validation.py` 对 Site 2 再做一层 bundle validation

文件：

```text
snn2/state_validation.py
```

函数：

```python
validate_site_state_bundle(...)
```

当前 Site 2 all-low branch 需要加强。

---

## 5.1 必须检查

对：

```python
site_index in GIF_ALL_LOW_SITE_IDS
```

至少要求：

```python
expected = {
    "gif_policy": GIF_ALL_LOW_POLICY,
    "base_bits": GIF_BASE_BITS,
    "add_bits": GIF_ADD_BITS,
    "low_qmin": 0,
    "low_qmax": GIF_LOW_QMAX,
    "temporal_steps": GIF_LOCAL_STEPS,
    "per_step_qmin": 0,
    "per_step_qmax": GIF_STEP_QMAX,
    "quantization_path": "low_only",
    "quantization_applied": True,
    "saliency_enabled": False,
    "temporal_policy": "low_at_t0_zero_at_t1",
}
```

必须完全匹配。

---

## 5.2 必须检查 forbidden fields

和 `AllLowStaticGIF` 保持一致。

至少：

```python
forbidden = {
    "mask_low",
    "mask_low_by_role",
    "mask_roles",
    "high_scale",
    "high_zero",
    "high_qmin",
    "high_qmax",
    "integer_decomposition",
}
```

存在任一字段都必须报错。

---

## 5.3 必须检查 parameter layout

Site 2 必须继续为：

```text
parameter_layout = attention_head_grouped
num_heads != None
channels_per_head != None
```

不能因为本轮 validation 加强而误改为 Site 6 的 last-dim layout。

如果现有 generic state validation 已经确保这一点，可以不重复代码，但 bundle validation 中至少应确保：

```text
site 2 != last_dim_grouped
```

避免错误 artifact 被接受。

---

# 6. 修正 3：calibration manifest 显式记录 saliency provenance

## 6.1 当前问题

当前：

```text
statistics.pt
gif_state.pt
```

已经保存：

```text
saliency_rule_by_role
saliency_accumulator_dtype_by_role
```

但：

```text
calibration_state_manifest.json
```

的 per-site summary 没有明确记录这些字段。

因此 artifact provenance 不完整。

本轮需要把以下内容写入 manifest 的每个 site entry：

```text
saliency_enabled
saliency_roles
saliency_rule_by_role
saliency_accumulator_dtype_by_role
gif_mask_policy
gif_mask_roles
```

---

# 7. 修改 `materialize_calibration_states()`

文件：

```text
snn2/calibration.py
```

函数：

```python
materialize_calibration_states(...)
```

当前会创建：

```python
summary = {
    ...
}
```

在这个 summary 中增加 saliency provenance。

---

## 7.1 所有 site 都要显式记录 `saliency_enabled`

从最终 `gif_state` 读取：

```python
"saliency_enabled": bool(gif_state.get("saliency_enabled", False)),
```

这样：

```text
Site 1  -> True
Site 2  -> False
Site 3  -> True
Site 4  -> True
Site 5  -> False
Site 6  -> True
Site 7  -> True
Site 8  -> False
Site 9  -> False
Site 10 -> True
```

---

## 7.2 `saliency_roles`

对于 salient sites：

### Site 1

```json
["q", "k", "v"]
```

### Site 7

```json
["gate", "up"]
```

### Site 3/4/6/10

```json
["default"]
```

### Site 2/5/8/9

```json
[]
```

推荐：

```python
"saliency_roles": (
    sorted(gif_state.get("saliency_rule_by_role", {}).keys())
    if gif_state.get("saliency_enabled", False)
    else []
),
```

不要直接依赖 statistics 原始字段，因为 manifest 应描述最终 materialized GIF state。

---

## 7.3 `saliency_rule_by_role`

对 salient sites，从 `gif_state` 复制：

```python
"saliency_rule_by_role": dict(
    gif_state.get("saliency_rule_by_role", {})
),
```

预期：

### Site 1 / Site 6 / Site 7 / Site 10

```text
spikellm_linear_fp32
```

### Site 3

```text
spikellm_qk_k_fp64
```

### Site 4

```text
spikellm_pv_v_fp64
```

不要把两者统一写成模糊的：

```text
spikellm_matmul_fp64
```

manifest 应保存真实 source string。

---

## 7.4 `saliency_accumulator_dtype_by_role`

复制：

```python
"saliency_accumulator_dtype_by_role": dict(
    gif_state.get("saliency_accumulator_dtype_by_role", {})
),
```

预期：

### Linear sites

```json
{"default":"float32"}
```

或：

```json
{"q":"float32","k":"float32","v":"float32"}
```

### Site 3/4

```json
{"default":"float64"}
```

### Identity / all-low

```json
{}
```

---

## 7.5 GIF mask provenance

增加：

```python
"gif_mask_policy": gif_state.get("mask_policy"),
"gif_mask_roles": list(gif_state.get("mask_roles", [])),
```

语义：

### Site 1

```text
gif_mask_policy = multi_role
gif_mask_roles = [q,k,v]
```

### Site 7

```text
gif_mask_policy = multi_role
gif_mask_roles = [gate,up]
```

### Site 3/4/6/10

```text
gif_mask_policy = single
gif_mask_roles = []
```

### Site 2/5/8/9

```text
gif_mask_policy = null
gif_mask_roles = []
```

若希望 schema 不出现 `null`，可以统一：

```text
none
```

但必须在 validation 中保持一致。

推荐直接保存：

```python
gif_state.get("mask_policy")
```

即可。

---

# 8. manifest 顶层增加 saliency policy metadata

除了 per-site summary，建议在：

```text
calibration_state_manifest.json
```

顶层新增全局 policy 字段。

例如：

```python
"gif_saliency_selection_policy": "spikellm_global_per_channel_threshold_leq",
"gif_saliency_tie_policy": "mask_low_equals_score_le_threshold",
"gif_linear_saliency_dtype": "float32",
"gif_matmul_saliency_dtype": "float64",
```

这些字符串应定义成常量，建议放在：

```text
snn2/temporal_ops.py
```

例如：

```python
GIF_SALIENCY_SELECTION_POLICY = "spikellm_global_per_channel_threshold_leq"
GIF_SALIENCY_TIE_POLICY = "mask_low_equals_score_le_threshold"
GIF_LINEAR_SALIENCY_DTYPE = "float32"
GIF_MATMUL_SALIENCY_DTYPE = "float64"
```

然后加入：

```python
temporal_policy_metadata()
```

或专门的：

```python
gif_saliency_policy_metadata()
```

推荐加入 `temporal_policy_metadata()`，因为这样：

- calibration manifest；
- conversion metadata；
- validation；

能自动获得一致 provenance。

但不要改变 temporal execution 本身。

---

# 9. `temporal_ops.py` metadata version 需要再次 bump

本轮修改了 artifact metadata schema。

因此应再次 bump：

```text
CALIBRATION_MANIFEST_FORMAT_VERSION
CONVERSION_METADATA_FORMAT_VERSION
```

因为 `temporal_policy_metadata()` 若加入新的 saliency policy字段，旧 conversion metadata 也会不完整。

推荐：

```text
CALIBRATION_MANIFEST_FORMAT_VERSION: 9 -> 10
CONVERSION_METADATA_FORMAT_VERSION: 10 -> 11
```

如果 Codex 检查后发现当前值已有后续变化，则：

> 在当前值基础上各加 1，不要强行写死 10/11。

`SITE_STATE_FORMAT_VERSION`：

- 如果只加强 validator，没有改变 Site 2 state schema，则可以不 bump；
- **但本方案要求 AllLowStaticGIF 现在强制要求现有 state 中已经存在的 numerical policy fields，因此 state 内容本身不变。**
- 所以 `SITE_STATE_FORMAT_VERSION` 无需 bump。

`STATISTICS_FORMAT_VERSION`：

- statistics 内容不变；
- 不需要 bump。

`SITE_TOPOLOGY_VERSION`：

- topology 不变；
- 不需要 bump。

`TEMPORAL_IMPLEMENTATION_VERSION`：

- temporal forward 不变；
- 不需要 bump。

---

# 10. `temporal_policy_metadata()` 建议新增字段

文件：

```text
snn2/temporal_ops.py
```

在：

```python
temporal_policy_metadata()
```

返回：

```python
"gif_saliency_selection_policy": GIF_SALIENCY_SELECTION_POLICY,
"gif_saliency_tie_policy": GIF_SALIENCY_TIE_POLICY,
"gif_linear_saliency_dtype": GIF_LINEAR_SALIENCY_DTYPE,
"gif_matmul_saliency_dtype": GIF_MATMUL_SALIENCY_DTYPE,
```

这样：

```text
calibration manifest
conversion metadata
evaluation deployment metadata
```

都可以继承/验证这些算法 provenance。

不要把：

```text
high_qmax=30
static GIF
```

改掉。

---

# 11. `state_validation.py` 加强 saliency provenance validation

除了 Site 2，本轮还要校验 manifest 中新增的 per-site saliency provenance。

文件：

```text
snn2/state_validation.py
```

函数：

```python
validate_site_state_bundle(...)
```

---

## 11.1 salient sites

对：

```text
1,3,4,6,7,10
```

要求 manifest site entry 中：

```text
saliency_enabled == true
```

且：

```text
saliency_roles
saliency_rule_by_role
saliency_accumulator_dtype_by_role
```

与实际 `gif_state.pt` 完全一致。

例如：

```python
if site_manifest.get("saliency_rule_by_role") != state.get("saliency_rule_by_role"):
    raise ValueError(...)
```

---

## 11.2 identity / all-low sites

对：

```text
2,5,8,9
```

必须：

```text
saliency_enabled == false
saliency_roles == []
saliency_rule_by_role == {}
saliency_accumulator_dtype_by_role == {}
```

不能允许旧 manifest 残留 saliency provenance。

---

## 11.3 multi-role sites

### Site 1

必须：

```text
gif_mask_policy == multi_role
gif_mask_roles == [q,k,v]
```

### Site 7

必须：

```text
gif_mask_policy == multi_role
gif_mask_roles == [gate,up]
```

且与：

```text
GIF_MULTI_MASK_ROLES
```

一致。

---

## 11.4 single-mask salient sites

对：

```text
3,4,6,10
```

要求：

```text
gif_mask_policy == single
gif_mask_roles == []
```

---

# 12. `conversion.py` provenance 同步

文件：

```text
snn2/conversion.py
```

当前：

```python
expected_grouping = {
    ...
}
```

以及 conversion metadata 中已经记录 temporal policy。

若 saliency policy 已进入：

```python
temporal_policy_metadata()
```

则优先依靠已有：

```python
validate_temporal_policy(...)
```

自动校验。

但需要检查：

```python
create_conversion(...)
validate_conversion_metadata(...)
```

是否把：

```text
temporal_policy_metadata()
```

完整展开到了 conversion metadata。

如果已经：

```python
**temporal_policy_metadata()
```

则不再手工重复。

如果不是，则补齐。

目标是：

```text
conversion_metadata.json
```

也明确保存：

```text
gif_saliency_selection_policy
gif_saliency_tie_policy
gif_linear_saliency_dtype
gif_matmul_saliency_dtype
```

---

# 13. `evaluation_forward_metadata()` 建议同步 saliency policy

除了 Site topology metadata，建议 evaluation result metadata 中加入：

```python
"gif_saliency_selection_policy": GIF_SALIENCY_SELECTION_POLICY,
"gif_saliency_tie_policy": GIF_SALIENCY_TIE_POLICY,
"gif_linear_saliency_dtype": GIF_LINEAR_SALIENCY_DTYPE,
"gif_matmul_saliency_dtype": GIF_MATMUL_SALIENCY_DTYPE,
```

这样最终 evaluation JSON 可独立证明该实验使用的是：

```text
SpikeLLM global per-channel threshold
<= tie rule
FP32 linear
FP64 matmul
```

---

# 14. 必须新增 / 更新的测试

本轮必须补测试。

---

## 14.1 `tests/test_neurons.py`

增加 Site 2 strict validation。

### Case A：合法 state

应成功：

```text
low_qmax=15
T=2
per_step_qmax=15
low_only
```

### Case B：错误 low_qmax

例如：

```python
state["low_qmax"] = 14
```

必须 fail。

### Case C：错误 temporal_steps

```python
state["temporal_steps"] = 4
```

必须 fail。

### Case D：错误 per_step_qmax

```python
state["per_step_qmax"] = 14
```

必须 fail。

### Case E：错误 policy

```python
state["quantization_path"] = "low_high"
```

必须 fail。

### Case F：伪造 high branch

加入：

```python
state["high_scale"] = ...
```

必须 fail。

### Case G：伪造 mask

加入：

```python
state["mask_low"] = ...
```

必须 fail。

---

## 14.2 `tests/test_controller_state_loading.py`

增加：

```text
Site 2 corrupted all-low state cannot be loaded in gif mode/deploy_gif
```

不要只单测 constructor，还应验证 controller path fail-fast。

---

## 14.3 `tests/test_calibration_profiles.py`

检查生成的 Site 2 state：

```text
gif_policy == all_low_static_qmax15
low_qmax == 15
temporal_steps == 2
per_step_qmax == 15
saliency_enabled == false
```

并检查不存在：

```text
high_qmax
high_scale
high_zero
mask_low
mask_low_by_role
```

---

## 14.4 calibration manifest provenance test

在：

```text
tests/test_calibration_profiles.py
```

或：

```text
tests/test_calibration_topology.py
```

新增一个 test，materialize 完整 10-site bundle 后读取：

```text
calibration_state_manifest.json
```

至少验证：

### Site 1

```json
{
  "saliency_enabled": true,
  "saliency_roles": ["k","q","v"] 或稳定顺序 ["q","k","v"],
  "saliency_accumulator_dtype_by_role": {
    "q":"float32",
    "k":"float32",
    "v":"float32"
  },
  "gif_mask_policy":"multi_role",
  "gif_mask_roles":["q","k","v"]
}
```

顺序应固定。

推荐按：

```text
GIF_MULTI_MASK_ROLES
```

顺序，不要用纯字典无序语义。

### Site 3

```text
saliency_enabled = true
saliency_roles = ["default"]
dtype = float64
gif_mask_policy = single
```

### Site 6

```text
saliency_enabled = true
dtype = float32
```

### Site 7

```text
gate/up multi-role
```

### Site 2/5/8/9

```text
saliency_enabled = false
saliency_roles = []
rule = {}
dtype = {}
```

---

## 14.5 manifest validation mismatch tests

手工修改 manifest：

```python
site_03["saliency_accumulator_dtype_by_role"]["default"] = "float32"
```

再调用：

```python
validate_site_state_bundle(...)
```

必须 fail。

类似增加：

```text
Site 1 mask role 少一个 v
Site 2 saliency_enabled 改 true
Site 7 mask_policy 改 single
```

至少覆盖其中 2–3 种。

---

## 14.6 `tests/test_evaluation_paths.py`

或者其他 evaluation metadata test：

对 GIF-aware ANN：

```python
metadata = evaluation_forward_metadata(...)
```

验证：

```text
static_replacement_impl
```

包含：

```text
StaticGIF
AllLowStaticGIF
IdentityGIF
SoftmaxIdentityGIF
```

同时验证：

```text
gif_salient_site_ids == [1,3,4,6,7,10]
gif_all_low_site_ids == [2]
gif_identity_site_ids == [5,8,9]
gif_multi_mask_roles == {
    "1":["q","k","v"],
    "7":["gate","up"]
}
```

以及 saliency dtype/policy metadata。

---

# 15. 不要修改的内容

本轮严禁修改：

- Site 1/7 multi-mask 算法；
- Site 2 all-low numerical behavior；
- Site 3/4 topology；
- Site 6 topology；
- Site 5/8/9 identity forward；
- GIF qparams calibration；
- Phase/MTN calibration；
- common Clip numerical algorithm；
- Prefix protocol；
- rotation protocol；
- training pipeline；
- ANN full-parameter fine-tuning；
- SNN temporal operator；
- `high_qmax=30`；
- `GIF_LOCAL_STEPS=2`；
- `group_size`；
- 10-site numbering。

这次是：

```text
metadata correctness
+
artifact validation strictness
+
provenance completeness
```

不是算法重构。

---

# 16. 建议修改文件清单

最低涉及：

```text
snn2/evaluation.py
snn2/neurons.py
snn2/state_validation.py
snn2/calibration.py
snn2/temporal_ops.py
snn2/conversion.py
tests/test_neurons.py
tests/test_controller_state_loading.py
tests/test_calibration_profiles.py
tests/test_calibration_topology.py
tests/test_evaluation_paths.py
```

如果文档存在旧描述，再同步：

```text
README.md
代码结构总结.md
实验执行总结.md
```

---

# 17. 最终 acceptance criteria

全部满足后才算完成。

## Evaluation metadata

- [ ] GIF-aware ANN 不再只描述为 `StaticGIF/SoftmaxIdentityGIF.forward`。
- [ ] metadata 显式包含四类 GIF runtime module。
- [ ] metadata 记录 salient/all-low/identity site IDs。
- [ ] metadata 记录 Site 1/7 multi-mask roles。
- [ ] metadata 记录 saliency threshold/tie/dtype policy。

## Site 2 validation

- [ ] `low_qmax` 必须严格等于 15。
- [ ] `temporal_steps` 必须严格等于 2。
- [ ] `per_step_qmax` 必须严格等于 15。
- [ ] `low_qmin/per_step_qmin` 必须等于 0。
- [ ] `base_bits=4/add_bits=1` 必须验证。
- [ ] `quantization_path=low_only` 必须验证。
- [ ] `saliency_enabled=false` 必须验证。
- [ ] `quantization_applied=true` 必须验证。
- [ ] high branch字段全部禁止。
- [ ] saliency mask字段全部禁止。
- [ ] corrupted Site 2 artifact必须 fail-fast。

## Calibration manifest provenance

- [ ] 每个 site 显式记录 `saliency_enabled`。
- [ ] 每个 site 显式记录 `saliency_roles`。
- [ ] 每个 salient site记录 `saliency_rule_by_role`。
- [ ] 每个 salient site记录 `saliency_accumulator_dtype_by_role`。
- [ ] manifest 记录 `gif_mask_policy`。
- [ ] manifest 记录 `gif_mask_roles`。
- [ ] 顶层记录 SpikeLLM threshold/tie/FP32/FP64 policy。
- [ ] state validation 会比较 manifest provenance 和 `gif_state.pt`。
- [ ] provenance不一致时 fail-fast。

## Versioning

- [ ] calibration manifest format version 已 bump。
- [ ] conversion metadata format version 已 bump。
- [ ] 不无意义 bump topology/statistics/state/temporal versions。

## Testing

- [ ] Site 2 corruption tests通过。
- [ ] manifest saliency provenance tests通过。
- [ ] evaluation metadata tests通过。
- [ ] 全量：

```bash
pytest -q
```

通过。

---

# 18. 完成后 Codex 应输出的简要总结

修改完成后，Codex 应报告：

```text
1. 修改了哪些文件；
2. evaluation metadata 新增了哪些 GIF topology/provenance 字段；
3. Site 2 all-low 新增了哪些 strict validation；
4. calibration manifest 新增了哪些 saliency provenance；
5. bump 了哪些 format version；
6. pytest 结果。
```

不要在没有实际运行测试时声称 `pytest` 已通过。
