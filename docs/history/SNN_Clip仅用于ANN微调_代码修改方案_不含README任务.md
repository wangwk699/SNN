# SNN 项目修改方案：Clip 仅用于 ANN aware 微调，并重构代码文档规范

## 0. 本文用途

本文面向部署在服务器上的 Codex。执行者应当假设自己**没有此前任何对话上下文**，仅依据本文和仓库当前 `main` 分支代码完成修改。

目标仓库：

```text
/home/wangwenkang/SNN
```

GitHub：

```text
https://github.com/wangwk699/SNN
```

本次修改包含三件事：

1. 修正 Clip 的语义：**Clip 只允许用于 ANN 的 `phase_aware` / `gif_aware` 微调；Post-finetuning SNN conversion 与 SNN deployment 完全不得使用 Clip。**
2. 重构根目录 `代码结构总结.md`：删除除“`2. 目录结构`”之外的说明性内容，以后该文件只维护仓库目录结构，并在每个文件后用一句话说明其功能。
3. 在仓库根目录创建标准 Codex 规则文件 `AGENTS.md`，本次只写两条规则：Clip 使用规则、`代码结构总结.md` 内容规则。

**README 不属于本次 Codex 修改任务。不要新建、改写或删除 `README.md`。**

---

# 1. 必须满足的最终语义

## 1.1 Clip 的唯一允许用途

Clip 只属于 **ANN aware fine-tuning**。

允许的两条路径：

```text
phase_aware ANN training:
activation
    ↓
Phase surrogate replacement
    ↓
Clipper.forward()
    ↓
后续 ANN 运算
```

```text
gif_aware ANN training:
activation
    ↓
StaticGIF.forward()
    ↓
Clipper.forward()
    ↓
后续 ANN 运算
```

也就是说：

```python
mode == "phase"
    -> PhaseSurrogate
    -> Clipper

mode == "gif"
    -> StaticGIF
    -> Clipper
```

ANN-training calibration 仍然必须生成：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

其中当前训练真正使用的是：

```text
phase_aware -> phase_state.pt + clip_state.pt
gif_aware   -> gif_state.pt   + clip_state.pt
```

不要删除 ANN-training `clip_state.pt`，不要改变 `phase_aware` / `gif_aware` 当前的静态 Clip 行为。

---

## 1.2 SNN conversion / deployment 中禁止 Clip

完成 ANN fine-tuning 后：

```text
final ANN checkpoint
    ↓
Post-finetuning Prefix
    ↓
Post-finetuning conversion calibration
    ↓
Phase / GIF / MTN state
    ↓
SNN conversion
    ↓
SNN deployment/evaluation
```

这条链路中**不允许出现任何 Clip**。

Post-finetuning conversion calibration 每个 site 最终只能生成：

```text
statistics.pt
statistics_summary.json

phase_state.pt
gif_state.pt
mtn_state.pt

calibration_summary.json
```

不得生成：

```text
clip_state.pt
```

SNN deployment 的 site 行为必须是：

```text
deploy_phase:
temporal input
    ↓
PhaseSurrogate.temporal()
    ↓
直接输出
```

```text
deploy_gif:
temporal input
    ↓
StaticGIF.temporal()
    ↓
直接输出
```

```text
deploy_mtn:
temporal input
    ↓
MultiThresholdNeuron.temporal()
    ↓
直接输出
```

不得再执行：

```text
Clipper.temporal()
temporal_clip()
cumulative -> Clip -> difference
```

特别注意：

**GIF 自己内部 `_quantize()` 的 `qmin/qmax clamp` 是 GIF 算法本身的一部分，不属于本次需要删除的 common Clip。**

例如：

```python
q = (...).clamp(qmin, qmax)
```

必须保留。

---

# 2. 当前代码中的错误位置

当前 `main` 中有四层错误，需要一起修。

## 2.1 Post-finetuning calibration 被标记成 with common Clip

当前 `snn2/calibration.py` 中：

```python
state_profile = {
    "ann_training": "ann_training_with_common_clip",
    "vanilla_analysis": "analysis_statistics_only",
    "post_finetuning": "snn_conversion_with_common_clip",
}[stage]
```

并且：

```python
"common_clip_required": stage in {"ann_training", "post_finetuning"}
```

这会把 Post-finetuning calibration 错误定义为需要 common Clip。

---

## 2.2 Post-finetuning calibration 会实际生成 `clip_state.pt`

当前 `collect_site_statistics()` 中：

```python
eligible_ann = purpose == "ann_training_calibration"
eligible_conversion = purpose == "post_finetuning_conversion_calibration"
```

然后 materialization 使用：

```python
include_clip = eligible_ann or eligible_conversion
```

所以 Step 5 与 Step 8 都生成 `clip_state.pt`。

这是本次必须修正的核心之一。

---

## 2.3 conversion 强制要求 `clip_state.pt`

当前 `snn2/conversion.py` 中：

```python
validate_site_state_bundle(
    ...,
    require_clip=True,
)
```

并且每个 site 显式要求：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

manifest 还要求：

```text
state_profile = snn_conversion_with_common_clip
common_clip_required = true
```

conversion metadata 中还有：

```text
common_clip_applied = true
```

这些都必须修改。

---

## 2.4 SNN deployment forward 会实际执行 temporal Clip

当前 `snn2/controller.py`：

```python
elif self.mode.startswith("deploy_"):
    ...
    required = (neuron, "clip")
```

并且：

```python
output = modules[neuron].temporal(temporal)
output = modules["clip"].temporal(output)
```

所以当前不是“只多保存一个无用的 `clip_state.pt`”，而是 SNN 实际前向确实执行了 Clip。

本次必须彻底删除这一行为。

---

# 3. 修改文件总览

至少检查并修改以下文件：

```text
configs/experiment_matrix.yaml

snn2/calibration.py
snn2/controller.py
snn2/conversion.py
snn2/neurons.py
snn2/temporal_ops.py
snn2/config.py
snn2/state_validation.py

scripts/verify_artifacts.py

tests/test_calibration_profiles.py
tests/test_calibration_topology.py
tests/test_controller_state_loading.py
tests/test_conversion_metadata.py
tests/test_generated_configs.py
tests/test_neurons.py
tests/test_temporal_ops.py

实验执行总结.md
代码结构总结.md

AGENTS.md            # 新建
```

还必须使用全文搜索检查是否有其它当前代码或当前文档依赖旧 Clip deployment 语义。

历史方案目录：

```text
docs/history/
```

属于历史记录，不要求为了本次语义重写旧方案。不要为了“全文零命中”而篡改历史文档。

---

# 4. `snn2/calibration.py` 修改方案

## 4.1 保留 ANN-training common Clip

下面语义必须保持：

```text
ann_training_calibration
    state_profile = ann_training_with_common_clip
    common_clip_required = true
    include_clip = true
```

`build_site_states(..., include_clip=True)` 的 common Clip 构造公式保持现状。

也就是说以下逻辑不要删除：

```python
phase_bound = ...
mtn_bound = ...
gif_lower = ...
gif_upper = ...
lower = ...
upper = ...
clip_state = ...
```

因为它仍然服务 ANN `phase_aware` / `gif_aware` 微调。

---

## 4.2 修改 Post-finetuning calibration profile

在 `calibration_provenance()` 中将：

```text
post_finetuning:
    state_profile = snn_conversion_with_common_clip
    common_clip_required = true
```

改成：

```text
post_finetuning:
    state_profile = snn_conversion_without_clip
    common_clip_required = false
```

建议最终映射：

```python
state_profile = {
    "ann_training": "ann_training_with_common_clip",
    "vanilla_analysis": "analysis_statistics_only",
    "post_finetuning": "snn_conversion_without_clip",
}[stage]
```

以及：

```python
"common_clip_required": stage == "ann_training",
```

不要再把 `post_finetuning` 包含进去。

---

## 4.3 修改 `collect_site_statistics()`

保留：

```python
eligible_ann = purpose == "ann_training_calibration"
eligible_conversion = purpose == "post_finetuning_conversion_calibration"
```

但修改 profile：

```python
state_profile = {
    "ann_training_calibration": "ann_training_with_common_clip",
    "vanilla_analysis_calibration": "analysis_statistics_only",
    "post_finetuning_conversion_calibration": "snn_conversion_without_clip",
}[purpose]
```

修改：

```python
"common_clip_required": eligible_ann
```

不要再写：

```python
eligible_ann or eligible_conversion
```

最重要的是：

```python
materialize_calibration_states(
    ...
    include_clip=eligible_ann,
    ...
)
```

而不是：

```python
include_clip=eligible_ann or eligible_conversion
```

因此：

```text
ANN-training calibration           -> include_clip=True
Post-finetuning conversion calib   -> include_clip=False
Vanilla analysis                   -> 不 materialize states
```

---

## 4.4 利用现有 stale-file 清理逻辑

当前 `materialize_calibration_states()` 已有：

```python
if not include_clip:
    (directory / "clip_state.pt").unlink(missing_ok=True)
```

保留这一逻辑。

这非常重要，因为用户可能在旧 Step 8 目录上直接重新执行 calibration。

新代码执行：

```text
--stage post_finetuning
```

时必须自动清除旧的：

```text
clip_state.pt
```

因此重新运行 Step 8 后目录会自然变成 clip-free。

---

## 4.5 calibration summary

当前：

```python
"clip_state_present": include_clip,
```

可以保留。

最终应满足：

ANN-training：

```json
{
  "clip_state_present": true,
  "clip_valid": true
}
```

Post-finetuning：

```json
{
  "clip_state_present": false
}
```

Post-finetuning summary 不应包含需要成功验证 temporal Clip 的字段。

---

# 5. `snn2/controller.py` 修改方案

这是决定 SNN runtime 是否真正执行 Clip 的核心文件。

## 5.1 ANN training 路径保持不变

保留：

```python
if self.mode == "phase":
    required = ("phase", "clip")
elif self.mode == "gif":
    required = ("gif", "clip")
```

并保留：

```python
if self.mode == "phase":
    return modules["clip"](modules["phase"](x))

if self.mode == "gif":
    return modules["clip"](modules["gif"](x))
```

这正是本次要求的 ANN Clip 行为。

---

## 5.2 deployment 只加载 neuron state

将：

```python
elif self.mode.startswith("deploy_"):
    ...
    required = (neuron, "clip")
```

改成：

```python
elif self.mode.startswith("deploy_"):
    ...
    required = (neuron,)
```

因此：

```text
deploy_phase -> 只加载 phase_state.pt
deploy_gif   -> 只加载 gif_state.pt
deploy_mtn   -> 只加载 mtn_state.pt
```

不得加载 `clip_state.pt`。

---

## 5.3 `set_deployment()` 不要求 Clip

当前：

```python
validation = validate_site_state_bundle(
    self.site_root,
    require_clip=True,
)
```

改成：

```python
validation = validate_site_state_bundle(
    self.site_root,
    require_clip=False,
)
```

如果实现时决定扩展 validator 为更清晰的 policy 枚举，也可以，但最终语义必须是：

```text
deployment bundle 不需要 clip_state.pt
```

---

## 5.4 删除 deployment 的 temporal Clip

当前：

```python
output = modules[neuron].temporal(temporal)

...

output = modules["clip"].temporal(output)

...

return from_temporal(output)
```

改成：

```python
output = modules[neuron].temporal(temporal)

if output.shape != temporal.shape:
    ...

if output.dtype != x.dtype or output.device != x.device:
    ...

return from_temporal(output)
```

也就是说 SNN site runtime 不再经过任何 Clip。

---

# 6. `snn2/neurons.py` 修改方案

## 6.1 `Clipper` 保留，但限定为 ANN static Clip

保留：

```python
class Clipper(nn.Module):
```

保留：

```python
def forward(self, x):
    ...
    return hard_clip(x, lower, upper)
```

因为 ANN `phase_aware` / `gif_aware` 仍然需要它。

---

## 6.2 删除 `Clipper.temporal()`

删除：

```python
def temporal(self, x):
    ...
    return temporal_clip(x, lower, upper)
```

这样从类接口层面阻止未来误把 Clip 再接回 SNN deployment。

---

## 6.3 删除 `temporal_clip` import

从：

```python
from .temporal_ops import (..., temporal_clip)
```

删除 `temporal_clip`。

---

## 6.4 不修改 GIF 自身 clamp

必须保留：

```python
q = (...).clamp(qmin, qmax)
```

以及：

```python
integer_chunks()
```

GIF `qmax=30`、两步各 `0..15` 的实现与本次 common Clip 删除无关。

---

# 7. `snn2/temporal_ops.py` 修改方案

## 7.1 删除 common Clip temporal policy

删除常量：

```python
COMMON_CLIP_TEMPORAL_POLICY = "cumulative_then_difference"
```

从：

```python
temporal_policy_metadata()
```

中删除：

```python
"common_clip_temporal_policy": COMMON_CLIP_TEMPORAL_POLICY,
```

SNN deployment policy 中不应再存在 Clip。

---

## 7.2 删除 `temporal_clip()`

整个删除：

```python
def temporal_clip(...):
    ...
```

因为新设计中没有任何合法调用方。

---

## 7.3 版本策略

建议：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 2    # 保持
SITE_STATE_FORMAT_VERSION = 2          # 保持
CALIBRATION_MANIFEST_FORMAT_VERSION = 3 # 保持
CONVERSION_METADATA_FORMAT_VERSION = 4  # 3 -> 4
```

原因：

- Phase/GIF/MTN temporal arithmetic 本身没有重写，因此 temporal implementation 不需要升版。
- neuron state 格式没有改变。
- ANN-training calibration 工件仍然有效，不应无必要地通过全局 calibration manifest 升版使已有 ANN-training calibration 失效。
- conversion descriptor 的语义发生了明确改变：旧 v3 表示带 common Clip，新 v4 表示 SNN 无 Clip，因此 conversion metadata 必须升版。

如果实现过程中发现 calibration manifest 的 schema 校验无法可靠区分新旧 Post-finetuning profile，也可以将 calibration manifest 升到 v4；但若这样做，必须明确处理旧 ANN-training manifest 的兼容性。优先采用“manifest v3 + profile 强校验 + 禁止 post-finetuning clip 文件”的方案。

---

# 8. `snn2/config.py` 与 `configs/experiment_matrix.yaml`

## 8.1 删除 deployment Clip 配置

当前 `deployment` 中：

```yaml
common_clip_temporal_policy: cumulative_then_difference
```

删除该字段。

最终 deployment 应只有类似：

```yaml
deployment:
  temporal_implementation: sparse_llm_temporal_v2
  temporal_layout: time_major_flattened_TB
  linear_bias_policy: first_timestep_once
  prefix_temporal_policy: uniform_kv_divide_by_T
```

对 `configs/experiment_matrix.yaml` 中所有 experiment block 都执行该修改。

---

## 8.2 修改 config validation

`config.py` 不再 import：

```python
COMMON_CLIP_TEMPORAL_POLICY
```

`expected_deployment` 中删除：

```python
"common_clip_temporal_policy": COMMON_CLIP_TEMPORAL_POLICY
```

建议同时让 validator 明确拒绝旧配置：

```text
deployment 中如果仍出现 common_clip_temporal_policy
-> ValueError
-> 提示重新运行 materialize_configs.py
```

最好要求当前 deployment key set 与新协议一致，避免旧 generated config 悄悄继续使用过期字段。

---

## 8.3 不删除 ANN replacement clip 配置

例如：

```yaml
replacement:
  train_mode: phase
  outer_clip_backward: hard_clip
  clip_rule: phase_mtn_gif_low_high_common_intersection
```

这些属于 ANN aware training，不是 SNN deployment Clip。

除非代码检查表明某字段完全未使用，否则本次不要因为名字中含 `clip` 就机械删除。

---

# 9. `snn2/state_validation.py` 修改方案

该模块需要继续同时支持：

```text
ANN-training bundle -> 有 Clip
Post-finetuning SNN bundle -> 无 Clip
```

因此可以继续保留：

```python
require_clip: bool
```

以及：

```python
_FACTORIES["clip"] = Clipper
```

但是所有 deployment / conversion 调用必须传：

```python
require_clip=False
```

ANN-training 相关验证仍可以：

```python
require_clip=True
```

不要把 validator 全局改成“永远不检查 Clip”。

---

# 10. `snn2/conversion.py` 修改方案

## 10.1 conversion calibration 不要求 Clip

`validate_calibration()` 中：

```python
validate_site_state_bundle(
    root,
    require_clip=False,
    ...
)
```

每个 site 必需文件改成：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

删除：

```text
clip_state.pt
```

---

## 10.2 强制 Post-finetuning conversion bundle 为 clip-free

为了防止旧 Step 8 工件被误用，建议 conversion validation 不只是“忽略 `clip_state.pt`”，而是**显式拒绝旧的 post-finetuning Clip 文件**。

例如遍历：

```text
layer_*/site_*/clip_state.pt
```

如果发现任何一个：

```text
raise ValueError(
    "Post-finetuning conversion calibration must be clip-free; "
    "re-run calibrate_sites.py --stage post_finetuning"
)
```

这样可以保证：

```text
旧 Step 8 工件
-> fail fast
-> 必须重新 calibration
```

不会出现“目录里还有旧 Clip，但代码恰好没加载”的语义歧义。

---

## 10.3 修改 manifest 条件

当前要求：

```text
purpose = post_finetuning_conversion_calibration
eligible_for_conversion = true
post_finetuning_recalibration = true
state_profile = snn_conversion_with_common_clip
common_clip_required = true
```

改成：

```text
purpose = post_finetuning_conversion_calibration
eligible_for_conversion = true
post_finetuning_recalibration = true
state_profile = snn_conversion_without_clip
common_clip_required = false
```

---

## 10.4 conversion metadata 改为 v4

删除：

```python
"common_clip_applied": True
```

建议增加一个显式反向约束字段：

```python
"snn_clip_applied": False
```

这样 descriptor 自己可以表明新协议。

`validate_conversion_metadata()` 应要求：

```text
format_version = 4
snn_clip_applied = false
```

如果读到旧 v3 descriptor，直接要求重新运行：

```bash
python scripts/convert_snn.py ...
```

不要尝试兼容旧的带 Clip descriptor。

---

# 11. `scripts/verify_artifacts.py` 修改方案

当前 artifact verifier 仍将 Post-finetuning calibration 认定为：

```text
state_profile = snn_conversion_with_common_clip
common_clip_required = true
```

修改为：

```text
state_profile = snn_conversion_without_clip
common_clip_required = false
```

同时继续保留 ANN-training：

```text
state_profile = ann_training_with_common_clip
common_clip_required = true
```

也就是说 verifier 必须明确检查两套不同的 Clip policy：

```text
ANN-training:
Clip REQUIRED

Post-finetuning conversion:
Clip FORBIDDEN
```

建议 verifier 对 Post-finetuning sites 显式确认：

```text
不存在任何 clip_state.pt
```

避免旧工件漏过。

---

# 12. 测试修改方案

所有测试必须表达新的协议，而不是仅让现有测试“能跑”。

## 12.1 `tests/test_calibration_profiles.py`

保留：

```python
test_build_site_states_with_common_clip
test_build_site_states_without_common_clip
```

因为底层 builder 仍需要同时支持 ANN 和 conversion 两种 profile。

将原来的：

```python
test_conversion_materialization_keeps_temporal_common_clip
```

改成类似：

```python
test_conversion_materialization_removes_common_clip
```

测试步骤：

1. 先人为写入旧 `clip_state.pt`。
2. 调用：

```python
materialize_calibration_states(
    ...,
    metadata={
        "state_profile": "snn_conversion_without_clip",
        "common_clip_required": False,
    },
    include_clip=False,
)
```

3. 断言所有 site：

```text
phase_state.pt exists
gif_state.pt exists
mtn_state.pt exists
clip_state.pt DOES NOT exist
```

4. 断言：

```json
"clip_state_present": false
```

ANN-training test 保持：

```text
clip_state.pt exists
clip_state_present = true
clip_valid = true
```

---

## 12.2 `tests/test_controller_state_loading.py`

deployment bundle 应改为 clip-free。

原断言：

```python
set(controller._modules[site_key(...)]) == {neuron, "clip"}
```

改为：

```python
set(controller._modules[site_key(...)]) == {neuron}
```

增加/保留以下关键测试：

### Test A：SNN deployment 无 Clip 也能运行

```text
只有 selected neuron state
没有 clip_state.pt
controller.set_deployment(neuron)
controller.apply(...)
成功
```

对：

```text
phase
gif
mtn
```

都覆盖。

### Test B：ANN phase 仍强制要求 Clip

```text
mode="phase"
phase_state.pt exists
clip_state.pt missing
-> FileNotFoundError
```

### Test C：ANN gif 仍强制要求 Clip

```text
mode="gif"
gif_state.pt exists
clip_state.pt missing
-> FileNotFoundError
```

### Test D：ANN GIF Clip 仍真正限制范围

现有：

```python
test_ann_gif_still_applies_common_clip
```

保留。

最好补一个 `phase` 对应测试，确认本次删除 SNN Clip 没有误伤 ANN Phase Clip。

---

## 12.3 `tests/test_calibration_topology.py`

当前 conversion validator 测试是基于：

```python
include_clip=True
```

改成：

```python
include_clip=False
```

并将：

```python
test_validate_calibration_requires_common_clip_state
```

替换成：

```text
test_validate_conversion_calibration_does_not_require_clip
```

以及建议增加：

```text
test_validate_conversion_calibration_rejects_stale_clip_state
```

确保 conversion 是真正“forbid”，而不是仅“不使用”。

---

## 12.4 `tests/test_conversion_metadata.py`

`_prepare()` 改为：

```python
materialize_calibration_states(
    ...,
    include_clip=False,
)
```

metadata：

```text
format_version = 4
snn_clip_applied = false
```

删除：

```text
common_clip_applied = true
```

旧测试：

```python
("common_clip_applied", False)
```

改成例如：

```python
("snn_clip_applied", True)
```

并要求 validation 失败。

增加旧 descriptor 测试：

```text
format_version = 3
-> fail
```

---

## 12.5 `tests/test_generated_configs.py`

删除 import：

```python
COMMON_CLIP_TEMPORAL_POLICY
```

deployment 预期字典不再包含：

```text
common_clip_temporal_policy
```

增加测试：

```text
人为把 common_clip_temporal_policy 放回 generated cfg
-> validate_config() 必须失败
```

---

## 12.6 `tests/test_neurons.py`

删除测试：

```python
test_neuron_temporal_then_differential_common_clip
```

因为这种行为不再合法。

保留所有：

```text
Clipper.forward()
Clip state validation
Phase static training
GIF static training
GIF temporal
MTN temporal
```

相关测试。

建议新增一个简洁测试，证明 `Clipper` 仍能用于 ANN static tensor：

```text
input 超范围
Clipper(input)
输出落在 lower/upper 内
```

但不再测试 `clip.temporal()`。

---

## 12.7 `tests/test_temporal_ops.py`

删除：

```python
temporal_clip
```

import。

删除：

```python
test_temporal_clip_is_cumulative_then_difference
```

版本测试修改为：

```python
assert CALIBRATION_MANIFEST_FORMAT_VERSION == 3
assert CONVERSION_METADATA_FORMAT_VERSION == 4
assert SITE_STATE_FORMAT_VERSION == 2
assert TEMPORAL_IMPLEMENTATION_VERSION == 2
```

如果最终实现选择 calibration manifest 也升 v4，则同步修改测试，但必须说明原因。

---

# 13. 全仓库旧 SNN Clip 语义清理

修改完成后在仓库根目录执行：

```bash
rg -n \
  "temporal_clip|clip\.temporal|COMMON_CLIP_TEMPORAL_POLICY|common_clip_temporal_policy|common_clip_applied|snn_conversion_with_common_clip" \
  . \
  --glob '!docs/history/**'
```

对**当前代码和当前根目录文档**，上述旧 SNN Clip 语义应为 0 命中。

然后执行：

```bash
rg -n \
  "clip_state\.pt|common_clip_required|ann_training_with_common_clip|Clipper" \
  snn2 scripts tests configs *.md
```

剩余命中必须逐项检查。

允许的命中应只属于：

```text
ANN-training calibration
ANN phase_aware
ANN gif_aware
Clipper static forward
ANN Clip tests
AGENTS.md / README.md / 实验执行总结.md 对 ANN Clip 规则的说明
```

不得再有：

```text
Post-finetuning conversion requires Clip
SNN deployment uses Clip
MTN deployment uses Clip
temporal common Clip
```

---

# 14. 修改 `实验执行总结.md`

这是当前用户发现错误的直接来源，必须同步改正。

## 14.1 Step 5 保留 ANN Clip 描述

Step 5 应明确写：

```text
ANN-training calibration 会生成 phase/gif/mtn/clip state。

clip_state.pt 只供 phase_aware / gif_aware ANN fine-tuning 使用。
```

不要删掉这一正确说明。

---

## 14.2 重写 Step 8

Step 8 应改成：

```text
Post-finetuning conversion calibration 针对每个 final ANN checkpoint
重新统计 10 个 site，并生成：

phase_state.pt
gif_state.pt
mtn_state.pt

不生成 clip_state.pt。
```

明确说明：

```text
Post-finetuning calibration 与 ANN-training calibration 的关键差异之一：
前者是 SNN conversion 用的 clip-free state bundle；
后者为了 aware ANN fine-tuning 包含 common Clip。
```

旧的以下描述必须全部删除：

```text
deployment 必需的 common clip_state.pt
Post-finetuning common Clip
三类 SNN full-temporal deployment 中统一使用 Clip
无 Clip 工件不能复用
```

反过来应说明：

```text
旧的带 clip_state.pt 的 Post-finetuning calibration 属于旧协议，不能复用。
```

---

## 14.3 重写 Step 10

删除：

```text
Phase/GIF/MTN conversion 同时读取 common clip_state.pt
neuron temporal output 后执行 temporal Clip
累计值 Clip 后做时间差分
```

改为：

```text
Phase conversion -> 只读取 Phase state
GIF conversion   -> 只读取 GIF state
MTN conversion   -> 只读取 MTN state
```

以及：

```text
SNN deployment 不应用任何额外 Clip。
GIF 内部 qmin/qmax clamp 属于 GIF 编码算法自身，继续保留。
```

---

## 14.4 检查整个当前文档

完成后执行：

```bash
rg -n "Clip|clip_state|common_clip|temporal_clip" 实验执行总结.md
```

逐项检查。

最终 Clip 相关内容只能表达：

```text
ANN phase_aware / gif_aware 使用 Clip
SNN conversion / deployment 不使用 Clip
```

---

# 15. 重构 `代码结构总结.md`

用户要求：

> 删除 `代码结构总结.md` 中除 `2. 目录结构` 以外的所有内容。以后该文件只要代码目录结构，以及每个文件后面用一句话描述其实现的功能。

## 15.1 最终文档结构

建议最终仅保留：

```markdown
# 代码结构总结

## 2. 目录结构

```text
...
```
```

不要再包含：

```text
总体数据流
Rotation 详细实现
Activation replacement 详细解释
Calibration 原理
Prefix 原理
训练流程
evaluation 流程
conversion 细节
公式
实验命令
```

这些内容全部删除。

---

## 15.2 目录树必须基于修改后的真实仓库

不要直接复制旧 section 2，因为当前旧目录树已经漏掉若干后续新增文件。

完成本轮代码修改后，应基于真实 tracked tree 重新整理，例如可先运行：

```bash
git ls-files
```

然后生成准确的目录树。

至少必须反映：

```text
AGENTS.md
README.md

snn2/state_validation.py
snn2/temporal_model.py
snn2/temporal_ops.py

tests/test_calibration_profiles.py
tests/test_controller_state_loading.py
tests/test_conversion_metadata.py
tests/test_temporal_model_integration.py
tests/test_temporal_ops.py
tests/test_temporal_prefix.py
...
```

不要继续使用旧的、不完整的目录列表。

---

## 15.3 每个文件后一行/一句话说明功能

推荐格式：

```text
├── scripts/
│   ├── prepare_rotation.py        — 生成并验证 Rotation 工件并保存 fused Base。
│   ├── discover_prefix.py         — 按执行阶段发现 Prefix token 并构造固定 KV cache。
│   ├── calibrate_sites.py         — 收集 10 个 replacement site 的统计并生成阶段对应 calibration state。
│   ├── train_ann.py               — 启动 ANN 全参数微调。
│   └── convert_snn.py             — 为指定 neuron 创建 SNN conversion descriptor。
```

每个文件只允许一句简洁描述。

不要在文件描述后继续展开第二段说明。

---

## 15.4 对目录本身

目录可以仅列名称；若希望描述目录，也只能一句话。

不要把 `代码结构总结.md` 重新变成设计文档。

---

# 16. 新建根目录 `AGENTS.md`

本节属于 Codex 任务；README 不属于本次 Codex 任务。

在：

```text
SNN/AGENTS.md
```

创建标准 Codex 项目规则文件。

**本次只能放以下两条规则，不得加入其它规则、背景、命令或说明。**

建议内容严格控制为：

```markdown
# AGENTS.md

1. Clip 只用于 ANN 的 `phase_aware` / `gif_aware` 微调：ANN-training calibration 可以生成并使用 `clip_state.pt`；Post-finetuning conversion calibration、SNN conversion、Phase/GIF/MTN SNN deployment 与 SNN evaluation 均不得生成、加载或执行任何 Clip。

2. `代码结构总结.md` 只允许保留 `2. 目录结构`，用于记录当前仓库目录结构；每个文件后只用一句话描述该文件实现的功能，任何代码修改导致目录或文件功能变化时都必须同步更新该文件。
```

不要加入第三条。

不要加入：

```text
测试规范
Git 规范
命名规范
Prefix 规范
Rotation 规范
代码风格规范
```

本次用户明确要求“其余不要放任何内容”。

---


# 18. `AGENTS.md` 加入代码结构总结

完成新文件后：

```text
代码结构总结.md
```

根目录树必须包含：

```text
├── AGENTS.md    — 记录后续代码修改必须遵守的项目规则。
```

`README.md` 不属于本次 Codex 修改任务；如果仓库中已经存在该文件，`代码结构总结.md` 应当按照“真实目录结构”原则正常列出它，但 Codex 不得因为本方案主动创建、改写或删除 README。


# 19. generated config 处理

修改：

```text
configs/experiment_matrix.yaml
```

后必须重新生成配置：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated
```

`configs/generated/` 当前由 Git ignore 管理，不要求把 generated 文件提交到仓库。

但是测试必须基于新 matrix 动态生成的 config 通过。

---

# 20. 旧实验工件处理

这是本次修改后实验恢复时必须遵守的规则。

## 20.1 可以继续保留的结果

本次不改变：

```text
Rotation
ANN-training Prefix
ANN-training calibration 的 ANN Clip 语义
已经完成的 ANN fine-tuning 参数
Post-finetuning Prefix 的定义
```

因此已有有效的：

```text
rotation/
ANN-training calibration/
final ANN checkpoint/
post-finetuning prefix/
```

不需要仅因为本次 Clip 修复而全部重新生成。

特别是：

```text
ANN-training calibration 下的 clip_state.pt
```

是正确工件，不要删除。

---

## 20.2 必须失效并重跑的结果

旧代码下生成的以下结果都使用了错误的 SNN Clip 语义：

```text
Post-finetuning conversion calibration
SNN conversion descriptor
SNN evaluation
```

因此修改完成后必须重新执行：

```text
Step 8
Step 10 conversion
Step 10 SNN evaluation
```

旧 SNN evaluation 指标不能再与新结果混用。

---

## 20.3 推荐清理方式

虽然新 `materialize_calibration_states(..., include_clip=False)` 会自动 unlink stale `clip_state.pt`，为了避免旧文件残留，建议在正式重跑某个 run 的 Step 8 前删除或移动该 run 的：

```text
post_finetuning/conversion_calibration/.../sites/
```

然后重新：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning
```

不要删除 final ANN checkpoint。

---

# 21. 修改后的实验协议示意

最终必须形成以下清晰分离。

## 21.1 ANN training

```text
Pre-finetuning model
    ↓
ANN-training calibration
    ├── phase_state.pt
    ├── gif_state.pt
    ├── mtn_state.pt
    └── clip_state.pt
          ↓
    ┌─────┴─────┐
    ↓           ↓
phase_aware   gif_aware
Phase+Clip    GIF+Clip
```

`vanilla` / `unaware` 不执行 activation replacement。

---

## 21.2 SNN conversion

```text
final ANN checkpoint
    ↓
Post-finetuning Prefix
    ↓
Post-finetuning conversion calibration
    ├── phase_state.pt
    ├── gif_state.pt
    └── mtn_state.pt
    ×  no clip_state.pt
    ↓
conversion
    ├── Phase SNN
    ├── GIF SNN
    └── MTN SNN
    ↓
deployment
    ×  no Clip
```

这就是本次修改最终必须实现的实验语义。

---

# 22. 建议的执行顺序

Codex 按以下顺序修改，避免中途测试处于混乱状态：

```text
1. 修改 temporal_ops.py
2. 修改 neurons.py
3. 修改 calibration.py
4. 修改 controller.py
5. 修改 conversion.py
6. 修改 state_validation.py（如需要）
7. 修改 config.py
8. 修改 experiment_matrix.yaml
9. 修改 verify_artifacts.py
10. 修改所有受影响测试
11. 修改 实验执行总结.md
12. 新建 AGENTS.md
13. 最后重建 代码结构总结.md
14. materialize configs
15. 执行测试
16. 执行 rg 语义扫描
```

---

# 23. 测试命令

先跑核心目标测试：

```bash
pytest -q \
  tests/test_calibration_profiles.py \
  tests/test_calibration_topology.py \
  tests/test_controller_state_loading.py \
  tests/test_conversion_metadata.py \
  tests/test_generated_configs.py \
  tests/test_neurons.py \
  tests/test_temporal_ops.py
```

再跑完整测试：

```bash
pytest -q
```

如果完整测试中其它 temporal integration / Prefix / evaluation 测试因为 metadata 字段变化失败，按新协议同步更新，但不要通过恢复 SNN Clip 来让旧测试通过。

---

# 24. 最终静态检查

## 24.1 禁止旧 SNN Clip 路径

执行：

```bash
rg -n \
  "temporal_clip|clip\.temporal|COMMON_CLIP_TEMPORAL_POLICY|common_clip_temporal_policy|common_clip_applied|snn_conversion_with_common_clip" \
  snn2 scripts configs tests \
  实验执行总结.md AGENTS.md
```

预期：

```text
0 个旧 SNN Clip 语义命中
```

如果为了测试“拒绝旧字符串”必须出现旧 key，可以在测试 fixture 中出现，但要确认生产代码不再接受/执行它。

---

## 24.2 ANN Clip 必须仍存在

执行：

```bash
rg -n \
  "ann_training_with_common_clip|clip_state\.pt|Clipper|modules\[\"clip\"\]" \
  snn2 tests 实验执行总结.md AGENTS.md
```

必须仍能看到 ANN training 的合法 Clip 路径。

重点人工确认 `controller.py` 最终仍有：

```python
mode == "phase":
    Phase -> Clip

mode == "gif":
    GIF -> Clip
```

但 `deploy_*` 中没有 Clip。

---

# 25. 验收标准

全部满足后才算完成。

## A. ANN Clip

- `phase_aware` ANN training 仍执行 Phase + Clip。
- `gif_aware` ANN training 仍执行 GIF + Clip。
- ANN-training calibration 仍生成 `clip_state.pt`。
- 缺少 ANN `clip_state.pt` 时 aware training 能 fail-fast。

## B. Post-finetuning calibration

- `--stage post_finetuning` 生成 Phase/GIF/MTN state。
- 不生成 `clip_state.pt`。
- 若目录中已有旧 `clip_state.pt`，重新 materialize 后自动删除。
- manifest：

```text
state_profile = snn_conversion_without_clip
common_clip_required = false
```

## C. conversion

- conversion 不要求 `clip_state.pt`。
- conversion 最好拒绝仍含旧 `clip_state.pt` 的 Post-finetuning calibration。
- descriptor 使用新 format version。
- metadata 明确 `snn_clip_applied=false`。

## D. SNN runtime

- `deploy_phase` 只加载 Phase。
- `deploy_gif` 只加载 GIF。
- `deploy_mtn` 只加载 MTN。
- 三者均不加载 `Clipper`。
- 三者均不执行 `temporal_clip()`。
- `temporal_clip()` / `Clipper.temporal()` 已从生产代码删除。

## E. 配置

- `deployment.common_clip_temporal_policy` 从 matrix 和当前 config protocol 删除。
- 新 generated config 不再包含它。
- legacy key 应被 validator 拒绝。

## F. 文档

- `实验执行总结.md` 的 Step 8 / Step 10 与新 clip-free SNN 协议一致。
- `代码结构总结.md` 只剩 `2. 目录结构`，每个文件一句话说明功能。
- 根目录存在 `AGENTS.md`，且只包含用户指定的两条规则。
- `代码结构总结.md` 已包含 `AGENTS.md`；若仓库中已有 `README.md`，则仅按真实目录结构列出，不在本任务中创建或改写。

## G. 测试

```bash
pytest -q
```

全部通过。

---

# 26. 明确禁止的错误修改

不要做以下事情：

1. **不要把 Clip 从 ANN `phase_aware` / `gif_aware` 中一起删掉。**
2. 不要删除 ANN-training `clip_state.pt` 的生成。
3. 不要因为删除 common Clip 而删除 GIF 自己的 qmin/qmax clamp。
4. 不要让 MTN SNN 继续加载一个“虽然不用但仍存在”的 Clip module。
5. 不要让 conversion 继续要求 `clip_state.pt`。
6. 不要保留 `Clipper.temporal()` 作为未使用的 legacy runtime 路径。
7. 不要为了兼容旧 SNN descriptor 而继续支持带 common Clip 的 conversion。
8. 不要修改或删除 final ANN checkpoint。
9. 不要把 `代码结构总结.md` 再写回长篇原理说明。
10. 不要在 `AGENTS.md` 加入第三条及更多规则。
11. 不要在本次 Codex 任务中创建、改写或删除 `README.md`。
12. 不要为了清理关键词而篡改 `docs/history/` 中的历史修改方案。

---

# 27. 本次修改完成后的核心一句话

最终项目必须严格遵守：

> **Clip 是 ANN-aware fine-tuning 的训练期约束，只服务 `phase_aware` / `gif_aware` ANN replacement；final ANN 完成后，Post-finetuning calibration、Phase/GIF/MTN SNN conversion 和所有 SNN deployment/evaluation 全部是 clip-free。**
