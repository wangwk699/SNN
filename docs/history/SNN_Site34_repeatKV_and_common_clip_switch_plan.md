# SNN Phase Site 3/4 严格对齐 + ANN-aware Common Clip 开关修改方案

## 0. 修改目标

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

当前基准 commit：

```text
e1e0effa1c6410bb9ee763e8281af0cf151220f0
```

本次只完成以下三件事：

1. 将 Site 3 / Site 4 的 **Phase calibration statistical view** 严格改成与 SpikingLLM 一致：Phase K/V statistics 在 `repeat_kv()` 之后构造；
2. 修正 `实验执行总结.md` 中“四个 Prefix 开关彼此独立”的错误表述；
3. 新增 ANN-aware common Clip 使用开关：
   ```yaml
   replacement:
     common_clip_enabled: true | false
   ```
   ANN-training calibration **无论该参数取 true 还是 false，都继续生成 `clip_state.pt`**；该参数只控制 `phase_aware` / `gif_aware` ANN fine-tuning forward 是否真正执行 common Clip。

本次不要修改：

- 10-site topology；
- Site 9；
- Phase `T/base/tau/v0`；
- Phase `surrogate_slope=1.0`；
- Phase τ EMA factor `0.99`、FP32 accumulator；
- Phase scalar τ；
- Prefix K/V runtime 必须经过 Site 3/4 neuron；
- Prefix K/V calibration statistics 必须排除 Prefix positions；
- full Softmax 必须经过 Site 5；
- Embedding temporal policy `x/T`；
- final RMSNorm global Phase neuron；
- aware conversion 继续复用 Pre-finetuning Prefix + ANN-training calibration；
- vanilla/unaware 继续使用 Post-finetuning Prefix + Post-finetuning conversion calibration；
- 所有 SNN deployment 均不得执行 common Clip。

---

# 1. 新参数命名与语义

统一使用：

```yaml
replacement:
  common_clip_enabled: true
```

参数名不要再另起别名。

新增统一 helper：

```python
def training_common_clip_enabled(cfg: dict[str, Any]) -> bool:
    return (
        is_aware_ann_mode(cfg)
        and bool(cfg["replacement"]["common_clip_enabled"])
    )
```

语义：

```text
phase_aware + true
    ANN forward = Clip(Phase(x))

phase_aware + false
    ANN forward = Phase(x)

gif_aware + true
    ANN forward = Clip(GIF(x))

gif_aware + false
    ANN forward = GIF(x)

vanilla / unaware
    不执行 activation replacement
    common_clip_enabled 在 resolved config 中固定为 false
```

注意：

> `common_clip_enabled=false` 只表示 ANN fine-tuning 不执行 common Clip，绝不表示 ANN-training calibration 不生成 `clip_state.pt`。

ANN-training calibration 对 aware mode 始终生成：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

Phase/GIF/MTN SNN conversion/deployment 无论该参数 true 还是 false，都只加载目标 neuron state，不加载 `clip_state.pt`。

---

# 2. `configs/experiment_matrix.yaml`

修改三个 experiment 的：

```yaml
replacement:
  train_mode: phase
  outer_clip_backward: hard_clip
  clip_rule: phase_mtn_gif_low_high_common_intersection
```

增加：

```yaml
  common_clip_enabled: true
```

默认先设为 `true`，保持当前实验行为。

后续如果要跑 no-common-Clip variant，只修改：

```yaml
replacement:
  common_clip_enabled: false
```

然后重新：

```bash
python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated
```

不需要新增第二套 generated config 文件命名规则。相同：

```text
exp1_qwen3_1_7b_tldr__phase_aware.yaml
exp1_qwen3_1_7b_tldr__gif_aware.yaml
```

在不同 `common_clip_enabled` 下会进入不同 artifact run root，因此不会互相覆盖。

---

# 3. `snn2/config.py`

## 3.1 `resolve_config()`

在读取 mode 前先：

```python
cfg["replacement"].setdefault(
    "common_clip_enabled",
    True,
)
```

这样旧 aware config 默认保持当前“使用 common Clip”的行为。

然后：

### vanilla

强制：

```python
cfg["replacement"]["train_mode"] = "none"
cfg["replacement"]["common_clip_enabled"] = False
```

### unaware

强制：

```python
cfg["replacement"]["train_mode"] = "none"
cfg["replacement"]["common_clip_enabled"] = False
```

### phase_aware

保留用户配置：

```python
cfg["replacement"]["train_mode"] = "phase"
# common_clip_enabled 保留 true / false
```

### gif_aware

保留用户配置：

```python
cfg["replacement"]["train_mode"] = "gif"
# common_clip_enabled 保留 true / false
```

---

## 3.2 `validate_config()`

要求：

```python
common_clip_enabled = cfg["replacement"].get(
    "common_clip_enabled"
)

if not isinstance(common_clip_enabled, bool):
    raise ValueError(
        "replacement.common_clip_enabled must be true or false"
    )
```

并要求：

```text
vanilla / unaware:
    resolved common_clip_enabled 必须 false

phase_aware / gif_aware:
    true / false 都合法
```

不要再强制 aware 必须为 true。

新增：

```python
def training_common_clip_enabled(
    cfg: dict[str, Any],
) -> bool:
    return (
        is_aware_ann_mode(cfg)
        and bool(
            cfg["replacement"]["common_clip_enabled"]
        )
    )
```

整个项目后续统一调用该 helper，不要直接散落：

```python
cfg["replacement"]["common_clip_enabled"]
```

---

# 4. Artifact run root 加入 common Clip 开关

修改：

```text
snn2/artifacts.py
```

## 4.1 不得修改 `prefix_enabled_dirname()`

当前：

```python
def prefix_enabled_dirname(enabled: bool) -> str:
    return (
        "prefix_enabled_ture"
        if enabled
        else "prefix_enabled_false"
    )
```

保留不动。

原因：

这个函数还被：

- ANN-training calibration；
- Post-finetuning calibration；
- evaluation；
- SNN conversion；

等其他目录使用。

如果直接修改，会错误地把 common Clip flag 引入共享 calibration 目录。

---

## 4.2 新增 ANN run variant dirname helper

例如：

```python
def ann_run_variant_dirname(
    *,
    prefix_enabled: bool,
    common_clip_enabled: bool,
    aware_mode: bool,
) -> str:
    result = prefix_enabled_dirname(prefix_enabled)

    if aware_mode:
        result += (
            "_common_clip_enabled_true"
            if common_clip_enabled
            else "_common_clip_enabled_false"
        )

    return result
```

也可以使用等价函数名，但最终字符串必须严格为：

```text
prefix_enabled_ture_common_clip_enabled_true
prefix_enabled_ture_common_clip_enabled_false
```

历史拼写：

```text
ture
```

继续保留。

---

## 4.3 `ArtifactLayout.root`

当前：

```python
self.root = (
    model_root
    / exp["ann_mode"]
    / learning_rate
    / prefix_enabled_dirname(ann_prefix)
    / seed
)
```

改成：

```python
run_variant = ann_run_variant_dirname(
    prefix_enabled=ann_prefix,
    common_clip_enabled=training_common_clip_enabled(cfg),
    aware_mode=is_aware_ann_mode(cfg),
)

self.root = (
    model_root
    / exp["ann_mode"]
    / learning_rate
    / run_variant
    / seed
)
```

### aware mode 结果

例如用户给出的 GIF-aware 路径：

```text
artifacts/snn2_main_v1/tldr/
Qwen_Qwen3-1.7B-Base/
gif_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_true/
seed42/
```

关闭 common Clip：

```text
artifacts/snn2_main_v1/tldr/
Qwen_Qwen3-1.7B-Base/
gif_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_false/
seed42/
```

Phase-aware 同样：

```text
phase_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_true/
seed42/
```

或：

```text
phase_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_false/
seed42/
```

---

## 4.4 vanilla / unaware 路径保持不变

common Clip 对这两种 mode 没有意义，因此不要增加无意义 suffix。

仍为：

```text
vanilla/.../prefix_enabled_false/seed42/
unaware/.../prefix_enabled_ture/seed42/
```

---

## 4.5 shared calibration 路径必须保持不变

非常重要。

`ann_training_calibration_dir` 仍只由：

```text
model
rotation policy
ann_training.prefix_enabled
```

决定：

```text
_shared/seed42/
rotated_prefix/
ann_training_calibration/
prefix_enabled_ture/
```

不要加入：

```text
common_clip_enabled_true
common_clip_enabled_false
```

因为：

> true / false 两个 aware training variant 使用完全同一套 ANN-training calibration；calibration 始终生成 Clip，只是训练时选择是否应用。

因此两个 variant 必须能够共享同一个：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

---

# 5. `SiteController` 增加 common Clip runtime 开关

修改：

```text
snn2/controller.py
```

## 5.1 Constructor

增加：

```python
def __init__(
    self,
    mode: str = "identity",
    site_root: str | Path | None = None,
    *,
    common_clip_enabled: bool = False,
):
    ...
    self.common_clip_enabled = bool(
        common_clip_enabled
    )
```

建议增加保护：

```python
if (
    self.mode not in {"phase", "gif"}
    and self.common_clip_enabled
):
    raise ValueError(
        "common_clip_enabled only applies to "
        "phase/gif ANN replacement modes"
    )
```

`deploy_phase/deploy_gif/deploy_mtn` 必须始终为 false。

---

## 5.2 `_load()`

当前：

```python
if self.mode == "phase":
    required = ("phase", "clip")
elif self.mode == "gif":
    required = ("gif", "clip")
```

改为：

```python
if self.mode == "phase":
    required = (
        ("phase", "clip")
        if self.common_clip_enabled
        else ("phase",)
    )

elif self.mode == "gif":
    required = (
        ("gif", "clip")
        if self.common_clip_enabled
        else ("gif",)
    )
```

因此：

```text
common_clip_enabled=false
```

时，ANN forward 连 `clip_state.pt` 都不加载。

但是 calibration bundle 中该文件仍必须存在。

---

## 5.3 `apply()`

当前：

```python
if self.mode == "phase":
    return modules["clip"](
        modules["phase"](x)
    )

if self.mode == "gif":
    return modules["clip"](
        modules["gif"](x)
    )
```

改成：

```python
if self.mode == "phase":
    output = modules["phase"](x)
    return (
        modules["clip"](output)
        if self.common_clip_enabled
        else output
    )

if self.mode == "gif":
    output = modules["gif"](x)
    return (
        modules["clip"](output)
        if self.common_clip_enabled
        else output
    )
```

deployment 逻辑完全不改。

---

# 6. `snn2/training.py`

导入：

```python
from .config import (
    is_aware_ann_mode,
    training_common_clip_enabled,
    training_prefix_enabled,
)
```

## 6.1 Controller

当前：

```python
controller = SiteController(
    mode=mode,
    site_root=layout.ann_training_site_dir,
)
```

改成：

```python
common_clip_enabled = (
    training_common_clip_enabled(cfg)
)

controller = SiteController(
    mode=mode,
    site_root=layout.ann_training_site_dir,
    common_clip_enabled=common_clip_enabled,
)
```

---

## 6.2 calibration validation 仍必须要求 Clip

当前：

```python
if is_aware_ann_mode(cfg):
    validate_site_state_bundle(
        layout.ann_training_site_dir,
        require_clip=True,
    )
```

必须保持。

即使：

```text
common_clip_enabled=false
```

仍然要求：

```text
clip_state.pt
```

存在。

原因：

ANN-training calibration protocol 本身没有变化，只有 ANN forward 使用策略变化。

---

## 6.3 保存到 model config

建议在训练前写：

```python
model.config.snn2_ann_mode = ...
model.config.snn2_ann_common_clip_enabled = (
    common_clip_enabled
)
```

这样：

```text
ann/final/config.json
```

能直接确认该 checkpoint 是否使用了 common Clip。

---

## 6.4 `training_result.json`

增加：

```text
ann_training_common_clip_enabled
ann_training_common_clip_applied
ann_training_common_clip_state_required
```

具体：

```python
"ann_training_common_clip_enabled":
    common_clip_enabled,

"ann_training_common_clip_applied":
    common_clip_enabled,

"ann_training_common_clip_state_required":
    is_aware_ann_mode(cfg),
```

对于：

```text
phase_aware/gif_aware + false
```

结果应为：

```json
{
  "ann_training_common_clip_enabled": false,
  "ann_training_common_clip_applied": false,
  "ann_training_common_clip_state_required": true
}
```

这个组合是合法且符合本次设计的。

---

# 7. Calibration metadata：明确“生成 Clip”和“训练是否使用 Clip”是两件事

修改：

```text
snn2/calibration.py
```

不要把现有：

```text
common_clip_required
```

改成随 `replacement.common_clip_enabled` 变化。

ANN-training calibration 仍必须：

```text
common_clip_required = true
```

因为 bundle 始终包含 Clip。

建议增加 manifest 字段：

```text
common_clip_generated: true
common_clip_application_control:
    replacement.common_clip_enabled
```

对于 ANN-training calibration：

```text
common_clip_generated = true
```

对于 Post-finetuning conversion calibration：

```text
common_clip_generated = false
```

这样避免之后把：

```text
bundle 是否包含 clip_state.pt
```

和：

```text
ANN training 是否执行 Clip
```

混为一谈。

不要把 `replacement.common_clip_enabled` 加到 shared calibration 路径，也不要让它改变 Phase/GIF/MTN/Clip state 数值。

---

# 8. aware training provenance

当前训练开始前已经冻结：

```text
Pre-finetuning Prefix hash
ANN-training calibration manifest hash
```

保持。

两个 common Clip variant：

```text
common_clip_enabled=true
common_clip_enabled=false
```

应允许引用完全相同的：

```text
ann_training_prefix_state_sha256
ann_training_prefix_kv_sha256
ann_training_calibration_manifest_sha256
```

不要因为开关不同重新 calibration。

训练结果本身通过：

```text
不同 ArtifactLayout.root
+
training_result.json 中 common_clip_enabled
+
final config.json 中 snn2_ann_common_clip_enabled
```

区分。

---

# 9. Conversion metadata 增加 ANN common Clip provenance

修改：

```text
snn2/conversion.py
```

建议在 conversion descriptor 中增加：

```text
source_ann_common_clip_enabled
```

值：

```python
training_common_clip_enabled(cfg)
```

aware：

```text
true / false
```

vanilla/unaware：

```text
false
```

`validate_conversion_metadata()` 同样校验：

```python
metadata["source_ann_common_clip_enabled"]
==
training_common_clip_enabled(cfg)
```

这样即使用户手工移动 descriptor，也不能把：

```text
common_clip_enabled=true
```

训练出的 ANN conversion 与：

```text
common_clip_enabled=false
```

配置混用。

### SNN Clip 语义保持

无论该字段 true / false：

```text
snn_clip_applied = false
```

必须保持。

不要在 SNN controller 中传递：

```text
common_clip_enabled=true
```

---

# 10. Evaluation metadata

修改：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

final ANN 与 SNN evaluation metadata 中增加：

```text
ann_training_common_clip_enabled
```

值：

```python
training_common_clip_enabled(cfg)
```

Base / rotated-pre-finetuning evaluation 建议：

```text
ann_training_common_clip_enabled = false
```

这样最终实验 JSON 能直接识别：

```text
phase_aware + Clip
phase_aware - Clip
gif_aware + Clip
gif_aware - Clip
```

artifact root 已经区分，因此 evaluation 子目录不需要再额外重复：

```text
common_clip_enabled_true/false
```

evaluation 仍在对应 run root 内使用原来的：

```text
prefix_enabled_ture
prefix_enabled_false
```

作为 final evaluation 的 Prefix on/off 子目录。

---

# 11. P0：Site 3 / Site 4 Phase statistical view 移到 `repeat_kv()` 后

修改：

```text
snn2/model_integration.py
```

这是严格对齐 SpikingLLM 的修改。

当前 collect 路径对 K/V：

```text
[B,Hkv,K,D]
```

直接构造 Phase view：

```text
[B,K,Hkv*D]
```

应改为：

```text
generic statistics:
    继续使用 pre-repeat K/V

Phase EMA statistics:
    current K/V
    -> repeat_kv(groups)
    -> [B,H,K,D]
    -> phase_statistical_view()
    -> [B,K,H*D]
```

---

## 11.1 `groups` 提前计算

把：

```python
groups = int(
    getattr(module, "num_key_value_groups", 1)
)
```

移动到 Site 3/4 collect 判断之前。

---

## 11.2 collect 路径统一显式记录 Site 3 / Site 4

不要只在：

```text
collect + past_length > 0
```

时特殊处理。

推荐整体改成：

```python
groups = int(
    getattr(module, "num_key_value_groups", 1)
)

if controller.mode == "collect":

    if past_length:
        statistics_key = key[
            ..., past_length:, :
        ]
        statistics_value = value[
            ..., past_length:, :
        ]
    else:
        statistics_key = key
        statistics_value = value

    phase_key = repeat_kv(
        statistics_key,
        groups,
    )
    phase_value = repeat_kv(
        statistics_value,
        groups,
    )

    controller.record_activation(
        layer_index,
        3,
        statistics_key,
        phase_activation=phase_statistical_view(
            3,
            phase_key,
        ),
    )

    controller.record_activation(
        layer_index,
        4,
        statistics_value,
        phase_activation=phase_statistical_view(
            4,
            phase_value,
        ),
    )

else:

    key = controller.apply(
        layer_index,
        3,
        key,
    )

    value = controller.apply(
        layer_index,
        4,
        value,
    )

key = repeat_kv(
    key,
    groups,
)

value = repeat_kv(
    value,
    groups,
)
```

---

## 11.3 必须保持的两个语义

### Generic K/V statistics

仍然：

```text
[B,Hkv,K_current,D]
```

因此 GIF/MTN/Clip statistics 不改变。

### Phase K/V statistics

改成：

```text
[B,H,K_current,D]
↓
[B,K_current,H*D]
```

并且 Prefix positions 仍排除。

---

## 11.4 Runtime 不要顺手改

本次只要求：

> Phase **statistical view** 严格位于 `repeat_kv` 后。

ANN-aware runtime Site 3/4 当前在 `repeat_kv` 前执行，主实验使用 scalar Phase τ，数值上与 repeat 后逐元素应用等价。

不要在本次方案中重构 runtime placement，避免额外影响 GIF replacement 和现有 10-site topology。

---

# 12. `phase_statistics.py`

现有：

```python
phase_statistical_view(
    site_index,
    x,
)
```

本身无需改变公式。

Site 3/4 仍：

```python
[B,H,K,D]
-> [B,K,H*D]
```

变化只在 caller：

> 传进来的 `x` 现在是 `repeat_kv()` 后的 tensor。

不要让 helper 自己调用 `repeat_kv()`，因为 helper 不应知道：

```text
num_key_value_groups
```

该信息属于 attention module。

---

# 13. 修正 `实验执行总结.md` 的 Prefix 开关表述

当前：

```text
“四个开关彼此独立”
```

不再正确。

必须删除这种绝对表述，以及删除例子：

```text
ann_training.prefix_enabled=false
rotated_pre_finetuning.prefix_enabled=true
```

因为当前主协议中：

```text
unaware/phase_aware/gif_aware
```

都强制要求 shared Pre-finetuning Prefix。

---

## 13.1 正确的 Prefix 协议

明确写：

| 配置 | vanilla | unaware | phase_aware | gif_aware |
|---|---:|---:|---:|---:|
| `ann_training.prefix_enabled` | 强制 `false` | 强制 `true` | 强制 `true` | 强制 `true` |
| `rotated_pre_finetuning.prefix_enabled` | 无此流程 | 可独立 true/false | 可独立 true/false | 可独立 true/false |
| `post_finetuning.prefix_enabled` | 可 true/false | 可 true/false | 不使用 | 不使用 |
| `evaluation.prefix_enabled` | 可 true/false | 可 true/false | 可 true/false | 可 true/false |

并说明：

```text
ann_training.prefix_enabled
并不是四种 mode 都可自由设置的 ablation 参数。
```

它由 mode protocol 约束：

```text
vanilla -> false
rotated modes -> true
```

真正可以独立控制的是对应 evaluation / post-finetuning 阶段的开关。

---

# 14. `实验执行总结.md` 增加 Common Clip 参数说明

在 Step 1 增加：

```yaml
replacement:
  common_clip_enabled: true
```

说明：

```text
该参数只控制 phase_aware/gif_aware ANN fine-tuning
是否在 Phase/GIF replacement 后执行 common Clip。

ANN-training calibration 无论 true/false 都生成 clip_state.pt。
SNN conversion/deployment 无论 true/false 都不执行 common Clip。
```

---

## 14.1 Step 5 ANN-training calibration

增加：

```text
common_clip_enabled=true/false 两种 aware training variant
共享完全相同的 ANN-training calibration。

因此切换该参数不需要重新发现 Prefix，
也不需要重新生成 ANN-training calibration，
前提是 calibration 的其他配置、代码版本和 artifact hash 未改变。
```

---

## 14.2 Step 6 ANN training

说明：

```text
phase_aware:
  common_clip_enabled=true  -> Clip(Phase(x))
  common_clip_enabled=false -> Phase(x)

gif_aware:
  common_clip_enabled=true  -> Clip(GIF(x))
  common_clip_enabled=false -> GIF(x)
```

训练仍然属于 ANN fine-tuning，不引入跨层 T。

---

## 14.3 结果路径

加入实例：

```text
.../phase_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_true/
seed42/
```

```text
.../phase_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_false/
seed42/
```

GIF-aware 同理。

说明：

```text
vanilla/unaware 因为不执行 activation replacement，
run root 不增加 common_clip_enabled suffix。
```

---

# 15. README / AGENTS 必要同步

## `README.md`

ANN Fine-tuning 一节改成：

```text
ANN-training calibration 始终为 aware mode 生成 common Clip state。
replacement.common_clip_enabled 决定 phase_aware/gif_aware
ANN forward 是否实际执行该 Clip。
SNN conversion/deployment 始终不使用 common Clip。
```

---

## `AGENTS.md`

增加两条规则：

```text
7. ANN-training calibration 对 phase_aware/gif_aware 始终生成
   clip_state.pt；replacement.common_clip_enabled 只控制 ANN
   training forward 是否应用 Clip，不得改变 shared calibration 内容。

8. aware run root 必须包含 common_clip_enabled_true/false；
   shared Pre-finetuning Prefix 和 ANN-training calibration
   不得因为该开关拆成两套。
```

---

# 16. `代码结构总结.md`

只有在文件功能描述需要变化时做最小更新。

新增文件没有必要，因为本次不新增代码文件。

可将：

```text
snn2/artifacts.py
```

一句话说明补充为：

```text
负责 mode/prefix/common-Clip-aware 的实验 artifact 路径组织。
```

`controller.py`：

```text
负责 ANN replacement 与 SNN deployment；
ANN Phase/GIF common Clip 可由配置开关控制。
```

不要新增额外章节。

---

# 17. 测试

至少修改：

```text
tests/test_statistics.py
tests/test_temporal_model_integration.py
tests/test_generated_configs.py
tests/test_controller_state_loading.py
tests/test_training.py
tests/test_conversion_metadata.py
tests/test_evaluation_paths.py
tests/test_post_finetuning_protocol.py
tests/test_sites.py
```

---

## 17.1 Site 3/4 Phase view 必须基于 repeat_kv 后 heads

修改当前 Prefix exclusion test。

构造：

```text
B = 1
Hkv = 2
groups = 2
H = 4
current K length = 2
D = 4
```

module：

```python
num_key_value_groups = 2
```

断言 generic statistics：

```text
key_stats.channels == 4
value_stats.channels == 4
```

保持不变。

断言：

```text
key_stats.phase_channels == H * D == 16
value_stats.phase_channels == H * D == 16
```

而不能再是：

```text
Hkv * D == 8
```

expected 必须用：

```python
expected_key = phase_statistical_view(
    3,
    repeat_kv(
        current_key,
        2,
    ),
).abs().amax(dim=(0, 1))

expected_value = phase_statistical_view(
    4,
    repeat_kv(
        current_value,
        2,
    ),
).abs().amax(dim=(0, 1))
```

同时继续断言 Prefix positions 没进入 statistics。

---

## 17.2 无 Prefix 时也必须走 repeat 后 Phase view

增加测试：

```text
past_length = 0
Hkv=2
groups=2
```

断言：

```text
generic channels = D
phase_channels = H*D
```

防止代码只修了 Prefix branch。

---

## 17.3 common Clip config

测试：

```text
phase_aware + true  -> 合法
phase_aware + false -> 合法
gif_aware + true    -> 合法
gif_aware + false   -> 合法
```

并断言：

```text
vanilla resolved -> false
unaware resolved -> false
```

非法：

```yaml
common_clip_enabled: "false"
```

字符串必须报错。

---

## 17.4 Artifact root

断言：

```text
phase_aware true:
prefix_enabled_ture_common_clip_enabled_true

phase_aware false:
prefix_enabled_ture_common_clip_enabled_false

gif_aware true:
prefix_enabled_ture_common_clip_enabled_true

gif_aware false:
prefix_enabled_ture_common_clip_enabled_false
```

vanilla/unaware root 保持旧格式。

---

## 17.5 Shared calibration root 不受 Clip 开关影响

创建两个完全相同 config，仅：

```text
replacement.common_clip_enabled
```

分别为 true / false。

断言：

```python
layout_true.ann_training_calibration_dir
==
layout_false.ann_training_calibration_dir
```

并断言：

```python
layout_true.ann_training_prefix_dir
==
layout_false.ann_training_prefix_dir
```

但是：

```python
layout_true.root
!=
layout_false.root
```

这是本次 artifact 设计最关键测试之一。

---

## 17.6 Controller Phase

准备一个会产生超出 Clip 区间输出的 Phase state。

### true

```python
controller = SiteController(
    mode="phase",
    ...,
    common_clip_enabled=True,
)
```

断言：

```text
_modules 中包含 phase + clip
output == Clip(Phase(x))
```

### false

```python
controller = SiteController(
    mode="phase",
    ...,
    common_clip_enabled=False,
)
```

断言：

```text
_modules 中只有 phase
没有 clip
output == Phase(x)
```

---

## 17.7 Controller GIF

同上：

```text
true  -> GIF + Clip
false -> GIF only
```

---

## 17.8 Calibration 始终生成 Clip

分别用：

```text
common_clip_enabled=true
common_clip_enabled=false
```

调用 ANN-training calibration state materialization。

两者都必须存在：

```text
clip_state.pt
```

并且如果其他 calibration 输入完全相同，应断言对应 state hashes 完全相同。

至少检查：

```text
phase_state
gif_state
mtn_state
clip_state
```

不受开关影响。

---

## 17.9 SNN deployment 永不 Clip

无论 cfg：

```text
common_clip_enabled=true
common_clip_enabled=false
```

断言：

```text
deploy_phase -> Phase only
deploy_gif   -> GIF only
deploy_mtn   -> MTN only
```

controller `_modules` 中都不能加载：

```text
clip
```

---

## 17.10 training_result

aware true：

```text
ann_training_common_clip_enabled = true
ann_training_common_clip_applied = true
ann_training_common_clip_state_required = true
```

aware false：

```text
ann_training_common_clip_enabled = false
ann_training_common_clip_applied = false
ann_training_common_clip_state_required = true
```

---

## 17.11 conversion metadata

true/false 都保存：

```text
source_ann_common_clip_enabled
```

并与 cfg 校验一致。

同时：

```text
snn_clip_applied = false
```

始终不变。

---

## 17.12 evaluation metadata

final ANN/SNN results 中：

```text
ann_training_common_clip_enabled
```

与 run config 一致。

---

# 18. Artifact schema

本次：

- 没有改变 Phase/GIF/MTN state 文件的 runtime 格式；
- Site 3/4 repeat_kv 后的 Phase view 在当前 scalar τ 策略下不改变最终 τ；
- common Clip 参数通过新的 run root 隔离旧 aware checkpoint。

因此：

```text
SITE_STATE_FORMAT_VERSION = 5
CALIBRATION_MANIFEST_FORMAT_VERSION = 6
CONVERSION_METADATA_FORMAT_VERSION = 7
TEMPORAL_IMPLEMENTATION_VERSION = 3
```

可以保持不变。

不要为了本次修改再次无意义升 schema。

但是：

```text
conversion metadata
training_result
evaluation metadata
```

应新增 common Clip provenance 字段。

由于 aware run root 已变化，旧 aware descriptor 不会与新 run 混用。

---

# 19. 本次修改后的实验执行方式

如果要重新跑 Qwen3-1.7B：

## 使用 common Clip

矩阵：

```yaml
replacement:
  common_clip_enabled: true
```

重新 materialize 后：

```bash
CFG=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml
```

输出进入：

```text
phase_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_true/
seed42/
```

---

## 不使用 common Clip

矩阵：

```yaml
replacement:
  common_clip_enabled: false
```

重新 materialize 后使用同一 config 文件名。

输出进入：

```text
phase_aware/
lr1e-06_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_false/
seed42/
```

---

# 20. 哪些前处理需要重跑

如果只切换：

```text
replacement.common_clip_enabled
```

且其他配置、代码和已有 shared artifact 都没有变化：

可以复用：

```text
data manifest
rotation
Pre-finetuning Prefix
ANN-training calibration
```

因为 ANN-training calibration 始终生成相同的：

```text
Phase/GIF/MTN/Clip states
```

必须重新执行：

```text
ANN fine-tuning
conversion descriptor
final ANN evaluation
Phase/GIF/MTN SNN evaluation
```

对于你接下来要重新跑的：

```text
Qwen3-1.7B phase_aware -> Phase SNN
```

由于本次还修改了 Site 3/4 的 Phase statistics 结构，为了确保新代码对应 artifact 完整一致，建议第一次正式重跑时从：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training
```

开始重新生成一次 ANN-training calibration。

之后如果只是把：

```text
common_clip_enabled=true
```

切换成：

```text
false
```

做 Clip ablation，就可以复用这次新生成的 shared calibration。

---

# 21. 完成条件

全部满足才算完成：

1. Site 3/4 generic statistics 仍是 pre-repeat K/V；
2. Site 3/4 Phase statistics 使用 `repeat_kv()` 后的 heads；
3. Site 3/4 Phase statistics 仍排除 Prefix positions；
4. Site 3/4 runtime Prefix K/V 仍经过 neuron；
5. `replacement.common_clip_enabled` 只接受 bool；
6. phase_aware/gif_aware true/false 都合法；
7. vanilla/unaware resolved 后 common Clip 固定 false；
8. ANN-training calibration 无论 true/false 都生成 `clip_state.pt`；
9. true 时 ANN Phase/GIF replacement 使用 common Clip；
10. false 时 ANN Phase/GIF replacement 不加载、不执行 common Clip；
11. SNN deployment 无论 true/false 都不加载、不执行 common Clip；
12. aware run root 包含：
    ```text
    prefix_enabled_ture_common_clip_enabled_true
    ```
    或：
    ```text
    prefix_enabled_ture_common_clip_enabled_false
    ```
13. vanilla/unaware run root 不增加 common Clip suffix；
14. true/false 共用同一 Pre-finetuning Prefix；
15. true/false 共用同一 ANN-training calibration；
16. `training_result.json` 记录 common Clip 状态；
17. `ann/final/config.json` 记录 common Clip 状态；
18. conversion metadata 记录 source ANN common Clip 状态；
19. evaluation metadata 记录 ANN training common Clip 状态；
20. `实验执行总结.md` 不再声称四个 Prefix 开关完全独立；
21. `实验执行总结.md` 清楚说明 common Clip 的生成/使用边界；
22. README / AGENTS 与代码一致；
23. 全部测试通过。

最终执行：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```
