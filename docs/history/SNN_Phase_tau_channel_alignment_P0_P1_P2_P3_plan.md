# SNN Phase τ Channel 对齐与 Provenance / Evaluation 收尾修改方案

## 0. 目标与基准

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

当前基准 commit：

```text
2ca4993685e7e982ae884d7661ff4775f4ed3e42
```

本次只处理以下 4 类问题：

1. **P0：Phase τ calibration 的 channel 维度仍未与 SpikingLLM 普通 Phase 完全对齐；**
2. **P1：aware ANN training 的 Prefix / calibration provenance hash 应在训练开始前冻结，并在训练结束后验证磁盘工件未变化；**
3. **P2：ANN evaluation 的 activation-neuron operator count 与 calibration metadata 语义错误；**
4. **P3：schema/error message/文档残留旧表述。**

除本文明确要求外，不修改：

- 10-site topology；
- Site 9；
- Prefix K/V runtime 经过 Site 3/4；
- Site 3/4 calibration statistics 排除 Prefix；
- Phase `surrogate_slope=1.0`；
- Phase τ EMA factor `0.99`、FP32 accumulator；
- Embedding `x/T`；
- full Softmax 经过 Site 5；
- final RMSNorm global Phase neuron；
- aware ANN common Clip；
- SNN deployment 不使用 common Clip；
- aware conversion 复用 Pre-finetuning Prefix + ANN-training calibration；
- vanilla/unaware conversion 使用 Post-finetuning Prefix + Post-finetuning conversion calibration；
- temporal QK/PV、Softmax cumulative-difference、MLP Hadamard 主公式。

---

# 1. P0：Phase τ 的 channel 维度与 SpikingLLM 对齐

## 1.1 问题

当前 `snn2/stats.py`：

```python
values = activation.detach().reshape(-1, activation.shape[-1])
...
current_phase_abs_max = work.abs().amax(dim=0).float().cpu()
```

因此 Phase EMA 永远把 **runtime tensor 的最后一维** 当 channel。

这对 attention sites 不等价于 SpikingLLM。

SpikingLLM 普通 Phase 在 Q/K/V neuron 前会先把 attention head 展平成 hidden channel：

```text
Q: [T,B,H,Q,D]   -> [T,B,Q,H*D]
K: [T,B,Hkv,K,D] -> [T,B,K,Hkv*D]
V: [T,B,Hkv,K,D] -> [T,B,K,Hkv*D]
```

Softmax 则把：

```text
[T,B,H,Q,K] -> [T,B,Q*K,H]
```

之后才进入 Phase neuron/statistics。

因此当前项目需要：

> **保持 runtime neuron tensor layout 不变，只给 Phase EMA calibration 单独提供 SpikingLLM-aligned statistical view。**

不能把通用 activation statistics 一起 reshape，因为 GIF/MTN/Clip 等仍依赖当前 site layout，尤其 Site 5 需要保留 key-position 维度。

---

## 1.2 新增统一 Phase statistical-view helper

建议新增文件：

```text
snn2/phase_statistics.py
```

或者放在 `snn2/model_integration.py` 中；优先独立文件，便于测试。

定义：

```python
def phase_statistical_view(
    site_index: int,
    x: torch.Tensor,
) -> torch.Tensor:
    ...
```

只做 reshape/permute，不修改数值。

### Site 2

runtime：

```text
[B,H,Q,D]
```

Phase view：

```text
[B,Q,H*D]
```

实现：

```python
B, H, Q, D = x.shape
return x.permute(0, 2, 1, 3).contiguous().reshape(B, Q, H * D)
```

### Site 3

runtime：

```text
[B,Hkv,K,D]
```

Phase view：

```text
[B,K,Hkv*D]
```

实现：

```python
B, Hkv, K, D = x.shape
return x.permute(0, 2, 1, 3).contiguous().reshape(B, K, Hkv * D)
```

### Site 4

同 Site 3：

```python
B, Hkv, K, D = x.shape
return x.permute(0, 2, 1, 3).contiguous().reshape(B, K, Hkv * D)
```

### Site 5

runtime：

```text
[B,H,Q,K]
```

Phase view：

```text
[B,Q*K,H]
```

实现：

```python
B, H, Q, K = x.shape
return x.permute(0, 2, 3, 1).contiguous().reshape(B, Q * K, H)
```

### Site 6

runtime：

```text
[B,H,Q,D]
```

Phase view：

```text
[B,Q,H*D]
```

实现同 Site 2：

```python
B, H, Q, D = x.shape
return x.permute(0, 2, 1, 3).contiguous().reshape(B, Q, H * D)
```

### Site 1 / 7 / 8 / 9 / 10

保持：

```python
return x
```

因为这些 site 当前最后一维已经是对应 activation channel。

### 输入检查

对 Site 2/3/4/5/6：

```python
if x.ndim != 4:
    raise ValueError(...)
```

不要静默 reshape 非预期 tensor。

---

# 2. Phase EMA storage 与通用 statistics 解耦

修改：

```text
snn2/stats.py
```

当前 `SiteStatistics` 用同一个：

```text
channels
```

同时决定：

- value_min/max
- abs_max
- sum_abs/sum_sq
- saliency
- phase_ema_abs_max

P0 修正后，Phase channel 数可能和通用 channel 数不同。

例如：

```text
Site 2 generic channels = D
Site 2 Phase channels   = H*D

Site 5 generic channels = K
Site 5 Phase channels   = H
```

所以必须解耦。

---

## 2.1 `SiteStatistics` 新增独立 Phase channel metadata

建议字段：

```python
phase_channels: int | None
phase_ema_abs_max: torch.Tensor
phase_ema_updates: torch.Tensor
```

`create()` 时不要再按 generic `channels` 固定 Phase 数组长度。

例如：

```python
phase_channels=None,
phase_ema_abs_max=torch.empty(0, dtype=torch.float32),
phase_ema_updates=torch.empty(0, dtype=torch.int64),
```

新增：

```python
def _ensure_phase_channels(self, channels: int) -> None:
    channels = int(channels)
    if channels <= 0:
        raise ValueError(...)

    if self.phase_channels is None:
        self.phase_channels = channels
        self.phase_ema_abs_max = torch.zeros(
            channels, dtype=torch.float32
        )
        self.phase_ema_updates = torch.zeros(
            channels, dtype=torch.int64
        )
        return

    if self.phase_channels != channels:
        raise ValueError(
            f"Phase channel dimension changed from "
            f"{self.phase_channels} to {channels}"
        )
```

---

## 2.2 `update()` 接受独立 `phase_activation`

将：

```python
def update(self, activation):
```

改成：

```python
def update(
    self,
    activation: torch.Tensor,
    *,
    phase_activation: torch.Tensor | None = None,
) -> None:
```

### generic statistics

仍完全使用：

```python
activation
```

计算：

```text
value_min
value_max
abs_max
sum_abs
sum_sq
row_count
tensor_count
```

不要改变。

### Phase EMA

使用：

```python
phase_source = (
    activation
    if phase_activation is None
    else phase_activation
)
```

然后：

```python
phase_values = phase_source.detach().reshape(
    -1, phase_source.shape[-1]
)
phase_work = phase_values.float()

self._ensure_phase_channels(
    int(phase_values.shape[-1])
)

current_phase_abs_max = (
    phase_work.abs()
    .amax(dim=0)
    .float()
    .cpu()
)
```

继续使用：

```python
ema <- 0.99 * ema + 0.01 * current
```

首个值直接初始化。

注意：

- Phase EMA 仍必须 FP32；
- generic statistics 仍可 FP64；
- `phase_ema_updates` 长度按 `phase_channels`；
- `row_count` 仍是 generic statistics 的 row count，不要拿来描述 Phase view。

---

## 2.3 metadata

`state_dict()` / `summary()` 增加：

```text
phase_channels
phase_statistical_view
phase_statistical_view_version
```

建议：

```text
phase_statistical_view = spikingllm_identity_input_layout
phase_statistical_view_version = 1
```

每个 site summary 至少记录：

```text
channels
phase_channels
```

方便检查：

```text
Site 2:
channels = head_dim
phase_channels = num_heads * head_dim

Site 5:
channels = key_position_capacity
phase_channels = num_heads
```

---

# 3. `StatisticsStore` 支持独立 Phase view

修改：

```text
snn2/stats.py
```

将：

```python
def update(self, layer_index, site_index, activation):
```

改为：

```python
def update(
    self,
    layer_index: int,
    site_index: int,
    activation: torch.Tensor,
    *,
    phase_activation: torch.Tensor | None = None,
) -> None:
```

最终：

```python
self.items[key].update(
    activation,
    phase_activation=phase_activation,
)
```

`update_global()` 可以继续默认：

```python
phase_activation=None
```

因为 final RMSNorm：

```text
[B,L,H]
```

本身已经是正确 Phase channel layout。

---

# 4. `SiteController` 支持 Phase statistical view

修改：

```text
snn2/controller.py
```

## 4.1 `record_activation()`

改成：

```python
def record_activation(
    self,
    layer_index: int,
    site_index: int,
    x: torch.Tensor,
    *,
    phase_activation: torch.Tensor | None = None,
) -> None:
    if self.mode == "collect":
        self.statistics.update(
            layer_index,
            site_index,
            x,
            phase_activation=phase_activation,
        )
```

---

## 4.2 `apply()` 的 collect 分支

增加参数：

```python
def apply(
    self,
    layer_index: int,
    site_index: int,
    x: torch.Tensor,
    *,
    phase_activation: torch.Tensor | None = None,
) -> torch.Tensor:
```

collect：

```python
if self.mode == "collect":
    self.statistics.update(
        layer_index,
        site_index,
        x,
        phase_activation=phase_activation,
    )
    return x
```

非 collect 模式：

```text
phase/gif/deploy_*
```

忽略 `phase_activation`，runtime neuron 仍对原始 `x` 执行。

这样可以保证：

> Phase statistical reshape 只影响 calibration，不影响 ANN-aware replacement 或 SNN deployment 的 tensor layout。

---

# 5. `model_integration.py` 在 Site 2–6 传入 Phase view

修改：

```text
snn2/model_integration.py
```

导入：

```python
from .phase_statistics import phase_statistical_view
```

---

## 5.1 Site 2

原：

```python
query = controller.apply(layer_index, 2, query)
```

改：

```python
query = controller.apply(
    layer_index,
    2,
    query,
    phase_activation=(
        phase_statistical_view(2, query)
        if controller.mode == "collect"
        else None
    ),
)
```

---

## 5.2 Site 3 / 4：继续排除 Prefix statistics

当前已经：

```python
if controller.mode == "collect" and past_length:
    controller.record_activation(
        layer_index, 3, key[..., past_length:, :]
    )
    controller.record_activation(
        layer_index, 4, value[..., past_length:, :]
    )
```

改成：

```python
if controller.mode == "collect" and past_length:
    current_key = key[..., past_length:, :]
    current_value = value[..., past_length:, :]

    controller.record_activation(
        layer_index,
        3,
        current_key,
        phase_activation=phase_statistical_view(
            3, current_key
        ),
    )
    controller.record_activation(
        layer_index,
        4,
        current_value,
        phase_activation=phase_statistical_view(
            4, current_value
        ),
    )
else:
    key = controller.apply(
        layer_index,
        3,
        key,
        phase_activation=(
            phase_statistical_view(3, key)
            if controller.mode == "collect"
            else None
        ),
    )
    value = controller.apply(
        layer_index,
        4,
        value,
        phase_activation=(
            phase_statistical_view(4, value)
            if controller.mode == "collect"
            else None
        ),
    )
```

runtime `phase/gif/deploy_*` 仍对完整 Prefix+Current K/V 执行 Site 3/4 neuron。

---

## 5.3 Site 5

原：

```python
weights = controller.apply(
    layer_index, 5, weights
)
```

改：

```python
weights = controller.apply(
    layer_index,
    5,
    weights,
    phase_activation=(
        phase_statistical_view(5, weights)
        if controller.mode == "collect"
        else None
    ),
)
```

**generic Site 5 statistics 继续使用 `[B,H,Q,K]` 的最后一维 K。**

只有 Phase EMA 使用：

```text
[B,Q*K,H]
```

---

## 5.4 Site 6

原：

```python
output = controller.apply(
    layer_index, 6, output
)
```

改：

```python
output = controller.apply(
    layer_index,
    6,
    output,
    phase_activation=(
        phase_statistical_view(6, output)
        if controller.mode == "collect"
        else None
    ),
)
```

---

## 5.5 Site 1/7/8/9/10

不需要显式传 `phase_activation`。

默认：

```text
phase_activation = activation
```

即可。

---

# 6. Phase τ 最终仍按 SpikingLLM 做“先逐 channel EMA，再全 channel max”

修改：

```text
snn2/calibration.py
```

当前：

```python
phase_stat = statistics["phase_ema_abs_max"]
tau = _group_reduce(...)
```

普通 Phase baseline 应明确：

```text
逐 forward：
每个 Phase channel 计算 abs-max
↓
每个 channel 独立 EMA
↓
EMA 完成后，对全部 Phase channels 取 max
↓
得到一个 scalar τ
```

也就是：

```python
tau = phase_stat.float().amax().reshape(1)
```

### 重要

普通 SpikingLLM Phase 的 `FSNeuron.find_activation_quant_param()` 最终得到的是 **一个 scalar τ**。

因此不要让通用 `calibration.group_size` 改变 Phase τ 的 grouping 语义。

建议 `build_phase_state()`：

```python
tau = phase_stat.float().amax().reshape(1)

return {
    ...
    "group_size": -1,
    "tau": tau,
    ...
}
```

这样 runtime `PhaseSurrogate` 使用 scalar τ 广播到任意 runtime layout。

GIF / MTN / Clip 的 group size 逻辑保持当前实现，不受影响。

如果项目中此前允许 Phase 使用正 `group_size`，本次普通 Phase baseline 直接废弃该行为，不再作为 main experiment 协议。

---

# 7. Phase state 增加 statistical-view metadata

`build_phase_state()` 增加：

```text
tau_channel_policy = spikingllm_flatten_attention_heads_before_channel_ema
tau_reduction_policy = per_channel_ema_then_global_max
phase_statistical_view_version = 1
```

Phase state validator 同步要求这些字段。

建议在：

```text
snn2/neurons.py
```

`PhaseSurrogate.__init__()` 中检查：

```python
state["tau_channel_policy"]
state["tau_reduction_policy"]
state["phase_statistical_view_version"]
```

防止旧 Phase state 静默进入新 runtime。

---

# 8. Artifact schema 再升级

P0 会改变 Phase τ 的实际数值，因此旧 calibration state 不可复用。

修改：

```text
snn2/temporal_ops.py
```

建议：

```text
SITE_STATE_FORMAT_VERSION:
4 -> 5

CALIBRATION_MANIFEST_FORMAT_VERSION:
5 -> 6

CONVERSION_METADATA_FORMAT_VERSION:
6 -> 7
```

保持：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 3
```

因为跨层 temporal arithmetic 没变。

`temporal_policy_metadata()` 增加：

```text
phase_statistical_view = spikingllm_identity_input_layout
phase_statistical_view_version = 1
phase_tau_reduction_policy = per_channel_ema_then_global_max
```

所有：

```text
state_validation.py
conversion.py
verify_artifacts.py
tests
README
实验执行总结.md
```

同步新版本。

---

# 9. P1：ANN training 开始前冻结 Prefix / calibration provenance

修改：

```text
snn2/training.py
```

当前 hash 是训练结束后从磁盘读取。

应改为：

```text
训练开始前 capture
↓
训练
↓
训练结束后 re-hash
↓
必须完全一致
↓
training_result 保存训练开始前 capture
```

---

## 9.1 新增 helper

建议：

```python
def capture_training_artifact_provenance(
    cfg,
    layout,
    *,
    prefix_ids,
) -> dict[str, Any]:
    ...
```

### Prefix

如果：

```python
training_prefix_enabled(cfg)
```

则记录：

```text
ann_training_prefix_root
ann_training_prefix_state_sha256
ann_training_prefix_kv_sha256
ann_training_prefix_token_ids
```

规则：

```text
prefix_state.json 必须存在
非空 Prefix -> prefixed_key_values.pt 必须存在
空 Prefix   -> KV hash = None
```

### aware calibration

如果：

```python
is_aware_ann_mode(cfg)
```

记录：

```text
ann_training_calibration_root
ann_training_calibration_manifest_sha256
```

并在 capture 前运行：

```python
validate_site_state_bundle(
    layout.ann_training_site_dir,
    require_clip=True,
)
```

---

## 9.2 capture 必须发生在 `trainer.train()` 前

推荐顺序：

```text
load model
validate calibration
load Prefix ids/KV
capture provenance
install Prefix/model integration
tokenize
construct Trainer
trainer.train()
```

只要保证：

```text
capture provenance
```

明确在：

```python
trainer.train(...)
```

之前即可。

---

## 9.3 训练结束后验证磁盘工件未变化

新增：

```python
def verify_training_artifact_provenance_unchanged(
    captured,
    cfg,
    layout,
) -> None:
    ...
```

重新计算所有相关 hash。

任何变化直接：

```python
raise RuntimeError(
    "ANN-training Prefix/calibration artifacts changed during training"
)
```

不要继续保存一个看似合法的 `training_result.json`。

---

## 9.4 `training_result.json`

不要训练结束后重新生成 provenance 值。

直接：

```python
metrics.update(captured_provenance)
```

确保记录的是：

> **训练开始前实际冻结的那一套工件。**

---

# 10. P2：ANN evaluation operator count 修正为 0

修改：

```text
snn2/evaluation.py
```

当前：

```python
base = num_hidden_layers * SITE_COUNT
if neuron == "phase":
    base += 1
return base
```

改成：

```python
def activation_neuron_operators_per_temporal_forward(
    *,
    num_hidden_layers: int,
    neuron: str,
) -> int:
    if neuron == "ann":
        return 0

    base = int(num_hidden_layers) * SITE_COUNT

    if neuron == "phase":
        return base + 1

    if neuron in {"gif", "mtn"}:
        return base

    raise ValueError(
        f"Unknown neuron: {neuron}"
    )
```

结果：

```text
ANN   -> 0
Phase -> L*10 + 1
GIF   -> L*10
MTN   -> L*10
```

---

# 11. P2：ANN evaluation 不再伪装成“使用 conversion calibration”

修改：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

对于：

```python
args.neuron == "ann"
```

以下 metadata：

```text
calibration_source_stage
calibration_root
```

必须为：

```text
None
```

以下字段：

```text
reused_ann_training_artifacts
post_finetuning_recalibration
```

必须为：

```text
False
```

因为 final ANN evaluation 不加载 neuron calibration。

### Prefix metadata 仍保留

ANN final evaluation 若启用 Prefix，仍应记录：

```text
prefix_enabled
prefix_source_stage
prefix_root
```

aware：

```text
prefix_source_stage = pre_finetuning
```

vanilla/unaware：

```text
prefix_source_stage = post_finetuning
```

不要因为 calibration metadata 清空而删除 Prefix provenance。

---

# 12. P3：修正 conversion schema 错误提示

修改：

```text
snn2/conversion.py
```

当前：

```python
if metadata.get("format_version") != CONVERSION_METADATA_FORMAT_VERSION:
    raise ValueError(
        "... format v5 is required"
    )
```

不要再硬编码旧版本。

改成：

```python
raise ValueError(
    f"{path} uses a legacy conversion schema; "
    f"format v{CONVERSION_METADATA_FORMAT_VERSION} is required"
)
```

本次升级后会自动显示：

```text
format v7 is required
```

同类 error message 全部搜索并去掉硬编码：

```text
v4
v5
v6
```

优先使用对应常量。

---

# 13. P3：修正 `实验执行总结.md` 的 clip-free 路径

当前 Post-finetuning conversion calibration 示例仍写：

```text
layer_xxx/site_xx_name/{phase,gif,mtn,clip}_state.pt
```

这是错误的。

改成：

```text
layer_xxx/site_xx_name/
├── statistics.pt
├── phase_state.pt
├── gif_state.pt
└── mtn_state.pt
```

明确：

```text
Post-finetuning conversion calibration 不生成 clip_state.pt。
```

Aware ANN-training calibration 才包含：

```text
clip_state.pt
```

---

# 14. `实验执行总结.md` 增加 Phase τ channel policy

在 Temporal/Phase 对齐章节增加简短说明：

```text
Phase τ calibration 不直接把 runtime tensor 的最后一维统一当 channel。
对于 attention Site 2/3/4/5/6，Phase EMA 使用与 SpikingLLM
Identity 输入一致的 statistical view：
Q/K/V/PV 先展平 attention heads；
Softmax 使用 head 作为 channel。
普通 value/GIF/MTN/Clip statistics 仍使用原 site tensor layout。
每个 Phase channel 独立执行 FP32 EMA，最后对所有 Phase channels
取 global max 得到 scalar τ。
```

并写明：

```text
Phase state 的 group_size 固定为 -1；
通用 calibration.group_size 不改变普通 Phase baseline 的 τ grouping。
```

---

# 15. README / AGENTS 必要同步

## README.md

增加一句：

```text
Phase τ 使用 SpikingLLM-aligned channel view：
attention heads 在 Phase statistics 中按参考实现 reshape，
逐 channel FP32 EMA 后 global max 得到 scalar τ；
该 reshape 只用于 Phase calibration，不改变 runtime tensor layout。
```

## AGENTS.md

增加规则：

```text
Phase calibration statistics 与 generic site statistics 必须解耦。
Site 2/3/4/5/6 的 Phase EMA 必须使用 SpikingLLM-aligned statistical view；
不得为了 Phase τ 对齐而改变 GIF/MTN/Clip statistics 或 runtime neuron layout。
```

---

# 16. 测试

至少修改/新增：

```text
tests/test_statistics.py
tests/test_calibration_profiles.py
tests/test_temporal_model_integration.py
tests/test_neurons.py
tests/test_evaluation_paths.py
tests/test_conversion_metadata.py
tests/test_post_finetuning_protocol.py
```

---

## 16.1 Phase statistical view shape

新增单元测试：

```text
Site 2:
[B=2,H=4,Q=3,D=5]
-> [2,3,20]

Site 3:
[2,Hkv=2,K=7,D=5]
-> [2,7,10]

Site 4:
同 Site 3

Site 5:
[2,H=4,Q=3,K=7]
-> [2,21,4]

Site 6:
[2,4,3,5]
-> [2,3,20]
```

并验证：

```text
reshape 前后元素完全对应；
不发生求和或数值变化。
```

---

## 16.2 generic channels 与 phase_channels 独立

例如 Site 5：

generic activation：

```text
[B,H,Q,K] = [1,4,3,7]
```

Phase view：

```text
[1,21,4]
```

断言：

```text
stats.channels == 7
stats.phase_channels == 4
stats.value_min.numel() == 7
stats.phase_ema_abs_max.numel() == 4
```

---

## 16.3 验证“EMA(max head)”与“max(EMA head)”不会被混淆

构造两次 update，使两个 head 在不同 forward 中交替成为最大值。

例如两个 Phase channels：

```text
forward 1: [100, 0]
forward 2: [0, 100]
```

正确：

```text
EMA channel 0 = 99
EMA channel 1 = 1
tau = 99
```

错误旧逻辑：

```text
每个 forward 先跨 head max
-> [100,100]
EMA -> 100
```

测试必须断言：

```text
tau == 99
```

按 FP32 tolerance 比较。

这是本次 P0 最关键 regression test。

---

## 16.4 Site 3/4 Prefix exclusion 仍有效

构造：

```text
Prefix length = P
Current length = L
```

断言：

```text
generic Site 3/4 row_count
只对应 Current

Phase Site 3/4 statistical view
也只由 Current K/V 构造
```

但 runtime phase/deploy：

```text
仍对 Prefix + Current 完整 K/V
执行 Site 3/4 neuron
```

---

## 16.5 Phase state 是 scalar τ

断言：

```python
phase_state["tau"].numel() == 1
phase_state["group_size"] == -1
phase_state["tau_accumulator_dtype"] == "float32"
phase_state["tau_reduction_policy"] == "per_channel_ema_then_global_max"
```

---

## 16.6 P1 provenance

新增测试：

### 正常情况

```text
capture
-> trainer mock
-> artifacts unchanged
-> verification passes
-> training_result 使用 captured hashes
```

### Prefix 中途变化

训练 mock 过程中改写：

```text
prefix_state.json
```

训练结束验证必须失败。

### calibration 中途变化

改写：

```text
calibration_state_manifest.json
```

必须失败。

### 非空 Prefix KV 变化

改写：

```text
prefixed_key_values.pt
```

必须失败。

---

## 16.7 operator count

断言：

```text
L=28

ann   -> 0
phase -> 281
gif   -> 280
mtn   -> 280
```

未知 neuron 抛异常。

---

## 16.8 ANN evaluation metadata

final ANN：

```text
calibration_source_stage is None
calibration_root is None
reused_ann_training_artifacts is False
post_finetuning_recalibration is False
```

但：

```text
prefix_source_stage
```

仍按 mode 正确。

---

# 17. 重跑边界

本次 P0 会改变 Phase τ，因此旧 calibration bundle 全部视为过期。

## phase_aware / gif_aware

必须重新：

```text
ANN-training calibration
↓
aware ANN fine-tuning
↓
conversion descriptor
↓
SNN evaluation
```

Pre-finetuning Prefix 如果内容/hash 未变可以复用。

不执行 aware Post-finetuning Prefix / calibration。

## vanilla / unaware

ANN checkpoint 可以复用。

但要重新：

```text
Post-finetuning conversion calibration
↓
conversion descriptor
↓
SNN evaluation
```

Post-finetuning Prefix 若 final ANN checkpoint 未变可复用。

---

# 18. 完成条件

全部满足才算完成：

1. Phase statistics 与 generic statistics 分离；
2. Site 2 Q Phase channel = `H*D`；
3. Site 3 K Phase channel = `Hkv*D`；
4. Site 4 V Phase channel = `Hkv*D`；
5. Site 5 Softmax Phase channel = `H`；
6. Site 6 PV output Phase channel = `H*D`；
7. Site 1/7/8/9/10 保持当前 channel layout；
8. Site 3/4 Phase calibration 仍排除 Prefix；
9. runtime Site 3/4 仍处理完整 Prefix+Current K/V；
10. Phase EMA 仍为 FP32、factor 0.99；
11. Phase τ 为“逐 channel EMA 后 global max”的 scalar；
12. Phase state `group_size=-1`；
13. GIF/MTN/Clip statistics 未因 P0 reshape；
14. aware training provenance 在训练前冻结；
15. 训练结束验证 Prefix/calibration hashes 未变化；
16. ANN operator count = 0；
17. ANN evaluation calibration metadata = None/False；
18. Phase operator count = `10L+1`；
19. GIF/MTN operator count = `10L`；
20. conversion schema error 不再硬编码旧版本；
21. Post-finetuning calibration 文档不再出现 `clip_state.pt`；
22. artifact schema 更新；
23. 全部测试通过。

执行：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```

---

# 19. 修改后建议的最小人工检查

重新跑 Qwen3-1.7B `phase_aware` ANN-training calibration 后，随机检查：

```text
layer_000/site_02_q_post_rope_r3/statistics.pt
layer_000/site_03_k_post_rope_r3/statistics.pt
layer_000/site_04_v_projection_r2/statistics.pt
layer_000/site_05_post_spiking_softmax/statistics.pt
layer_000/site_06_post_attention_value_dot_r2/statistics.pt
```

以 Qwen3-1.7B 为例，应看到类似：

```text
Site 2:
generic channels = head_dim
phase_channels = num_attention_heads * head_dim

Site 3:
generic channels = head_dim
phase_channels = num_key_value_heads * head_dim

Site 4:
同 Site 3

Site 5:
generic channels = max key-position capacity
phase_channels = num_attention_heads

Site 6:
generic channels = head_dim
phase_channels = num_attention_heads * head_dim
```

并检查对应：

```text
phase_state.pt
```

满足：

```text
tau.shape == [1]
tau.dtype == float32
group_size == -1
tau_calibration == spikingllm_ema_channel_abs_max
tau_ema_factor == 0.99
tau_accumulator_dtype == float32
tau_reduction_policy == per_channel_ema_then_global_max
```

确认这些都正确后，再重新执行 `phase_aware` ANN fine-tuning。
