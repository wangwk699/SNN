# SNN 项目修改方案：Final RMSNorm Phase/MTN 对齐 + Phase/MTN EMA 参数统计与 Clamp

> 目标仓库：`https://github.com/wangwk699/SNN`，以当前 `main` 分支为基线。  
> 本文档用于部署在服务器上的 Codex **在没有任何额外上下文的情况下直接完成代码修改**。  
> 本次只修改 SNN 项目，不修改 `wangwk699/SparseLLM`；SparseLLM 仅作为 MTN 参数统计规则与 Final RMSNorm MTN topology 的参考。

---

## 0. 本次修改的最终目标

本次修改包含三个相互关联的目标：

1. **统一 Final RMSNorm → LM Head 之间的 neuron topology**
   - GIF：ANN aware training/evaluation 与 GIF SNN deployment 都不加 GIF Neuron。
   - Phase：Phase-aware ANN fine-tuning / final ANN evaluation 加 `PhaseSurrogate`；Phase SNN deployment 加 temporal Phase Neuron。
   - MTN：MTN SNN deployment 加 temporal MTN Neuron。
   - Final RMSNorm 全局位置在 aware ANN 中 **永远不使用 Clip**。
   - 该位置是全局位置，不计入每层 10 个 activation replacement sites。

2. **将 MTN parameter calibration 改为与当前 Phase τ、SparseLLM MTN 相同的 EMA channel abs-max 规则**
   - 逐 channel/head-channel 统计 activation absolute maximum。
   - calibration 样本按既有顺序执行 factor `0.99` 的有序 EMA。
   - EMA 完成后再按本项目现有 group granularity 在 group 内取 max。
   - MTN 不再使用当前 `value_min/value_max -> absolute -> 2*absolute` 的统计规则。

3. **Phase 与 MTN 参数在 state materialization 时加入固定 clamp**
   - Phase：
     `tau = clamp(group_max(EMA_channel_abs_max), 5e-4, 1e4)`
   - MTN：
     `base_scale = clamp(2 * group_max(EMA_channel_abs_max), 5e-4, 1e4)`
   - `phase_state.pt` / `mtn_state.pt` 直接保存 clamp 后的 tensor。
   - 不要把未 clamp 的 tensor 保存为运行时参数。
   - 参数 state 是 calibration-derived frozen state，因此不需要像 SparseLLM trainable parameter 那样在每次 forward 再做 STE clamp；应在 state validation 中严格验证保存值和 metadata。

---

# 1. 必须保持不变的行为

以下内容是本次已经明确决定 **不得修改** 的部分。

## 1.1 Embedding temporal encoding 保持现状

当前 deployment policy：

```text
uniform_embedding_divide_by_T
```

即 embedding 在 T 个 timestep 上均匀拆分：

```text
[E/T, E/T, ..., E/T]
```

不要改成 SparseLLM 的：

```text
[E, 0, ..., 0]
```

不要修改 `snn2/model_integration.py` 当前 embedding temporal hook 的数值语义。

---

## 1.2 Prefix K/V temporal decomposition 保持现状

当前 policy：

```text
uniform_kv_divide_by_T
```

即 Prefix K/V 在 T 个 timestep 上均匀拆分。

不要改成 SparseLLM 的 t0 impulse：

```text
[KV, 0, ..., 0]
```

不要修改 `snn2/prefix_cache.py` 当前 Prefix temporal decomposition 的数值语义。

---

## 1.3 MTN / Phase 的 group granularity 保持本项目现状

`calibration.group_size` 及当前 site-specific logical layout 继续有效：

- Site 2：原生 per-head `[B,H,L,D]` layout，在每个 head 的 D 内 grouping。
- Site 3/4：`repeat_kv()` 后以 query heads 为 logical per-head grouping。
- Site 5：
  - Phase τ `[H,1]`
  - MTN `base_scale` `[H,1]`
  - 忽略全局 G。
- Site 6：head merge 后 ordinary last-dim grouping。
- 其他普通 site：当前 last-dim grouping。
- Final RMSNorm：ordinary last-dim grouping，受同一个 `calibration.group_size` 控制。

**只改 parameter statistics source，不改 group 划分规则。**

---

## 1.4 每层 activation replacement site 数仍然是 10

不要增加第 11 个 site。

Final RMSNorm 后的 Phase/MTN 是：

```text
_global/final_rmsnorm
```

下的 global replacement position，不属于：

```text
layer_xxx/site_xx
```

因此：

```text
calibration.expected_sites_per_layer
```

仍然保持 10。

---

# 2. Final RMSNorm 的最终行为矩阵

这是本次最重要的 topology 约束，必须严格实现。

| 运行模式 | Final RMSNorm 本身 | Final RMSNorm 后、LM Head 前 | Clip |
|---|---|---|---|
| Base ANN | ordinary RMSNorm | identity | 无 |
| rotated-pre-finetuning ANN diagnostic | ordinary RMSNorm | identity | 无 |
| vanilla ANN | ordinary RMSNorm | identity | 无 |
| unaware ANN | ordinary RMSNorm | identity | 无 |
| GIF-aware ANN fine-tuning | ordinary RMSNorm | **identity** | **无** |
| GIF-aware final ANN evaluation | ordinary RMSNorm | **identity** | **无** |
| Phase-aware ANN fine-tuning | ordinary RMSNorm | **PhaseSurrogate** | **无** |
| Phase-aware final ANN evaluation | ordinary RMSNorm | **PhaseSurrogate** | **无** |
| GIF SNN deployment | **Temporal RMSNorm** | **identity** | 无 |
| Phase SNN deployment | **Temporal RMSNorm** | **temporal Phase Neuron** | 无 |
| MTN SNN deployment | **Temporal RMSNorm** | **temporal MTN Neuron** | 无 |

必须特别注意：

```text
Phase-aware ANN:
ordinary Final RMSNorm
    -> PhaseSurrogate
    -> LM Head
```

而不是 temporal RMSNorm。

只有 `deploy_*` 模式才启用 temporal RMSNorm。

---

# 3. Final RMSNorm state 的来源

Final RMSNorm 不单独运行一套特殊 calibration 数据。

继续使用当前 calibration stage 生命周期与同一份 Final RMSNorm statistics。

## 3.1 ANN-training calibration

ANN-training calibration 必须产生：

```text
_global/final_rmsnorm/statistics.pt
_global/final_rmsnorm/statistics_summary.json
_global/final_rmsnorm/phase_state.pt
_global/final_rmsnorm/mtn_state.pt
```

不得产生：

```text
_global/final_rmsnorm/gif_state.pt
_global/final_rmsnorm/clip_state.pt
```

Phase-aware ANN fine-tuning 和 final ANN evaluation 使用 ANN-training calibration 中的：

```text
_global/final_rmsnorm/phase_state.pt
```

不能使用 post-finetuning state，因为训练开始前还不存在 final ANN checkpoint。

---

## 3.2 Post-finetuning conversion calibration

post-finetuning calibration 同样产生：

```text
_global/final_rmsnorm/statistics.pt
_global/final_rmsnorm/phase_state.pt
_global/final_rmsnorm/mtn_state.pt
```

Phase / MTN SNN conversion/deployment 使用当前 conversion bundle 对应的 global state：

```text
Phase:
_global/final_rmsnorm/phase_state.pt

MTN:
_global/final_rmsnorm/mtn_state.pt
```

GIF SNN 不加载任何 Final RMSNorm GIF state。

---

## 3.3 Vanilla analysis calibration

如果当前代码仍要求 vanilla analysis calibration 保存 global Final RMSNorm statistics，则保持现有 calibration 统计流程；是否 materialize Phase/MTN state 应与当前 Stage A state materialization 统一，不要引入另一套特殊规则。

analysis-only artifact 仍然不得被误用为 training/conversion state。

---

# 4. Phase / MTN parameter 的统一统计公式

本节是本次 calibration 更改的唯一数值标准。

设 calibration 第 `i` 次 update 中，第 `c` 个 logical channel/head-channel 的 activation 为：

```text
x_c^(i)
```

首先计算当前 sample/update 的 channel absolute maximum：

```text
m_c^(i) = max |x_c^(i)|
```

第一条 observation：

```text
e_c^(1) = m_c^(1)
```

后续有序 EMA：

```text
e_c^(i) = 0.99 * e_c^(i-1) + 0.01 * m_c^(i)
```

EMA accumulator 必须：

```text
dtype = float32
factor = 0.99
```

calibration 继续保持当前：

```text
batch_size = 1
single process
```

因为 EMA 是 order-dependent。

---

# 5. Phase τ 的新定义

在完成逐 channel/head-channel EMA 后，按照本项目当前 grouping rule，在每个 group 内取 max：

```text
tau_raw_g = max_{c in group g} e_c
```

然后固定 clamp：

```text
tau_g = clamp(tau_raw_g, min=5e-4, max=1e4)
```

最终保存：

```python
state["tau"] = tau
```

其中保存的 `tau` **必须已经是 clamp 后的 tensor**。

不允许保存 raw τ 后再仅依靠 runtime clamp。

---

## 5.1 Site 5

Site 5 继续使用每个 head 一个 scalar：

```text
tau_raw[h] = EMA_abs_max[h]
tau[h] = clamp(tau_raw[h], 5e-4, 1e4)
shape = [H, 1]
```

不要因为加入 clamp 改变 Site 5 grouping policy。

---

# 6. MTN base_scale 的新定义

MTN 必须使用与 Phase τ **同一个 EMA statistics source**。

普通 grouped site：

```text
ema_group_max_g = max_{c in group g} e_c
```

然后：

```text
base_scale_raw_g = 2 * ema_group_max_g
```

最后：

```text
base_scale_g = clamp(base_scale_raw_g, min=5e-4, max=1e4)
```

保存：

```python
state["base_scale"] = base_scale
```

即：

```text
base_scale =
    clamp(
        2 * group_max(EMA_0.99(channel_abs_max)),
        5e-4,
        1e4
    )
```

---

## 6.1 删除当前 MTN extrema-based initialization

当前 `snn2/calibration.py::build_site_states()` 中类似下面的逻辑：

```python
minimum = ...
maximum = ...
absolute = torch.maximum(minimum.abs(), maximum.abs()).clamp_min(1e-8)

mtn_state = {
    ...
    "base_scale": (2.0 * absolute).float(),
}
```

**不得再用于 MTN base_scale。**

`minimum/maximum` 仍可继续服务 GIF qparams、其他 statistics/summary，不要因为 MTN 不再使用而误删 GIF 所需逻辑。

MTN 必须改为读取当前 Phase 已使用的 EMA channel abs-max statistics。

---

## 6.2 Site 5

保持：

```text
base_scale shape = [H, 1]
```

新规则：

```text
base_scale[h] =
    clamp(2 * ema_abs_max[h], 5e-4, 1e4)
```

不使用全局 G。

---

# 7. Clamp 常量与 metadata

不要把 clamp magic number 分散写在多个文件。

建议定义统一常量，例如：

```python
NEURON_PARAMETER_CLAMP_MIN = 5e-4
NEURON_PARAMETER_CLAMP_MAX = 1e4
NEURON_PARAMETER_CLAMP_POLICY = "materialize_clamped_state_5e-4_to_1e4"
```

名字可按项目风格调整，但必须有 **唯一 source of truth**。

Phase state 至少应记录类似：

```text
tau_calibration = ema_channel_abs_max_then_group_max
tau_ema_factor = 0.99
tau_accumulator_dtype = float32
tau_channel_policy = native_site_layout_per_channel
tau_reduction_policy = within_group_max_after_channel_ema
tau_clamp_min = 5e-4
tau_clamp_max = 1e4
tau_clamp_policy = materialize_clamped_state
```

MTN state 至少应明确记录：

```text
base_scale_calibration = ema_channel_abs_max_then_group_max_then_times_2
base_scale_ema_factor = 0.99
base_scale_accumulator_dtype = float32
base_scale_channel_policy = native_site_layout_per_channel
base_scale_reduction_policy = within_group_max_after_channel_ema
base_scale_multiplier = 2.0
base_scale_clamp_min = 5e-4
base_scale_clamp_max = 1e4
base_scale_clamp_policy = materialize_clamped_state
```

不要只靠代码隐式表达 MTN 的新统计规则。

---

# 8. statistics schema：将 EMA 明确声明为 Phase/MTN shared calibration statistics

当前 statistics 中已有：

```text
phase_ema_abs_max
phase_ema_updates
```

这份 tensor 的数学统计已经正好满足新 MTN 所需数据。

为了避免 artifact metadata 继续误导为 “Phase-only statistics”，本次应把统计 provenance 明确扩展为 **Phase/MTN shared EMA calibration**。

推荐做法：

1. 将字段重命名为更通用的名字，例如：

```text
parameter_ema_abs_max
parameter_ema_updates
```

2. metadata 改为 shared policy，例如：

```text
parameter_calibration = ema_channel_abs_max_then_group_max
parameter_ema_factor = 0.99
parameter_accumulator_dtype = float32
parameter_channel_policy = native_site_layout_per_channel
parameter_reduction_policy = within_group_max_after_channel_ema
parameter_consumers = ["phase_tau", "mtn_base_scale"]
```

3. `Phase` 和 `MTN` state builder 都使用同一 shared statistics tensor。

如果为减少 diff 而保留内部 tensor 名 `phase_ema_abs_max`，也必须至少在 statistics metadata 中显式加入 MTN consumer 及 MTN calibration provenance；**不能让新 MTN state 的来源在 metadata 中不可追溯。**

由于本项目当前明确要求拒绝旧 schema，优先推荐执行通用字段重命名并 bump statistics schema，而不是保留 legacy fallback。

---

# 9. `snn2/calibration.py` 必须修改的逻辑

## 9.1 提取统一 EMA group reduction helper

建议增加一个 helper，用于 Phase / MTN 共用，例如：

```python
def grouped_ema_abs_max(statistics, cfg):
    ...
```

行为：

1. validation statistics schema。
2. 读取 shared EMA tensor。
3. 读取 `calibration.group_size`。
4. 调用当前 `_layout_metadata()`。
5. Site 5：
   - reshape `[H] -> [H,1]`
6. 其他 site：
   - `group_reduce_last_dim(..., reduction="max")`
7. 返回：
   - grouped EMA tensor
   - layout metadata

不要让 Phase 与 MTN 分别复制一套 grouping 实现，否则后续容易漂移。

---

## 9.2 `build_phase_state()`

改为：

```text
grouped_ema
    -> clamp(5e-4, 1e4)
    -> phase_state["tau"]
```

必须写入 clamp metadata。

---

## 9.3 新增独立 `build_mtn_state()`

不要继续在 `build_site_states()` 尾部直接用 extrema 构造字典。

新增类似：

```python
def build_mtn_state(statistics, cfg):
    grouped_ema, layout = ...
    base_scale = (2.0 * grouped_ema).clamp(
        min=NEURON_PARAMETER_CLAMP_MIN,
        max=NEURON_PARAMETER_CLAMP_MAX,
    )
    return {...}
```

然后 ordinary per-site 与 global Final RMSNorm 都统一调用该函数。

这样可以保证：

```text
ordinary site MTN
global Final RMSNorm MTN
```

使用完全相同的 parameter semantics。

---

## 9.4 `build_site_states()`

改为：

```python
phase_state = build_phase_state(statistics, cfg)
mtn_state = build_mtn_state(statistics, cfg)
```

GIF min/max qparam 路径保持原样。

---

# 10. Final RMSNorm state materialization

当前 `materialize_calibration_states()` 只对：

```text
_global/final_rmsnorm/statistics.pt
```

生成：

```text
phase_state.pt
```

必须改为同时生成：

```text
phase_state.pt
mtn_state.pt
```

伪代码：

```python
final_statistics = torch.load(...)

final_phase_state = build_phase_state(final_statistics, cfg)
final_mtn_state = build_mtn_state(final_statistics, cfg)

torch.save(final_phase_state, ... / "phase_state.pt")
torch.save(final_mtn_state, ... / "mtn_state.pt")
```

必须确保：

```text
_global/final_rmsnorm/gif_state.pt
_global/final_rmsnorm/clip_state.pt
```

不存在。

如果旧 artifact 目录中残留这些文件，应删除或 validator 直接报错；不要 silently ignore。

---

# 11. calibration manifest 的 global state schema

当前 global Final RMSNorm manifest 只有 Phase 信息。

修改后至少包含：

```yaml
global_states:
  final_rmsnorm:
    parameter_layout: ...
    configured_group_size: ...
    effective_group_size: ...

    phase_state_path: _global/final_rmsnorm/phase_state.pt
    phase_state_sha256: ...
    tau_shape: ...
    phase_tau_calibration: ...
    phase_tau_clamp_min: 0.0005
    phase_tau_clamp_max: 10000.0

    mtn_state_path: _global/final_rmsnorm/mtn_state.pt
    mtn_state_sha256: ...
    base_scale_shape: ...
    mtn_base_scale_calibration: ...
    mtn_base_scale_multiplier: 2.0
    mtn_base_scale_clamp_min: 0.0005
    mtn_base_scale_clamp_max: 10000.0

    gif_state_present: false
    clip_state_present: false
```

具体字段名可遵循仓库已有风格，但必须：

- 能验证 Phase state hash。
- 能验证 MTN state hash。
- 明确没有 global GIF。
- 明确没有 global Clip。
- 明确记录 parameter calibration 与 clamp provenance。

---

# 12. `snn2/controller.py`：把 Phase-only Final Norm hook 重构为 generic Final Norm neuron hook

当前：

```python
apply_final_norm_phase()
```

只支持：

```text
deploy_phase
```

这已经不能满足新 topology。

应重构为语义明确的 generic 方法，例如：

```python
apply_final_norm_neuron()
```

并增加：

```python
self._final_norm_phase
self._final_norm_mtn
```

缓存。

---

## 12.1 Static Phase-aware ANN

当：

```python
self.mode == "phase"
```

时：

1. 加载：

```text
site_root/_global/final_rmsnorm/phase_state.pt
```

2. 构造：

```python
PhaseSurrogate(
    state,
    T=self.phase_T,
    surrogate_slope=self.phase_surrogate_slope,
)
```

3. 输入是普通 Final RMSNorm 输出。
4. 直接：

```python
output = module(x)
```

5. **不要调用 Clipper。**
6. 即使：

```text
replacement.common_clip_enabled = true
```

Final RMSNorm 仍不得 Clip。

---

## 12.2 GIF-aware ANN

当：

```python
self.mode == "gif"
```

时：

```python
return x
```

不得：

- 加载 global GIF state
- 实例化 GIF
- 运行 GIF
- 加载 Clip
- 执行 Clip

---

## 12.3 Phase deployment

当：

```python
self.mode == "deploy_phase"
```

时保持当前 temporal Phase 行为：

```text
Temporal Final RMSNorm
    -> to_temporal()
    -> PhaseSurrogate.temporal()
    -> from_temporal()
```

Phase SNN 使用：

```text
_global/final_rmsnorm/phase_state.pt
```

---

## 12.4 MTN deployment

新增：

```python
self.mode == "deploy_mtn"
```

行为：

1. 加载：

```text
_global/final_rmsnorm/mtn_state.pt
```

2. 构造：

```python
MultiThresholdNeuron(
    state,
    T=self.mtn_T,
    K=self.mtn_K,
    threshold_factor=self.mtn_threshold_factor,
)
```

3. 输入 Final Temporal RMSNorm 输出。
4.：

```python
temporal = to_temporal(x, self.temporal_steps)
output = mtn.temporal(temporal)
output = from_temporal(output)
```

5. 保持 shape/dtype/device validation。
6. 不允许 Clip。

最终：

```text
Final Temporal RMSNorm
    -> MTN temporal neuron
    -> LM Head
```

---

## 12.5 GIF deployment

当：

```python
self.mode == "deploy_gif"
```

时：

```python
return x
```

Final RMSNorm 后不加 GIF Neuron。

注意：Final RMSNorm 本身仍由 `_install_temporal_rmsnorm()` temporal 化，因为这是 deploy 模式。

---

## 12.6 identity / none / collect

保持 identity forward。

不要让这个 global hook 干扰：

- Base ANN
- vanilla
- unaware
- rotated-pre-finetuning
- calibration collect

Final RMSNorm statistics 的 collect 仍沿用当前专门的 statistics hook/record path。

---

# 13. `snn2/model_integration.py`

当前 Final RMSNorm hook：

```python
parts.final_norm.register_forward_hook(
    lambda ...: controller.apply_final_norm_phase(output)
)
```

改成 generic hook，例如：

```python
parts.final_norm.register_forward_hook(
    lambda ...: controller.apply_final_norm_neuron(output)
)
```

不要改变：

```python
_install_temporal_rmsnorm(parts.final_norm, controller)
```

的 deploy-only 逻辑。

因此最终顺序必须自然成立：

### Phase-aware ANN

```text
original Final RMSNorm.forward()
    -> generic final-norm hook
    -> PhaseSurrogate.forward()
    -> LM Head
```

### Phase/MTN deployment

```text
temporal RMSNorm wrapper
    -> generic final-norm hook
    -> Phase.temporal() / MTN.temporal()
    -> LM Head
```

### GIF deployment

```text
temporal RMSNorm wrapper
    -> generic final-norm hook returns identity
    -> LM Head
```

---

# 14. Final RMSNorm 永远不使用 Clip

这是硬约束。

不要：

- 在 global final_rmsnorm 目录生成 `clip_state.pt`
- 在 controller Final Norm path 读取 `clip_root`
- 调用 `Clipper`
- 受 `replacement.common_clip_enabled` 影响

因此：

```text
common_clip_enabled
```

只继续控制现有 layer-wise eligible sites 的 aware ANN Clip。

Final RMSNorm global position 不属于 Clip eligible sites。

---

# 15. Phase / MTN Neuron class 的 state validation

## 15.1 PhaseSurrogate

在初始化时除现有 calibration metadata 外，还应验证：

```text
tau_clamp_min == 5e-4
tau_clamp_max == 1e4
tau_clamp_policy == expected policy
```

并严格验证：

```python
torch.all(tau >= 5e-4)
torch.all(tau <= 1e4)
torch.isfinite(tau).all()
```

如果 artifact 不满足，直接报旧 state / incompatible state，要求重新 calibration。

不要运行时 silently clamp 一个非法旧 state。

---

## 15.2 MultiThresholdNeuron

新增 MTN calibration provenance validation：

```text
EMA factor
FP32 accumulator
channel policy
group reduction
multiplier = 2.0
clamp range
clamp policy
```

并验证：

```python
torch.all(base_scale >= 5e-4)
torch.all(base_scale <= 1e4)
torch.isfinite(base_scale).all()
```

不要为了兼容旧 extrema-based state 自动 fallback。

---

# 16. Stage B Clip 的连锁变化

普通 layer-wise eligible sites 的 common Clip 仍然保留现有设计。

但是 Stage B Clip bounds 当前依赖：

```text
phase_state["tau"]
mtn_state["base_scale"]
```

因此本次修改后：

- Phase τ 变为 clamp 后 EMA-group-max。
- MTN base_scale 变为 clamp 后 `2 * EMA-group-max`。
- Stage B Clip profile 必须基于新的 Stage A states 重新 materialize。
- 原先基于 extrema MTN / unclamped Phase 的 Stage B Clip artifact 不得复用。

Final RMSNorm 本身仍然完全 Clip-free。

---

# 17. temporal policy / schema / artifact version 必须升级

本次不是纯实现细节变更，而是：

- MTN calibration semantics 改变。
- Phase state semantics增加 clamp。
- Final RMSNorm topology 改变。
- global state schema 新增 MTN。
- calibration provenance 改变。

因此必须 bump 相应版本，拒绝旧 artifact。

当前基线可见：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 7
SITE_STATE_FORMAT_VERSION = 9
STATISTICS_FORMAT_VERSION = 3
CALIBRATION_MANIFEST_FORMAT_VERSION = 11
CONVERSION_METADATA_FORMAT_VERSION = 12
```

建议至少升级为下一版本，例如：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 8
SITE_STATE_FORMAT_VERSION = 10
STATISTICS_FORMAT_VERSION = 4
CALIBRATION_MANIFEST_FORMAT_VERSION = 12
CONVERSION_METADATA_FORMAT_VERSION = 13
```

如果 Codex 在修改时发现当前 main 已再次 bump，则不要机械使用上述数字，应在当前值基础上继续递增，保持全项目一致。

---

## 17.1 Temporal implementation name

当前：

```text
sparse_llm_temporal_v7_two_stage_calibration
```

必须改名，使旧 conversion/manifest 无法假装兼容。

例如：

```text
sparse_llm_temporal_v8_final_norm_phase_mtn_ema_clamp
```

名字可调整，但必须代表新语义，并同步所有 validator/test/docs。

---

# 18. temporal policy metadata 需要新增/修改

当前只存在类似：

```text
phase_final_norm_policy
```

应改为能够完整描述三种 neuron 的 Final RMSNorm policy。

建议：

```text
final_norm_neuron_policy:
    phase_ann: static_phase_surrogate
    phase_snn: temporal_phase
    mtn_snn: temporal_mtn
    gif_ann: identity
    gif_snn: identity
    clip: forbidden
```

如果 manifest 只允许 string，则可使用一个稳定 versioned string，例如：

```text
phase_ann_surrogate_phase_snn_temporal_mtn_snn_temporal_gif_identity_clip_forbidden_v1
```

同时记录 shared parameter calibration：

```text
phase_parameter_calibration
mtn_parameter_calibration
parameter_ema_factor
parameter_clamp_min
parameter_clamp_max
```

`validate_temporal_policy()` 必须严格检查。

---

# 19. conversion / evaluation / artifact verification

全仓库 grep 并处理以下旧语义：

```text
apply_final_norm_phase
_final_norm_phase
final_norm_phase_state
PHASE_FINAL_NORM_POLICY
phase_final_norm_policy
final_rmsnorm/phase_state.pt
global_states.final_rmsnorm
```

重点检查：

```text
snn2/conversion.py
snn2/state_validation.py
scripts/verify_artifacts.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

以及 tests 中相关断言。

---

## 19.1 `snn2/state_validation.py`

`validate_site_state_bundle()` 当前只验证 global Phase state。

必须改为同时要求：

```text
phase_state.pt
mtn_state.pt
```

并验证：

- manifest path/hash。
- state constructor 可成功实例化。
- Phase/MTN grouping metadata 一致。
- clamp metadata 正确。
- tensor 全部位于 `[5e-4, 1e4]`。
- global directory 中不存在：
  - `gif_state.pt`
  - `clip_state.pt`

返回 validation info 时不要再只有：

```text
final_norm_phase_state
```

应改成能够表达两个 global states，例如：

```text
final_norm_states:
    phase: ...
    mtn: ...
```

或两个独立字段。

---

## 19.2 conversion metadata

conversion descriptor 必须记录新的：

- temporal implementation version/name。
- final norm neuron policy。
- Phase/MTN shared EMA calibration rule。
- clamp range。
- Final RMSNorm Phase state hash。
- Final RMSNorm MTN state hash。

旧 conversion descriptor 必须被拒绝。

---

## 19.3 `scripts/verify_artifacts.py`

增加/修改校验：

1. 每个 Stage A calibration bundle：
   - global Final RMSNorm Phase state 必须存在。
   - global Final RMSNorm MTN state 必须存在。
   - hash 必须匹配 manifest。
   - global GIF state 必须不存在。
   - global Clip state 必须不存在。

2. Phase/MTN state：
   - parameter metadata 正确。
   - 数值范围符合 `[5e-4, 1e4]`。
   - MTN 不再允许旧 extrema-based provenance。

3. conversion：
   - 选择 phase 时 global Phase state required。
   - 选择 mtn 时 global MTN state required。
   - 选择 gif 时禁止依赖 global GIF state。

---

# 20. regression recorder 命名

当前 regression checkpoint 使用 Phase-only 命名：

```text
final_norm/before_global_phase
final_norm/after_global_phase
```

应改成不误导的 generic 命名，例如：

```text
final_norm/before_global_neuron
final_norm/after_global_neuron
```

如果需要区分 neuron，可追加：

```text
final_norm/after_global_phase
final_norm/after_global_mtn
final_norm/after_global_identity
```

但不要继续让 MTN 经过一个名为 `apply_final_norm_phase` 的路径。

如果存在：

```text
regression_bypass_final_norm_phase
```

应重构成 generic final norm bypass，或者提供明确的 Phase/MTN independent semantics，并同步测试。

---

# 21. `AGENTS.md` 必须同步修改

当前仓库根目录 `AGENTS.md` 是项目硬约束，本次代码完成时必须同步更新。

至少修改以下语义。

---

## 21.1 修改当前关于 Final RMSNorm grouping 的规则

当前类似：

```text
calibration.group_size 同时控制普通 Site 与 final RMSNorm 的 Phase/GIF/MTN/Clip grouping
```

应改为：

```text
calibration.group_size 必须为 -1 或正整数，并控制普通 Site grouping，
以及 Final RMSNorm 的 Phase/MTN grouping。
Final RMSNorm 永远不生成或执行 GIF/Clip，因此不存在 Final RMSNorm GIF/Clip grouping。
改变 G 后必须重新 calibration，禁止跨 G 复用 statistics/state/manifest/conversion/SNN 工件。
```

---

## 21.2 替换/扩展 Phase-only calibration 规则

应明确写成：

```text
Phase tau 与 MTN base_scale 必须共享同一 calibration statistics：
在原生 site layout 上逐 channel/head-channel 计算 abs-max，
使用 FP32、factor=0.99 的有序 EMA，
再按本项目当前 group policy 在 group 内取 max。
Phase tau = clamp(group_max(EMA), 5e-4, 1e4)；
MTN base_scale = clamp(2 * group_max(EMA), 5e-4, 1e4)。
phase_state.pt / mtn_state.pt 必须直接保存 clamp 后的 parameter。
不得使用旧 extrema-based MTN initialization。
calibration 固定 batch_size=1 且 single-process。
```

---

## 21.3 新增/修改 Final RMSNorm topology 硬约束

必须明确：

```text
Final RMSNorm 是 global replacement position，不计入每层 10 sites。

GIF-aware ANN training/final ANN evaluation：
Final RMSNorm 后 identity，不加 GIF，不加 Clip。

GIF SNN deployment：
Final Temporal RMSNorm 后 identity，不加 GIF。

Phase-aware ANN training/final ANN evaluation：
ordinary Final RMSNorm 后必须执行 PhaseSurrogate，
使用当前 ANN-training calibration 的 _global/final_rmsnorm/phase_state.pt；
该位置禁止 Clip。

Phase SNN deployment：
Final Temporal RMSNorm 后执行 temporal Phase neuron。

MTN SNN deployment：
Final Temporal RMSNorm 后执行 temporal MTN neuron，
使用 _global/final_rmsnorm/mtn_state.pt。

Global Final RMSNorm 只允许 phase_state.pt 与 mtn_state.pt；
禁止 gif_state.pt 与 clip_state.pt。
```

---

## 21.4 把本次明确保留不变的三条策略写入 AGENTS.md

加入硬约束：

```text
Embedding temporal encoding 固定 uniform_embedding_divide_by_T，
不得改为 t0 impulse。

Prefix K/V temporal decomposition 固定 uniform_kv_divide_by_T，
不得改为 t0 impulse。

Phase/MTN grouping 继续使用本项目现有 site-specific grouping；
本项目只借鉴 SparseLLM 的 MTN parameter statistics rule，
不得为了“对齐 SparseLLM”改写 group granularity。
```

---

# 22. `代码结构总结.md`

`AGENTS.md` 当前要求：

```text
代码结构总结.md 只保留 2. 目录结构；
每个文件后只用一句话描述功能；
职责变化必须同步更新。
```

本次若以下文件职责发生变化：

```text
snn2/calibration.py
snn2/controller.py
snn2/stats.py
snn2/neurons.py
snn2/state_validation.py
snn2/temporal_ops.py
snn2/model_integration.py
```

应按现有格式同步一句话职责描述。

不要向 `代码结构总结.md` 添加本修改方案正文。

---

# 23. 必须新增/修改的测试

不要只修改代码后依赖现有 pytest。

必须新增针对本次语义的 regression/unit tests。

---

## 23.1 Phase clamp test

构造 statistics，使 raw grouped EMA 出现：

```text
< 5e-4
within range
> 1e4
```

断言：

```text
tau == [5e-4, original, 1e4]
```

并验证 state metadata。

---

## 23.2 MTN EMA-source test

构造一个 statistics，使：

```text
global value_min/value_max extrema
```

与：

```text
EMA channel abs-max
```

明显不同。

断言新：

```text
base_scale
```

来自：

```text
2 * grouped EMA
```

而不是：

```text
2 * max(abs(value_min), abs(value_max))
```

这是防止代码“看似改了 metadata、实际仍走旧 extrema”最重要的测试之一。

---

## 23.3 MTN clamp test

断言：

```text
base_scale = clamp(2 * grouped_ema, 5e-4, 1e4)
```

注意 clamp 在乘 `2` **之后**。

必须包含一个能够区分：

```text
2 * clamp(ema)
```

和：

```text
clamp(2 * ema)
```

的测试值。

最终必须是：

```text
clamp(2 * ema, ...)
```

---

## 23.4 Group layout tests

确保本次改动没有改变：

- ordinary last-dim grouping。
- Site 2 per-head grouping。
- Site 3/4 post-repeat-kv logical per-head grouping。
- Site 5 `[H,1]`。
- Site 6 merged last-dim grouping。
- Final RMSNorm last-dim grouping。

---

## 23.5 Global state materialization test

断言：

```text
_global/final_rmsnorm/phase_state.pt exists
_global/final_rmsnorm/mtn_state.pt exists
_global/final_rmsnorm/gif_state.pt does not exist
_global/final_rmsnorm/clip_state.pt does not exist
```

manifest hashes 必须一致。

---

## 23.6 Final Norm Phase-aware ANN test

构造：

```text
controller.mode = "phase"
```

断言：

```text
Final RMSNorm output
    -> PhaseSurrogate.forward
```

且：

- 使用 ANN-training global Phase state。
- `surrogate_slope` 来自 runtime config。
- 不使用 Clip。
- `common_clip_enabled=true/false` 均不改变该位置是否有 Clip：都没有 Clip。

---

## 23.7 Final Norm GIF-aware ANN test

构造：

```text
controller.mode = "gif"
```

断言 Final RMSNorm hook exact identity。

即使：

```text
common_clip_enabled = true
```

仍不能加载/执行 Final Norm Clip。

---

## 23.8 Final Norm deployment tests

分别测试：

### deploy_phase

```text
Temporal RMSNorm output -> Phase.temporal()
```

### deploy_mtn

```text
Temporal RMSNorm output -> MTN.temporal()
```

### deploy_gif

```text
Temporal RMSNorm output -> identity
```

并验证 shape/dtype/device。

---

## 23.9 State validation negative tests

至少覆盖：

- missing global MTN state -> fail。
- wrong global MTN SHA -> fail。
- global GIF state exists -> fail。
- global Clip state exists -> fail。
- Phase τ 超 clamp range -> fail。
- MTN base_scale 超 clamp range -> fail。
- old MTN extrema calibration metadata -> fail。
- old temporal implementation / manifest version -> fail。

---

## 23.10 Non-regression tests

必须确保：

- Embedding 仍是 `E/T` 每步。
- Prefix KV 仍是 `KV/T` 每步。
- expected sites per layer 仍为 10。
- Site 5 GIF 仍 identity。
- GIF Final Norm 仍 identity。
- Phase/MTN group policy未变化。

---

# 24. Artifact 兼容策略

本项目明确采用 strict schema。

因此：

- 不要兼容旧 `phase_state.pt`。
- 不要兼容旧 extrema-based `mtn_state.pt`。
- 不要兼容只有 global Phase、没有 global MTN 的 calibration manifest。
- 不要在 runtime 自动补 state。
- 不要 silently clamp 旧 state。
- 不要根据文件缺失自动回退为旧 topology。

出现旧 artifact 时应给出明确错误：

```text
re-run calibration / re-materialize Stage A / regenerate Stage B / reconvert
```

---

# 25. 修改完成后必须重新生成的 artifact

因为 calibration semantics 与 topology 均发生变化，旧 artifact 不应继续用于最终实验。

至少需要重新生成：

1. Stage A calibration statistics/state/manifest。
2. Stage B Clip profiles（对于需要 common Clip 的 aware ANN）。
3. Phase-aware ANN training 所依赖的 ANN-training Final RMSNorm Phase state。
4. post-finetuning conversion calibration。
5. conversion descriptor。
6. Phase / GIF / MTN SNN evaluation artifact。
7. 任何依赖旧 manifest/hash 的 verify artifact。

Rotation、Prefix discovery、Embedding policy本身没有因为本次方案改变；是否可以复用必须继续按现有 provenance/hash policy 判断，不要因为本次改动无条件删除 G-independent shared artifact。

---

# 26. 建议修改文件清单

至少检查并按职责修改：

```text
AGENTS.md
snn2/phase_statistics.py        # 或将 shared parameter statistics 常量抽到更通用模块
snn2/stats.py
snn2/calibration.py
snn2/neurons.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_ops.py
snn2/state_validation.py
snn2/conversion.py
scripts/verify_artifacts.py
代码结构总结.md
tests/...
```

同时全仓库 grep：

```text
phase_ema_abs_max
phase_ema_updates
PHASE_TAU_CALIBRATION
apply_final_norm_phase
_final_norm_phase
regression_bypass_final_norm_phase
PHASE_FINAL_NORM_POLICY
phase_final_norm_policy
final_norm_phase_state
global_states
final_rmsnorm
base_scale
value_min
value_max
SITE_STATE_FORMAT_VERSION
STATISTICS_FORMAT_VERSION
CALIBRATION_MANIFEST_FORMAT_VERSION
CONVERSION_METADATA_FORMAT_VERSION
TEMPORAL_IMPLEMENTATION_VERSION
```

不要只修改上述显式文件而遗漏脚本/tests/docs 中的 schema 检查。

---

# 27. 完成后的目标前向图

## 27.1 Phase-aware ANN

```text
Embedding
↓
Transformer layers
  └─ 10 个现有 static Phase replacement sites
↓
Final ordinary RMSNorm
↓
Global PhaseSurrogate
  └─ ANN-training calibration phase_state.pt
  └─ no Clip
↓
LM Head
```

---

## 27.2 GIF-aware ANN

```text
Embedding
↓
Transformer layers
  └─ 10 个现有 GIF replacement topology
↓
Final ordinary RMSNorm
↓
identity
  └─ no GIF
  └─ no Clip
↓
LM Head
```

---

## 27.3 Phase SNN

```text
uniform E/T temporal embedding
↓
Full temporal Transformer
↓
Final Temporal RMSNorm
↓
Global temporal Phase Neuron
↓
LM Head
↓
sum over T
```

---

## 27.4 GIF SNN

```text
uniform E/T temporal embedding
↓
Full temporal Transformer
↓
Final Temporal RMSNorm
↓
identity
↓
LM Head
↓
sum over T
```

---

## 27.5 MTN SNN

```text
uniform E/T temporal embedding
↓
Full temporal Transformer
↓
Final Temporal RMSNorm
↓
Global temporal MTN Neuron
↓
LM Head
↓
sum over T
```

---

# 28. 最终验收标准

代码修改只有同时满足以下条件才算完成：

- [ ] Phase τ 使用 `EMA channel abs-max -> group max -> clamp`。
- [ ] MTN base_scale 使用 `EMA channel abs-max -> group max -> ×2 -> clamp`。
- [ ] Phase clamp 直接作用于 τ。
- [ ] MTN clamp 在乘 2 后作用于 base_scale。
- [ ] clamp 固定为 `[5e-4, 1e4]`。
- [ ] `phase_state.pt` / `mtn_state.pt` 保存的是 clamp 后 tensor。
- [ ] ordinary sites 与 Final RMSNorm 共用同一 parameter statistics semantics。
- [ ] group granularity 完全保持现有项目设计。
- [ ] Embedding temporal encoding 仍为 uniform divide-by-T。
- [ ] Prefix KV temporal decomposition 仍为 uniform divide-by-T。
- [ ] 每层 site 数仍为 10。
- [ ] ANN-training Final RMSNorm 生成 Phase + MTN global states。
- [ ] Phase-aware ANN Final RMSNorm 后执行 PhaseSurrogate。
- [ ] Phase-aware ANN Final RMSNorm 后无 Clip。
- [ ] GIF-aware ANN Final RMSNorm 后 identity、无 GIF、无 Clip。
- [ ] Phase SNN Final Temporal RMSNorm 后有 Phase Neuron。
- [ ] MTN SNN Final Temporal RMSNorm 后有 MTN Neuron。
- [ ] GIF SNN Final Temporal RMSNorm 后没有 GIF Neuron。
- [ ] global Final RMSNorm 不生成 GIF state。
- [ ] global Final RMSNorm 不生成 Clip state。
- [ ] manifest / conversion / validator 都记录并检查新的 global MTN 与 clamp provenance。
- [ ] 旧 artifact/schema 被严格拒绝。
- [ ] Stage B Clip 根据新 Phase/MTN states 重建。
- [ ] `AGENTS.md` 已同步写入本次全部硬约束。
- [ ] `代码结构总结.md` 按现有项目规则同步职责变化。
- [ ] 新增测试覆盖 topology、EMA source、clamp、global states 与 negative validation。
- [ ] `pytest -q` 全部通过。

---

# 29. 实施原则

1. 不要为了“对齐 SparseLLM”修改本方案明确要求保留的 Embedding、Prefix KV 或 group granularity。
2. 不要新增第 11 个 site；Final RMSNorm 是 global position。
3. 不要让 Final RMSNorm 路径复用 layer-wise Clip 逻辑。
4. 不要保留旧 MTN extrema-based fallback。
5. 不要让 Phase 与 MTN 分别实现两套 EMA/grouping 代码；应共享同一 calibration helper/statistics source。
6. 不要仅修改 forward 而遗漏 calibration manifest、state validation、conversion metadata、verify_artifacts 和 tests。
7. 本次完成后以新的 artifact schema 为唯一合法格式。
8. 修改完成后运行：

```bash
pytest -q
```

必须全部通过后再认为任务完成。
