# SNN 项目完整修改方案：Per-Head Grouped Calibration 与 Softmax Site 5 特殊策略（最终版）

> **历史说明：**
> 本文记录的是当时的 Q16 Site 5 设计，该设计现已被
> `SNN_Site5_GIF_Strict_SpikeLLM_Identity_Modification_Plan.md` 替代。
> 当前实现以 SpikeLLM `n_bits=16` sentinel 的真实 identity 行为为准；
> 本文中的 Q16 / `SoftmaxFixedGIF` / cumulative-difference 内容仅用于历史追溯，
> 不再代表当前实现。

> 本文档是本次修改的唯一实施依据，面向部署在服务器上的 Codex。无需依赖此前对话上下文。
>
> 基线：`https://github.com/wangwk699/SNN/tree/main` 当前实现。
>
> 本方案已经冻结以下设计：Self-Attention per-head 独立参数、`calibration.group_size` 统一控制普通 site 的 Phase/GIF/MTN/Clip、Site 3/4 使用原生 KV head、Site 5 特殊 Phase/GIF/MTN 规则且永久无 Clip、Site 5 GIF 显式 fixed-[0,1] Q16、GIF temporal 使用 quantized cumulative difference、group-size-dependent artifact/provenance 隔离、旧统计/state/conversion artifact 全部拒绝复用。

---

# 1. 修改目标

本次修改必须完成以下目标：

1. Self-Attention 中 Site 2/3/4/6 改成 **per-head 独立统计量和独立参数**。
2. `calibration.group_size` 同时控制 Phase、GIF、MTN、Clip 的分组粒度；不再仅影响 generic GIF/MTN/Clip。
3. 对 Self-Attention 普通多头 site，grouping 只能发生在 **每个 head 内部**，禁止把 `head_num × head_dim` flatten 后跨 head 分组。
4. `calibration.group_size=-1`：
   - 非 attention site：整个最后一维 channel dimension 为一组；
   - Site 2/3/4/6：每个 head 的整个 `head_dim` 各自为一组，因此每个 head 有独立参数；
   - Site 5：忽略全局 group_size，固定为“每个 head 的整个可变 key-position 维度为一组”。
5. Phase 不再使用当前 SpikingLLM-aligned attention reshape / flatten-head statistical view，也不再对所有 attention heads 做 global scalar τ。
6. Softmax Site 5：
   - Phase：每个 attention head 一个 scalar τ；
   - MTN：每个 attention head 一个 scalar `base_scale`；
   - GIF：fixed `[0,1]` 显式 16-bit quantization；
   - Clip：永久禁止；
   - 不受全局 `calibration.group_size` 控制。
7. `calibration.group_size` 对 Site 1/7/8/9/10 和 global final RMSNorm Phase 同样生效。
8. 保持当前 10-site topology、Prefix runtime、ANN/SNN 边界、Phase surrogate slope 等其他实验协议不变。
9. 因 statistics/state/temporal/conversion 数学语义改变，所有旧 artifact 必须拒绝加载，不能静默兼容。

---

# 2. 必须保持不变的现有项目边界

以下行为不得被本次修改破坏：

- 仍是固定 10 个 activation replacement sites。
- `vanilla/unaware` ANN training 与 final ANN evaluation 仍是 identity activation semantics。
- `phase_aware` final ANN evaluation 仍复现训练期 `PhaseSurrogate.forward()`；`gif_aware` 仍复现训练期 GIF static semantics。
- aware ANN training 的局部 replacement 在 site 内展开/聚合，时间维度不能跨 Transformer layer 传播；它仍是 ANN fine-tuning。
- `--neuron phase|gif|mtn` 正式评估仍必须走 full-temporal `deploy_*`。
- SNN conversion/deployment/evaluation 永远不执行 common Clip。
- Prefix K/V runtime 必须经过 Site 3/4 neuron。
- Softmax runtime 必须对完整 Softmax tensor（包含 Prefix columns）执行 Site 5 neuron。
- calibration statistics 可以排除 Prefix positions，但不能导致 runtime bypass neuron。
- Phase `surrogate_slope` 仍只作为 phase-aware ANN runtime backward 参数，不写入 `phase_state.pt`。
- Phase EMA accumulator 仍固定 FP32，EMA factor 仍为 0.99。
- shared rotation、data manifest、Prefix 不因为 group_size 无意义复制；只有真正依赖 group_size 的 artifact 隔离。

---

# 3. `calibration.group_size` 的新精确定义

配置只允许：

```yaml
calibration:
  group_size: -1
```

或：

```yaml
calibration:
  group_size: G
```

其中 `G` 为正整数。

必须拒绝：

```text
0
-2
-3
...
```

## 3.1 普通固定宽度 activation

最后一维为 `C`：

- `G=-1`：effective group width = `C`，group 数 = 1；
- `G>0`：要求 `C % G == 0`，group 数 = `C/G`；每相邻 G channels 共享一组参数。

## 3.2 Self-Attention 普通多头 activation

张量：

```text
[B, H, L, D]
```

其中 `D=head_dim`。

- `G=-1`：每个 head 的整个 D 为一组；参数推荐保存为 `[H,1]`；
- `G>0`：要求 `D % G == 0`；每个 head 有 `D/G` 组；参数保存为 `[H,D/G]`。

绝不能先 reshape 为 `[B,L,H*D]` 后做 grouping。

## 3.3 Site 5

Site 5 不读取 global `calibration.group_size`。

它的 effective policy 固定为：

```text
per-head full variable key-position axis
```

state 中可记录：

```text
configured_group_size = cfg.calibration.group_size   # provenance only
group_size = -1                                      # effective Site 5 override
group_size_source = site5_fixed_override
```

---

# 4. 10 个 Site 的最终 layout

| Site | 名称 | 典型 runtime shape | grouping/layout |
|---|---|---|---|
| 1 | post_input_rmsnorm | `[B,L,C]` | last-dim grouped |
| 2 | q_post_rope_r3 | `[B,Hq,L,D]` | query-head independent + within-head grouping |
| 3 | k_post_rope_r3 | `[B,Hkv,L,D]` | KV-head independent + within-head grouping |
| 4 | v_projection_r2 | `[B,Hkv,L,D]` | KV-head independent + within-head grouping |
| 5 | post_spiking_softmax | `[B,Hq,Q,K]` | special per-head scalar/fixed-range policy |
| 6 | post_attention_value_dot_r2 | `[B,Hq,Q,D]` | query-head independent + within-head grouping |
| 7 | post_mlp_rmsnorm | `[B,L,C]` | last-dim grouped |
| 8 | post_spiking_silu | `[B,L,I]` | last-dim grouped |
| 9 | post_mlp_up_proj | `[B,L,I]` | last-dim grouped |
| 10 | post_mlp_product_r4 | `[B,L,I]` | last-dim grouped |

Global final RMSNorm Phase：按普通 `[B,L,C]` last-dim grouping，受 `calibration.group_size` 控制。

---

# 5. GQA / MQA：Site 3/4 必须使用原生 KV Head

当前 Site 3/4 replacement 发生在 `repeat_kv()` 前，因此：

- Site 3 K 的 state head 数必须是 `num_key_value_heads`；
- Site 4 V 的 state head 数必须是 `num_key_value_heads`；
- calibration 不能为了 Phase 再 `repeat_kv()`；
- runtime replacement 继续发生在 `repeat_kv()` 前。

否则 calibration state 与 runtime tensor head layout 不一致。

## 5.1 Site 3/4 GIF saliency 聚合

当前 consumer saliency 是在 `repeat_kv()` 后的 query-head tensor 上计算的。

假设：

```text
Hq = Hkv * num_key_value_groups
score.shape = [B,Hq,L,D]
```

必须聚合回 KV head：

```python
score_kv = score.reshape(
    B, Hkv, num_key_value_groups, L, D
).sum(dim=2)
```

结果：

```text
[B,Hkv,L,D]
```

使用 **sum**，不是 mean、第一份拷贝或保留 repeat 后 head。

原因：同一 KV head 被多个 query heads 消费，saliency 应累计所有 consumer contribution。

Site 3 和 Site 4 都必须按该规则处理。

---

# 6. Statistics schema 必须重构并版本化

这是本次修改的强制项，不能只 bump `phase_state.pt`/conversion version。

当前 raw `statistics.pt` 的 tensor shape 和 Site 5 schema 都会发生变化，因此必须新增/提升 statistics schema version，例如：

```python
STATISTICS_FORMAT_VERSION = 2
```

如果当前 main 已有该常量，则在当前最新值基础上 +1。

每个 `statistics.pt` 必须保存至少：

```text
format_version
site_index
layout_kind
num_heads
channels_per_head
channels
value_min
value_max
abs_max
sum_abs
sum_sq
saliency_sum
saliency_row_count
row_count
tensor_count
phase_ema_abs_max
phase_ema_updates
phase_tau_ema_factor
phase_tau_accumulator_dtype
```

其中与 layout 不适用的字段允许为 `None`/空 tensor，但 schema 必须明确。

`statistics_manifest.json` 的 `format_version` 同样必须升级，并记录新的 layout policy。

`calibration.materialize_calibration_states()` 必须验证 statistics version。旧 `statistics.pt` 不能被新代码重新 materialize 成新 state。

如果 calibration 目录中存在旧 statistics manifest/state，必须报错要求删除/移动旧目录后重新 calibration。

---

# 7. StatisticsStore 的三类 layout

## 7.1 `last_dim`

用于：Site 1/7/8/9/10、global final RMSNorm。

输入如 `[B,L,C]`，保存：

```text
value_min[C]
value_max[C]
abs_max[C]
sum_abs[C]
sum_sq[C]
saliency_sum[C]
saliency_row_count[C]
phase_ema_abs_max[C]
phase_ema_updates[C]
```

## 7.2 `attention_head`

用于 Site 2/3/4/6。

输入 `[B,H,L,D]`，保存：

```text
value_min[H,D]
value_max[H,D]
abs_max[H,D]
sum_abs[H,D]
sum_sq[H,D]
saliency_sum[H,D]
saliency_row_count[H,D]
phase_ema_abs_max[H,D]
phase_ema_updates[H,D]
```

统计样本只沿 batch/token 等非 head/non-channel 维 reduction，不跨 head。

## 7.3 `attention_softmax`

用于 Site 5。

输入 `[B,H,Q,K]`，K/Q 可以变化。

不再分配固定 `max_seq_length` key-position statistics buffer。

只保存 per-head：

```text
value_min[H]
value_max[H]
abs_max[H]
phase_ema_abs_max[H]
phase_ema_updates[H]
```

Site 5 不需要 GIF saliency，不再保存/依赖：

```text
saliency_sum[K]
saliency_row_count[K]
variable_channels
max_channels_by_site
```

若为了统一 schema 保留 saliency 字段，应为空且 validator 明确 Site 5 不要求它。

删除 Site 5 当前 `record_saliency_reduced()` 的 position-score 路径；若全仓库无其他用途，可直接删除该 API。

---

# 8. Phase 新统计规则

## 8.1 删除旧 SpikingLLM attention statistical view

当前以下语义必须从 active code/current docs 中删除：

```text
spikingllm_identity_input_layout
spikingllm_flatten_attention_heads_before_channel_ema
per_channel_ema_then_global_max
phase_statistical_view(...)
phase_statistical_view_version
```

历史文档 `docs/history/` 可以保留旧记录，但 active code、README、AGENTS、实验执行总结不得再把它描述为当前行为。

## 8.2 EMA

继续：

第一次：

\[
EMA = current\_abs\_max
\]

后续：

\[
EMA \leftarrow 0.99 EMA + 0.01 current\_abs\_max
\]

accumulator = FP32。

calibration 继续要求确定性 forward 顺序：

- `calibration.batch_size == 1`；
- **calibration 必须单进程执行**；
- 如果 distributed 已初始化且 `world_size > 1`，继续直接报错；
- 错误信息改成 Phase EMA order-dependent，而不是 SpikingLLM-specific。

## 8.3 普通 site / final RMSNorm

输入 `[B,L,C]`：

\[
m_c=\max_{B,L}|x_{B,L,c}|
\]

得到 `EMA[C]`。

`G=-1`：

\[
\tau = \max_c EMA_c
\]

保存 `[1]`。

`G>0`：

\[
\tau_g=\max_{c\in g} EMA_c
\]

保存 `[C/G]`。

## 8.4 Site 2/3/4/6

输入 `[B,H,L,D]`：

\[
m_{h,d}=\max_{B,L}|x_{B,h,L,d}|
\]

得到 `EMA[H,D]`。

`G=-1`：

\[
\tau_h=\max_d EMA_{h,d}
\]

保存：

```text
tau[H,1]
```

`G>0`：

\[
\tau_{h,g}=\max_{d\in g}EMA_{h,d}
\]

保存：

```text
tau[H,D/G]
```

禁止跨 H global max。

## 8.5 Site 5

输入 `[B,H,Q,K]`：

\[
m_h=\max_{B,Q,K}|x_{B,h,Q,K}|
\]

做 per-head EMA：

```text
phase_ema_abs_max[H]
```

最终：

```text
tau[H,1]
```

固定：

```text
effective group_size = -1
parameter_layout = attention_head_scalar
```

不读取 global G。

## 8.6 Phase state

必须继续保存现有算法必要字段：

```text
T
base
max_spikes
```

以及新 grouping metadata：

```text
parameter_layout
configured_group_size
group_size
num_heads
channels_per_head
groups_per_head
tau
v0
tau_calibration = ema_channel_abs_max_then_group_max
tau_ema_factor = 0.99
tau_accumulator_dtype = float32
tau_channel_policy = native_site_layout_per_channel
tau_reduction_policy = within_group_max_after_channel_ema
```

`tau.shape == v0.shape`。

继续：

\[
v_0=0.5\tau 2^{-T}
\]

Phase threshold 时间规则本次不改变。

---

# 9. Generic group reduction helper

替换当前只支持 1D vector 的 `_group_reduce()`。

推荐：

```python
def group_reduce_last_dim(values, group_size, reduction):
    width = values.shape[-1]
    effective = width if group_size == -1 else group_size
    if effective <= 0:
        raise ValueError(...)
    if width % effective != 0:
        raise ValueError(...)
    grouped = values.reshape(
        *values.shape[:-1],
        width // effective,
        effective,
    )
    ... reduce dim=-1 ...
```

输入输出：

```text
[C]   -> [C/G]
[H,D] -> [H,D/G]
```

Site 5 不走该函数。

对 divisibility error 必须报告：

```text
layer
site
layout
channel/head_dim
configured group_size
```

---

# 10. GIF 普通 Site 新语义

Site 1/2/3/4/6/7/8/9/10 保留当前 ordinary GIF：

```text
base_bits=4
high_qmax=30
temporal_steps=2
per_step_qmax=15
low_ratio/salient_ratio
static min-max qparams
```

但 qparams 按新 grouping 保存。

## 10.1 非 attention

```text
low_scale[C/G]
low_zero[C/G]
high_scale[C/G]
high_zero[C/G]
mask_low[C]
```

## 10.2 attention Site 2/3/4/6

```text
low_scale[H,D/G]
low_zero[H,D/G]
high_scale[H,D/G]
high_zero[H,D/G]
mask_low[H,D]
```

## 10.3 `mask_low`

仍逐 channel，不按 group 合并。

attention 中每个 head 独立排序：

```text
low_channels_per_head = floor(low_ratio * D)
```

每个 head 根据自己的 saliency 选 low/high channels，禁止跨 heads 统一排序。

Site 3/4 的 saliency 必须先按第 5 节聚合回 KV head。

---

# 11. Site 5 GIF：Fixed [0,1] Q16

Site 5 不使用 ordinary GIF state。

不生成：

```text
low_scale
low_zero
high_scale
high_zero
mask_low
low_ratio saliency
qmax=30 two-chunk quantization parameters
```

固定：

\[
q_{max}=65535
\]

\[
Q_{16}(x)=\frac{\operatorname{round}(65535x)}{65535}
\]

Softmax 理论上满足：

\[
0\le x\le1
\]

可以做数值安全 clamp `[0,1]`，但不能重新统计 xmin/xmax。

本项目应明确表述：采用 SpikeLLM fixed-[0,1] 思路，但 **显式执行 Q16 fake quantization**；不要声称逐行复制其 `n_bits>=16` 直接 identity 的代码行为。

## 11.1 Static / ANN forward

- FP32 中执行 Q16；
- round 使用 STE，保证 gif-aware ANN training 梯度；
- hard forward 数值必须等于 `round(65535*x)/65535`；
- 返回原 activation dtype。

## 11.2 Site 5 GIF state

推荐：

```text
state_kind = gif
format_version = new
parameter_layout = softmax_fixed_range
gif_policy = softmax_fixed_range_u16
configured_group_size = G
group_size = -1
group_size_source = site5_fixed_override
num_heads = H
range_min = 0.0
range_max = 1.0
quantization_bits = 16
qmin = 0
qmax = 65535
scale = 1/65535
zero_point = 0
temporal_steps = 2
temporal_policy = quantized_cumulative_difference
```

---

# 12. Site 5 GIF Temporal 定义

不能把 `[0,65535]` 强行拆成当前 ordinary GIF 的两个 `[0,15]` chunks。

新增独立 module，例如：

```text
SoftmaxFixedGIF
```

不要把大量 Site 5 special case 塞进 ordinary `StaticGIF`。

输入：

```text
incoming[T,B,H,Q,K]
```

计算 cumulative：

\[
c_t=\sum_{i=0}^{t} incoming_i
\]

逐 timestep 对 cumulative 做 Q16：

\[
\hat c_t = Q_{16}(c_t)
\]

再恢复 temporal increment：

\[
y_0=\hat c_0
\]

\[
y_t=\hat c_t-\hat c_{t-1}
\]

因此：

\[
\sum_t y_t=Q_{16}\left(\sum_t incoming_t\right)
\]

要求：

- temporal shape 不变；
- dtype/device 不变；
- temporal 中允许负 increment；
- 不使用 low/high qparams；
- 不使用 qmax30 decomposition；
- `temporal_steps=2`，保持全模型 GIF T 一致。

Temporal deployment 使用 hard round，不需要 STE。

---

# 13. MTN 新语义

## 普通非 attention

`base_scale` 按 grouped min/max：

```text
[C/G]
```

## Site 2/3/4/6

```text
base_scale[H,D/G]
```

## Site 5

不读取 global G。

每个 head：

\[
absolute_h=\max(|min_h|,|max_h|)
\]

沿用当前：

\[
base\_scale_h=2\cdot absolute_h
\]

保存：

```text
base_scale[H,1]
parameter_layout=attention_head_scalar
group_size=-1
```

---

# 14. Clip 新语义

## 14.1 clip-eligible sites

在 `sites.py` 集中定义：

```python
ATTENTION_HEAD_GROUPED_SITE_IDS = {2,3,4,6}
SOFTMAX_SITE_ID = 5
CLIP_ELIGIBLE_SITE_IDS = {1,2,3,4,6,7,8,9,10}
```

增加 helper：

```text
is_attention_head_grouped_site()
is_softmax_site()
site_supports_clip()
```

## 14.2 ordinary Clip

Site 1/2/3/4/6/7/8/9/10 在 ANN-training calibration 仍生成 Clip。

形状：

- non-attention：`[C/G]`；
- attention：`[H,D/G]`。

Clip 仍是 Phase/MTN/GIF representable interval intersection。

如果继续支持 `phase.base != 2`，计算 Phase representable bound 时不要硬编码 `2^{-T}` 的几何和；应使用当前 `phase.base` 的真实 amplitude sum。若项目只允许 base=2，也可以在 config 中明确严格验证 base=2。

## 14.3 Site 5

永久：

```text
no clip_state.pt
no Clipper load
no Clip forward
```

即使：

```yaml
replacement:
  common_clip_enabled: true
```

Site 5 仍然跳过 Clip。

若新 validator 在 Site 5 发现 stale `clip_state.pt`，必须拒绝 artifact 并要求重新 calibration。

## 14.4 manifest 语义

ANN-training bundle 仍可描述为：

```text
ann_training_with_common_clip
common_clip_generated = true
```

但必须明确该布尔值表示：

> 所有 clip-eligible sites 均生成 common Clip；Site 5 是协议定义的永久例外。

manifest 建议新增：

```text
clip_eligible_site_ids = [1,2,3,4,6,7,8,9,10]
clip_excluded_site_ids = [5]
softmax_site5_clip_policy = disabled
```

---

# 15. Runtime parameter broadcasting

当前 `_channel_values()` 只支持 last-dim 1D 参数，必须替换为 layout-aware helper。

state 至少支持：

```text
last_dim_grouped
attention_head_grouped
attention_head_scalar
softmax_fixed_range
```

## `last_dim_grouped`

输入 `[B,L,C]`，参数 `[Ng]`，expand 到 `[1,1,C]`。

## `attention_head_grouped`

输入 `[B,H,L,D]`，参数 `[H,Ng]`，group repeat 后 expand 到：

```text
[1,H,1,D]
```

严格验证：

```text
runtime H == state.num_heads
runtime D == state.channels_per_head
Ng == state.groups_per_head
```

## `attention_head_scalar`

Site 5 输入 `[B,H,Q,K]`，参数 `[H,1]`，expand 到：

```text
[1,H,1,1]
```

因此天然支持可变 K。

## GIF mask

新增 mask expansion：

- non-attention `[C]`；
- attention `[H,D]`。

删除当前 silent padding/truncation。新 state shape 不匹配必须直接报错。

---

# 16. `model_integration.py` 必改项

1. 删除 `phase_statistical_view` import 和所有 `phase_activation=` 参数。
2. Site 2：原生 `[B,Hq,L,D]` 直接 collect/apply。
3. Site 3/4 collect：
   - 继续排除 Prefix/past positions；
   - 不再 `repeat_kv()` 后做 Phase stats；
   - statistics 全部保持原生 KV head。
4. Site 3/4 saliency：repeat 后 score 按 GQA groups sum 回 KV head。
5. Site 5：
   - collect 时 statistics 可以切掉 Prefix key columns；
   - runtime 仍对完整 Softmax tensor apply；
   - 删除 `position_score` / `record_saliency_reduced(site5)`。
6. Site 6：保持 `[B,Hq,Q,D]` 进入 controller，不 flatten head。

Site 5 calibration 建议：

```python
statistics_weights = (
    weights[..., past_length:]
    if past_length
    else weights
)
controller.record_activation(layer_index, 5, statistics_weights)
```

仅 statistics 切 Prefix；实际 `weights` 继续完整 forward。

---

# 17. `stats.py` 必改项

- 重构 `SiteStatistics` 支持三类 layout；
- 记录 site/layout/head metadata；
- attention stats 保留 `[H,D]`；
- Site 5 保留 `[H]`；
- Phase accumulator独立、FP32；
- 删除 `variable_channels/max_channels_by_site` 设计；
- 删除 Site 5 fixed max-position buffer；
- 删除不再使用的 reduced saliency API；
- `statistics.pt` 加 format version；
- `statistics_manifest.json` bump format；
- summary 增加 layout/head/shape 信息；
- single-process Phase EMA 约束继续保留。

---

# 18. `phase_statistics.py` / Phase policy constants

可以保留文件名，但删除旧 reshape 函数和旧 metadata。

新 policy 推荐：

```text
PHASE_TAU_CALIBRATION = ema_channel_abs_max_then_group_max
PHASE_TAU_EMA_FACTOR = 0.99
PHASE_TAU_ACCUMULATOR_DTYPE = float32
PHASE_TAU_CHANNEL_POLICY = native_site_layout_per_channel
PHASE_TAU_REDUCTION_POLICY = within_group_max_after_channel_ema
```

不得存在 active fallback 到旧 SpikingLLM layout。

---

# 19. `calibration.py` 必改项

## `calibration_provenance()`

必须新增并 hash/validate：

```text
calibration_group_size
calibration_grouping_policy = per_head_within_head_groups_v1
statistics_format_version
softmax_site5_grouping_policy
softmax_site5_gif_policy
softmax_site5_clip_policy
```

这样旧/不同 G calibration 不能被错误复用。

## `build_phase_state()`

根据 `layout_kind` 分支：

```text
last_dim
attention_head
attention_softmax
```

删除固定 scalar τ / fixed group_size=-1 假设。

## group reduction

用 preserve-head last-dim reduction。

## `build_site_states()`

必须知道 `site_index`（从 statistics metadata 读取或显式传入）。

Site != 5：

```text
phase + ordinary gif + mtn + optional clip
```

Site 5：

```text
phase + softmax_fixed_q16 gif + mtn
```

永远 no Clip。

## GIF saliency

attention 每个 head 单独排序。

## materialization summary

每个 site 记录至少：

```text
site_index
layout_kind
parameter_layout
configured_group_size
effective_group_size
num_heads
channels_per_head
groups_per_head
phase_tau_shape
gif_policy
gif_parameter_shape
mtn_parameter_shape
clip_policy
clip_state_present
state_sha256
```

Site 5 special-case generic GIF summary 字段。

## collect

删除：

```python
StatisticsStore(max_channels_by_site={5: ...})
```

改为新 `StatisticsStore()`。

继续强制 batch_size=1 和 single-process calibration。

---

# 20. `neurons.py` 必改项

## PhaseSurrogate

删除：

```text
SpikingLLM state 校验
scalar tau 限制
group_size 必须 -1 限制
```

增加 layout/shape/group metadata 校验和广播。

保留：

```text
T
base
max_spikes
surrogate slope runtime override
hard threshold forward
```

## StaticGIF

只处理 ordinary GIF。

使用 grouped qparams 和 per-head mask。

不允许 silent mask pad/truncate。

## SoftmaxFixedGIF

新增独立 module，实现第 11/12 节。

## MultiThresholdNeuron

使用新 grouped parameter expansion。

## Clipper

支持 `[Ng]` 和 `[H,Ng]`；Site 5 不构造 Clipper。

建议提供统一 factory，例如：

```python
def gif_module_from_state(state):
    if state["gif_policy"] == "softmax_fixed_range_u16":
        return SoftmaxFixedGIF(state)
    return StaticGIF(state)
```

controller 和 validator 共享该 factory，避免各自重复分支。

---

# 21. `controller.py` 必改项

- collect API 删除 `phase_activation`；
- Site 5 common Clip special-case；
- `common_clip_enabled=true`：只有 `site_supports_clip(site)` 才加载/执行 Clip；
- GIF module 使用 state policy factory；
- deployment 仍只加载 selected neuron；
- Site 5 deploy_gif 使用 `SoftmaxFixedGIF.temporal()`。

---

# 22. `state_validation.py` 必改项

## ANN-training / `require_clip=True`

Site 1/2/3/4/6/7/8/9/10 要求：

```text
phase
gif
mtn
clip
```

Site 5 要求：

```text
phase
gif
mtn
```

并要求 `clip_state.pt` 不存在。

## conversion clip-free

10 个 site 全部：

```text
phase
gif
mtn
```

任何 Clip 均禁止。

## GIF validation

ordinary GIF 继续验证 qmax30/2-step/15。

Site 5 验证：

```text
range=[0,1]
bits=16
qmax=65535
temporal_steps=2
temporal_policy=quantized_cumulative_difference
```

## Phase validation

验证：

```text
parameter layout
head count
channels_per_head
groups_per_head
configured/effective group_size
tau/v0 shape
```

final RMSNorm grouped Phase 同样验证。

---

# 23. Artifact/schema/temporal version 全部升级

当前 main 大致为：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 3
SITE_STATE_FORMAT_VERSION = 5
CALIBRATION_MANIFEST_FORMAT_VERSION = 6
CONVERSION_METADATA_FORMAT_VERSION = 7
```

实施时以当前最新值为准，各自 +1。

同时新增/提升：

```text
STATISTICS_FORMAT_VERSION
```

示意：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 4
TEMPORAL_IMPLEMENTATION = sparse_llm_temporal_v4
SITE_STATE_FORMAT_VERSION = 6
STATISTICS_FORMAT_VERSION = 2
CALIBRATION_MANIFEST_FORMAT_VERSION = 7
CONVERSION_METADATA_FORMAT_VERSION = 8
```

`SITE_TOPOLOGY_VERSION` 不变，因为仍是相同 10 site。

Temporal/calibration policy metadata 新增：

```text
calibration_grouping_policy = per_head_within_head_groups_v1
phase_tau_calibration = ema_channel_abs_max_then_group_max
phase_tau_channel_policy = native_site_layout_per_channel
phase_tau_reduction_policy = within_group_max_after_channel_ema
softmax_site5_grouping_policy = per_head_full_variable_key_axis
softmax_site5_gif_policy = fixed_range_u16_quantized_cumulative_difference
softmax_site5_clip_policy = disabled
```

删除旧 SpikingLLM statistical-view metadata。

---

# 24. `config.py` / experiment configs

## validation

`group_size`：

```text
-1 or positive integer
```

Site 2/3/4/6 的 `D % G` 和普通 site `C % G` 可在实际 calibration 观察 shape 后验证。

Site 5 不做 divisibility check。

## deployment config

更新：

```text
temporal_implementation
phase_tau_calibration
```

并保留：

```text
phase_tau_ema_factor=0.99
phase_tau_accumulator_dtype=float32
```

## `configs/experiment_matrix.yaml`

默认仍：

```yaml
calibration:
  group_size: -1
```

但文档解释必须改成新的 per-head 含义。

更新 deployment policy 后重新运行：

```bash
python scripts/materialize_configs.py
```

---

# 25. Artifact 路径必须按 group_size 隔离

`group_size` 会改变 Phase/GIF/MTN/Clip 以及 aware ANN forward，因此不能共用路径。

在 `artifacts.py` 增加稳定 helper：

```python
def calibration_group_dirname(group_size):
    return f"calibration_group_size_{int(group_size)}"
```

## 必须隔离

### ANN-training calibration

```text
.../ann_training_calibration/prefix_enabled_.../calibration_group_size_<G>/
```

### vanilla analysis calibration

加入 `calibration_group_size_<G>`。

### post-finetuning conversion calibration

加入 `calibration_group_size_<G>`。

### aware ANN run root

`phase_aware/gif_aware` training 依赖 G，因此 run root 必须加入：

```text
calibration_group_size_<G>
```

保持 slope/warmup/common-clip 等现有隔离规则。

### vanilla/unaware ANN checkpoint

identity ANN training 不依赖 G，因此不要因为 G 改变强制重训 identical checkpoint。

但它们的：

```text
post-finetuning calibration
SNN conversion
SNN evaluation
```

必须按 G 隔离。

### SNN output

所有 mode 的 SNN conversion/evaluation root 必须带 `calibration_group_size_<G>`。

## provenance

所有 calibration/training/conversion/evaluation metadata 显式保存：

```text
calibration_group_size
calibration_grouping_policy
```

禁止仅通过路径推断。

---

# 26. `training.py` / `evaluation.py` / `conversion.py`

## training.py

aware training frozen provenance 增加：

```text
ann_training_calibration_group_size
ann_training_calibration_grouping_policy
statistics_format_version
```

final ANN aware evaluation 必须验证同一 G。

## evaluation.py

forward/result metadata 增加：

```text
calibration_group_size
calibration_grouping_policy
softmax_site5_gif_policy
softmax_site5_clip_applied=false
```

确保：

- phase-aware final ANN 使用 grouped Phase；
- gif-aware final ANN Site 5 使用 Q16；
- Site 5 不使用 Clip；
- SNN Site 5 GIF 使用 special temporal operator。

## conversion.py

`allow_clip_bundle=True` 不再意味着 10/10 site 都必须有 Clip。

aware calibration bundle 合法定义：

```text
9 个 clip-eligible sites 有 Clip
Site 5 无 Clip
```

conversion metadata 增加并严格验证：

```text
calibration_group_size
calibration_grouping_policy
statistics_format_version
softmax_site5_grouping_policy
softmax_site5_gif_policy
softmax_site5_clip_policy
```

---

# 27. `verify_artifacts.py` 与 scripts

`verify_artifacts.py` 必须验证：

- statistics/state/manifest/conversion 新版本；
- grouping metadata；
- group_size path/config/provenance 一致；
- Site 2/6 是 query-head；
- Site 3/4 是 KV-head；
- Site 5 Q16/no-mask/no-clip；
- ANN-training Clip 数为 `num_layers * 9`；
- post-finetuning conversion Clip 数为 0；
- no legacy SpikingLLM Phase metadata。

同时检查以下 scripts 是否含旧 path/schema 假设，必要时同步：

```text
scripts/calibrate_sites.py
scripts/convert_snn.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
scripts/regress_phase_conversion.py
```

不要只改 `snn2/` 而遗漏脚本层 metadata/path 逻辑。

---

# 28. Phase conversion regression

全仓库清理任何：

```text
tau.numel() == 1
scalar tau
group_size must be -1
SpikingLLM statistical view
```

重点检查：

```text
snn2/phase_conversion_regression.py
scripts/regress_phase_conversion.py
tests/test_phase_conversion_regression.py
```

新 regression 支持：

- non-attention grouped τ；
- attention `[H,Ng]` τ；
- Site 5 `[H,1]` τ；
- final RMSNorm grouped τ。

不能为 regression 方便再 global-max/flatten。

---

# 29. Prefix 行为

## Site 3/4 calibration

Prefix positions 可以排除，但 statistics 用原生 KV head。

## Site 3/4 runtime

Prefix KV 必须经过 grouped Phase/GIF/MTN。

## Site 5 calibration

可以排除 Prefix key columns。

## Site 5 runtime ANN/SNN

完整 Softmax tensor，包括 Prefix columns，必须经过 Site 5：

- Phase per-head τ；
- GIF fixed Q16；
- MTN per-head base scale。

---

# 30. 必须更新/新增的测试

本次路径、schema、provenance 改动范围较大，以下测试都必须检查并按需要更新，不能只跑 neuron 单测。

## 核心数学测试

### `tests/test_statistics.py`

新增：

- attention stats 保留 `[H,D]`；
- per-head EMA 不互相污染；
- `H=2,D=4,G=2`：tau 为 `[[2,4],[20,40]]` 等手工结果；
- `G=-1`：每 head 一个 tau，不是跨 head scalar；
- Site 5 两次不同 K/Q 长度仍统计 `[H]`；
- statistics format/version/layout metadata。

删除/重写旧 SpikingLLM statistical-view tests。

### `tests/test_neurons.py`

覆盖：

- Phase `[H,Ng]` broadcasting；
- MTN `[H,Ng]`；
- GIF grouped qparams；
- GIF mask `[H,D]`；
- Clip `[H,Ng]`；
- wrong H/D/shape 报错；
- 无 silent pad/truncate；
- `SoftmaxFixedGIF.forward()` 精确 Q16；
- `SoftmaxFixedGIF.temporal().sum(0) == Q16(incoming.sum(0))`；
- Site 5 special GIF validator；
- ordinary GIF qmax30 仍正常。

### `tests/test_controller_state_loading.py`

覆盖：

- Site 5 common_clip=true 仍不加载 Clip；
- Site 5 GIF factory 加载 SoftmaxFixedGIF；
- stale Site 5 clip 被 validator 拒绝；
- ordinary sites common Clip 仍正常；
- deploy_gif Site 5 temporal。

## Calibration tests

必须检查/更新：

```text
tests/test_calibration_gif.py
tests/test_calibration_profiles.py
tests/test_calibration_topology.py
```

ANN-training Clip count：

```text
num_layers * 9
```

post-finetuning：0。

## GQA test

构造：

```text
Hq=4
Hkv=2
groups=2
```

验证 repeat 后 saliency sum 回 `[Hkv,D]`，Site 3/4 mask/state 均为 KV-head shape。

## Temporal / Prefix tests

检查/更新：

```text
tests/test_temporal_model_integration.py
tests/test_temporal_ops.py
tests/test_temporal_prefix.py
```

覆盖：

- Site 3/4 repeat_kv 前 replacement；
- Prefix columns经过 Site 5；
- Site 5 Q16 temporal；
- tensor shape/dtype/device invariants。

## Path/provenance/schema tests

必须检查/更新：

```text
tests/test_conversion_metadata.py
tests/test_evaluation_paths.py
tests/test_training.py
tests/test_post_finetuning_protocol.py
tests/test_generated_configs.py
tests/test_phase_conversion_regression.py
```

至少覆盖：

- G=-1 与 G=32 calibration path 不同；
- aware run root 不同；
- SNN conversion/eval path 不同；
- vanilla/unaware identity ANN checkpoint不因 G 强制分叉；
- conversion metadata 锁定 G/grouping policy；
- training provenance 锁定 G；
- statistics/state/manifest/conversion legacy version 均拒绝；
- generated config 使用新 deployment policy。

## Config validation

覆盖：

```text
-1 valid
1 valid
positive divisor valid
0 invalid
-2 invalid
D % G != 0 -> calibration error
Site 5 不因 K % G != 0 报错
```

---

# 31. 最低测试命令

先运行：

```bash
pytest -q \
  tests/test_statistics.py \
  tests/test_neurons.py \
  tests/test_controller_state_loading.py \
  tests/test_calibration_gif.py \
  tests/test_calibration_profiles.py \
  tests/test_calibration_topology.py \
  tests/test_temporal_model_integration.py \
  tests/test_temporal_ops.py \
  tests/test_temporal_prefix.py \
  tests/test_phase_conversion_regression.py \
  tests/test_conversion_metadata.py \
  tests/test_evaluation_paths.py \
  tests/test_training.py \
  tests/test_post_finetuning_protocol.py \
  tests/test_generated_configs.py
```

然后：

```bash
pytest -q
```

重新生成 configs：

```bash
python scripts/materialize_configs.py
pytest -q tests/test_generated_configs.py
```

---

# 32. 两组实际 smoke test

## A. `group_size=-1`

Qwen3-1.7B ANN-training calibration 至少检查一层。

Site 2：

```text
Phase tau = [num_attention_heads,1]
GIF scales = [num_attention_heads,1]
GIF mask = [num_attention_heads,head_dim]
MTN base_scale = [num_attention_heads,1]
Clip = [num_attention_heads,1]
```

Site 3/4：

```text
head count = num_key_value_heads
```

Site 5：

```text
Phase tau = [num_attention_heads,1]
MTN base_scale = [num_attention_heads,1]
GIF policy = softmax_fixed_range_u16
clip_state.pt 不存在
```

Site 6 同 Site 2。

Site 1/7/8/9/10：每个 site 一个 group。

final RMSNorm Phase：一个 group。

## B. 正 group_size

根据模型实际维度选择同时能整除 head_dim/hidden/intermediate 的 G，例如 32 或 64。

Site 2/3/4/6：

```text
[H,D/G]
```

Site 1/7/8/9/10：

```text
[C/G] or [I/G]
```

Site 5 始终：

```text
Phase/MTN [H,1]
GIF Q16
no Clip
```

若普通 site dimension 不能整除 G，必须清晰报错。

---

# 33. Artifact/provenance smoke check

同一 model/seed：

```text
G=-1
G=32
```

必须满足：

- calibration root 不同；
- calibration manifest hash/path 不同；
- aware checkpoint root 不同；
- SNN conversion root 不同；
- SNN evaluation root 不同；
- conversion metadata 中 G 不同；
- 不得交叉引用另一 G 的 state hash。

Vanilla/unaware identity ANN checkpoint可以共享，但其 post-finetuning calibration/SNN artifact 必须分 G。

---

# 34. 文档同步

## README.md

删除旧：

```text
SpikingLLM attention reshape
per-channel EMA -> global scalar tau
```

新增：

- native site layout；
- attention per-head；
- group_size head-local grouping；
- Site 5 Q16/no Clip；
- final RMSNorm grouped Phase。

## AGENTS.md

必须重写旧 Phase statistical-view 规则。

新增不可违反规则：

1. `calibration.group_size` 同时控制 ordinary Phase/GIF/MTN/Clip；
2. Site 2/3/4/6 per-head 且只在 head 内 grouping；
3. Site 3/4 原生 KV head；
4. Site 5 忽略 global G；
5. Site 5 GIF fixed `[0,1]` Q16；
6. Site 5 永远 no Clip；
7. Prefix runtime仍经过 Site 3/4/5；
8. Phase EMA FP32、0.99；
9. 不允许重新引入跨-head global τ；
10. ANN-training calibration 对所有 clip-eligible sites 生成 Clip，Site 5 永久例外。

## 实验执行总结.md

写清：

- 默认 G=-1 新含义；
- 改 G 后必须重新 calibration；
- aware training 改 G 必须重新训练；
- vanilla/unaware identity ANN checkpoint不需要因 G 重训，但 conversion calibration/SNN conversion必须重做；
- 旧 statistics/state/conversion artifact 不可复用。

## 代码结构总结.md

遵守项目既有规则：只保留 `2. 目录结构`，每个文件一句功能说明。

更新所有本次职责变化的文件说明。

---

# 35. 全仓库旧语义清理

执行：

```bash
rg -n \
  "spikingllm_identity_input_layout|spikingllm_flatten_attention_heads_before_channel_ema|per_channel_ema_then_global_max|phase_statistical_view|phase_statistical_view_version" \
  .
```

除 `docs/history/` 历史记录外，active code/current docs 不应存在。

检查 scalar τ 假设：

```bash
rg -n "tau.*numel|numel.*tau|scalar tau|scalar τ|group_size.*-1" \
  snn2 scripts tests README.md AGENTS.md 实验执行总结.md
```

检查 Site 5 Clip：

```bash
rg -n "clip_state.pt|require_clip|common_clip" snn2 scripts tests
```

检查 generic GIF metadata没有错误套到 Site 5：

```bash
rg -n "GIF_HIGH_QMAX|GIF_INTEGER_DECOMPOSITION|per_step_qmax|high_qmax" \
  snn2 scripts tests
```

检查 statistics legacy version：

```bash
rg -n "statistics_manifest|statistics.pt|STATISTICS_FORMAT_VERSION" snn2 scripts tests
```

---

# 36. 推荐实施顺序

1. `sites.py`：site policy constants/helper。
2. `phase_statistics.py` / `temporal_ops.py`：新 policy + all schema/version constants。
3. `config.py` / `experiment_matrix.yaml`：config semantics。
4. `stats.py`：raw statistics schema/layout/version。
5. `model_integration.py`：per-head collect、GQA saliency、Site 5 collect。
6. `calibration.py`：grouped state materialization、Site 5 special、provenance。
7. `neurons.py`：layout-aware broadcast、SoftmaxFixedGIF。
8. `controller.py`：Site 5 no Clip / GIF factory。
9. `state_validation.py`：per-site bundle rules。
10. `artifacts.py`：group-size path isolation。
11. `training.py` / `evaluation.py` / `conversion.py`：provenance/path/metadata。
12. `phase_conversion_regression.py` / scripts：去 scalar τ 假设。
13. `verify_artifacts.py`：完整验证。
14. tests。
15. materialize generated configs。
16. docs。
17. 全量 pytest + 两组 smoke test。

---

# 37. 禁止事项

1. 不得改变 10 个 activation site 的位置/数量。
2. 不得把 Site 3/4 replacement 移到 `repeat_kv()` 后。
3. 不得 flatten heads 后做 group_size。
4. 不得让 Site 5 使用 ordinary GIF low/high scale + mask。
5. 不得让 Site 5 生成/加载/执行 Clip。
6. 不得用两个 `[0,15]` chunks 表示 Site 5 Q16 integer。
7. 不得保留旧 SpikingLLM Phase attention reshape fallback。
8. 不得 silent pad/truncate grouped parameters 或 mask。
9. 不得允许旧 statistics/state/manifest/conversion version 加载。
10. 不得因 calibration 排除 Prefix 就让 Prefix runtime bypass Site 3/4/5。
11. 不得把 `surrogate_slope` 写入 Phase state。
12. 不得因为 group_size 改变复制 rotation/data/Prefix 等与 G 无关的 shared artifacts。
13. 不得让 Site 3/4 的 saliency mask 使用 repeat 后 Hq shape。
14. 不得让 final RMSNorm Phase 继续无条件 scalar τ。

---

# 38. 最终验收标准

只有以下全部满足才算完成：

- [ ] Site 2/3/4/6 per-head 独立；
- [ ] Site 3/4 为原生 KV heads；
- [ ] G=-1 在 attention 中为每 head 一组；
- [ ] 正 G 只在 head 内 grouping；
- [ ] Phase ordinary sites 全部受 G 控制；
- [ ] Phase no SpikingLLM flatten/global scalar τ；
- [ ] final RMSNorm Phase 受 G 控制；
- [ ] GIF/MTN/Clip ordinary sites受 G 控制；
- [ ] GIF mask attention shape `[H,D]` 且每 head 独立排序；
- [ ] GQA Site 3/4 saliency sum 回 KV heads；
- [ ] Site 5 Phase `[H,1]`；
- [ ] Site 5 MTN `[H,1]`；
- [ ] Site 5 GIF fixed `[0,1]` Q16；
- [ ] Site 5 GIF temporal = quantized cumulative difference；
- [ ] Site 5 永远 no Clip；
- [ ] Site 5 不受 global group_size；
- [ ] Prefix runtime经过 Site 3/4/5；
- [ ] raw statistics schema/version 已升级；
- [ ] state/manifest/conversion/temporal versions 已升级；
- [ ] calibration provenance记录 G/grouping policy；
- [ ] group-size-dependent artifact path 已隔离；
- [ ] aware training provenance锁定 G；
- [ ] legacy statistics/state/conversion 全部拒绝；
- [ ] scalar τ / SpikingLLM old semantics 已清理；
- [ ] 相关测试全通过；
- [ ] `pytest -q` 全通过；
- [ ] G=-1 smoke test通过；
- [ ] 正 G smoke test通过；
- [ ] README / AGENTS / 实验执行总结 / 代码结构总结已同步。

---

# 39. Codex 最终回复要求

完成后必须简要报告：

1. 修改的核心文件；
2. 新 per-head/group_size 数学规则；
3. Site 3/4 GQA/KV-head 处理；
4. Site 5 Phase/GIF/MTN/Clip 特殊策略；
5. statistics/state/manifest/conversion/temporal version 变化；
6. artifact path/provenance 如何按 G 隔离；
7. 新增/更新的测试；
8. `pytest -q` 最终结果；
9. 两组 smoke test 结果；
10. 明确提示旧 calibration/statistics/conversion artifact 是否必须删除后重跑。

不要只回复“修改完成”。
