# SNN 项目修改方案：Per-Head Grouped Calibration 与 Softmax Site 5 特殊策略

> **历史说明：**
> 本文记录的是当时的 Q16 Site 5 设计，该设计现已被
> `SNN_Site5_GIF_Strict_SpikeLLM_Identity_Modification_Plan.md` 替代。
> 当前实现以 SpikeLLM `n_bits=16` sentinel 的真实 identity 行为为准；
> 本文中的 Q16 / `SoftmaxFixedGIF` / cumulative-difference 内容仅用于历史追溯，
> 不再代表当前实现。

## 0. 任务目标与实施原则

本次修改基于当前 `main` 分支实现，目标是重构 activation calibration / state materialization / ANN replacement / Temporal SNN deployment 的参数粒度，使：

1. Self-Attention 中的多头 activation replacement site 支持 **per-head 独立统计量与独立参数**。
2. `calibration.group_size` 统一控制：
   - Phase；
   - GIF；
   - MTN；
   - Clip；
   - Self-Attention 之外的所有普通 activation replacement site；
   - Self-Attention 的普通多头 site 在 **每个 head 内部** 分组。
3. `calibration.group_size=-1` 的含义改为：
   - 普通非 attention site：整个最后一维 channel dimension 为一组；
   - Self-Attention 普通多头 site：**每个 head 的整个 head dimension 各自为一组**，绝不能再将 `head_num × head_dim` flatten 后共享一组；
   - Softmax Site 5：固定 special policy，不读取全局 `calibration.group_size`，等价于每个 head 的整个可变 key-position 维度为一组。
4. Phase 不再采用当前 SpikingLLM-aligned 的 `flatten attention heads before channel EMA → global max` 统计方式。
5. Softmax Site 5 改成单独实现：
   - Phase：per-head scalar τ；
   - MTN：per-head scalar `base_scale`；
   - GIF：固定 `[0,1]`、显式 16-bit fixed-range quantization；
   - Clip：永远禁用，不生成、不加载、不执行；
   - Site 5 不受配置项 `calibration.group_size` 控制。
6. 保持当前已有的重要实验边界：
   - 仍然是 10 个 activation replacement site；
   - Prefix runtime 仍必须经过 Site 3/4 neuron；
   - calibration 可继续排除 Prefix positions；
   - ANN-aware training 中时间维度仍只在 site 内局部展开后聚合，不能变成跨层 Temporal SNN；
   - SNN deployment 仍必须走 full-temporal `deploy_*` 路径；
   - SNN deployment 永远不使用 common Clip。
7. 本次修改会改变 state 数学语义，**旧 calibration/state/conversion artifact 必须拒绝复用**，不得做兼容性“猜测加载”。

不要只做局部 patch。必须从 statistics → state materialization → runtime broadcasting → controller → validation → conversion provenance → tests → docs 全链路改完整。

---

# 1. 冻结后的新数学语义

## 1.1 `calibration.group_size`

配置值只允许：

```yaml
calibration:
  group_size: -1
```

或：

```yaml
calibration:
  group_size: G   # G 为正整数
```

禁止 `0`，禁止 `< -1`。

对于固定 channel width 为 `C` 的普通 site：

- `group_size=-1`：
  - effective group width = `C`
  - group 数 = 1
- `group_size=G>0`：
  - 要求 `C % G == 0`
  - group 数 = `C / G`
  - 每相邻 `G` 个 channels 共享一套参数

对于 Self-Attention 普通多头 tensor：

\[
x\in\mathbb{R}^{B\times H\times L\times D}
\]

其中 `H` 是该 site 实际 runtime head 数、`D=head_dim`：

- `group_size=-1`：
  - 每个 head 独立；
  - 每个 head 的整个 `D` 为一组；
  - 每个 head 有 1 组；
  - 总参数组形状推荐保存为 `[H, 1]`。
- `group_size=G>0`：
  - 要求 `D % G == 0`；
  - 每个 head 有 `D/G` 组；
  - 参数形状保存为 `[H, D/G]`；
  - **绝不能跨 head 分组**。

注意：`G` 的定义永远是 “组内 channel 数”，不是 “group 数”。

---

# 2. 各 activation site 的新 layout

保持当前 10 site 定义和 `SITE_TOPOLOGY_VERSION`，不要增删 site。

| Site | 位置 | runtime activation 典型形状 | 新统计/参数 layout |
|---|---|---|---|
| 1 | post_input_rmsnorm | `[B,L,C]` | 普通 last-dim grouped |
| 2 | q_post_rope_r3 | `[B,Hq,L,D]` | **query-head 独立，head 内 grouped** |
| 3 | k_post_rope_r3 | `[B,Hkv,L,D]` | **KV-head 独立，head 内 grouped** |
| 4 | v_projection_r2 | `[B,Hkv,L,D]` | **KV-head 独立，head 内 grouped** |
| 5 | post_spiking_softmax | `[B,Hq,Q,K]` | special：每个 query head 一组，忽略 global group_size |
| 6 | post_attention_value_dot_r2 | `[B,Hq,Q,D]` | **query-head 独立，head 内 grouped** |
| 7 | post_mlp_rmsnorm | `[B,L,C]` | 普通 last-dim grouped |
| 8 | post_spiking_silu | `[B,L,I]` | 普通 last-dim grouped |
| 9 | post_mlp_up_proj | `[B,L,I]` | 普通 last-dim grouped |
| 10 | post_mlp_product_r4 | `[B,L,I]` | 普通 last-dim grouped |

额外的 global final RMSNorm Phase state：

- 形状 `[B,L,C]`
- 按普通 last-dim grouped 处理
- 也必须受 `calibration.group_size` 控制

---

# 3. Self-Attention head 语义必须严格区分

## 3.1 Site 2 / 6

使用 `num_attention_heads`：

- Site 2 Q：`H = num_attention_heads`
- Site 6 attention output：`H = num_attention_heads`

## 3.2 Site 3 / 4

必须使用 `num_key_value_heads`：

- Site 3 K：在 `repeat_kv()` 前做 calibration / replacement
- Site 4 V：在 `repeat_kv()` 前做 calibration / replacement
- state 的 head 维必须与原始 KV head 数一致

**不要**为 Site 3/4 在 calibration 中先 `repeat_kv()` 再统计。

否则：
- calibration state 是 query-head 粒度；
- runtime replacement 却发生在 KV-head tensor；
- state 与 runtime layout 不一致。

## 3.3 GQA 下 Site 3/4 GIF saliency

当前 GIF saliency 是在 `repeat_kv()` 后的 query-head tensor 上得到的。

新方案中 Site 3/4 的 GIF mask 必须是：

```text
[Hkv, D]
```

因此必须把 repeat 后的 saliency 聚合回原生 KV head：

假设：

```text
Hq = Hkv * num_key_value_groups
score.shape = [B, Hq, L, D]
```

先 reshape：

```text
[B, Hkv, num_key_value_groups, L, D]
```

然后沿 `num_key_value_groups` 求和：

```text
score_kv = score.reshape(...).sum(dim=2)
```

得到：

```text
[B, Hkv, L, D]
```

这里使用 **sum**，不是简单取第一份，也不要保留重复 head。原因是原始 KV head 被多个 query heads 共同消费，求和对应这些 consumer contributions 对原 KV head 的总敏感度。

Site 3 和 Site 4 都必须做这个处理。

---

# 4. 新 StatisticsStore 语义

当前 `stats.py` 把普通 activation：

```python
activation.reshape(-1, activation.shape[-1])
```

会把 attention head 维聚合掉，因此必须重构。

## 4.1 三种 layout

建议统计 state 明确记录：

```text
last_dim
attention_head
attention_softmax
```

### A. `last_dim`

用于 Site 1/7/8/9/10 和 global final RMSNorm。

原始 channel statistics 形状：

```text
[C]
```

### B. `attention_head`

用于 Site 2/3/4/6。

原始 channel statistics 形状：

```text
[H, D]
```

必须保存 head 维，不允许 flatten 为 `[H*D]` 后统一统计。

### C. `attention_softmax`

用于 Site 5。

不再保存长度为 `K` 的可变位置 statistics。

只保存 per-head 聚合统计：

```text
[H]
```

即每次 forward 对：

```text
[B,H,Q,K]
```

在 `B/Q/K` 维做 reduction，只保留 head。

这样可以完全删除当前 Site 5 的 `max_channels_by_site` / 固定最大 key-position buffer 设计。

---

# 5. Generic statistics 的具体统计方式

## 5.1 普通非 attention site

对 `[B,L,C]`：

- `value_min[C]`
- `value_max[C]`
- `abs_max[C]`
- `sum_abs[C]`
- `sum_sq[C]`
- `saliency_sum[C]`
- `saliency_row_count[C]`

与当前数学意义保持一致。

## 5.2 Site 2/3/4/6

对 `[B,H,L,D]`：

保留：

```text
value_min[H,D]
value_max[H,D]
abs_max[H,D]
sum_abs[H,D]
sum_sq[H,D]
saliency_sum[H,D]
saliency_row_count[H,D]
```

每个 `[h,d]` 的统计样本来自 batch/token 维，不跨 head。

## 5.3 Site 5

对 `[B,H,Q,K]`：

只需要支持 Phase 和 MTN calibration。

保存：

```text
value_min[H]
value_max[H]
abs_max[H]
phase_ema_abs_max[H]
phase_ema_updates[H]
```

其中 generic min/max：

```text
current_min[h] = min over B,Q,K
current_max[h] = max over B,Q,K
```

跨 calibration forward 继续做 dataset-level min/max 累积。

Site 5：

- 不再统计 GIF saliency；
- 不再需要 `saliency_sum[K]`；
- 不再需要 `saliency_row_count[K]`；
- 不再需要 `variable_channels=True`；
- 不再需要 `max_seq_length + prefix_length` 作为固定 statistics width。

---

# 6. Phase 新统计规则

## 6.1 删除 SpikingLLM-aligned attention reshape 语义

当前以下概念必须删除/替换：

```text
spikingllm_identity_input_layout
spikingllm_flatten_attention_heads_before_channel_ema
per_channel_ema_then_global_max
phase_statistical_view(...)
```

不能再在 Site 2/3/4/5/6 为 Phase 单独构造 SpikingLLM Identity-input layout。

Phase 与 generic statistics 仍然可以保留独立 accumulator，但 **layout 必须使用 site 原生 head 结构**。

## 6.2 EMA 顺序

EMA factor 继续保持：

\[
0.99
\]

accumulator 继续固定：

```text
float32
```

第一次更新：

\[
EMA = current\_abs\_max
\]

后续：

\[
EMA \leftarrow 0.99 EMA + 0.01 current\_abs\_max
\]

## 6.3 普通非 attention site

对 `[B,L,C]`：

每次 forward：

\[
m_c=\max_{B,L}|x_{B,L,c}|
\]

得到：

```text
phase_ema_abs_max[C]
```

最终：

- `G=-1`：

\[
\tau=\max_{c=1}^{C} EMA_c
\]

保存 `[1]`

- `G>0`：

每相邻 `G` channels：

\[
\tau_g=\max_{c\in g} EMA_c
\]

保存 `[C/G]`

## 6.4 Site 2/3/4/6

对 `[B,H,L,D]`：

每次 forward：

\[
m_{h,d}=\max_{B,L}|x_{B,h,L,d}|
\]

EMA：

```text
phase_ema_abs_max[H,D]
```

最终只在 **head 内** 做 group max。

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
\tau_{h,g}=\max_{d\in g} EMA_{h,d}
\]

保存：

```text
tau[H,D/G]
```

绝不能再执行跨 `H` 的 global maximum。

## 6.5 Site 5

对 Softmax：

```text
[B,H,Q,K]
```

每次 forward：

\[
m_h=\max_{B,Q,K}|x_{B,h,Q,K}|
\]

然后做 per-head EMA：

```text
phase_ema_abs_max[H]
```

最终：

```text
tau[H,1]
```

Site 5 的：

```text
group_size = -1
```

是 **effective special group size**，含义是每个 head 的整个可变 key-position 维是一组。

它不读取 global `calibration.group_size`。

## 6.6 final RMSNorm Phase

必须按普通 non-attention grouped 规则：

- `G=-1`：scalar τ
- `G>0`：`tau[C/G]`

当前“final Phase 永远 scalar τ”的假设必须删除。

## 6.7 Phase state metadata

推荐新 state 使用类似字段：

```python
{
    "state_kind": "phase",
    "format_version": ...,
    "temporal_implementation_version": ...,
    "parameter_layout": "last_dim_grouped"
        | "attention_head_grouped"
        | "attention_head_scalar",
    "configured_group_size": G,
    "group_size": effective_G,
    "num_heads": ...,
    "channels_per_head": ...,
    "groups_per_head": ...,
    "tau": ...,
    "v0": ...,
    "tau_calibration": "ema_channel_abs_max_then_group_max",
    "tau_ema_factor": 0.99,
    "tau_accumulator_dtype": "float32",
    "tau_channel_policy": "native_site_layout_per_channel",
    "tau_reduction_policy": "within_group_max_after_channel_ema",
}
```

`attention_head_scalar` 用于 Site 5。

不要再保存：

```text
phase_statistical_view
phase_statistical_view_version
spikingllm_identity_input_layout
```

`v0` 继续按当前公式逐参数计算：

\[
v_0 = 0.5\tau 2^{-T}
\]

因此 `v0` 与 `tau` 形状相同。

---

# 7. Generic group reduction

当前 `_group_reduce()` 只适合 1D vector，需要改成“仅沿最后一维分组，同时保留前面的 head 维”。

建议语义：

```python
def group_reduce_last_dim(values, group_size, reduction):
    width = values.shape[-1]
    effective = width if group_size == -1 else group_size
    if width % effective != 0:
        raise ValueError(...)
    grouped = values.reshape(
        *values.shape[:-1],
        width // effective,
        effective,
    )
    return reduce(grouped, dim=-1)
```

这样：

- `[C] -> [C/G]`
- `[H,D] -> [H,D/G]`

Site 5 不走该函数。

---

# 8. GIF 新语义

## 8.1 普通非 Site 5

保持当前 low/high GIF 算法：

- low 4-bit；
- high qmax = 30；
- 2 temporal steps；
- 每步 qmax = 15；
- `low_ratio` / `salient_ratio` 保留；
- high/low qparams 仍由 calibration min/max 生成。

但参数粒度改为新的 grouped layout。

### 非 attention site

```text
low_scale[C/G]
low_zero[C/G]
high_scale[C/G]
high_zero[C/G]
mask_low[C]
```

### Site 2/3/4/6

```text
low_scale[H,D/G]
low_zero[H,D/G]
high_scale[H,D/G]
high_zero[H,D/G]
mask_low[H,D]
```

## 8.2 GIF mask_low

mask 仍然是逐 channel，不受 group_size 合并。

但在 attention Site 2/3/4/6 必须变成 **per-head channel mask**：

```text
mask_low[H,D]
```

`low_ratio` 应当 **每个 head 内独立应用**：

例如：

```text
low_channels_per_head = floor(low_ratio * D)
```

每个 head 根据自己的 saliency 排序后选低比特 channels。

不要把所有 heads 混在一起排序。

Site 3/4 使用前文定义的 GQA saliency 聚合回 KV-head 后再生成 mask。

---

# 9. Softmax Site 5 GIF：固定 [0,1] 16-bit

Site 5 不再使用普通 GIF：

- 不生成 low/high scale；
- 不生成 zero；
- 不生成 `mask_low`；
- 不做 dynamic min/max；
- 不使用 `low_ratio`；
- 不使用 qmax=30 的两段整数表示作为量化本体。

## 9.1 ANN/static forward

固定：

\[
q_{\max}=2^{16}-1=65535
\]

\[
x_q=\frac{\operatorname{round}(65535x)}{65535}
\]

Softmax 理论范围：

\[
0\le x\le1
\]

实现建议：

- 运算在 FP32 完成；
- ANN training 为保证梯度可传播，round 使用 STE；
- 返回原 activation dtype；
- 可以对数值做 `[0,1]` 安全 clamp，但不得重新统计 xmin/xmax。

必须明确：
这采用 SpikeLLM 的 fixed-[0,1] quantization 思路，但本项目是 **显式执行 16-bit fake quantization**。不要在文档中错误声称完全复制其 `n_bits>=16 -> return x` 的具体代码行为。

## 9.2 新 Site 5 GIF state

建议：

```python
{
    "state_kind": "gif",
    "format_version": ...,
    "temporal_implementation_version": ...,
    "gif_policy": "softmax_fixed_range_u16",
    "parameter_layout": "softmax_fixed_range",
    "configured_group_size": G,      # 仅记录 provenance
    "group_size": -1,                # effective special policy
    "group_size_source": "site5_fixed_override",
    "num_heads": H,
    "range_min": 0.0,
    "range_max": 1.0,
    "quantization_bits": 16,
    "qmin": 0,
    "qmax": 65535,
    "scale": 1.0 / 65535.0,
    "zero_point": 0,
    "temporal_steps": 2,
    "temporal_policy": "quantized_cumulative_difference",
}
```

不要给 Site 5 state 填入伪造的：

```text
low_scale
high_scale
mask_low
high_qmax=30
integer_decomposition=...
```

---

# 10. Softmax Site 5 GIF 的 Temporal SNN 语义

这是本次修改的关键点。

当前普通 GIF temporal 是：

```text
high_qmax=30
-> two unsigned chunks
-> 每步 [0,15]
```

该机制不能表达 `[0,65535]`。

因此 Site 5 必须使用独立 temporal operator，**不把 16-bit integer 强行拆成两个 4-bit chunk**。

## 10.1 新 temporal 定义

输入：

```text
incoming[T,B,H,Q,K]
```

其中当前 GIF `T=2`。

先恢复每个时刻累计的 Softmax 概率：

\[
c_t=\sum_{i=0}^{t} incoming_i
\]

对每个累计状态执行 fixed Q16：

\[
\hat c_t = Q_{16}(c_t)
\]

其中：

\[
Q_{16}(x)=\frac{\operatorname{round}(65535x)}{65535}
\]

再转换回 temporal increment：

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

- 输出 temporal shape 与输入完全相同；
- 不使用 low/high GIF qparams；
- 不使用 `[0,15]` chunk decomposition；
- 不要求输出 increment 全为正，因为 temporal softmax 的 difference 本身也可能带符号；
- 最终 cumulative result 必须等于对最终 Softmax probability 执行 Q16 的结果。

建议新增独立模块：

```text
SoftmaxFixedGIF
```

而不是把大量 special if 塞进普通 `StaticGIF` 主路径。

Controller 在读取 `gif_state.pt` 后根据：

```text
gif_policy
```

选择：

- 普通 site -> `StaticGIF`
- Site 5 -> `SoftmaxFixedGIF`

`SoftmaxFixedGIF.temporal_steps` 仍返回 2，以保持全模型 GIF temporal step 一致。

---

# 11. MTN 新语义

## 11.1 普通非 attention site

由 grouped absolute range 构造：

```text
base_scale[C/G]
```

`G=-1` 为 `[1]`。

## 11.2 Site 2/3/4/6

按 head 内 group：

```text
base_scale[H,D/G]
```

## 11.3 Site 5

不读取 global group_size。

每个 head 统计：

```text
value_min[H]
value_max[H]
```

定义：

\[
absolute_h=\max(|min_h|,|max_h|)
\]

沿用当前 MTN 公式：

\[
base\_scale_h=2\cdot absolute_h
\]

保存：

```text
base_scale[H,1]
group_size=-1
parameter_layout="attention_head_scalar"
```

---

# 12. Clip 新语义

## 12.1 Site 1/2/3/4/6/7/8/9/10

ANN-training calibration 仍生成 common Clip state。

Clip 仍是：

```text
intersection(
    Phase representable bound,
    MTN bound,
    GIF low/high representable intersection
)
```

只是参数 shape 变成：

- 普通 site：`[C/G]`
- attention Site 2/3/4/6：`[H,D/G]`

## 12.2 Site 5

永久禁用 Clip。

无论：

```yaml
replacement:
  common_clip_enabled: true
```

还是 false，Site 5 都必须：

- 不生成 `clip_state.pt`
- 不加载 `clip_state.pt`
- 不执行 Clip

如果发现旧 artifact 中：

```text
layer_xxx/site_05_post_spiking_softmax/clip_state.pt
```

新 validator 应当拒绝或要求重新 materialize，不能静默使用。

## 12.3 `include_clip` 的新含义

当前 `include_clip=True` 实际隐含“所有 site 都生成 Clip”。

修改后语义应改为：

> 为所有 **clip-eligible sites** 生成 Clip。

在 `sites.py` 中集中定义，例如：

```python
CLIP_ELIGIBLE_SITE_IDS = (1,2,3,4,6,7,8,9,10)
SOFTMAX_SITE_ID = 5
```

不要在多个文件硬编码重复集合。

---

# 13. Runtime 参数广播必须重构

当前 `_channel_values()` 只支持最后一维的一维 group 参数，需要替换为 layout-aware helper。

建议 state 统一带：

```text
parameter_layout
```

支持至少：

```text
last_dim_grouped
attention_head_grouped
attention_head_scalar
softmax_fixed_range
```

## 13.1 `last_dim_grouped`

输入例如：

```text
[B,L,C]
```

参数：

```text
[Ng]
```

展开成：

```text
[1,1,C]
```

## 13.2 `attention_head_grouped`

输入：

```text
[B,H,L,D]
```

参数：

```text
[H,Ng]
```

每个 group repeat `G` 次后展开：

```text
[1,H,1,D]
```

要求 runtime：

```text
x.shape[1] == num_heads
x.shape[-1] == channels_per_head
```

否则直接报错。

## 13.3 `attention_head_scalar`

用于 Site 5。

输入：

```text
[B,H,Q,K]
```

参数：

```text
[H,1]
```

广播为：

```text
[1,H,1,1]
```

不依赖当前 `K`，因此天然支持可变 sequence length / Prefix length。

## 13.4 GIF mask

增加独立 mask expansion helper：

- non-attention `[C]`
- attention `[H,D]`

不能继续使用当前只按最后一维截断/padding 的逻辑。

对新 state 应严格校验 shape，不允许 silent pad/truncate 修复错误 state。

---

# 14. `model_integration.py` 的具体修改

## 14.1 删除 Phase special statistical view

删除：

```python
from .phase_statistics import phase_statistical_view
```

以及所有：

```python
phase_activation=phase_statistical_view(...)
```

`controller.apply()` / `record_activation()` 不再需要 `phase_activation` 参数。

## 14.2 Site 2

直接对原生：

```text
[B,Hq,L,D]
```

进行 collect。

## 14.3 Site 3 / 4

保留 Prefix/past positions 排除逻辑：

```python
statistics_key = key[..., past_length:, :]
statistics_value = value[..., past_length:, :]
```

但删除为 Phase 做的：

```python
repeat_kv(statistics_key, groups)
repeat_kv(statistics_value, groups)
```

所有 statistics 都按原生 KV head 保存。

runtime replacement 仍然在 `repeat_kv()` 前执行。

## 14.4 Site 3 / 4 saliency

在 repeat 后算出的：

```text
key_score/value_score [B,Hq,L,D]
```

按第 3.3 节规则 sum 回：

```text
[B,Hkv,L,D]
```

再传给 `controller.record_saliency()`。

## 14.5 Site 5 collect

Softmax runtime tensor：

```text
weights[B,Hq,Q,K_total]
```

Prefix runtime 必须仍经过 Site 5 neuron。

但 calibration statistics 继续遵守“可排除 Prefix positions”。

因此 collect 模式：

```python
statistics_weights = (
    weights[..., past_length:]
    if past_length
    else weights
)
controller.record_activation(layer_index, 5, statistics_weights)
```

然后保持完整 `weights` 不变继续 forward。

非 collect 模式：

```python
weights = controller.apply(layer_index, 5, weights)
```

注意：
- training/deployment 对 **完整 softmax tensor，包括 Prefix columns** 执行 Site 5；
- 只有 calibration statistics 可切掉 Prefix columns。

## 14.6 删除 Site 5 saliency

删除当前：

```text
position_score
record_saliency_reduced(site 5)
```

及相关 chunk loop。

Site 5 fixed GIF 不需要 mask，因此不再需要该 saliency。

## 14.7 Site 6

继续保持 head-shaped tensor进入 controller：

```text
[B,Hq,Q,D]
```

不要在 Site 6 前 flatten heads。

---

# 15. `stats.py` 的具体修改

建议彻底重构 `SiteStatistics`：

1. 保存 layout metadata：
   - `site_index`
   - `layout_kind`
   - `num_heads`
   - `channels_per_head`
2. 根据 site 第一次看到的 activation 建立对应 shape。
3. 后续每次 update 严格校验：
   - head 数不变；
   - head_dim 不变；
   - Site 5 允许 Q/K 长度变化；
   - 普通 site channel width 不变。
4. Phase EMA accumulator 与 generic statistics 同 layout，但仍是独立 tensor。
5. Phase accumulator 始终 FP32。
6. Site 5 不再维护 variable-position buffers。
7. 删除：
   - `variable_channels`
   - `max_channels_by_site`
   - Site 5 fixed max-position storage
   - 不再使用的 `update_saliency_reduced()` 路径（如果全仓库无其他调用，直接删除）
8. `statistics.pt` 中必须记录完整 layout metadata。
9. `statistics_summary.json` 中增加：
   - `layout_kind`
   - `num_heads`
   - `channels_per_head`
   - `configured_group_size` 可由 materialization summary 补充
   - Phase EMA shape / update metadata
10. 不能在 summary 中继续使用 “SpikingLLM statistical view” 相关字段。

---

# 16. `phase_statistics.py`

保留该文件也可以，但必须重构语义。

删除：

```text
PHASE_STATISTICAL_VIEW
PHASE_STATISTICAL_VIEW_VERSION
phase_statistical_view()
spikingllm_flatten_attention_heads_before_channel_ema
```

只保留/新增 Phase calibration policy metadata，例如：

```text
PHASE_TAU_CALIBRATION = "ema_channel_abs_max_then_group_max"
PHASE_TAU_EMA_FACTOR = 0.99
PHASE_TAU_ACCUMULATOR_DTYPE = "float32"
PHASE_TAU_CHANNEL_POLICY = "native_site_layout_per_channel"
PHASE_TAU_REDUCTION_POLICY = "within_group_max_after_channel_ema"
```

若常量目前定义在 `temporal_ops.py`，可以统一来源，避免两个文件重复定义。

重点不是文件名，而是全仓库不得再出现旧 SpikingLLM attention reshape 语义。

---

# 17. `calibration.py` 的具体修改

## 17.1 `build_phase_state`

必须读取 statistics layout，并应用新 group_size。

不允许：

```python
tau = phase_stat.float().amax().reshape(1)
group_size = -1
```

这种固定 scalar 逻辑。

分支：

- `last_dim`
- `attention_head`
- `attention_softmax`

Site 5 固定 effective `group_size=-1`。

## 17.2 `_group_reduce`

替换为 preserve-head 的 last-dim group reduction。

## 17.3 `build_site_states`

建议签名中显式获得 `site_index`，或者直接从 statistics metadata 读取。

### Site != 5

构造：

```text
phase
gif
mtn
clip（仅 include_clip 且 clip eligible）
```

### Site == 5

构造：

```text
phase
gif(softmax_fixed_range_u16)
mtn
```

永远不构造 Clip。

## 17.4 GIF saliency ranking

attention head layout：

- saliency `[H,D]`
- 每个 head 单独排序
- `mask_low[H,D]`

普通 layout 保持一维排序。

## 17.5 materialize summary

当前 summary 假设所有 GIF 都有：

```text
gif_base_bits
gif_high_qmax
gif_per_step_qmax
gif_low_ratio
```

必须 special-case Site 5。

推荐 manifest 每 site 至少记录：

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

Site 5：

```text
gif_policy = softmax_fixed_range_u16
clip_policy = disabled
clip_state_present = false
effective_group_size = -1
```

## 17.6 collect

删除：

```python
StatisticsStore(max_channels_by_site={5: ...})
```

改回新：

```python
StatisticsStore()
```

当前 calibration 仍保持：

```text
batch_size = 1
```

因为 Phase EMA 顺序依赖 forward 顺序。

错误信息不要再说：

```text
SpikingLLM Phase EMA calibration requires batch_size=1
```

改为：

```text
Phase EMA calibration is order-dependent and requires batch_size=1
```

---

# 18. `neurons.py` 的具体修改

## 18.1 新 parameter expansion helper

替换 `_channel_values()`。

统一为 state/layout-aware parameter expansion。

Phase / GIF / MTN / Clip 全部共用相同 layout 语义。

## 18.2 `PhaseSurrogate`

删除旧校验：

```text
tau_calibration == spikingllm_ema_channel_abs_max
phase_statistical_view == ...
group_size == -1
tau.numel() == 1
```

新校验应验证：

- state version
- tau calibration policy
- EMA factor
- FP32 accumulator provenance
- parameter_layout
- parameter shape
- num_heads/head_dim/group metadata
- `tau.shape == v0.shape`

支持：
- scalar/grouped non-attention
- per-head grouped
- per-head scalar

forward/temporal 数学循环不需要改变阈值时间规则，只改变 tau/v0 broadcast。

## 18.3 `StaticGIF`

只负责普通 GIF site。

删除对 attention mask 的 silent truncate/pad。

使用严格 layout-aware：
- grouped qparams expansion
- per-head channel mask expansion

## 18.4 新 `SoftmaxFixedGIF`

新增独立 module：

```text
forward(x)
temporal(incoming)
temporal_steps
```

按第 9、10 节实现。

ANN forward：
- Q16 fixed range
- round STE
- 返回原 dtype

Temporal：
- cumulative
- hard Q16
- temporal difference
- 返回原 dtype
- shape 不变

## 18.5 `MultiThresholdNeuron`

使用新的 parameter expansion helper。

## 18.6 `Clipper`

支持：
- `[Ng]`
- `[H,Ng]`

并严格验证 parameter layout。

Site 5 不应被构造为 Clipper。

---

# 19. `controller.py`

## 19.1 collect API

删除：

```text
phase_activation
```

相关参数和传递。

## 19.2 Site 5 Clip special rule

定义集中 helper：

```python
site_supports_clip(site_index)
```

ANN phase/gif mode：

若：

```text
common_clip_enabled=True
```

则：

- Site 5：只加载 phase 或 gif；
- 其他 clip-eligible site：加载 neuron + clip。

apply 时同样：

```python
if common_clip_enabled and site_supports_clip(site_index):
    output = clip(output)
```

## 19.3 GIF factory

不能继续：

```python
"gif": StaticGIF
```

无条件加载。

先读取 state：

- `gif_policy == normal_grouped_gif` -> `StaticGIF`
- `gif_policy == softmax_fixed_range_u16` -> `SoftmaxFixedGIF`

## 19.4 Deployment

仍只加载选中的 neuron state。

Site 5 GIF special module 的 temporal 输出必须仍保持当前 full-temporal contract。

---

# 20. `state_validation.py`

当前：

```python
required = ("phase","gif","mtn","clip") if require_clip else ...
```

必须改成 per-site required state。

## 20.1 ANN-training bundle / `require_clip=True`

对于：

```text
Site 1,2,3,4,6,7,8,9,10
```

要求：

```text
phase
gif
mtn
clip
```

对于 Site 5：

要求：

```text
phase
gif
mtn
```

并明确要求：

```text
clip_state.pt 不存在
```

## 20.2 Conversion clip-free bundle

所有 site：

```text
phase
gif
mtn
```

所有 site 都不允许 Clip。

## 20.3 GIF state validation

普通 GIF：
- 继续检查 qmax=30 / 2-step / 15

Site 5 GIF：
- 检查：
  - fixed `[0,1]`
  - bits=16
  - qmax=65535
  - temporal policy
  - temporal_steps=2
- 不检查 generic GIF low/high 字段

## 20.4 Phase shape validation

增加：
- attention head 数；
- groups_per_head；
- channels_per_head；
- tau/v0 shape；
- effective/configured group size；
- final RMSNorm grouped shape。

---

# 21. `temporal_ops.py` 与 artifact schema version

本次修改不仅改变 calibration state，还改变 Site 5 GIF temporal operator，因此必须 bump temporal implementation。

当前大致为：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 3
SITE_STATE_FORMAT_VERSION = 5
CALIBRATION_MANIFEST_FORMAT_VERSION = 6
CONVERSION_METADATA_FORMAT_VERSION = 7
```

修改为下一版本，例如：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 4
TEMPORAL_IMPLEMENTATION = "sparse_llm_temporal_v4"

SITE_STATE_FORMAT_VERSION = 6
CALIBRATION_MANIFEST_FORMAT_VERSION = 7
CONVERSION_METADATA_FORMAT_VERSION = 8
```

如果当前 main 在实际实施前已有新版本，则不要机械使用这些数字，原则是 **在当前最新值基础上各自 +1**。

`SITE_TOPOLOGY_VERSION` 不变，因为仍是同样 10 site。

## 21.1 temporal policy metadata

删除旧 Phase SpikingLLM statistical-view metadata。

新增：

```text
phase_tau_calibration
phase_tau_channel_policy
phase_tau_reduction_policy
calibration_grouping_policy = "per_head_within_head_groups_v1"
softmax_site5_grouping_policy = "per_head_full_variable_key_axis"
softmax_site5_gif_policy = "fixed_range_u16_quantized_cumulative_difference"
softmax_site5_clip_policy = "disabled"
```

generic GIF qmax=30 metadata继续保留，但必须明确它只适用于 ordinary GIF sites，不适用于 Site 5。

---

# 22. `config.py`

## 22.1 group_size validation

增加：

```text
group_size == -1
or
group_size > 0
```

拒绝：

```text
0
-2
-3 ...
```

不要在 config load 阶段假设所有模型维度，因为此时未必已加载 model config。

真正的：

```text
C % G == 0
D % G == 0
```

可以在 calibration 第一次观察到对应 site shape 时严格检查，并给出：

```text
layer
site
head_dim/channel_dim
configured group_size
```

的清晰报错。

Site 5 不做 divisibility check。

## 22.2 deployment config

将：

```yaml
temporal_implementation: sparse_llm_temporal_v3
phase_tau_calibration: spikingllm_ema_channel_abs_max
```

更新到新 policy。

如果仍保留：

```yaml
phase_tau_ema_factor: 0.99
phase_tau_accumulator_dtype: float32
```

继续严格校验。

---

# 23. `configs/experiment_matrix.yaml` 与 generated configs

统一修改所有实验：

```yaml
deployment:
  temporal_implementation: sparse_llm_temporal_v4
  phase_tau_calibration: ema_channel_abs_max_then_group_max
```

保留：

```yaml
calibration:
  group_size: -1
```

默认实验仍使用 `-1`。

这里的新含义必须在文档中写清楚：

- attention：per-head full head_dim；
- non-attention：full last dimension；
- Site 5：忽略此值，固定 per-head full variable key axis。

重新运行：

```bash
python scripts/materialize_configs.py
```

生成所有 configs。

---

# 24. Artifact 路径必须加入 group_size 隔离

这是必须修改项。

现在 `group_size` 将改变：

- Phase state
- GIF state
- MTN state
- Clip state
- phase_aware/gif_aware ANN forward
- final ANN checkpoint（aware mode）
- SNN conversion/evaluation

因此不同 group_size 绝不能落到同一 artifact path。

## 24.1 新路径 helper

在 `artifacts.py` 增加例如：

```python
def calibration_group_dirname(group_size):
    return f"calibration_group_size_{int(group_size)}"
```

## 24.2 必须隔离的路径

### shared ANN-training calibration

当前：

```text
.../ann_training_calibration/prefix_enabled_.../
```

改为：

```text
.../ann_training_calibration/prefix_enabled_.../calibration_group_size_<G>/
```

### vanilla analysis calibration

加入：

```text
calibration_group_size_<G>
```

### post-finetuning conversion calibration

加入：

```text
calibration_group_size_<G>
```

### aware ANN run root

`phase_aware` / `gif_aware` 的训练 forward 会受 group_size 影响，因此 run root 必须加入：

```text
calibration_group_size_<G>
```

位置可以放在 run variant 后、phase slope/warmup 前，保持稳定一致。

### vanilla / unaware ANN checkpoint

identity ANN training 不依赖 group_size，因此不要强制为每个 group_size 重训一套 identical checkpoint。

但其：

- post-finetuning calibration；
- conversion；
- SNN evaluation

必须按 group_size 隔离。

### SNN conversion/evaluation root

无论 mode，所有 SNN 输出都应带：

```text
calibration_group_size_<G>
```

避免不同 grouping 的 conversion/eval 相互覆盖。

## 24.3 provenance

training/conversion/evaluation metadata 显式保存：

```text
calibration_group_size
calibration_grouping_policy
```

不能只依赖路径猜测。

---

# 25. `training.py`

重点检查 aware training frozen provenance。

新增/校验：

```text
ann_training_calibration_group_size
ann_training_calibration_grouping_policy
```

phase_aware/gif_aware：

- checkpoint 必须和同一 group_size 的 ANN-training calibration bundle 绑定；
- final ANN evaluation 必须加载相同 grouping state；
- 不允许当前 config G 与训练记录 G 不一致。

Vanilla/unaware identity training不需要 calibration state参与 forward，但 conversion阶段仍按当前 G 选择对应 post-finetuning calibration。

---

# 26. `evaluation.py`

需要确保：

1. final ANN aware evaluation：
   - 从训练 provenance 加载同一个 grouped state；
   - Site 5 Clip 永远不执行；
   - Site 5 GIF 使用 fixed Q16。
2. SNN evaluation：
   - grouped Phase/GIF/MTN 正确 broadcast；
   - Site 5 GIF 使用特殊 temporal operator。
3. forward metadata 写入：
   - `calibration_group_size`
   - `calibration_grouping_policy`
   - `softmax_site5_gif_policy`
   - `softmax_site5_clip_applied=false`

---

# 27. `conversion.py`

## 27.1 calibration bundle validation

当前 `allow_clip_bundle=True` 不能再意味着每个 site 都必须存在 clip。

必须调用新 per-site validator。

## 27.2 provenance

conversion metadata 增加：

```text
calibration_group_size
calibration_grouping_policy
softmax_site5_grouping_policy
softmax_site5_gif_policy
softmax_site5_clip_policy
```

并在 validate conversion metadata 时严格比对。

## 27.3 aware bundle

允许 ANN-training calibration bundle 中：

- 9 个 clip-eligible sites 存在 Clip
- Site 5 没有 Clip

这仍然属于合法 `ann_training_with_common_clip` bundle。

不要因为 Site 5 无 Clip 把整个 bundle 判断为 “clip-free”。

---

# 28. `verify_artifacts.py`

同步修改验证规则：

1. ANN-training calibration：
   - 9 个 site 有 Clip；
   - Site 5 无 Clip；
2. conversion calibration：
   - 10 site 全部无 Clip；
3. state version 必须是新版本；
4. grouping metadata 完整；
5. attention state 参数 shape 与 per-head policy 相符；
6. Site 3/4 标记为 KV-head layout；
7. Site 5：
   - GIF fixed u16；
   - no clip；
   - no mask；
8. group_size 与 path / config / manifest / training result / conversion metadata 一致；
9. 不允许 legacy SpikingLLM Phase view metadata。

---

# 29. Phase conversion regression 相关代码

必须全仓库搜索任何默认：

```text
tau.numel() == 1
group_size == -1
scalar tau
spikingllm
phase_statistical_view
```

尤其检查：

```text
snn2/phase_conversion_regression.py
scripts/regress_phase_conversion.py
tests/test_phase_conversion_regression.py
```

新的 Phase regression 必须支持：

- non-attention grouped τ；
- attention `[H,Ng]` τ；
- Site 5 `[H,1]` τ；
- final norm grouped τ。

不要在 regression code 为了方便重新把 per-head tau flatten/global max。

Regression 应验证 state-expanded threshold 与 runtime element 对齐。

---

# 30. `sites.py`

保留 10-site topology。

新增集中定义：

```text
ATTENTION_HEAD_GROUPED_SITE_IDS = {2,3,4,6}
SOFTMAX_SITE_ID = 5
CLIP_ELIGIBLE_SITE_IDS = {1,2,3,4,6,7,8,9,10}
```

可再加 helper：

```text
is_attention_head_grouped_site()
is_softmax_site()
site_supports_clip()
```

所有 calibration/controller/validator 使用这些 helper，避免散落 magic numbers。

---

# 31. state schema 推荐

## 31.1 attention Phase example

假设：

```text
H=32
D=128
G=32
```

则：

```python
{
    "parameter_layout": "attention_head_grouped",
    "configured_group_size": 32,
    "group_size": 32,
    "num_heads": 32,
    "channels_per_head": 128,
    "groups_per_head": 4,
    "tau": Tensor[32,4],
    "v0": Tensor[32,4],
}
```

## 31.2 attention G=-1

```python
{
    "parameter_layout": "attention_head_grouped",
    "configured_group_size": -1,
    "group_size": -1,
    "num_heads": 32,
    "channels_per_head": 128,
    "groups_per_head": 1,
    "tau": Tensor[32,1],
}
```

## 31.3 non-attention G=128

若 `C=4096`：

```text
tau[32]
low_scale[32]
base_scale[32]
clip.lower[32]
```

## 31.4 Site 5

若 `H=32`：

Phase：

```text
tau[32,1]
```

MTN：

```text
base_scale[32,1]
```

GIF：

```text
无可学习/校准 qparams
```

Clip：

```text
不存在
```

---

# 32. Prefix 与可变长度必须保持的行为

## 32.1 Site 3/4

Calibration：

- Prefix positions 可以排除；
- statistics 使用原生 KV heads。

Runtime：

- Prefix K/V 必须仍通过 Site 3/4 neuron；
- 不能因为 calibration 排除 Prefix 就 bypass。

## 32.2 Site 5

Calibration：

- 可以排除 Prefix key columns；
- Phase/MTN statistics 只对非 Prefix columns 更新。

Runtime ANN/SNN：

- Site 5 对完整 Softmax tensor执行；
- 包括 Prefix columns。

fixed Q16 GIF 对 Prefix columns 使用完全相同 `[0,1]` 映射。

---

# 33. 必须更新的测试

不要只改现有测试以“通过”，必须新增针对新数学语义的测试。

## 33.1 `tests/test_statistics.py`

删除/重写旧测试：

```text
test_phase_statistical_view_matches_spikingllm_layout
test_phase_tau_is_global_max_after_per_channel_ema
```

新增：

### A. attention stats preserve heads

输入：

```text
[B=2,H=3,L=4,D=5]
```

断言：

```text
value_min.shape == [3,5]
phase_ema_abs_max.shape == [3,5]
```

不同 head 注入不同极值，确认不互相污染。

### B. Phase per-head EMA

构造两个 head 不同数据，手工验证：

```text
EMA[h,d]
```

### C. Phase grouped tau

例如：

```text
H=2,D=4,G=2
```

手工给 EMA：

```text
[[1,2,3,4],
 [10,20,30,40]]
```

期望：

```text
tau =
[[2,4],
 [20,40]]
```

### D. G=-1

期望：

```text
[[4],
 [40]]
```

不是 scalar 40。

### E. Site 5 variable length

先：

```text
[B,2,4,4]
```

再：

```text
[B,2,7,7]
```

统计 shape 始终：

```text
[2]
```

不会因为 K 变化报错。

## 33.2 `tests/test_neurons.py`

新增：

1. Phase `[H,Ng]` broadcasting；
2. MTN `[H,Ng]` broadcasting；
3. GIF qparams `[H,Ng]`；
4. GIF mask `[H,D]`；
5. Clipper `[H,Ng]`；
6. 错误 head 数必须报错；
7. 错误 head_dim 必须报错；
8. 禁止 silent mask padding/truncation；
9. `SoftmaxFixedGIF.forward()` 精确匹配：

\[
round(65535x)/65535
\]

10. `SoftmaxFixedGIF.temporal()`：

```text
temporal.sum(0) == fixed_q16(incoming.sum(0))
```

11. Site 5 GIF state 不含 generic qmax30 fields 也能合法加载；
12. 普通 GIF 仍拒绝错误 qmax。

## 33.3 `tests/test_controller_state_loading.py`

新增：

- common_clip=true 时 Site 5 phase 只加载 phase；
- Site 5 gif 只加载 SoftmaxFixedGIF；
- Site 5 不要求 clip_state；
- 若 Site 5 存在 stale clip，validator 拒绝；
- 其他 site 仍要求 Clip；
- deploy_gif Site 5 使用 special temporal module。

## 33.4 `tests/test_calibration_profiles.py`

ANN-training profile：

```text
clip_state_count = num_layers * 9
```

而不是 `num_layers * 10`。

Post-finetuning：

```text
clip_state_count = 0
```

## 33.5 `tests/test_calibration_topology.py`

site topology 仍为 10，不改变。

但 state presence rules更新。

## 33.6 `tests/test_temporal_model_integration.py`

覆盖：

- Site 3/4 state 在 repeat_kv 前应用；
- Site 5 special GIF temporal；
- head shape 保持；
- Prefix columns仍通过 Site 5。

## 33.7 新 GQA test

构造：

```text
Hq=4
Hkv=2
groups=2
```

手工设置 repeat 后 saliency，验证 sum 回：

```text
[Hkv=2,D]
```

且 Site 3/4 mask 生成使用 KV-head shape。

## 33.8 group_size validation test

覆盖：

```text
-1 -> valid
1 -> valid
D divisor -> valid
0 -> invalid
-2 -> invalid
D % G != 0 -> calibration fail
```

Site 5 在 `G` 不整除 K 时仍不受影响。

## 33.9 artifact path test

不同：

```text
group_size=-1
group_size=32
```

必须得到不同：

- calibration path；
- aware run path；
- SNN conversion/evaluation path。

Vanilla/unaware ANN identity checkpoint是否保持共享按第 24 节规则断言。

---

# 34. 文档必须同步修改

## 34.1 `README.md`

删除：

```text
Phase τ 按 SpikingLLM statistical view reshape
per-channel EMA 后 global max scalar τ
```

改为：

- Phase 按 site native layout；
- attention per-head；
- group_size 控制 head 内 grouping；
- Site 5 special per-head；
- Site 5 GIF fixed Q16；
- Site 5 no Clip。

## 34.2 `AGENTS.md`

当前规则 6 必须重写。

旧：

```text
Site 2/3/4/5/6 Phase EMA 使用 SpikingLLM-aligned statistical view
```

删除。

新增不可违反规则：

1. `calibration.group_size` 对 Phase/GIF/MTN/Clip 统一生效；
2. attention Site 2/3/4/6 必须 per-head、只在 head 内 grouping；
3. Site 3/4 使用原生 KV head；
4. Site 5 忽略 global group_size；
5. Site 5 GIF fixed `[0,1]` Q16；
6. Site 5 永远 no Clip；
7. Prefix runtime仍通过 Site 3/4/5；
8. Phase EMA FP32、factor=0.99；
9. 不允许重新引入跨-head global τ。

当前“ANN-training calibration 始终生成 clip_state.pt”的规则也要改成：

> ANN-training calibration 为所有 clip-eligible sites 生成 `clip_state.pt`；Site 5 永远除外。

## 34.3 `实验执行总结.md`

写清：

- 默认 `group_size=-1` 新含义；
- 如果改 G，必须重新 calibration；
- aware training若 G 改变必须重新训练；
- unaware/vanilla identity ANN checkpoint不需要因为 G 改变重训，但 conversion calibration/SNN conversion要重做；
- 旧 v3/v5/v6/v7 artifact 不可复用。

## 34.4 `代码结构总结.md`

遵守当前 AGENTS 要求：

- 只保留 `2. 目录结构`
- 每个文件只一句功能说明

如果没有新增文件，只更新：
- `phase_statistics.py`
- `stats.py`
- `calibration.py`
- `neurons.py`
- `controller.py`
- `model_integration.py`
- `state_validation.py`
- `temporal_ops.py`
- `artifacts.py`
- `training.py`
- `evaluation.py`
- tests 说明

---

# 35. 必须全仓库清理的旧语义

实施后执行：

```bash
rg -n \
  "spikingllm_identity_input_layout|spikingllm_flatten_attention_heads_before_channel_ema|per_channel_ema_then_global_max|phase_statistical_view|phase_statistical_view_version" \
  .
```

除非仅出现在 `docs/history/` 的历史文档中，否则代码、当前 README、AGENTS、当前实验文档中都不应再存在。

再检查：

```bash
rg -n "tau.*numel|numel.*tau|group_size.*-1|scalar tau|scalar τ" snn2 scripts tests README.md AGENTS.md 实验执行总结.md
```

人工确认没有任何 runtime/validator/regression 仍假设 Phase τ 必须 scalar。

再检查：

```bash
rg -n "clip_state.pt|require_clip|common_clip" snn2 scripts tests
```

确认 Site 5 special rule 全链路一致。

再检查：

```bash
rg -n "GIF_HIGH_QMAX|GIF_INTEGER_DECOMPOSITION|per_step_qmax|high_qmax" snn2 scripts tests
```

确认 generic GIF policy 没有错误套到 Site 5。

---

# 36. 实施顺序

建议 Codex 严格按下面顺序执行，避免半成品状态混乱。

## Step 1：定义 policy/constants/version

修改：

```text
sites.py
phase_statistics.py
temporal_ops.py
config.py
experiment_matrix.yaml
```

先把新语义常量确定。

## Step 2：重构 StatisticsStore

修改：

```text
stats.py
model_integration.py
```

先让 statistics 能正确保留 per-head shape。

为 Site 3/4 加 GQA saliency reduction。

删除 Site 5 variable-position statistics/saliency。

## Step 3：重构 state materialization

修改：

```text
calibration.py
```

实现：
- grouped Phase；
- grouped GIF；
- grouped MTN；
- grouped Clip；
- Site 5 special states。

## Step 4：重构 runtime neuron broadcasting

修改：

```text
neurons.py
controller.py
```

增加：
- layout-aware expansion；
- SoftmaxFixedGIF；
- Site 5 no Clip。

## Step 5：validation / conversion

修改：

```text
state_validation.py
conversion.py
verify_artifacts.py
```

完成 state presence / metadata / version 规则。

## Step 6：artifact path / provenance

修改：

```text
artifacts.py
training.py
evaluation.py
```

加入 group_size 隔离和 provenance。

## Step 7：phase regression

检查并修改：

```text
phase_conversion_regression.py
regress_phase_conversion.py
```

去掉 scalar τ 假设。

## Step 8：tests

先跑小单测，再全量。

## Step 9：docs

最后统一更新 README / AGENTS / 实验执行总结 / 代码结构总结。

---

# 37. 最低测试命令

先运行：

```bash
pytest -q \
  tests/test_statistics.py \
  tests/test_neurons.py \
  tests/test_controller_state_loading.py \
  tests/test_calibration_profiles.py \
  tests/test_calibration_topology.py \
  tests/test_temporal_model_integration.py \
  tests/test_temporal_ops.py \
  tests/test_phase_conversion_regression.py
```

再运行：

```bash
pytest -q
```

然后重新生成配置：

```bash
python scripts/materialize_configs.py
```

再运行 generated config tests：

```bash
pytest -q tests/test_generated_configs.py
```

---

# 38. 必须做的最小实际 calibration smoke test

至少使用 Qwen3-1.7B 跑一次默认：

```yaml
calibration:
  group_size: -1
```

完成 ANN-training calibration 后检查任意一层：

## Site 2

应满足：

```text
phase tau shape = [num_attention_heads, 1]
GIF scale shape = [num_attention_heads, 1]
GIF mask shape = [num_attention_heads, head_dim]
MTN base_scale shape = [num_attention_heads, 1]
Clip lower/upper shape = [num_attention_heads, 1]
```

## Site 3/4

应满足：

```text
head dimension = num_key_value_heads
```

不是 `num_attention_heads`。

## Site 5

应满足：

```text
phase tau = [num_attention_heads,1]
mtn base_scale = [num_attention_heads,1]
gif policy = softmax_fixed_range_u16
clip_state.pt 不存在
```

## Site 6

同 Site 2：

```text
[num_attention_heads,1]
```

## Site 1/7/8/9/10

`G=-1` 时各参数仍为单 group。

---

# 39. 再做一个正 group_size smoke test

选择一个能整除：

- hidden dimension；
- intermediate dimension；
- head_dim；

的 G，例如先根据模型实际维度选择 `G=32` 或 `G=64`。

要求：

- Site 2/3/4/6：
  - shape `[H, D/G]`
- Site 1/7/8/9/10：
  - shape `[C/G]` 或 `[I/G]`
- Site 5：
  - 与 `G=-1` 一样仍是 `[H,1]`
  - GIF 仍 fixed Q16
  - no Clip

如果某普通 site 的 channel width 不能整除 G，应明确报错并指出 layer/site/dimension/G。

---

# 40. 数值回归验收

必须确保：

## 40.1 Identity 路径

`vanilla` / `unaware` final ANN identity 语义不因本次 calibration 重构改变。

## 40.2 aware ANN

`phase_aware`：

- Phase grouped state正确；
- slope仍只作为 runtime backward parameter；
- Site 5 no Clip。

`gif_aware`：

- ordinary site 使用 grouped GIF；
- Site 5 使用 Q16 fixed GIF；
- Site 5 no Clip。

## 40.3 Temporal SNN

Phase/GIF/MTN：

- T 仍跨层传播；
- Site 3/4 Prefix runtime仍通过 neuron；
- Site 5 Prefix columns仍通过 Site 5 neuron；
- GIF Site 5 temporal sum与 static Q16 final值一致。

## 40.4 Artifact provenance

同一 model / seed 下：

```text
G=-1
G=32
```

不得引用同一 calibration state hash/path。

aware checkpoint也不得跨 G 复用。

---

# 41. 不要做的事情

1. 不要改变 10 个 activation site 的位置。
2. 不要把 Site 3/4 移到 `repeat_kv()` 后。
3. 不要把 head flatten 后再按 group_size 分组。
4. 不要让 Site 5 使用普通 GIF `low/high scale + mask`。
5. 不要让 Site 5 生成 Clip。
6. 不要让 Site 5 的 16-bit integer 强行使用现有两步 `[0,15]` decomposition。
7. 不要保留旧 SpikingLLM attention Phase reshape 作为隐藏 fallback。
8. 不要 silent pad/truncate 不匹配的 grouped parameter 或 GIF mask。
9. 不要允许旧 state version 继续加载。
10. 不要因为 Site 5 calibration 排除 Prefix columns，就让 Prefix runtime bypass Site 5。
11. 不要改变 `surrogate_slope` 只在 Phase-aware ANN runtime 注入的现有规则。
12. 不要因为 group_size 改变而把 shared Prefix / rotation / data manifest 无意义复制；只隔离真正依赖 group_size 的 artifact。

---

# 42. 最终完成标准

只有同时满足以下条件，本次修改才算完成：

- [ ] Self-Attention Site 2/3/4/6 全部 per-head 独立；
- [ ] Site 3/4 使用原生 KV heads；
- [ ] `group_size=-1` 在 attention 中表示每 head 一组；
- [ ] 正 G 只在 head 内分组；
- [ ] Phase 所有普通 site 都受 group_size 控制；
- [ ] Phase 不再使用 SpikingLLM attention flatten/global scalar τ；
- [ ] final RMSNorm Phase 也受 group_size 控制；
- [ ] GIF/MTN/Clip ordinary sites 都受 group_size 控制；
- [ ] GIF attention mask_low 为 `[H,D]` 且每 head 独立排序；
- [ ] GQA Site 3/4 saliency 聚合回 KV head；
- [ ] Site 5 Phase 为 per-head τ；
- [ ] Site 5 MTN 为 per-head base_scale；
- [ ] Site 5 GIF 为 fixed `[0,1]` Q16；
- [ ] Site 5 GIF temporal 使用 quantized cumulative difference；
- [ ] Site 5 永远 no Clip；
- [ ] Site 5 不受 global group_size 影响；
- [ ] Prefix runtime 仍通过 Site 3/4/5；
- [ ] statistics/state/manifest/conversion schema version 已升级；
- [ ] temporal implementation version 已升级；
- [ ] group_size-dependent artifact path 已隔离；
- [ ] aware training provenance锁定 group_size；
- [ ] 所有 scalar τ 假设已清理；
- [ ] 所有旧 SpikingLLM statistical-view 当前代码表述已清理；
- [ ] 单测全通过；
- [ ] 两组 group_size smoke test 通过；
- [ ] README / AGENTS / 实验执行总结 / 代码结构总结 已同步。

---

# 43. Codex 最后需要输出的修改总结

完成代码后，请在终端最终回复中简要列出：

1. 修改了哪些核心文件；
2. 新的 per-head/group_size 统计规则；
3. Site 5 special policy；
4. GQA Site 3/4 如何处理；
5. artifact/version/path 如何变化；
6. 新增/更新了哪些测试；
7. `pytest -q` 最终结果；
8. 是否需要删除旧 calibration/conversion artifact 后重新运行。

不要只说“实现完成”，必须明确报告上述结果。
