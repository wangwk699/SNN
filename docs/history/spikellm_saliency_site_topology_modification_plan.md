# SNN 项目修改方案：GIF Salient Channels 严格对齐 SpikeLLM + Site 3/4/6 Attention Topology 重构

> 目标仓库：`https://github.com/wangwk699/SNN`  
> 目标分支：`main`  
> 本文档面向部署在服务器上的 Codex。**假设 Codex 没有此前对话上下文，必须仅依据本文档完成修改。**
>
> 参考实现：
>
> - SpikeLLM 官方仓库：`https://github.com/Xingrun-Xing2/SpikeLLM`
>   - `spike_driven_quant/spike_linear.py`
>   - `spike_driven_quant/spike_matmul.py`
>   - `models/spike_llama_layer.py`
>   - `main.py`
> - SpikingLLM Phase 参考实现：  
>   `https://github.com/njzhenghy/SpikingLLM/blob/main/phase/phase_layer.py`
>   - 重点参考 `phaseSnnSdpaLlamaAttention`
>
> **本文档中的决定均为最终决定，不要重新设计算法，不要擅自改动下文明确要求保留的实验设定。**

---

## 1. 修改目标

本轮修改包含两个相互关联的目标。

### 1.1 GIF salient-channel 计算与 SpikeLLM 官方代码对齐

在本项目保留自身 static quantization、grouped quantization、训练流程等实验设定的前提下，使 **GIF salient channels 的敏感度公式、计算精度、channel ranking、threshold/tie 规则以及各 operator 的 mask 语义** 与 SpikeLLM 官方实现严格对齐。

这里“对齐 SpikeLLM”只针对 **salient-channel 计算与 mask 选择**，不要求把本项目改造成 SpikeLLM 的完整 OmniQuant 流程。

### 1.2 重构 Attention 中 Site 3 / Site 4 / Site 6 的 topology

最终采用以下方案：

- **Site 3 / Site 4：方案 B**
  - GIF、Phase、MTN 三种 neuron 全部移动到 `repeat_kv` **之后**。
  - GIF 的 Site 3/4 参考 SpikeLLM 官方 K/V `SpikeQuantMatMul` 语义。
  - Phase/MTN 的 Site 3/4 topology 参考 SpikingLLM 的 `phaseSnnSdpaLlamaAttention`：
    `repeat_kv -> merge heads -> neuron -> restore heads -> attention matmul`。
- **Site 6**
  - `attn_output = seq_matmul(attn_weights, value_states)` / ANN 对应 matmul 得到 head-wise attention output 后，
    **必须先 merge heads，再进入 Site 6**。
  - Site 6 从此已经没有独立 `H` 维度，不再使用 per-head 参数布局，只保留 `group_size`。

---

# 2. 不允许修改的实验设定

以下内容是明确要求保留的，不要为了“更像 SpikeLLM”而修改。

## 2.1 GIF 仍然使用 static quantization

保持本项目现有 static calibration / static GIF 方案。

**不要**改成 SpikeLLM 官方运行时 per-token dynamic quantization。

## 2.2 保留当前 GIF integer range

保持：

```text
base_bits = 4
add_bits = 1
low_qmax = 15
high_qmax = 30
GIF temporal_steps = 2
per_step_qmax = 15
```

即继续使用本项目当前可被两个 `[0, 15]` timestep 精确分解的：

```text
high_qmax = 30
```

**不要**改成 SpikeLLM fake-quant 的 31。

## 2.3 保留 `group_size`

`calibration.group_size` 的定义与用户配置不变。

其中：

- 对仍属于 per-head layout 的 site：继续按每个 head 内部的 `group_size` 分组。
- 对普通 last-dim site：继续沿最后一维按 `group_size` 分组。
- `group_size = -1` 仍表示当前逻辑 channel dimension 内全部 channel 共用一组参数。

## 2.4 Site 3 / Site 4 仍保留 per-head + group_size 参数布局

虽然 Site 3/4 的 **实际 replacement tensor** 在 ANN 中将是 `[B,L,HD]`，SNN deployment 的逻辑 temporal tensor 将是 `[T,B,L,HD]`，但 Site 3/4 的参数仍必须保留 logical per-head 结构：

```text
[H, D / group_size]
```

适用于：

- Phase `tau`
- Phase `v0`
- MTN `base_scale`
- GIF `low_scale / low_zero`
- GIF `high_scale / high_zero`
- common Clip `lower / upper`

这里的 `H` 是 **`repeat_kv` 之后的 attention heads 数量**，即 `num_attention_heads`，不是 `num_key_value_heads`。

## 2.5 Site 6 不再保留 per-head

这是 Site 6 与 Site 3/4 的关键区别。

Site 6 进入 replacement site 前已经完成：

```text
H,L,D -> L,H,D -> L,HD
```

因此 Site 6 只把 `HD` 看作普通最后一维 channel space。

Site 6 参数布局必须改成：

```text
[HD / group_size]
```

不再有 `[H, ...]` 结构。

适用于：

- Phase `tau`
- Phase `v0`
- MTN `base_scale`
- GIF `low_scale / low_zero`
- GIF `high_scale / high_zero`
- GIF salient mask
- common Clip `lower / upper`

若：

```yaml
calibration:
  group_size: -1
```

则 Site 6 的整个 `HD` 共享一个参数 group。

## 2.6 保留当前 calibration 数据流程

不要改：

- calibration 数据来源；
- sample 数量；
- Prefix calibration protocol；
- rotation calibration protocol；
- 当前 whole-model calibration 流程；
- 当前 token/sample reduction 逻辑，除非是为了支持下文明确要求的 role-specific saliency state。

本轮只要求 salient score 的 **operator formula / dtype / channel mask selection** 与 SpikeLLM 对齐。

## 2.7 暂时不要修改全参数 ANN fine-tuning

本项目现有 full-parameter ANN fine-tuning 保留。

不要改成 SpikeLLM 的 LET/LWC/OmniQuant 训练方式。

---

# 3. 关于维度 `T` 的强制约定

这是本轮实现中必须严格遵守的约定。

## 3.1 ANN calibration / ANN fine-tuning 中没有外部 `T` 维

ANN 前向传播的 replacement-site tensor **不能人为增加 leading T dimension**。

例如：

### Site 3 / Site 4

ANN 实际 replacement tensor：

```text
[B, L, HD]
```

不是：

```text
[T, B, L, HD]
```

### Site 6

ANN 实际 replacement tensor：

```text
[B, L, HD]
```

不是：

```text
[T, B, L, HD]
```

Phase/GIF surrogate/neuron 可以在 **replacement module 内部**使用自己的 `T` 或 temporal coding 逻辑，但对 ANN 模型其输入输出 shape 必须仍与普通 ANN tensor 一致。

## 3.2 只有 SNN conversion/deployment 才有真实 temporal dimension

SNN deployment 逻辑 tensor 才是：

```text
[T, B, ...]
```

但当前项目的全模型 temporal layout 是：

```text
time_major_flattened_TB
```

即模块之间通常实际保存：

```text
[T*B, ...]
```

并通过：

```python
to_temporal(...)
from_temporal(...)
```

在 neuron / temporal operator 内部恢复 `[T,B,...]`。

**不要为了 Site 3/4/6 重构而把整个项目改成显式传递 `[T,B,...]`。**

应继续遵循现有 flattened `T*B` architecture。

---

# 4. 最终 10 个 replacement site 的 GIF 行为

本轮修改后 GIF site policy 必须明确分为三类。

## 4.1 Saliency-aware GIF sites

```text
Site 1
Site 3
Site 4
Site 6
Site 7
Site 10
```

这些 site 使用 SpikeLLM 对齐的 salient-channel selection。

其中：

- Site 1：3 个 independent masks
- Site 7：2 个 independent masks
- Site 3/4/6/10：各 1 个 mask

## 4.2 All-low GIF site

```text
Site 2
```

Site 2：

```text
all-low static 4-bit GIF
```

即：

- 仍然 quantize；
- 仍然使用 static `low_scale / low_zero`；
- 仍然保留当前 per-head + group_size 参数布局；
- 所有 channel 都走 low 4-bit path；
- **不计算 saliency**；
- **不产生 salient/high mask**；
- forward 不能执行 high quantization 分支；
- SNN GIF deployment 时 low code 全部由第一个 GIF local timestep 输出，第二个 timestep 为 0，保持两个 timestep 的总和等于 low fake-quant output。

可以为了 artifact schema compatibility 保存未使用的 `high_scale/high_zero` 元数据，但：

> 它们不得参与 Site 2 forward、temporal coding、mask selection 或 Site 2 common-Clip 的 GIF representable-range 约束。

优先建议新增明确的：

```text
gif_policy = all_low_static_qmax15
```

而不是伪造一个“全 True mask”后仍计算 high branch。

## 4.3 GIF identity sites

```text
Site 5
Site 8
Site 9
```

这里的 `identity` 是严格恒等变换：

```text
GIF(x) = x
```

意味着对 GIF：

- 不计算 salient channels；
- 不生成 `mask_low`；
- 不执行 low quantization；
- 不执行 high quantization；
- 不使用 `low_scale/low_zero/high_scale/high_zero`；
- ANN `gif_aware` 前向中输入原样通过；
- SNN `deploy_gif` 中 temporal increments 原样通过。

### Site 5

继续完全保持当前 SpikeLLM-aligned Softmax identity 方案。

不要改变当前 Site 5：

```text
n_bits=16 / fix0to1 sentinel -> identity
```

语义。

### Site 8 / Site 9

采用旋转方案 A：

```text
Site 8 GIF = identity
Site 9 GIF = identity
```

删除当前：

```python
product_saliency = gate.square() * up.square()
```

用于 Site 8/9 GIF saliency 的逻辑。

Site 8/9 仍然需要收集 activation statistics，因为 Phase / MTN / common Clip 仍可能使用这些 activation statistics。

---

# 5. SpikeLLM-aligned salient score：公式和精度

## 5.1 Linear consumer：严格 FP32

SpikeLLM `SpikeQuantLinear.add_batch()` 的核心行为是：

```python
inp = inp.to(torch.float32)
weight = weight.to(torch.float32)

grad = inp @ weight.T
grad = grad @ weight
score = grad * inp
```

数学上：

\[
S(X)=X\odot(XW^\top W)
\]

对所有 linear-consumer saliency，必须按该公式重新计算。

**不要继续使用当前：**

```python
inputs * (output @ weight)
```

作为严格对齐实现。

原因：

1. 当前实现通常先在 BF16 中得到 `output`；
2. 若未来存在 bias，`output @ weight` 还会把 bias contribution 混入；
3. SpikeLLM 官方明确用 FP32 input/weight 重新计算。

因此新增统一 helper，例如：

```python
@torch.no_grad()
def spikellm_linear_saliency(x, weight):
    x32 = x.detach().to(torch.float32)
    w32 = weight.detach().to(torch.float32)
    projected = torch.matmul(x32, w32.transpose(-1, -2))
    grad = torch.matmul(projected, w32)
    return grad * x32
```

实际实现需支持 `[B,L,C]` 等 batch shape，不能只支持官方 `B=1` 的 `torch.mm` 写法。

Linear saliency 对应：

```text
Site 1 q role
Site 1 k role
Site 1 v role
Site 6 o_proj input
Site 7 gate role
Site 7 up role
Site 10 down_proj input
```

## 5.2 QK 中 K saliency：严格 FP64

SpikeLLM `SpikeQuantMatMul.add_batch()`：

```python
qp = Q.float64()
kv = K_transposed.float64()

grad = Q.T @ Q
grad = grad @ K_transposed
score = grad * K_transposed
```

因此 Site 3 必须按 FP64 计算。

若项目使用 K layout：

```text
[B,H,L,D]
```

可以等价写为：

```python
q64 = query.detach().to(torch.float64)
k64 = key.detach().to(torch.float64)

qk = torch.matmul(q64, k64.transpose(-2, -1))
score_k = k64 * torch.matmul(qk.transpose(-2, -1), q64)
```

得到：

```text
[B,H,L,D]
```

这与官方在 `[B,H,D,L]` 的 `K^T` 上计算后转置回来等价。

**禁止在 BF16/FP32 中先计算后再 `.double()`。**

## 5.3 PV 中 V saliency：严格 FP64

SpikeLLM 对 V：

```python
P64 = attn_weights.float64()
V64 = value.float64()

grad = P64.T @ P64
grad = grad @ V64
score = grad * V64
```

等价实现：

```python
p64 = attn_weights.detach().to(torch.float64)
v64 = value.detach().to(torch.float64)
pv64 = torch.matmul(p64, v64)
score_v = v64 * torch.matmul(p64.transpose(-2, -1), pv64)
```

或者复用已经严格以 FP64 重新计算的等价表达式。

不要复用 BF16 `attn_output` 再 cast FP64 作为“严格官方精度”。

## 5.4 Saliency accumulator dtype 也必须对齐

当前 `SiteStatistics.saliency_sum` 固定为 FP64，会破坏“linear saliency 官方 FP32 accumulator”的严格精度语义。

本轮应将 saliency accumulator 改成按 saliency source 保存 dtype：

```text
Linear saliency:
    score dtype = float32
    accumulator dtype = float32

MatMul K/V saliency:
    score dtype = float64
    accumulator dtype = float64
```

activation min/max、Phase EMA 等当前统计 dtype 不要求因此改变。

建议在 statistics artifact 中保存：

```text
saliency_accumulator_dtype
saliency_rule
saliency_roles
```

用于 provenance 和校验。

---

# 6. Mask selection 必须使用 SpikeLLM 官方 threshold / tie 规则

不要继续用：

```python
argsort(...)
mask_low[ordering[:low_channels]] = True
```

严格改成官方语义：

```python
low_quota = int(low_p * width)
threshold = torch.sort(channel_score.flatten())[0][low_quota - 1]
mask_low = channel_score <= threshold
```

其中：

```text
low_p = 0.9
salient ratio ≈ 0.1
```

由于使用：

```text
<= threshold
```

若 threshold 上存在 tie：

- low channel 数可能略大于 90%；
- salient channel 数可能略少于 10%。

这是允许且要求保留的官方行为。

---

# 7. 各 Site 的最终详细语义

---

## 7.1 Site 1：3 个 independent GIF masks

Site 1 是共同的 input RMSNorm output，但 SpikeLLM 官方的：

```text
q_proj
k_proj
v_proj
```

是三个独立 `SpikeQuantLinear`，因此必须保存三个 independent masks：

```text
mask_low_q
mask_low_k
mask_low_v
```

或者等价的：

```python
mask_low_by_role = {
    "q": ...,
    "k": ...,
    "v": ...,
}
```

### Saliency

分别使用各 consumer 自己的 weight：

\[
S_q(X)=X\odot(XW_q^\top W_q)
\]

\[
S_k(X)=X\odot(XW_k^\top W_k)
\]

\[
S_v(X)=X\odot(XW_v^\top W_v)
\]

全部 FP32。

### qparams

Site 1 仍然只有一套 shared static qparams：

```text
low_scale
low_zero
high_scale
high_zero
```

继续按 Site 1 原来的 `last_dim + group_size` 统计。

不要给 q/k/v 再各自创建一套 scale/zero。

### ANN/deployment 应用位置

由于三个 mask 不同，不能继续在 RMSNorm hook 中先对 Site 1 做一次统一 GIF 后再把同一个 tensor 同时送给 q/k/v。

对 GIF 模式应改成：

```text
RMSNorm output X
   ├─ GIF(X, role=q) -> q_proj
   ├─ GIF(X, role=k) -> k_proj
   └─ GIF(X, role=v) -> v_proj
```

推荐实现：

- `norm1_hook`
  - collect：记录 Site 1 activation 一次；
  - Phase / deploy_phase / deploy_mtn：继续在共享 Site 1 位置执行；
  - GIF / deploy_gif：不要在 norm hook 统一执行 GIF。
- 为 `q_proj/k_proj/v_proj` 安装 role-aware `forward_pre_hook`：
  - GIF / deploy_gif 时调用 Site 1 GIF；
  - 分别传入 `role="q" / "k" / "v"`。

Phase/MTN Site 1 的 topology 不要拆成三份。

---

## 7.2 Site 2：all-low 4-bit GIF

最终规定：

```text
Site 2 GIF = all-low static 4-bit
```

Site 2 对应 Q 在 QK matmul 的输入。

SpikeLLM 官方 `q_quant_params` 没有 `addbit/low_p`，因此 Q 不使用 salient/high path。

### GIF 行为

```text
所有 channel -> low quantizer
qmax = 15
```

不计算 Site 2 saliency。

不生成 Site 2 salient mask。

### 参数布局

保持当前：

```text
attention-head + group_size
```

即：

```text
[H, D/group_size]
```

### Phase / MTN

Site 2 Phase/MTN topology 和已有逻辑保持不变。

---

## 7.3 Site 3：K，在 `repeat_kv` 之后

这是本轮主要 topology 修改之一。

最终顺序：

```text
K after projection/RoPE/R3
    -> prefix/cache handling
    -> repeat_kv
    -> Site 3
    -> restore head layout
    -> QK matmul
```

### ANN shape

`repeat_kv` 后：

```text
[B,H,L,D]
```

然后：

```text
[B,H,L,D]
 -> transpose head/sequence
 -> [B,L,H,D]
 -> reshape
 -> [B,L,HD]
 -> Site 3 neuron
 -> [B,L,H,D]
 -> [B,H,L,D]
```

ANN 中绝对不要创建外部 `[T,B,...]`。

### SNN deployment

当前项目模块之间仍用：

```text
[T*B,H,L,D]
```

逻辑上：

```text
[T,B,H,L,D]
 -> [T,B,L,HD]
 -> Site 3 neuron
 -> [T,B,H,L,D]
```

应通过当前 `to_temporal/from_temporal` 体系实现，不要推翻 flattened-TB layout。

### Phase / MTN

Phase/MTN 同 GIF 一起移动到 `repeat_kv` 后。

topology 参考 `phaseSnnSdpaLlamaAttention`：

```text
repeat_kv
-> merge H,D
-> neuron
-> restore H,D
-> attention matmul
```

### 参数布局

虽然实际 Site 3 tensor 是 merged `[B,L,HD]`，但 parameter layout 保留：

```text
[H,D/group_size]
```

其中：

```text
H = num_attention_heads
```

### Calibration statistics

activation statistics 必须基于 **repeat 后的 K**。

为了保留 per-head parameter layout，calibration 内部可以直接用：

```text
[B,H,L,D]
```

进行统计，再在实际 replacement 时 merge 成 `[B,L,HD]`。

不要把 Site 3 statistics 改成 global last-dim grouped。

### GIF saliency

使用 repeat 后 K：

```text
Q: [B,H,L,D]
K: [B,H,L,D]
```

FP64 计算。

**删除当前 GQA score collapse：**

```text
H_attn
 -> H_kv × groups
 -> sum(groups)
 -> H_kv
```

这一逻辑不再允许。

Site 3 mask 必须属于 repeated attention-head coordinate：

```text
[H_attn,D]
```

### Mask ranking

虽然 mask 保存为 `[H,D]`，但 low/salient threshold 必须在：

```text
全部 H*D channels
```

上进行一次全局 ranking。

**不能每个 head 独立取 90% low。**

---

## 7.4 Site 4：V，在 `repeat_kv` 之后

Site 4 与 Site 3 使用相同 topology：

```text
V
 -> prefix/cache handling
 -> repeat_kv
 -> [B,H,L,D]
 -> merge -> [B,L,HD]
 -> Site 4
 -> restore [B,H,L,D]
 -> PV matmul
```

SNN deployment 同样按 logical：

```text
[T,B,H,L,D]
 -> [T,B,L,HD]
 -> Site 4
 -> [T,B,H,L,D]
```

### 参数布局

继续：

```text
[H,D/group_size]
```

其中 `H=num_attention_heads`。

### GIF saliency

FP64，严格按 SpikeLLM PV `x2=V` 公式。

mask 为：

```text
[H,D]
```

但 threshold 对全部：

```text
H*D
```

channels 一次性全局计算。

### 删除 GQA collapse

和 Site 3 一样，不再把 repeated-head saliency sum 回 native KV heads。

---

## 7.5 Site 5：GIF identity

继续当前行为。

GIF：

```text
identity
no saliency
no low/high quantization
no GIF mask
```

Site 5 继续永久不生成 common Clip state。

不要修改 Phase/MTN 的既有 Site 5 行为。

---

## 7.6 Site 6：head merge 后的 `o_proj` input

这是本轮第二个主要 topology 修改。

当前 attention value matmul：

```python
attn_output = matmul(attn_weights, value_states)
```

先得到：

```text
[B,H,L,D]               # ANN
[T,B,H,L,D] logical     # SNN deployment
```

必须先执行：

```text
H,L,D -> L,H,D -> L,HD
```

然后才进入 Site 6。

### ANN

必须是：

```text
attn_output: [B,H,L,D]
 -> transpose
 -> [B,L,H,D]
 -> reshape
 -> [B,L,HD]
 -> Site 6
 -> o_proj
```

Site 6 ANN replacement tensor 是：

```text
[B,L,HD]
```

### SNN deployment

logical：

```text
[T,B,H,L,D]
 -> [T,B,L,H,D]
 -> [T,B,L,HD]
 -> Site 6
 -> o_proj
```

current flattened implementation 中可以是：

```text
[T*B,H,L,D]
 -> [T*B,L,HD]
 -> controller.apply(...)
```

controller 内部再通过 `to_temporal` 恢复 `[T,B,L,HD]`。

### Site 6 不再 per-head

从 `ATTENTION_HEAD_GROUPED_SITE_IDS` / 等价逻辑中移除 Site 6。

Site 6 statistics layout 必须变成：

```text
last_dim
```

参数 shape：

```text
[HD/group_size]
```

若 `group_size=-1`：

```text
[1]
```

### Site 6 GIF saliency

Site 6 对应 `o_proj` input，严格使用 SpikeLLM Linear saliency：

\[
S(X)=X\odot(XW_o^\top W_o)
\]

FP32。

score shape：

```text
[B,L,HD]
```

mask shape：

```text
[HD]
```

threshold 对全部 `HD` channels 全局执行。

删除当前 `o_proj` saliency hook 中：

```python
score.reshape(..., heads, ...).transpose(...)
```

的 per-head reshape。

---

## 7.7 Site 7：2 个 independent GIF masks

SpikeLLM 官方：

```text
gate_proj
up_proj
```

是独立 `SpikeQuantLinear`。

因此 Site 7 保存：

```text
mask_low_gate
mask_low_up
```

或：

```python
mask_low_by_role = {
    "gate": ...,
    "up": ...,
}
```

### Saliency

分别 FP32：

\[
S_{gate}(X)=X\odot(XW_{gate}^\top W_{gate})
\]

\[
S_{up}(X)=X\odot(XW_{up}^\top W_{up})
\]

### qparams

仍只保存一套 Site 7 shared：

```text
low_scale
low_zero
high_scale
high_zero
```

按当前 last-dim + group_size。

### 应用

GIF 模式：

```text
post-MLP-RMSNorm X
   ├─ GIF(X, gate role) -> gate_proj
   └─ GIF(X, up role)   -> up_proj
```

不要先生成一个统一 GIF(X) 再同时送给两个 projection。

Phase/MTN Site 7 保持共享 replacement topology。

---

## 7.8 Site 8：GIF identity

删除 Site 8 GIF saliency。

不要再用：

```text
gate^2 * up^2
```

产生 GIF mask。

GIF forward / temporal 均 identity。

Phase/MTN 仍保持现有 neuron 行为。

---

## 7.9 Site 9：GIF identity

同 Site 8。

GIF：

```text
identity
no saliency
```

Phase/MTN 保持现有行为。

---

## 7.10 Site 10：保留单一 SpikeLLM Linear saliency

Site 10 是 down-proj consumer input。

继续使用一个 mask，但 saliency helper 必须改为严格 FP32 的 SpikeLLM Linear 公式：

\[
S(X)=X\odot(XW_{down}^\top W_{down})
\]

不要继续使用 BF16 output-derived score。

Site 10 参数继续普通：

```text
last_dim + group_size
```

---

# 8. Site 3/4 与 Site 6 的 shape helper

建议新增明确 helper，避免在多个路径复制容易出错的 transpose/reshape。

例如：

```python
def merge_attention_heads(x):
    # [B,H,L,D] -> [B,L,HD]
    if x.ndim != 4:
        raise ...
    B, H, L, D = x.shape
    return x.transpose(1, 2).contiguous().reshape(B, L, H * D)

def restore_attention_heads(x, *, num_heads, head_dim):
    # [B,L,HD] -> [B,H,L,D]
    if x.ndim != 3:
        raise ...
    B, L, HD = x.shape
    assert HD == num_heads * head_dim
    return (
        x.reshape(B, L, num_heads, head_dim)
         .transpose(1, 2)
         .contiguous()
    )
```

注意：

- ANN 的 `B` 就是真实 batch。
- SNN deployment 中这里的 `B` 实际可以是 flattened `T*B`。
- helper 本身不需要知道 `T`。
- neuron/controller 内部负责 temporal reshape。

---

# 9. 让 per-head state 可以作用于 merged `[B,L,HD]`

当前 `neurons.py::_parameter_values()` 的 `attention_head_grouped` 只接受：

```text
[B,H,L,D]
```

Site 3/4 改造后，实际 neuron 输入是：

```text
[B,L,HD]
```

因此必须扩展其 runtime broadcasting。

对于 state：

```text
num_heads = H
channels_per_head = D
groups_per_head = D/group_size
parameter tensor = [H, groups_per_head]
```

需要同时支持两种 runtime：

### 原有 native-head runtime

```text
[B,H,L,D]
```

继续保留，用于 Site 2 等。

### 新增 merged runtime

```text
[B,L,HD]
```

将：

```text
[H,groups]
 -> repeat_interleave(group_size)
 -> [H,D]
 -> flatten
 -> [HD]
 -> broadcast to [1,1,HD]
```

同理扩展：

```python
_mask_values(...)
```

使 `[H,D]` GIF mask 可 flatten 成 `[HD]` 作用于 Site 3/4 merged tensor。

这样：

- Site 3/4 实际 neuron tensor 是 merged；
- calibration/state 仍保留 per-head/group 参数；
- Site 2 原有 head-layout 不受影响。

Clipper、Phase、GIF、MTN 都必须通过同一 helper 获得一致 broadcasting。

---

# 10. Site 1 / Site 7 role-specific saliency 与 mask state

当前 statistics 只支持每个 site 一个：

```text
saliency_sum
saliency_row_count
```

本轮需要支持：

```text
Site 1: q / k / v
Site 7: gate / up
```

建议重构成 role-aware statistics。

例如：

```python
saliency = {
    "q": {
        "sum": ...,
        "row_count": ...,
        "accumulator_dtype": "float32",
    },
    ...
}
```

单 mask site 可以统一使用：

```text
role = "default"
```

推荐 role 约定：

```text
Site 1: q, k, v
Site 3: default
Site 4: default
Site 6: default
Site 7: gate, up
Site 10: default
```

Site 2/5/8/9 不应存在 GIF saliency role。

`StatisticsStore.record/update_saliency` 应显式接收：

```text
role
expected dtype / source kind
```

不要在 Site 1 把 q/k/v score 相加。

不要在 Site 7 把 gate/up score 相加。

---

# 11. Calibration：activation statistics 与 saliency statistics 的分工

## 11.1 Site 3/4 activation statistics

Site 3/4 move 到 `repeat_kv` 后。

因此 calibration 收集 activation 时必须使用 repeated K/V。

如果 Prefix/past KV 当前 calibration 逻辑只统计 current-token slice，则继续保留该 policy：

```text
past_length / prefix 部分仍从 calibration statistics 中排除
```

但排除操作应发生在 **repeat 后的 K/V** 上。

## 11.2 Site 3/4 saliency

同样基于 repeated K/V。

不 collapse。

## 11.3 Site 6 activation statistics

必须在 head merge 后记录：

```text
[B,L,HD]
```

因此 Site 6：

```text
layout_kind = last_dim
num_heads = None
channels = HD
```

## 11.4 Site 1/7 activation statistics

仍只记录共享 activation 一次。

不同 role 只影响 GIF saliency mask，不影响 shared static qparams。

---

# 12. GIF mask ranking 与 qparams grouping 必须彻底解耦

这是本轮最重要的原则之一。

## 12.1 Saliency mask：per-channel

所有 saliency-aware site 都按 **单 channel** score 进行 threshold。

mask 不是 per-group。

`group_size` 不参与 saliency score 聚合和 mask ranking。

## 12.2 qparams：仍按原 grouped policy

mask 确定后：

```text
low_scale / low_zero
high_scale / high_zero
```

仍按配置的 grouped policy 统计。

例如 Site 3：

```text
mask:
    [H,D] per-channel,
    threshold global over H*D

qparams:
    [H,D/group_size]
```

Site 6：

```text
mask:
    [HD]

qparams:
    [HD/group_size]
```

这两部分不能混淆。

---

# 13. Common Clip 需要同步改造

方案 B 使 Site 3/4 的三种 neuron 都处于 `repeat_kv` 后，因此 Site 3/4 不再需要 pre-repeat / post-repeat 两套 Clip。

每个 Site 3/4 继续只有一份：

```text
clip_state.pt
```

参数 shape：

```text
[H_attn,D/group_size]
```

## 13.1 普通 saliency GIF sites

对：

```text
Site 1,3,4,6,7,10
```

保留现有 intersection 思路：

```text
Phase range
∩ MTN range
∩ GIF low representable range
∩ GIF high representable range
```

## 13.2 Site 2 all-low

Site 2 没有 high branch。

因此 Clip 的 GIF constraint 必须只使用：

```text
GIF low representable range
```

即：

```text
Phase
∩ MTN
∩ GIF low
```

不要因为 artifact 中可能保留了 unused `high_scale/high_zero` 就把 high range 纳入 Site 2 Clip。

## 13.3 Site 8/9 GIF identity

GIF identity 没有 representable range 约束。

因此 Site 8/9 common Clip state若仍为 Phase ANN training 生成，应使用：

```text
Phase
∩ MTN
```

而不是要求不存在的 GIF low/high range。

### GIF ANN mode

即使：

```yaml
replacement:
  common_clip_enabled: true
```

Site 8/9 的 GIF path 仍必须保持严格 identity。

因此 **GIF mode 不得在 Site 8/9 identity 后继续应用 common Clip**。

Phase mode仍可按当前 common-Clip policy 在 Site 8/9 应用对应 `clip_state.pt`。

建议新增 mode-aware helper，例如：

```text
site_supports_clip_for_mode(site_index, mode)
```

而不是继续只靠一个全局 `CLIP_ELIGIBLE_SITE_IDS`。

## 13.4 Site 5

仍永久：

```text
no clip
```

---

# 14. GIF state policy 建议

保持每个 site 仍只有一个：

```text
gif_state.pt
```

不要把 Site 1/7 拆成新的 site directory。

建议新增明确 policy：

```text
ordinary_salient_static_qmax30
all_low_static_qmax15
identity
softmax_identity
```

可以具体命名为项目一致的字符串，但必须在：

- `calibration.py`
- `neurons.py`
- `state_validation.py`
- manifest
- tests

中统一。

## 14.1 Site 1/7 multi-mask state

建议：

```python
{
    ...
    "mask_policy": "multi_role",
    "mask_roles": ["q","k","v"],   # Site 1
    "mask_low_by_role": {...},
    "saliency_score_by_role": {...},
}
```

Site 7：

```text
gate, up
```

qparams shared。

## 14.2 单 mask ordinary state

Site 3/4/6/10 可继续：

```text
mask_low
saliency_score
```

## 14.3 Site 2 state

明确：

```text
saliency_enabled = false
quantization_path = low_only
```

不要求 saliency artifact。

## 14.4 Site 8/9 identity state

不应保存：

```text
mask_low
low_scale
low_zero
high_scale
high_zero
```

应保存足够的：

```text
site id
channels
temporal_steps
gif_policy
quantization_applied=false
saliency_enabled=false
temporal_policy=identity
```

用于 shape / provenance validation。

---

# 15. `snn2/sites.py` 必改

当前大致有：

```python
ATTENTION_HEAD_GROUPED_SITE_IDS = {2,3,4,6}
GIF_IDENTITY_SITE_IDS = {5}
```

修改为等价语义：

```text
PER_HEAD_GROUPED_SITE_IDS = {2,3,4}
GIF_IDENTITY_SITE_IDS = {5,8,9}
GIF_ALL_LOW_SITE_IDS = {2}
GIF_SALIENT_SITE_IDS = {1,3,4,6,7,10}
GIF_MULTI_MASK_ROLES = {
    1: ("q","k","v"),
    7: ("gate","up"),
}
```

可保留旧函数名以减少调用方改动，但语义必须更新。

Site 6 必须不再被判定为 `attention_head` statistics。

### Topology version

本轮改变：

- Site 3/4 physical replacement location；
- Site 6 tensor coordinate；
- GIF policy；
- state schema。

因此必须 bump：

```text
SITE_TOPOLOGY_VERSION
```

不要让旧 calibration artifact 被静默复用。

10 个 site 的编号和 directory name 不要改变。

---

# 16. `snn2/model_integration.py` 必改

重点修改以下区域。

## 16.1 Replace `_linear_score`

现有：

```python
inputs * torch.matmul(output, weight)
```

改成严格 SpikeLLM FP32 helper。

只在：

```text
controller.mode == "collect"
```

时计算 expensive saliency，ANN training/evaluation 不要无意义计算。

## 16.2 Site 1 hooks

- norm hook负责 shared activation / Phase/MTN；
- q/k/v projection 前新增 GIF role-aware prehook；
- q/k/v projection hook分别记录 q/k/v FP32 saliency；
- 不再将三者写入同一个 aggregate score。

## 16.3 Attention forward：Site 2

- query 仍在现有 Site 2 位置；
- GIF mode由 Site 2 all-low module处理；
- collect 时不要记录 Site 2 saliency。

## 16.4 Attention forward：Site 3/4

当前顺序大致是：

```text
Site3/4 -> repeat_kv
```

必须改成：

```text
repeat_kv -> Site3/4
```

具体：

```python
groups = ...
key = repeat_kv(key, groups)
value = repeat_kv(value, groups)

# collect:
#   record repeated K/V activation statistics
#
# replacement:
#   merge [B,H,L,D] -> [B,L,HD]
#   controller.apply(site3/site4)
#   restore [B,H,L,D]
```

然后才计算 QK。

### Saliency

- Site 3：FP64；
- Site 4：FP64；
- 删除 repeated-group collapse。

## 16.5 Site 6

当前：

```text
PV -> Site6 on [B,H,L,D] -> transpose -> [B,L,HD]
```

改成：

```text
PV -> transpose/reshape -> [B,L,HD] -> Site6
```

attention backend直接返回 merged Site 6 output。

## 16.6 `o_proj` saliency hook

输入本来就是 `[B,L,HD]`。

使用 FP32 SpikeLLM Linear saliency后直接记录。

删除 reshape 成 `[B,H,L,D]` 的代码。

## 16.7 Site 7 hooks

仿照 Site 1：

- norm2 shared activation；
- GIF role-specific `gate/up` prehook；
- gate/up 分别 FP32 saliency；
- 不再 aggregate。

## 16.8 Site 8/9

删除：

```python
product_saliency = gate.square() * up.square()
record_saliency(site8)
record_saliency(site9)
```

但仍调用：

```python
controller.apply(site8, ...)
controller.apply(site9, ...)
```

由 identity GIF module保证 GIF 不改变 tensor；Phase/MTN继续正常工作。

## 16.9 Site 10

使用新的 strict FP32 linear helper。

---

# 17. `snn2/temporal_model.py` 必改

当前 deployment 顺序是：

```text
Site2 Q
Site3 K
Site4 V
repeat_kv
...
PV
Site6
transpose
```

必须改为：

```text
Site2 Q
repeat_kv K/V
merge K/V
Site3 K
Site4 V
restore K/V heads
QK temporal matmul
...
PV temporal matmul
merge attn_output heads
Site6
return merged output
```

伪代码：

```python
query = controller.apply(site2, query)

key = repeat_kv(key, groups)
value = repeat_kv(value, groups)

key_merged = merge_attention_heads(key)       # [TB,L,HD]
value_merged = merge_attention_heads(value)

key_merged = controller.apply(site3, key_merged)
value_merged = controller.apply(site4, value_merged)

key = restore_attention_heads(key_merged, H, D)
value = restore_attention_heads(value_merged, H, D)

# temporal QK / softmax / PV

flat_output_heads = from_temporal(output_increment)   # [TB,H,L,D]
flat_output = merge_attention_heads(flat_output_heads) # [TB,L,HD]

flat_output = controller.apply(site6, flat_output)

return flat_output, flat_weights
```

不要在 return 前再次 transpose Site 6 output。

---

# 18. `snn2/stats.py` 必改

## 18.1 Site 6

`statistics_layout(6)` 必须成为：

```text
last_dim
```

不是 `attention_head`。

## 18.2 Site 3/4

仍按：

```text
attention_head
```

保存 calibration parameters。

但输入 statistics 是 **repeat 后的 H_attn**。

## 18.3 Role-aware saliency

支持 Site 1/7 多 role。

## 18.4 Precision

`update_saliency()` 不得无条件：

```python
score.float()
```

也不得无条件保存 FP64。

需要保留 caller 已严格计算好的：

```text
FP32 linear
FP64 matmul
```

dtype。

activation statistics 的现有 `.float()` reduction 可以继续保留。

## 18.5 Artifact schema

更新 `statistics.pt` 与 summary：

至少记录：

```text
saliency roles
saliency accumulator dtype per role
saliency source/operator
```

旧统计 artifact 必须因 format version 不匹配而拒绝。

---

# 19. `snn2/calibration.py` 必改

## 19.1 Mask build

将 per-head argsort 删除。

统一实现 official threshold helper，例如：

```python
def spikellm_mask_low(score, low_p):
    flat = score.flatten()
    low_quota = int(low_p * flat.numel())
    threshold = torch.sort(flat)[0][low_quota - 1]
    return score <= threshold
```

对 Site 3/4：

```text
score shape = [H,D]
flat numel = H*D
```

全局 threshold。

## 19.2 Site 1/7

分别 materialize role masks。

## 19.3 Site 2

不要求 saliency statistics。

构建 all-low GIF state。

## 19.4 Site 5/8/9

不要求 saliency statistics。

构建 identity GIF state。

Site 5 保持其专用 softmax metadata。

## 19.5 Site 6

由于 stats layout 已改为 last_dim：

```text
qparams / tau / base_scale / Clip
```

自然按 `HD + group_size` 构建。

不要再写任何 `num_heads` Site 6 参数。

## 19.6 Site 3/4

layout仍为 attention-head grouped，但 `num_heads` 应来自 repeated H_attn。

## 19.7 Clip

按第 13 节新增三类 rule：

```text
ordinary salient
all-low
identity
```

---

# 20. `snn2/neurons.py` 必改

至少完成以下内容。

## 20.1 `_parameter_values`

`attention_head_grouped` 同时支持：

```text
[B,H,L,D]
[B,L,HD]
```

## 20.2 `_mask_values`

同理支持：

```text
mask [H,D] -> runtime [B,L,HD]
```

## 20.3 StaticGIF multi-role

支持 Site 1/7：

```python
forward(x, role=...)
temporal(incoming, role=...)
```

或等价接口。

如果 state 声明 `multi_role` 而 caller 未提供 role，必须 fail fast。

如果 role 非法，也必须报错。

## 20.4 AllLowStaticGIF

实现 Site 2 low-only。

ANN：

```text
output = low_quantize(x)
```

SNN GIF：

```text
t0 = low_quantized_value
t1 = 0
```

总和等于 ANN low fake-quant output。

## 20.5 IdentityGIF

Site 8/9 使用普通 last-dim identity class。

Site 5 可以继续使用现有 `SoftmaxIdentityGIF`。

`gif_module_from_state()` 根据 policy返回：

```text
StaticGIF
AllLowStaticGIF
IdentityGIF
SoftmaxIdentityGIF
```

---

# 21. `snn2/controller.py` 必改

## 21.1 role-aware GIF API

建议给：

```python
apply(...)
```

增加可选参数：

```text
gif_role
```

或者新增专门 helper。

要求：

- Site 1/7 GIF state需要 role；
- 其他 site 不需要 role；
- regression checkpoint 若 Site 1/7 被调用多次，label 应带 role，避免三次覆盖/混淆，例如：

```text
site_01/gif_q/pre
site_01/gif_q/post
site_01/gif_k/pre
...
```

## 21.2 GIF identity + Clip

Site 8/9 在：

```text
mode == gif
mode == deploy_gif
```

时必须保持 overall identity，不要在 identity GIF 后继续套 common Clip。

Phase mode仍可套 Site 8/9 Clip。

## 21.3 Temporal

保持现有：

```python
to_temporal(x, T)
neuron.temporal(...)
from_temporal(...)
```

Site 3/4/6 的 merged tensor只需要让该通用路径接受 `[TB,L,HD]`。

不要在 controller 外部 ANN path创建 `T`。

---

# 22. `snn2/state_validation.py` 必改

validation 必须理解新的 site-specific GIF policy。

不要再假设所有非 Site 5 GIF 都有：

```text
high_qmax=30
mask_low
ordinary StaticGIF
```

应分别验证：

### Saliency-aware

```text
Site 1,3,4,6,7,10
```

### All-low

```text
Site 2
```

### Identity

```text
Site 5,8,9
```

另外验证：

- Site 1 role恰好 `q,k,v`
- Site 7 role恰好 `gate,up`
- Site 3/4 `num_heads == num_attention_heads`（若 manifest有模型 topology metadata）
- Site 6 parameter layout 为 last-dim grouped
- Site 6 `num_heads is None`
- Site 6 width为 hidden size
- Site 8/9 identity state无量化参数/mask

---

# 23. `snn2/temporal_ops.py` / artifact version 必须 bump

本轮旧 artifacts 与新 topology 不兼容。

至少 bump：

```text
TEMPORAL_IMPLEMENTATION_VERSION
SITE_STATE_FORMAT_VERSION
STATISTICS_FORMAT_VERSION
CALIBRATION_MANIFEST_FORMAT_VERSION
CONVERSION_METADATA_FORMAT_VERSION
```

并同步更新：

```text
TEMPORAL_IMPLEMENTATION
CALIBRATION_GROUPING_POLICY
```

当前全局 grouping policy 名称类似：

```text
per_head_within_head_groups_v1
```

已经不再准确，因为 Site 6 变成 merged last-dim grouping。

改成能明确表达：

```text
Site 2/3/4 logical per-head grouped
Site 6 merged-last-dim grouped
other sites last-dim grouped
```

的新 policy/version string。

`TEMPORAL_LAYOUT = time_major_flattened_TB` 保持不变。

---

# 24. `snn2/conversion.py` / training provenance 必须同步

本轮不改变 aware-mode calibration reuse policy，但由于 artifact schema/topology 改变：

- conversion validation；
- metadata；
- training provenance；
- calibration manifest hash；
- grouping policy；
- statistics version；

必须全部使用新版本。

旧 calibration / training provenance / conversion metadata 必须 fail fast，不得继续使用。

`convert_snn.py` 的外部命令接口如果不必要不要改变。

---

# 25. 当前代码中必须删除/替换的旧逻辑

Codex 应全仓库 grep 并清除以下旧假设。

## 25.1 Site 6 per-head

所有：

```text
Site 6 ∈ ATTENTION_HEAD_GROUPED
Site 6 statistics [B,H,L,D]
Site 6 parameter [H,groups]
Site 6 saliency reshape back to heads
```

全部删除。

## 25.2 Site 3/4 pre-repeat

所有：

```text
controller.apply(site3/site4) before repeat_kv
record Site3/4 activation before repeat_kv
```

全部移到 repeat 后。

## 25.3 GQA saliency collapse

删除：

```python
reshape(... kv_heads, groups, ...)
.sum(dim=groups)
```

用于 Site 3/4 GIF saliency的逻辑。

## 25.4 Per-head salient quota

删除：

```python
for head:
    argsort(...)
    choose low_ratio * D
```

Site 3/4 以及任何其他 site 都不应以 group_size/per-head quota 替代 global per-channel threshold。

## 25.5 Site 1 aggregation

删除 q/k/v saliency写入同一累计 tensor的逻辑。

## 25.6 Site 7 aggregation

删除 gate/up saliency写入同一累计 tensor的逻辑。

## 25.7 Site 8/9 product saliency

删除：

```text
gate^2 * up^2
```

用于 GIF salient mask。

## 25.8 BF16-first linear saliency

删除依赖 projection `output` 的 `_linear_score`。

---

# 26. 回归检查与单元测试

本轮不能只修改实现而不补测试。

至少更新/新增以下测试。

## 26.1 `tests/test_sites.py`

验证：

```text
10 sites 不变
Site 2/3/4 = logical per-head grouping
Site 6 != per-head grouping
GIF identity = 5,8,9
GIF all-low = 2
GIF saliency-aware = 1,3,4,6,7,10
Site1 roles=q,k,v
Site7 roles=gate,up
```

## 26.2 `tests/test_statistics.py`

验证：

- Site 3/4 repeated H statistics；
- Site 6 `last_dim`；
- Site 6 channels=`HD`；
- role-specific saliency；
- Linear saliency accumulator FP32；
- MatMul saliency accumulator FP64。

## 26.3 `tests/test_calibration_gif.py`

新增：

### official threshold tie

构造：

```text
score 中 threshold 位置存在多个相同值
```

验证：

```text
mask_low = score <= threshold
```

导致 low count可以大于精确 quota。

### Site 3/4 global threshold

确保不是 per-head 90%。

### Site 1 multi-mask

q/k/v 三个 weight构造不同 saliency排序，验证三个 mask不同。

### Site 7 multi-mask

同理。

### Site 2

验证无 saliency也能 materialize；forward只用 low。

### Site 8/9

验证无 saliency也能 materialize identity GIF。

### Site 6

验证 qparams shape为：

```text
[HD/group_size]
```

不是 `[H,D/group_size]`。

## 26.4 `tests/test_neurons.py`

新增：

- `attention_head_grouped` state作用于 merged `[B,L,HD]`；
- mask `[H,D]` flatten broadcast正确；
- Site 2 all-low ANN/SNN等价；
- Site 8/9 identity ANN/SNN输入输出 bitwise equal；
- multi-role GIF role选择；
- missing/invalid role fail fast。

## 26.5 `tests/test_controller_state_loading.py`

验证：

- Site 1/7 role-aware load；
- GIF Site8/9不会因 `common_clip_enabled=true` 改变 tensor；
- Phase Site8/9仍可使用 common Clip。

## 26.6 Attention topology test

建议新增独立 test，例如：

```text
tests/test_attention_site_topology.py
```

使用小 tensor / mock controller记录每个 Site 输入 shape。

### ANN

必须看到：

```text
Site 3: [B,L,HD]
Site 4: [B,L,HD]
Site 6: [B,L,HD]
```

且调用 Site 3/4 时 K/V已经 repeat 到 `num_attention_heads`。

**测试中不得出现 ANN external T。**

### SNN

controller/neuron内部 logical：

```text
Site 3: [T,B,L,HD]
Site 4: [T,B,L,HD]
Site 6: [T,B,L,HD]
```

模型模块之间仍为 flattened：

```text
[T*B,L,HD]
```

## 26.7 Phase conversion regression

更新现有：

```text
tests/test_phase_conversion_regression.py
```

使其覆盖：

```text
repeat_kv -> merge -> Site3/4 -> restore
PV -> merge -> Site6
```

Phase topology 应与 SpikingLLM `phaseSnnSdpaLlamaAttention` 的对应结构一致。

## 26.8 Conversion metadata / calibration topology tests

同步更新：

- `test_calibration_topology.py`
- `test_calibration_profiles.py`
- `test_conversion_metadata.py`
- `test_post_finetuning_protocol.py`

确保旧 state/version被拒绝。

---

# 27. 必须增加的数值公式测试

不要只测 shape。

## 27.1 Linear FP32 saliency reference

随机生成 FP16/BF16：

```text
x
W
```

expected 必须明确用：

```python
x32 = x.float()
w32 = W.float()
expected = ((x32 @ w32.T) @ w32) * x32
```

项目 helper 输出应与 expected 一致。

## 27.2 K FP64 saliency

expected 使用纯 FP64 reference。

## 27.3 V FP64 saliency

expected 使用纯 FP64 reference。

测试必须能发现“先 BF16 matmul，再 cast FP64”的错误实现。

---

# 28. Regression recorder / debug metadata

由于 Site 3/4/6 topology改变，建议同步更新 regression checkpoint 命名，使 shape语义明确。

例如：

```text
attn/k_after_repeat_before_site3
attn/v_after_repeat_before_site4
attn/pv_head_output_before_merge
attn/pv_merged_before_site6
```

Site 1/7 role-specific GIF：

```text
site_01/gif_q/pre
site_01/gif_q/post
...
site_07/gif_gate/pre
...
```

不要让多个 role 都写同一个 checkpoint name。

---

# 29. 文档与注释同步

全仓库 markdown / comments 搜索以下旧说法并更新：

```text
Site 3/4 before repeat_kv
Site 6 per-head
ATTENTION_HEAD_GROUPED_SITE_IDS includes 6
Site 8/9 GIF saliency
one mask per replacement site
```

统一为本方案。

不要改变 10-site numbering。

---

# 30. 重新实验要求

本轮 topology、statistics schema、GIF mask 都改变，已有 calibration / ANN-aware training artifacts 不能复用。

修改完成后必须重新：

```text
1. ANN-training calibration
2. aware ANN fine-tuning
3. SNN conversion
4. SNN evaluation
```

如果某 mode 使用当前项目既有的 aware conversion reuse policy，则仍按该 policy，但 **必须复用本轮重新生成的新 calibration artifact**。

旧：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
calibration_state_manifest.json
training provenance
conversion_metadata.json
```

应因 version mismatch 被拒绝。

---

# 31. 最终 acceptance criteria

代码修改完成必须同时满足以下条件。

## A. GIF algorithm

- [ ] Site 1 有 q/k/v 三个 GIF masks。
- [ ] Site 7 有 gate/up 两个 GIF masks。
- [ ] Site 2 是 all-low static 4-bit。
- [ ] Site 5/8/9 GIF strict identity。
- [ ] Site 8/9 不再计算 product saliency。
- [ ] Site 3/4 K/V mask来自 repeat 后 attention-head coordinate。
- [ ] Site 3/4 不再 collapse GQA saliency。
- [ ] Site 3/4 salient threshold对 H*D 全局执行。
- [ ] Site 6 salient threshold对 HD 全局执行。
- [ ] Site 1/7/10 threshold对完整 channel dimension全局执行。
- [ ] mask使用官方 `<= threshold` tie语义。
- [ ] Linear saliency严格 FP32。
- [ ] K/V MatMul saliency严格 FP64。
- [ ] `high_qmax=30` 保持。
- [ ] static GIF 保持。

## B. Topology

- [ ] Site 3/4 的 GIF、Phase、MTN 全在 `repeat_kv` 后。
- [ ] Site 3/4 actual replacement tensor ANN 为 `[B,L,HD]`。
- [ ] Site 3/4 SNN logical tensor为 `[T,B,L,HD]`。
- [ ] Site 3/4 parameters仍为 `[H,D/group_size]`。
- [ ] Site 6 在 PV output merge heads 后。
- [ ] Site 6 ANN tensor为 `[B,L,HD]`。
- [ ] Site 6 SNN logical tensor为 `[T,B,L,HD]`。
- [ ] Site 6 完全取消 per-head parameter layout。
- [ ] Site 6只保留 `group_size`。

## C. ANN/SNN temporal convention

- [ ] ANN calibration没有 external T。
- [ ] ANN fine-tuning没有 external T。
- [ ] ANN replacement module输入输出 shape与 ANN tensor一致。
- [ ] 只有 SNN deployment有真实 temporal T。
- [ ] 全模型继续使用 `time_major_flattened_TB`。

## D. Clip

- [ ] Site 3/4仍各只有一份 common Clip。
- [ ] Site 3/4 Clip layout为 repeated `[H_attn,groups]`。
- [ ] Site 6 Clip仅 `[HD/group_size]`。
- [ ] Site 2 Clip只受 GIF low range约束。
- [ ] Site 8/9 identity GIF不应用 Clip。
- [ ] Site 8/9 Phase仍可使用 Phase/MTN common Clip。
- [ ] Site 5仍无 Clip。

## E. Artifacts

- [ ] 所有相关 format/topology version已 bump。
- [ ] 旧 artifacts fail fast。
- [ ] manifest明确记录新 GIF site policy / grouping policy / saliency precision。
- [ ] 所有 unit tests 通过。

---

# 32. 不要做的事情

本轮明确禁止顺手修改以下内容：

- 不要把 static GIF 改成 dynamic；
- 不要把 `high_qmax=30` 改成 31；
- 不要改 `base_bits=4/add_bits=1/T_GIF=2`；
- 不要取消 `group_size`；
- 不要把 Site 3/4 改成 global-last-dim parameter grouping；
- 不要让 Site 6继续 per-head；
- 不要恢复 Site 8/9 GIF saliency；
- 不要给 Site 2增加 saliency/high path；
- 不要改全参数 ANN fine-tuning 策略；
- 不要改变 calibration dataset / sample selection protocol；
- 不要改变 Prefix protocol；
- 不要改变 rotation algorithm；
- 不要改变 10 个 site 的编号；
- 不要在 ANN calibration/fine-tuning tensor上增加 external T；
- 不要因为参考 SpikingLLM 就直接复制其完整模型类；只参考其 attention topology，并适配当前项目的 controller/flattened-TB architecture。

---

# 33. 实现优先顺序

建议 Codex 按以下顺序实施，减少交叉错误：

```text
1. sites.py：site policy / grouping / topology version
2. temporal_ops.py：artifact/version/policy metadata
3. stats.py：role-aware saliency + precision + Site6 layout
4. calibration.py：official threshold + new state policies + Clip
5. neurons.py：merged per-head broadcast + multi-role/all-low/identity GIF
6. controller.py：role-aware apply + mode-aware Clip
7. model_integration.py：ANN/collect topology与 saliency
8. temporal_model.py：SNN deployment topology
9. state_validation.py
10. training.py / conversion.py provenance
11. tests
12. markdown/comments
13. run complete pytest
```

完成后再进行真实 Qwen/Llama calibration/training/conversion，不要用旧 artifact 做兼容性绕过。
