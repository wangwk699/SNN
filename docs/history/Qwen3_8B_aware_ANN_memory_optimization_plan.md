# Qwen3-8B `phase_aware` / `gif_aware` ANN Fine-tuning 显存优化修改方案

## 0. 文档目的

本方案用于修改仓库：

```text
https://github.com/wangwk699/SNN
```

目标是降低 **Qwen3-8B `phase_aware` / `gif_aware` 全参数 ANN fine-tuning** 的 GPU 峰值显存，使训练尽可能在现有 GPU 条件下运行，同时保持当前实验的算法语义不变。

本次只采用已经确认的三个方向：

1. **开启 Gradient Checkpointing**
2. **仅优化 Phase/GIF 的 ANN training forward 显存，不修改任何 SNN temporal deployment 路径**
3. **DeepSpeed ZeRO-3 开启 CPU optimizer offload**

本次修改必须遵守一个最高优先级原则：

> **这是一轮 memory-equivalent implementation optimization，不是算法修改。**
>
> 不允许改变 Phase/GIF 的数学定义、surrogate gradient 定义、量化参数、量化 bit-width、temporal decomposition、replacement site topology、训练 batch 语义、数据、Prefix、rotation、Clip、loss 或 SNN conversion/evaluation 规则。

---

# 1. 当前代码基线与问题定位

实施修改前，先基于当前 `main` 分支确认以下事实。不要根据旧文档或旧配置推断，直接看实际代码。

## 1.1 ANN fine-tuning 入口

训练入口：

```text
scripts/train_ann.py
```

实际调用：

```python
from snn2.training import train_full_parameters
```

核心实现：

```text
snn2/training.py
```

当前训练为全参数训练：

```python
for parameter in model.parameters():
    parameter.requires_grad_(True)
```

`TrainingArguments` 已经从配置读取：

```python
gradient_checkpointing=bool(
    training_cfg.get("gradient_checkpointing", False)
)
```

所以本次不需要重新实现 Hugging Face gradient checkpointing，只需使 Qwen3-8B 训练配置真正启用它。

模型加载代码已经设置：

```python
model.config.use_cache = False
```

因此不要再额外修改 `use_cache` 语义。

## 1.2 当前 Qwen3-8B 训练设置

`configs/experiment_matrix.yaml` 中 `exp1_qwen3_8b_tldr` 当前关键训练配置为：

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  bf16: true
  fp16: false
  dtype: bfloat16
  gradient_checkpointing: false
  deepspeed_config: configs/deepspeed_zero3.json
```

本次：

- `per_device_train_batch_size` **保持 1**
- `gradient_accumulation_steps` **保持 16**
- `max_seq_length` **保持 2048**
- BF16 **保持不变**
- learning rate / optimizer 超参数 **全部保持不变**

特别注意：

```text
gradient_accumulation_steps: 16
```

不能为了省显存改成 8。

`gradient_accumulation_steps` 主要决定 effective global batch 和 optimizer update 频率，而单个 micro-batch 的 forward/backward peak memory 基本不由它决定。

例如 2 GPU 时：

```text
effective global batch
= 1 × 2 × 16
= 32
```

若改成 8 会变成 16，属于训练优化语义改变，因此本次禁止修改。

## 1.3 当前 DeepSpeed 配置

当前：

```text
configs/deepspeed_zero3.json
```

为 ZeRO-3，但没有 optimizer CPU offload：

```json
{
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 50,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

本次不要直接破坏/替换这个共享配置。新增一个专门的 CPU optimizer offload 配置，供 Qwen3-8B 使用。

---

# 2. 本次修改的硬性边界

以下规则是实现约束，不是建议。

## 2.1 允许修改

允许：

```text
✓ Qwen3-8B 开启 gradient checkpointing

✓ 新增 ZeRO-3 optimizer CPU offload 配置

✓ PhaseSurrogate 新增 ANN-only memory-optimized forward

✓ ordinary salient StaticGIF 新增 ANN-only mixed fake-quant forward

✓ 增加 forward/backward equivalence tests

✓ 增加 temporal deployment regression tests

✓ 增加配置/DeepSpeed regression tests
```

## 2.2 严禁修改

本次禁止：

```text
✗ gradient_accumulation_steps 16 → 8
✗ per_device_train_batch_size 改动
✗ max_seq_length 2048 改动
✗ phase.T 改动
✗ GIF base_bits/add_bits/qmax 改动
✗ learning rate 改动
✗ optimizer beta / epsilon / weight decay 改动
✗ Prefix 使用规则改动
✗ rotation 改动
✗ common Clip 改动
✗ calibration 统计规则改动
✗ replacement site 数量、位置或 topology 改动
✗ attention backend / snn2_eager 改动
✗ FlashAttention / SDPA 重构
✗ LoRA / QLoRA
✗ Phase surrogate derivative 改动
✗ Phase custom backward
✗ Phase SNN temporal() 改动
✗ GIF SNN temporal() 改动
✗ GIF integer_chunks() 改动
✗ ordinary GIF 原 _quantize() 语义改动
✗ AllLowStaticGIF 改动
✗ SoftmaxIdentityGIF 改动
✗ IdentityGIF 改动
✗ optimizer parameter offload（本轮只 offload optimizer state）
```

如果完成本方案后 Qwen3-8B 仍然 OOM，不要自行追加上述方法；先保留当前修改结果，再单独评估下一轮显存优化。

---

# 3. 修改一：Qwen3-8B 开启 Gradient Checkpointing

## 3.1 修改文件

修改：

```text
configs/experiment_matrix.yaml
```

找到：

```text
exp1_qwen3_8b_tldr
```

将：

```yaml
gradient_checkpointing: false
```

改为：

```yaml
gradient_checkpointing: true
```

其余训练字段保持完全不变。

不要修改：

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

不要修改：

```yaml
data:
  max_seq_length: 2048
```

## 3.2 为什么这样不会改变算法定义

Gradient Checkpointing 只改变 activation 的保存策略：

原实现：

```text
forward
  ↓
保存大量中间 activation
  ↓
backward 直接使用
```

开启后：

```text
forward
  ↓
少保存 activation
  ↓
backward 时重新计算相应 forward
  ↓
求同一个梯度
```

模型函数、loss、Phase/GIF operator、batch、optimizer update rule 都不改变。

理论算法语义相同。

但不要把验收标准写成“最终 checkpoint 必须逐 bit 一致”。GPU 浮点运算、recompute、CPU optimizer execution 都可能造成最后几位数值差异。正确要求是严格 `allclose` 和训练行为一致。

## 3.3 生成配置

`configs/generated/*.yaml` 是 materialized config。

不要只手工修改生成后的 YAML。

修改 `configs/experiment_matrix.yaml` 后运行：

```bash
python scripts/materialize_configs.py
```

确认：

```text
configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml
configs/generated/exp1_qwen3_8b_tldr__gif_aware.yaml
```

以及同一 Qwen3-8B experiment 下生成的配置均含：

```yaml
training:
  gradient_checkpointing: true
```

由于当前 materializer 是 experiment-level config × ann_mode，本方案允许 Qwen3-8B 的四个 ANN mode 都使用 gradient checkpointing。

不要为只限制 `phase_aware/gif_aware` 而额外引入复杂的 mode-specific config override 机制。

该改变不影响算法定义，只改变 Qwen3-8B 的训练内存/算力策略。

---

# 4. 修改二：新增 ZeRO-3 CPU optimizer offload

## 4.1 不直接覆盖原共享配置

保留：

```text
configs/deepspeed_zero3.json
```

原样。

新增：

```text
configs/deepspeed_zero3_cpu_offload.json
```

建议完整内容为：

```json
{
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 50,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

本轮只增加：

```json
"offload_optimizer": {
  "device": "cpu",
  "pin_memory": true
}
```

不要增加：

```json
"offload_param": ...
```

不要把：

```json
"overlap_comm": true
```

改成 false。

## 4.2 修改 Qwen3-8B experiment 的 DeepSpeed 路径

在：

```text
configs/experiment_matrix.yaml
```

的：

```text
exp1_qwen3_8b_tldr
```

中将：

```yaml
deepspeed_config: configs/deepspeed_zero3.json
```

改成：

```yaml
deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

然后再次运行：

```bash
python scripts/materialize_configs.py
```

确认 Qwen3-8B 生成配置使用新文件。

其他 experiment（例如 Qwen3-1.7B、Llama3-8B）不要因为本次修改自动切换到 CPU offload；本次目标是 Qwen3-8B。

## 4.3 CPU optimizer offload 的语义约束

CPU offload 的目的：

```text
原来：
GPU
├ model/parameter shard
├ gradient shard
├ optimizer m state
└ optimizer v state

修改后：
GPU
├ model/parameter shard
└ 当前计算需要的数据

CPU RAM
├ optimizer m state
└ optimizer v state
```

它改变 optimizer state 的存储位置，而不是主动改变：

```text
learning_rate
weight_decay
adam_beta1
adam_beta2
adam_epsilon
gradient accumulation
gradient clipping
```

`training.py` 中这些参数全部保持现状。

不要为了 CPU offload 在 DeepSpeed JSON 中新增另一套不同超参数的 `optimizer` 配置。

如果 DeepSpeed 运行时由于 ZeRO-Offload 选择 CPU-compatible Adam 实现，必须确认：

- optimizer 仍为当前训练所要求的 Adam/AdamW 更新语义；
- learning rate 与所有 beta/epsilon/weight-decay 与 `TrainingArguments` 一致；
- 不允许静默换成不同的优化算法。

理论算法语义相同，但不要求 CPU 与 GPU optimizer 的浮点更新逐 bit 相同。

---

# 5. 修改三：Phase ANN-only 显存优化

## 5.1 当前问题

文件：

```text
snn2/neurons.py
```

当前 `PhaseSurrogate`：

```python
def encode(self, x: torch.Tensor, return_temporal: bool) -> torch.Tensor:
    sign = x.sign().detach()
    tau = _parameter_values(x, self.tau, self.layout)
    v0 = _parameter_values(x, self.v0, self.layout)
    membrane = x.abs() + v0
    outputs = []
    for timestep in range(self.T):
        amplitude = tau * 2.0 ** (-(timestep + 1))
        distance = membrane - amplitude
        spike = (
            (distance > 0).to(distance.dtype)
            if self.slope is None
            else HeavisideSigmoid.apply(distance, self.slope)
        )
        outputs.append(sign * amplitude * spike)
        membrane = membrane - amplitude * spike
    temporal = torch.stack(outputs, dim=0)
    return temporal if return_temporal else temporal.sum(dim=0)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.encode(x, return_temporal=False)

def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
    if incoming.shape[0] != self.T:
        raise ValueError(...)
    return self.encode(incoming.sum(dim=0), return_temporal=True)
```

问题是 ANN fine-tuning 最终只需要：

```text
sum_t(output_t)
```

但当前 ANN path 仍然：

1. 保存 `T` 个 full-size `outputs`
2. `torch.stack` 成 `[T, ...]`
3. 再 `sum(dim=0)`

Qwen3-8B 当前 Phase `T=6`，尤其在大型 activation / attention probability 上会显著增加峰值显存。

## 5.2 最关键约束：不要修改 SNN reference path

当前：

```text
ANN:
forward()
  → encode(..., return_temporal=False)

SNN:
temporal()
  → encode(..., return_temporal=True)
```

因此不能把共享 `encode()` 改成只返回 streaming sum。

本次必须采用：

```text
PhaseSurrogate
│
├── forward()
│     └── 新 ANN-only streaming path
│
├── encode()
│     └── 保持现有 temporal/reference 行为
│
└── temporal()
      └── 原样继续调用 encode(..., return_temporal=True)
```

## 5.3 推荐实现

在 `PhaseSurrogate` 中新增一个明确只给 ANN 使用的私有方法，例如：

```python
def _forward_ann_streaming(self, x: torch.Tensor) -> torch.Tensor:
    sign = x.sign().detach()
    tau = _parameter_values(x, self.tau, self.layout)
    v0 = _parameter_values(x, self.v0, self.layout)
    membrane = x.abs() + v0

    accumulated = None

    for timestep in range(self.T):
        amplitude = tau * 2.0 ** (-(timestep + 1))
        distance = membrane - amplitude

        spike = (
            (distance > 0).to(distance.dtype)
            if self.slope is None
            else HeavisideSigmoid.apply(distance, self.slope)
        )

        contribution = sign * amplitude * spike

        accumulated = (
            contribution
            if accumulated is None
            else accumulated + contribution
        )

        membrane = membrane - amplitude * spike

    if accumulated is None:
        raise RuntimeError("Phase ANN streaming forward produced no timestep output")

    return accumulated
```

然后仅修改：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward_ann_streaming(x)
```

## 5.4 `encode()` 必须保留为 temporal reference

原：

```python
def encode(...)
```

不要改变其 temporal 计算含义。

最保守做法是本轮直接保持它的主体代码不变，包括：

```python
outputs = []
...
outputs.append(...)
...
temporal = torch.stack(outputs, dim=0)
return temporal if return_temporal else temporal.sum(dim=0)
```

尽管新的 ANN production path 不再使用：

```python
encode(..., return_temporal=False)
```

也不要删除 `return_temporal=False` 支持，因为测试要把它作为 **legacy ANN reference implementation**。

这样可以直接验证：

```text
new forward()
vs
old encode(..., return_temporal=False)
```

## 5.5 `temporal()` 必须完全保持

以下逻辑不允许改变：

```python
def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
    if incoming.shape[0] != self.T:
        raise ValueError(f"Phase expects T={self.T}, got {incoming.shape[0]}")
    return self.encode(incoming.sum(dim=0), return_temporal=True)
```

即 Phase SNN 仍然：

```text
incoming [T, ...]
→ incoming.sum(dim=0)
→ 原 encode()
→ T 个 Phase firing outputs
→ torch.stack
→ [T, ...]
```

不得改变：

- `T`
- `tau`
- `v0`
- amplitude/threshold 公式
- `distance`
- membrane recurrence
- spike rule
- SNN hard spike
- temporal output shape
- timestep order

## 5.6 不要实现 Phase custom backward

当前 surrogate：

```python
class HeavisideSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ...
```

以及：

```python
membrane = membrane - amplitude * spike
```

使后续 timestep 与前面 spike 存在递归依赖。

本轮绝对不要为了进一步省显存自己写新的 Phase backward/recompute backward。

必须保留：

```python
HeavisideSigmoid.apply(...)
```

和 PyTorch 现有 autograd graph。

Gradient Checkpointing 已负责更大范围的 recompute。

---

# 6. 修改四：ordinary salient GIF ANN-only mixed fake quantization

## 6.1 当前问题

当前普通 `StaticGIF.forward()`：

```python
low, _, _ = self._quantize(
    x,
    self.low_scale,
    self.low_zero,
    qmin=0,
    qmax=GIF_LOW_QMAX,
)

high, _, _ = self._quantize(
    x,
    self.high_scale,
    self.high_zero,
    qmin=0,
    qmax=self.high_qmax,
)

mask = _mask_values(x, self._mask(role), self.layout)

return torch.where(mask, low, high)
```

而 `_quantize()` 内部会：

```python
x.float()
```

并生成 full-size FP32 fake-quant intermediates。

因此当前实际为：

```text
整个 x → low FP32 fake quant
+
整个 x → high FP32 fake quant
+
torch.where(mask)
```

即便 low/high channel 只分别使用自己的部分 channel，也会对完整 activation 计算两套 FP32 path。

## 6.2 GIF SNN temporal 依赖原 `_quantize()`，禁止修改

当前 `StaticGIF.temporal()` 明确调用：

```python
_, low_q, low_zero = self._quantize(...)
_, high_q, high_zero = self._quantize(...)
```

然后执行：

```python
high_chunks = self.integer_chunks(high_q)
```

再按两个 temporal timestep 构造 output。

因此：

```text
StaticGIF._quantize()
StaticGIF.integer_chunks()
StaticGIF.temporal()
```

全部作为 **SNN immutable reference path**。

本轮不要重写 `_quantize()` 以实现 mixed quantization。

## 6.3 只新增 ANN-only mixed fake quant 方法

建议在普通 `StaticGIF` 中新增：

```python
def _forward_ann_mixed_quant(
    self,
    x: torch.Tensor,
    *,
    role: str | None = None,
) -> torch.Tensor:
    ...
```

然后让：

```python
def forward(...)
```

只调用这个新方法。

## 6.4 mixed quant 的数学定义

原逻辑对每个 channel：

low channel：

```text
scale = low_scale
zero  = low_zero
qmax  = 15
```

high channel：

```text
scale = high_scale
zero  = high_zero
qmax  = 30
```

先根据 mask 选择 channel-specific effective parameters：

```text
effective_scale
effective_zero
effective_qmax
```

再只对 full activation 做一次：

```text
x.float()
→ round_ste
→ clamp
→ dequantize
```

数学上：

```text
low channel 仍执行原 low quantizer
high channel 仍执行原 high quantizer
```

只是避免计算未被选择的另一整套 full-tensor branch。

## 6.5 推荐实现骨架

实现时必须继续复用：

```python
_parameter_values(...)
_mask_values(...)
self._mask(role)
self.round_ste(...)
```

建议结构：

```python
def _forward_ann_mixed_quant(
    self,
    x: torch.Tensor,
    *,
    role: str | None = None,
) -> torch.Tensor:
    mask = _mask_values(x, self._mask(role), self.layout)

    low_scale = _parameter_values(
        x, self.low_scale, self.layout
    ).clamp_min(1e-8)

    high_scale = _parameter_values(
        x, self.high_scale, self.layout
    ).clamp_min(1e-8)

    low_zero = _parameter_values(
        x, self.low_zero, self.layout
    )

    high_zero = _parameter_values(
        x, self.high_zero, self.layout
    )

    scale = torch.where(mask, low_scale, high_scale)
    zero = torch.where(mask, low_zero, high_zero)

    x32 = x.float()
    scale32 = scale.float()
    zero32 = zero.float()

    q = self.round_ste(x32 / scale32) + zero32

    qmax = torch.where(
        mask,
        torch.as_tensor(
            GIF_LOW_QMAX,
            dtype=q.dtype,
            device=q.device,
        ),
        torch.as_tensor(
            self.high_qmax,
            dtype=q.dtype,
            device=q.device,
        ),
    )

    q = torch.minimum(
        torch.clamp_min(q, 0.0),
        qmax,
    )

    dequantized = (q - zero32) * scale32
    return dequantized.to(x.dtype)
```

然后：

```python
def forward(
    self,
    x: torch.Tensor,
    *,
    role: str | None = None,
) -> torch.Tensor:
    return self._forward_ann_mixed_quant(x, role=role)
```

## 6.6 实现时必须保持的 GIF 语义

必须保持：

```text
low qmin  = 0
low qmax  = GIF_LOW_QMAX = 15

high qmin = 0
high qmax = self.high_qmax = 30

round_ste 完全保持
group expansion 完全保持
mask broadcast 完全保持
role-specific mask 完全保持
```

特别要覆盖：

```text
last_dim_grouped
attention_head_grouped
single-mask
multi-role mask
```

Site 1 / Site 7 的 role-specific mask 不能被合并或简化。

---

# 7. GIF temporal / 特殊 GIF 类型必须完全不动

普通 `StaticGIF` 之外，当前 factory 还可能返回：

```text
AllLowStaticGIF
SoftmaxIdentityGIF
IdentityGIF
```

本次不要修改它们。

## 7.1 `StaticGIF.temporal()`

保持原逻辑：

```text
incoming.sum(dim=0)
↓
原 low _quantize()
↓
原 high _quantize()
↓
integer_chunks(high_q)
↓
high q ∈ [0, 30]
拆为两个 [0, 15] chunk
↓
两个 temporal timestep
↓
torch.stack
```

禁止让 SNN temporal 使用新 ANN mixed quant。

## 7.2 `AllLowStaticGIF`

当前 temporal 会调用自己的 forward：

```python
quantized = self.forward(incoming.sum(dim=0), role=role)
```

而 All-low 本来就只有一套 low quantization，不存在 low/high 双 full-tensor 分支浪费。

所以：

```text
AllLowStaticGIF.forward()
AllLowStaticGIF.temporal()
```

全部不修改。

## 7.3 `SoftmaxIdentityGIF`

Site 5 GIF identity 的：

```text
forward()
temporal()
```

全部不改。

SNN temporal identity 行为必须保持 exact identity。

## 7.4 `IdentityGIF`

全部不改。

---

# 8. 修改后的执行路径必须明确变成下面这样

## 8.1 Phase ANN training

修改后：

```text
phase_aware ANN
      ↓
SiteController mode == "phase"
      ↓
PhaseSurrogate.forward()
      ↓
_forward_ann_streaming()
      ↓
逐 timestep 计算 contribution
      ↓
streaming sum
      ↓
返回普通 ANN tensor
```

没有 `[T, ...]` 的最终 `torch.stack`。

但 surrogate autograd 与 membrane recurrence 仍然保留。

## 8.2 Phase SNN deployment

必须仍然：

```text
deploy_phase
      ↓
SiteController
      ↓
PhaseSurrogate.temporal()
      ↓
原 encode(..., return_temporal=True)
      ↓
原 outputs list
      ↓
原 torch.stack
      ↓
原 temporal output
```

## 8.3 GIF-aware ANN training

普通 salient GIF：

```text
gif_aware ANN
      ↓
StaticGIF.forward()
      ↓
ANN-only mixed quant
      ↓
按 mask 选择 scale/zero/qmax
      ↓
一次 full-tensor FP32 fake quant
      ↓
普通 ANN tensor
```

## 8.4 GIF SNN deployment

必须仍然：

```text
deploy_gif
      ↓
StaticGIF.temporal()
      ↓
原 low _quantize()
      ↓
原 high _quantize()
      ↓
原 integer_chunks()
      ↓
原 two-step temporal decomposition
```

---

# 9. 测试要求：必须先建立“旧语义 reference”，再验收新实现

主要修改：

```text
tests/test_neurons.py
```

可以新增独立测试文件，例如：

```text
tests/test_ann_memory_equivalence.py
tests/test_training_memory_config.py
```

推荐独立文件，避免把现有 `test_neurons.py` 继续膨胀。

---

# 10. Phase ANN forward 等价性测试

## 10.1 利用保留的旧 `encode()` 作为 reference

由于本方案明确要求保留：

```python
encode(x, return_temporal=False)
```

的旧实现，因此测试可以直接做：

```text
reference = module.encode(x, return_temporal=False)
new       = module(x)
```

必须覆盖：

```text
last_dim_grouped
attention_head_grouped native [B,H,L,D]
attention_head_grouped merged [B,L,HD]
```

## 10.2 Phase forward test

固定随机种子。

对同一输入 clone：

```python
x_ref = x.detach().clone().requires_grad_(True)
x_new = x.detach().clone().requires_grad_(True)
```

创建相同 state 的 module。

比较：

```python
reference = module.encode(
    x_ref,
    return_temporal=False,
)

optimized = module(x_new)
```

要求：

```python
torch.testing.assert_close(
    optimized,
    reference,
    rtol=严格容差,
    atol=严格容差,
)
```

对 FP32 小型 unit test 可优先尝试：

```text
rtol=1e-6
atol=1e-7
```

若 reduction 顺序造成最后几位差异，可根据实际观察仅适度放宽，不允许使用宽松到掩盖算法错误的阈值。

## 10.3 Phase backward test

用同一固定 upstream gradient：

```python
grad = torch.randn_like(reference)
```

分别：

```python
reference.backward(grad)
optimized.backward(grad)
```

比较：

```text
x_ref.grad
x_new.grad
```

必须 `assert_close`。

这是关键验收项。

只检查 forward 不够，因为 `phase_aware` 的核心是 surrogate gradient training。

---

# 11. Phase temporal reference test

本轮 Phase SNN temporal path 不修改。

仍应增加一个 regression，防止未来改 ANN 时误伤 temporal。

使用：

```python
PhaseSurrogate(
    state,
    T=...,
    surrogate_slope=None,
)
```

模拟 deployment hard-spike module。

构造固定：

```text
incoming [T,B,...]
```

比较：

```python
temporal = module.temporal(incoming)

reference = module.encode(
    incoming.sum(dim=0),
    return_temporal=True,
)
```

要求优先：

```python
torch.equal(...)
```

如果当前 dtype/实现无法 exact equal，再使用极严格 `assert_close`。

同时保持现有：

```python
test_phase_hard_forward_and_temporal_sum_match()
```

等测试继续通过。

---

# 12. GIF ANN forward 等价性测试

由于生产 `forward()` 将变成 mixed quant，需要在测试中显式重建旧 forward。

测试 helper 可写成：

```python
def _legacy_static_gif_forward(
    module,
    x,
    *,
    role=None,
):
    low, _, _ = module._quantize(
        x,
        module.low_scale,
        module.low_zero,
        qmin=0,
        qmax=GIF_LOW_QMAX,
    )

    high, _, _ = module._quantize(
        x,
        module.high_scale,
        module.high_zero,
        qmin=0,
        qmax=module.high_qmax,
    )

    mask = _mask_values(
        x,
        module._mask(role),
        module.layout,
    )

    return torch.where(mask, low, high)
```

测试可以从：

```text
snn2.neurons
```

导入测试所需的 private helper：

```python
_mask_values
```

这是测试代码，不要求为此把 private function 改成 public API。

## 12.1 GIF forward test 必须覆盖

至少：

```text
1. last_dim_grouped + single mask
2. attention_head_grouped native [B,H,L,D]
3. attention_head_grouped merged [B,L,HD]
4. multi-role mask
5. 数据包含：
   - 不触发 clamp 的值
   - low branch clamp 边界附近值
   - high branch clamp 边界附近值
   - 超出 qrange 的值
```

比较：

```text
legacy full-low/full-high + where
vs
new mixed quant
```

使用严格 `torch.testing.assert_close`。

如果实际能 `torch.equal`，可以使用 exact equality；但不把 exact bitwise equality 作为整个训练实验的最终要求。

---

# 13. GIF backward 等价性测试

`StaticGIF` 的 scale/zero/mask 都是 calibration buffer，关键是输入 `x` 的 STE gradient。

构造：

```python
x_ref = ...
x_new = ...
```

然后：

```python
reference = _legacy_static_gif_forward(...)
optimized = module(x_new, role=...)
```

对同一：

```python
grad = torch.randn_like(reference)
```

分别 backward。

必须比较：

```text
x_ref.grad
x_new.grad
```

确保：

```text
round_ste
+
clamp
+
selected low/high branch
```

的 gradient semantics 没改变。

尤其要包含 clamp 内、clamp 外和边界附近值。

---

# 14. GIF temporal reference test

普通 `StaticGIF.temporal()` 不修改，但必须通过 regression 锁住。

建议在测试中写一个 `_legacy_static_gif_temporal()`，严格复刻当前 temporal：

```text
x = incoming.sum(dim=0)

low_q / low_zero
  ← 原 _quantize()

high_q / high_zero
  ← 原 _quantize()

mask
scale_low
scale_high

high_chunks = integer_chunks(high_q)

t=0:
  high_output -= high_zero * scale_high
  low_output = (low_q - low_zero) * scale_low

t=1:
  low_output = zeros

每 timestep:
  torch.where(mask, low_output, high_output)

torch.stack(outputs)
```

比较：

```python
module.temporal(incoming)
```

与 reference。

要求 exact 或非常严格 allclose。

---

# 15. 特殊 GIF temporal regression

现有测试已经包含部分行为，必须保留并确保继续通过：

```text
AllLowStaticGIF:
  timestep 1 必须全零
  temporal.sum(0) == quantized result

SoftmaxIdentityGIF:
  forward exact identity
  temporal exact identity

IdentityGIF:
  forward exact identity
  temporal exact identity
```

不要因为 ordinary `StaticGIF.forward()` 重构而修改这些类。

---

# 16. 配置 regression tests

新增或扩展：

```text
tests/test_generated_configs.py
```

至少检查：

## 16.1 Qwen3-8B

materialize 后：

```text
exp1_qwen3_8b_tldr__vanilla
exp1_qwen3_8b_tldr__unaware
exp1_qwen3_8b_tldr__phase_aware
exp1_qwen3_8b_tldr__gif_aware
```

均应满足：

```python
cfg["training"]["gradient_checkpointing"] is True
cfg["training"]["deepspeed_config"] == \
    "configs/deepspeed_zero3_cpu_offload.json"
```

同时必须保持：

```python
cfg["training"]["per_device_train_batch_size"] == 1
cfg["training"]["gradient_accumulation_steps"] == 16
cfg["data"]["max_seq_length"] == 2048
cfg["training"]["bf16"] is True
cfg["training"]["fp16"] is False
```

## 16.2 非 Qwen3-8B experiment 不应被意外修改

确认 Qwen3-1.7B / Llama experiment 的：

```text
deepspeed_config
gradient_checkpointing
```

只有在它们原本被明确修改时才变化。

本方案默认不改。

---

# 17. DeepSpeed JSON regression test

新增：

```text
tests/test_training_memory_config.py
```

读取：

```text
configs/deepspeed_zero3_cpu_offload.json
```

检查：

```python
zero = config["zero_optimization"]

assert zero["stage"] == 3
assert zero["offload_optimizer"]["device"] == "cpu"
assert zero["offload_optimizer"]["pin_memory"] is True
assert "offload_param" not in zero
```

还应检查原本关键设置没有被意外改变：

```python
assert zero["overlap_comm"] is True
assert zero["contiguous_gradients"] is True
assert zero["stage3_gather_16bit_weights_on_model_save"] is True
```

以及：

```python
assert config["bf16"]["enabled"] is True
assert config["fp16"]["enabled"] is False
```

---

# 18. 不需要修改的文件/逻辑

本轮正常情况下不需要修改：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/convert_snn.py
snn2/conversion.py
snn2/evaluation.py
snn2/temporal_model.py
snn2/temporal_ops.py
snn2/model_integration.py
snn2/controller.py
snn2/calibration.py
```

尤其：

```text
snn2/controller.py
```

当前已经通过不同 `mode` 区分：

```text
phase / gif ANN
deploy_phase / deploy_gif SNN
```

不要为了本次显存优化改变 controller routing。

---

# 19. `snn2/training.py` 原则上不需要结构性修改

当前已经把：

```python
gradient_checkpointing=...
```

传给 `TrainingArguments`。

因此本轮不要重新包装 Trainer 或手写 checkpoint function。

也不要改变：

```python
for parameter in model.parameters():
    parameter.requires_grad_(True)
```

仍保持 full-parameter ANN fine-tuning。

不要为了显存改成 freeze/LoRA。

如需增加少量日志，最多可以记录：

```text
gradient_checkpointing
deepspeed config path
```

但这不是本方案的必要功能，避免扩大修改范围。

---

# 20. 数值等价性的正确验收标准

本方案要求：

## 20.1 算法语义必须相同

必须保持：

```text
模型函数定义
Phase surrogate gradient 定义
GIF STE 定义
training loss
batch semantics
optimizer update rule
SNN temporal coding
conversion rule
calibration state semantics
```

## 20.2 不要求整个训练逐 bit 一致

以下修改都可能带来微小 floating-point trajectory 差异：

```text
Gradient Checkpointing recomputation
Phase reduction execution order
GIF parameter selection execution graph
CPU optimizer execution
```

所以：

```text
old final checkpoint SHA256
==
new final checkpoint SHA256
```

不是验收要求。

也不能要求最终 ROUGE 每一位都完全相同。

正确的底层验收为：

```text
Phase ANN forward: strict allclose
Phase ANN x.grad:  strict allclose

GIF ANN forward:   strict allclose
GIF ANN x.grad:    strict allclose

Phase temporal:    exact/strict regression
GIF temporal:      exact/strict regression

所有原 pytest:     PASS
```

然后再进行真实 Qwen3-8B train smoke test。

---

# 21. 推荐实施顺序

Codex 按以下顺序执行，不要一次混改后再排查。

## Step 1：先补 regression tests

先在旧代码上写：

```text
Phase legacy reference
GIF legacy reference
temporal reference
DeepSpeed/config expectations（可先预期失败）
```

旧实现的 reference test 本身必须能运行。

## Step 2：新增 DeepSpeed CPU offload 文件

新增：

```text
configs/deepspeed_zero3_cpu_offload.json
```

不要修改原共享 JSON。

## Step 3：修改 Qwen3-8B experiment matrix

修改：

```text
configs/experiment_matrix.yaml
```

Qwen3-8B：

```yaml
gradient_checkpointing: true
deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

其他核心训练超参数不动。

## Step 4：重新 materialize configs

运行：

```bash
python scripts/materialize_configs.py
```

检查生成 diff。

## Step 5：实现 Phase ANN-only streaming forward

只改：

```text
PhaseSurrogate.forward()
```

production routing。

新增：

```text
_forward_ann_streaming()
```

保留：

```text
encode()
temporal()
HeavisideSigmoid
```

reference/deployment 语义。

## Step 6：运行 Phase unit tests

至少：

```bash
pytest -q tests/test_neurons.py
pytest -q tests/test_ann_memory_equivalence.py
```

若 Phase forward/backward equivalence 不通过，先修复，不继续 GIF。

## Step 7：实现 StaticGIF ANN-only mixed quant

只改 ordinary：

```text
StaticGIF.forward()
```

新增 ANN-only method。

保留：

```text
_quantize()
integer_chunks()
temporal()
```

以及全部特殊 GIF class。

## Step 8：运行 GIF unit tests

再次：

```bash
pytest -q tests/test_neurons.py
pytest -q tests/test_ann_memory_equivalence.py
```

## Step 9：运行 config / DeepSpeed tests

例如：

```bash
pytest -q tests/test_generated_configs.py
pytest -q tests/test_training_memory_config.py
```

## Step 10：运行完整测试

必须：

```bash
pytest -q
```

全部通过后再进行 GPU 训练。

---

# 22. GPU smoke test

完整 pytest 通过后，使用 Qwen3-8B `phase_aware` / `gif_aware` 的真实训练命令进行 smoke test。

不要为了 smoke test 修改正式 experiment 的算法参数。

可以仅让训练跑到确认：

```text
model load 完成
DeepSpeed ZeRO-3 初始化完成
optimizer CPU offload 生效
第一个 forward 完成
第一个 backward 完成
第一个 optimizer step 完成
显存不 OOM
loss finite
```

如果项目已有通过 train sample 数进行快速测试的工作流，可以使用独立测试配置/临时副本；不要污染正式生成配置的实验定义。

---

# 23. 运行时必须检查的 DeepSpeed 信息

训练启动日志中确认：

```text
ZeRO stage = 3
optimizer offload = cpu
```

同时确认没有意外：

```text
parameter offload = cpu
```

本轮不要求 parameter offload。

如果 DeepSpeed 输出 optimizer implementation 信息，确认 optimizer 仍遵循当前 Adam/AdamW 及以下超参数：

```text
learning_rate
adam_beta1
adam_beta2
adam_epsilon
weight_decay
```

如发生不兼容，不要通过改变 optimizer 算法来强行绕过。

---

# 24. 显存验证建议

如果服务器环境允许，在 smoke test 前后记录同一 workload 下的：

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

或者使用系统 GPU 监控。

对比至少：

```text
旧 phase_aware peak
新 phase_aware peak

旧 gif_aware peak
新 gif_aware peak
```

目的不是设定固定下降百分比，而是确认修改确实减少峰值显存。

预期来源：

```text
Gradient Checkpointing:
  大幅减少 Transformer activation 常驻显存

Phase ANN streaming:
  去掉 ANN path 最终 T-way outputs list + stack 的额外峰值

GIF ANN mixed quant:
  去掉 ordinary GIF low/high 两套完整 FP32 fake-quant branch

Optimizer CPU offload:
  将 Adam optimizer state 从 GPU 移到 CPU RAM
```

---

# 25. 必须确认 SNN conversion/evaluation 不受影响

修改后使用同一个已有 ANN checkpoint / calibration artifact 做验证时：

## Phase

```text
Phase temporal implementation
必须与修改前相同
```

检查：

```text
T
tau
v0
threshold/amplitude
membrane recurrence
spike
temporal layout
```

均未发生代码语义变化。

## GIF

检查：

```text
StaticGIF.temporal()
_quantize()
integer_chunks()
AllLowStaticGIF
SoftmaxIdentityGIF
IdentityGIF
```

均没有被 ANN memory optimization 改写。

特别是 ordinary salient GIF 仍然：

```text
high integer q ∈ [0,30]
→ 两个 [0,15] integer chunks
→ 原 two-step temporal decomposition
```

不能让 SNN temporal 调用新的 ANN-only mixed quant。

---

# 26. 最终验收清单

提交修改前逐项确认：

```text
[ ] Qwen3-8B gradient_checkpointing = true

[ ] Qwen3-8B gradient_accumulation_steps 仍为 16

[ ] per_device_train_batch_size 仍为 1

[ ] max_seq_length 仍为 2048

[ ] 新建 deepspeed_zero3_cpu_offload.json

[ ] 原 deepspeed_zero3.json 未被破坏

[ ] 仅 optimizer offload 到 CPU

[ ] 未启用 parameter offload

[ ] Phase 新增 ANN-only streaming forward

[ ] Phase encode() temporal/reference 行为未改变

[ ] Phase temporal() 未改变

[ ] HeavisideSigmoid surrogate backward 未改变

[ ] 未新增 Phase custom backward

[ ] ordinary StaticGIF 新增 ANN-only mixed quant

[ ] StaticGIF._quantize() 未改变

[ ] StaticGIF.integer_chunks() 未改变

[ ] StaticGIF.temporal() 未改变

[ ] AllLowStaticGIF 未改变

[ ] SoftmaxIdentityGIF 未改变

[ ] IdentityGIF 未改变

[ ] Phase ANN forward equivalence PASS

[ ] Phase ANN backward equivalence PASS

[ ] GIF ANN forward equivalence PASS

[ ] GIF ANN backward equivalence PASS

[ ] Phase temporal regression PASS

[ ] GIF temporal regression PASS

[ ] special GIF temporal tests PASS

[ ] generated config tests PASS

[ ] DeepSpeed config tests PASS

[ ] pytest -q 全部 PASS

[ ] Qwen3-8B phase_aware GPU smoke test 不 OOM

[ ] Qwen3-8B gif_aware GPU smoke test 不 OOM（若资源允许）

[ ] loss finite

[ ] SNN temporal path 没有被修改
```

---

# 27. 最终实现原则总结

本次代码修改的最终架构应当是：

```text
                 ANN TRAINING
                     │
        ┌────────────┴────────────┐
        │                         │
      Phase                      GIF
        │                         │
new ANN streaming         new ANN mixed quant
        │                         │
        └──────────┬──────────────┘
                   │
          lower activation memory


              SNN DEPLOYMENT
                   │
        ┌──────────┴───────────┐
        │                      │
      Phase                   GIF
        │                      │
 original temporal()    original temporal()
        │                      │
 original encode()      original _quantize()
                               │
                       original integer_chunks()
        │                      │
        └──────────┬───────────┘
                   │
             semantics unchanged
```

核心要求只有一句：

> **优化 ANN training 的内存实现；Phase/GIF temporal deployment 保持为不可修改的 reference path。**

完成后，算法语义应保持一致；允许由于浮点 execution path 差异产生极小的数值差异，但不允许出现 forward/backward 语义变化或 SNN temporal coding 变化。
