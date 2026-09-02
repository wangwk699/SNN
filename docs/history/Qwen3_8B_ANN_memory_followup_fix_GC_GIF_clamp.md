# Qwen3-8B ANN 显存优化后续修正方案：关闭 Gradient Checkpointing + 修复 GIF qmax 边界梯度

## 0. 目的与适用仓库

仓库：

```text
https://github.com/wangwk699/SNN
```

本方案基于当前 `main` 最新代码（当前相关提交为 `7d220094ea0fdb0a8829eb0db4de1dd0448120b8`）制定。

上一轮已经完成：

1. Qwen3-8B 使用 `configs/deepspeed_zero3_cpu_offload.json`
2. Phase ANN 使用 `_forward_ann_streaming()`
3. ordinary salient `StaticGIF.forward()` 使用 ANN-only mixed quant
4. Phase/GIF SNN temporal 路径保持原实现
5. Qwen3-8B 开启了 `gradient_checkpointing: true`

重新检查后发现两个必须修正的问题：

1. 当前实验中 `phase_aware` / `gif_aware` ANN fine-tuning 一定启用 Pre-finetuning Prefix，而项目锁定的 Transformers Gradient Checkpointing 会清除 decoder layer 的 `past_key_value`，因此不能继续在这些训练中启用 Gradient Checkpointing。
2. 新的 GIF ANN mixed quant 使用 `torch.minimum(torch.clamp_min(...), qmax)`，在 `q == qmax` 边界处的 backward 与旧 `.clamp(qmin, qmax)` 不完全一致，必须恢复旧 clamp 的梯度语义。

本轮只修这两个问题。

---

# 1. 本轮最终显存优化策略

完成本轮后，Qwen3-8B `phase_aware` / `gif_aware` ANN fine-tuning 最终保留的显存优化为：

```text
Phase-aware ANN
├── Phase ANN-only streaming forward
└── ZeRO-3 optimizer CPU offload

GIF-aware ANN
├── ordinary StaticGIF ANN-only mixed fake quant
└── ZeRO-3 optimizer CPU offload
```

明确取消：

```text
Gradient Checkpointing
```

也就是说，本轮不要实现任何：

```text
Prefix-aware Gradient Checkpointing
Gradient Checkpointing-compatible Prefix cache
运行时 Prefix 检测后动态关闭 GC
```

最保守做法就是直接把 Qwen3-8B experiment 的：

```yaml
gradient_checkpointing: true
```

恢复为：

```yaml
gradient_checkpointing: false
```

---

# 2. 为什么不需要增加 Prefix 判断逻辑

当前项目实验定义中：

```text
vanilla       ANN fine-tuning：不使用 Prefix
unaware       ANN fine-tuning：使用 Pre-finetuning Prefix
phase_aware   ANN fine-tuning：使用 Pre-finetuning Prefix
gif_aware     ANN fine-tuning：使用 Pre-finetuning Prefix
```

当前 `snn2/config.py` 也明确要求：

```python
if mode == "vanilla" and training_prefix_enabled(cfg):
    raise ValueError(...)

if mode != "vanilla" and not training_prefix_enabled(cfg):
    raise ValueError(...)
```

本次需要解决 OOM 的目标训练就是：

```text
Qwen3-8B phase_aware
Qwen3-8B gif_aware
```

二者 100% 使用 Prefix。

因此没有必要在 `snn2/training.py` 增加：

```python
if training_prefix_enabled(cfg):
    gradient_checkpointing = False
```

之类的运行时分支。

本轮应保持实现简单：

```text
Qwen3-8B experiment
    ↓
gradient_checkpointing = false
```

`TrainingArguments` 继续照常读取配置即可。

不要改 `snn2/training.py` 的 Gradient Checkpointing routing。

---

# 3. 修改一：Qwen3-8B 恢复 `gradient_checkpointing: false`

## 3.1 修改文件

修改：

```text
configs/experiment_matrix.yaml
```

找到 Qwen3-8B TL;DR experiment：

```text
exp1_qwen3_8b_tldr
```

当前训练配置中有：

```yaml
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-06
  weight_decay: 0.0
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-08
  lr_scheduler_type: cosine
  warmup_ratio: 0.0
  bf16: true
  fp16: false
  dtype: bfloat16
  max_grad_norm: 1.0
  gradient_checkpointing: true
  attn_implementation: eager
  flash_attention: false
  eval_strategy: 'no'
  save_strategy: 'no'
  load_best_model_at_end: false
  logging_steps: 10
  deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

只把：

```yaml
gradient_checkpointing: true
```

改为：

```yaml
gradient_checkpointing: false
```

---

## 3.2 CPU optimizer offload 必须保留

不要回退：

```yaml
deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

仍然使用：

```text
configs/deepspeed_zero3_cpu_offload.json
```

不要重新切回：

```text
configs/deepspeed_zero3.json
```

不要删除：

```json
"offload_optimizer": {
  "device": "cpu",
  "pin_memory": true
}
```

也不要增加：

```json
"offload_param": ...
```

最终 Qwen3-8B 训练配置必须同时满足：

```yaml
gradient_checkpointing: false
deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

---

# 4. 不修改 Prefix 实现

本轮不要修改：

```text
snn2/prefix_cache.py
```

不要修改：

```python
install_prefix_kv_forward(...)
```

不要修改：

```python
_fresh_dynamic_cache(...)
```

不要修改 Prefix 的：

```text
attention_mask extension
position_ids offset
cache_position offset
fixed K/V injection
```

原因是本轮已经选择最保守方案：

```text
不用 Gradient Checkpointing
```

因此没有必要为了 GC 去重构 Prefix path。

---

# 5. 不修改 `snn2/training.py`

当前 `snn2/training.py` 已有：

```python
gradient_checkpointing=bool(
    training_cfg.get("gradient_checkpointing", False)
)
```

保持不变。

不要增加：

```python
training_prefix_enabled(...)
```

相关的动态判断。

不要增加：

```python
if prefix_enabled:
    gradient_checkpointing = False
```

不要修改 Trainer 初始化方式。

最终行为由 materialized config 直接决定：

```yaml
gradient_checkpointing: false
```

---

# 6. 重新 materialize configs

修改：

```text
configs/experiment_matrix.yaml
```

后运行：

```bash
python scripts/materialize_configs.py
```

重新生成所有 main experiment config。

重点检查：

```text
configs/generated/exp1_qwen3_8b_tldr__vanilla.yaml
configs/generated/exp1_qwen3_8b_tldr__unaware.yaml
configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml
configs/generated/exp1_qwen3_8b_tldr__gif_aware.yaml
```

四个 Qwen3-8B config 均应为：

```yaml
training:
  gradient_checkpointing: false
  deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

不需要为了 vanilla 单独开启 GC。

这次目标是保证 Qwen3-8B experiment 行为简单一致，并确保 `phase_aware/gif_aware` Prefix training 不受 GC 干扰。

---

# 7. 修改二：修复 GIF mixed quant 的 qmax 边界 backward

## 7.1 修改文件

修改：

```text
snn2/neurons.py
```

只修改：

```python
StaticGIF._forward_ann_mixed_quant(...)
```

不要修改：

```text
StaticGIF._quantize()
StaticGIF.temporal()
StaticGIF.integer_chunks()
AllLowStaticGIF
SoftmaxIdentityGIF
IdentityGIF
```

---

# 8. 当前有问题的代码

当前：

```python
def _forward_ann_mixed_quant(
    self, x: torch.Tensor, *, role: str | None = None
) -> torch.Tensor:
    mask = _mask_values(x, self._mask(role), self.layout)
    low_scale = _parameter_values(x, self.low_scale, self.layout).clamp_min(1e-8)
    high_scale = _parameter_values(x, self.high_scale, self.layout).clamp_min(1e-8)
    low_zero = _parameter_values(x, self.low_zero, self.layout)
    high_zero = _parameter_values(x, self.high_zero, self.layout)

    scale = torch.where(mask, low_scale, high_scale)
    zero = torch.where(mask, low_zero, high_zero)

    q = self.round_ste(x.float() / scale.float()) + zero.float()

    qmax = torch.where(
        mask,
        torch.as_tensor(GIF_LOW_QMAX, dtype=q.dtype, device=q.device),
        torch.as_tensor(self.high_qmax, dtype=q.dtype, device=q.device),
    )

    q = torch.minimum(torch.clamp_min(q, 0.0), qmax)

    return ((q - zero.float()) * scale.float()).to(x.dtype)
```

问题只在：

```python
q = torch.minimum(torch.clamp_min(q, 0.0), qmax)
```

forward 数值与旧 quantizer 一致，但在：

```text
q == qmax
```

处 backward 不完全等价。

---

# 9. 为什么这个边界必须修

旧 ordinary GIF `_quantize()`：

```python
q = (
    self.round_ste(x.float() / scale.float())
    + zero.float()
).clamp(qmin, qmax)
```

其中：

```text
low  qmax = 15
high qmax = 30
```

新 mixed quant 为了支持每 channel 不同 qmax，把上界写成：

```python
torch.minimum(..., qmax)
```

但是：

```text
torch.clamp
```

和：

```text
torch.minimum
```

在输入恰好等于 upper bound 时的 autograd 行为不同。

GIF 使用：

```python
round_ste(...)
```

forward 会产生离散 integer-like code，因此：

```text
q == 15
q == 30
```

并不是可以忽略的极端浮点事件。

这会改变 GIF-aware ANN fine-tuning 的 surrogate/STE gradient，违反本轮“数学语义不变”的要求。

---

# 10. 正确修改方式

将：

```python
q = torch.minimum(torch.clamp_min(q, 0.0), qmax)
```

修改为 tensor upper bound 的 `torch.clamp`：

```python
q = torch.clamp(
    q,
    min=0.0,
    max=qmax,
)
```

最终建议：

```python
def _forward_ann_mixed_quant(
    self, x: torch.Tensor, *, role: str | None = None
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

    q = self.round_ste(
        x.float() / scale.float()
    ) + zero.float()

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

    q = torch.clamp(
        q,
        min=0.0,
        max=qmax,
    )

    return (
        (q - zero.float()) * scale.float()
    ).to(x.dtype)
```

---

# 11. 必须保留 GIF mixed quant 的其他逻辑

不要因为修 clamp 又恢复到旧的双 full-tensor quantization。

仍然保留：

```text
mask
  ↓
选择 effective scale
  ↓
选择 effective zero
  ↓
选择 per-channel qmax
  ↓
一次 x.float()
  ↓
一次 round_ste
  ↓
一次 clamp
  ↓
一次 dequantize
```

不能退回：

```text
full low quant
+
full high quant
+
torch.where
```

否则会失去上一轮 GIF ANN 显存优化。

---

# 12. GIF SNN temporal 路径必须继续完全不动

当前 `StaticGIF.temporal()` 仍然：

```text
incoming.sum(dim=0)
    ↓
原 low _quantize()
    ↓
原 high _quantize()
    ↓
integer_chunks(high_q)
    ↓
two-step temporal decomposition
```

本轮不要修改。

特别是：

```python
self._quantize(...)
```

仍然保持旧 `.clamp(qmin, qmax)`。

不要让 temporal path 调用：

```python
_forward_ann_mixed_quant(...)
```

因此完成修改后仍必须满足：

```text
GIF ANN:
    new memory-optimized mixed quant

GIF SNN:
    original reference temporal quantization
```

---

# 13. Phase ANN / SNN 路径不修改

上一轮 Phase 修改是正确的。

当前：

```python
def forward(self, x):
    return self._forward_ann_streaming(x)
```

保持。

当前：

```python
def temporal(self, incoming):
    ...
    return self.encode(
        incoming.sum(dim=0),
        return_temporal=True,
    )
```

保持。

不要修改：

```text
PhaseSurrogate.encode()
PhaseSurrogate._forward_ann_streaming()
PhaseSurrogate.forward()
PhaseSurrogate.temporal()
HeavisideSigmoid
```

本轮 Phase 不需要任何代码修改。

---

# 14. 修改现有 generated-config regression

修改：

```text
tests/test_generated_configs.py
```

当前：

```python
def test_qwen3_8b_memory_optimized_training_configuration(generated_configs):
    ...
    for cfg in qwen3_8b:
        training = cfg["training"]
        assert training["gradient_checkpointing"] is True
        assert training["deepspeed_config"] == \
            "configs/deepspeed_zero3_cpu_offload.json"
```

把：

```python
assert training["gradient_checkpointing"] is True
```

改为：

```python
assert training["gradient_checkpointing"] is False
```

其他断言全部保留：

```python
assert training["deepspeed_config"] == \
    "configs/deepspeed_zero3_cpu_offload.json"

assert training["per_device_train_batch_size"] == 1
assert training["gradient_accumulation_steps"] == 16
assert cfg["data"]["max_seq_length"] == 2048
assert training["bf16"] is True
assert training["fp16"] is False
```

---

# 15. 保留 DeepSpeed regression

当前：

```python
def test_deepspeed_zero3_cpu_offload_is_optimizer_only():
```

继续保留。

必须继续检查：

```python
zero["stage"] == 3
zero["offload_optimizer"] == {
    "device": "cpu",
    "pin_memory": True,
}
"offload_param" not in zero
```

也就是说关闭 GC 不代表撤销 CPU optimizer offload。

---

# 16. 非 Qwen3-8B config test 的处理

当前已有：

```python
def test_non_qwen3_8b_training_memory_settings_remain_unchanged(...):
    ...
    assert cfg["training"]["gradient_checkpointing"] is False
    assert cfg["training"]["deepspeed_config"] == \
        "configs/deepspeed_zero3.json"
```

该测试应继续保持。

最终配置规则：

```text
Qwen3-8B:
  gradient_checkpointing = false
  deepspeed = zero3_cpu_offload

其他现有 experiment:
  gradient_checkpointing = false
  deepspeed = 原 zero3
```

---

# 17. GIF 边界 regression：必须新增

修改：

```text
tests/test_neurons.py
```

现有测试：

```python
test_static_gif_ann_mixed_quant_matches_legacy_forward_and_input_gradient
```

虽然已经检查普通输入的 forward/backward 等价性，但没有明确锁住：

```text
q == 15
q == 30
```

因此必须增加专门边界测试。

---

# 18. 边界测试目标

至少覆盖下面五类 integer-code 位置：

```text
q < 0
q == 0
0 < q < qmax
q == qmax
q > qmax
```

并且分别覆盖：

```text
low branch:  qmax = 15
high branch: qmax = 30
```

最关键的是：

```text
q == 15
q == 30
```

必须显式出现，而不是依赖随机数碰到。

---

# 19. 推荐 GIF 边界测试方式

继续使用当前已经存在的 legacy helper：

```python
def _legacy_static_gif_forward(module, x, *, role=None):
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

    return torch.where(
        _mask_values(
            x,
            module._mask(role),
            module.layout,
        ),
        low,
        high,
    )
```

不要自己写一个新的“理论 reference clamp”。

直接以项目原 `_quantize()` 作为 reference。

---

# 20. 构造恰好命中 qmax 的输入

现有 test state 中：

```text
low_scale  = 0.1
low_zero   = 0
high_scale = 0.05
high_zero  = 0
```

因此可以精确构造：

low branch：

```text
x = 15 * 0.1 = 1.5
```

使：

```text
round(x / low_scale) = 15
```

high branch：

```text
x = 30 * 0.05 = 1.5
```

同样使：

```text
round(x / high_scale) = 30
```

由于当前 test state 两个边界都可以用：

```text
x = 1.5
```

命中，需通过 mask 分别让某 channel 走 low/high branch。

还可以构造：

```text
low above qmax:
x = 1.6  → round(16) > 15

high above qmax:
x = 1.55 → round(31) > 30
```

实际测试值应避免 banker rounding 的半整数歧义，使用明确落在整数 code 的输入。

---

# 21. 必须比较 forward 和 x.grad

测试结构：

```python
x_reference = x.detach().clone().requires_grad_(True)
x_optimized = x.detach().clone().requires_grad_(True)

reference = _legacy_static_gif_forward(
    module,
    x_reference,
)

optimized = module(
    x_optimized,
)

torch.testing.assert_close(
    optimized,
    reference,
    rtol=0,
    atol=0,
)
```

如果 exact forward equality 因 dtype/实现细节无法满足，可使用极严格容差；但 FP32 unit test 应优先要求 exact。

然后使用固定 upstream gradient：

```python
grad = torch.ones_like(reference)
```

或固定非均匀 gradient：

```python
grad = torch.tensor(...)
```

执行：

```python
reference.backward(grad)
optimized.backward(grad)
```

必须：

```python
torch.testing.assert_close(
    x_optimized.grad,
    x_reference.grad,
    rtol=0,
    atol=0,
)
```

至少对于 FP32 boundary unit test 应尝试 exact equality。

---

# 22. 建议新增测试名称

例如：

```python
def test_static_gif_ann_mixed_quant_matches_legacy_clamp_boundary_gradients():
    ...
```

或者拆成：

```python
@pytest.mark.parametrize(...)
def test_static_gif_ann_mixed_quant_matches_legacy_at_qmax_boundary(...):
    ...
```

重点是测试名明确指出：

```text
qmax boundary gradient
```

避免以后误删。

---

# 23. multi-role GIF test 继续保留

当前：

```python
test_static_gif_ann_mixed_quant_matches_legacy_multi_role_masks
```

必须继续通过。

本轮修改不能破坏：

```text
Site 1 q/k/v role-specific mask
Site 7 gate/up role-specific mask
```

或项目实际保存的 role policy。

---

# 24. temporal regression 继续保留

当前：

```python
test_static_gif_temporal_still_matches_legacy_reference
```

必须继续通过。

这用于确保修 ANN clamp 时没有误改：

```text
StaticGIF.temporal()
_quantize()
integer_chunks()
```

---

# 25. 本轮不需要修改的主要文件

正常情况下不要修改：

```text
snn2/training.py
snn2/config.py
snn2/prefix_cache.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_ops.py
snn2/evaluation.py
scripts/train_ann.py
scripts/evaluate_tldr.py
scripts/convert_snn.py
```

也不要修改：

```text
configs/deepspeed_zero3_cpu_offload.json
```

除非检查发现文件内容与上一轮不同；当前 main 下它已经正确。

---

# 26. 最小修改文件集合

本轮预计只需要修改：

```text
configs/experiment_matrix.yaml

snn2/neurons.py

tests/test_generated_configs.py

tests/test_neurons.py
```

然后执行：

```bash
python scripts/materialize_configs.py
```

如果 `configs/generated/` 为未提交生成物，只需要重新 materialize 并在本地用于运行/测试；按项目当前版本控制规则处理即可。

---

# 27. 实施顺序

按以下顺序执行。

## Step 1

修改：

```text
configs/experiment_matrix.yaml
```

Qwen3-8B：

```yaml
gradient_checkpointing: false
```

保留：

```yaml
deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

---

## Step 2

修改：

```text
tests/test_generated_configs.py
```

把 Qwen3-8B：

```python
gradient_checkpointing is True
```

期望改为：

```python
gradient_checkpointing is False
```

---

## Step 3

重新：

```bash
python scripts/materialize_configs.py
```

---

## Step 4

先运行：

```bash
pytest -q tests/test_generated_configs.py
```

必须通过。

---

## Step 5

修改：

```text
snn2/neurons.py
```

只把：

```python
q = torch.minimum(
    torch.clamp_min(q, 0.0),
    qmax,
)
```

改为：

```python
q = torch.clamp(
    q,
    min=0.0,
    max=qmax,
)
```

---

## Step 6

在：

```text
tests/test_neurons.py
```

增加 qmax boundary forward/backward regression。

---

## Step 7

运行：

```bash
pytest -q tests/test_neurons.py
```

必须通过。

---

## Step 8

运行所有相关 regression：

```bash
pytest -q \
  tests/test_generated_configs.py \
  tests/test_neurons.py
```

---

## Step 9

运行完整测试：

```bash
pytest -q
```

全部通过。

---

# 28. 完成后必须满足的代码状态

## Qwen3-8B training

```text
gradient_checkpointing = false
```

```text
optimizer CPU offload = enabled
```

```text
parameter CPU offload = disabled
```

---

## Phase-aware ANN

继续：

```text
PhaseSurrogate.forward()
    ↓
_forward_ann_streaming()
```

---

## Phase SNN

继续：

```text
PhaseSurrogate.temporal()
    ↓
原 encode(..., return_temporal=True)
```

---

## GIF-aware ANN

继续：

```text
StaticGIF.forward()
    ↓
_forward_ann_mixed_quant()
    ↓
effective low/high scale/zero/qmax
    ↓
single FP32 quant path
    ↓
torch.clamp(min=0, max=qmax)
```

---

## GIF SNN

继续：

```text
StaticGIF.temporal()
    ↓
原 _quantize()
    ↓
原 integer_chunks()
    ↓
原 temporal decomposition
```

---

# 29. 完整验收清单

提交前确认：

```text
[ ] Qwen3-8B gradient_checkpointing 已恢复 false

[ ] 没有新增 Prefix runtime GC 判断

[ ] 没有修改 prefix_cache.py

[ ] Qwen3-8B 仍使用 deepspeed_zero3_cpu_offload.json

[ ] optimizer CPU offload 仍启用

[ ] 没有启用 parameter offload

[ ] Phase _forward_ann_streaming 未改变

[ ] Phase temporal 未改变

[ ] StaticGIF mixed quant 仍为单 full-tensor FP32 path

[ ] GIF mixed quant qmax 截断改为 torch.clamp

[ ] StaticGIF._quantize() 未改变

[ ] StaticGIF.temporal() 未改变

[ ] StaticGIF.integer_chunks() 未改变

[ ] AllLowStaticGIF 未改变

[ ] SoftmaxIdentityGIF 未改变

[ ] IdentityGIF 未改变

[ ] q == 15 的 low branch forward regression PASS

[ ] q == 15 的 low branch backward regression PASS

[ ] q == 30 的 high branch forward regression PASS

[ ] q == 30 的 high branch backward regression PASS

[ ] q > qmax saturation regression PASS

[ ] existing mixed quant equivalence tests PASS

[ ] multi-role GIF tests PASS

[ ] GIF temporal regression PASS

[ ] generated config tests PASS

[ ] pytest -q 全部 PASS
```

---

# 30. 后续真实 GPU smoke test

本轮测试全部通过后，再执行 Qwen3-8B：

```text
phase_aware
gif_aware
```

真实 DeepSpeed smoke test。

此时重点确认：

```text
Gradient Checkpointing 没有启用
Prefix KV 正常参与 ANN training
ZeRO Stage 3 正常
optimizer offload = CPU
parameter offload 未启用
第一个 forward 正常
第一个 backward 正常
第一个 optimizer step 正常
loss finite
没有 OOM
```

由于取消了 Gradient Checkpointing，峰值 activation memory 会高于上一版“理论上启用 GC”的配置；但上一版 GC 与当前 Prefix 训练语义不兼容，因此不能作为有效方案。

本轮最终有效的显存优化来源为：

```text
Phase ANN streaming
GIF ANN single mixed fake quant
ZeRO-3 CPU optimizer offload
```

---

# 31. 最终原则

本次修正后必须满足：

> **Prefix-enabled Qwen3-8B aware ANN fine-tuning 不使用 Gradient Checkpointing；Phase/GIF 的 ANN-only neuron 显存优化与 ZeRO-3 optimizer CPU offload 保留。GIF mixed quant 必须在 qmax 边界处与旧 `_quantize().clamp()` 的 forward/backward 语义一致。**

本轮不要扩展到其他显存优化方案。
