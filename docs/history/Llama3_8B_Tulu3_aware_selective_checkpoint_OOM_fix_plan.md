# Llama3-8B Tulu-3 `phase_aware` / `gif_aware` ANN Training OOM 修复方案

> 仓库：`https://github.com/wangwk699/SNN`  
> 基线：实施时以 `main` 分支最新代码为准。  
> 目标：在 **保持 `max_seq_length: 2048`** 的前提下，通过 **Attention-Core Selective Checkpoint + MLP Selective Checkpoint** 解决 Llama3-8B Tulu-3 `phase_aware` / `gif_aware` ANN Training 的 OOM。  
> 本轮属于 **memory-equivalent implementation optimization**，不是算法修改。

---

## 1. 本轮最终决定

正式实验继续保持：

```yaml
data:
  max_seq_length: 2048
  truncation: true
  truncation_side: right
  packing: false
```

训练超参继续保持：

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: false
  bf16: true
  deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

本轮只新增两项：

1. **Attention-Core Selective Activation Checkpointing**
2. **MLP Selective Activation Checkpointing**

明确 **不启用** Hugging Face / Transformers whole-decoder Gradient Checkpointing：

```yaml
gradient_checkpointing: false
```

原因：当前 `phase_aware` / `gif_aware` ANN Training 必须使用 Pre-finetuning Prefix / fixed KV cache，不能简单恢复之前已撤销的 decoder-level Gradient Checkpointing。

---

## 2. 为什么这轮应该这样改

当前 Llama3-8B Tulu-3 aware ANN Training 的主要峰值来自：

```text
Tulu-3 长序列
    +
max_seq_length = 2048
    +
自定义 snn2 eager attention 显式 materialize [B,H,L,L+P]
    +
gradient_checkpointing = false
    +
Phase/GIF aware replacement 的额外 backward graph
    +
Llama3 较宽的 MLP intermediate activation
    =
单卡 activation peak OOM
```

当前 micro-batch 已经是 1，因此改变 `gradient_accumulation_steps` 不能有效降低单步峰值显存。

Selective checkpoint 的目标是：

```text
Forward 少保存 activation
        ↓
Backward 时重新计算 attention / MLP core
        ↓
降低 peak GPU memory
```

代价是训练时间增加，但数学函数、数据和训练目标保持不变。

---

## 3. 本轮绝对不能修改的内容

### 数据与 batch

不要修改：

```text
max_seq_length = 2048
truncation_side = right
packing = false
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
```

### Hugging Face Gradient Checkpointing

必须继续：

```yaml
gradient_checkpointing: false
```

### DeepSpeed

继续使用：

```text
configs/deepspeed_zero3_cpu_offload.json
```

本轮不要新增：

```json
"offload_param": ...
```

### Attention backend

不要改成：

```text
flash_attention_2
sdpa
```

当前 Site 5 需要显式 attention probability，直接替换会改变现有 replacement 语义。

### Prefix

不要修改：

```text
snn2/prefix_cache.py
install_prefix_kv_forward()
_fresh_dynamic_cache()
attention mask extension
position_ids offset
cache_position offset
fixed past_key_values
```

### Phase / GIF

不得修改：

```text
PhaseSurrogate 数学定义
Phase T / tau / surrogate_slope
Phase streaming ANN forward
Phase temporal deployment
StaticGIF 数学定义
GIF base_bits / add_bits / qmax
GIF mask / saliency
GIF mixed fake quant
GIF temporal decomposition
```

### 其他实验语义

不得修改：

```text
rotation
R3 / R4
Hadamard 语义
calibration
Clip
replacement site topology
loss
labels / assistant-token mask
optimizer
learning rate
scheduler
epochs
ANN evaluation
SNN conversion/deployment
```

---

## 4. 涉及的主要文件

本轮预计修改：

```text
configs/experiment_matrix.yaml
snn2/config.py
snn2/controller.py
snn2/training.py
snn2/model_integration.py
tests/test_selective_activation_checkpointing.py
```

并重新运行：

```bash
python scripts/materialize_configs.py
```

不要因为本轮修改去重构：

```text
snn2/neurons.py
snn2/prefix_cache.py
snn2/temporal_model.py
snn2/temporal_ops.py
snn2/calibration.py
snn2/conversion.py
snn2/evaluation.py
```

---

## 5. 新增独立的 ANN-training memory 配置

为了避免 `configs/experiment_matrix.yaml` 中 YAML nested merge 的浅覆盖问题，建议不要为了 mode-specific 开关重写整个 `training:` block。

新增顶层 section：

```yaml
ann_training_memory:
  attention_core_checkpoint: false
  mlp_checkpoint: false
```

在 `snn2/config.py::resolve_config()` 中增加默认值：

```python
cfg.setdefault("ann_training_memory", {})
cfg["ann_training_memory"].setdefault(
    "attention_core_checkpoint", False
)
cfg["ann_training_memory"].setdefault(
    "mlp_checkpoint", False
)
```

这样旧配置和 Qwen 实验默认都保持原行为。

---

## 6. `configs/experiment_matrix.yaml` 修改

### 6.1 Llama3 base / vanilla / unaware

保持关闭：

```yaml
ann_training_memory:
  attention_core_checkpoint: false
  mlp_checkpoint: false
```

### 6.2 Llama3 `phase_aware`

在当前 variant 中增加：

```yaml
- name: exp2_llama3_8b_tulu3
  ann_modes: [phase_aware]
  config:
    <<: *exp2_llama3_8b_tulu3_base
    conversion:
      use_post_finetuning_artifacts: false
    ann_training_memory:
      attention_core_checkpoint: true
      mlp_checkpoint: true
```

### 6.3 Llama3 `gif_aware`

增加：

```yaml
- name: exp2_llama3_8b_tulu3
  ann_modes: [gif_aware]
  config:
    <<: *exp2_llama3_8b_tulu3_base
    conversion:
      use_post_finetuning_artifacts: false
    ann_training_memory:
      attention_core_checkpoint: true
      mlp_checkpoint: true
```

### 6.4 Qwen3

Qwen3-1.7B / Qwen3-8B 所有 mode 都保持：

```yaml
attention_core_checkpoint: false
mlp_checkpoint: false
```

本轮不要改变已经验证过的 Qwen 训练路径。

---

## 7. `snn2/config.py` validation

在 `validate_config()` 中严格检查：

```python
memory_cfg = cfg["ann_training_memory"]
```

只允许：

```text
attention_core_checkpoint
mlp_checkpoint
```

两个字段都必须是 `bool`。

建议拒绝未知 key。

### 7.1 aware-only invariant

若任意 flag 为 True：

```python
enabled = (
    memory_cfg["attention_core_checkpoint"]
    or memory_cfg["mlp_checkpoint"]
)
```

则必须：

```python
is_aware_ann_mode(cfg)
```

也就是禁止：

```text
vanilla + checkpoint=true
unaware + checkpoint=true
```

### 7.2 禁止和 Transformers GC 叠加

若 `enabled=True`，必须要求：

```python
cfg["training"]["gradient_checkpointing"] is False
```

否则报错，例如：

```python
raise ValueError(
    "Selective ANN checkpointing must not be combined with "
    "Transformers gradient_checkpointing"
)
```

---

## 8. `snn2/controller.py`

在 `SiteController.__init__()` 增加两个 keyword-only 参数：

```python
checkpoint_attention_core: bool = False,
checkpoint_mlp: bool = False,
```

保存：

```python
self.checkpoint_attention_core = bool(checkpoint_attention_core)
self.checkpoint_mlp = bool(checkpoint_mlp)
```

除此之外，不修改：

```text
_load()
apply()
set_deployment()
apply_final_norm_neuron()
```

这些 flag 只是 memory implementation metadata，不改变 neuron 行为。

---

## 9. `snn2/training.py`

在 `train_full_parameters()` 创建 `SiteController` 前读取：

```python
memory_cfg = cfg.get("ann_training_memory", {})
```

创建 controller 时传入：

```python
controller = SiteController(
    ...,
    checkpoint_attention_core=bool(
        memory_cfg.get("attention_core_checkpoint", False)
    ),
    checkpoint_mlp=bool(
        memory_cfg.get("mlp_checkpoint", False)
    ),
)
```

其他 calibration / evaluation / conversion / deployment 创建 `SiteController` 的地方不需要主动传值，默认 `False` 即可。

---

## 10. Selective checkpoint 的统一启用条件

在 `snn2/model_integration.py` 中增加统一 helper，例如：

```python
def _selective_checkpoint_allowed(
    module: torch.nn.Module,
    controller: SiteController,
    *,
    kind: str,
) -> bool:
    if controller.mode not in {"phase", "gif"}:
        return False

    if not module.training:
        return False

    if not torch.is_grad_enabled():
        return False

    # Regression recorder 有 forward side effects。
    # checkpoint backward 会 recompute，因此 recorder 存在时禁用。
    if controller.regression_recorder is not None:
        return False

    if kind == "attention":
        return bool(controller.checkpoint_attention_core)
    if kind == "mlp":
        return bool(controller.checkpoint_mlp)

    raise ValueError(kind)
```

必须同时满足：

```text
controller.mode == phase/gif
module.training == True
grad enabled
regression_recorder is None
对应 checkpoint flag == True
```

才允许 checkpoint。

因此以下路径自动保持原实现：

```text
vanilla
unaware
collect
ANN evaluation
SNN evaluation
deploy_phase
deploy_gif
deploy_mtn
任何 regression recorder 检查
```

---

# 11. Attention-Core Selective Checkpoint

## 11.1 只 checkpoint attention 的核心大张量区域

当前显存热点大致为：

```python
qk = torch.matmul(query, key.transpose(2, 3))
weights = qk * scale

if attention_mask is not None:
    weights = weights + ...

if softcap is not None:
    weights = ...

weights = F.softmax(
    weights,
    dim=-1,
    dtype=torch.float32,
).to(query.dtype)

weights = controller.apply(layer_index, 5, weights)
weights = F.dropout(...)
output_heads = torch.matmul(weights, value)
```

这里会显式生成 `[B,H,L,L+P]` attention activation，是 2048-token Tulu-3 的核心 OOM 来源。

---

## 11.2 正确 checkpoint 边界

建议只把以下部分抽成 `_attention_core()`：

```text
QK^T
→ scaling
→ attention mask
→ softcap（若存在）
→ FP32 softmax
→ cast 回 query dtype
→ Site 5 controller.apply()
→ dropout
→ P @ V
```

以下仍留在 core 外：

```text
R3
Site 2
repeat_kv
Site 3
Site 4
head merge/restore
Site 6
```

这样修改范围最小。

---

## 11.3 `_attention_core()` 必须保持原计算顺序

示意：

```python
def _attention_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
):
    qk = torch.matmul(query, key.transpose(2, 3))

    scale = float(
        scaling
        if scaling is not None
        else getattr(
            module,
            "scaling",
            1.0 / math.sqrt(query.shape[-1]),
        )
    )

    weights = qk * scale

    if attention_mask is not None:
        weights = weights + attention_mask[..., : key.shape[-2]]

    if softcap is not None:
        cap = float(softcap)
        weights = torch.tanh(weights / cap) * cap

    weights = F.softmax(
        weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)

    weights = controller.apply(
        layer_index,
        5,
        weights,
    )

    weights = F.dropout(
        weights,
        p=dropout,
        training=module.training,
    )

    output_heads = torch.matmul(weights, value)

    return output_heads, weights
```

变量名可按当前代码风格调整，但数学顺序不能变。

---

## 11.4 checkpoint 调用

导入：

```python
from torch.utils.checkpoint import checkpoint
```

当允许 checkpoint：

```python
output_heads, weights = checkpoint(
    _attention_core,
    query,
    key,
    value,
    attention_mask,
    use_reentrant=False,
)
```

否则：

```python
output_heads, weights = _attention_core(
    query,
    key,
    value,
    attention_mask,
)
```

必须使用：

```python
use_reentrant=False
```

不要使用旧 reentrant checkpoint。

---

## 11.5 必须继续返回 `weights`

不要为了省显存把 attention backend 改为：

```python
return output, None
```

当前函数对外是：

```python
return output, weights
```

本轮必须保留该语义，避免破坏 `output_attentions` 或 Transformers attention interface 的行为。

在常规 training `output_attentions=False` 时，外层不长期保留 `weights`；checkpoint 的主要收益仍来自不为 backward 保存内部 QK / softmax / Site5 / PV 中间 graph。

---

## 11.6 RNG / Dropout

不要设置：

```python
preserve_rng_state=False
```

保持 PyTorch checkpoint 默认 RNG preservation。

即使当前 attention dropout 为 0，也要保证以后 dropout 非零时 backward recompute 使用一致 RNG 状态。

---

## 11.7 Regression recorder 必须禁用 checkpoint

当前项目 regression recording 有 forward side effects。

若 checkpoint 在 backward recompute 时再次执行：

```text
forward record
+
backward recompute 再 record
```

会污染回归数据。

因此：

```python
controller.regression_recorder is not None
```

时必须走原始非-checkpoint 路径。

不要尝试重构 recorder 去区分 recompute；本轮采用最保守方案。

---

## 11.8 `controller.apply(site5)` 的 lazy load 不需要改

第一次 forward 可能 lazy load `phase_state.pt` / `gif_state.pt` 并把 buffer 移到 GPU。

这是允许的。

Backward recompute 会复用已经加载好的 controller module。

不要为了 checkpoint 重构 `_load()` 或 state cache。

---

# 12. MLP Selective Checkpoint

当前 `_make_mlp_forward()` 大致：

```python
gate_projection = mlp.gate_proj(x)
up_projection = mlp.up_proj(x)

gate = mlp.act_fn(gate_projection)
gate = controller.apply(layer_index, 8, gate)

up = controller.apply(layer_index, 9, up_projection)

product = gate * up

if r4 is not None:
    product = random_hadamard(...)

product = controller.apply(layer_index, 10, product)
output = mlp.down_proj(product)
```

Llama3-8B 的 MLP intermediate activation 很宽，`gate_projection / up_projection / product` 等 `[B,L,d_ff]` tensor 是第二个主要显存来源。

---

## 12.1 把整个 MLP body 放进 checkpoint core

推荐：

```python
def _mlp_core(x: torch.Tensor) -> torch.Tensor:
    # 保持当前 gate_proj / up_proj / SiLU / Site8 / Site9
    # / product / R4 / Site10 / down_proj 的原顺序
    ...
    return output
```

外层：

```python
if _selective_checkpoint_allowed(
    mlp,
    controller,
    kind="mlp",
):
    return checkpoint(
        _mlp_core,
        x,
        use_reentrant=False,
    )

return _mlp_core(x)
```

---

## 12.2 R4 必须留在 MLP checkpoint core 内

保持原顺序：

```text
gate_proj / up_proj
→ SiLU
→ Site8 / Site9
→ gate * up
→ R4 Hadamard
→ Site10
→ down_proj
```

不要为了 checkpoint 把 R4 移到其他位置。

把 R4 放在 checkpoint core 内，可以避免在 checkpoint 外长期保存大 `product` graph。

---

## 12.3 Deployment MLP 不 checkpoint

由于 helper 要求：

```python
controller.mode in {"phase", "gif"}
```

以下模式全部自动关闭：

```text
deploy_phase
deploy_gif
deploy_mtn
```

不要在 temporal deployment 分支添加任何 checkpoint。

---

# 13. 不要给 PhaseSurrogate / StaticGIF 再单独套 checkpoint

当前：

```text
PhaseSurrogate.forward()
```

已有 ANN-only streaming forward。

当前：

```text
StaticGIF.forward()
```

已有 ANN-only mixed fake quant forward。

本轮不要再做：

```python
checkpoint(PhaseSurrogate.forward, ...)
checkpoint(StaticGIF.forward, ...)
```

因为 attention / MLP 外层 checkpoint 已经会在 backward 重算里面的 Phase/GIF replacement。

避免 nested checkpoint。

---

# 14. Prefix 路径保持完全不变

Selective checkpoint 位于已经建立 fixed Prefix KV 之后的 decoder compute core。

因此不要修改：

```text
snn2/prefix_cache.py
```

尤其不要修改：

```text
past_key_values injection
DynamicCache
attention_mask extension
position_ids
cache_position
Prefix KV tensor layout
```

这正是本方案相比 whole-layer HF Gradient Checkpointing 的核心优势。

---

# 15. Training metadata

建议在 `training_result.json` 增加：

```json
{
  "attention_core_checkpoint": true,
  "mlp_checkpoint": true,
  "transformers_gradient_checkpointing": false
}
```

用于 provenance。

本轮不要因为 selective checkpoint 改 artifact path；它是数学等价的 memory implementation，不是新的实验变量。

---

# 16. 重新 materialize configs

运行：

```bash
python scripts/materialize_configs.py
```

重点检查：

```text
configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml
configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml
configs/generated/exp2_llama3_8b_tulu3__vanilla.yaml
configs/generated/exp2_llama3_8b_tulu3__unaware.yaml
```

---

# 17. Generated config 验收

## Llama3 phase-aware / gif-aware

必须：

```yaml
data:
  max_seq_length: 2048

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: false
  deepspeed_config: configs/deepspeed_zero3_cpu_offload.json

ann_training_memory:
  attention_core_checkpoint: true
  mlp_checkpoint: true
```

## Llama3 vanilla / unaware

必须：

```yaml
ann_training_memory:
  attention_core_checkpoint: false
  mlp_checkpoint: false
```

## Qwen3 所有实验

全部保持 false。

---

# 18. 必须新增的测试

建议新增：

```text
tests/test_selective_activation_checkpointing.py
```

如果已有更合适的 test 文件可以复用，但以下覆盖不能减少。

---

## Test A：默认配置兼容

旧配置没有 `ann_training_memory` 时，resolve 后应：

```python
attention_core_checkpoint is False
mlp_checkpoint is False
```

---

## Test B：aware-only validation

以下必须 raise：

```text
vanilla + checkpoint=true
unaware + checkpoint=true
```

以下允许：

```text
phase_aware + checkpoint=true
gif_aware + checkpoint=true
```

---

## Test C：禁止叠加 HF GC

以下必须 raise：

```yaml
training:
  gradient_checkpointing: true

ann_training_memory:
  attention_core_checkpoint: true
```

MLP true 同理。

---

## Test D：Attention forward equivalence

构造小 tensor，例如：

```text
B=2
H=4
L=8
D=16
```

比较 checkpoint OFF / ON：

```text
output_heads
attention weights
```

必须 `assert_close`。

---

## Test E：Attention backward equivalence

用相同 input / parameter 初值，定义 scalar loss：

```python
loss = output.float().square().mean()
```

分别 backward，比较：

```text
query.grad
key.grad
value.grad
相关 trainable parameter grad
```

FP32 unit test 建议严格到约：

```text
rtol=1e-5
atol=1e-6
```

若 custom op 只能 BF16，则使用合理浮点 tolerance，但不能只测 forward。

---

## Test F：Phase-aware Attention checkpoint

至少覆盖：

```text
controller.mode = phase
Site 5 Phase replacement
forward equivalence
backward equivalence
```

优先复用现有 minimal Phase state fixture。

---

## Test G：GIF-aware Attention checkpoint

至少覆盖：

```text
controller.mode = gif
Site 5 GIF path
forward equivalence
backward equivalence
```

---

## Test H：Prefix-length attention shape

模拟：

```text
Q length = L
K/V length = L + P
attention_mask length = L + P
```

例如：

```text
L=8
P=3
```

验证 checkpoint OFF / ON 均正确并等价。

这是 Prefix compatibility 的关键 shape regression。

---

## Test I：Dropout RNG

设置非零 attention dropout，OFF / ON 前使用同一 random seed。

比较 forward/backward，确保 checkpoint 没有关闭 RNG preservation。

---

## Test J：MLP forward/backward equivalence

覆盖：

```text
gate_proj
up_proj
SiLU
Site8
Site9
product
R4
Site10
down_proj
```

比较：

```text
output
input.grad
gate_proj.weight.grad
up_proj.weight.grad
down_proj.weight.grad
```

---

## Test K：R4-enabled MLP

不能只测试 `r4=None`。

至少一个测试启用真实或测试用等价 R4 spec，保证 checkpoint OFF / ON 输出和 gradient 一致。

---

## Test L：Regression recorder guard

即使 flags=true，只要：

```python
controller.regression_recorder is not None
```

helper 必须返回 false。

---

## Test M：eval / deployment guard

验证以下全部不 checkpoint：

```text
module.eval()
collect
identity
none
deploy_phase
deploy_gif
deploy_mtn
```

---

# 19. 完整测试

完成后必须：

```bash
pytest -q
```

全部通过。

不要只跑新增测试。

如果已有 snapshot / config regression 因新 section 需要更新，必须同步更新。

---

# 20. 数值等价要求

本轮的核心验收不是“能跑”，而是：

```text
checkpoint OFF
vs
checkpoint ON
```

在相同：

```text
input
weights
random seed
Prefix
Phase/GIF state
```

下：

```text
forward output 一致
gradient 一致到正常浮点误差
```

如果出现明显 gradient mismatch：

> 不允许通过放宽 tolerance 掩盖问题。

必须检查：

```text
checkpoint boundary
RNG
regression side effect
custom autograd
controller lazy load
R4 recompute
```

---

# 21. GPU 实际验收顺序

不要直接全量训练后才判断。

推荐先用临时配置做三步 A/B；临时配置不要提交。

### Stage 1：Attention only

```yaml
attention_core_checkpoint: true
mlp_checkpoint: false
```

运行 Llama3 phase-aware，确认：

```text
无 Prefix 错误
无 autograd 错误
显存明显下降
```

### Stage 2：MLP only

```yaml
attention_core_checkpoint: false
mlp_checkpoint: true
```

确认 MLP checkpoint 独立可训练。

### Stage 3：最终正式组合

```yaml
attention_core_checkpoint: true
mlp_checkpoint: true
```

先运行 phase-aware，再运行 gif-aware。

最终 `experiment_matrix.yaml` 只保留 Stage 3 正式配置。

---

# 22. 4×A100-80G 正式验收

Phase-aware 示例：

```bash
CFG=configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/train_ann.py \
  --config "$CFG"
```

GIF-aware：

```bash
CFG=configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/train_ann.py \
  --config "$CFG"
```

如果项目已有专用 shell wrapper，优先使用已有 wrapper。

必须确认：

```text
不再 CUDA OOM
能越过此前 OOM 的长样本
Prefix 正常
DeepSpeed ZeRO-3 正常
loss 正常变化
无 checkpoint backward error
Final ANN checkpoint 正常保存
training_result.json 正常生成
```

不能只因为第一步 forward 成功就判定 OOM 已修复。

---

# 23. 建议观察峰值显存

可用：

```bash
watch -n 1 nvidia-smi
```

如果方便，在临时 profiling 中记录：

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

比较：

```text
checkpoint OFF
attention only
MLP only
attention + MLP
```

预期：

```text
Attention checkpoint → 最大显存收益
MLP checkpoint       → 进一步增加余量
```

因为 attention 主要随：

```text
O(L^2)
```

增长，而 MLP activation 主要随：

```text
O(L * d_ff)
```

增长。

---

# 24. ANN evaluation / SNN deployment 必须保持无 checkpoint

Final ANN evaluation 通常 `model.eval()`，因此 helper 必须返回 false。

SNN deployment mode：

```text
deploy_phase
deploy_gif
deploy_mtn
```

也必须返回 false。

本轮不要修改 temporal Phase/GIF/MTN 路径。

---

# 25. 不修改 artifact path

Selective checkpoint 属于数学等价的 memory implementation。

不要在输出路径里加入：

```text
attention_checkpoint_true
mlp_checkpoint_true
```

只在 resolved config / training metadata 中记录即可。

---

# 26. 本轮禁止顺手加入的其他优化

Codex 不要自行加入：

```text
FlashAttention
SDPA
sequence parallel
context parallel
tensor parallel
activation CPU offload
ZeRO offload_param
LoRA / QLoRA
8-bit optimizer
batch size 改动
gradient accumulation 改动
max_seq_length 改动
packing
length bucketing
按样本长度动态 checkpoint
torch.compile
FSDP
```

本轮只做两项 selective checkpoint。

---

# 27. 不要硬编码模型名或任务名

不要写：

```python
if "Llama" in model_name:
    checkpoint = True
```

也不要写：

```python
if task == "tulu3":
    checkpoint = True
```

是否启用必须由：

```yaml
ann_training_memory
```

控制。

实现保持通用；当前只在 Llama3 Tulu-3 aware config 中启用。

---

# 28. 不要按 sequence length 动态切换

不要写：

```python
if seq_len > 1024:
    checkpoint(...)
else:
    normal(...)
```

正式 aware 训练统一：

```text
Attention checkpoint = ON
MLP checkpoint = ON
```

避免不同样本走不同计算实现，增加复现复杂度。

---

# 29. 推荐注释

在 helper 上加入类似：

```python
# ANN-aware selective activation checkpointing.
#
# This checkpoints only the attention/MLP compute cores after fixed Prefix KV
# injection has already been established. It is intentionally separate from
# Transformers decoder-level gradient checkpointing, which remains disabled
# for Prefix-aware ANN training.
```

在 recorder guard 上：

```python
# Regression recording has forward side effects. Disable checkpoint
# recomputation whenever a recorder is attached.
```

---

# 30. 最终 Checklist

## Config

- [ ] Llama3 Tulu-3 `max_seq_length=2048`
- [ ] micro-batch=1
- [ ] gradient accumulation=16
- [ ] `gradient_checkpointing=false`
- [ ] DeepSpeed 仍是 optimizer CPU offload 配置
- [ ] Llama3 phase-aware attention checkpoint=true
- [ ] Llama3 phase-aware MLP checkpoint=true
- [ ] Llama3 gif-aware attention checkpoint=true
- [ ] Llama3 gif-aware MLP checkpoint=true
- [ ] Llama3 vanilla/unaware=false
- [ ] Qwen3 所有 mode=false

## Semantics

- [ ] Prefix 未修改
- [ ] Phase 未修改
- [ ] GIF 未修改
- [ ] Site topology 未修改
- [ ] R3/R4 未修改
- [ ] Attention 数学顺序未修改
- [ ] MLP 数学顺序未修改
- [ ] temporal deployment 未修改

## Gating

- [ ] 只允许 controller `phase` / `gif`
- [ ] 只允许 `module.training`
- [ ] 只允许 grad enabled
- [ ] recorder 存在时关闭
- [ ] deploy/collect/eval 自动关闭

## Tests

- [ ] default config regression
- [ ] aware-only validation
- [ ] HF GC conflict validation
- [ ] attention forward equivalence
- [ ] attention backward equivalence
- [ ] Phase attention checkpoint
- [ ] GIF attention checkpoint
- [ ] Prefix-length shape
- [ ] dropout RNG
- [ ] MLP forward/backward
- [ ] R4-enabled MLP
- [ ] regression recorder guard
- [ ] eval/deployment guard
- [ ] `pytest -q` 全部通过

## GPU

- [ ] 4×A100-80G phase-aware 不再 OOM
- [ ] 4×A100-80G gif-aware 不再 OOM
- [ ] 能越过此前 OOM 长样本
- [ ] loss 正常
- [ ] Prefix 正常
- [ ] DeepSpeed 正常
- [ ] Final checkpoint 正常保存

---

# 31. 如果两项 checkpoint 后仍然 OOM

不要擅自改成 `max_seq_length=1024`。

先记录：

```text
max_memory_allocated
max_memory_reserved
OOM requested allocation
allocated/reserved/free
触发 OOM 样本 token length
```

然后停止本轮修改。

下一轮再单独评估：

```text
ZeRO-3 parameter CPU offload
activation CPU offload
更大的 checkpoint boundary
sequence/context parallel
```

这些都不属于当前方案。

---

# 32. 最终目标状态

```text
Llama3-8B + Tulu-3
max_seq_length = 2048
micro batch = 1
gradient accumulation = 16
BF16
Transformers gradient checkpointing = OFF
ZeRO-3 optimizer CPU offload = ON
Pre-finetuning fixed Prefix KV = ON

phase_aware:
  Phase ANN streaming = 保持当前实现
  Attention-Core selective checkpoint = ON
  MLP selective checkpoint = ON

gif_aware:
  StaticGIF ANN mixed fake quant = 保持当前实现
  Attention-Core selective checkpoint = ON
  MLP selective checkpoint = ON

ANN evaluation:
  selective checkpoint = OFF

SNN deployment:
  selective checkpoint = OFF
```

核心原则：

> **通过 backward recomputation 降低 2048-token Tulu-3 aware ANN Training 的 activation peak，而不是通过缩短 context、改变 Prefix、改变 neuron 或改变训练数据来规避 OOM。**
