# Qwen3-8B ZeRO-3 CPU Optimizer Offload 兼容性修正方案

## 0. 目的

仓库：

```text
https://github.com/wangwk699/SNN
```

本方案针对当前 `main` 中 Qwen3-8B ANN fine-tuning 的 ZeRO-3 CPU optimizer offload 配置做最后一轮兼容性修正。

当前已经确认正确并保持不变的内容：

```text
Qwen3-8B:
  gradient_checkpointing = false

Phase-aware ANN:
  PhaseSurrogate.forward()
    → _forward_ann_streaming()

GIF-aware ANN:
  StaticGIF.forward()
    → _forward_ann_mixed_quant()
    → tensor-bound torch.clamp()

Phase/GIF SNN temporal:
  保持原 reference path

DeepSpeed:
  ZeRO stage 3
  optimizer CPU offload enabled
  parameter offload disabled
```

本轮只解决一个问题：

> 当前 `configs/deepspeed_zero3_cpu_offload.json` 启用了 `offload_optimizer: cpu`，但没有显式配置 DeepSpeed optimizer。Transformers Trainer 因此会创建客户端 `torch.optim.AdamW`；DeepSpeed 0.16.9 默认 `zero_force_ds_cpu_optimizer=True`，在 CPU optimizer offload 场景下会拒绝非 `DeepSpeedCPUAdam` 的客户端 optimizer，导致真实训练可能在 DeepSpeed optimizer 初始化阶段直接失败。

本轮修改目标：

```text
在 DeepSpeed JSON 中显式声明 AdamW，
参数全部使用 "auto"，
让 Transformers 用现有 TrainingArguments 填充超参数，
并让 DeepSpeed 在 CPU offload 下自动创建 DeepSpeedCPUAdam(adamw_mode=True)。
```

---

# 1. 当前问题

项目锁定版本：

```text
transformers==4.53.2
deepspeed==0.16.9
```

当前：

```text
configs/deepspeed_zero3_cpu_offload.json
```

大致为：

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

问题是其中没有：

```json
"optimizer": ...
```

---

# 2. 为什么真实训练会有问题

当前 `snn2/training.py` 创建 Hugging Face `Trainer` / `TrainingArguments`，并没有手工创建 DeepSpeed optimizer。

当 DeepSpeed JSON 不含：

```json
"optimizer"
```

时，Transformers 会创建自己的客户端 optimizer，即当前默认的 AdamW。

但 DeepSpeed 0.16.9 的 ZeRO-Offload 逻辑在：

```text
offload_optimizer.device == cpu
```

且客户端 optimizer 不是：

```text
DeepSpeedCPUAdam
```

时，会检查：

```text
zero_force_ds_cpu_optimizer
```

该参数默认值为：

```text
True
```

因此会直接抛出 `ZeRORuntimeException`。

不要通过：

```json
"zero_force_ds_cpu_optimizer": false
```

来绕过。

本项目当前目标就是使用真正的 CPU optimizer offload，所以应使用 DeepSpeed 原生 CPU AdamW 实现。

---

# 3. 最终推荐方案

修改：

```text
configs/deepspeed_zero3_cpu_offload.json
```

显式增加：

```json
"optimizer": {
  "type": "AdamW",
  "params": {
    "lr": "auto",
    "betas": "auto",
    "eps": "auto",
    "weight_decay": "auto"
  }
}
```

最终完整配置应为：

```json
{
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
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

---

# 4. 为什么使用 `"auto"`

不要把当前训练超参数手工重复写入 DeepSpeed JSON。

当前 `snn2/training.py` 已经通过 `TrainingArguments` 提供：

```text
learning_rate
adam_beta1
adam_beta2
adam_epsilon
weight_decay
```

Transformers 的 DeepSpeed 集成会把：

```json
"lr": "auto"
"betas": "auto"
"eps": "auto"
"weight_decay": "auto"
```

分别替换为当前 `TrainingArguments` 中的值。

因此实际 optimizer 参数仍来自当前实验配置。

例如当前 Qwen3-8B experiment 中：

```yaml
learning_rate: 1.0e-06
weight_decay: 0.0
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1.0e-08
```

最终 DeepSpeed optimizer 应对应：

```text
lr = 1e-6
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.0
```

如果后续做 learning-rate sweep，`"auto"` 也能继续使用不同生成配置中的 learning rate，而不需要修改 DeepSpeed JSON。

---

# 5. AdamW 数学语义保持

DeepSpeed 配置：

```json
"optimizer": {
  "type": "AdamW"
}
```

在 CPU optimizer offload 场景下，DeepSpeed 0.16.9 会使用：

```text
DeepSpeedCPUAdam
```

并以：

```text
adamw_mode = true
```

运行。

因此 optimizer 仍然是 AdamW 更新规则。

本轮不得改动：

```text
learning rate
betas
epsilon
weight decay
gradient clipping
gradient accumulation
scheduler
warmup
```

允许 CPU optimizer 与原 GPU/PyTorch AdamW 在浮点最后几位存在数值差异，但算法更新规则必须保持 AdamW。

---

# 6. 不要采用的替代方案

不要添加：

```json
"zero_force_ds_cpu_optimizer": false
```

原因：

这会允许 DeepSpeed 使用客户端 `torch.optim.AdamW` 配合 CPU offload，但 DeepSpeed 自身明确提示这种组合通常性能较差，而且不是当前最标准的 ZeRO-Offload optimizer 路径。

本项目需要的是：

```text
ZeRO-3
+
DeepSpeed CPU optimizer offload
+
AdamW semantics
```

因此应使用：

```text
DeepSpeedCPUAdam(adamw_mode=True)
```

---

# 7. 不要修改 `snn2/training.py`

本轮不要修改：

```text
snn2/training.py
```

尤其不要：

```text
手工 import DeepSpeedCPUAdam
手工创建 optimizer
向 Trainer 传自定义 optimizer
修改 TrainingArguments optimizer routing
```

DeepSpeed JSON 已足够完成 optimizer 选择。

继续让：

```text
TrainingArguments
    ↓
Transformers DeepSpeed integration
    ↓
DeepSpeed JSON optimizer=AdamW
    ↓
DeepSpeedCPUAdam
```

完成配置。

---

# 8. 不要修改 experiment matrix

当前：

```text
configs/experiment_matrix.yaml
```

中的 Qwen3-8B 训练设置应保持：

```yaml
gradient_checkpointing: false
deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

不要再次修改 GC。

不要改：

```text
per_device_train_batch_size
gradient_accumulation_steps
max_seq_length
learning_rate
BF16
Prefix
```

---

# 9. 不需要重新设计 Prefix

不要修改：

```text
snn2/prefix_cache.py
```

当前 Gradient Checkpointing 已关闭，因此 Prefix 与 GC 的冲突已经通过最保守方案规避。

本轮只处理 optimizer。

---

# 10. 不修改 Phase/GIF neuron

不要修改：

```text
PhaseSurrogate._forward_ann_streaming()
PhaseSurrogate.forward()
PhaseSurrogate.encode()
PhaseSurrogate.temporal()

StaticGIF._forward_ann_mixed_quant()
StaticGIF.forward()
StaticGIF._quantize()
StaticGIF.integer_chunks()
StaticGIF.temporal()

AllLowStaticGIF
SoftmaxIdentityGIF
IdentityGIF
```

上一轮对应修改已经通过回归测试。

---

# 11. 修改配置回归测试

修改：

```text
tests/test_generated_configs.py
```

当前已有：

```python
def test_deepspeed_zero3_cpu_offload_is_optimizer_only():
```

在该测试中增加 optimizer 断言。

例如：

```python
def test_deepspeed_zero3_cpu_offload_is_optimizer_only():
    config = json.loads(
        (
            ROOT
            / "configs"
            / "deepspeed_zero3_cpu_offload.json"
        ).read_text(encoding="utf-8")
    )

    optimizer = config["optimizer"]

    assert optimizer["type"] == "AdamW"
    assert optimizer["params"] == {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto",
    }

    zero = config["zero_optimization"]

    assert zero["stage"] == 3
    assert zero["offload_optimizer"] == {
        "device": "cpu",
        "pin_memory": True,
    }

    assert "offload_param" not in zero

    assert zero["overlap_comm"] is True
    assert zero["contiguous_gradients"] is True
    assert (
        zero["stage3_gather_16bit_weights_on_model_save"]
        is True
    )

    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False
```

---

# 12. 增加防回归断言

同一个测试建议增加：

```python
assert "zero_force_ds_cpu_optimizer" not in config
```

确保后续没有通过关闭安全检查来绕过 DeepSpeedCPUAdam。

也可以增加：

```python
assert "torch_adam" not in optimizer["params"]
```

确保没有强制回 PyTorch AdamW。

核心预期是：

```text
optimizer.type = AdamW
offload_optimizer = cpu
```

让 DeepSpeed 自动选择 CPU AdamW implementation。

---

# 13. 是否需要重新 materialize configs

这次只修改：

```text
configs/deepspeed_zero3_cpu_offload.json
```

而 `configs/generated/*.yaml` 中只保存：

```yaml
deepspeed_config: configs/deepspeed_zero3_cpu_offload.json
```

路径没有变化。

因此理论上不需要因为 JSON 内容改变而重新 materialize configs。

不过为了保持运行前流程统一，可以运行：

```bash
python scripts/materialize_configs.py
```

然后确认 generated config 没出现意外 diff。

这不是本轮必要修改点。

---

# 14. 测试顺序

首先运行：

```bash
pytest -q tests/test_generated_configs.py
```

确认 DeepSpeed JSON regression 通过。

然后：

```bash
pytest -q tests/test_neurons.py
```

确认前几轮 Phase/GIF ANN optimization 和 temporal regression 没被误伤。

然后完整：

```bash
pytest -q
```

必须全部通过。

---

# 15. 静态测试仍不足以完成最终验收

即使：

```text
pytest -q
```

全部通过，也不能证明 DeepSpeedCPUAdam 能在目标服务器环境正常初始化。

本轮修正的最终验收必须包含真实 DeepSpeed smoke test。

---

# 16. Qwen3-8B smoke test 目标

优先测试：

```text
phase_aware
```

然后：

```text
gif_aware
```

使用现有正式训练入口：

```bash
torchrun \
  --standalone \
  --nproc_per_node="$NGPU" \
  scripts/train_ann.py \
  --config "$CFG"
```

可以使用现有的短训练/测试配置机制缩短运行，但不要修改正式实验数学参数来掩盖问题。

至少跑到：

```text
DeepSpeed initialization 完成
optimizer 初始化完成
第一个 forward 完成
第一个 backward 完成
第一个 optimizer step 完成
loss finite
```

---

# 17. smoke test 日志必须确认的内容

必须确认实际 optimizer 是 DeepSpeed CPU optimizer。

日志中应能看到与以下内容一致的信息：

```text
DeepSpeedCPUAdam
```

或能够明确证明：

```text
optimizer offload = cpu
AdamW
DeepSpeed CPU optimizer path
```

同时确认：

```text
ZeRO stage 3
```

确认：

```text
offload_optimizer device = cpu
```

确认没有：

```text
offload_param = cpu
```

因为本轮仍然只 offload optimizer state。

---

# 18. 不允许出现的错误

修正后不应再出现：

```text
ZeRORuntimeException:
You are using ZeRO-Offload with a client provided optimizer
(<class 'torch.optim.adamw.AdamW'>)
```

如果仍出现该错误，说明 DeepSpeed 没有读取到 JSON 中显式配置的：

```json
"optimizer": {
  "type": "AdamW"
}
```

需要检查实际运行使用的 resolved config / DeepSpeed config 路径，而不要通过：

```json
zero_force_ds_cpu_optimizer=false
```

绕过。

---

# 19. optimizer 参数核对

真实启动后确认最终 optimizer 参数与 experiment 保持一致。

当前 Qwen3-8B 应为：

```text
lr = 当前 generated config 的 learning_rate
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.0
```

如果运行的是 learning-rate sweep，则：

```text
lr
```

应随当前生成/临时配置变化。

这是采用 `"auto"` 的重要原因。

---

# 20. 最小修改文件集合

本轮正常情况下只需要修改：

```text
configs/deepspeed_zero3_cpu_offload.json
tests/test_generated_configs.py
```

不要修改其它生产代码。

---

# 21. 最终状态

完成本轮后，Qwen3-8B aware ANN fine-tuning 应为：

```text
Qwen3-8B phase_aware / gif_aware
        │
        ├── Prefix enabled
        │
        ├── Gradient Checkpointing = false
        │
        ├── Full-parameter fine-tuning
        │
        ├── BF16
        │
        ├── micro-batch = 1
        │
        ├── gradient accumulation = 16
        │
        ├── ZeRO stage 3
        │
        ├── optimizer = AdamW
        │       ↓
        │   DeepSpeedCPUAdam(adamw_mode=True)
        │
        ├── optimizer state offload → CPU
        │
        └── parameter offload → disabled
```

Neuron 路径：

```text
Phase ANN:
  _forward_ann_streaming()

GIF ANN:
  _forward_ann_mixed_quant()

Phase/GIF SNN:
  original temporal reference path
```

---

# 22. 最终验收清单

```text
[ ] deepspeed_zero3_cpu_offload.json 新增 optimizer

[ ] optimizer.type == "AdamW"

[ ] optimizer.params.lr == "auto"

[ ] optimizer.params.betas == "auto"

[ ] optimizer.params.eps == "auto"

[ ] optimizer.params.weight_decay == "auto"

[ ] zero_optimization.stage == 3

[ ] offload_optimizer.device == "cpu"

[ ] offload_optimizer.pin_memory == true

[ ] 未添加 offload_param

[ ] 未添加 zero_force_ds_cpu_optimizer=false

[ ] Qwen3-8B gradient_checkpointing 仍为 false

[ ] Qwen3-8B 仍引用 deepspeed_zero3_cpu_offload.json

[ ] training.py 未修改

[ ] prefix_cache.py 未修改

[ ] Phase neuron 未修改

[ ] GIF neuron 未修改

[ ] temporal deployment 未修改

[ ] tests/test_generated_configs.py 增加 optimizer regression

[ ] pytest -q tests/test_generated_configs.py PASS

[ ] pytest -q tests/test_neurons.py PASS

[ ] pytest -q 全部 PASS

[ ] 真实 DeepSpeed 初始化成功

[ ] 日志确认使用 DeepSpeedCPUAdam / CPU AdamW path

[ ] 第一个 forward 成功

[ ] 第一个 backward 成功

[ ] 第一个 optimizer step 成功

[ ] loss finite

[ ] 未出现 client AdamW + ZeRO-Offload ZeRORuntimeException
```

---

# 23. 最终原则

> **不要关闭 `zero_force_ds_cpu_optimizer` 来绕过 DeepSpeed 检查；应在 CPU-offload DeepSpeed JSON 中显式配置 `AdamW`，由 Transformers 的 `"auto"` 参数同步当前 TrainingArguments，并由 DeepSpeed 0.16.9 在 optimizer CPU offload 场景下创建 `DeepSpeedCPUAdam(adamw_mode=True)`。**

本轮除此之外不要扩大修改范围。
