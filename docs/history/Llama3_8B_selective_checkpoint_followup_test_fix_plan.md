# Llama3-8B Tulu-3 Selective Checkpoint 后续修补方案
## —— 修复测试可复现性，并补齐真实 Phase / GIF / R4 回归覆盖

> 适用仓库：`https://github.com/wangwk699/SNN`  
> 基线：当前 `main` 分支已经完成 Llama3-8B Tulu-3 `phase_aware` / `gif_aware` 的 Attention-Core Selective Checkpoint 与 MLP Selective Checkpoint。  
> 本轮目标：**不重构现有 selective checkpoint 主体实现，只修复当前测试中的确定问题，并补齐关键 regression coverage。**

---

# 1. 本轮结论

当前 selective checkpoint 主体实现整体正确，本轮不要修改其核心数学路径。

本轮只处理以下 3 个问题：

1. **修复 `tests/test_selective_activation_checkpointing.py` 直接依赖 `configs/generated/` 的问题**
2. **补充真实 `PhaseSurrogate` / `StaticGIF` 与 checkpoint 的 forward/backward regression**
3. **补充真实 R4 Hadamard 与 MLP checkpoint 的 forward/backward regression**

完成后运行：

```bash
pytest -q
```

全部通过。

本轮不需要再次修改：

```text
Attention-Core checkpoint boundary
MLP checkpoint boundary
Prefix implementation
Phase/GIF neuron implementation
SNN temporal deployment
DeepSpeed
max_seq_length
batch size
training hyperparameters
artifact path
```

---

# 2. 当前已确认无需修改的主体实现

当前以下逻辑保持不动。

## 2.1 Selective checkpoint gating

继续保持：

```python
if controller.mode not in {"phase", "gif"}:
    return False

if not module.training:
    return False

if not torch.is_grad_enabled():
    return False

if controller.regression_recorder is not None:
    return False
```

并继续读取：

```python
controller.checkpoint_attention_core
controller.checkpoint_mlp
```

因此：

```text
phase/gif ANN training → selective checkpoint 可开启
vanilla/unaware       → 不开启
collect               → 不开启
deploy_*              → 不开启
model.eval()          → 不开启
regression recorder   → 不开启
```

这一部分不要改。

---

# 3. Attention-Core checkpoint 主体保持不动

当前：

```text
QKᵀ
→ scale
→ mask
→ softcap
→ FP32 softmax
→ Site 5
→ dropout
→ P×V
```

放在 checkpoint core 内是正确的。

以下仍在 core 外：

```text
R3
Site 2
repeat_kv
Site 3
Site 4
Site 6
```

继续保持。

当前：

```python
checkpoint(
    attention_core,
    query,
    key,
    value,
    attention_mask,
    use_reentrant=False,
)
```

继续保持。

不要增加：

```python
preserve_rng_state=False
```

不要改变 attention 返回：

```python
return output, weights
```

---

# 4. MLP checkpoint 主体保持不动

当前：

```text
gate_proj
up_proj
SiLU
Site 8
Site 9
gate * up
R4
Site 10
down_proj
```

放在 `mlp_core()` 中 checkpoint 是正确的。

继续：

```python
checkpoint(
    mlp_core,
    x,
    use_reentrant=False,
)
```

不要拆成更细的 nested checkpoint。

不要给：

```text
PhaseSurrogate.forward()
StaticGIF.forward()
```

单独增加 checkpoint。

---

# 5. 问题一：测试不能依赖 `configs/generated/`

当前：

```text
configs/generated/
```

位于 `.gitignore`：

```gitignore
configs/generated/
```

因此全新 clone 后，不保证存在：

```text
configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml
configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml
```

当前：

```text
tests/test_selective_activation_checkpointing.py
```

直接读取这些 generated YAML，导致：

```text
fresh clone
→ pytest
→ FileNotFoundError
```

即使开发机因为之前运行过：

```bash
python scripts/materialize_configs.py
```

而测试通过，也不能说明测试可复现。

这是本轮必须修的问题。

---

# 6. 修复方式：测试内部 materialize 到 `tmp_path`

不要再定义：

```python
def _generated_config(mode: str) -> dict:
    return yaml.safe_load(
        (
            ROOT
            / "configs"
            / "generated"
            / f"exp2_llama3_8b_tulu3__{mode}.yaml"
        ).read_text(...)
    )
```

改为测试 fixture。

参考项目中：

```text
tests/test_generated_configs.py
```

已有写法。

推荐在：

```text
tests/test_selective_activation_checkpointing.py
```

新增：

```python
from scripts.materialize_configs import materialize_configs
```

然后：

```python
@pytest.fixture()
def generated_configs(tmp_path):
    paths = materialize_configs(
        ROOT / "configs" / "experiment_matrix.yaml",
        tmp_path / "generated",
    )

    result = {}
    for path in paths:
        if path.stem.startswith("exp2_llama3_8b_tulu3__"):
            mode = path.stem.split("__", 1)[1]
            result[mode] = yaml.safe_load(
                path.read_text(encoding="utf-8")
            )

    return result
```

最终应能得到：

```python
generated_configs["vanilla"]
generated_configs["unaware"]
generated_configs["phase_aware"]
generated_configs["gif_aware"]
```

---

# 7. 修改所有 config tests

例如当前：

```python
cfg = _generated_config("phase_aware")
```

改成：

```python
cfg = copy.deepcopy(
    generated_configs["phase_aware"]
)
```

所有需要修改 cfg 的测试都必须：

```python
copy.deepcopy(...)
```

避免同一个 fixture object 被前一个 test mutation 污染。

例如：

```python
def test_memory_config_rejects_transformers_gradient_checkpointing(
    generated_configs,
):
    cfg = copy.deepcopy(
        generated_configs["phase_aware"]
    )
    cfg["training"]["gradient_checkpointing"] = True
    ...
```

---

# 8. 保留 legacy default regression

原测试想验证：

```text
没有 ann_training_memory 的旧配置
→ resolve_config()
→ 两个 flag 自动 false
```

继续保留。

改为：

```python
def test_memory_config_defaults_and_llama3_variants(
    generated_configs,
):
    legacy = copy.deepcopy(
        generated_configs["vanilla"]
    )
    legacy.pop("ann_training_memory")

    resolved = resolve_config(legacy)

    assert resolved["ann_training_memory"] == {
        "attention_core_checkpoint": False,
        "mlp_checkpoint": False,
    }

    validate_config(resolved)

    ...
```

---

# 9. 问题二：当前 Phase/GIF regression 只是 fake controller

当前测试类似：

```python
class _ReplacementController:
    ...
    def apply(...):
        return value * self.gain
```

即使：

```python
mode="phase"
mode="gif"
```

实际计算也只是：

```text
value * gain
```

所以当前测试并没有验证：

```text
real PhaseSurrogate
real HeavisideSigmoid custom autograd
real StaticGIF fake quant
```

与：

```text
torch.utils.checkpoint(use_reentrant=False)
```

的兼容性。

本轮必须增加真实 neuron regression。

---

# 10. 不要删除现有 lightweight tests

当前 `_ReplacementController` 测试仍然有价值。

它可以快速验证：

```text
attention checkpoint boundary
prefix-shaped K/V
dropout RNG
generic backward equivalence
checkpoint gating
```

所以不要删。

本轮是在此基础上新增：

```text
real Phase
real GIF
real R4
```

测试。

---

# 11. 复用现有 neuron state helper

当前：

```text
tests/test_neurons.py
```

已有：

```python
_phase_state(...)
_gif_state(...)
```

但不建议跨 test module 直接 import 私有 helper：

```python
from tests.test_neurons import _phase_state
```

避免 test 文件之间产生隐式依赖。

推荐把最小 state helper 在：

```text
tests/test_selective_activation_checkpointing.py
```

中重新定义一份，或者提取到公共：

```text
tests/helpers.py
```

如果项目目前没有统一 `tests/helpers.py`，本轮优先选择：

> **在新 test 文件内部定义最小 helper**

避免为了测试顺手重构整个 tests 目录。

---

# 12. 新增真实 Phase state helper

根据当前 `PhaseSurrogate` 的 state contract，构造最小：

```python
def _real_phase_state():
    ...
```

必须包含当前代码要求的：

```text
state_kind
format_version
temporal_implementation_version

parameter_layout
configured_group_size
group_size
num_heads
channels_per_head
groups_per_head

tau
tau_calibration
tau_ema_factor
tau_accumulator_dtype
tau_channel_policy
tau_reduction_policy
tau_clamp_min
tau_clamp_max
tau_clamp_policy
```

不要自己硬编码版本常量。

从当前项目模块 import：

```python
from snn2.phase_statistics import ...
from snn2.temporal_ops import ...
```

与现有 `tests/test_neurons.py` 保持一致。

建议测试 shape 用简单：

```text
last_dim_grouped
channels = 4
group_size = 2
```

例如：

```python
x.shape = [B, L, 4]
```

这样 state 简单。

---

# 13. 新增真实 StaticGIF state helper

同样定义：

```python
def _real_gif_state():
    ...
```

使用普通 grouped StaticGIF，而不是 Site 5 的 softmax identity GIF。

原因：

> Site 5 GIF 当前是 identity，无法覆盖真实 StaticGIF fake-quant backward。

因此真实 GIF checkpoint regression 应重点放在：

```text
MLP Site 8 / Site 9 / Site 10
```

而不是强行要求 attention Site 5 做 ordinary GIF。

必须符合当前 state contract：

```text
base_bits = 4
add_bits = 1
low_qmin = 0
low_qmax = 15
high_qmin = 0
high_qmax = 30
temporal_steps = 2
per_step_qmin = 0
per_step_qmax = 15
integer_decomposition = current constant
gif_policy = current ordinary GIF policy
runtime layout fields
low_scale
low_zero
high_scale
high_zero
mask_low
```

---

# 14. 增加真实 Phase Attention checkpoint regression

因为 Attention Site 5 对 Phase 是真实 Phase replacement，因此这里适合直接覆盖真实 custom autograd。

新增一个 controller fixture，例如：

```python
class _RealPhaseController:
    mode = "phase"
    checkpoint_attention_core = ...
    checkpoint_mlp = ...
    regression_recorder = None

    def __init__(...):
        self.phase = PhaseSurrogate(
            _real_phase_state(),
            T=4,
            surrogate_slope=1.0,
        )

    def apply(self, layer, site, x, **kwargs):
        if site == 5:
            return self.phase(x)
        return x
```

注意：

```text
Attention Site 5 tensor shape = [B,H,L,K]
```

因此 `_real_phase_state()` 的 parameter layout 必须兼容最后一维。

为了简单，可使用：

```text
configured_group_size = -1
group_size = -1
```

或根据项目当前 grouped Phase state contract 构造能匹配 Site 5 最后一维的最小状态。

如果当前 Phase Site 5 state 实际采用不同 layout，则按正式 calibration state contract 构造，不能为了测试绕过 shape validation。

---

# 15. Real Phase Attention 测试内容

创建：

```python
def _real_phase_attention_run(checkpoint_enabled: bool):
    ...
```

使用：

```text
B=1
H=2
L=4
K=4 or L+P
D=4
```

避免测试过大。

执行：

```text
checkpoint OFF
checkpoint ON
```

使用完全相同：

```text
random seed
query
key
value
mask
Phase state
```

loss：

```python
loss = output.float().square().mean()
loss.backward()
```

至少比较：

```text
output
weights
query.grad
key.grad
value.grad
```

使用：

```python
torch.testing.assert_close(...)
```

推荐 CPU FP32：

```text
rtol=1e-5
atol=1e-6
```

如果严格相等可以更严。

---

# 16. 必须显式覆盖 `HeavisideSigmoid`

Real Phase test 的目的之一就是确保：

```python
HeavisideSigmoid.apply(...)
```

内部：

```python
ctx.save_for_backward(x)
```

与 non-reentrant checkpoint backward recompute 相容。

因此不要在该测试中 monkeypatch：

```text
PhaseSurrogate
HeavisideSigmoid
controller.apply
```

成 identity。

---

# 17. GIF 的真实回归不要依赖 Attention Site 5

当前项目 GIF Site 5 是：

```text
softmax identity GIF
```

因此 Attention checkpoint 的：

```text
mode="gif"
```

现有 lightweight controller test 继续保留即可，用于验证 checkpoint generic path。

真正的 StaticGIF fake quant regression 应放到 MLP。

这是与当前项目实际 topology 一致的测试方式。

---

# 18. 新增真实 GIF MLP controller

例如：

```python
class _RealGIFController:
    mode = "gif"
    checkpoint_attention_core = False
    checkpoint_mlp = ...

    regression_recorder = None

    def __init__(self):
        self.gif8 = StaticGIF(_real_gif_state())
        self.gif9 = StaticGIF(_real_gif_state())
        self.gif10 = StaticGIF(_real_gif_state())

    def apply(self, layer, site, x, **kwargs):
        if site == 8:
            return self.gif8(x)
        if site == 9:
            return self.gif9(x)
        if site == 10:
            return self.gif10(x)
        return x
```

如果 Site 8/9/10 的 actual layout 不完全相同，则分别构造匹配各 site 的 state。

重点是：

> 测试 shape/layout 必须与当前真实 controller state contract 一致。

---

# 19. 新增真实 GIF MLP checkpoint regression

新增：

```python
def test_real_static_gif_mlp_checkpoint_forward_backward_match(...):
    ...
```

构造小 MLP：

```python
class _MLP(torch.nn.Module):
    gate_proj
    up_proj
    down_proj
    act_fn = SiLU()
```

输入：

```text
[B,L,C]
```

执行：

```text
checkpoint OFF
checkpoint ON
```

分别 backward。

至少比较：

```text
output
input.grad
gate_proj.weight.grad
up_proj.weight.grad
down_proj.weight.grad
```

不要比较 GIF buffer grad，因为：

```text
scale / zero / mask
```

是 buffers，不是 trainable parameters。

---

# 20. 新增真实 Phase MLP regression

除了 Attention Site 5，还建议 MLP 再覆盖真实 Phase：

```python
def test_real_phase_mlp_checkpoint_forward_backward_match(...)
```

controller：

```text
Site 8 → PhaseSurrogate
Site 9 → PhaseSurrogate
Site 10 → PhaseSurrogate
```

这样验证：

```text
MLP checkpoint
+
多个 PhaseSurrogate
+
多个 HeavisideSigmoid custom autograd
```

可以在一次 backward 中正常 recompute。

这是比单独 Attention Phase 更强的 coverage。

---

# 21. 问题三：当前 R4 test 实际是 identity

当前测试虽然：

```python
@pytest.mark.parametrize("r4", [None, object()])
```

但又：

```python
monkeypatch.setattr(
    model_integration,
    "random_hadamard",
    lambda value, _spec: value,
)
```

因此：

```text
r4 != None
```

并没有真的执行 Hadamard。

本轮要增加一个真实 R4 regression。

---

# 22. 真实 R4 fixture

从：

```python
from snn2.hadamard import make_spec
```

构造真实：

```python
r4 = make_spec(
    "R4_test",
    dimension=12,
    seed=123,
)
```

12 是当前实现支持的 Paley Hadamard dimension。

对应 MLP：

```python
class _MLP12(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            6, 12, bias=False
        )
        self.up_proj = torch.nn.Linear(
            6, 12, bias=False
        )
        self.down_proj = torch.nn.Linear(
            12, 6, bias=False
        )
        self.act_fn = torch.nn.SiLU()
```

这样：

```text
gate * up
```

最后一维正好 12，能真正执行：

```python
random_hadamard(product.float(), r4)
```

---

# 23. Real R4 test 不允许 monkeypatch Hadamard

新增：

```python
def test_real_r4_mlp_checkpoint_forward_backward_match():
```

此 test 内：

> 不要 monkeypatch `random_hadamard`

执行真实：

```text
sign multiplication
structured Hadamard
Paley H12
```

比较：

```text
checkpoint OFF
checkpoint ON
```

至少比较：

```text
output
input.grad
gate_proj.weight.grad
up_proj.weight.grad
down_proj.weight.grad
```

---

# 24. Real R4 test 最好与真实 Phase 或 GIF 组合

最强的 test 是：

```text
real R4
+
real Phase
+
MLP checkpoint
```

或者：

```text
real R4
+
real StaticGIF
+
MLP checkpoint
```

推荐至少实现一个：

```python
test_real_phase_r4_mlp_checkpoint_forward_backward_match
```

这样能够一次覆盖：

```text
SiLU
PhaseSurrogate
HeavisideSigmoid
multiply
real R4
Site 10 Phase
down_proj
checkpoint recompute
```

如果该 test 已经覆盖真实 R4，则原 lightweight：

```text
r4=object() + identity monkeypatch
```

可以保留，也可以删除冗余 parameterized `object()` case。

推荐保留基础 test 的 `r4=None`，把真实 R4 单独作为新测试，结构更清晰。

---

# 25. 关于 CPU Hadamard

当前：

```text
snn2.hadamard._fast_fht()
```

在 CPU 上会走 pure PyTorch FHT。

因此真实 R4 test：

```text
不需要 CUDA
不需要 fast-hadamard-transform CUDA kernel
```

可以进入普通 pytest。

这非常适合 regression。

---

# 26. 必须保证 OFF / ON 初始化完全一致

Checkpoint OFF/ON 两次运行必须重新：

```python
torch.manual_seed(...)
```

然后分别创建：

```text
MLP
input
controller/neuron state
```

不能：

```text
先跑 OFF backward
再复用同一个 module 跑 ON
```

否则参数 grad 与 RNG state 已被污染。

推荐 pattern：

```python
def run(checkpoint_enabled):
    torch.manual_seed(1234)
    mlp = ...
    x = torch.randn(..., requires_grad=True)
    controller = ...
    ...
    loss.backward()
    return ...
```

---

# 27. 不要比较 buffer object identity

测试只比较：

```text
forward tensor
input gradient
model trainable parameter gradient
```

不要要求：

```text
Phase/GIF module object identity
buffer storage identity
controller cache identity
```

因为 checkpoint backward recompute 可能重新进入 forward，但数学结果才是关键。

---

# 28. Lazy-loading controller 的测试范围

当前正式 `SiteController.apply()` 会 lazy load：

```text
phase_state.pt
gif_state.pt
```

本轮不强制在 selective checkpoint test 中完整构造磁盘 calibration tree。

原因：

- `controller.py` 已有独立 state loading tests；
- 本轮核心关注 checkpoint/autograd compatibility；
- 不应让一个 regression test 同时绑定 calibration filesystem。

因此：

```text
真实 neuron module
+
轻量 controller wrapper
```

是合理的。

不要为了这个测试重构 SiteController lazy-loading。

---

# 29. 增加一个 `torch.is_grad_enabled()` guard 测试

当前 guard 已检查：

```python
if not torch.is_grad_enabled():
    return False
```

现有测试没有明确覆盖这一点。

建议补：

```python
def test_checkpoint_guard_rejects_no_grad():
    controller = _ReplacementController(
        "phase",
        attention=True,
        mlp=True,
    )
    module = torch.nn.Linear(2, 2).train()

    with torch.no_grad():
        assert not _selective_checkpoint_allowed(
            module,
            controller,
            kind="attention",
        )
        assert not _selective_checkpoint_allowed(
            module,
            controller,
            kind="mlp",
        )
```

这是小补充，不是主要问题，但成本很低。

---

# 30. 保留 dropout RNG test

当前：

```text
test_attention_checkpoint_preserves_dropout_rng
```

继续保留。

不要改：

```python
checkpoint(... preserve_rng_state=False)
```

当前测试是必要的。

---

# 31. Prefix shape test 继续保留

当前：

```text
Q length = L
K/V length = L + P
```

的测试继续保留。

它能覆盖：

```text
fixed Prefix KV 已经进入 attention 后
selective checkpoint 仍支持 rectangular attention
```

不要删除。

---

# 32. Config test 还应显式验证 Qwen 保持 OFF

虽然当前 experiment matrix 已正确配置，但新 test 最好明确检查：

```text
exp1_qwen3_1_7b_tldr
exp1_qwen3_8b_tldr
```

所有 mode：

```yaml
ann_training_memory:
  attention_core_checkpoint: false
  mlp_checkpoint: false
```

推荐 fixture materialize 12 个 configs 后增加：

```python
for path in paths:
    cfg = ...
    task = cfg["experiment"]["task"]
    model = cfg["experiment"]["model_name"]
    mode = cfg["experiment"]["ann_mode"]

    if model.startswith("Qwen/"):
        assert attention_core_checkpoint is False
        assert mlp_checkpoint is False
```

避免未来 YAML anchor 改动误把 checkpoint 打开到 Qwen。

---

# 33. Config test 应显式验证 Llama aware only

同一个 test 中：

```text
Llama vanilla  → false false
Llama unaware  → false false
Llama phase    → true true
Llama gif      → true true
```

不要只检查两个 aware config。

---

# 34. 不修改 generated configs 的 git 管理策略

不要为了让 test 通过而：

```text
从 .gitignore 删除 configs/generated/
```

也不要提交 generated YAML。

当前：

```text
configs/generated/
```

被忽略是项目既定行为。

正确做法是：

> 测试自己 materialize 临时 configs。

---

# 35. 不修改 `scripts/materialize_configs.py`

当前 materializer 已能：

```text
读取 experiment_matrix
resolve
validate
生成 12 个 main configs
```

本轮没有理由改它。

---

# 36. 推荐的新测试结构

最终：

```text
tests/test_selective_activation_checkpointing.py
```

建议包含这些 test。

## Config

```text
test_memory_config_defaults_and_llama3_variants
test_memory_config_rejects_non_aware_modes
test_memory_config_allows_aware_modes
test_memory_config_rejects_transformers_gradient_checkpointing
test_memory_config_rejects_unknown_or_non_boolean_keys
test_qwen_selective_checkpoint_is_disabled
```

## Generic attention

```text
test_attention_checkpoint_forward_and_backward_match
test_attention_checkpoint_supports_prefix_key_value_length
test_attention_checkpoint_preserves_dropout_rng
```

## Real neuron

```text
test_real_phase_attention_checkpoint_forward_backward_match
test_real_phase_mlp_checkpoint_forward_backward_match
test_real_static_gif_mlp_checkpoint_forward_backward_match
```

## Real R4

```text
test_real_r4_mlp_checkpoint_forward_backward_match
```

或者更强：

```text
test_real_phase_r4_mlp_checkpoint_forward_backward_match
```

## Guard

```text
test_checkpoint_guard_rejects_non_aware_modes
test_checkpoint_guard_rejects_eval_and_regression_recording
test_checkpoint_guard_rejects_no_grad
test_checkpoint_guard_rejects_unknown_kind
```

---

# 37. 不要求真实 Llama3 模型进入 pytest

不要在普通 pytest 中：

```text
下载 Meta-Llama-3-8B
加载 8B model
初始化 DeepSpeed
申请 CUDA
```

这些属于 integration test，不属于 unit regression。

普通 pytest 必须：

```text
CPU 可运行
离线可运行
不依赖 HuggingFace 下载
不依赖本地 generated configs
```

---

# 38. 测试 tolerance 原则

CPU FP32：

```python
rtol=1e-5
atol=1e-6
```

通常足够。

如果真实 Phase custom autograd 在 checkpoint OFF/ON 下误差明显超过这一数量级：

> 先排查实现，不要直接放宽 tolerance。

真实 Hadamard CPU 应是 deterministic。

---

# 39. 真实 Phase forward 选择输入时要避免全部远离 threshold

测试输入不要全部是非常大的正数/负数，否则 surrogate gradient 可能过于简单。

建议使用较小值，例如：

```python
torch.randn(...) * 0.5
```

并选择合适 tau，使部分值位于 firing threshold 附近。

这样：

```text
HeavisideSigmoid backward
```

能产生非平凡 gradient，更能验证 backward equivalence。

可以增加：

```python
assert torch.count_nonzero(x.grad) > 0
```

避免测试“两个路径 gradient 都是 0”而假通过。

---

# 40. Real GIF 输入也应避免全部 clamp 饱和

StaticGIF test 输入建议限制：

```text
[-0.5, 0.5]
```

或者按 scale/zero 设计，使部分值：

```text
不在 qmin/qmax 饱和区
```

这样 STE backward 是非平凡的。

同样可验证：

```python
assert torch.isfinite(x.grad).all()
assert torch.count_nonzero(x.grad) > 0
```

---

# 41. Real R4 test 要检查 gradient 非零

除了 OFF/ON close：

```python
assert torch.count_nonzero(
    mlp.gate_proj.weight.grad
) > 0
```

至少对一个 trainable parameter 做非零检查。

避免 test 因 zero loss/zero gradient 假通过。

---

# 42. 不修改正式 config

本轮：

```text
configs/experiment_matrix.yaml
```

原则上不需要再改。

除非修测试时发现当前实际 main 的 config 与上一轮结论不一致。

正常情况下继续：

```text
Llama phase-aware → attention=true, mlp=true
Llama gif-aware   → attention=true, mlp=true
其他              → false
```

---

# 43. 不修改 `snn2/config.py`

当前：

```text
default false
unknown key rejection
bool validation
aware-only validation
禁止 HF GC 同时开启
```

已经正确。

本轮正常情况下不要再改。

---

# 44. 不修改 `snn2/controller.py`

当前新增：

```python
checkpoint_attention_core
checkpoint_mlp
```

只是 flags，设计正确。

不要改 neuron state load 或 apply。

---

# 45. 不修改 `snn2/training.py`

当前能：

```text
从 cfg 读取 memory flags
传入 SiteController
在 training_result.json 记录 metadata
```

已经正确。

本轮不用改。

---

# 46. 不修改 `snn2/model_integration.py`

除非新增真实 neuron test 暴露：

```text
checkpoint OFF/ON numerical mismatch
runtime error
custom autograd incompatibility
```

否则：

> 不要为了“代码更漂亮”再次重构 checkpoint implementation。

本轮目标是验证而不是重写。

---

# 47. 完成后运行 targeted tests

先：

```bash
pytest -q tests/test_selective_activation_checkpointing.py
```

确保新增 tests 通过。

然后：

```bash
pytest -q tests/test_neurons.py
pytest -q tests/test_generated_configs.py
```

最后：

```bash
pytest -q
```

必须全部通过。

---

# 48. Fresh-clone 可复现性验收

必须额外验证一次：

```bash
git clean -xfd
```

这一命令非常危险，会删除 ignored/untracked 文件。

**不要直接在有重要 artifact 的主工作目录运行。**

推荐：

```bash
git clone <repo> /tmp/SNN-test
cd /tmp/SNN-test
```

安装/复用环境后直接：

```bash
pytest -q tests/test_selective_activation_checkpointing.py
```

在没有：

```text
configs/generated/
```

预先存在的情况下，也必须通过。

这是问题一真正的验收方式。

---

# 49. 不需要因为测试修补重新跑全部 calibration

本轮只是 test 修补。

不需要重新：

```text
prepare_data
prepare_rotation
discover_prefix
calibrate_sites
```

除非代码实际主体被额外修改。

---

# 50. GPU integration test

所有 unit tests 通过后，再继续上一轮 GPU 验收：

```text
Llama3-8B Tulu-3
max_seq_length=2048
4×A100-80G
phase_aware
attention checkpoint ON
MLP checkpoint ON
```

先运行少量步骤，确认：

```text
无 OOM
无 Prefix error
无 checkpoint backward error
loss 正常
```

然后再运行完整 10k samples。

之后同样验证：

```text
gif_aware
```

---

# 51. 如果新增真实 Phase test 失败

不要马上删除 test。

先判断属于哪一种：

## A. forward mismatch

说明 checkpoint core 存在：

```text
non-deterministic forward
side effect
state mutation
```

需要修主体实现。

## B. forward 一致，backward mismatch

重点检查：

```text
HeavisideSigmoid custom autograd
checkpoint recompute
Phase membrane iterative graph
```

## C. 只在 R4 时 mismatch

重点检查：

```text
random_hadamard
dtype cast FP32 → original dtype
checkpoint recompute
```

## D. 只在 GIF 时 mismatch

重点检查：

```text
round STE
clamp
float32 fake quant intermediate
mask / scale broadcast
```

不要用放宽 tolerance 代替定位。

---

# 52. 如果全部新测试通过

则可以认为：

```text
Attention checkpoint generic behavior
Prefix-shaped attention
dropout RNG
Phase custom autograd
StaticGIF fake quant
MLP checkpoint
R4 Hadamard
checkpoint gating
```

均已有 unit-level regression 保护。

此时当前 selective checkpoint 实现就具备进入正式 2048-token Llama3 Tulu-3 训练的代码质量基础。

---

# 53. 本轮最终修改文件

正常情况下，本轮只需要修改：

```text
tests/test_selective_activation_checkpointing.py
```

如果为了避免 helper 复制新增公共 helper，也最多：

```text
tests/helpers.py
```

但优先不要引入新 helper 文件。

正常情况下不要修改：

```text
configs/experiment_matrix.yaml
snn2/config.py
snn2/controller.py
snn2/training.py
snn2/model_integration.py
snn2/neurons.py
snn2/prefix_cache.py
snn2/hadamard.py
```

---

# 54. 最终验收 Checklist

## 测试独立性

- [ ] 不再读取 repository `configs/generated/`
- [ ] test 内使用 `materialize_configs(..., tmp_path/...)`
- [ ] fresh clone 下 test 可直接运行

## Config regression

- [ ] Llama vanilla checkpoint=false
- [ ] Llama unaware checkpoint=false
- [ ] Llama phase checkpoint=true
- [ ] Llama gif checkpoint=true
- [ ] Qwen 所有 mode checkpoint=false
- [ ] legacy config default false
- [ ] HF GC + selective checkpoint 仍被拒绝

## Attention

- [ ] generic Phase/GIF mode OFF/ON forward match
- [ ] backward match
- [ ] Prefix-shaped K/V match
- [ ] dropout RNG match
- [ ] real PhaseSurrogate attention OFF/ON match

## MLP

- [ ] generic MLP OFF/ON match
- [ ] real PhaseSurrogate MLP match
- [ ] real StaticGIF MLP match
- [ ] real R4 Hadamard MLP match

## Gradient quality

- [ ] input grad finite
- [ ] relevant parameter grad finite
- [ ] 至少一个关键 gradient 非零
- [ ] OFF/ON grad close

## Guard

- [ ] identity false
- [ ] none false
- [ ] collect false
- [ ] deploy_* false
- [ ] eval false
- [ ] no_grad false
- [ ] regression recorder false

## Full suite

- [ ] `pytest -q tests/test_selective_activation_checkpointing.py`
- [ ] `pytest -q`
- [ ] 全部通过

---

# 55. 本轮最终目标

最终状态应为：

```text
Selective checkpoint 主体代码：
    不改

正式实验配置：
    不改

测试：
    不依赖 ignored generated files
    有真实 Phase custom autograd regression
    有真实 StaticGIF fake quant regression
    有真实 R4 Hadamard regression
```

核心原则：

> **上一轮已经完成显存优化实现；本轮只把测试从“简化算子证明 checkpoint 基本可用”提升到“真实 Phase/GIF/R4 路径也有数值和梯度回归保护”。**

只有新增真实 regression test 暴露实际 mismatch 时，才进一步修改 `snn2/model_integration.py`。

否则不要再次重构训练实现。
