# SNN Site 5 GIF 严格对齐 SpikeLLM Identity 行为修改方案

> 基线：`wangwk699/SNN` 当前 `main`，提交 `e5905980fe11422e6610ac9c24ac55dae1df83cf`
>
> 目标：将当前 Site 5 GIF 从显式 Q16 `SoftmaxFixedGIF` / `quantized_cumulative_difference` 实现，修改为**严格遵循 Xingrun-Xing2/SpikeLLM 当前代码真实行为的 identity**。
>
> 本次修改必须同步覆盖代码、schema / metadata、config、测试和当前有效 Markdown 文档。旧 Q16 artifact 必须严格拒绝，不能保留兼容 fallback。
>
> 本方案只修改 **GIF 在 Site 5 的行为**。Site 5 的 Phase / MTN 行为、Site 1/2/3/4/6/7/8/9/10 的 ordinary GIF、per-head/grouped calibration、Clip 三态协议、Prefix、Rotation 等规则均保持不变。

---

# 1. SpikeLLM 当前真实行为：作为本次修改的唯一参考

严格按 `Xingrun-Xing2/SpikeLLM` 当前代码：

## 1.1 Softmax probability 的配置

`main.py`：

```python
args.p_quant_params = {
    "n_bits": 16,
    "metric": "fix0to1",
}
```

## 1.2 `n_bits=16` 实际不会执行 `fix0to1`

`spike_driven_quant/quantizer.py::UniformAffineQuantizer.forward()`：

```python
def forward(self, x):
    if self.n_bits >= 16 or not self.enable:
        return x

    if self.metric == "fix0to1":
        return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
```

由于 `n_bits = 16`，先满足 `self.n_bits >= 16`，因此直接 `return x`，后面的 `round(65535*x)/65535` 不会执行。

所以 SpikeLLM 的 Softmax probability 实际为：

\[
P_q = P.
\]

## 1.3 P 不进入 multibit temporal decomposition

`models/spike_llama_layer.py`：

```python
attn_output = self.pv_matmul(
    (attn_weights, value_states)
)
```

这里 `x1 = attn_weights = P`，`x2 = value_states = V`。

`spike_driven_quant/spike_matmul.py` 中：

```python
x1 = self.quant_x1(x1)
x2 = self.quant_x2(x2)
```

`multibit_simulate` 的 membrane / spike decomposition 只存在于 `quant_x2()`；而 `x1=P` 只经过 `self.x1_quantizer(x1)`，其 `n_bits=16` 又直接 identity。

因此本项目要对齐的是：

\[
\boxed{\text{Site 5 GIF}(P)=P}
\]

而不是：

\[
\frac{\operatorname{round}(65535P)}{65535}.
\]

也不是把 `[0,65535]` 整数拆成两个 `[0,15]` timestep。

---

# 2. 本项目最终目标语义

## 2.1 `gif_aware` ANN training / final ANN evaluation

Site 5：

```python
GIF_site5(P) = P
```

即 exact identity：

```text
不 round
不 clamp
不乘 65535
不除 65535
不 STE fake quant
不使用 scale / zero point
不执行 ordinary GIF qmax30
```

## 2.2 `deploy_gif` temporal SNN

Site 5 对传入的 temporal Softmax increments 必须：

```python
return incoming
```

即：

\[
Y_t = X_t.
\]

禁止当前：

\[
Q_{16}\left(\sum_{i=0}^{t}X_i\right)
-
Q_{16}\left(\sum_{i=0}^{t-1}X_i\right)
\]

这种 Q16 cumulative-difference 处理。

## 2.3 一个必须明确的边界

本项目仍然是自己的 **full-temporal SNN execution framework**。

因此 `temporal_softmax()` 仍然负责把 cumulative attention score 映射成 temporal Softmax increments。

本次修改只要求：

> **Site 5 GIF replacement 本身是 identity。**

即：

```text
temporal_softmax
    ↓
Site 5 GIF identity
    ↓
temporal PV
```

不要为了 Site 5 identity 去删除 `temporal_softmax()`，也不要把整个项目的 full-temporal attention 重写成 SpikeLLM 的内部 simulation 方式。

严格对齐的是 **Site 5 GIF operator 的真实 SpikeLLM 行为**，不是把整个 SNN temporal engine 改写成 SpikeLLM。

---

# 3. 保持不变的内容

以下内容禁止因本次修改而改变：

```text
Site 1/2/3/4/6/7/8/9/10 ordinary GIF:
    base_bits = 4
    add_bits = 1
    high_qmax = 30
    temporal_steps = 2
    per_step_qmax = 15
    two [0,15] integer chunks

Site 5 Phase:
    per-head tau [H,1]

Site 5 MTN:
    per-head base_scale [H,1]

Site 5 Clip:
    永久 disabled / no clip_state.pt

Site 5 statistics:
    attention_softmax / per-head statistics
```

Site 5 statistics 仍然需要，因为 Phase Site 5 calibration 与 MTN Site 5 calibration 仍依赖它，因此不能删除 Site 5 statistics collection。

---

# 4. 修改 `snn2/temporal_ops.py`

当前：

```python
TEMPORAL_IMPLEMENTATION_VERSION = 4
TEMPORAL_IMPLEMENTATION = "sparse_llm_temporal_v4"

SITE_STATE_FORMAT_VERSION = 6
STATISTICS_FORMAT_VERSION = 2
CALIBRATION_MANIFEST_FORMAT_VERSION = 7
CONVERSION_METADATA_FORMAT_VERSION = 8

SOFTMAX_SITE5_GIF_POLICY = (
    "fixed_range_u16_quantized_cumulative_difference"
)
```

必须升级为：

```python
TEMPORAL_IMPLEMENTATION_VERSION = 5
TEMPORAL_IMPLEMENTATION = "sparse_llm_temporal_v5"

SITE_STATE_FORMAT_VERSION = 7

# statistics layout / accumulation 本次没有变化
STATISTICS_FORMAT_VERSION = 2

CALIBRATION_MANIFEST_FORMAT_VERSION = 8
CONVERSION_METADATA_FORMAT_VERSION = 9

SOFTMAX_SITE5_GIF_POLICY = (
    "spikellm_nbits16_sentinel_identity"
)
```

保持：

```python
SOFTMAX_SITE5_GROUPING_POLICY = (
    "per_head_full_variable_key_axis"
)

SOFTMAX_SITE5_CLIP_POLICY = "disabled"
```

---

# 5. ordinary GIF metadata 必须明确作用域

当前 `temporal_policy_metadata()` 同时输出：

```text
gif_high_qmax = 30
gif_local_decomposition_steps = 2
gif_per_step_qmax = 15
```

以及：

```text
softmax_site5_gif_policy = ...
```

这很容易让人误认为 Site 5 也使用 qmax=30 / `[0,15]×2`。

本次建议改为：

```python
"ordinary_gif_high_qmax": GIF_HIGH_QMAX,
"ordinary_gif_local_decomposition_steps": GIF_LOCAL_STEPS,
"ordinary_gif_per_step_qmax": GIF_STEP_QMAX,
"ordinary_gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,

"softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
```

不要继续保留无作用域的：

```text
gif_high_qmax
gif_local_decomposition_steps
gif_per_step_qmax
```

新 schema 会严格拒绝旧 metadata，不需要兼容旧 key。

---

# 6. 修改 `snn2/calibration.py`

当前 Site 5 GIF state 是：

```python
{
    "parameter_layout": "softmax_fixed_range",
    "group_size_source": "site5_fixed_override",
    "gif_policy": "softmax_fixed_range_u16",
    "range_min": 0.0,
    "range_max": 1.0,
    "quantization_bits": 16,
    "qmin": 0,
    "qmax": 65535,
    "scale": 1.0 / 65535.0,
    "zero_point": 0,
    "temporal_steps": GIF_LOCAL_STEPS,
    "temporal_policy": "quantized_cumulative_difference",
}
```

必须完全删除这一语义。

## 6.1 新 Site 5 GIF state

推荐：

```python
if is_site5:
    gif_state = {
        "state_kind": "gif",
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version":
            TEMPORAL_IMPLEMENTATION_VERSION,

        "parameter_layout": "softmax_identity",

        # G 仍只保存 provenance；Site 5 GIF identity 不使用 grouping 参数
        "configured_group_size": configured_group,
        "group_size": -1,
        "group_size_source": "site5_identity_override",

        "num_heads": int(layout["num_heads"]),
        "channels_per_head": None,
        "groups_per_head": 1,

        "gif_policy": SOFTMAX_SITE5_GIF_POLICY,

        # 明确记录为什么是 identity
        "reference_n_bits": 16,
        "reference_metric": "fix0to1",
        "quantization_applied": False,

        # 全局 GIF temporal deployment 仍为 T=2，但 Site 5 本身只是逐 timestep identity
        "temporal_steps": GIF_LOCAL_STEPS,
        "temporal_policy": "identity",
    }
```

## 6.2 Site 5 state 禁止保存 Q16 参数

新 state 中不得存在：

```text
range_min
range_max
quantization_bits
qmin
qmax
scale
zero_point
```

尤其不要保存：

```text
qmax = 65535
scale = 1 / 65535
```

否则仍会制造“Site 5 做 16-bit quantization”的错误语义。

## 6.3 Site 5 GIF 不使用 calibration statistics 生成参数

Site 5 statistics 仍正常收集，但构造 `gif_state` 时不得使用 `value_min`、`value_max`、`phase_ema_abs_max`、saliency 去产生任何 GIF quantization parameter。

Site 5 statistics 只继续服务 Phase 与 MTN。

---

# 7. 修改 `snn2/neurons.py`

## 7.1 删除 `SoftmaxFixedGIF`

完整删除当前：

```python
class SoftmaxFixedGIF(nn.Module):
```

包括：

```python
_hard_q16()
```

以及所有：

```text
torch.round
clamp(0, 1)
* 65535
/ 65535
cumsum
quantized - previous
```

不要保留兼容 alias。

## 7.2 新增 `SoftmaxIdentityGIF`

推荐：

```python
class SoftmaxIdentityGIF(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()

        _validate_state_header(state, "gif")

        expected = {
            "parameter_layout": "softmax_identity",
            "gif_policy": SOFTMAX_SITE5_GIF_POLICY,
            "group_size": -1,
            "group_size_source": "site5_identity_override",
            "reference_n_bits": 16,
            "reference_metric": "fix0to1",
            "quantization_applied": False,
            "temporal_steps": GIF_LOCAL_STEPS,
            "temporal_policy": "identity",
        }

        mismatched = {
            key: (expected_value, state.get(key))
            for key, expected_value in expected.items()
            if state.get(key) != expected_value
        }

        if mismatched:
            raise ValueError(
                "Invalid SpikeLLM-aligned Site 5 GIF identity state: "
                f"{mismatched}"
            )

        self.num_heads = int(state["num_heads"])
        self._temporal_steps = int(state["temporal_steps"])

        if self.num_heads <= 0:
            raise ValueError(
                "Site 5 GIF identity requires num_heads > 0"
            )
```

## 7.3 `forward()` 必须真正 exact identity

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    self._validate_input(x)
    return x
```

必须禁止：

```text
float()
clone()
round()
clamp()
STE
multiply
divide
```

直接 `return x`。

## 7.4 `temporal()` 必须真正 exact identity

```python
@property
def temporal_steps(self) -> int:
    return self._temporal_steps


def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
    if (
        incoming.ndim != 5
        or incoming.shape[0] != self.temporal_steps
    ):
        raise ValueError(...)

    self._validate_input(incoming[0])
    return incoming
```

禁止 `cumsum`、Q16、difference。

## 7.5 Factory 修改

当前：

```python
def gif_module_from_state(state):
    if state.get("gif_policy") == "softmax_fixed_range_u16":
        return SoftmaxFixedGIF(state)
    return StaticGIF(state)
```

改为：

```python
def gif_module_from_state(state):
    if state.get("gif_policy") == SOFTMAX_SITE5_GIF_POLICY:
        return SoftmaxIdentityGIF(state)
    return StaticGIF(state)
```

旧 `softmax_fixed_range_u16` 必须因 schema/policy 不匹配而拒绝。

---

# 8. `StaticGIF` ordinary GIF 完全不改

`StaticGIF` 的：

```text
low qmax = 15
high qmax = 30
T = 2
per-step qmax = 15
```

全部保持现状。

`integer_chunks()` 仍只服务 ordinary GIF site。

不要为了 Site 5 identity 删除 `GIF_HIGH_QMAX`、`GIF_STEP_QMAX`、`GIF_INTEGER_DECOMPOSITION`，这些仍然是 Site 1/2/3/4/6/7/8/9/10 的真实算法。

---

# 9. 修改 `snn2/state_validation.py`

当前 Site 5 检查中的旧：

```text
softmax_fixed_range_u16
```

改成统一比较：

```python
SOFTMAX_SITE5_GIF_POLICY
```

最好完全消除硬编码字符串。

## 9.1 新 identity state 必须严格校验

validator / `SoftmaxIdentityGIF` 必须拒绝：

```text
softmax_fixed_range_u16
fixed_range_u16_quantized_cumulative_difference
softmax_fixed_range
site5_fixed_override
quantization_applied = True
```

以及旧：

```text
state format v6
temporal implementation v4
```

## 9.2 `steps_by_neuron["gif"]` 保持一致

`SoftmaxIdentityGIF.temporal_steps` 仍返回 `2`，因为整个 `deploy_gif` 仍然使用 `GIF_LOCAL_STEPS = 2`。

这里 T=2 的含义只是“接收并原样返回两个 temporal frames”，不是把任何 16-bit integer 拆成两个 timestep。

---

# 10. `snn2/controller.py` 原则上不需要重构

当前 controller 通过：

```python
modules[name] = gif_module_from_state(state)
```

即可实现：

```text
ordinary Site -> StaticGIF
Site 5       -> SoftmaxIdentityGIF
```

并继续保持 Site 5 no-Clip。

不要在 controller 里重新写一套 Q16/identity special case。

---

# 11. `snn2/model_integration.py`：static 路径不绕过 Site 5

当前：

```python
weights = F.softmax(...)
weights = controller.apply(layer_index, 5, weights)
```

保持这一调用。

不要写：

```python
if gif:
    skip controller.apply(...)
```

Site 5 仍是统一的 replacement site；identity 行为应由 state/module policy 表达，而不是散落在 attention forward 中写 special case。

这样 Phase / MTN Site 5 仍正常工作。

---

# 12. `snn2/temporal_model.py`：temporal 路径同样保持 Site 5 调用

当前：

```python
weight_increment = temporal_softmax(...)
flat_weights = from_temporal(weight_increment)
flat_weights = controller.apply(layer_index, 5, flat_weights)
weight_increment = to_temporal(flat_weights, steps)
```

保持结构不变。

当 `deploy_gif` 时：

```python
SoftmaxIdentityGIF.temporal()
```

返回原输入，所以数学上：

\[
\Delta P_t \rightarrow \Delta P_t.
\]

当 `deploy_phase` / `deploy_mtn` 时，Site 5 仍执行对应 Phase / MTN neuron。

---

# 13. 修改 `snn2/evaluation.py`

当前 final gif-aware ANN metadata：

```python
implementation = "StaticGIF/SoftmaxFixedGIF.forward"
```

改为：

```python
implementation = "StaticGIF/SoftmaxIdentityGIF.forward"
```

同时 `softmax_site5_gif_policy` 记录新 identity policy。

---

# 14. GIF activation-neuron operator 计数必须修改

当前 GIF 也按每层 10 个 activation-neuron operator 计数。

Site 5 GIF 变为 identity 后：

```text
GIF 每层只有 9 个实际 GIF neuron replacement
```

因此应改为：

```python
if neuron == "ann":
    return 0

if neuron == "phase":
    return num_hidden_layers * SITE_COUNT + 1

if neuron == "gif":
    return num_hidden_layers * len(GIF_ACTIVE_SITE_IDS)

if neuron == "mtn":
    return num_hidden_layers * SITE_COUNT
```

不要直接长期硬编码 `SITE_COUNT - 1`，建议在 `sites.py` 定义活跃 site 集合。

---

# 15. `snn2/sites.py`

建议新增：

```python
GIF_IDENTITY_SITE_IDS = frozenset({SOFTMAX_SITE_ID})

GIF_ACTIVE_SITE_IDS = frozenset(
    site
    for site in SITE_IDS
    if site not in GIF_IDENTITY_SITE_IDS
)
```

并测试：

```python
GIF_ACTIVE_SITE_IDS == {
    1, 2, 3, 4, 6, 7, 8, 9, 10
}
```

---

# 16. conversion / evaluation metadata

以下文件都会记录 `SOFTMAX_SITE5_GIF_POLICY`，因此常量修改后需要确认生成的新 metadata 为：

```text
spikellm_nbits16_sentinel_identity
```

至少检查：

```text
snn2/conversion.py
snn2/evaluation.py
scripts/verify_artifacts.py
```

同时 conversion schema：

```text
v8 -> v9
```

旧 v8 conversion metadata 必须拒绝。

---

# 17. `snn2/config.py` 与 `configs/experiment_matrix.yaml`

## 17.1 temporal implementation

`configs/experiment_matrix.yaml` 中全部：

```yaml
temporal_implementation: sparse_llm_temporal_v4
```

改为：

```yaml
temporal_implementation: sparse_llm_temporal_v5
```

所有 model-task 实验统一修改。

## 17.2 推荐显式写入 Site 5 GIF policy

在 `deployment:` 增加：

```yaml
softmax_site5_gif_policy: spikellm_nbits16_sentinel_identity
```

`config.py` 导入 `SOFTMAX_SITE5_GIF_POLICY`，并在 `expected_deployment` 加入：

```python
"softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
```

这样 generated config 会显式锁定 Site 5 行为。

## 17.3 `gif:` section 仍表示 ordinary GIF

当前：

```yaml
gif:
  base_bits: 4
  add_bits: 1
  high_qmax: 30
  temporal_steps: 2
  per_step_qmax: 15
```

不必大规模重构配置结构，但所有代码错误信息、metadata 和文档都必须明确：

```text
这些是 ordinary GIF site 的配置，
不适用于 Site 5。
```

---

# 18. schema 升级与旧 artifact 严格拒绝

本次属于真正的 runtime semantic change，统一升级：

```text
TEMPORAL_IMPLEMENTATION_VERSION: 4 -> 5
TEMPORAL_IMPLEMENTATION:
    sparse_llm_temporal_v4
    ->
    sparse_llm_temporal_v5

SITE_STATE_FORMAT_VERSION:
    6 -> 7

STATISTICS_FORMAT_VERSION:
    2 -> 2
    # 不变

CALIBRATION_MANIFEST_FORMAT_VERSION:
    7 -> 8

CONVERSION_METADATA_FORMAT_VERSION:
    8 -> 9
```

## 18.1 为什么 statistics v2 不变

本次没有改变 `last_dim`、`attention_head`、`attention_softmax`、Phase EMA、saliency accumulation。

Site 5 statistics 仍然用于 Phase / MTN，因此无需 bump statistics schema。

## 18.2 旧 Q16 state 必须拒绝

任何包含：

```text
gif_policy = softmax_fixed_range_u16
parameter_layout = softmax_fixed_range
temporal_policy = quantized_cumulative_difference
```

的 Site 5 GIF state 都必须报 legacy / incompatible，禁止自动转换。

---

# 19. 测试修改：`tests/test_neurons.py`

当前导入：

```python
SoftmaxFixedGIF
```

改成：

```python
SoftmaxIdentityGIF
```

## 19.1 新 `_softmax_gif_state()`

改成：

```python
def _softmax_gif_state():
    return {
        **_header("gif"),
        "parameter_layout": "softmax_identity",
        "configured_group_size": 32,
        "group_size": -1,
        "group_size_source": "site5_identity_override",
        "num_heads": 2,
        "channels_per_head": None,
        "groups_per_head": 1,
        "gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "reference_n_bits": 16,
        "reference_metric": "fix0to1",
        "quantization_applied": False,
        "temporal_steps": 2,
        "temporal_policy": "identity",
    }
```

## 19.2 删除 Q16 数学测试

删除：

```text
test_softmax_fixed_gif_is_exact_q16_and_factory_selects_it
test_softmax_fixed_gif_temporal_is_quantized_cumulative_difference
```

## 19.3 新 static identity test

```python
def test_softmax_gif_is_exact_identity_and_factory_selects_it():
    module = gif_module_from_state(_softmax_gif_state())
    assert isinstance(module, SoftmaxIdentityGIF)

    x = torch.tensor(
        [[[
            [-0.1, 0.1, 0.5, 1.1]
        ], [
            [0.2, 0.3, 0.4, 0.1]
        ]]]
    )

    output = module(x)

    assert output is x
    assert torch.equal(output, x)
```

故意包含 `-0.1`、`1.1`，用来确认 Site 5 module 本身没有 `clamp(0,1)`，虽然正常 Softmax 输入不会超出 `[0,1]`。

## 19.4 新 temporal identity test

```python
def test_softmax_gif_temporal_is_exact_identity():
    module = SoftmaxIdentityGIF(_softmax_gif_state())
    incoming = torch.randn(2, 1, 2, 3, 5)
    output = module.temporal(incoming)

    assert output is incoming
    assert torch.equal(output, incoming)
```

这比只检查 `output.sum(0)` 更严格。

## 19.5 旧 Q16 state rejection test

新增测试，旧：

```text
softmax_fixed_range_u16
softmax_fixed_range
quantized_cumulative_difference
```

必须被拒绝。

---

# 20. `tests/test_controller_state_loading.py`

当前类似：

```text
test_site5_common_clip_uses_q16_gif_without_loading_clipper
```

改名：

```text
test_site5_common_clip_uses_identity_gif_without_loading_clipper
```

断言：

```python
output = controller.apply(0, 5, x)
assert torch.equal(output, x)

assert set(
    controller._modules[site_key(0, 5)]
) == {"gif"}

assert isinstance(
    controller._modules[site_key(0, 5)]["gif"],
    SoftmaxIdentityGIF,
)
```

Site 5 仍不能出现 `clip`。

同时新增 `deploy_gif` Site 5 identity 测试，要求 controller 经过 `to_temporal -> identity -> from_temporal` 后数值 bitwise-equivalent。

---

# 21. `tests/test_calibration_profiles.py`

修改 Site 5 GIF state 断言：

```python
site5_gif["gif_policy"] == SOFTMAX_SITE5_GIF_POLICY
site5_gif["quantization_applied"] is False
site5_gif["temporal_policy"] == "identity"
```

并确认以下 key 不存在：

```python
for key in (
    "range_min",
    "range_max",
    "quantization_bits",
    "qmin",
    "qmax",
    "scale",
    "zero_point",
):
    assert key not in site5_gif
```

仍确认 Site 5 不存在 `clip_state.pt`。

---

# 22. `tests/test_evaluation_paths.py`

更新：

```text
StaticGIF/SoftmaxFixedGIF.forward
```

为：

```text
StaticGIF/SoftmaxIdentityGIF.forward
```

新增/更新 operator count。假设 `N = num_hidden_layers`：

```text
phase: 10N + 1
gif:   9N
mtn:   10N
```

---

# 23. `tests/test_generated_configs.py`

更新预期：

```text
sparse_llm_temporal_v5
```

如果按本方案在 deployment 显式增加 policy，再断言：

```python
cfg["deployment"]["softmax_site5_gif_policy"] == (
    "spikellm_nbits16_sentinel_identity"
)
```

---

# 24. `tests/test_temporal_ops.py`

更新 temporal implementation v5 以及 metadata key。

必须断言：

```python
metadata["softmax_site5_gif_policy"] == (
    "spikellm_nbits16_sentinel_identity"
)
```

并建议断言：

```text
ordinary_gif_high_qmax = 30
ordinary_gif_local_decomposition_steps = 2
ordinary_gif_per_step_qmax = 15
```

不存在旧模糊 key：

```text
gif_high_qmax
gif_local_decomposition_steps
gif_per_step_qmax
```

---

# 25. `tests/test_conversion_metadata.py`

更新：

```text
conversion metadata v8 -> v9
```

更新 `softmax_site5_gif_policy` 以及 ordinary GIF metadata key。

新增 old-Q16 conversion metadata rejection：

```python
metadata["softmax_site5_gif_policy"] = (
    "fixed_range_u16_quantized_cumulative_difference"
)

with pytest.raises(ValueError):
    ...
```

---

# 26. `tests/test_calibration_topology.py`

如测试显式构造 state v6 / manifest v7 / temporal v4，全部改成使用新常量或新版本，避免测试继续硬编码旧版本数字。

---

# 27. `tests/test_temporal_model_integration.py`

至少新增一个语义测试：

> `deploy_gif` 下，Site 5 operator 对 temporal Softmax increments 不改变任何数值。

如果完整 attention 构造太重，可以保持测试在 controller/module 层完成，不必为了这一项增加大量 tracing 代码。

---

# 28. `scripts/verify_artifacts.py`

检查并更新所有：

```text
temporal v4
site state v6
manifest v7
conversion v8
fixed_range_u16...
Q16
```

对应 validator 应使用新常量，并明确拒绝旧 Site 5 Q16 policy。

---

# 29. 当前 Markdown 文档必须同步修改

当前有效文档至少：

```text
README.md
AGENTS.md
实验执行总结.md
代码结构总结.md
```

必须全部从 Q16 语义改为 identity。

---

# 30. `README.md` 修改

当前：

```text
gif_aware:
ordinary sites 使用 StaticGIF，
Site 5 使用 Q16 SoftmaxFixedGIF
```

改为：

```text
gif_aware:
ordinary sites 使用 StaticGIF；
Site 5 严格跟随 SpikeLLM 当前代码的
16-bit sentinel 行为，实际为 identity，
不执行 Q16 fake quantization。
```

当前：

```text
Site 5 忽略 G：
Phase/MTN 为 per-head [H,1]，
GIF 显式执行 round(65535*x)/65535，
temporal 使用 quantized cumulative difference
```

改为：

```text
Site 5 忽略 G：
Phase/MTN 仍为 per-head [H,1]；
GIF 严格按 SpikeLLM 的 n_bits=16 sentinel
执行 identity，不做 GIF calibration、
Q16 fake quantization 或 Site 5 temporal encoding；
Site 5 永远 no-Clip。
```

并补一句：

```text
gif.high_qmax=30 / per_step_qmax=15
只适用于 ordinary GIF sites，不适用于 Site 5。
```

---

# 31. `AGENTS.md` 修改

当前第 3 条 Q16 规则改成：

```text
3. Site 5 忽略全局 G：
Phase tau 与 MTN base_scale 固定为 [H,1]；
GIF 必须严格跟随 SpikeLLM 当前 n_bits=16 sentinel
的真实代码行为，static 与 temporal 均为 identity。
Site 5 GIF 不得执行 Q16、round/clamp、
scale/zero-point calibration、qmax30/[0,15] chunk
或 cumulative-difference quantization。
Site 5 永远不得生成、加载或执行 Clip。
```

并建议新增一条：

```text
ordinary GIF qmax30/T=2/[0,15]x2
只允许用于 Site 1/2/3/4/6/7/8/9/10；
禁止将这些 metadata 解释为 Site 5 策略。
```

---

# 32. `实验执行总结.md` 修改

全文件查找并修改：

```text
fixed-Q16 GIF
SoftmaxFixedGIF
Site 5 Q16
round(65535*x)/65535
quantized cumulative difference
```

## 32.1 Step 5 calibration

补充：

```text
ANN-training / Post-finetuning calibration
仍记录 Site 5 statistics，因为 Phase/MTN 需要；
但 Site 5 GIF 不从 statistics 生成任何 quantization parameter，
只物化显式 identity policy state。
```

## 32.2 Step 6 aware ANN

当前：

```text
Site 5 不论开关均只执行 Phase 或 fixed-Q16 GIF
```

改为：

```text
Site 5：
phase_aware 执行 Phase；
gif_aware 严格执行 identity；
两者均不执行 common Clip。
```

## 32.3 Final ANN table

当前：

```text
gif_aware:
ordinary StaticGIF / Site 5 SoftmaxFixedGIF
```

改成：

```text
gif_aware:
ordinary sites = StaticGIF
Site 5 = SoftmaxIdentityGIF
(SpikeLLM n_bits=16 sentinel identity)
```

## 32.4 schema 说明

更新为：

```text
statistics v2
site state v7
calibration manifest v8
conversion metadata v9
temporal implementation v5
```

---

# 33. `代码结构总结.md`

该文档继续只保留 `## 2. 目录结构`，每个文件一句功能说明。

至少更新：

```text
snn2/calibration.py
— 将 statistics v2 物化为 grouped Phase/ordinary GIF/MTN、
  9-site Clip、Site 5 GIF identity policy 与 grouped final RMSNorm state。

snn2/neurons.py
— 实现 grouped Phase/ordinary GIF/MTN/Clip
  以及严格对齐 SpikeLLM 的 Site 5 SoftmaxIdentityGIF。

snn2/temporal_ops.py
— 实现 temporal v5 基础算子并定义
  ordinary GIF 与 Site 5 identity 的分离 policy/schema。
```

更新测试说明，删除 Q16 字样。

---

# 34. `docs/history/` 的处理规则

历史文档正文不要篡改成“当时就已经是 identity”，否则会破坏历史可追溯性。

但必须让读者明确旧 Q16 方案已被当前 identity 方案 supersede。

Codex 执行：

```bash
rg -n \
  "Q16|SoftmaxFixedGIF|softmax_fixed_range_u16|quantized_cumulative_difference|65535" \
  --glob '*.md'
```

## 当前有效文档

```text
README.md
AGENTS.md
实验执行总结.md
代码结构总结.md
```

直接更新正文为当前 identity 语义。

## `docs/history/*.md`

不要重写历史正文。

凡是包含旧 Q16 Site 5 设计的历史文档，在最顶部标题后添加统一提示：

```markdown
> **历史说明：**
> 本文记录的是当时的 Q16 Site 5 设计，
> 该设计现已被
> `SNN_Site5_GIF_Strict_SpikeLLM_Identity_Modification_Plan.md`
> 替代。当前实现以 SpikeLLM `n_bits=16`
> sentinel 的真实 identity 行为为准；
> 本文中的 Q16 / `SoftmaxFixedGIF` /
> cumulative-difference 内容仅用于历史追溯，
> 不再代表当前实现。
```

---

# 35. 新方案文档归档

本方案实施时将本文保存到：

```text
docs/history/SNN_Site5_GIF_Strict_SpikeLLM_Identity_Modification_Plan.md
```

并在 `代码结构总结.md` 的 `docs/history/` 中增加一句：

```text
— 保存 Site 5 GIF 从 Q16 特例切换为
  SpikeLLM 16-bit sentinel identity 的实施方案。
```

---

# 36. 不要修改哪些历史结论

不要把 ordinary GIF 的：

```text
qmax=30
T=2
per-step qmax=15
```

删除。

这些对 ordinary sites 仍然成立，只需要把所有表述明确成 `ordinary GIF policy`，而非全局 GIF policy。

---

# 37. Artifact 重跑边界

本次变更不能直接复用旧 Q16 state。

## 可继续复用

```text
data manifest
rotation
Pre-finetuning Prefix
Post-finetuning Prefix
```

这些与 Site 5 GIF policy 无关。

## 必须重新生成

所有 G 下的：

```text
ANN-training calibration
Post-finetuning conversion calibration
SNN conversion metadata
SNN evaluation outputs
```

因为 site state schema、manifest schema、temporal policy、Site 5 GIF state 均已改变。

---

# 38. aware ANN checkpoint 重跑规则

## `gif_aware`

必须重新训练。

原因：旧 `Site5(P)=Q16(P)`，新 `Site5(P)=P`，ANN training forward 已改变。

## `phase_aware`

Phase forward 本身没有改变。

但是当前项目 frozen provenance 会记录 ANN-training calibration manifest SHA-256；本次重新 materialize calibration 后 manifest hash 必然变化，所以旧 phase-aware `training_result.json` 会与新 calibration 不一致。

**不要手工改 hash。**

在保持当前严格 provenance 协议的前提下：

```text
phase_aware 也重新训练
```

## `vanilla / unaware`

ANN training 是 identity，不依赖 Site 5 GIF state，因此 final ANN checkpoint 可以继续复用。

只需要重新：

```text
Post-finetuning conversion calibration
convert_snn
SNN evaluation
```

---

# 39. 路径规则保持不变

本次不改变：

```text
calibration_group_size_<G>
aware SNN path 去重
calibration config/log G 隔离
Prefix/rotation/data shared
```

不要为 Site 5 identity 再增加新的 artifact 路径层级；schema + manifest provenance 已足以隔离旧 artifact。

---

# 40. 最低单元测试集合

至少运行：

```bash
pytest -q \
  tests/test_neurons.py \
  tests/test_controller_state_loading.py \
  tests/test_calibration_profiles.py \
  tests/test_calibration_topology.py \
  tests/test_evaluation_paths.py \
  tests/test_temporal_ops.py \
  tests/test_temporal_model_integration.py \
  tests/test_conversion_metadata.py \
  tests/test_generated_configs.py
```

然后：

```bash
pytest -q
```

---

# 41. 必须增加的关键 regression

至少保证以下断言存在。

## 41.1 Static Site 5 GIF exact identity

```python
y = module(x)
assert y is x
```

## 41.2 Temporal Site 5 GIF exact identity

```python
y = module.temporal(x)
assert y is x
```

## 41.3 Site 5 不存在 Q16 参数

```python
assert "qmax" not in state
assert "scale" not in state
assert "zero_point" not in state
```

## 41.4 Old Q16 state rejected

```text
softmax_fixed_range_u16
```

必须失败。

## 41.5 Ordinary GIF 不受影响

继续验证：

\[
q_{\max}^{high}=30
\]

且：

\[
q=q_0+q_1,
\qquad
q_0,q_1\in[0,15].
\]

## 41.6 GIF operator count

N 层模型：

```text
GIF = 9N
Phase = 10N + 1
MTN = 10N
```

---

# 42. 实际 smoke test

重新 materialize config：

```bash
python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated
```

检查：

```bash
grep -R \
  "sparse_llm_temporal_v5\|spikellm_nbits16_sentinel_identity" \
  configs/generated
```

然后按当前 mode-aware protocol 重跑 calibration / aware ANN / conversion。

至少先用 Qwen3-1.7B 做 smoke test。

## 42.1 gif-aware static ANN

重新训练快速 gif-aware checkpoint 后：

```bash
CUDA_VISIBLE_DEVICES=<GPU> \
accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__gif_aware.yaml \
  --neuron ann
```

metadata 必须出现：

```text
StaticGIF/SoftmaxIdentityGIF.forward
softmax_site5_gif_policy = spikellm_nbits16_sentinel_identity
```

## 42.2 GIF conversion

```bash
python scripts/convert_snn.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__gif_aware.yaml \
  --neuron gif
```

必须生成 conversion metadata v9。

## 42.3 GIF temporal evaluation

```bash
CUDA_VISIBLE_DEVICES=<GPU> \
accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__gif_aware.yaml \
  --neuron gif
```

要求：

```text
Site 5 不执行 Q16
Site 5 不执行 common Clip
Site 5 temporal operator identity
ordinary GIF sites 仍正常 temporal decomposition
```

---

# 43. 最终全仓库 grep 验收

实施完成后运行：

```bash
rg -n \
  "SoftmaxFixedGIF|softmax_fixed_range_u16|fixed_range_u16_quantized_cumulative_difference|quantized_cumulative_difference|round\(65535|65535\.0" \
  --glob '!docs/history/**'
```

在当前有效代码 / 文档中应无旧 Site 5 Q16 实现残留。

允许 `docs/history/` 中保留历史正文，但必须带 superseded 提示。

---

# 44. 推荐再 grep 模糊 qmax metadata

运行：

```bash
rg -n \
  '"gif_high_qmax"|"gif_local_decomposition_steps"|"gif_per_step_qmax"' \
  snn2 tests
```

如果按本方案完成 metadata scope 重命名，应无旧无作用域 key。

新 key 应为：

```text
ordinary_gif_high_qmax
ordinary_gif_local_decomposition_steps
ordinary_gif_per_step_qmax
```

---

# 45. 最终目标状态

## Ordinary GIF sites

```text
Site 1
Site 2
Site 3
Site 4
Site 6
Site 7
Site 8
Site 9
Site 10
```

使用：

\[
q_{\max}^{high}=30,
\qquad
T=2,
\qquad
q_{\max}^{step}=15.
\]

Temporal：

\[
q=q_0+q_1,
\qquad
q_0,q_1\in[0,15].
\]

## Site 5 GIF

严格跟随 SpikeLLM 当前实现：

```text
n_bits = 16
metric = fix0to1
n_bits >= 16 -> return x
```

所以：

\[
\boxed{GIF_{Site5}(P)=P}
\]

Static：

```python
return x
```

Temporal：

```python
return incoming
```

不做：

```text
Q16
qmax65535
scale
zero point
STE
cumulative Q16
integer chunks
Clip
```

---

# 46. 最终验收清单

- [ ] 删除 `SoftmaxFixedGIF`。
- [ ] 新增 `SoftmaxIdentityGIF`。
- [ ] Site 5 static GIF exact identity。
- [ ] Site 5 temporal GIF exact identity。
- [ ] Site 5 GIF state 不含 Q16 qparams。
- [ ] Site 5 GIF state 明确记录 SpikeLLM `n_bits=16` sentinel provenance。
- [ ] ordinary GIF qmax30 / T2 / step15 完全保持。
- [ ] Site 5 Phase / MTN 完全保持。
- [ ] Site 5 statistics 继续收集。
- [ ] Site 5 永远 no-Clip。
- [ ] temporal implementation v5。
- [ ] site state v7。
- [ ] statistics 保持 v2。
- [ ] calibration manifest v8。
- [ ] conversion metadata v9。
- [ ] generated config 使用 `sparse_llm_temporal_v5`。
- [ ] config 显式锁定 Site 5 identity policy。
- [ ] metadata 的 ordinary GIF qmax 字段作用域明确。
- [ ] GIF activation-neuron operator count 改为每层 9 个。
- [ ] old Q16 state / manifest / conversion 严格拒绝。
- [ ] README 更新。
- [ ] AGENTS.md 更新。
- [ ] 实验执行总结.md 更新。
- [ ] 代码结构总结.md 更新。
- [ ] `docs/history` 中旧 Q16 文档增加 superseded 提示但正文保留。
- [ ] `pytest -q` 全通过。
- [ ] gif-aware static ANN smoke test 通过。
- [ ] gif-aware GIF conversion 通过。
- [ ] gif-aware temporal GIF evaluation 通过。
- [ ] 当前有效代码/文档中不存在 `SoftmaxFixedGIF` / Q16 Site 5 残留。

---

# 47. Codex 最终回复要求

完成修改后必须明确报告：

1. `SoftmaxFixedGIF` 是否已彻底删除；
2. `SoftmaxIdentityGIF` 的 static / temporal 具体行为；
3. 新 Site 5 GIF state 保存哪些字段、删除哪些 Q16 字段；
4. ordinary GIF qmax30/T2/step15 是否保持不变；
5. Site 5 Phase / MTN 是否保持不变；
6. schema 最终版本号；
7. `temporal_policy_metadata()` 如何区分 ordinary GIF 与 Site 5；
8. GIF operator count 是否改为每层 9 个；
9. 修改了哪些当前 Markdown 文档；
10. 哪些 `docs/history` 文档增加了 superseded 提示；
11. 更新了哪些测试；
12. `pytest -q` 结果；
13. gif-aware ANN / conversion / temporal evaluation smoke test 结果。

不要只回复“修改完成”。
