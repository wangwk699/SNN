# SNN 项目：GIF-aware ANN Fine-tuning 增加 STE / HTGE 可选 Round 代理梯度——完整修改方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 适用代码基线：本文档编写时 `main` 分支当前实现。  
> 本文档必须能够让服务器上的 Codex **在没有任何此前对话上下文的情况下**完成修改。  
> 修改完成后，现有 STE 行为必须保持为默认行为；HTGE 只改变 GIF-aware ANN Fine-tuning 中 `round` 的反向代理梯度，不改变 GIF forward quantization、校准数学、Clip 数学、SNN temporal deployment 数学或其他 ANN mode。

---

## 1. 修改目标

当前 `gif_aware` ANN Fine-tuning 中，GIF fake quantization 的不可导 `round()` 使用 STE：

```python
@staticmethod
def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x
```

因此：

- forward：`round(x)`
- backward：近似 `d round(x) / dx = 1`

本轮修改需要新增另一种代理梯度 **HTGE (Hyperbolic Tanh Gradient Estimator)**，并通过 YAML 配置在 STE / HTGE 之间选择。

最终要求：

```yaml
gif:
  ...
  round_gradient_estimator: STE
  htge_t: 16.0
```

其中：

- `round_gradient_estimator` 只允许：
  - `STE`
  - `HTGE`
- 默认：`STE`
- `htge_t` 默认：`16.0`
- `htge_t` 必须是有限正数，可以为浮点数，不限制为整数。
- `htge_t` 只有在 `round_gradient_estimator: HTGE` 时影响训练数学。
- STE 下 `htge_t` 是无效参数，**不得因为 STE 下的 `htge_t` 不同而创建不同运行路径**。
- HTGE 下 `htge_t` 必须进入 GIF-aware ANN run path，从而不同 `t` 不会互相覆盖。
- STE 和 HTGE 必须进入不同 GIF-aware ANN run path。

---

# 2. 数学定义：必须严格遵守

## 2.1 当前 GIF fake quantization 数学保持不变

当前代码的实际 round 输入是：

\[
u = \frac{x}{s},
\]

然后类似：

\[
q =
\operatorname{clamp}
\left(
\operatorname{round}(u)+z,\,
q_{\min},q_{\max}
\right),
\]

最后：

\[
\hat{x}=(q-z)s.
\]

**本轮禁止把这一数学改写成论文自己的完整 quantizer。**

尤其禁止：

- 不得把 `zero` 移入当前 `round()`；
- 不得修改 scale / zero 的现有定义；
- 不得修改 low/high GIF bit-width；
- 不得修改 salient mask；
- 不得引入论文中的 learnable clipping-range parameters；
- 不得修改当前 common Clip 数学；
- 不得修改当前 hard `clamp` 的 backward；
- 不得修改 SNN temporal integer decomposition。

HTGE 只替换：

\[
\frac{\partial\,\operatorname{round}(u)}{\partial u}
\]

这一处代理梯度。

---

## 2.2 STE

STE 保持当前实现：

\[
y=\operatorname{round}(u)
\]

forward，并在 backward 使用：

\[
\boxed{
\frac{\partial y}{\partial u}\approx 1
}
\]

当前 detach trick 可以保留：

```python
return (x.round() - x).detach() + x
```

---

## 2.3 HTGE

对于当前 `round()` 的输入 `u`，定义：

\[
a=\lfloor u\rfloor,\qquad b=\lceil u\rceil.
\]

HTGE **forward 仍然必须精确使用真正的**：

\[
\boxed{y=\operatorname{round}(u)}
\]

也就是说：

```text
STE forward == HTGE forward == 现有 round forward
```

两者必须 bit-exact 相同。

HTGE backward 使用：

\[
\boxed{
\frac{\partial y}{\partial u}
\approx
\frac{t}{2}
\left[
1-\tanh^2
\left(
t
\left(
u-\frac{a+b}{2}
\right)
\right)
\right]
}
\]

其中：

\[
a=\lfloor u\rfloor,\qquad b=\lceil u\rceil.
\]

### 非常重要：论文公式 (25) 的修正

实现时必须使用：

\[
\boxed{
u-\frac{a+b}{2}
}
\]

而不是：

\[
u\frac{a+b}{2}
\]

或者任何缺少减号的形式。

论文正文的公式 (25) 排版有笔误；其前面的 \(H(x)\) 定义以及后续 quantization-dequantization 梯度展开都对应：

\[
x-\frac{a+b}{2}.
\]

因此代码和测试全部以带减号的修正版为准。

---

## 2.4 推荐实现方式：自定义 autograd Function

建议在 `snn2/neurons.py` 新增一个非常局部、明确的自定义 autograd Function，例如：

```python
class HTGERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.t = float(t)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        t = ctx.t

        a = torch.floor(x)
        b = torch.ceil(x)
        midpoint = 0.5 * (a + b)

        z = t * (x - midpoint)
        tanh_z = torch.tanh(z)
        surrogate = 0.5 * t * (1.0 - tanh_z.square())

        return grad_output * surrogate, None
```

命名可以按项目风格调整，但数学不能调整。

### 精度要求

当前 GIF quantization 在 round 前已经执行：

```python
x.float() / scale.float()
```

所以 HTGE 的 round 输入本身是 float32。保持这一行为。

不要为了 HTGE 把整个 GIF forward 强制成 fp64，也不要修改最终输出 dtype。

---

# 3. HTGE 的作用范围

本轮已经确认以下口径。

## 3.1 作用于 GIF-aware ANN Fine-tuning 中所有真正执行 GIF quantization round 的位置

`HTGE` 应覆盖：

- ordinary salient mixed low/high GIF；
- `AllLowStaticGIF`；
- Site 1 / Site 7 的 multi-role GIF quantization；
- 其他实际复用同一 GIF rounding path 的 quantized GIF site。

不应作用于：

- `IdentityGIF`；
- `SoftmaxIdentityGIF` / Site 5 identity policy；
- 任何本身不执行 `round()` 的 GIF identity site。

identity site 必须继续保持 exact identity。

---

## 3.2 只改变 round backward，不改 forward trajectory

对于相同：

- input；
- GIF state；
- scale；
- zero；
- mask；
- qmin/qmax；
- Clip；
- model weights；

必须满足：

```text
STE GIF forward == HTGE GIF forward
```

不仅近似相同，而是应当在相同 dtype/device 下 **bit-exact 相同**，因为二者 forward 都必须调用同一个真实 `torch.round()` 语义。

---

## 3.3 ANN-training calibration / post-finetuning calibration 不因为 STE/HTGE 分叉

STE / HTGE 是 **ANN training backward policy**。

它不是 calibration-derived state。

因此：

- `ann_training` calibration 不因为 estimator 或 `t` 分裂；
- shared pre-finetuning Prefix 不因为 estimator 或 `t` 分裂；
- calibration `gif_state.pt` 不保存 estimator；
- calibration `clip_state.pt` 不保存 estimator；
- calibration manifest 不把 estimator 当作 calibration identity；
- `htge_t` 不进入 shared calibration path。

原因：

```text
STE forward == HTGE forward
```

因此 pre-finetuning forward calibration statistics 完全相同。

---

# 4. YAML / Config 修改

## 4.1 `configs/experiment_matrix.yaml`

在所有 experiment base config 的：

```yaml
gif:
```

下新增：

```yaml
round_gradient_estimator: STE
htge_t: 16.0
```

建议放在现有 runtime quantization 参数附近，例如：

```yaml
gif:
  base_bits: 4
  add_bits: 1
  high_qmax: 30
  temporal_steps: 2
  per_step_qmax: 15
  low_ratio: 0.9
  salient_ratio: 0.1
  scale_initialization: direct_min_max
  mse_scale_refinement: false
  runtime_quantization: static
  round_gradient_estimator: STE
  htge_t: 16.0
  saliency_rule: operator_aware_spikellm_extension
```

默认必须是 STE，从而不主动改变当前实验数学。

---

## 4.2 `snn2/config.py`

### 4.2.1 默认值

在 `resolve_config()` 中为 `gif` 添加兼容默认值：

```python
gif_cfg = cfg.setdefault("gif", {})
gif_cfg.setdefault("round_gradient_estimator", "STE")
gif_cfg.setdefault("htge_t", 16.0)
```

目的：

- 老的 hand-written config 若缺字段，默认仍恢复当前 STE 行为；
- materialized generated config 仍应显式包含这两个字段。

---

### 4.2.2 Validation

新增严格 validation：

```python
estimator = cfg["gif"].get("round_gradient_estimator")
if estimator not in {"STE", "HTGE"}:
    raise ValueError(...)

try:
    htge_t = float(cfg["gif"]["htge_t"])
except (TypeError, ValueError):
    raise ValueError(...)

if not math.isfinite(htge_t) or htge_t <= 0.0:
    raise ValueError(...)
```

要求：

- 大小写严格；
- 不静默接受 `ste` / `htge`；
- 不静默 canonicalize；
- `0`、负数、`NaN`、`inf` 必须拒绝。

推荐增加常量：

```python
GIF_ROUND_GRADIENT_ESTIMATORS = {"STE", "HTGE"}
GIF_DEFAULT_HTGE_T = 16.0
```

避免 magic values 分散。

---

# 5. Generated YAML

运行：

```bash
python scripts/materialize_configs.py
```

重新生成：

```text
configs/generated/*.yaml
```

当前 materializer 生成 12 个主实验 YAML；不要手工只改 `gif_aware` generated YAML。

所有 generated config 都应保留：

```yaml
gif:
  round_gradient_estimator: STE
  htge_t: 16.0
```

即使某个 mode 本次不会使用 HTGE，也保持 schema 一致。

---

# 6. `snn2/neurons.py` 修改

这是本轮数学实现的核心。

## 6.1 不删除现有 STE

保留现有：

```python
@staticmethod
def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x
```

现有 STE 数学必须 regression 保持。

---

## 6.2 新增 HTGE round

新增 HTGE autograd 实现，满足第 2 节公式。

推荐：

```python
class HTGERound(torch.autograd.Function):
    ...
```

并提供一个统一 dispatch，例如：

```python
def _round_with_surrogate(self, x: torch.Tensor) -> torch.Tensor:
    if self.round_gradient_estimator == "STE":
        return self.round_ste(x)
    if self.round_gradient_estimator == "HTGE":
        return HTGERound.apply(x, self.htge_t)
    raise RuntimeError(...)
```

不要在多个 quantization 分支复制 HTGE 数学。

---

## 6.3 `StaticGIF`

把构造函数改为支持 training backward policy，例如：

```python
class StaticGIF(nn.Module):
    def __init__(
        self,
        state: dict[str, Any],
        *,
        round_gradient_estimator: str = "STE",
        htge_t: float = 16.0,
    ):
        ...
```

保存为普通 Python runtime attributes：

```python
self.round_gradient_estimator = ...
self.htge_t = ...
```

**不要注册成 calibration buffer。**

它们不是 `gif_state.pt` 的一部分。

---

## 6.4 替换实际 round 调用

当前类似：

```python
q = self.round_ste(x.float() / scale.float()) + zero.float()
```

改成统一 dispatch：

```python
q = self._round_with_surrogate(
    x.float() / scale.float()
) + zero.float()
```

必须覆盖：

- `_quantize()`
- `_forward_ann_mixed_quant()`

以及任何其他实际 GIF fake quantization round call。

---

## 6.5 `AllLowStaticGIF`

当前 `AllLowStaticGIF` 继承 `StaticGIF`，但其 `__init__` 使用：

```python
nn.Module.__init__(self)
```

而不是调用完整 `StaticGIF.__init__()`。

因此必须特别处理 estimator/t runtime configuration，不能只改 `StaticGIF.__init__()` 然后以为 AllLow 会自动得到字段。

推荐抽出一个小 helper，例如：

```python
def _configure_round_gradient(
    module: nn.Module,
    estimator: str,
    htge_t: float,
) -> None:
    ...
```

或者在 `StaticGIF` 中做专用 method，由 `AllLowStaticGIF.__init__()` 显式调用。

必须保证：

```text
ordinary StaticGIF
AllLowStaticGIF
```

都使用同一套 dispatch。

---

## 6.6 `IdentityGIF` / `SoftmaxIdentityGIF`

不要加入 fake quantization。

它们必须继续 exact identity。

允许 factory 接收 estimator/t 后忽略这些参数，但 identity 类本身不要为了“接口统一”新增任何 round。

---

# 7. `gif_module_from_state()` 修改

当前 factory：

```python
def gif_module_from_state(state: dict[str, Any]) -> nn.Module:
```

建议变为：

```python
def gif_module_from_state(
    state: dict[str, Any],
    *,
    round_gradient_estimator: str = "STE",
    htge_t: float = 16.0,
) -> nn.Module:
```

行为：

```text
ordinary GIF policy  -> StaticGIF(... estimator/t ...)
all-low GIF policy   -> AllLowStaticGIF(... estimator/t ...)
identity policy      -> IdentityGIF(state)
softmax identity     -> SoftmaxIdentityGIF(state)
```

identity policy 不使用 estimator/t。

默认值必须保持 STE，避免 calibration / deployment 旧调用点因为新参数而破坏。

---

# 8. `snn2/controller.py` 修改

## 8.1 `SiteController.__init__`

新增 runtime 参数，例如：

```python
gif_round_gradient_estimator: str = "STE",
gif_htge_t: float = 16.0,
```

保存：

```python
self.gif_round_gradient_estimator = ...
self.gif_htge_t = ...
```

对其做 defensive validation。

---

## 8.2 `_load()` 中 GIF module 构造

关键要求：

### ANN `mode == "gif"`

必须把 YAML 中的 estimator/t 传给 GIF module：

```python
modules["gif"] = gif_module_from_state(
    state,
    round_gradient_estimator=self.gif_round_gradient_estimator,
    htge_t=self.gif_htge_t,
)
```

### temporal SNN / sequential calibration deployment

不要让 estimator/t 成为 deployment identity。

推荐：

```python
if self.mode == "gif":
    # ANN static GIF replacement
    pass estimator/t
else:
    # deploy_gif / calibration_deploy
    use factory defaults / hard forward-equivalent path
```

即使 temporal evaluation accidentally 使用带 HTGE 的 module，forward 数值也必须相同，但架构语义上应保持：

```text
STE / HTGE = ANN backward policy
SNN deployment = 与该 backward policy 无关
```

---

# 9. `snn2/training.py` 修改

在 `train_full_parameters()` 构造 `SiteController` 时：

```python
controller = SiteController(
    ...
    gif_round_gradient_estimator=str(
        cfg["gif"]["round_gradient_estimator"]
    ),
    gif_htge_t=float(cfg["gif"]["htge_t"]),
    ...
)
```

至少在 `mode == "gif"` 时传入有效配置。

---

## 9.1 训练 provenance metadata

建议在 GIF-aware training result 中显式记录：

```json
{
  "gif_round_gradient_estimator": "STE"
}
```

或 HTGE：

```json
{
  "gif_round_gradient_estimator": "HTGE",
  "gif_htge_t": 16.0
}
```

推荐规则：

- `gif_round_gradient_estimator`：gif-aware 时记录；
- `gif_htge_t`：HTGE 时记录；
- STE 下可以不记录有效 `t`，或者记录一个 `null`；
- 不要让 STE 下无意义 `htge_t` 看起来像 active training parameter。

也可以在 saved Hugging Face `model.config` 中写入：

```python
model.config.snn2_gif_round_gradient_estimator = ...
model.config.snn2_gif_htge_t = ...  # only semantically active for HTGE
```

用于 checkpoint provenance。

---

# 10. `snn2/evaluation.py` 修改

Final ANN evaluation 对 `gif_aware` 仍然会构造：

```text
controller.mode == "gif"
```

为了配置/metadata 一致，建议在 final ANN evaluation 的 `SiteController` 构造中也传入：

```python
gif_round_gradient_estimator=cfg["gif"]["round_gradient_estimator"],
gif_htge_t=cfg["gif"]["htge_t"],
```

注意：

- evaluation 使用 `torch.no_grad()`；
- STE/HTGE forward 必须相同；
- 所以这不会改变 ANN evaluation 数值；
- 但可以保证 runtime configuration 与训练 checkpoint 的 config 一致。

---

## 10.1 Evaluation metadata

对 final `gif_aware` ANN evaluation，建议 metadata 增加：

```json
{
  "gif_round_gradient_estimator": "HTGE",
  "gif_htge_t": 16.0
}
```

STE：

```json
{
  "gif_round_gradient_estimator": "STE"
}
```

对：

- vanilla
- unaware
- phase_aware
- temporal phase/gif/mtn SNN

不要错误宣称 HTGE 是 active runtime backward policy。

---

# 11. Artifact 路径：必须修改

文件：

```text
snn2/artifacts.py
```

当前 `gif_aware` 的 run root 由：

```python
gif_training_dirname(...)
```

构造。

必须把 estimator 加入该函数。

---

## 11.1 路径规则

### STE

```text
round_gradient_estimator_STE
```

### HTGE, t=16

```text
round_gradient_estimator_HTGE_htge_t_16
```

### HTGE, t=8

```text
round_gradient_estimator_HTGE_htge_t_8
```

`t` 建议统一使用：

```python
format(float(htge_t), ".12g")
```

因此：

```text
16
16.0
16.000
```

canonicalize 成：

```text
16
```

避免同一数值形成多个目录。

---

## 11.2 STE 下不要把 `htge_t` 放入路径

这是明确要求。

即：

```yaml
round_gradient_estimator: STE
htge_t: 16
```

和：

```yaml
round_gradient_estimator: STE
htge_t: 8
```

路径应相同，因为 `htge_t` 对 STE 完全不起作用。

不要产生：

```text
round_gradient_estimator_STE_htge_t_16
round_gradient_estimator_STE_htge_t_8
```

这种无意义分叉。

---

## 11.3 推荐修改 `gif_training_dirname()`

示意：

```python
def gif_training_dirname(
    *,
    phase_T,
    mtn_T,
    warmup_ratio,
    round_gradient_estimator,
    htge_t,
    lr_scheduler_type=None,
    gradient_accumulation_steps=None,
):
    ...
    estimator = str(round_gradient_estimator)

    if estimator == "STE":
        result += "_round_gradient_estimator_STE"
    elif estimator == "HTGE":
        t_string = format(float(htge_t), ".12g")
        result += (
            "_round_gradient_estimator_HTGE"
            f"_htge_t_{t_string}"
        )
    else:
        raise ValueError(...)
```

在 `ArtifactLayout.__init__()` 的 `gif_aware` 分支传：

```python
round_gradient_estimator=cfg["gif"]["round_gradient_estimator"],
htge_t=cfg["gif"]["htge_t"],
```

---

# 12. 哪些路径必须分开，哪些必须共用

这是本轮最重要的 artifact 语义之一。

## 12.1 必须共用，不因 STE / HTGE / t 改变

以下属于 pre-finetuning shared artifacts：

```text
_shared/.../pre_finetuning_prefix/
_shared/.../ann_training_calibration/
```

以及对应：

```text
ann_training_site_dir
ann_training_clip_profile_dir
```

它们不应包含：

```text
round_gradient_estimator
htge_t
```

原因是 STE / HTGE forward 完全相同。

---

## 12.2 必须隔离

因为 STE 和 HTGE 会得到不同的 trained ANN weights，以下全部必须自然位于不同 run root：

```text
ann/
  final/
  trainer/
  training_result.json
  evaluation/...

post_finetuning/
  prefix/
  conversion_calibration/
  ...

snn/
  ...

logs/
config/
```

也就是说，只要把 estimator/t 加进 `gif_aware` run root，后续：

- Final ANN checkpoint；
- ANN evaluation；
- post-finetuning Prefix；
- post-finetuning calibration；
- SNN conversion；
- SNN evaluation；

都会自然隔离。

不要再在这些子目录各自重复创建第二套 estimator suffix。

---

# 13. `gif_state.pt` / calibration state 禁止修改语义

不要向 `gif_state.pt` 写入：

```text
round_gradient_estimator
htge_t
```

不要因此 bump GIF calibration state format version。

原因：

```text
gif_state.pt = calibration-derived forward quantization parameters
STE/HTGE     = ANN-training backward surrogate policy
```

两者语义不同。

同理：

- `clip_state.pt` 不加；
- shared calibration manifest 不加 estimator identity；
- Stage A / Stage B 统计数学不改。

---

# 14. Common Clip 保持完全不变

当前项目的 Clip 仍保持现有：

```text
hard_clip
```

backward。

论文 HTGE 还讨论了利用 HTGE 学习 clipping range，但 **本项目不移植这一部分**。

本项目中的：

- calibration-derived Clip lower/upper；
- GIF scale / zero；
- salient mask；

仍是现有固定 calibration state。

本轮 HTGE 只影响通过 fake-quantized activation 传播回 upstream ANN parameters 的 gradient。

---

# 15. SNN conversion / temporal deployment 保持不变

必须确保：

```text
HTGE 不改变任何 SNN temporal forward 数学
```

包括：

- GIF 2-step temporal decomposition；
- high qmax = 30；
- per-step qmax = 15；
- low/high integer code；
- zero-point handling；
- Site 5 identity；
- all-low temporal policy；
- salient mask；
- temporal aggregation。

HTGE 是训练期 backward estimator，不是 SNN neuron deployment rule。

---

# 16. 需要重点检查的代码调用链

修改完成后，Codex 必须从头检查以下链路，不要只改 `neurons.py`：

```text
configs/experiment_matrix.yaml
        ↓
scripts/materialize_configs.py
        ↓
configs/generated/*.yaml
        ↓
snn2/config.py
        ↓
ArtifactLayout
        ↓
gif_aware run root
        ↓
snn2/training.py
        ↓
SiteController(mode="gif")
        ↓
gif_module_from_state(...)
        ↓
StaticGIF / AllLowStaticGIF
        ↓
round surrogate dispatch
        ↓
STE or HTGE backward
```

以及 final ANN evaluation：

```text
evaluate
  ↓
build_evaluation_controller()
  ↓
SiteController(mode="gif")
  ↓
Static GIF forward
```

和 deployment：

```text
SNN evaluation
  ↓
SiteController(mode="deploy_gif")
  ↓
temporal GIF
```

确保 deployment 不依赖 HTGE。

---

# 17. 测试要求

建议优先扩展：

```text
tests/test_neurons.py
tests/test_generated_configs.py
tests/test_evaluation_paths.py
```

必要时新建：

```text
tests/test_gif_round_gradient.py
```

建议把纯数学 gradient tests 放入独立文件，避免 `test_neurons.py` 继续膨胀。

---

## 17.1 HTGE forward 必须等于 `torch.round`

测试多个输入：

```python
x = torch.tensor([
    -2.3, -2.0, -1.5, -1.1,
     0.0,  0.2,  0.5,  0.9,
     1.0,  1.5,  2.7,
], dtype=torch.float32)
```

要求：

```python
torch.equal(htge_round(x, t=16), torch.round(x))
```

---

## 17.2 STE / HTGE forward 必须 bit-exact 一致

```python
torch.equal(
    ste_round(x),
    htge_round(x, t=16),
)
```

---

## 17.3 STE backward regression

对：

```python
y = ste_round(x).sum()
y.backward()
```

要求 round surrogate 本身的：

```text
x.grad == 1
```

注意：这是只测试 round surrogate，不包含后续 clamp saturation。

---

## 17.4 HTGE backward analytical test

手工构造：

\[
a=\lfloor x\rfloor,\quad
b=\lceil x\rceil,
\]

\[
g(x)=
\frac{t}{2}
\left[
1-\tanh^2
\left(
t\left(x-\frac{a+b}{2}\right)
\right)
\right].
\]

然后：

```python
loss = HTGERound.apply(x, t).sum()
loss.backward()
```

要求：

```python
torch.testing.assert_close(
    x.grad,
    expected_gradient,
)
```

必须显式覆盖论文公式中修正后的减号。

---

## 17.5 不同 `t` 产生不同 backward

例如：

```text
t = 4
t = 8
t = 16
```

对非特殊输入要求 gradient profile 不同。

同时保证 forward 相同。

---

## 17.6 HTGE 数值稳定性

对：

- 很靠近 integer；
- 很靠近 half-integer；
- 普通随机输入；

验证：

```text
gradient finite
no NaN
no inf
```

---

## 17.7 StaticGIF STE vs HTGE forward

对普通 `StaticGIF`：

- 同一 state；
- 同一 input；
- STE module；
- HTGE module；

要求：

```python
torch.equal(out_ste, out_htge)
```

或在项目实际 dtype path 下严格检查 bit-exact。

---

## 17.8 StaticGIF backward 必须不同

构造不触发 qmin/qmax clamp saturation 的输入，分别反传。

要求：

```text
STE upstream gradient != HTGE upstream gradient
```

并验证 HTGE 与 analytical formula 对应。

---

## 17.9 `AllLowStaticGIF`

必须单独测试：

```text
STE forward == HTGE forward
```

且 HTGE backward 确实生效。

这是必须项，因为 AllLow 类没有直接调用完整 `StaticGIF.__init__()`。

---

## 17.10 multi-role GIF

至少测试一个 multi-role state：

```text
q/k/v or current project实际 role
```

保证：

- role mask selection 不变；
- STE/HTGE forward 相同；
- HTGE 只影响 round backward。

---

## 17.11 identity GIF

对：

```text
IdentityGIF
SoftmaxIdentityGIF
```

保证：

- estimator=STE / HTGE 都不改变 identity；
- 不调用 HTGE round；
- output 仍 exact same object / exact same tensor semantics（按现有测试约束）。

---

## 17.12 Temporal deployment regression

同一 input/state：

```text
temporal output with STE-configured training policy
==
temporal output with HTGE-configured training policy
```

确保 SNN temporal math 不变。

---

# 18. Config 测试

## 18.1 默认值

缺少新字段的旧 config 经 `resolve_config()`：

```yaml
round_gradient_estimator: STE
htge_t: 16.0
```

---

## 18.2 合法值

接受：

```yaml
round_gradient_estimator: STE
htge_t: 16
```

接受：

```yaml
round_gradient_estimator: HTGE
htge_t: 16.0
```

接受：

```yaml
round_gradient_estimator: HTGE
htge_t: 0.5
```

---

## 18.3 非法值

必须拒绝：

```text
ste
htge
foo
""
None
```

以及：

```text
htge_t = 0
htge_t < 0
NaN
+inf
-inf
```

---

# 19. Artifact path 测试

必须新增明确 regression。

假设其他参数完全相同。

## 19.1 STE vs HTGE

```yaml
STE, t=16
HTGE, t=16
```

要求：

```text
layout_ste.root != layout_htge.root
```

---

## 19.2 HTGE 不同 t

```yaml
HTGE, t=16
HTGE, t=8
```

要求：

```text
layout_htge16.root != layout_htge8.root
```

并包含：

```text
round_gradient_estimator_HTGE_htge_t_16
round_gradient_estimator_HTGE_htge_t_8
```

---

## 19.3 STE 不同无效 t

```yaml
STE, t=16
STE, t=8
```

要求：

```text
layout_ste16.root == layout_ste8.root
```

因为 STE 不使用 `htge_t`。

---

## 19.4 Shared pre-finetuning artifacts 必须相同

对：

```text
STE
HTGE t=16
HTGE t=8
```

要求以下路径相同：

```python
ann_training_prefix_dir
ann_training_calibration_dir
ann_training_site_dir
ann_training_clip_profile_dir
```

---

## 19.5 Trained / post-finetuning artifacts 必须不同

要求：

```python
ann_dir
ann_checkpoint_dir
post_finetuning_dir
post_finetuning_prefix_dir
post_finetuning_conversion_calibration_dir
```

STE / HTGE 必须不同。

HTGE 不同 `t` 也必须不同。

因为它们都挂在不同 GIF-aware run root。

---

# 20. Generated config 测试

`tests/test_generated_configs.py` 中加入：

- 12 个 generated config 都有：
  - `gif.round_gradient_estimator`
  - `gif.htge_t`
- 默认：
  - `STE`
  - `16.0`
- config validation 全部通过；
- materialize 后的配置与 `experiment_matrix.yaml` 一致。

---

# 21. Evaluation metadata / path 测试

对 `gif_aware` Final ANN evaluation：

- path 自动继承 training run root；
- STE / HTGE 不覆盖；
- HTGE t=8 / 16 不覆盖；
- forward metadata 可以识别当前 estimator；
- 但 ANN evaluation logits/forward 不应因为 estimator 本身产生差异（checkpoint weights 相同的单元测试场景下）。

对 SNN evaluation：

- 仍使用现有 temporal GIF；
- estimator 不被报告为 active SNN runtime operator。

---

# 22. 不要修改的内容

除非测试暴露出真正依赖问题，本轮不要借机修改：

```text
Hadamard rotation
Prefix discovery 数学
saliency statistics
GIF mask policy
calibration group_size
Phase surrogate
MTN
Clip interval mathematics
common_clip_enabled semantics
Site topology
Site 5 identity policy
GIF qmax/chunk policy
post-finetuning artifact source policy
evaluation task definitions
optimizer
learning-rate scheduler
teacher forcing
data sampling
```

---

# 23. 兼容性要求

## 23.1 旧实验默认行为

新字段默认：

```yaml
round_gradient_estimator: STE
htge_t: 16.0
```

因此数学上旧 GIF-aware STE 实验保持不变。

注意：

由于新的 GIF-aware run path 显式加入：

```text
round_gradient_estimator_STE
```

所以旧版已经保存的 GIF-aware checkpoint 所在旧目录不会被新代码自动视为同一路径。

这是预期行为，有利于避免新旧目录语义混淆。

---

## 23.2 Calibration artifact 不要求重跑，仅因为 estimator 变化时

若其他 calibration 配置完全相同，仅：

```text
STE -> HTGE
```

或：

```text
HTGE t=16 -> HTGE t=8
```

不应要求重新生成：

```text
ann_training Prefix
ann_training Stage-A GIF state
ann_training Clip profile
```

因为这些 pre-finetuning artifacts 与 backward estimator 无关。

但是不同 trained ANN checkpoint 的：

```text
post_finetuning Prefix
post_finetuning conversion calibration
```

仍然必须各自重新生成，因为训练后的模型权重已经不同。

---

# 24. 建议的实现顺序

按以下顺序修改，避免路径和 runtime 改一半：

1. `snn2/config.py`
   - constants/defaults/validation。

2. `configs/experiment_matrix.yaml`
   - 新字段。
   - regenerate generated configs。

3. `snn2/neurons.py`
   - HTGE autograd；
   - STE/HTGE dispatch；
   - StaticGIF；
   - AllLowStaticGIF；
   - factory。

4. `snn2/controller.py`
   - runtime config plumbing。

5. `snn2/training.py`
   - YAML -> controller；
   - provenance metadata。

6. `snn2/evaluation.py`
   - final ANN controller / metadata。

7. `snn2/artifacts.py`
   - GIF-aware run root path isolation。

8. tests
   - math；
   - config；
   - path；
   - forward invariance；
   - deployment invariance。

9. `pytest -q`

---

# 25. 建议的测试命令

先跑局部：

```bash
pytest -q \
  tests/test_neurons.py \
  tests/test_generated_configs.py \
  tests/test_evaluation_paths.py
```

如果新增：

```text
tests/test_gif_round_gradient.py
```

则：

```bash
pytest -q tests/test_gif_round_gradient.py
```

最后必须：

```bash
pytest -q
```

全部通过。

---

# 26. 最低验收标准

代码修改完成后，至少满足以下全部条件：

- [ ] YAML `gif` 下存在 `round_gradient_estimator`。
- [ ] YAML `gif` 下存在 `htge_t`。
- [ ] 默认 estimator = `STE`。
- [ ] 默认 `htge_t = 16.0`。
- [ ] estimator 只允许 `STE` / `HTGE`。
- [ ] `htge_t` 为 finite positive float。
- [ ] STE forward 保持当前实现。
- [ ] HTGE forward 精确等于 `torch.round`。
- [ ] HTGE backward 使用本文修正公式。
- [ ] 公式使用 `x - (a+b)/2`，不能漏掉减号。
- [ ] ordinary StaticGIF 支持 HTGE。
- [ ] AllLowStaticGIF 支持 HTGE。
- [ ] multi-role GIF 支持 HTGE。
- [ ] IdentityGIF 不受影响。
- [ ] SoftmaxIdentityGIF 不受影响。
- [ ] common Clip 数学不变。
- [ ] GIF calibration state schema 不因 HTGE 改变。
- [ ] pre-finetuning shared calibration path 不因 STE/HTGE 改变。
- [ ] STE / HTGE 的 GIF-aware run root 不同。
- [ ] HTGE 不同 `t` 的 run root 不同。
- [ ] STE 不同无效 `htge_t` 的 run root 相同。
- [ ] Final ANN checkpoints 不覆盖。
- [ ] post-finetuning Prefix/calibration 不覆盖。
- [ ] 后续 SNN results 不覆盖。
- [ ] SNN temporal GIF forward 数学不变。
- [ ] STE / HTGE 相同 state/input 的 GIF forward bit-exact 相同。
- [ ] HTGE analytical gradient unit test 通过。
- [ ] 全部 `pytest -q` 通过。

---

# 27. 最终期望的配置示例

## 27.1 当前默认 STE

```yaml
gif:
  base_bits: 4
  add_bits: 1
  high_qmax: 30
  temporal_steps: 2
  per_step_qmax: 15
  low_ratio: 0.9
  salient_ratio: 0.1
  scale_initialization: direct_min_max
  mse_scale_refinement: false
  runtime_quantization: static
  round_gradient_estimator: STE
  htge_t: 16.0
  saliency_rule: operator_aware_spikellm_extension
```

路径包含：

```text
round_gradient_estimator_STE
```

不包含：

```text
htge_t_16
```

---

## 27.2 HTGE, t=16

```yaml
gif:
  ...
  round_gradient_estimator: HTGE
  htge_t: 16.0
```

路径包含：

```text
round_gradient_estimator_HTGE_htge_t_16
```

---

## 27.3 HTGE, t=8

```yaml
gif:
  ...
  round_gradient_estimator: HTGE
  htge_t: 8.0
```

路径包含：

```text
round_gradient_estimator_HTGE_htge_t_8
```

---

# 28. 本轮设计的核心边界

一句话概括：

> **保持当前 GIF quantizer 和所有 forward/deployment 数学完全不变，只把 GIF-aware ANN Fine-tuning 中不可导 `round` 的 backward estimator 从固定 STE 扩展为 YAML 可选的 STE / HTGE，并通过 GIF-aware run root 对 estimator 及 HTGE 的 `t` 做完整 artifact 隔离。**

这条边界优先级高于“为了代码统一而顺手重构”的需求。任何超出这一边界的行为都应避免。
