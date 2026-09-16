# GIF Static MSE Refinement — 第三轮修正方案

> 仓库：`https://github.com/wangwk699/SNN`
>
> 基准分支：当前 `main`
>
> 本轮仅修复两个问题：
>
> 1. 统一 GIF scale 最小值 `1e-8`，保证 MSE calibration、StaticGIF ANN forward、Temporal GIF 三者数值语义一致；
> 2. 当 `gif.mse_scale_refinement=false` 时，完全忽略无效的 MSE search config，保证 direct-min-max artifact 的 path semantics 与 validation semantics 一致。
>
> 本文档用于服务器上的 Codex 在无额外上下文时直接完成修改。

---

## 1. 不得改变的已有设计

本轮不得改变：

- static GIF qparams 原则；
- two-pass MSE calibration；
- histogram provenance；
- `calibration.group_size` 配置驱动；
- low/high 独立 refinement；
- Site 1/7 role-specific mask + shared qparams；
- `gif_previous_layers_snn=false/true` 两条 trajectory；
- MSE path suffix；
- saliency、site topology、common Clip；
- Phase/MTN；
- Prefix/rotation；
- training hyperparameters；
- evaluation task 定义。

---

# 2. Problem A：统一 GIF scale floor

## 2.1 当前问题

Direct qparams 当前通过：

```python
scale = ((maximum - minimum) / (qmax - qmin)).clamp_min(1e-8)
```

因此 direct scale 总有：

```text
scale >= 1e-8
```

但 MSE local search 中有：

```python
scale = clip_best.scale * ratio
```

默认：

```text
local_scale_ratio_min = 0.95
```

所以：

```text
clip_best.scale = 1e-8
ratio = 0.95
```

会生成：

```text
scale = 9.5e-9
```

当前 MSE helper 只检查 `scale > 0`，因此这个 candidate 可能参与 ranking，甚至被保存进 `gif_state.pt`。

而实际 StaticGIF ANN runtime 会：

```python
scale.clamp_min(1e-8)
```

于是会出现：

```text
saved/MSE search scale = 9.5e-9
runtime effective scale = 1e-8
```

另外 Temporal GIF 的 qcode 计算通过 `_quantize()` 使用 clamped scale，但 reconstruction 阶段重新读取 `self.low_scale/self.high_scale` 时没有再次 clamp，所以极端边界下 MSE evaluator、ANN StaticGIF、Temporal GIF 三者可能不一致。

本轮必须彻底统一。

---

# 3. 定义唯一常量

在：

```text
snn2/temporal_ops.py
```

新增唯一 canonical 常量：

```python
GIF_SCALE_MIN = 1e-8
```

不要再新增其他同义常量，例如：

```text
GIF_SCALE_EPS
GIF_MIN_SCALE
MSE_SCALE_MIN
```

所有 GIF quantization scale floor 都统一 import `GIF_SCALE_MIN`。

---

# 4. 修改 `_qparams()`

文件：

```text
snn2/calibration.py
```

将：

```python
.clamp_min(1e-8)
```

改为：

```python
.clamp_min(GIF_SCALE_MIN)
```

Direct-min-max 数学不变，只消除 magic number。

---

# 5. 新增 runtime scale canonicalization helper

文件建议：

```text
snn2/gif_mse_calibration.py
```

新增：

```python
def canonical_runtime_scale(scale: float) -> float:
    value = torch.tensor(float(scale), dtype=torch.float32)
    value = value.clamp_min(GIF_SCALE_MIN)

    if not torch.isfinite(value):
        raise ValueError("GIF scale must be finite")

    return float(value.item())
```

其语义固定为：

```text
Python/raw candidate scale
→ FP32
→ clamp_min(GIF_SCALE_MIN)
→ runtime实际有效 scale
```

所有 MSE candidate 在评价之前都必须经过这个 helper。

---

# 6. 修改 `runtime_fake_quant()`

当前已经正确采用：

```text
FP32 quantization arithmetic
+
FP64 MSE accumulation
```

继续保留。

但 scale 必须先 canonicalize。

推荐：

```python
def runtime_fake_quant(
    values: torch.Tensor,
    scale: float,
    zero: int,
    *,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be positive and finite")

    if int(zero) != zero or not qmin <= int(zero) <= qmax:
        raise ValueError(
            "zero must be integer-valued and inside the quantization range"
        )

    x32 = values.to(torch.float32)

    scale32 = torch.tensor(
        canonical_runtime_scale(scale),
        dtype=torch.float32,
        device=x32.device,
    )

    zero32 = torch.tensor(
        float(zero),
        dtype=torch.float32,
        device=x32.device,
    )

    q = torch.round(x32 / scale32) + zero32
    q = torch.clamp(q, qmin, qmax)
    y32 = (q - zero32) * scale32

    return y32.to(torch.float64)
```

注意：

- quantization decision 仍在 FP32；
- reconstructed output 转 FP64 后再做 MSE accumulation；
- `scale < GIF_SCALE_MIN` 时 evaluator 必须表现得与 runtime 一样。

---

# 7. Candidate ranking 前必须 canonicalize scale

在：

```python
optimize_static_qparams(...)
```

中，任何 candidate：

```text
direct
coarse
fine
local
```

都必须在计算 MSE 之前：

```python
scale = canonical_runtime_scale(scale)
```

不要只在最终 persistence 时才 clamp。

正确流程必须是：

```text
raw candidate scale
→ FP32 cast
→ clamp to GIF_SCALE_MIN
→ evaluate MSE
→ candidate ranking
→ persist same exact scale
```

因此：

```text
search scale
==
saved scale
==
runtime scale
```

---

# 8. Direct scale 同样 canonicalize

即使 `_qparams()` 已经 floor：

```python
direct_scale = canonical_runtime_scale(direct_scale)
```

然后再算：

```python
direct_metrics
```

保证 baseline 也使用真正 runtime arithmetic。

---

# 9. Local search 的正确行为

保持：

```python
scale = clip_best.scale * ratio
```

但是 `add()` 内部必须 canonicalize。

例如：

```text
clip_best.scale = 1e-8
ratio = 0.95
raw scale = 9.5e-9
canonical scale = 1e-8
```

MSE evaluator与最终 persistence都使用：

```text
1e-8
```

可选优化：

对 canonicalized：

```python
(scale, zero)
```

做去重，避免 floor 附近多个 ratio 重复评价。

这不是 correctness 必需项。

---

# 10. Persisted state invariant

最终写入：

```text
low_scale
high_scale
direct_low_scale
direct_high_scale
```

的所有 scale 必须满足：

\[
scale \ge GIF\_SCALE\_MIN.
\]

在：

```text
refine_gif_state()
```

最终 assignment 前增加 defensive assertion，例如：

```python
if torch.any(refined_scale < GIF_SCALE_MIN):
    raise RuntimeError(
        "Refined GIF scale violates runtime GIF_SCALE_MIN"
    )
```

Direct baseline tensor也应满足同样 invariant。

---

# 11. 修改 StaticGIF ANN path

文件：

```text
snn2/neurons.py
```

以下已有：

```python
.clamp_min(1e-8)
```

全部改成：

```python
.clamp_min(GIF_SCALE_MIN)
```

至少包括：

```text
StaticGIF._quantize()
StaticGIF._forward_ann_mixed_quant()
```

不要改变：

```text
STE
HTGE
round forward
backward surrogate
```

---

# 12. 修复 Temporal GIF reconstruction scale

当前：

```python
_, low_q, low_zero = self._quantize(...)
_, high_q, high_zero = self._quantize(...)
```

qcode 使用 clamped scale。

但后续：

```python
scale_low = _parameter_values(x, self.low_scale, self.layout)
scale_high = _parameter_values(x, self.high_scale, self.layout)
```

没有 clamp。

改为：

```python
scale_low = _parameter_values(
    x, self.low_scale, self.layout
).clamp_min(GIF_SCALE_MIN)

scale_high = _parameter_values(
    x, self.high_scale, self.layout
).clamp_min(GIF_SCALE_MIN)
```

这样保证：

```text
ANN quantizer scale
Temporal qcode scale
Temporal reconstruction scale
MSE evaluator scale
```

全部一致。

对正常合法 artifact，该修改不改变数值，因为 persisted scale本来就应该 `>= GIF_SCALE_MIN`。

---

# 13. 强化 MSE state validation

文件：

```text
snn2/gif_mse_validation.py
```

当前检查：

```python
scale > 0
```

对 MSE-enabled state 改成：

```python
scale >= GIF_SCALE_MIN
```

必须覆盖：

```text
low_scale
high_scale
direct_low_scale
direct_high_scale
```

建议报错：

```text
MSE GIF scale is below runtime GIF_SCALE_MIN
```

---

# 14. 普通 direct state

如果项目已有统一 direct GIF state qparam validation，也建议检查：

```text
scale >= GIF_SCALE_MIN
```

但不得因此改变 state format version。

历史 direct artifact由 `_qparams()` 生成，本来就是 `>=1e-8`，所以正常 legacy direct artifact不会受到影响。

---

# 15. 新增 scale floor 单元测试

至少新增：

```python
test_mse_local_scale_search_respects_runtime_scale_floor()
```

构造：

```text
direct/clip scale = GIF_SCALE_MIN
local ratio = 0.95
```

要求：

```python
optimized["scale"] >= GIF_SCALE_MIN
```

并且最终 state不会保存：

```text
9.5e-9
```

---

# 16. `runtime_fake_quant()` floor test

新增：

```python
test_runtime_fake_quant_clamps_scale_to_runtime_floor()
```

比较：

```python
runtime_fake_quant(
    values,
    scale=0.95 * GIF_SCALE_MIN,
    ...
)
```

与：

```python
runtime_fake_quant(
    values,
    scale=GIF_SCALE_MIN,
    ...
)
```

要求输出完全一致。

---

# 17. ANN / Temporal / MSE 三路一致性测试

新增关键 regression test：

```python
test_gif_scale_floor_matches_mse_ann_and_temporal_paths()
```

构造：

```text
scale = GIF_SCALE_MIN
```

以及量化 threshold 附近 values。

分别计算：

1. `runtime_fake_quant()`
2. `StaticGIF.forward()`
3. `StaticGIF.temporal(...).sum(dim=0)`

要求逻辑 activation相同条件下三者一致。

至少覆盖：

```text
low branch qmax=15
high branch qmax=30
```

Temporal 输入构造要保证：

```python
incoming.sum(dim=0) == x
```

---

# 18. Defensive malformed-state test

可额外构造：

```text
state scale = 0.95 * GIF_SCALE_MIN
```

验证 runtime defensive clamp：

```text
ANN
Temporal
```

仍保持一致。

但正式 persisted state必须被 validation拒绝。

---

# 19. Problem B：MSE disabled 时完全忽略 MSE search config

## 19.1 当前问题

现有：

```python
gif_qparam_calibration_suffix(cfg)
```

在：

```yaml
mse_scale_refinement: false
```

时返回：

```python
None
```

因此以下参数在 direct mode下：

```text
histogram_bins
coarse_alpha_values
fine_alpha_*
local_scale_ratio_*
local_zero_radius
```

都：

- 不影响算法；
- 不影响 artifact；
- 不影响 path。

这是正确设计。

但：

```python
validate_gif_qparam_manifest_compatibility(...)
```

目前如果 direct manifest 中存在：

```text
gif_mse_refinement_config
```

还会拿它与当前 config比较。

这会导致：

```text
mse=false
histogram_bins: 4096 → 2048
```

虽然 direct算法/path完全没变，但 validator失败。

本轮必须修复。

---

# 20. Direct mode 的 compatibility rule

当：

```python
gif_mse_refinement_enabled(cfg) == False
```

真正需要约束的字段只有：

```text
gif_scale_initialization
gif_mse_scale_refinement
gif_qparam_calibration_method
gif_mse_refinement_signature
```

语义：

```python
gif_scale_initialization == "direct_min_max"
gif_mse_scale_refinement == False
gif_qparam_calibration_method == "direct_min_max"
gif_mse_refinement_signature in {None, missing}
```

而：

```text
gif_mse_refinement_config
```

在 direct mode下必须视为：

```text
informational-only
```

不能参与 artifact compatibility。

---

# 21. 修改 `validate_gif_qparam_manifest_compatibility()`

文件：

```text
snn2/gif_mse_validation.py
```

建议明确拆成两个分支。

## 21.1 MSE enabled

继续严格：

```python
if enabled:
    require:
        gif_scale_initialization == "direct_min_max"
        gif_mse_scale_refinement == True
        gif_qparam_calibration_method == "offline_static_mse"
        gif_mse_refinement_signature == current signature
        gif_mse_refinement_config == current normalized config
```

不允许放松。

---

## 21.2 MSE disabled

改成：

```python
else:
    direct_expected = {
        "gif_scale_initialization": "direct_min_max",
        "gif_mse_scale_refinement": False,
        "gif_qparam_calibration_method": "direct_min_max",
        "gif_mse_refinement_signature": None,
    }

    for key, expected in direct_expected.items():
        if key not in manifest:
            continue  # legacy direct artifact

        if manifest[key] != expected:
            raise ValueError(...)

    # IMPORTANT:
    # gif_mse_refinement_config is deliberately ignored
    # when MSE refinement is disabled.
```

删除当前这种特殊判断：

```python
if key == "gif_mse_refinement_config" and ...
```

Direct mode下根本不应该比较它。

---

# 22. Legacy direct compatibility

以下 artifact继续允许：

### 旧 artifact，无新字段

```python
{}
```

相对于 MSE metadata而言允许。

### 新 direct artifact

```python
{
    "gif_scale_initialization": "direct_min_max",
    "gif_mse_scale_refinement": False,
    "gif_qparam_calibration_method": "direct_min_max",
    "gif_mse_refinement_signature": None,
    "gif_mse_refinement_config": <任意历史合法search config>,
}
```

也允许。

---

# 23. Conflicting metadata仍必须拒绝

Direct mode下以下必须失败：

```python
gif_mse_scale_refinement = True
```

或：

```python
gif_qparam_calibration_method = "offline_static_mse"
```

或：

```python
gif_mse_refinement_signature = "non-null-signature"
```

或：

```python
gif_scale_initialization != "direct_min_max"
```

即：

> 只忽略 unused search config，不忽略真正的 mode identity。

---

# 24. Manifest writer无需删除 config

当前：

```text
materialize_calibration_states()
```

即使 direct mode也可能写：

```text
gif_mse_refinement_config
```

可以继续。

它只是 informational metadata。

本轮无需改 manifest writer，除非为了代码清理。

---

# 25. Artifact path不要改

当前：

```text
MSE false → no MSE suffix
MSE true  → gif_qparams_mse_refined_v1_<signature8>
```

行为正确。

不要修改：

```text
gif_qparam_calibration_suffix()
```

Problem B 是 validator语义问题，不是 path问题。

---

# 26. Direct unused-config compatibility test

新增：

```python
test_direct_artifact_ignores_unused_mse_refinement_config()
```

步骤：

1. Config A：

```yaml
mse_scale_refinement: false
mse_refinement:
  histogram_bins: 4096
```

2. 用 A 生成 direct manifest。

3. Config B：

```yaml
mse_scale_refinement: false
mse_refinement:
  histogram_bins: 2048
```

4. 用 B validation manifest A：

```python
validate_gif_qparam_manifest_compatibility(...)
```

必须通过。

---

# 27. Direct path equality test

同样 A / B：

要求：

```text
MSE disabled
不同 unused search config
→ artifact paths相同
```

至少验证：

```text
ann_training_site_dir
post_finetuning_site_dir
ann_dir
```

以及已有 evaluation path helper中与 aware ANN output绑定的路径。

---

# 28. MSE enabled control test

Config C：

```yaml
mse_scale_refinement: true
histogram_bins: 4096
```

Config D：

```yaml
mse_scale_refinement: true
histogram_bins: 2048
```

必须：

```text
signature(C) != signature(D)
path(C) != path(D)
```

并且：

```text
manifest generated by C
```

用 D 验证时必须失败。

这证明：

```text
disabled → search config不是artifact identity
enabled  → search config是artifact identity
```

---

# 29. Config validation继续保持严格

上一轮已经要求：

```yaml
preserve_zero: true
fallback_to_direct_min_max: true
```

固定 true。

本轮继续。

“Direct mode忽略 MSE search config”只表示：

```text
artifact compatibility忽略其差异
```

并不表示 YAML 可以是非法结构。

`validate_config()` 仍然正常执行。

---

# 30. 推荐修改文件

本轮主要修改：

```text
snn2/temporal_ops.py
snn2/calibration.py
snn2/gif_mse_calibration.py
snn2/gif_mse_state.py
snn2/gif_mse_validation.py
snn2/neurons.py
```

测试至少检查：

```text
tests/test_gif_mse_state.py
tests/test_verify_artifacts.py
tests/test_evaluation_paths.py
```

若仓库已有更合适 path/artifact test 文件，应复用。

---

# 31. 正常情况下不要修改

本轮不应重新修改以下核心逻辑：

```text
snn2/gif_mse_integration.py
snn2/gif_mse_provenance.py
snn2/blockwise_calibration.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
scripts/calibrate_sites.py
```

除非仅为了 import canonical constant或测试所需的极小调整。

尤其不要重构上一轮已经正确的：

```text
statistics ↔ histogram SHA binding
stale histogram cleanup
Pass 2 no-saliency
core bundle MSE validation
```

---

# 32. 推荐实施顺序

### Step 1
新增：

```python
GIF_SCALE_MIN = 1e-8
```

并替换 GIF scale floor magic numbers。

### Step 2
统一：

```text
_qparams
runtime_fake_quant
candidate ranking
state persistence
StaticGIF ANN
StaticGIF temporal
MSE validation
```

### Step 3
补：

```text
scale floor tests
MSE/ANN/Temporal consistency tests
```

### Step 4
修改：

```python
validate_gif_qparam_manifest_compatibility()
```

使 direct mode忽略 `gif_mse_refinement_config`。

### Step 5
补：

```text
direct unused-config compatibility
direct path equality
MSE enabled strict control
```

### Step 6
完整运行：

```bash
pytest -q
```

必须全部通过。

---

# 33. Scale floor验收标准

最终必须满足：

\[
\boxed{
scale_{\mathrm{persisted}} \ge GIF\_SCALE\_MIN
}
\]

并且：

\[
\boxed{
scale_{\mathrm{MSE}}
=
scale_{\mathrm{ANN}}
=
scale_{\mathrm{Temporal}}
}
\]

这里指 runtime effective FP32 arithmetic。

---

# 34. Direct compatibility验收标准

当：

```yaml
mse_scale_refinement: false
```

时，仅修改：

```text
histogram_bins
alpha grid
fine search参数
local scale search参数
local zero radius
```

不得导致：

```text
artifact path变化
validator失败
training/evaluation/conversion拒绝原 direct artifact
```

前提是真正的 direct mode metadata没有冲突。

---

# 35. MSE enabled验收标准

当：

```yaml
mse_scale_refinement: true
```

时，MSE search config变化必须继续：

```text
改变 signature
改变 artifact path
使旧 MSE artifact被新 config拒绝
```

Direct compatibility逻辑不得削弱这一点。

---

# 36. 测试清单

至少确保：

- [ ] `GIF_SCALE_MIN` 只有一个 canonical定义；
- [ ] `_qparams()` 使用该常量；
- [ ] MSE evaluator使用同一 floor；
- [ ] candidate ranking前 scale已 FP32 + floor canonicalize；
- [ ] persisted refined scale不低于 floor；
- [ ] direct baseline scale不低于 floor；
- [ ] StaticGIF ANN使用同一 floor；
- [ ] Temporal GIF qcode/reconstruction使用同一 floor；
- [ ] MSE validator检查 `scale >= GIF_SCALE_MIN`；
- [ ] floor边界下 MSE/ANN/Temporal输出一致；
- [ ] malformed `< floor` persisted MSE state被 validator拒绝；
- [ ] direct mode忽略历史 `gif_mse_refinement_config`差异；
- [ ] direct mode path不受 MSE search config影响；
- [ ] MSE enabled config变化仍改变 signature/path；
- [ ] legacy direct missing新字段仍通过；
- [ ] conflicting direct mode metadata仍失败；
- [ ] `pytest -q` 全部通过。

---

# 37. Codex完成后必须汇报

请输出：

1. 修改文件列表；
2. `GIF_SCALE_MIN` 定义位置；
3. 哪些代码路径统一使用该 scale floor；
4. local `<1e-8` candidate如何 canonicalize；
5. MSE / StaticGIF ANN / Temporal GIF一致性测试；
6. direct-mode compatibility最终规则；
7. direct disabled时不同 search config的 path是否相同；
8. MSE enabled时不同 search config的 path是否不同；
9. 新增/修改测试列表；
10. `pytest -q` 完整结果。

---

# 38. 最终原则

本轮完成后必须满足：

\[
\boxed{
\text{MSE calibration quantizer}
=
\text{StaticGIF ANN quantizer}
=
\text{Temporal GIF effective quantizer}
}
\]

包括 scale-floor 边界。

同时：

\[
\boxed{
\text{MSE disabled}
\Rightarrow
\text{MSE search config 不属于 artifact identity}
}
\]

以及：

\[
\boxed{
\text{MSE enabled}
\Rightarrow
\text{MSE search config 属于 artifact identity}
}
\]

本轮不要修改除此之外的实验方法。
