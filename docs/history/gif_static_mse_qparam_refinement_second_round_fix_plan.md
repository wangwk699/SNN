# GIF Static MSE QParam Refinement — 第二轮修正方案

> 仓库：`https://github.com/wangwk699/SNN`
>
> 基准代码：当前 `main`，本轮检查基于提交：
>
> ```text
> a90f39e2fc70c66bb5358ad9cec47e5790a78fe0
> ```
>
> 本文档用于部署在服务器上的 Codex 在**没有任何对话上下文**的情况下完成本轮修正。
>
> 本轮不是重新设计 Static MSE refinement，而是在当前已经实现的基础上修复 correctness、provenance、validation、backward compatibility 与 calibration 性能问题。

---

# 1. 当前实现中已经正确、必须保留的设计

以下部分已经正确，本轮不得破坏：

1. GIF qparams 仍然是 static：
   - calibration 阶段生成；
   - ANN training / ANN evaluation / SNN conversion 全程固定；
   - 不允许 dynamic per-token / per-batch scale/zero；
   - `scale/zero` 不是 trainable parameter。

2. MSE refinement 仍然是：
   ```text
   Pass 1 statistics
   → Pass 2 histogram
   → offline MSE search
   → fixed qparams
   → gif_state.pt
   ```

3. `calibration.group_size` 已经配置驱动：
   ```python
   configured_group_size = int(cfg["calibration"]["group_size"])
   ```
   不得硬编码 128。

4. 必须继续支持：
   ```text
   group_size = -1
   group_size = 64
   group_size = 128
   其他合法正整数
   ```

5. `gif_previous_layers_snn=false/true` 两条 trajectory 均保留。

6. low/high qparams 独立优化：
   ```text
   low:  qmax=15
   high: qmax=30
   ```

7. Site 1 / Site 7 保持：
   ```text
   role-specific mask
   + shared qparams
   ```

8. MSE artifact path 继续与：
   ```text
   group_size
   MSE config signature
   calibration trajectory
   ```
   隔离。

9. `mse_scale_refinement=false` 时，direct-min-max runtime 数学行为不得改变。

---

# 2. 本轮需要修复的问题总览

当前代码存在以下 6 类问题。

## P0-1：MSE histogram 没有绑定具体 statistics 文件

当前 histogram provenance 虽然记录了：

```text
num_samples
configured_group_size
previous_layers_snn
trajectory_source
calibration_data_manifest_sha256
prefix_state_sha256
prefix_kv_sha256
rotation_state_sha256
mse_refinement_signature
```

但是没有记录它所依赖的：

```text
statistics.pt
```

或者：

```text
gif_statistics.pt
```

的 SHA256。

因此理论上可能出现：

```text
新的 statistics.pt
+
旧的 gif_mse_histogram.pt
```

仍然通过 provenance 检查。

必须修复。

---

## P0-2：MSE evaluator 不是 runtime-equivalent FP32 quantizer

当前：

```python
runtime_fake_quant(...)
```

把 activation 转成：

```python
float64
```

后做：

```text
x / scale
round
clamp
dequant
```

但实际 `StaticGIF` runtime 明确使用：

```python
x.float()
scale.float()
```

即 quantization decision 发生在 FP32。

这意味着 calibration search 当前优化的是：

\[
Q_{\mathrm{FP64}}
\]

而真正 ANN/SNN runtime 使用：

\[
Q_{\mathrm{FP32}}.
\]

必须改成：

```text
quantization arithmetic = FP32
MSE accumulation        = FP64
```

---

## P1-1：普通 training/evaluation validation 没有强制 MSE-specific validation

当前已经有：

```python
validate_gif_mse_state(...)
```

但是普通：

```python
validate_site_state_bundle(...)
```

没有统一调用它。

因此如果用户直接：

```bash
python scripts/train_ann.py ...
```

而没有提前运行：

```bash
python scripts/verify_artifacts.py ...
```

则 MSE metadata / histogram / qparams provenance 的错误可能不会在训练开始前被阻止。

必须把 MSE validation 接入核心 state bundle validation。

---

## P1-2：direct-min-max artifact 只做到 path backward compatibility，没有做到 manifest backward compatibility

当前 `mse_scale_refinement=false` 时：

- artifact path 保持旧路径；
- 这是正确的。

但是新版 `verify_artifacts.py` 又强制要求旧 direct artifact 中存在新字段，例如：

```text
gif_scale_initialization
gif_mse_scale_refinement
gif_qparam_calibration_method
gif_mse_refinement_signature
```

旧 artifact 并没有这些字段。

所以目前实际语义是：

```text
旧 path 可找到
但旧 artifact 会被 verifier 拒绝
```

这与前一轮约定的 backward compatibility 不一致。

本轮按本文建议修成：

> 对 `mse=false` 的 legacy direct artifact，缺失新 MSE metadata 字段时允许按旧 direct-min-max 默认值解释；如果字段存在但与 direct 语义冲突，则报错。

---

## P2-1：`preserve_zero` / `fallback_to_direct_min_max` 当前是伪可配置项

当前 config 中：

```yaml
preserve_zero: true
fallback_to_direct_min_max: true
```

虽然会参与 validation / signature，但实现实际上始终：

- 搜索区间包含 0；
- direct-min-max 永远作为 candidate；
- persist 后如果无严格改善则直接回退 direct。

因此把它们设置为 false 并不会产生一个真正有定义的另一套算法。

本轮建议：

> 第一版不要实现 false 分支，直接固定二者必须为 `true`。

这样 config signature 不会因为无意义的假参数变化而改变。

---

## P2-2：Pass 2 histogram collection 仍然重新计算昂贵 saliency

`gif_mse_collect` 当前被算入：

```python
collecting_statistics == True
```

虽然：

```python
record_saliency(...)
```

最终会 return，不写入 statistics，

但昂贵的 saliency 数值已经在调用它之前被计算了。

例如 attention 中仍然会重新计算：

```text
q64
k64
qk64
key_score

p64
v64
pv64
value_score
```

linear consumer saliency 也会重新计算。

对 Llama3-8B 的 second calibration pass，这是纯浪费。

必须增加：

```python
collecting_saliency
```

只让 Pass 1 / sequential Pass 1 做 saliency。

---

# 3. 修复 P0-1：statistics ↔ histogram 强绑定

## 3.1 目标

对于每一个：

```text
layer_xxx/site_xx/gif_mse_histogram.pt
```

必须明确记录：

```python
"source_statistics_file"
"source_statistics_sha256"
```

并在 state materialization / validation 时重新计算当前 statistics 文件 hash。

要求：

\[
\boxed{
\text{histogram.source\_statistics\_sha256}
=
\text{sha256(current statistics file)}
}
\]

否则：

```python
raise ValueError(...)
```

绝对禁止静默 fallback。

---

# 4. common trajectory 的 source statistics

当：

```yaml
calibration:
  gif_previous_layers_snn: false
```

MSE histogram 必须绑定：

```text
statistics.pt
```

因此每个 site 的 histogram metadata 写入：

```python
"source_statistics_file": "statistics.pt",
"source_statistics_sha256": sha256_file(
    site_dir / "statistics.pt"
),
```

---

# 5. sequential GIF trajectory 的 source statistics

当：

```yaml
calibration:
  gif_previous_layers_snn: true
```

MSE histogram 必须绑定：

```text
gif_statistics.pt
```

因此写入：

```python
"source_statistics_file": "gif_statistics.pt",
"source_statistics_sha256": sha256_file(
    site_dir / "gif_statistics.pt"
),
```

---

# 6. 推荐修改 `gif_mse_integration.py`

当前：

```python
save_histogram_store(
    store,
    site_root,
    metadata,
)
```

建议修改为由 save 层根据每个 site 的实际 source statistics 自动写入 source hash。

例如：

```python
def save_histogram_store(
    store,
    site_root,
    metadata,
    *,
    statistics_name: str,
):
```

然后：

```python
source_path = root / key / statistics_name
if not source_path.exists():
    raise FileNotFoundError(source_path)

site_metadata = {
    **metadata,
    "source_statistics_file": statistics_name,
    "source_statistics_sha256": sha256_file(source_path),
}
```

再：

```python
torch.save(
    store.state_for(key, site_metadata),
    histogram_path,
)
```

这样 source hash 是 per-site，而不是错误地写成一个全局 hash。

---

# 7. common calibration 必须清理 stale histogram

当前 common trajectory 没有像 blockwise GIF 一样显式清理旧：

```text
gif_mse_histogram.pt
```

本轮增加 helper，例如：

```python
def clear_common_gif_mse_histograms(site_root: str | Path) -> None:
    root = Path(site_root)
    for path in root.glob("layer_*/site_*/gif_mse_histogram.pt"):
        path.unlink(missing_ok=True)
```

在：

```python
collect_site_statistics(...)
```

开始新的 MSE run 时：

```python
if gif_mse_refinement_enabled(cfg) and not previous_layers_snn_enabled(cfg, "gif"):
    clear_common_gif_mse_histograms(root)
```

清理必须发生在生成新 histogram 之前。

不要依赖后续覆盖来防 stale artifact。

---

# 8. sequential trajectory stale 清理继续保留

当前：

```python
clear_target_trajectory_artifacts(..., neuron="gif")
```

已经删除：

```text
gif_mse_histogram.pt
```

这个行为保留。

另外新增 test 锁定这一点，避免以后 regression。

---

# 9. `validate_histogram_provenance()` 必须检查 source statistics

修改：

```text
snn2/gif_mse_provenance.py
```

当前 expected 增加：

```python
expected_statistics_name = (
    "gif_statistics.pt"
    if previous_layers_snn_enabled(cfg, "gif")
    else "statistics.pt"
)
```

但 source hash 不能只靠 manifest 推导。

函数建议增加 `histogram_path` 或 `site_directory` 参数：

```python
def validate_histogram_provenance(
    histogram,
    manifest,
    cfg,
    *,
    site_directory: str | Path,
) -> None:
```

然后：

```python
source_file = histogram.get("source_statistics_file")

if source_file != expected_statistics_name:
    raise ValueError(...)

source_path = Path(site_directory) / expected_statistics_name

if not source_path.exists():
    raise FileNotFoundError(source_path)

actual_sha = sha256_file(source_path)

if histogram.get("source_statistics_sha256") != actual_sha:
    raise ValueError(...)
```

---

# 10. 所有 provenance validation 调用点更新

至少检查：

```text
snn2/calibration.py
snn2/gif_mse_state.py
scripts/verify_artifacts.py
snn2/state_validation.py
```

凡是调用：

```python
validate_histogram_provenance(...)
```

都必须提供当前 site directory。

---

# 11. 修复 P0-2：MSE quantizer 必须严格模拟 StaticGIF FP32 arithmetic

当前 `runtime_fake_quant()` 不允许继续把 quantization decision 放在 float64。

新的 calibration helper 必须与：

```text
snn2/neurons.py::StaticGIF._quantize
snn2/neurons.py::StaticGIF._forward_ann_mixed_quant
```

保持一致。

---

# 12. 推荐新的 `runtime_fake_quant()`

实现语义：

```python
def runtime_fake_quant(
    values: torch.Tensor,
    scale: float,
    zero: int,
    *,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError(...)

    if int(zero) != zero or not qmin <= int(zero) <= qmax:
        raise ValueError(...)

    x32 = values.to(torch.float32)
    s32 = torch.tensor(float(scale), dtype=torch.float32, device=x32.device)
    z32 = torch.tensor(float(zero), dtype=torch.float32, device=x32.device)

    q = torch.round(x32 / s32) + z32
    q = torch.clamp(q, qmin, qmax)
    y32 = (q - z32) * s32

    return y32.to(torch.float64)
```

注意：

- quantization arithmetic：FP32；
- return 转成 FP64；
- MSE accumulation 仍在 FP64。

---

# 13. 不要直接复用 STE forward

MSE calibration 是 offline deterministic search。

不要调用：

```text
round_ste
HTGE
autograd.Function
```

因为前向只需要真实：

```python
torch.round(...)
```

目标只是保持：

```text
forward numerical semantics
```

一致。

---

# 14. threshold-boundary regression test

新增一个非常关键的 test。

人为构造：

```text
scale = 某 FP32 值
zero = 某整数
```

activation 放在：

\[
x=(k+0.5)s+\epsilon
\]

附近，其中：

```text
epsilon
```

选足够小，使 FP32 / FP64 rounding 可能不同。

然后比较：

```python
runtime_fake_quant(...)
```

和真正：

```python
StaticGIF
```

所执行的 forward quantization。

要求：

```python
torch.equal(...)
```

或按照 runtime dtype 做严格等价检查。

不要只测随机大范围值。

---

# 15. MSE diagnostics 仍用 FP64 accumulation

虽然 quantization 用 FP32：

```text
x32
scale32
round32
clamp32
dequant32
```

但误差：

\[
(x-y)^2
\]

建议转回 FP64 后累加：

```python
error64 = values.double() - reconstructed.double()
```

Histogram center 本身仍可存 FP64。

---

# 16. persisted-scale consistency

当前实现已经把 final scale cast 到：

```python
float32
```

后重新评估。

这个行为继续保留。

完整 candidate evaluation 应确保：

```text
候选 scale 最终以 FP32 表达后
仍然按 FP32 runtime quantizer重新评估
```

不要只评价 float64 candidate 再直接存 float32。

---

# 17. 修复 P1-1：把 MSE validation 接进核心 state bundle validation

当前：

```python
validate_gif_mse_state(...)
```

只在 `verify_artifacts.py` 中显式使用。

本轮要求：

```text
training
evaluation
conversion
verify_artifacts
```

只要经过：

```python
validate_site_state_bundle(..., cfg=cfg)
```

就必须自动验证 MSE state。

---

# 18. 修改 `snn2/state_validation.py`

在加载：

```python
gif_state.pt
```

后调用：

```python
from .gif_mse_validation import validate_gif_mse_state
```

例如：

```python
gif_state_path = directory / "gif_state.pt"

validate_gif_mse_state(
    states["gif"],
    cfg,
    path=gif_state_path,
)
```

但只在：

```python
cfg is not None
```

时调用。

如果 `cfg is None` 的老测试/通用 schema validation 需要继续工作，不要强行推断 MSE config。

---

# 19. 避免 circular import

如果：

```text
state_validation.py
→ gif_mse_validation.py
→ state_validation.py
```

产生 circular import，优先调整模块依赖。

推荐：

```text
gif_mse_validation.py
```

只依赖：

```text
config.py
artifacts.py（如需要 sha）
torch
pathlib
```

不要 import `state_validation.py`。

如果仍存在循环，可在函数内部局部 import：

```python
from .gif_mse_validation import validate_gif_mse_state
```

---

# 20. training 必须因此自动阻止坏 MSE state

当前：

```text
snn2/training.py
```

已经调用：

```python
validate_site_state_bundle(
    layout.ann_training_site_dir,
    cfg=cfg,
    clip_policy="forbid_all",
)
```

因此只要把 MSE validation 接入 `validate_site_state_bundle()`，training 自动获得保护。

不需要再在 `training.py` 复制一套验证。

---

# 21. evaluation / conversion 同理

检查所有现有：

```python
validate_site_state_bundle(..., cfg=cfg)
```

调用。

目标是通过核心 bundle validator 统一保证。

不要在多个上层脚本里复制同样的 MSE-specific validation。

---

# 22. `verify_artifacts.py` 可保留额外显式检查

即使核心 validator 已经接入：

```text
scripts/verify_artifacts.py
```

仍可以继续显式调用：

```python
validate_gif_mse_state(...)
```

作为更清晰的 artifact verifier。

但不要产生两套不一致判断。

---

# 23. 修复 P1-2：legacy direct artifact compatibility

本轮明确选择：

> **保留旧 direct-min-max artifact 可复用。**

因此当：

```yaml
gif:
  mse_scale_refinement: false
```

时：

旧 manifest 缺少新 MSE 字段不应直接失败。

---

# 24. Legacy direct metadata 解释规则

对于：

```text
mse_scale_refinement=false
```

以下新字段缺失时：

```text
gif_scale_initialization
gif_mse_scale_refinement
gif_qparam_calibration_method
gif_mse_refinement_signature
gif_mse_refinement_config
```

解释为：

```python
gif_scale_initialization = "direct_min_max"
gif_mse_scale_refinement = False
gif_qparam_calibration_method = "direct_min_max"
gif_mse_refinement_signature = None
gif_mse_refinement_config = None
```

---

# 25. 但 conflicting metadata 必须报错

如果 legacy/direct artifact 中字段存在，例如：

```python
"gif_mse_scale_refinement": True
```

但当前 config：

```yaml
mse_scale_refinement: false
```

必须 raise。

如果：

```python
"gif_qparam_calibration_method": "offline_static_mse"
```

也必须 raise。

即：

> missing 可以兼容，conflict 不能兼容。

---

# 26. 修改 `verify_artifacts.py`

当前 `_verify_grouped_calibration()` 不要再简单：

```python
metadata.get(key) != expected
```

检查 direct-only MSE 新字段。

改成 helper，例如：

```python
def _validate_gif_qparam_manifest_compatibility(
    manifest,
    cfg,
    *,
    context,
):
```

逻辑：

### MSE enabled

要求所有新字段完整存在、严格匹配。

### MSE disabled

允许字段：

- 缺失；
- 或存在且严格等于 direct 默认值。

---

# 27. `state_validation.load_calibration_manifest()` 同样考虑 legacy direct

如果后续在该函数中加入 MSE manifest validation，也必须使用相同兼容规则。

不要让：

```text
verify_artifacts 可以
但 training 不可以
```

或者反过来。

建议把 compatibility helper 放到一个共用模块，例如：

```text
snn2/gif_mse_validation.py
```

---

# 28. Legacy direct state 本身也要兼容

旧 `gif_state.pt` 可能有：

```python
"mse_refinement": False
```

但没有：

```text
qparam_calibration_method
mse_refinement_version
mse_objective
```

当 `mse=false` 时这是合法的。

`validate_gif_mse_state()` 当前对 disabled 分支只要：

```python
mse_refinement != True
```

即可，这一点可以继续。

---

# 29. MSE enabled 时绝不允许 legacy fallback

当：

```yaml
mse_scale_refinement: true
```

则：

```text
gif_state.pt
gif_mse_histogram.pt
manifest
```

必须完整是新 schema。

不能因为 direct legacy compatibility 而允许缺字段。

---

# 30. 修复 P2-1：固定 `preserve_zero=true`

当前 first-version static MSE 算法就是：

```text
candidate interval 必须包含 0
```

因此配置必须显式规定：

```yaml
preserve_zero: true
```

如果为：

```yaml
preserve_zero: false
```

`validate_config()` 直接报错：

```python
raise ValueError(
    "gif.mse_refinement.preserve_zero is fixed and must be true"
)
```

---

# 31. 固定 `fallback_to_direct_min_max=true`

同理：

```yaml
fallback_to_direct_min_max: true
```

是当前方法不可变原则。

如果 false：

```python
raise ValueError(
    "gif.mse_refinement.fallback_to_direct_min_max is fixed and must be true"
)
```

---

# 32. 为什么保留字段但固定 true

不要删除这两个字段。

原因：

- artifact provenance 更清晰；
- 后续若真正实现新版本算法，可以升级：
  ```text
  static_mse_v2
  ```
- 当前 `static_mse_v1` 的语义因此完全确定。

---

# 33. signature 行为

这两个字段继续进入 normalized config/signature。

但由于 validation 强制为 true：

```text
不会再出现 false 导致 path signature 改变、
但实际算法不变的伪实验。
```

---

# 34. 修复 P2-2：Pass 2 禁止重新算 saliency

核心思路：

把：

```text
collecting activation/statistics
```

和：

```text
collecting saliency
```

拆开。

---

# 35. 修改 `SiteController`

新增：

```python
@property
def collecting_saliency(self) -> bool:
    return self.mode in {"collect", "calibration_collect"}
```

或者等价实现。

要求：

| controller mode | activation stats | saliency | MSE histogram |
|---|---:|---:|---:|
| `collect` | yes | yes | no |
| `calibration_collect` | yes | yes | no |
| `gif_mse_collect` | no normal stats | **no** | yes |
| `gif` | no | no | no |
| `deploy_gif` | no | no | no |

---

# 36. `record_saliency()` 继续保留防御式 early return

当前：

```python
if self.mode == "gif_mse_collect":
    return
```

可以保留。

但它不能作为唯一优化。

真正需要避免的是：

> 不要在调用 `record_saliency()` 之前计算 saliency tensor。

---

# 37. 修改 `model_integration.py`

所有昂贵 saliency computation 条件从：

```python
if controller.collecting_statistics:
```

或等价 fallback：

```python
if getattr(controller, "collecting_statistics", ...):
```

改成：

```python
if getattr(controller, "collecting_saliency", False):
```

仅限 **saliency 计算部分**。

不要误改 activation histogram collection。

---

# 38. Attention saliency

以下 Pass 2 必须跳过：

```text
q64
k64
qk64
key_score
```

和：

```text
p64
v64
pv64
value_score
```

即只有：

```python
controller.collecting_saliency
```

时才计算。

---

# 39. Linear consumer saliency

以下 hook：

```text
q_proj / k_proj / v_proj consumer score
gate_proj / up_proj consumer score
o_proj score
down_proj score
```

都必须改为只在：

```python
controller.collecting_saliency
```

时调用：

```python
_linear_score(...)
```

否则 `gif_mse_collect` Pass 2 会白算大矩阵乘。

---

# 40. `temporal_model.py` 同样修

sequential GIF histogram Pass 2 经过 temporal attention。

必须把 temporal attention 中的：

```text
key_score
value_score
```

计算条件改成：

```python
collecting_saliency
```

但以下 activation recording 仍然保留：

```text
Site 3
Site 4
Site 5
Site 6
```

因为 histogram 仍然需要 activation。

---

# 41. 不要影响 Pass 1 saliency

必须保证：

```text
collect
calibration_collect
```

仍然和修改前一样生成完整 saliency。

本轮优化只让：

```text
gif_mse_collect
```

不再重复做 saliency。

---

# 42. 需要新增的测试：source statistics hash

新增 test：

```python
test_histogram_binds_current_source_statistics_hash()
```

步骤：

1. 写：
   ```text
   statistics.pt
   ```
2. 保存 histogram；
3. validate → pass；
4. 修改/覆盖 `statistics.pt`；
5. validate → 必须 fail。

分别覆盖：

```text
ann_common → statistics.pt
sequential_temporal_gif → gif_statistics.pt
```

---

# 43. stale common histogram test

新增：

```python
test_common_calibration_clears_stale_gif_histogram()
```

预先创建：

```text
layer_000/site_02/gif_mse_histogram.pt
```

内容写入 sentinel。

重新启动 common MSE calibration。

要求：

- stale file 在新 histogram 写入前被清理；
- 不可能被 materialization 误读。

---

# 44. sequential stale cleanup test

当前已有逻辑，但新增/扩展 test：

```python
clear_target_trajectory_artifacts(root, "gif")
```

后：

```text
gif_statistics.pt    removed
gif_state.pt         removed
gif_mse_histogram.pt removed
```

---

# 45. runtime arithmetic equivalence test

新增：

```python
test_mse_fake_quant_matches_staticgif_fp32_forward()
```

至少测试：

- normal random values；
- quantization bin midpoint；
- midpoint ± tiny epsilon；
- positive-only；
- negative-only；
- qmax 15；
- qmax 30。

目标：

```text
calibration evaluator output
==
runtime StaticGIF forward output
```

按 runtime dtype 精确比较。

---

# 46. candidate ranking regression test

构造一个例子，使：

```text
FP64 quantizer candidate A 看起来更好
但 FP32 runtime candidate B 才更好
```

不一定必须非常复杂。

至少保证 optimizer 使用新 FP32 evaluator，而不是旧 FP64 evaluator。

---

# 47. core bundle validation test

新增：

```python
test_validate_site_state_bundle_rejects_invalid_mse_state()
```

构造合法 bundle 后篡改：

```text
mse_refinement_signature
```

或者：

```text
refined_mse > baseline_mse
```

然后：

```python
validate_site_state_bundle(..., cfg=cfg)
```

必须直接失败。

不允许必须依赖：

```text
verify_artifacts.py
```

才能发现。

---

# 48. legacy direct artifact compatibility test

构造旧 direct manifest：

不含：

```text
gif_scale_initialization
gif_mse_scale_refinement
gif_qparam_calibration_method
gif_mse_refinement_signature
gif_mse_refinement_config
```

在：

```yaml
mse_scale_refinement: false
```

下：

```text
verify/group validation
```

应通过。

---

# 49. legacy conflict test

同样旧/direct path，但 manifest 人为加入：

```python
"gif_mse_scale_refinement": True
```

或者：

```python
"gif_qparam_calibration_method": "offline_static_mse"
```

在 `mse=false` config 下必须 fail。

---

# 50. MSE enabled legacy rejection test

当：

```yaml
mse_scale_refinement: true
```

但 manifest 缺：

```text
gif_mse_refinement_signature
```

必须 fail。

---

# 51. 固定配置项 test

新增：

```python
cfg["gif"]["mse_refinement"]["preserve_zero"] = False
```

必须：

```python
validate_config(cfg)
```

fail。

以及：

```python
fallback_to_direct_min_max = False
```

也必须 fail。

---

# 52. Pass 2 no-saliency regression test

给 toy controller/model 增加 counters：

```text
activation_updates
histogram_updates
saliency_computations
```

要求：

### Pass 1

```text
saliency_computations > 0
```

### MSE Pass 2

```text
histogram_updates > 0
saliency_computations == 0
```

注意测试的是：

> 不只是 `record_saliency()` 没有写入。

而是 `_linear_score`、QK/PV saliency 本身没有执行。

可通过 monkeypatch `_linear_score` 为计数器/抛错函数实现。

---

# 53. common two-pass sample order test继续保留

当前已有：

```text
mse=false → [11,22]
mse=true  → [11,22,11,22]
```

这项测试保留。

---

# 54. sequential layer order test继续保留并扩展

要求严格顺序：

```text
Pass 1 statistics
→ source statistics saved
→ Pass 2 histogram
→ source statistics hash recorded
→ state materialization
→ deployment
→ cached activation propagation
```

特别验证：

```text
source_statistics_sha256
```

确实对应 deployment 前保存的当前 layer `gif_statistics.pt`。

---

# 55. 推荐修改文件范围

至少检查/修改：

```text
snn2/gif_mse_calibration.py
snn2/gif_mse_integration.py
snn2/gif_mse_provenance.py
snn2/gif_mse_validation.py
snn2/gif_mse_state.py
snn2/calibration.py
snn2/blockwise_calibration.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
snn2/state_validation.py
snn2/config.py
scripts/verify_artifacts.py
```

测试至少涉及：

```text
tests/test_gif_mse_refinement.py
tests/test_gif_mse_state.py
tests/test_blockwise_calibration_runtime.py
tests/test_verify_artifacts.py
```

如结构更清晰，可新增：

```text
tests/test_gif_mse_provenance.py
tests/test_gif_mse_runtime_equivalence.py
```

---

# 56. 不要修改的文件/逻辑范围

除非测试确实需要，不要动：

```text
Phase neuron math
MTN neuron math
GIF temporal integer decomposition
saliency mask selection rule
site topology
common Clip math
prefix discovery
rotation
training optimizer
LR scheduler
warmup
training sample count
lm-eval task definitions
```

---

# 57. `gif_state.pt` schema 不需要大改

本轮不需要再改变 final qparam key：

```text
low_scale
low_zero
high_scale
high_zero
```

也不需要把 source statistics hash写进 runtime state。

source hash应主要存在：

```text
gif_mse_histogram.pt
```

以及 manifest/summary 如有需要。

Runtime module不应读取：

```text
statistics hash
histogram
MSE config
alpha
```

---

# 58. histogram schema 更新建议

建议：

```python
{
    "format_version": 2,
    "refinement_version": "static_mse_v1",

    "configured_group_size": ...,
    "effective_group_size": ...,
    "parameter_layout": ...,

    "histogram_bins": ...,

    "trajectory_source": ...,
    "previous_layers_snn": ...,

    "source_statistics_file": "statistics.pt" or "gif_statistics.pt",
    "source_statistics_sha256": "...",

    "mse_refinement_signature": "...",

    ...
}
```

如果将：

```text
format_version
```

从 1 升为 2，则 MSE enabled 模式必须拒绝旧 version 1 histogram。

这是合理的，因为 version 1 不包含 source statistics hash。

---

# 59. direct artifact不受 histogram format version影响

当：

```yaml
mse_scale_refinement: false
```

根本不要求：

```text
gif_mse_histogram.pt
```

因此 histogram schema升级不会影响 legacy direct artifacts。

---

# 60. MSE path suffix不需要再改

本轮不改变：

```text
gif_qparams_mse_refined_v1_<signature8>
```

因为算法仍然是：

```text
static_mse_v1
```

这次修的是 correctness bug，不是方法定义变化。

但注意：

如果你认为 FP32 evaluator修复属于改变方法数学版本，也可以将：

```text
version
```

升级为：

```text
static_mse_v1_1
```

不过我不建议。

原因是文档原本就要求：

> evaluator 必须与 runtime一致。

当前 FP64 是实现 bug，不应被当成新方法。

---

# 61. MSE signature 是否应包含 histogram format version

建议 MSE signature继续只来自：

```text
gif.mse_refinement config
```

不要把代码内部 schema version塞进 config signature。

Artifact schema mismatch由：

```text
histogram.format_version
```

和 verifier 管理。

---

# 62. numerical tolerance

继续使用：

```text
1e-12
```

做 FP64 MSE比较可以。

但注意：

candidate 的 reconstructed output来自 FP32 runtime arithmetic。

因此：

\[
MSE
\]

本身是 FP64 accumulation后的值。

---

# 63. `optimized_lower/upper` 与 representable range

保留两套 diagnostics：

```text
optimized_lower
optimized_upper
```

表示 search requested interval；

```text
representable_lower
representable_upper
```

表示实际 FP32 persisted `(scale,zero)` 对应的量化区间。

clipping ratio继续按：

```text
representable range
```

计算。

---

# 64. training provenance 可额外记录 MSE mode

当前 training artifact provenance已经绑定 calibration manifest hash，因此理论上足够。

但建议为了可读性额外记录：

```python
"ann_training_gif_mse_scale_refinement": ...
"ann_training_gif_mse_refinement_signature": ...
```

不是 correctness 必需项，但有利于：

```text
training_result.json
```

直接查看。

如果加入，validation/replay也要同步。

---

# 65. 推荐实施顺序

## Step 1 — source statistics hash

先完成：

```text
histogram format v2
source_statistics_file
source_statistics_sha256
stale cleanup
provenance validation
```

运行 provenance tests。

---

## Step 2 — FP32 runtime-equivalent evaluator

修改：

```text
runtime_fake_quant
evaluate_histogram_mse
optimize_static_qparams
```

运行：

```text
runtime equivalence tests
synthetic MSE tests
```

---

## Step 3 — core MSE state validation

把：

```text
validate_gif_mse_state
```

接入：

```text
validate_site_state_bundle
```

运行 training/conversion/evaluation state tests。

---

## Step 4 — legacy direct compatibility

修改：

```text
verify_artifacts
manifest compatibility helper
```

运行：

```text
legacy direct pass
legacy conflict fail
MSE legacy fail
```

---

## Step 5 — fixed config semantics

强制：

```text
preserve_zero=true
fallback_to_direct_min_max=true
```

更新 config tests。

---

## Step 6 — Pass 2 performance

新增：

```text
collecting_saliency
```

修改：

```text
model_integration.py
temporal_model.py
```

运行：

```text
Pass1 saliency preserved
Pass2 saliency skipped
```

---

## Step 7 — full regression

最后：

```bash
pytest -q
```

必须全部通过。

---

# 66. 建议 smoke test

完整 pytest 后，再进行一个非常小的真实模型 calibration smoke test。

不要直接上完整 8B training。

例如使用当前可运行的最小配置：

```text
num_samples = 很小测试值
group_size = 当前合法值
mse_scale_refinement = true
```

验证生成：

```text
statistics.pt
gif_mse_histogram.pt
gif_state.pt
calibration_state_manifest.json
```

---

# 67. smoke test 重点检查

随机选：

```text
layer_000/site_02
layer_000/site_01
layer_000/site_07
layer_000/site_10
```

检查：

```python
hist["source_statistics_file"]
hist["source_statistics_sha256"]

state["mse_refinement"]
state["qparam_calibration_method"]
state["direct_low_scale"]
state["low_scale"]
state["mse_diagnostics"]
```

并确认：

```text
sha256(current statistics file)
==
histogram source_statistics_sha256
```

---

# 68. smoke test runtime equivalence

随机抽一个 site/group：

读取：

```text
gif_state.pt
```

对同一小批 activation：

1. 用 calibration evaluator量化；
2. 用真正 StaticGIF量化；

要求输出一致。

---

# 69. 不允许为了 smoke test 改正式实验 config

测试参数应使用临时 config / override。

不要把：

```text
num_samples很小
```

提交进正式：

```text
configs/experiment_matrix.yaml
```

---

# 70. 本轮完成标准

只有以下全部满足才算完成。

## Correctness

- [ ] MSE histogram绑定具体 source statistics SHA256。
- [ ] common trajectory 清理 stale histogram。
- [ ] sequential trajectory继续清理 stale histogram。
- [ ] source statistics变化后旧 histogram必定 validation fail。
- [ ] MSE evaluator quantization arithmetic 与 StaticGIF FP32一致。
- [ ] MSE accumulation仍为 FP64。
- [ ] persisted FP32 scale重新评价。
- [ ] `refined_mse <= baseline_mse + tolerance`。

## Validation

- [ ] `validate_site_state_bundle(..., cfg=cfg)` 自动检查 MSE state。
- [ ] training 不依赖 `verify_artifacts.py` 才能发现坏 MSE artifact。
- [ ] evaluation/conversion 同样共享核心 validation。
- [ ] MSE enabled 不允许 legacy/incomplete state。

## Backward compatibility

- [ ] direct-min-max 旧 path不变。
- [ ] `mse=false` 时旧 direct manifest缺新字段仍可通过。
- [ ] conflicting新字段仍会报错。
- [ ] `mse=true` 时必须完整严格匹配新 schema。

## Config semantics

- [ ] `preserve_zero` 必须为 true。
- [ ] `fallback_to_direct_min_max` 必须为 true。
- [ ] false 不再生成“signature不同但算法相同”的伪实验。

## Performance

- [ ] Pass 1 saliency计算不变。
- [ ] `gif_mse_collect` Pass 2 不再计算 QK/PV saliency。
- [ ] `gif_mse_collect` Pass 2 不再计算 linear consumer saliency。
- [ ] histogram activation collection仍正常。

## Tests

- [ ] source statistics hash tests通过。
- [ ] stale histogram tests通过。
- [ ] FP32 runtime equivalence tests通过。
- [ ] threshold-boundary tests通过。
- [ ] core bundle validation tests通过。
- [ ] legacy direct compatibility tests通过。
- [ ] fixed-config tests通过。
- [ ] no-saliency Pass 2 tests通过。
- [ ] existing group_size 64/128/-1 tests通过。
- [ ] existing two-pass sample order tests通过。
- [ ] existing blockwise trajectory tests通过。
- [ ] `pytest -q` 全部通过。

---

# 71. Codex 完成后必须汇报

修改完成后，请输出：

1. 修改文件列表；
2. histogram schema最终版本；
3. source statistics SHA256如何保存与验证；
4. common / sequential stale cleanup如何实现；
5. MSE evaluator如何与 StaticGIF FP32 forward对齐；
6. `validate_site_state_bundle` 如何接入 MSE validation；
7. legacy direct artifact兼容规则；
8. `preserve_zero` / `fallback_to_direct_min_max` 最终语义；
9. Pass 2 跳过了哪些 saliency计算；
10. 新增/修改测试列表；
11. `pytest -q` 最终结果；
12. smoke test结果。

---

# 72. 最终原则

本轮修改完成后，应满足：

\[
\boxed{
\text{Static MSE calibration}
=
\text{同 trajectory activation}
+
\text{与 runtime 一致的 FP32 quantizer}
+
\text{FP64 objective accumulation}
+
\text{强 provenance}
+
\text{强 validation}
}
\]

并且：

\[
\boxed{
\text{不同 group size / MSE config 不覆盖}
}
\]

同时：

\[
\boxed{
\text{legacy direct-min-max artifacts 在 mse=false 下继续兼容}
}
\]

最后：

> 本轮不允许改变正式实验方法本身，只修复当前 Static MSE refinement 实现中的 correctness、artifact safety、validation 与 second-pass 性能问题。
