# GIF Static MSE QParam Refinement 修改方案

> 仓库：`https://github.com/wangwk699/SNN`
>
> 目标：在当前 static GIF 量化框架中，将 GIF qparams 从单纯 `direct_min_max` 扩展为可选的 **offline MSE-optimal fixed scale/zero refinement**。本文档必须足够让 Codex 在没有其他上下文的情况下直接完成修改。

---

## 1. 不可破坏的核心约束

本轮修改必须严格满足：

1. GIF `scale/zero` 只能在 calibration 阶段离线确定。
2. ANN fine-tuning、ANN evaluation、SNN conversion 运行时 `scale/zero` 全程固定。
3. 禁止 dynamic per-token / per-batch scale/zero。
4. 禁止把 `scale/zero` 变成 trainable parameter。
5. 禁止通过 STE/HTGE 优化 `scale/zero`。
6. StaticGIF runtime forward 数学不变。
7. GIF bit policy 不变：
   - low branch: `qmin=0, qmax=15`
   - high branch: `qmin=0, qmax=30`
8. saliency 规则、site topology、common Clip、Phase/MTN 统计均不改。
9. `calibration.group_size` 不能写死成 128。
10. 本轮不调整 LR、scheduler、warmup、epoch、train_samples、low_ratio 等训练参数。

最终仍然是：

\[
\text{calibration activations}
\rightarrow
\text{offline MSE search}
\rightarrow
(s^*,z^*)
\rightarrow
\text{gif_state.pt}
\rightarrow
\text{fixed runtime qparams}.
\]

---

## 2. `group_size` 必须从配置读取

所有 MSE refinement 的 grouping 都必须读取：

```python
configured_group_size = int(cfg["calibration"]["group_size"])
```

不得出现算法级：

```python
group_size = 128
```

必须继续支持当前语义：

- `group_size=-1`：整个当前 grouping dimension 为一个 group；
- `group_size>0`：按配置值分组。

必须复用当前项目已有的 `_layout_metadata()`、`group_reduce_last_dim()` 和 site layout 语义，不得为了 MSE refinement 改 grouping topology。

---

## 3. 当前 direct-min-max 必须保留为 backward-compatible baseline

当前 `snn2/calibration.py::build_gif_state()` 使用：

```python
_qparams(minimum, maximum, qmin=0, qmax=GIF_LOW_QMAX)
_qparams(minimum, maximum, qmin=0, qmax=GIF_HIGH_QMAX)
```

并保存：

```python
"scale_initialization": "direct_min_max",
"mse_refinement": False,
"original_spikellm_dynamic_quantization": False,
```

修改后要求：

### `gif.mse_scale_refinement=false`

必须与旧逻辑 tensor-for-tensor 等价：

- `low_scale`
- `low_zero`
- `high_scale`
- `high_zero`
- saliency mask
- artifact 路径

都不得因为本轮修改改变。

### `gif.mse_scale_refinement=true`

流程变为：

```text
direct-min-max baseline
→ offline MSE clipping-range search
→ local (scale, zero) refinement
→ select lowest-MSE candidate
→ save final fixed qparams
```

Direct-min-max 必须永远作为候选和 fallback。

---

## 4. 配置设计

保留现有：

```yaml
gif:
  scale_initialization: direct_min_max
  mse_scale_refinement: false
  runtime_quantization: static
```

新增：

```yaml
gif:
  mse_refinement:
    version: static_mse_v1
    objective: elementwise_mse
    histogram_bins: 4096

    coarse_alpha_values:
      - 1.0
      - 0.995
      - 0.99
      - 0.98
      - 0.97
      - 0.95
      - 0.925
      - 0.90
      - 0.875
      - 0.85
      - 0.80
      - 0.75
      - 0.70

    fine_alpha_radius: 0.03
    fine_alpha_step: 0.005
    fine_alpha_min: 0.70
    fine_alpha_max: 1.0

    local_scale_ratio_min: 0.95
    local_scale_ratio_max: 1.05
    local_scale_ratio_step: 0.01
    local_zero_radius: 2

    preserve_zero: true
    fallback_to_direct_min_max: true
```

`resolve_config()` 必须给旧配置补默认值，旧 YAML 不得失效。

`validate_config()` 必须验证：

- `histogram_bins` 为正整数；
- alpha 均 finite 且 `0 < alpha <= 1`；
- coarse list 非空且无重复；
- fine radius/step 为正；
- min/max 合法；
- scale ratio min/max/step 合法；
- `local_zero_radius` 为非负整数；
- `preserve_zero`、`fallback_to_direct_min_max` 为 bool；
- `objective == "elementwise_mse"`；
- `version == "static_mse_v1"`。

---

## 5. Static affine quantization 数学

继续沿用当前 runtime 的 rounding/clamp 语义。

对：

\[
q_{min}=0,\qquad q_{max}\in\{15,30\},
\]

定义：

\[
q(x;s,z)=
\operatorname{clip}
\left(
\operatorname{round}(x/s)+z,
q_{min},q_{max}
\right),
\]

\[
Q(x;s,z)=s(q(x;s,z)-z).
\]

Direct-min-max：

\[
s_0=
\max\left(
\frac{x_{max}-x_{min}}{q_{max}-q_{min}},
10^{-8}
\right),
\]

\[
z_0=
\operatorname{clip}
\left(
\operatorname{round}
\left(q_{min}-\frac{x_{min}}{s_0}\right),
q_{min},q_{max}
\right).
\]

必须复用当前 `_qparams()` 逻辑，不能新写一套不一致的 rounding 规则。

---

## 6. MSE objective

对某一：

```text
layer × site × group × branch
```

的 calibration values：

\[
\mathcal X=\{x_i\}_{i=1}^{N},
\]

优化：

\[
(s^*,z^*)=
\arg\min_{s,z}
\frac{1}{N}
\sum_i
\left[x_i-Q(x_i;s,z)\right]^2.
\]

第一版只允许 **element-wise activation reconstruction MSE**。

不要加入：

- Hessian weighting；
- gradient weighting；
- task loss；
- KL；
- teacher loss；
- downstream weight reconstruction。

---

## 7. 必须采用 two-pass calibration

当前 `StatisticsStore` 只有 min/max、sum、sum_sq、saliency、Phase/MTN EMA 等统计，无法重建任意 candidate qparams 的 MSE。

因此 `mse_scale_refinement=true` 时增加第二遍 calibration forward。

### Pass 1

完全保留当前行为，得到：

- `value_min/value_max`
- saliency score / mask
- Phase / MTN statistics
- trajectory metadata

### Pass 2

使用：

- 同样 calibration samples；
- 同样顺序；
- 同样 prefix；
- 同样 rotation；
- 同样 `*_previous_layers_snn` trajectory；

再次 forward，仅收集 GIF MSE histogram。

Pass 2：

- 不重新决定 saliency mask；
- 不改变 Phase/MTN state；
- 不改变模型；
- 不改变 activation trajectory。

`mse_scale_refinement=false` 时不得多跑第二遍。

---

## 8. 新增独立模块

推荐新增：

```text
snn2/gif_mse_calibration.py
```

职责包括：

```python
normalize_mse_refinement_config(...)
build_gif_branch_mask_spec(...)
build_histogram_spec(...)
GIFMSEHistogramStore
evaluate_histogram_mse(...)
search_clip_range(...)
refine_scale_zero(...)
optimize_static_qparams(...)
```

函数名可调整，但不要把全部搜索逻辑继续堆进 `calibration.py`。

---

## 9. Histogram 而非保存 raw activations

不要持久化全部 activation。

每个：

```text
layer × site × group × branch
```

保存固定 bins，例如 4096。

建议每个 site 保存：

```text
gif_mse_histogram.pt
```

内容至少包含：

```python
{
    "format_version": ...,
    "refinement_version": "static_mse_v1",
    "configured_group_size": ...,
    "effective_group_size": ...,
    "parameter_layout": ...,
    "histogram_bins": 4096,

    "low": {
        "counts": ...,
        "bin_edges": ...,
        "sample_count": ...,
        "sum_sq": ...,
    },

    "high": {
        ...
    },

    "trajectory_source": ...,
}
```

Identity GIF site 不需要 histogram。

Histogram 只服务 calibration，不进入 runtime module buffer。

---

## 10. Low / High 分别优化

Salient GIF site 对 low/high branch 分别求：

\[
(s_{g,low}^*,z_{g,low}^*)
\]

和：

\[
(s_{g,high}^*,z_{g,high}^*).
\]

Low 使用：

```text
qmax=15
```

High 使用：

```text
qmax=30
```

不能混成一个 objective。

All-low site 只优化 low。

若某 group 某 branch 的 `sample_count==0`：

- 不搜索；
- 直接保留 direct-min-max qparams；
- `refinement_applied=false`；
- 不得产生 NaN。

---

## 11. Site 1 / Site 7 multi-role

当前 Site 1/7 是 role-specific mask，但 qparams shared：

```text
Site 1: q / k / v masks
Site 7: gate / up masks
```

本轮必须保持：

> role-specific masks，shared low/high qparams。

不能新增：

```text
low_scale_by_role
high_scale_by_role
```

Histogram 按 runtime use 聚合：

- 某 role 判为 low → 加入共享 low histogram；
- 某 role 判为 high → 加入共享 high histogram。

同一 channel 在不同 role 中可分别贡献 low/high objective，这是允许且正确的。

---

## 12. Branch-specific bounds

Pass 1 已有 per-channel/per-head：

```python
value_min
value_max
```

结合最终 mask，先得到：

\[
x_{min}^{branch},\qquad x_{max}^{branch}.
\]

然后用这些 bounds 建 histogram。

Multi-role 情况下，一个 channel 若在任一 role 中属于 low，则可贡献 low bound；high 同理。

---

## 13. `gif_previous_layers_snn` 两种 trajectory 都必须支持

不得只为当前某个 false 配置写死。

### `gif_previous_layers_snn=false`

GIF state 使用 common：

```text
statistics.pt
```

MSE histogram 也必须来自同一个 common ANN trajectory。

### `gif_previous_layers_snn=true`

GIF state 使用：

```text
gif_statistics.pt
```

MSE histogram 必须来自 block-wise sequential GIF trajectory。

对每层推荐顺序：

```text
1. collect gif statistics
2. derive masks / histogram spec
3. rerun same cached layer inputs, collect histogram only
4. build gif_state.pt
5. deploy current GIF layer
6. propagate cached activations to next layer
```

绝对不能先 deploy 当前层再收 histogram。

Phase/MTN blockwise calibration逻辑不改。

---

## 14. Post-finetuning calibration 同样使用该逻辑

`gif.mse_scale_refinement=true` 必须同时控制：

- `ann_training` Stage A GIF calibration；
- `post_finetuning` Stage A GIF conversion calibration。

Post-finetuning 仍是：

```text
final ANN checkpoint
→ Pass 1
→ Pass 2 histogram
→ MSE-refined fixed qparams
→ conversion
```

---

## 15. Asymmetric clipping-range coarse search

若：

\[
x_{min}<0<x_{max},
\]

搜索：

\[
l=\alpha_l x_{min},\qquad
u=\alpha_u x_{max},
\]

且：

\[
\alpha_l
\]

与：

\[
\alpha_u
\]

必须独立搜索。

若全正：

\[
x_{min}\ge0
\]

则固定：

\[
l=0
\]

只搜索 upper。

若全负：

\[
x_{max}\le0
\]

则固定：

\[
u=0
\]

只搜索 lower。

每个 candidate 通过当前 `_qparams()` 转换成 `(scale,zero)`。

---

## 16. Histogram MSE

Histogram center/count：

\[
(v_k,c_k).
\]

计算：

\[
E(s,z)=
\frac{
\sum_k c_k[v_k-Q(v_k;s,z)]^2
}{
\sum_k c_k
}.
\]

建议：

- search 在 CPU；
- centers / scale search / MSE 使用 float64；
- counts 使用 int64；
- 最终保存 state 时 scale 转回当前使用的 float32。

---

## 17. Fine clipping search

Coarse 最优：

\[
(\alpha_l^c,\alpha_u^c).
\]

围绕它：

```text
±0.03
```

按：

```text
0.005
```

细搜，并 clamp 到：

```text
[0.70, 1.0]
```

边界候选去重。

---

## 18. Local `(scale, zero)` refinement

从 clipping search 最优：

\[
s_c,z_c
\]

继续搜索：

\[
s=\gamma s_c,
\]

其中：

```text
gamma = 0.95 ... 1.05, step=0.01
```

以及：

```text
z = z_c + {-2,-1,0,1,2}
```

并 clamp 到 `[qmin,qmax]`。

这是 offline discrete/local search，不使用梯度。

---

## 19. Direct-min-max 必须永远参与最终比较

最终候选集合必须包含旧 qparams：

```text
direct candidate
+ clipping-search candidates
+ local qparam candidates
```

选择最低 histogram MSE。

必须保证：

\[
E_{final}\le E_{direct}+\epsilon.
\]

若 direct 最优，则合法 fallback：

```python
refinement_applied = False
fallback_reason = "direct_min_max_is_best"
```

Provenance mismatch、非法 histogram 等代码/数据错误不得静默 fallback，应 raise。

---

## 20. Deterministic tie-break

若两个 candidate MSE 在：

```text
abs(delta) <= 1e-12
```

内相同，则优先选择更接近 direct-min-max 的 candidate：

1. covered range 更接近/更大；
2. scale 更接近 direct scale；
3. zero 更接近 direct zero；
4. 最后固定 tuple 顺序。

不要在没有 MSE 收益时无意义偏离旧 qparams。

---

## 21. Diagnostics

实际 representable range：

\[
r_{min}=(q_{min}-z)s,
\]

\[
r_{max}=(q_{max}-z)s.
\]

保存：

\[
r_{clip-low}=P(x<r_{min}),
\]

\[
r_{clip-high}=P(x>r_{max}),
\]

\[
r_{clip-total}=r_{clip-low}+r_{clip-high}.
\]

同时保存：

\[
MSE=
\frac1N\sum_i(x_i-Q(x_i))^2,
\]

和：

\[
NMSE=
\frac{
\sum_i(x_i-Q(x_i))^2
}{
\sum_i x_i^2+10^{-12}
}.
\]

MSE 用作 objective，NMSE 只用于诊断。

---

## 22. `gif_state.pt` metadata

Runtime key 名不改：

```python
low_scale
low_zero
high_scale
high_zero
```

新增：

```python
"scale_initialization": "direct_min_max",
"mse_refinement": True,
"qparam_calibration_method": "offline_static_mse",
"mse_refinement_version": "static_mse_v1",
"mse_objective": "elementwise_mse",
"runtime_quantization": "static",
```

保存 baseline：

```python
direct_low_scale
direct_low_zero
direct_high_scale
direct_high_zero
```

以及 grouped diagnostics，例如：

```python
"mse_diagnostics": {
    "low": {
        "sample_count": ...,
        "baseline_mse": ...,
        "refined_mse": ...,
        "baseline_nmse": ...,
        "refined_nmse": ...,
        "optimized_lower": ...,
        "optimized_upper": ...,
        "representable_lower": ...,
        "representable_upper": ...,
        "clip_ratio_low": ...,
        "clip_ratio_high": ...,
        "refinement_applied": ...,
    },
    "high": {...},
}
```

不要把 histogram counts 塞进 `gif_state.pt`。

---

## 23. `calibration_summary.json` / manifest

每个 site summary 增加：

```text
gif_qparam_calibration_method
gif_mse_refinement_version
gif_histogram_bins
gif_low_mse_reduction_mean
gif_low_mse_reduction_median
gif_low_clip_ratio_mean
gif_high_mse_reduction_mean
gif_high_mse_reduction_median
gif_high_clip_ratio_mean
gif_refined_group_count
gif_fallback_group_count
```

Global provenance 增加：

```python
"gif_scale_initialization": ...
"gif_mse_scale_refinement": ...
"gif_qparam_calibration_method": ...
"gif_mse_refinement_config": ...
"gif_mse_refinement_signature": ...
"calibration_group_size": int(cfg["calibration"]["group_size"])
```

---

## 24. Artifact 路径：group size 不得互相覆盖

当前代码已有：

```text
calibration_group_size_<value>
```

必须保留。

因此：

```text
group_size=128
```

和：

```text
group_size=64
```

必须落入不同目录。

至少覆盖：

- ANN-training calibration；
- post-finetuning calibration；
- aware ANN training output；
- evaluation/conversion aliases。

---

## 25. Direct 与 MSE 也必须隔离

如果 group_size 相同，但：

```text
mse_scale_refinement=false
```

和：

```text
mse_scale_refinement=true
```

也不能覆盖。

新增 stable helper，例如：

```python
gif_qparam_calibration_suffix(cfg)
```

### `mse_scale_refinement=false`

**保持修改前路径完全不变。**

不要新增 `gif_qparams_direct_min_max` 目录，否则历史 artifact 会失效。

### `mse_scale_refinement=true`

新增 suffix：

```text
gif_qparams_mse_refined_v1_<signature8>
```

其中 `<signature8>` 是 normalized `gif.mse_refinement` config 的稳定 SHA256 前 8 位。

这样 search 参数变化也不会覆盖旧 MSE artifact。

---

## 26. 路径示例

假设：

```yaml
calibration:
  group_size: 128
  num_samples: 128
gif:
  mse_scale_refinement: true
```

ANN-training calibration 示例：

```text
.../
ann_training_calibration/
prefix_enabled_ture/
calibration_group_size_128_num_samples_128/
gif_qparams_mse_refined_v1_<signature8>/
<calibration_trajectory>/
sites/
```

如果 `group_size=64`：

```text
.../
calibration_group_size_64_num_samples_128/
gif_qparams_mse_refined_v1_<signature8>/
...
```

Aware ANN output 也必须包含 MSE suffix，例如：

```text
.../
gif_aware/
epochs_1_num_samples_128_lr5e-06_train_samples_10000_calibration_group_size_128_gif_qparams_mse_refined_v1_<signature8>/
...
```

Direct mode (`false`) 保持旧路径不变。

---

## 27. 需要修改/检查的文件

至少检查：

```text
configs/experiment_matrix.yaml
snn2/config.py
snn2/artifacts.py
snn2/stats.py
snn2/calibration.py
snn2/controller.py
snn2/blockwise_calibration.py
scripts/calibrate_sites.py
scripts/verify_artifacts.py
```

推荐新增：

```text
snn2/gif_mse_calibration.py
tests/test_gif_mse_refinement.py
```

并更新相关已有测试：

```text
tests/test_calibration_gif.py
tests/test_calibration_profiles.py
tests/test_calibration_topology.py
tests/test_generated_configs.py
tests/test_conversion_metadata.py
tests/test_post_finetuning_protocol.py
tests/test_evaluation_paths.py
tests/test_verify_artifacts.py
```

---

## 28. Controller / runtime 修改边界

StaticGIF ANN/SNN runtime 不得改数学。

若为了 Pass 2 需要 collector，可增加 calibration-only hook，例如：

```python
self.gif_mse_collector = None
```

但只有 calibration collection 时调用。

ANN training：

```python
mode="gif"
```

不得更新 histogram、min/max 或 qparams。

---

## 29. Stale artifact 防护

MSE run 开始时必须防止：

```text
new statistics.pt + old gif_mse_histogram.pt
```

混用。

Histogram metadata 至少记录：

```text
configured_group_size
num_samples
previous_layers_snn
trajectory_source
calibration_data_manifest_sha256
prefix_state_sha256
prefix_kv_sha256
rotation_state_sha256
mse_refinement_signature
```

Materialization 时不匹配直接报错。

`gif_previous_layers_snn=true` 时，清理旧 `gif_statistics.pt` 的逻辑也要同步清理对应旧 MSE histogram。

---

## 30. Runtime-equivalent fake quant helper

MSE evaluator 必须与 `snn2/neurons.py` 当前 StaticGIF forward 的 rounding/cast/clamp 完全一致。

可以新增 calibration-only纯函数，但实现前要核对 runtime。

不能出现：

```text
calibration fake quant semantics != runtime fake quant semantics
```

---

## 31. Numerical policy

统一：

- histogram edges/centers：float64；
- MSE/NMSE：float64；
- qparam search：float64；
- counts：int64；
- 最终 scale：按当前 state 习惯保存 float32。

所有 scale/MSE/bounds 必须 `isfinite`。

scale 必须：

```text
> 0
```

zero 必须为整数值且满足：

```text
qmin <= zero <= qmax
```

---

## 32. Config / path / qparam 单元测试

新增 `tests/test_gif_mse_refinement.py`，至少覆盖：

1. **Synthetic outlier**
   - 大量值在 `[-1,1]`
   - 极少数 outlier，例如 10
   - 验证 `refined_mse < direct_mse`

2. **No-outlier / uniform**
   - direct 已接近最优时允许 fallback
   - `refined_mse <= direct_mse + tolerance`

3. **Positive-only**
   - lower=0 语义正确

4. **Negative-only**
   - upper=0 语义正确

5. **Zero representability**
   - `Q(0)==0`

6. **Empty branch**
   - direct fallback
   - 无 NaN

7. **Low/high qmax**
   - low=15
   - high=30

8. **Group size**
   - `128`
   - `64`
   - `-1`
   - shape/group 数正确
   - 证明算法没有写死 128

9. **Multi-role**
   - 同 channel 在不同 role 可分别贡献 low/high histogram
   - qparams仍 shared

10. **Determinism**
    - 相同 histogram/config 两次结果完全一致

---

## 33. Backward compatibility 测试

给固定 synthetic statistics：

```python
cfg["gif"]["mse_scale_refinement"] = False
```

修改后的 `build_gif_state()` 必须与旧 direct-min-max 输出一致：

```text
low_scale
low_zero
high_scale
high_zero
mask
```

并验证：

```text
旧 direct artifact path == 修改后的 direct artifact path
```

---

## 34. Path isolation 测试

至少构造：

```text
A: group_size=128, mse=false
B: group_size=64,  mse=false
C: group_size=128, mse=true, refinement config X
D: group_size=128, mse=true, refinement config Y
```

必须：

```text
path(A) != path(B)
path(A) != path(C)
path(C) != path(D)
```

同时 `path(A)` 必须保持旧 direct path。

对以下都测：

- ann_training calibration path；
- post_finetuning calibration path；
- aware ANN checkpoint path；
- evaluation/conversion path。

---

## 35. Two-pass integration test

至少写轻量 fake/tiny model test，验证：

```text
mse=false → 只执行原 Pass 1
mse=true  → Pass 1 + Pass 2
```

并验证 calibration sample 顺序一致。

`gif_previous_layers_snn=true` 时验证每层严格：

```text
statistics
→ histogram
→ state
→ deployment
→ next layer
```

---

## 36. `verify_artifacts.py` 新检查

MSE-refined state 至少检查：

```text
configured_group_size == cfg calibration.group_size
mse_refinement == true
qparam_calibration_method == offline_static_mse
mse_refinement_version == static_mse_v1
runtime_quantization == static
scale > 0
zero integer-valued
qmin <= zero <= qmax
refined_mse <= baseline_mse + tolerance
diagnostics finite
```

Identity site 不要求 histogram。

All-low site 只要求 low。

Salient site 要求 low/high。

---

## 37. CLI 不变

用户仍使用现有命令：

```bash
CFG=configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml

python scripts/calibrate_sites.py   --config "$CFG"   --stage ann_training   --calibration-phase A
```

当：

```yaml
gif:
  mse_scale_refinement: true
```

内部自动执行 Pass 2。

Post-finetuning 同理。

不要新增一个容易与 YAML 不一致的 `--mse-refinement` CLI 开关。

---

## 38. 第一组实验的变量控制

本轮代码完成后，第一组 Llama3-8B 实验只改变：

```yaml
gif.mse_scale_refinement: false → true
```

其余保持完全一致：

- `calibration.group_size` 保持当前配置值；
- calibration samples 不改；
- low_ratio 不改；
- STE/HTGE 不改；
- LR 不改；
- scheduler 不改；
- warmup 不改；
- epoch 不改；
- train samples 不改。

先比较：

```text
Direct min-max
vs
Offline static MSE qparams
```

---

## 39. 推荐实现顺序

### Step 1
完成 config defaults/validation/signature。

### Step 2
完成 artifact path isolation，确保 direct 旧路径不变。

### Step 3
独立实现 pure histogram MSE search，并用 synthetic tests验证。

### Step 4
实现 histogram collector/grouping/multi-role。

### Step 5
接入 common calibration (`gif_previous_layers_snn=false`)。

### Step 6
接入 blockwise GIF (`gif_previous_layers_snn=true`)。

### Step 7
接入 `build_gif_state()` final qparams/diagnostics。

### Step 8
更新 manifests / provenance / verify_artifacts。

### Step 9
运行：

```bash
pytest -q
```

所有测试必须通过。

---

## 40. 完成标准

### 数学
- [ ] qparams全程 static
- [ ] MSE仅 offline calibration
- [ ] low/high独立优化
- [ ] asymmetric clipping search
- [ ] direct candidate永久保留
- [ ] final MSE不高于 direct
- [ ] runtime quantization数学不变

### Group size
- [ ] 无硬编码 128
- [ ] 始终读取 `calibration.group_size`
- [ ] `-1` 支持
- [ ] 64/128/-1 tests通过

### Trajectory
- [ ] common GIF trajectory正确
- [ ] sequential GIF trajectory正确
- [ ] post-finetuning正确

### Path
- [ ] group_size不同不覆盖
- [ ] direct/MSE不覆盖
- [ ] MSE config变化不覆盖
- [ ] direct旧路径保持
- [ ] ANN checkpoint隔离
- [ ] post-finetuning隔离

### Compatibility
- [ ] `mse=false` 与旧逻辑一致
- [ ] identity site不受影响
- [ ] Phase/MTN不受影响
- [ ] common Clip数学不受影响
- [ ] training/eval命令不变

### Tests
- [ ] 新 MSE tests通过
- [ ] path tests通过
- [ ] blockwise tests通过
- [ ] `pytest -q` 全部通过

---

## 41. Codex 修改完成后必须汇报

最终请给出：

1. 修改文件列表；
2. 实际 MSE search 算法；
3. `group_size` 从配置读取的位置；
4. direct 与 MSE artifact path 示例；
5. `group_size=128` 与 `64` path 示例；
6. `gif_previous_layers_snn=false/true` 两条 histogram trajectory；
7. `gif_state.pt` 新 metadata；
8. backward compatibility 保证；
9. 新增/修改测试；
10. `pytest -q` 结果。

实现过程中可调整函数名或代码拆分，但不得改变本文规定的：

- static qparam 原则；
- MSE 数学；
- group-size 配置来源；
- trajectory 一致性；
- artifact 隔离；
- direct backward compatibility。
