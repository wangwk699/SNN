# SNN Stage-A Calibration Path 去冗余与 Phase Runtime Dependency 修订方案

> 目标仓库：`https://github.com/wangwk699/SNN`  
> 目标分支：`main`  
> 本轮检查基准提交：`aa4be0dc8ec3872115c3fc84e65c2eadcdb99834`
>
> 本文档用于指导 Codex 在没有额外上下文的情况下完成本轮修改。
>
> 本轮**不修改 Phase 数学、不修改 calibration 算法、不修改 Stage-B Clip 公式**，只修正 Stage-A artifact 路径的 runtime dependency 语义，并补齐相应测试。

---

## 1. 本轮问题

当前 `snn2/artifacts.py` 中，ANN-training Stage A 路径为：

```python
return _with_gif_qparam_suffix(
    ...
) / f"phase_base_{format_phase_base(self._training_phase_base)}" \
  / calibration_trajectory_dirname(self._source_cfg)
```

因此当前会出现类似：

```text
.../
calibration_group_size_128_num_samples_128/
phase_base_2/
phase_previous_layers_snn_false_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false/
```

以及：

```text
.../
calibration_group_size_128_num_samples_128/
phase_base_2/
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false_
phase_base_2_T_4/
```

这里前置的：

```text
phase_base_2/
```

是冗余的。

原因是：

- 当 `phase_previous_layers_snn=false` 时，Stage A trajectory 不依赖 `phase.base`；
- 当 `phase_previous_layers_snn=true` 时，`calibration_trajectory_dirname()` 已经在最终 trajectory 目录中记录：
  ```text
  phase_base_<BASE>_T_<T>
  ```
  因此前面再放一层 `phase_base_<BASE>` 没有额外信息。

---

# 2. 本轮核心设计原则

Stage A 的路径只编码：

> **真正会改变 Stage-A activation trajectory / statistics 的参数。**

因此：

```text
calibration_trajectory_dirname()
```

应成为 Stage-A runtime dependency 的**唯一命名来源**。

不要再在它外面无条件增加：

```text
phase_base_<BASE>
```

---

# 3. Phase Stage-A dependency 规则

## 3.1 `phase_previous_layers_snn=false`

配置：

```yaml
calibration:
  phase_previous_layers_snn: false
```

此时 Stage A 采用 common ANN trajectory。

因此 Stage A 不依赖：

```text
phase.base
phase.T
```

不同的：

```text
phase.base
phase.T
```

可以复用同一份 Stage A。

例如：

```text
base=2.0, T=4
base=1.5, T=4
base=1.5, T=6
```

只要其它 Stage-A trajectory dependency 完全一致，就应指向同一个 Stage-A artifact。

---

## 3.2 `phase_previous_layers_snn=true`

配置：

```yaml
calibration:
  phase_previous_layers_snn: true
```

此时每一层 calibration input 都受前层 Temporal Phase 输出影响。

Temporal Phase trajectory 依赖：

```text
phase.base
phase.T
```

因此：

```text
phase.base
phase.T
```

必须属于 Stage-A artifact identity。

修改任意一个，都必须重新运行 Stage A。

例如：

```text
base=2.0, T=4
```

与：

```text
base=1.5, T=4
```

必须使用不同 Stage A。

同样：

```text
base=2.0, T=4
```

与：

```text
base=2.0, T=6
```

也必须使用不同 Stage A。

---

# 4. 注意：这里说的 Phase runtime 参数只包括 `phase.base` 与 `phase.T`

不要把：

```text
phase.surrogate_slope
```

加入 Stage-A trajectory identity。

原因：

- sequential Stage A 使用 Temporal Phase；
- `surrogate_slope` 是 ANN `PhaseSurrogate` 的训练/梯度参数；
- 它不决定 Temporal Phase deployment trajectory；
- Stage-B Phase bound 也不依赖 `surrogate_slope`。

所以本轮 Phase Stage-A runtime identity 仍然严格是：

```text
phase.base
phase.T
```

---

# 5. 修改后的 ANN-training Stage A 路径

当前：

```text
.../
calibration_group_size_128_num_samples_128/
phase_base_<BASE>/
<calibration_trajectory>/
```

修改为：

```text
.../
calibration_group_size_128_num_samples_128/
<calibration_trajectory>/
```

具体而言：

## all previous_layers_snn=false

修改前：

```text
.../
calibration_group_size_128_num_samples_128/
phase_base_2/
phase_previous_layers_snn_false_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false/
```

修改后：

```text
.../
calibration_group_size_128_num_samples_128/
phase_previous_layers_snn_false_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false/
```

此时不同：

```text
phase.base
phase.T
```

使用同一个 Stage A。

---

## `phase_previous_layers_snn=true`

修改前：

```text
.../
calibration_group_size_128_num_samples_128/
phase_base_2/
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false_
phase_base_2_T_4/
```

修改后：

```text
.../
calibration_group_size_128_num_samples_128/
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false_
phase_base_2_T_4/
```

Phase runtime identity 已经完整保留在 trajectory dirname 中。

---

# 6. `snn2/artifacts.py` 必改项

修改：

```python
@property
def ann_training_calibration_dir(self) -> Path:
```

当前尾部：

```python
), self._cfg) \
    / f"phase_base_{format_phase_base(self._training_phase_base)}" \
    / calibration_trajectory_dirname(self._source_cfg)
```

改为：

```python
), self._cfg) \
    / calibration_trajectory_dirname(self._source_cfg)
```

也就是删除：

```python
f"phase_base_{format_phase_base(self._training_phase_base)}"
```

这一层。

---

# 7. Post-finetuning Stage A 同样删除冗余 `phase_base_*`

当前：

```python
@property
def post_finetuning_conversion_calibration_dir(self) -> Path:
```

也存在：

```python
/ f"phase_base_{format_phase_base(self._training_phase_base)}" \
/ calibration_trajectory_dirname(self._source_cfg)
```

应统一改为：

```python
/ calibration_trajectory_dirname(self._source_cfg)
```

原因完全相同。

当：

```text
phase_previous_layers_snn=false
```

时，post-finetuning Stage A 也不应仅因为 deployment `phase.base/T` 不同而重新统计。

当：

```text
phase_previous_layers_snn=true
```

时，trajectory dirname 本身已经编码：

```text
phase_base_<BASE>_T_<T>
```

无需额外外层 `phase_base_*`。

---

# 8. Vanilla-analysis Stage A 同样删除冗余 `phase_base_*`

当前：

```python
@property
def vanilla_analysis_calibration_dir(self) -> Path:
```

也包含：

```python
/ f"phase_base_{format_phase_base(self._training_phase_base)}" \
/ calibration_trajectory_dirname(self._source_cfg, effective=False)
```

这里尤其没有必要。

`effective=False` 时：

```text
effective_previous_layers_snn
```

全部按 false 处理。

Vanilla analysis statistics 本身不依赖 Phase deployment runtime。

因此改为：

```python
/ calibration_trajectory_dirname(
    self._source_cfg,
    effective=False,
)
```

不要保留外层：

```text
phase_base_<BASE>
```

---

# 9. Stage B 路径不要删除 `phase_base/T`

这一点非常重要。

当前：

```python
def clip_profile_dirname(
    phase_base,
    phase_T,
    mtn_T,
):
    return (
        f"phase_base_{...}_T_{...}_"
        f"mtn_T_{...}"
    )
```

以及：

```python
@property
def ann_training_clip_profile_dir(self) -> Path:
```

必须保留。

Stage B 是 runtime-specific Clip profile。

Phase bound：

\[
B_{\mathrm{phase}}
=
\tau\frac{1-b^{-T}}{b-1}
\]

明确依赖：

```text
phase.base
phase.T
```

所以 Stage B profile 必须继续按 runtime 参数区分。

不要因为本轮删除 Stage-A 外层 `phase_base_*`，就把 Stage-B：

```text
phase_base_<BASE>_T_<T>_mtn_T_<T>
```

也删掉。

---

# 10. all previous_layers_snn=false 时的最终目录结构

目标结构：

```text
ann_training_calibration/
└── prefix_enabled_.../
    └── calibration_group_size_128_num_samples_128/
        └── phase_previous_layers_snn_false_
            gif_previous_layers_snn_false_
            mtn_previous_layers_snn_false/
            ├── sites/
            │   ├── ...
            │   └── calibration_state_manifest.json
            │
            ├── config/
            ├── logs/
            │
            └── clip_profiles/
                ├── phase_base_2_T_4_mtn_T_4/
                ├── phase_base_1.5_T_4_mtn_T_4/
                ├── phase_base_1.5_T_6_mtn_T_4/
                └── ...
```

也就是：

```text
一个 Stage A
    │
    ├── Stage B(base=2.0,T=4)
    ├── Stage B(base=1.5,T=4)
    └── Stage B(base=1.5,T=6)
```

---

# 11. `phase_previous_layers_snn=true` 时的最终目录结构

目标结构：

```text
ann_training_calibration/
└── prefix_enabled_.../
    └── calibration_group_size_128_num_samples_128/
        └── phase_previous_layers_snn_true_
            gif_previous_layers_snn_false_
            mtn_previous_layers_snn_false_
            phase_base_2_T_4/
            ├── sites/
            ├── config/
            ├── logs/
            └── clip_profiles/
                └── phase_base_2_T_4_mtn_T_4/
```

如果改成：

```text
phase.base=1.5
phase.T=4
```

则必须产生另一个 Stage A：

```text
.../
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false_
phase_base_1.5_T_4/
```

然后在它下面生成对应 Stage B。

---

# 12. Stage A / Stage B 的重新运行规则

必须满足：

| Phase trajectory | Phase 参数变化 | Stage A | Stage B |
|---|---|---:|---:|
| `phase_previous_layers_snn=false` | `phase.base` 改变 | 不重跑 | 重跑 |
| `phase_previous_layers_snn=false` | `phase.T` 改变 | 不重跑 | 重跑 |
| `phase_previous_layers_snn=true` | `phase.base` 改变 | **重跑** | **重跑** |
| `phase_previous_layers_snn=true` | `phase.T` 改变 | **重跑** | **重跑** |

---

# 13. 一般化 dependency 原则

本轮不要硬编码成“只有 Phase 特殊”。

继续以当前：

```python
calibration_trajectory_config()
```

和：

```python
calibration_trajectory_dirname()
```

作为统一 dependency 规则。

一般原则：

```text
某 neuron 的 previous_layers_snn=false
    → 该 neuron 的 trajectory runtime 参数
      不属于 Stage-A identity

某 neuron 的 previous_layers_snn=true
    → 该 neuron 的 trajectory runtime 参数
      属于 Stage-A identity
```

当前代码已经实现：

### Phase active

Stage-A identity 包含：

```text
phase.base
phase.T
```

### GIF active

Stage-A identity 包含：

```text
gif.temporal_steps
```

### MTN active

Stage-A identity 包含：

```text
mtn.T
mtn.K
mtn.threshold_factor
```

本轮不要改变这些 dependency 定义。

---

# 14. 一个重要示例：Phase sequential、MTN common

例如：

```yaml
calibration:
  phase_previous_layers_snn: true
  gif_previous_layers_snn: false
  mtn_previous_layers_snn: false
```

此时 Stage A 路径：

```text
.../
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false_
phase_base_2_T_4/
```

如果只把：

```yaml
mtn:
  T: 8
```

由于：

```text
mtn_previous_layers_snn=false
```

Stage A 不应变化。

只需要重新运行 Stage B。

Stage B 从：

```text
phase_base_2_T_4_mtn_T_4
```

变成：

```text
phase_base_2_T_4_mtn_T_8
```

这正是本轮目标 dependency 语义。

---

# 15. Stage A 与 Stage B 的 Phase 一致性约束

当：

```text
phase_previous_layers_snn=true
```

时：

```text
Stage-A phase.base == Stage-B phase.base
Stage-A phase.T    == Stage-B phase.T
```

必须保持一致。

现有 sequential Phase provenance guard 已经检查：

```text
calibration_trajectory.phase.phase_base
calibration_trajectory.phase.phase_T
```

与 deployment runtime 是否一致。

不要删除或放宽这个 guard。

因此：

```text
Stage A(base=2,T=4)
+
Stage B(base=1.5,T=4)
```

在 `phase_previous_layers_snn=true` 下必须拒绝。

---

# 16. 当 `phase_previous_layers_snn=false` 时不要新增错误限制

此时必须允许：

```text
同一 Stage A
+
Stage B(base=2,T=4)
+
Stage B(base=1.5,T=4)
+
Stage B(base=1.5,T=6)
```

不要新增：

```text
Stage-A source config phase.base
==
Stage-B phase.base
```

这种约束。

Stage A manifest 中已有：

```text
stage_a_parameter_independence
```

应继续明确包含：

```text
phase.T
phase.base
```

---

# 17. 不要修改 ANN training run identity

本轮只删除 **Stage-A calibration artifact path** 中多余的外层：

```text
phase_base_<BASE>
```

不要修改 aware ANN training 的：

```text
phase_training_dirname(...)
gif_training_dirname(...)
```

例如 `phase_aware` ANN training 本身仍然需要保留：

```text
phase_base_<BASE>_T_<T>_...
```

因为 ANN fine-tuning graph / Clip profile / surrogate runtime 的确依赖这些训练参数。

---

# 18. 不要修改 SNN deployment output identity

以下保持不变：

```python
phase_snn_dirname(
    phase_base,
    phase_T,
)
```

SNN deployment output 仍然必须区分：

```text
phase_base_2_T_4
phase_base_1.5_T_4
phase_base_1.5_T_6
```

本轮只修改 Stage-A calibration root。

---

# 19. 不要修改 `ArtifactLayout` source/deployment freeze 机制

以下设计保持：

```python
self._source_cfg = copy.deepcopy(cfg)
self._training_phase_base = ...
self._training_phase_T = ...
self._training_mtn_T = ...
```

deployment override 仍然必须发生在：

```text
ArtifactLayout
```

创建之后。

不要因为 Stage-A path 去掉外层 `phase_base_*` 而改变 source/deployment separation。

---

# 20. 推荐代码改动

## `snn2/artifacts.py`

修改三个 property：

```text
ann_training_calibration_dir
vanilla_analysis_calibration_dir
post_finetuning_conversion_calibration_dir
```

统一删除：

```python
/ f"phase_base_{format_phase_base(self._training_phase_base)}"
```

保留：

```python
/ calibration_trajectory_dirname(...)
```

---

# 21. Stage-A path 的唯一 runtime dependency 来源

不需要额外新增重复 helper。

保持：

```python
calibration_trajectory_dirname()
```

作为 Stage-A path 的唯一 runtime dependency 编码函数即可。

今后不要在 Stage-A root property 外面再次手动追加：

```text
phase_base
phase_T
mtn_T
mtn_K
...
```

---

# 22. 必须新增路径测试

建议在现有 artifact/path tests 中新增以下覆盖。

## Test 1：all false 时 Stage A 对 Phase runtime 独立

构造两个 cfg：

```text
cfg1:
  phase.base=2.0
  phase.T=4
  all previous_layers_snn=false

cfg2:
  phase.base=1.5
  phase.T=6
  all previous_layers_snn=false
```

断言：

```python
layout1.ann_training_calibration_dir \
    == layout2.ann_training_calibration_dir
```

以及：

```python
layout1.ann_training_site_dir \
    == layout2.ann_training_site_dir
```

---

# 23. Test 2：all false 时 Stage B 仍然区分 runtime

同样两个 cfg：

```text
base=2,T=4
base=1.5,T=6
```

断言：

```python
layout1.ann_training_clip_profile_dir \
    != layout2.ann_training_clip_profile_dir
```

并检查目录名分别包含：

```text
phase_base_2_T_4
phase_base_1.5_T_6
```

---

# 24. Test 3：Phase sequential 时 Stage A 区分 Phase runtime

构造：

```text
phase_previous_layers_snn=true
```

并分别：

```text
base=2,T=4
base=1.5,T=4
base=2,T=6
```

必须满足：

```python
layout_2_4.ann_training_calibration_dir \
    != layout_1_5_4.ann_training_calibration_dir

layout_2_4.ann_training_calibration_dir \
    != layout_2_6.ann_training_calibration_dir
```

---

# 25. Test 4：Phase sequential path 不再重复 base

对：

```text
phase_previous_layers_snn=true
phase.base=2
phase.T=4
```

检查 path parts。

应该只存在 trajectory component：

```text
...phase_base_2_T_4
```

不应再存在一个独立 parent component：

```text
phase_base_2
```

即禁止旧结构：

```text
/.../phase_base_2/...phase_base_2_T_4/
```

---

# 26. Test 5：Post-finetuning path 同样遵循 dependency

至少验证：

### common Phase trajectory

```text
phase_previous_layers_snn=false
base=2,T=4
```

和：

```text
phase_previous_layers_snn=false
base=1.5,T=6
```

其：

```python
post_finetuning_conversion_calibration_dir
```

一致。

### sequential Phase trajectory

`phase_previous_layers_snn=true` 时，不同 base/T 的：

```python
post_finetuning_conversion_calibration_dir
```

不同。

---

# 27. Test 6：Vanilla-analysis path 不依赖 Phase runtime

由于 vanilla analysis 使用：

```python
calibration_trajectory_dirname(
    cfg,
    effective=False,
)
```

不同：

```text
phase.base
phase.T
```

不应改变：

```python
vanilla_analysis_calibration_dir
```

---

# 28. 建议增加 Stage A → Stage B integration test

对 all-false 配置：

1. 用：
   ```text
   base=2,T=4
   ```
   materialize 一次 Stage A；

2. 构造：
   ```text
   base=1.5,T=4
   ```
   的 cfg / layout；

3. 验证新 cfg 的：
   ```python
   ann_training_site_dir
   ```
   与旧 Stage A 完全相同；

4. 直接调用：
   ```python
   materialize_clip_profile(...)
   ```

5. 必须成功生成：
   ```text
   clip_profiles/phase_base_1.5_T_4_mtn_T_4
   ```

这个测试直接证明本轮核心目标：

> `phase_previous_layers_snn=false` 时，不需要重新 Stage A，可以直接在旧 Stage A 上生成新 Stage B。

---

# 29. 建议保留 sequential fail-closed tests

上一轮已有：

```text
missing trajectory -> fail
previous_layers_snn=false provenance -> fail
base mismatch -> fail
matching runtime -> pass
```

全部保留。

尤其不要因为共享 Stage A 路径修改而放宽：

```text
phase_previous_layers_snn=true
```

下的：

```text
phase.base
phase.T
```

一致性检查。

---

# 30. 旧 artifact 的处理

修改路径后，已有旧 artifact 不会自动出现在新路径。

本轮**不建议实现自动 legacy fallback**，因为：

- 容易产生新旧目录并存；
- 容易造成 provenance ambiguity；
- 后续很难判断实际用了哪一份 Stage A。

推荐显式迁移或重新生成。

---

# 31. 已有 all-false Stage A 可以直接迁移，不需要重新计算

例如旧路径：

```text
.../
calibration_group_size_128_num_samples_128/
phase_base_2/
phase_previous_layers_snn_false_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false/
```

可直接移动到：

```text
.../
calibration_group_size_128_num_samples_128/
phase_previous_layers_snn_false_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false/
```

由于 all-false Stage A 本身不依赖 Phase runtime，这种迁移是语义安全的。

迁移后，不同 base/T 可以直接运行 Stage B。

---

# 32. 已有 sequential Stage A 也可做纯路径迁移

例如旧路径：

```text
.../
phase_base_2/
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false_
phase_base_2_T_4/
```

可移动到：

```text
.../
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_false_
phase_base_2_T_4/
```

这只是删除冗余 parent component。

Stage A 数值本身无需重新计算。

但：

```text
base=2,T=4
```

的 sequential Stage A 仍然不能拿去服务：

```text
base=1.5,T=4
```

---

# 33. 不要自动合并冲突 artifact

如果新目标目录已经存在：

```text
<new_stage_a_root>
```

不要自动覆盖。

迁移脚本如果存在，应：

```text
destination exists -> fail
```

让用户手动检查。

本轮代码本身无需提供迁移脚本；文档说明即可。

---

# 34. 检查所有依赖路径的 provenance

修改后重点检查：

```text
training_result.json
conversion_metadata.json
clip_profile_manifest.json
calibration_state_manifest.json
verify_artifacts.py
```

确保所有 path / SHA provenance 都使用新的 `ArtifactLayout` 结果。

不要保留硬编码旧：

```text
/.../phase_base_<BASE>/<trajectory>/
```

路径。

---

# 35. 推荐全仓库搜索

修改后执行：

```bash
grep -R 'phase_base_.*calibration' -n snn2 scripts tests
grep -R 'ann_training_calibration_dir' -n snn2 scripts tests
grep -R 'post_finetuning_conversion_calibration_dir' -n snn2 scripts tests
grep -R 'vanilla_analysis_calibration_dir' -n snn2 scripts tests
grep -R 'calibration_trajectory_dirname' -n snn2 scripts tests
grep -R 'clip_profile_dirname' -n snn2 scripts tests
```

重点寻找是否还有人为追加：

```text
phase_base_<BASE>
```

到 Stage-A root 的地方。

---

# 36. 不要修改的代码/语义

本轮不要修改：

```text
snn2/phase_math.py
PhaseSurrogate
Temporal Phase
Phase v0
Phase amplitude
tau calibration
Stage-B PhaseBound
build_clip_state()
clip_profile_dirname()
phase_snn_dirname()
mtn_snn_dirname()
ANN run path
source/deployment override separation
sequential Phase fail-closed provenance guard
```

---

# 37. 最终期望行为矩阵

| previous-layers 配置 | 参数变化 | Stage-A path | Stage A 重跑 | Stage B 重跑 |
|---|---|---|---:|---:|
| Phase=false, GIF=false, MTN=false | Phase base 改变 | 不变 | 否 | 是 |
| Phase=false, GIF=false, MTN=false | Phase T 改变 | 不变 | 否 | 是 |
| Phase=true, GIF=false, MTN=false | Phase base 改变 | 改变 | 是 | 是 |
| Phase=true, GIF=false, MTN=false | Phase T 改变 | 改变 | 是 | 是 |
| Phase=true, GIF=false, MTN=false | MTN T 改变 | 不变 | 否 | 是 |
| Phase=false, GIF=false, MTN=true | MTN T/K/factor 改变 | 改变 | 是 | 视 runtime profile 而定 |

最后一行仍遵循现有 `calibration_trajectory_config()` dependency，不要在本轮重新定义 MTN 数学。

---

# 38. 必须运行测试

```bash
pytest -q
```

必须全部通过。

建议定向运行：

```bash
pytest -q tests/test_artifacts.py
pytest -q tests/test_conversion_metadata.py
pytest -q tests/test_calibration_profiles.py
pytest -q tests/test_blockwise_calibration_config.py
pytest -q tests/test_blockwise_calibration_runtime.py
```

如果实际测试文件名与上述不同，以仓库现有路径测试文件为准。

---

# 39. 最终验收标准

本轮完成后必须满足：

1. `ann_training_calibration_dir` 不再有无条件的外层：
   ```text
   phase_base_<BASE>
   ```

2. `post_finetuning_conversion_calibration_dir` 不再有该冗余层。

3. `vanilla_analysis_calibration_dir` 不再有该冗余层。

4. Stage-A runtime dependency 只由：
   ```python
   calibration_trajectory_dirname()
   ```
   决定。

5. `phase_previous_layers_snn=false` 时：
   ```text
   phase.base/T
   ```
   改变不改变 Stage-A path。

6. 同一份 common Stage A 可以直接生成多个不同 Phase base/T 的 Stage-B Clip profiles。

7. `phase_previous_layers_snn=true` 时：
   ```text
   phase.base/T
   ```
   仍然编码在 trajectory dirname 中。

8. sequential Phase 的 base/T 发生变化时，必须指向新的 Stage A。

9. Stage-B Clip profile path 继续包含：
   ```text
   phase_base
   phase_T
   mtn_T
   ```

10. 不放宽现有 sequential provenance validation。

11. `pytest -q` 全部通过。

---

# 40. 最终设计总结

修改前的 Stage-A path 同时存在：

```text
phase_base_<BASE>/
```

和：

```text
<trajectory ... phase_base_<BASE>_T_<T>>
```

导致：

- common trajectory 时错误地把 Stage A 按 base 分开；
- sequential trajectory 时 base 信息重复。

修改后统一采用：

```text
calibration_variant/
    └── calibration_trajectory_dirname(...)
```

作为 Stage-A identity。

这样自然得到：

```text
previous_layers_snn=false
    → runtime 参数不进入 Stage-A path
    → Stage A 可复用
    → runtime 参数变化只重新 Stage B
```

以及：

```text
previous_layers_snn=true
    → 对应 trajectory 参数进入 Stage-A path
    → runtime 参数变化自动切换到新的 Stage A
    → 必须重新 Stage A，再重新 Stage B
```

这使 artifact 路径与真实数学依赖一一对应，并消除当前冗余的外层 `phase_base_*`。
