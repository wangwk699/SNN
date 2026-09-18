# Tulu-3 Canonical Preprocessing 与 Rotation Regression Seed Scope 修正方案

> 目标仓库：`https://github.com/wangwk699/SNN`
> 目标分支：`main`
> 本轮检查基准提交：`f038ee688cbb776ed3728f1b7f311099102ed8d1`
>
> 本文档用于指导 Codex 在没有额外上下文的情况下完成本轮修正。
>
> 本轮只修复一个核心问题：
>
> **Tulu-3 的 `canonical_preprocessing_calibration` 当前错误地绑定到了 `training.train_seed` 的 data-selection scope，而 Rotation weights / Rotation regression 又仍处于 experiment-level shared scope，导致不同 seed run 之间出现 Rotation regression provenance 冲突。**
>
> 已经正确的以下逻辑全部保持不变：
>
> - Tulu-3：先按 `training.train_seed` 从 validation-excluded shared pool 抽 ANN training subset，再按 `calibration.seed` 从 ANN training subset 内抽 Stage-A calibration subset；
> - 当前默认关系：`ANN Training 10000 -> Stage-A Calibration 128`；
> - Tulu-3 concrete run path 使用同一级目录：
>   `experiment_seed_<E>_train_seed_<T>_calibration_seed_<C>`；
> - TL;DR run path 保持 `seed<experiment.seed>`；
> - Tulu-3 Pre-finetuning Prefix、ANN-training Stage A/B 继续按三 seed data-selection identity 隔离；
> - `canonical_preprocessing_calibration` 仍是独立用于 Rotation regression 的 128 samples，不改成 ANN 10000 的子集。

---

# 1. 当前问题

当前 `ArtifactLayout` 中：

```python
self.shared_task_root = (
    task_root
    / "_shared"
    / data_selection_seed_name
)
```

而 Tulu-3：

```text
data_selection_seed_name
=
experiment_seed_<E>_train_seed_<T>_calibration_seed_<C>
```

同时：

```python
@property
def data_dir(self) -> Path:
    return self.shared_task_root / "data"
```

以及：

```python
@property
def canonical_preprocessing_calibration_dir(self) -> Path:
    return self.data_dir / "canonical_preprocessing" / "num_samples_128"
```

因此 Tulu-3 canonical preprocessing 当前路径类似：

```text
.../
_shared/
experiment_seed_42_train_seed_42_calibration_seed_42/
data/
canonical_preprocessing/
num_samples_128/
calibration_manifest.json
```

这意味着 `training.train_seed` 变化时，即使 canonical preprocessing samples 的数学定义完全没有变化，其 artifact path 也会变化。

这是不正确的 dependency。

---

# 2. Canonical preprocessing 的真实 dependency

当前 canonical preprocessing 的数据选择代码是：

```python
canonical_positions, canonical_indices = _calibration_selection(
    list(range(len(raw_train))),
    seed=int(cfg["calibration"]["seed"]),
    num_samples=CANONICAL_PREPROCESSING_NUM_SAMPLES,
    with_replacement=False,
)
```

所以它依赖：

- dataset / raw train split
- `calibration.seed`
- `CANONICAL_PREPROCESSING_NUM_SAMPLES (=128)`

在 Tulu-3 下它**不依赖**：

- `training.train_seed`
- `training.train_samples`

也不依赖 Stage-A ANN training subset。

因此 `training.train_seed` 不能出现在 canonical preprocessing artifact identity 中。

---

# 3. Rotation weights 与 Rotation regression 的 dependency 必须拆开

当前：

```python
@property
def rotation_dir(self) -> Path:
    return (
        self.shared_model_root
        / "rotated_prefix"
        / "rotation"
    )
```

`shared_model_root` 仍按 `experiment.seed` 隔离。

因此 Rotation 权重路径类似：

```text
.../
_shared/
seed42/
rotated_prefix/
rotation/
```

这是合理的。

Rotation fusion 本身不应因为：

- `training.train_seed`
- `calibration.seed`

变化而重新保存不同 fused weights。

因此：

```text
rotation_state.pt
fused_base/
```

应继续保持 experiment/model-policy shared。

---

# 4. 真正冲突发生在 Rotation regression metadata

`prepare_rotation.py` 当前把 regression 写入：

```text
layout.rotation_dir / "rotation_regression.json"
```

同时 regression 记录：

```text
calibration_manifest_path
calibration_manifest_sha256
```

其来源为：

```python
layout.canonical_preprocessing_calibration_manifest_path
```

而 verifier 又要求：

```python
Path(recorded_manifest).resolve()
==
layout.canonical_preprocessing_calibration_manifest_path.resolve()
```

以及：

```python
regression["calibration_manifest_sha256"]
==
sha256_file(layout.canonical_preprocessing_calibration_manifest_path)
```

所以只要 `training.train_seed` 改变，而 canonical path 又被错误绑定到 train seed，就会出现：

```text
共享 Rotation regression
+
不同 canonical manifest path
```

的 provenance 冲突。

---

# 5. 冲突示例

Run A：

```yaml
experiment:
  seed: 42
training:
  train_seed: 42
calibration:
  seed: 42
```

Run B：

```yaml
experiment:
  seed: 42
training:
  train_seed: 43
calibration:
  seed: 42
```

canonical preprocessing 的实际 sample selection 完全相同，因为它不使用 `train_seed`。

但当前 canonical path 会不同。

而 Rotation regression 仍共享：

```text
.../_shared/seed42/rotated_prefix/rotation/rotation_regression.json
```

于是 Run A / Run B 不能同时维持严格 provenance valid。

---

# 6. `calibration.seed` 改变时是真实 dependency

与 `training.train_seed` 不同，`calibration.seed` 会真实改变 canonical preprocessing 128 samples。

因此：

```text
calibration.seed=42
```

和：

```text
calibration.seed=43
```

必须有不同的 canonical preprocessing artifact identity。

同时 Rotation regression 结果也必须按 canonical calibration selection 隔离。

---

# 7. 最终 dependency 设计

目标关系：

```text
Concrete ANN run:
experiment.seed
+ training.train_seed
+ calibration.seed
```

继续使用：

```text
experiment_seed_<E>_train_seed_<T>_calibration_seed_<C>
```

Stage-A data / Prefix / ANN-training Stage A/B：

```text
experiment.seed
+ training.train_seed
+ calibration.seed
```

继续保持三 seed identity。

Canonical preprocessing：

```text
experiment.seed
+ calibration.seed
```

**不依赖 `training.train_seed`。**

Rotation weights：

```text
experiment/model/rotation identity
```

不依赖 train/calibration seed。

Rotation regression：

```text
Rotation weights
+
canonical preprocessing selection
```

因此至少按 `calibration.seed` 隔离。

---

# 8. 推荐最终目录设计

Canonical preprocessing：

```text
.../<task>/_shared/
seed42/
canonical_preprocessing/
calibration_seed_42/
num_samples_128/
calibration_manifest.json
```

Rotation：

```text
.../<task>/<model>/_shared/seed42/
└── rotated_prefix/
    └── rotation/
        ├── rotation_state.pt
        ├── fused_base/
        └── regression/
            └── calibration_seed_42/
                ├── rotation_regression.json
                └── rotation_summary.json
```

---

# 9. Canonical preprocessing path 不再复用 `data_selection_seed_name`

不要继续让：

```python
canonical_preprocessing_calibration_dir
```

挂在：

```python
self.data_dir
```

下面，因为 Tulu-3 `data_dir` 当前就是三 seed data-selection scope。

应为 canonical preprocessing 单独建立 root。

---

# 10. 推荐新增 canonical preprocessing root

例如：

```python
self.canonical_preprocessing_root = (
    task_root
    / "_shared"
    / experiment_seed_name
    / "canonical_preprocessing"
    / f"calibration_seed_{int(cfg['calibration']['seed'])}"
)
```

然后：

```python
@property
def canonical_preprocessing_calibration_dir(self) -> Path:
    return (
        self.canonical_preprocessing_root
        / "num_samples_128"
    )
```

核心是 dependency 必须是：

```text
experiment.seed + calibration.seed
```

而不是：

```text
experiment.seed + training.train_seed + calibration.seed
```

---

# 11. Stage-A `data_dir` 保持三 seed scope

当前 Tulu-3：

```python
self.shared_task_root = (
    task_root
    / "_shared"
    / data_selection_seed_name
)
```

这对于：

```text
train_manifest.json
validation_manifest.json
Stage-A calibration manifest
```

是合理的。

不要因为修 canonical preprocessing，又把 Stage-A data manifest 改回只按 `experiment.seed` 共享。

---

# 12. Rotation weights path 保持不变

继续：

```text
.../<model>/_shared/seed42/rotated_prefix/rotation/
```

以下保持共享：

```text
rotation_state.pt
fused_base/
```

不要改成 three-seed composite scope。

---

# 13. Rotation regression 单独增加 calibration seed scope

当前：

```text
rotation/
├── rotation_regression.json
└── rotation_summary.json
```

推荐改为：

```text
rotation/
├── rotation_state.pt
├── fused_base/
└── regression/
    └── calibration_seed_<C>/
        ├── rotation_regression.json
        └── rotation_summary.json
```

例如：

```text
rotation/regression/calibration_seed_42/rotation_regression.json
```

---

# 14. `rotation_summary.json` 也应随 regression 隔离

当前 `rotation_summary.json` 包含：

```text
rotation_regression_format_version
rotation_regression_path
rotation_regression_passed
```

所以它也绑定某一次 regression。

因此推荐：

```text
regression/calibration_seed_<C>/
    rotation_regression.json
    rotation_summary.json
```

而不是只移动 regression 文件。

---

# 15. 推荐 `ArtifactLayout` 新增 properties

例如：

```python
@property
def rotation_regression_dir(self) -> Path:
    return (
        self.rotation_dir
        / "regression"
        / f"calibration_seed_{int(self._cfg['calibration']['seed'])}"
    )
```

```python
@property
def rotation_regression_path(self) -> Path:
    return self.rotation_regression_dir / "rotation_regression.json"
```

```python
@property
def rotation_summary_path(self) -> Path:
    return self.rotation_regression_dir / "rotation_summary.json"
```

后续 caller 不要继续手写：

```python
layout.rotation_dir / "rotation_regression.json"
layout.rotation_dir / "rotation_summary.json"
```

---

# 16. `scripts/prepare_rotation.py` 修改

当前：

```python
regression_path = layout.rotation_dir / "rotation_regression.json"
```

改为：

```python
regression_path = layout.rotation_regression_path
```

保存 summary 当前：

```python
write_json(
    layout.rotation_dir / "rotation_summary.json",
    {...}
)
```

改为：

```python
write_json(
    layout.rotation_summary_path,
    {...}
)
```

其中：

```text
rotation_regression_path
```

继续记录当前实际 regression path。

---

# 17. Rotation state / fused Base 保存位置不要改

继续：

```python
save_rotation_state(
    state,
    layout.rotation_dir / "rotation_state.pt",
)
```

以及：

```python
destination = layout.rotation_dir / "fused_base"
```

这两者继续共享。

---

# 18. `scripts/verify_artifacts.py` 同步修改

required artifacts 当前若写：

```python
layout.rotation_dir / "rotation_regression.json"
layout.rotation_dir / "rotation_summary.json"
```

全部改为：

```python
layout.rotation_regression_path
layout.rotation_summary_path
```

读取也统一使用新的 properties。

---

# 19. Verifier 保留 strict canonical path/hash 校验

不要删除或放宽：

```python
Path(recorded_manifest).resolve()
==
layout.canonical_preprocessing_calibration_manifest_path.resolve()
```

以及 manifest hash 校验。

应该修 artifact identity，而不是降低 provenance 强度。

---

# 20. `prepare_rotation.py` 仍使用独立 canonical selection

继续：

```python
calibration = load_canonical_preprocessing_raw(
    cfg, layout
)
```

以及：

```python
manifest_path = (
    layout.canonical_preprocessing_calibration_manifest_path
)
```

不要切换到 Stage-A calibration manifest。

---

# 21. `load_canonical_preprocessing_raw()` 数学逻辑不变

继续验证：

```text
manifest_role = canonical_preprocessing_calibration
num_samples = 128
sampling = seeded_without_replacement
duplicates_preserved = false
```

本轮只改 path scope，不改 selection 数学。

---

# 22. `prepare_manifests()` 自然写入新 canonical path

完整 Step 2 仍写：

```python
layout.canonical_preprocessing_calibration_manifest_path
```

只需让 layout property 指向新位置。

`--calibration-only` 继续只写 Stage-A calibration，不重写 canonical preprocessing。

---

# 23. 关键 dependency 验收

## 改 `training.train_seed`

例如：

```text
A: exp=42 train=42 cal=42
B: exp=42 train=43 cal=42
```

必须满足：

```python
A.root != B.root
A.calibration_data_manifest_path != B.calibration_data_manifest_path
A.ann_training_prefix_dir != B.ann_training_prefix_dir
A.ann_training_calibration_dir != B.ann_training_calibration_dir
```

但：

```python
A.canonical_preprocessing_calibration_manifest_path
==
B.canonical_preprocessing_calibration_manifest_path

A.rotation_dir == B.rotation_dir
A.rotation_regression_path == B.rotation_regression_path
```

---

## 改 `calibration.seed`

例如：

```text
A: exp=42 train=42 cal=42
C: exp=42 train=42 cal=43
```

必须：

```python
A.root != C.root
A.calibration_data_manifest_path != C.calibration_data_manifest_path
A.canonical_preprocessing_calibration_manifest_path
!=
C.canonical_preprocessing_calibration_manifest_path
```

同时：

```python
A.rotation_dir == C.rotation_dir
```

但：

```python
A.rotation_regression_path
!=
C.rotation_regression_path
```

---

# 24. Tulu-3 three-seed concrete run path 保持现状

继续：

```text
experiment_seed_<E>_train_seed_<T>_calibration_seed_<C>
```

这是**一个目录名**。

本轮不要修改这一点。

---

# 25. Stage-A data / Prefix / A-B 仍按 three-seed scope

改变 `training.train_seed` 时，以下继续变化：

```text
train manifest
Stage-A calibration manifest
Pre-finetuning Prefix
ANN-training Stage A
ANN-training Stage B
concrete ANN run root
```

---

# 26. `calibration.seed` 改变时的正确影响范围

必须变化：

```text
Stage-A calibration manifest
Pre-finetuning Prefix
ANN-training Stage A/B
concrete ANN run
canonical preprocessing manifest
Rotation regression
```

保持不变：

```text
rotation_state.pt
fused_base/
```

---

# 27. 推荐新增测试：canonical ignores train seed

```python
def test_tulu_canonical_preprocessing_ignores_train_seed():
    ...
    assert (
        first.canonical_preprocessing_calibration_manifest_path
        ==
        second.canonical_preprocessing_calibration_manifest_path
    )
    assert first.rotation_dir == second.rotation_dir
    assert (
        first.rotation_regression_path
        ==
        second.rotation_regression_path
    )
```

---

# 28. 推荐新增测试：canonical follows calibration seed

```python
def test_tulu_canonical_preprocessing_tracks_calibration_seed():
    ...
    assert (
        first.canonical_preprocessing_calibration_manifest_path
        !=
        second.canonical_preprocessing_calibration_manifest_path
    )
    assert first.rotation_dir == second.rotation_dir
    assert (
        first.rotation_regression_path
        !=
        second.rotation_regression_path
    )
```

---

# 29. 推荐新增测试：Rotation weights remain shared

分别改变：

```text
training.train_seed
calibration.seed
```

都必须：

```python
first.rotation_dir == second.rotation_dir
```

并确认：

```python
first.rotation_dir / "rotation_state.pt"
==
second.rotation_dir / "rotation_state.pt"
```

以及：

```python
first.rotation_dir / "fused_base"
==
second.rotation_dir / "fused_base"
```

---

# 30. 推荐新增 verifier integration test

先生成：

```text
experiment.seed=42
calibration.seed=42
```

的 canonical manifest 与 regression。

然后只改：

```text
training.train_seed
```

新 layout 仍必须找到同一个：

```text
canonical manifest
rotation regression
rotation summary
```

并通过 path/hash provenance 检查。

---

# 31. `实验执行总结.md` 同步修正

附录 I 中保留：

> `calibration.seed` 也用于独立的 canonical preprocessing 128 samples。

同时增加：

```text
Tulu-3 three-seed composite identity 只用于 concrete run 和 Stage-A data-dependent shared artifacts。
canonical_preprocessing_calibration 不依赖 training.train_seed；
其 identity 仅按 experiment.seed 与 calibration.seed 隔离。
Rotation weights / fused_base 继续按 experiment/model-policy 共享；
Rotation regression 按 canonical preprocessing 的 calibration.seed 隔离。
```

---

# 32. 更新关键路径示例

加入：

```text
# Tulu-3 concrete run
.../
experiment_seed_42_train_seed_42_calibration_seed_42/
ann/

# Tulu-3 Stage-A data
.../_shared/
experiment_seed_42_train_seed_42_calibration_seed_42/
data/

# canonical preprocessing
.../_shared/
seed42/
canonical_preprocessing/
calibration_seed_42/
num_samples_128/
calibration_manifest.json

# shared rotation weights
.../<model>/_shared/
seed42/
rotated_prefix/
rotation/
├── rotation_state.pt
├── fused_base/
└── regression/
    └── calibration_seed_42/
        ├── rotation_regression.json
        └── rotation_summary.json
```

---

# 33. 不要修改四种 seed 的职责

继续：

```text
experiment.seed
→ global training randomness
→ Tulu validation/shared-pool split

training.tldr_train_seed
→ TL;DR ANN training subset

training.train_seed
→ Tulu-3 ANN training subset

calibration.seed
→ Stage-A calibration subset
→ independent canonical preprocessing 128
```

---

# 34. 不要修改 Stage-A 10000→128 逻辑

继续保证：

```text
Tulu-3:
shared pool
→ train_seed
→ ANN 10000
→ calibration.seed
→ Stage-A 128
```

并：

```text
Calibration 128 ⊆ ANN Training 10000
```

---

# 35. 不要修改 TL;DR path

TL;DR concrete run 继续：

```text
seed42
```

本轮如果使用通用 helper，也必须加测试确保 TL;DR 路径不变。

---

# 36. 旧 artifact 处理

用户计划：

```text
artifacts/snn2_main_v1/tulu3
→
artifacts/snn2_main_v1/tulu3_old
```

然后从 Step 2 重新跑。

所以本轮不要实现 legacy fallback，也不要从 `tulu3_old` 自动迁移 artifact。

---

# 37. 建议全仓库搜索

修改后：

```bash
grep -R 'rotation_regression.json' -n snn2 scripts tests *.md
grep -R 'rotation_summary.json' -n snn2 scripts tests *.md
grep -R 'canonical_preprocessing_calibration' -n snn2 scripts tests *.md
grep -R 'canonical_preprocessing_calibration_manifest_path' -n snn2 scripts tests
grep -R 'rotation_dir / "rotation_regression.json"' -n snn2 scripts tests
grep -R 'rotation_dir / "rotation_summary.json"' -n snn2 scripts tests
```

确保所有 caller 均采用新的 layout properties。

---

# 38. 本轮不要修改

不要修改：

- Tulu-3 10000→128 nested sampling；
- TL;DR sampling；
- Tulu three-seed concrete run dirname；
- Stage-A data three-seed identity；
- Prefix three-seed identity；
- ANN-training Stage A/B three-seed identity；
- Base baseline path；
- Rotation fusion math；
- Hadamard policy；
- rotation seed；
- Phase/GIF/MTN math；
- Trainer seed/data_seed；
- lm-eval；
- evaluation sampling。

---

# 39. 必须运行测试

至少：

```bash
pytest -q
```

全部通过。

建议重点：

```bash
pytest -q tests/test_tulu3_lm_eval_protocol.py
pytest -q tests/test_verify_artifacts.py
pytest -q tests/test_generated_configs.py
```

---

# 40. 最终验收矩阵

| 改动参数 | Concrete ANN run | Stage-A data/Prefix/A-B | Canonical preprocessing | Rotation weights | Rotation regression |
|---|---:|---:|---:|---:|---:|
| `experiment.seed` | 变化 | 变化 | 变化 | 变化 | 变化 |
| `training.train_seed` | 变化 | 变化 | **不变** | **不变** | **不变** |
| `calibration.seed` | 变化 | 变化 | **变化** | **不变** | **变化** |

---

# 41. 最终设计总结

修正后 Tulu-3 应形成三个清晰 scope：

```text
Scope A：Concrete run / Stage-A data selection
experiment.seed + training.train_seed + calibration.seed
```

例如：

```text
experiment_seed_42_train_seed_42_calibration_seed_42
```

```text
Scope B：Canonical preprocessing
experiment.seed + calibration.seed
```

例如：

```text
seed42/
canonical_preprocessing/
calibration_seed_42/
```

```text
Scope C：Rotation weights
experiment/model/rotation identity
```

继续共享：

```text
seed42/rotated_prefix/rotation/
rotation_state.pt
fused_base/
```

而 Rotation regression 作为：

```text
Rotation weights + canonical preprocessing selection
```

的验证结果，放在：

```text
rotation/regression/calibration_seed_42/
```

下面。

这样：

- 改 `training.train_seed` 不会错误改变 canonical preprocessing；
- 改 `training.train_seed` 不会让共享 Rotation regression provenance 失效；
- 改 `calibration.seed` 会正确生成新的 canonical 128 和新的 Rotation regression；
- Rotation weights / fused_base 不会因为 calibration data 变化而无意义复制；
- 所有 artifact path dependency 与真实数学依赖保持一致。
