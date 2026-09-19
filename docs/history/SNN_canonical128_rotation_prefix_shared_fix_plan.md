# Canonical 128 / Rotation Regression / Pre-finetuning Prefix 统一共享方案

> 目标仓库：`https://github.com/wangwk699/SNN`
>
> 本文档基于当前 `main` 最新代码设计，供 Codex 在没有额外上下文的情况下完成本轮修改。
>
> 本轮目标：
>
> 1. 修正 `实验执行总结.md` 中关于 Pre-finetuning Prefix “每个 model-task 一次”与当前 data-selection identity 之间的表述矛盾；
> 2. 将 **Pre-finetuning Prefix** 改为与 **canonical preprocessing / Rotation regression** 共用同一套固定 128 条数据；
> 3. 让这套固定 128 条数据仅由 `experiment.seed` 决定，不再依赖：
>    - `training.tldr_train_samples`
>    - `training.train_samples`
>    - `training.tldr_train_seed`
>    - `training.train_seed`
>    - `calibration.seed`
>    - `calibration.num_samples`
> 4. 因此保证：
>    - canonical 128：每个 task + `experiment.seed` 一套；
>    - Rotation regression：每个 model-task + `experiment.seed` 一套；
>    - Pre-finetuning Prefix：每个 model-task + `experiment.seed` 一套；
>    - 三者都不因 `train_samples` sweep 或 Stage-A calibration seed sweep 重跑；
> 5. ANN-training Stage A/B 仍然严格依赖完整 data-selection identity；
> 6. 清理 canonical preprocessing manifest 中容易误导的 metadata；
> 7. 更新 Prefix provenance / validator / rotated-pre-finetuning evaluation path；
> 8. 保持此前已经完成的 **manifest-only ANN training** 规则不变。

---

# 1. 最终数据依赖关系

本轮修改后，应明确存在两套完全不同的数据选择链。

## 1.1 Canonical fixed 128

用于：

```text
Rotation regression
Pre-finetuning Prefix discovery
```

来源：

```text
raw train
    ↓ experiment.seed
fixed canonical 128
```

因此它只依赖：

```text
dataset identity
experiment.seed
```

不依赖：

```text
task-specific train seed
train_samples
calibration.seed
calibration.num_samples
```

---

## 1.2 ANN-training Stage-A calibration

用于：

```text
ANN-training calibration Stage A/B
aware ANN fine-tuning
use_post_finetuning_artifacts=false 的 SNN source
```

来源：

TL;DR：

```text
raw train
    ↓ tldr_train_seed + tldr_train_samples
fixed ANN training subset
    ↓ calibration.seed
Stage-A calibration subset
```

Tulu-3：

```text
raw train
    ↓ experiment.seed
fixed validation + shared pool
    ↓ train_seed + train_samples
fixed ANN training subset
    ↓ calibration.seed
Stage-A calibration subset
```

这套数据仍然依赖完整 data-selection identity。

---

# 2. 最终职责边界

修改后：

```text
canonical 128
├── Rotation regression
└── Pre-finetuning Prefix

Step-2 training manifest
└── Stage-A calibration
    ├── ANN-training Stage A/B
    ├── ANN training
    └── post-finetuning chain
```

禁止再让 Pre-finetuning Prefix 使用：

```text
layout.calibration_data_manifest_path
```

或：

```text
load_selected_raw(...).calibration
```

---

# 3. Canonical 128 的抽样 seed 改为 `experiment.seed`

当前 `prepare_manifests()` 中 canonical 128 使用：

```python
seed=int(cfg["calibration"]["seed"])
```

必须改为：

```python
seed=int(cfg["experiment"]["seed"])
```

即：

```python
canonical_positions, canonical_indices = _calibration_selection(
    list(range(len(raw_train))),
    seed=int(cfg["experiment"]["seed"]),
    num_samples=CANONICAL_PREPROCESSING_NUM_SAMPLES,
    with_replacement=False,
)
```

---

# 4. Canonical 128 不再随 `calibration.seed` 变化

例如：

```yaml
experiment:
  seed: 42

calibration:
  seed: 42
```

和：

```yaml
experiment:
  seed: 42

calibration:
  seed: 123
```

必须得到完全相同的：

```text
canonical indices
canonical manifest path
canonical manifest SHA
Rotation regression
Pre-finetuning Prefix
```

---

# 5. Canonical preprocessing path 去掉 `calibration_seed`

当前：

```text
.../<task>/_shared/
seed42/
canonical_preprocessing/
calibration_seed_42/
num_samples_128/
calibration_manifest.json
```

改为：

```text
.../<task>/_shared/
seed42/
canonical_preprocessing/
num_samples_128/
calibration_manifest.json
```

即：

```python
self.canonical_preprocessing_root = (
    task_root
    / "_shared"
    / experiment_seed_name
    / "canonical_preprocessing"
)
```

不要再追加：

```text
calibration_seed_<C>
```

---

# 6. Canonical manifest metadata 清理

当前 canonical manifest 中存在：

```python
"calibration_seed": ...
"positions_in_selected_train": ...
"retained_in_training": True
```

这些字段不再合适。

---

# 7. 删除 `calibration_seed`

canonical selection 以后不再由：

```text
calibration.seed
```

决定，因此 canonical manifest 不应再写：

```json
"calibration_seed": 42
```

---

# 8. 删除 `retained_in_training`

当前：

```json
"retained_in_training": true
```

语义不正确。

canonical 128 是从 raw train 独立抽取的，并不保证：

```text
canonical 128 ⊆ 当前 ANN training subset
```

因此彻底删除：

```text
retained_in_training
```

---

# 9. 删除或重命名 `positions_in_selected_train`

canonical selection 并不是：

```text
selected ANN training subset
```

上的 position。

当前：

```python
positions_in_selected_train
```

会误导。

推荐直接删除。

如果确实需要保留抽样位置，可改成：

```text
positions_in_raw_train
```

但由于 canonical selection 是：

```python
list(range(len(raw_train)))
```

其 position 与 raw index 实际等价，因此没有必要重复存储。

推荐：

> 直接删除 `positions_in_selected_train`。

---

# 10. Canonical manifest 推荐 metadata

建议变成：

```json
{
  "dataset_name": "...",
  "dataset_config_name": null,
  "dataset_revision": "...",
  "seed": 42,
  "manifest_role": "canonical_preprocessing_calibration",
  "selection_scope": "raw_train_split",
  "selection_seed_source": "experiment.seed",
  "selection_seed": 42,
  "split": "train",
  "sampling": "seeded_without_replacement",
  "num_samples": 128,
  "indices": [...],
  "record_ids": [...],
  "duplicates_preserved": false
}
```

---

# 11. `load_canonical_preprocessing_raw()` 加强 validator

当前只验证：

```text
manifest_role
num_samples
sampling
duplicates_preserved
```

建议增加：

```python
expected = {
    "manifest_role": "canonical_preprocessing_calibration",
    "selection_scope": "raw_train_split",
    "selection_seed_source": "experiment.seed",
    "selection_seed": int(cfg["experiment"]["seed"]),
    "num_samples": CANONICAL_PREPROCESSING_NUM_SAMPLES,
    "sampling": "seeded_without_replacement",
    "duplicates_preserved": False,
}
```

并继续：

```python
raw[manifest["split"]].select(manifest["indices"])
```

---

# 12. Rotation regression path 去掉 `calibration_seed`

当前：

```text
rotation/
regression/
calibration_seed_42/
rotation_regression.json
```

改为：

```text
rotation/
regression/
rotation_regression.json
```

以及：

```text
rotation/
regression/
rotation_summary.json
```

---

# 13. `rotation_regression_dir`

推荐：

```python
@property
def rotation_regression_dir(self) -> Path:
    if self._cfg["experiment"]["task"] in {"tldr", "tulu3"}:
        return self.rotation_dir / "regression"
    return self.rotation_dir
```

不要再引用：

```python
cfg["calibration"]["seed"]
```

---

# 14. Rotation weights 保持不变

继续：

```text
.../<model>/_shared/
seed<E>/
rotated_prefix/
rotation/
├── rotation_state.pt
├── fused_base/
└── regression/
```

---

# 15. Rotation regression 继续读取 canonical 128

`scripts/prepare_rotation.py` 当前：

```python
calibration = load_canonical_preprocessing_raw(
    cfg,
    layout,
)
```

这是正确的。

无需改变数据 loader。

---

# 16. Pre-finetuning Prefix 改用 canonical 128

当前 `scripts/discover_prefix.py` 不区分数据来源，统一：

```python
load_selected_raw(cfg, layout).calibration
```

这必须修改。

---

# 17. `discover_prefix.py` Pre/Post 分支必须使用不同数据源

推荐结构：

```python
if canonical_stage == "pre_finetuning":
    calibration_raw = load_canonical_preprocessing_raw(
        cfg,
        layout,
    )
    manifest_path = (
        layout.canonical_preprocessing_calibration_manifest_path
    )
    discovery_num_samples = (
        CANONICAL_PREPROCESSING_NUM_SAMPLES
    )
    discovery_data_source = (
        "canonical_preprocessing_calibration"
    )
else:
    calibration_raw = load_selected_raw(
        cfg,
        layout,
    ).calibration
    manifest_path = (
        layout.calibration_data_manifest_path
    )
    discovery_num_samples = int(
        cfg["calibration"]["num_samples"]
    )
    discovery_data_source = (
        "stage_a_calibration_selection"
    )
```

然后：

```python
state = discover_prefix_tokens(
    model,
    tokenizer,
    calibration_raw,
    cfg,
    output_dir / "prefix_state.json",
)
```

---

# 18. Pre-finetuning Prefix 必须完全绕过 Step-2 Stage-A calibration

Pre-finetuning branch 不允许再调用：

```python
load_selected_raw(...).calibration
```

因为该函数会重新引入：

```text
train_samples
train seed
calibration.seed
```

dependency。

---

# 19. Pre-finetuning Prefix provenance

Pre-finetuning `prefix_state.json` 应记录：

```json
{
  "discovery_num_samples": 128,
  "discovery_data_source": "canonical_preprocessing_calibration",
  "discovery_manifest_path": ".../canonical_preprocessing/num_samples_128/calibration_manifest.json",
  "discovery_manifest_sha256": "..."
}
```

---

# 20. Post-finetuning Prefix 保持现有语义

Post-finetuning Prefix 仍然使用当前 run 的 Stage-A calibration subset：

```json
{
  "discovery_num_samples": calibration.num_samples,
  "discovery_data_source": "stage_a_calibration_selection",
  "discovery_manifest_path": ".../data/calibration/num_samples_N/calibration_manifest.json",
  "discovery_manifest_sha256": "..."
}
```

不要把 Post-finetuning Prefix 也切到 canonical 128。

---

# 21. Pre-finetuning Prefix path 从 data-selection root 移出

当前：

```python
self.ann_training_prefix_base_dir = (
    self.data_selection_policy_root
    / "pre_finetuning_prefix"
)
```

必须改为：

```python
@property
def ann_training_prefix_base_dir(self) -> Path:
    return self.policy_root / "pre_finetuning_prefix"
```

---

# 22. Pre-finetuning Prefix 固定使用 `num_samples_128`

当前：

```python
ann_training_prefix_dir
=
... / f"num_samples_{cfg['calibration']['num_samples']}"
```

必须改为固定：

```text
num_samples_128
```

---

# 23. 避免魔法数字

建议将：

```python
CANONICAL_PREPROCESSING_NUM_SAMPLES = 128
```

移动到中立模块，例如：

```text
snn2/data_constants.py
```

然后：

```text
artifacts.py
data.py
discover_prefix.py
```

统一 import。

避免 `artifacts.py -> data.py` 循环依赖。

---

# 24. `ann_training_prefix_dir`

最终类似：

```python
@property
def ann_training_prefix_dir(self) -> Path:
    return (
        self.ann_training_prefix_base_dir
        / f"num_samples_{CANONICAL_PREPROCESSING_NUM_SAMPLES}"
    )
```

---

# 25. ANN-training calibration 继续依赖完整 data-selection identity

不要修改：

```python
self.ann_training_calibration_dir
```

继续挂：

```python
self.data_selection_policy_root
```

因为 Stage A/B 仍依赖：

```text
train_samples
task-specific train seed
calibration.seed
```

---

# 26. 最终路径结构

TL;DR / Tulu-3 都应遵循：

```text
<task>/
├── _shared/
│   ├── seed42/
│   │   └── canonical_preprocessing/
│   │       └── num_samples_128/
│   │           └── calibration_manifest.json
│   └── <full-data-selection-identity>/
│       └── data/
│           ├── train_manifest.json
│           └── calibration/
│               └── num_samples_<N>/
│                   └── calibration_manifest.json
└── <model>/
    └── _shared/
        └── seed42/
            └── rotated_prefix/
                ├── rotation/
                │   ├── rotation_state.pt
                │   ├── fused_base/
                │   └── regression/
                │       ├── rotation_regression.json
                │       └── rotation_summary.json
                ├── pre_finetuning_prefix/
                │   └── num_samples_128/
                │       ├── prefix_state.json
                │       └── prefixed_key_values.pt
                └── <full-data-selection-identity>/
                    └── ann_training_calibration/
```

---

# 27. `validate_prefix_discovery_state()` 必须 stage-aware

当前 validator 默认所有 Prefix 都来自 Stage-A calibration，这不再正确。

推荐签名：

```python
def validate_prefix_discovery_state(
    cfg,
    layout,
    prefix_dir,
    *,
    stage: str,
):
```

允许：

```text
pre_finetuning
post_finetuning
```

---

# 28. Pre-finetuning Prefix validator

当：

```text
stage == "pre_finetuning"
```

expected：

```python
manifest_path = (
    layout.canonical_preprocessing_calibration_manifest_path
)

expected = {
    "discovery_num_samples":
        CANONICAL_PREPROCESSING_NUM_SAMPLES,
    "discovery_data_source":
        "canonical_preprocessing_calibration",
    "discovery_manifest_path":
        str(manifest_path.resolve()),
    "discovery_manifest_sha256":
        sha256_file(manifest_path),
}
```

Prefix root 应为：

```text
num_samples_128
```

---

# 29. Post-finetuning Prefix validator

当：

```text
stage == "post_finetuning"
```

继续：

```python
manifest_path = layout.calibration_data_manifest_path
```

并按：

```text
calibration.num_samples
stage_a_calibration_selection
```

验证。

---

# 30. 更新所有 Prefix validator caller

全仓库：

```bash
grep -R "validate_prefix_discovery_state" -n snn2 scripts tests
```

所有调用方必须显式传递 Prefix stage。

不要通过猜路径判断 provenance 类型。

---

# 31. ANN training 使用的 Prefix

ANN training：

```text
ann_training.prefix_enabled=true
```

时继续读取 Pre-finetuning Prefix。

但现在：

```text
train_samples=2000
5000
10000
```

全部共享同一个 Prefix。

---

# 32. ANN-training Stage A/B

当 Prefix enabled 时：

```text
固定 canonical Pre Prefix
+
当前 data-selection-specific Stage-A calibration
```

这是本轮最终语义。

---

# 33. rotated-pre-finetuning evaluation 也应移出 data-selection root

当前：

```python
return (
    self.data_selection_policy_root
    / "rotated_pre_finetuning"
)
```

改为：

```python
return (
    self.policy_root
    / "rotated_pre_finetuning"
)
```

原因：

```text
rotated Base
fixed Pre Prefix
evaluation dataset
```

均不依赖 `train_samples / train seed / calibration.seed`。

---

# 34. rotated-pre-finetuning evaluation 不再按 `calibration.num_samples` 加 suffix

当前：

```python
if rotated_pre_finetuning:
    return rotated_pre_finetuning_prefix_enabled(cfg)
```

应改为：

```python
if rotated_pre_finetuning:
    return False
```

或等价移除这一 dependency。

---

# 35. 不影响 unaware Final ANN 的 Post Prefix path

Final unaware ANN 若使用 Prefix，仍是 Post-finetuning Prefix。

因此 unaware Final ANN 对：

```text
calibration.num_samples
```

的已有 path isolation 保持不变。

只移除：

```text
rotated_pre_finetuning
```

对 `calibration.num_samples` 的依赖。

---

# 36. Canonical manifest 推荐 write-or-validate

因为不同 `train_samples` 的 Step 2 会写到同一个 canonical path，推荐：

```text
不存在 → 写入
已存在 → 校验
一致 → 复用
不一致 → fail closed
```

不要静默覆盖不同 canonical selection。

---

# 37. 新增 canonical manifest validator

建议：

```python
validate_canonical_preprocessing_manifest_for_config(
    cfg,
    manifest,
)
```

校验：

```text
dataset_name
dataset_config_name
dataset_revision
experiment.seed
split
selection_seed_source
selection_seed
num_samples=128
sampling
duplicates_preserved
indices validity
```

---

# 38. `prepare_manifests()` canonical handling

推荐：

```text
1. 按 experiment.seed 算 expected canonical indices
2. canonical manifest 不存在 → 写
3. 已存在 → read
4. metadata + indices 与 expected 完全一致 → reuse
5. 不一致 → error
```

---

# 39. `--calibration-only` 不触碰 canonical

继续保持：

```text
--calibration-only
只读 Step-2 train manifest
只写 Stage-A calibration manifest
```

不创建、不覆盖 canonical manifest。

---

# 40. Pre Prefix 缺 canonical manifest 时 fail closed

`discover_prefix.py --stage pre_finetuning` 若缺：

```text
canonical_preprocessing/num_samples_128/calibration_manifest.json
```

直接报错：

```text
Pre-finetuning Prefix requires the fixed canonical 128 manifest.
Run Step 2 prepare_data.py first.
```

---

# 41. Pre Prefix 仍要求 Rotation artifacts

继续要求：

```text
rotation_state.pt
fused_base/config.json
```

存在。

Step 4 顺序仍然：

```text
Rotation
↓
Pre-finetuning Prefix
↓
rotated-pre-finetuning evaluation
```

---

# 42. `实验执行总结.md`：修正 Stage-A subset 表述

把：

```text
Calibration 128 ⊆ ANN Training 10000
```

改为：

```text
Stage-A Calibration 128 ⊆ 当前 Step 2 固定的 ANN Training subset
```

并补充：

```text
默认 train_samples=10000：
Stage-A Calibration 128 ⊆ ANN Training 10000

若 train_samples=N：
Stage-A Calibration 128 ⊆ ANN Training N
```

---

# 43. `实验执行总结.md`：明确两套 128

必须明确：

```text
1. fixed canonical 128
   - raw train
   - experiment.seed
   - Rotation regression
   - Pre-finetuning Prefix
   - 不要求属于 ANN training subset

2. Stage-A calibration 128
   - 当前 Step-2 ANN training subset
   - calibration.seed
   - ANN-training Stage A/B
   - 必须属于 ANN training subset
```

---

# 44. `实验执行总结.md`：Step 4 Prefix 表述

建议写：

> Pre-finetuning Prefix 使用与 Rotation regression 相同的 fixed canonical 128 samples。该 canonical subset 直接从 raw train 按 `experiment.seed` 固定、随机、无放回抽取，因此 Pre-finetuning Prefix 不依赖 task-specific training seed、`train_samples`、`calibration.seed` 或 `calibration.num_samples`。在 model-task、`experiment.seed`、Rotation、模型/tokenizer 与 Prefix 算法配置不变时，只需生成一次，并可被 unaware / phase-aware / gif-aware 以及所有 `train_samples` sweep 共享。

---

# 45. `实验执行总结.md`：Step 4 命令仍每个 model-task 一次

继续：

```bash
for CFG in "$CFG_17_U" "$CFG_8_U" "$CFG_L_U"; do
  python scripts/discover_prefix.py     --config "$CFG"     --stage pre_finetuning
done
```

明确：

```text
无需为 2k / 5k / 10k train_samples 分别重跑。
```

---

# 46. `train_samples` sweep 重跑表更新

新的表必须是：

| Step | `train_samples` 改变后 | 说明 |
|---|---|---|
| Step 1 | 需要准备对应 config | 新 ANN run identity |
| Step 2 | **必须完整重跑** | 新 training manifest / Stage-A calibration |
| Step 3 Base | 不用 | Base 不依赖 train samples |
| Step 4 Rotation | **不用** | fixed canonical 128 |
| Step 4 Pre-finetuning Prefix | **不用** | fixed canonical 128 |
| Step 4 rotated-pre-finetuning eval | **不用** | rotated Base + fixed Prefix 均不变 |
| Step 5 Stage A/B | **必须重跑** | Stage-A calibration 改变 |
| Step 5.1 GIF MSE | 启用时必须重跑 | 依赖 Stage-A calibration |
| Step 6 ANN training | **必须重跑** | training subset 改变 |
| Step 7 Post Prefix | **必须重跑** | final checkpoint 改变 |
| Step 8 Post Stage A | **必须重跑** | checkpoint / calibration 改变 |
| Step 9 Final ANN eval | **必须重跑** | checkpoint 改变 |
| Step 10 SNN conversion/eval | **必须重跑** | run-specific artifacts 改变 |

---

# 47. `calibration.seed` sweep 的共享行为

只要：

```text
experiment.seed
```

不变：

```text
canonical 128
Rotation
Rotation regression
Pre-finetuning Prefix
rotated-pre-finetuning evaluation
```

全部复用。

Stage-A chain 按新的 calibration seed 隔离。

---

# 48. `calibration.num_samples` sweep 的共享行为

例如：

```text
128
256
512
```

只影响：

```text
Stage-A calibration
ANN-training Stage A/B
run-specific Post Prefix / Post Stage A
```

不影响：

```text
canonical 128
Rotation regression
Pre Prefix
```

---

# 49. Prefix provenance 测试

Pre Prefix：

```text
discovery_num_samples=128
discovery_data_source=canonical_preprocessing_calibration
manifest=canonical manifest
```

Post Prefix：

```text
discovery_num_samples=calibration.num_samples
discovery_data_source=stage_a_calibration_selection
manifest=Stage-A manifest
```

---

# 50. 推荐路径测试

改变：

```text
train_samples
task train seed
calibration.seed
calibration.num_samples
```

必须保证：

```python
ann_training_prefix_dir
```

不变。

同时：

```python
ann_training_calibration_dir
```

按当前 dependency 正确变化。

---

# 51. 推荐 canonical independence 测试

保持 `experiment.seed` 不变，分别改变：

```text
train_samples
task train seed
calibration.seed
calibration.num_samples
```

都必须：

```python
canonical_preprocessing_calibration_manifest_path
```

相同。

---

# 52. 推荐 Rotation regression independence 测试

保持：

```text
experiment.seed
```

不变时：

```python
rotation_regression_path
```

不应因：

```text
train_samples
task train seed
calibration.seed
calibration.num_samples
```

改变。

---

# 53. 推荐 rotated-pre-finetuning path 测试

上述参数变化时：

```python
rotated_pre_finetuning_dir
```

应相同。

---

# 54. 推荐 Pre/Post Prefix validator 测试

Pre：

```text
canonical manifest → pass
Stage-A manifest → fail
```

Post：

```text
Stage-A manifest → pass
canonical manifest → fail
```

---

# 55. `verify_artifacts.py`

检查所有：

```text
Pre Prefix path/provenance
Rotation regression path
canonical manifest path
rotated-pre-finetuning path
```

expectation。

Pre 与 Post Prefix 不能再使用同一 Stage-A provenance 规则。

---

# 56. 不修改 manifest-only training

此前修好的：

```text
Step 2 train_manifest
→ Step 6 only consumes manifest indices
```

必须保持。

不得恢复：

```text
use_configured_train_subset
runtime random.sample
```

---

# 57. 全仓库 grep

修改后：

```bash
grep -R "canonical_preprocessing" -n snn2 scripts tests
grep -R "calibration_seed_" -n snn2 scripts tests *.md
grep -R "ann_training_prefix_dir" -n snn2 scripts tests
grep -R "validate_prefix_discovery_state" -n snn2 scripts tests
grep -R "discovery_data_source" -n snn2 scripts tests
grep -R "rotated_pre_finetuning_dir" -n snn2 scripts tests
grep -R "evaluation_depends_on_prefix_num_samples" -n snn2 scripts tests
```

确认旧 dependency 已清理。

---

# 58. 测试

必须：

```bash
pytest -q
```

全部通过。

重点：

```bash
pytest -q tests/test_generated_configs.py
pytest -q tests/test_evaluation_paths.py
pytest -q tests/test_rotated_pre_finetuning_protocol.py
pytest -q tests/test_post_finetuning_protocol.py
pytest -q tests/test_tulu3_lm_eval_protocol.py
pytest -q tests/test_verify_artifacts.py
```

---

# 59. 本轮不要修改

不要修改：

- Rotation 数学；
- Hadamard fusion；
- Rotation pass threshold；
- PrefixQuant outlier detection 算法；
- `PrefixOutlierCollector` 数学；
- Phase/GIF/MTN neuron math；
- Stage-A calibration 数学；
- common Clip；
- manifest-only ANN training；
- Post-finetuning Prefix 算法；
- SNN conversion selector；
- lm-eval / TL;DR evaluation metric。

---

# 60. 最终验收语义

对于：

```text
train_samples=2000
train_samples=5000
train_samples=10000
```

三条实验：

```text
共享：
- canonical 128
- Rotation weights
- Rotation regression
- Pre-finetuning Prefix
- rotated-pre-finetuning evaluation

独立：
- Step-2 train manifest
- Stage-A calibration
- ANN-training Stage A/B
- ANN checkpoint
- Post-finetuning Prefix
- Post-finetuning Stage A
- Final ANN eval
- SNN conversion/eval
```

---

# 61. `实验执行总结.md` 必须明确写出的最终结论

请加入或等价明确表述：

> 项目存在两套不同用途的数据选择。第一套是 **fixed canonical 128**：直接从 raw train 中按 `experiment.seed` 固定、随机、无放回抽取 128 条，用于 Rotation regression 与 Pre-finetuning Prefix discovery；它不依赖 task-specific training seed、`train_samples`、`calibration.seed` 或 `calibration.num_samples`。因此在 model-task、`experiment.seed`、模型/Rotation/Prefix 算法配置不变时，Rotation、Rotation regression、Pre-finetuning Prefix 与 rotated-pre-finetuning evaluation 都只需执行一次，并可被所有 `train_samples` sweep 共享。
>
> 第二套是 **Stage-A calibration subset**：先由 Step 2 根据 task-specific training seed 与 `train_samples` 固定 ANN training subset，再从该 subset 内按 `calibration.seed` 抽取 `calibration.num_samples` 条。该 subset 用于 ANN-training Stage A/B，并随 data-selection identity 改变。
>
> **只要 sweep 的 `train_samples` 改变，都应该把它视为一条新的数据选择 → calibration → training → conversion 实验链。** 新 sample count 必须重新执行 Step 2、Step 5、Step 6、Step 7、Step 8、Step 9、Step 10；若启用 GIF MSE，则相应 MSE calibration 也必须重跑。但 Step 3 Base baseline 与 Step 4 的 Rotation、fixed canonical Pre-finetuning Prefix、rotated-pre-finetuning evaluation 均可复用，无需因 `train_samples` 改变而重新执行。
