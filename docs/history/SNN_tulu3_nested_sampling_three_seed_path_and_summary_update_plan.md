# Tulu-3 训练/Calibration 嵌套抽样、三 Seed Run Path 与实验执行总结更新方案

> 目标仓库：`https://github.com/wangwk699/SNN`
>
> 本文档用于指导 Codex 在没有额外上下文的情况下完成本轮修改。
>
> 本轮目标：
>
> 1. 将 Tulu-3 的 Stage-A calibration 数据抽样逻辑改为与 TL;DR 一致：**先固定 ANN training subset，再从该 training subset 内抽 calibration subset**。
> 2. Tulu-3 的 ANN run identity 不再只使用 `seed42`，而是同时显式编码：
>    - `experiment.seed`
>    - `training.train_seed`
>    - `calibration.seed`
> 3. TL;DR 的 run path 本轮不修改。
> 4. 更新 `实验执行总结.md`，特别是 Step 2 和附录，写清四种 seed 的真实作用以及 TL;DR / Tulu-3 的“先 10000，再从这 10000 中抽 128”关系。
> 5. `canonical_preprocessing_calibration` 保持现有独立逻辑，不改为 ANN 10000 的子集。
> 6. 修改完成后用户会将旧目录：
>
>    ```text
>    SNN/artifacts/snn2_main_v1/tulu3
>    ```
>
>    rename 为：
>
>    ```text
>    SNN/artifacts/snn2_main_v1/tulu3_old
>    ```
>
>    然后从 `实验执行总结.md` 的 **Step 2：准备固定数据 manifest** 开始重新执行 Tulu-3 实验。
>
> 本轮不要实现旧 artifact fallback，也不要自动读取 `tulu3_old`。

---

# 1. 当前问题

当前 Tulu-3 数据关系为：

```text
raw train
    │
    │ experiment.seed
    ▼
fixed validation + validation-excluded shared training pool
    │
    ├── training.train_seed
    │      └── ANN training 10000
    │
    └── calibration.seed
           └── Stage-A calibration 128
```

因此当前：

```text
Calibration 128
```

与：

```text
ANN Training 10000
```

只是从同一个 shared pool 独立抽样。

当前代码没有保证：

```text
Calibration 128 ⊆ ANN Training 10000
```

而 TL;DR 已经是：

```text
raw train
    │
    │ training.tldr_train_seed
    ▼
ANN training 10000
    │
    │ calibration.seed
    ▼
Stage-A calibration 128
```

本轮要让 Tulu-3 的 **Stage-A calibration** 与 TL;DR 使用相同的嵌套抽样语义。

---

# 2. 修改后的 Tulu-3 数据关系

最终必须变为：

```text
Tulu-3 raw train
        │
        │ experiment.seed
        ▼
fixed validation
+
validation-excluded shared training pool
        │
        │ training.train_seed
        ▼
ANN Training subset
当前为 10000 samples
        │
        │ calibration.seed
        ▼
Stage-A Calibration subset
当前为 128 samples
```

因此必须满足：

\[
\mathcal D_{\mathrm{calibration}}
\subseteq
\mathcal D_{\mathrm{ANN\ train}}
\subseteq
\mathcal D_{\mathrm{validation\ excluded\ pool}}.
\]

当前默认实验规模下：

\[
|\mathcal D_{\mathrm{ANN\ train}}| = 10000,
\qquad
|\mathcal D_{\mathrm{calibration}}| = 128.
\]

也就是：

> Tulu-3 先从 validation-excluded shared pool 中按 `training.train_seed` 无放回抽 10000 条，再从这 10000 条中按 `calibration.seed` 无放回抽 128 条。

---

# 3. TL;DR 保持现有数据逻辑

TL;DR 本轮不要修改抽样算法。

继续保持：

```text
TL;DR raw train
       │
       │ training.tldr_train_seed
       ▼
ANN Training subset
当前为 10000 samples
       │
       │ calibration.seed
       ▼
Stage-A Calibration subset
当前为 128 samples
```

即：

\[
\mathcal D_{\mathrm{calibration}}
\subseteq
\mathcal D_{\mathrm{ANN\ train}}.
\]

---

# 4. 不要混淆 Stage-A calibration 128 与 canonical preprocessing 128

项目当前存在两类“128 samples”。

## 4.1 Stage-A calibration

这是本轮要修改 Tulu-3 抽样关系的对象。

它用于：

- Pre-finetuning Prefix discovery；
- ANN-training Stage A calibration；
- Post-finetuning Prefix / calibration 所引用的数据选择；
- Phase/GIF/MTN calibration states；
- aware ANN training 的 calibration provenance；
- SNN conversion provenance。

修改后：

```text
TL;DR Stage-A calibration 128 ⊆ TL;DR ANN train 10000
Tulu-3 Stage-A calibration 128 ⊆ Tulu-3 ANN train 10000
```

## 4.2 `canonical_preprocessing_calibration`

保持现有逻辑不变。

当前：

```python
CANONICAL_PREPROCESSING_NUM_SAMPLES = 128
```

它通过：

```python
load_canonical_preprocessing_raw(...)
```

供：

```text
scripts/prepare_rotation.py
```

的 rotation regression 使用。

本轮不要把它改成 ANN 10000 的子集。

文档中必须明确：

> “先 10000，再从这 10000 中抽 128”仅指 Stage-A calibration selection，不包括 canonical preprocessing calibration。

---

# 5. `snn2/data.py`：Tulu-3 calibration pool 修改

当前 `prepare_calibration_manifest()` 中：

```python
raw_train, train_split, train_indices, _, _, _ = _manifest_split_selection(
    cfg, raw
)

calibration_pool_indices = train_indices

if cfg["experiment"]["task"] == "tulu3":
    _, calibration_pool_indices = _tulu3_shared_split_selection(
        raw_train, cfg
    )
```

这里 Tulu-3 特殊覆盖了：

```text
calibration_pool_indices
```

导致 calibration 绕过 ANN 10000，回到整个 validation-excluded shared pool。

## 必须删除该特殊覆盖

修改后统一：

```python
raw_train, train_split, train_indices, _, _, _ = _manifest_split_selection(
    cfg, raw
)

calibration_pool_indices = train_indices
```

然后：

```python
calibration_positions, calibration_indices = _calibration_selection(
    calibration_pool_indices,
    seed=int(cfg["calibration"]["seed"]),
    num_samples=calibration_num_samples,
    with_replacement=with_replacement,
)
```

这样：

```text
positions_in_selected_train
```

在 TL;DR 与 Tulu-3 中都具有一致语义：

> calibration sample 在当前 ANN training subset 中的位置。

---

# 6. `prepare_manifests()` 同样修改

当前 `prepare_manifests()` 里也存在：

```python
calibration_pool_indices = train_indices

if task == "tulu3":
    _, calibration_pool_indices = _tulu3_shared_split_selection(
        raw_train, cfg
    )
```

同样必须删除 Tulu-3 特殊覆盖。

统一为：

```python
calibration_pool_indices = train_indices
```

然后从 `train_indices` 内调用：

```python
_calibration_selection(...)
```

---

# 7. `prepare_data.py --calibration-only` 必须与完整 Step 2 完全一致

以下两种方式：

```bash
python scripts/prepare_data.py --config "$CFG_L_V"
```

和：

```bash
python scripts/prepare_data.py   --config "$CFG_L_V"   --calibration-only
```

在相同配置下生成的 Stage-A calibration selection 必须完全一致。

不能出现：

```text
完整 prepare_manifests:
    calibration 从 ANN 10000 抽

--calibration-only:
    calibration 仍从 shared pool 抽
```

因此必须同时修改：

```text
prepare_manifests()
prepare_calibration_manifest()
```

---

# 8. Tulu-3 calibration manifest metadata 更新

当前 Tulu-3 calibration manifest 中类似：

```json
{
  "selection_pool": "validation_excluded_shared_training_pool",
  "retained_in_shared_training_pool": true,
  "retained_in_ann_training_subset": null
}
```

修改后这个 provenance 已经过时。

推荐改为：

```json
{
  "selection_pool": "selected_ann_training_subset",
  "retained_in_shared_training_pool": true,
  "retained_in_ann_training_subset": true
}
```

要求：

1. 必须明确 calibration 是从 ANN training subset 中抽样；
2. `retained_in_ann_training_subset` 必须为 `true`；
3. 保留 `calibration_seed`；
4. 保留 `positions_in_selected_train`，且它实际是相对于 ANN training subset 的 position；
5. `indices` 仍记录 raw train split 中的实际 index；
6. `record_ids` 继续与这些 indices 一一对应。

---

# 9. Tulu-3 train manifest 继续记录 train seed

继续包含：

```json
{
  "train_samples": 10000,
  "train_seed": 42,
  "validation_size": 1000,
  "selection_scope": "current_ann_training_config"
}
```

不要删除这些 provenance。

---

# 10. 推荐增加 calibration parent training provenance

可选但推荐在 calibration manifest 中增加：

```json
{
  "parent_training_samples": 10000,
  "parent_training_seed": 42
}
```

TL;DR 的 parent seed 来自：

```text
training.tldr_train_seed
```

Tulu-3 来自：

```text
training.train_seed
```

如果不希望扩大 schema，本项可以不做；核心要求仍是：

```text
calibration.indices ⊆ train.indices
```

---

# 11. Tulu-3 ANN run path：编码三种 seed

当前：

```python
seed = f"seed{int(exp['seed'])}"
...
self.root = run_root / seed
```

导致 Tulu-3 ANN checkpoint 类似：

```text
.../<calibration_trajectory>/seed42/ann/
```

本轮要求 Tulu-3 run identity 同时编码：

```text
experiment.seed
training.train_seed
calibration.seed
```

---

# 12. 推荐 Tulu-3 seed dirname

统一使用：

```text
experiment_seed_<E>_train_seed_<T>_calibration_seed_<C>
```

例如：

```text
experiment_seed_42_train_seed_42_calibration_seed_42
```

因此 ANN checkpoint 变为：

```text
.../
phase_previous_layers_snn_.../
experiment_seed_42_train_seed_42_calibration_seed_42/
ann/
final/
```

不要使用：

```text
seed42_42_42
```

这种无法直接辨认字段含义的缩写。

---

# 13. TL;DR run path 本轮完全不变

TL;DR 继续：

```text
.../seed42/ann/
```

不要改成 composite seed dirname。

---

# 14. 推荐增加 `run_seed_dirname()`

例如：

```python
def run_seed_dirname(cfg: dict[str, Any]) -> str:
    exp_seed = int(cfg["experiment"]["seed"])

    if cfg["experiment"]["task"] == "tulu3":
        train_seed = int(cfg["training"]["train_seed"])
        calibration_seed = int(cfg["calibration"]["seed"])
        return (
            f"experiment_seed_{exp_seed}_"
            f"train_seed_{train_seed}_"
            f"calibration_seed_{calibration_seed}"
        )

    return f"seed{exp_seed}"
```

然后：

```python
self.root = run_root / run_seed_dirname(cfg)
```

---

# 15. 不要机械替换所有现有 `seed` 路径

同一个旧 `seed` 还用于：

```text
base_root
shared_task_root
shared_model_root
```

它们的 dependency 与 concrete ANN run 不完全相同。

建议拆成：

```python
experiment_seed_name = f"seed{int(exp['seed'])}"
run_seed_name = run_seed_dirname(cfg)
```

其中：

```python
self.root = run_root / run_seed_name
```

而真正只依赖 experiment seed 的 artifact 保留：

```python
experiment_seed_name
```

---

# 16. Base baseline 路径不要绑定 train/calibration seed

Base baseline 不依赖 ANN training subset。

因此继续：

```text
.../base/seed42/
```

即：

```python
self.base_root = model_root / "base" / experiment_seed_name
```

---

# 17. Run-local downstream artifact 应跟随 composite root

不要只特殊改：

```python
ann_dir
```

正确结构应是：

```text
.../
experiment_seed_42_train_seed_42_calibration_seed_42/
├── ann/
├── post_finetuning/
├── snn/
├── analysis/
└── ...
```

否则会造成 ANN、post-finetuning、SNN 三套路径身份分裂。

---

# 18. Shared artifact 的 seed scope 必须避免 collision

新的 Tulu-3 Stage-A calibration subset 同时依赖：

```text
experiment.seed
training.train_seed
calibration.seed
```

因此必须检查所有由该 Stage-A data selection 派生且跨 mode 共享的 artifact：

```text
data manifest
Pre-finetuning Prefix
ANN-training Stage A
ANN-training Stage B
```

原则：

> 如果 artifact 内容依赖 `training.train_seed` 或 `calibration.seed`，其 identity 必须能区分这些 seed，不能在不同 seed 配置下静默覆盖同一路径。

---

# 19. 推荐 shared-data seed scope

建议 Tulu-3 的 data-dependent shared artifacts 使用同一 seed scope：

```text
experiment_seed_<E>_train_seed_<T>_calibration_seed_<C>
```

TL;DR 继续原有：

```text
seed<experiment.seed>
```

可以新增：

```python
def data_selection_seed_dirname(cfg: dict[str, Any]) -> str:
    if cfg["experiment"]["task"] == "tulu3":
        return (
            f"experiment_seed_{int(cfg['experiment']['seed'])}_"
            f"train_seed_{int(cfg['training']['train_seed'])}_"
            f"calibration_seed_{int(cfg['calibration']['seed'])}"
        )
    return f"seed{int(cfg['experiment']['seed'])}"
```

---

# 20. Rotation identity 不应无意义绑定 `training.train_seed`

Rotation fusion 本身不依赖：

```text
training.train_seed
```

因此不要为了路径统一让每个 train seed 都重新存一套 Rotation 权重。

推荐区分：

```text
rotation/model-policy identity
```

与：

```text
data selection / calibration identity
```

例如：

```text
_shared/
└── seed42/
    └── rotated_prefix/
        ├── rotation/
        └── <data-selection-seed-scope>/
            ├── pre_finetuning_prefix/
            └── ann_training_calibration/
```

实现可做最小调整，但必须保证：

- Rotation 不因 `training.train_seed` 无意义复制；
- Prefix / Stage A 不因不同 train/calibration seed 发生路径碰撞。

---

# 21. canonical preprocessing / rotation 逻辑保持不变

`canonical_preprocessing_calibration` 本轮不改变抽样来源与算法。

当前它仍由：

```text
calibration.seed
```

控制独立的 128 条 sample。

不要把它改成：

```text
ANN train 10000 的子集
```

也不要和 Stage-A calibration manifest 合并。

---

# 22. 四种 seed 的最终语义

`实验执行总结.md` 附录必须写清：

## `experiment.seed`

作用包括：

- Python `random` 全局 seed；
- NumPy seed；
- PyTorch seed；
- CUDA seed；
- Hugging Face `TrainingArguments.seed`；
- Hugging Face `TrainingArguments.data_seed`；
- Tulu-3 validation / validation-excluded shared pool 划分；
- experiment-level provenance。

## `training.tldr_train_seed`

只针对 TL;DR ANN training subset selection：

```text
raw TL;DR train
    → tldr_train_seed
    → ANN train subset
```

它不是 Trainer shuffle seed。

## `training.train_seed`

只针对 Tulu-3 ANN training subset selection：

```text
validation-excluded shared pool
    → train_seed
    → ANN train subset
```

它不是 Trainer shuffle seed。

## `calibration.seed`

控制 Stage-A calibration subset：

```text
ANN train subset
    → calibration.seed
    → Stage-A calibration subset
```

此外，当前代码中 `canonical_preprocessing_calibration` 也使用 `calibration.seed` 生成其独立 128 samples；附录必须单独注明。

---

# 23. “TL;DR 与 Tulu-3 都先 10000，再 128”的准确表述

建议附录写成：

> 在当前默认配置下，TL;DR 与 Tulu-3 的 **Stage-A calibration selection** 均采用嵌套抽样：先按任务对应的 training subset seed 无放回选取 10000 条 ANN training samples，再从这 10000 条中按 `calibration.seed` 无放回选取 128 条 Stage-A calibration samples。因此 Stage-A calibration subset 必然是 ANN training subset 的子集。这里的 128 不包括独立用于 rotation regression 的 `canonical_preprocessing_calibration` 128 samples。

并分别给出：

```text
TL;DR:
raw train
  └── tldr_train_seed → ANN train 10000
                         └── calibration.seed → Stage-A calibration 128
```

```text
Tulu-3:
raw train
  └── experiment.seed → fixed validation + shared pool
                         └── train_seed → ANN train 10000
                                          └── calibration.seed → Stage-A calibration 128
```

---

# 24. 更新 `实验执行总结.md` Step 2

当前旧表述：

```text
shared calibration 从 validation-excluded shared training pool 独立取样；
ANN training 再从同一 pool 按 train seed/sample 独立取样。
```

必须删除。

改成：

```text
Tulu-3 先在 experiment.seed 固定的 validation-excluded training pool 中，
按 training.train_seed 随机无放回选取 training.train_samples（当前 10000）；
Stage-A calibration 再从这批 ANN training samples 内按 calibration.seed
随机无放回选取 calibration.num_samples（当前 128）。
```

并明确：

```text
Calibration 128 ⊆ ANN Training 10000
```

---

# 25. 更新文档底部旧 Tulu shared artifact 关系

当前旧图：

```text
source train → fixed 1000 validation → shared training pool
                                  ├─ shared calibration / Pre-finetuning Prefix / shared Stage A
                                  └─ training.train_samples-specific ANN subset
```

必须更新为：

```text
source train
    ↓ experiment.seed
fixed 1000 validation + validation-excluded shared training pool
    ↓ training.train_seed
ANN training subset (10000)
    ↓ calibration.seed
Stage-A calibration subset (128)
    ↓
Pre-finetuning Prefix / shared Stage A
```

并删除旧结论：

```text
_shared/seed42/ 不包含 train_samples 或 train_seed；
这些字段只进入具体 ANN run path。
```

---

# 26. 新增附录 I

建议新增：

```text
## 附录 I：随机种子与训练/Calibration 数据抽样关系
```

至少包含：

| 参数 | 任务 | 作用 |
|---|---|---|
| `experiment.seed` | TL;DR / Tulu-3 | 全局训练随机性；Tulu-3 validation/shared-pool split；Trainer `seed/data_seed` |
| `training.tldr_train_seed` | TL;DR | 从 raw train 选 ANN training subset |
| `training.train_seed` | Tulu-3 | 从 validation-excluded shared pool 选 ANN training subset |
| `calibration.seed` | TL;DR / Tulu-3 | 从 ANN training subset 选 Stage-A calibration subset；另用于独立 canonical preprocessing 128 |

并加入两条数据链以及 Tulu-3 三 seed path 示例。

---

# 27. 文档路径示例同步更新

凡是 Tulu-3 concrete ANN run identity 仍写：

```text
seed42
```

的地方，应更新为：

```text
experiment_seed_42_train_seed_42_calibration_seed_42
```

但：

- TL;DR 示例仍保留 `seed42`；
- Base baseline 若代码仍只按 experiment seed 保存，则仍保留 `base/seed42`；
- Rotation-only shared path 不要误写成依赖 `training.train_seed`。

---

# 28. 写入旧 artifact migration note

可加入：

```text
本轮 Tulu-3 数据抽样/provenance 变更与旧 artifact 不兼容。
正式重跑前，将：

artifacts/snn2_main_v1/tulu3

改名保存为：

artifacts/snn2_main_v1/tulu3_old

然后从 Step 2：prepare_data 开始重新执行 Tulu-3 流程。
不要将 tulu3_old 中的 manifest、Prefix、Stage A/B、ANN checkpoint、
post-finetuning 或 SNN artifact 复制回新 tulu3 目录。
```

---

# 29. 数据抽样测试

至少新增：

## Test A：Tulu calibration 是 train subset 的子集

```python
assert len(train_indices) == 10000
assert len(calibration_indices) == 128
assert set(calibration_indices).issubset(set(train_indices))
```

## Test B：positions 语义正确

```python
assert [
    train_indices[position]
    for position in calibration_positions
] == calibration_indices
```

## Test C：只改 calibration.seed

必须：

```text
train_indices 不变
calibration_indices 改变
两组 calibration 都属于相同 train subset
```

## Test D：只改 training.train_seed

必须：

```text
train subset 改变
calibration 随新的 train subset 重新产生
calibration 仍属于对应 train subset
```

## Test E：完整 prepare 与 calibration-only 一致

比较：

```text
indices
positions_in_selected_train
record_ids
calibration_seed
num_samples
```

## Test F：canonical preprocessing 保持独立

不要新增：

```text
canonical_preprocessing ⊆ ANN train subset
```

这种错误约束。

---

# 30. Path 测试

## Test G：Tulu root 编码三个 seed

例如：

```yaml
experiment.seed: 42
training.train_seed: 43
calibration.seed: 44
```

断言：

```python
layout.root.name == (
    "experiment_seed_42_"
    "train_seed_43_"
    "calibration_seed_44"
)
```

## Test H：改任一 seed 都改变 Tulu concrete run root

分别修改：

```text
experiment.seed
training.train_seed
calibration.seed
```

都应改变：

```python
layout.root
```

## Test I：TL;DR path 不变

仍：

```python
layout.root.name == "seed42"
```

## Test J：Tulu downstream run-local path 同属 composite root

确认：

```text
ann/
post_finetuning/
snn/
analysis/
```

都挂在同一个 composite seed run root 下。

---

# 31. Shared artifact collision 测试

如果按推荐隔离 data-dependent shared artifacts，则验证：

改变：

```text
training.train_seed
```

或：

```text
calibration.seed
```

时，至少：

```text
calibration_data_manifest_path
ann_training_prefix_dir
ann_training_calibration_dir
```

不会静默覆盖同一路径。

同时：

```text
rotation_dir
```

不应仅因为 `training.train_seed` 改变而无意义改变。

---

# 32. Trainer seed 语义保持不变

继续：

```python
TrainingArguments(
    ...
    seed=int(cfg["experiment"]["seed"]),
    data_seed=int(cfg["experiment"]["seed"]),
)
```

不要改成 `training.train_seed`。

---

# 33. 不改 TL;DR train selection

继续使用：

```python
random.Random(
    int(cfg["training"].get("tldr_train_seed", 42))
).sample(...)
```

---

# 34. 不改 Tulu validation split

继续：

```python
random.Random(
    int(cfg["experiment"]["seed"])
).shuffle(permutation)
```

不要改成 `training.train_seed`。

---

# 35. 不改无放回 policy

继续：

```text
ANN train subset：无放回
Stage-A calibration subset：无放回
```

---

# 36. `training.train_samples: null`

仍支持：

```text
ANN training subset = 完整 validation-excluded shared pool
```

Stage-A calibration 再从这个实际 ANN training selection 中抽取。

统一原则：

```text
calibration pool = actual ANN training selection
```

---

# 37. `calibration.num_samples` sweep

当 training selection 不变，仅改变：

```text
calibration.num_samples
```

不同 calibration manifest 都应来自同一个 ANN train subset。

保留：

```text
data/calibration/num_samples_<N>/
```

隔离。

---

# 38. 推荐增加 invariant

建议在 manifest 构造或测试层确保：

```python
if not set(calibration_indices).issubset(set(train_indices)):
    raise RuntimeError(...)
```

更严格：

```python
[
    train_indices[position]
    for position in calibration_positions
] == calibration_indices
```

---

# 39. 检查 verifier

检查：

```text
scripts/verify_artifacts.py
```

是否仍期待：

```text
selection_pool = validation_excluded_shared_training_pool
retained_in_ann_training_subset = null
```

如有必须同步更新。

推荐验证：

```text
retained_in_ann_training_subset == true
calibration.indices ⊆ train_manifest.indices
```

---

# 40. Prefix / Stage A provenance 保持

继续保留：

```text
discovery_manifest_path
discovery_manifest_sha256
calibration manifest hash
```

新的 calibration manifest hash 应自然传播到后续 provenance。

---

# 41. 旧 Tulu artifact 不兼容是预期行为

用户会执行：

```bash
mv   /home/wangwenkang/SNN/artifacts/snn2_main_v1/tulu3   /home/wangwenkang/SNN/artifacts/snn2_main_v1/tulu3_old
```

执行前自行确认 `tulu3_old` 不已存在，避免覆盖。

代码不要自动访问旧目录。

---

# 42. 重跑边界

从 Step 2 开始重跑 Tulu-3 是正确的。

后续至少重新建立：

```text
Step 2 data manifests
Step 4 rotation regression / Pre-finetuning Prefix
Step 5 ANN-training Stage A/B
ANN fine-tuning
Final ANN evaluation
Post-finetuning Prefix
Post-finetuning Stage A
SNN conversion
SNN evaluation
```

---

# 43. 全仓库搜索

修改后建议：

```bash
grep -R 'validation_excluded_shared_training_pool' -n snn2 scripts tests *.md
grep -R 'retained_in_ann_training_subset' -n snn2 scripts tests *.md
grep -R 'train_seed' -n snn2 scripts tests *.md
grep -R 'tldr_train_seed' -n snn2 scripts tests *.md
grep -R 'calibration_seed' -n snn2 scripts tests *.md
grep -R 'seed42' -n 实验执行总结.md tests snn2
grep -R 'self.root = run_root' -n snn2
```

---

# 44. 本轮不要修改

不要修改：

- TL;DR sampling algorithm；
- TL;DR concrete run seed path；
- Tulu-3 validation split algorithm；
- Hugging Face Trainer `seed/data_seed` 来源；
- calibration without-replacement policy；
- `calibration.num_samples` path isolation；
- canonical preprocessing sampling algorithm；
- Phase/GIF/MTN calibration 数学；
- Stage A/B dependency；
- Phase base/T sweep；
- Prefix 算法；
- lm-eval task selection；
- evaluation test seed；
- model architecture；
- training hyperparameters。

---

# 45. `实验执行总结.md` 必须消除正文/附录矛盾

至少同步修改：

1. Step 2 的旧 Tulu calibration 描述；
2. 底部旧 Tulu shared artifact 关系；
3. Tulu concrete run path 中单独 `seed42` 的示例；
4. 新增附录 I 四种 seed 总结。

不要只追加附录而保留正文旧逻辑。

---

# 46. 最终验收矩阵

| 项目 | TL;DR | Tulu-3 |
|---|---|---|
| ANN subset seed | `training.tldr_train_seed` | `training.train_seed` |
| ANN subset 默认规模 | 10000 | 10000 |
| Stage-A calibration seed | `calibration.seed` | `calibration.seed` |
| Stage-A calibration 默认规模 | 128 | 128 |
| Calibration pool | ANN subset | **ANN subset** |
| `calibration ⊆ ANN train` | 是 | **是** |
| Trainer seed | `experiment.seed` | `experiment.seed` |
| Validation/shared-pool split | dataset 原有 split | `experiment.seed` |
| run seed dirname | `seed42` | **三 seed composite dirname** |
| canonical preprocessing 128 | 独立 | 独立 |

---

# 47. 测试

必须：

```bash
pytest -q
```

建议额外：

```bash
pytest -q tests/test_data.py
pytest -q tests/test_post_finetuning_protocol.py
pytest -q tests/test_verify_artifacts.py
```

实际测试文件名以仓库为准。

---

# 48. 最终实际路径验收

当前 Tulu 默认：

```yaml
experiment:
  seed: 42

training:
  train_samples: 10000
  train_seed: 42

calibration:
  num_samples: 128
  seed: 42
```

ANN run 必须出现：

```text
.../
experiment_seed_42_train_seed_42_calibration_seed_42/
ann/
```

而不是：

```text
.../seed42/ann/
```

TL;DR 仍保持：

```text
.../seed42/ann/
```

---

# 49. 最终 manifest 验收

Tulu Step 2 后：

```python
len(train_manifest["indices"]) == 10000
len(calibration_manifest["indices"]) == 128

set(calibration_manifest["indices"]).issubset(
    set(train_manifest["indices"])
)
```

并：

```python
[
    train_manifest["indices"][position]
    for position in calibration_manifest["positions_in_selected_train"]
] == calibration_manifest["indices"]
```

---

# 50. 最终设计总结

修改后：

```text
TL;DR
raw train
  → tldr_train_seed
  → ANN 10000
  → calibration.seed
  → Stage-A 128
```

以及：

```text
Tulu-3
raw train
  → experiment.seed
  → fixed validation + shared pool
  → train_seed
  → ANN 10000
  → calibration.seed
  → Stage-A 128
```

Tulu-3 concrete run path 同时记录：

```text
experiment.seed
training.train_seed
calibration.seed
```

例如：

```text
experiment_seed_42_train_seed_42_calibration_seed_42
```

TL;DR 路径本轮仍为：

```text
seed42
```

`实验执行总结.md` 必须在 Step 2 与新增附录中同时反映这一事实，并明确：

> “10000 → 128”只指 Stage-A calibration；独立用于 rotation regression 的 `canonical_preprocessing_calibration` 128 不属于该嵌套抽样关系。
