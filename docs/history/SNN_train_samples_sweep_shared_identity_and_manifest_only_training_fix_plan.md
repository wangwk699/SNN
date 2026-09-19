# TL;DR + Tulu-3 `train_samples` Sweep：Shared Identity + Manifest-Only Training 完整修正方案

> 目标仓库：`https://github.com/wangwk699/SNN`
>
> 本文档用于指导 Codex 在没有额外上下文的情况下完成本轮修改。
>
> 本轮合并修复两类问题：
>
> 1. **TL;DR 与 Tulu-3 在 sweep 不同 `train_samples` 时，shared data / Prefix / ANN-training calibration artifact identity 冲突；**
> 2. **ANN training 当前仍可绕过 Step 2 已生成的 `train_manifest.json`，根据 YAML 中的 seed / train_samples 在训练阶段再次重采样。**
>
> 用户已确认：
>
> - 不兼容旧 TL;DR shared artifacts；
> - 直接采用新的完整 data-selection identity；
> - 所有 ANN training 使用的数据必须事先通过修改 YAML 并执行 Step 2 固化到 `train_manifest.json`；
> - ANN training 阶段禁止重新采样、替换或忽略 `train_manifest.json["indices"]`；
> - `--calibration-only` 也只能基于已有 Step 2 `train_manifest.json` 工作，不能自己重新生成 training subset。
>
> `实验执行总结.md` 中必须明确写明：
>
> **只要 sweep 的 `train_samples` 改变，都应该把它视为一条新的数据选择 → calibration → training → conversion 实验链。**

---

# 1. 本轮最终原则

## 1.1 Step 2 是 ANN training 数据选择的唯一入口

TL;DR：

```text
修改 YAML:
training.tldr_train_samples
training.tldr_train_seed
        ↓
Step 2: prepare_data.py
        ↓
固定 train_manifest.json
        ↓
Step 5 calibration / Prefix
        ↓
Step 6 ANN training
```

Tulu-3：

```text
修改 YAML:
training.train_samples
training.train_seed
        ↓
Step 2: prepare_data.py
        ↓
固定 train_manifest.json
        ↓
Step 5 calibration / Prefix
        ↓
Step 6 ANN training
```

一旦 Step 2 完成：

> **所有后续阶段只能消费 manifest，不允许重新决定训练 indices。**

---

# 2. 当前训练阶段仍存在的错误旁路

当前 `snn2/training.py`：

```python
bundle = load_selected_raw(
    cfg, layout, use_configured_train_subset=True
)
```

而当前 `snn2/data.py::load_selected_raw()` 中：

```python
if (
    name == "train"
    and use_configured_train_subset
    and cfg["experiment"]["task"] in {"tldr", "tulu3"}
):
    ...
```

会忽略原 manifest 已有：

```python
manifest["indices"]
```

并重新执行：

TL;DR：

```python
indices, sampling = _tldr_train_selection(
    raw_train, cfg
)
```

Tulu-3：

```python
permutation = list(range(len(raw_train)))
random.Random(
    int(cfg["experiment"]["seed"])
).shuffle(permutation)

indices, sampling = _tulu3_train_selection(
    permutation[
        int(cfg["data"]["validation_size"]):
    ],
    cfg,
)
```

这条逻辑必须删除。

---

# 3. 为什么训练阶段二次采样不允许存在

假设 Step 2 已经生成：

```text
train_manifest.json
```

其中：

```json
{
  "indices": [...]
}
```

那么这个文件应代表：

```text
本次实验唯一、固定、可复现的 ANN training dataset selection
```

如果 Step 6 又根据：

```text
experiment.seed
train_seed
train_samples
```

重新 `random.sample()`：

```text
YAML
 ↓
runtime resampling
 ↓
新的 training indices
```

则会导致：

- Step 2 manifest 与真实训练数据不再是同一对象；
- Stage-A calibration 是从 manifest training subset 中抽的，但 ANN training 可能使用另一套 runtime subset；
- Prefix / calibration / training provenance 断裂；
- artifact verifier 即使验证了 manifest，也无法保证模型真正用过这套数据；
- sweep `train_samples` 时无法确保每个 experiment 的训练数据由 Step 2 唯一定义。

因此必须彻底移除。

---

# 4. 正确的 source-of-truth 规则

对于所有 TL;DR / Tulu-3 ANN training：

```text
train_manifest.json["indices"]
```

必须是：

```text
唯一 ANN training indices
```

后续：

```text
train_ann.py
training.py
load_selected_raw()
tokenize_dataset()
Trainer
```

都必须沿用这套 indices。

---

# 5. 修改 `snn2/training.py`

当前：

```python
bundle = load_selected_raw(
    cfg,
    layout,
    use_configured_train_subset=True,
)
```

改为：

```python
bundle = load_selected_raw(
    cfg,
    layout,
)
```

更推荐在 `load_selected_raw()` 中彻底删除：

```text
use_configured_train_subset
```

参数，因此最终可以简化成：

```python
bundle = load_selected_raw(cfg, layout)
```

---

# 6. 修改 `load_selected_raw()`

当前签名：

```python
def load_selected_raw(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    *,
    use_configured_train_subset: bool = False,
) -> DatasetBundle:
```

改为：

```python
def load_selected_raw(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
) -> DatasetBundle:
```

---

# 7. 删除 training-time resampling 分支

整个：

```python
if (
    name == "train"
    and use_configured_train_subset
    and cfg["experiment"]["task"] in {"tldr", "tulu3"}
):
    ...
else:
    selected[name] = raw[
        manifest["split"]
    ].select(manifest["indices"])
```

改为统一：

```python
for name, manifest in manifests.items():
    selected[name] = raw[
        manifest["split"]
    ].select(manifest["indices"])
```

也就是说：

```text
train
validation
calibration
evaluation
```

全部由各自 manifest 决定。

---

# 8. 禁止在 ANN training 阶段调用 `_tldr_train_selection()`

`_tldr_train_selection()` 只能用于：

```text
Step 2 prepare_manifests()
```

或其他显式数据准备逻辑。

禁止用于：

```text
Step 6 ANN training
```

---

# 9. 禁止在 ANN training 阶段调用 `_tulu3_train_selection()`

同理，`_tulu3_train_selection()` 只能用于 Step 2。

以下流程：

```python
permutation = list(range(len(raw_train)))
random.Random(experiment.seed).shuffle(permutation)
training_pool = ...
indices = _tulu3_train_selection(...)
```

不能再出现在 ANN training data load 路径中。

---

# 10. ANN training 必须 fail closed 校验当前 YAML 与 train manifest 一致

仅仅“读取 manifest”还不够。

训练启动前必须检查：

```text
当前 YAML 的 data-selection identity
==
train_manifest 对应的 identity
```

若不一致：

```text
直接报错
```

不能自动重新采样。

---

# 11. TL;DR train manifest 校验

当前 TL;DR manifest 已记录：

```text
tldr_train_samples
tldr_train_seed
```

训练前必须校验：

```python
manifest["tldr_train_samples"]
==
cfg["training"].get("tldr_train_samples")
```

以及：

```python
int(manifest["tldr_train_seed"])
==
int(cfg["training"].get("tldr_train_seed", 42))
```

如果：

```text
tldr_train_samples = N
```

为整数：

```python
len(manifest["indices"]) == N
```

必须成立。

若：

```text
tldr_train_samples = None
```

则 manifest 应对应 full split。

---

# 12. Tulu-3 train manifest 校验

训练前必须校验：

```python
manifest["train_samples"]
==
cfg["training"].get("train_samples")
```

```python
int(manifest["train_seed"])
==
int(cfg["training"].get("train_seed", 42))
```

```python
int(manifest["validation_size"])
==
int(cfg["data"]["validation_size"])
```

若配置为整数 N：

```python
len(manifest["indices"]) == N
```

必须成立。

若 `train_samples=None`：

```text
应对应 validation-excluded full training pool
```

---

# 13. 推荐新增统一 validator

推荐在 `snn2/data.py` 中新增：

```python
def validate_train_manifest_for_config(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    ...
```

用于：

- ANN training；
- `--calibration-only`；
- verifier；
- 其他依赖 fixed training subset 的路径。

不要在多个文件复制相同检查。

---

# 14. 推荐 validator 失败信息

例如：

```text
Training manifest does not match the current configuration.
The ANN training subset must be prepared by Step 2.
Re-run:
python scripts/prepare_data.py --config <config>
```

重点是：

> 不允许 fallback 到 runtime sampling。

---

# 15. 当前 `prepare_calibration_manifest()` 也需要修改

当前：

```python
raw = _load_raw(cfg)

raw_train, train_split, train_indices, _, _, _ = (
    _manifest_split_selection(cfg, raw)
)
```

这意味着：

```text
--calibration-only
```

仍然会自己根据 YAML：

```text
train_seed
train_samples
```

重新构造 ANN training subset。

这也必须删除。

---

# 16. `--calibration-only` 的新语义

`--calibration-only` 必须：

```text
已有 Step 2 train_manifest.json
        ↓
读取 train_manifest.json["indices"]
        ↓
验证 manifest 与 YAML 完全一致
        ↓
只从已有 train indices 中抽 calibration subset
        ↓
写 calibration_manifest.json
```

不能：

```text
重新运行 _manifest_split_selection()
```

---

# 17. 新 `prepare_calibration_manifest()` 推荐结构

逻辑应类似：

```python
def prepare_calibration_manifest(
    cfg,
    layout,
):
    train_manifest_path = (
        layout.data_dir
        / "train_manifest.json"
    )

    if not train_manifest_path.exists():
        raise FileNotFoundError(
            "Training manifest is missing. "
            "Run Step 2 prepare_data.py without "
            "--calibration-only first."
        )

    train_manifest = read_json(
        train_manifest_path
    )

    validate_train_manifest_for_config(
        cfg,
        train_manifest,
    )

    raw = _load_raw(cfg)
    raw_train = raw[
        train_manifest["split"]
    ]
    train_indices = [
        int(i)
        for i
        in train_manifest["indices"]
    ]

    calibration_positions, calibration_indices = (
        _calibration_selection(
            train_indices,
            seed=int(
                cfg["calibration"]["seed"]
            ),
            num_samples=int(
                cfg["calibration"]["num_samples"]
            ),
            with_replacement=bool(
                cfg["calibration"].get(
                    "with_replacement",
                    False,
                )
            ),
        )
    )

    ...
```

---

# 18. `--calibration-only` 缺少 train manifest 时必须报错

例如：

```text
FileNotFoundError:
Step 2 training manifest is missing.
Run:
python scripts/prepare_data.py --config <config>
before using --calibration-only.
```

不要自动 fallback。

---

# 19. `--calibration-only` 遇到 YAML / manifest mismatch 必须报错

例如：

Step 2：

```yaml
train_samples: 10000
```

后来用户直接把 YAML 改成：

```yaml
train_samples: 20000
```

但没有重新执行完整 Step 2。

此时运行：

```bash
prepare_data.py --calibration-only
```

必须 fail：

```text
configured train_samples=20000
manifest train_samples=10000
```

而不是临时生成 20000 subset。

---

# 20. train_samples sweep 的唯一正确流程

例如 Tulu-3：

```text
train_samples=2000
修改 YAML
↓
Step 1 config
↓
Step 2 full prepare_data
↓
生成 2000-specific train manifest
↓
后续实验链

train_samples=5000
修改 YAML
↓
Step 1 config
↓
Step 2 full prepare_data
↓
生成 5000-specific train manifest
↓
后续实验链

train_samples=10000
修改 YAML
↓
Step 1 config
↓
Step 2 full prepare_data
↓
生成 10000-specific train manifest
↓
后续实验链
```

禁止：

```text
只跑一次 Step 2
↓
Step 6 根据 YAML 临时 sample 2000 / 5000 / 10000
```

---

# 21. TL;DR 同样严格适用

例如：

```text
tldr_train_samples=2000
↓
Step 2
↓
2000 train manifest

tldr_train_samples=5000
↓
Step 2
↓
5000 train manifest
```

每个 sample count 都有独立固定 manifest。

---

# 22. Shared identity 修正仍然必须保留

删除 runtime resampling 不能替代路径隔离。

两类修改都必须完成：

```text
A. manifest-only training semantics
B. train_samples-aware artifact identity
```

缺一不可。

---

# 23. TL;DR 新完整 data-selection identity

推荐：

```text
experiment_seed_<E>_
tldr_train_seed_<S>_
train_samples_<N>_
calibration_seed_<C>
```

拼成一个目录名。

例如：

```text
experiment_seed_42_tldr_train_seed_42_train_samples_2000_calibration_seed_42
```

---

# 24. Tulu-3 新完整 data-selection identity

推荐：

```text
experiment_seed_<E>_
train_seed_<S>_
train_samples_<N>_
calibration_seed_<C>
```

例如：

```text
experiment_seed_42_train_seed_42_train_samples_2000_calibration_seed_42
```

---

# 25. `train_samples=None`

统一编码：

```text
train_samples_full
```

不要编码成：

```text
train_samples_None
```

---

# 26. 推荐 helper

例如：

```python
def train_samples_dir_value(
    value: Any,
) -> str:
    if value is None:
        return "full"

    value = int(value)

    if value <= 0:
        raise ValueError(
            "train_samples must be a "
            "positive integer or null"
        )

    return str(value)
```

---

# 27. 推荐修改 `run_seed_dirname()`

TL;DR：

```python
return (
    f"experiment_seed_{exp_seed}_"
    f"tldr_train_seed_{...}_"
    f"train_samples_{samples}_"
    f"calibration_seed_{...}"
)
```

Tulu-3：

```python
return (
    f"experiment_seed_{exp_seed}_"
    f"train_seed_{...}_"
    f"train_samples_{samples}_"
    f"calibration_seed_{...}"
)
```

---

# 28. `data_selection_seed_dirname()`

可继续：

```python
return run_seed_dirname(cfg)
```

确保：

```text
run identity
=
data-selection identity
```

在 TL;DR / Tulu-3 下成立。

---

# 29. Shared data path

TL;DR：

```text
.../tldr/_shared/
experiment_seed_42_tldr_train_seed_42_train_samples_2000_calibration_seed_42/
data/
```

Tulu-3：

```text
.../tulu3/_shared/
experiment_seed_42_train_seed_42_train_samples_2000_calibration_seed_42/
data/
```

---

# 30. `data_selection_policy_root` 两类 task 都要使用完整 identity

推荐：

```python
self.data_selection_policy_root = (
    self.policy_root
    / data_selection_seed_name
)
```

不要再只对 Tulu-3 加这一层。

---

# 31. TL;DR Pre-finetuning Prefix

新路径：

```text
.../<model>/_shared/seed42/
rotated_prefix/
experiment_seed_42_tldr_train_seed_42_train_samples_2000_calibration_seed_42/
pre_finetuning_prefix/
```

---

# 32. Tulu-3 Pre-finetuning Prefix

新路径：

```text
.../<model>/_shared/seed42/
rotated_prefix/
experiment_seed_42_train_seed_42_train_samples_2000_calibration_seed_42/
pre_finetuning_prefix/
```

---

# 33. ANN-training Stage A/B 同样挂在完整 identity 下

TL;DR / Tulu-3：

```text
.../rotated_prefix/
<data-selection-identity>/
ann_training_calibration/
...
```

---

# 34. canonical preprocessing 不能绑定 `train_samples`

canonical preprocessing 是独立 rotation-regression selection。

不能因为：

```text
train_samples
train seed
```

变化而变化。

---

# 35. 推荐统一 canonical preprocessing root

TL;DR 与 Tulu-3 均：

```text
.../<task>/_shared/
seed<E>/
canonical_preprocessing/
calibration_seed_<C>/
num_samples_128/
calibration_manifest.json
```

其 dependency：

```text
experiment.seed
+
calibration.seed
```

不依赖：

```text
train_samples
task-specific train seed
```

---

# 36. Rotation weights 不随 `train_samples` 改变

继续：

```text
.../<model>/_shared/seed<E>/
rotated_prefix/
rotation/
├── rotation_state.pt
└── fused_base/
```

---

# 37. Rotation regression 不随 `train_samples` 改变

只要：

```text
experiment.seed
calibration.seed
```

相同：

```text
rotation_regression_path
```

应相同。

因为 canonical preprocessing 不变。

---

# 38. Rotation regression 可统一为 calibration-seed scope

若本轮接受不兼容旧 TL;DR shared artifact，推荐 TL;DR / Tulu-3 都采用：

```text
rotation/
└── regression/
    └── calibration_seed_<C>/
        ├── rotation_regression.json
        └── rotation_summary.json
```

若保留现有 TL;DR regression 根路径也不影响 `train_samples` 语义，但统一更整洁。

---

# 39. concrete run identity

建议 TL;DR / Tulu-3 的 concrete run identity 也统一加入 train sample count。

TL;DR：

```text
experiment_seed_<E>_
tldr_train_seed_<S>_
train_samples_<N>_
calibration_seed_<C>
```

Tulu-3：

```text
experiment_seed_<E>_
train_seed_<S>_
train_samples_<N>_
calibration_seed_<C>
```

上层已有：

```text
..._train_samples_<N>...
```

可以保留。

路径存在重复信息不是本轮问题，不做清理。

---

# 40. `实验执行总结.md` 必须新增核心规则

必须明确写：

> **只要 sweep 的 `train_samples` 改变，都应该把它视为一条新的数据选择 → calibration → training → conversion 实验链。**
>
> 对 TL;DR，`train_samples` 指 `training.tldr_train_samples`；对 Tulu-3，指 `training.train_samples`。

---

# 41. `实验执行总结.md` 必须新增 manifest-only training 原则

建议原样或等价写：

> **所有 ANN training 使用的训练样本集合必须事先通过修改对应 YAML 配置并执行 Step 2 `prepare_data.py` 固化到 `train_manifest.json`。Step 6 ANN training 只能消费该 manifest 中的 `indices`，禁止根据 `train_samples`、task-specific training seed 或 `experiment.seed` 在训练阶段重新生成、重采样或替换 `train_manifest.json` 中的 training indices。**

---

# 42. `实验执行总结.md` 必须新增 calibration-only 原则

建议：

> **`--calibration-only` 只能基于已经存在且与当前 YAML 完全匹配的 Step 2 `train_manifest.json` 生成新的 Stage-A calibration manifest；它不得创建、重建或重新采样 ANN training subset。若 train manifest 不存在或与当前 YAML 中的 training sample count / seed 不一致，必须先重新执行完整 Step 2。**

---

# 43. `train_samples` sweep 时 Step 重跑规则

必须写清：

| Step | `train_samples` 改变后 | 原因 |
|---|---|---|
| Step 1 | 需要准备新 config | 新 sample count 必须进入配置与 artifact identity |
| Step 2 | **必须完整重跑** | 生成新的固定 training manifest；禁止只跑 `--calibration-only` |
| Step 3 Base | 无需 | 与 train samples 无关 |
| Step 4 Rotation | 无需 | canonical + Rotation 不依赖 train samples |
| Step 4 Pre-finetuning Prefix | **必须重跑** | Stage-A selection 改变 |
| Step 4 rotated-pre-finetuning eval | Prefix enabled 时重跑 | Prefix 已变 |
| Step 5 Stage A | **必须重跑** | calibration subset 改变 |
| Step 5 Stage B | **必须重跑** | 依赖新的 Stage A |
| Step 5.1 GIF MSE | 启用时必须重跑 | histogram/qparams 依赖新 calibration |
| Step 6 ANN training | **必须重跑** | training dataset 变化 |
| Step 7 Post Prefix | **必须重跑** | final checkpoint 变化 |
| Step 8 Post Stage A | **必须重跑** | checkpoint / calibration 变化 |
| Step 9 Final ANN eval | **必须重跑** | checkpoint 变化 |
| Step 10 SNN conversion/eval | **必须重跑** | source artifacts 全属于新链 |

---

# 44. Step 4 必须拆开描述

不能简单写：

```text
Step 4 不重跑
```

正确是：

```text
Rotation fusion / regression：
不重跑

Pre-finetuning Prefix：
必须重跑

rotated-pre-finetuning evaluation：
若实际加载 Prefix，必须重跑
```

---

# 45. Step 2 对每个 `train_samples` value 都必须运行

例如：

```text
2000
5000
10000
```

每一个都必须对应：

```bash
python scripts/prepare_data.py \
  --config <对应配置>
```

不能只生成一份 manifest 后在训练阶段切 sample count。

---

# 46. `--calibration-only` 的适用范围

只适用于：

```text
training data identity 不变
```

例如只修改：

```text
calibration.num_samples
```

且：

```text
train samples / train seed / experiment seed
```

不变。

这时可以：

```bash
python scripts/prepare_data.py \
  --config "$CFG" \
  --calibration-only
```

---

# 47. Train manifest provenance

TL;DR 至少记录：

```text
tldr_train_samples
tldr_train_seed
indices
record_ids
sampling
```

Tulu-3 至少记录：

```text
train_samples
train_seed
validation_size
indices
record_ids
sampling
```

如已有则保留。

---

# 48. 推荐记录 `data_selection_identity`

推荐 train / calibration manifest 增加：

```json
{
  "data_selection_identity":
  "experiment_seed_..._train_samples_..."
}
```

方便 verifier 和 debugging。

可选但推荐。

---

# 49. calibration manifest parent provenance

继续记录：

```text
parent_training_samples
parent_training_seed
```

并确保来自实际 Step 2 train manifest。

不要重新计算 parent training subset。

---

# 50. 推荐 verifier 新增 train-manifest config 校验

`verify_artifacts.py` 应调用统一：

```python
validate_train_manifest_for_config()
```

这样可验证：

```text
artifact path identity
manifest metadata
current YAML
```

三者一致。

---

# 51. 训练启动前继续检查 selected count

当前 `training.py` 中已有：

```python
configured_train_samples = ...
if configured_train_samples is not None:
    if len(bundle.train) != int(
        configured_train_samples
    ):
        raise RuntimeError(...)
```

这个检查可以保留。

但它只能作为：

```text
sanity check
```

不能通过二次采样来“修正”长度。

---

# 52. 建议新增测试：training consumes manifest exactly

构造：

```text
manifest indices = [具体固定列表]
```

配置中的 seed / sample count 与 manifest 匹配。

mock raw dataset。

调用：

```python
load_selected_raw(cfg, layout)
```

断言：

```text
bundle.train
```

严格对应 manifest indices。

---

# 53. 建议新增测试：禁止 runtime resampling

可以 monkeypatch：

```python
_tldr_train_selection
_tulu3_train_selection
```

使其一旦在 `load_selected_raw()` 阶段调用就抛异常。

然后：

```python
load_selected_raw()
```

必须正常完成。

用于锁死：

```text
training load 不允许 sampling
```

---

# 54. 建议新增测试：calibration-only requires Step 2 manifest

没有：

```text
train_manifest.json
```

时调用：

```python
prepare_calibration_manifest()
```

必须：

```text
FileNotFoundError
```

---

# 55. 建议新增测试：calibration-only rejects stale 10k manifest

例如：

```text
manifest:
train_samples=10000

cfg:
train_samples=20000
```

调用：

```python
prepare_calibration_manifest()
```

必须失败。

不能自动生成 20000 subset。

---

# 56. 建议新增测试：calibration-only derives from manifest indices

manifest：

```text
indices=[...固定集合...]
```

然后 calibration selection 必须：

```text
subset of manifest indices
```

且 positions 对应：

```python
[
    manifest_indices[p]
    for p in positions
] == calibration_indices
```

---

# 57. 建议新增 TL;DR train_samples identity test

```python
10k.data_dir != 20k.data_dir
10k.ann_training_prefix_dir != 20k.ann_training_prefix_dir
10k.ann_training_calibration_dir != 20k.ann_training_calibration_dir
10k.root != 20k.root
```

同时：

```python
10k.canonical_preprocessing_calibration_manifest_path
==
20k.canonical_preprocessing_calibration_manifest_path
```

---

# 58. 建议新增 Tulu-3 train_samples identity test

同样：

```python
10k.data_dir != 20k.data_dir
10k.ann_training_prefix_dir != 20k.ann_training_prefix_dir
10k.ann_training_calibration_dir != 20k.ann_training_calibration_dir
10k.root != 20k.root
```

canonical / Rotation 保持相同。

---

# 59. train seed sweep

TL;DR：

```text
tldr_train_seed
```

改变：

```text
data identity
Prefix
Stage A/B
run root
```

都变化。

Tulu-3：

```text
train_seed
```

同理。

但 canonical preprocessing / Rotation weights 不变，只要：

```text
experiment.seed
calibration.seed
```

不变。

---

# 60. calibration seed sweep

改变：

```text
calibration.seed
```

应改变：

```text
data-selection identity
Stage-A calibration
Prefix
Stage A/B
canonical preprocessing
Rotation regression
```

但：

```text
Rotation weights / fused_base
```

不变。

---

# 61. `calibration.num_samples` 与 training sample count 继续分离

例如：

```text
train_samples_10000
...
data/calibration/num_samples_128/
```

不要把两个不同概念合并。

---

# 62. Base baseline 不进入 data-selection identity

保持：

```text
base/seed<E>/
```

不依赖：

```text
train_samples
train seed
calibration seed
```

---

# 63. Post-finetuning artifacts 自然跟随新 run root

只要 concrete run identity 正确：

```text
post_finetuning/
snn/
analysis/
```

都会自动随 train sample count 隔离。

---

# 64. SNN selector 不允许跨 train_samples identity 混用

`use_post_finetuning_artifacts=false`：

```text
checkpoint
+
Pre Prefix
+
ANN-training Stage A
```

必须来自同一完整 identity。

`true`：

```text
checkpoint
+
Post Prefix
+
Post Stage A
```

也必须来自同一 run identity。

---

# 65. 旧 TL;DR artifact 策略

用户确认：

```text
不兼容旧 TL;DR shared artifacts
```

因此：

- 不做 fallback；
- 不读旧 `_shared/seed42/data`；
- 不自动复制；
- 新实验按完整 identity 重建。

---

# 66. 旧 Tulu-3 artifact 策略

旧：

```text
experiment_seed_<E>_
train_seed_<S>_
calibration_seed_<C>
```

不再作为新 data-selection identity。

新：

```text
experiment_seed_<E>_
train_seed_<S>_
train_samples_<N>_
calibration_seed_<C>
```

---

# 67. `实验执行总结.md` 路径示例必须更新

TL;DR：

```text
.../tldr/_shared/
experiment_seed_42_tldr_train_seed_42_train_samples_10000_calibration_seed_42/
data/
```

Tulu-3：

```text
.../tulu3/_shared/
experiment_seed_42_train_seed_42_train_samples_10000_calibration_seed_42/
data/
```

---

# 68. Model-level shared 路径也更新

TL;DR：

```text
.../<model>/_shared/seed42/
rotated_prefix/
experiment_seed_42_tldr_train_seed_42_train_samples_10000_calibration_seed_42/
```

Tulu-3：

```text
.../<model>/_shared/seed42/
rotated_prefix/
experiment_seed_42_train_seed_42_train_samples_10000_calibration_seed_42/
```

---

# 69. `实验执行总结.md` 中明确不能直接改 YAML 后跳到 Step 6

必须写明：

```text
如果修改 train_samples / task-specific train seed，
禁止直接运行 train_ann.py。
必须先重新执行 Step 2，
生成当前配置对应的新 train_manifest.json。
```

---

# 70. 推荐训练前错误提示

若配置与 manifest 不匹配：

```text
The current training configuration does not match
the fixed Step-2 train manifest.

Do not resample training data at runtime.

Re-run:
python scripts/prepare_data.py --config <config>
```

---

# 71. 不能在 loader 中静默修复 stale manifest

禁止类似：

```python
if mismatch:
    indices = resample(...)
```

只能：

```python
raise ...
```

---

# 72. `prepare_manifests()` 仍是完整 selection 的唯一实现

以下函数继续负责 Step 2：

```text
_tldr_train_selection()
_tulu3_shared_split_selection()
_tulu3_train_selection()
_manifest_split_selection()
```

这是允许 sampling 的阶段。

后续 loader 只负责：

```text
read + validate + select
```

---

# 73. 推荐职责划分

```text
prepare_manifests()
→ 决定 indices

prepare_calibration_manifest()
→ 读取已有 train indices，再决定 calibration indices

load_selected_raw()
→ 不决定 indices，只加载 indices

training.py
→ 不决定 indices，只训练
```

这是本轮最重要的代码结构。

---

# 74. 全仓库检查 runtime sampling caller

修改后搜索：

```bash
grep -R "_tldr_train_selection" -n snn2 scripts tests
grep -R "_tulu3_train_selection" -n snn2 scripts tests
grep -R "_manifest_split_selection" -n snn2 scripts tests
grep -R "use_configured_train_subset" -n snn2 scripts tests
```

验收：

```text
use_configured_train_subset
```

生产代码应不再存在。

训练路径不得调用 selection helpers。

---

# 75. 全仓库检查旧 shared identity

```bash
grep -R "data_selection_seed_dirname" -n snn2 scripts tests
grep -R "data_selection_policy_root" -n snn2 scripts tests
grep -R "_shared/seed42/data" -n . --exclude-dir=.git
```

清理文档与测试中的旧预期。

---

# 76. 必须更新测试

至少覆盖：

- TL;DR 10k vs 20k identity；
- Tulu-3 10k vs 20k identity；
- full identity；
- loader 严格使用 manifest indices；
- training loader 不调用 sampling helpers；
- calibration-only 缺 manifest fail；
- calibration-only stale manifest fail；
- calibration-only subset 来源为 Step 2 manifest；
- canonical path 对 train_samples 不敏感；
- Rotation path/regression 对 train_samples 不敏感。

---

# 77. 必须执行测试

```bash
pytest -q
```

全部通过。

重点：

```bash
pytest -q tests/test_tulu3_lm_eval_protocol.py
pytest -q tests/test_verify_artifacts.py
pytest -q tests/test_generated_configs.py
```

以及新增/修改的 data/training tests。

---

# 78. 最终 dependency matrix

| 参数变化 | train manifest | Prefix | ANN Stage A/B | ANN run | canonical | Rotation weights | Rotation regression |
|---|---:|---:|---:|---:|---:|---:|---:|
| `experiment.seed` | 变 | 变 | 变 | 变 | 变 | 变 | 变 |
| task train seed | 变 | 变 | 变 | 变 | 不变* | 不变* | 不变* |
| `train_samples` | **变** | **变** | **变** | **变** | **不变** | **不变** | **不变** |
| `calibration.seed` | 变 | 变 | 变 | 变 | 变 | 不变 | 变 |
| `calibration.num_samples` | train manifest 不变 | 通常变 | 变 | aware path变 | 不变 | 不变 | 不变 |

`*`：在 `experiment.seed` / `calibration.seed` 不变时。

---

# 79. 最终流程语义

## TL;DR

```text
YAML
tldr_train_seed + tldr_train_samples
        ↓
Step 2
        ↓
fixed train_manifest indices
        ↓
calibration.seed
        ↓
fixed Stage-A calibration
        ↓
Prefix / Stage A/B
        ↓
ANN training consumes train_manifest only
        ↓
Final ANN
        ↓
Post artifacts
        ↓
SNN
```

---

## Tulu-3

```text
YAML
experiment.seed
        ↓
fixed validation/shared pool
        ↓
train_seed + train_samples
        ↓
Step 2 fixed train_manifest
        ↓
calibration.seed
        ↓
fixed Stage-A calibration
        ↓
Prefix / Stage A/B
        ↓
ANN training consumes train_manifest only
        ↓
Final ANN
        ↓
Post artifacts
        ↓
SNN
```

---

# 80. `实验执行总结.md` 必须出现的最终规则

请明确写入：

> **只要 sweep 的 `train_samples` 改变，都应该把它视为一条新的数据选择 → calibration → training → conversion 实验链。**
>
> 对 TL;DR，`train_samples` 指 `training.tldr_train_samples`；对 Tulu-3，指 `training.train_samples`。每一个新的 sample count 都必须先修改对应 YAML 并重新执行完整 Step 2 `prepare_data.py`，由 Step 2 生成当前实验唯一有效的 `train_manifest.json`。Step 6 ANN training 只能使用该 manifest 中的 training indices，禁止在训练阶段根据 `train_samples`、task-specific training seed 或 `experiment.seed` 再次采样、替换或忽略这些 indices。
>
> 当 `train_samples` 改变时，需要重新执行：Step 1 对应 config、Step 2 完整数据 manifest、Step 4 Pre-finetuning Prefix、Step 5 ANN-training Stage A/B、Step 6 ANN training、Step 7 Post-finetuning Prefix、Step 8 Post-finetuning Stage A、Step 9 Final ANN evaluation、Step 10 SNN conversion/evaluation；若启用 GIF MSE，也必须重新执行对应 MSE calibration。若 rotated-pre-finetuning evaluation 实际使用 Prefix，也必须重新评估。
>
> 无需因 `train_samples` 改变而重新执行：Step 3 Base baseline，以及 Step 4 中的 Rotation fusion / canonical preprocessing / Rotation regression。
>
> `--calibration-only` 只能读取已有且与当前 YAML 匹配的 Step 2 `train_manifest.json`，并在该固定 training subset 内重新抽取 Stage-A calibration；它不能创建、重建或重新采样 ANN training subset。若 manifest 不存在或其 sample count / seed 与当前 YAML 不一致，必须先重新执行完整 Step 2。

---

# 81. 本轮不要修改

不要修改：

- neuron math；
- Phase/GIF/MTN；
- Rotation 数学；
- Hadamard；
- Prefix discovery 算法；
- common Clip；
- Stage-A calibration 数学；
- optimizer/scheduler；
- lm-eval task；
- evaluation metric；
- SNN selector 语义。

本轮核心只有：

```text
1. train_samples-aware artifact identity
2. Step 2 manifest-only source of truth
3. 禁止 training-time resampling
4. calibration-only fail closed
5. 文档明确新的实验链与重跑规则
```
