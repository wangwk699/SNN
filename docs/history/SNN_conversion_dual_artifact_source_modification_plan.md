# SNN Conversion 双套 Prefix / Calibration Artifact 可切换方案

> 目标仓库：`wangwk699/SNN`，基于 2026-09-03 当前 `main`。  
> 本文面向**没有本次对话上下文的 Codex**。请严格按本文修改代码、测试与 `实验执行总结.md`，不要恢复旧的 mode-aware 强制策略。

---

## 1. 修改目标

当前代码把四种 ANN mode 的 SNN conversion artifact source 固定成两类：

- `vanilla / unaware`
  - Post-finetuning Prefix
  - Post-finetuning conversion calibration
- `phase_aware / gif_aware`
  - Pre-finetuning Prefix
  - ANN-training calibration
  - 并且当前 `resolve_config()`、`discover_prefix.py`、`calibrate_sites.py` 会阻止 aware mode 重新生成 post-finetuning artifacts。

本次修改后，必须改成：

### 1.1 两套 artifact 都允许生成并同时存在

对于 `unaware / phase_aware / gif_aware`，同时保留：

**套件 A：Pre-finetuning bundle**

- Step 4：Pre-finetuning Prefix
- Step 5：ANN-training Calibration Stage A
- Prefix 基于 shared rotated fused Base
- Calibration 也基于 shared rotated fused Base + Pre-finetuning Prefix
- `unaware / phase_aware / gif_aware` 共享同一份 Prefix 和同一份 Stage A calibration artifact

**套件 B：Post-finetuning bundle**

- Step 7：Post-finetuning Prefix
- Step 8：Post-finetuning conversion Calibration Stage A
- 二者均针对各 mode 自己训练得到的 Final ANN checkpoint
- `vanilla / unaware / phase_aware / gif_aware` 都允许生成
- 每个 Final ANN run 自己保存，不共享

`vanilla` 没有 Pre-finetuning Prefix / ANN-training calibration，因此仍只能使用 Post-finetuning bundle。

### 1.2 新增一个唯一的 SNN artifact source 选择参数

在配置中新增：

```yaml
conversion:
  use_post_finetuning_artifacts: true
```

语义严格固定为：

- `true`
  - `convert_snn.py` 使用 **Post-finetuning Prefix + Post-finetuning calibration**
  - SNN Evaluation 使用同一套 Post-finetuning bundle
- `false`
  - `convert_snn.py` 使用 **Pre-finetuning Prefix + ANN-training calibration**
  - SNN Evaluation 使用同一套 Pre-finetuning bundle

约束：

- `vanilla`：只允许 `true`
- `vanilla + false`：必须在 config validation 阶段直接报错，不能静默改成 `true`
- `unaware / phase_aware / gif_aware`：允许 `true` 或 `false`

**这个参数只控制 SNN conversion / SNN Evaluation。不要让它改变 ANN Training、Final ANN Evaluation 或 rotated-pre-finetuning ANN Evaluation 的语义。**

---

# 2. 必须保持不变的语义

## 2.1 Final ANN checkpoint 不因新参数而变化

同一个 ANN checkpoint 可以在不重新训练的情况下分别跑：

```yaml
conversion:
  use_post_finetuning_artifacts: false
```

和：

```yaml
conversion:
  use_post_finetuning_artifacts: true
```

只需要重新执行对应的：

```bash
python scripts/convert_snn.py ...
accelerate launch ... scripts/evaluate_tldr.py --neuron phase|gif|mtn
# 或 evaluate_lm_harness.py
```

两种设置必须指向**同一个 `ann/final` checkpoint**，只是 SNN conversion source 和 SNN 输出目录不同。

## 2.2 Final ANN Evaluation 的 Prefix 规则不由新参数控制

本次不要顺带改变 Final ANN evaluation。

当前 Final ANN evaluation 的 mode-aware 语义继续保留：

- `vanilla`：不加载 Prefix
- `unaware`：按现有规则使用 Post-finetuning Prefix（受 `evaluation.prefix_enabled` 控制）
- `phase_aware / gif_aware`：按现有规则使用 Pre-finetuning Prefix（受 `evaluation.prefix_enabled` 控制）

因此必须把当前同时服务 ANN/SNN 的 `final_evaluation_prefix_artifact_stage()` 逻辑拆开，避免 `conversion.use_post_finetuning_artifacts` 误改 Final ANN Evaluation。

建议明确拆成：

```python
final_ann_evaluation_prefix_artifact_stage(cfg)
final_snn_evaluation_prefix_artifact_stage(cfg)
```

其中：

```python
final_ann_evaluation_prefix_artifact_stage(cfg)
```

保持旧规则：

- aware -> `pre_finetuning`
- unaware -> `post_finetuning`
- vanilla 实际不加载

而：

```python
final_snn_evaluation_prefix_artifact_stage(cfg)
```

严格跟随：

```yaml
conversion.use_post_finetuning_artifacts
```

## 2.3 `evaluation.prefix_enabled` 不要删除

新参数选择的是 **Prefix/calibration artifact source bundle**，不是替代现有 Prefix enable/disable 开关。

继续保留：

```yaml
evaluation:
  prefix_enabled: true|false
```

其现有“evaluation 是否实际注入 Prefix KV”的职责不变。

对于 SNN：

- `use_post_finetuning_artifacts` 决定从哪个 Prefix artifact 目录读取
- `evaluation.prefix_enabled` 决定 evaluation 是否实际注入 Prefix

对于 calibration：

- ANN-training calibration 使用 `ann_training.prefix_enabled`
- Post-finetuning calibration 使用 `post_finetuning.prefix_enabled`

不要把这些布尔值混为一个参数。

---

# 3. `snn2/config.py` 修改

## 3.1 `resolve_config()`

新增：

```python
cfg.setdefault("conversion", {})
cfg["conversion"].setdefault("use_post_finetuning_artifacts", True)
```

默认值为 `true`。

### 删除 aware mode 的强制禁用逻辑

当前 `phase_aware` / `gif_aware` 分支中存在类似：

```python
cfg["post_finetuning"].update({
    "rediscover_prefix": False,
    "recalibrate_sites": False,
    "post_finetuning_recalibration": False,
    "prefix_enabled": False,
})
```

**全部删除。**

`phase_aware / gif_aware` 不再由 `resolve_config()`：

- 禁止 Post-finetuning Prefix
- 禁止 Post-finetuning calibration
- 强制 `post_finetuning.prefix_enabled=false`

`configs/experiment_matrix.yaml` 当前 Post-finetuning 字段是 `true`，materialize 后 aware config 应保留这些值。

### vanilla

不要在 `resolve_config()` 中把：

```yaml
conversion.use_post_finetuning_artifacts
```

强制写成 `true`。

如果用户显式写 `false`，保留该值并在 `validate_config()` 报错。

## 3.2 `validate_config()`

将 `conversion` 加入 required section，并验证：

```python
value = cfg["conversion"].get("use_post_finetuning_artifacts")
if not isinstance(value, bool):
    raise ValueError(...)
```

vanilla 增加：

```python
if mode == "vanilla" and not cfg["conversion"]["use_post_finetuning_artifacts"]:
    raise ValueError(
        "vanilla requires conversion.use_post_finetuning_artifacts=true"
    )
```

### 删除旧的 mode-aware post-finetuning 强制校验

当前存在：

```python
expected_post = not is_aware_ann_mode(cfg)
for key in (...):
    if bool(cfg["post_finetuning"].get(key, False)) != expected_post:
        ...
```

该规则必须删除。

`phase_aware / gif_aware` 现在必须允许：

```yaml
post_finetuning:
  rediscover_prefix: true
  recalibrate_sites: true
  post_finetuning_recalibration: true
  prefix_enabled: true
```

这些旧字段可以继续保留并做布尔类型校验，但**不得再负责选择 conversion source**。

## 3.3 增加统一 helper

建议新增：

```python
def use_post_finetuning_artifacts(cfg) -> bool:
    return bool(cfg["conversion"]["use_post_finetuning_artifacts"])
```

以及：

```python
def conversion_calibration_stage(cfg) -> str:
    return (
        "post_finetuning"
        if use_post_finetuning_artifacts(cfg)
        else "ann_training"
    )

def conversion_prefix_artifact_stage(cfg) -> str:
    return (
        "post_finetuning"
        if use_post_finetuning_artifacts(cfg)
        else "pre_finetuning"
    )

def conversion_uses_ann_training_artifacts(cfg) -> bool:
    return not use_post_finetuning_artifacts(cfg)
```

当前 `conversion_reuses_ann_training_artifacts()` 若大量代码依赖，可以保留名字作为兼容 wrapper，但其返回值必须由**新参数**决定，而不能再由 `is_aware_ann_mode()` 决定。

### `conversion_prefix_enabled(cfg)`

改为：

```python
def conversion_prefix_enabled(cfg):
    if use_post_finetuning_artifacts(cfg):
        return post_finetuning_prefix_enabled(cfg)
    return training_prefix_enabled(cfg)
```

### Final ANN / SNN Prefix source helper 必须拆开

旧的：

```python
final_evaluation_prefix_artifact_stage(cfg)
```

不要继续同时用于 ANN 和 SNN。

改为两个明确 helper，具体规则见第 2.2 节。

## 3.4 `requires_*` helper

`requires_pre_finetuning_prefix(cfg)`：

- 继续保持 `mode != vanilla`
- Step 4 仍只为 rotated modes 生成

`requires_ann_training_calibration(cfg)`：

- 可继续保持当前 aware-only 语义，因为它表示“ANN Training 本身需要 Stage A/B”
- **不要为了 unaware 改成 true**
- unaware 在 `use_post_finetuning_artifacts=false` 时只是消费 Step 5 已经由 phase-aware config 生成的共享 Stage A

`requires_post_finetuning_artifacts(cfg)`：

当前“aware=false”的语义已经失效。

可以：

1. 改名为 `supports_post_finetuning_artifacts()` 并对四种 mode 都返回 true；或
2. 保留旧函数名以减少改动，但对四种 mode 都返回 true。

关键要求是：

- `discover_prefix --stage post_finetuning` 四种 mode 都允许
- `calibrate_sites --stage post_finetuning --calibration-phase A` 四种 mode 都允许

**不要让是否生成 Post-finetuning artifacts 依赖 `conversion.use_post_finetuning_artifacts`。两套 artifact 必须可以同时存在。**

---

# 4. `configs/experiment_matrix.yaml` / generated configs

在每个 experiment 的共享 config 中加入：

```yaml
conversion:
  use_post_finetuning_artifacts: true
```

默认使用 Post-finetuning bundle。

`materialize_configs.py` 仍生成原来的 12 个配置，不需要为 true/false 再扩成 24 个配置。

用户需要比较两套 SNN 时，可以对同一 ANN run config 切换该字段，不应改变 ANN run identity。

生成配置中四种 mode 都必须显式包含该字段。

### vanilla

`true` 正常。

如果用户手工改成：

```yaml
conversion:
  use_post_finetuning_artifacts: false
```

load/validate config 必须报错。

---

# 5. `scripts/discover_prefix.py`

当前代码会对 aware mode 的：

```bash
--stage post_finetuning
```

报错，原因来自 `requires_post_finetuning_artifacts()` 的旧 mode-aware 规则。

删除该限制。

修改后：

```bash
python scripts/discover_prefix.py \
  --config <vanilla|unaware|phase_aware|gif_aware> \
  --stage post_finetuning
```

四种 mode 都允许。

Post-finetuning Prefix 仍必须：

- 加载各自 `layout.ann_checkpoint_dir`
- 即 `seed42/ann/final`
- 使用当前 Final ANN checkpoint 构建 Prefix state/KV
- 保存到各自 run 的：
  `seed42/post_finetuning/prefix/num_samples_<N>/`

不要改变 Pre-finetuning Prefix：

- vanilla 仍禁止
- unaware / phase_aware / gif_aware 仍共享 rotated fused Base 的 Prefix
- 共享路径不按 mode 拆开

Post-finetuning Prefix discovery 对 aware mode 采用和当前 vanilla/unaware Post-finetuning discovery **同一条执行逻辑**，不要增加 PhaseSurrogate / StaticGIF 特判。

---

# 6. `scripts/calibrate_sites.py`

## 6.1 ANN-training Stage A/B

保留当前 Step 5 的 canonical 生成方式：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG_*_P" \
  --stage ann_training \
  --calibration-phase A
```

以及 aware ANN training 所需的 Stage B。

不要要求 unaware 自己再运行一份 ANN-training calibration。

原因：

`ArtifactLayout.ann_training_calibration_dir` 本来就是：

```text
_shared/seed42/rotated_prefix/ann_training_calibration/...
```

不包含 `ann_mode`，因此 unaware / phase_aware / gif_aware 应消费同一份 Stage A。

修改后的 Step 5 文档必须明确：

> ANN-training Stage A 除了服务 phase/gif aware ANN training，也作为 `unaware/phase_aware/gif_aware + use_post_finetuning_artifacts=false` 的共享 SNN conversion calibration source。

Stage B 仍只服务 aware ANN fine-tuning common Clip，不是 SNN deployment artifact。

## 6.2 Post-finetuning Stage A

删除 aware mode 的拒绝逻辑。

以下四种 config 都必须允许：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning \
  --calibration-phase A
```

Post-finetuning Stage A：

- source model = 各自 Final ANN checkpoint
- Prefix = 各自 Post-finetuning Prefix（当 `post_finetuning.prefix_enabled=true`）
- site statistics / Phase/GIF/MTN states = 各自 Final ANN 独立重新统计
- 保存到各自：
  `seed42/post_finetuning/conversion_calibration/.../sites`

`phase_aware/gif_aware` 的 Post-finetuning 路径不得放到 `_shared`，必须和 vanilla/unaware 一样在自己的 run root 中。

当前 `ArtifactLayout.post_finetuning_dir = self.root / "post_finetuning"` 已经满足“与 `seed42/ann` 同级”；不要为 aware mode 新造特殊路径。

---

# 7. Calibration manifest provenance 修改

涉及 `snn2/calibration.py`。

当前 ANN-training calibration manifest 中存在类似：

```text
conversion_reuse_policy = "aware_modes_only"
```

这已经不正确，因为新方案允许：

```text
unaware + use_post_finetuning_artifacts=false
```

消费同一份 ANN-training Stage A。

将该 provenance policy 改为能够表达：

> shared rotated pre-finetuning Stage A 可被所有 non-vanilla mode 在显式选择 pre-finetuning bundle 时用于 conversion。

例如统一改成稳定枚举：

```text
"non_vanilla_when_selected"
```

或同等清晰名称。

需要同步修改：

- `calibration_provenance()`
- `collect_site_statistics()`
- conversion manifest validator
- tests
- `verify_artifacts.py`

不要只改字符串一处。

保持：

- `eligible_for_ann_training=True`：仍表示该 Stage A 可服务 aware ANN training
- `eligible_for_conversion=True`
- `post_finetuning_recalibration=False`
- source model stage = rotated fused Base

Post-finetuning Stage A 继续：

- source model stage = final ANN checkpoint
- source_ann_mode = 当前 mode
- source_ann_checkpoint = 当前 run `ann/final`
- `post_finetuning_recalibration=True`

现在该规则同样适用于 aware mode。

---

# 8. `snn2/artifacts.py`：source 路径与 SNN 输出路径

## 8.1 Post-finetuning 路径

保持现有：

```python
self.post_finetuning_dir = self.root / "post_finetuning"
```

或等价 property。

必须保证四种 mode 都是：

```text
.../seed42/
├── ann/
│   └── final/
├── post_finetuning/
│   ├── prefix/
│   └── conversion_calibration/
└── snn/
```

aware mode 不得把 Post-finetuning artifact 放入 `_shared`。

## 8.2 `conversion_prefix_dir`

当前按 `is_aware_ann_mode()` 选择，必须改成按新参数：

```python
@property
def conversion_prefix_dir(self):
    return (
        self.post_finetuning_prefix_dir
        if use_post_finetuning_artifacts(self._cfg)
        else self.ann_training_prefix_dir
    )
```

vanilla false 在 config validation 已经被拒绝。

## 8.3 `conversion_calibration_dir` / `conversion_site_dir`

同理：

```python
@property
def conversion_calibration_dir(self):
    return (
        self.post_finetuning_conversion_calibration_dir
        if use_post_finetuning_artifacts(self._cfg)
        else self.ann_training_calibration_dir
    )
```

`conversion_site_dir` 继续为：

```python
return self.conversion_calibration_dir / "sites"
```

## 8.4 SNN 路径必须新增 source-selector 目录

新增 helper，例如：

```python
def conversion_artifact_source_dirname(enabled: bool) -> str:
    return (
        "use_post_finetuning_artifacts_true"
        if enabled
        else "use_post_finetuning_artifacts_false"
    )
```

注意这里 `true` 拼写正常，不沿用 `prefix_enabled_ture` 的历史 typo。

所有 SNN output path 在 `snn/` 后立即插入：

```text
snn/use_post_finetuning_artifacts_true/...
```

或：

```text
snn/use_post_finetuning_artifacts_false/...
```

也就是说修改 `ArtifactLayout.snn_dir()` 的 base：

```python
base = (
    self.root
    / "snn"
    / conversion_artifact_source_dirname(
        use_post_finetuning_artifacts(self._cfg)
    )
)
```

然后再接现有：

- calibration variant（当前 vanilla/unaware 有）
- neuron
- phase_T / mtn_T,K

不要删除现有 `prefix_enabled_ture|false` conversion/evaluation 子目录。

示意：

### phase/gif aware

```text
.../seed42/snn/
  use_post_finetuning_artifacts_true/
    phase/phase_T_.../
    gif/
    mtn/mtn_T_..._mtn_K_.../

  use_post_finetuning_artifacts_false/
    phase/phase_T_.../
    gif/
    mtn/mtn_T_..._mtn_K_.../
```

### unaware

保留其现有 calibration variant 层，只在 `snn/` 后插入 selector：

```text
.../seed42/snn/
  use_post_finetuning_artifacts_true/
    calibration_group_size_<G>_num_samples_<N>/
      phase/...
  use_post_finetuning_artifacts_false/
    calibration_group_size_<G>_num_samples_<N>/
      phase/...
```

### vanilla

只有：

```text
.../snn/use_post_finetuning_artifacts_true/...
```

false config 直接报错。

---

# 9. `snn2/modeling.py`：Prefix stage 必须拆开

当前：

```python
prefix_ids_for_stage(..., stage="final_ann_evaluation")
prefix_ids_for_stage(..., stage="final_snn_evaluation")
```

共享 `final_evaluation_prefix_artifact_stage(cfg)`。

必须拆开。

## Final ANN

保持当前 mode-aware source，不受新参数影响。

## Final SNN

当：

```yaml
conversion.use_post_finetuning_artifacts: true
```

读取：

```text
layout.post_finetuning_prefix_dir
```

当 false：

```text
layout.ann_training_prefix_dir
```

`prefix_key_values_for_stage()` 同样修改，不能只改 token ids。

确保 Prefix state 与 fixed KV 总是来自同一个 selected directory。

---

# 10. `snn2/conversion.py`

这是本次修改的核心。

## 10.1 source selection

所有以下信息都必须从新参数统一派生，禁止继续按 `ann_mode` 推断：

- conversion Prefix root
- conversion calibration root
- `prefix_source_stage`
- `calibration_source_stage`
- `reused_ann_training_artifacts`
- `post_finetuning_recalibration`

对应：

### true

```text
prefix_source_stage = post_finetuning
calibration_source_stage = post_finetuning
reused_ann_training_artifacts = false
post_finetuning_recalibration = true
```

### false

```text
prefix_source_stage = pre_finetuning
calibration_source_stage = ann_training
reused_ann_training_artifacts = true
post_finetuning_recalibration = false
```

并建议 conversion metadata 显式新增：

```json
"use_post_finetuning_artifacts": true
```

这样 descriptor 自描述，不需要仅从两个 stage 字段反推。

## 10.2 aware training provenance 与 artifact source selection 必须解耦

当前 `_source_bundle()` 在 `reused=True` 时会调用 `_validate_aware_training_provenance()`。

新方案下：

```text
unaware + false
```

也会 `reused_ann_training_artifacts=true`，但 unaware 没有 aware ANN surrogate training provenance，不能因此要求 `training_result.json` 里存在 aware calibration/Clip provenance。

因此改成：

```python
selected_pre_finetuning_bundle = not use_post_finetuning_artifacts(cfg)

needs_aware_training_provenance = (
    selected_pre_finetuning_bundle
    and is_aware_ann_mode(cfg)
)
```

只有：

- `phase_aware + false`
- `gif_aware + false`

需要校验“当前 conversion 使用的 Pre-finetuning Prefix / ANN-training calibration 确实就是训练时固定的那些 artifact”。

以下情况不要调用 aware training provenance validator：

- unaware + false
- vanilla + true
- unaware + true
- phase_aware + true
- gif_aware + true

特别是：

**aware + true 使用的是 Final ANN 重新统计的 Post-finetuning bundle，不应被强制要求 conversion calibration hash 等于 ANN-training calibration。**

## 10.3 `_validate_source_manifest()`

不要再把 `reused` 与“aware mode”绑定。

它应只根据 selected source 判断：

- pre/ANN-training bundle -> 检查 ANN-training calibration manifest
- post bundle -> 检查 Post-finetuning calibration manifest

ANN-training bundle 的 `conversion_reuse_policy` 必须接受 unaware。

## 10.4 conversion metadata schema

建议 bump：

```python
CONVERSION_METADATA_FORMAT_VERSION
```

因为：

- artifact source policy 语义改变
- metadata 新增 selector
- old descriptor 不应被新 validator 当作等价 descriptor

`create_conversion()` 与 `validate_conversion_metadata()` 都加入：

```json
"use_post_finetuning_artifacts": <bool>
```

并严格验证与当前 config 一致。

旧 descriptor 应被拒绝并提示重新运行 `convert_snn.py`。

---

# 11. `snn2/evaluation.py`

## 11.1 SNN controller

`build_evaluation_controller()` 的 SNN：

```python
site_root=layout.conversion_site_dir
```

可以保留，只要 `layout.conversion_site_dir` 已经按新参数选源。

## 11.2 `evaluation_calibration_metadata()`

ANN 分支保持原逻辑：

- aware final ANN -> ANN-training calibration
- identity ANN -> no SNN calibration metadata

SNN 分支改成由新参数决定：

### true

```json
{
  "calibration_source_stage": "post_finetuning",
  "reused_ann_training_artifacts": false,
  "post_finetuning_recalibration": true,
  "calibration_root": "<post_finetuning_site_dir>",
  "use_post_finetuning_artifacts": true
}
```

### false

```json
{
  "calibration_source_stage": "ann_training",
  "reused_ann_training_artifacts": true,
  "post_finetuning_recalibration": false,
  "calibration_root": "<ann_training_site_dir>",
  "use_post_finetuning_artifacts": false
}
```

建议只对 SNN metadata 写 selector；若为了 schema 统一对 ANN 也写，可以写 `None`，但测试要明确，不要产生歧义。

---

# 12. `evaluate_tldr.py` / `evaluate_lm_harness.py`

两者必须同步修改。

## 12.1 Prefix actual load

SNN：

```python
prefix_stage = "final_snn_evaluation"
```

继续保留，但 `modeling.py` 必须按 selector 选源。

## 12.2 metrics 中的 `prefix_root` / `prefix_source_stage`

当前代码对 Final ANN 和 SNN 都可能使用：

```python
layout.conversion_prefix_dir
final_evaluation_prefix_artifact_stage(cfg)
```

这在新方案下会造成 Final ANN metadata 被 conversion selector 污染。

必须分开：

### Final ANN

根据原来的 ANN evaluation 规则决定 root/stage。

### Final SNN

使用：

```python
layout.conversion_prefix_dir
conversion_prefix_artifact_stage(cfg)
```

或 `final_snn_evaluation_prefix_artifact_stage(cfg)`。

SNN metrics 建议显式保存：

```json
"use_post_finetuning_artifacts": true|false,
"prefix_source_stage": "pre_finetuning|post_finetuning",
"calibration_source_stage": "ann_training|post_finetuning"
```

必须与 conversion descriptor 完全一致。

## 12.3 SNN evaluation output path

由于 `model_output_dir = layout.snn_dir(args.neuron)`，只要 `ArtifactLayout.snn_dir()` 按第 8 节修改，TL;DR 与 lm-harness 的结果都会自动隔离。

确认两种 selector 不会覆盖：

```text
snn/use_post_finetuning_artifacts_true/...
snn/use_post_finetuning_artifacts_false/...
```

---

# 13. `scripts/verify_artifacts.py`

所有当前按：

```python
is_aware_ann_mode()
conversion_reuses_ann_training_artifacts()
```

隐式推断 source 的地方都要改成按 selector。

必须验证：

## true

- conversion Prefix root 是 Post-finetuning Prefix
- calibration root 是 Post-finetuning Stage A
- manifest source checkpoint 是当前 Final ANN
- `prefix_source_stage=post_finetuning`
- `calibration_source_stage=post_finetuning`
- `post_finetuning_recalibration=true`
- descriptor/metrics `use_post_finetuning_artifacts=true`

## false

- mode 不能是 vanilla
- Prefix root 是 shared Pre-finetuning Prefix
- calibration root 是 shared ANN-training Stage A
- `prefix_source_stage=pre_finetuning`
- `calibration_source_stage=ann_training`
- `post_finetuning_recalibration=false`
- descriptor/metrics `use_post_finetuning_artifacts=false`

对于：

```text
unaware + false
```

不要要求 aware ANN training `training_result.json` provenance。

对于：

```text
phase/gif aware + false
```

继续校验 selected shared artifacts 与 ANN training recorded provenance 一致。

对于：

```text
phase/gif aware + true
```

验证 Post-finetuning Stage A 的 source checkpoint/hash，而不是要求其等于 ANN-training states。

---

# 14. `实验执行总结.md` 必须重写的内容

这是用户明确要求同步修改的主执行文档。

## Step 1

新增：

```yaml
conversion:
  use_post_finetuning_artifacts: true
```

解释：

- 只控制 SNN conversion / SNN Evaluation artifact source
- true = Step 7 + Step 8
- false = Step 4 + Step 5 Stage A
- vanilla 只能 true

删除旧表述：

> aware 的 Post-finetuning Prefix 与 calibration 被 resolved config 强制关闭

## Step 4

继续：

- Rotation 每个 model-task 一次
- Pre-finetuning Prefix 用 unaware config 每个 model-task 生成一次
- unaware / phase-aware / gif-aware 共享

并明确：

> 该 Prefix 除了用于 ANN training/原有流程，也作为 `use_post_finetuning_artifacts=false` 时三种 non-vanilla mode 的 SNN conversion Prefix。

## Step 5

改成：

- ANN-training Stage A 仍用 phase-aware config 每个 model-task/group/num_samples 生成一次
- unaware / phase-aware / gif-aware 共享同一 Stage A
- Stage B 仍服务 phase/gif aware ANN training common Clip
- **unaware 不需要单独再跑一份 Step 5**
- SNN false 分支只消费 Stage A，不消费 Stage B Clip

## Step 7

从原来的仅 vanilla/unaware 改成 **12 个 Final ANN checkpoint 全部生成 Post-finetuning Prefix**：

```bash
for CFG in "${ALL_CFGS[@]}"; do
  python scripts/discover_prefix.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

删除：

> phase_aware / gif_aware 必须复用 Pre-finetuning Prefix，脚本拒绝 post_finetuning

改成：

> 四种 mode 都允许生成自己的 Final-ANN-specific Post-finetuning Prefix。

## Step 8

从原来的仅 vanilla/unaware 改成 **12 个 Final ANN checkpoint 全部生成 Post-finetuning conversion Stage A**：

```bash
for CFG in "${ALL_CFGS[@]}"; do
  python scripts/calibrate_sites.py \
    --config "$CFG" \
    --stage post_finetuning \
    --calibration-phase A
done
```

明确：

- 只跑 Stage A
- 不用于 ANN fine-tuning
- 保存于各自 `seed42/post_finetuning/...`
- aware 同样允许

## Step 9

Final ANN Evaluation 规则保持不变，并明确：

> `conversion.use_post_finetuning_artifacts` 不影响 Step 9。

## Step 10

改成 selector matrix：

| mode | `use_post_finetuning_artifacts=false` | `true` |
|---|---|---|
| vanilla | 非法 | Step 7 Prefix + Step 8 Stage A |
| unaware | Step 4 Prefix + Step 5 Stage A | Step 7 Prefix + Step 8 Stage A |
| phase_aware | Step 4 Prefix + Step 5 Stage A | Step 7 Prefix + Step 8 Stage A |
| gif_aware | Step 4 Prefix + Step 5 Stage A | Step 7 Prefix + Step 8 Stage A |

明确：

- 两套 artifacts 在 Step 10 前已经同时存在
- 切换 selector 不重新训练 ANN
- 也不需要重新跑 Step 4/5/7/8
- 只重新 `convert_snn.py` 和 SNN evaluation
- 输出路径用 selector 隔离

---

# 15. 测试修改

至少更新：

- `tests/test_generated_configs.py`
- `tests/test_post_finetuning_protocol.py`
- `tests/test_conversion_metadata.py`
- `tests/test_evaluation_paths.py`
- `tests/test_verify_artifacts.py`（若当前文件名/结构不同，更新对应 verify tests）
- 与 calibration manifest provenance 相关的 tests

## 15.1 Config tests

覆盖：

1. 四种 generated config 都含：
   ```yaml
   conversion.use_post_finetuning_artifacts: true
   ```
2. unaware/phase/gif `false` 合法
3. vanilla `false` 报错
4. aware config 不再被 `resolve_config()` 强制：
   - `post_finetuning.prefix_enabled=false`
   - `rediscover_prefix=false`
   - `recalibrate_sites=false`

## 15.2 protocol table

至少覆盖：

```text
vanilla + true
unaware + true
unaware + false
phase_aware + true
phase_aware + false
gif_aware + true
gif_aware + false
```

验证 selected:

- Prefix stage
- calibration stage
- Prefix root
- calibration root
- reused flag
- post recalibration flag

## 15.3 Final ANN independence

同一 mode 下分别：

```yaml
use_post_finetuning_artifacts=true
use_post_finetuning_artifacts=false
```

验证：

- `ann_checkpoint_dir` 完全相同
- Final ANN evaluation Prefix source 完全相同
- `snn_dir()` 不同
- conversion source root 不同

特别为 `phase_aware` 加测试，防止 selector true 导致 Final ANN evaluation 错误读取 Post-finetuning Prefix。

## 15.4 shared unaware pre bundle

验证：

```text
ArtifactLayout(unaware false).conversion_prefix_dir
==
ArtifactLayout(phase_aware false).ann_training_prefix_dir
```

以及：

```text
ArtifactLayout(unaware false).conversion_site_dir
==
ArtifactLayout(phase_aware false).ann_training_site_dir
```

不要要求 unaware 有自己的 ANN-training calibration 目录。

## 15.5 aware Post-finetuning

验证：

- phase/gif aware Post-finetuning prefix path 位于自身 `seed42/post_finetuning`
- Post-finetuning site path 位于自身 `seed42/post_finetuning`
- 不等于 `_shared/.../ann_training_calibration`
- 两个 aware mode 的 Final ANN checkpoint 不共享 Post-finetuning artifact

## 15.6 SNN path

所有 mode 的 `snn_dir()` 必须在 `snn` 后紧跟：

```text
use_post_finetuning_artifacts_true
```

或：

```text
use_post_finetuning_artifacts_false
```

selector segment 必须恰好出现一次。

vanilla false 不应能构造合法 resolved config。

## 15.7 conversion metadata

两套 source 分别测试 hash/provenance。

重点回归：

- `unaware + false` 不触发 aware training provenance validator
- `phase_aware + false` 仍触发 ANN-training artifact provenance 一致性检查
- `phase_aware + true` 使用 Post-finetuning source，不要求 conversion calibration hash 等于 training calibration

---

# 16. 建议的实现后行为表

最终必须得到：

| ANN mode | selector | SNN Prefix | SNN calibration | source model of calibration |
|---|---:|---|---|---|
| vanilla | true | Post-finetuning Prefix | Post-finetuning Stage A | Final vanilla ANN |
| vanilla | false | **非法** | **非法** | - |
| unaware | true | Post-finetuning Prefix | Post-finetuning Stage A | Final unaware ANN |
| unaware | false | shared Pre-finetuning Prefix | shared ANN-training Stage A | rotated fused Base |
| phase_aware | true | Post-finetuning Prefix | Post-finetuning Stage A | Final phase-aware ANN |
| phase_aware | false | shared Pre-finetuning Prefix | shared ANN-training Stage A | rotated fused Base |
| gif_aware | true | Post-finetuning Prefix | Post-finetuning Stage A | Final gif-aware ANN |
| gif_aware | false | shared Pre-finetuning Prefix | shared ANN-training Stage A | rotated fused Base |

注意：

`phase_aware/gif_aware + true` 的 Post-finetuning calibration 是在 Final ANN checkpoint 上重新统计的独立 conversion calibration，不用于 ANN fine-tuning，也不覆盖原有 shared ANN-training calibration。

---

# 17. 完成标准

修改完成后必须满足：

1. `pytest -q` 全部通过。
2. `scripts/materialize_configs.py` 正常生成 12 个 config。
3. aware config 不再禁止：
   ```bash
   discover_prefix.py --stage post_finetuning
   calibrate_sites.py --stage post_finetuning --calibration-phase A
   ```
4. Step 4/5 与 Step 7/8 artifacts 能同时存在，互不覆盖。
5. `unaware + false` 能直接消费 Step 4/5 的 shared artifacts。
6. `phase/gif aware + true` 能使用 Final ANN 自己的 Step 7/8 artifacts。
7. vanilla false 在 config validation 阶段失败。
8. selector true/false 不改变 ANN checkpoint path。
9. selector true/false 的 SNN descriptor/evaluation path 完全隔离：
   ```text
   snn/use_post_finetuning_artifacts_true/...
   snn/use_post_finetuning_artifacts_false/...
   ```
10. conversion descriptor、SNN metrics、`verify_artifacts.py` 对 Prefix source、calibration source、hash 和 selector 的判断完全一致。
11. Final ANN Evaluation 不受 selector 影响。
12. `实验执行总结.md` 的 Step 1、4、5、7、8、9、10 与新代码完全一致。

---

# 18. 不要做的事情

- 不要为 true/false 重新训练两份 ANN。
- 不要把 selector 加入 ANN run root；它只应进入 `snn/` 子树。
- 不要让 selector 改变 Final ANN Evaluation。
- 不要让 unaware 单独重新生成一份 ANN-training Stage A；它应复用 Step 5 shared artifact。
- 不要把 aware Post-finetuning artifacts 放进 `_shared`。
- 不要删除原有 Pre-finetuning artifacts。
- 不要在生成 Post-finetuning artifacts 时覆盖 ANN-training artifacts。
- 不要通过 `resolve_config()` 静默把 vanilla false 改回 true；必须 validation error。
- 不要继续使用 `is_aware_ann_mode()` 作为 SNN conversion artifact source 的决定条件。
- 不要因为 `unaware + false` 使用 ANN-training Stage A 就误要求 aware ANN training provenance。
- 不要让 `evaluation.prefix_enabled` 与 `conversion.use_post_finetuning_artifacts` 合并成同一个开关。
