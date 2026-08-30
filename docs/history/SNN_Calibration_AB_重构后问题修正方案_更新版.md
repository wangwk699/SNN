# SNN Calibration A/B 重构后问题修正方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 适用基线：已完成 `SNN_Calibration_AB_Phase_MTN_Sweep_重构实施方案.md` 中 A/B 两阶段重构后的当前 `main`。  
> 本文档只处理本次代码检查中发现的剩余问题，不重复上一份总体重构方案。  
> 要求：部署在服务器上的 Codex 在无对话上下文的情况下，仅凭本文档即可完成本轮修正。

---

# 1. 本轮修正范围与优先级

本轮共修 6 项。

## Blocker，必须优先修

1. `calibration.num_samples` sweep 与 Rotation regression 仍然冲突。
2. Prefix discovery 仍然跟随 `calibration.num_samples`，但 Prefix artifact 路径未按 `num_samples` 隔离。
3. Stage B mask-aware Clip validator 只验证 schema/hash，没有重新验证真实 mask-aware 语义。
4. 多次 Stage B 会覆盖 Stage A calibration root 中的 `resolved_config.yaml` 和共用日志语义。

## 收尾项

5. 删除 `build_site_states(..., include_clip=True)` 旧入口，使 A/B API 真正分离。
6. `regress_phase_conversion.py` 增加 deployment `phase.T` override 支持，允许 regression 检查 training T 与 deployment T 不同。

完成后必须运行：

```bash
pytest -q
```

全部通过后再进行正式 `calibration.num_samples` sweep。

---

# 2. 修正 1：Rotation regression 与 `calibration.num_samples` 解耦

## 2.1 当前问题

当前：

```python
scripts/prepare_rotation.py
```

会读取：

```python
load_selected_raw(cfg, layout).calibration
```

这意味着 rotation regression 会跟随当前：

```yaml
calibration:
  num_samples: N
```

但是：

```python
validate_rotation_regression_suite()
```

仍硬编码要求：

```text
N == 128
```

同时 `verify_artifacts.py` 又要求 rotation regression 中记录的 calibration manifest 必须等于当前 config 的：

```text
data/calibration/num_samples_N/calibration_manifest.json
```

因此：

```text
num_samples != 128
```

时：

- 重新跑 rotation → 直接失败；
- 不重新跑 rotation → verify_artifacts 又因为 manifest/path/hash 不匹配失败。

---

## 2.2 最终设计

Rotation regression **永久使用独立固定的 canonical 128-sample selection**。

它不属于 Stage A `calibration.num_samples` sweep。

原则：

```text
Rotation regression samples = fixed 128
Stage A calibration samples  = cfg.calibration.num_samples
```

二者完全解耦。

---

## 2.3 新数据路径

新增 canonical preprocessing calibration manifest：

```text
_shared/seed42/data/
├── train_manifest.json
├── validation_manifest.json
├── evaluation_manifest.json
├── calibration/
│   ├── num_samples_64/
│   │   └── calibration_manifest.json
│   ├── num_samples_128/
│   │   └── calibration_manifest.json
│   └── ...
└── canonical_preprocessing/
    └── num_samples_128/
        └── calibration_manifest.json
```

也可以命名为：

```text
canonical_calibration/
num_samples_128/
```

关键要求是：

- 与 Stage A `calibration/num_samples_N/` 分离；
- 永远固定 128；
- 使用固定 calibration seed；
- Rotation regression 和 Prefix discovery 共用这套 canonical 128 samples。

建议 ArtifactLayout 新增：

```python
canonical_preprocessing_calibration_dir
canonical_preprocessing_calibration_manifest_path
```

---

## 2.4 Data 层实现

在：

```text
snn2/data.py
```

新增 helper，例如：

```python
CANONICAL_PREPROCESSING_NUM_SAMPLES = 128
```

或者统一放入 `temporal_ops.py` / dedicated constants module。

新增：

```python
prepare_canonical_preprocessing_manifest(cfg, layout)
load_canonical_preprocessing_raw(cfg, layout)
```

canonical sampling 必须：

```text
without replacement
seed = cfg["calibration"]["seed"]
num_samples = 128
```

注意：

- 不要临时修改 `cfg["calibration"]["num_samples"] = 128` 再复用普通 Stage A loader；
- canonical 和 sweep calibration 必须在代码语义上明确分离。

---

## 2.5 `prepare_data.py`

`prepare_data.py` 必须同时保证：

```text
普通 calibration manifest：
    num_samples = cfg.calibration.num_samples

canonical preprocessing manifest：
    num_samples = 128
```

如果两者恰好都是 128，也不要让 canonical path 指向普通 sweep path。

---

## 2.6 `prepare_rotation.py`

改为：

```python
calibration = load_canonical_preprocessing_raw(cfg, layout)
manifest_path = layout.canonical_preprocessing_calibration_manifest_path
```

不要再使用：

```python
load_selected_raw(cfg, layout).calibration
```

作为 rotation regression 数据来源。

---

## 2.7 Rotation regression validation

`validate_rotation_regression_suite()`：

保留固定 128 要求：

```python
expected_samples = 128
```

但是不能再从：

```python
cfg["calibration"]["num_samples"]
```

读取。

建议：

```python
if len(calibration_dataset) != 128:
    raise RuntimeError(...)
```

result metadata：

```json
{
  "num_samples": 128,
  "sample_source": "canonical_preprocessing_calibration",
  "calibration_manifest_path": "...canonical_preprocessing/num_samples_128/...",
  "calibration_manifest_sha256": "..."
}
```

---

## 2.8 `verify_artifacts.py`

不要再将 rotation regression manifest 与：

```python
layout.calibration_data_manifest_path
```

比较。

改为：

```python
layout.canonical_preprocessing_calibration_manifest_path
```

要求：

```text
rotation regression num_samples == 128
rotation regression manifest path == canonical path
rotation regression manifest hash == canonical manifest hash
```

Rotation verification 完全不受当前：

```text
cfg.calibration.num_samples
```

影响。

---

# 3. 修正 2：Prefix discovery 随 `calibration.num_samples` sweep，并按 `num_samples` 隔离 artifact

## 3.1 当前问题

当前 `scripts/discover_prefix.py` 使用：

```python
load_selected_raw(cfg, layout).calibration
```

因此 Prefix discovery 实际会随：

```text
calibration.num_samples = 64 / 128 / 256 / ...
```

使用不同的 calibration sample selection。

但当前 Prefix artifact 路径仍是共享的，例如 aware 模式：

```text
_shared/.../pre_finetuning_prefix/
├── prefix_state.json
└── prefixed_key_values.pt
```

这里没有 `num_samples`，因此不同 sweep 会互相覆盖。

例如：

```text
num_samples=64  -> Prefix A
num_samples=128 -> Prefix B
```

第二次 Prefix discovery 会覆盖第一次 Prefix artifact，从而导致：

- 已有 Stage A provenance 中 Prefix hash 失效；
- 已训练 aware ANN provenance 中 Prefix hash 失效；
- 不同 `num_samples` 实验无法并行保留对应 Prefix；
- calibration / training / conversion 无法稳定追踪实际使用的 Prefix。

---

## 3.2 最终设计

本轮确认：

> Prefix discovery 必须随 Stage A `calibration.num_samples` sweep 改变。

因此：

```text
Prefix discovery samples = cfg.calibration.num_samples
Stage A samples          = cfg.calibration.num_samples
```

二者使用同一个当前 sweep calibration selection。

Rotation regression 与此不同：

```text
Rotation regression = 固定 canonical 128
Prefix discovery     = 当前 cfg.calibration.num_samples
Stage A calibration  = 当前 cfg.calibration.num_samples
```

---

## 3.3 Prefix artifact 路径按 `num_samples` 隔离

### Pre-finetuning / ANN-training Prefix

当前：

```text
_shared/seed42/rotated_prefix/pre_finetuning_prefix/
```

改为：

```text
_shared/seed42/rotated_prefix/pre_finetuning_prefix/
└── num_samples_<N>/
    ├── prefix_state.json
    └── prefixed_key_values.pt
```

例如：

```text
pre_finetuning_prefix/
├── num_samples_64/
│   ├── prefix_state.json
│   └── prefixed_key_values.pt
├── num_samples_128/
│   ├── prefix_state.json
│   └── prefixed_key_values.pt
└── num_samples_256/
    ├── prefix_state.json
    └── prefixed_key_values.pt
```

### Post-finetuning Prefix

同样必须按 `num_samples` 隔离：

```text
post_finetuning/
└── prefix/
    └── num_samples_<N>/
        ├── prefix_state.json
        └── prefixed_key_values.pt
```

不能只修改 aware 的 pre-finetuning Prefix。

---

## 3.4 `ArtifactLayout` 修改

建议保留 base dir 与当前 variant dir 两层。

例如：

```python
@property
def ann_training_prefix_base_dir(self):
    return self.policy_root / "pre_finetuning_prefix"

@property
def ann_training_prefix_dir(self):
    return (
        self.ann_training_prefix_base_dir
        / f"num_samples_{int(self._cfg['calibration']['num_samples'])}"
    )
```

Post-finetuning 同理：

```python
@property
def post_finetuning_prefix_base_dir(self):
    return self.post_finetuning_dir / "prefix"

@property
def post_finetuning_prefix_dir(self):
    return (
        self.post_finetuning_prefix_base_dir
        / f"num_samples_{int(self._cfg['calibration']['num_samples'])}"
    )
```

所有调用方继续统一通过：

```text
layout.ann_training_prefix_dir
layout.post_finetuning_prefix_dir
layout.conversion_prefix_dir
```

访问 Prefix，禁止在脚本中手工拼接 `num_samples`。

---

## 3.5 `discover_prefix.py`

Prefix discovery 继续使用当前 Stage A calibration selection：

```python
load_selected_raw(cfg, layout).calibration
```

这一点不要改成 canonical 128 loader。

例如：

```text
cfg.calibration.num_samples = 64
```

则使用：

```text
data/calibration/num_samples_64/calibration_manifest.json
```

中的 64 条样本。

```text
cfg.calibration.num_samples = 256
```

则使用对应 256 条样本。

---

## 3.6 `prefix_state.json` 增加数据 provenance

建议增加：

```json
{
  "discovery_num_samples": 64,
  "discovery_data_source": "stage_a_calibration_selection",
  "discovery_manifest_path": ".../data/calibration/num_samples_64/calibration_manifest.json",
  "discovery_manifest_sha256": "..."
}
```

具体来源必须是当前：

```python
layout.calibration_data_manifest_path
```

不能只靠 Prefix 目录名隐式表达。

---

## 3.7 Prefix KV provenance

若 Prefix 非空并生成：

```text
prefixed_key_values.pt
```

必须保证：

```text
prefix_state.json
prefixed_key_values.pt
```

来自同一个：

```text
num_samples_N
```

Prefix root。

禁止从另一个 `num_samples` 目录复用 KV。

---

## 3.8 Calibration provenance

`calibration_provenance()` 中已有：

```text
prefix_state_path
prefix_state_sha256
prefix_kv_path
prefix_kv_sha256
```

路径改造后，这些字段必须自动指向当前：

```text
.../pre_finetuning_prefix/num_samples_N/
```

或：

```text
.../post_finetuning/prefix/num_samples_N/
```

并验证：

```text
prefix discovery num_samples == cfg.calibration.num_samples
prefix discovery manifest path == layout.calibration_data_manifest_path
prefix discovery manifest hash == current calibration manifest hash
```

---

## 3.9 Training provenance

aware ANN training 必须绑定当前 `num_samples` 对应 Prefix。

例如：

```text
num_samples_64 ANN training
```

必须记录：

```text
ann_training_prefix_root = .../pre_finetuning_prefix/num_samples_64
```

不能指向其他 sweep。

现有：

```text
ann_training_prefix_state_sha256
ann_training_prefix_kv_sha256
ann_training_prefix_token_ids
```

继续保留。

建议增加：

```text
ann_training_prefix_num_samples
ann_training_prefix_discovery_manifest_sha256
```

---

## 3.10 Conversion / evaluation

### Aware

Aware conversion 继续复用训练时的 pre-finetuning Prefix，但必须是该 ANN training 对应的：

```text
num_samples_N
```

Prefix。

由于 aware ANN run path 已包含：

```text
num_samples_N_...
```

conversion provenance 必须校验：

```text
training_result 中 Prefix root
==
当前 layout.ann_training_prefix_dir
```

### Vanilla / unaware

Post-finetuning Prefix discovery 同样使用当前：

```text
calibration.num_samples=N
```

并保存到：

```text
post_finetuning/prefix/num_samples_N/
```

其 post-finetuning conversion calibration 已经按：

```text
calibration_group_size_G_num_samples_N
```

隔离，两者必须一致。

---

## 3.11 `verify_artifacts.py`

Prefix validation 必须检查：

```text
Prefix root dirname == num_samples_<cfg.calibration.num_samples>
```

并验证：

```text
prefix_state.discovery_num_samples
==
cfg.calibration.num_samples
```

以及：

```text
prefix_state.discovery_manifest_path
==
layout.calibration_data_manifest_path
```

和对应 SHA-256 一致。

对于 aware：

```text
training_result 中记录的 Prefix root/hash
```

必须与当前 `num_samples` 对应 Prefix 一致。

对于 vanilla/unaware：

```text
post-finetuning Prefix
```

必须与当前 post-finetuning calibration 的 `num_samples` 一致。

# 4. Rotation / Prefix / Stage A 的最终数据关系

最终数据依赖关系：

```text
prepare_data
│
├── train/validation/evaluation shared manifests
│
├── canonical preprocessing calibration = fixed 128
│      └── prepare_rotation
│
└── Stage A calibration selection = cfg.calibration.num_samples
       ├── discover_prefix
       └── calibrate_sites --calibration-phase A
```

也就是说：

```text
Rotation regression
    固定 canonical 128

Prefix discovery
    跟随 cfg.calibration.num_samples

Stage A
    跟随 cfg.calibration.num_samples
```

改变：

```yaml
calibration.num_samples
```

时：

### 不需要重跑

```text
prepare_rotation
```

### 必须重跑

```text
discover_prefix
Stage A
Stage B
aware ANN fine-tuning
non-aware post-finetuning Prefix/calibration/conversion as appropriate
```

不同 `num_samples` 对应的 Prefix artifact 通过：

```text
prefix/.../num_samples_N/
```

并行保留，不允许互相覆盖。

# 5. 修正 3：Stage B validator 必须真正验证 mask-aware Clip

## 5.1 当前问题

当前：

```python
build_clip_state()
```

的计算逻辑基本正确：

```text
all-low  -> gif_low
all-high -> gif_high
mixed    -> gif_low ∩ gif_high
```

Site 1：

```text
q/k/v role-specific
```

Site 7：

```text
gate/up role-specific
```

但：

```python
validate_clip_profile()
```

目前主要验证：

```text
profile hash
Stage A hash
T
role schema
Clipper(state)
lower < upper
```

并没有从 Stage A 的真实：

```text
mask_low
mask_low_by_role
low_scale/zero
high_scale/zero
```

重新计算 expected Clip。

因此如果以后 Stage B generator 被误改，validator 可能仍然放行。

---

# 6. 新 mask-aware validation helper

建议在：

```text
snn2/state_validation.py
```

新增纯函数：

```python
recompute_expected_clip_from_stage_a(
    phase_state,
    gif_state,
    mtn_state,
    *,
    phase_T,
    mtn_T,
)
```

最好直接复用 calibration 中的数学 helper，而不是复制公式。

更推荐把以下纯逻辑提取到独立公共模块，例如：

```text
snn2/clip_policy.py
```

包括：

```python
gif_group_classification(...)
gif_representable_ranges(...)
phase_bound(...)
mtn_bound(...)
apply_gif_constraints(...)
expected_clip_intervals(...)
```

然后：

```text
calibration Stage B
state_validation
tests
```

全部调用同一套低层 helper。

注意：

> validator 不能直接“相信并返回 build_clip_state() 的结果”然后只比较同源代码，否则 generator 和 validator 同时出错时测试失效。

可以共用数学原子 helper，但 validation 必须独立对 serialized state 做完整比较。

---

# 7. 单 role salient Site validation

适用于：

```text
Site 3/4/6/10
```

从 Stage A：

```python
mask = gif_state["mask_low"]
```

重新计算：

```text
all_low / all_high / mixed
```

断言：

```python
torch.equal(
    state["gif_group_classification"],
    expected_classification,
)
```

然后逐 group 验证：

### all-low

```text
lower >= phase/mtn base lower
upper <= phase/mtn base upper
lower >= gif_low_min
upper <= gif_low_max
```

并且最终 interval 必须等于预期：

```text
max(base_lower, gif_low_min)
min(base_upper, gif_low_max)
```

### all-high

同理只使用：

```text
gif_high
```

### mixed

使用：

```text
gif_low ∩ gif_high
```

最终建议直接：

```python
torch.testing.assert_close(
    serialized_lower,
    expected_lower,
)
torch.testing.assert_close(
    serialized_upper,
    expected_upper,
)
```

validation 代码中使用显式 tolerance。

---

# 8. Site 1 / Site 7 role-specific validation

## Site 1

必须独立验证：

```text
q
k
v
```

## Site 7

独立验证：

```text
gate
up
```

对于每个 role：

```python
expected_classification = classify(mask_low_by_role[role])
```

然后验证：

```text
gif_group_classification_by_role[role]
lower_by_role[role]
upper_by_role[role]
```

不得将多个 role 的 mask 合并。

---

# 9. Stage B summary counts validation

`calibration_summary.json` 中：

单 role：

```json
"gif_group_class_counts": {
  "all_low": ...,
  "all_high": ...,
  "mixed": ...
}
```

Site 1/7：

```json
"gif_group_class_counts_by_role": ...
```

validator 必须根据 actual classification tensor 重算 count 并比较。

不能只相信 summary。

---

# 10. Site 2 / 8 / 9 validation

同时补全：

## Site 2

验证：

```text
rule = intersection(phase, mtn, gif_low)
```

以及 serialized interval 等于：

```text
base ∩ gif_low
```

## Site 8/9

验证：

```text
rule = intersection(phase, mtn)
```

以及：

```text
lower == -min(phase_bound, mtn_bound)
upper == +min(phase_bound, mtn_bound)
```

即最终应对称。

## Site 5

继续：

```text
no clip_state.pt
```

---

# 11. 修正 4：Stage A config/log 与 Stage B profile config/log 分离

## 11.1 当前问题

`calibrate_sites.py` 无论 A/B 都执行：

```python
setup(args.config, config_scope=scope)
```

而：

```python
setup()
```

会写：

```text
resolved_config.yaml
```

当前 A/B 又共用：

```text
ann_training_calibration/.../config/
ann_training_calibration/.../logs/
```

因此：

```text
Stage A T=4/M=4
Stage B T=2/M=2
Stage B T=8/M=8
```

最终：

```text
ann_training_calibration/.../config/resolved_config.yaml
```

会被最后一次 Stage B 的 8/8 配置覆盖。

虽然 Stage A state 没变，但 metadata 表达错误。

---

# 12. 新 Stage A config/log 路径

Calibration root：

```text
calibration_group_size_-1_num_samples_128/
├── config/
│   └── resolved_config.yaml          # Stage A only
├── logs/
│   ├── calibrate_sites_*_phase_A.*
│   └── ...
├── sites/
└── clip_profiles/
```

Stage A：

```text
config/
logs/
```

仍位于 calibration root。

---

# 13. 新 Stage B config/log 路径

每个 profile：

```text
clip_profiles/
└── phase_T_4_mtn_T_8/
    ├── config/
    │   └── resolved_config.yaml
    ├── logs/
    │   ├── calibrate_sites_*_phase_B.log
    │   ├── ...
    │   └── *_result.json
    ├── layer_000/...
    └── clip_profile_manifest.json
```

新增 ArtifactLayout：

```python
ann_training_clip_profile_config_dir
ann_training_clip_profile_logs_dir

post_finetuning_clip_profile_config_dir
post_finetuning_clip_profile_logs_dir
```

---

# 14. `setup()` 增加 Stage B scope

建议增加：

```text
ann_training_clip_profile
post_finetuning_clip_profile
```

例如：

```python
elif config_scope == "ann_training_clip_profile":
    config_dir = layout.ann_training_clip_profile_config_dir
```

---

# 15. `calibrate_sites.py`

不要在解析 A/B 前先使用同一个 scope。

改为：

```python
if args.calibration_phase == "A":
    scope = ...
else:
    scope = ...
```

Stage B：

```python
cfg, layout = setup(
    args.config,
    config_scope="ann_training_clip_profile",
)
```

logs_dir 也使用 profile-specific logs。

---

# 16. Stage A manifest 不受 Stage B config overwrite

增加测试：

1. Stage A 使用：
   ```text
   phase.T=4
   mtn.T=4
   ```
2. 记录：
   ```text
   Stage A resolved_config.yaml bytes/hash
   ```
3. Stage B 运行：
   ```text
   2/2
   8/8
   ```
4. 断言 Stage A：
   ```text
   resolved_config.yaml
   logs
   Stage A state
   ```
   均未被 Stage B 修改。

---

# 17. 修正 5：彻底删除 `include_clip` 旧 API

## 17.1 当前问题

当前仍存在：

```python
build_site_states(
    statistics,
    cfg,
    include_clip=True,
)
```

能够直接生成：

```text
phase/gif/mtn/clip
```

这与 A/B 两阶段职责冲突。

测试也仍有：

```text
test_build_site_states_with_common_clip
```

---

## 17.2 最终要求

改成：

```python
build_site_states(statistics, cfg)
```

永远只返回：

```python
{
    "phase": ...,
    "gif": ...,
    "mtn": ...,
}
```

删除参数：

```text
include_clip
```

---

## 17.3 Clip 唯一入口

Clip 只能从：

```python
build_clip_state(...)
materialize_clip_profile(...)
```

产生。

禁止 Stage A helper 直接生成 Clip。

---

## 17.4 `materialize_calibration_states`

删除：

```python
include_clip
```

参数。

所有调用统一：

```python
materialize_calibration_states(
    site_root,
    cfg,
    metadata,
    expected_num_hidden_layers=...,
)
```

---

## 17.5 Tests

删除/重构：

```text
test_build_site_states_with_common_clip
```

增加：

```text
test_build_site_states_is_stage_a_only
```

断言：

```python
set(states) == {"phase", "gif", "mtn"}
```

---

# 18. 修正 6：Phase conversion regression 支持 deployment T override

## 18.1 当前问题

正式：

```text
convert_snn.py
evaluate_tldr.py
evaluate_lm_harness.py
```

已经支持 deployment：

```text
--phase-T
--mtn-T
--mtn-K
```

但：

```text
scripts/regress_phase_conversion.py
```

仍直接使用 YAML 中的：

```python
cfg["phase"]["T"]
```

因此它不能测试：

```text
ANN training phase.T = 4
deployment phase.T = 8
```

---

# 19. 新 CLI

改：

```python
regression_parser = parser(
    "...",
    deployment_overrides=True,
)
```

或显式加入：

```text
--phase-T
```

然后必须遵循与 `convert_snn.py` 相同的关键顺序：

```python
cfg, layout = setup(args.config)
apply_deployment_overrides(args, cfg)
```

注意：

> `layout` 必须先根据 training config 构造，再 apply deployment override。

否则 aware ANN root 会被 deployment T 改写到不存在的 training path。

当前 `_common.apply_deployment_overrides()` 的注释就是：

```text
Apply deployment-only T/K after ArtifactLayout fixed the source ANN run.
```

保持该语义。

---

# 20. Regression source T / deployment T 分开

regression metadata 中增加：

```text
source_ann_training_phase_T
deployment_phase_T
```

`source_ann_training_phase_T` 从：

```text
training_result.json
```

读取。

`deployment_phase_T` 来自 override 后：

```python
cfg["phase"]["T"]
```

不要假设相等。

---

# 21. Phase static oracle 的语义

需要特别注意。

当前 regression 比较：

```text
phase_static
vs
phase_temporal
```

如果 deployment T 与 training T 不同，那么需要明确 static oracle 使用哪个 T。

本轮建议：

> **static Phase oracle 使用 deployment phase.T。**

原因：

该 regression 的目标是检查：

```text
同一个 Phase neuron T 下
static sum representation
vs
temporal decomposition
```

而不是比较 ANN training 时 T 与 deployment T 的性能差异。

因此：

```text
phase_static T = deployment_phase_T
phase_temporal T = deployment_phase_T
```

source ANN checkpoint 仍来自：

```text
training phase.T = training_result 中记录的 T
```

metadata 同时记录二者。

---

# 22. 需要同步修改/审计的文件

本轮至少审计：

```text
scripts/prepare_data.py
scripts/prepare_rotation.py
scripts/discover_prefix.py
scripts/calibrate_sites.py
scripts/regress_phase_conversion.py
scripts/verify_artifacts.py

snn2/artifacts.py
snn2/data.py
snn2/calibration.py
snn2/state_validation.py
snn2/training.py
snn2/conversion.py
snn2/phase_conversion_regression.py
```

若抽出公共 Clip policy：

```text
snn2/clip_policy.py
```

同时更新 import。

Tests 至少：

```text
tests/test_calibration_profiles.py
tests/test_controller_state_loading.py
tests/test_generated_configs.py
tests/test_training.py
tests/test_conversion_metadata.py
tests/test_phase_conversion_regression.py
tests/test_verify_artifacts.py
```

建议新增：

```text
tests/test_canonical_preprocessing_data.py
```

---

# 23. 必须新增的 tests

## 23.1 canonical preprocessing 固定 128

config：

```text
calibration.num_samples=64
```

断言：

```text
Stage A calibration manifest -> 64
canonical preprocessing manifest -> 128
```

---

## 23.2 canonical 与 Stage A path 分离

即使：

```text
calibration.num_samples=128
```

仍断言：

```text
canonical path != Stage A sweep path
```

---

## 23.3 Rotation 不随 Stage A num_samples

分别：

```text
num_samples=64
num_samples=256
```

构造 layout，断言：

```text
canonical rotation regression manifest path 相同
```

---

## 23.4 Prefix 随 Stage A num_samples 变化并隔离

分别：

```text
num_samples=64
num_samples=256
```

断言：

```text
ann_training_prefix_dir 不同
post_finetuning_prefix_dir 不同
```

并分别包含：

```text
num_samples_64
num_samples_256
```

Prefix discovery 使用的 manifest 也分别为：

```text
data/calibration/num_samples_64/calibration_manifest.json
data/calibration/num_samples_256/calibration_manifest.json
```

---

## 23.5 Stage A 随 num_samples 变化

64/256：

```text
ann_training_calibration_dir 不同
calibration_data_manifest_path 不同
```

并验证对应 Prefix root 与 Stage A `num_samples` 一致。

---

## 23.6 mask-aware validator tamper tests

### all-low tamper

把：

```text
lower/upper
```

篡改为 high-only 区间，validator 必须 fail。

### all-high tamper

篡改为 low-only，fail。

### mixed tamper

只保留 low，fail。

---

## 23.7 Site1 role tamper

例如：

```text
q classification 正确
k classification 改错
```

validator fail。

也测试：

```text
q lower interval 替换成 k interval
```

必须 fail。

---

## 23.8 Stage B summary count tamper

篡改：

```text
gif_group_class_counts
```

validator fail。

---

## 23.9 Stage B 不覆盖 Stage A config

记录：

```text
Stage A resolved_config hash
```

生成两组 Stage B 后不变。

---

## 23.10 Stage B profile config 隔离

断言：

```text
phase_T_2_mtn_T_4/config/resolved_config.yaml
phase_T_8_mtn_T_4/config/resolved_config.yaml
```

分别存在且内容对应各自 config。

---

## 23.11 build_site_states 不可能生成 Clip

API 不再接受：

```text
include_clip
```

且返回值始终：

```text
phase/gif/mtn
```

---

## 23.12 Phase regression deployment override

training provenance：

```text
training phase.T = 4
```

CLI override：

```text
--phase-T 8
```

断言：

```text
layout 仍指向 phase_T_4 ANN training run
deployment Phase T = 8
metadata:
source_ann_training_phase_T=4
deployment_phase_T=8
```

---

# 24. 更新文档中的执行语义

项目执行文档必须明确：

## Shared preprocessing / per-num-samples Prefix

```bash
python scripts/prepare_data.py --config "$CFG"
python scripts/prepare_rotation.py --config "$ROT_CFG"
python scripts/discover_prefix.py --config "$CFG" --stage ann_training
```

其中：

```text
Rotation regression
    使用固定 canonical 128 samples

Prefix discovery
    使用当前 cfg.calibration.num_samples 对应的 Stage A calibration selection
```

因此只改变 `calibration.num_samples` 时：

```text
prepare_rotation 不需要重跑
discover_prefix 必须针对新的 num_samples 重跑
```

---

## Stage A sweep

例如：

```text
num_samples=64
num_samples=128
num_samples=256
```

只重新：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase A
```

---

## Stage B sweep

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase B
```

---

# 25. 不允许的实现方式

以下实现不要采用。

## 25.1 不要让 Rotation 跟随 num_samples

禁止：

```python
cfg["calibration"]["num_samples"]
```

决定 rotation regression selection。

---

## 25.2 不要继续共享单一 Prefix artifact 路径

本轮明确要求：

```text
Prefix discovery 跟随 calibration.num_samples
```

因此必须使用：

```text
prefix/num_samples_64
prefix/num_samples_128
prefix/num_samples_256
...
```

禁止继续让所有 sweep 共用：

```text
pre_finetuning_prefix/prefix_state.json
post_finetuning/prefix/prefix_state.json
```

否则不同 `num_samples` 会互相覆盖。

---

## 25.3 不要让 validator 只比较 `lower < upper`

mask-aware Clip 必须按 Stage A mask 复算。

---

## 25.4 不要让 Stage B 写 Stage A config/log 目录

A/B config/log 必须隔离。

---

## 25.5 不要保留 include_clip backward compatibility

这是 breaking architecture change，直接删除旧 API。

---

# 26. 最终验收标准

全部满足才算完成。

1. `pytest -q` 全部通过。
2. `calibration.num_samples=64/128/256` 均能通过 config validation。
3. Rotation regression 始终使用 canonical 128 samples。
4. Prefix discovery 始终使用当前 `cfg.calibration.num_samples` 对应的 Stage A calibration selection。
5. 改变 Stage A num_samples 不需要重新生成 Rotation。
6. 改变 Stage A num_samples 必须重新运行 Prefix discovery，并生成独立的 `num_samples_N` Prefix artifact。
7. verify_artifacts 不再拿 current Stage A manifest 去验证 Rotation regression。
8. Prefix provenance 指向当前 `data/calibration/num_samples_N/calibration_manifest.json`，Prefix root 按 `num_samples_N` 隔离。
9. Stage A calibration manifest 仍按 num_samples 分离。
10. Stage B mask-aware Clip validation 会根据 Stage A mask 重新计算。
11. tamper all-low/all-high/mixed Clip 都会被 validator 检出。
12. Site1 q/k/v 独立 validation。
13. Site7 gate/up 独立 validation。
14. Stage B 不覆盖 Stage A `resolved_config.yaml`。
15. 每个 Stage B profile 有自己的 `config/` 和 `logs/`。
16. `build_site_states()` 永远只返回 phase/gif/mtn。
17. 不存在 `include_clip=True` production/test API。
18. Phase regression 可以使用：
    ```text
    training T != deployment T
    ```
19. regression metadata 同时记录 source training T 和 deployment T。
20. 正式 ANN training / SNN conversion / evaluation 原有 A/B 语义保持不变。

---

# 27. 推荐实施顺序

按下面顺序改：

1. 新增 canonical preprocessing data path/helper。
2. 修改 `prepare_data.py` 同时生成 canonical 128 Rotation manifest 和当前 `num_samples` Stage A manifest。
3. 修改 Rotation 使用 canonical 128。
4. 修改 Prefix artifact path 按 `num_samples_N` 隔离；Prefix discovery 继续使用当前 Stage A calibration selection。
5. 修改 calibration/training/conversion/verify_artifacts 的 Prefix provenance 与路径校验。
6. 将 Stage B config/log 移入 profile root。
7. 抽取/完善 Clip policy validation。
8. 删除 `include_clip`。
9. 修改 Phase regression deployment override。
10. 更新 tests。
11. 更新执行文档。
12. `pytest -q`。
13. 做最小 smoke test：
    ```text
    num_samples=64 Prefix + Stage A
    num_samples=128 Prefix + Stage A
    同一 Rotation hash
    不同 Prefix root/hash
    两个 Stage B profile
    common_clip_enabled=true ANN forward
    Phase deployment T != training T
    ```

