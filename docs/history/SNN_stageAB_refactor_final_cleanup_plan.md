# SNN Stage A/B 重构收尾修复方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`
>
> 当前基线：`main` 最新一轮 Stage A / Stage B calibration 重构已经完成主体实现，但仍存在若干必须收尾的问题。
>
> 本文档供部署在服务器上的 Codex 直接执行。Codex 不应依赖本次聊天上下文；仅依据本文档与仓库当前 `main` 分支完成修复。
>
> **明确排除项：**
>
> - `ann_training/train_and_evaluate_qwen3_1_7b_phase_aware_GROUP_SIZE.sh`
> - 其它 GROUP_SIZE sweep 脚本
>
> 本次不修改、不检查、不要求修复任何 GROUP_SIZE sweep 脚本问题。

---

# 1. 本次收尾必须解决的问题

本次只处理以下 5 类问题：

1. `snn2/evaluation.py::evaluation_forward_metadata()` 中存在未定义变量，当前 final ANN/SNN evaluation 会触发 `NameError`。
2. SNN conversion 虽然已经加入 neuron-scoped fingerprint，但 `validate_conversion_metadata()` 仍把整个 Stage-B `calibration_root` 当成强制一致性条件，导致无关 Stage-B 参数变化错误地使已有单-neuron conversion 失效。
3. tests 仍大量按旧 one-shot calibration API / mixed statistics+state root 编写，当前测试套件与生产 API 不一致，必须完整迁移到 Stage A/B 新 schema。
4. `实验执行总结.md` 的正常执行流程缺少 Stage B 命令，且部分版本号仍停留在旧 state v7 / manifest v8。
5. `gif.salient_ratio` 当前 validation 允许缺省，但 ArtifactLayout 直接索引该字段；必须统一 schema，改成必填。

完成后必须保证：

```bash
python -m compileall snn2 scripts tests
pytest -q
```

全部通过。

---

# 2. 修复 `evaluation_forward_metadata()` 的 NameError

## 2.1 当前问题

当前：

```python
def evaluation_forward_metadata(...):
    ...
    return {
        ...
        "state_variant": None if inactive else Path(
            str(
                layout.ann_training_site_dir
                if aware_ann
                else layout.conversion_site_dir
            )
        ).name,
        ...
    }
```

但是该函数中没有定义：

```python
inactive
aware_ann
```

这两个变量只存在于另一个函数：

```python
evaluation_calibration_metadata()
```

因此当前代码在执行 evaluation metadata 写出时会直接报：

```text
NameError: name 'inactive' is not defined
```

## 2.2 正确修复方式

不要从另一个函数隐式“借变量”。

在 `evaluation_forward_metadata()` 内部明确计算：

```python
diagnostic_identity = base or rotated_pre_finetuning
aware_ann = (
    neuron == "ann"
    and is_aware_ann_mode(cfg)
    and not diagnostic_identity
)
identity_ann = (
    neuron == "ann"
    and not aware_ann
)
inactive_calibration = diagnostic_identity or identity_ann
```

然后：

```python
if inactive_calibration:
    state_variant = None
elif aware_ann:
    state_variant = Path(layout.ann_training_site_dir).name
else:
    state_variant = Path(layout.conversion_site_dir).name
```

最终 metadata：

```python
"state_variant": state_variant
```

不要再次写三层嵌套条件表达式。

## 2.3 语义要求

### Base / rotated-pre-finetuning

```text
state_variant = None
replacement_state_root = None
```

### vanilla / unaware final ANN

```text
state_variant = None
replacement_state_root = None
```

### phase_aware / gif_aware final ANN

```text
state_variant = 当前 ann_training Stage-B variant 目录名
replacement_state_root = layout.ann_training_site_dir
```

例如：

```text
phase_T_4_mtn_T_4_mtn_K_6_gif_low_ratio_0.9_gif_salient_ratio_0.1
```

### phase/gif/mtn SNN

```text
state_variant = 当前 conversion Stage-B variant 目录名
replacement_state_root = layout.conversion_site_dir
```

## 2.4 必须补测试

在 `tests/test_evaluation_paths.py` 或独立测试中覆盖：

- base ANN
- rotated-pre-finetuning ANN
- vanilla ANN
- unaware ANN
- phase_aware ANN
- gif_aware ANN
- phase SNN
- gif SNN
- mtn SNN

至少断言：

```python
metadata["state_variant"]
metadata["replacement_state_root"]
```

符合上述规则。

---

# 3. 完成 SNN neuron-scoped provenance 解耦

这是本次最重要的收尾项。

## 3.1 当前已经正确实现的部分

当前已有：

```python
deployment_state_fingerprint(
    state_root,
    neuron,
)
```

其规则：

```text
phase -> 所有 phase_state.pt
         + _global/final_rmsnorm/phase_state.pt

gif   -> 所有 gif_state.pt

mtn   -> 所有 mtn_state.pt
```

这一设计保持不变。

## 3.2 当前剩余错误

当前 `validate_conversion_metadata()` 的 expected 中仍包含：

```python
"calibration_root": str(layout.conversion_site_dir.resolve())
```

这会造成错误耦合。

例：

初始 Stage-B：

```text
states/
phase_T_4_mtn_T_4_mtn_K_6_gif_low_ratio_0.9_gif_salient_ratio_0.1/
```

已经创建：

```text
snn/phase/T_4/conversion/...
```

之后只修改：

```yaml
mtn:
  K: 8
```

新 Stage-B root：

```text
states/
phase_T_4_mtn_T_4_mtn_K_8_gif_low_ratio_0.9_gif_salient_ratio_0.1/
```

对于 Phase：

```text
phase_state.pt 不变
global final RMSNorm phase_state.pt 不变
Phase deployment fingerprint 不变
SNN path 仍是 snn/phase/T_4/
```

因此旧 Phase conversion 语义仍然有效。

但如果校验：

```text
calibration_root 必须等于当前 Stage-B root
```

就会错误判 stale。

## 3.3 修改原则

`calibration_root` 仍然可以写入：

```text
conversion_metadata.json
```

作为 informational provenance。

但是：

> `validate_conversion_metadata()` 不得再要求旧 metadata 的 `calibration_root` 与当前 Stage-B root 完全相同。

同样：

```text
calibration_state_manifest_sha256
```

也可以继续写入 metadata 供追溯，但不得作为单-neuron conversion semantic identity 的强制一致性条件。

## 3.4 单-neuron conversion 真正的强制依赖

### 公共依赖

所有 neuron 必须继续强校验：

```text
source_ann_checkpoint
source_ann_config_sha256
rotation_enabled
rotation_state_sha256
prefix_enabled
prefix_token_ids
prefix_state_sha256
prefix_kv_sha256
calibration_group_size
calibration_num_samples
calibration_grouping_policy
statistics_format_version
source_statistics_manifest_sha256
temporal implementation/schema
```

### Phase

强制：

```text
deployment_neuron == "phase"
deployment_state_fingerprint_sha256 == 当前 phase fingerprint
deployment_parameters.phase_T == 当前 cfg phase.T
```

### MTN

强制：

```text
deployment_neuron == "mtn"
deployment_state_fingerprint_sha256 == 当前 mtn fingerprint
deployment_parameters.mtn_T == 当前 cfg mtn.T
deployment_parameters.mtn_K == 当前 cfg mtn.K
```

### GIF

强制：

```text
deployment_neuron == "gif"
deployment_state_fingerprint_sha256 == 当前 gif fingerprint
deployment_parameters.gif_low_ratio == 当前 cfg gif.low_ratio
deployment_parameters.gif_salient_ratio == 当前 cfg gif.salient_ratio
```

---

# 4. `source_statistics_manifest_sha256` 必须直接进入 conversion metadata

为了移除 `calibration_root` 的强绑定后仍保持完整追溯，`create_conversion()` 应显式记录：

```python
"source_statistics_manifest_sha256": manifest[
    "source_statistics_manifest_sha256"
]
```

最好同时记录 informational：

```python
"source_statistics_manifest_path": manifest[
    "source_statistics_manifest_path"
]
```

`validate_conversion_metadata()` 强制比较：

```python
metadata["source_statistics_manifest_sha256"]
==
current_manifest["source_statistics_manifest_sha256"]
```

不要比较绝对 statistics root 路径作为 semantic identity。

---

# 5. aware ANN provenance 不要回退成 Stage-B root 强绑定

当前 ANN training 已经采用 dependency-scoped fingerprint。

保持：

```text
phase_aware + clip=false -> phase
phase_aware + clip=true  -> phase + clip
gif_aware + clip=false   -> gif
gif_aware + clip=true    -> gif + clip
```

但是检查：

```python
validate_recorded_training_artifact_provenance()
_validate_aware_training_provenance()
```

确保它们没有再次把：

```text
ann_training_state_root_at_training_time
ann_training_calibration_root
完整 calibration manifest hash
```

作为强制等价条件。

这些路径字段可以保留用于日志/追溯，但 ANN semantic identity 必须由：

```text
ann_training_state_dependency_kinds
ann_training_state_fingerprint_sha256
ann_training_statistics_manifest_sha256
Prefix hashes
G/N
```

决定。

---

# 6. 必须新增 provenance 回归测试

至少新增以下测试。

## 6.1 `mtn.K` 变化不应使 Phase conversion 失效

流程：

1. 同一 Stage-A statistics；
2. materialize：
   ```text
   mtn.K=6
   ```
3. 创建 Phase conversion；
4. materialize：
   ```text
   mtn.K=8
   ```
5. 当前 Phase state fingerprint 应与旧值一致；
6. `validate_conversion_metadata(..., neuron="phase")` 仍应通过。

## 6.2 `mtn.K` 变化必须使 MTN conversion 不匹配

同样流程：

```text
mtn.K=6 -> 8
```

断言：

```text
MTN deployment fingerprint 改变
旧 MTN conversion metadata validation 失败
```

## 6.3 GIF ratio 变化不应使 Phase/MTN conversion 因 root 变化误失效

如果修改：

```text
gif.low_ratio
gif.salient_ratio
```

但 Phase/MTN states 实际不变：

```text
Phase conversion 仍有效
MTN conversion 仍有效
GIF conversion 必须失效
```

## 6.4 phase.T 变化

断言：

```text
Phase conversion 失效
MTN conversion 不因 phase.T 目录 root 变化而失效
GIF conversion 不因 phase.T 目录 root 变化而失效
```

前提是相应 MTN/GIF state hash 确实不变。

---

# 7. 修复 tests 对旧 materialize API 的使用

当前生产 API：

```python
materialize_calibration_states(
    statistics_root,
    state_root,
    cfg,
    metadata,
    include_clip=...,
    expected_num_hidden_layers=...,
)
```

所有测试必须统一迁移。

---

# 8. 重构 calibration test fixture

建议在 tests 中新增共享 helper，例如：

```python
def write_stage_a_statistics(root, cfg, *, layers=1):
    ...
```

它必须生成：

```text
statistics/
├── layer_000/site_xx/statistics.pt
├── layer_000/site_xx/statistics_summary.json
├── _global/final_rmsnorm/statistics.pt
├── _global/final_rmsnorm/statistics_summary.json
└── statistics_manifest.json
```

manifest 必须满足当前 Stage-B validator：

```text
statistics_manifest_format_version
statistics_format_version
calibration_group_size
calibration_num_samples
expected_num_hidden_layers
expected_layer_names
purpose
source_model_stage
source_ann_mode
source_ann_checkpoint
source_ann_config_sha256
prefix_enabled
prefix_state_sha256
prefix_kv_sha256
rotation_enabled
rotation_state_sha256
calibration_data_manifest_sha256
calibration_grouping_policy
per-site statistics_sha256
global statistics_sha256
```

不要通过 monkeypatch 大量绕开 Stage-A manifest 校验；至少核心 calibration tests 应使用真实 schema。

---

# 9. 修复 `tests/test_calibration_profiles.py`

当前此文件仍按旧接口：

```python
materialize_calibration_states(
    tmp_path,
    _cfg(),
    metadata,
    ...
)
```

必须改成：

```python
statistics_root = tmp_path / "statistics"
state_root = tmp_path / "states"

_write_stage_a_statistics(statistics_root, cfg)

manifest = materialize_calibration_states(
    statistics_root,
    state_root,
    cfg,
    metadata,
    include_clip=...,
    expected_num_hidden_layers=1,
)
```

断言必须对应新 schema：

### Stage A

```text
statistics_root/.../statistics.pt 存在
state 文件不存在
```

### Stage B

```text
state_root/.../phase_state.pt 存在
state_root/.../gif_state.pt 存在
state_root/.../mtn_state.pt 存在
statistics.pt 不存在
```

### ANN training

普通 eligible sites：

```text
clip_state.pt 存在
```

Site 5：

```text
clip_state.pt 不存在
```

### post-finetuning

所有 site：

```text
clip_state.pt 不存在
```

---

# 10. 修复 `tests/test_conversion_metadata.py`

当前该文件还使用旧 mixed root，必须整体迁移。

`_Layout` 至少区分：

```text
post_finetuning_statistics_dir
post_finetuning_state_dir
ann_training_statistics_dir
ann_training_state_dir
conversion_site_dir
```

测试准备流程必须变成：

```text
write Stage-A statistics
        ↓
materialize Stage-B states
        ↓
write ANN checkpoint config
        ↓
create/validate conversion metadata
```

不允许再：

```text
statistics.pt 和 state 共目录
```

同时 `_cfg()` 必须补齐当前 schema 所需：

```python
"calibration": {
    "group_size": -1,
    "num_samples": 128,
    "seed": 42,
    "expected_sites_per_layer": 10,
},
"gif": {
    ...
    "low_ratio": 0.5,
    "salient_ratio": 0.5,
},
"replacement": {
    "common_clip_enabled": False,
},
```

若测试调用真实 `ArtifactLayout`，按完整 cfg 构造。

---

# 11. 修复 `tests/test_evaluation_paths.py`

当前该文件仍存在：

```python
"calibration": {"group_size": -1}
```

但 production code 已读取：

```python
cfg["calibration"]["num_samples"]
```

所有相关 fixture 必须补：

```python
"calibration": {
    "group_size": -1,
    "num_samples": 128,
}
```

并更新 expected metadata，加入：

```text
calibration_num_samples
state_variant
```

---

# 12. 修复 data manifest fake layout

当前 `prepare_manifests()` 已使用：

```python
layout.calibration_data_manifest_path
```

所以 tests 中不能再只写：

```python
SimpleNamespace(data_dir=tmp_path)
```

建议新增小型 fake layout：

```python
class _DataLayout:
    def __init__(self, root, *, seed=42, num_samples=4):
        self.data_dir = root
        self.calibration_data_manifest_path = (
            root
            / "calibration"
            / f"calibration_seed_{seed}_num_samples_{num_samples}"
            / "calibration_manifest.json"
        )
```

或者直接用真实 `ArtifactLayout`。

所有：

```text
prepare_manifests
load_manifests
load_selected_raw
```

相关测试必须更新。

---

# 13. 增加 `num_samples` 隔离测试

必须覆盖：

```text
num_samples=128
num_samples=256
```

断言：

```text
calibration_data_manifest_path 不同
calibration scope root 不同
Stage-A statistics root 不同
Stage-B states root 不同
aware ANN run root 不同
vanilla/unaware ANN checkpoint root 不因 num_samples 改变
non-aware SNN calibration scope 不同
```

---

# 14. 增加 Stage A / Stage B 结构测试

必须至少有一个测试完整验证：

Stage A 后：

```text
statistics/
  layer_000/site_xx/statistics.pt
  layer_000/site_xx/statistics_summary.json
  _global/final_rmsnorm/statistics.pt
  statistics_manifest.json
```

不存在：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

Stage B 后：

```text
states/<variant>/
  layer_000/site_xx/phase_state.pt
  layer_000/site_xx/gif_state.pt
  layer_000/site_xx/mtn_state.pt
  [clip_state.pt]
  calibration_summary.json
  calibration_state_manifest.json
```

不存在：

```text
statistics.pt
statistics_summary.json
```

---

# 15. `gif.salient_ratio` 改成配置必填

## 15.1 当前不一致

当前 validation：

```python
salient = float(
    cfg["gif"].get(
        "salient_ratio",
        1.0 - float(cfg["gif"]["low_ratio"])
    )
)
```

意味着：

```yaml
gif:
  low_ratio: 0.9
```

没有 `salient_ratio` 也会通过。

但是 ArtifactLayout 中：

```python
cfg["gif"]["salient_ratio"]
```

会直接 `KeyError`。

## 15.2 正确方案

既然当前实验路径已经明确要求保存：

```text
low_ratio_<...>_salient_ratio_<...>
```

则 `gif.salient_ratio` 设为必填。

修改 `validate_config()`：

```python
if "salient_ratio" not in cfg["gif"]:
    raise ValueError("gif.salient_ratio is required")
```

然后：

```python
try:
    low_ratio = float(cfg["gif"]["low_ratio"])
    salient_ratio = float(cfg["gif"]["salient_ratio"])
except (TypeError, ValueError) as exc:
    raise ValueError(...) from exc
```

要求：

```text
0 < low_ratio <= 1
0 <= salient_ratio < 1
low_ratio + salient_ratio == 1
```

推荐：

```python
if not math.isfinite(low_ratio) or not math.isfinite(salient_ratio):
    raise ValueError(...)
```

不要再使用隐式：

```python
1.0 - low_ratio
```

fallback。

---

# 16. generated config tests

`tests/test_generated_configs.py` 增加：

```python
assert "salient_ratio" in cfg["gif"]
assert math.isclose(
    float(cfg["gif"]["low_ratio"])
    + float(cfg["gif"]["salient_ratio"]),
    1.0,
)
assert "max_spikes" not in cfg["phase"]
```

增加：

```python
cfg = deepcopy(...)
del cfg["gif"]["salient_ratio"]

with pytest.raises(ValueError, match="salient_ratio"):
    validate_config(cfg)
```

---

# 17. 修复 `实验执行总结.md` Step 5

当前 Step 5 只明确给了 Stage A loop，而 Stage B 只有一个 `$CFG_17_P` 示例。

必须改成完整执行顺序。

## ANN-training Stage A

```bash
for CFG in "$CFG_17_P" "$CFG_8_P" "$CFG_L_P"; do
  python scripts/collect_calibration_statistics.py \
    --config "$CFG" \
    --stage ann_training
done
```

## ANN-training Stage B

紧接着必须写：

```bash
for CFG in "$CFG_17_P" "$CFG_8_P" "$CFG_L_P"; do
  python scripts/materialize_calibration_states.py \
    --config "$CFG" \
    --stage ann_training
done
```

然后再明确：

> 若仅修改 `phase.T / mtn.T / mtn.K / gif.low_ratio / gif.salient_ratio`，不要重复 Stage A，只针对对应配置重新运行 Stage B。

---

# 18. 修复 `实验执行总结.md` Step 8

当前 Step 8 只有 post-finetuning Stage A。

必须补完整 Stage B：

## Stage A

```bash
for CFG in "$CFG_17_V" "$CFG_17_U" "$CFG_8_V" "$CFG_8_U" "$CFG_L_V" "$CFG_L_U"; do
  python scripts/collect_calibration_statistics.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

## Stage B

```bash
for CFG in "$CFG_17_V" "$CFG_17_U" "$CFG_8_V" "$CFG_8_U" "$CFG_L_V" "$CFG_L_U"; do
  python scripts/materialize_calibration_states.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

并明确：

```text
Stage A = 针对 final ANN 重新前向采 statistics
Stage B = 从已有 statistics 生成 Phase/GIF/MTN state
```

---

# 19. 修正文档版本号

当前代码已经是：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 6
SITE_STATE_FORMAT_VERSION = 8
STATISTICS_FORMAT_VERSION = 2
STATISTICS_MANIFEST_FORMAT_VERSION = 1
CALIBRATION_MANIFEST_FORMAT_VERSION = 9
CONVERSION_METADATA_FORMAT_VERSION = 10
```

所以所有文档中旧表述：

```text
site state v7
calibration manifest v8
conversion metadata v9
temporal v5
```

统一修正为：

```text
site state v8
calibration manifest v9
conversion metadata v10
temporal v6
```

重点检查：

```text
实验执行总结.md
代码结构总结.md
README.md
AGENTS.md
```

以及其它非 history 的当前协议文档。

`docs/history/` 中历史方案可以保留历史版本描述，不需要强行改写历史记录。

---

# 20. 修复 `代码结构总结.md` 中测试说明

当前部分描述仍有旧版本，例如：

```text
test_conversion_metadata.py — 验证 v9 ...
```

应改成：

```text
test_conversion_metadata.py — 验证 v10 neuron-scoped conversion provenance ...
```

同时：

```text
state_validation.py
conversion.py
evaluation.py
```

描述中明确：

```text
Stage A statistics / Stage B states 分离
ANN dependency-scoped fingerprint
SNN neuron-scoped fingerprint
```

---

# 21. `verify_artifacts.py` 增强

当前该脚本仍需要检查是否过度依赖 Stage-B root。

必须新增/确认以下逻辑。

## 21.1 State root

可以检查：

```text
当前 cfg 对应 Stage-B root 是否存在
当前 conversion metadata 是否能通过 validate_conversion_metadata()
```

但是不要额外自行要求：

```text
metadata.calibration_root == 当前 root
```

如果 `validate_conversion_metadata()` 已改成 fingerprint identity，`verify_artifacts.py` 不应再次引入 root 强绑定。

## 21.2 Stage-A statistics provenance

对当前 Stage-B manifest：

```python
source_manifest = read_json(
    layout.conversion_site_dir / "calibration_state_manifest.json"
)
```

继续验证：

```text
source_statistics_manifest_path 存在
source_statistics_manifest_sha256 正确
```

## 21.3 conversion

每个 neuron 只通过：

```python
validate_conversion_metadata(cfg, layout, neuron)
```

来判断 semantic consistency。

不要再额外比较整个 calibration state manifest hash。

---

# 22. `create_conversion()` metadata 建议最终 schema

保留 informational：

```json
{
  "calibration_root": "...",
  "calibration_state_manifest_sha256": "...",
  "source_statistics_manifest_path": "...",
  "source_statistics_manifest_sha256": "..."
}
```

真正 semantic fields：

```json
{
  "deployment_neuron": "phase",
  "deployment_state_kinds": ["phase"],
  "deployment_state_fingerprint_sha256": "...",
  "deployment_state_file_hashes": {...},

  "deployment_parameters": {
    "phase_T": 4
  },

  "source_ann_config_sha256": "...",
  "prefix_state_sha256": "...",
  "prefix_kv_sha256": "...",
  "rotation_state_sha256": "...",
  "calibration_group_size": -1,
  "calibration_num_samples": 128,
  "source_statistics_manifest_sha256": "..."
}
```

---

# 23. `validate_conversion_metadata()` 最终 expected 规则

建议拆成两组。

## 23.1 semantic expected

强校验：

```python
semantic_expected = {
    "deployment_neuron": neuron,
    "deployment_state_kinds": [neuron],
    "deployment_state_fingerprint_sha256": deployment_fingerprint["sha256"],
    "full_temporal_steps": ...,
    "source_ann_checkpoint": ...,
    "source_ann_config_sha256": ...,
    "calibration_source_stage": ...,
    "prefix_source_stage": ...,
    "reused_ann_training_artifacts": ...,
    "post_finetuning_recalibration": ...,
    "prefix_enabled": ...,
    "prefix_token_ids": ...,
    "prefix_state_sha256": ...,
    "prefix_kv_sha256": ...,
    "rotation_enabled": ...,
    "rotation_state_sha256": ...,
    "expected_num_hidden_layers": ...,
    "snn_clip_applied": False,
    "source_ann_common_clip_enabled": ...,
    "calibration_group_size": ...,
    "calibration_num_samples": ...,
    "calibration_grouping_policy": ...,
    "statistics_format_version": ...,
    "source_statistics_manifest_sha256": ...,
}
```

以及 neuron-specific deployment parameters。

## 23.2 informational only

不要强制当前值等于旧值：

```text
calibration_root
calibration_state_manifest_sha256
source_statistics_manifest_path
```

这些只用于追溯“当初从哪个 Stage-B bundle 创建 descriptor”。

---

# 24. 注意 `full_temporal_steps`

当前：

```text
phase -> phase.T
mtn -> mtn.T
gif -> GIF_LOCAL_STEPS
```

继续严格校验。

不要把：

```text
mtn.K
gif.low_ratio
```

错误当成 temporal steps。

---

# 25. 收尾后必须执行 grep

```bash
git grep -n "max_spikes"
git grep -n "spike_count"
```

生产代码/配置中应继续为零。

然后：

```bash
git grep -n "site state v7"
git grep -n "manifest v8"
git grep -n "conversion metadata v9"
git grep -n "temporal v5"
```

当前非 history 文档中不应残留旧版本说明。

再检查：

```bash
git grep -n "materialize_calibration_states("
```

所有测试必须使用新：

```python
statistics_root,
state_root,
cfg,
metadata,
```

接口。

---

# 26. 本次不修改 GROUP_SIZE sweep

再次强调：

不要修改：

```text
ann_training/train_and_evaluate_qwen3_1_7b_phase_aware_GROUP_SIZE.sh
```

也不要把该脚本加入本轮验收失败项。

本次 pytest / compileall 不依赖它。

---

# 27. 推荐修改顺序

1. `snn2/evaluation.py`
   - 修 NameError
   - clean state_variant logic

2. `snn2/conversion.py`
   - 去掉 calibration_root 强绑定
   - 增加 source_statistics_manifest_sha256
   - semantic/informational provenance 分离

3. `snn2/state_validation.py`
   - 如需要补 fingerprint helper / manifest helper

4. `scripts/verify_artifacts.py`
   - 不再自行恢复 Stage-B root 强绑定

5. `snn2/config.py`
   - `gif.salient_ratio` 必填

6. tests shared fixtures

7. `tests/test_calibration_profiles.py`

8. `tests/test_conversion_metadata.py`

9. `tests/test_evaluation_paths.py`

10. 其它受 Stage A/B API 影响的 tests

11. `实验执行总结.md`

12. `代码结构总结.md / README.md / AGENTS.md`

13. 全测试

---

# 28. 最终测试命令

先：

```bash
python scripts/materialize_configs.py
```

然后：

```bash
python -m compileall snn2 scripts tests
```

针对性：

```bash
pytest -q \
  tests/test_calibration_profiles.py \
  tests/test_conversion_metadata.py \
  tests/test_evaluation_paths.py \
  tests/test_generated_configs.py \
  tests/test_post_finetuning_protocol.py \
  tests/test_training.py \
  tests/test_neurons.py
```

然后：

```bash
pytest -q
```

---

# 29. 必须新增的关键回归测试清单

- [ ] `evaluation_forward_metadata()` 不再 NameError。
- [ ] Base ANN `state_variant=None`。
- [ ] vanilla/unaware final ANN `state_variant=None`。
- [ ] aware final ANN `state_variant=<ann Stage-B variant>`。
- [ ] SNN `state_variant=<conversion Stage-B variant>`。
- [ ] Stage A statistics root 不含 state。
- [ ] Stage B state root 不含 statistics。
- [ ] post-finetuning Stage B 无 Clip。
- [ ] ANN-training Stage B true/false Clip 开关生成 state 完全一致。
- [ ] mtn.K 改变不会使 Phase conversion 因 root 改变而失效。
- [ ] mtn.K 改变会使 MTN conversion 失效。
- [ ] GIF ratio 改变只使 GIF conversion 失效。
- [ ] phase.T 改变只使 Phase conversion 失效（除非其它 state hash 真实变化）。
- [ ] `gif.salient_ratio` 缺失会在 config validation 阶段明确报错。
- [ ] `num_samples=128/256` data/calibration roots 不覆盖。
- [ ] `max_spikes` 不再出现。
- [ ] 全测试通过。

---

# 30. 最终验收标准

本轮收尾完成后，必须满足：

1. final ANN 和 SNN evaluation 均可正常生成 metadata，不存在未定义变量。
2. Stage-B full bundle 路径可以随无关 neuron 参数变化，而单-neuron conversion 的 semantic validity 只由该 neuron state fingerprint 决定。
3. ANN training provenance 仍按实际使用的 Phase/GIF/Clip state fingerprint 绑定。
4. tests 全部使用 Stage A statistics root + Stage B state root 新 schema。
5. tests 不再假设 statistics 与 states 共目录。
6. `gif.salient_ratio` 成为显式必填配置。
7. `实验执行总结.md` 的 ANN-training 和 Post-finetuning calibration 均明确包含 Stage A + Stage B 两条命令。
8. 当前协议文档版本号与 temporal v6 / state v8 / calibration manifest v9 / conversion metadata v10 一致。
9. `pytest -q` 全部通过。
10. 本轮不修改 GROUP_SIZE sweep 脚本。

