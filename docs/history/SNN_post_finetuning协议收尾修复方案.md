# SNN 项目本轮收尾修复方案：Post-finetuning Calibration 协议加固、Provenance 完善与代码结构文档重构

## 0. 本文档用途

本文档用于指导部署在服务器上的 Codex 在**没有其他对话上下文**的情况下，对当前 SNN 项目完成上一轮
“分离 ANN-training calibration 与 Post-finetuning conversion calibration”修改后的**最后一轮收尾修复**。

目标仓库：

```text
/home/wangwenkang/SNN
```

远程仓库：

```text
https://github.com/wangwk699/SNN
```

当前主分支：

```text
main
```

本轮不是重新设计实验协议，也不是推翻已经完成的主流程。当前以下核心逻辑已经正确，不要改回旧行为：

```text
1. ANN-training Prefix 与 post-finetuning Prefix 已分离。
2. ANN-training calibration 与 post-finetuning conversion calibration 已分离。
3. Vanilla analysis calibration 独立存在，只用于分析。
4. 每个 final ANN checkpoint 都独立 rediscover post-finetuning Prefix。
5. 每个 final ANN checkpoint 都独立执行 post-finetuning calibration。
6. Vanilla final ANN checkpoint 在训练结束后也执行 post-finetuning Prefix discovery。
7. Final ANN evaluation 使用 post-finetuning Prefix。
8. SNN evaluation 使用与对应 ANN evaluation 相同的 post-finetuning Prefix。
9. SNN conversion 只允许使用 run-specific post-finetuning conversion calibration。
10. conversion metadata 中 post_finetuning_recalibration = true。
```

本轮仅修复目前仍未完全落实的内容：

```text
A. calibration manifest provenance 信息不完整；
B. Vanilla analysis calibration 的 resolved config / logs 归档层级错误；
C. conversion 对 post-finetuning Prefix 前置条件检查不足；
D. verify_artifacts 对三类 calibration 的语义验证不足；
E. test_post_finetuning_protocol.py 覆盖不足；
F. generated config test 未覆盖 post_finetuning 协议；
G. 代码结构总结.md 仍包含旧协议，并且项目结构没有写全；
H. Markdown 中所有当前实现描述必须统一，不能再出现“ANN 微调后禁止重新校准”等旧表述。
```

用户会在本轮代码修改完成后自行执行：

```bash
pytest -q
```

因此 Codex 本轮需要**补齐/修改测试文件**，但不要求替用户执行完整 pytest。

---

# 1. 开始修改前必须先检查当前仓库

进入仓库：

```bash
cd /home/wangwenkang/SNN
```

先检查：

```bash
git status
git branch --show-current
git log -5 --oneline
```

确认：

```text
branch = main
```

然后读取以下文件的当前内容：

```text
snn2/artifacts.py
snn2/calibration.py
snn2/config.py
snn2/conversion.py
snn2/modeling.py
snn2/training.py

scripts/_common.py
scripts/calibrate_sites.py
scripts/discover_prefix.py
scripts/convert_snn.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py

tests/test_post_finetuning_protocol.py
tests/test_generated_configs.py
tests/test_calibration_topology.py

configs/experiment_matrix.yaml

代码结构总结.md
实验执行总结.md
环境配置.md
```

同时执行全仓库搜索：

```bash
rg -n \
  'post_finetuning_recalibration|ANN 微调后禁止重新校准|禁止重新校准|微调前 frozen calibration|layout\.prefix_dir|layout\.site_dir|layout\.calibration_dir|vanilla_analysis|ann_training_calibration|post_finetuning_conversion_calibration|eligible_for_conversion|eligible_for_ann_training' \
  .
```

要求逐项判断搜索结果是否属于：

```text
当前代码
当前说明文档
历史迁移文档
```

`docs/history/` 中明确标记为历史方案的文档可以保留历史上下文，但不能让根目录下的当前说明文档继续包含旧协议。

---

# 2. 当前实验协议：本轮修改不得破坏

## 2.1 三类 Calibration

### Vanilla analysis calibration

对象：

```text
Original pretrained Base
```

条件：

```text
no rotation
no prefix
no activation replacement
```

目的：

```text
仅用于比较加入 Rotation + Prefix 前后相同 10 个 site 的激活统计变化。
```

必须满足：

```text
analysis_only = true
eligible_for_ann_training = false
eligible_for_conversion = false
post_finetuning_recalibration = false
```

推荐：

```text
只 materialize statistics，不生成 Phase/GIF/MTN/clip deployment states。
```

---

### ANN-training calibration

对象：

```text
Rotated/fused Base
+
ANN-training Prefix
```

目的：

```text
只服务于 phase_aware / gif_aware ANN 微调。
```

必须满足：

```text
purpose = ann_training_calibration
analysis_only = false
eligible_for_ann_training = true
eligible_for_conversion = false
post_finetuning_recalibration = false
```

使用关系：

```text
vanilla     -> 不使用
unaware     -> 不使用
phase_aware -> 使用 Phase + Clip
gif_aware   -> 使用 GIF + Clip
```

---

### Post-finetuning conversion calibration

对象：

```text
每个 final ANN checkpoint
+
该 final ANN checkpoint 独立 rediscover 的 post-finetuning Prefix
```

目的：

```text
为该 final ANN 的 Phase/GIF/MTN SNN conversion 和 SNN evaluation 提供 states。
```

必须满足：

```text
purpose = post_finetuning_conversion_calibration
analysis_only = false
eligible_for_ann_training = false
eligible_for_conversion = true
post_finetuning_recalibration = true
```

该 calibration 必须是：

```text
run-specific
ann_mode-specific
learning-rate-specific
seed-specific
final-checkpoint-specific
```

---

# 3. 修复一：完善 Calibration Manifest Provenance

## 3.1 当前问题

当前 `snn2/calibration.py` 中已经预留了类似：

```python
"source_ann_checkpoint": None,
"source_ann_config_sha256": None,
"calibration_data_manifest_sha256": None,
"prefix_enabled": ...,
"prefix_token_ids": None,
"prefix_state_sha256": None,
"prefix_kv_sha256": None,
"rotation_enabled": None,
"rotation_state_sha256": None,
```

但大量字段最终仍然写成 `null`。

这不符合实验可追溯性要求。

本轮必须真正填充这些字段。

---

## 3.2 Manifest 的统一字段

对于三类 calibration，`statistics_manifest.json` 和
`calibration_state_manifest.json` 都应尽可能包含同一套 provenance 字段：

```json
{
  "purpose": "...",
  "analysis_only": false,
  "eligible_for_ann_training": false,
  "eligible_for_conversion": false,
  "post_finetuning_recalibration": false,

  "source_model_stage": "...",
  "source_ann_mode": null,
  "source_ann_checkpoint": null,
  "source_ann_config_sha256": null,

  "calibration_data_manifest_path": "...",
  "calibration_data_manifest_sha256": "...",

  "prefix_protocol_enabled": false,
  "prefix_token_ids": [],
  "prefix_state_path": null,
  "prefix_state_sha256": null,
  "prefix_kv_path": null,
  "prefix_kv_sha256": null,

  "rotation_enabled": false,
  "rotation_state_path": null,
  "rotation_state_sha256": null,

  "learning_rate": null,
  "seed": 42
}
```

字段名称可以在保持向后兼容的前提下略作调整，但语义必须完整。

---

## 3.3 不要用 `prefix_key_values is not None` 表示 Prefix protocol 是否开启

当前代码如果使用：

```python
"prefix_enabled": prefix_key_values is not None
```

语义不够准确。

原因：

Qwen Prefix discovery 允许得到：

```text
prefix_token_ids = []
```

此时：

```text
post_finetuning Prefix protocol 是开启的
但是没有非空 Prefix token
因此 prefixed_key_values.pt 可以不存在
```

所以必须区分：

```text
prefix_protocol_enabled
prefix_token_ids
prefix_kv_present
```

建议定义：

```python
prefix_protocol_enabled: bool
prefix_token_ids: list[int]
prefix_kv_present: bool
```

如果保留旧的：

```python
prefix_enabled
```

则它必须表示：

```text
该阶段是否启用 Prefix protocol
```

而不能表示：

```text
KV cache 是否非空
```

---

## 3.4 三类 calibration 的 Prefix provenance

### Vanilla analysis

必须为：

```json
{
  "prefix_protocol_enabled": false,
  "prefix_token_ids": [],
  "prefix_state_path": null,
  "prefix_state_sha256": null,
  "prefix_kv_path": null,
  "prefix_kv_sha256": null
}
```

---

### ANN-training calibration

读取：

```text
layout.ann_training_prefix_dir / prefix_state.json
layout.ann_training_prefix_dir / prefixed_key_values.pt
```

规则：

```text
prefix_state.json 必须存在。

如果 prefix_state.json 中 prefix_token_ids 非空：
    prefixed_key_values.pt 必须存在，并记录 sha256。

如果 prefix_token_ids 为空：
    prefixed_key_values.pt 可以不存在，
    prefix_kv_path = null，
    prefix_kv_sha256 = null。
```

---

### Post-finetuning conversion calibration

读取：

```text
layout.post_finetuning_prefix_dir / prefix_state.json
layout.post_finetuning_prefix_dir / prefixed_key_values.pt
```

规则同上。

---

## 3.5 Rotation provenance

### Vanilla analysis

```json
{
  "rotation_enabled": false,
  "rotation_state_path": null,
  "rotation_state_sha256": null
}
```

### ANN-training calibration

```json
{
  "rotation_enabled": true,
  "rotation_state_path": "<.../_shared/.../rotated_prefix/rotation/rotation_state.pt>",
  "rotation_state_sha256": "<real sha256>"
}
```

### Post-finetuning conversion calibration

对于：

```text
unaware
phase_aware
gif_aware
```

记录：

```text
rotation_enabled = true
rotation_state_path
rotation_state_sha256
```

对于：

```text
vanilla
```

记录：

```text
rotation_enabled = false
rotation_state_path = null
rotation_state_sha256 = null
```

---

## 3.6 Calibration dataset provenance

三类 calibration 必须指向**同一套固定 calibration manifest**。

当前 task-level calibration manifest 位于：

```text
layout.data_dir / calibration_manifest.json
```

必须记录：

```text
calibration_data_manifest_path
calibration_data_manifest_sha256
```

使用现有：

```python
sha256_file(...)
```

不要重新实现 hash 函数。

这样后续可以验证：

```text
Vanilla analysis calibration
ANN-training calibration
Post-finetuning conversion calibration
```

是否确实使用了同一批 calibration samples。

---

## 3.7 Final ANN provenance

只有 post-finetuning conversion calibration 需要：

```text
source_ann_mode
source_ann_checkpoint
source_ann_config_sha256
learning_rate
```

其中：

```text
source_ann_checkpoint = layout.ann_checkpoint_dir.resolve()
source_ann_config_sha256 = sha256(layout.ann_checkpoint_dir / "config.json")
source_ann_mode = cfg["experiment"]["ann_mode"]
learning_rate = cfg["training"]["learning_rate"]
```

ANN-training calibration 和 Vanilla analysis calibration：

```text
source_ann_checkpoint = null
source_ann_config_sha256 = null
```

---

## 3.8 推荐实现方式

不要让 `snn2/calibration.py` 自己猜 ArtifactLayout。

推荐：

在 `scripts/calibrate_sites.py` 中，根据 stage 组装完整：

```python
extra_metadata
```

或者新增一个集中 helper，例如：

```python
def calibration_provenance(
    cfg,
    layout,
    *,
    stage: str,
) -> dict[str, Any]:
    ...
```

更推荐把 provenance 构造逻辑放在 `snn2/` 内的公共模块中，而不是脚本里堆大量路径逻辑。

无论采用哪种方式，最终必须让两个 manifest 都获得真实值。

---

# 4. 修复二：Vanilla Analysis 的 config / logs 必须归入 shared vanilla_original

## 4.1 当前错误

当前 `scripts/calibrate_sites.py --stage vanilla_analysis` 的：

```text
statistics
resolved config
logs
```

没有落在同一个 shared policy 层级。

它属于：

```text
model-level shared analysis preprocessing
```

不属于：

```text
Base evaluation
```

也不属于：

```text
vanilla/lr.../seed... run
```

---

## 4.2 正确目录

必须统一为：

```text
artifacts/<experiment>/<task>/<model>/_shared/seed42/
└── vanilla_original/
    ├── config/
    │   └── resolved_config.yaml
    ├── logs/
    │   ├── calibrate_sites_vanilla_analysis.log
    │   ├── calibrate_sites_vanilla_analysis.jsonl
    │   └── ...
    └── vanilla_analysis_calibration/
        └── sites/
```

---

## 4.3 修改 `scripts/calibrate_sites.py`

对于：

```bash
--stage vanilla_analysis
```

不要继续使用：

```python
config_scope="base"
```

应使用：

```python
config_scope="policy_shared"
```

由于 Vanilla config 下：

```python
layout.policy_root
=
layout.shared_model_root / "vanilla_original"
```

所以 resolved config 会自然进入：

```text
_shared/.../vanilla_original/config/
```

StageRun 也必须使用：

```python
layout.policy_logs_dir
```

因此建议：

```python
if args.stage in {"ann_training", "vanilla_analysis"}:
    config_scope = "policy_shared"
    logs_dir = layout.policy_logs_dir
elif args.stage == "post_finetuning":
    config_scope = "run"
    logs_dir = layout.logs_dir
```

不要把 Vanilla analysis 放到 Base evaluation 的：

```text
base/seed42/
```

中。

---

# 5. 修复三：SNN Conversion 必须强制检查 Post-finetuning Prefix

## 5.1 当前问题

当前 `create_conversion()` 已经会强制 post-finetuning calibration：

```text
purpose == post_finetuning_conversion_calibration
eligible_for_conversion == true
post_finetuning_recalibration == true
```

这一点保持不变。

但是 post-finetuning Prefix 目前检查不够严格。

---

## 5.2 Conversion 前必须存在 `prefix_state.json`

当前主协议要求：

```text
post_finetuning.prefix_enabled = true
```

所以所有四种 final ANN mode 在 conversion 前都应该已执行：

```bash
python scripts/discover_prefix.py \
  --config "$CFG" \
  --stage post_finetuning
```

因此：

```text
layout.post_finetuning_prefix_dir / prefix_state.json
```

必须存在。

如果不存在：

```python
raise FileNotFoundError(...)
```

错误信息中明确提示重新运行：

```bash
python scripts/discover_prefix.py --config <CFG> --stage post_finetuning
```

---

## 5.3 KV cache 条件检查

读取：

```text
prefix_state.json
```

得到：

```python
prefix_token_ids
```

如果：

```python
prefix_token_ids
```

非空：

```text
prefixed_key_values.pt 必须存在
```

否则 conversion 直接报错。

如果：

```text
prefix_token_ids == []
```

则：

```text
prefixed_key_values.pt 可以不存在
```

这和当前 verify 中对 Qwen empty Prefix 的处理保持一致。

---

## 5.4 Conversion metadata

必须至少记录：

```json
{
  "prefix_root": ".../post_finetuning/prefix",
  "prefix_state_sha256": "...",
  "prefix_token_ids": [],
  "prefix_kv_sha256": null,
  "post_finetuning_recalibration": true
}
```

对于非空 Prefix：

```json
"prefix_kv_sha256": "<real sha256>"
```

建议 conversion metadata 与 calibration manifest 对这些 provenance 字段保持一致。

---

# 6. 修复四：增强 `verify_artifacts.py`

当前 verify 已经检查：

```text
post_finetuning prefix
post_finetuning calibration
conversion metadata
evaluation results
```

但还需要增加对**三类 calibration 语义**的明确验证。

---

## 6.1 ANN-training calibration validation

仅 rotated mode：

```text
unaware
phase_aware
gif_aware
```

需要存在：

```text
layout.ann_training_site_dir / calibration_state_manifest.json
```

并验证：

```python
purpose == "ann_training_calibration"
analysis_only is False
eligible_for_ann_training is True
eligible_for_conversion is False
post_finetuning_recalibration is False
```

同时：

```text
rotation_enabled == true
calibration_data_manifest_sha256 不为空
```

Prefix protocol：

```text
prefix_protocol_enabled == true
```

如果 Prefix token 非空，还要检查：

```text
ann_training_prefix/prefixed_key_values.pt
```

---

## 6.2 Vanilla analysis validation

所有 model-task pair 最终都应有：

```text
layout.vanilla_analysis_site_dir / statistics_manifest.json
```

验证：

```python
purpose == "vanilla_analysis_calibration"
analysis_only is True
eligible_for_ann_training is False
eligible_for_conversion is False
post_finetuning_recalibration is False
rotation_enabled is False
prefix_protocol_enabled is False
```

如果 Vanilla analysis 当前按设计不 materialize neuron states：

```text
不要调用 validate_calibration() 去要求 phase_state.pt 等文件。
```

只验证：

```text
10-site statistics topology
statistics_manifest
provenance
```

---

## 6.3 Post-finetuning conversion calibration validation

已有检查继续保留，并补充：

```python
purpose == "post_finetuning_conversion_calibration"
analysis_only is False
eligible_for_ann_training is False
eligible_for_conversion is True
post_finetuning_recalibration is True
source_ann_checkpoint is not None
source_ann_config_sha256 is not None
calibration_data_manifest_sha256 is not None
prefix_protocol_enabled is True
```

Rotated mode：

```text
rotation_enabled == true
rotation_state_sha256 != null
```

Vanilla：

```text
rotation_enabled == false
rotation_state_sha256 == null
```

---

## 6.4 Hash 一致性

如果 manifest 保存了：

```text
calibration_data_manifest_sha256
prefix_state_sha256
prefix_kv_sha256
rotation_state_sha256
source_ann_config_sha256
```

verify 应重新计算实际文件 hash 并比较。

至少对 post-finetuning conversion calibration 做严格 hash validation。

如果 hash 不一致：

```python
raise ValueError(...)
```

不能只检查文件存在。

---

# 7. 修复五：补齐测试覆盖

用户会自行执行：

```bash
pytest -q
```

Codex 需要补测试，但本轮无需替用户执行完整 pytest。

---

## 7.1 扩充 `tests/test_post_finetuning_protocol.py`

当前只有基础路径和 Prefix enable 测试，明显不足。

至少新增以下测试。

### Test A：stage-specific paths

已有测试保留：

```text
ann_training_prefix
ann_training_calibration/sites
vanilla_analysis_calibration/sites
post_finetuning/prefix
post_finetuning/conversion_calibration/sites
```

---

### Test B：Vanilla Prefix policy

验证：

```python
training_prefix_enabled(vanilla) is False
post_finetuning_prefix_enabled(vanilla) is True
```

已有则保留。

---

### Test C：Rotated Prefix policy

验证：

```python
training_prefix_enabled(unaware/phase_aware/gif_aware) is True
post_finetuning_prefix_enabled(...) is True
```

---

### Test D：Vanilla analysis manifest flags

构造最小临时 statistics/calibration artifact 或直接测试 metadata helper，验证：

```text
purpose = vanilla_analysis_calibration
analysis_only = true
eligible_for_ann_training = false
eligible_for_conversion = false
post_finetuning_recalibration = false
rotation_enabled = false
prefix_protocol_enabled = false
```

---

### Test E：ANN-training manifest flags

验证：

```text
purpose = ann_training_calibration
eligible_for_ann_training = true
eligible_for_conversion = false
post_finetuning_recalibration = false
```

---

### Test F：Post-finetuning manifest flags

验证：

```text
purpose = post_finetuning_conversion_calibration
eligible_for_ann_training = false
eligible_for_conversion = true
post_finetuning_recalibration = true
```

---

### Test G：Conversion 拒绝 ANN-training calibration

构造临时：

```text
calibration_state_manifest.json
purpose = ann_training_calibration
```

调用 conversion validation/helper，应抛：

```text
ValueError
```

---

### Test H：Conversion 拒绝 Vanilla analysis calibration

同样验证：

```text
purpose = vanilla_analysis_calibration
```

不能用于 conversion。

---

### Test I：Conversion 缺失 post-FT prefix_state 时失败

主协议中：

```text
post_finetuning Prefix enabled
```

如果：

```text
prefix_state.json 缺失
```

conversion 应失败。

---

### Test J：Empty Prefix 合法

如果：

```json
{"prefix_token_ids": []}
```

则：

```text
prefixed_key_values.pt 不存在
```

也应该允许通过 Prefix artifact validation。

---

### Test K：Non-empty Prefix 缺 KV 时失败

如果：

```json
{"prefix_token_ids": [123]}
```

但没有：

```text
prefixed_key_values.pt
```

应失败。

---

### Test L：Vanilla analysis shared logs/config path

验证 Vanilla ArtifactLayout + shared policy：

```text
policy_root ends with vanilla_original
policy_config_dir under _shared
policy_logs_dir under _shared
```

如果新增 stage helper，也直接测试 stage mapping。

---

## 7.2 更新 `tests/test_generated_configs.py`

当前 generated config test 需要额外验证 12 个配置全部包含：

```yaml
post_finetuning:
  rediscover_prefix: true
  recalibrate_sites: true
  prefix_enabled: true
  post_finetuning_recalibration: true
```

测试：

```python
for path in configs:
    cfg = yaml.safe_load(...)
    assert cfg["post_finetuning"]["rediscover_prefix"] is True
    assert cfg["post_finetuning"]["recalibrate_sites"] is True
    assert cfg["post_finetuning"]["prefix_enabled"] is True
    assert cfg["post_finetuning"]["post_finetuning_recalibration"] is True
```

并保留现有：

```text
12 configs
SITE_COUNT = 10
Qwen3-1.7B unaware LR = 5e-6
evaluation batch size
```

---

# 8. 修复六：完整重构 `代码结构总结.md`

这是本轮文档修改的重点之一。

当前 `代码结构总结.md` 有两个问题：

```text
1. 仍然保留旧的 calibration 描述：
   “所有 state 都在 Base checkpoint 的 calibration 阶段生成；
    ANN 微调后禁止重新校准。”

2. 当前仓库结构没有写全。
```

本轮要求：

> **从文件开头开始按当前代码整体重构 `代码结构总结.md`，不要只在末尾追加“当前 calibration 分层”补丁。**

---

# 9. `代码结构总结.md` 必须首先写完整项目根目录结构

当前 SNN 项目根目录按工作树语义包含 **6 个目录 + 7 个根文件**。

必须完整写出：

```text
SNN/
├── configs/
├── docs/
├── fast-hadamard-transform/
├── scripts/
├── snn2/
├── tests/
├── .gitignore
├── demo.py
├── pytest.ini
├── requirements.txt
├── 代码结构总结.md
├── 实验执行总结.md
└── 环境配置.md
```

说明：

```text
fast-hadamard-transform/
```

是 Git submodule / gitlink，在 GitHub API 中可能显示为 `commit` 类型，但在本地项目结构文档中必须作为项目根目录下的子模块目录列出。

不允许继续只写：

```text
configs/
scripts/
snn2/
requirements.txt
几个 md
```

这种不完整版本。

---

# 10. `代码结构总结.md` 必须完整展开六个目录

## 10.1 `configs/`

至少写：

```text
configs/
├── deepspeed_zero3.json
├── experiment_matrix.yaml
└── generated/
    ├── exp1_qwen3_1_7b_tldr__vanilla.yaml
    ├── exp1_qwen3_1_7b_tldr__unaware.yaml
    ├── exp1_qwen3_1_7b_tldr__phase_aware.yaml
    ├── exp1_qwen3_1_7b_tldr__gif_aware.yaml
    ├── exp1_qwen3_8b_tldr__vanilla.yaml
    ├── exp1_qwen3_8b_tldr__unaware.yaml
    ├── exp1_qwen3_8b_tldr__phase_aware.yaml
    ├── exp1_qwen3_8b_tldr__gif_aware.yaml
    ├── exp2_llama3_8b_tulu3__vanilla.yaml
    ├── exp2_llama3_8b_tulu3__unaware.yaml
    ├── exp2_llama3_8b_tulu3__phase_aware.yaml
    └── exp2_llama3_8b_tulu3__gif_aware.yaml
```

并说明：

```text
experiment_matrix.yaml = 单一实验矩阵源
generated/ = materialize_configs.py 生成的 12 个 resolved experiment config
deepspeed_zero3.json = 全参 ANN fine-tuning 的 ZeRO-3 配置
```

---

## 10.2 `docs/`

当前至少包含：

```text
docs/
└── history/
    ├── SNN 项目本轮修改方案：分离 ANN-training calibration 与 Post-finetuning conversion calibration.md
    ├── SNN_10_site_activation_replacement_实施方案.md
    └── SNN_10site_后续问题完整修复方案.md
```

这些是：

```text
历史迁移 / 修改方案
```

不是当前主流程说明。

当前主流程以根目录：

```text
代码结构总结.md
实验执行总结.md
环境配置.md
```

为准。

---

## 10.3 `fast-hadamard-transform/`

说明：

```text
第三方 Fast Hadamard Transform 子模块
用于 CUDA tensor 的高效 Hadamard transform
项目正式 rotation 实验依赖该实现
```

不要尝试把整个第三方子模块的所有源码复制到项目结构文档中。

只需标明：

```text
Git submodule
来源 Dao-AILab/fast-hadamard-transform
用途
```

---

## 10.4 `scripts/`

必须完整列出当前脚本：

```text
scripts/
├── _common.py
├── calibrate_sites.py
├── convert_snn.py
├── discover_prefix.py
├── evaluate_lm_harness.py
├── evaluate_tldr.py
├── materialize_configs.py
├── prepare_data.py
├── prepare_rotation.py
├── train_ann.py
└── verify_artifacts.py
```

每个文件给一句准确作用说明。

必须反映新的 stage：

```text
discover_prefix.py:
  --stage ann_training
  --stage post_finetuning

calibrate_sites.py:
  --stage ann_training
  --stage vanilla_analysis
  --stage post_finetuning
```

---

## 10.5 `snn2/`

必须根据修改完成后的当前目录实际内容重新读取并完整列出。

不要依赖旧文档中的手写列表。

Codex 在服务器上执行：

```bash
find snn2 -maxdepth 1 -type f -printf '%f\n' | sort
```

并把所有当前文件完整写入文档。

至少应包含当前已经存在的：

```text
snn2/
├── __init__.py
├── artifacts.py
├── calibration.py
├── config.py
├── controller.py
├── conversion.py
├── data.py
├── evaluation.py
├── hadamard.py
├── logging_utils.py
├── model_integration.py
├── modeling.py
├── neurons.py
├── prefix.py
├── prefix_cache.py
├── rotation.py
├── sites.py
├── stats.py
└── training.py
```

如果服务器当前 `main` 还有其它文件，以实际 `find` 结果为准，必须补入。

每个文件给一句职责说明。

特别注意当前文档以前漏掉/容易漏掉：

```text
__init__.py
prefix_cache.py
sites.py
```

等文件。

---

## 10.6 `tests/`

必须按修改完成后的实际内容完整列出。

当前至少：

```text
tests/
├── test_calibration_topology.py
├── test_evaluation_paths.py
├── test_generated_configs.py
├── test_hadamard.py
├── test_neurons.py
├── test_post_finetuning_protocol.py
├── test_prefix.py
├── test_sites.py
└── test_statistics.py
```

如果本轮新增测试文件，必须同步加入。

每个测试文件写明其验证目的。

---

# 11. `代码结构总结.md` 的总体数据流必须改成新协议

旧流程：

```text
Base calibration
-> ANN fine-tuning
-> frozen calibration
-> SNN conversion
```

必须删除。

新的总体数据流应明确写成：

```text
固定 data manifest
        ↓
每个 model-task pair 的 shared pre-finetuning 阶段
        ├── prepare rotation / fused Base
        ├── discover ANN-training Prefix
        ├── ANN-training calibration
        └── Vanilla analysis calibration
        ↓
四种 ANN mode 全参 fine-tuning
        ↓
得到 12 个 ann/final checkpoint
        ↓
每个 final checkpoint 独立
        ├── rediscover post-finetuning Prefix
        └── post-finetuning conversion calibration
        ↓
        ├── Final ANN evaluation
        └── Phase/GIF/MTN conversion
                 ↓
             SNN evaluation
```

必须说明：

```text
Final ANN evaluation 和对应 SNN evaluation 使用同一个 post-finetuning Prefix。
```

---

# 12. `代码结构总结.md` 必须重新解释四种 ANN mode

写清：

```text
vanilla:
  training:
    no rotation
    no ANN-training Prefix
    no activation replacement
  post-finetuning:
    rediscover Prefix
    recalibrate
    ANN/SNN evaluation 使用 post-finetuning Prefix

unaware:
  training:
    rotation
    ANN-training Prefix
    no activation replacement
  不使用 ANN-training calibration
  post-finetuning:
    独立 Prefix + calibration

phase_aware:
  training:
    rotation
    ANN-training Prefix
    Phase + Clip from ANN-training calibration
  post-finetuning:
    独立 Prefix + calibration

gif_aware:
  training:
    rotation
    ANN-training Prefix
    GIF + Clip from ANN-training calibration
  post-finetuning:
    独立 Prefix + calibration
```

---

# 13. `代码结构总结.md` 必须完整说明 Prefix 两阶段

写清：

## ANN-training Prefix

```text
rotated/fused Base 上 discover
保存于 _shared
服务 unaware / phase_aware / gif_aware ANN training
Vanilla training 不使用
```

## Post-finetuning Prefix

```text
每个 ann/final checkpoint 独立 rediscover
保存于 run-specific post_finetuning/prefix
四种 mode 都使用，包括 Vanilla final
用于：
  Final ANN evaluation
  post-finetuning conversion calibration
  SNN evaluation
```

并明确：

```text
fixed Prefix 通过 fixed past_key_values / KV cache 注入；
不是简单把 prefix token 永久拼到训练 labels 前。
```

---

# 14. `代码结构总结.md` 必须完整说明三类 Calibration

删除：

```text
所有 state 都在 Base checkpoint calibration 阶段生成
ANN 微调后禁止重新校准
```

替换为：

```text
Vanilla analysis calibration:
  original Base
  analysis only
  statistics only

ANN-training calibration:
  rotated Base + ANN-training Prefix
  pre-finetuning
  仅 phase_aware / gif_aware training 使用

Post-finetuning conversion calibration:
  每个 final ANN 独立
  final checkpoint + post-ft Prefix
  生成新的 Phase/GIF/MTN/clip states
  conversion 使用
  post_finetuning_recalibration = true
```

---

# 15. `代码结构总结.md` 必须继续保留并正确说明 10-site topology

当前每层固定 10 site：

```text
1  post_input_rmsnorm
2  q_post_rope_r3
3  k_post_rope_r3
4  v_projection_r2
5  post_spiking_softmax
6  post_attention_value_dot_r2
7  post_mlp_rmsnorm
8  post_spiking_silu
9  post_mlp_up_proj
10 post_mlp_product_r4
```

坐标：

```text
1:R1
2:R3
3:R3
4:R2
5:I
6:R2
7:R1
8:I
9:I
10:R4
```

说明：

```text
Site 9 = up_proj 后，product 前
Site 9 不额外 rotation
Site 10 = gate ⊙ up 后经过 R4
```

不要重新引入旧 9-site 表述。

---

# 16. `代码结构总结.md` 必须完整说明 Artifact 目录层级

至少写：

```text
artifacts/<experiment>/<task>/<model>/
├── _shared/seed42/
│   ├── rotated_prefix/
│   │   ├── config/
│   │   ├── logs/
│   │   ├── rotation/
│   │   ├── ann_training_prefix/
│   │   └── ann_training_calibration/
│   └── vanilla_original/
│       ├── config/
│       ├── logs/
│       └── vanilla_analysis_calibration/
│
├── base/seed42/
│   ├── config/
│   ├── logs/
│   └── evaluation/
│
└── <ann_mode>/lr<learning_rate>/seed42/
    ├── config/
    ├── logs/
    ├── ann/
    │   └── final/
    ├── post_finetuning/
    │   ├── prefix/
    │   └── conversion_calibration/
    │       └── sites/
    └── snn/
        ├── phase/
        ├── gif/
        └── mtn/
```

注意 Vanilla analysis 本轮修复后：

```text
config/logs
```

也必须在：

```text
_shared/.../vanilla_original/
```

下面。

---

# 17. 更新 `实验执行总结.md` 的一致性检查

`实验执行总结.md` 当前已经整体重构为新协议，不要重新大改。

但本轮修改完成后必须检查以下内容是否与代码保持一致：

```text
--stage ann_training
--stage vanilla_analysis
--stage post_finetuning

post_finetuning Prefix
post_finetuning conversion calibration

Vanilla final 也 rediscover Prefix

SNN conversion 使用 post-ft calibration
post_finetuning_recalibration = true
```

如果 Vanilla analysis 的 logs/config 路径在文档中有说明，则同步改成：

```text
_shared/seed42/vanilla_original/config/
_shared/seed42/vanilla_original/logs/
```

不得出现：

```text
Base calibration frozen across fine-tuning
ANN 微调后禁止 recalibration
conversion 使用微调前 calibration
```

---

# 18. `环境配置.md`

当前本轮原则上无需修改实验环境。

但是 Codex 仍需快速检查：

```bash
rg -n \
  'calibration|prefix|post_finetuning|site_dir|ann_training' \
  环境配置.md
```

如果没有旧协议描述，不做无意义改动。

---

# 19. Legacy alias 的处理

当前 `ArtifactLayout` 可能仍保留：

```python
prefix_dir
calibration_dir
site_dir
```

作为 legacy alias。

本轮不强制删除。

但必须确认当前主流程：

```text
training
prefix discovery
calibration
evaluation
conversion
verification
```

均不再通过这些模糊 alias 选择 stage。

执行：

```bash
rg -n \
  'layout\.(prefix_dir|calibration_dir|site_dir)' \
  snn2 scripts tests
```

如果当前主流程仍有使用：

```text
必须改为明确的 stage-specific path。
```

如果只剩兼容 alias 定义本身，可以保留，并在注释中明确：

```text
legacy only
new code must not use
```

---

# 20. 不要改变的内容

本轮不要顺手修改：

```text
10-site 数学位置
R1/R2/R3/R4 定义
Phase neuron 数学
GIF quantization 数学
MTN 数学
TL;DR ROUGE 实现
lm-eval task specs
training learning rates
DeepSpeed 策略
dataset sample policy
128 calibration sample 数
max_seq_length
```

除非为了本轮 provenance / validation / documentation 必须做最小改动。

---

# 21. 推荐修改文件清单

本轮预计至少需要修改：

```text
snn2/calibration.py
snn2/conversion.py

scripts/calibrate_sites.py
scripts/verify_artifacts.py

tests/test_post_finetuning_protocol.py
tests/test_generated_configs.py

代码结构总结.md
```

根据实现方式可能还需修改：

```text
snn2/artifacts.py
snn2/modeling.py
snn2/config.py
实验执行总结.md
```

不要为了“看起来整洁”修改无关文件。

---

# 22. 修改完成后的静态检查

用户会自行运行完整 pytest。

Codex 修改完成后至少自行执行以下低成本检查：

```bash
python -m compileall -q snn2 scripts tests
```

以及搜索：

```bash
rg -n \
  'ANN 微调后禁止重新校准|转换只加载微调前|微调前 frozen calibration state' \
  --glob '!docs/history/**' \
  .
```

预期：

```text
当前实现文档中无旧协议描述。
```

再执行：

```bash
rg -n \
  'layout\.(prefix_dir|calibration_dir|site_dir)' \
  snn2 scripts tests
```

预期：

```text
除 ArtifactLayout 中明确 legacy alias 定义外，
当前主流程代码不依赖模糊 alias。
```

---

# 23. 最终验收标准

本轮 Codex 修改完成后必须满足：

## 23.1 Provenance

三类 calibration manifest 都真实记录：

```text
purpose
analysis_only
eligible_for_ann_training
eligible_for_conversion
post_finetuning_recalibration

calibration_data_manifest_sha256

prefix protocol / token ids / state hash / KV hash
rotation enabled / rotation hash

post-ft 时：
source_ann_checkpoint
source_ann_config_sha256
source_ann_mode
learning_rate
seed
```

不应再无条件写大量 `null`。

---

## 23.2 Vanilla analysis 归档

必须：

```text
_shared/.../vanilla_original/config/
_shared/.../vanilla_original/logs/
_shared/.../vanilla_original/vanilla_analysis_calibration/
```

不能进入：

```text
base/
```

也不能进入：

```text
vanilla/lr.../seed.../logs
```

---

## 23.3 Conversion validation

必须：

```text
只接受 post_finetuning_conversion_calibration
必须存在 post-ft prefix_state.json
non-empty prefix 必须存在 fixed KV
empty prefix 允许无 KV
```

---

## 23.4 Verify

必须验证：

```text
Vanilla analysis semantics
ANN-training calibration semantics
Post-ft calibration semantics
Prefix conditional KV requirement
重要 provenance hashes
```

---

## 23.5 Tests

至少补齐本方案第 7 节所列测试。

用户会在服务器自行执行：

```bash
pytest -q
```

---

## 23.6 `代码结构总结.md`

必须：

```text
从头整体重构
不再保留旧协议矛盾描述
完整写出项目根目录 6 个目录 + 7 个文件
完整列出 scripts/
完整列出 snn2/
完整列出 tests/
说明 configs/ 与 docs/
说明 fast-hadamard-transform 是 submodule
说明新的 Prefix/Calibration 两阶段/三阶段语义
说明 run-specific post-finetuning pipeline
说明完整 artifact tree
说明当前 10-site topology
```

根目录必须明确展示：

```text
6 个目录：
configs/
docs/
fast-hadamard-transform/
scripts/
snn2/
tests/

7 个文件：
.gitignore
demo.py
pytest.ini
requirements.txt
代码结构总结.md
实验执行总结.md
环境配置.md
```

---

# 24. 最终实验语义不得发生变化

完成本轮收尾修复后，项目最终语义仍然必须是：

```text
                    PRE-FINETUNING
                         │
        ┌────────────────┴─────────────────┐
        │                                  │
 Original Base                        Rotated Base
        │                                  │
 Vanilla analysis                 ANN-training Prefix
 calibration                            │
 [analysis only]               ANN-training calibration
                                           │
                                  phase/gif-aware training
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                    unaware / phase / gif       vanilla training
                              │                         │
                              └────────────┬────────────┘
                                           │
                                   Final ANN checkpoints
                                           │
                                   POST-FINETUNING
                                           │
                         each checkpoint independently
                                           │
                              rediscover Prefix
                                           │
                                  fixed Prefix KV
                                           │
                              recalibrate 10 sites
                                           │
                    post-finetuning conversion calibration
                                           │
                         ┌─────────────────┴────────────────┐
                         │                                  │
                   ANN evaluation                    SNN conversion
              uses post-ft Prefix                 Phase / GIF / MTN
                                                            │
                                                     SNN evaluation
                                                  uses same post-ft Prefix
```

其中：

```text
post_finetuning_recalibration = true
```

必须始终保持。

---

# 25. 一句话任务总结

本轮不是再改实验设计，而是把已经实现的新协议真正“封口”：

```text
补齐 provenance，
把 Vanilla analysis 完整归入 shared policy，
让 conversion/verify 拒绝任何错误阶段的工件，
补齐单元测试，
并从头重构 代码结构总结.md，
使代码、artifact、测试和文档对 ANN-training / post-finetuning calibration 的描述完全一致。
```
