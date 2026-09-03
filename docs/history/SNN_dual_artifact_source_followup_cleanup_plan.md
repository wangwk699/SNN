# SNN 双套 Artifact Source 二次收尾修改方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 当前检查基于 `main` 最新提交 `dca0b861e9c6d49688ed55c372a74766dc15c7bd`。  
> 本文面向部署在服务器上的 Codex；请在没有本次对话上下文的情况下，仅依据本文完成本轮修改。

---

# 1. 本轮范围

当前 dual-artifact source 主逻辑已经正确，本轮只处理上一轮检查后仍剩余的收尾问题。

需要修改：

1. 补齐 `selector=false` 的真实 conversion 端到端测试；
2. 补齐 `verify_artifacts.py` 对 true/false source matrix 的回归测试；
3. 修正 `validate_calibration()` 在 selector=false 场景下的误导性错误提示；
4. 同步 `代码结构总结.md` 与 `实验执行总结.md` 的少量残留文档问题。

明确**不修改**：

```bash
ann_training/train_and_evaluate_qwen3_8b_phase_aware.sh
```

中当前：

```bash
LEARNING_RATES=(
  5.0e-05
)
```

这是本轮确认保留的实验设置，不要恢复之前的多学习率 sweep。

---

# 2. 不要改动当前已经正确的核心逻辑

以下行为已经正确，本轮禁止重构或改变：

## 2.1 Selector 语义

```yaml
conversion:
  use_post_finetuning_artifacts: true|false
```

是 SNN conversion / SNN evaluation artifact source 的唯一 selector。

### true

```text
Post-finetuning Prefix
+
Post-finetuning Stage A
```

### false

仅 non-vanilla：

```text
shared Pre-finetuning Prefix
+
shared ANN-training Stage A
```

### vanilla

```text
use_post_finetuning_artifacts=false
```

继续非法。

---

## 2.2 Final ANN Evaluation

selector 不影响 Final ANN Evaluation。

保持：

```text
vanilla      -> no Prefix
unaware      -> Post-finetuning Prefix
phase_aware  -> Pre-finetuning Prefix
gif_aware    -> Pre-finetuning Prefix
```

实际是否加载 Prefix 继续由：

```yaml
evaluation:
  prefix_enabled: true|false
```

控制。

---

## 2.3 Provenance

保持：

```text
unaware + false
    -> 使用 shared Pre Prefix + shared ANN-training Stage A
    -> 不要求 aware ANN-training provenance

phase/gif aware + false
    -> 使用 shared Pre Prefix + shared ANN-training Stage A
    -> 必须校验训练期 frozen provenance

phase/gif aware + true
    -> 使用各自 Final ANN 的 Post Prefix + Post Stage A
    -> 不要求 Post Stage A hash == ANN-training Stage A hash
```

---

# 3. 补齐 `tests/test_conversion_metadata.py` 的 selector=false 真实端到端测试

当前该文件主要覆盖 `use_post_finetuning_artifacts=true`，并且部分 `create_conversion()` 测试通过 monkeypatch `_source_bundle()`，没有真正验证 selector=false 的实际 source resolution。

本轮必须补真实路径测试。

---

## 3.1 重构 test helper

当前 `_cfg()` 类似固定：

```python
"conversion": {
    "use_post_finetuning_artifacts": True,
}
```

改成支持：

```python
def _cfg(
    rotation_enabled=False,
    *,
    ann_mode="vanilla",
    use_post=True,
):
    ...
```

其中：

```python
cfg["experiment"]["ann_mode"] = ann_mode
cfg["conversion"]["use_post_finetuning_artifacts"] = use_post
```

注意：

```text
vanilla + use_post=False
```

不应作为正常 conversion fixture 使用。

---

# 4. 新增 `unaware + false` conversion 端到端测试

构造真实：

```text
Final unaware ANN checkpoint
shared Pre-finetuning Prefix
shared ANN-training Stage A
```

不要 monkeypatch `_source_bundle()`。

调用：

```python
create_conversion(cfg, layout, "phase")
validate_conversion_metadata(cfg, layout, "phase")
```

验证至少：

```python
metadata["use_post_finetuning_artifacts"] is False
metadata["prefix_source_stage"] == "pre_finetuning"
metadata["calibration_source_stage"] == "ann_training"
metadata["reused_ann_training_artifacts"] is True
metadata["post_finetuning_recalibration"] is False
```

并验证 source calibration manifest：

```text
source_model_stage = rotated_fused_base
source_ann_mode = None
source_ann_checkpoint = None
```

---

## 4.1 最关键的 regression

monkeypatch：

```python
snn2.conversion._validate_aware_training_provenance
```

令其一旦被调用就：

```python
raise AssertionError("unaware must not require aware training provenance")
```

然后执行真实：

```python
create_conversion(...)
validate_conversion_metadata(...)
```

测试必须通过。

这用于锁死：

```text
unaware + false
```

绝不能误进入 aware training provenance 分支。

---

# 5. 新增 `phase_aware + false` conversion provenance 测试

构造真实：

```text
shared Pre Prefix
shared ANN-training Stage A
Final phase-aware ANN
training_result.json
Stage B clip_profile_manifest.json
```

`training_result.json` 至少完整记录当前 `_validate_aware_training_provenance()` 需要的字段，包括：

```text
ann_training_prefix_root
ann_training_prefix_state_sha256
ann_training_prefix_kv_sha256
ann_training_prefix_num_samples
ann_training_prefix_discovery_manifest_sha256
ann_training_prefix_token_ids
ann_training_calibration_root
ann_training_calibration_manifest_sha256
ann_training_calibration_group_size
ann_training_calibration_grouping_policy
statistics_format_version
ann_training_phase_T
ann_training_mtn_T
ann_training_calibration_num_samples
ann_training_clip_profile_root
ann_training_clip_profile_manifest_sha256
```

然后：

```python
metadata = create_conversion(cfg, layout, "phase")
validate_conversion_metadata(cfg, layout, "phase")
```

应通过。

验证：

```python
metadata["reused_ann_training_artifacts"] is True
metadata["post_finetuning_recalibration"] is False
metadata["source_ann_training_phase_T"] == expected_phase_T
metadata["source_ann_training_mtn_T"] == expected_mtn_T
```

---

## 5.1 篡改 provenance 后必须 fail-fast

创建 conversion 后，修改：

```text
ANN-training calibration manifest
```

或至少使：

```text
training_result.json 中记录的 manifest hash
!= 当前文件 hash
```

然后：

```python
validate_conversion_metadata(...)
```

必须抛 `ValueError`。

错误应明确指向：

```text
Aware conversion artifacts differ from those fixed during ANN training
```

或等价 provenance mismatch。

---

# 6. 新增 `phase_aware + true` Post bundle regression

构造：

```text
Final phase-aware ANN
Post-finetuning Prefix
Post-finetuning Stage A
```

同时可以存在 ANN-training Stage A，但故意让两者内容/hash 不同。

调用：

```python
create_conversion(cfg, layout, "phase")
validate_conversion_metadata(cfg, layout, "phase")
```

必须通过。

验证：

```python
metadata["use_post_finetuning_artifacts"] is True
metadata["prefix_source_stage"] == "post_finetuning"
metadata["calibration_source_stage"] == "post_finetuning"
metadata["reused_ann_training_artifacts"] is False
metadata["post_finetuning_recalibration"] is True
```

source manifest 必须是：

```text
source_model_stage = final_ann_checkpoint
source_ann_mode = phase_aware
source_ann_checkpoint = 当前 Final ANN checkpoint
```

---

## 6.1 明确验证不调用 aware training provenance

monkeypatch：

```python
_validate_aware_training_provenance
```

令其调用即失败。

`phase_aware + true` 的真实 conversion 必须仍正常通过。

这用于防止以后又把 aware 模式固定绑定到 training bundle。

---

# 7. GIF-aware 至少补一个等价 source regression

不必重复 phase-aware 的所有测试，但至少补一个：

```text
gif_aware + false
```

或：

```text
gif_aware + true
```

真实 `_source_bundle -> create_conversion -> validate_conversion_metadata` 流程。

建议优先补：

```text
gif_aware + false
```

确保 aware false 的 provenance 规则不只在 phase-aware 生效。

---

# 8. 补齐 `tests/test_verify_artifacts.py` selector source matrix regression

当前该文件主要覆盖：

```text
Site 2/3/4/6 topology
GIF provenance
Final ANN forward metadata
SNN forward metadata
```

还需要增加 selector-aware source regression。

无需完整运行真实大模型，只需构造 verify 所需的 artifact tree + metadata。

---

# 9. 新增 `unaware + false` verify regression

构造：

```text
ann_mode = unaware
use_post_finetuning_artifacts = false
```

并准备：

```text
shared Pre Prefix
shared ANN-training Stage A
conversion_metadata.json
SNN evaluation metrics
```

验证 `verify_artifacts.py` 选择：

```text
layout.conversion_prefix_dir == layout.ann_training_prefix_dir
layout.conversion_site_dir == layout.ann_training_site_dir
```

并验证 metrics / conversion metadata：

```text
use_post_finetuning_artifacts = false
prefix_source_stage = pre_finetuning
calibration_source_stage = ann_training
```

最重要：

```text
不得要求 aware training_result / Stage B provenance
```

---

# 10. 新增 `phase_aware + false` verify regression

构造 valid aware training provenance。

verify 必须：

```text
选择 shared Pre bundle
要求 training_result.json
验证 selected Stage B profile provenance
```

然后篡改任一 frozen provenance hash。

`verify_artifacts.py` 必须 fail-fast。

---

# 11. 新增 `phase_aware + true` verify regression

构造：

```text
Post Prefix
Post Stage A
```

并使 Post Stage A 与 ANN-training Stage A hash 不同。

verify 必须正常通过。

确认：

```text
use_post_finetuning_artifacts = true
prefix_source_stage = post_finetuning
calibration_source_stage = post_finetuning
post_finetuning_recalibration = true
```

且不要求：

```text
Post Stage A == ANN-training Stage A
```

---

# 12. selector regression 测试应直接验证的 canonical matrix

测试中至少覆盖：

| mode | selector | Prefix source | Calibration source | aware training provenance |
|---|---:|---|---|---|
| unaware | false | Pre | ANN-training A | 不要求 |
| phase_aware | false | Pre | ANN-training A | 必须 |
| gif_aware | false | Pre | ANN-training A | 必须 |
| phase_aware | true | Post | Post A | 不要求 |

现有 config test 已覆盖 vanilla=false 非法，无需重复完整 conversion。

---

# 13. 修正 `snn2/conversion.py::validate_calibration()` 的错误提示

当前代码：

```python
if clip_policy == "forbid_all" and clip_states:
    raise ValueError(
        "Post-finetuning conversion calibration must be clip-free; "
        "re-run calibrate_sites.py --stage post_finetuning --calibration-phase A"
    )
```

问题：

`validate_calibration()` 现在同时用于：

```text
selector=true
    -> Post-finetuning Stage A

selector=false
    -> ANN-training Stage A
```

因此如果 selector=false 的 shared ANN-training Stage A 出现旧 `clip_state.pt`，错误却指导用户重跑 Post-finetuning calibration，这是错误的操作建议。

---

## 13.1 修改方案

直接改成 source-neutral 文案：

```python
if clip_policy == "forbid_all" and clip_states:
    raise ValueError(
        "Conversion Stage A calibration must be clip-free; "
        "rebuild the currently selected Stage A calibration artifact."
    )
```

推荐使用这一简单方案，不必为了错误文案给函数额外传 `source_stage`。

---

## 13.2 测试

现有 stale `clip_state.pt` rejection test 保留。

更新断言，使其至少匹配：

```text
Conversion Stage A calibration must be clip-free
```

不要再断言 Post-finetuning 专用文案。

---

# 14. 同步 `代码结构总结.md`

当前文件存在两个遗漏。

---

## 14.1 补 history 文件

在：

```text
docs/history/
```

中加入：

```text
SNN_dual_artifact_source_final_cleanup_plan.md
```

一句话说明，例如：

```text
— 保存 dual-artifact source 首轮收尾修复与测试补强方案。
```

---

## 14.2 补全 `ann_training/` 当前脚本列表

当前实际 `ann_training/` 不只有：

```text
train_and_evaluate_qwen3_1_7b_phase_aware.sh
```

应把当前目录中的脚本全部同步到 `代码结构总结.md`，至少包括当前存在的：

```text
train_and_evaluate_qwen3_1_7b_gif_aware_LEARNING_RATE.sh
train_and_evaluate_qwen3_1_7b_phase_aware.sh
train_and_evaluate_qwen3_1_7b_phase_aware_GROUP_SIZE.sh
train_and_evaluate_qwen3_1_7b_phase_aware_LEARNING_RATE.sh
train_and_evaluate_qwen3_8b_gif_aware.sh
train_and_evaluate_qwen3_8b_phase_aware.sh
train_discover_prefix_and_evaluate_qwen3_8b_tldr_vanilla_unaware_LEARNING_RATE.sh
```

若目录中还有其他当前文件，以实际 `ann_training/` 为准，一并补齐。

每个文件仍然只写**一句话职责**，遵守 `AGENTS.md` Rule 13。

不要新建其他章节。

---

# 15. `train_and_evaluate_qwen3_8b_phase_aware.sh` 的文档描述

该脚本当前只跑：

```bash
5.0e-05
```

这是确认保留的状态。

在 `代码结构总结.md` 描述时不要写成：

```text
多学习率 sweep
```

如果其当前逻辑只运行单 LR，应描述为当前真实行为，例如：

```text
— 以当前固定学习率执行 Qwen3-8B phase-aware ANN training 与后续流程。
```

---

# 16. 同步 `实验执行总结.md`

只修文案，不改变实验协议。

---

## 16.1 Step 1 开头

当前：

```text
四个开关分别控制四类实际执行阶段：
```

但实际列出了：

```text
rotated_pre_finetuning.prefix_enabled
ann_training.prefix_enabled
post_finetuning.prefix_enabled
evaluation.prefix_enabled
conversion.use_post_finetuning_artifacts
replacement.common_clip_enabled
```

共 6 个控制项。

改成：

```text
以下配置项分别控制 Prefix、SNN artifact source 与 common Clip 的实际执行阶段：
```

不要写具体“4 个”或“6 个”，避免以后配置项变化再次过时。

---

## 16.2 附录 B 标题

当前：

```text
附录 B：四个 prefix_enabled 的边界
```

但表中已经包含：

```text
4 个 prefix_enabled
+
conversion.use_post_finetuning_artifacts
```

改成：

```text
附录 B：Prefix 开关与 SNN artifact source selector 的边界
```

正文与表本身的现有语义保持不变。

---

# 17. 不要修改 `README.md` / `AGENTS.md` 主协议

当前以下内容已经正确，不要再次改写：

```text
README.md dual-artifact source matrix
AGENTS.md Rule 8 / 8a selector semantics
AGENTS.md Stage A / Stage B / Clip 规则
Final ANN Prefix source 独立于 selector
```

除非实现测试时发现实际代码与这些规则矛盾，否则只做必要修改。

---

# 18. 测试描述同步

修改 `代码结构总结.md` 中测试文件说明时，建议更新：

```text
test_conversion_metadata.py
```

为：

```text
— 验证 v14 selector-aware conversion、true/false source selection、aware provenance、schema、工件哈希、Rotation 与无 Clip 合同。
```

以及：

```text
test_verify_artifacts.py
```

改为类似：

```text
— 验证 Site topology、GIF provenance、Final ANN forward metadata 以及 selector-aware conversion/evaluation source matrix。
```

---

# 19. 最终测试

修改完成后执行：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```

或：

```bash
conda run -n snn2 python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

conda run -n snn2 pytest -q
```

---

# 20. 建议额外执行定向测试

先快速跑本轮直接相关测试：

```bash
pytest -q \
  tests/test_conversion_metadata.py \
  tests/test_post_finetuning_protocol.py \
  tests/test_verify_artifacts.py
```

然后再：

```bash
pytest -q
```

---

# 21. 完成标准

以下全部满足才算本轮完成：

1. Qwen3-8B phase-aware `LEARNING_RATES=(5.0e-05)` 保持不变；
2. `unaware + false` 有真实 `_source_bundle -> create_conversion -> validate_conversion_metadata` 回归；
3. `unaware + false` 明确测试不调用 aware training provenance；
4. `phase_aware + false` 有真实 frozen provenance validation；
5. 篡改 aware false training provenance 后 validation fail-fast；
6. `phase_aware + true` 明确允许 Post Stage A 与 ANN-training Stage A hash 不同；
7. `phase_aware + true` 不调用 aware training provenance validator；
8. gif-aware 至少有一个真实 selector source regression；
9. `verify_artifacts.py` 测试覆盖 unaware=false；
10. `verify_artifacts.py` 测试覆盖 aware=false；
11. `verify_artifacts.py` 测试覆盖 aware=true；
12. `validate_calibration()` clip-free 错误信息不再错误指向 Post-finetuning；
13. `代码结构总结.md` 加入 `SNN_dual_artifact_source_final_cleanup_plan.md`；
14. `代码结构总结.md` 补齐当前 `ann_training/` 脚本；
15. `实验执行总结.md` 不再写“4 个开关”这种已过时数量；
16. 附录 B 标题改成 Prefix + selector 边界；
17. `pytest -q` 全部通过。

---

# 22. 本轮禁止事项

- 不要恢复 Qwen3-8B phase-aware 多学习率 sweep；
- 不要改变 selector 的业务语义；
- 不要改变 ANN checkpoint path；
- 不要改变 Final ANN Prefix source；
- 不要让 unaware 单独生成 ANN-training Stage A；
- 不要让 unaware=false 校验 aware training provenance；
- 不要让 aware=true 强制复用 training calibration；
- 不要让 Post-finetuning 重新允许 Stage B；
- 不要修改已正确的 Site topology、Phase/GIF/MTN 数学、Prefix temporal policy 或 Final RMSNorm 语义。
