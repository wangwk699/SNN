# SNN dual-artifact source：verify_artifacts 最终收尾修复方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 基于当前 `main` 最新提交：`49f4063ec7caea4fcbfdcc5e7c4b921a26c63cb9`  
> 本文面向部署在服务器上的 Codex；请在没有本次对话上下文的情况下，仅依据本文完成本轮修改。
>
> 本轮不修改训练、conversion、SNN forward 或 selector 核心语义，只修复 `verify_artifacts.py` 的两个验证缺口，并补对应 regression tests 与文档同步。

---

# 1. 本轮明确不修改的内容

以下内容已经确认正确，本轮禁止重构或改变。

## 1.1 Qwen3-8B phase-aware 学习率

保持：

```bash
LEARNING_RATES=(
  5.0e-05
)
```

文件：

```text
ann_training/train_and_evaluate_qwen3_8b_phase_aware.sh
```

不要恢复旧的多学习率 sweep。

---

## 1.2 SNN artifact source selector

继续保持：

```yaml
conversion:
  use_post_finetuning_artifacts: true|false
```

为 SNN conversion / SNN evaluation artifact source 的唯一 selector。

### `true`

使用：

```text
Post-finetuning Prefix
+
Post-finetuning Stage A
```

### `false`

仅允许：

```text
unaware
phase_aware
gif_aware
```

使用：

```text
shared Pre-finetuning Prefix
+
shared ANN-training Stage A
```

### vanilla

```text
vanilla + use_post_finetuning_artifacts=false
```

继续在 config validation 阶段非法。

---

## 1.3 Final ANN Prefix 规则

selector 不改变 Final ANN evaluation。

保持：

```text
vanilla
    -> 不加载 Prefix

unaware
    -> Post-finetuning Prefix

phase_aware
    -> Pre-finetuning Prefix

gif_aware
    -> Pre-finetuning Prefix
```

实际是否加载仍由：

```yaml
evaluation:
  prefix_enabled: true|false
```

控制。

---

## 1.4 Conversion provenance

保持当前已经正确的规则：

```text
unaware + selector=false
    -> shared Pre Prefix + shared ANN-training Stage A
    -> 不要求 aware ANN-training provenance

phase/gif aware + selector=false
    -> shared Pre Prefix + shared ANN-training Stage A
    -> 必须校验 ANN-training frozen provenance

phase/gif aware + selector=true
    -> Final ANN 的 Post Prefix + Post Stage A
    -> 不要求 Post Stage A hash == ANN-training Stage A hash
```

当前 `snn2/conversion.py` 的这些逻辑不改。

---

# 2. 问题一：aware + selector=true 时 verify 没有独立验证 ANN-training frozen provenance

当前 `scripts/verify_artifacts.py` 中大致存在：

```python
reused = conversion_reuses_ann_training_artifacts(cfg)

if reused and is_aware_ann_mode(cfg):
    training_result = read_json(...)
    ...
    validate_clip_profile(...)
```

这里把：

```text
Final aware ANN training provenance validation
```

错误地绑定到了：

```text
SNN conversion 是否复用 ANN-training Stage A
```

这两个职责必须拆开。

---

# 3. 为什么当前逻辑不完整

对于：

```text
phase_aware + selector=true
gif_aware + selector=true
```

SNN conversion 正确使用：

```text
Post-finetuning Prefix
+
Post-finetuning Stage A
```

因此 conversion 本身确实不应该要求：

```text
Post Stage A hash == ANN-training Stage A hash
```

这一点当前代码是正确的。

但是 Final aware ANN 本身仍然来自：

```text
Pre-finetuning Prefix
ANN-training Stage A
ANN-training selected Stage B Clip profile
```

并且训练时这些 artifacts 已经被冻结进：

```text
ann/training_result.json
```

因此：

```text
selector=true
```

不能让 `verify_artifacts.py` 放弃验证：

```text
Final ANN 的训练 provenance
```

否则可能出现：

```text
phase_aware
selector=true
ANN training 已完成
↓
Stage B Clip profile 后来被修改
↓
Post-finetuning Prefix / Post Stage A 完整
↓
conversion metadata 也完整
↓
verify_artifacts.py 仍可能通过
```

这与 `verify_artifacts.py` 的“完整 artifact validation”职责不符。

---

# 4. 修改 `scripts/verify_artifacts.py`：aware Final ANN provenance 与 selector 解耦

在：

```text
scripts/verify_artifacts.py
```

中导入：

```python
from snn2.training import validate_recorded_training_artifact_provenance
```

然后在当前 run 的基础 artifact existence check 完成后、正式验证 conversion/evaluation metadata 前，对所有 aware mode 执行：

```python
if is_aware_ann_mode(cfg):
    validate_recorded_training_artifact_provenance(cfg, layout)
```

注意：

```python
if is_aware_ann_mode(cfg):
```

不能写成：

```python
if reused and is_aware_ann_mode(cfg):
```

---

# 5. 保留 reused-aware conversion-specific validation

当前：

```python
if reused and is_aware_ann_mode(cfg):
    ...
    validate_clip_profile(...)
```

若其职责是：

```text
验证 selector=false 当前所选 shared ANN-training bundle 与训练期 frozen provenance
```

则可以继续保留。

但是必须区分两层验证：

## 第一层：Final aware ANN provenance

所有 aware mode：

```text
phase_aware + true
phase_aware + false
gif_aware + true
gif_aware + false
```

都必须执行：

```python
validate_recorded_training_artifact_provenance(cfg, layout)
```

## 第二层：selector=false conversion-source provenance

只有：

```text
reused=True && aware
```

才继续校验：

```text
当前 SNN selected ANN-training Stage A
+
selected Stage B / training provenance
```

不要把第二层当成第一层的替代。

---

# 6. 问题二：unaware + selector=false 时 verify 漏掉 Final ANN 使用的 Post Prefix

这是本轮最重要的 Prefix 边界问题。

对于：

```text
ann_mode = unaware
conversion.use_post_finetuning_artifacts = false
evaluation.prefix_enabled = true
```

实际 runtime 是：

```text
Final ANN Evaluation
    -> Post-finetuning Prefix

SNN conversion/evaluation
    -> shared Pre-finetuning Prefix
```

因此该 run 同时依赖两套不同 Prefix。

---

# 7. 当前 verify 的问题

当前 `verify_artifacts.py` 使用类似：

```python
prefix_state_path = layout.conversion_prefix_dir / "prefix_state.json"
```

然后：

```python
if conversion_prefix_enabled(cfg) or evaluation_prefix_enabled(cfg):
    required.append(prefix_state_path)
```

问题是：

```text
selector=false
```

时：

```python
layout.conversion_prefix_dir
```

指向：

```text
shared Pre-finetuning Prefix
```

因此在：

```text
unaware + selector=false
```

下，即使：

```python
evaluation_prefix_enabled(cfg) == True
```

上述检查仍然检查的是：

```text
Pre Prefix
```

而 Final unaware ANN 实际需要的是：

```text
Post-finetuning Prefix
```

这会造成 Final ANN Prefix artifact verification 漏检。

---

# 8. 将 verify 中 Prefix 校验拆成三种职责

不要再使用：

```text
一个 layout.conversion_prefix_dir 同时代表所有 evaluation Prefix
```

必须明确拆成：

```text
A. ANN-training / shared Pre Prefix
B. Final ANN evaluation Prefix
C. SNN conversion/evaluation selected Prefix
```

---

# 9. A：ANN-training / shared Pre Prefix

已有逻辑可以继续保留：

```python
if requires_pre_finetuning_prefix(cfg) and training_prefix_enabled(cfg):
    ...
```

root：

```python
layout.ann_training_prefix_dir
```

验证：

```text
prefix_state.json
validate_prefix_discovery_state(...)
非空 prefix_token_ids -> prefixed_key_values.pt
```

此部分主要服务：

```text
unaware / phase_aware / gif_aware
```

的 Pre Prefix。

---

# 10. B：新增 Final ANN evaluation Prefix validation

增加一个独立 helper，推荐：

```python
def _final_ann_prefix_root(cfg, layout):
    ...
```

或直接内联实现。

按照：

```python
final_ann_evaluation_prefix_artifact_stage(cfg)
```

解析 Final ANN Prefix root。

建议在 `scripts/verify_artifacts.py` 导入：

```python
from snn2.config import final_ann_evaluation_prefix_artifact_stage
```

规则：

```text
vanilla
    -> Final ANN Prefix disabled
    -> 不需要 root

unaware
    -> stage = post_finetuning
    -> layout.post_finetuning_prefix_dir

phase_aware
    -> stage = pre_finetuning
    -> layout.ann_training_prefix_dir

gif_aware
    -> stage = pre_finetuning
    -> layout.ann_training_prefix_dir
```

只有：

```python
final_ann_evaluation_prefix_enabled(cfg)
```

为 true 时要求 Prefix artifact。

---

# 11. Final ANN Prefix artifact 必须验证的内容

如果：

```python
final_ann_evaluation_prefix_enabled(cfg)
```

为 true：

必须验证：

```text
prefix_state.json
```

存在。

然后调用：

```python
validate_prefix_discovery_state(
    cfg,
    layout,
    final_ann_prefix_root,
)
```

如果：

```python
prefix_token_ids
```

非空，则必须存在：

```text
prefixed_key_values.pt
```

并由现有 validator 验证 hash / manifest provenance。

---

# 12. unaware + selector=false 的关键要求

必须确保以下两个 root 都被验证：

```text
Final ANN:
layout.post_finetuning_prefix_dir

SNN:
layout.ann_training_prefix_dir
```

不能因为：

```text
layout.conversion_prefix_dir == layout.ann_training_prefix_dir
```

而漏掉 Post Prefix。

---

# 13. C：SNN conversion/evaluation Prefix

SNN 部分继续使用：

```python
layout.conversion_prefix_dir
```

因为这本来就是 selector-aware source：

```text
selector=true
    -> Post Prefix

selector=false
    -> Pre Prefix
```

继续调用：

```python
validate_conversion_prefix(cfg, layout)
```

或等价 validation。

这部分不要改成 Final ANN Prefix root。

---

# 14. 加强 `_verify_final_ann_forward_metadata()`

当前 `_verify_final_ann_forward_metadata()` 主要验证：

```text
evaluation_forward_kind
controller_mode
temporal_execution
static replacement
common Clip
calibration root
GIF provenance
```

还应加入 Final ANN Prefix runtime metadata 验证。

---

# 15. Final ANN metrics 应校验的 Prefix metadata

Final ANN evaluation metrics 当前已经记录：

```text
prefix_enabled
prefix_stage
prefix_root
prefix_source_stage
```

因此 verify 应显式验证。

建议在 `_verify_final_ann_forward_metadata()` 中加入：

```python
expected_prefix_enabled = final_ann_evaluation_prefix_enabled(cfg)
```

验证：

```python
metadata.get("prefix_enabled") == expected_prefix_enabled
```

对于 Final ANN：

```python
metadata.get("prefix_stage") == "final_ann_evaluation"
```

若 enabled 为 false：

```python
metadata.get("prefix_root") is None
```

若 enabled 为 true：

根据：

```python
final_ann_evaluation_prefix_artifact_stage(cfg)
```

计算：

```text
expected source stage
expected Prefix root
```

并验证：

```python
metadata.get("prefix_source_stage") == expected_stage
```

以及：

```python
Path(metadata["prefix_root"]).resolve()
==
Path(expected_root).resolve()
```

---

# 16. Final ANN Prefix canonical matrix

verify tests 必须锁死：

| ann_mode | Final ANN Prefix enabled 条件 | Prefix source |
|---|---|---|
| vanilla | 永远 false | none |
| unaware | `evaluation.prefix_enabled` | Post-finetuning |
| phase_aware | `evaluation.prefix_enabled` | Pre-finetuning |
| gif_aware | `evaluation.prefix_enabled` | Pre-finetuning |

这一矩阵完全独立于：

```yaml
conversion.use_post_finetuning_artifacts
```

---

# 17. 修改 `tests/test_verify_artifacts.py`

当前新增的：

```python
test_verify_selector_source_matrix(...)
```

只测试：

```python
_validate_snn_source_metadata(...)
```

仍然保留，但不足以覆盖本轮两个实际问题。

必须新增真实 provenance regression。

---

# 18. Regression 1：phase_aware + selector=true 仍验证 ANN-training provenance

新增测试，建议名称：

```python
test_verify_aware_post_selector_still_validates_training_provenance(...)
```

场景：

```text
ann_mode = phase_aware
use_post_finetuning_artifacts = true
```

构造：

```text
valid ANN-training Pre Prefix
valid ANN-training Stage A
valid Stage B Clip profile
valid training_result.json
valid Post Prefix
valid Post Stage A
```

首先验证：

```text
aware Final ANN training provenance
```

可以通过。

然后篡改：

```text
ANN-training Stage B clip_profile_manifest.json
```

或：

```text
training_result.json 中的 ann_training_clip_profile_manifest_sha256
```

再执行对应 verify provenance helper。

必须失败。

期望错误来自：

```text
Recorded ANN-training Prefix/calibration provenance ...
```

或等价 frozen provenance mismatch。

最关键的是：

```text
selector=true
```

不能绕过该失败。

---

# 19. Regression 2：gif_aware + selector=true

建议至少再参数化一次：

```python
@pytest.mark.parametrize("ann_mode", ["phase_aware", "gif_aware"])
```

确保两个 aware mode 都遵守同一规则。

---

# 20. Regression 3：unaware + selector=false 必须验证 Final ANN Post Prefix

新增测试，建议名称：

```python
test_verify_unaware_pre_selector_requires_post_prefix_for_final_ann(...)
```

场景：

```text
ann_mode = unaware
use_post_finetuning_artifacts = false
evaluation.prefix_enabled = true
```

准备：

```text
shared Pre Prefix       # SNN source
Post-finetuning Prefix  # Final ANN source
```

首先两者都存在，Final ANN Prefix validation 应通过。

然后仅删除：

```text
layout.post_finetuning_prefix_dir / "prefix_state.json"
```

保持：

```text
layout.ann_training_prefix_dir / "prefix_state.json"
```

完整。

verify 必须失败。

错误必须明确指向：

```text
Post-finetuning Final ANN Prefix
```

而不是 Pre Prefix。

---

# 21. Regression 4：unaware + selector=false 的 SNN Pre Prefix 仍然独立验证

与上一测试相反：

保留 Post Prefix。

删除：

```text
shared Pre Prefix
```

SNN source validation 必须失败。

这样两个测试组合能证明：

```text
Final ANN Post Prefix
!=
SNN selected Pre Prefix
```

没有被 verify 混在一起。

---

# 22. Regression 5：Final ANN metrics Prefix source mismatch

对：

```text
unaware + selector=false
```

构造合法 Final ANN metrics：

```text
prefix_enabled = true
prefix_stage = final_ann_evaluation
prefix_source_stage = post_finetuning
prefix_root = layout.post_finetuning_prefix_dir
```

应通过。

然后篡改：

```text
prefix_source_stage = pre_finetuning
```

或：

```text
prefix_root = layout.ann_training_prefix_dir
```

`_verify_final_ann_forward_metadata()` 必须 fail-fast。

---

# 23. Regression 6：aware Final ANN Prefix 与 selector 独立

参数化：

```text
phase_aware + selector=true
phase_aware + selector=false
gif_aware + selector=true
gif_aware + selector=false
```

在：

```text
evaluation.prefix_enabled=true
```

时 Final ANN 必须始终：

```text
prefix_source_stage = pre_finetuning
prefix_root = ann_training_prefix_dir
```

selector 不得改变。

---

# 24. 不要错误修改 evaluator

当前：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

Final ANN / SNN Prefix source 已经分开。

本轮原则上不需要修改 evaluator 的 source selection。

尤其不要把 Final ANN Prefix 改成：

```python
layout.conversion_prefix_dir
```

否则会重新引入 dual-source bug。

---

# 25. 不要修改 `snn2/conversion.py` selector 逻辑

当前 conversion tests 已真实覆盖：

```text
unaware + false
phase_aware + false
gif_aware + false
phase_aware + true
```

并已确认：

```text
unaware + false
    -> 不进入 aware provenance

aware + false
    -> 校验 frozen provenance

aware + true
    -> 不要求 Post Stage A == training Stage A
```

本轮不要继续改 `_source_bundle()`。

---

# 26. 文档同步：`代码结构总结.md`

当前仓库新增：

```text
docs/history/SNN_dual_artifact_source_followup_cleanup_plan.md
```

必须在：

```text
代码结构总结.md
```

的：

```text
docs/history/
```

列表中补充一行。

建议：

```text
SNN_dual_artifact_source_followup_cleanup_plan.md — 保存 dual-artifact source 二次收尾、selector=false conversion 回归与 verify 补强方案。
```

本轮完成后，如果把本文件也放入：

```text
docs/history/
```

例如：

```text
SNN_dual_artifact_source_verify_final_cleanup_plan.md
```

则也应同步加入 `代码结构总结.md`。

---

# 27. `实验执行总结.md`

当前这两个问题已修正：

```text
Step 1 不再错误称“四个开关”
附录 B 已改为 Prefix 开关与 SNN artifact source selector
```

本轮不需要再改这些内容。

如果本轮 `verify_artifacts.py` 的职责说明发生变化，可仅在现有一句文件职责描述中补充：

```text
Final ANN Prefix source
aware training frozen provenance
```

但不要新增大段说明。

---

# 28. 推荐的 verify helper 结构

为了避免 `main()` 内重复逻辑，可以增加小 helper，但不要大规模重构。

推荐：

```python
def _validate_prefix_artifact(cfg, layout, root, *, label):
    ...
```

用于：

```text
Pre Prefix
Final ANN Prefix
SNN selected Prefix
```

或者至少增加：

```python
def _final_ann_prefix_root(cfg, layout):
    ...
```

核心要求不是函数命名，而是三个 Prefix responsibility 必须分开。

---

# 29. `verify_artifacts.py` 最终应体现的验证结构

推荐逻辑：

```text
1. config / data / checkpoint existence

2. shared Rotation artifacts

3. shared Pre Prefix
   if required by training

4. Final ANN Prefix
   according to final_ann_evaluation_prefix_artifact_stage()

5. selected SNN Prefix
   according to conversion selector

6. aware Final ANN training frozen provenance
   for every phase_aware/gif_aware
   independent of selector

7. selected conversion Stage A
   according to selector

8. selector=false + aware
   extra selected ANN-training conversion-source provenance

9. Final ANN evaluation metadata
   including Prefix metadata

10. phase/gif/mtn conversion metadata

11. phase/gif/mtn SNN evaluation metadata
```

---

# 30. Canonical dependency matrix for verify

## vanilla + true

```text
Final ANN:
    no Prefix

SNN:
    Post Prefix
    Post Stage A

aware training provenance:
    N/A
```

---

## unaware + true

```text
Final ANN:
    Post Prefix

SNN:
    Post Prefix
    Post Stage A

aware training provenance:
    N/A
```

---

## unaware + false

```text
Final ANN:
    Post Prefix

SNN:
    Pre Prefix
    ANN-training Stage A

aware training provenance:
    N/A
```

这一行必须同时验证：

```text
Post Prefix
Pre Prefix
```

---

## phase_aware + true

```text
Final ANN:
    Pre Prefix
    ANN-training Stage A
    selected Stage B

SNN:
    Post Prefix
    Post Stage A

aware training provenance:
    必须
```

---

## phase_aware + false

```text
Final ANN:
    Pre Prefix
    ANN-training Stage A
    selected Stage B

SNN:
    Pre Prefix
    ANN-training Stage A

aware training provenance:
    必须
```

---

## gif_aware + true

```text
Final ANN:
    Pre Prefix
    ANN-training Stage A
    selected Stage B

SNN:
    Post Prefix
    Post Stage A

aware training provenance:
    必须
```

---

## gif_aware + false

```text
Final ANN:
    Pre Prefix
    ANN-training Stage A
    selected Stage B

SNN:
    Pre Prefix
    ANN-training Stage A

aware training provenance:
    必须
```

---

# 31. 必须保持的关键 invariant

完成后必须满足：

### invariant A

```text
conversion selector
```

只控制：

```text
SNN Prefix source
SNN Stage A source
```

---

### invariant B

selector 不控制：

```text
Final ANN Prefix source
Final aware ANN training provenance
```

---

### invariant C

```text
unaware + false
```

同时依赖：

```text
Post Prefix for Final ANN
Pre Prefix for SNN
```

---

### invariant D

```text
aware + true
```

虽然 SNN 使用 Post bundle，但：

```text
Final ANN 的 ANN-training Prefix/Stage A/Stage B frozen provenance
```

仍必须通过 verify。

---

# 32. 测试执行

修改后至少运行：

```bash
cd /home/wangwenkang/SNN

python -m py_compile \
  scripts/verify_artifacts.py \
  snn2/conversion.py \
  snn2/training.py

pytest -q tests/test_verify_artifacts.py
pytest -q tests/test_conversion_metadata.py
pytest -q
```

如果使用 conda：

```bash
conda run -n snn2 python -m py_compile \
  scripts/verify_artifacts.py \
  snn2/conversion.py \
  snn2/training.py

conda run -n snn2 pytest -q tests/test_verify_artifacts.py
conda run -n snn2 pytest -q tests/test_conversion_metadata.py
conda run -n snn2 pytest -q
```

---

# 33. 最终检查清单

完成本轮后逐项确认：

- [ ] Qwen3-8B phase-aware 仍只保留 `5e-05`
- [ ] selector 核心语义未改变
- [ ] Final ANN Prefix source 未改成 selector-driven
- [ ] `validate_recorded_training_artifact_provenance()` 被 `verify_artifacts.py` 用于所有 aware mode
- [ ] aware + true 仍验证 ANN-training Stage B frozen provenance
- [ ] aware + false 仍验证 ANN-training frozen provenance
- [ ] unaware + false 不要求 aware training provenance
- [ ] unaware + false Final ANN Post Prefix 被独立验证
- [ ] unaware + false SNN Pre Prefix 被独立验证
- [ ] Final ANN metrics 校验 `prefix_enabled`
- [ ] Final ANN metrics 校验 `prefix_stage`
- [ ] Final ANN metrics 校验 `prefix_source_stage`
- [ ] Final ANN metrics 校验 `prefix_root`
- [ ] phase/gif aware Final ANN Prefix 始终是 Pre，不受 selector 影响
- [ ] SNN Prefix 仍由 selector 决定
- [ ] selector source metadata regression 继续保留
- [ ] 新增 aware=true Stage B tamper regression
- [ ] 新增 unaware=false Post Prefix missing/tamper regression
- [ ] 新增 Final ANN Prefix metadata mismatch regression
- [ ] `代码结构总结.md` 补齐 followup cleanup history 文档
- [ ] 本轮新 history 文档若入库也同步到 `代码结构总结.md`
- [ ] `pytest -q` 全部通过

---

# 34. 本轮完成标准

本轮完成后，应不存在这种情况：

```text
SNN Post bundle 正常
但 Final aware ANN 的 training artifacts 已被修改
verify 仍通过
```

也不应存在：

```text
unaware + selector=false
Pre Prefix 正常
Post Prefix 已缺失/损坏
verify 仍通过
```

最终 `verify_artifacts.py` 必须同时回答两个独立问题：

```text
1. Final ANN 当前实际依赖的 artifacts 是否仍与训练/评估协议一致？
2. 当前 selector 选中的 SNN artifacts 是否完整且与 conversion/evaluation metadata 一致？
```

只有二者同时通过，整个 run 才算 artifact verification 成功。
