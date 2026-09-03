# SNN verify_artifacts Prefix summary 最终修复方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 当前检查基于 `main` 最新提交：`b5b10e2dba09152e2058f62e6d1547796d2206ab`  
> 本文面向部署在服务器上的 Codex；请在没有本次对话上下文的情况下，仅依据本文完成本轮修改。
>
> 本轮只修复 `scripts/verify_artifacts.py` 中一个运行时 `NameError` 风险，并补对应 regression test。  
> 不修改训练、conversion、SNN forward、Prefix source selector、aware provenance 逻辑，也不修改用户当前用于测试的 T/K sweep 脚本参数。

---

# 1. 明确保留，不修改

## 1.1 保留当前 T/K sweep 测试脚本改动

文件：

```text
snn_evaluate/evaluate_qwen3_1_7b_phase_aware_T_K_sweep.sh
```

当前改动是用户主动进行的实验测试，**保持原样，不恢复、不重构、不纳入问题修复范围**。

---

## 1.2 保留 Qwen3-8B phase-aware 单学习率设置

文件：

```text
ann_training/train_and_evaluate_qwen3_8b_phase_aware.sh
```

继续保持：

```bash
LEARNING_RATES=(
  5.0e-05
)
```

不要恢复旧的多学习率 sweep。

---

## 1.3 不修改 dual-artifact source 核心逻辑

继续保持：

```yaml
conversion:
  use_post_finetuning_artifacts: true|false
```

语义：

```text
true
    -> SNN 使用 Post-finetuning Prefix + Post-finetuning Stage A

false
    -> 非 vanilla SNN 使用 shared Pre-finetuning Prefix
       + shared ANN-training Stage A
```

Final ANN Prefix 规则继续独立于 selector：

```text
vanilla      -> no Prefix
unaware      -> Post-finetuning Prefix
phase_aware  -> Pre-finetuning Prefix
gif_aware    -> Pre-finetuning Prefix
```

不要再修改：

```text
snn2/config.py
snn2/artifacts.py
snn2/conversion.py
snn2/modeling.py
snn2/evaluation.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

除非测试暴露出与本轮问题直接相关的新错误。

---

# 2. 当前唯一必须修复的问题

文件：

```text
scripts/verify_artifacts.py
```

本轮 Prefix validation 重构以后，旧代码中的：

```python
prefix_token_ids = []
```

以及后续：

```python
prefix_token_ids = [...]
```

已经被移除。

现在 Prefix validation 改为：

```python
_validate_prefix_artifact(...)
```

例如 selected SNN Prefix：

```python
if conversion_prefix_enabled(cfg) or evaluation_prefix_enabled(cfg):
    _validate_prefix_artifact(
        cfg,
        layout,
        layout.conversion_prefix_dir,
        label="Selected SNN",
    )
```

但是 `_validate_prefix_artifact()` 的返回值被丢弃。

---

# 3. 当前会触发的运行时错误

`main()` 最后仍然存在：

```python
result = {
    "required_files": len(required),
    "calibration": calibration,
    "conversion_descriptors": conversions,
    "prefix_length": len(prefix_token_ids),
    "prefix_kv_required": bool(prefix_token_ids),
    **topology_metadata(),
}
```

此时：

```python
prefix_token_ids
```

已经没有任何局部定义。

因此只要：

```text
verify_artifacts.py
```

成功执行到最终结果构造，就会触发：

```text
NameError: name 'prefix_token_ids' is not defined
```

这是运行时错误，普通：

```bash
python -m py_compile scripts/verify_artifacts.py
```

无法检测。

---

# 4. 修复原则

`artifact_verification.json` 中：

```text
prefix_length
prefix_kv_required
```

应该描述的是**当前 selected SNN Prefix**，而不是 Final ANN Prefix。

原因：

```text
selector=true
    -> SNN Prefix = Post-finetuning Prefix

selector=false
    -> SNN Prefix = shared Pre-finetuning Prefix
```

而 Final ANN Prefix source 是另一套独立规则。

因此不要重新引入一个模糊的：

```python
prefix_token_ids
```

变量。

建议使用明确命名：

```python
selected_snn_prefix_info
selected_snn_prefix_ids
```

---

# 5. 推荐的最小代码修改

在：

```text
scripts/verify_artifacts.py
```

selected SNN Prefix validation 之前初始化：

```python
selected_snn_prefix_info = {
    "token_ids": [],
}
```

然后将当前：

```python
if conversion_prefix_enabled(cfg) or evaluation_prefix_enabled(cfg):
    _validate_prefix_artifact(
        cfg,
        layout,
        layout.conversion_prefix_dir,
        label="Selected SNN",
    )
```

改成：

```python
if conversion_prefix_enabled(cfg) or evaluation_prefix_enabled(cfg):
    selected_snn_prefix_info = _validate_prefix_artifact(
        cfg,
        layout,
        layout.conversion_prefix_dir,
        label="Selected SNN",
    )
```

之后：

```python
selected_snn_prefix_ids = [
    int(value)
    for value in selected_snn_prefix_info.get("token_ids", [])
]
```

最终：

```python
result = {
    "required_files": len(required),
    "calibration": calibration,
    "conversion_descriptors": conversions,
    "prefix_length": len(selected_snn_prefix_ids),
    "prefix_kv_required": bool(selected_snn_prefix_ids),
    **topology_metadata(),
}
```

---

# 6. 为什么不能使用 Final ANN Prefix result

当前 verify 已经正确拆成：

```text
A. ANN-training / shared Pre Prefix
B. Final ANN evaluation Prefix
C. Selected SNN Prefix
```

必须保持这个边界。

尤其：

```text
unaware + selector=false
```

实际是：

```text
Final ANN
    -> Post Prefix

SNN
    -> Pre Prefix
```

因此以下写法是错误的：

```python
prefix_length = len(final_ann_prefix_info["token_ids"])
```

也不要：

```python
prefix_length = len(training_prefix_info["token_ids"])
```

正确来源只能是：

```python
layout.conversion_prefix_dir
```

对应的 selected SNN Prefix validation result。

---

# 7. selector-aware summary 必须满足

修复后：

## selector=true

```text
selected_snn_prefix_info
    -> layout.post_finetuning_prefix_dir
```

最终：

```text
prefix_length
prefix_kv_required
```

描述 Post Prefix。

---

## selector=false

```text
selected_snn_prefix_info
    -> layout.ann_training_prefix_dir
```

最终：

```text
prefix_length
prefix_kv_required
```

描述 shared Pre Prefix。

---

# 8. Prefix disabled 场景

若：

```python
conversion_prefix_enabled(cfg) is False
and
evaluation_prefix_enabled(cfg) is False
```

则：

```python
selected_snn_prefix_info = {
    "token_ids": [],
}
```

保持默认值。

最终：

```text
prefix_length = 0
prefix_kv_required = false
```

不能触发未定义变量。

---

# 9. 不要回退本轮已经正确的 Prefix validation

当前以下修复已经正确，本轮不要改回去：

## Final ANN Prefix root

继续使用：

```python
_final_ann_prefix_root(cfg, layout)
```

并根据：

```python
final_ann_evaluation_prefix_artifact_stage(cfg)
```

选择：

```text
unaware     -> Post
phase/gif   -> Pre
vanilla     -> none
```

---

## Selected SNN Prefix

继续使用：

```python
layout.conversion_prefix_dir
```

其 source 由：

```yaml
conversion.use_post_finetuning_artifacts
```

控制。

---

## aware training provenance

继续无条件对 aware mode：

```python
_validate_aware_final_ann_training_provenance(cfg, layout)
```

不要重新绑定：

```python
reused
```

---

# 10. 新增 regression test

文件：

```text
tests/test_verify_artifacts.py
```

当前已有大量 helper-level tests，但没有覆盖 `main()` 最终 Prefix summary 变量的生命周期。

本轮至少增加一个 regression，确保：

```text
Prefix validation refactor
```

不会再次留下悬空局部变量。

---

# 11. 推荐做法 A：抽一个小 helper

为了避免为了测试启动完整 `verify_artifacts.main()`，建议抽一个非常小的纯函数：

```python
def _selected_snn_prefix_summary(prefix_info):
    token_ids = [
        int(value)
        for value in prefix_info.get("token_ids", [])
    ]
    return {
        "prefix_length": len(token_ids),
        "prefix_kv_required": bool(token_ids),
    }
```

然后：

```python
result = {
    "required_files": len(required),
    "calibration": calibration,
    "conversion_descriptors": conversions,
    **_selected_snn_prefix_summary(selected_snn_prefix_info),
    **topology_metadata(),
}
```

这样可以直接单元测试。

注意：

- 这是推荐的小重构；
- 不要为此大规模重构 `main()`；
- 如果 Codex 更倾向保持最小 diff，也可以不抽 helper，见下一节。

---

# 12. 推荐 regression tests

若采用 helper：

```python
@pytest.mark.parametrize(
    ("token_ids", "expected_length", "expected_required"),
    [
        ([], 0, False),
        ([151643], 1, True),
        ([1, 2, 3], 3, True),
    ],
)
def test_selected_snn_prefix_summary(
    token_ids,
    expected_length,
    expected_required,
):
    summary = _VERIFY._selected_snn_prefix_summary(
        {"token_ids": token_ids}
    )

    assert summary == {
        "prefix_length": expected_length,
        "prefix_kv_required": expected_required,
    }
```

同时补 selector source regression：

```python
def test_selected_snn_prefix_summary_uses_conversion_prefix_result(...):
    ...
```

至少锁死：

```text
unaware + selector=false
```

时：

```text
Final ANN Post Prefix token_ids
!=
Selected SNN Pre Prefix token_ids
```

最终 summary 必须取：

```text
Selected SNN Pre Prefix
```

而不是 Final ANN Prefix。

---

# 13. 推荐做法 B：不抽 helper

如果坚持最小 diff，可以直接保留：

```python
selected_snn_prefix_info
selected_snn_prefix_ids
```

然后增加一个针对 Prefix-validation flow 的测试。

关键要求是测试必须实际证明：

```text
selected SNN validation result
```

被保存并用于 summary，而不是仅测试：

```python
_validate_prefix_artifact()
```

本身。

---

# 14. 建议增加一个完整 main smoke regression

如果当前测试基础设施允许，最好增加一个：

```python
test_verify_main_reaches_result_without_prefix_name_error(...)
```

通过 monkeypatch：

```text
parser/setup
StageRun
conversion validation
evaluation paths
artifact tree
```

构造最小 fixture，让：

```python
_VERIFY.main()
```

真正执行到：

```python
write_json(layout.root / "artifact_verification.json", result)
```

然后验证：

```text
artifact_verification.json
```

存在，且：

```python
payload["prefix_length"] == expected
payload["prefix_kv_required"] is expected_bool
```

这样可以直接防止：

```text
NameError
UnboundLocalError
```

此类 helper tests 检测不到的 `main()` 局部变量问题。

如果实现完整 main smoke 会引入大量脆弱 monkeypatch，则不强制；优先保证前述 summary helper regression。

---

# 15. 文档

本轮属于单点 bugfix。

`README.md`、`AGENTS.md`、`实验执行总结.md` 不需要修改。

`代码结构总结.md` 只有在本轮新增 history 文档入库时才需要同步。

若将本文加入：

```text
docs/history/
```

建议文件名：

```text
SNN_verify_artifacts_prefix_summary_nameerror_fix_plan.md
```

并在：

```text
代码结构总结.md
```

加入一行：

```text
SNN_verify_artifacts_prefix_summary_nameerror_fix_plan.md — 保存 verify_artifacts selected SNN Prefix summary 悬空变量修复方案。
```

---

# 16. 本轮禁止顺手修改的内容

不要因为本轮 bugfix 顺手调整：

```text
evaluate_qwen3_1_7b_phase_aware_T_K_sweep.sh
Qwen3-8B phase-aware LR
conversion selector
Final ANN Prefix source
SNN Prefix source
Stage A/B provenance
Post-finetuning protocol
temporal T/K
Site topology
GIF policy
Phase/MTN calibration
evaluation result directory
```

---

# 17. 测试命令

修改后运行：

```bash
cd /home/wangwenkang/SNN

python -m py_compile \
  scripts/verify_artifacts.py

pytest -q tests/test_verify_artifacts.py
pytest -q
```

如果使用 conda：

```bash
conda run -n snn2 python -m py_compile \
  scripts/verify_artifacts.py

conda run -n snn2 pytest -q tests/test_verify_artifacts.py
conda run -n snn2 pytest -q
```

注意：

```text
py_compile
```

本身不能证明本次 `NameError` 已修复。

必须至少运行：

```text
tests/test_verify_artifacts.py
```

中的新 regression。

---

# 18. 最终检查清单

- [ ] `evaluate_qwen3_1_7b_phase_aware_T_K_sweep.sh` 保持用户测试改动
- [ ] Qwen3-8B phase-aware 仍只保留 `5e-05`
- [ ] dual-artifact selector 核心逻辑未改
- [ ] Final ANN Prefix source 未改
- [ ] Selected SNN Prefix 仍使用 `layout.conversion_prefix_dir`
- [ ] aware final ANN provenance validation 未改
- [ ] `prefix_token_ids` 悬空引用已完全删除
- [ ] selected SNN Prefix validation return value 被保存
- [ ] `prefix_length` 来自 selected SNN Prefix
- [ ] `prefix_kv_required` 来自 selected SNN Prefix
- [ ] Prefix disabled 时 summary 为 `0 / false`
- [ ] selector=true summary 对应 Post Prefix
- [ ] selector=false summary 对应 Pre Prefix
- [ ] unaware+false 不会误用 Final ANN Post Prefix 做 SNN summary
- [ ] 新增 regression 能防止悬空局部变量再次出现
- [ ] `pytest -q tests/test_verify_artifacts.py` 通过
- [ ] `pytest -q` 全部通过

---

# 19. 本轮完成标准

完成后：

```text
python scripts/verify_artifacts.py --config ...
```

必须能够正常执行到最终：

```text
artifact_verification.json
```

写出阶段，不再发生：

```text
NameError: prefix_token_ids is not defined
```

并且最终：

```json
{
  "prefix_length": ...,
  "prefix_kv_required": ...
}
```

必须严格描述：

```text
当前 selector 选中的 SNN Prefix
```

而不是 Final ANN Prefix。
