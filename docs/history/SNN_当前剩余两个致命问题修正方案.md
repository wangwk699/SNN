# SNN 当前剩余两个致命问题修正方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`
> 当前检查基线：`main` commit `7004c7641ab54129e964231b20f8021176e4de1c`
> 本文档只处理当前仍存在的 **2 个必须修复的致命问题**。其他已经完成的 Calibration A/B、Rotation canonical 128、Prefix `num_samples` 隔离、Stage B Clip profile、runtime T/K 解耦等逻辑保持不变。

---

# 1. 本轮必须修复的两个问题

1. `scripts/verify_artifacts.py` 存在真实语法错误，并且 SNN evaluation path verification 逻辑存在重复/错误调用。
2. Vanilla final ANN evaluation 与 Vanilla SNN evaluation 共用同一个 `final_evaluation` Prefix stage，导致 Vanilla SNN evaluation 被错误关闭 post-finetuning Prefix。

完成后必须运行：

```bash
pytest -q
```

并全部通过。

---

# 2. 修正 1：`verify_artifacts.py` 语法错误与 SNN evaluation path verification

## 2.1 当前致命语法错误

当前 `scripts/verify_artifacts.py` 的 SNN verification 中存在类似：

```python
if (
    int(metadata.get("full_temporal_steps", -1))
    != int(...)
):
    raise ValueError(
metrics_path = evaluation_paths(
    layout.snn_dir(neuron), neuron=neuron
)[0]
    )
```

这里 `metrics_path = ...` 被错误插入 `raise ValueError(` 内部，导致文件无法被 Python parse。

正确恢复为：

```python
metadata = validate_conversion_metadata(cfg, layout, neuron)

expected_steps = {
    "phase": int(cfg["phase"]["T"]),
    "gif": 2,
    "mtn": int(cfg["mtn"]["T"]),
}[neuron]

if int(metadata.get("full_temporal_steps", -1)) != expected_steps:
    raise ValueError(
        f"{neuron} conversion metadata has incompatible temporal steps"
    )

metrics_path = evaluation_paths(
    layout.snn_dir(neuron),
    neuron=neuron,
)[0]
```

---

## 2.2 补齐缺失 import

当前 `evaluation_paths()` 内部对 SNN 使用：

```python
evaluation_prefix_enabled(cfg)
```

但顶部缺少 import。

补：

```python
from snn2.config import (
    conversion_prefix_enabled,
    conversion_reuses_ann_training_artifacts,
    evaluation_prefix_enabled,
    final_ann_evaluation_prefix_enabled,
    requires_ann_training_calibration,
    requires_pre_finetuning_prefix,
    training_common_clip_enabled,
    training_prefix_enabled,
)
```

---

## 2.3 删除重复的 SNN `evaluation_paths()` 调用

当前存在类似：

```python
required.extend(
    [
        *evaluation_paths(layout.snn_dir(neuron), neuron=neuron),
        *evaluation_paths(layout.snn_dir(neuron)),
    ]
)
```

第二个调用默认 `neuron="ann"`，会让同一 SNN root 同时按 ANN Prefix policy 和 SNN Prefix policy 被验证。

改为：

```python
required.extend(
    evaluation_paths(
        layout.snn_dir(neuron),
        neuron=neuron,
    )
)
```

---

## 2.4 删除错误的 `metrics_path` 覆盖

当前类似：

```python
metrics_path = evaluation_paths(
    layout.snn_dir(neuron),
    neuron=neuron,
)[0]

metrics_path = evaluation_paths(
    layout.snn_dir(neuron),
)[0]
```

第二个会再次退回 `neuron="ann"`。

只保留：

```python
metrics_path = evaluation_paths(
    layout.snn_dir(neuron),
    neuron=neuron,
)[0]
```

---

## 2.5 TL;DR selection verification 保留真实 neuron

不要把所有 SNN root 都统一当成固定 neuron。

建议：

```python
evaluation_roots = [
    ("ann", layout.ann_dir),
    ("phase", layout.snn_dir("phase")),
    ("gif", layout.snn_dir("gif")),
    ("mtn", layout.snn_dir("mtn")),
]

for neuron, root in evaluation_roots:
    selection_path = evaluation_paths(
        root,
        neuron=neuron,
    )[1]
```

---

## 2.6 `evaluation_paths()` 最终语义

保持：

```python
def evaluation_paths(root, *, neuron="ann"):
```

内部：

```python
if neuron == "ann":
    enabled = final_ann_evaluation_prefix_enabled(cfg)
else:
    enabled = evaluation_prefix_enabled(cfg)

directory = directory / prefix_enabled_dirname(enabled)

directory = append_evaluation_num_samples_if_needed(
    directory,
    cfg,
    neuron=neuron,
)
```

即：

```text
ANN -> final ANN evaluation Prefix policy
SNN -> SNN/final evaluation Prefix policy
```

---

# 3. 修正 2：拆分 final ANN evaluation 与 final SNN evaluation Prefix stage

## 3.1 当前问题

当前 `snn2/modeling.py` 对：

```python
stage == "final_evaluation"
```

使用：

```python
if not final_ann_evaluation_prefix_enabled(cfg):
    return []
```

对于 Vanilla：

```python
final_ann_evaluation_prefix_enabled(vanilla) == False
```

这对 Vanilla final ANN evaluation 是正确的。

但 `evaluate_tldr.py` 和 `evaluate_lm_harness.py` 对 final ANN 和 final SNN 都传：

```python
stage="final_evaluation"
```

于是 Vanilla SNN evaluation 也会得到：

```text
prefix_ids = []
prefix KV = None
```

这是错误的。

---

# 4. 最终确认的 Vanilla Prefix 语义

必须满足：

## Vanilla final ANN evaluation

```text
final vanilla ANN checkpoint
+ identity forward
+ no Prefix
+ no replacement calibration
```

即：

```text
Prefix = OFF
```

## Vanilla post-finetuning preprocessing

仍然必须执行：

```text
Post-finetuning Prefix discovery
Post-finetuning Stage A calibration
```

用于：

```text
Vanilla ANN
-> SNN conversion
-> SNN evaluation
```

## Vanilla SNN evaluation

当：

```yaml
evaluation:
  prefix_enabled: true
```

时必须使用：

```text
post-finetuning Prefix
```

---

# 5. 正确做法：拆分两个 Prefix stage

推荐将统一的：

```text
final_evaluation
```

在 Prefix artifact resolution 层拆成：

```text
final_ann_evaluation
final_snn_evaluation
```

---

# 6. `prefix_ids_for_stage()` 修改

新增：

```python
elif stage == "final_ann_evaluation":
    if not final_ann_evaluation_prefix_enabled(cfg):
        return []

    artifact_stage = final_evaluation_prefix_artifact_stage(cfg)

    path = (
        layout.ann_training_prefix_dir
        if artifact_stage == "pre_finetuning"
        else layout.post_finetuning_prefix_dir
    ) / "prefix_state.json"
```

以及：

```python
elif stage == "final_snn_evaluation":
    if not evaluation_prefix_enabled(cfg):
        return []

    artifact_stage = final_evaluation_prefix_artifact_stage(cfg)

    path = (
        layout.ann_training_prefix_dir
        if artifact_stage == "pre_finetuning"
        else layout.post_finetuning_prefix_dir
    ) / "prefix_state.json"
```

---

# 7. `prefix_key_values_for_stage()` 同步拆分

同样增加：

```text
final_ann_evaluation
final_snn_evaluation
```

并先通过：

```python
prefix_ids_for_stage(...)
```

判断是否需要 KV。

artifact root 仍通过：

```python
artifact_stage = final_evaluation_prefix_artifact_stage(cfg)

directory = (
    layout.ann_training_prefix_dir
    if artifact_stage == "pre_finetuning"
    else layout.post_finetuning_prefix_dir
)
```

---

# 8. Prefix artifact source 规则保持不变

保持：

```text
phase_aware / gif_aware
    -> pre_finetuning Prefix

vanilla / unaware
    -> post_finetuning Prefix
```

本轮只拆分“是否启用 Prefix”，不要修改“Prefix 从哪里取”。

---

# 9. `evaluate_tldr.py` 修改

当前：

```python
prefix_stage = (
    "base_evaluation"
    if args.base
    else (
        "rotated_pre_finetuning"
        if args.rotated_pre_finetuning
        else "final_evaluation"
    )
)
```

改成：

```python
if args.base:
    prefix_stage = "base_evaluation"

elif args.rotated_pre_finetuning:
    prefix_stage = "rotated_pre_finetuning"

elif args.neuron == "ann":
    prefix_stage = "final_ann_evaluation"

else:
    prefix_stage = "final_snn_evaluation"
```

之后所有：

```python
prefix_ids_for_stage(...)
prefix_key_values_for_stage(...)
```

统一使用 `prefix_stage`。

---

# 10. `evaluate_lm_harness.py` 同步

同样先统一构造：

```python
if args.base:
    prefix_stage = "base_evaluation"

elif args.rotated_pre_finetuning:
    prefix_stage = "rotated_pre_finetuning"

elif args.neuron == "ann":
    prefix_stage = "final_ann_evaluation"

else:
    prefix_stage = "final_snn_evaluation"
```

后面 `prefix_ids_for_stage()` 和 `prefix_key_values_for_stage()` 全部使用该变量。

---

# 11. Final evaluation Prefix policy 总表

| 模式 | Evaluation | Prefix |
|---|---|---|
| Vanilla | final ANN | OFF |
| Vanilla | Phase/GIF/MTN SNN | post-finetuning Prefix |
| Unaware | final ANN | `evaluation.prefix_enabled` 控制，post-finetuning Prefix |
| Unaware | SNN | `evaluation.prefix_enabled` 控制，post-finetuning Prefix |
| Phase-aware | final ANN | `evaluation.prefix_enabled` 控制，pre-finetuning Prefix |
| Phase-aware | SNN | `evaluation.prefix_enabled` 控制，pre-finetuning Prefix |
| GIF-aware | final ANN | `evaluation.prefix_enabled` 控制，pre-finetuning Prefix |
| GIF-aware | SNN | `evaluation.prefix_enabled` 控制，pre-finetuning Prefix |

另外：

```text
Base evaluation
    -> no Prefix

Rotated pre-finetuning evaluation
    -> rotated_pre_finetuning.prefix_enabled
    -> pre-finetuning Prefix
```

---

# 12. Vanilla final ANN path 保持不变

Vanilla final ANN：

```text
prefix_enabled_false
```

且不增加：

```text
num_samples_N
```

例如：

```text
ann/evaluation/tldr/
└── test_samples_256/
    └── prefix_enabled_false/
```

---

# 13. Vanilla SNN path 保持现有 non-aware isolation

Vanilla SNN 的：

```python
layout.snn_dir(neuron)
```

已经包含：

```text
calibration_group_size_G_num_samples_N
```

因此无需额外修改路径。

只需要确保真实 SNN forward 加载：

```text
layout.post_finetuning_prefix_dir
```

对应 Prefix。

---

# 14. Metadata 必须与实际 forward 一致

修复后以下字段必须和真实行为一致：

```text
prefix_enabled
prefix_root
prefix_stage
prefix_source_stage
prefix_token_ids
actual installed Prefix KV
```

建议：

## Vanilla final ANN

```json
{
  "prefix_enabled": false,
  "prefix_stage": "final_ann_evaluation",
  "prefix_source_stage": null,
  "prefix_root": null
}
```

## Vanilla SNN

```json
{
  "prefix_enabled": true,
  "prefix_stage": "final_snn_evaluation",
  "prefix_source_stage": "post_finetuning",
  "prefix_root": ".../post_finetuning/prefix/num_samples_128"
}
```

---

# 15. `verify_artifacts.py` 同步验证 ANN/SNN Prefix stage

应继续使用：

```python
if neuron == "ann":
    enabled = final_ann_evaluation_prefix_enabled(cfg)
else:
    enabled = evaluation_prefix_enabled(cfg)
```

并额外验证：

```text
ANN -> prefix_stage == final_ann_evaluation
SNN -> prefix_stage == final_snn_evaluation
```

以及 `prefix_root` 与：

```python
final_evaluation_prefix_artifact_stage(cfg)
```

一致。

---

# 16. 必须新增的 tests

至少新增/更新：

1. `verify_artifacts.py` 可正常 import。
2. SNN evaluation paths 不再重复调用 ANN policy。
3. Vanilla `final_ann_evaluation` 返回空 Prefix。
4. Vanilla `final_snn_evaluation` 读取 post-finetuning Prefix。
5. Vanilla SNN 非空 Prefix 时读取对应 `prefixed_key_values.pt`。
6. 同一 Vanilla config 下：
   ```text
   final_ann_evaluation -> OFF
   final_snn_evaluation -> ON
   ```
7. Unaware ANN/SNN 均按 `evaluation.prefix_enabled` 使用 post-finetuning Prefix。
8. Phase-aware/GIF-aware ANN/SNN 均使用 pre-finetuning Prefix。
9. Base evaluation 和 rotated pre-finetuning evaluation 逻辑不被破坏。
10. evaluation metadata 与实际 installed Prefix 一致。

---

# 17. 推荐修改文件

至少：

```text
snn2/modeling.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
```

测试：

```text
tests/test_verify_artifacts.py
tests/test_evaluation.py
tests/test_post_finetuning_protocol.py
tests/test_rotated_pre_finetuning_protocol.py
```

---

# 18. 不允许的实现方式

禁止：

```text
为了让 Vanilla SNN 使用 Prefix
-> 又把 Vanilla final ANN Prefix 打开
```

必须通过：

```text
final_ann_evaluation
final_snn_evaluation
```

拆分解决。

同时禁止：

```python
evaluation_paths(layout.snn_dir(neuron))
```

这种默认 `neuron="ann"` 的 SNN verification 调用。

Rotation canonical 128 逻辑本轮保持不动：

```text
Rotation regression = independent fixed canonical 128
```

---

# 19. 最终验收标准

1. `scripts/verify_artifacts.py` 可 parse/import。
2. `pytest -q` 全部通过。
3. SNN required evaluation paths 不再重复 ANN-policy path。
4. SNN metrics path 不再被 ANN-policy path 覆盖。
5. Vanilla final ANN Prefix OFF。
6. Vanilla post-finetuning Prefix 继续生成。
7. Vanilla post-finetuning Stage A calibration 继续生成。
8. Vanilla Phase/GIF/MTN SNN evaluation 使用 post-finetuning Prefix。
9. Unaware ANN/SNN Prefix policy 不被破坏。
10. Aware ANN/SNN 继续使用 pre-finetuning Prefix。
11. Base evaluation 无 Prefix。
12. Rotated pre-finetuning evaluation 使用原有 Prefix policy。
13. evaluation metadata 与实际 installed Prefix 一致。
14. Vanilla final ANN output path 不增加 `num_samples_N`。
15. Vanilla SNN non-aware path 继续通过 calibration variant 隔离。

---

# 20. 最小 smoke test

完成修改后：

```bash
pytest -q
```

并检查：

```text
A. Vanilla final ANN
   -> prefix_enabled=false
   -> no Prefix KV installed

B. Vanilla Phase SNN
   -> prefix_enabled=true
   -> prefix_stage=final_snn_evaluation
   -> Prefix root=post_finetuning/prefix/num_samples_N

C. Unaware ANN Prefix=true
   -> post-finetuning Prefix

D. Phase-aware ANN
   -> pre-finetuning Prefix

E. verify_artifacts.py
   -> can import
   -> SNN evaluation path verification 不再出现 duplicate ANN-policy paths
```

以上通过后，再开始正式实验。
