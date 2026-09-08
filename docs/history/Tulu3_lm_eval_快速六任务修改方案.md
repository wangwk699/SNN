# Tulu-3 / lm-eval 快速六任务修改方案

## 0. 目的

基于当前仓库：

```text
https://github.com/wangwk699/SNN/tree/main
```

仅对 **Tulu-3 / lm-evaluation-harness 评估任务配置与相应协议代码**进行修改，并清理 `configs/experiment_matrix.yaml` 中 TL;DR 与 Tulu-3 互相混入的 evaluation 参数。

本轮修改后的 Tulu-3 **启用** 6 个 benchmark：

1. `truthfulqa_mc1`
2. `agieval`
3. `arc_challenge`
4. `piqa`
5. `winogrande`
6. `boolq`

原先以下 4 个慢任务**不删除**，但统一放到 `lm_eval_task_specs` 最后，并设为：

```yaml
enabled: false
```

即：

7. `mmlu_pro`
8. `bbh`
9. `gsm8k_cot`
10. `minerva_math`

固定的 lm-eval revision **保持不变**：

```text
6d2abda4fd171e68a8789330c4149e37c1ca0bda
```

---

# 1. 硬性约束

## 1.1 只修改本轮明确涉及的内容

不要顺手修改：

- rotation
- Prefix
- calibration
- ANN/SNN replacement
- Phase/GIF/MTN
- training 超参数
- artifact path 规则
- conversion
- post-finetuning
- scheduler / warmup
- Tulu-3 数据划分与 `training.train_samples`
- TL;DR 数据/训练/评估语义

尤其：

> `configs/experiment_matrix.yaml` 中每个实验除 `evaluation:` section 外，其他 section 不要改。

---

## 1.2 TL;DR 与 Tulu-3 的 evaluation 参数彻底分离

### TL;DR 实验

以下两个 base experiment：

```text
exp1_qwen3_1_7b_tldr
exp1_qwen3_8b_tldr
```

其 `evaluation:` 中只保留 TL;DR 实际需要的字段，不再保留任何 Tulu-3 / lm-eval 专属字段。

### Tulu-3 实验

```text
exp2_llama3_8b_tulu3
```

其 `evaluation:` 中只保留 Tulu-3 / lm-eval 实际需要的字段，不再保留 TL;DR 专属字段。

---

# 2. 修改 `configs/experiment_matrix.yaml`

文件：

```text
configs/experiment_matrix.yaml
```

## 2.1 `exp1_qwen3_1_7b_tldr`

其 `evaluation:` 最终应只保留 TL;DR 所需内容：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8
  max_new_tokens: 32
  tldr_input_length: 512
  tldr_test_samples: 1000
  tldr_test_seed: 42
  rouge_types:
  - rouge1
  - rouge2
  - rougeL
  - rougeLsum
```

删除：

```yaml
limit: null
lm_eval_revision: ...
apply_chat_template: ...
lm_eval_task_specs: ...
```

## 2.2 `exp1_qwen3_8b_tldr`

同样处理。其 `evaluation:` 最终只保留：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8
  max_new_tokens: 32
  tldr_input_length: 512
  tldr_test_samples: 1000
  tldr_test_seed: 42
  rouge_types:
  - rouge1
  - rouge2
  - rougeL
  - rougeLsum
```

删除：

```text
limit
lm_eval_revision
apply_chat_template
lm_eval_task_specs
```

两个 TL;DR experiment 的其他 section 与 evaluation 数值不要改。

## 2.3 `exp2_llama3_8b_tulu3`

从 Tulu-3 `evaluation:` 删除所有 TL;DR 专属字段：

```text
max_new_tokens
tldr_input_length
tldr_test_samples
tldr_test_seed
rouge_types
```

保留：

```yaml
prefix_enabled: true
batch_size: 8
lm_eval_revision: 6d2abda4fd171e68a8789330c4149e37c1ca0bda
apply_chat_template: true
lm_eval_task_specs:
  ...
```

不要重新引入全局 `limit`。Tulu-3 已使用每任务独立 `test_samples`。

---

# 3. Tulu-3 最终 benchmark 配置

`exp2_llama3_8b_tulu3` 的 `evaluation.lm_eval_task_specs` 最终按以下顺序与参数设置：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8
  lm_eval_revision: 6d2abda4fd171e68a8789330c4149e37c1ca0bda
  apply_chat_template: true
  lm_eval_task_specs:

  - name: truthfulqa_mc1
    enabled: true
    num_fewshot: 0
    metric: acc
    cot: false
    test_samples: null
    test_seed: 42

  - name: agieval
    enabled: true
    num_fewshot: 0
    metric: acc
    cot: false
    test_samples: null
    test_seed: 42

  - name: arc_challenge
    enabled: true
    num_fewshot: 0
    metric: acc_norm
    cot: false
    test_samples: null
    test_seed: 42

  - name: piqa
    enabled: true
    num_fewshot: 0
    metric: acc_norm
    cot: false
    test_samples: null
    test_seed: 42

  - name: winogrande
    enabled: true
    num_fewshot: 0
    metric: acc
    cot: false
    test_samples: null
    test_seed: 42

  - name: boolq
    enabled: true
    num_fewshot: 0
    metric: acc
    cot: false
    test_samples: null
    test_seed: 42

  - name: mmlu_pro
    enabled: false
    num_fewshot: 5
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: bbh
    enabled: false
    num_fewshot: 3
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: gsm8k_cot
    enabled: false
    num_fewshot: 4
    metric: exact_match
    cot: true
    test_samples: null
    test_seed: 42

  - name: minerva_math
    enabled: false
    num_fewshot: 4
    metric: exact_match
    cot: false
    test_samples: null
    test_seed: 42
```

---

# 4. 四个新增任务的 pinned lm-eval 语义

固定 revision：

```text
6d2abda4fd171e68a8789330c4149e37c1ca0bda
```

官方 task 定义位置：

```text
lm_eval/tasks/arc/arc_challenge.yaml
lm_eval/tasks/arc/arc_easy.yaml
lm_eval/tasks/piqa/piqa.yaml
lm_eval/tasks/winogrande/default.yaml
lm_eval/tasks/super_glue/boolq/default.yaml
```

四个新任务统一：

```text
num_fewshot = 0
cot = false
```

具体 canonical metric：

```text
arc_challenge -> acc_norm
piqa          -> acc_norm
winogrande    -> acc
boolq         -> acc
```

ARC-Challenge 和 PIQA 官方都同时报告 `acc` 与 `acc_norm`，本项目统一选择 `acc_norm` 作为 canonical metric。

BoolQ 必须使用标准 task：

```text
task: boolq
output_type: multiple_choice
```

不要使用 `boolq-seq2seq` 或其他 BoolQ variant。

---

# 5. 修改 `snn2/lm_eval_protocol.py`

文件：

```text
snn2/lm_eval_protocol.py
```

当前存在项目侧 audited whitelist：

```python
LM_EVAL_0_4_8_TASK_COT
LM_EVAL_0_4_8_TASK_METRIC
```

必须扩展到 10 个 task。

## 5.1 `LM_EVAL_0_4_8_TASK_COT`

最终应等价于：

```python
LM_EVAL_0_4_8_TASK_COT = {
    "truthfulqa_mc1": False,
    "mmlu_pro": True,
    "bbh": True,
    "agieval": False,
    "gsm8k_cot": True,
    "minerva_math": False,
    "arc_challenge": False,
    "piqa": False,
    "winogrande": False,
    "boolq": False,
}
```

不要删除旧四个 disabled benchmark 的映射。`validate_lm_eval_task_specs()` 会验证所有 entry，包括 `enabled: false` 的 entry。

## 5.2 `LM_EVAL_0_4_8_TASK_METRIC`

最终应等价于：

```python
LM_EVAL_0_4_8_TASK_METRIC = {
    "truthfulqa_mc1": "acc",
    "mmlu_pro": "exact_match",
    "bbh": "exact_match",
    "agieval": "acc",
    "gsm8k_cot": "exact_match",
    "minerva_math": "exact_match",
    "arc_challenge": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
}
```

## 5.3 保持现有验证语义

继续保持：

- revision 必须等于 pinned revision；
- `enabled` 必须 bool；
- `cot` 必须 bool；
- `num_fewshot` 必须非负整数；
- `metric` 必须与 audited canonical metric 一致；
- `test_samples` 只能是正整数或 `null`；
- `test_seed` 必须整数；
- CoT 必须与 audited semantic 一致；
- 至少一个 task enabled。

不要为了新增四个任务放宽 validation。

---

# 6. `scripts/evaluate_lm_harness.py`

当前脚本已经通过：

```python
enabled_lm_eval_task_specs(cfg)
```

动态遍历 enabled tasks，并通过：

```python
TaskManager().load_task_or_group([name])
```

加载 task。

因此不需要为四个新增任务增加 task-specific branch。

禁止添加类似：

```python
if name == "arc_challenge":
    ...
elif name == "piqa":
    ...
```

继续保持：

```python
simple_evaluate(
    ...,
    tasks=[spec["name"]],
    num_fewshot=int(spec["num_fewshot"]),
    limit=None,
    ...
)
```

并保持现有：

- per-task deterministic test selection；
- `test_samples: null` = full evaluation population；
- result metric existence check；
- per-task timing；
- per-task execution counter；
- provenance metadata；
- task-specific result directory。

---

# 7. `scripts/verify_artifacts.py`

当前 verifier 已通过：

```python
for spec in enabled_lm_eval_task_specs(cfg):
```

动态验证所有 enabled Tulu tasks。

因此不需要硬编码四个新增 task 的 verifier 分支。

必须保持：

- ANN 对所有 enabled task 验证；
- Phase/GIF/MTN 对所有 enabled task 验证；
- `results.json`；
- `test_selection.json`；
- exact selection replay；
- metric existence；
- `actual_test_samples`；
- `n-samples.effective`；
- ANN/SNN selection provenance。

如果检查当前代码确认没有旧的“六任务固定列表”，则不要改 verifier。

---

# 8. `snn2/config.py`

当前 `validate_config()` 只在：

```python
cfg["experiment"]["task"] == "tulu3"
```

时调用：

```python
validate_lm_eval_task_specs(cfg)
```

因此：

- TL;DR 删除 lm-eval 字段后，不需要补默认 lm-eval 字段；
- Tulu-3 保留完整 lm-eval 字段即可。

本轮原则上不需要修改 `snn2/config.py`。

---

# 9. 更新测试

至少修改：

```text
tests/test_tulu3_lm_eval_protocol.py
```

并补充：

```text
tests/test_generated_configs.py
```

## 9.1 扩展 metric mapping 测试

将 `test_pinned_metric_mapping` 扩展为 10 task：

```python
@pytest.mark.parametrize(
    "task, metric",
    [
        ("truthfulqa_mc1", "acc"),
        ("mmlu_pro", "exact_match"),
        ("bbh", "exact_match"),
        ("agieval", "acc"),
        ("gsm8k_cot", "exact_match"),
        ("minerva_math", "exact_match"),
        ("arc_challenge", "acc_norm"),
        ("piqa", "acc_norm"),
        ("winogrande", "acc"),
        ("boolq", "acc"),
    ],
)
def test_pinned_metric_mapping(task, metric):
    assert LM_EVAL_0_4_8_TASK_METRIC[task] == metric
```

## 9.2 增加 CoT mapping 测试

import：

```python
LM_EVAL_0_4_8_TASK_COT
```

至少覆盖新四项：

```python
@pytest.mark.parametrize(
    "task",
    ["arc_challenge", "piqa", "winogrande", "boolq"],
)
def test_new_fast_tasks_are_non_cot(task):
    assert LM_EVAL_0_4_8_TASK_COT[task] is False
```

## 9.3 canonical metric 冲突测试

ARC-Challenge / PIQA 的项目 canonical metric 是 `acc_norm`，因此至少验证：

```text
arc_challenge: metric=acc -> validation fail
piqa: metric=acc -> validation fail
```

WinoGrande / BoolQ：

```text
metric=acc -> validation pass
```

## 9.4 generated config evaluation 隔离测试

对所有 TL;DR generated configs：

```python
assert "lm_eval_revision" not in cfg["evaluation"]
assert "apply_chat_template" not in cfg["evaluation"]
assert "lm_eval_task_specs" not in cfg["evaluation"]
assert "limit" not in cfg["evaluation"]
```

同时仍必须包含：

```text
max_new_tokens
tldr_input_length
tldr_test_samples
tldr_test_seed
rouge_types
```

对所有 Tulu generated configs：

```python
for key in (
    "max_new_tokens",
    "tldr_input_length",
    "tldr_test_samples",
    "tldr_test_seed",
    "rouge_types",
):
    assert key not in cfg["evaluation"]
```

同时必须包含：

```text
prefix_enabled
batch_size
lm_eval_revision
apply_chat_template
lm_eval_task_specs
```

## 9.5 task 顺序和 enabled 状态

验证 exact names：

```python
expected_names = [
    "truthfulqa_mc1",
    "agieval",
    "arc_challenge",
    "piqa",
    "winogrande",
    "boolq",
    "mmlu_pro",
    "bbh",
    "gsm8k_cot",
    "minerva_math",
]
```

实际 enabled names 必须恰好为：

```python
[
    "truthfulqa_mc1",
    "agieval",
    "arc_challenge",
    "piqa",
    "winogrande",
    "boolq",
]
```

旧四个必须全部 `enabled is False`。

---

# 10. 重新生成 `configs/generated`

修改 matrix 和 protocol 后运行：

```bash
cd /home/wangwenkang/SNN
python scripts/materialize_configs.py
```

该脚本会重新生成 12 个 config 到：

```text
configs/generated/
```

必须确认：

### Qwen3 TL;DR generated configs

不再出现：

```text
lm_eval_revision
apply_chat_template
lm_eval_task_specs
limit
```

### Llama3 Tulu generated configs

不再出现：

```text
tldr_input_length
tldr_test_samples
tldr_test_seed
rouge_types
max_new_tokens
```

并包含完整 10-task `lm_eval_task_specs`。

---

# 11. 最终 Tulu-3 enabled benchmark 集合

| Task | enabled | num_fewshot | canonical metric | cot | test_samples | test_seed |
|---|---|---:|---|---|---|---:|
| `truthfulqa_mc1` | true | 0 | `acc` | false | null | 42 |
| `agieval` | true | 0 | `acc` | false | null | 42 |
| `arc_challenge` | true | 0 | `acc_norm` | false | null | 42 |
| `piqa` | true | 0 | `acc_norm` | false | null | 42 |
| `winogrande` | true | 0 | `acc` | false | null | 42 |
| `boolq` | true | 0 | `acc` | false | null | 42 |

禁用但保留：

| Task | enabled |
|---|---|
| `mmlu_pro` | false |
| `bbh` | false |
| `gsm8k_cot` | false |
| `minerva_math` | false |

---

# 12. 输出路径原则

不要修改当前 Tulu lm-eval output path 结构。

每个 enabled task 继续独立输出到：

```text
.../evaluation/
  prefix_enabled_<...>/
    <task_name>/
      num_fewshot_<N>_cot_<true|false>_test_samples_<full|N>_test_seed_<seed>/
        results.json
        test_selection.json
```

例如：

```text
.../boolq/
  num_fewshot_0_cot_false_test_samples_full_test_seed_42/
```

```text
.../arc_challenge/
  num_fewshot_0_cot_false_test_samples_full_test_seed_42/
```

不要新增 `lm_harness/` 目录层。

---

# 13. 不要修改旧四任务的协议参数

虽然旧四项禁用，但其现有参数保留：

```text
mmlu_pro:
  num_fewshot=5
  metric=exact_match
  cot=true

bbh:
  num_fewshot=3
  metric=exact_match
  cot=true

gsm8k_cot:
  num_fewshot=4
  metric=exact_match
  cot=true

minerva_math:
  num_fewshot=4
  metric=exact_match
  cot=false
```

只改变：

```yaml
enabled: false
```

以及移动到 task list 最后。

---

# 14. 验证步骤

完成后至少执行：

```bash
cd /home/wangwenkang/SNN
python scripts/materialize_configs.py
pytest -q
```

要求 `pytest -q` 全部通过。

建议再做轻量 config smoke check：

```bash
python - <<'PY'
from pathlib import Path
import yaml

root = Path("configs/generated")

for path in sorted(root.glob("*.yaml")):
    cfg = yaml.safe_load(path.read_text())
    ev = cfg["evaluation"]
    task = cfg["experiment"]["task"]

    if task == "tldr":
        assert "lm_eval_revision" not in ev
        assert "apply_chat_template" not in ev
        assert "lm_eval_task_specs" not in ev
        assert "limit" not in ev
        for key in (
            "max_new_tokens",
            "tldr_input_length",
            "tldr_test_samples",
            "tldr_test_seed",
            "rouge_types",
        ):
            assert key in ev

    elif task == "tulu3":
        for key in (
            "max_new_tokens",
            "tldr_input_length",
            "tldr_test_samples",
            "tldr_test_seed",
            "rouge_types",
        ):
            assert key not in ev

        specs = ev["lm_eval_task_specs"]
        assert [x["name"] for x in specs] == [
            "truthfulqa_mc1",
            "agieval",
            "arc_challenge",
            "piqa",
            "winogrande",
            "boolq",
            "mmlu_pro",
            "bbh",
            "gsm8k_cot",
            "minerva_math",
        ]

        assert [x["name"] for x in specs if x["enabled"]] == [
            "truthfulqa_mc1",
            "agieval",
            "arc_challenge",
            "piqa",
            "winogrande",
            "boolq",
        ]

print("evaluation config separation: PASS")
PY
```

---

# 15. 完成标准

本轮修改只有在以下条件全部满足时才算完成：

1. 两个 TL;DR experiment 的 `evaluation:` 中没有任何 lm-eval task 参数。
2. Tulu-3 experiment 的 `evaluation:` 中没有任何 TL;DR 参数。
3. Tulu-3 有 10 个 task spec。
4. 实际 enabled task 恰好为：TruthfulQA MC1、AGIEval、ARC-Challenge、PIQA、WinoGrande、BoolQ。
5. `mmlu_pro`、`bbh`、`gsm8k_cot`、`minerva_math` 全部保留但 disabled。
6. 四个新增 task 的 protocol 为：
   - `arc_challenge`: `0-shot`, `acc_norm`, `cot=false`
   - `piqa`: `0-shot`, `acc_norm`, `cot=false`
   - `winogrande`: `0-shot`, `acc`, `cot=false`
   - `boolq`: `0-shot`, `acc`, `cot=false`
7. 四个新增 task 都是 `enabled: true`, `test_samples: null`, `test_seed: 42`。
8. pinned lm-eval revision 不变。
9. `LM_EVAL_0_4_8_TASK_COT` 与 `LM_EVAL_0_4_8_TASK_METRIC` 已扩展到 10 task。
10. evaluator/verifier 继续动态处理 task spec，不增加 task-specific 分支。
11. `configs/generated` 已重新生成。
12. `pytest -q` 全部通过。
13. 除本轮明确要求外，没有改变 TL;DR、Tulu 数据、训练、rotation、Prefix、calibration、ANN/SNN conversion 与输出路径语义。
