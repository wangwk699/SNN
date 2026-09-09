# Llama3-8B / Tulu3 lm-eval ANN/SNN Batch Size 拆分修改方案

## 1. 修改目标

仅对：

```text
exp2_llama3_8b_tulu3
```

对应的 Tulu3 / lm-eval 实验，将 evaluation batch size 拆分为：

```yaml
evaluation:
  batch_size: 8
  snn_batch_size: 1
```

语义固定为：

```text
Base ANN evaluation                → batch_size = 8
Rotated pre-finetuning ANN eval   → batch_size = 8
Final ANN evaluation              → batch_size = 8

Phase SNN evaluation              → snn_batch_size = 1
GIF SNN evaluation                → snn_batch_size = 1
MTN SNN evaluation                → snn_batch_size = 1
```

本轮必须保证：

```text
Qwen3-1.7B / TL;DR
Qwen3-8B   / TL;DR
```

的 evaluation 配置和实际执行逻辑保持原样。

---

# 2. 修改文件

预计主要修改：

```text
configs/experiment_matrix.yaml
snn2/config.py
snn2/evaluation.py
scripts/evaluate_lm_harness.py
tests/test_generated_configs.py
```

如现有测试组织更合适，也可以在已有 lm-eval 测试文件中增加测试。

不要修改：

```text
snn2/model_integration.py
snn2/temporal_model.py
snn2/temporal_ops.py
snn2/controller.py
snn2/neurons.py
snn2/prefix_cache.py
snn2/conversion.py
```

本轮不改变任何 SNN 数学计算。

---

# 3. `configs/experiment_matrix.yaml`

只修改：

```text
exp2_llama3_8b_tulu3
```

的 base configuration。

当前：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8
  lm_eval_revision: ...
```

修改为：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8
  snn_batch_size: 1
  lm_eval_revision: ...
```

因为四个：

```text
vanilla
unaware
phase_aware
gif_aware
```

都是从 `exp2_llama3_8b_tulu3_base` 派生，因此四份 generated Tulu3 config 都应得到：

```yaml
evaluation:
  batch_size: 8
  snn_batch_size: 1
```

## 明确禁止修改 TL;DR

以下配置不要增加：

```yaml
snn_batch_size:
```

即 Qwen3 TL;DR 仍然保持例如：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8
  max_new_tokens: 32
  tldr_input_length: 512
  ...
```

不要变成：

```yaml
evaluation:
  batch_size: 8
  snn_batch_size: 1
```

---

# 4. `snn2/config.py`

对 Tulu3 配置增加 batch size 合法性校验。

在：

```python
elif cfg["experiment"].get("task") == "tulu3":
```

分支中增加：

```python
evaluation = cfg["evaluation"]

batch_size = evaluation.get("batch_size")
if (
    not isinstance(batch_size, int)
    or isinstance(batch_size, bool)
    or batch_size <= 0
):
    raise ValueError(
        "evaluation.batch_size must be a positive integer"
    )

snn_batch_size = evaluation.get("snn_batch_size")
if (
    not isinstance(snn_batch_size, int)
    or isinstance(snn_batch_size, bool)
    or snn_batch_size <= 0
):
    raise ValueError(
        "Tulu-3 evaluation.snn_batch_size must be a positive integer"
    )
```

这里 Tulu3 的：

```yaml
snn_batch_size
```

应作为**必需字段**。

不要使用：

```python
evaluation.get("snn_batch_size", 1)
```

静默默认。

原因是该值直接关系到是否发生 SNN OOM，generated config 应明确记录它。

---

# 5. TL;DR validation 不要增加新要求

不要要求 Qwen TL;DR：

```yaml
evaluation:
  snn_batch_size: ...
```

也不要改变 TL;DR 当前：

```python
evaluation["batch_size"]
```

语义。

因此：

```text
task == "tldr"
```

继续完全按照现有规则工作。

本轮不需要专门禁止 TL;DR 出现 `snn_batch_size`，只需保证 experiment matrix 中没有添加即可，避免无关的 strict-validation 改动。

---

# 6. 在 `snn2/evaluation.py` 增加统一 batch size resolver

推荐新增一个很小、容易测试的 helper：

```python
def lm_eval_batch_size(
    cfg: dict[str, object],
    *,
    neuron: str,
) -> int:
    evaluation = cfg["evaluation"]

    if (
        cfg["experiment"]["task"] == "tulu3"
        and neuron != "ann"
    ):
        return int(
            evaluation["snn_batch_size"]
        )

    return int(
        evaluation["batch_size"]
    )
```

语义：

```text
Tulu3 + ann   → batch_size
Tulu3 + phase → snn_batch_size
Tulu3 + gif   → snn_batch_size
Tulu3 + mtn   → snn_batch_size

TL;DR + ann   → batch_size
TL;DR + phase → batch_size
TL;DR + gif   → batch_size
TL;DR + mtn   → batch_size
```

也就是说：

> `snn_batch_size` 这个新字段严格限定在 Tulu3 lm-eval 上。

---

# 7. 为什么判断 `neuron != "ann"` 就够了

当前：

```text
--base
```

只允许：

```text
--neuron ann
```

而：

```text
--rotated-pre-finetuning
```

同样只允许：

```text
--neuron ann
```

因此 resolver 不需要再传：

```text
base
rotated_pre_finetuning
```

只看：

```python
neuron
```

即可：

```text
ann   → 8
phase → 1
gif   → 1
mtn   → 1
```

---

# 8. `scripts/evaluate_lm_harness.py`

当前代码类似：

```python
batch_size = int(
    cfg["evaluation"].get(
        "batch_size",
        1,
    )
)
```

改为调用统一 resolver。

首先 import：

```python
from snn2.evaluation import (
    ...
    lm_eval_batch_size,
)
```

然后：

```python
batch_size = lm_eval_batch_size(
    cfg,
    neuron=args.neuron,
)
```

后续保持：

```python
harness_model = DistributedPreinitializedHFLM(
    accelerator=accelerator,
    pretrained=proxy,
    tokenizer=tokenizer,
    batch_size=batch_size,
    max_length=int(
        cfg["data"]["max_seq_length"]
    ),
)
```

以及：

```python
simple_evaluate(
    ...
    batch_size=batch_size,
)
```

全部使用同一个已经解析好的 runtime：

```python
batch_size
```

---

# 9. 运行时实际行为

修改后：

## Final ANN

例如：

```bash
accelerate launch \
  scripts/evaluate_lm_harness.py \
  --config "$CFG" \
  --neuron ann
```

解析：

```python
batch_size = 8
```

每张 GPU：

```text
logical batch = 8
model batch   = 8
```

---

## Phase SNN

```bash
accelerate launch \
  scripts/evaluate_lm_harness.py \
  --config "$CFG" \
  --neuron phase
```

解析：

```python
batch_size = 1
```

Phase：

```text
T = 4
```

所以：

```text
logical batch = 1
temporal model batch = 4 × 1 = 4
```

不再是原来的：

```text
8 × 4 = 32
```

---

## MTN SNN

同理：

```text
logical batch = 1
T = 4
temporal model batch = 4
```

---

## GIF SNN

GIF full temporal execution按照当前 controller 实际 deployment steps 工作。

lm-eval 外层仍然：

```text
logical batch = 1
```

因此 GIF 也不会再以逻辑 batch 8 进入 SNN deployment。

---

# 10. 多 GPU document data parallel 不变

例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2
accelerate launch --num_processes 3 ...
```

仍然是：

```text
rank0 / GPU0 → 一份完整 Llama3-8B
rank1 / GPU1 → 一份完整 Llama3-8B
rank2 / GPU2 → 一份完整 Llama3-8B
```

每个 rank 使用：

```text
SNN logical batch = 1
```

因此三个 rank 同时处理：

```text
3 个 logical samples
```

document sharding：

```python
indices_for_rank(...)
```

保持不变。

不要把本轮修改成 model parallel。

---

# 11. Metadata 保持现有语义

当前 `common_snn2_metadata` 中已有类似：

```python
"batch_size": batch_size,
"evaluation_batch_size_per_rank": batch_size,
```

这些字段继续使用**实际 runtime batch size**。

因此：

### ANN result

```json
{
  "batch_size": 8,
  "evaluation_batch_size_per_rank": 8
}
```

### SNN result

```json
{
  "batch_size": 1,
  "evaluation_batch_size_per_rank": 1
}
```

这是正确行为。

本轮不需要新增：

```text
configured_ann_batch_size
configured_snn_batch_size
```

避免扩大 metadata schema。

---

# 12. Artifact 路径不要改变

不要把 batch size 添加到：

```text
evaluation/
```

路径。

也不要出现：

```text
batch_size_8/
snn_batch_size_1/
```

新的目录层。

原因是 batch size 是纯 evaluation throughput / memory 参数：

```text
不改变选中的样本
不改变 model
不改变 neuron
不改变 metric
不改变结果数学定义
```

因此现有路径继续保持。

---

# 13. 与上一轮 incremental persistence 修改兼容

上一轮方案还会将：

```python
simple_evaluate(
    ...
    log_samples=False,
)
```

以及逐 task 保存结果。

这和本轮 batch size 拆分完全独立。

最终 `simple_evaluate()` 应类似：

```python
task_result = simple_evaluate(
    model=harness_model,
    tasks=[spec["name"]],
    task_manager=task_manager,
    num_fewshot=int(
        spec["num_fewshot"]
    ),
    batch_size=batch_size,
    limit=None,
    ...
    log_samples=False,
)
```

其中：

```text
ANN:
batch_size = 8

SNN:
batch_size = 1
```

不要把：

```python
log_samples=False
```

和：

```python
snn_batch_size
```

混成同一个功能开关。

---

# 14. `tests/test_generated_configs.py`

当前 task-specific evaluation test 需要更新。

现有 Tulu3 大致要求：

```python
{
    "prefix_enabled",
    "batch_size",
    "lm_eval_revision",
    "apply_chat_template",
    "lm_eval_task_specs",
} <= set(evaluation)
```

修改为：

```python
{
    "prefix_enabled",
    "batch_size",
    "snn_batch_size",
    "lm_eval_revision",
    "apply_chat_template",
    "lm_eval_task_specs",
} <= set(evaluation)
```

同时明确：

```python
assert evaluation["batch_size"] == 8
assert evaluation["snn_batch_size"] == 1
```

---

# 15. 明确增加 TL;DR 不变回归测试

在：

```python
test_generated_evaluation_configs_are_task_specific
```

或新增独立测试中明确：

```python
if cfg["experiment"]["task"] == "tldr":
    assert "snn_batch_size" not in evaluation
```

同时保留：

```python
assert evaluation["batch_size"] == 8
```

如果当前所有 Qwen TL;DR config 的 batch size 都确实为 8。

这样以后不会误把 Tulu3 的新字段扩散给 Qwen。

---

# 16. 增加 resolver 单元测试

针对：

```python
lm_eval_batch_size(...)
```

至少测试：

```python
@pytest.mark.parametrize(
    ("neuron", "expected"),
    [
        ("ann", 8),
        ("phase", 1),
        ("gif", 1),
        ("mtn", 1),
    ],
)
def test_tulu3_lm_eval_batch_size(...):
    ...
```

---

# 17. 增加 TL;DR resolver 回归

对于 Qwen TL;DR：

```python
@pytest.mark.parametrize(
    "neuron",
    ["ann", "phase", "gif", "mtn"],
)
def test_tldr_uses_existing_batch_size_for_all_neurons(...):
    assert (
        lm_eval_batch_size(
            cfg,
            neuron=neuron,
        )
        == cfg["evaluation"]["batch_size"]
    )
```

这是本轮非常重要的隔离测试。

---

# 18. 增加配置错误测试

Tulu3：

```yaml
snn_batch_size: 0
```

应拒绝。

测试：

```python
@pytest.mark.parametrize(
    "value",
    [0, -1, 1.5, True, "1", None],
)
def test_tulu3_snn_batch_size_must_be_positive_integer(...):
    ...
```

以及：

```yaml
batch_size: 0
```

也应拒绝。

---

# 19. Generated config 验收

运行：

```bash
python scripts/materialize_configs.py
```

然后确认四份：

```text
configs/generated/exp2_llama3_8b_tulu3__vanilla.yaml
configs/generated/exp2_llama3_8b_tulu3__unaware.yaml
configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml
configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml
```

全部包含：

```yaml
evaluation:
  batch_size: 8
  snn_batch_size: 1
```

同时所有：

```text
exp1_qwen3_1_7b_tldr__*.yaml
exp1_qwen3_8b_tldr__*.yaml
```

均：

```text
不存在 snn_batch_size
```

---

# 20. 测试命令

至少运行：

```bash
pytest -q tests/test_generated_configs.py
```

如果 resolver 测试放在其它文件：

```bash
pytest -q tests/test_evaluation_paths.py
```

或对应新增测试文件。

最后：

```bash
pytest -q
```

必须全部通过。

---

# 21. GPU smoke test

先不要直接重新跑整个 sweep。

建议先用一个 Llama3/Tulu3 SNN config 做最小实际验证，例如 Phase：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

accelerate launch \
  --num_processes 3 \
  scripts/evaluate_lm_harness.py \
  --config configs/generated/exp2_llama3_8b_tulu3__vanilla.yaml \
  --neuron phase
```

具体必须使用与你当前 conversion artifacts 匹配的配置/参数。

启动日志或 metadata 应确认：

```text
evaluation_batch_size_per_rank = 1
```

而不是 8。

---

# 22. OOM 验收

之前错误发生于：

```text
Running loglikelihood requests: 0/1205
```

并尝试一次分配：

```text
12.38 GiB
```

修改后必须至少满足：

```text
成功越过第一个最长 loglikelihood batch
```

不能只以：

```text
模型成功加载
```

作为验收。

最好跑过多个 batch，确认：

```text
0/1205
→ 1/1205
→ 2/1205
→ ...
```

显存稳定。

---

# 23. 不要顺手做的修改

本轮不要同时做以下优化：

```text
temporal attention streaming
LM Head temporal aggregation
FlashAttention
SNN selective checkpoint
CPU offload
model parallel
auto batch size
动态按 T 调 batch size
```

这些都属于后续独立优化。

当前先采用最简单、低风险的：

```text
Tulu ANN = 8
Tulu SNN = 1
```

验证稳定性。

---

# 24. 最终配置语义

## Qwen3 / TL;DR

保持：

```yaml
evaluation:
  batch_size: 8
```

所有 ANN/SNN 继续按照原代码的 `batch_size`。

---

## Llama3-8B / Tulu3

变为：

```yaml
evaluation:
  prefix_enabled: true
  batch_size: 8
  snn_batch_size: 1
  lm_eval_revision: ...
  apply_chat_template: true
  lm_eval_task_specs:
    ...
```

运行：

```text
Base ANN                         8
Rotated pre-finetuning ANN      8
Final ANN                       8
Phase SNN                       1
GIF SNN                         1
MTN SNN                         1
```

---

# 25. 最终验收清单

- [ ] 只有 Llama3/Tulu3 增加 `evaluation.snn_batch_size`
- [ ] Tulu3 `batch_size == 8`
- [ ] Tulu3 `snn_batch_size == 1`
- [ ] Qwen3/TL;DR YAML 完全不增加 `snn_batch_size`
- [ ] Tulu3 ANN evaluation 使用 8
- [ ] Tulu3 Base ANN 使用 8
- [ ] Tulu3 rotated pre-finetuning ANN 使用 8
- [ ] Tulu3 Phase SNN 使用 1
- [ ] Tulu3 GIF SNN 使用 1
- [ ] Tulu3 MTN SNN 使用 1
- [ ] SNN temporal `T` 和 neuron 数学逻辑完全不改
- [ ] distributed document sharding 完全不改
- [ ] evaluation output path 完全不改
- [ ] metadata 中 runtime `batch_size` 正确反映 8 或 1
- [ ] Tulu3 `snn_batch_size <= 0` 被 config validation 拒绝
- [ ] Qwen TL;DR regression tests 证明执行语义不变
- [ ] generated configs 测试通过
- [ ] `pytest -q` 全部通过
- [ ] 3×GPU SNN smoke test 成功越过之前 OOM 的第一个长 batch

## 核心原则

本轮只做：

```text
Llama3/Tulu3 ANN throughput setting
             ↓
        batch_size = 8

Llama3/Tulu3 SNN memory-safe setting
             ↓
    snn_batch_size = 1
```

而：

```text
Qwen3 / TL;DR
```

完全保持当前行为。