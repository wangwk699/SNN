# Tulu-3 / lm-eval 多卡数据并行评估修改方案

## 0. 修改目标

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

目标是让 `scripts/evaluate_lm_harness.py` 真正支持通过：

```bash
accelerate launch --num_processes N
```

进行 **Tulu-3 / lm-eval benchmark 的多 GPU 数据并行评估**。

必须覆盖：

- Base ANN
- Final ANN
- Phase SNN
- GIF SNN
- MTN SNN

本轮修改 **只针对 Tulu-3 / lm-eval**。

### 绝对禁止修改

不要修改：

```text
scripts/evaluate_tldr.py
```

不要修改 TL;DR summarization 的数据选择、batch、generation、ROUGE、Prefix、ANN/SNN forward、多卡逻辑、输出路径和 artifact provenance。

也不要为了本轮功能修改 training、rotation、calibration、Phase/GIF/MTN 数学、conversion、Prefix KV 语义、Tulu benchmark selection 语义、lm-eval pinned revision，以及 benchmark metric / cot / few-shot 定义。

---

## 1. 当前问题

当前命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG_L_V" \
  --neuron ann \
  --base
```

会启动 4 个进程，并且每个进程都把完整模型加载到自己的 GPU，这一点本身没有问题。

真正的问题在：

```python
proxy = EvaluationModelProxy(...)
harness_model = HFLM(
    pretrained=proxy,
    ...
)
```

项目传给 `HFLM` 的是一个 **已经初始化的 PyTorch model/proxy object**，而不是 model name 字符串。

固定的 lm-eval revision：

```text
6d2abda4fd171e68a8789330c4149e37c1ca0bda
```

中，`HFLM` 对这种 pre-initialized model 明确走单进程逻辑：

```python
self._rank = 0
self._world_size = 1
```

所以虽然外部 accelerate 启动了 4 个 process，lm-eval 内部却在每个进程里都认为：

```text
lm.rank = 0
lm.world_size = 1
```

因此每个进程都会构建整个 benchmark 的 requests。

---

## 2. 正确的并行方案

正确设计应为：

```text
rank 0 -> GPU 0 -> 一份完整 Base/ANN/SNN model -> evaluation shard 0
rank 1 -> GPU 1 -> 一份完整 Base/ANN/SNN model -> evaluation shard 1
rank 2 -> GPU 2 -> 一份完整 Base/ANN/SNN model -> evaluation shard 2
rank 3 -> GPU 3 -> 一份完整 Base/ANN/SNN model -> evaluation shard 3
```

即：

> one full model replica per process + lm-eval request/document data parallelism

不要使用 `DistributedDataParallel(model)`，也不要使用 `accelerator.prepare(model)`。

本任务没有训练梯度，不需要 parameter synchronization。Phase/GIF/MTN temporal forward 和 Prefix KV 都绑定在每个进程自己的 model/controller 上，复制模型并分 evaluation documents 是最安全的方案。

---

## 3. 为什么只需要让 HFLM 正确认识 rank/world_size

固定 lm-eval 中，`evaluate()` 构建 task requests 时会使用：

```python
task.build_all_requests(
    ...,
    rank=lm.rank,
    world_size=lm.world_size,
    ...
)
```

本项目 `_selected_lm_eval_tasks()` 已经把 leaf task 的 `doc_iterator` 包装成基于 `rank/world_size` 的分片逻辑。

所以本轮不要重写 benchmark sampling / selection。

只需要：

1. 正确初始化 Accelerate distributed context；
2. 把实际 rank/world_size 注入项目使用的 preinitialized HFLM；
3. 处理 distributed `simple_evaluate()` 非主 rank 返回 `None`；
4. 聚合 execution counters；
5. 保证每个 task 之间所有 rank 同步；
6. 只由 global rank 0 写 artifact。

---

## 4. 推荐新增模块

新增：

```text
snn2/lm_eval_distributed.py
```

不要把这套逻辑塞进 `snn2/evaluation.py`，因为后者同时被 TL;DR 使用。

新模块只服务 `evaluate_lm_harness.py`。

---

## 5. 新增 DistributedPreinitializedHFLM

在 `snn2/lm_eval_distributed.py` 实现项目侧 wrapper，例如：

```python
from __future__ import annotations

from accelerate import Accelerator
from lm_eval.models.huggingface import HFLM


class DistributedPreinitializedHFLM(HFLM):
    """HFLM adapter for an already-instantiated SNN2 model proxy."""

    def __init__(self, *, accelerator: Accelerator, **kwargs):
        super().__init__(**kwargs)

        self.accelerator = accelerator
        self._rank = int(accelerator.process_index)
        self._world_size = int(accelerator.num_processes)
        self._device = accelerator.device
```

必须使用：

```python
accelerator.process_index
```

作为全局 rank，不要用 `local_process_index` 覆盖 lm-eval global rank。

---

## 6. 不要调用 accelerator.prepare(model)

`Accelerator` 在这里只负责：

- process-group initialization
- rank/world_size
- local device
- barriers

不要负责 DDP wrapping。

---

## 7. 修改 scripts/evaluate_lm_harness.py

在 config/setup 完成后、model load 之前创建：

```python
from accelerate import Accelerator

accelerator = Accelerator()

rank = int(accelerator.process_index)
world_size = int(accelerator.num_processes)
device = accelerator.device
```

CUDA 下：

```python
if device.type == "cuda":
    torch.cuda.set_device(device)
```

删除当前脚本里自己根据 `LOCAL_RANK` 重新构造 device 的重复逻辑。

之后整个 `evaluate_lm_harness.py` 统一使用：

```python
rank
world_size
device
accelerator
```

建议加防御性检查：

```python
env_rank = int(os.environ.get("RANK", rank))
if env_rank != rank:
    raise RuntimeError(...)
```

---

## 8. 每个 rank 加载完整模型到自己的 GPU

当前：

```python
model = load_model(
    cfg,
    source,
    training=False,
)

model.to(device)
model.eval()
```

继续保留。

不要使用 `device_map=auto` 去把一份模型跨多卡切分。

本轮需要的是 data-parallel replication，而不是 model parallelism。

---

## 9. 使用 DistributedPreinitializedHFLM

当前：

```python
harness_model = HFLM(
    pretrained=proxy,
    tokenizer=tokenizer,
    batch_size=batch_size,
    max_length=int(cfg["data"]["max_seq_length"]),
)
```

改为：

```python
harness_model = DistributedPreinitializedHFLM(
    accelerator=accelerator,
    pretrained=proxy,
    tokenizer=tokenizer,
    batch_size=batch_size,
    max_length=int(cfg["data"]["max_seq_length"]),
)
```

之后立即检查：

```python
if harness_model.rank != rank:
    raise RuntimeError(...)

if harness_model.world_size != world_size:
    raise RuntimeError(...)
```

4 卡时必须满足：

```text
rank0: HFLM rank=0 world_size=4
rank1: HFLM rank=1 world_size=4
rank2: HFLM rank=2 world_size=4
rank3: HFLM rank=3 world_size=4
```

---

## 10. `_selected_lm_eval_tasks()` 不要改 sampling 语义

保留现有：

```python
selection = build_test_selection(
    {leaf_name: len(task.eval_docs) for leaf_name, task in leaves},
    task=name,
    test_samples=spec["test_samples"],
    test_seed=int(spec["test_seed"]),
)
```

所有 rank 都必须得到完全相同的一套全局 `test_selection`。

之后只用 rank/world_size 把 selected docs 互斥分片。

禁止改成每个 rank 独立 `random.sample()`。

---

## 11. 建议提取纯 partition helper

建议增加：

```python
def indices_for_rank(indices, *, rank: int, world_size: int):
    if world_size <= 0:
        raise ValueError(...)
    if rank < 0 or rank >= world_size:
        raise ValueError(...)
    return list(indices)[rank::world_size]
```

`doc_iterator` 基于该 helper。

这样可独立验证：

- 各 rank 无 overlap；
- union 等于 selected set；
- world_size=1 时行为不变。

---

## 12. simple_evaluate 返回值必须按 rank 处理

真正启用 distributed rank/world_size 后，固定 lm-eval 中 `simple_evaluate()` 只会在 `lm.rank == 0` 返回完整 result；非 rank 0 返回 `None`。

所以当前：

```python
task_result = simple_evaluate(...)

correct_effective_sample_counts(task_result, test_selection)

if not result_contains_metric(task_result, spec["metric"]):
    ...
```

会让 rank 1/2/3 报错。

必须改为：

```python
task_result = simple_evaluate(...)

if rank == 0:
    if task_result is None:
        raise RuntimeError(...)

    correct_effective_sample_counts(
        task_result,
        test_selection,
    )

    if not result_contains_metric(
        task_result,
        spec["metric"],
    ):
        raise ValueError(...)
else:
    if task_result is not None:
        raise RuntimeError(...)
```

---

## 13. execution counter 必须跨 rank 求和

当前 task-local counter 只是本 rank 局部执行量。

建议在 `snn2/lm_eval_distributed.py` 增加：

```python
def sum_execution_counters(counters):
    result = {}
    for counter in counters:
        for key, value in counter.items():
            result[key] = result.get(key, 0) + int(value)
    return result
```

以及：

```python
import torch.distributed as dist


def gather_sum_execution_counter(
    local_counter,
    *,
    world_size,
):
    if world_size == 1:
        return dict(local_counter)

    gathered = [None] * world_size

    dist.all_gather_object(
        gathered,
        dict(local_counter),
    )

    return sum_execution_counters(gathered)
```

最终保存到 `results.json` 的 `execution_counter` 使用 global sum。

---

## 14. timing 必须记录真实多卡 wall time

多个 rank 的耗时不应相加。

数据并行 wall time 应取：

```text
max(per-rank elapsed time)
```

建议：

```python
def distributed_max_seconds(local_seconds, *, device, world_size):
    if world_size == 1:
        return float(local_seconds)

    tensor = torch.tensor(
        [float(local_seconds)],
        dtype=torch.float64,
        device=device,
    )

    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MAX,
    )

    return float(tensor.item())
```

至少对 `lm_eval_seconds` 和 `total_seconds` 使用 global max。

`evaluation_timing.json` 中 `total_lm_eval_seconds` 继续等于各 task global `lm_eval_seconds` 求和。

---

## 15. 每个 task 结束后必须 barrier

每个 task 推荐执行顺序：

```text
1. all ranks build identical global selection
2. simple_evaluate
3. all ranks finish lm-eval internal gather
4. collect task-local execution counters
5. reduce timing
6. rank 0:
     - effective counts
     - metric validation
     - store task_results
7. accelerator.wait_for_everyone()
8. next task
```

这个 barrier 要保留，避免某些 rank 过早进入下一个 task 的 distributed collective。

---

## 16. rank 0 写 artifact，其它 rank 禁止写

继续保持只有 global rank 0 写：

```text
results.json
test_selection.json
evaluation_timing.json
```

推荐统一使用：

```python
if accelerator.is_main_process:
```

所有 worker rank 可以保留自己的 StageRun rank log，但不能覆盖 evaluation result。

---

## 17. 新增 distributed provenance

建议在 `common_snn2_metadata` 增加：

```python
"evaluation_parallelism": (
    "lm_eval_document_data_parallel"
    if world_size > 1
    else "single_process"
),
"evaluation_world_size": int(world_size),
"evaluation_batch_size_per_rank": int(batch_size),
"evaluation_model_replication": "one_full_model_per_process",
```

注意：

```text
evaluation.batch_size=8
```

在多卡下表示每个 rank / 每张 GPU 的 batch size 是 8。

---

## 18. scripts/verify_artifacts.py

不要为 Base / ANN / Phase / GIF / MTN 新增五套 verifier。

保持现有动态 Tulu task verifier。

只建议增加轻量 provenance validation：

- `evaluation_world_size` 是 >=1 的整数；
- 如果 `evaluation_world_size > 1`：
  - `evaluation_parallelism == "lm_eval_document_data_parallel"`
  - `evaluation_model_replication == "one_full_model_per_process"`

不要固定 `world_size == 4`，以后 1/2/4/8 GPU 都应合法。

---

## 19. scripts/_common.py 不需要修改

当前 `setup()` 已经只在 `RANK=0` 写 resolved config，本轮不要修改 `_common.py`。

---

## 20. snn2/evaluation.py 不要加入 distributed 逻辑

`EvaluationModelProxy`、`temporal_forward`、`greedy_generate` 的数学执行都不需要知道 rank/world_size。

多卡分片属于 lm-eval orchestration，不属于 neuron forward。

因此不要把 `torch.distributed` / `Accelerator` / rank/world_size 塞进 Phase/GIF/MTN forward，避免影响 TL;DR。

---

## 21. 测试

建议新建：

```text
tests/test_lm_eval_distributed.py
```

至少覆盖：

### 21.1 分片完整性

```python
def test_indices_for_rank_partition_is_disjoint_and_complete():
    indices = list(range(17))

    shards = [
        indices_for_rank(indices, rank=r, world_size=4)
        for r in range(4)
    ]

    flattened = [x for shard in shards for x in shard]

    assert sorted(flattened) == indices
    assert len(flattened) == len(set(flattened))
```

并测试：

```python
indices_for_rank(indices, rank=0, world_size=1) == indices
```

### 21.2 HFLM rank/world binding

用 fake accelerator 验证：

```text
process_index=2
num_processes=4
```

最终：

```text
rank=2
world_size=4
```

### 21.3 execution counter merge

验证多个局部 counter 求和正确。

### 21.4 单进程回归

world_size=1 时：

- selected docs 不变；
- counter 不变；
- timing 不变；
- output path 不变。

---

## 22. 真实 distributed smoke test

单元测试通过后，先不要跑完整六任务。

复制一个临时 Tulu config，只启用：

```text
truthfulqa_mc1
```

并设置：

```yaml
test_samples: 64
```

其它 benchmark 临时 `enabled: false`。

不要修改正式 matrix。

---

## 23. 单卡 smoke test

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  scripts/evaluate_lm_harness.py \
  --config /tmp/tulu_multigpu_smoke.yaml \
  --neuron ann \
  --base
```

保存 `results.json` 与 `test_selection.json`。

---

## 24. 四卡 smoke test

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py \
  --config /tmp/tulu_multigpu_smoke.yaml \
  --neuron ann \
  --base
```

要求：

- 不再每个 rank 都完整跑 64 docs；
- 每个 rank 约 16 docs；
- 各 rank candidate loglikelihood request 数约为总量 1/4；
- `test_selection.json` 与单卡完全一致；
- `actual_test_samples == 64`，而不是 256；
- TruthfulQA `acc` 与单卡一致。

---

## 25. Base 通过后再测试 Final ANN / Phase / GIF / MTN

用小样本临时 config 分别验证：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py --config "$CFG_L_V" --neuron ann
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py --config "$CFG_L_P" --neuron phase
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py --config "$CFG_L_G" --neuron gif
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py --config "$CFG_L_P" --neuron mtn
```

确认每个 rank 都是：

```text
one local model
one local controller
one local Prefix KV state
one disjoint evaluation shard
```

---

## 26. 正式运行命令

修改完成后，以下命令应真正成为四卡数据并行。

### Base

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG_L_V" \
  --neuron ann \
  --base
```

### Final ANN

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG_L_V" \
  --neuron ann
```

### Phase SNN

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG_L_P" \
  --neuron phase
```

### GIF SNN

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG_L_G" \
  --neuron gif
```

### MTN SNN

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG_L_P" \
  --neuron mtn
```

---

## 27. 实验执行总结.md

只更新 Tulu-3 / lm-eval 相关命令和说明。

明确：

```text
evaluation.batch_size
```

在多卡下是 per-rank / per-GPU batch size。

不要修改文档中的 TL;DR 命令。

---

## 28. configs/experiment_matrix.yaml 不需要修改

本轮多卡支持不需要新增：

```text
evaluation.world_size
evaluation.num_gpus
```

GPU 数量由运行命令决定：

```bash
CUDA_VISIBLE_DEVICES=...
accelerate launch --num_processes ...
```

正式 benchmark config 与 GPU 数保持解耦。

---

## 29. 完整验证

修改后执行：

```bash
cd /home/wangwenkang/SNN
pytest -q
```

必须全部通过。

之后执行真实 1 GPU vs 4 GPU TruthfulQA 小样本 regression。

---

## 30. 最终验收标准

1. `scripts/evaluate_tldr.py` 未修改。
2. TL;DR generated config / path / evaluation 行为未修改。
3. `evaluate_lm_harness.py` 使用 `Accelerator()` 获取真实 global rank/world_size/device。
4. 每个 rank 加载一份完整 Base/ANN/SNN model 到自己的 GPU。
5. 不使用 DDP。
6. 不使用 `accelerator.prepare(model)`。
7. preinitialized HFLM 使用真实 `accelerator.process_index` 和 `accelerator.num_processes`。
8. lm-eval `build_all_requests()` 收到真实 rank/world_size。
9. 全局 `test_selection` 语义保持不变。
10. 各 rank evaluation docs 无 overlap。
11. 各 rank evaluation docs union 等于全局 selected docs。
12. 非 rank0 正确处理 `simple_evaluate() -> None`。
13. execution counter 跨 rank 求和。
14. timing 使用跨 rank最大耗时表示 wall time。
15. 每个 benchmark 结束前有 distributed barrier。
16. 只有 global rank0 写 evaluation artifacts。
17. Base / Final ANN / Phase / GIF / MTN 均支持多卡。
18. single-GPU lm-eval 行为保持兼容。
19. 1 GPU 与 4 GPU 使用同一 selection 时 metric 一致。
20. TruthfulQA 4 卡日志不再出现 4 份完整 `4114/4114`。
21. `pytest -q` 全部通过。

---

## 31. 核心原则

```text
不要并行模型内部计算，
而是让每个 GPU 持有完整的 SNN2 model replica，
由 lm-eval 按真实 rank/world_size 对 benchmark documents 做数据并行分片，
最后聚合完整指标。
```

这样改动最小，并且不会触碰 ANN/Phase/GIF/MTN forward、Prefix、calibration、conversion 和 TL;DR summarization。
