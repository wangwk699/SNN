# SAT-LLM Energy Profiling 最终部署协议修正方案（第三轮）

> 适用仓库：`https://github.com/wangwk699/SNN`  
> 目标：将 Energy profiling 从“遍历所有 ANN training modes”改为“只分析最终实际部署模型”。  
> 本文档面向服务器上的 Codex；即使没有本轮对话上下文，也应能仅凭本文件完成代码和文档修改。  
> **最高优先级约束：不得改变现有 ANN/SNN evaluation 数值路径，不得改变 Step 9/Step 10 的性能评估语义。**

---

# 1. 背景与最终结论

当前项目存在四种 ANN training mode：

```text
vanilla
unaware
phase_aware
gif_aware
```

其中：

- `vanilla`：普通 ANN training；
- `unaware`：带 Hadamard rotation，但 ANN training 不使用 replacement-aware surrogate；
- `phase_aware`：训练时使用 Phase-aware static surrogate replacement；
- `gif_aware`：训练时使用 GIF-aware static replacement。

但 **Energy profiling 的目标不是分析训练时 forward graph，而是分析最终部署模型的计算能耗**。

因此：

- `phase_aware ANN` forward 和 `gif_aware ANN` forward 只是训练/静态评估语义，不是最终部署时要运行的 graph；
- `unaware ANN` 也只是对照实验，不是最终部署的 ANN baseline；
- 最终 ANN baseline 应使用 `vanilla` Final ANN checkpoint + 原始 dense ANN graph；
- 最终 SNN Energy 应使用“最终真正选定部署的 aware checkpoint”作为统一权重来源，并分别转换/运行 Phase、GIF、MTN 三种 full-temporal SNN graph。

---

# 2. 最终 Energy protocol

对每一个 model/task，只产生 4 行正式 Energy 结果：

| Energy row | checkpoint source | deployment graph |
|---|---|---|
| ANN | `vanilla` Final ANN checkpoint | ordinary dense ANN |
| Phase | **最终选定的 aware checkpoint** | `deploy_phase` full-temporal SNN |
| GIF | **同一个最终选定的 aware checkpoint** | `deploy_gif` full-temporal SNN |
| MTN | **同一个最终选定的 aware checkpoint** | `deploy_mtn` full-temporal SNN |

其中“最终选定的 aware checkpoint”必须是：

```text
phase_aware
或
gif_aware
```

二选一。

对于同一个模型：

```text
Phase / GIF / MTN 三行必须使用同一个 selected-aware checkpoint。
```

禁止：

```text
Phase -> phase_aware checkpoint
GIF   -> gif_aware checkpoint
MTN   -> 任意另一个 checkpoint
```

这种混用方式。

---

# 3. 为什么 SNN Energy 必须使用最终 selected-aware checkpoint

MAC 中一部分由结构决定，但 AC 不是纯结构常数。

Phase/GIF/MTN 的：

```text
spike count
event multiplicity
firing activity
```

会受到 Final ANN checkpoint 权重影响。

因此，如果最终实际部署使用的是某一个 `phase_aware` 或 `gif_aware` checkpoint，那么其真实理论 Energy 应基于该 checkpoint 产生的实际 temporal activity。

所以正式 Energy 不能再统一使用 `unaware` checkpoint。

`unaware` 仍保留为性能对照实验，但不参与最终 Energy Table。

---

# 4. ANN baseline 的定义

正式 ANN Energy baseline 固定为：

```text
vanilla Final ANN checkpoint
+
ordinary dense ANN forward
```

即：

```bash
python scripts/profile_energy.py \
  --config "$CFG_V" \
  --neuron ann
```

由于 config 必须是 vanilla：

```text
controller mode = identity
replacement sites不执行 Phase/GIF surrogate
无 temporal SNN
```

这才代表最终传统 ANN deployment baseline。

不要使用：

```text
unaware + --neuron ann
phase_aware + --neuron ann
gif_aware + --neuron ann
```

生成正式 Energy 结果。

---

# 5. 三种 SNN 的定义

对于最终选定 aware config：

```bash
python scripts/profile_energy.py \
  --config "$CFG_SELECTED_AWARE" \
  --neuron phase

python scripts/profile_energy.py \
  --config "$CFG_SELECTED_AWARE" \
  --neuron gif

python scripts/profile_energy.py \
  --config "$CFG_SELECTED_AWARE" \
  --neuron mtn
```

当前 Step 10 已明确：

```text
--neuron phase -> deploy_phase
--neuron gif   -> deploy_gif
--neuron mtn   -> deploy_mtn
```

无论源 ANN mode 是 `phase_aware` 还是 `gif_aware`，三种 SNN 都进入对应的 full-temporal deployment graph，并且不使用 ANN-training common Clip。

Energy profiler 必须继续复用这一语义。

---

# 6. Hadamard Rotation 的处理

最终 selected-aware checkpoint 本身属于 rotated mode，因此：

```text
Hadamard rotation 继续保留
```

SNN Energy profiling 必须沿用最终部署时实际存在的 rotation/model integration。

不要为了 Energy profiling 把 selected-aware checkpoint 转回 vanilla graph。

---

# 7. Prefix / Stage-A / selector 的处理

SNN Energy 必须完全跟随最终部署 config 的：

```text
conversion.use_post_finetuning_artifacts
evaluation.prefix_enabled
Prefix artifact source
Stage-A source
Phase T/base
MTN T/K
GIF temporal policy
```

也就是说，Energy profiler 只改变“统计 operation”，不能自己创造一套简化 deployment graph。

Prefix 规则继续保持此前约定：

- Prefix KV cache 的**生成成本不计入 per-sample Energy**；
- cache 提前计算一次并重复使用；
- Prefix 导致的实际 K/V length 增长必须计入当前样本 attention MAC/AC。

---

# 8. GIF-aware MSE branch 的特别规则

如果最终 selected-aware checkpoint 是：

```text
gif_aware MSE-refined branch
```

则 Energy profiling 必须继续使用该实际最终 config，例如已有文档中的：

```text
$CFG_L_G_MSE
```

不要在 Energy 阶段切回：

```text
$CFG_L_G_DIRECT
```

Energy 必须与最终真实部署 checkpoint / Stage-A branch 一致。

---

# 9. 当前 Step 11 的问题

当前 `实验执行总结.md` Step 11：

```bash
for CFG in "${ALL_CFGS[@]}"; do
  for NEURON in ann phase gif mtn; do
    python scripts/profile_energy.py --config "$CFG" --neuron "$NEURON"
  done
done
```

是错误的正式 Energy protocol。

它会产生：

```text
vanilla ANN
unaware ANN
phase-aware static ANN
gif-aware static ANN

以及每一种 ANN training mode 下的三种 SNN
```

这既包含不会实际部署的 ANN graph，也会产生大量并非最终部署 checkpoint 对应的 SNN Energy。

必须修改。

---

# 10. Step 11 新的配置选择方式

Step 11 应为每个模型明确指定：

```text
1 个 vanilla config
1 个 selected-aware config
```

当前项目变量为：

```bash
CFG_17_V
CFG_17_P
CFG_17_G

CFG_8_V
CFG_8_P
CFG_8_G

CFG_L_V
CFG_L_P
CFG_L_G
```

其中每个模型的 SNN Energy config 必须由最终实验结果决定。

建议 Step 11 增加三个显式变量：

```bash
# 对每个模型只选择一个最终实际部署的 aware checkpoint。
# 必须在 phase_aware / gif_aware 中二选一。
#
# 以下仅为写法示例，实际值按最终部署选择修改：
ENERGY_CFG_17_SNN="$CFG_17_P"   # 或 "$CFG_17_G"
ENERGY_CFG_8_SNN="$CFG_8_P"     # 或 "$CFG_8_G"
ENERGY_CFG_L_SNN="$CFG_L_P"     # 或 "$CFG_L_G"/"$CFG_L_G_MSE"
```

不要由代码自动根据 metric 猜测或选择最终 checkpoint。

最终选择由用户根据正式实验结果决定。

---

# 11. Step 11 推荐命令

## Qwen3-1.7B

```bash
python scripts/profile_energy.py \
  --config "$CFG_17_V" \
  --neuron ann

for NEURON in phase gif mtn; do
  python scripts/profile_energy.py \
    --config "$ENERGY_CFG_17_SNN" \
    --neuron "$NEURON"
done
```

## Qwen3-8B

```bash
python scripts/profile_energy.py \
  --config "$CFG_8_V" \
  --neuron ann

for NEURON in phase gif mtn; do
  python scripts/profile_energy.py \
    --config "$ENERGY_CFG_8_SNN" \
    --neuron "$NEURON"
done
```

## Llama-3-8B / Tulu-3

```bash
python scripts/profile_energy.py \
  --config "$CFG_L_V" \
  --neuron ann

for NEURON in phase gif mtn; do
  python scripts/profile_energy.py \
    --config "$ENERGY_CFG_L_SNN" \
    --neuron "$NEURON"
done
```

因此每个 model/task 最终只产生 4 行正式 Energy 结果。

---

# 12. 不修改 Step 9 / Step 10

本轮只修改 Energy profiling protocol。

不要修改：

```text
Step 9 Final ANN evaluation
Step 10 SNN conversion/evaluation
```

Step 9/10 仍然用于完整性能实验和对照实验，因此可以继续遍历：

```text
vanilla
unaware
phase_aware
gif_aware
```

本轮限制只适用于：

```text
Step 11 Energy profiling
```

---

# 13. 给 `profile_energy.py` 增加正式协议校验

为了避免未来误运行，Energy profiler 应显式拒绝不属于最终部署协议的组合。

建议新增纯函数：

```python
def validate_energy_deployment_protocol(cfg: dict, neuron: str) -> None:
    ...
```

规则如下。

## 13.1 `--neuron ann`

要求：

```python
cfg["experiment"]["ann_mode"] == "vanilla"
```

否则直接报错。

因此正式支持：

```text
vanilla + ann
```

拒绝：

```text
unaware + ann
phase_aware + ann
gif_aware + ann
```

## 13.2 `--neuron phase|gif|mtn`

要求：

```python
cfg["experiment"]["ann_mode"] in {"phase_aware", "gif_aware"}
```

正式支持：

```text
phase_aware + phase
phase_aware + gif
phase_aware + mtn

gif_aware + phase
gif_aware + gif
gif_aware + mtn
```

正式 Energy protocol 拒绝：

```text
vanilla + phase/gif/mtn
unaware + phase/gif/mtn
```

注意：这只是 `profile_energy.py` 的正式 Energy policy。

不要改变：

```text
convert_snn.py
evaluate_tldr.py
evaluate_lm_harness.py
```

这些脚本仍需支持 Step 10 的完整四种 mode SNN evaluation。

---

# 14. 校验应放在哪里

推荐在：

```text
scripts/profile_energy.py
```

CLI 参数解析和 deployment override 后、真正加载模型/大 artifact 前调用。

也可以把纯 helper 放在：

```text
snn2/energy_profiler.py
```

然后脚本调用。

为了防止其他 Python caller 直接绕过 CLI，建议：

```python
profile_energy(...)
```

入口内部也调用同一个 validator。

不要复制两套判断逻辑。

---

# 15. 不要把 selected-aware 的选择写进 config schema

不要新增：

```yaml
energy:
  selected_checkpoint: ...
```

也不要修改：

```text
configs/experiment_matrix.yaml
materialize_configs.py
```

原因：

selected-aware checkpoint 是最终实验决策，不是训练配置属性。

Step 11 通过 shell 变量：

```text
ENERGY_CFG_*_SNN
```

明确选择即可。

---

# 16. 输出路径不需要重新设计

当前输出路径已经与 source run 绑定。

因此：

```text
vanilla + ann
```

自然保存在 vanilla run 下：

```text
<vanilla-run>/energy/ann/...
```

而：

```text
selected-aware + phase/gif/mtn
```

自然保存在 selected-aware run 下：

```text
<selected-aware-run>/energy/snn/.../phase/...
<selected-aware-run>/energy/snn/.../gif/...
<selected-aware-run>/energy/snn/.../mtn/...
```

这正是需要的语义。

本轮不需要把 SNN Energy 移动到 vanilla 路径。

---

# 17. Metadata 增加最终部署协议字段

当前 metadata 已包含：

```text
ann_mode
neuron
checkpoint_source
controller_mode
```

建议再增加：

```json
{
  "energy_deployment_protocol":
    "vanilla_ann_vs_single_selected_aware_checkpoint_snn",

  "energy_deployment_role":
    "ann_baseline"
}
```

ANN 时：

```text
energy_deployment_role = "ann_baseline"
```

SNN 时：

```text
energy_deployment_role = "selected_aware_snn"
```

同时增加：

```json
"source_ann_mode": "vanilla"
```

或：

```json
"source_ann_mode": "phase_aware"
```

/ `"gif_aware"`。

即使 `ann_mode` 已经存在，也建议保留 `source_ann_mode`，使 Energy metadata 自解释。

---

# 18. SNN metadata 的一致性

对一个模型的 Phase/GIF/MTN 三个正式 Energy 结果：

```text
source_ann_mode
checkpoint_source
experiment id
selected-aware run identity
```

必须来自同一个 selected-aware config。

程序无法自动知道“三个独立命令是不是同一 config”，所以这条主要由 Step 11 命令保证。

可以在文档中明确：

> 汇总正式 Energy Table 前，必须检查三个 SNN `energy_metadata.json` 的 `checkpoint_source` 和 `source_ann_mode` 一致。

无需新增复杂的跨目录 registry。

---

# 19. CSV 主表不增加 Training Mode 列

保持此前确认的 8 列：

```text
Model
Neuron
T
MACs (G)
Synaptic ACs (G)
Neuron ACs (G)
Total ACs (G)
Energy (J)
```

不要增加：

```text
Training Mode
Checkpoint Mode
```

因为正式协议下一种 neuron 只对应唯一的 deployment source。

source checkpoint 信息由：

```text
结果路径
energy_metadata.json
```

记录。

---

# 20. `--neuron ann` 的实际 forward 不需要重新实现

只要 validator 强制：

```text
ann -> vanilla config
```

现有：

```python
build_evaluation_controller(cfg, layout, neuron="ann")
```

对于 vanilla 会自然得到：

```text
mode = identity
```

因此就是普通 dense ANN graph。

不要为了 Energy profiler 新写一个 ANN model forward。

继续复用现有数值路径。

---

# 21. SNN forward 不需要重新实现

对于 selected-aware：

```text
--neuron phase
--neuron gif
--neuron mtn
```

继续使用：

```python
build_evaluation_controller(...)
temporal_forward(...)
```

对应：

```text
deploy_phase
deploy_gif
deploy_mtn
```

不要为 Energy 新写 temporal deployment。

---

# 22. `unaware` 在 Energy 中的角色

正式 Energy Table 中：

```text
unaware 不参与
```

但不要删除：

```text
unaware config
unaware ANN checkpoint
unaware SNN conversion
unaware performance evaluation
```

它仍是论文性能对照实验的一部分。

只是：

```text
profile_energy.py
```

的正式 deployment protocol 不允许用 unaware 产生最终 Energy 行。

---

# 23. 未选中的 aware mode 在 Energy 中的角色

假设某模型最终选择：

```text
phase_aware
```

则：

```text
gif_aware checkpoint 不参与该模型正式 Energy profiling。
```

反之亦然。

不要同时跑两套 aware Energy，然后挑更好看的 Energy。

Energy source checkpoint 应由最终实际 deployment decision 决定。

---

# 24. 选择最终 aware checkpoint 的依据

本轮代码不要实现自动选择逻辑。

最终：

```text
phase_aware
vs
gif_aware
```

选择由用户根据完整性能实验结果决定。

Energy profiling 只是对已经选定的最终 checkpoint 进行 measurement/accounting。

---

# 25. 测试要求

修改：

```text
tests/test_energy_profiler.py
```

增加 protocol validation tests。

至少包括：

```python
vanilla + ann              -> pass
unaware + ann              -> raise
phase_aware + ann          -> raise
gif_aware + ann            -> raise

phase_aware + phase        -> pass
phase_aware + gif          -> pass
phase_aware + mtn          -> pass

gif_aware + phase          -> pass
gif_aware + gif            -> pass
gif_aware + mtn            -> pass

vanilla + phase/gif/mtn    -> raise
unaware + phase/gif/mtn    -> raise
```

建议参数化测试。

---

# 26. 不要破坏现有 Energy accounting tests

以下已有测试仍必须全部保留并通过：

```text
MAC/AC unit conversion
Phase event multiplicity
MTN event multiplicity
GIF integer-code unit event expansion
GIF nonzero zero-point policy
dynamic × event temporal matmul
event × dynamic temporal matmul
event × event temporal matmul
_pair() reduction
GIF PV regression
fixed 512-token input
held-out selection
Prefix length accounting
Energy path isolation
profiler on/off numerical equivalence
```

---

# 27. 文档修改要求

重点修改：

```text
实验执行总结.md
```

Step 11 必须清楚说明：

1. Energy 只分析最终部署 graph；
2. ANN baseline 只使用 vanilla；
3. SNN 三行只使用同一个 selected-aware checkpoint；
4. selected-aware 由 Phase-aware / GIF-aware 二选一；
5. unaware 不参与正式 Energy Table；
6. 未选中的 aware checkpoint 不参与正式 Energy Table；
7. SNN source config 必须已经具备对应 Phase/GIF/MTN conversion descriptors；
8. GIF-aware MSE branch 若被选中，必须使用同一 MSE config。

可以同步更新：

```text
代码结构总结.md
```

如果其中存在 Energy profiler 的使用说明。

---

# 28. Step 11 推荐完整文本结构

建议写成：

```markdown
## Step 11：Energy profiling of final deployment models

Energy profiling 不遍历所有 ANN training modes。它只统计最终实际部署 graph。

每个模型固定统计四行：

- Vanilla Final ANN checkpoint -> ordinary ANN
- Selected aware Final ANN checkpoint -> Phase SNN
- Same selected aware checkpoint -> GIF SNN
- Same selected aware checkpoint -> MTN SNN

其中 selected aware checkpoint 必须在 phase_aware / gif_aware 中二选一，并由最终性能实验决定。
```

然后给出三个：

```text
ENERGY_CFG_17_SNN
ENERGY_CFG_8_SNN
ENERGY_CFG_L_SNN
```

变量和命令。

---

# 29. 不改变前两轮 Energy 基本协议

以下继续保持：

```text
held-out validation
calibration.seed 随机无放回
num_samples = calibration.num_samples
exactly 512 tokens
batch size = 1
full sample, not prompt-only
one forward, no generation
Prefix cache generation cost excluded
Prefix K/V length effect included
raw operation count internally
4.6 pJ/MAC
0.9 pJ/AC
Synaptic AC + Neuron AC
no Reduction column
memory energy excluded
special-function energy excluded
dense residual/bias additions excluded
Hadamard implementation cost excluded from main energy
GIF zero-point fixed compensation excluded
```

不要因为本轮 source-checkpoint protocol 改动而修改这些规则。

---

# 30. Energy accounting policy version

本轮改变的是：

```text
允许进入正式 Energy profiler 的 source checkpoint protocol
```

不是 MAC/AC 数学公式。

建议仍然升级 profiler policy/version，以避免旧的“任意 ann_mode 都可 profile”结果和新的正式部署结果混淆。

例如：

```python
ENERGY_PROFILER_VERSION = 3
ENERGY_ACCOUNTING_POLICY = "sat_llm_mac_ac_v3_final_deployment_protocol"
```

虽然 MAC/AC 数学仍沿用 v2，但正式结果 provenance 已改变，因此升级 version 更安全。

---

# 31. 旧 v2 Energy 结果

如果之前已经使用 v2 对：

```text
unaware
phase_aware ANN
gif_aware ANN
或非 selected-aware SNN
```

运行过 Energy profiling：

这些结果可以保留做 debug，但：

```text
不得作为正式论文最终 Energy Table 数据。
```

正式结果应按 v3 protocol 重新运行。

---

# 32. 推荐修改文件

本轮主要修改：

```text
scripts/profile_energy.py
snn2/energy_profiler.py
tests/test_energy_profiler.py
实验执行总结.md
```

可选：

```text
代码结构总结.md
docs/history/
```

原则上不要修改：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/convert_snn.py
snn2/evaluation.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
snn2/temporal_ops.py
snn2/neurons.py
```

因为本轮不改变正式 evaluation / conversion 数值语义。

---

# 33. 验收标准

完成后必须满足：

### Protocol

- [ ] `vanilla + --neuron ann` 可运行；
- [ ] `unaware/phase_aware/gif_aware + --neuron ann` 被 Energy profiler 拒绝；
- [ ] `phase_aware + --neuron phase/gif/mtn` 可运行；
- [ ] `gif_aware + --neuron phase/gif/mtn` 可运行；
- [ ] `vanilla/unaware + --neuron phase/gif/mtn` 被 Energy profiler 拒绝；
- [ ] Step 10 完整 evaluation 能力不受影响。

### Step 11

- [ ] 不再遍历 `ALL_CFGS`；
- [ ] 每个模型只跑一个 vanilla ANN；
- [ ] 每个模型只指定一个 selected-aware config；
- [ ] 同一 selected-aware config 连续跑 Phase/GIF/MTN；
- [ ] 文档说明 GIF MSE branch 的一致性。

### Metadata

- [ ] 写入 `energy_deployment_protocol`；
- [ ] 写入 `energy_deployment_role`；
- [ ] 写入 `source_ann_mode`；
- [ ] ANN metadata 显示 vanilla；
- [ ] SNN metadata 显示实际 selected aware mode。

### Safety

- [ ] evaluation forward 数值路径无改动；
- [ ] conversion forward 数值路径无改动；
- [ ] existing Energy accounting tests 全通过；
- [ ] `pytest -q` 全部通过。

---

# 34. 修改完成后的正式运行形态

最终对每个 model/task：

```text
1 × ANN Energy
3 × SNN Energy
= 4 rows
```

若共有三个 model/task：

```text
3 × 4 = 12 rows
```

而不是：

```text
12 configs × 4 neurons = 48 rows
```

---

# 35. Codex 完成后需汇报

完成修改后请返回：

```text
1. 修改/新增的文件
2. 新 Energy deployment protocol validator 的规则
3. Step 11 新命令
4. metadata 新字段
5. ENERGY_PROFILER_VERSION / policy
6. 新增测试
7. pytest -q 结果
8. 一个 vanilla ANN Energy 命令示例
9. 一个 selected-aware Phase/GIF/MTN Energy 命令示例
```

不要自动决定哪个 aware checkpoint 是最终 selected-aware；该选择留给用户根据最终性能结果显式设置。
