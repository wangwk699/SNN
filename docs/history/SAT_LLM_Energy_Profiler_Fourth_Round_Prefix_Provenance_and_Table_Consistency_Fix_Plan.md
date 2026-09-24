# SAT-LLM Energy Profiler 第四轮修正方案：Prefix Provenance 与正式 Energy Table 一致性校验

> 适用仓库：`https://github.com/wangwk699/SNN`  
> 当前基线：`main` 分支已完成 v3 最终部署 Energy protocol。  
> 本轮目标：修正 Energy metadata 中 Prefix 配置态与实际执行态混淆的问题，补充正式四行 Energy Table 的 profiling sample identity 一致性要求，并清理 v3 协议下已经不可达的 dead code。  
> **本轮不改变 MAC/AC accounting 数学、不改变 ANN/SNN 数值 forward、不改变 Step 9/Step 10 的 evaluation/conversion 语义。**

---

# 1. 当前状态

现有 v3 Energy protocol 已正确规定：

```text
vanilla + --neuron ann
```

用于正式 ANN Energy baseline；

以及：

```text
selected phase_aware/gif_aware checkpoint
    + --neuron phase
    + --neuron gif
    + --neuron mtn
```

用于最终三种 SNN deployment Energy。

现有 `validate_energy_deployment_protocol()` 已正确拒绝：

```text
unaware + ann
phase_aware + ann
gif_aware + ann

vanilla/unaware + phase/gif/mtn
```

因此本轮不需要再修改正式 deployment source protocol。

上一轮的：

```text
dynamic × event
event × dynamic
_pair() reduction
GIF PV regression
GIF zero-point policy
```

均保持不变。

---

# 2. 本轮需要修正的 3 个问题

本轮修正以下问题。

## 2.1 Prefix metadata 将“配置值”误写成“实际执行状态”

当前 `energy_metadata.json` 中：

```python
"evaluation_prefix_enabled": cfg["evaluation"]["prefix_enabled"]
```

这只是 raw config value，不一定等于最终 Energy forward 实际是否加载 Prefix。

典型例子：

```text
vanilla + --neuron ann
```

当前 config 中可能：

```yaml
evaluation:
  prefix_enabled: true
```

但项目正式 Final ANN protocol 对 vanilla 强制：

```text
actual Prefix disabled
```

因此 Energy forward 实际：

```text
prefix_length = 0
不加载 prefixed_key_values.pt
```

如果 metadata 仍写：

```json
"evaluation_prefix_enabled": true
```

就会产生 provenance 歧义。

---

## 2.2 Prefix 未实际启用时仍写 `prefix_artifact_stage`

当前 metadata 无条件写：

```python
final_ann_evaluation_prefix_artifact_stage(cfg)
```

或：

```python
final_snn_evaluation_prefix_artifact_stage(cfg)
```

但如果 Prefix 实际未启用，则本次 Energy forward 并未消费任何 Prefix artifact。

此时继续写：

```text
pre_finetuning
或
post_finetuning
```

容易被误解为实际使用了该 Prefix cache。

正确行为应为：

```json
"prefix_artifact_stage": null
```

---

## 2.3 正式 Energy Table 缺少“四行使用同一 profiling 样本”的显式汇总检查

正式每个 model/task 有四行：

```text
ANN
Phase
GIF
MTN
```

虽然当前 generated config 默认：

```text
calibration.seed = 42
calibration.num_samples = 128
```

且同一 model/task 的 validation manifest 通常一致，但正式论文表格必须显式保证：

```text
ANN / Phase / GIF / MTN
```

使用完全相同的 held-out 512-token sequence 集。

需要在文档中增加明确检查要求。

本轮不必增加复杂跨目录 aggregator，但 metadata 必须具备足够信息供人工/脚本核对。

---

# 3. Prefix metadata 重新定义

Energy metadata 需要同时记录：

```text
configured state
actual runtime state
```

不能只记录一个模糊的：

```text
evaluation_prefix_enabled
```

---

# 4. 建议新增 Prefix metadata 字段

推荐在 `snn2/energy_profiler.py` 中计算：

```python
configured_prefix_enabled = bool(
    cfg["evaluation"]["prefix_enabled"]
)

actual_prefix_enabled = bool(
    layout.energy_prefix_enabled(neuron)
)
```

然后 metadata 写：

```json
{
  "configured_evaluation_prefix_enabled": true,
  "actual_evaluation_prefix_enabled": false
}
```

---

# 5. 旧字段处理

当前：

```json
"evaluation_prefix_enabled": ...
```

建议删除。

原因：

```text
该名称无法区分 configured 与 actual。
```

如果为了旧结果兼容必须保留，则必须改成：

```json
"evaluation_prefix_enabled": false
```

即让该旧字段表示：

```text
actual runtime state
```

但更推荐：

```text
彻底删除旧字段
+
使用 configured_/actual_ 两个显式字段
```

Energy profiler 目前仍处于新功能阶段，不建议保留容易误解的历史字段。

---

# 6. `prefix_artifact_stage` 的正确规则

推荐先计算：

```python
actual_prefix_enabled = layout.energy_prefix_enabled(neuron)
```

然后：

```python
if not actual_prefix_enabled:
    prefix_artifact_stage = None
elif neuron == "ann":
    prefix_artifact_stage = final_ann_evaluation_prefix_artifact_stage(cfg)
else:
    prefix_artifact_stage = final_snn_evaluation_prefix_artifact_stage(cfg)
```

最终写：

```json
"prefix_artifact_stage": prefix_artifact_stage
```

---

# 7. `prefix_length` 必须与 actual state 一致

已有实际 forward 已经通过：

```python
cache = prefix_key_values_for_stage(...)
prefix_tokens = prefix_length(cache) if cache is not None else 0
```

正确得到实际 Prefix length。

本轮增加 defensive consistency check：

```python
if actual_prefix_enabled and cache is None:
    raise RuntimeError(
        "Energy protocol says Prefix is enabled but no Prefix KV cache was loaded"
    )

if not actual_prefix_enabled and cache is not None:
    raise RuntimeError(
        "Energy protocol says Prefix is disabled but a Prefix KV cache was loaded"
    )
```

然后：

```python
if not actual_prefix_enabled and prefix_tokens != 0:
    raise RuntimeError(...)
```

目的是确保：

```text
metadata
actual cache load
prefix_length
```

三者完全一致。

---

# 8. vanilla ANN 正确 metadata 示例

对于：

```bash
python scripts/profile_energy.py \
  --config "$CFG_V" \
  --neuron ann
```

如果 raw config：

```yaml
evaluation:
  prefix_enabled: true
```

但 Final vanilla ANN protocol 强制 Prefix disabled，则 metadata 应为：

```json
{
  "source_ann_mode": "vanilla",
  "configured_evaluation_prefix_enabled": true,
  "actual_evaluation_prefix_enabled": false,
  "prefix_artifact_stage": null,
  "prefix_length": 0
}
```

这是本轮最重要的 provenance 修正。

---

# 9. selected-aware SNN Prefix metadata 示例

例如：

```text
phase_aware checkpoint
+
--neuron phase
+
evaluation.prefix_enabled = true
+
conversion.use_post_finetuning_artifacts = false
```

则实际 Prefix stage 应按现有 Final SNN selector 规则解析。

metadata 例如：

```json
{
  "source_ann_mode": "phase_aware",
  "configured_evaluation_prefix_enabled": true,
  "actual_evaluation_prefix_enabled": true,
  "prefix_artifact_stage": "pre_finetuning",
  "prefix_length": 16
}
```

如果：

```text
evaluation.prefix_enabled = false
```

则：

```json
{
  "configured_evaluation_prefix_enabled": false,
  "actual_evaluation_prefix_enabled": false,
  "prefix_artifact_stage": null,
  "prefix_length": 0
}
```

---

# 10. `energy_path_prefix_enabled` 保持使用 actual state

当前：

```python
layout.energy_prefix_enabled(neuron)
```

已经按真实 Final ANN / Final SNN protocol 解析。

因此继续保留：

```json
"energy_path_prefix_enabled": ...
```

并要求：

```python
metadata["energy_path_prefix_enabled"]
==
metadata["actual_evaluation_prefix_enabled"]
```

建议增加运行时 assert：

```python
assert layout.energy_prefix_enabled(neuron) == actual_prefix_enabled
```

或显式 consistency check。

---

# 11. 正式 Energy Table 的四行 sample identity

每个 model/task 正式汇总：

```text
ANN
Phase
GIF
MTN
```

必须使用同一 profiling sample identity。

至少检查以下 metadata：

```text
validation_manifest_sha256
selected_validation_positions
profile_num_samples
profile_seed
profile_sequence_length
```

全部一致。

---

# 12. 为什么这项检查必要

当前 Energy sampling 定义为：

```text
source = held-out validation
seed = calibration.seed
N = calibration.num_samples
without replacement
sequence length = 512
```

如果未来 selected-aware config 处于某个 sweep：

```text
calibration.num_samples = 256
```

而 vanilla baseline 仍为：

```text
calibration.num_samples = 128
```

则即使 model/task 相同：

```text
ANN 和 SNN Energy
```

也不再基于完全相同的序列集。

这会破坏正式对比的公平性。

---

# 13. Step 11 增加正式汇总检查要求

在 `实验执行总结.md` Step 11 中增加：

> 汇总每个模型的四行正式 Energy Table 前，必须同时核对 ANN、Phase、GIF、MTN 四份 `energy_metadata.json` 中以下字段完全一致：
>
> - `validation_manifest_sha256`
> - `selected_validation_positions`
> - `profile_num_samples`
> - `profile_seed`
> - `profile_sequence_length`
>
> 此外三种 SNN 还必须具有一致的：
>
> - `checkpoint_source`
> - `source_ann_mode`
> - `experiment_id`

注意：

```text
ANN checkpoint_source
```

本来就应与 SNN 不同，因此不要要求 ANN 与 SNN 的 checkpoint 一致。

---

# 14. selected-aware 三种 SNN 还必须继续检查 source 一致性

Phase/GIF/MTN 三行：

```text
checkpoint_source
source_ann_mode
experiment_id
```

必须相同。

这表示三种 SNN 都来自同一个 selected-aware checkpoint。

不要放松这一条。

---

# 15. 推荐增加 sample identity helper

可在 `snn2/energy_profiler.py` 中新增纯 helper：

```python
def energy_sample_identity(metadata: dict) -> dict:
    keys = (
        "validation_manifest_sha256",
        "selected_validation_positions",
        "profile_num_samples",
        "profile_seed",
        "profile_sequence_length",
    )
    return {key: metadata[key] for key in keys}
```

本轮不强制写 aggregator，但 helper 可以用于 test 或未来 table collection script。

如果觉得没有实际 caller，也可以不新增 helper，仅文档说明。

---

# 16. 不新增跨目录 aggregator

本轮不要增加：

```text
collect_energy_results.py
aggregate_energy_table.py
```

除非仓库已经有统一结果汇总框架可直接复用。

当前要求只需：

```text
metadata 足够明确
+
文档明确核对规则
```

即可。

---

# 17. 清理 v3 协议下不可达的 dead code

当前 `profile_energy()` 内：

```python
if neuron == "ann" and cfg["experiment"]["ann_mode"] in {
    "phase_aware",
    "gif_aware",
}:
    validate_recorded_training_artifact_provenance(cfg, layout)
```

在 v3 protocol 下已经永远不可达。

因为函数入口首先执行：

```python
validate_energy_deployment_protocol(cfg, neuron)
```

而：

```text
neuron == ann
```

时只允许：

```text
ann_mode == vanilla
```

所以这段应删除。

---

# 18. 删除对应无用 import

若删除上述 dead branch，则：

```python
from snn2.training import validate_recorded_training_artifact_provenance
```

若无其他用途，应同时删除。

保持 `energy_profiler.py` 干净。

---

# 19. 不删除 Final ANN training provenance 的原有项目能力

本轮仅从：

```text
Energy profiler
```

移除不可达调用。

不要修改：

```text
evaluate_tldr.py
evaluate_lm_harness.py
evaluation.py
training.py
verify_artifacts.py
```

中已有 Final ANN provenance validation。

---

# 20. Metadata 建议最终结构

正式建议至少包含：

```json
{
  "energy_profiler_version": 3,
  "energy_accounting_policy": "sat_llm_mac_ac_v3_final_deployment_protocol",

  "experiment_id": "...",
  "task": "...",
  "model_name": "...",

  "ann_mode": "vanilla",
  "source_ann_mode": "vanilla",

  "energy_deployment_protocol":
    "vanilla_ann_vs_single_selected_aware_checkpoint_snn",

  "energy_deployment_role": "ann_baseline",

  "neuron": "ann",
  "deployment_T": null,

  "configured_evaluation_prefix_enabled": true,
  "actual_evaluation_prefix_enabled": false,
  "prefix_artifact_stage": null,
  "prefix_length": 0,

  "profile_sequence_length": 512,
  "profile_num_samples": 128,
  "profile_seed": 42,

  "validation_manifest_sha256": "...",
  "selected_validation_positions": [...],

  "checkpoint_source": "...",

  "energy_path_prefix_enabled": false,

  "mac_energy_pj": 4.6,
  "ac_energy_pj": 0.9
}
```

SNN metadata 相同，只是：

```text
source_ann_mode = selected aware
energy_deployment_role = selected_aware_snn
actual Prefix state按真实部署解析
```

---

# 21. 是否升级 profiler version

本轮只修正 metadata/provenance 与文档汇总约束。

不改变：

```text
MAC/AC mathematics
deployment source protocol
```

因此：

```python
ENERGY_PROFILER_VERSION = 3
ENERGY_ACCOUNTING_POLICY = "sat_llm_mac_ac_v3_final_deployment_protocol"
```

可以保持不变。

不需要升级到 v4。

原因：

```text
正式数值语义不变；
只是 provenance metadata 修正。
```

但如果项目习惯任何 metadata schema 改动都升级版本，也可以升 schema version；不建议修改 `ENERGY_ACCOUNTING_POLICY`。

---

# 22. 建议增加独立 metadata schema version

如果想区分旧 v3 metadata 与修正后的 metadata，优先新增：

```python
ENERGY_METADATA_SCHEMA_VERSION = 2
```

而不是把 accounting policy 升为 v4。

例如：

```json
{
  "energy_profiler_version": 3,
  "energy_accounting_policy":
    "sat_llm_mac_ac_v3_final_deployment_protocol",
  "energy_metadata_schema_version": 2
}
```

这是推荐方案。

如果不想增加新常量，也可以不加。

---

# 23. 测试：vanilla ANN actual Prefix state

新增 test，构造：

```text
ann_mode = vanilla
cfg evaluation.prefix_enabled = true
```

确认：

```text
layout.energy_prefix_enabled("ann") == false
```

并验证 metadata helper/逻辑最终得到：

```text
configured = true
actual = false
prefix_artifact_stage = None
prefix_length = 0
```

---

# 24. 测试：SNN Prefix enabled

对 selected-aware config：

```text
phase_aware 或 gif_aware
evaluation.prefix_enabled = true
```

确认：

```text
actual_evaluation_prefix_enabled = true
```

并且：

```text
prefix_artifact_stage
```

与当前：

```text
conversion.use_post_finetuning_artifacts
```

对应。

---

# 25. 测试：Prefix disabled

对 selected-aware config：

```text
evaluation.prefix_enabled = false
```

确认：

```text
actual = false
prefix_artifact_stage = None
prefix_length = 0
```

即使对应 Prefix artifacts 已经存在，也不能被 metadata 误报为实际使用。

---

# 26. 测试：cache 与 actual state consistency

可将 Prefix consistency 逻辑提取成 helper，例如：

```python
def validate_energy_prefix_runtime(
    *,
    actual_enabled: bool,
    cache: Any | None,
    prefix_tokens: int,
) -> None:
    ...
```

测试：

```text
actual=true, cache=None -> raise
actual=false, cache!=None -> raise
actual=false, prefix_tokens>0 -> raise
actual=true, cache!=None, prefix_tokens>0 -> pass
actual=false, cache=None, prefix_tokens=0 -> pass
```

若不新增 helper，也应至少覆盖关键分支。

---

# 27. 测试：sample identity

新增一个轻量纯函数或 test-local comparison：

```python
ann_meta = {...}
phase_meta = {...}
gif_meta = {...}
mtn_meta = {...}
```

检查相同：

```text
validation_manifest_sha256
selected_validation_positions
profile_num_samples
profile_seed
profile_sequence_length
```

时通过。

修改其中任意一项：

```text
num_samples
seed
positions
manifest hash
sequence length
```

应被识别为不一致。

如果本轮不增加 aggregator/helper，可以只把该规则写入文档，不强制测试。

---

# 28. 更新 `实验执行总结.md`

Step 11 当前已经正确说明：

```text
vanilla ANN
selected-aware Phase/GIF/MTN
```

本轮追加两个小节。

## 28.1 Prefix provenance

说明：

> `configured_evaluation_prefix_enabled` 记录 YAML 配置值；`actual_evaluation_prefix_enabled` 记录最终 Energy forward 实际是否加载 Prefix。  
> 对 vanilla ANN，即使 YAML `evaluation.prefix_enabled=true`，Final ANN protocol 仍强制不使用 Prefix，因此 actual=false、`prefix_artifact_stage=null`、`prefix_length=0`。

## 28.2 四行 sample identity

说明：

> 汇总正式 Energy Table 前，ANN/Phase/GIF/MTN 的 held-out profiling sample identity 必须完全一致。

并列出五个字段。

---

# 29. 更新 `代码结构总结.md`

不需要大改。

若 `energy_profiler.py` 描述当前为：

```text
校验 vanilla ANN/selected-aware SNN 来源并用临时 hook 统计理论能耗
```

可改为：

```text
校验 vanilla ANN/selected-aware SNN 最终部署来源、记录实际 Prefix provenance，并用临时 hook 统计固定 held-out 序列的理论能耗。
```

仅此即可。

---

# 30. 不修改以下文件的数值逻辑

本轮原则上不要修改：

```text
snn2/evaluation.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
snn2/temporal_ops.py
snn2/neurons.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/convert_snn.py
```

因为本轮仅处理：

```text
Energy provenance
metadata
documentation
dead code cleanup
```

---

# 31. 不改变以下 Energy accounting 规则

保持：

```text
4.6 pJ/MAC
0.9 pJ/AC

Phase nonzero spike = 1 event
MTN nonzero spike = 1 event
GIF integer code -> unit-event expansion

dynamic × dynamic -> MAC
event × event -> AC
mixed dynamic/event -> active MAC

Synaptic AC
Neuron AC

512-token full sequence
held-out validation
calibration.seed sampling
calibration.num_samples
no generation
per-sequence mean
Prefix cache construction energy excluded
Prefix attention expansion included
```

本轮不能动这些数学。

---

# 32. 正式四行 Energy 的最终核对规则

对于每个模型：

## ANN 行

必须：

```text
source_ann_mode = vanilla
energy_deployment_role = ann_baseline
neuron = ann
actual_evaluation_prefix_enabled = false
prefix_length = 0
```

其中最后两条基于当前 Final vanilla ANN protocol。

## Phase/GIF/MTN 三行

必须：

```text
energy_deployment_role = selected_aware_snn
source_ann_mode ∈ {phase_aware, gif_aware}
```

而且三行的：

```text
checkpoint_source
source_ann_mode
experiment_id
```

必须相同。

## 四行共同

以下全部必须相同：

```text
validation_manifest_sha256
selected_validation_positions
profile_num_samples
profile_seed
profile_sequence_length
```

---

# 33. 旧结果处理

如果此前已经生成 v3 Energy：

其数值可以继续有效，只要 MAC/AC protocol 与 source protocol 相同。

但旧 metadata 若存在：

```json
"evaluation_prefix_enabled": true
```

同时实际 vanilla ANN 未使用 Prefix，则 provenance 有歧义。

正式论文结果建议修复代码后重新跑，生成新的 metadata。

---

# 34. 推荐修改文件

主要：

```text
snn2/energy_profiler.py
tests/test_energy_profiler.py
实验执行总结.md
代码结构总结.md
```

`scripts/profile_energy.py` 通常不需要改。

---

# 35. 回归测试

修改完成后运行：

```bash
python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```

全部通过。

不要修改旧 evaluation 数值测试的 expected outputs。

---

# 36. 最终验收标准

### Prefix metadata

- [ ] configured 与 actual Prefix state 分开记录；
- [ ] vanilla ANN actual Prefix=false；
- [ ] vanilla ANN `prefix_artifact_stage=null`；
- [ ] vanilla ANN `prefix_length=0`；
- [ ] selected-aware SNN Prefix state与最终部署一致；
- [ ] disabled Prefix 不再写虚假的 artifact stage。

### Sample fairness

- [ ] Step 11 明确要求四行 sample identity 一致；
- [ ] 五个 sample identity 字段完整存在；
- [ ] 三种 SNN source metadata 一致性要求继续保留。

### Cleanup

- [ ] 删除不可达的 aware-ANN provenance branch；
- [ ] 删除对应无用 import。

### Safety

- [ ] MAC/AC 数学无变化；
- [ ] Step 9/10 无变化；
- [ ] ANN/SNN forward 无变化；
- [ ] `pytest -q` 全通过。

---

# 37. Codex 完成后需要汇报

完成后返回：

```text
1. 修改文件
2. Prefix configured/actual metadata 的新定义
3. prefix_artifact_stage 的新规则
4. 是否新增 ENERGY_METADATA_SCHEMA_VERSION
5. 删除的 dead code
6. Step 11 新增的四行 sample identity 检查
7. 新增测试
8. pytest -q 结果
9. 一个 vanilla ANN metadata 示例
10. 一个 selected-aware SNN metadata 示例
```

不要修改当前 v3 的 MAC/AC accounting 数学和最终 deployment source protocol。
