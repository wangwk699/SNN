# SNN Phase 对齐收尾修改方案：Prefix K/V、Surrogate、EMA 精度、计数与 Provenance

## 0. 修改目标

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

当前基准 commit：

```text
92f83298b93392a2c1d1017f3eceda40b934e78f
```

本次是对上一轮 `SNN_Phase_SpikingLLM_alignment_and_artifact_reuse_plan.md` 实施后的收尾修正。

本次只处理以下 5 类问题：

1. Prefix K/V 在 Site 3 / Site 4 仍绕过 neuron；
2. Phase surrogate backward slope 与 SpikingLLM 普通 Phase 不一致；
3. Phase τ EMA 虽然公式已一致，但递归精度应改成 FP32；
4. final RMSNorm 后新增 Phase neuron 后，evaluation 的 SNN operator count 少算该 global neuron；
5. aware calibration 的 conversion eligibility、Prefix stage 命名和 `实验执行总结.md` 仍存在语义矛盾。

除本文明确要求的内容外，不要再次改动：

- `phase_aware` / `gif_aware` 仍是 ANN fine-tuning，不引入跨层真实时间维度；
- 保留 10 个 activation sites；
- Site 9 保留；
- aware ANN common Clip 保留；
- SNN deployment 不使用 Clip；
- aware conversion 继续复用 Pre-finetuning Prefix + ANN-training calibration；
- vanilla / unaware 继续使用 Post-finetuning Prefix + Post-finetuning conversion calibration；
- temporal QK/PV、temporal Softmax cumulative-difference、temporal MLP Hadamard 主公式保持不变；
- final RMSNorm Phase neuron 仍是 global auxiliary state，不定义成 Site 11。

---

# 1. Prefix K/V 必须在 runtime 经过 Site 3 / Site 4 neuron

当前代码在：

```text
snn2/model_integration.py
snn2/temporal_model.py
```

对 Prefix K/V 做了 bypass：

```python
prefix_key, current_key = ...
prefix_value, current_value = ...

current_key = controller.apply(layer_index, 3, current_key)
current_value = controller.apply(layer_index, 4, current_value)

key = torch.cat((prefix_key, current_key), ...)
value = torch.cat((prefix_value, current_value), ...)
```

这与 SpikingLLM 普通 Phase runtime 不一致。

SpikingLLM 的实际 runtime 语义是：

```text
先把 cached Prefix K/V 合并进完整 K/V
        ↓
完整 K 进入 k_Identity
完整 V 进入 v_Identity
```

因此本项目也要改成：

```text
Prefix K + Current K -> Site 3
Prefix V + Current V -> Site 4
```

## 1.1 修改 `snn2/model_integration.py`

在 `snn2_eager_attention_forward()` 中，不要再把 Prefix K/V 从 Site 3/4 中切掉。

将逻辑改成等价于：

```python
query = controller.apply(layer_index, 2, query)

key = controller.apply(layer_index, 3, key)
value = controller.apply(layer_index, 4, value)
```

然后再继续：

```python
repeat_kv(...)
qk = ...
softmax = ...
```

### 重要要求

ANN `phase_aware` / `gif_aware` static replacement：

```text
完整 K -> Site 3
完整 V -> Site 4
```

`identity` / `collect` 模式也使用同一个 topology。

但是 **calibration statistics 对 Prefix K/V 的取样策略不要跟着改成“统计 Prefix”**，见第 1.3 节。

## 1.2 修改 `snn2/temporal_model.py`

在 `deployment_attention_forward()` 中同样删除：

```python
if past_length:
    prefix_key, current_key = ...
    ...
```

改成完整：

```python
query = controller.apply(layer_index, 2, query)
key = controller.apply(layer_index, 3, key)
value = controller.apply(layer_index, 4, value)
```

这里的 Prefix KV 已经由：

```text
uniform_kv_divide_by_T
```

处理成 temporal Prefix，所以 Site 3/4 对完整 temporal K/V 执行 neuron 即可。

## 1.3 calibration statistics 仍排除 Prefix K/V

SpikingLLM 的 activation statistic 会排除 prefixed positions，因此本项目也应保持：

```text
runtime：Prefix K/V 经过 Site 3/4 neuron
τ / GIF / MTN calibration statistic：Prefix K/V 不参与 Site 3/4 parameter estimation
```

因此需要把“runtime apply”和“collect statistics”解耦。

推荐做法：

在 `SiteController` 新增一个仅记录 statistics、不做 replacement 的 API，例如：

```python
def record_activation(
    self,
    layer_index: int,
    site_index: int,
    x: torch.Tensor,
) -> None:
    if self.mode == "collect":
        self.statistics.update(layer_index, site_index, x)
```

然后 attention 中写成：

```python
if controller.mode == "collect" and past_length:
    current_key = key[..., past_length:, :]
    current_value = value[..., past_length:, :]

    controller.record_activation(layer_index, 3, current_key)
    controller.record_activation(layer_index, 4, current_value)

    # collect 模式保持 identity，完整 key/value 继续向后传播
else:
    key = controller.apply(layer_index, 3, key)
    value = controller.apply(layer_index, 4, value)
```

无 Prefix 时：

```python
key = controller.apply(layer_index, 3, key)
value = controller.apply(layer_index, 4, value)
```

如果 controller 已存在等价能力，可复用，不必机械新增同名函数。

## 1.4 saliency

Site 3 / Site 4 的 saliency 仍只统计 current-token K/V，不包含 Prefix positions，保持现在的裁剪逻辑：

```python
if past_length:
    key_score = key_score[..., past_length:, :]
    value_score = value_score[..., past_length:, :]
```

这一点不要改。

---

# 2. Phase surrogate slope 改成 SpikingLLM 普通 Phase 的 1.0

当前：

```yaml
phase:
  surrogate_slope: 4.0
```

必须改成：

```yaml
phase:
  surrogate_slope: 1.0
```

修改：

```text
configs/experiment_matrix.yaml
snn2/config.py
tests/test_generated_configs.py
tests/test_neurons.py
```

## 2.1 `configs/experiment_matrix.yaml`

所有 experiment 的：

```yaml
phase:
  surrogate_slope:
```

统一设为：

```yaml
surrogate_slope: 1.0
```

不要只改 Qwen3-1.7B。

## 2.2 `snn2/config.py`

主实验固定要求：

```python
if float(cfg["phase"]["surrogate_slope"]) != 1.0:
    raise ValueError(
        "Main experiments require SpikingLLM Phase surrogate_slope=1.0"
    )
```

避免以后 generated config 漂移。

## 2.3 说明

这里只影响 ANN `phase_aware` fine-tuning 的 backward surrogate。

硬阈值 forward：

```text
(x > 0)
```

不变。

Phase SNN inference forward 不因该参数发生结构性变化。

---

# 3. Phase τ EMA accumulator 改成 FP32

当前：

```text
phase_ema_abs_max -> float64
current_abs_max   -> double()
```

需要与 SpikingLLM：

```python
comming_max = ... .float().cpu()
```

一致。

修改：

```text
snn2/stats.py
snn2/calibration.py
snn2/temporal_ops.py
tests/test_statistics.py
tests/test_calibration_profiles.py
```

## 3.1 `snn2/stats.py`

`SiteStatistics.create()`：

```python
phase_ema_abs_max=torch.zeros(
    channels,
    dtype=torch.float32,
)
```

`update()`：

其他 min/max/sum 统计仍可使用 FP64。

Phase EMA 单独写成：

```python
current_phase_abs_max = (
    work.abs()
    .amax(dim=0)
    .float()
    .cpu()
)
```

递归：

```python
ema = self.phase_ema_abs_max[:active]

ema.copy_(
    torch.where(
        first,
        current_phase_abs_max,
        0.99 * ema + 0.01 * current_phase_abs_max,
    )
)
```

不要中间转 FP64。

## 3.2 `snn2/calibration.py`

`build_phase_state()` 中：

```python
tau = _group_reduce(
    phase_stat.float(),
    reduction_group_size,
    "max",
).float()
```

不要先 `.double()`。

## 3.3 metadata

现有：

```text
phase_tau_statistic = spikingllm_ema_channel_abs_max
phase_tau_ema_factor = 0.99
```

保留。

新增：

```text
phase_tau_accumulator_dtype = float32
```

写入：

```text
statistics.pt
statistics_summary.json
calibration_state_manifest.json
temporal_policy_metadata()
```

建议同步升 artifact schema：

```text
SITE_STATE_FORMAT_VERSION: 3 -> 4
CALIBRATION_MANIFEST_FORMAT_VERSION: 4 -> 5
CONVERSION_METADATA_FORMAT_VERSION: 5 -> 6
```

`TEMPORAL_IMPLEMENTATION_VERSION` 可继续为 3，因为 temporal arithmetic 本身未变化。

所有 validator、错误提示、测试同步更新，旧 FP64 EMA artifact 不得静默复用。

---

# 4. final RMSNorm Phase neuron 计入 Phase SNN operator count

当前统计仍按：

\[
L 	imes 10
\]

但 `deploy_phase` 实际为：

\[
L 	imes 10 + 1
\]

最后的 `+1` 是：

```text
_global/final_rmsnorm/phase_state.pt
```

修改：

```text
snn2/evaluation.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
tests/test_evaluation_paths.py
tests/test_temporal_model_integration.py
```

## 4.1 新增统一 helper

在：

```text
snn2/evaluation.py
```

新增：

```python
def activation_neuron_operators_per_temporal_forward(
    *,
    num_hidden_layers: int,
    neuron: str,
) -> int:
    base = int(num_hidden_layers) * SITE_COUNT
    if neuron == "phase":
        base += 1
    return base
```

## 4.2 两个 evaluation script

把：

```python
temporal_sample_step_forwards * layers * SITE_COUNT
```

改成：

```python
per_forward_operators = activation_neuron_operators_per_temporal_forward(
    num_hidden_layers=layers,
    neuron=args.neuron,
)

activation_site_temporal_operator_calls = (
    temporal_sample_step_forwards
    * per_forward_operators
)

batched_activation_site_temporal_slots = (
    batched_temporal_sample_slots
    * per_forward_operators
)
```

## 4.3 metadata

增加：

```text
per_temporal_forward_activation_neuron_operators
global_final_norm_phase_neuron_present
```

语义：

```text
phase -> true
gif/mtn/ann -> false
```

`SITE_COUNT` 继续等于 10。

---

# 5. aware calibration manifest 的 conversion eligibility 改正确

当前 aware conversion 正式复用：

```text
ANN-training calibration
```

但 manifest 仍写：

```text
eligible_for_conversion = false
```

这是 provenance 自相矛盾。

修改：

```text
snn2/calibration.py
snn2/conversion.py
scripts/verify_artifacts.py
tests/test_calibration_profiles.py
tests/test_conversion_metadata.py
tests/test_post_finetuning_protocol.py
```

## 5.1 ANN-training calibration

对于：

```text
purpose = ann_training_calibration
```

改成：

```text
eligible_for_ann_training = true
eligible_for_conversion = true
conversion_reuse_policy = aware_modes_only
post_finetuning_recalibration = false
state_profile = ann_training_with_common_clip
common_clip_required = true
```

`eligible_for_conversion=true` 不代表任意 mode 都可用，必须由：

```text
conversion_reuse_policy = aware_modes_only
```

限制。

## 5.2 Post-finetuning calibration

继续：

```text
eligible_for_ann_training = false
eligible_for_conversion = true
conversion_reuse_policy = final_ann_only
post_finetuning_recalibration = true
state_profile = snn_conversion_without_clip
common_clip_required = false
```

## 5.3 `snn2/conversion.py`

aware source manifest 必须检查：

```python
{
    "purpose": "ann_training_calibration",
    "eligible_for_ann_training": True,
    "eligible_for_conversion": True,
    "conversion_reuse_policy": "aware_modes_only",
    "post_finetuning_recalibration": False,
    "state_profile": "ann_training_with_common_clip",
    "common_clip_required": True,
}
```

vanilla/unaware：

```python
{
    "purpose": "post_finetuning_conversion_calibration",
    "eligible_for_ann_training": False,
    "eligible_for_conversion": True,
    "conversion_reuse_policy": "final_ann_only",
    "post_finetuning_recalibration": True,
    "state_profile": "snn_conversion_without_clip",
    "common_clip_required": False,
}
```

---

# 6. Prefix artifact stage 命名统一为 `pre_finetuning`

当前同一个 aware Prefix：

```text
conversion metadata -> prefix_source_stage = pre_finetuning
evaluation metadata -> prefix_source_stage = ann_training
```

必须统一。

修改：

```text
snn2/config.py
snn2/modeling.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
tests/test_post_finetuning_protocol.py
tests/test_evaluation_paths.py
```

## 6.1 helper

把：

```python
final_evaluation_prefix_artifact_stage(cfg)
```

改成：

```python
def final_evaluation_prefix_artifact_stage(cfg) -> str:
    return (
        "pre_finetuning"
        if is_aware_ann_mode(cfg)
        else "post_finetuning"
    )
```

## 6.2 `snn2/modeling.py`

`prefix_ids_for_stage(..., stage="final_evaluation")`：

```text
pre_finetuning -> layout.ann_training_prefix_dir
post_finetuning -> layout.post_finetuning_prefix_dir
```

`prefix_key_values_for_stage()` 同样修改。

不要再用 `"ann_training"` 同时表示物理目录和 artifact provenance stage。

## 6.3 metadata

最终 aware ANN/SNN evaluation：

```text
prefix_source_stage = pre_finetuning
```

vanilla/unaware：

```text
prefix_source_stage = post_finetuning
```

conversion 与 evaluation 对同一 Prefix 必须一致。

---

# 7. `实验执行总结.md` 必须彻底清理旧协议

当前文档仍有多处旧语义，必须完整清理。

修改：

```text
实验执行总结.md
README.md
AGENTS.md
代码结构总结.md（仅文件一句话功能描述变化时）
```

## 7.1 Step 10

删除：

```text
Conversion 使用 Step 8 中由 post_finetuning.prefix_enabled 指定的 calibration
```

改成：

```text
vanilla / unaware:
    使用 Step 7 Post-finetuning Prefix
    + Step 8 Post-finetuning conversion calibration

phase_aware / gif_aware:
    不执行 Step 7 / Step 8
    使用 Step 4 Pre-finetuning Prefix
    + Step 5 ANN-training calibration
```

## 7.2 附录 A 依赖树

Vanilla 不得再出现：

```text
Pre-finetuning Prefix
```

应为：

```text
Vanilla
├── ANN training（无 Prefix / 无 replacement）
└── final ANN
    ├── Post-finetuning Prefix
    ├── Post-finetuning conversion calibration
    └── conversion / evaluation
```

Unaware：

```text
Rotation
├── Pre-finetuning Prefix
└── ANN training
    └── final ANN
        ├── Post-finetuning Prefix
        ├── Post-finetuning conversion calibration
        └── conversion / evaluation
```

Aware：

```text
Rotation
├── shared Pre-finetuning Prefix
├── shared ANN-training calibration
└── aware ANN training
    └── final ANN
        └── conversion / evaluation
            复用训练前 Prefix + calibration
```

## 7.3 附录 B Prefix 开关表

`evaluation.prefix_enabled` 的 Prefix 来源：

```text
phase_aware/gif_aware -> Pre-finetuning Prefix
vanilla/unaware       -> Post-finetuning Prefix
```

`post_finetuning.prefix_enabled` 只控制：

```text
vanilla/unaware 的 Post-finetuning calibration / conversion Prefix
```

## 7.4 附录 C

删除：

```text
vanilla 从原始 pretrained Base 发现 Pre-finetuning Prefix
```

改成：

```text
Pre-finetuning Prefix 只存在于：
unaware / phase_aware / gif_aware
```

同一 model-task 共享：

```text
_shared/.../rotated_prefix/pre_finetuning_prefix/
```

## 7.5 附录 D

删除 vanilla ANN-training calibration 的旧描述。

改成：

```text
ANN-training calibration 只用于 phase_aware / gif_aware
```

## 7.6 附录 E

删除或修正：

```text
vanilla_original/pre_finetuning_prefix/
vanilla ANN-training calibration
```

以及所有把 SNN conversion descriptor suffix 一律解释为：

```text
post_finetuning.prefix_enabled
```

的描述。

aware conversion suffix 应解释为：

```text
conversion_prefix_enabled(cfg)
```

即 aware 对应 `ann_training.prefix_enabled`。

---

# 8. README / AGENTS 收尾

## README.md

只做必要同步：

- Prefix K/V runtime 同样经过 Site 3/4 neuron；
- Phase surrogate slope 固定为 1.0；
- Phase τ EMA accumulator 为 FP32；
- aware ANN-training calibration 明确可用于 aware conversion reuse。

## AGENTS.md

增加：

```text
Prefix K/V 在实际 ANN-aware replacement 与 SNN deployment 中必须经过 Site 3/4 neuron；
calibration statistic 可排除 Prefix positions，但不得因此让 Prefix runtime bypass neuron。
```

再增加：

```text
普通 Phase main experiment 固定 surrogate_slope=1.0，Phase τ EMA accumulator 固定 FP32。
```

---

# 9. 测试要求

必须增加以下测试。

## 9.1 Prefix K/V Site 3/4 runtime

在：

```text
tests/test_temporal_model_integration.py
tests/test_controller_state_loading.py
```

增加：

### ANN/static replacement

构造有 Prefix 的 attention，断言：

```text
输入 Site 3 的 K length == prefix_length + current_length
输入 Site 4 的 V length == prefix_length + current_length
```

### deployment

Phase/GIF/MTN 至少验证：

```text
Prefix K/V 会进入 Site 3/4 runtime neuron
```

## 9.2 collect statistic 排除 Prefix

构造有 Prefix 的 collect forward。

断言 Site 3 / Site 4 statistics 只统计 current-token positions，不包含 Prefix positions。

## 9.3 surrogate slope

`test_generated_configs.py`：

```python
cfg["phase"]["surrogate_slope"] == 1.0
```

`test_neurons.py`：

- forward hard threshold 不变；
- backward slope 使用 1.0。

## 9.4 EMA dtype

`test_statistics.py`：

```python
phase_ema_abs_max.dtype == torch.float32
```

并使用 FP32 手算 EMA。

`phase_state["tau"].dtype == torch.float32`。

## 9.5 operator count

构造：

```text
layers = 28
SITE_COUNT = 10
```

断言：

```text
phase -> 281
gif   -> 280
mtn   -> 280
```

## 9.6 manifest eligibility

aware：

```text
eligible_for_conversion = true
conversion_reuse_policy = aware_modes_only
```

vanilla/unaware Post-finetuning：

```text
eligible_for_conversion = true
conversion_reuse_policy = final_ann_only
```

并验证 unaware 不能用 ANN-training calibration。

## 9.7 Prefix stage metadata

aware final ANN/SNN evaluation：

```text
prefix_source_stage == pre_finetuning
```

vanilla/unaware：

```text
prefix_source_stage == post_finetuning
```

---

# 10. Artifact version / 重跑边界

由于本次：

- Phase EMA accumulator precision 改变；
- aware calibration manifest schema 改变；
- Prefix K/V runtime topology 语义改变；
- Phase surrogate slope 改变；

旧 aware ANN checkpoint 不能继续作为正式对齐实验使用。

## 10.1 phase_aware / gif_aware

必须从 ANN-training calibration 开始：

```text
Pre-finetuning Prefix
        ↓
重新 ANN-training calibration
        ↓
重新 aware ANN fine-tuning
        ↓
重新 conversion descriptor
        ↓
重新 Phase/GIF/MTN evaluation
```

不要做 Post-finetuning Prefix / calibration。

## 10.2 vanilla / unaware

ANN checkpoint 可复用，但：

```text
Post-finetuning conversion calibration
conversion descriptor
SNN evaluation
```

必须重新执行。

如果 Post-finetuning Prefix 对应的 final ANN checkpoint 未变化，可复用。

## 10.3 可复用项

如果 hash / config 未变化：

```text
data manifest
rotation
Pre-finetuning Prefix
```

可复用。

---

# 11. 最终完成条件

全部满足才算完成：

1. Prefix K/V 在 runtime 中进入 Site 3 / Site 4 neuron；
2. Site 3/4 calibration statistics 仍排除 Prefix positions；
3. Site 3/4 saliency 仍排除 Prefix positions；
4. `phase.surrogate_slope == 1.0`；
5. Phase τ EMA accumulator 为 FP32；
6. aware ANN-training calibration manifest 明确可被 aware conversion 复用；
7. unaware / vanilla 不得误用 aware calibration；
8. final aware evaluation `prefix_source_stage == pre_finetuning`；
9. Phase operator count 使用 `layers * 10 + 1`；
10. GIF/MTN operator count仍为 `layers * 10`；
11. `SITE_COUNT == 10`；
12. final RMSNorm Phase 不成为 Site 11；
13. `实验执行总结.md` 不再出现 vanilla Pre-finetuning Prefix、vanilla ANN-training calibration、aware Post-finetuning conversion 依赖等旧语义；
14. README / AGENTS 与代码一致；
15. 全部测试通过：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py   --matrix configs/experiment_matrix.yaml   --output-dir configs/generated

pytest -q
```

---

# 12. 实施后最小验证

代码修改完成后先执行：

```bash
pytest -q
```

然后用 Qwen3-1.7B `phase_aware`：

```bash
CFG=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml

python scripts/calibrate_sites.py   --config "$CFG"   --stage ann_training
```

检查任意几个 `phase_state.pt`：

```text
tau_calibration = spikingllm_ema_channel_abs_max
tau_ema_factor = 0.99
tau dtype = float32
surrogate_slope = 1.0
```

检查 `calibration_state_manifest.json`：

```text
eligible_for_ann_training = true
eligible_for_conversion = true
conversion_reuse_policy = aware_modes_only
```

确认后再重新进行 `phase_aware` ANN fine-tuning 和 SNN conversion。
