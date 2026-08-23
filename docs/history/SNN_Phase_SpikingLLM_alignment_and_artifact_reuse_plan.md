# SNN Phase / SpikingLLM 对齐与感知微调工件复用修改方案

## 0. 目标与边界

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

参考实现：

```text
https://github.com/njzhenghy/SpikingLLM
```

本次修改以 SpikingLLM 的普通 single-grain Phase coding 为参考，重点对齐：

```text
phase/phase_util.py::get_act_stat
phase/phase_neuron.py::FSNeuron.find_activation_quant_param
phase/phase_layer.py::phaseSnnEmbedding
phase/phase_layer.py 中 Softmax temporal + softmax_Identity 路径
```

必须遵守以下边界，禁止顺手改掉：

1. `phase_aware` / `gif_aware` 仍然是 **ANN fine-tuning**。ANN 微调阶段不允许把整个 Transformer 展开成真实的层间时间维度 `T`；只保留当前局部 neuron replacement 的静态模拟方式。
2. 保留当前 10 个 activation replacement sites，**Site 9 必须保留**，不得退回 9 sites。
3. `phase_aware` / `gif_aware` ANN 微调中的 common Clip 保留；SNN deployment 仍不得加载或执行 common Clip。
4. Phase neuron 不训练 neuron parameter；本次只修改初始化 / calibration 来源与后续工件复用协议。
5. 不修改当前 Phase 主体编码公式、`base=2`、`v0=0.5*tau*2^-T`、当前 `T`、`max_spikes=T` 等已与普通 Phase coding 等价的部分。
6. 不修改已经对齐的 temporal QK/PV matmul、temporal Softmax 的 cumulative-difference 主公式以及 MLP temporal Hadamard 主公式。

---

# 1. 修改后的最终实验协议

新增统一的 mode-aware 协议判断，禁止继续把所有 ANN mode 都强制走相同 Post-finetuning 流程。

最终依赖关系必须是：

| ANN mode | Pre-finetuning Prefix | ANN-training calibration | Post-finetuning Prefix | Post-finetuning conversion calibration | SNN conversion 使用的 Prefix | SNN conversion 使用的 calibration |
|---|---|---|---|---|---|---|
| `vanilla` | 不需要 | 不需要 | 需要 | 需要 | Post-finetuning Prefix | Post-finetuning calibration |
| `unaware` | 需要 | 不需要 | 需要 | 需要 | Post-finetuning Prefix | Post-finetuning calibration |
| `phase_aware` | 需要 | 需要 | **不执行** | **不执行** | **Pre-finetuning Prefix** | **ANN-training calibration** |
| `gif_aware` | 需要 | 需要 | **不执行** | **不执行** | **Pre-finetuning Prefix** | **ANN-training calibration** |

其中：

```text
phase_aware:
Pre-finetuning Prefix
        +
ANN-training calibration
        ↓
Phase + common Clip 静态 ANN-aware fine-tuning
        ↓
final ANN
        ↓
不重新发现 Prefix
不重新 calibration
        ↓
Phase / GIF / MTN SNN 全部继续使用训练前同一套 Prefix 与同一套 calibration
```

```text
gif_aware:
Pre-finetuning Prefix
        +
ANN-training calibration
        ↓
GIF + common Clip 静态 ANN-aware fine-tuning
        ↓
final ANN
        ↓
不重新发现 Prefix
不重新 calibration
        ↓
Phase / GIF / MTN SNN 全部继续使用训练前同一套 Prefix 与同一套 calibration
```

这里“同一套 calibration”必须是**同一个 state bundle，而不是训练后重新估计一套参数**。因此：

- `phase_aware` 最终 Phase SNN 使用的 `tau / v0 / base / T` 必须与 ANN-aware training 前生成的 `phase_state.pt` 完全相同。
- `gif_aware` / `phase_aware` 最终 GIF SNN 使用 ANN-training calibration 中同一个 `gif_state.pt`。
- 最终 MTN SNN 同样复用 ANN-training calibration 中同一个 `mtn_state.pt`。
- ANN-training calibration 中存在的 `clip_state.pt` 继续保留给 ANN-aware training 使用，但 Phase/GIF/MTN SNN conversion 和 deployment **不得加载或执行它**。

`vanilla` / `unaware` 没有 activation replacement，因此最终 SNN 仍然按 final ANN 重新生成 Post-finetuning Prefix 与 Post-finetuning conversion calibration。

---

# 2. 建立统一的 mode-aware protocol helper

修改：

```text
snn2/config.py
```

至少新增以下等价语义的 helper，具体函数名可以调整，但整个项目不得再到处手写 `ann_mode` 分支：

```python
AWARE_ANN_MODES = {"phase_aware", "gif_aware"}

def is_aware_ann_mode(cfg) -> bool:
    ...

def requires_pre_finetuning_prefix(cfg) -> bool:
    # unaware / phase_aware / gif_aware -> True
    # vanilla -> False
    ...

def requires_ann_training_calibration(cfg) -> bool:
    # phase_aware / gif_aware -> True
    # vanilla / unaware -> False
    ...

def requires_post_finetuning_artifacts(cfg) -> bool:
    # vanilla / unaware -> True
    # phase_aware / gif_aware -> False
    ...

def conversion_reuses_ann_training_artifacts(cfg) -> bool:
    # phase_aware / gif_aware -> True
    ...

def conversion_prefix_enabled(cfg) -> bool:
    # aware -> ann_training.prefix_enabled
    # vanilla/unaware -> post_finetuning.prefix_enabled
    ...

def conversion_calibration_stage(cfg) -> str:
    # aware -> "ann_training"
    # vanilla/unaware -> "post_finetuning"
    ...

def final_evaluation_prefix_artifact_stage(cfg) -> str:
    # aware -> "ann_training"
    # vanilla/unaware -> "post_finetuning"
    ...
```

`evaluation.prefix_enabled` 继续作为最终 evaluation 的 Prefix on/off 开关保留，不能删除原来的无 Prefix ablation 能力；但当它为 `true` 时：

```text
phase_aware / gif_aware -> 从 Pre-finetuning Prefix 目录读取
vanilla / unaware       -> 从 Post-finetuning Prefix 目录读取
```

## 2.1 resolve_config / validate_config

修改 `resolve_config()`：

- `vanilla`
  - `rotation.enabled = false`
  - `replacement.train_mode = none`
  - `ann_training.prefix_enabled = false`
  - Post-finetuning rediscovery / recalibration 保持开启。
- `unaware`
  - Rotation 开启。
  - `replacement.train_mode = none`
  - 使用 Pre-finetuning Prefix。
  - 不要求 ANN-training calibration。
  - Post-finetuning rediscovery / recalibration 保持开启。
- `phase_aware` / `gif_aware`
  - Rotation 开启。
  - 使用 Pre-finetuning Prefix。
  - 使用 ANN-training calibration。
  - `post_finetuning.rediscover_prefix = false`
  - `post_finetuning.recalibrate_sites = false`
  - `post_finetuning.post_finetuning_recalibration = false`
  - `post_finetuning.prefix_enabled` 不再作为 conversion 的 Prefix 来源；建议 resolved config 中直接设为 `false`，避免误导。

修改 `validate_config()`，不要再无条件要求所有 mode：

```text
post_finetuning.rediscover_prefix == true
post_finetuning.recalibrate_sites == true
post_finetuning_recalibration == true
```

改成按 mode 校验：

```text
vanilla / unaware:
    上述三项必须 true

phase_aware / gif_aware:
    上述三项必须 false
```

同时确保：

```text
vanilla 不依赖 Pre-finetuning Prefix
unaware 依赖 Pre-finetuning Prefix，但不依赖 ANN-training calibration
phase_aware / gif_aware 同时依赖 Pre-finetuning Prefix 和 ANN-training calibration
```

同步修改：

```text
configs/experiment_matrix.yaml
tests/test_generated_configs.py
```

---

# 3. Phase τ calibration 改成与 SpikingLLM 普通 Phase 完全一致

这是本次最重要的数值修改之一。

参考 SpikingLLM：

```python
tensor = tensor.view(-1, hidden_dim).abs().detach()
comming_max = torch.max(tensor, dim=0)[0]

if key_name in act_stat:
    act_stat[key_name] = 0.99 * act_stat[key_name] + 0.01 * comming_max
else:
    act_stat[key_name] = comming_max
```

然后：

```python
x = quantized_item_stat.reshape(-1, group_size)
xmax = x.amax([-1], keepdim=True)
tau = xmax
```

## 3.1 `snn2/stats.py`

当前 `abs_max/value_min/value_max` 的 global extreme 统计继续保留，因为 GIF、MTN、common Clip 仍然需要它们。

**另外新增 Phase 专用 EMA statistic，不得用 global abs max 代替。**

每个 site 保存：

```text
phase_ema_abs_max
phase_ema_updates
```

更新公式必须严格为：

对于第 `i` 次 calibration forward 的 activation `x_i`：

\[
m_i[c] = \max_{\text{该次 forward 中除 channel 外的全部维度}} |x_i[...,c]|
\]

第一次：

\[
e_1[c] = m_1[c]
\]

之后：

\[
e_i[c] = 0.99 e_{i-1}[c] + 0.01 m_i[c]
\]

最终保存：

```text
phase_ema_abs_max = e_N
```

要求：

- EMA factor 固定为 `0.99`，与 SpikingLLM 一致。
- 主实验 calibration 当前 `batch_size=1`；继续保持这一点，避免 batch 聚合改变 EMA 更新次数 / 顺序。
- 如果未来用 distributed calibration，不能对 EMA statistic 做 `MAX/SUM all_reduce` 伪装成等价结果。当前主流程是单进程 `python scripts/calibrate_sites.py`，建议检测 world size > 1 时直接报错，避免静默产生错误 Phase τ。
- Site 5 是 variable-length Softmax site：只更新当前实际存在的 channel；未出现的位置保持原状态。最终默认 `group_size=-1` 时仍归约成一个 scalar τ。

在 `statistics.pt` / `statistics_summary.json` 中记录：

```text
phase_tau_statistic: spikingllm_ema_channel_abs_max
phase_tau_ema_factor: 0.99
phase_ema_updates: ...
```

## 3.2 `snn2/calibration.py`

把 Phase state 的构造从 GIF/MTN 所用的 global min/max 中拆出来。

建议抽出：

```python
def build_phase_state(statistics, cfg):
    ...
```

Phase τ 不再使用：

```python
absolute = max(abs(minimum), abs(maximum))
phase_tau = absolute
```

而使用：

```python
phase_stat = statistics["phase_ema_abs_max"]

# 与 SpikingLLM:
# quantized_item_stat.reshape(-1, group_size).amax(-1)
# 等价
phase_tau = _group_reduce(
    phase_stat,
    reduction_group_size,
    "max",
).float()
```

最终：

```python
phase_state = {
    ...
    "tau": phase_tau,
    "v0": 0.5 * phase_tau * 2 ** (-T),
    "base": 2.0,
    "T": T,
    ...
    "tau_calibration": "spikingllm_ema_channel_abs_max",
    "tau_ema_factor": 0.99,
}
```

注意：

- GIF、MTN 和 Clip 原来的 `minimum / maximum / absolute` 统计逻辑不因本次修改被删除。
- common Clip 中的 `phase_bound` 要改成使用新的 `phase_tau`。
- 不改变 Phase neuron forward / temporal 的编码公式。

---

# 4. Phase/GIF aware：训练后不再重新生成 Prefix / calibration

修改：

```text
scripts/discover_prefix.py
scripts/calibrate_sites.py
snn2/modeling.py
snn2/artifacts.py
snn2/conversion.py
snn2/state_validation.py
snn2/training.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
```

## 4.1 `scripts/discover_prefix.py`

`--stage pre_finetuning`：

```text
unaware / phase_aware / gif_aware -> 允许
vanilla                           -> 不需要，建议直接报清晰错误
```

因为三个 rotated mode 使用同一个 fused Base，所以仍共享同一份：

```text
_shared/.../rotated_prefix/pre_finetuning_prefix/
```

`--stage post_finetuning`：

```text
vanilla / unaware       -> 允许
phase_aware / gif_aware -> 禁止
```

aware mode 若误执行，报错明确说明：

```text
This ANN mode reuses the pre-finetuning Prefix for SNN conversion;
do not rediscover a post-finetuning Prefix.
```

## 4.2 `scripts/calibrate_sites.py`

`--stage ann_training`：

```text
phase_aware / gif_aware -> 允许
vanilla / unaware       -> 不需要，建议拒绝
```

`phase_aware` 与 `gif_aware` 的 ANN-training calibration 仍可按当前 shared policy 只生成一次。

`--stage post_finetuning`：

```text
vanilla / unaware       -> 允许
phase_aware / gif_aware -> 禁止
```

aware mode 若误执行必须报清晰错误，不能生成一套随后被误用的新 neuron state。

## 4.3 mode-aware artifact root

在 `ArtifactLayout` 或统一 protocol helper 中提供：

```python
conversion_prefix_dir
conversion_site_dir
conversion_calibration_dir
```

语义：

```text
phase_aware / gif_aware:
    conversion_prefix_dir = ann_training_prefix_dir
    conversion_site_dir   = ann_training_site_dir

vanilla / unaware:
    conversion_prefix_dir = post_finetuning_prefix_dir
    conversion_site_dir   = post_finetuning_site_dir
```

`snn_conversion_dir(neuron)` 的 `prefix_enabled_ture/false` 后缀必须基于：

```python
conversion_prefix_enabled(cfg)
```

而不能继续固定读取：

```python
post_finetuning.prefix_enabled
```

---

# 5. 用 hash 保证 aware training 与 SNN conversion 真正使用“同一个”工件

仅复用相同路径还不够，因为路径内容可能在训练后被重新覆盖。

## 5.1 calibration manifest 必须绑定 state 文件 hash

修改 calibration materialization。

在 `calibration_state_manifest.json` 的每个 site 中保存至少：

```json
{
  "state_sha256": {
    "phase": "...",
    "gif": "...",
    "mtn": "...",
    "clip": "..."
  }
}
```

无 Clip bundle 则无 `clip` 项。

后面第 8 节新增的 final RMSNorm Phase state 也要保存 SHA-256。

`validate_site_state_bundle()` 必须重新计算文件 hash，并与 manifest 比较。

这样：

```text
calibration_state_manifest.json SHA
```

可以传递地锁定真正的 neuron state 文件，而不是只锁定 summary。

## 5.2 `snn2/training.py`

在 ANN training 完成时，向 `training_result.json` 写入训练实际使用的前置工件 provenance。

所有使用 Pre-finetuning Prefix 的 mode 记录：

```text
ann_training_prefix_root
ann_training_prefix_state_sha256
ann_training_prefix_kv_sha256
ann_training_prefix_token_ids
```

`phase_aware` / `gif_aware` 额外记录：

```text
ann_training_calibration_root
ann_training_calibration_manifest_sha256
```

## 5.3 `snn2/conversion.py`

aware mode conversion 必须：

1. 读取 `training_result.json`。
2. 读取 Pre-finetuning Prefix 当前文件 hash。
3. 读取 ANN-training calibration manifest 当前 hash。
4. 与训练时保存的 hash 对比。
5. 任意不一致直接拒绝 conversion。

这一步保证：

```text
Phase/GIF aware ANN training
与
最终 Phase/GIF/MTN SNN conversion
```

确实使用训练时已经固定的同一套 Prefix / neuron state。

---

# 6. aware conversion 复用含 Clip 的 ANN-training bundle，但 SNN 不得使用 Clip

当前 ANN-training calibration 中：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

必须继续保留。

因此 `conversion.py` 不能再把：

```text
conversion root 中存在 clip_state.pt
```

一律判断为非法。

改成 mode-aware validation：

### aware mode

允许 calibration source：

```text
purpose = ann_training_calibration
state_profile = ann_training_with_common_clip
common_clip_required = true
```

并允许目录中存在：

```text
clip_state.pt
```

但 conversion / deployment 必须仍然：

```text
required state = 当前 neuron 自己的 state
```

即：

```text
deploy_phase -> 只加载 phase_state.pt
deploy_gif   -> 只加载 gif_state.pt
deploy_mtn   -> 只加载 mtn_state.pt
```

不得加载 `clip_state.pt`。

### vanilla / unaware

仍要求：

```text
purpose = post_finetuning_conversion_calibration
state_profile = snn_conversion_without_clip
common_clip_required = false
```

且 Post-finetuning bundle 中存在 `clip_state.pt` 继续视为错误。

Conversion metadata 新增：

```text
calibration_source_stage: ann_training | post_finetuning
prefix_source_stage: pre_finetuning | post_finetuning
reused_ann_training_artifacts: true | false
post_finetuning_recalibration: false | true
snn_clip_applied: false
```

---

# 7. Prefix + Softmax：整个 Softmax tensor 都经过 neuron

修改：

```text
snn2/model_integration.py
snn2/temporal_model.py
snn2/calibration.py
```

当前有 Prefix 时，代码把 Softmax 输出拆成：

```text
prefix columns  -> bypass Site 5
current columns -> Site 5
```

必须删除这一特殊处理。

## 7.1 ANN / collect 路径

在 `snn2_eager_attention_forward()` 中：

Softmax 完成后，无论有没有 Prefix，都直接：

```python
weights = controller.apply(layer_index, 5, weights)
```

不得再：

```python
prefix_weights = ...
current_weights = controller.apply(...)
torch.cat(...)
```

因此：

```text
[Prefix key positions + 当前 token key positions]
```

对应的完整 attention probability tensor 全部进入 Site 5。

## 7.2 SNN deployment 路径

`deployment_attention_forward()` 中：

```python
weight_increment = temporal_softmax(...)
flat_weights = from_temporal(weight_increment)
```

之后直接对完整：

```python
flat_weights
```

执行 Site 5 neuron，再恢复 temporal layout。

不得按 `past_length` 把 Prefix columns 绕过 Phase/GIF/MTN neuron。

## 7.3 Site 5 calibration capacity

当前：

```python
max_channels_by_site={5: max_seq_length}
```

在 Prefix 也进入 Site 5 后不够。

改成：

```text
site5_max_channels =
    data.max_seq_length
    + actual_prefix_length
```

其中 `actual_prefix_length` 必须从本次 calibration 实际注入的 fixed Prefix KV 中得到，而不是写死。

同时 Site 5 saliency 的 Prefix 部分也不要再裁掉；Site 5 的 activation/statistics 与实际 forward 必须覆盖完整 Softmax 最后一维。

注意：

- Site 3 / Site 4 对 cached Prefix K/V 的现有特殊处理本次不改。
- 本次只修正 Softmax Site 5。

---

# 8. Embedding temporal 编码改成 SpikingLLM Phase embedding

修改：

```text
snn2/model_integration.py
```

当前：

```text
[x, 0, 0, ..., 0]
```

必须改成 SpikingLLM 的：

\[
x_t = x / T,\quad t=0,\ldots,T-1
\]

即：

```text
[x/T, x/T, ..., x/T]
```

当前 `temporal_forward()` 已经把相同 `input_ids` repeat 成 `T*B`，因此 embedding hook 中：

```python
temporal = output.reshape(T, B, ...)
temporal = temporal / T
return temporal.reshape_as(output)
```

即可得到与 SpikingLLM `phaseSnnEmbedding` 相同的行为。

增加 invariant test：

\[
\sum_t embedding_t = original\_embedding
\]

并检查所有 temporal frame 相同。

不要改变 Prefix KV 的现有：

```text
KV / T then repeat T
```

策略，该部分已经与 SpikingLLM 对齐。

---

# 9. 最终 RMSNorm 后增加 Phase neuron

SpikingLLM 普通 Phase baseline 会给最终 RMSNorm output 也安装 Phase neuron。

本项目保持“每层 10 sites”不变，因此：

**不得把最终 RMSNorm 定义成 Site 11。**

使用一个独立的 global Phase state：

```text
<calibration_site_root>/
└── _global/
    └── final_rmsnorm/
        ├── statistics.pt
        ├── statistics_summary.json
        └── phase_state.pt
```

## 9.1 calibration

在 `collect_site_statistics()` 中给：

```python
parts.final_norm
```

注册一个额外 statistics hook。

它只需要收集 Phase 所需 statistic，尤其是：

```text
phase_ema_abs_max
```

使用第 3 节完全相同的 SpikingLLM EMA 规则。

`materialize_calibration_states()` 额外生成：

```text
_global/final_rmsnorm/phase_state.pt
```

该 state 的：

```text
tau
v0
base
T
```

与普通 site Phase state 使用相同构造函数。

该 global state：

- 不计入 `SITE_COUNT`。
- 不改变 `expected_sites_per_layer=10`。
- 不生成 common Clip。
- 不生成 GIF / MTN state。

`calibration_state_manifest.json` 必须记录：

```text
global_states.final_rmsnorm.phase_state_path
global_states.final_rmsnorm.phase_state_sha256
```

## 9.2 deployment

在 `SiteController` 增加专门的：

```python
apply_final_norm_phase(...)
```

只在：

```text
mode == deploy_phase
```

时加载：

```text
_global/final_rmsnorm/phase_state.pt
```

并执行 Phase temporal neuron。

`install_model_integration()` 中 final RMSNorm 已经先通过 temporal RMSNorm；在其输出之后增加 hook：

```text
temporal RMSNorm
        ↓
final RMSNorm Phase neuron
        ↓
lm_head
```

本次不要把该 auxiliary final-norm Phase neuron加入 10-site topology，也不要给它加 common Clip。

对于：

```text
deploy_gif
deploy_mtn
```

final RMSNorm 后不额外执行该 Phase neuron。

---

# 10. Artifact schema / temporal policy 必须升版本，拒绝旧工件

本次修改同时改变：

- Phase τ 的统计定义；
- Prefix Softmax Site 5 行为；
- Embedding temporal 编码；
- final RMSNorm Phase deployment；
- conversion calibration source protocol。

旧 calibration / conversion 不能静默复用。

修改：

```text
snn2/temporal_ops.py
```

建议至少：

```text
SITE_STATE_FORMAT_VERSION: 2 -> 3
CALIBRATION_MANIFEST_FORMAT_VERSION: 3 -> 4
CONVERSION_METADATA_FORMAT_VERSION: 4 -> 5
TEMPORAL_IMPLEMENTATION_VERSION: 2 -> 3
TEMPORAL_IMPLEMENTATION: sparse_llm_temporal_v2 -> sparse_llm_temporal_v3
```

新增并写入 `temporal_policy_metadata()`：

```text
embedding_temporal_policy = uniform_embedding_divide_by_T
softmax_prefix_neuron_policy = full_softmax_tensor_including_prefix
phase_final_norm_policy = phase_neuron_after_final_temporal_rmsnorm
phase_tau_calibration = spikingllm_ema_channel_abs_max
phase_tau_ema_factor = 0.99
```

同步修改：

```text
configs/experiment_matrix.yaml
snn2/config.py
snn2/state_validation.py
snn2/conversion.py
scripts/verify_artifacts.py
tests
```

所有旧 v2/v3/v4 对应工件必须给出明确错误，要求重新 materialize，而不是兼容运行。

---

# 11. `modeling.py` 的 Prefix stage 解析

修改：

```text
snn2/modeling.py
```

`prefix_ids_for_stage(..., stage="final_evaluation")` 和
`prefix_key_values_for_stage(..., stage="final_evaluation")`
不能再永远读取：

```text
post_finetuning_prefix_dir
```

改成：

```text
phase_aware / gif_aware:
    evaluation.prefix_enabled == true
        -> ann_training_prefix_dir

vanilla / unaware:
    evaluation.prefix_enabled == true
        -> post_finetuning_prefix_dir
```

Conversion 的 Prefix validator 也改成 generic validator，不要继续命名 / 实现为只接受 Post-finetuning Prefix。

建议把：

```python
validate_post_finetuning_prefix(...)
```

重构成：

```python
validate_conversion_prefix(cfg, layout, ...)
```

并在 metadata 中保存实际 prefix source stage。

---

# 12. Evaluation 脚本改成 mode-aware calibration root

修改：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

当前 SNN controller 固定使用：

```python
layout.post_finetuning_site_dir
```

改成 mode-aware：

```python
conversion_site_dir(cfg, layout)
```

即：

```text
phase_aware / gif_aware -> ann_training_site_dir
vanilla / unaware       -> post_finetuning_site_dir
```

输出 metadata 也必须反映真实来源：

```text
calibration_root
calibration_source_stage
prefix_source_stage
reused_ann_training_artifacts
post_finetuning_recalibration
```

最终 ANN evaluation：

- aware mode：若 `evaluation.prefix_enabled=true`，使用 Pre-finetuning Prefix。
- vanilla/unaware：若 `evaluation.prefix_enabled=true`，使用 Post-finetuning Prefix。

SNN evaluation 同样如此。

保留已有：

```text
prefix_enabled_ture
prefix_enabled_false
```

结果目录隔离规则，不改变历史拼写。

---

# 13. `verify_artifacts.py` 改成按 mode 验证

当前 verifier 无条件要求 Post-finetuning Prefix / Post-finetuning calibration，必须重构。

最终要求：

## vanilla

必须有：

```text
final ANN
Post-finetuning Prefix（启用时）
Post-finetuning conversion calibration
Phase/GIF/MTN conversion
evaluation
```

不要求：

```text
Pre-finetuning Prefix
ANN-training calibration
```

## unaware

必须有：

```text
Rotation
Pre-finetuning Prefix（ANN training 使用）
final ANN
Post-finetuning Prefix
Post-finetuning conversion calibration
Phase/GIF/MTN conversion
evaluation
```

不要求 ANN-training calibration。

即使 shared rotated policy 目录因为 phase-aware 实验物理上存在 ANN-training calibration，`unaware` 也不能把它当自己的 conversion dependency。

## phase_aware / gif_aware

必须有：

```text
Rotation
Pre-finetuning Prefix
ANN-training calibration
final ANN
training_result 中固定的 Prefix/calibration hash
Phase/GIF/MTN conversion
evaluation
```

**不得要求**：

```text
Post-finetuning Prefix
Post-finetuning conversion calibration
```

aware conversion validation 必须确认：

```text
conversion metadata 的 prefix/calibration hash
==
training_result 固定的 pre-finetuning / ANN-training hash
```

---

# 14. 必须修改的当前文档

## 14.1 `实验执行总结.md`

这是必改项。

至少重写 Step 4–10 与依赖图：

### Step 4

Pre-finetuning Prefix 只为：

```text
unaware / phase_aware / gif_aware
```

准备；三个 rotated mode 对同一 model-task pair 共用同一份 rotated Pre-finetuning Prefix。

不再为 vanilla 生成 Pre-finetuning Prefix。

### Step 5

ANN-training calibration 只服务：

```text
phase_aware / gif_aware
```

同一 model-task pair 生成一次共享 bundle。

明确写：

```text
unaware 不执行、不依赖 ANN-training calibration
vanilla 不执行、不依赖 ANN-training calibration
```

### Step 6

训练仍是 ANN fine-tuning。

明确说明：

```text
phase_aware / gif_aware 只在 replacement site 局部静态模拟 neuron conversion；
层间不展开真实 temporal T，因此仍属于 ANN fine-tuning。
```

### Step 7

Post-finetuning Prefix 只对：

```text
vanilla / unaware
```

执行。

`phase_aware / gif_aware` 明确跳过。

### Step 8

Post-finetuning conversion calibration 只对：

```text
vanilla / unaware
```

执行。

`phase_aware / gif_aware` 明确跳过，并写明：

```text
最终所有 Phase/GIF/MTN conversion 继续复用 ANN-training calibration。
```

### Step 9

final ANN evaluation 的 Prefix 来源：

```text
phase_aware/gif_aware -> Pre-finetuning Prefix
vanilla/unaware       -> Post-finetuning Prefix
```

`evaluation.prefix_enabled=false` 时仍允许无 Prefix ablation。

### Step 10

Conversion 分两类命令依赖：

```text
phase_aware / gif_aware:
    Step 4 Pre-finetuning Prefix
    + Step 5 ANN-training calibration

vanilla / unaware:
    Step 7 Post-finetuning Prefix
    + Step 8 Post-finetuning calibration
```

明确写：

```text
aware ANN-training bundle 虽然含 clip_state.pt，
SNN conversion / deployment 只加载当前 neuron state，不加载 Clip。
```

重画附录依赖树。

## 14.2 `README.md`

当前 README 中“每个 final ANN checkpoint 都重新做 Post-finetuning Prefix + calibration”的描述会变成错误，必须同步修改为上述 mode-aware protocol。

## 14.3 `AGENTS.md`

保留原来的“Clip 只用于 ANN aware fine-tuning”规则，并补充一句：

```text
phase_aware/gif_aware 的 SNN conversion 允许复用含 clip_state.pt 的 ANN-training calibration bundle；
Clip 文件仅因 ANN-aware training 而存在，SNN conversion/deployment 不得加载或执行它。
```

## 14.4 `代码结构总结.md`

遵守现有 AGENTS 规则。若本次修改后某个文件的一句话功能描述已经不准确，才同步更新；不要加入额外章节。

`docs/history/` 不需要回写旧历史方案。

---

# 15. 测试要求

至少覆盖以下测试。

## 15.1 Phase τ

在 `tests/test_statistics.py` / `tests/test_calibration_profiles.py` 增加：

1. 人工给出多次 activation，手算：

\[
e_1=m_1,\quad
e_2=0.99e_1+0.01m_2,\quad ...
\]

断言保存的 `phase_ema_abs_max` 完全一致。

2. `group_size=-1` 时：

```text
tau = max(final EMA channel vector)
```

3. 构造一个第一批存在巨大 outlier、后续正常值的例子，断言 Phase τ 不再等于全局历史最大值。

4. `phase_state` 的：

```text
v0 = 0.5 * tau * 2^-T
base = 2
T = config T
```

不变。

## 15.2 Prefix + Softmax

增加测试：

- 有 Prefix 的 ANN/collect forward，Site 5 收到最后一维长度：

```text
prefix_length + current_key_length
```

- deployment Phase/GIF/MTN 同样对整个 Softmax tensor执行 Site 5。
- 不再存在 Prefix columns bypass Site 5。
- calibration Site 5 capacity 为 `max_seq_length + actual_prefix_length`。

## 15.3 Embedding

增加：

```text
每个 temporal frame == original_embedding / T
sum_t(frame) == original_embedding
```

并断言不再是：

```text
[x, 0, ..., 0]
```

## 15.4 final RMSNorm Phase

增加测试：

- `_global/final_rmsnorm/phase_state.pt` 被 materialize。
- 不计入 `SITE_COUNT`，每层仍严格 10 sites。
- `deploy_phase` 在 final temporal RMSNorm 后执行 Phase neuron。
- `deploy_gif` / `deploy_mtn` 不执行这个 Phase neuron。
- final-norm Phase 不加载 Clip。

## 15.5 mode-aware protocol

在：

```text
tests/test_post_finetuning_protocol.py
tests/test_generated_configs.py
tests/test_conversion_metadata.py
tests/test_evaluation_paths.py
```

覆盖四种 mode：

```text
vanilla
unaware
phase_aware
gif_aware
```

断言第 1 节表格完全成立。

尤其：

- aware `discover_prefix --stage post_finetuning` 被拒绝。
- aware `calibrate_sites --stage post_finetuning` 被拒绝。
- vanilla/unaware conversion 仍要求 Post-finetuning artifacts。
- phase/gif aware conversion 只要求 Pre-finetuning Prefix + ANN-training calibration。
- unaware 不要求 ANN-training calibration。
- vanilla 不要求 Pre-finetuning Prefix。
- aware final evaluation Prefix 从 Pre-finetuning 目录读取。

## 15.6 provenance

增加测试：

- 训练后篡改 Prefix KV -> aware conversion 必须失败。
- 训练后篡改任意 `phase_state.pt/gif_state.pt/mtn_state.pt` -> manifest/hash validation 必须失败。
- aware conversion metadata 中必须记录 `reused_ann_training_artifacts=true`。
- vanilla/unaware 为 `false`。

## 15.7 Clip 回归保护

保留并增强：

```text
phase_aware ANN -> Phase + Clip
gif_aware ANN   -> GIF + Clip

deploy_phase -> Phase only
deploy_gif   -> GIF only
deploy_mtn   -> MTN only
```

aware conversion 虽然允许 source 目录存在 `clip_state.pt`，但测试必须证明 SNN controller 从未实例化 / 加载 Clipper。

## 15.8 不得误改的行为

增加或保留 regression test：

- ANN `phase_aware/gif_aware` training output 仍是普通 ANN tensor shape，不出现 `[T,B,...]` 的层间真实 temporal propagation。
- `SITE_COUNT == 10`。
- Site 9 仍存在。
- Prefix KV 仍采用 `/T + repeat T`。
- temporal QK/PV/Softmax/MLP 主公式不被本次修改破坏。

最后运行：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```

---

# 16. 对现有工件的重跑边界

因为 Phase τ / temporal policy / calibration manifest / conversion descriptor 都升版本，旧 conversion state 不允许继续复用。

## 16.1 `phase_aware` / `gif_aware`

必须从 **ANN-training calibration** 开始重跑，并重新 ANN fine-tuning：

```text
Pre-finetuning Prefix
    ↓
ANN-training calibration（重新生成，新 EMA τ + final norm state）
    ↓
ANN phase_aware / gif_aware fine-tuning（重新训练）
    ↓
直接 conversion
    ↓
SNN evaluation
```

**不要执行**：

```text
Post-finetuning Prefix
Post-finetuning conversion calibration
```

Rotation 与固定数据 manifest 若 hash / config 未变化可复用；Pre-finetuning Prefix 算法本身未修改时也可复用，但训练结果必须记录并锁定它的 hash。

Qwen3-1.7B `phase_aware` 示例流程应变成：

```bash
CFG=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml

# Rotation / Pre-finetuning Prefix 已存在且通过验证时可复用
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training

torchrun \
  --standalone \
  --nproc_per_node="$NGPU" \
  scripts/train_ann.py \
  --config "$CFG"

# 这里不运行 post_finetuning Prefix / calibration

for NEURON in phase gif mtn; do
  python scripts/convert_snn.py \
    --config "$CFG" \
    --neuron "$NEURON"
done
```

然后正常执行 TL;DR evaluation。

## 16.2 `vanilla` / `unaware`

最终 SNN conversion 仍需要重新生成新版：

```text
Post-finetuning Prefix
Post-finetuning conversion calibration
conversion descriptor
```

Post-finetuning calibration 必须重跑，因为新的 Phase τ schema 与 final RMSNorm Phase state 已改变。

如果已有 vanilla ANN 是按旧协议“带 Pre-finetuning Prefix”训练的，则它不符合本次新 protocol；正式比较时需要按 `vanilla` 不使用 Pre-finetuning Prefix 的新配置重新训练。

---

# 17. 完成标准

只有同时满足以下条件才算本次修改完成：

1. Phase τ 数值来源已经从 global extreme max 改成 SpikingLLM 的 `0.99 EMA of per-forward channel abs-max`。
2. `phase_aware/gif_aware` 训练结束后不会再生成或依赖 Post-finetuning Prefix / Post-finetuning calibration。
3. aware conversion 的 Prefix 与 calibration hash 与训练时完全一致。
4. vanilla/unaware 仍使用 Post-finetuning Prefix + Post-finetuning calibration。
5. unaware 不依赖 ANN-training calibration；vanilla 不依赖 Pre-finetuning Prefix。
6. Prefix 存在时整个 Softmax tensor（包含 Prefix columns）全部经过 Site 5 neuron。
7. Embedding temporal 输入为 `[x/T, ..., x/T]`。
8. Phase SNN 在最终 RMSNorm 后执行额外 Phase neuron，但 `SITE_COUNT` 仍等于 10。
9. Site 9 保留。
10. ANN aware fine-tuning 仍是静态局部 replacement，不变成 full-temporal SNN fine-tuning。
11. ANN common Clip 保留；SNN deployment 不加载 / 不执行 Clip。
12. 旧 calibration / conversion artifact 会因版本不匹配被明确拒绝。
13. `实验执行总结.md`、README、AGENTS 与代码行为一致。
14. `pytest -q` 全部通过。
