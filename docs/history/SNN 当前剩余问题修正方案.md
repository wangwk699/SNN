# SNN 当前剩余问题修正方案

目标仓库：`https://github.com/wangwk699/SNN`，基于当前 `main`。

本次只修下面 3 项，不改已经完成的 Phase/MTN calibration、Embedding temporal encoding、Prefix KV decomposition、group granularity、10-site topology。

## 1. 修正 MTN activation-neuron operator count

修改：

```text
snn2/evaluation.py
tests/test_evaluation_paths.py
```

当前 MTN 仍按：

```python
base = num_hidden_layers * SITE_COUNT

if neuron == "phase":
    return base + 1

if neuron == "mtn":
    return base
```

但当前 topology 已经是：

```text
Phase SNN = 每层 10 sites + Final RMSNorm global Phase
MTN SNN   = 每层 10 sites + Final RMSNorm global MTN
GIF SNN   = 保持当前 GIF 计数逻辑，Final RMSNorm 无 GIF
```

因此改为：

```python
if neuron in {"phase", "mtn"}:
    return base + 1
```

28 层模型：

```text
Phase = 281
MTN   = 281
```

同步把测试中的：

```python
("mtn", 280)
```

改成：

```python
("mtn", 281)
```

确保 TL;DR 与 lm-eval 中以下指标不再少算 Final RMSNorm MTN：

```text
per_temporal_forward_activation_neuron_operators
activation_site_temporal_operator_calls
batched_activation_site_temporal_slots
```

## 2. 修正 Final RMSNorm evaluation metadata

至少检查：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
tests/...
```

删除/替换旧字段：

```python
"global_final_norm_phase_neuron_present": args.neuron == "phase"
```

推荐统一改为：

```python
"global_final_norm_replacement": (
    "phase_surrogate"
    if args.neuron == "ann" and controller.mode == "phase"
    else "temporal_phase"
    if args.neuron == "phase"
    else "temporal_mtn"
    if args.neuron == "mtn"
    else "identity"
),
"global_final_norm_clip_applied": False,
```

必须准确表达：

```text
Phase-aware ANN -> phase_surrogate
GIF-aware ANN   -> identity
Phase SNN       -> temporal_phase
MTN SNN         -> temporal_mtn
GIF SNN         -> identity
```

GIF 永远不得被记录为存在 Final RMSNorm GIF neuron。

`scripts/verify_artifacts.py` 必须同步验证新的 metadata；旧的或与实际 topology 不一致的 evaluation artifact 应直接报错。

## 3. 补 Final RMSNorm topology regression tests

至少直接覆盖以下 5 种情况。

### Phase-aware ANN

```text
ordinary Final RMSNorm
-> PhaseSurrogate.forward()
-> LM Head
```

断言：

- 加载 global `phase_state.pt`；
- 确实执行 `PhaseSurrogate.forward()`；
- `common_clip_enabled=True/False` 两种情况下，Final RMSNorm 均不执行 Clip。

### GIF-aware ANN

```text
Final RMSNorm -> identity
```

断言：

- 不加载 global GIF state；
- 不执行 GIF；
- 不执行 Clip。

### deploy_phase

断言：

```text
Final Temporal RMSNorm
-> temporal Phase neuron
```

### deploy_mtn

断言：

```text
Final Temporal RMSNorm
-> temporal MTN neuron
```

### deploy_gif

断言：

```text
Final Temporal RMSNorm
-> identity
```

不得存在 Final RMSNorm GIF neuron。

## 4. 同步 metadata / operator tests

增加断言：

```text
phase-aware ANN -> global_final_norm_replacement == "phase_surrogate"
GIF-aware ANN   -> "identity"
Phase SNN       -> "temporal_phase"
MTN SNN         -> "temporal_mtn"
GIF SNN         -> "identity"
```

所有模式都应满足：

```text
global_final_norm_clip_applied == False
```

operator count：

```text
L=28:
Phase = 281
MTN   = 281
```

GIF 保持当前项目已有计算规则。

## 5. 本次不要修改

不要顺手改动：

```text
Phase tau calibration
MTN base_scale calibration
[5e-4, 1e4] clamp
Embedding uniform divide-by-T
Prefix KV uniform divide-by-T
MTN/Phase group granularity
每层 10 个 activation replacement sites
Final RMSNorm global Phase/MTN state materialization
Stage B Clip 公式
```

这些部分当前实现已经正确。

## 6. 完成标准

- [ ] MTN operator count 包含 Final RMSNorm global MTN。
- [ ] 28 层模型 Phase/MTN 均为 `281`。
- [ ] evaluation metadata 正确区分 `phase_surrogate / temporal_phase / temporal_mtn / identity`。
- [ ] GIF ANN/SNN Final RMSNorm 均为 identity。
- [ ] Final RMSNorm Clip metadata 永远为 false。
- [ ] `verify_artifacts.py` 校验新的 metadata。
- [ ] 补齐 5 种 Final RMSNorm topology regression tests。
- [ ] 所有相关旧测试同步更新。
- [ ] `pytest -q` 全部通过。
::: ​​