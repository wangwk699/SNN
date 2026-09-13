# SNN 项目：新增 Block-wise SNN-aware Neuron Calibration 修改方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 本方案基于当前 `main` 分支代码结构制定。审阅时仓库树对应 commit：`22e47b84f8bbc48179fa0b21cc504d91574ff149`。  
> 本文档是**独立实施文档**：服务器上的 Codex 不需要任何额外对话上下文，应仅依据本文档完成代码修改、测试和验证。

---

## 1. 修改目标

当前项目对 Phase / GIF / MTN 三类神经元参数的 Stage A calibration，核心逻辑是：

1. 模型中 replacement site 处于 `collect` 模式；
2. 所有 replacement site 都只记录 ANN activation / saliency，不改变前向 tensor；
3. calibration samples 对完整网络做一次 ANN forward；
4. 由同一套 `statistics.pt` 同时生成：
   - `phase_state.pt`
   - `gif_state.pt`
   - `mtn_state.pt`
5. 因而第 \(l\) 层神经元参数统计使用的是纯 ANN trajectory：
   \[
   X_l^{ANN}.
   \]

本轮修改需要增加可选的 **SparseLLM-style block-wise SNN-aware calibration**。

当某类神经元配置为启用该模式时，第 \(l\) 个 Transformer block 的神经元参数应基于：

\[
X_l^{(n)}
=
F_{l-1}^{(n)}
\circ \cdots \circ
F_0^{(n)}(X_0),
\qquad
n \in \{\text{phase},\text{gif},\text{mtn}\},
\]

也就是：

- 前面已经完成 calibration 的 Transformer blocks 使用该神经元的**真实 Temporal SNN forward**；
- 当前 Transformer block 尚未插入该神经元，只收集本 block 全部 replacement sites 的统计量；
- 当前 block 全部 site 的 state 一次性生成完成后，才把整个当前 block 切换为对应 Temporal SNN；
- 用该 block 的 SNN 输出作为下一个 block 的输入；
- **严格按 Transformer block 逐层推进，禁止 site-wise “Site 1 校准→插入→Site 2 校准→插入”**。

本轮修改主要是为了改善 `gif_aware` ANN fine-tuning，但三个神经元均提供独立开关。

---

# 2. 已确认、不得再修改的设计决策

以下设计已经确认，实施时不要再次询问。

## 2.1 严格使用 block-wise 语义

采用 SparseLLM 的 block-wise 思路：

```text
Block 0:
  current block sites = collect only
  -> 得到 Block 0 neuron state
  -> Block 0 真实 Temporal SNN forward
  -> 得到 X_1^SNN

Block 1:
  input = X_1^SNN
  current block sites = collect only
  -> 得到 Block 1 neuron state
  -> Block 1 真实 Temporal SNN forward
  -> 得到 X_2^SNN

...

Block L:
  input = 前面所有 blocks 的 SNN 输出
  current block sites = collect only
  -> 得到 Block L neuron state
  -> Block L 真实 Temporal SNN forward
```

同一个 Transformer block 内所有 replacement sites 必须在**同一次 collect pass** 中统计，当前 block 内不能提前启用某个已统计完成的 site。

---

## 2.2 新增三个独立 bool 参数

在 `configs/experiment_matrix.yaml` 的 `calibration:` 下新增：

```yaml
calibration:
  phase_previous_layers_snn: false
  gif_previous_layers_snn: false
  mtn_previous_layers_snn: false
```

含义：

- `false`：保持当前纯 ANN single-pass statistics；
- `true`：该神经元使用 block-wise previous-layers-SNN calibration。

三个字段必须是真正 YAML bool：

```yaml
true
false
```

禁止接受字符串：

```yaml
"true"
"false"
```

默认值全部为 `false`，以保证旧配置缺少这些字段时仍可通过 `resolve_config()` 得到旧行为。

---

## 2.3 三条 trajectory 完全独立

三个开关可以任意组合。

例如：

```yaml
phase_previous_layers_snn: false
gif_previous_layers_snn: true
mtn_previous_layers_snn: false
```

则：

- Phase state 来自 ANN common trajectory；
- GIF state 来自 GIF Temporal SNN trajectory；
- MTN state 来自 ANN common trajectory。

绝对禁止因为 GIF=true，就让 Phase/MTN 使用 GIF-SNN-conditioned activation。

形式上：

\[
\mathcal T_{\rm phase}
\neq
\mathcal T_{\rm gif}
\neq
\mathcal T_{\rm mtn}
\]

三类 state 必须能分别追溯到自己的 activation trajectory。

---

## 2.4 GIF=true 时整个 GIF state 都从 GIF-SNN-conditioned trajectory 重算

不能只重算 low/high quantization scale/zero。

当：

```yaml
gif_previous_layers_snn: true
```

时，以下内容全部必须来自 GIF-SNN-conditioned trajectory：

- activation `value_min/value_max`
- `low_scale`
- `low_zero`
- `high_scale`
- `high_zero`
- saliency statistics
- salient mask
- Site 1 的 Q/K/V role-specific masks
- Site 7 的 gate/up role-specific masks
- 其它所有依赖 GIF statistics 的字段

禁止：

```text
ANN saliency mask
+
SNN-conditioned low/high scale
```

这种混合 provenance。

Site 5 / GIF identity sites 仍保持项目现有特殊策略不变。

---

## 2.5 三个开关同时作用于两个真正的 Stage A calibration

三个 bool 同时控制：

1. `ann_training` Stage A calibration；
2. `post_finetuning` Stage A conversion calibration。

`vanilla_analysis` 必须保持纯 ANN analysis-only single-pass，不受三个开关影响。

即：

```text
ann_training:
    obey three previous_layers_snn flags

post_finetuning:
    obey three previous_layers_snn flags

vanilla_analysis:
    always legacy ANN single-pass
```

---

## 2.6 Stage B common Clip 数学规则完全不变

现有：

```python
build_clip_state(
    phase_state,
    gif_state,
    mtn_state,
    phase_T=...,
    mtn_T=...,
)
```

以及 Phase / GIF / MTN interval intersection 规则保持不变。

如果三类 state 来自不同 trajectory，就直接使用各自 state 生成 common Clip。

例如：

```yaml
phase_previous_layers_snn: false
gif_previous_layers_snn: true
mtn_previous_layers_snn: false
```

Stage B 自然使用：

```text
Phase state from ANN
+
GIF state from GIF-SNN-conditioned trajectory
+
MTN state from ANN
→ current build_clip_state()
```

不修改 Clip 数学定义。

---

# 3. 当前代码中需要保留的关键行为

实施时不要破坏以下已有 contract。

## 3.1 当前 Stage A

入口：

```text
scripts/calibrate_sites.py
```

当前 Stage A 调用：

```python
collect_site_statistics(...)
```

并使用：

```python
controller = SiteController(mode="collect")
```

当前 `SiteController.apply()` 在 `collect` 下：

```python
self.statistics.update(...)
return x
```

因此现有 calibration 是不改变网络 activation 的完整 ANN forward。

---

## 3.2 当前统计规则保持不变

不要修改 Phase / MTN 的统计数学规则。

Phase：

```text
EMA per-channel abs-max
→ configured grouping
→ group max
→ clamp
→ tau
```

MTN：

```text
同一 EMA/group-max
→ × 2
→ clamp
→ base_scale
```

GIF：

```text
direct min-max
→ low/high qparams
→ operator-aware saliency
→ salient mask
```

本轮只改变：

> “统计量来自哪一条 activation trajectory”

不要额外引入 MSE threshold search，也不要把 SparseLLM 的 MSE-init 机制照搬进来。

---

# 4. 非常重要：Sequential calibration 统计的是 temporal trajectory 对应的 logical activation

当前项目真实 Temporal neuron 的行为决定了 calibration 不应把 `[T,B,...]` 每个 timestep 当成互不相关的 ANN sample。

Phase：

```python
PhaseSurrogate.temporal(incoming):
    encode(incoming.sum(dim=0), ...)
```

GIF：

```python
StaticGIF.temporal(incoming):
    x = incoming.sum(dim=0)
    ...
```

因此在 sequential calibration 的 current block 中，replacement site 收到 temporal increment：

```text
[T, B, ...]
```

时，用于保持现有 Phase/GIF/MTN statistics contract 的“逻辑 activation”定义为：

\[
X_{\rm logical}
=
\sum_{t=0}^{T-1} X_t.
\]

实现要求：

- current block 的非 replacement 运算仍走真实 temporal algebra；
- current block 的 replacement site **不运行 neuron**；
- 但在 site 处，把 temporal input reshape 为 `[T,B,...]`；
- 对 timestep 求和得到 `[B,...]`；
- 把这个 logical activation 交给现有 `StatisticsStore.update()`；
- 原始 temporal increments 原样继续向当前 block 后续运算传播。

这样：

1. 原有 Phase/MTN EMA 统计公式不用改；
2. GIF min/max 仍是 logical activation 的 static quantization range；
3. 统计维度继续与现有 ANN statistics 一致；
4. 当前 block 内不会因为 T 被错误当作额外 batch rows 而改变 EMA 定义。

建议在 `SiteController` 中加入统一 helper，例如：

```python
def logical_activation_for_calibration(self, x: torch.Tensor) -> torch.Tensor:
    if not self.sequential_calibration_active:
        return x
    return to_temporal(x, self.temporal_steps).sum(dim=0)
```

禁止在不同 site 各自手写不同 reshape/sum 规则。

---

# 5. GIF saliency 在 sequential temporal calibration 下的统一定义

GIF=true 时 saliency 也必须来自 GIF-SNN-conditioned trajectory，但现有 saliency 公式是针对 logical ANN operator 定义的。

因此 temporal calibration 下必须使用：

> **由 temporal increments 重构得到的 logical tensor，再套用当前项目已有的 SpikeLLM-style saliency 公式。**

不要改变 saliency 数学定义，只改变其输入 tensor 来源。

统一 helper：

```python
logical = to_temporal(x, GIF_LOCAL_STEPS).sum(dim=0)
```

然后继续调用现有：

- `_linear_score(...)`
- QK key saliency
- PV value saliency
- Site 6 o_proj consumer saliency
- Site 10 down_proj consumer saliency

具体要求：

### Site 1 Q/K/V

对进入 q/k/v linear 的 temporal activation 先求 timestep sum，再使用现有：

```python
_linear_score(logical_input, weight)
```

分别记录 q/k/v role。

### Site 3 K

对 temporal query / key 分别重构 logical Q/K，再运行当前：

```text
qk = Q @ K^T
key_score = K * (qk^T @ Q)
```

的 FP64 规则。

### Site 4 V

对 temporal attention probability / V 重构 logical tensor，再运行当前 PV saliency 规则。

### Site 6

对进入 `o_proj` 的 temporal merged activation 做 timestep sum，再使用现有 linear-consumer FP32 saliency。

### Site 7 gate/up

分别对 gate/up branch input 的 logical activation 使用当前 `_linear_score()`。

### Site 10

对 `down_proj` input 的 logical activation使用当前 `record_down_proj_saliency()` 数学规则。

必须继续保持当前：

- linear saliency FP32；
- QK/PV saliency FP64；
- role-specific policies；
- saliency source/rule metadata。

---

# 6. 推荐的总体 Stage A 架构

为了兼顾兼容性、诊断能力和三条独立 trajectory，**不要删除现有 ANN common statistics**。

即使有一个或多个 `previous_layers_snn=true`，Stage A 仍先生成当前项目已有的 ANN common baseline：

```text
layer_xxx/site_xx/statistics.pt
```

然后仅对 `true` 的神经元额外生成 target-specific statistics。

推荐目录：

```text
sites/
  layer_000/
    site_01/
      statistics.pt                 # 永远保留：ANN common baseline

      phase_statistics.pt           # 仅 phase_previous_layers_snn=true 时存在
      gif_statistics.pt             # 仅 gif_previous_layers_snn=true 时存在
      mtn_statistics.pt             # 仅 mtn_previous_layers_snn=true 时存在

      phase_state.pt
      gif_state.pt
      mtn_state.pt

      statistics_summary.json
      phase_statistics_summary.json
      gif_statistics_summary.json
      mtn_statistics_summary.json
```

对于 Final RMSNorm：

```text
sites/_global/final_rmsnorm/
    statistics.pt
    phase_statistics.pt     # phase=true 时
    mtn_statistics.pt       # mtn=true 时
    phase_state.pt
    mtn_state.pt
```

GIF Final RMSNorm 继续 identity，不生成 GIF global state。

---

# 7. State 使用哪份 statistics 的规则

定义 helper：

```python
def neuron_statistics_source(
    site_dir: Path,
    neuron: str,
    cfg: dict,
) -> Path:
    if cfg["calibration"][f"{neuron}_previous_layers_snn"]:
        return site_dir / f"{neuron}_statistics.pt"
    return site_dir / "statistics.pt"
```

然后：

```text
phase_state.pt
← phase_statistics.pt if phase=true
← statistics.pt       if phase=false

gif_state.pt
← gif_statistics.pt   if gif=true
← statistics.pt       if gif=false

mtn_state.pt
← mtn_statistics.pt   if mtn=true
← statistics.pt       if mtn=false
```

建议把当前：

```python
build_site_states(statistics, cfg)
```

拆开为明确的三类 builder：

```python
build_phase_state(statistics, cfg)
build_gif_state(statistics, cfg)
build_mtn_state(statistics, cfg)
```

`build_phase_state()` / `build_mtn_state()` 当前已有。

把 `build_site_states()` 中 GIF 部分抽出为：

```python
build_gif_state(statistics, cfg)
```

再新增：

```python
build_site_states_from_sources(
    *,
    phase_statistics,
    gif_statistics,
    mtn_statistics,
    cfg,
)
```

禁止再假定三类 state 一定来自同一个 `statistics.pt`。

---

# 8. Stage A 执行流程

## 8.1 `vanilla_analysis`

完全保留旧逻辑：

```text
ANN single pass
→ statistics.pt
→ analysis only
```

不生成 sequential target statistics。

---

## 8.2 `ann_training` / `post_finetuning`

### Step 1：始终运行 legacy ANN common pass

保持当前 `collect_site_statistics()` 的数值行为。

目的：

- 为所有 `false` neuron 提供 state；
- 保留 ANN baseline diagnostic statistics；
- 支持 true/false mixed configurations；
- 便于 regression test。

如果三个 bool 都为 `false`：

```text
直接走旧路径
```

必须做到 state tensor 与修改前代码一致。

### Step 2：对每一个 true neuron 分别运行独立 sequential trajectory

伪代码：

```python
targets = []

if calibration.phase_previous_layers_snn:
    targets.append("phase")

if calibration.gif_previous_layers_snn:
    targets.append("gif")

if calibration.mtn_previous_layers_snn:
    targets.append("mtn")

for neuron in targets:
    collect_blockwise_snn_conditioned_statistics(
        model=model,
        neuron=neuron,
        ...
    )
```

三次运行必须互相 reset：

- controller module cache；
- statistics store；
- temporal runtime state；
- hidden-state cache；
- prefix dynamic cache；
- final norm collector。

绝对禁止前一个 target 的 temporal output 成为下一个 target 的输入。

---

# 9. Block-wise sequential calibration engine

建议新增独立模块：

```text
snn2/blockwise_calibration.py
```

不要继续把所有逻辑堆进 `snn2/calibration.py`。

建议主要 API：

```python
@torch.no_grad()
def collect_blockwise_snn_conditioned_statistics(
    model,
    controller,
    tokenizer,
    calibration_raw,
    cfg,
    prefix_key_values,
    site_root,
    *,
    neuron: Literal["phase", "gif", "mtn"],
) -> dict:
    ...
```

---

## 9.1 不允许使用 PhaseSurrogate / StaticGIF ANN forward 代替前层 SNN

`previous_layers_snn=true` 的语义是：

```text
真实 Temporal Phase
真实 Temporal GIF
真实 Temporal MTN
```

必须复用当前 deployment temporal math。

禁止：

```text
PhaseSurrogate.forward()
StaticGIF.forward()
```

作为 previous-layers-SNN trajectory。

---

## 9.2 SiteController 增加 calibration temporal mode

当前字符串判断：

```python
controller.mode.startswith("deploy_")
```

分散在：

- `model_integration.py`
- `prefix_cache.py`
- temporal RMSNorm
- attention backend
- embedding hook
- 其它 temporal path

不要简单增加大量：

```python
or mode.startswith("calibration_")
```

建议给 `SiteController` 增加显式属性/方法：

```python
@property
def temporal_execution_enabled(self) -> bool:
    ...

@property
def sequential_calibration_active(self) -> bool:
    ...

@property
def sequential_calibration_neuron(self) -> str | None:
    ...
```

并逐步把：

```python
mode.startswith("deploy_")
```

替换成语义 helper。

建议 controller 支持至少三种行为：

```text
legacy collect
sequential collect for current block
sequential deploy for calibrated block
```

例如内部状态：

```python
self.calibration_neuron: str | None
self.calibration_block_index: int | None
self.calibration_collect_current_block: bool
```

不要污染正式 SNN evaluation 的 `set_deployment()` 行为。

---

## 9.3 current block collect pass

对于当前 block \(l\)：

```text
input:
    temporal output from blocks 0 ... l-1

non-site operations:
    temporal execution

replacement sites:
    collect logical activation
    return original temporal increments unchanged

current block neuron:
    NOT active
```

同一 block 内所有 sites 必须一起 collect。

---

## 9.4 current block state materialization

收集完 calibration set 上当前 block 的统计后：

```text
Phase target:
    write phase_statistics.pt
    build only phase_state.pt

GIF target:
    write gif_statistics.pt
    build only gif_state.pt

MTN target:
    write mtn_statistics.pt
    build only mtn_state.pt
```

必须在当前 block SNN propagation 之前完成对应 state。

---

## 9.5 current block deployment pass

state 生成后，对同一份 current-block input 再跑一次 block：

```text
non-site:
    temporal execution

sites:
    target temporal neuron enabled

output:
    temporal increments
```

输出缓存为下一 block 输入。

这一步必须严格使用新生成的当前 block state。

---

# 10. Hidden-state caching：必须采用 SparseLLM-style block cache，避免 O(L^2) full-model rerun

不要为了实现方便，在每个 block 都从 token IDs 重新完整跑一遍：

```text
Block 0 calibration: full model prefix
Block 1 calibration: rerun Block 0 + Block 1
Block 2 calibration: rerun Block 0 + Block 1 + Block 2
...
```

这种实现虽然语义可以勉强做到，但会变成约 \(O(L^2)\) 计算，不接受作为最终实现。

必须采用 block-wise hidden-state propagation。

建议实现：

### A. bootstrap first-block input

使用 first-layer catcher / pre-hook：

1. 让模型执行 embedding、position preparation、prefix injection、causal mask preparation；
2. 在第一个 decoder layer 真正执行之前捕获：
   - `hidden_states`
   - `attention_mask`
   - `position_ids`
   - `cache_position`
   - `position_embeddings`（若当前 Transformers 版本传入）
   - 其它 decoder layer 必需 kwargs；
3. 抛出内部专用 exception 中止 full model forward；
4. 把 first-block input offload 到 CPU cache。

Sequential target 启动时 temporal embedding policy 必须生效：

```text
uniform_embedding_divide_by_T
```

所以 first-block hidden input 必须已经是：

```text
[T * B, L, D]
```

的 time-major flattened layout。

### B. per-block cache

对每个 block：

```text
CPU cached current inputs
→ move sample/batch to current layer device
→ collect pass
→ materialize current state
→ deployment pass
→ output offload CPU
→ replace cache
```

任何时候都不要把所有 layers 的 activation 同时留在 GPU。

---

# 11. Prefix KV 的处理

当前 `install_prefix_kv_forward()` 会根据 deployment temporal steps：

- logical batch size；
- temporal steps；
- fixed Prefix K/V；
- uniform `KV / T`；

构造 temporal prefix cache。

Block-wise direct layer forward 时不能依赖 model-level wrapper 自动重新创建 cache。

应在 `snn2/prefix_cache.py` 提供一个**公开 helper**，复用当前 `_fresh_dynamic_cache()` 逻辑，例如：

```python
def fresh_prefix_dynamic_cache(
    prefix_key_values,
    *,
    logical_batch_size: int,
    temporal_steps: int | None,
    device,
):
    ...
```

或者更合适的公共 API。

要求：

- 每个 sample / block forward 使用 fresh cache；
- 禁止复用已经被上一轮 attention update 修改过的 mutable `DynamicCache`；
- prefix-disabled 时保持 `None`；
- temporal prefix policy 必须继续是：
  ```text
  uniform_kv_divide_by_T
  ```

不要改变 prefix semantics。

---

# 12. Final RMSNorm 的 sequential calibration

当前项目：

- Phase deployment：Final RMSNorm 后有 Phase temporal neuron；
- MTN deployment：Final RMSNorm 后有 MTN temporal neuron；
- GIF deployment：Final RMSNorm 后 identity。

因此：

## phase_previous_layers_snn=true

所有 decoder blocks 都完成 Phase temporal propagation 后：

```text
final temporal hidden
→ temporal RMSNorm
→ reconstruct logical output
→ phase_statistics.pt
→ phase_state.pt
```

## mtn_previous_layers_snn=true

同理：

```text
final temporal hidden
→ temporal RMSNorm
→ reconstruct logical output
→ mtn_statistics.pt
→ mtn_state.pt
```

## gif_previous_layers_snn=true

Final RMSNorm GIF 仍为 identity：

```text
不生成 global gif_state
```

当前项目已有：

```text
_global/final_rmsnorm/phase_state.pt
_global/final_rmsnorm/mtn_state.pt
```

结构继续保留。

---

# 13. 一个关键的 provenance 修正：Stage A 不再总是与 T/K 无关

当前代码 manifest 中声明：

```python
"stage_a_parameter_independence": [
    "phase.T",
    "mtn.T",
    "mtn.K",
]
```

这在 legacy ANN single-pass 下成立。

但是当：

```yaml
phase_previous_layers_snn: true
```

时，Phase trajectory 明确依赖：

```text
phase.T
```

当：

```yaml
mtn_previous_layers_snn: true
```

时，MTN trajectory 明确依赖：

```text
mtn.T
mtn.K
mtn.threshold_factor
```

因此必须修改 provenance contract。

建议 top-level manifest：

```json
{
  "calibration_trajectory": {
    "phase": {
      "previous_layers_snn": true,
      "source": "sequential_temporal_phase",
      "phase_T": 4
    },
    "gif": {
      "previous_layers_snn": false,
      "source": "ann_common"
    },
    "mtn": {
      "previous_layers_snn": true,
      "source": "sequential_temporal_mtn",
      "mtn_T": 4,
      "mtn_K": 6,
      "mtn_threshold_factor": 0.75
    }
  }
}
```

当 bool=false 时，不需要把对应 runtime parameter 视为 statistics dependency。

不要继续无条件声称：

```text
Stage A independent of phase.T / mtn.T / mtn.K
```

---

# 14. State 自身也必须记录 trajectory provenance

建议在：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
```

增加：

```python
"previous_layers_snn": bool
"calibration_trajectory": str
```

当 true 时还记录对应 runtime signature。

Phase：

```python
"calibration_phase_T": cfg["phase"]["T"]
```

MTN：

```python
"calibration_mtn_T": cfg["mtn"]["T"]
"calibration_mtn_K": cfg["mtn"]["K"]
"calibration_mtn_threshold_factor": cfg["mtn"]["threshold_factor"]
```

GIF：

```python
"calibration_gif_temporal_steps": GIF_LOCAL_STEPS
```

目的：

- state 文件即使被手工移动，也能被 validator 拒绝错误 runtime；
- ANN training / SNN conversion 不仅依赖目录路径正确；
- provenance 可审计。

相应修改：

### `PhaseSurrogate.__init__`

若：

```python
state["previous_layers_snn"] is True
```

必须验证：

```python
state["calibration_phase_T"] == T
```

### `MultiThresholdNeuron.__init__`

若 true，验证：

```python
calibration_mtn_T == T
calibration_mtn_K == K
calibration_mtn_threshold_factor == threshold_factor
```

GIF fixed T=2，校验其 recorded temporal steps 与 `GIF_LOCAL_STEPS` 一致。

---

# 15. Config 修改

修改：

```text
configs/experiment_matrix.yaml
snn2/config.py
scripts/materialize_configs.py（仅如测试/生成逻辑需要）
configs/generated/*.yaml（重新生成，不手工逐个编辑）
```

## 15.1 experiment_matrix.yaml

三个 experiment base config 的 `calibration:` 全部加入：

```yaml
phase_previous_layers_snn: false
gif_previous_layers_snn: false
mtn_previous_layers_snn: false
```

如果当前实验目标是立刻测试 Tulu-3 `gif_aware`，不要在代码实现阶段偷偷把默认改成 true。

先保持 matrix 中默认 false。

之后用户可单独把 Tulu-3 配置改为：

```yaml
gif_previous_layers_snn: true
```

进行实验。

---

## 15.2 resolve_config()

加入 backward-compatible defaults：

```python
cal = cfg.setdefault("calibration", {})
cal.setdefault("phase_previous_layers_snn", False)
cal.setdefault("gif_previous_layers_snn", False)
cal.setdefault("mtn_previous_layers_snn", False)
```

---

## 15.3 validate_config()

三个字段必须 `type(value) is bool`。

建议：

```python
for neuron in ("phase", "gif", "mtn"):
    key = f"{neuron}_previous_layers_snn"
    value = cfg["calibration"].get(key)
    if not isinstance(value, bool):
        raise ValueError(
            f"calibration.{key} must be true or false"
        )
```

注意 Python：

```python
isinstance(1, bool) == False
```

可接受；不要把整数 0/1 当 bool。

---

# 16. Artifact path：三个 bool 必须全部进入路径

新增统一 helper，例如：

```python
def calibration_trajectory_dirname(cfg: dict[str, Any]) -> str:
    ...
```

基础部分固定包含三个 bool：

```text
phase_previous_layers_snn_false_
gif_previous_layers_snn_true_
mtn_previous_layers_snn_false
```

建议完整格式：

```text
phase_previous_layers_snn_false_gif_previous_layers_snn_true_mtn_previous_layers_snn_false
```

不要使用历史 `ture` 拼写；新增字段统一使用正确 `true/false`。

---

# 17. 当 sequential=true 时，相关 temporal runtime 也必须进入 calibration path

仅三个 bool 仍不足以避免 artifact 冲突。

例如：

```yaml
phase_previous_layers_snn: true
phase:
  T: 4
```

和：

```yaml
phase_previous_layers_snn: true
phase:
  T: 8
```

会产生不同 previous-layer trajectory。

因此 dirname 必须条件性追加：

## Phase=true

```text
_phase_T_4
```

## GIF=true

GIF T 当前固定为 2，建议显式加入：

```text
_gif_temporal_steps_2
```

这样未来 GIF temporal policy 变化不会复用旧目录。

## MTN=true

必须加入：

```text
_mtn_T_4_mtn_K_6_mtn_threshold_factor_0.75
```

建议对 float 使用稳定格式函数，避免 `0.750000` / `0.75` 不一致。

最终示例：

```text
phase_previous_layers_snn_false_
gif_previous_layers_snn_true_
mtn_previous_layers_snn_false_
gif_temporal_steps_2
```

或：

```text
phase_previous_layers_snn_true_
gif_previous_layers_snn_false_
mtn_previous_layers_snn_true_
phase_T_4_
mtn_T_4_
mtn_K_6_
mtn_threshold_factor_0.75
```

---

# 18. 哪些路径必须隔离

## 18.1 ann_training calibration

当前：

```text
.../ann_training_calibration/
  prefix_enabled_*/
  calibration_group_size_*_num_samples_*/
```

修改为：

```text
.../ann_training_calibration/
  prefix_enabled_*/
  calibration_group_size_*_num_samples_*/
  <calibration_trajectory_dirname>/
```

或者把 trajectory dirname 合并到 variant dirname。

---

## 18.2 post_finetuning conversion calibration

同样加入 trajectory dirname：

```text
.../post_finetuning/conversion_calibration/
  prefix_enabled_*/
  calibration_group_size_*_num_samples_*/
  <calibration_trajectory_dirname>/
```

---

## 18.3 aware ANN training path

`phase_aware` / `gif_aware` ANN training 会读取 Stage A/Stage B artifact，因此其 run path 也必须体现 trajectory。

在 aware run variant 中加入：

```text
<calibration_trajectory_dirname>
```

这样不同 calibration state 训练得到的 ANN checkpoint 不会覆盖。

---

## 18.4 vanilla / unaware ANN checkpoint

vanilla / unaware ANN training 本身不使用 replacement neuron state。

不要因为 calibration flag 改变而重复训练完全相同的 vanilla/unaware ANN checkpoint。

因此不要求把 trajectory 加到 vanilla/unaware ANN checkpoint root。

但是它们后续 SNN conversion/evaluation 会使用不同 calibration artifact，因此：

```python
ArtifactLayout.snn_dir()
```

对非-aware mode 必须加入 trajectory dirname，避免不同 conversion/eval 相互覆盖。

---

## 18.5 不应改变的共享目录

以下 artifact 不依赖三个 bool，不要复制：

- raw selected calibration data；
- calibration sample manifest；
- rotation state；
- Pre-finetuning Prefix discovery；
- Prefix KV；
- canonical preprocessing data。

尤其：

```text
calibration_data_dir
```

继续只由 `num_samples` 等数据采样参数决定。

---

# 19. Stage B Clip path

Stage B 已经位于对应 Stage A calibration root 下：

```text
ann_training_calibration/.../clip_profiles/...
```

因此只要 Stage A root 已含 trajectory dirname，Clip 自然隔离。

无需再次修改 Clip 数学 rule。

但 `clip_profile_manifest.json` 必须复制/引用 Stage A trajectory metadata，并继续记录：

```text
state_a_phase_sha256
state_a_gif_sha256
state_a_mtn_sha256
```

保证 mixed trajectory 可审计。

---

# 20. `calibration_provenance()` 修改

在：

```text
snn2/calibration.py
```

的 `calibration_provenance()` 增加：

```python
"phase_previous_layers_snn": ...
"gif_previous_layers_snn": ...
"mtn_previous_layers_snn": ...
"calibration_trajectory": ...
```

`ann_training` / `post_finetuning` 使用真实配置。

`vanilla_analysis` 的 effective trajectory 必须明确写：

```text
ANN-only
```

即使 config 中三个 bool 有 true，也不能让 vanilla_analysis 改行为。

建议同时保存：

```python
"requested_previous_layers_snn": {...}
"effective_previous_layers_snn": {...}
```

其中 vanilla_analysis：

```python
effective = {
    "phase": False,
    "gif": False,
    "mtn": False,
}
```

---

# 21. Statistics / manifest schema

当前一个 `statistics.pt` 同时对应三个 state 的假设已经不再成立。

必须升级 manifest schema，禁止旧代码静默读取 mixed-trajectory artifact。

建议至少 bump：

```text
STATISTICS_FORMAT_VERSION
CALIBRATION_MANIFEST_FORMAT_VERSION
SITE_STATE_FORMAT_VERSION
```

如果 conversion metadata 新增 trajectory provenance，也 bump：

```text
CONVERSION_METADATA_FORMAT_VERSION
```

不要只增加字段而继续沿用旧 version。

---

# 22. `statistics_manifest.json`

继续表示 ANN common baseline：

```text
statistics.pt
```

不要把它伪装成 mixed trajectory。

另新增一个 Stage A trajectory summary，例如：

```text
trajectory_statistics_manifest.json
```

示例：

```json
{
  "phase": {
    "previous_layers_snn": false,
    "statistics_source": "statistics.pt"
  },
  "gif": {
    "previous_layers_snn": true,
    "statistics_source": "gif_statistics.pt",
    "temporal_steps": 2
  },
  "mtn": {
    "previous_layers_snn": false,
    "statistics_source": "statistics.pt"
  }
}
```

也可以直接合并到 `calibration_state_manifest.json`，但必须保证：

- 每个 state 的 source statistics 文件明确；
- source file SHA256 明确；
- trajectory 明确；
- runtime dependency 明确。

---

# 23. Site summary 必须增加 source provenance

当前 `calibration_state_manifest.json["sites"][key]` 记录：

```text
state_sha256
```

新增：

```json
"state_statistics_source": {
  "phase": {
    "file": "statistics.pt",
    "sha256": "...",
    "previous_layers_snn": false
  },
  "gif": {
    "file": "gif_statistics.pt",
    "sha256": "...",
    "previous_layers_snn": true
  },
  "mtn": {
    "file": "statistics.pt",
    "sha256": "...",
    "previous_layers_snn": false
  }
}
```

这样 Stage B / training / conversion 都能验证 state 的真实来源。

---

# 24. Training 必须验证配置与 Stage A provenance 一致

当前：

```text
snn2/training.py
```

已经通过：

```python
validate_site_state_bundle(...)
validate_clip_profile(...)
capture_training_artifact_provenance(...)
```

锁定 calibration artifact。

需要扩展这些验证。

`train_full_parameters()` 必须从 cfg 读取：

```python
cfg["calibration"]["phase_previous_layers_snn"]
cfg["calibration"]["gif_previous_layers_snn"]
cfg["calibration"]["mtn_previous_layers_snn"]
```

并确认 `calibration_state_manifest.json` 完全一致。

如果：

```text
config says gif=true
manifest says gif=false
```

必须直接报错，不能自动 fallback。

---

# 25. `training_result.json` 记录三个开关

在 `capture_training_artifact_provenance()` 和最终 training result 中增加：

```text
ann_training_phase_previous_layers_snn
ann_training_gif_previous_layers_snn
ann_training_mtn_previous_layers_snn
ann_training_calibration_trajectory_signature
```

以及 target=true 时对应 runtime dependencies。

后续：

```python
validate_recorded_training_artifact_provenance(...)
```

和 conversion reuse 都必须检查这些值。

---

# 26. Conversion provenance 修改

修改：

```text
snn2/conversion.py
```

当前 `validate_calibration()` 假定 site 下必须存在：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

`statistics.pt` 继续保留，因此该最低要求可继续存在。

但需要额外验证：

- true target 对应 `*_statistics.pt` 必须存在；
- false target 对应 target-specific statistics 最好禁止存在，避免 stale artifact；
- state manifest source hash 与实际文件一致；
- config flags 与 manifest 一致；
- conditional T/K runtime signature 一致。

`conversion_metadata.json` 应记录：

```text
phase_previous_layers_snn
gif_previous_layers_snn
mtn_previous_layers_snn
calibration_trajectory_signature
```

必要时 bump metadata version。

---

# 27. `state_validation.py` 修改

当前 Stage A validator 会递归禁止：

```text
phase_T
mtn_T
mtn_K
```

因为旧 Stage A 假定 runtime-independent。

新逻辑必须改为**条件约束**：

### flag=false

继续要求对应 state/statistics 不依赖 temporal runtime。

### phase=true

允许并要求记录：

```text
calibration_phase_T
```

并验证与 cfg 一致。

### mtn=true

允许并要求：

```text
calibration_mtn_T
calibration_mtn_K
calibration_mtn_threshold_factor
```

### gif=true

要求：

```text
calibration_gif_temporal_steps == GIF_LOCAL_STEPS
```

不要简单删除所有 runtime dependency validation。

---

# 28. `model_integration.py` 必须支持“temporal execution + collect sites”

当前代码把：

```text
temporal execution
```

几乎等同于：

```python
controller.mode.startswith("deploy_")
```

而本轮需要新的组合：

```text
non-site operators = temporal
sites             = collect / bypass
```

因此需要重构。

至少以下地方要改成 controller semantic property：

- attention temporal backend dispatch；
- RMSNorm temporal wrapper；
- embedding temporal policy；
- linear bias temporal policy；
- MLP temporal SiLU/Hadamard path；
- Prefix temporal KV 判断；
- regression recorder temporal flag。

目标是让以下状态合法：

```text
controller.temporal_execution_enabled == True
controller.site_behavior == "collect"
```

---

# 29. `temporal_model.py` 必须支持 calibration collector

`deployment_attention_forward()` 当前直接：

```python
controller.apply(...)
```

并不执行 `collect` 模式下已有的 GIF saliency 路径。

需要将 temporal attention path 扩展为同时支持：

```text
deployment
sequential calibration
```

建议重命名为更通用的：

```python
temporal_attention_forward(...)
```

或者保留名字但通过 controller capability dispatch。

要求：

- Site 2/3/4/5/6 在 sequential collect 时记录 logical activation；
- GIF target 时记录 logical saliency；
- collect 后返回原 temporal tensor，不运行 current block neuron；
- deployment 时保持现有输出 bit-for-bit。

---

# 30. 不得改变正式 SNN deployment 数值

本轮 calibration refactor 不允许改变已有：

```text
evaluate --neuron phase
evaluate --neuron gif
evaluate --neuron mtn
```

的 temporal forward 数学。

建议增加 regression：

```text
before change deployment reference
vs
new controller semantic-property refactor
```

在旧 all-false artifact 下，SNN forward 必须一致。

---

# 31. all-false 必须保持旧 Stage A 数值行为

当：

```yaml
phase_previous_layers_snn: false
gif_previous_layers_snn: false
mtn_previous_layers_snn: false
```

要求：

- calibration sample selection 不变；
- ANN single-pass order 不变；
- Phase EMA update order 不变；
- GIF min/max 不变；
- GIF saliency 不变；
- generated `phase_state.pt` tensor 值不变；
- generated `gif_state.pt` tensor 值不变；
- generated `mtn_state.pt` tensor 值不变；
- common Clip 数学结果不变。

允许因为 schema bump 导致 metadata/version/hash 改变，但核心 tensor 必须一致。

---

# 32. mixed flags 时的运行次数

本方案始终保留一遍 ANN common baseline。

额外 sequential trajectories 数量：

```python
N = sum(
    [
        phase_previous_layers_snn,
        gif_previous_layers_snn,
        mtn_previous_layers_snn,
    ]
)
```

所以：

```text
false false false:
    1 ANN common pass

false true false:
    1 ANN common pass
    + 1 GIF block-wise pass

true true false:
    1 ANN common pass
    + 1 Phase block-wise pass
    + 1 GIF block-wise pass

true true true:
    1 ANN common pass
    + 3 independent block-wise passes
```

三条 sequential pass 禁止共享 hidden-state trajectory。

---

# 33. Memory policy

Sequential calibration 面向 Llama-3-8B / Qwen3-8B。

不要把整个：

```text
num_samples × num_layers × seq_len × hidden
```

缓存下来。

只缓存“当前 block input”。

每完成一个 block：

```text
old current-block input
→ current block deploy
→ next-block input
→ free old input
```

建议 CPU offload。

每个 sample 的：

- hidden state；
- mask / position metadata；

可以单独缓存。

每层结束后：

```python
del ...
torch.cuda.empty_cache()
```

但不要在 inner tensor operation 中频繁调用 empty_cache。

---

# 34. Device-map 支持

当前 calibration 支持：

```yaml
calibration:
  device_map: auto
```

block-wise runner 必须兼容 model layers 分布在多 GPU。

不要假定所有 layer 在：

```text
cuda:0
```

处理当前 block 时：

```python
layer_device = next(layer.parameters()).device
```

将：

- hidden states；
- attention mask；
- position tensors；
- fresh prefix cache；

移动到正确设备。

输出再 offload CPU。

---

# 35. EMA 顺序

Phase/MTN 使用 order-dependent EMA：

```text
ema_factor = 0.99
```

因此 sequential calibration 仍必须：

- single process；
- calibration dataset order 与 legacy selection 一致；
- batch_size 必须继续为 1；
- 每个 sample 对每个 site 只执行一次 logical-statistics update。

不能通过并行不同 calibration sample 改变 EMA order。

---

# 36. GIF qparam 仍然 direct_min_max

不要因为参考 SparseLLM 就启用：

```yaml
gif:
  mse_scale_refinement: true
```

保持：

```yaml
scale_initialization: direct_min_max
mse_scale_refinement: false
runtime_quantization: static
```

本轮没有 MSE search。

---

# 37. Stage B 仍然只在 ann_training 生成 Clip profile

保持项目当前 A/B contract：

```text
ann_training:
    Stage A + Stage B

post_finetuning:
    Stage A only

vanilla_analysis:
    Stage A analysis only
```

不要借本次修改重新引入 post-finetuning Stage B。

---

# 38. 建议新增 helper API

建议在 `snn2/config.py`：

```python
def previous_layers_snn_enabled(cfg, neuron: str) -> bool:
    ...

def calibration_trajectory_config(cfg) -> dict:
    ...

def any_previous_layers_snn_enabled(cfg) -> bool:
    ...
```

在 `snn2/artifacts.py`：

```python
def calibration_trajectory_dirname(cfg) -> str:
    ...
```

在 `snn2/calibration.py`：

```python
def statistics_path_for_neuron(site_dir, neuron, cfg) -> Path:
    ...

def build_gif_state(statistics, cfg) -> dict:
    ...

def materialize_neuron_state(
    statistics,
    cfg,
    neuron,
) -> dict:
    ...
```

在 `snn2/controller.py`：

```python
@property
def temporal_execution_enabled(...)

def begin_sequential_calibration(...)

def begin_sequential_deployment(...)

def end_sequential_calibration(...)
```

命名可略有调整，但语义必须集中，避免 scattered bool logic。

---

# 39. 需要重点修改的文件

至少检查和修改：

```text
configs/experiment_matrix.yaml
configs/generated/*.yaml

scripts/materialize_configs.py
scripts/calibrate_sites.py

snn2/config.py
snn2/artifacts.py
snn2/calibration.py
snn2/stats.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
snn2/prefix_cache.py
snn2/neurons.py
snn2/state_validation.py
snn2/training.py
snn2/conversion.py
snn2/temporal_ops.py
```

建议新增：

```text
snn2/blockwise_calibration.py
```

并补充/新增对应 tests。

---

# 40. `scripts/calibrate_sites.py`

Stage A 入口不要增加新的用户 CLI flag。

行为必须只由 config 控制。

仍然使用：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase A
```

脚本内部读取：

```yaml
calibration.phase_previous_layers_snn
calibration.gif_previous_layers_snn
calibration.mtn_previous_layers_snn
```

并自动决定是否运行 sequential trajectories。

Stage B 命令不变。

---

# 41. ANN training 不新增 CLI flag

训练命令保持：

```bash
torchrun ... scripts/train_ann.py --config "$CFG"
```

训练通过：

```text
config
→ ArtifactLayout
→ Stage A manifest validation
→ correct phase/gif state
```

自动选取匹配 artifact。

不要要求用户手工传：

```text
--gif-snn-calibration
```

之类参数。

---

# 42. Tests：必须覆盖以下内容

## 42.1 Config

新增测试：

1. 三个字段缺失 → resolve 后全部 false；
2. true/false 合法；
3. `"true"`、`1`、`None` 非法；
4. generated configs 都包含三个字段。

---

## 42.2 Path isolation

遍历 8 种组合：

```text
FFF
FFT
FTF
FTT
TFF
TFT
TTF
TTT
```

确保 calibration path 全部不同。

另外：

- Phase=true 时 `phase.T` 不同 → calibration path 不同；
- MTN=true 时 `mtn.T/K/threshold_factor` 不同 → path 不同；
- 对应 flag=false 时，这些 runtime 参数不应无意义地改变 shared Stage A calibration path；
- GIF=true 时 path 包含固定 temporal policy signature。

---

## 42.3 all-false regression

构造 tiny model / fixture。

比较修改前等价 legacy collector 与新 all-false collector：

```text
phase tau
gif low/high scale/zero
gif masks
mtn base_scale
```

必须 tensor equal 或严格可解释的 bitwise equal。

---

## 42.4 Block-wise 语义

构造至少 3-block toy model。

令 Temporal fake neuron 明确改变输出。

验证：

```text
Block 0 statistics
    不受 previous layer neuron 影响

Block 1 statistics
    来自 Block 0 SNN output

Block 2 statistics
    来自 Block 0 + Block 1 SNN output
```

---

## 42.5 禁止 site-wise insertion

同一 toy block 放两个 sites：

```text
site A
site B
```

验证 current block collect 时：

```text
site B 看到的不是 “site A 已经 neuronized” 的 activation
```

而是 current block 所有 sites 都处于 collect/bypass。

---

## 42.6 Mixed trajectories

配置：

```yaml
phase: false
gif: true
mtn: false
```

验证：

```text
phase state == state built from statistics.pt
mtn state   == state built from statistics.pt
gif state   == state built from gif_statistics.pt
```

Phase/MTN 不得读取 GIF trajectory。

---

## 42.7 Independent true trajectories

例如：

```yaml
phase: true
gif: true
mtn: false
```

分别修改 Phase/GIF fake temporal neuron 输出，使 trajectories 不同。

验证：

```text
phase_statistics != gif_statistics
```

且执行顺序互换不会改变结果。

---

## 42.8 GIF saliency

GIF=true 时：

- `gif_statistics.pt` 中 saliency source 存在；
- Site 1 roles 恰好 Q/K/V；
- Site 7 roles 恰好 gate/up；
- mask 由 `gif_statistics.pt` saliency 计算；
- 修改 common `statistics.pt` saliency 不应改变 GIF state；
- 修改 `gif_statistics.pt` saliency 应改变 GIF mask。

---

## 42.9 Temporal logical aggregation

人工构造：

```text
T=2
x0
x1
```

验证 sequential collector 送入 `StatisticsStore` 的 activation 为：

```python
x0 + x1
```

而不是把 `x0`、`x1` 当两个独立 EMA update。

`phase_ema_updates` 对一个 calibration sample 仍增加 1。

---

## 42.10 Final RMSNorm

Phase=true：

```text
all Phase SNN blocks
→ temporal final RMSNorm
→ phase final statistics
```

MTN=true 同理。

验证 final state 不来自 ANN common statistics。

GIF 不应生成 global gif state。

---

## 42.11 Clip

给定相同三份 state：

```python
build_clip_state()
```

修改前后数值必须一致。

mixed trajectory 时只验证它读取最终三份 state，不增加新的数学分支。

---

## 42.12 Training provenance

制造：

```text
cfg gif=true
manifest gif=false
```

必须 fail fast。

同样验证：

```text
Phase calibration T mismatch
MTN calibration T/K/factor mismatch
```

必须 fail。

---

## 42.13 Conversion provenance

`convert_snn.py` / evaluation 必须拒绝：

- trajectory flags mismatch；
- target statistics missing；
- state source SHA mismatch；
- conditional runtime signature mismatch。

---

## 42.14 Deployment regression

controller temporal-mode refactor 后，已有：

```text
deploy_phase
deploy_gif
deploy_mtn
```

forward 必须保持数值不变。

特别关注：

- prefix temporal division；
- embedding `/T`；
- temporal RMSNorm；
- temporal attention；
- bias first timestep once；
- Site 5 GIF identity；
- Site 6 merged-head topology；
- Final RMSNorm policy。

---

# 43. Artifact stale-file policy

不同 trajectory 已通过 path 隔离，但同一路径 rerun 仍可能残留文件。

Stage A 开始时要验证目录。

若 manifest/version/trajectory 与当前 cfg 不一致：

```text
fail fast
```

不要静默复用。

在 target=false 的目录中发现：

```text
phase_statistics.pt
gif_statistics.pt
mtn_statistics.pt
```

对应不该存在的 target-specific stats 时，建议报 stale-artifact error，而不是忽略。

---

# 44. Manifest/hash 要求

以下文件全部需要 SHA provenance：

```text
statistics.pt
phase_statistics.pt（若存在）
gif_statistics.pt（若存在）
mtn_statistics.pt（若存在）

phase_state.pt
gif_state.pt
mtn_state.pt
```

Final RMSNorm 同理。

Stage B profile 继续锁定 Stage A state SHA。

Training result 继续锁定 Stage A + Stage B manifest SHA。

Conversion metadata 继续锁定 selected Stage A manifest SHA。

---

# 45. Tulu-3 / gif_aware 目标实验配置

代码完成并通过 tests 后，用户可以针对 Tulu-3 GIF-aware 单独设置：

```yaml
calibration:
  phase_previous_layers_snn: false
  gif_previous_layers_snn: true
  mtn_previous_layers_snn: false
```

含义：

```text
Phase state:
    ANN common statistics

GIF state:
    block-wise previous GIF-SNN-conditioned statistics

MTN state:
    ANN common statistics

Stage B Clip:
    current unchanged rule over the above three states

gif_aware ANN fine-tuning:
    loads GIF state from GIF-SNN trajectory
    and the corresponding Stage B Clip profile
```

这是本轮最主要的目标实验。

---

# 46. 推荐运行流程

修改完成后：

## 46.1 重新生成 configs

```bash
python scripts/materialize_configs.py
```

确认 generated YAML 均包含三个新字段。

---

## 46.2 准备 calibration data / rotation / prefix

沿用项目当前流程，不因本轮修改重复设计。

---

## 46.3 ANN-training Stage A

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase A
```

预期：

- 先有 ANN common `statistics.pt`；
- GIF=true 时逐 block 生成 `gif_statistics.pt`；
- 最终 `gif_state.pt` source 指向 GIF statistics。

---

## 46.4 ANN-training Stage B Clip

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase B
```

Clip 数学规则不变。

---

## 46.5 ANN fine-tuning

```bash
torchrun --standalone --nproc_per_node=$NGPU \
  scripts/train_ann.py \
  --config "$CFG"
```

训练启动前必须验证：

```text
cfg trajectory flags
==
Stage A manifest
==
Stage B provenance
```

---

## 46.6 Post-finetuning Stage A

重新发现 post-finetuning prefix 后：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning \
  --calibration-phase A
```

同样遵循三个 bool。

不要生成 post-finetuning Stage B。

---

# 47. Logging

Sequential calibration 日志至少记录：

```text
stage
target neuron
previous_layers_snn
block index
num blocks
num calibration samples
temporal steps
statistics output path
state output path
runtime seconds
```

例如：

```text
[gif sequential calibration]
block=17/32
T=2
samples=128
statistics=.../layer_017/site_*/gif_statistics.pt
```

不要输出每个 tensor 的完整内容。

---

# 48. 性能与正确性优先级

优先级：

1. 正确的 independent trajectories；
2. 正确的 block-wise temporal propagation；
3. 正确的 prefix / attention / RMSNorm temporal semantics；
4. state/provenance/path 不串；
5. GPU memory 可控；
6. 再考虑速度。

但最终实现不能采用明显的 O(L²) full-model-per-block 方案。

---

# 49. 不在本轮范围内的内容

不要顺手修改：

- GIF bit-width policy；
- GIF `high_qmax=30`；
- GIF local steps=2；
- Site 5 GIF identity；
- Site 2 all-low policy；
- Site 3/4 post-repeat-kv topology；
- Site 6 merged-head topology；
- Phase threshold equation；
- Phase base=2；
- Phase EMA factor=0.99；
- MTN ×2 base-scale statistics rule；
- common Clip mathematics；
- Prefix discovery algorithm；
- rotation algorithm；
- Tulu-3 train/eval dataset selection；
- lm-eval tasks；
- ANN teacher forcing；
- optimizer/scheduler；
- learning rate；
- ANN checkpointing policy。

本轮只实现 calibration trajectory 选择与相应 artifact/provenance plumbing。

---

# 50. Acceptance criteria

本轮修改只有同时满足以下条件才算完成：

- [ ] `experiment_matrix.yaml` 有三个 bool 参数；
- [ ] old config 缺字段默认 false；
- [ ] all-false 数值行为与当前代码一致；
- [ ] true 使用真实 Temporal SNN previous layers；
- [ ] strictly block-wise，不是 site-wise；
- [ ] 三种 neuron trajectory 互相独立；
- [ ] GIF=true 时 qparams + saliency + masks 全部重新统计；
- [ ] ann_training 和 post_finetuning 都遵循 flags；
- [ ] vanilla_analysis 永远 ANN-only；
- [ ] common Clip 数学规则完全不变；
- [ ] ANN common `statistics.pt` 继续保留；
- [ ] true target 有独立 `*_statistics.pt`；
- [ ] state manifest 明确记录每种 state 的 statistics source；
- [ ] sequential=true 时正确记录并隔离 temporal runtime dependency；
- [ ] artifact path 包含三个 bool；
- [ ] conditional Phase/MTN temporal parameters 进入 calibration path；
- [ ] aware training path 不会因 calibration trajectory 不同而覆盖；
- [ ] non-aware SNN conversion/eval path 不会覆盖；
- [ ] training fail-fast 验证 trajectory provenance；
- [ ] conversion fail-fast 验证 trajectory provenance；
- [ ] deployment temporal 数值未被 controller refactor 改坏；
- [ ] tests 全部通过；
- [ ] `pytest -q` 全部通过。

---

# 51. 最终建议的核心实现流程伪代码

```python
def run_stage_a(...):
    # --------------------------------------------------
    # 1. Legacy ANN common baseline: always keep it
    # --------------------------------------------------
    collect_ann_common_statistics(...)
    # writes statistics.pt

    if stage == "vanilla_analysis":
        return

    # --------------------------------------------------
    # 2. Independent optional SNN-aware trajectories
    # --------------------------------------------------
    for neuron in ("phase", "gif", "mtn"):
        if not previous_layers_snn_enabled(cfg, neuron):
            continue

        reset_sequential_runtime()

        current_inputs = bootstrap_temporal_first_layer_inputs(
            model=model,
            neuron=neuron,
            ...
        )

        for layer_idx, layer in enumerate(layers):
            # A. current block = temporal math, sites collect/bypass
            target_stats = collect_current_block_logical_statistics(
                layer=layer,
                temporal_inputs=current_inputs,
                neuron=neuron,
                ...
            )

            save_target_statistics(
                target_stats,
                filename=f"{neuron}_statistics.pt",
            )

            # B. materialize only target neuron state for current block
            materialize_target_state(
                neuron=neuron,
                stats=target_stats,
                cfg=cfg,
            )

            # C. rerun same current input with current block target SNN enabled
            next_inputs = propagate_current_block_temporal_snn(
                layer=layer,
                temporal_inputs=current_inputs,
                neuron=neuron,
                ...
            )

            free(current_inputs)
            current_inputs = next_inputs

        if neuron in ("phase", "mtn"):
            collect_and_materialize_target_final_norm_state(
                temporal_inputs=current_inputs,
                neuron=neuron,
                ...
            )

        free(current_inputs)

    # --------------------------------------------------
    # 3. Final state materialization / manifest
    # --------------------------------------------------
    # false target -> statistics.pt
    # true target  -> <neuron>_statistics.pt
    materialize_all_states_from_selected_sources(...)

    write_calibration_state_manifest_with_trajectory_provenance(...)
```

---

# 52. 最重要的实现约束总结

Codex 实施时请始终记住下面五句话：

1. **false = 当前 ANN single-pass；true = 前面 blocks 真实 Temporal SNN。**
2. **当前 block 所有 sites 一起 collect，整个 block state 生成后才插入 neuron。**
3. **Phase / GIF / MTN 三条 trajectory 永远独立。**
4. **GIF=true 时 scale/zero 与 saliency/mask 必须全部来自 GIF trajectory。**
5. **sequential=true 使 Stage A 对 temporal runtime 参数产生条件依赖，路径和 provenance 必须一起修正。**

不要只实现“多三个 YAML bool”，而漏掉第 5 点，否则不同 T/K 的 calibration artifact 会发生错误复用。
