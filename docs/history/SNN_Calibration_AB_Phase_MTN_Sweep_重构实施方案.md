# SNN 项目 Calibration A/B 两阶段重构与 Phase/MTN 参数 Sweep 支持实施方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 本文档按 2026-08-30 `main` 分支当前实现制定。  
> 目的：让部署在服务器上的 Codex **在没有任何对话上下文的情况下，仅依据本文档即可完成全部代码修改**。  
> 本次修改不是局部路径重命名，而是一次 calibration state/runtime hyperparameter 职责拆分。必须同步修改 calibration、neuron runtime、controller、artifact layout、training provenance、conversion/evaluation、artifact verification 与 tests，不能只改 `scripts/calibrate_sites.py`。

---

## 1. 最终目标

当前代码将 activation statistics、Phase/MTN runtime 超参数和 common Clip materialization 混合在一次 calibration 中，导致：

- 更换 `phase.T` 后需要重新生成包含 `T/v0` 的 `phase_state.pt`；
- 更换 `mtn.T / mtn.K` 后，`mtn_state.pt` 与 runtime 紧耦合；
- common Clip 与通用 calibration state 共址，无法让多个 `phase.T × mtn.T` 组合复用同一套 statistics；
- aware ANN fine-tuning 与 SNN conversion/evaluation 的 artifact 路径不能完整表达本次要 sweep 的参数；
- 当前 salient GIF Clip 对所有 group 无条件使用 `gif_low ∩ gif_high`，没有利用真实 saliency mask 判断 all-low / all-high / mixed；
- Site 1、Site 7 是 multi-role GIF site，但当前 Clip 只有 site-level 一套 `lower/upper`，不能做真正的 role-specific mask-aware Clip。

本次重构后必须满足：

1. **Calibration Stage A**：只运行一次昂贵的模型 forward/statistics collection，生成与
   - `phase.T`
   - `mtn.T`
   - `mtn.K`

   无关的通用 calibration artifacts。

2. **Calibration Stage B**：不再加载模型、不再遍历 calibration dataset，只读取 Stage A artifacts，根据当前 YAML 中
   - `phase.T`
   - `mtn.T`

   快速生成对应的 common Clip profile。

3. 同一套 Stage A 可以反复复用：

   ```text
   Stage A
      ├── Stage B: phase_T_2_mtn_T_2
      ├── Stage B: phase_T_2_mtn_T_4
      ├── Stage B: phase_T_4_mtn_T_2
      └── Stage B: phase_T_4_mtn_T_4
   ```

4. ANN fine-tuning：
   - `phase_aware` 支持不同 `phase.T × mtn.T`；
   - `gif_aware` 支持不同 `phase.T × mtn.T`；
   - `common_clip_enabled=true` 时必须加载**与当前 config 完全匹配**的 Stage B Clip profile；
   - `common_clip_enabled=false` 时可以不实际 apply Clip，但 Stage B 仍作为本次 aware 实验选定的 Clip profile/provenance 保留。

5. SNN conversion/evaluation：
   - Phase SNN 的 `phase.T` 是 deployment runtime 参数，可以与 ANN fine-tuning 时使用的 `phase.T` 不同；
   - MTN SNN 的 `mtn.T / mtn.K` 是 deployment runtime 参数，可以与 ANN fine-tuning / Clip profile 中使用的 `mtn.T` 不同；
   - Phase/MTN SNN 路径必须包含 deployment 参数，防止结果覆盖。

6. `calibration.num_samples` 改为真正可 sweep 的正整数，不能再硬编码必须等于 128。

---

# 2. 本次已确认、不可再改变的设计决定

实现时以本节为最终规格。

## 2.1 Phase 固定规则

- `phase.base` **永久固定为 `2.0`**。
- 配置中仍保留：

  ```yaml
  phase:
    T: ...
    base: 2.0
    surrogate_slope: ...
  ```

- `validate_config()` 必须强制：

  ```python
  float(cfg["phase"]["base"]) == 2.0
  ```

  非 2.0 直接报错。

- 删除配置项：

  ```yaml
  phase.max_spikes
  ```

- 删除所有代码中的：
  - `self.max_spikes`
  - `spike_count`
  - `spike_count < self.max_spikes`
  - 对 `max_spikes` 的 state 保存、加载、validation、metadata、tests。

Phase neuron 不再有额外 spike-count 上限，循环次数仅由 runtime `phase.T` 决定。

---

## 2.2 MTN 固定规则

Stage A 的 MTN calibration 核心量定义保持：

\[
absolute = \max(|minimum|,\ |maximum|)
\]

\[
base\_scale = 2.0 \times absolute
\]

`mtn.T / mtn.K / threshold_factor` 均视为 **runtime hyperparameters**，不能写进 Stage A `mtn_state.pt`。

---

## 2.3 Stage A state 的职责

原则：

> **State 保存 calibration-derived parameters；Config 保存 runtime hyperparameters。**

Stage A 中：

- `phase_state.pt`：核心只保存 `tau`；
- `mtn_state.pt`：核心只保存 `base_scale`；
- `gif_state.pt`：继续保存 GIF static quantization 所需的 calibration-derived 参数、mask、scale/zero 以及必要的 schema/protocol metadata；
- state 可以保留必要的：
  - `state_kind`
  - format/schema version
  - parameter layout
  - group metadata
  - calibration policy/provenance metadata

  但不得把本次要求动态 sweep 的 runtime 参数重新塞回 Stage A state。

特别地：

### `phase_state.pt` 必须删除

```text
T
base
max_spikes
v0
```

以及任何等价 runtime 字段。

### `mtn_state.pt` 必须删除

```text
T
K
threshold_factor
```

以及任何等价 runtime 字段。

> 之前讨论中“从 phase_state.pt 删除 mtn.T/mtn.K”是笔误，实际指 `mtn_state.pt`。

---

# 3. 新 Calibration 两阶段架构

建议保留现有 lifecycle `--stage`：

```text
ann_training
vanilla_analysis
post_finetuning
```

另外新增一个明确的 CLI 参数：

```text
--calibration-phase A
--calibration-phase B
```

不要把 lifecycle stage 和 A/B phase 混成一个 enum。

建议 `scripts/calibrate_sites.py` 最终调用形式如下。

## 3.1 Stage A

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase A
```

## 3.2 Stage B

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase B
```

## 3.3 Post-finetuning 正常流程

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning \
  --calibration-phase A
```

Post-finetuning 同样使用新的 A/B 架构和共用代码，但**正常最终 SNN conversion 只需要 Stage A，不需要运行 Stage B**，因为 SNN deployment 不应用 common Clip。

可以让 Stage B 的底层 materializer 对 `post_finetuning` 也可调用，便于测试/分析，但 production conversion 不得依赖它。

---

# 4. Stage A：具体行为

Stage A 是唯一允许：

- load model
- install model integration
- load Prefix KV
- iterate calibration dataset
- collect activation statistics/saliency

的阶段。

Stage B 禁止做这些工作。

Stage A 当前应基于现有：

```python
collect_site_statistics(...)
StatisticsStore
```

重构。

---

## 4.1 Stage A 每个 replacement site 生成文件

例如：

```text
sites/
└── layer_000/
    └── site_01_post_input_rmsnorm/
        ├── statistics.pt
        ├── statistics_summary.json
        ├── phase_state.pt
        ├── gif_state.pt
        └── mtn_state.pt
```

Stage A **不再生成**：

```text
clip_state.pt
calibration_summary.json
```

这两个文件全部属于 Stage B。

必须确保反复运行不同 `phase.T / mtn.T / mtn.K` 的 Stage B 时，以上 5 个 Stage A 文件完全不发生修改。

---

# 5. `statistics.pt / statistics_summary.json` 要求

当前 statistics 本身原则上已经不依赖 `phase.T / mtn.T / mtn.K`。

必须全面检查并保证：

```text
statistics.pt
statistics_summary.json
statistics_manifest.json
```

中不存在：

```text
phase.T
phase_T
mtn.T
mtn_T
mtn.K
mtn_K
max_spikes
v0
```

若当前某处 metadata 间接保存上述动态参数，一并删除。

### 保留的 Phase statistics

必须继续保留现有 Phase tau calibration 统计：

```text
phase_ema_abs_max
phase_ema_updates
phase_tau_calibration
phase_tau_ema_factor
phase_tau_accumulator_dtype
phase_tau_channel_policy
phase_tau_reduction_policy
```

现有规则保持：

```text
ema_channel_abs_max_then_group_max
EMA factor = 0.99
FP32 accumulator
```

不要修改 tau 的统计定义。

---

# 6. 新 `phase_state.pt`

## 6.1 Stage A materialization

`build_phase_state()` 不再读取：

```python
cfg["phase"]["T"]
cfg["phase"]["base"]
cfg["phase"]["max_spikes"]
```

它只根据 statistics 和 calibration grouping 生成 tau。

建议结构：

```python
{
    "state_kind": "phase",
    "format_version": ...,
    "temporal_implementation_version": ...,

    # layout / grouping
    "parameter_layout": ...,
    "configured_group_size": ...,
    "group_size": ...,
    "num_heads": ...,
    "channels_per_head": ...,
    "groups_per_head": ...,

    # calibration-derived payload
    "tau": tau,

    # tau provenance
    "tau_calibration": ...,
    "tau_ema_factor": 0.99,
    "tau_accumulator_dtype": "float32",
    "tau_channel_policy": ...,
    "tau_reduction_policy": ...,
}
```

不得再出现：

```text
T
base
max_spikes
v0
surrogate_slope
```

---

## 6.2 Phase runtime

重构 `PhaseSurrogate`。

建议接口：

```python
PhaseSurrogate(
    state,
    *,
    T: int,
    surrogate_slope: float | None = None,
)
```

`T` 必须来自当前 config/runtime，不再来自 state。

由于 `phase.base=2.0` 永久固定，运行时使用：

\[
v_0 = 0.5 \times \tau \times 2^{-T}
\]

不要预先保存 `v0`。

建议在构造器内：

```python
self.T = int(T)
self.register_buffer("tau", state["tau"].float())
self.register_buffer(
    "v0",
    (0.5 * self.tau * 2 ** (-self.T)).float(),
)
```

也可以不注册 `v0`，在 encode 前动态计算；关键要求是**不能从 state 加载**。

Phase amplitude 保持：

\[
a_t = \tau \times 2^{-(t+1)}
\]

`t=0,...,T-1`。

删除 spike count：

原逻辑类似：

```python
spike_count = torch.zeros_like(x)
...
if self.max_spikes > 0:
    spike = spike * (spike_count < self.max_spikes)
spike_count += spike.detach()
```

全部删除。

最终循环只由：

```python
for timestep in range(self.T):
```

控制。

---

# 7. 新 `mtn_state.pt`

## 7.1 Stage A materialization

`mtn_state.pt` 只保存 calibration-derived `base_scale` 与必要 layout/schema metadata。

建议：

```python
{
    "state_kind": "mtn",
    "format_version": ...,
    "temporal_implementation_version": ...,

    "parameter_layout": ...,
    "configured_group_size": ...,
    "group_size": ...,
    "num_heads": ...,
    "channels_per_head": ...,
    "groups_per_head": ...,

    "base_scale": base_scale,
}
```

其中：

\[
base\_scale = 2 \times
\max(|minimum|,\ |maximum|)
\]

不得出现：

```text
T
K
threshold_factor
```

---

## 7.2 MTN runtime

重构：

```python
MultiThresholdNeuron(
    state,
    *,
    T: int,
    K: int,
    threshold_factor: float,
)
```

全部 runtime 参数来自 config。

例如 controller/deployment 根据当前配置传入：

```python
T=int(cfg["mtn"]["T"])
K=int(cfg["mtn"]["K"])
threshold_factor=float(cfg["mtn"]["threshold_factor"])
```

`MultiThresholdNeuron` 不得再通过：

```python
state["T"]
state["K"]
```

获得 runtime 参数。

---

# 8. `_global/final_rmsnorm` 重构

Stage A 继续生成：

```text
sites/
└── _global/
    └── final_rmsnorm/
        ├── statistics.pt
        ├── statistics_summary.json
        └── phase_state.pt
```

这里的 `phase_state.pt` 也必须是 T-independent：

- 保留 `tau`
- 删除 `T`
- 删除 `base`
- 删除 `v0`
- 删除 `max_spikes`

Phase SNN deployment 加载 final RMSNorm phase state 时，必须使用**当前 deployment `phase.T`** 创建 `PhaseSurrogate`。

不能再要求：

```text
final RMSNorm state 中的 T == per-site state 中的 T
```

因为 state 中根本不保存 T。

---

# 9. Stage A `calibration_state_manifest.json`

仍保留在：

```text
sites/calibration_state_manifest.json
```

但语义改为：

> **Stage A common calibration-state manifest**

必须 T/K-independent。

每个 site 的 `state_sha256` 只记录：

```text
phase
gif
mtn
```

不能记录 `clip`。

删除 per-site summary 中现有类似：

```text
phase_T
phase_base
mtn_T
mtn_K_positive_and_negative
```

以及任何与本次 runtime sweep 参数相关的字段。

建议新增：

```json
{
  "calibration_architecture": "two_stage_A_common_B_clip_profiles",
  "calibration_phase": "A",
  "stage_a_parameter_independence": [
    "phase.T",
    "mtn.T",
    "mtn.K"
  ]
}
```

现有 provenance（model / prefix / rotation / data manifest hash / grouping / topology / saliency rule 等）继续保留。

---

# 10. Stage B：Clip profile

Stage B：

- 不 load model；
- 不 load tokenizer；
- 不 load Prefix KV 做 forward；
- 不读取 calibration dataset；
- 不调用 `collect_site_statistics()`；
- 只读取 Stage A：
  - `phase_state.pt`
  - `gif_state.pt`
  - `mtn_state.pt`
  - Stage A manifest
- 根据当前 `phase.T / mtn.T` materialize Clip。

---

# 11. Stage B 目录结构

Calibration root 改为包含 `num_samples`，见后文路径章节。

在一个 calibration root 下：

```text
calibration_group_size_-1_num_samples_128/
├── sites/                                  # Stage A
│   ├── layer_000/
│   │   ├── site_01_.../
│   │   │   ├── statistics.pt
│   │   │   ├── statistics_summary.json
│   │   │   ├── phase_state.pt
│   │   │   ├── gif_state.pt
│   │   │   └── mtn_state.pt
│   │   └── ...
│   ├── _global/final_rmsnorm/...
│   ├── statistics_manifest.json
│   └── calibration_state_manifest.json
│
└── clip_profiles/                          # Stage B
    ├── phase_T_2_mtn_T_2/
    │   ├── layer_000/
    │   │   ├── site_01_.../
    │   │   │   ├── clip_state.pt
    │   │   │   └── calibration_summary.json
    │   │   └── ...
    │   └── clip_profile_manifest.json
    │
    ├── phase_T_2_mtn_T_4/
    ├── phase_T_4_mtn_T_2/
    └── phase_T_4_mtn_T_4/
```

Stage B profile dirname 必须精确包含：

```text
phase_T_<phase.T>_mtn_T_<mtn.T>
```

例如：

```text
phase_T_4_mtn_T_8
```

`mtn.K` **不进入 Clip profile 路径**，因为 Clip 计算不使用 `mtn.K`。

---

# 12. PhaseBound 与 MTNBound

## 12.1 PhaseBound

`phase.base=2.0` 固定。

\[
B_{phase}
=
\tau
\sum_{t=0}^{T_{phase}-1}2^{-(t+1)}
\]

等价：

\[
B_{phase}
=
\tau(1-2^{-T_{phase}})
\]

Phase 基础范围：

\[
[-B_{phase},\ B_{phase}]
\]

对称于 0。

---

## 12.2 MTNBound

Stage A：

\[
base\_scale = 2absolute
\]

当前旧公式：

\[
B_{mtn}=2T_{mtn}absolute
\]

因此 Stage B 直接使用：

\[
B_{mtn}=T_{mtn}\times base\_scale
\]

MTN 基础范围：

\[
[-B_{mtn},\ B_{mtn}]
\]

同样对称于 0。

---

## 12.3 Phase + MTN base intersection

```python
base_lower = torch.maximum(-phase_bound, -mtn_bound)
base_upper = torch.minimum( phase_bound,  mtn_bound)
```

也就是：

\[
[-\min(B_{phase},B_{mtn}),\ \min(B_{phase},B_{mtn})]
\]

仍然对称。

最终 Clip 与 GIF asymmetric affine representable range 相交后**可以不对称**，这是正确行为，不要强制最终 `lower=-upper`。

---

# 13. 不同 GIF Site 的 Stage B Clip 规则

当前 10 Site GIF policy 保持：

```text
Site 1  salient
Site 2  all-low
Site 3  salient
Site 4  salient
Site 5  identity
Site 6  salient
Site 7  salient
Site 8  identity
Site 9  identity
Site 10 salient
```

---

## 13.1 Site 5

Site 5：

```text
不生成 clip_state.pt
不生成普通 common Clip
```

保持现有 `SOFTMAX_SITE5_CLIP_POLICY = disabled` 语义。

Stage B 的 profile manifest/calibration summary 必须明确记录 Site 5 被排除，而不能误报缺文件。

---

## 13.2 Site 8 / 9：GIF identity

只取：

\[
Phase \cap MTN
\]

即：

```text
intersection(phase, mtn)
```

最终仍是对称范围。

---

## 13.3 Site 2：all-low

取：

\[
Phase \cap MTN \cap GIF_{low}
\]

即：

```text
intersection(phase, mtn, gif_low)
```

---

# 14. Salient Site：mask-aware per-group intersection

适用于：

```text
Site 1 / 3 / 4 / 6 / 7 / 10
```

这是本次必须修改的核心逻辑。

当前旧实现对所有 salient group 都无条件：

```text
phase ∩ mtn ∩ gif_low ∩ gif_high
```

必须改为根据**该 group 的真实 low-mask**分类。

GIF mask 当前语义：

```text
mask_low == True   -> 普通/low channel
mask_low == False  -> salient/high channel
```

---

## 14.1 Group 分类

对一个 group 内所有 channel：

### all-low

```python
all(mask_low == True)
```

只使用：

\[
GIF_{low}
\]

Clip：

\[
Phase \cap MTN \cap GIF_{low}
\]

---

### all-high / all-salient

```python
all(mask_low == False)
```

只使用：

\[
GIF_{high}
\]

Clip：

\[
Phase \cap MTN \cap GIF_{high}
\]

---

### mixed

既有 low 又有 high：

```python
not all_low and not all_high
```

才使用：

\[
GIF_{low}\cap GIF_{high}
\]

Clip：

\[
Phase \cap MTN \cap GIF_{low}\cap GIF_{high}
\]

---

## 14.2 Group reshape 必须服从现有 parameter layout

### `last_dim_grouped`

mask：

```text
[C]
```

按 `group_size` reshape：

```text
[num_groups, group_size]
```

逐 group 判断 all-low/all-high/mixed。

### `attention_head_grouped`

mask：

```text
[H, D]
```

按每个 head 内的 D 做：

```text
[H, groups_per_head, group_size]
```

不能跨 head 合并。

### Site 5

不参与。

---

# 15. Site 1 / Site 7：必须 role-specific Clip

这是本次最容易实现错的地方。

现有 GIF multi-role：

```text
Site 1: q / k / v
Site 7: gate / up
```

不同 role 的 mask **不允许合并**。

不得将：

```text
q/k/v mask union/intersection
gate/up mask union/intersection
```

生成一个保守的 site-level mask。

必须分别 materialize。

---

## 15.1 Site 1

Stage B 对：

```text
q
k
v
```

分别完成 group 分类与 Clip 计算。

---

## 15.2 Site 7

对：

```text
gate
up
```

分别完成。

---

# 16. 新 role-specific `clip_state.pt` schema

建议对 Site 1/7：

```python
{
    "state_kind": "clip",
    "format_version": ...,
    "temporal_implementation_version": ...,

    "phase_T": int(...),
    "mtn_T": int(...),

    **layout,

    "clip_role_policy": "role_specific",
    "clip_roles": ["q", "k", "v"],   # Site 1
    # 或 ["gate", "up"]

    "lower_by_role": {
        "q": tensor(...),
        "k": tensor(...),
        "v": tensor(...),
    },
    "upper_by_role": {
        "q": tensor(...),
        "k": tensor(...),
        "v": tensor(...),
    },

    "gif_group_classification_by_role": {
        "q": ...,   # 可用 small int tensor / enum metadata
        "k": ...,
        "v": ...,
    },

    "gif_low_range": (...),
    "gif_high_range": (...),

    "gif_constraint_policy": ...,
    "rule": "mask_aware_per_group_role_specific",
}
```

推荐 group classification 使用稳定 enum：

```text
0 = all_low
1 = all_high
2 = mixed
```

并在 metadata 中保存 mapping，避免 magic number。

Site 7 同理。

---

# 17. 单-role salient `clip_state.pt`

Site：

```text
3 / 4 / 6 / 10
```

可以继续保存普通：

```text
lower
upper
```

但必须增加 mask-aware group classification metadata，例如：

```python
{
    ...
    "clip_role_policy": "single",
    "lower": ...,
    "upper": ...,
    "gif_group_classification": ...,
    "rule": "mask_aware_per_group",
}
```

---

# 18. Stage B `calibration_summary.json`

Stage B 每个 Site 保存：

```text
calibration_summary.json
```

它是 profile-specific summary。

至少包含：

```text
site_index
layout_kind
parameter_layout
configured_group_size
effective_group_size
num_heads
channels_per_head
groups_per_head

phase_T
mtn_T

phase_bound_shape
mtn_bound_shape

gif_policy
clip_state_present
clip_valid
clip_rule
clip_role_policy
clip_roles

state_a_phase_sha256
state_a_gif_sha256
state_a_mtn_sha256
clip_state_sha256
```

对 salient Site 还应保存 group 分类数量。

单 role：

```json
"gif_group_class_counts": {
  "all_low": ...,
  "all_high": ...,
  "mixed": ...
}
```

Site 1/7：

```json
"gif_group_class_counts_by_role": {
  "q": {
    "all_low": ...,
    "all_high": ...,
    "mixed": ...
  },
  ...
}
```

这对后续 `verify_artifacts.py` 很重要。

---

# 19. `clip_profile_manifest.json`

每个：

```text
clip_profiles/phase_T_X_mtn_T_Y/
```

根目录新增：

```text
clip_profile_manifest.json
```

至少记录：

```text
format_version
calibration_phase: "B"

phase_T
mtn_T
phase_base: 2.0

stage_a_root
stage_a_calibration_manifest_path
stage_a_calibration_manifest_sha256

calibration_group_size
calibration_num_samples

expected_num_hidden_layers
site topology metadata

clip policy version
mask-aware policy
role-specific policy

per-site calibration_summary
per-site clip_state hashes
```

Stage B manifest 必须绑定 Stage A manifest SHA-256。

这样：

> Stage A 任何 state 被修改后，旧 Stage B profile 会自动 provenance mismatch，不能继续使用。

---

# 20. Clipper runtime 重构

当前 `Clipper` 假定所有 Site 都有：

```text
lower
upper
```

必须支持两种 schema：

```text
single
role_specific
```

建议：

```python
class Clipper(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *,
        role: str | None = None,
    ) -> torch.Tensor:
        ...
```

### single

要求：

```text
role is None
```

或忽略 default role。

### role_specific

要求：

```text
role in clip_roles
```

然后从：

```text
lower_by_role[role]
upper_by_role[role]
```

取参数。

若 Site 1/7 使用 role-specific state 而 runtime 未传 role，必须直接报错，不能偷偷 fallback 到任意 role 或做 role merge。

---

# 21. Phase-aware Site 1/7 的特殊 runtime 路径

**必须特别处理，不能只修改 `Clipper.forward(role=...)`。**

当前模型 integration 中：

- Site 1 是 RMSNorm 后共享激活，之后才分别进入 q/k/v；
- Site 7 是 MLP RMSNorm 后共享激活，之后才分别进入 gate/up；
- Phase replacement 当前在共享 RMSNorm output 上执行；
- GIF 已经有 branch-specific pre-hook，按 q/k/v、gate/up 传 `gif_role`。

因此 role-specific common Clip 在 `phase_aware` 下不能在共享 RMSNorm output 处选择一个 role。

正确设计：

## 21.1 Phase-aware Site 1

1. RMSNorm 输出处：
   - 执行 **一次** PhaseSurrogate；
   - **不在这里做 common Clip**。

2. q/k/v 三个 projection 的 forward pre-hook：
   - 不重复执行 PhaseSurrogate；
   - 若 `common_clip_enabled=true`：
     - q 输入执行 Site 1 `role=q` Clip；
     - k 输入执行 Site 1 `role=k` Clip；
     - v 输入执行 Site 1 `role=v` Clip。

即：

```text
RMSNorm
   ↓
PhaseSurrogate once
   ├── Clip(role=q) → q_proj
   ├── Clip(role=k) → k_proj
   └── Clip(role=v) → v_proj
```

---

## 21.2 Phase-aware Site 7

```text
RMSNorm
   ↓
PhaseSurrogate once
   ├── Clip(role=gate) → gate_proj
   └── Clip(role=up)   → up_proj
```

---

## 21.3 推荐 Controller API

增加一个明确方法，例如：

```python
controller.apply_role_clip(
    layer_index,
    site_index,
    x,
    *,
    role,
)
```

或等价设计。

不要通过“再次调用 `controller.apply()`”实现 Phase branch clip，否则有风险重复执行 Phase neuron。

对于 `mode=="phase"`：

- Site 1/7：
  - shared `apply()`：Phase only；
  - branch pre-hook：Clip only。

- 其他 clip-eligible Site：
  - `apply()` 可以保持 Phase → Clip 一体化。

---

# 22. GIF-aware Site 1/7

当前 GIF branch pre-hook 已经按 role 调用 GIF。

保持：

```text
q/k/v
gate/up
```

branch-specific replacement。

当 `common_clip_enabled=true` 时：

```text
GIF(role)
→ Clip(role)
```

在同一个 branch 调用链完成。

不要增加第二次 GIF quantization。

---

# 23. SNN deployment 不使用 common Clip

保持当前设计：

```text
deploy_phase
deploy_gif
deploy_mtn
```

均：

```text
snn_clip_applied = False
```

SNN conversion / evaluation 只加载 Stage A Phase/GIF/MTN state，不需要 Stage B Clip。

aware ANN checkpoint 的 provenance 仍需记录它训练时对应的 Stage B profile，因为 ANN 权重是在哪个 common Clip 条件下训练出来的必须可追踪。

---

# 24. `SiteController` runtime 参数注入

当前 Controller/Factories 通过 state 反推出 Phase/MTN T/K，必须重构。

建议 `SiteController` 构造时接收当前 runtime config，例如：

```python
SiteController(
    ...,
    phase_T=int(cfg["phase"]["T"]),
    mtn_T=int(cfg["mtn"]["T"]),
    mtn_K=int(cfg["mtn"]["K"]),
    mtn_threshold_factor=float(cfg["mtn"]["threshold_factor"]),
    ...
)
```

或者传完整精简 runtime-neuron config dataclass。

要求：

- `PhaseSurrogate` 从 controller runtime 参数拿 T；
- `MultiThresholdNeuron` 从 controller runtime 参数拿 T/K/threshold factor；
- `set_deployment()` 不再从 state bundle 推断 Phase/MTN T。

### deployment temporal steps

```python
phase -> cfg["phase"]["T"]
mtn   -> cfg["mtn"]["T"]
gif   -> GIF fixed/local temporal steps
```

GIF 现有两步 static policy保持不变。

---

# 25. `validate_site_state_bundle()` 重构

当前 validation 会：

- 实例化 PhaseSurrogate；
- 实例化 MultiThresholdNeuron；
- 从 module.T 收集 temporal steps；
- 要求 final RMSNorm Phase T 与 per-site Phase T 相同。

这些逻辑全部需要修改。

Stage A validation 只验证：

```text
statistics/state schema
layout/grouping
tau shape
base_scale shape
GIF policy/mask/saliency provenance
hash
site topology
global final_rmsnorm tau state
```

**不能从 Stage A state 推断 Phase/MTN T/K。**

若需要 runtime validation，显式传当前 config/runtime：

```text
phase.T > 0
mtn.T > 0
mtn.K > 0
```

Stage B Clip validation单独验证 profile。

建议拆成：

```python
validate_stage_a_site_state_bundle(...)
validate_clip_profile(...)
```

或者保留一个总入口但内部职责明确分离。

---

# 26. `calibration.num_samples` 支持 sweep

删除当前硬编码：

```python
if int(cfg["calibration"]["num_samples"]) != 128:
    raise ValueError(...)
```

改成：

```python
num_samples = cfg["calibration"]["num_samples"]
if not isinstance(num_samples, int) ...
if num_samples <= 0:
    raise ValueError(...)
```

仍保留：

```text
with_replacement = false
```

主实验约束。

若 `num_samples > 可选 train 样本数`，继续由 data sampling 阶段报错。

---

# 27. Calibration data manifest 必须随 `num_samples` 隔离

这是支持真实 `num_samples` sweep 的必要修改。

当前所有 config 共用类似：

```text
_shared/seed42/data/calibration_manifest.json
```

如果用不同 `num_samples` 执行 `prepare_data.py`，会互相覆盖。

改成：

```text
_shared/seed42/data/
├── train_manifest.json
├── validation_manifest.json
├── evaluation_manifest.json
└── calibration/
    ├── num_samples_64/
    │   └── calibration_manifest.json
    ├── num_samples_128/
    │   └── calibration_manifest.json
    └── num_samples_256/
        └── calibration_manifest.json
```

若希望更严谨，也可以把 calibration seed 纳入子路径，但本次最低要求必须有 `num_samples`。

建议新增 ArtifactLayout：

```python
calibration_data_dir
calibration_data_manifest_path
```

例如：

```python
@property
def calibration_data_dir(self):
    return self.data_dir / "calibration" / f"num_samples_{...}"
```

---

## 27.1 `snn2/data.py`

重构：

```python
prepare_manifests()
load_manifests()
load_selected_raw()
```

使 calibration manifest 根据当前 cfg 的 `num_samples` 读取。

`load_manifests()` 当前只接 `layout`，建议改为：

```python
load_manifests(cfg, layout)
```

或让 layout 已绑定 cfg 并提供专用 property。

train/validation/evaluation manifest 保持共享，不因 `num_samples` 重复复制。

---

## 27.2 Calibration provenance

`calibration_provenance()` 必须 hash 当前：

```text
.../data/calibration/num_samples_N/calibration_manifest.json
```

并记录：

```text
calibration_num_samples
calibration_data_manifest_path
calibration_data_manifest_sha256
```

Stage A manifest 与 Stage B profile manifest 都必须记录。

---

# 28. Calibration artifact 路径修改

当前：

```text
calibration_group_size_<group>
```

统一改为：

```text
calibration_group_size_<group>_num_samples_<N>
```

例如：

```text
calibration_group_size_-1_num_samples_128
```

必须应用于：

- `ann_training_calibration`
- `vanilla_analysis_calibration`
- `post_finetuning/conversion_calibration`

不要只改 ann_training。

建议将 helper：

```python
calibration_group_dirname(group_size)
```

重构为：

```python
calibration_variant_dirname(group_size, num_samples)
```

输出：

```text
calibration_group_size_-1_num_samples_128
```

并让所有相关 path property 共用这个 helper，避免各处手写。

---

# 29. Aware ANN fine-tuning 路径

`vanilla`、`unaware` ANN fine-tuning 路径保持不变。

只修改：

```text
phase_aware
gif_aware
```

---

## 29.1 Aware 公共上层路径增加 num_samples

当前 aware path 中类似：

```text
lr5e-05_train_samples_10000_calibration_group_size_-1
```

改成：

```text
num_samples_128_lr5e-05_train_samples_10000_calibration_group_size_-1
```

即 `num_samples_<N>_` 放在 `lr...` 前面。

只对 aware mode 这样做。

---

## 29.2 phase_aware

当前：

```text
surrogate_slope_<x>_warmup_ratio_<y>
```

改为：

```text
phase_T_<P>_mtn_T_<M>_surrogate_slope_<S>_warmup_ratio_<W>
```

完整示例：

```text
.../
phase_aware/
num_samples_128_lr5e-05_train_samples_10000_calibration_group_size_-1/
prefix_enabled_ture_common_clip_enabled_true/
phase_T_4_mtn_T_8_surrogate_slope_1.0_warmup_ratio_0.0/
seed42/
```

---

## 29.3 gif_aware

在：

```text
prefix_enabled_..._common_clip_enabled_.../
seed42
```

之间插入：

```text
phase_T_<P>_mtn_T_<M>_warmup_ratio_<W>
```

示例：

```text
.../
gif_aware/
num_samples_128_lr5e-05_train_samples_10000_calibration_group_size_-1/
prefix_enabled_ture_common_clip_enabled_true/
phase_T_4_mtn_T_8_warmup_ratio_0.0/
seed42/
```

---

# 30. ANN training 如何选择 Stage B profile

新增 layout property：

```text
ann_training_clip_profile_dir
```

它必须解析为：

```text
ann_training_calibration/
prefix_enabled_.../
calibration_group_size_<G>_num_samples_<N>/
clip_profiles/
phase_T_<P>_mtn_T_<M>/
```

`training.py`：

- Stage A state root：

  ```python
  layout.ann_training_site_dir
  ```

- Stage B clip root：

  ```python
  layout.ann_training_clip_profile_dir
  ```

不得再假设：

```text
phase/gif/mtn/clip_state.pt
```

全部共址。

---

# 31. `common_clip_enabled` 两种情况

## 31.1 `true`

aware ANN training 必须：

1. validate Stage A；
2. validate Stage B profile；
3. Stage B manifest 的：
   - `phase_T`
   - `mtn_T`
   - `group_size`
   - `num_samples`
   - Stage A manifest hash

   必须和当前 config/Stage A 完全一致；
4. 实际 runtime apply Clip。

任何 mismatch 必须 fail fast。

---

## 31.2 `false`

当前 false 只是暂时不用 Clip，后续仍必须支持 true。

建议 aware 实验流程仍要求先运行 Stage B，使实验目录有明确的：

```text
phase_T_X_mtn_T_Y
```

profile 与 provenance。

training 可以：

```text
common_clip_enabled = false
common_clip_applied = false
```

但仍记录 selected Stage B profile path/hash。

这样同一套实验路径语义稳定，不会因为当前暂时 false 而将 Stage B 架构做成 optional hack。

---

# 32. Training provenance 更新

当前 `training.py` 会记录 calibration manifest hash。

重构后至少记录：

```text
ann_training_stage_a_root
ann_training_stage_a_manifest_sha256

ann_training_clip_profile_root
ann_training_clip_profile_manifest_sha256

ann_training_calibration_group_size
ann_training_calibration_num_samples

ann_training_phase_T
ann_training_mtn_T

ann_training_common_clip_enabled
ann_training_common_clip_applied

prefix provenance
rotation provenance
statistics format/state format versions
```

训练结束后的 provenance unchanged check 必须同时检查：

- Stage A manifest 未变化；
- 当前 selected Stage B profile manifest 未变化。

---

# 33. SNN conversion runtime 参数与路径

## 33.1 Phase SNN

现有：

```text
.../snn/phase/
```

改为：

```text
.../snn/phase/phase_T_<deployment phase.T>/
```

例如：

```text
.../snn/phase/phase_T_8/
```

---

## 33.2 MTN SNN

现有：

```text
.../snn/mtn/
```

改为：

```text
.../snn/mtn/mtn_T_<deployment mtn.T>_mtn_K_<deployment mtn.K>/
```

例如：

```text
.../snn/mtn/mtn_T_8_mtn_K_6/
```

---

## 33.3 GIF SNN

本次没有要求改变 GIF deployment path：

```text
.../snn/gif/
```

保持。

---

# 34. aware training T 与 deployment T 必须区分

这是本次重要实验语义。

例如：

```text
phase-aware ANN training:
phase.T = 4
mtn.T   = 4
```

最后可以：

```text
Phase SNN deployment:
phase.T = 8
```

这是允许的。

因此 conversion metadata 不能只写一个模糊的 `phase_T`。

建议明确区分：

```text
source_ann_training_phase_T
source_ann_training_mtn_T

deployment_phase_T
deployment_mtn_T
deployment_mtn_K
```

对 Phase conversion：

```text
deployment_phase_T = cfg["phase"]["T"]
```

对 MTN conversion：

```text
deployment_mtn_T = cfg["mtn"]["T"]
deployment_mtn_K = cfg["mtn"]["K"]
```

source ANN training T 从 `training_result.json` provenance 读取，而不是假设和当前 conversion config 一样。

---

# 35. Non-aware SNN 与 num_samples collision

`vanilla / unaware` ANN checkpoint 路径按要求保持不变。

但是它们的 post-finetuning calibration 现在也支持不同 `num_samples`，因此 SNN output 如果仍只有：

```text
snn/calibration_group_size_-1/...
```

不同 `num_samples` 会覆盖。

所以对 non-aware SNN 必须把现有 calibration prefix 改成新的完整 calibration variant：

```text
snn/
calibration_group_size_-1_num_samples_128/
phase/
phase_T_4/
...
```

和：

```text
snn/
calibration_group_size_-1_num_samples_128/
mtn/
mtn_T_4_mtn_K_6/
...
```

这是 `num_samples` sweep 的必要防覆盖修改。

aware SNN 因为其 `layout.root` 本身已包含：

```text
num_samples_<N>_lr...
```

不需要再重复插入一次 num_samples。

---

# 36. Post-finetuning calibration

`vanilla / unaware` 保持：

- 使用 final ANN checkpoint；
- 使用 post-finetuning Prefix policy；
- Stage A 重新收集最终 ANN activation statistics；
- 输出 T/K-independent common states。

路径：

```text
post_finetuning/
conversion_calibration/
prefix_enabled_.../
calibration_group_size_<G>_num_samples_<N>/
sites/
...
```

最终 SNN conversion：

```text
只依赖 Stage A
不要求 clip_profiles
```

Stage B 代码结构可以复用，但 production workflow 不需要执行。

---

# 37. `conversion.py` 重构

当前 conversion 会从 state bundle 得到 `temporal_steps`。

修改为：

1. validate Stage A common state；
2. 从当前 config 获取 deployment runtime 参数；
3. `SiteController.set_deployment()` 显式接收或已持有 runtime 参数；
4. conversion metadata 写出当前 deployment T/K；
5. 不要求任何 Stage B Clip。

Aware conversion 额外校验：

- source ANN training Stage A provenance；
- source ANN training selected Clip profile provenance；
- 当前 ANN checkpoint 与 training_result 对应。

但是不要错误要求：

```text
deployment phase.T == ANN training phase.T
deployment mtn.T == ANN training mtn.T
deployment mtn.K == 某个 calibration state K
```

这些都允许不同。

---

# 38. Evaluation scripts

审计：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/regress_phase_conversion.py
```

以及它们调用的 helper。

要求：

- 使用新的 `layout.snn_conversion_dir(neuron)`，自动得到 Phase/MTN 参数化路径；
- validate conversion metadata 时按 deployment 参数验证；
- 不从 Stage A state 读取 T/K；
- 输出目录与 conversion path 同样不会发生参数组合覆盖。

如果 evaluation path 当前在别处自己拼 `snn/phase`、`snn/mtn`，全部改成统一 ArtifactLayout helper。

---

# 39. `ArtifactLayout` 必须集中提供的新 helper/property

建议至少加入：

```text
calibration_variant_dirname(group_size, num_samples)

clip_profile_dirname(phase_T, mtn_T)

phase_training_dirname(
    phase_T,
    mtn_T,
    surrogate_slope,
    warmup_ratio,
)

gif_training_dirname(
    phase_T,
    mtn_T,
    warmup_ratio,
)

phase_snn_dirname(phase_T)
mtn_snn_dirname(mtn_T, mtn_K)
```

以及 property：

```text
ann_training_site_dir
ann_training_clip_profiles_dir
ann_training_clip_profile_dir

post_finetuning_site_dir
post_finetuning_clip_profiles_dir
post_finetuning_clip_profile_dir

calibration_data_manifest_path
```

不要把 dirname 字符串散落在 training/conversion/evaluation 脚本里。

---

# 40. Config 修改

## 40.1 删除

从：

```text
configs/experiment_matrix.yaml
所有 generated configs
tests fixtures
docs examples
```

删除：

```yaml
phase:
  max_spikes: ...
```

重新运行：

```bash
python scripts/materialize_configs.py
```

生成最新 config。

---

## 40.2 validation

`validate_config()`：

### Phase

```text
phase.T        positive integer
phase.base     exactly 2.0
surrogate_slope positive finite
max_spikes     不再支持
```

建议 stale config 若仍出现 `phase.max_spikes` 直接报错，促使用户重新 materialize config，而不是静默忽略。

### MTN

```text
mtn.T positive integer
mtn.K positive integer
threshold_factor valid finite positive value
```

### Calibration

```text
calibration.num_samples positive integer
group_size = -1 or positive integer
with_replacement = false
```

删除 `num_samples == 128` 限制。

---

# 41. Stage A/B provenance compatibility key

Stage A 能否复用，不应该看 `phase.T / mtn.T / mtn.K`。

Stage A compatibility 应至少取决于：

```text
source model / source ANN checkpoint
rotation state
prefix state/KV
calibration data manifest
calibration num_samples
calibration seed
calibration group_size
site topology
statistics schema
GIF policy/saliency policy
Phase tau calibration policy
```

Changing only：

```text
phase.T
mtn.T
mtn.K
phase.surrogate_slope
training.warmup_ratio
```

不得迫使 Stage A 重新生成。

Stage B compatibility 取决于：

```text
Stage A manifest hash
phase.T
mtn.T
Clip policy/schema
```

不依赖 `mtn.K`。

---

# 42. Schema/version 处理

这是 breaking artifact change。

必须升级：

```text
SITE_STATE_FORMAT_VERSION
CALIBRATION_MANIFEST_FORMAT_VERSION
CONVERSION_METADATA_FORMAT_VERSION
TEMPORAL_IMPLEMENTATION_VERSION
```

并更新相应 policy/version 字符串。

`SITE_TOPOLOGY_VERSION` 不需要因为本次修改改变，10 个 site topology 没变。

`STATISTICS_FORMAT_VERSION`：
- 如果实际 `statistics.pt` schema 没有变化，只是确认没有 T/K 字段，可以保持；
- 如果实现过程中确实修改了 statistics serialized schema，再升级。
- 不要为了路径变化无意义地升级 statistics format。

旧 calibration states 不允许被新代码误加载。

错误信息明确提示：

```text
re-run Stage A / Stage B
```

而不是旧的“re-materialize calibration states”模糊提示。

---

# 43. `verify_artifacts.py`

必须完整适配新结构。

至少校验：

## Stage A

- 所有 expected layer；
- 每层恰好 10 sites；
- site statistics files；
- phase/gif/mtn state；
- **Stage A site 中不存在 `clip_state.pt`**；
- **Stage A site 中不存在 `calibration_summary.json`**；
- state hash；
- forbidden runtime fields 不存在：
  - phase state 无 T/base/v0/max_spikes
  - mtn state 无 T/K/threshold_factor；
- `_global/final_rmsnorm/phase_state.pt` 同样 T-independent；
- manifest 不出现动态 T/K。

## Stage B

- profile dirname 与 manifest `phase_T/mtn_T` 一致；
- Stage A manifest hash 一致；
- Site 5 无 clip；
- eligible sites 有 clip；
- role-specific Site 1/7 roles 完整：
  - Site1 q/k/v
  - Site7 gate/up；
- single-role sites schema 正确；
- mask-aware group counts 与 state classification 一致；
- 每个 lower < upper；
- Clip SHA-256 一致。

## Training

- aware training result 路径中的 T/num_samples 与 provenance 一致；
- selected Stage B profile 与 training_result hash 一致。

## SNN conversion

- Phase path T 与 metadata/config 一致；
- MTN path T/K 与 metadata/config 一致；
- conversion 不依赖 Stage B Clip。

---

# 44. `calibration_summary.json` 与旧 manifest 的职责迁移

旧代码的 `calibration_summary.json` 同时描述 Phase/GIF/MTN/Clip。

新架构中必须重新定义：

- Stage A `calibration_state_manifest.json`：负责 common state summary；
- Stage B `calibration_summary.json`：负责当前 T profile 的 Clip summary；
- Stage B `clip_profile_manifest.json`：负责 profile-level provenance。

不要在 Stage A 再复制一份 profile-dependent calibration summary。

---

# 45. 需要重点修改/审计的文件

以下不是可选清单，Codex 必须逐一审计。

## 核心

```text
scripts/calibrate_sites.py

snn2/calibration.py
snn2/stats.py
snn2/neurons.py
snn2/controller.py
snn2/model_integration.py
snn2/state_validation.py
snn2/artifacts.py
snn2/config.py
snn2/training.py
snn2/conversion.py
snn2/data.py
snn2/temporal_ops.py
```

## Conversion/evaluation/verification

```text
scripts/convert_snn.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/regress_phase_conversion.py
scripts/verify_artifacts.py
```

## Config

```text
configs/experiment_matrix.yaml
scripts/materialize_configs.py
configs/generated/*    # 通过 materialize 重新生成，不要手工漂移
```

## Tests

至少审计：

```text
tests/test_calibration_gif.py
tests/test_calibration_profiles.py
tests/test_calibration_topology.py
tests/test_controller_state_loading.py
tests/test_conversion_metadata.py
tests/test_evaluation_paths.py
tests/test_generated_configs.py
tests/test_neurons.py
tests/test_phase_conversion_regression.py
tests/test_post_finetuning_protocol.py
tests/test_statistics.py
tests/test_verify_artifacts.py
```

以及 grep 到所有引用旧：

```text
max_spikes
phase_state["T"]
mtn_state["T"]
mtn_state["K"]
clip_state.pt
calibration_summary.json
calibration_group_size_
snn/phase
snn/mtn
```

的文件。

---

# 46. 建议新增/重构的 calibration 函数

不要继续让一个 `build_site_states()` 同时生成 Phase/GIF/MTN/Clip。

建议拆分为：

```python
build_phase_state(statistics, cfg)
build_gif_state(statistics, cfg)
build_mtn_state(statistics, cfg)

materialize_stage_a_states(...)

build_clip_state(
    phase_state,
    gif_state,
    mtn_state,
    *,
    phase_T,
    mtn_T,
    ...
)

materialize_clip_profile(...)
```

`build_clip_state()` 不要直接依赖 model/statistics collection。

---

# 47. Stage B GIF representable range

Stage B 可以直接由 Stage A `gif_state.pt` 重建 representable range。

例如 low：

\[
low_{min} = (qmin-low\_zero)\times low\_scale
\]

\[
low_{max} = (qmax-low\_zero)\times low\_scale
\]

当前：

```text
qmin = 0
low_qmax = 15
```

high：

```text
qmin = 0
high_qmax = 30
```

也可以复用一个统一 helper。

不要为了 Stage B 再跑 `_qparams()` 所需的 dataset forward。

---

# 48. `Clipper` validation 注意事项

旧 Clipper 会要求 salient Clip 同时包含 low/high range。

新 mask-aware policy 下这个 validation 必须变成**按 group/role**验证：

### all-low group

最终 Clip 必须落在 `gif_low_range` 中，不需要落在 high range。

### all-high group

必须落在 `gif_high_range` 中，不要求落在 low range。

### mixed group

必须同时落在 low + high。

Site 1/7 分 role 做上述验证。

否则如果保留旧 validation，即使 Stage B 算对了，runtime 也会错误拒绝 all-low/all-high optimized Clip。

---

# 49. `common_clip_enabled=true` regression 风险点

必须重点测试以下路径。

## Phase-aware Site 1

确保：

```text
Phase 只执行一次
q/k/v 分别 clip
```

不能：

```text
Phase(q) + Phase(k) + Phase(v)
```

重复三次 replacement。

## Phase-aware Site 7

同理 Phase 只一次，gate/up 两套 Clip。

## GIF-aware Site 1/7

GIF replacement 本来就是 role-specific，要保证：

```text
GIF(role) → Clip(role)
```

而不是：

```text
GIF(role) → Clip(default)
```

---

# 50. 必须新增的测试

下面测试建议全部实现。

---

## 50.1 Phase state T-independent

同一 statistics：

```text
cfg phase.T=2
cfg phase.T=8
```

Stage A `phase_state` 内容必须相同。

断言无：

```text
T
base
max_spikes
v0
```

---

## 50.2 MTN state T/K-independent

对：

```text
mtn.T=2,K=4
mtn.T=8,K=16
```

Stage A `mtn_state` 相同。

断言只有相同 `base_scale` 与 layout/schema metadata。

---

## 50.3 Phase runtime v0

给固定 tau：

验证：

\[
v0 = 0.5\tau2^{-T}
\]

分别测试 T=1/2/4/8。

---

## 50.4 删除 max_spikes

测试：
- generated config 无 `max_spikes`；
- stale config 带 `max_spikes` 时按设计报错；
- neuron 内无 spike count 截断。

---

## 50.5 phase.base

```text
2.0 -> pass
非 2.0 -> fail
```

---

## 50.6 num_samples

测试：

```text
1 -> pass
64 -> pass
128 -> pass
256 -> pass
0 -> fail
negative -> fail
```

并验证 without replacement 超过可用样本会 fail。

---

## 50.7 Calibration data manifest path

不同 num_samples：

```text
64
128
```

manifest 路径不同，不互相覆盖。

---

## 50.8 Calibration artifact path

断言：

```text
calibration_group_size_-1_num_samples_128
```

---

## 50.9 Stage A/B independence

1. 运行/materialize Stage A；
2. 记录所有 Stage A state hash；
3. 生成 B `(Tphase=2,Tmtn=2)`；
4. 生成 B `(4,8)`；
5. 再检查 Stage A hashes 完全不变。

---

## 50.10 Clip profile path

分别存在：

```text
phase_T_2_mtn_T_2
phase_T_4_mtn_T_8
```

无覆盖。

---

## 50.11 mask-aware all-low

构造一个 group mask 全 True。

断言 Clip 只受：

```text
phase/mtn/gif_low
```

约束，不额外 intersect gif_high。

---

## 50.12 mask-aware all-high

mask 全 False。

只用 gif_high。

---

## 50.13 mixed

同时 True/False。

必须 low/high 都取交集。

---

## 50.14 Attention per-head grouping

Site 3/4 mask `[H,D]`：

确保 group classification 不跨 head。

---

## 50.15 Site 1 role-specific

人为构造：

```text
q -> all-low
k -> all-high
v -> mixed
```

验证：

```text
lower_by_role["q"]
lower_by_role["k"]
lower_by_role["v"]
```

按各自规则不同。

---

## 50.16 Site 7 role-specific

gate/up 同理。

---

## 50.17 Clipper role validation

role-specific state：

```text
role=None -> fail
非法 role -> fail
合法 role -> pass
```

---

## 50.18 Phase-aware Site1/7 integration

用 mock/count：

- Phase module forward 只执行一次；
- q/k/v clip 各执行一次；
- gate/up clip 各执行一次。

---

## 50.19 GIF-aware Site1/7 integration

确认正确 role 传到 GIF 和 Clip。

---

## 50.20 Site 5

所有 Stage B profile：

```text
Site5 no clip_state.pt
```

---

## 50.21 Stage A manifest forbidden fields

递归检查 manifest 不包含：

```text
phase_T
mtn_T
mtn_K
max_spikes
v0
```

作为 runtime value。

注意名称出现在说明字符串/independence 列表时可以允许，但不能作为当前动态参数值。

---

## 50.22 Stage B manifest provenance

改变 Stage A manifest/hash 后，旧 B profile validation 必须失败。

---

## 50.23 ANN path

phase-aware：

```text
num_samples_128_lr.../
.../
phase_T_4_mtn_T_8_surrogate_slope_1.0_warmup_ratio_0.0/
seed42
```

gif-aware：

```text
num_samples_128_lr.../
.../
phase_T_4_mtn_T_8_warmup_ratio_0.0/
seed42
```

vanilla/unaware 断言原 ANN path 不变。

---

## 50.24 SNN path

Phase：

```text
snn/phase/phase_T_8/
```

MTN：

```text
snn/mtn/mtn_T_8_mtn_K_6/
```

non-aware 还要包含：

```text
calibration_group_size_*_num_samples_*
```

---

## 50.25 Deployment T 与 training T 不同

构造 training provenance：

```text
train phase.T=4
```

conversion config：

```text
deployment phase.T=8
```

必须允许。

MTN 同理。

---

## 50.26 Post-finetuning A-only

证明：

- 没有任何 Stage B profile；
- vanilla/unaware SNN conversion 仍可正常创建；
- conversion 不错误要求 clip profile。

---

# 51. 推荐运行流程

## 51.1 Aware ANN：第一次组合

例如：

```bash
CFG=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml
```

### Stage A

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase A
```

### Stage B

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase B
```

### ANN fine-tuning

按原 train 命令运行。

---

## 51.2 改 `phase.T/mtn.T` 后

若以下条件都没变：

```text
model
rotation
prefix
calibration data
num_samples
group_size
GIF policy
```

只需：

```bash
python scripts/calibrate_sites.py \
  --config "$NEW_CFG" \
  --stage ann_training \
  --calibration-phase B
```

**禁止重新跑 Stage A。**

然后运行新组合 ANN fine-tuning。

---

## 51.3 改 `mtn.K`

Stage A 不需要重跑。

Stage B 也不需要因为 `mtn.K` 重跑，因为 Clip 不依赖 K。

如果只是最终 MTN SNN sweep：

```text
直接重新 convert/evaluate 对应 mtn.T/mtn.K
```

---

## 51.4 改 `num_samples`

必须：

1. prepare 对应 calibration manifest；
2. 运行新的 Stage A；
3. 运行所需 Stage B；
4. aware ANN training 使用新 num_samples path。

---

## 51.5 Vanilla/Unaware post-finetuning

完成 final ANN 与 post-finetuning Prefix 后：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning \
  --calibration-phase A
```

然后直接 conversion。

不需要 Stage B。

---

# 52. 旧 artifact 兼容策略

本次是 breaking change。

不要尝试透明兼容旧：

```text
phase_state.pt(T/v0/max_spikes)
mtn_state.pt(T/K)
site directory 内 clip_state.pt
旧 calibration_state_manifest
旧 conversion_metadata
```

新代码发现旧 schema 应明确 fail。

建议提示：

```text
Legacy pre-A/B calibration artifact detected.
Re-run calibration Stage A, then Stage B if ANN common Clip is required.
```

### 重建范围

- aware calibration：必须重建 A/B；
- aware ANN：旧 checkpoint 的 training provenance 与新 path/schema 不一致，若要用于新实验体系应重新训练；
- vanilla/unaware ANN checkpoint 路径可以保持；
- 但其 post-finetuning Stage A 与 SNN conversion metadata 应按新 schema 重建；
- 旧 Phase/MTN SNN conversion/evaluation artifact 不再复用。

---

# 53. 文档与 shell 命令同步

完成代码后，grep 项目内所有 markdown/shell：

```bash
grep -R "calibrate_sites.py" -n .
grep -R "calibration_group_size_" -n .
grep -R "max_spikes" -n .
grep -R "snn/phase" -n .
grep -R "snn/mtn" -n .
```

更新项目执行文档中的：

- calibration 命令为 A/B；
- 新路径；
- `num_samples`；
- Phase/MTN deployment path；
- 删除 max_spikes；
- post-finetuning 只跑 A。

不要留下“calibration 一次同时生成 Clip”的旧表述。

---

# 54. 实施顺序

建议 Codex 严格按以下顺序修改，减少中间状态混乱：

1. **Config/schema**
   - 删除 max_spikes；
   - phase.base=2 validation；
   - num_samples positive；
   - version bump。

2. **ArtifactLayout + data manifest**
   - calibration variant dirname；
   - num_samples-specific data manifest；
   - aware ANN path；
   - Clip profile path；
   - SNN dynamic path。

3. **Stage A state**
   - Phase T-independent；
   - MTN T/K-independent；
   - manifest T/K-independent。

4. **Neuron runtime**
   - Phase runtime T/v0；
   - remove max_spikes；
   - MTN runtime T/K。

5. **Stage B**
   - PhaseBound；
   - MTNBound；
   - mask-aware per-group；
   - role-specific Site1/7；
   - profile manifest。

6. **Controller + model integration**
   - separate Stage A state root / Clip root；
   - Site1/7 Phase shared replacement + branch clip；
   - GIF role clip。

7. **Training provenance**

8. **Conversion/deployment**

9. **Evaluation**

10. **verify_artifacts**

11. **Tests**

12. **Regenerate generated configs**

13. **Update docs**

14. 运行完整测试。

---

# 55. 最终验收标准

只有同时满足以下条件才算本次修改完成。

1. `pytest -q` 全部通过。
2. 项目中不存在任何 Phase `max_spikes` runtime 逻辑。
3. generated configs 中不存在 `phase.max_spikes`。
4. Stage A `phase_state.pt` 无 T/base/v0/max_spikes。
5. Stage A `mtn_state.pt` 无 T/K/threshold_factor。
6. Stage A manifest 无 runtime T/K 值。
7. 同一 Stage A 可以生成至少两组不同 T 的 Stage B profile，且 Stage A hash 不变。
8. Stage B 不 load model / 不遍历 calibration dataset。
9. Site1 q/k/v Clip 完全 role-specific。
10. Site7 gate/up Clip 完全 role-specific。
11. salient Site 实现 all-low/all-high/mixed mask-aware per-group intersection。
12. `common_clip_enabled=true` 的 phase-aware/gif-aware 路径有单元测试覆盖。
13. Phase-aware Site1/7 不重复执行 Phase neuron。
14. `calibration.num_samples` 可取任意合法正整数。
15. 不同 num_samples calibration data manifest 不覆盖。
16. calibration path 包含 `group_size + num_samples`。
17. aware ANN path 包含 num_samples。
18. phase-aware ANN path包含 `phase_T + mtn_T + surrogate_slope + warmup_ratio`。
19. gif-aware ANN path包含 `phase_T + mtn_T + warmup_ratio`。
20. vanilla/unaware ANN fine-tuning path保持原样。
21. Phase SNN path包含 deployment `phase_T`。
22. MTN SNN path包含 deployment `mtn_T + mtn_K`。
23. deployment T/K 可以和 ANN training T 不同。
24. post-finetuning conversion 只需要 Stage A。
25. SNN deployment 不应用 common Clip。
26. conversion/training provenance 能准确区分 Stage A、Stage B、training T、deployment T/K。
27. `scripts/verify_artifacts.py` 能验证新 A/B topology、hash、role-specific Clip 和路径。
28. 旧 schema artifact 会 fail fast，而不是被静默加载。

---

# 56. 最重要的实现原则总结

整个重构务必遵守下面四句话：

> **第一，Stage A 只负责“从数据中学到什么”，不能负责“这次实验运行多少个 timestep”。**

> **第二，Stage B 只负责“给定当前 phase.T/mtn.T 后 Clip 应该是多少”，不能重新跑模型或重新统计 activation。**

> **第三，Phase/MTN 的 T/K 是 runtime hyperparameter，不是 calibration state。**

> **第四，Site 1/7 的 common Clip 必须真正 role-specific；Phase-aware 下 Phase 在共享激活处只执行一次，然后 q/k/v 或 gate/up 各自使用自己的 Clip。**

完成修改后先运行：

```bash
pytest -q
```

全部通过后，再用一个小模型/少量数据执行最小 smoke test，至少覆盖：

```text
Stage A
→ Stage B(T1)
→ Stage B(T2)
→ aware ANN common_clip_enabled=true
→ Phase conversion with deployment T != training T
→ MTN conversion with deployment T/K
```

并人工检查 artifact tree 与本文档规定完全一致。
