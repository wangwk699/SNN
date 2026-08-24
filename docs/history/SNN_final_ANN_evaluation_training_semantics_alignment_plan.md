# SNN Final ANN Evaluation 与 ANN Training Static Replacement 语义对齐完整修改方案

## 0. 适用仓库与基线

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

以 GitHub `main` 最新代码为唯一基准。制定本方案时最新提交为：

```text
4718a585234f83bd36eabaae2a1c9ed9fd0fa83a
phase_aware -> Phase SNN 单元级 Conversion Regression 检查与修正方案
```

该提交已经加入：

```text
scripts/regress_phase_conversion.py
snn2/phase_conversion_regression.py
```

并已经把以下诊断图明确区分：

```text
Graph I   = Identity ANN
Graph P   = Static Phase ANN（training-time semantics）
Graph S   = Temporal Phase SNN
Graph S0  = Temporal Phase SNN + regression-only final-norm bypass
```

本轮修改必须建立在这套 regression 基础上，不得回滚、删除或绕开已有 Phase conversion regression。

---

# 1. 本轮唯一核心目标

当前 `phase_aware` / `gif_aware` 的 ANN training 使用 static activation replacement，但官方 final ANN evaluation：

```bash
scripts/evaluate_tldr.py --neuron ann
scripts/evaluate_lm_harness.py --neuron ann
```

仍统一创建：

```python
SiteController(mode="identity")
```

因此 aware mode 的 ANN training 与 final ANN evaluation 语义不一致。

本轮要把正式实验固定为以下三阶段。

## 1.1 Phase 三阶段

```text
① Phase-aware ANN training
   controller.mode = "phase"
   10 个 replacement site 使用 PhaseSurrogate.forward()
   非 temporal，T 不跨 Transformer layer 传播

② Phase-aware final ANN evaluation
   controller.mode = "phase"
   10 个 replacement site 继续使用同一套 PhaseSurrogate.forward()
   与训练时使用同一套 ANN-training Phase states
   非 temporal，T 不跨 Transformer layer 传播

③ Phase SNN evaluation
   controller.mode = "deploy_phase"
   使用 Temporal Phase neuron
   full-temporal execution
   T 在模型 temporal 路径中传播
```

最关键比较：

```text
② Static Phase ANN
vs
③ Temporal Phase SNN
```

用于观察：

```text
PhaseSurrogate ANN -> Temporal Phase SNN
```

的 conversion loss。

对于 `replacement.common_clip_enabled=false` 的 regression / 主比较：

```text
② = PhaseSurrogate
③ = Temporal Phase neuron
```

中间不再混入 ANN identity graph。

---

## 1.2 GIF 三阶段

同理：

```text
① GIF-aware ANN training
   static GIF surrogate

② GIF-aware final ANN evaluation
   使用与训练完全相同的 static GIF surrogate

③ GIF SNN evaluation
   使用 Temporal GIF neuron
```

当前代码中 static GIF surrogate 的实现类名是：

```python
StaticGIF
```

即：

```python
StaticGIF.forward()
```

承担本文所说的：

```text
GIFSurrogate / static GIF surrogate
```

功能。

**本轮不要仅为了命名把 `StaticGIF` 大规模重命名为 `GIFSurrogate`。**

文档中统一写成：

```text
GIFSurrogate（当前代码实现：StaticGIF.forward()）
```

即可，避免无必要的兼容性改动。

---

# 2. 四种 ANN mode 的正式语义必须固定

这是本轮最重要的项目协议，必须同时写入：

```text
实验执行总结.md
AGENTS.md
README.md
```

并由代码和测试强制保证。

| `ann_mode` | ANN training | final ANN evaluation `--neuron ann` | SNN evaluation `--neuron phase/gif/mtn` |
|---|---|---|---|
| `vanilla` | `identity(x)`；代码训练 mode 可继续为 `none`，其 `apply()` 语义就是返回 `x` | `identity(x)` | 始终为所选 neuron 的 Temporal SNN |
| `unaware` | `identity(x)`；有 Rotation/Prefix，但无 activation replacement | `identity(x)`；保留 Rotation/Prefix | 始终为所选 neuron 的 Temporal SNN |
| `phase_aware` | `PhaseSurrogate.forward(x)`，可选 ANN common Clip | `PhaseSurrogate.forward(x)`，必须与训练使用同一 states 和同一 common-Clip 设置 | 始终为所选 neuron 的 Temporal SNN |
| `gif_aware` | `GIFSurrogate(x)`，当前实现为 `StaticGIF.forward(x)`，可选 ANN common Clip | 同一 `StaticGIF.forward(x)`，必须与训练使用同一 states 和同一 common-Clip 设置 | 始终为所选 neuron 的 Temporal SNN |

必须明确：

```text
--neuron ann
```

从本轮开始表示：

> **non-temporal final ANN execution**

而不是：

> identity execution

也就是说：

```text
--neuron ann + vanilla      -> identity ANN
--neuron ann + unaware      -> identity ANN
--neuron ann + phase_aware  -> static PhaseSurrogate ANN
--neuron ann + gif_aware    -> static GIFSurrogate ANN
```

而：

```text
--neuron phase
--neuron gif
--neuron mtn
```

无论 final ANN checkpoint 来自：

```text
vanilla
unaware
phase_aware
gif_aware
```

都必须表示：

```text
Temporal SNN deployment
```

不得根据 `ann_mode` 把 SNN evaluation 偷换成 static surrogate。

---

# 3. 不改变的边界

本轮只修正 **final ANN evaluation semantics**，以下内容默认不得改变。

## 3.1 不改变 ANN training

`snn2/training.py` 当前已经正确按：

```python
mode = cfg["replacement"]["train_mode"]
```

创建：

```python
SiteController(...)
```

其中 resolve 后：

```text
vanilla     -> train_mode = none
unaware     -> train_mode = none
phase_aware -> train_mode = phase
gif_aware   -> train_mode = gif
```

`SiteController.apply()` 中：

```python
if self.mode in {"identity", "none"}:
    return x
```

因此：

```text
none == identity(x)
```

在前向语义上已经满足 vanilla/unaware 要求。

不要为了表面统一把训练配置中的 `none` 强制改成 `identity`。

---

## 3.2 不改变 Temporal SNN deployment

以下逻辑必须保留：

```python
controller.set_deployment("phase")
controller.set_deployment("gif")
controller.set_deployment("mtn")
```

得到：

```text
deploy_phase
deploy_gif
deploy_mtn
```

SNN evaluation 继续走：

```text
temporal_forward(...)
temporal attention
temporal MLP
temporal neuron
```

不得因为本轮修改 final ANN evaluation 而让 `--neuron phase` 退化成 `PhaseSurrogate.forward()`。

---

## 3.3 SNN 永远不使用 common Clip

继续遵守当前项目协议：

```text
common Clip 只属于 phase_aware / gif_aware ANN static replacement
```

所以：

```text
ANN training:
    aware mode 可由 replacement.common_clip_enabled 控制 Clip

final ANN evaluation:
    aware mode 必须镜像训练的 common_clip_enabled

SNN conversion / SNN evaluation:
    永远不实例化、不加载、不执行 common Clip
```

GIF `_quantize()` 内部的 qmin/qmax clamp 属于 GIF 算法自身，不属于 common Clip。

---

## 3.4 不改变 regression 中 Graph I

`regress_phase_conversion.py` 中：

```text
Graph I = Identity ANN
```

仍然作为诊断 baseline 保留。

本轮修改后：

```text
phase_aware 官方 --neuron ann
```

应与：

```text
Graph P = Static Phase ANN
```

语义对齐，而不是与 Graph I 对齐。

不要删除 Graph I。

---

# 4. 核心代码修改：统一构造 evaluation controller

修改：

```text
snn2/evaluation.py
```

不要继续让：

```text
evaluate_tldr.py
evaluate_lm_harness.py
```

各自手写 controller 选择逻辑。

应把 mode-aware evaluation 语义集中到 `snn2/evaluation.py`。

建议新增以下三个 helper；函数名可按项目风格微调，但语义必须一致。

---

## 4.1 新增 final ANN static replacement mode 映射

建议：

```python
def final_ann_replacement_mode(cfg: dict[str, object]) -> str:
    mode = cfg["experiment"]["ann_mode"]

    mapping = {
        "vanilla": "identity",
        "unaware": "identity",
        "phase_aware": "phase",
        "gif_aware": "gif",
    }

    try:
        return mapping[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ANN mode for final evaluation: {mode}"
        ) from exc
```

该函数必须是唯一真源。

不要在 TL;DR 与 lm-eval 两个脚本里各写一份：

```python
if phase_aware ...
elif gif_aware ...
```

避免以后两套 evaluation 漂移。

---

## 4.2 新增统一 `build_evaluation_controller`

建议语义：

```python
def build_evaluation_controller(
    cfg,
    layout,
    *,
    neuron: str,
    base: bool = False,
    rotated_pre_finetuning: bool = False,
):
    ...
```

### A. Base evaluation

必须：

```text
controller.mode = identity
steps = 1
site_root = None
common_clip_enabled = false
```

原始 Base 不安装 aware static surrogate。

---

### B. Rotated pre-finetuning ANN evaluation

必须继续：

```text
controller.mode = identity
steps = 1
common_clip_enabled = false
```

即使使用的 config 是：

```text
phase_aware
gif_aware
```

也不要在：

```text
--rotated-pre-finetuning
```

中启用 Phase/GIF surrogate。

原因：

```text
rotated-pre-finetuning evaluation
```

的用途是诊断：

```text
Rotation + Pre-finetuning Prefix
```

在 ANN 微调前的影响，不是 final aware training-semantics evaluation。

---

### C. Final ANN evaluation：vanilla/unaware

当：

```text
neuron == "ann"
ann_mode in {vanilla, unaware}
```

构造：

```python
SiteController(
    mode="identity",
    site_root=None,
    common_clip_enabled=False,
)
```

`unaware` 虽然需要 SNN2 Rotation integration，但 10 个 replacement site 上仍必须：

```text
identity(x)
```

---

### D. Final ANN evaluation：phase_aware

当：

```text
neuron == "ann"
ann_mode == "phase_aware"
```

必须构造：

```python
SiteController(
    mode="phase",
    site_root=layout.ann_training_site_dir,
    common_clip_enabled=training_common_clip_enabled(cfg),
)
```

必须读取：

```text
layout.ann_training_site_dir
```

中的：

```text
phase_state.pt
```

若：

```yaml
replacement:
  common_clip_enabled: true
```

则和训练一致执行：

```text
PhaseSurrogate
-> common Clip
```

若：

```yaml
replacement:
  common_clip_enabled: false
```

则执行：

```text
PhaseSurrogate
```

不得加载/执行 Clip。

---

### E. Final ANN evaluation：gif_aware

当：

```text
neuron == "ann"
ann_mode == "gif_aware"
```

必须构造：

```python
SiteController(
    mode="gif",
    site_root=layout.ann_training_site_dir,
    common_clip_enabled=training_common_clip_enabled(cfg),
)
```

实际 site operator 为：

```text
StaticGIF.forward()
```

也就是本文所说的：

```text
GIFSurrogate
```

必须读取 ANN training 使用的同一套：

```text
gif_state.pt
```

及可选：

```text
clip_state.pt
```

---

### F. SNN evaluation：所有 ann_mode

当：

```text
neuron in {"phase", "gif", "mtn"}
```

必须始终：

```python
controller = SiteController(
    mode="identity",
    site_root=layout.conversion_site_dir,
    common_clip_enabled=False,
)

steps = controller.set_deployment(neuron)
```

最终 controller 必须分别为：

```text
deploy_phase
deploy_gif
deploy_mtn
```

这里不得读取：

```text
final_ann_replacement_mode(cfg)
```

来决定是否 temporal。

SNN neuron 类型只由：

```text
--neuron phase|gif|mtn
```

决定。

---

# 5. aware final ANN evaluation 必须验证 ANN-training state bundle

当前 ANN training 在 aware mode 开始前会验证：

```python
validate_site_state_bundle(
    layout.ann_training_site_dir,
    require_clip=True,
)
```

final ANN evaluation 在 `phase_aware` / `gif_aware` 下也应做同级别验证。

即使：

```text
common_clip_enabled=false
```

ANN-training calibration 按当前项目协议仍应包含：

```text
clip_state.pt
```

只是 controller 不应用它。

因此 aware final ANN evaluation 建议在 controller 创建前验证：

```python
validate_site_state_bundle(
    layout.ann_training_site_dir,
    require_clip=True,
)
```

确保：

```text
② final static ANN
```

确实依赖训练阶段冻结的完整 ANN-training bundle，而不是某个不完整 conversion-only bundle。

---

# 6. aware final ANN evaluation 必须复用训练时 calibration provenance

本轮目标不是：

```text
重新 calibration 后再评估 static surrogate
```

而是：

```text
加载训练这个 final checkpoint 时实际使用的那套 ANN-training states
```

因此对：

```text
phase_aware
gif_aware
```

建议增加训练 provenance 校验。

当前 `training_result.json` 已记录：

```text
ann_training_prefix_state_sha256
ann_training_prefix_kv_sha256
ann_training_calibration_manifest_sha256
ann_training_prefix_token_ids
```

并且 `snn2/training.py` 已有：

```text
capture_training_artifact_provenance()
verify_training_artifact_provenance_unchanged()
```

建议在 `snn2/training.py` 增加一个轻量 helper，例如：

```python
def validate_recorded_training_artifact_provenance(
    cfg,
    layout,
) -> dict[str, Any]:
    ...
```

语义：

1. 读取：

```text
layout.ann_dir / "training_result.json"
```

2. 从其中恢复训练结束时保存的 frozen provenance；
3. 调用现有 provenance 检查逻辑；
4. 当前磁盘的：

```text
Pre-finetuning Prefix
ANN-training calibration manifest
```

任一 hash 与训练时不一致立即失败。

然后在 aware final ANN evaluation 启动时调用。

目的：

> 保证 ② 真正使用训练 ① 时冻结的 static replacement states。

不要为 vanilla/unaware 强行要求 aware calibration provenance。

---

# 7. 修改 `evaluation_calibration_metadata`

当前：

```text
snn2/evaluation.py::evaluation_calibration_metadata()
```

把：

```text
neuron == "ann"
```

全部视为 calibration inactive。

本轮之后这个逻辑已经不正确。

应改成：

## 7.1 Base / rotated-pre-finetuning

继续：

```json
{
  "calibration_source_stage": null,
  "reused_ann_training_artifacts": false,
  "post_finetuning_recalibration": false,
  "calibration_root": null
}
```

---

## 7.2 vanilla/unaware final ANN

继续：

```json
{
  "calibration_source_stage": null,
  "reused_ann_training_artifacts": false,
  "post_finetuning_recalibration": false,
  "calibration_root": null
}
```

因为它们的 final ANN evaluation 是：

```text
identity(x)
```

不读取 neuron calibration states。

注意：

```text
unaware 的 SNN conversion
```

仍然使用 Post-finetuning conversion calibration；这里只说的是：

```text
unaware final ANN evaluation
```

---

## 7.3 phase_aware/gif_aware final ANN

必须改成：

```json
{
  "calibration_source_stage": "ann_training",
  "reused_ann_training_artifacts": true,
  "post_finetuning_recalibration": false,
  "calibration_root": "<layout.ann_training_site_dir>"
}
```

这里的：

```text
reused_ann_training_artifacts
```

表示 final ANN evaluation 复用 ANN training 使用的 frozen bundle。

如果担心旧字段原本主要描述 conversion，可以额外增加更清晰的新字段：

```text
ann_evaluation_reuses_ann_training_artifacts
```

但不要让旧 metadata 继续错误地声称：

```text
phase_aware final ANN 不依赖 calibration
```

---

# 8. 增加统一 forward-semantics metadata

为了明确区分 12 个 final ANN checkpoint 实际使用的是：

```text
identity(x)
PhaseSurrogate
GIFSurrogate
```

建议在：

```text
snn2/evaluation.py
```

增加：

```python
def evaluation_forward_metadata(...):
    ...
```

TL;DR 和 lm-eval 共用。

至少保存以下字段：

```text
ann_mode
evaluation_forward_kind
controller_mode
temporal_execution
static_replacement_enabled
static_replacement_impl
evaluation_common_clip_applied
replacement_state_root
```

推荐固定枚举：

### vanilla / unaware final ANN

```json
{
  "evaluation_forward_kind": "identity_ann",
  "controller_mode": "identity",
  "temporal_execution": false,
  "static_replacement_enabled": false,
  "static_replacement_impl": null,
  "evaluation_common_clip_applied": false,
  "replacement_state_root": null
}
```

### phase_aware final ANN

```json
{
  "ann_mode": "phase_aware",
  "evaluation_forward_kind": "phase_surrogate_ann",
  "controller_mode": "phase",
  "temporal_execution": false,
  "static_replacement_enabled": true,
  "static_replacement_impl": "PhaseSurrogate.forward",
  "evaluation_common_clip_applied": false,
  "replacement_state_root": "<ANN-training site root>"
}
```

若 common Clip 开：

```text
evaluation_common_clip_applied = true
```

### gif_aware final ANN

```json
{
  "ann_mode": "gif_aware",
  "evaluation_forward_kind": "gif_surrogate_ann",
  "controller_mode": "gif",
  "temporal_execution": false,
  "static_replacement_enabled": true,
  "static_replacement_impl": "StaticGIF.forward",
  "evaluation_common_clip_applied": false,
  "replacement_state_root": "<ANN-training site root>"
}
```

### Temporal SNN

例如 Phase：

```json
{
  "evaluation_forward_kind": "temporal_phase_snn",
  "controller_mode": "deploy_phase",
  "temporal_execution": true,
  "static_replacement_enabled": false,
  "static_replacement_impl": null,
  "evaluation_common_clip_applied": false,
  "replacement_state_root": "<conversion site root>"
}
```

GIF / MTN 分别：

```text
temporal_gif_snn
temporal_mtn_snn
```

---

# 9. 保留现有 metadata，但修正容易误解的字段

当前已有：

```text
ann_training_common_clip_enabled
```

这个字段表示：

> checkpoint 在 ANN training 时的 common Clip 设置

它不能再被误解成：

> evaluation 当前是否执行 Clip

所以必须保留该字段的历史语义，并新增：

```text
evaluation_common_clip_applied
```

示例：

### phase_aware，训练和 final ANN eval 都开启 Clip

```json
{
  "ann_training_common_clip_enabled": true,
  "evaluation_common_clip_applied": true
}
```

### phase_aware 的 Temporal Phase SNN

```json
{
  "ann_training_common_clip_enabled": true,
  "evaluation_common_clip_applied": false
}
```

这样能清楚表达：

```text
checkpoint 来源训练时开过 Clip
```

但：

```text
SNN 本身没有执行 Clip
```

---

# 10. 修改 `scripts/evaluate_tldr.py`

当前核心问题是：

```python
controller = SiteController(
    mode="identity",
    ...
)

steps = 1 if args.neuron == "ann" else controller.set_deployment(args.neuron)
```

把这段替换为统一 helper，例如：

```python
controller, steps = build_evaluation_controller(
    cfg,
    layout,
    neuron=args.neuron,
    base=args.base,
    rotated_pre_finetuning=args.rotated_pre_finetuning,
)
```

之后现有：

```python
if args.neuron != "ann" or cfg["rotation"]["enabled"]:
    install_model_integration(...)
```

总体可以保留。

它会自然得到：

```text
vanilla final ANN:
    rotation false
    identity
    不安装 SNN2 integration

unaware final ANN:
    rotation true
    identity controller
    安装 Rotation/SNN2 integration，但 sites 为 identity

phase_aware final ANN:
    rotation true
    phase controller
    安装同训练语义 static Phase replacement

gif_aware final ANN:
    rotation true
    gif controller
    安装同训练语义 static GIF replacement
```

这正是目标。

---

## 10.1 TL;DR greedy generation 不需要改成 temporal

`greedy_generate()` 当前只有：

```text
controller.mode.startswith("deploy_")
```

才进入：

```text
temporal_forward()
```

因此：

```text
mode = phase
mode = gif
```

仍然会走普通非 temporal：

```python
model(...).logits
```

这是正确行为。

不要修改该条件为：

```text
phase/gif 也走 temporal_forward
```

否则会把 aware ANN evaluation 错误改成完整 SNN execution。

---

## 10.2 TL;DR metrics 增加 forward metadata

在：

```text
metrics.json
```

中加入第 8 节定义的统一 metadata。

尤其必须能看到：

```text
evaluation_forward_kind
controller_mode
temporal_execution
static_replacement_impl
evaluation_common_clip_applied
replacement_state_root
```

保持现有：

```text
model_variant = finetuned_ann
```

也可以，不要求破坏旧字段。

不要依赖：

```text
model_variant
```

单独表达 static replacement 类型，新字段专门负责这件事。

---

# 11. 修改 `scripts/evaluate_lm_harness.py`

与 TL;DR 完全相同。

把当前：

```python
controller = SiteController(
    mode="identity",
    ...
)

steps = (
    1
    if args.neuron == "ann"
    else controller.set_deployment(args.neuron)
)
```

替换为同一个：

```text
build_evaluation_controller(...)
```

不得在 lm-eval 中重新实现一套 mode mapping。

---

## 11.1 `EvaluationModelProxy` 不需要改成 temporal static surrogate

当前：

```python
if self.controller.mode.startswith("deploy_"):
    temporal_forward(...)
else:
    self.model(...)
```

必须继续保留。

因此：

```text
controller.mode = phase
controller.mode = gif
```

仍属于：

```text
non-temporal ANN forward
```

这是正确的。

---

## 11.2 lm-eval results 增加同样 metadata

写入：

```text
results.json
-> snn2_metadata
```

字段与 TL;DR 保持同名、同枚举。

禁止 TL;DR 写：

```text
phase_surrogate_ann
```

而 lm-eval 写另一套：

```text
static_phase
```

两套脚本必须共享 helper。

---

# 12. Prefix 行为保持当前协议

本轮不改变：

```text
evaluation.prefix_enabled
```

final ANN 与 SNN evaluation 继续使用当前 Prefix 开关。

Aware mode：

```text
phase_aware
gif_aware
```

final evaluation 的 Prefix artifact 来源仍为：

```text
Pre-finetuning Prefix
```

即：

```text
final_evaluation_prefix_artifact_stage(cfg) == "pre_finetuning"
```

vanilla/unaware 仍按当前 post-finetuning 规则。

必须继续保证：

```text
Prefix K/V runtime 在 aware static replacement 下经过 Site 3/4 surrogate
Prefix K/V runtime 在 SNN deployment 下经过 Site 3/4 temporal neuron
```

不得因为 final ANN controller 从 identity 改为 phase/gif 而让 Prefix K/V bypass replacement。

---

## 12.1 关于“训练与评估完全一致”的边界

本轮要求一致的是：

```text
activation replacement semantics
state source
common-Clip semantics
```

训练和 evaluation 本身仍然存在正常差异：

```text
train() vs eval()
teacher forcing vs generation / lm-eval scoring
梯度开启 vs no_grad
```

另外用户仍可通过：

```yaml
evaluation:
  prefix_enabled: false
```

做 Prefix ablation。

因此不要在文档中宣称：

> 训练和评估整个计算过程 bitwise 完全相同。

正确表述是：

> final ANN evaluation 的 activation replacement forward 与对应 ANN training 使用相同的 static surrogate operator、同一 ANN-training states，并镜像 common-Clip 设置。

对于 `② vs ③` 的严格 conversion 对比，ANN 和 SNN evaluation 必须使用相同：

```text
evaluation.prefix_enabled
test samples
decode policy
```

---

# 13. Phase global final RMSNorm neuron 边界保持不变

当前 regression 已确认：

```text
Static Phase ANN
```

与：

```text
Temporal Phase deployment
```

在 global final RMSNorm Phase topology 上存在设计差异：

```text
Static Phase ANN:
    不执行 global final RMSNorm Phase

Temporal Phase SNN:
    执行 global final RMSNorm Phase
```

最新 regression 结论指出：

```text
final RMSNorm Phase bypass 未显著缩小 P vs S
```

因此本轮：

```text
不要给 phase_aware ANN training/final ANN evaluation 新增 global final Phase
不要从 Temporal Phase SNN 删除 global final Phase
```

只让官方 final ANN evaluation 与当前 Graph P 对齐。

---

# 14. `scripts/verify_artifacts.py` 必须防止旧 identity ANN 结果被误认为新结果

因为本轮建议不改变 evaluation 输出目录，所以旧的：

```text
phase_aware/ann/evaluation/...
gif_aware/ann/evaluation/...
```

可能已经存在由旧 identity controller 生成的结果。

如果只改代码、不加强 verifier，旧结果可能被误认为新语义。

因此必须修改：

```text
scripts/verify_artifacts.py
```

读取 final ANN evaluation metadata，并强制校验。

---

## 14.1 预期 final ANN metadata

### vanilla

```text
evaluation_forward_kind == identity_ann
controller_mode == identity
temporal_execution == false
static_replacement_enabled == false
evaluation_common_clip_applied == false
```

### unaware

同样：

```text
identity_ann
identity
false
```

但 Rotation 仍按现有 metadata 验证。

### phase_aware

```text
evaluation_forward_kind == phase_surrogate_ann
controller_mode == phase
temporal_execution == false
static_replacement_enabled == true
static_replacement_impl == PhaseSurrogate.forward
calibration_source_stage == ann_training
replacement_state_root == layout.ann_training_site_dir
evaluation_common_clip_applied == training_common_clip_enabled(cfg)
```

### gif_aware

```text
evaluation_forward_kind == gif_surrogate_ann
controller_mode == gif
temporal_execution == false
static_replacement_enabled == true
static_replacement_impl == StaticGIF.forward
calibration_source_stage == ann_training
replacement_state_root == layout.ann_training_site_dir
evaluation_common_clip_applied == training_common_clip_enabled(cfg)
```

旧结果没有这些字段时：

```text
verification 必须失败
```

提示重新运行 final ANN evaluation。

这样不需要修改目录结构，也能避免旧 identity aware ANN 指标混入新实验。

---

## 14.2 SNN verifier 保持 temporal hard gate

对已有 SNN metadata 至少继续验证：

```text
neuron == phase/gif/mtn
full_temporal_steps 与 conversion metadata 一致
```

若新 metadata 已写入，则额外校验：

```text
controller_mode == deploy_<neuron>
temporal_execution == true
evaluation_common_clip_applied == false
```

不要因为 final ANN evaluation 改为 static surrogate 而放松 SNN temporal 验证。

---

# 15. 不修改 ANN/SNN evaluation 输出目录层级

本轮建议：

```text
不新增
ann_forward_identity/
ann_forward_phase/
ann_forward_gif/
```

原因：

1. 12 个 final checkpoint 本身已经位于各自 `ann_mode` run root；
2. 新 metadata 可以精确标明实际 forward；
3. 保持现有工具、脚本和结果目录兼容；
4. `verify_artifacts.py` 可以通过新字段拒绝旧语义结果。

因此 TL;DR 仍：

```text
.../<run>/ann/evaluation/tldr/<test_samples_dir>/
└── prefix_enabled_ture|prefix_enabled_false/
```

lm-eval 仍：

```text
.../<run>/ann/evaluation/lm_harness/
└── prefix_enabled_ture|prefix_enabled_false/
```

重新 evaluation 时直接覆盖该 checkpoint 对应旧结果即可。

---

# 16. 12 个 final ANN checkpoint 的正式评估规则

当前实验矩阵为：

```text
3 个 model/task 实验
×
4 个 ann_mode
=
12 个 final ANN checkpoint
```

分别是：

```text
Qwen3-1.7B / TL;DR:
    vanilla
    unaware
    phase_aware
    gif_aware

Qwen3-8B / TL;DR:
    vanilla
    unaware
    phase_aware
    gif_aware

Llama3-8B / Tulu3:
    vanilla
    unaware
    phase_aware
    gif_aware
```

12 个 checkpoint 使用同样的：

```bash
--neuron ann
```

但内部必须自动区分：

```text
vanilla     -> identity(x)
unaware     -> identity(x)
phase_aware -> PhaseSurrogate.forward(x)
gif_aware   -> StaticGIF.forward(x)
```

用户不需要增加：

```text
--static-phase
--static-gif
```

之类的新 CLI flag。

**模式由 final checkpoint 所属 config 的 `experiment.ann_mode` 唯一决定。**

---

# 17. Phase-aware 的正式两条评估命令

修改完成后：

## 17.1 ② Static Phase ANN evaluation

```bash
CFG_17_P=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml

export CUDA_VISIBLE_DEVICES=6

accelerate launch --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$CFG_17_P" \
  --neuron ann
```

必须实际得到：

```text
final phase_aware checkpoint
+
Rotation
+
evaluation Prefix（按 evaluation.prefix_enabled）
+
10 × PhaseSurrogate.forward()
+
optional common Clip（严格镜像训练设置）
```

不得再得到：

```text
10 × identity
```

---

## 17.2 ③ Temporal Phase SNN evaluation

```bash
export CUDA_VISIBLE_DEVICES=6

accelerate launch --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$CFG_17_P" \
  --neuron phase
```

必须实际得到：

```text
same final checkpoint
+
conversion artifacts
+
Temporal Phase SNN
+
controller.mode = deploy_phase
```

这两条命令才构成正式：

```text
② vs ③
```

比较。

---

# 18. GIF-aware 的正式两条评估命令

## 18.1 Static GIF ANN

```bash
CFG_17_G=configs/generated/exp1_qwen3_1_7b_tldr__gif_aware.yaml

accelerate launch --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$CFG_17_G" \
  --neuron ann
```

必须：

```text
controller.mode = gif
StaticGIF.forward()
```

---

## 18.2 Temporal GIF SNN

```bash
accelerate launch --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$CFG_17_G" \
  --neuron gif
```

必须：

```text
controller.mode = deploy_gif
Temporal GIF neuron
```

---

# 19. 注意：SNN neuron 类型不等于 ANN training mode

必须在所有文档中写清楚。

例如：

```text
phase_aware final checkpoint
```

不仅能：

```bash
--neuron phase
```

也仍能按当前实验协议：

```bash
--neuron gif
--neuron mtn
```

它们分别表示：

```text
phase_aware weights -> Temporal GIF SNN
phase_aware weights -> Temporal MTN SNN
```

同理：

```text
gif_aware weights
```

也可以转换到：

```text
Temporal Phase / GIF / MTN
```

所以：

```text
ann_mode
```

控制的是：

> ANN training / final ANN static forward 使用什么 replacement。

而：

```text
--neuron
```

在 SNN evaluation 时控制的是：

> 最终部署成哪一种 Temporal SNN neuron。

二者禁止混淆。

---

# 20. 修改 `实验执行总结.md`

这是本轮文档修改的最高优先级。

必须至少修改以下部分。

---

## 20.1 ANN training 一节

明确写：

```text
vanilla:
    identity(x)

unaware:
    identity(x)
    但保留 Rotation + Pre-finetuning Prefix

phase_aware:
    PhaseSurrogate.forward()
    optional common Clip

gif_aware:
    GIFSurrogate
    当前实现 StaticGIF.forward()
    optional common Clip
```

并说明：

```text
phase_aware/gif_aware 仍属于 ANN fine-tuning
```

因为 site 内 temporal/local coding 会立即聚合回 static tensor，T 不跨层传播。

---

## 20.2 Final ANN evaluation 一节

把旧的“所有 `--neuron ann` 都是 identity”表述全部删除。

加入一张与第 2 节等价的四模式表。

必须明确：

```text
--neuron ann != identity 的同义词
```

而是：

```text
--neuron ann = non-temporal final ANN evaluation
```

并按 `ann_mode` 恢复训练语义。

---

## 20.3 SNN evaluation 一节

必须明确：

```text
vanilla
unaware
phase_aware
gif_aware
```

四种 final checkpoint 进入 SNN evaluation 后：

```text
--neuron phase -> deploy_phase
--neuron gif   -> deploy_gif
--neuron mtn   -> deploy_mtn
```

全部为 Temporal SNN。

---

## 20.4 Step 9：12 个 final ANN checkpoint evaluation

Step 9 必须写明：

```text
12 个 final ANN checkpoint 虽然统一使用 --neuron ann，
但实际 forward 由 ann_mode 自动决定。
```

并写出：

```text
vanilla/unaware = identity
phase_aware     = PhaseSurrogate
gif_aware       = GIFSurrogate / StaticGIF
```

命令本身无需增加新 flag。

---

## 20.5 Step 10：SNN conversion / evaluation

明确：

```text
Step 10 与 Step 9 的 static ANN semantics 是两回事。
```

Step 10 无论哪个 `ann_mode`，只要：

```text
--neuron != ann
```

就是 full-temporal SNN。

---

## 20.6 附录 A：整体依赖图

Aware 分支改成：

```text
aware ANN training
    └── final ANN
        ├── final ANN evaluation
        │   └── 复用 ANN-training static surrogate states
        └── SNN conversion/evaluation
            └── 复用 conversion 对应 states，进入 temporal deployment
```

不要再把 aware final ANN evaluation 描述成 identity graph。

---

## 20.7 附录 D：Calibration

ANN-training calibration 的用途补充：

```text
不仅服务 phase_aware/gif_aware ANN training，
还服务对应 final ANN static-surrogate evaluation。
```

对于 aware：

```text
training 和 final ANN evaluation
```

读取同一套：

```text
layout.ann_training_site_dir
```

---

## 20.8 附录 E.7：Final ANN 与 SNN evaluation

必须补充：

```text
ANN evaluation result metadata 会记录实际 forward kind：
identity_ann
phase_surrogate_ann
gif_surrogate_ann
```

SNN：

```text
temporal_phase_snn
temporal_gif_snn
temporal_mtn_snn
```

---

## 20.9 附录 F：逐点命令说明

把：

```text
evaluate_tldr.py / evaluate_lm_harness.py
```

的说明改成：

> `--neuron ann` 对 final checkpoint 执行 mode-aware non-temporal ANN forward；vanilla/unaware 为 identity，phase_aware/gif_aware 分别复用 ANN-training Phase/GIF static surrogate。`--neuron phase/gif/mtn` 始终执行 full-temporal SNN。

---

## 20.10 附录 H：Phase Conversion Debug / Regression

当前存在旧表述：

```text
官方 evaluate_tldr.py --neuron ann 使用 identity controller
```

本轮必须更新为：

```text
历史旧实现中官方 --neuron ann 对 phase_aware 仍使用 identity；
本轮修正后，phase_aware final --neuron ann 已正式对齐 Graph P，
即 training-time static Phase graph。
```

然后明确：

```text
Graph I
```

只保留 regression diagnostic baseline。

对于：

```text
common_clip_enabled=false
```

且 ANN/SNN evaluation 使用相同 Prefix/test/decode 设置时：

```text
官方 phase_aware --neuron ann
≈ Graph P 语义

官方 --neuron phase
= Graph S 正式 temporal deployment 语义
```

因此正式性能比较可以直接对应：

```text
② vs ③
```

---

# 21. 修改 `AGENTS.md`

必须新增不可违反的项目规则。

建议追加以下规则，措辞可微调但语义必须完整。

## Rule 9：Final ANN mode-aware semantics

```text
Final ANN evaluation (`--neuron ann`) 必须复现对应 ANN training 的 static activation semantics：
vanilla/unaware = identity(x)；
phase_aware = PhaseSurrogate.forward()；
gif_aware = static GIF surrogate（当前 StaticGIF.forward()）。
```

---

## Rule 10：ANN 与 SNN CLI 语义

```text
`--neuron ann` 只表示 non-temporal ANN execution，不等价于 identity。
`--neuron phase|gif|mtn` 在正式 evaluation 中始终表示 full-temporal SNN deployment。
```

---

## Rule 11：Aware evaluation state reuse

```text
phase_aware/gif_aware final ANN evaluation 必须读取与对应 ANN training 相同的 ANN-training calibration states，并镜像 replacement.common_clip_enabled；不得改用 Post-finetuning conversion calibration。
```

---

## Rule 12：SNN 不受 ANN static mode 替代

```text
任何 ann_mode 进入 SNN evaluation 后都必须使用选定 neuron 的 deploy_* temporal path；不得因为 checkpoint 来自 phase_aware/gif_aware 而用 static surrogate 代替 Temporal SNN neuron。
```

---

## Rule 13：Base / rotated-pre-finetuning 边界

```text
Base baseline 与 rotated-pre-finetuning ANN diagnostic 保持 identity activation semantics；本轮 mode-aware static surrogate 只作用于 final ANN checkpoint evaluation。
```

---

# 22. 修改 `README.md`

README 至少增加一个：

```text
## ANN Training / Final ANN Evaluation / SNN Evaluation Semantics
```

并放入下表：

| Mode | ANN training | Final ANN `--neuron ann` | SNN `--neuron phase/gif/mtn` |
|---|---|---|---|
| vanilla | identity | identity | temporal selected neuron |
| unaware | identity | identity | temporal selected neuron |
| phase_aware | PhaseSurrogate | PhaseSurrogate | temporal selected neuron |
| gif_aware | GIF surrogate / StaticGIF | GIF surrogate / StaticGIF | temporal selected neuron |

README 还必须明确：

```text
phase_aware/gif_aware 的 static surrogate 不使其成为完整 SNN training
```

和：

```text
Temporal SNN 只由 deploy_* 路径定义
```

---

# 23. 同步修改 `代码结构总结.md`

`AGENTS.md` 已规定：

> `代码结构总结.md` 只保留目录结构，但任何文件功能发生变化必须同步更新一句话描述。

因此至少更新：

```text
snn2/evaluation.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
```

如果增加训练 provenance helper，还更新：

```text
snn2/training.py
```

建议描述变为：

```text
snn2/evaluation.py
— 统一实现 mode-aware static ANN evaluation、full-temporal SNN evaluation、forward metadata 与 generation/proxy。

evaluate_tldr.py
— 按 final ann_mode 执行 identity/Phase/GIF static ANN 或按 --neuron 执行 Temporal SNN 的 TL;DR 评估。

evaluate_lm_harness.py
— 按 final ann_mode 执行 identity/Phase/GIF static ANN 或按 --neuron 执行 Temporal SNN 的 lm-eval。

verify_artifacts.py
— 验证 mode-aware final ANN forward metadata、conversion provenance 与 SNN temporal evaluation 完整性。
```

不要把其它说明章节加入 `代码结构总结.md`。

---

# 24. 测试修改

优先扩展：

```text
tests/test_evaluation_paths.py
tests/test_phase_conversion_regression.py
tests/test_generated_configs.py
```

如果认为职责过重，也可新增：

```text
tests/test_evaluation_semantics.py
```

但若新增文件必须同步 `代码结构总结.md`。

---

## 24.1 Final ANN mode mapping test

参数化：

```text
vanilla     -> identity
unaware     -> identity
phase_aware -> phase
gif_aware   -> gif
```

---

## 24.2 Controller build test

测试：

### vanilla

```text
neuron=ann
controller.mode == identity
steps == 1
common_clip_enabled == false
```

### unaware

同上。

### phase_aware

分别测试：

```text
common_clip_enabled=true
common_clip_enabled=false
```

要求：

```text
controller.mode == phase
controller.site_root == layout.ann_training_site_dir
controller.common_clip_enabled == config
steps == 1
```

### gif_aware

要求：

```text
controller.mode == gif
controller.site_root == layout.ann_training_site_dir
controller.common_clip_enabled == config
steps == 1
```

---

## 24.3 Base / rotated pre-finetuning test

即使 cfg 的：

```text
ann_mode = phase_aware
```

只要：

```text
base=True
```

或：

```text
rotated_pre_finetuning=True
```

都必须：

```text
controller.mode == identity
```

---

## 24.4 所有 ann_mode 的 SNN temporal test

参数化：

```text
ann_mode ∈ {
    vanilla,
    unaware,
    phase_aware,
    gif_aware
}

neuron ∈ {
    phase,
    gif,
    mtn
}
```

共 12 种组合。

必须验证最终：

```text
controller.mode == deploy_<neuron>
```

而不是：

```text
phase
gif
identity
```

并验证：

```text
common_clip_enabled == false
```

---

## 24.5 `evaluation_calibration_metadata` test

旧测试：

```text
test_final_ann_evaluation_has_no_calibration_metadata
```

不能再对四种 mode 全部期待 `None`。

应拆成：

### vanilla/unaware

仍然：

```text
calibration_source_stage = None
calibration_root = None
```

### phase_aware/gif_aware

必须：

```text
calibration_source_stage = ann_training
reused_ann_training_artifacts = true
calibration_root = ann_training_site_dir
```

---

## 24.6 Forward metadata test

参数化验证：

```text
vanilla     -> identity_ann
unaware     -> identity_ann
phase_aware -> phase_surrogate_ann
gif_aware   -> gif_surrogate_ann

phase SNN -> temporal_phase_snn
gif SNN   -> temporal_gif_snn
mtn SNN   -> temporal_mtn_snn
```

---

## 24.7 common Clip metadata test

对 aware final ANN：

```text
evaluation_common_clip_applied
==
training_common_clip_enabled(cfg)
```

对 SNN：

```text
evaluation_common_clip_applied == false
```

---

## 24.8 Phase regression 与正式 ANN eval 对齐 test

在：

```text
tests/test_phase_conversion_regression.py
```

增加一个轻量 test：

对于：

```text
phase_aware
common_clip_enabled=false
neuron=ann
```

正式 evaluation controller 的关键语义必须是：

```text
mode == phase
site_root == ann_training_site_dir
common_clip_enabled == false
```

与 Graph P constructor 条件一致。

不要让 regression 自己依赖正式 evaluation 来生成 Graph P；只测试二者协议一致即可，保留 regression 的独立性。

---

## 24.9 12 份 generated config semantic test

在：

```text
tests/test_generated_configs.py
```

对 materialize 后 12 个 config 检查：

```text
vanilla/unaware -> expected final ANN identity
phase_aware     -> expected final ANN phase
gif_aware       -> expected final ANN gif
```

避免未来 config resolve 改动导致 evaluation 语义漂移。

---

# 25. 验收测试命令

必须至少运行：

```bash
conda run -n snn2 \
python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated
```

然后：

```bash
conda run -n snn2 pytest -q
```

全部通过。

---

# 26. 必须做一个真实 Phase smoke test

使用已经训练完成且：

```text
replacement.common_clip_enabled=false
```

的 Qwen3-1.7B `phase_aware` final checkpoint。

先跑：

```bash
CFG_17_P=<对应 common_clip_false 的 phase_aware config>

CUDA_VISIBLE_DEVICES=6 \
accelerate launch --num_processes 1 \
scripts/evaluate_tldr.py \
  --config "$CFG_17_P" \
  --neuron ann
```

检查新 `metrics.json` 至少：

```text
evaluation_forward_kind = phase_surrogate_ann
controller_mode = phase
temporal_execution = false
static_replacement_enabled = true
static_replacement_impl = PhaseSurrogate.forward
evaluation_common_clip_applied = false
calibration_source_stage = ann_training
```

然后跑：

```bash
CUDA_VISIBLE_DEVICES=6 \
accelerate launch --num_processes 1 \
scripts/evaluate_tldr.py \
  --config "$CFG_17_P" \
  --neuron phase
```

检查：

```text
evaluation_forward_kind = temporal_phase_snn
controller_mode = deploy_phase
temporal_execution = true
evaluation_common_clip_applied = false
```

---

# 27. 与已有 Phase conversion regression 结果交叉验证

修改完成后再次运行现有：

```bash
CUDA_VISIBLE_DEVICES=6 \
conda run --no-capture-output -n snn2 \
python scripts/regress_phase_conversion.py \
  --config <common_clip_false_resolved_config.yaml> \
  --sample-index 0 \
  --max-input-tokens 64 \
  --decode-steps 16
```

要求：

1. regression Graph P 行为不变；
2. 最新 BF16 RMSNorm 修正不回退；
3. first-divergence 诊断结果不应因为“官方 ANN evaluation 改 mode”而改变；
4. 正式 `--neuron ann` 的 Phase static semantics 与 Graph P 协议一致；
5. 正式 `--neuron phase` 仍与 Graph S 的 temporal semantics 一致。

---

# 28. 重跑边界

本轮修改：

```text
不改变 ANN training 数学
不改变 calibration 数学
不改变 final checkpoint weights
不改变 SNN conversion 数学
不改变 Temporal SNN 算子
```

因此正常情况下：

## 不需要重跑

```text
prepare_data
prepare_rotation
Pre-finetuning Prefix
ANN-training calibration
12 个 ANN training
Post-finetuning Prefix/calibration
SNN conversion
```

只要现有 artifact provenance 校验通过即可。

---

## 必须重新跑

```text
12 个 final ANN checkpoint 的 ANN evaluation
```

因为：

```text
phase_aware/gif_aware
```

旧 ANN result 是错误的 identity semantics；

而：

```text
vanilla/unaware
```

虽然数值语义本身仍是 identity，但建议也统一重跑，以生成新的明确 forward metadata，并让 verifier 使用一致 schema。

---

## SNN evaluation

本轮没有修改 SNN 数学。

已有合法 SNN evaluation：

```text
数值结果可继续复用
```

不应仅因为本轮 ANN evaluation 修正就强制重新跑昂贵的 SNN evaluation。

未来重新运行 SNN evaluation 时，新 metadata 应明确记录：

```text
temporal_<neuron>_snn
deploy_<neuron>
```

---

# 29. 结果解释规则

本轮完成后，对 Phase：

旧比较：

```text
identity final ANN
vs
Temporal Phase SNN
```

不再作为：

```text
Phase conversion loss
```

的正式定义。

正式定义改为：

```text
phase_aware final ANN with PhaseSurrogate
vs
Temporal Phase SNN
```

即：

```text
② vs ③
```

如果：

```text
② 已经很差
③ 与 ② 接近
```

则主要问题是：

```text
static Phase replacement / training adaptation
```

不是 temporal conversion。

如果：

```text
② 很好
③ 明显下降
```

才说明：

```text
Static Phase -> Temporal Phase
```

存在真正 conversion loss。

如果：

```text
② 已下降
③ 又进一步下降
```

则两种损失同时存在，继续使用已有 unit-level regression 定位：

```text
Graph P vs Graph S
```

的额外 conversion gap。

---

# 30. 最终必须满足的硬性验收条件

完成后以下条件必须全部成立。

## ANN training

```text
vanilla     -> identity(x)
unaware     -> identity(x)
phase_aware -> PhaseSurrogate.forward()
gif_aware   -> StaticGIF.forward()
```

---

## Final ANN evaluation：`--neuron ann`

```text
vanilla     -> identity(x)
unaware     -> identity(x)
phase_aware -> PhaseSurrogate.forward()
gif_aware   -> StaticGIF.forward()
```

其中 aware：

```text
复用 ANN-training site states
镜像 replacement.common_clip_enabled
```

---

## SNN evaluation

对任意：

```text
ann_mode
```

只要：

```text
--neuron phase
--neuron gif
--neuron mtn
```

就必须分别：

```text
deploy_phase
deploy_gif
deploy_mtn
```

且：

```text
Temporal execution = true
common Clip = false
```

---

## Phase conversion 正式性能比较

```text
② phase_aware --neuron ann
=
Static PhaseSurrogate ANN

③ phase_aware --neuron phase
=
Temporal Phase SNN
```

正式 conversion loss：

```text
② vs ③
```

---

# 31. 禁止事项

本轮禁止顺手进行以下改动：

```text
不要修改 Phase τ 算法
不要修改 Phase T/base/surrogate_slope
不要修改 GIF qmax/chunk 算法
不要修改 MTN
不要修改 10-site topology
不要新增 Site 11
不要修改 Rotation
不要修改 Hadamard precision
不要修改 Prefix discovery 算法
不要修改 calibration sample 数
不要修改 training LR
不要删除 global final RMSNorm Phase
不要让 static Phase/GIF 走 temporal_forward
不要让 deploy_phase/gif/mtn 退化成 static surrogate
不要让 SNN 加 common Clip
不要把历史 regression Graph I 删除
```

---

# 32. Codex 完成后应返回的检查摘要

完成代码后请输出简短检查摘要，至少包含：

```text
1. 修改了哪些文件；
2. final_ann_replacement_mode 四种 mode 的映射；
3. TL;DR 与 lm-eval 是否共用同一 evaluation controller helper；
4. phase_aware --neuron ann 最终 controller.mode；
5. gif_aware --neuron ann 最终 controller.mode；
6. vanilla/unaware --neuron ann 最终 controller.mode；
7. 任意 --neuron phase/gif/mtn 是否仍为 deploy_*；
8. aware final ANN 是否复用 ANN-training site states；
9. aware final ANN common Clip 是否镜像训练设置；
10. SNN 是否仍完全不使用 common Clip；
11. 新 evaluation metadata 字段；
12. pytest 结果；
13. Phase smoke test 的 metrics 中 forward kind/controller mode；
14. 是否确认无需重新训练 12 个 final ANN checkpoint。
```
