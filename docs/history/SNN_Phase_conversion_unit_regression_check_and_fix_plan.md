# Qwen3-1.7B `phase_aware -> Phase SNN` 单元级 Conversion Regression 检查与修正方案

## 0. 背景与目标

仓库：

```text
/home/wangwenkang/SNN
https://github.com/wangwk699/SNN
```

本文基于当前 `main`：

```text
a33b22bc91ed3d677b31c250baaf0eaf575e352e
```

当前重新实验结果：

| common_clip_enabled | LR | Graph / Mode | ROUGE Avg. |
|---|---:|---|---:|
| true | 1e-8 | final ANN（当前官方 ANN evaluation） | 0.2133 |
| true | 1e-8 | Phase SNN | 0.0179 |
| false | 1e-8 | final ANN（当前官方 ANN evaluation） | 0.2133 |
| false | 1e-8 | Phase SNN | 0.0149 |
| false | 1e-7 | final ANN（当前官方 ANN evaluation） | 0.2096 |
| false | 1e-7 | Phase SNN | 0.0199 |

即使：

```yaml
replacement:
  common_clip_enabled: false
```

Phase SNN 仍发生明显性能坍塌。

本次不再继续凭静态代码猜测原因，而是加入 **Phase conversion unit-level regression**，直接比较：

```text
同一个 final phase_aware ANN checkpoint
+ 同一套 Pre-finetuning Prefix
+ 同一套 ANN-training Phase states
+ common_clip_enabled=false
```

在以下两条实际计算图中的中间结果：

```text
A. training-semantics static Phase graph
   即 phase_aware ANN fine-tuning 时的前向语义：
   controller.mode = "phase"
   common_clip_enabled = false

B. converted temporal Phase SNN graph
   controller.mode = "deploy_phase"
```

目标是定位：

> **第一个发生显著数值不一致的 layer / operator / site，并只修正经 regression 证实的错误。**

不要先修改 Phase 参数、τ、Prefix、训练 LR、T、base、Site 数量或其它算法设置。

---

## 1. 必须先澄清：当前官方 `--neuron ann` 不是 training-time Phase graph

当前：

```text
scripts/evaluate_tldr.py --neuron ann
```

创建：

```python
SiteController(mode="identity")
```

因此表格里的：

```text
ANN = 0.2133 / 0.2096
```

实际评估的是：

```text
final checkpoint weights
+
identity activation graph
```

而 `phase_aware` ANN fine-tuning 时实际使用：

```python
SiteController(
    mode="phase",
    common_clip_enabled=False,
)
```

即训练计算图是：

```text
Phase replacement ON
Clip OFF
```

所以本次 regression 必须同时区分三条 graph：

```text
Graph I   : identity ANN
Graph P   : static Phase ANN（training-time semantics）
Graph SNN : temporal Phase SNN
```

最核心比较是：

```text
Graph P vs Graph SNN
```

而不是：

```text
Graph I vs Graph SNN
```

Graph I 只用于判断 Phase replacement 本身造成了多少损失。

---

## 2. 当前代码中的高优先级检查点：global final RMSNorm Phase

当前代码存在以下结构：

```text
phase_aware ANN training:
    final RMSNorm
    -> lm_head
```

而 `deploy_phase`：

```text
temporal final RMSNorm
-> global final RMSNorm Phase neuron
-> lm_head
```

也就是说，即使：

```text
common_clip_enabled=false
```

training graph 与 Phase deployment graph 仍然不是完全相同的 neuron topology。

当前：

```python
controller.apply_final_norm_phase(...)
```

只在：

```text
mode == deploy_phase
```

时执行 global Phase neuron。

因此 regression 必须专门比较：

```text
final_norm_before_global_phase
final_norm_after_global_phase
lm_head
```

并额外运行一个 **仅用于诊断** 的：

```text
deploy_phase_no_final_norm_phase
```

ablation。

禁止在看到 regression 结果之前直接删除或加入该 neuron。

---

## 3. 新增文件

新增：

```text
snn2/phase_conversion_regression.py
scripts/regress_phase_conversion.py
```

不要把 regression 主逻辑塞进：

```text
evaluate_tldr.py
convert_snn.py
```

官方实验入口保持不变。

同时对以下文件只增加 **默认关闭、无正常运行开销** 的 trace hook：

```text
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
```

正常训练 / calibration / evaluation 不启用 recorder 时，数值行为必须完全不变。

---

## 4. Regression 前置条件

`scripts/regress_phase_conversion.py` 启动后立即验证：

```text
experiment.ann_mode == phase_aware
replacement.common_clip_enabled == false
ann_training.prefix_enabled == true
```

并验证以下 artifact 存在：

```text
layout.ann_checkpoint_dir
layout.ann_training_prefix_dir/prefix_state.json
layout.ann_training_prefix_dir/prefixed_key_values.pt   # Prefix 非空时
layout.ann_training_site_dir/calibration_state_manifest.json
layout.snn_conversion_dir("phase")/conversion_metadata.json
```

然后严格检查：

### final ANN

```text
ann/final/config.json:
snn2_ann_mode == phase_aware
snn2_ann_common_clip_enabled == false
```

### training_result

```text
ann_training_common_clip_enabled == false
ann_training_common_clip_applied == false
ann_training_common_clip_state_required == true
```

### provenance

训练记录中的：

```text
ann_training_prefix_state_sha256
ann_training_prefix_kv_sha256
ann_training_calibration_manifest_sha256
```

必须与当前磁盘完全一致。

### conversion metadata

必须：

```text
deployment_neuron == phase
reused_ann_training_artifacts == true
post_finetuning_recalibration == false
source_ann_common_clip_enabled == false
snn_clip_applied == false
```

并确认 conversion 读取的 Prefix、calibration manifest、final ANN checkpoint 与 training_result 完全一致。

任意 provenance 不一致：

```text
立即失败
不要继续做数值 regression
```

---

## 5. Regression 输入

默认只使用：

```text
1 个 TL;DR evaluation sample
```

并限制真实 token 长度：

```text
max_input_tokens = 64
```

目的不是测 ROUGE，而是使全部 attention 中间 tensor 可以安全保存并精确比较。

要求：

```text
batch_size = 1
model.eval()
dropout = 0
use_cache = false
torch.no_grad()
```

输入必须由项目现有 TL;DR prompt/tokenization 逻辑构造，不允许 regression 自己发明模板。

建议复用：

```text
load_selected_raw()
encode_tldr_generation_prompt()
```

选择：

```text
evaluation manifest 中第 0 个样本
```

并保存：

```text
dataset record id
input_ids
attention_mask
prefix_token_ids
```

到 regression metadata。

---

## 6. 三条计算图必须分别重新加载模型

不要在同一个 model instance 上来回切换：

```text
identity -> phase -> deploy_phase
```

因为当前项目会动态安装 attention backend、RMSNorm wrapper、forward hooks、MLP forward override、Prefix KV hook。

正确方式：

```text
Graph I:
    load final ANN
    run
    del model
    empty_cache

Graph P:
    重新 load 同一个 final ANN
    安装 static Phase integration
    run
    del model
    empty_cache

Graph SNN:
    再次 load 同一个 final ANN
    安装 deploy_phase integration
    run
```

三次 source 必须都是：

```text
layout.ann_checkpoint_dir
```

---

## 7. Graph I：Identity ANN

构造：

```python
controller = SiteController(
    mode="identity",
    site_root=layout.ann_training_site_dir,
)
```

安装 Rotation integration 与同一套 Pre-finetuning Prefix。

Graph I 只是 identity baseline。

输出：

```text
identity_logits
```

重点记录最后一个有效 token 的 logits。

---

## 8. Graph P：Static Phase ANN，作为真正 conversion reference

构造：

```python
controller = SiteController(
    mode="phase",
    site_root=layout.ann_training_site_dir,
    common_clip_enabled=False,
)
```

它必须加载：

```text
phase_state.pt
```

但不得加载：

```text
clip_state.pt
```

Graph P 是本次最重要的 reference，因为它对应：

```text
phase_aware ANN fine-tuning 的实际 replacement 前向行为
```

输出：

```text
phase_static_logits
```

---

## 9. Graph SNN：Temporal Phase deployment

构造：

```python
controller = SiteController(
    mode="identity",
    site_root=layout.ann_training_site_dir,
)

controller.set_deployment("phase")
```

随后安装同一 Rotation、同一 Pre-finetuning Prefix KV 与 temporal integration。

输出：

```text
phase_temporal_logits
```

---

## 10. Level 0：artifact / graph triage

第一份报告直接输出：

```text
Identity ANN      vs Static Phase ANN
Static Phase ANN  vs Temporal Phase SNN
Identity ANN      vs Temporal Phase SNN
```

指标：

```text
relative_l2_error
mean_abs_error
max_abs_error
cosine_similarity
last_token_top1_equal
last_token_top1_id_ref
last_token_top1_id_test
last_token_top5_overlap
```

定义：

\[
\mathrm{relative\_l2}
=
\frac{\|x-y\|_2}
{\max(\|x\|_2, 10^{-12})}
\]

所有 metric 计算前先 `.float()`。

---

## 11. Level 0 的判定分支

### 情况 A

```text
Identity ANN 好
Static Phase ANN 已经和 Identity ANN 差很多
Temporal Phase SNN 与 Static Phase ANN 很接近
```

结论：

> 主要问题不是 temporal conversion。

说明此前看到的：

```text
ANN 0.21 vs Phase SNN 0.01
```

主要是在比较：

```text
identity final ANN
vs
Phase graph
```

而不是：

```text
training-time static Phase
vs
temporal Phase
```

此时不要为了追求 0.21 而修改 temporal operator。

后续应分析：

```text
Phase state / τ / coding resolution
Phase replacement information loss
训练 LR 是否足以让权重适应 Phase replacement
```

---

### 情况 B

```text
Static Phase ANN 接近 Identity ANN
Temporal Phase SNN 明显偏离 Static Phase ANN
```

确认存在真实 Phase conversion implementation error，继续 Level 1–4。

---

### 情况 C

```text
Static Phase ANN 已经差
Temporal Phase SNN 又进一步明显变差
```

说明同时存在：

```text
Phase replacement approximation loss
+
temporal conversion loss
```

本次先定位：

```text
Static Phase ANN vs Temporal Phase SNN
```

的 conversion loss。

---

## 12. Level 1：Phase neuron 本体 unit regression

对 ANN-training calibration 中：

```text
每层 10 个 site 的 phase_state.pt
+ _global/final_rmsnorm/phase_state.pt
```

至少抽取：

```text
layer 0
middle layer
last layer
```

的全部 site。

对每个 Phase state 创建相同总输入 `x`，检查：

```python
static = phase(x)
```

与：

```python
temporal = phase.temporal(increments).sum(dim=0)
```

至少测试三种 temporal decomposition：

### A

```text
[x, 0, 0, ...]
```

### B

```text
[x/T, x/T, ..., x/T]
```

### C

随机：

```text
r_0, ..., r_{T-2}
r_{T-1} = x - sum(r_0...r_{T-2})
```

当前 `PhaseSurrogate.temporal()` 是：

```python
encode(incoming.sum(dim=0), return_temporal=True)
```

因此三种 decomposition 的 temporal sum 理论上都应与 `PhaseSurrogate.forward(x)` 相同。

要求：

```text
relative_l2 <= 1e-7
max_abs <= 1e-7
```

若可 bitwise identical，再记录：

```text
exact_equal
```

这里失败时优先修：

```text
snn2/neurons.py
```

不要继续查 Transformer。

---

## 13. Level 2：Temporal primitive unit regression

为以下函数加入独立 sum-preservation test：

```text
temporal_rmsnorm
temporal_silu
temporal_seq_matmul
temporal_softmax
temporal_symmetric_hadamard
temporal_bias_once
embedding x/T
Prefix KV /T
```

核心 invariant：

```text
sum_t(temporal_operator(increments))
≈
static_operator(sum_t(increments))
```

### RMSNorm

```python
temporal_rmsnorm(x_t).sum(0)
```

应等于：

```python
rmsnorm(x_t.sum(0))
```

### SiLU

```python
temporal_silu(x_t).sum(0)
```

应等于：

```python
F.silu(x_t.sum(0))
```

### QK/PV

```python
temporal_seq_matmul(a_t, b_t).sum(0)
```

应等于：

```python
torch.matmul(a_t.sum(0), b_t.sum(0))
```

### Softmax

固定同一 attention mask：

```python
temporal_softmax(score_increment, mask).sum(0)
```

应等于：

```python
softmax(score_increment.sum(0) + mask)
```

有 Prefix columns 时再单独测一次。

### MLP Hadamard

```python
temporal_symmetric_hadamard(gate_t, up_t).sum(0)
```

应等于：

```python
gate_t.sum(0) * up_t.sum(0)
```

### Linear bias

temporal frames 总和中 bias 必须只出现一次。

FP32 unit test 要求：

```text
relative_l2 <= 1e-6
```

再额外跑 BF16 并记录误差。

任何 FP32 invariant 失败都视为真实实现 bug。

---

## 14. Level 3：完整模型 paired checkpoint recorder

新增：

```python
class PhaseConversionRegressionRecorder:
    ...
```

放在：

```text
snn2/phase_conversion_regression.py
```

正常运行时 recorder 为 `None`。

---

## 15. Recorder 的统一比较语义

Static Phase Graph P 中 tensor：

```text
[B, ...]
```

直接保存 reference。

Temporal Graph SNN 中大多数 tensor：

```text
[T*B, ...]
```

record 时必须：

```python
to_temporal(x, T).sum(dim=0)
```

后再和 static reference 比较。

也就是说，比较的是：

> temporal signal 的总模拟值是否等于 static Phase graph 对应节点的值。

不要比较单独 timestep。

---

## 16. Recorder 默认不保存长序列大 tensor 到磁盘

默认只在内存中保存：

```text
CPU float32 tensor
```

并保持：

```text
input length <= 64
batch = 1
```

报告落盘只写 metrics。

仅对 first failing checkpoint 可选：

```text
--dump-first-failure-tensor
```

保存完整 `.pt`。

---

## 17. SiteController 自动记录 neuron pre / post

修改：

```text
snn2/controller.py
```

增加：

```python
self.regression_recorder = None
```

以及：

```python
def set_regression_recorder(self, recorder):
    self.regression_recorder = recorder
```

在：

```python
apply(...)
```

中仅 recorder 非空时记录：

```text
layer_xxx/site_yy/pre
layer_xxx/site_yy/post
```

Static：

```text
pre = x
post = Phase(x)
```

Temporal：

```text
pre = sum_t incoming
post = sum_t Phase temporal output
```

正常实验 recorder 为 `None`，不得产生 clone/cpu 开销。

---

## 18. 必须记录的 operator checkpoint

按每层执行顺序统一命名。

### Layer 入口

```text
layer_000/input
```

### Attention

```text
layer_000/site_01/pre
layer_000/site_01/post

layer_000/site_02/pre
layer_000/site_02/post

layer_000/site_03/pre
layer_000/site_03/post

layer_000/site_04/pre
layer_000/site_04/post

layer_000/attn/qk_scaled
layer_000/attn/softmax_before_site5

layer_000/site_05/pre
layer_000/site_05/post

layer_000/attn/pv_before_site6

layer_000/site_06/pre
layer_000/site_06/post

layer_000/attn/o_proj_output
layer_000/post_attention_residual
```

`qk_scaled` 两边都在加 mask/softcap 前记录。

`softmax_before_site5` 两边都在应用 mask/softcap/softmax 后、进入 Site 5 前记录。

### MLP

```text
layer_000/site_07/pre
layer_000/site_07/post

layer_000/mlp/gate_proj
layer_000/mlp/up_proj

layer_000/site_08/pre
layer_000/site_08/post

layer_000/site_09/pre
layer_000/site_09/post

layer_000/mlp/product_before_site10

layer_000/site_10/pre
layer_000/site_10/post

layer_000/mlp/down_proj_output
layer_000/output
```

---

## 19. Final head 必须单独记录

统一记录：

```text
final_norm/before_global_phase
final_norm/after_global_phase
lm_head/output
model/logits
```

Static Phase graph：

```text
before_global_phase == after_global_phase
```

因为目前 static training 不执行 global final Phase。

Temporal Phase graph：

```text
after_global_phase
=
global Phase(before_global_phase)
```

这一处必须在报告中标记：

```text
topology_mismatch_candidate = true
```

---

## 20. 增加 regression-only final norm bypass

只在 regression script 中支持：

```text
--bypass-final-norm-phase
```

不要暴露为正式实验 config。

可在 controller 中增加 regression-only flag：

```python
controller.regression_bypass_final_norm_phase = True
```

此时：

```python
apply_final_norm_phase(x)
```

只在 regression run 中返回 `x`。

正式 `convert_snn.py` / `evaluate_*` 行为不变。

---

## 21. 必须运行四个 fixed-forward graph

对同一输入：

```text
I  = Identity ANN
P  = Static Phase ANN
S  = Temporal Phase SNN（当前正式实现）
S0 = Temporal Phase SNN（regression-only，bypass global final norm Phase）
```

比较：

```text
I vs P
P vs S
P vs S0
S vs S0
```

---

## 22. final RMSNorm Phase 判定

如果：

```text
P vs S 很差
P vs S0 很接近
```

且 detailed trace 显示在：

```text
final_norm/before_global_phase
```

之前都很接近，则确认：

> 当前主要 conversion gap 来自 training graph 中没有、deployment graph 中额外存在的 global final RMSNorm Phase neuron。

此时不得把问题归结为 tolerance。

---

## 23. 若确认 global final Phase 是首要问题，如何修

先保存 regression 报告。

按当前目标：

> `phase_aware` ANN fine-tuning 与 Phase conversion 使用一致的 Phase neuron graph

修正。

由于项目此前加入 global final RMSNorm Phase 是为了对齐 SpikingLLM topology，优先方案：

```text
保留 SNN global final Phase
并让 phase_aware ANN training 也使用同一 global final Phase static replacement
```

不要定义成 Site 11：

```text
SITE_COUNT = 10
```

保持。

它继续使用：

```text
_global/final_rmsnorm/phase_state.pt
```

phase_aware training 中：

```text
final RMSNorm
-> PhaseSurrogate.forward(final_norm_output)
-> lm_head
```

且：

```text
common_clip_enabled=false
```

时不加 Clip。

本轮只处理：

```text
phase_aware -> Phase SNN
```

不要顺手修改 gif_aware final norm 行为。

如果采用此修复：

```text
旧 phase_aware final ANN checkpoint 必须重新训练
```

因为 training graph 已改变。

---

## 24. 如果 upstream 更早出现 divergence

如果第一个明显误差在 global final Phase 之前出现，不要先改 final norm。

按 first failure 处理。

---

## 25. 每个 checkpoint metrics

统一生成：

```json
{
  "name": "...",
  "shape": [...],
  "reference_l2": 0.0,
  "test_l2": 0.0,
  "diff_l2": 0.0,
  "relative_l2_error": 0.0,
  "mean_abs_error": 0.0,
  "max_abs_error": 0.0,
  "cosine_similarity": 0.0,
  "reference_zero_fraction": 0.0,
  "test_zero_fraction": 0.0
}
```

Site pre/post 再增加：

```text
local_error_amplification
```

定义：

```text
post_relative_l2 /
max(pre_relative_l2, 1e-12)
```

---

## 26. First-divergence 规则

按 forward 顺序找：

```text
第一个 relative_l2_error > 1e-2
```

同时输出：

```text
第一个 > 1e-3 的节点
最大 relative_l2 节点
最大 error amplification 节点
```

阈值只用于定位，不应掩盖完整 metrics。

---

## 27. Level 4：Locked-token decoding regression

fixed forward 可能还没有 argmax 分叉，但 autoregressive generation 会放大误差。

加入：

```text
--decode-steps 16
```

使用：

```text
Static Phase Graph P
```

作为 token oracle。

每个 step：

1. Graph P 对当前相同 token history 求 logits；
2. Graph SNN 对同一 history 求 logits；
3. 比较最后 token logits；
4. 取：
   ```python
   next_token = phase_static_logits.argmax()
   ```
5. 无论 SNN argmax 是什么，都把 Static Phase 的 token 追加给两边。

每步保存：

```text
step
context_length
logits_relative_l2
mean_abs
max_abs
top1_equal
static_top1_id
snn_top1_id
```

找：

```text
first_top1_disagreement_step
first_logits_relative_l2_gt_1e-2_step
```

然后只对第一失败 step 重新跑 detailed trace。

---

## 28. Identity ANN 也可做 locked-token 对照

额外记录：

```text
Identity ANN vs Static Phase ANN
```

用于量化 Phase replacement approximation。

但 conversion pass/fail 仍以：

```text
Static Phase ANN vs Temporal Phase SNN
```

为主。

---

## 29. 输出目录

使用当前 run root：

```text
<layout.root>/
analysis/
phase_conversion_regression/
```

例如：

```text
artifacts/snn2_main_v1/tldr/
Qwen_Qwen3-1.7B-Base/
phase_aware/
lr1e-07_train_samples_10000/
prefix_enabled_ture_common_clip_enabled_false/
seed42/
analysis/
phase_conversion_regression/
```

保存：

```text
regression_metadata.json
artifact_validation.json
micro_phase_neuron.json
temporal_primitives.json
fixed_forward_summary.json
checkpoint_metrics.jsonl
first_divergence.json
locked_decode.jsonl
final_norm_ablation.json
```

可选：

```text
first_failure_reference.pt
first_failure_temporal_sum.pt
```

---

## 30. 新脚本 CLI

建议：

```bash
CUDA_VISIBLE_DEVICES=6 \
python scripts/regress_phase_conversion.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml \
  --sample-index 0 \
  --max-input-tokens 64 \
  --decode-steps 16
```

支持：

```text
--sample-index
--max-input-tokens
--decode-steps
--dump-first-failure-tensor
--skip-locked-decode
```

不要求手工传 checkpoint / Prefix / calibration 路径，全部从 `ArtifactLayout(cfg)` 解析。

---

## 31. 建议先使用哪组实验

先用已有：

```text
common_clip_enabled = false
LR = 1e-7
```

或 `1e-8` 均可。

该 regression 是结构检查，理论上不依赖 LR。

建议先跑 `1e-7`，之后用 `1e-8, common_clip=false` 重复 1 个 sample，确认 first-divergence 位置一致。

---

## 32. 修复映射

| 第一失败点 | 优先检查 |
|---|---|
| Phase micro regression | `snn2/neurons.py::PhaseSurrogate` |
| embedding | `temporal_embedding_hook` |
| Prefix K/V sum | `prefix_cache.py` |
| RMSNorm | `temporal_rmsnorm()` |
| Site 1/7 pre | residual / RMSNorm |
| Site 2/3/4 pre | q/k/v、RoPE、R3、Prefix K/V |
| QK | `temporal_seq_matmul()` |
| Softmax | `temporal_softmax()`、mask、softcap |
| Site 5 | Phase Site 5 |
| PV | `temporal_seq_matmul()` |
| Site 6 后 | attention output / `o_proj` / bias |
| SiLU | `temporal_silu()` |
| Site 8/9 | gate/up projection / Phase |
| MLP product | `temporal_symmetric_hadamard()` |
| Site 10 后 | down_proj / bias |
| layer output | residual |
| final norm before global phase | temporal final RMSNorm |
| only after global final Phase | training/deployment topology mismatch |
| only lm_head | lm_head temporal bias / logits summation |

---

## 33. 修复原则

每发现一个真实 bug：

1. 保存失败 regression；
2. 写最小 unit test；
3. 只修改对应 operator；
4. 跑该 unit test；
5. 跑：
   ```bash
   pytest -q
   ```
6. 重新跑 regression；
7. 确认 first divergence 消失或后移；
8. 再处理下一个。

禁止一次同时重写多个 temporal operator。

---

## 34. Regression pass 标准

### Micro / primitive FP32

```text
relative_l2 <= 1e-6
```

Phase direct static/temporal：

```text
<= 1e-7
```

### Full model BF16

主要比较：

```text
Static Phase ANN vs Temporal Phase SNN
```

目标：

```text
final logits relative_l2 <= 1e-2
last-token top1 agreement = true
```

更重要的是不存在中间节点误差突然从约 `1e-4` 跳到 `1e-1 / 1e0`。

如果 16 个 locked decode steps top1 全一致，可以认为结构 regression 基本通过。

---

## 35. 如果 P 与 S 已经高度一致，不要“修” conversion

如果：

```text
Static Phase ANN vs Temporal Phase SNN:
    logits relative_l2 很小
    locked decode 基本一致
```

但：

```text
Identity ANN vs Static Phase ANN:
    差异极大
```

则报告必须明确：

> conversion 本身没有发现足以解释 ROUGE collapse 的实现错误；性能差主要发生在 identity ANN 与 Phase replacement graph 之间。

此时下一步应评估：

```text
Static Phase ANN 的实际 TL;DR ROUGE
```

并分析：

```text
Phase τ / coding resolution
Phase replacement information loss
学习率与 Phase-aware adaptation 是否足够
```

不要为了让 SNN 接近 identity ANN 而破坏 conversion 数学语义。

---

## 36. Diagnostic-only Static Phase TL;DR evaluation

`regress_phase_conversion.py` 可支持：

```text
--static-phase-generation-samples 8
```

在同一 final checkpoint：

```text
controller.mode = phase
common_clip_enabled = false
```

做少量 greedy TL;DR generation。

只输出到：

```text
analysis/phase_conversion_regression/
```

不要混入正式 `ann/evaluation/`。

如果：

```text
Static Phase ROUGE ≈ Temporal Phase SNN ROUGE
```

强烈说明 conversion 并非主要问题。

---

## 37. 更新 `实验执行总结.md`

增加一个 Debug / Regression 小节。

说明：

```text
regress_phase_conversion.py
```

只用于：

```text
phase_aware + common_clip_enabled=false
```

下检查 training-time static Phase graph 与 temporal Phase conversion 的数值一致性。

必须写明：

```text
官方 --neuron ann 是 identity final ANN evaluation，
不等价于 training-time static Phase graph。
```

---

## 38. 测试文件

至少新增：

```text
tests/test_phase_conversion_regression.py
```

并补充：

```text
tests/test_neurons.py
tests/test_temporal_ops.py
tests/test_temporal_model_integration.py
```

覆盖：

- Phase static vs temporal sum；
- RMSNorm invariant；
- SiLU invariant；
- QK/PV invariant；
- Softmax + causal mask invariant；
- MLP symmetric Hadamard invariant；
- linear bias once；
- final norm regression bypass 只在 regression 开启；
- recorder 关闭时不改变正常 forward；
- recorder temporal reduction 使用 `sum(dim=0)`；
- common Clip 必须 false；
- provenance mismatch 必须拒绝。

---

## 39. Codex 完成后的执行顺序

先：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```

确保目标 config：

```yaml
experiment:
  ann_mode: phase_aware

replacement:
  common_clip_enabled: false
```

然后：

```bash
CUDA_VISIBLE_DEVICES=6 \
python scripts/regress_phase_conversion.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml \
  --sample-index 0 \
  --max-input-tokens 64 \
  --decode-steps 16
```

重点读取：

```text
fixed_forward_summary.json
first_divergence.json
final_norm_ablation.json
locked_decode.jsonl
```

根据 first divergence 做最小修复。

修后再次：

```bash
pytest -q
```

和：

```bash
CUDA_VISIBLE_DEVICES=6 \
python scripts/regress_phase_conversion.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml \
  --sample-index 0 \
  --max-input-tokens 64 \
  --decode-steps 16
```

直到得到：

```text
conversion_passed = true
```

或明确：

```text
conversion_temporal_path_matches_static_phase = true
identity_to_phase_gap_is_dominant = true
```

---

## 40. 何时需要重新训练

Codex 加入 regression checker 后，先使用现有 `common_clip=false` checkpoint 诊断。

只有以下情况重新训练：

### 修复改变 training-time graph

例如确认 global final RMSNorm Phase 必须加入 phase_aware training，则重新训练。

### 只修 temporal conversion operator

如只修：

```text
temporal_softmax
temporal_seq_matmul
temporal bias
```

则 final ANN checkpoint 可以复用，只需重新 conversion / SNN evaluation。

---

## 41. 最终交付要求

Codex 完成后必须提供：

```text
1. regression checker 代码
2. 新增/修改的 unit tests
3. pytest 结果
4. regression 输出目录
5. first divergence 明确位置
6. 修复前 metrics
7. 修复后 metrics
8. 是否需要重新训练 ANN 的结论
```

如果没有发现 temporal conversion bug，也必须明确说明：

```text
没有为了改善结果而做未经 regression 证实的算法修改。
```

本次目标不是“强行让 ROUGE 变高”，而是先回答：

> **Phase-aware training-time static Phase graph 和 Phase temporal conversion 到底从哪一个算子开始不一致。**
