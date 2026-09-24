# SAT-LLM 独立 Energy Profiler 修改方案

> 目标：在 **不改变现有 ANN / SNN evaluation 数值路径** 的前提下，为当前 `wangwk699/SNN` 项目增加一个独立的理论能耗分析工具。  
> 本文档面向部署在服务器上的 Codex；即使没有本轮对话上下文，也应能仅凭本文档完成代码修改、测试与使用说明更新。

---

## 1. 项目背景与本轮目标

仓库：

```text
https://github.com/wangwk699/SNN
```

当前实验主流程由：

```bash
python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated
```

生成 12 个主配置。当前 `实验执行总结.md` 中：

- Step 6：逐个 `config` 训练 12 个 Final ANN checkpoint；
- Step 9：评估 Final ANN；
- Step 10：同一个 Final ANN checkpoint 分别转换/评估 `phase`、`gif`、`mtn` 三种 full-temporal SNN。

本轮新增一个**独立 Energy Profiler**，用于对某一个：

```text
--config <generated config>
--neuron ann|phase|gif|mtn
```

执行一次固定长度、固定 held-out 样本集的 inference profiling，并输出该模型/神经元组合的一行理论 MAC/AC/energy 结果。

### 1.1 本轮最重要的硬约束

**Energy Profiler 不得改变现有 evaluation 数值路径。**

尤其不要为了计数而改变下列已有数值实现：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
snn2/evaluation.py 中现有 evaluation / generation 数值逻辑
snn2/model_integration.py 中现有 ANN/SNN forward 数值逻辑
snn2/temporal_model.py 中现有 temporal attention 数值逻辑
snn2/temporal_ops.py 中现有 temporal operators 数值逻辑
snn2/neurons.py 中 Phase / GIF / MTN 的现有 forward/temporal 数值逻辑
snn2/controller.py 中现有 SiteController 数值逻辑
```

允许在这些文件之外新增 profiler 专用代码，也允许在 `snn2/artifacts.py` 中增加只负责路径解析的 property/helper。

若确有必要在公共模块中增加**纯辅助、默认不执行**的只读 helper，也必须确保：

1. 原有函数参数、默认行为和返回值不变；
2. `evaluate_tldr.py` / `evaluate_lm_harness.py` 不需要传任何新参数；
3. 未启用 profiler 时不存在任何额外 hook、额外 tensor 变换或额外 forward；
4. 现有全部测试必须继续通过。

优先方案仍然是：**Energy Profiler 自己注册临时 hook / wrapper，仅在 `scripts/profile_energy.py` 进程内生效。**

---

# 2. 已确认的实验协议

以下所有规则均已确认，不需要再次询问。

## 2.1 一次命令对应一行结果

CLI 采用：

```bash
python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron ann
```

或：

```bash
python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron phase

python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron gif

python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron mtn
```

即：

```text
一个 config + 一个 neuron = 一行 Energy Analysis 结果
```

需要继续支持当前已有 deployment override 语义：

```bash
--phase-T
--phase-base
--mtn-T
--mtn-K
```

要求完全复用 `scripts/_common.py` 当前 `apply_deployment_overrides()` 的约束：

- `ann` / `gif` 不接受这些 deployment overrides；
- `phase` 只接受 `--phase-T` / `--phase-base`；
- `mtn` 只接受 `--mtn-T` / `--mtn-K`；
- override 只能改变 deployment，不得改变源 ANN run identity。

---

## 2.2 Profiling 样本数

Energy profiling 样本数**不增加新的 YAML 配置项**，直接复用：

```yaml
calibration:
  num_samples: 128
```

定义：

\[
N_{\mathrm{profile}}=\texttt{calibration.num\_samples}.
\]

默认当前为 128，但用户后续可以通过修改 `calibration.num_samples` 并重新 materialize config 来改变 Energy Profiling 样本数。

不要硬编码 128。

---

## 2.3 Profiling 数据必须来自 held-out validation

不要复用 Stage-A calibration subset。

当前代码中的 Stage-A calibration subset 来自 ANN training subset，因此不是 held-out。

Energy Profiler 统一从现有：

```text
validation_manifest.json
```

对应的 validation 数据中抽取：

```text
N_profile = calibration.num_samples
```

条样本。

当前项目中：

- TL;DR：validation manifest 对应原始 `validation` split；
- Tulu-3：validation manifest 对应 Step 2 中由 `experiment.seed` 固定隔离的 1000 条 validation，这部分不进入 ANN training pool。

抽样规则：

```python
seed = int(cfg["calibration"]["seed"])
num_samples = int(cfg["calibration"]["num_samples"])
rng = random.Random(seed)
selected_positions = rng.sample(range(len(validation)), k=num_samples)
selected_positions.sort()
```

要求：

- 无放回；
- deterministic；
- 若 `num_samples > len(validation)`，直接报错；
- 相同 model-task、相同 calibration seed / num_samples 的所有 mode 和所有 neuron 必须得到完全相同的 held-out sample indices。

建议 profiler 在结果 metadata 中保存：

```text
selected_validation_positions
calibration.seed
calibration.num_samples
validation manifest path
validation manifest SHA256
```

以保证复现。

---

# 3. 固定 512-token 输入协议

本 Energy Analysis **不是 generation benchmark**，而是模型级固定长度 forward profiling。

每个选中的 held-out 样本都构造成：

```text
exactly 512 tokens
```

并且只做一次 512-token forward，不生成新 token。

## 3.1 TL;DR

使用**完整样本**：

```text
prompt + reference/completion
```

而不是只使用 generation prompt。

尽量复用当前 `snn2.data.tokenize_row()` 中 TL;DR 的完整样本编码语义，即 `_encode_tldr()`：

- prompt：`add_special_tokens=True`
- completion：`add_special_tokens=False`
- 若 tokenizer 有 EOS，则 completion 末尾加 EOS。

## 3.2 Tulu-3

使用**完整 conversation**，包括 assistant answer。

尽量复用当前 `snn2.data.tokenize_row()` / `_encode_messages()` 的完整 conversation chat-template 语义：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    ...
)
```

不要使用 `encode_generation_prompt()`，因为 Energy Profiling 不是 prompt-only generation。

## 3.3 512-token 截断与 padding

统一：

```text
sequence_length = 512
```

该值本轮固定，不增加 YAML 参数。

要求：

1. 先按任务的完整样本语义编码；
2. 超过 512：按照当前 config 的 `data.truncation_side` 截断；
3. 少于 512：right-pad 到 512；
4. attention mask：有效 token = 1，padding token = 0；
5. 每次实际模型输入 shape 固定为 `[B, 512]`。

为了统计最简单、最可验证，Energy Profiler 默认：

```text
logical profiling batch size = 1
```

不要复用 `evaluation.batch_size`。逐条执行 `N_profile` 个样本并取平均。

---

# 4. Prefix KV cache 协议

当前项目的 Prefix KV cache 已经提前计算一次，并在 inference 中重复使用。

## 4.1 不计 Prefix cache 的生成成本

不要把：

```text
discover_prefix.py
prefix token forward
prefixed_key_values.pt 的构造
```

计入单样本 energy。

Profiler 只能**加载已经存在的 fixed Prefix KV cache**。

## 4.2 必须计入 Prefix 对当前样本 attention 的影响

Prefix 虽然提前缓存，但它会增加当前样本 attention 的 K/V length。

因此：

```text
QK^T
P/V matmul
```

中由 prefix length 导致的额外运算必须自然进入 profiling 计数。

不要为了 Energy Profiling 手工把 prefix token 拼入 `input_ids`。

必须继续复用当前：

```python
install_prefix_kv_forward(...)
```

的语义。当前 Prefix temporal policy 已经是 `uniform_kv_divide_by_T`，SNN profiling 必须沿用，不得另写不同的 prefix temporal path。

## 4.3 Prefix 来源与现有 evaluation 保持一致

对于 `--neuron ann`，Prefix stage 采用：

```text
final_ann_evaluation
```

继续遵守现有 mode protocol：

- vanilla Final ANN：固定不使用 Prefix；
- unaware Final ANN：按 `evaluation.prefix_enabled`，使用 Post-finetuning Prefix；
- phase_aware / gif_aware Final ANN：按当前代码定义使用相应 Pre-finetuning Prefix。

对于 `--neuron phase|gif|mtn`，Prefix stage 采用：

```text
final_snn_evaluation
```

并继续由：

```yaml
conversion.use_post_finetuning_artifacts
evaluation.prefix_enabled
```

决定实际 Prefix artifact source。不要自己重新定义 Prefix selector。

---

# 5. Profiling 的模型与 controller 必须复用现有 evaluation 语义

Profiler 的数值 forward 必须与当前最终 evaluation 使用完全相同的 checkpoint / controller / integration。

推荐顺序如下。

## 5.1 读取 config 与固定源 run identity

优先复用：

```python
cfg, layout = setup(args.config)
```

`ArtifactLayout(cfg)` 必须在 deployment override 之前构造，这与现有 `convert_snn.py` / evaluation 的语义一致。

然后：

```python
apply_deployment_overrides(args, cfg)
```

## 5.2 模型来源

Energy Analysis 一律基于 Final ANN checkpoint：

```python
source = model_source_for_stage(
    cfg,
    layout,
    stage="post_finetuning",
)
```

即 `layout.ann_checkpoint_dir`。不要使用 Base model，也不要使用 rotated-pre-finetuning model。

## 5.3 SNN dependency 验证

对于 `phase / gif / mtn`，要和现有 evaluation 一样调用：

```python
validate_conversion_metadata(cfg, layout, args.neuron)
```

缺 conversion descriptor / metadata 时直接报错，并提示先执行：

```bash
python scripts/convert_snn.py \
  --config "$CFG" \
  --neuron "$NEURON"
```

## 5.4 Final ANN training provenance

对于 `phase_aware` / `gif_aware` 且 `--neuron ann`，继续调用：

```python
validate_recorded_training_artifact_provenance(cfg, layout)
```

不要绕过当前 provenance validation。

## 5.5 Controller

必须调用：

```python
controller, steps = build_evaluation_controller(
    cfg,
    layout,
    neuron=args.neuron,
)
```

从而保证：

- `ann`：final ANN mode-aware forward；
- `phase`：`deploy_phase` full-temporal；
- `gif`：`deploy_gif` full-temporal；
- `mtn`：`deploy_mtn` full-temporal；
- SNN 不使用 common Clip；
- aware ANN 使用现有 static surrogate + common Clip 规则。

## 5.6 Model integration

遵守当前 evaluation 条件：

```python
if args.neuron != "ann" or cfg["rotation"]["enabled"]:
    install_model_integration(
        model,
        controller,
        rotation_state(cfg, layout),
    )
```

不得另写一套 attention / MLP forward。

## 5.7 Prefix injection

使用现有：

```python
prefix_key_values_for_stage(...)
install_prefix_kv_forward(
    model,
    prefix_key_values,
    controller=controller,
)
```

Profiler 不生成 Prefix cache，只加载已有 cache。

---

# 6. Energy Analysis 的统一理论模型

Energy Profiler 不是 GPU wall-power profiler，也不是 CUDA FLOP profiler。

它统计：

```text
theoretical operation counts under a unified MAC/AC abstraction
```

核心输出：

```text
MACs (G)
Synaptic ACs (G)
Neuron ACs (G)
Total ACs (G)
Energy (J)
```

---

# 7. 内部计数单位与最终换算

内部不要直接累计 G，而使用 raw operation count，即 Python `int` 或不会溢出的整数累计。

最后：

\[
N_{\mathrm{MAC}}[\mathrm G]=\frac{N_{\mathrm{MAC,raw}}}{10^9}
\]

\[
N_{\mathrm{AC}}[\mathrm G]=\frac{N_{\mathrm{AC,raw}}}{10^9}.
\]

不要在每一层提前除以 `1e9`。

---

# 8. 能耗公式与单位

采用理论单次运算能耗：

\[
E_{\mathrm{MAC}}=4.6\ \mathrm{pJ/op},\qquad
E_{\mathrm{AC}}=0.9\ \mathrm{pJ/op}.
\]

单位关系：

\[
1\ \mathrm{pJ}=10^{-12}\ \mathrm J,\qquad 1\ \mathrm G=10^9.
\]

raw count 形式：

\[
\boxed{
E[\mathrm J]
=
4.6\times 10^{-12}N_{\mathrm{MAC,raw}}
+
0.9\times 10^{-12}N_{\mathrm{AC,raw}}
}
\]

若输入已经是 G：

\[
\boxed{
E[\mathrm J]
=
0.0046N_{\mathrm{MAC}}[\mathrm G]
+
0.0009N_{\mathrm{AC}}[\mathrm G]
}
\]

其中：

```text
0.0046 的单位 = J / G-MAC
0.0009 的单位 = J / G-AC
```

程序内部推荐用 raw count 版本计算 Energy。

---

# 9. 三种 neuron 的统一 event 定义

公平性的核心原则：

> 不因为模块名叫 Phase / GIF / MTN 就决定 cost，而根据实际 temporal representation 对应多少个 unit events 决定。

定义每个 temporal element 的：

\[
M^{(t)}=\text{unit-event multiplicity}.
\]

## 9.1 Phase

当前 `PhaseSurrogate.temporal()` 的每个 element、每个 timestep 最多产生一个 signed spike contribution。

因此：

\[
\boxed{M_{\mathrm P}^{(t)}=\mathbf 1[y_{\mathrm P}^{(t)}\neq0]}
\]

正负都算 1 个 event。

Phase timestep-dependent amplitude `tau * base ** (-(t + 1))` 视为 frozen neuron coefficient。

主分析假设：

```text
fixed neuron coefficients can be implemented by pre-folding /
event-type lookup / fixed routing
```

因此非零 signed Phase event 在 downstream event-driven synaptic accumulation 中按 1 个 unit event 统计。

## 9.2 MTN

当前 `MultiThresholdNeuron.temporal()` 每个 element / timestep 最终至多输出一个 signed threshold spike。

因此：

\[
\boxed{M_{\mathrm{MTN}}^{(t)}=\mathbf 1[y_{\mathrm{MTN}}^{(t)}\neq0]}
\]

正 spike 与负 spike 均算 1 个 unit event。

不要因为 `K=6` 就把一次实际 firing 算成 6 个 event。K 只影响 threshold search / selected level。

## 9.3 GIF

GIF 不能简单用 `(output != 0)` 统计，因为 current GIF 是 multi-level code。

当前 ordinary GIF：

```text
high_qmax = 30
temporal_steps = 2
per_step_qmax = 15
```

并通过两个 unsigned chunks：

```text
chunk0 in [0,15]
chunk1 in [0,15]
chunk0 + chunk1 = q_high
```

表示 high branch。

根据 neuromorphic 1-bit expansion 语义，multi-level code 应展开成 unit spikes：

\[
\boxed{M_{\mathrm{GIF}}^{(t)}=|q^{(t)}_{\mathrm{unit\ code}}|}
\]

对于当前 unsigned 实现即 code 本身。

例：

```text
chunk = 0  -> 0 events
chunk = 1  -> 1 event
chunk = 7  -> 7 events
chunk = 15 -> 15 events
```

若 `q_high = 30`，则 `chunk0=15, chunk1=15`，总共 30 unit events。

### 重要

不能从 `abs(dequantized_output)` 直接得到 GIF event multiplicity。

Profiler 必须根据 GIF state 中的 `scale / zero / mask / q / chunk decomposition` 恢复或同步计算 integer code。

建议在**新的 profiler-only helper** 中复现只读 integer-code 计算，不修改 `StaticGIF.temporal()` 数值返回值。

必须覆盖当前不同 GIF policy：

```text
StaticGIF
AllLowStaticGIF
IdentityGIF
SoftmaxIdentityGIF
```

其中：

- `StaticGIF`：按 low/high mask 与 integer chunks 统计 unit events；
- `AllLowStaticGIF`：low code 只在 t0，有效 code 个数按 unit events 计；t1 为 0；
- `IdentityGIF`：不是 spike quantization，不产生 GIF unit-event multiplicity；
- `SoftmaxIdentityGIF`：Site 5 identity，不应假装成 spike event。

---

# 10. Static identity 与 event tensor 的区分

Profiler 内部至少区分：

```text
dense_dynamic
event_phase
event_gif
event_mtn
fixed_weight
```

其中：

- `IdentityGIF` / `SoftmaxIdentityGIF` 输出仍视为 `dense_dynamic` temporal signal；
- 不能因为整个 model 是 `deploy_gif`，就把所有 temporal tensors 都当成 spike；
- Site 5 / 8 / 9 GIF identity 是本轮统计公平性的重要边界。

当前 topology：

```text
Site 1  post_input_rmsnorm
Site 2  q_post_rope_r3
Site 3  k_post_rope_r3
Site 4  v_projection_r2
Site 5  post_spiking_softmax
Site 6  post_attention_value_dot_r2
Site 7  post_mlp_rmsnorm
Site 8  post_spiking_silu
Site 9  post_mlp_up_proj
Site 10 post_mlp_product_r4
```

当前 GIF policy：

```text
GIF all-low: Site 2
GIF salient: Site 1,3,4,6,7,10
GIF identity: Site 5,8,9
```

以及：

```text
Site 1 roles: q,k,v
Site 7 roles: gate,up
```

Energy Profiler 必须尊重这些现有 policy。

---

# 11. Linear 的统一 MAC / AC 计数

考虑：

\[
Y=XW^\top
\]

其中 `X shape = [..., Din]`，`W shape = [Dout, Din]`。

## 11.1 Dense ANN / dense dynamic input

若 input 是 `dense_dynamic`：

\[
\boxed{N_{\mathrm{MAC}}=N_{\mathrm{rows}}D_{\mathrm{in}}D_{\mathrm{out}}}
\]

其中 `N_rows` 是除最后 feature dim 外所有逻辑位置的乘积。

对普通 `[B,L,Din]`：

\[
BLD_{\mathrm{in}}D_{\mathrm{out}}.
\]

## 11.2 Event input × fixed weight

若 input 是 Phase/GIF/MTN event stream，weight 是 frozen model weight：

\[
\boxed{N_{\mathrm{AC}}^{\mathrm{Linear}}=D_{\mathrm{out}}\sum M(X)}
\]

归入 `Synaptic ACs`，不要再额外计同一 event 的 MAC。

## 11.3 Bias / residual

本轮主表聚焦：

```text
matrix/event synaptic compute + neuron dynamics
```

Linear bias、residual add 等普通 dense elementwise addition不单独加入 `Synaptic ACs`。

`Neuron ACs` 只表示 spiking neuron dynamics，不能把 residual/bias 混进去。

metadata 写：

```text
dense_elementwise_additions_included = false
```

以后若扩展完整芯片 energy，可另增加 `Other ACs`，本轮不要改变表结构。

---

# 12. Attention QK 与 PV 的统一统计

当前 SNN temporal attention 使用：

```python
temporal_seq_matmul(a, b)
```

数值实现为：

\[
\mathrm{cumsum}(A)B + A\mathrm{cumsum}(B)-AB.
\]

Profiler 必须统计**实际 deployment algorithm 的三个 temporal matrix-product terms**，不能仅按一个最终等价 matmul 计数。

## 12.1 两侧均为 event representation

定义一般 pair count：

\[
\mathrm{Pair}(A,B)
=
\sum_{t,b,i,k,j}
M_A(t,b,i,k)M_B(t,b,k,j).
\]

定义 cumulative event multiplicity：

\[
\widehat M_A(t)=\sum_{\tau\le t}M_A(\tau),\qquad
\widehat M_B(t)=\sum_{\tau\le t}M_B(\tau).
\]

三个实际 temporal terms 分别按：

\[
\mathrm{Pair}(\widehat A,B),\qquad
\mathrm{Pair}(A,\widehat B),\qquad
\mathrm{Pair}(A,B)
\]

统计。

总 event-driven matmul AC：

\[
\boxed{
N_{\mathrm{AC}}^{\mathrm{temporal\_seq\_matmul}}
=
\mathrm{Pair}(\widehat A,B)
+
\mathrm{Pair}(A,\widehat B)
+
\mathrm{Pair}(A,B)
}
\]

负号只表示 subtraction，仍然是 accumulation 类 operation，不改变 operation count。

## 12.2 至少一侧是非-event dynamic tensor

采用保守规则：

```text
event × event        -> AC
event × fixed weight -> AC
dynamic × dynamic    -> MAC
dynamic × event      -> MAC for active event-expanded pairs
event × dynamic      -> MAC for active event-expanded pairs
```

对于 `dynamic × event`，利用 event multiplicity 跳过 0，但每一个有效 unit event 与动态值的 product 仍计一个 MAC。

这特别重要于 GIF：Site 5 是 `SoftmaxIdentityGIF`，因此 GIF attention weight 不是 spiked code，不能把整个 PV 全部按 event-event AC。

## 12.3 Prefix

Prefix KV cache 的 key/value 同样参与上述实际 attention shape。

由于 Prefix cache 已经被 `install_prefix_kv_forward()` 注入，实际 K length 为：

```text
prefix length + 512 current-token length
```

Profiler 从实际 operand shape / multiplicity 统计即可。

不要额外人为加一次 prefix preprocessing cost，也不要把 prefix K/V 排除掉。

---

# 13. MLP temporal Hadamard product

当前 SNN MLP 使用：

```python
temporal_symmetric_hadamard(a, b)
```

数值形式：

\[
0.5\left(A_t\sum_\tau B_\tau+\sum_\tau A_\tau B_t\right).
\]

Energy Profiler 按两个实际 product branches 分开计。

若两个 operand 都是 event representation，则按 unit-event pair operation 统计为 AC。

若任意一侧为 `dense_dynamic`，则相应 active pair product 计 MAC。

这会自然产生：

- Phase / MTN：Site 8 与 Site 9 都是真正 temporal neuron output，可进入 event-event accounting；
- GIF：Site 8、Site 9 是 identity，因此该 MLP product 不得错误当成 GIF spike-driven AC。

固定常数 `0.5` 视为固定 implementation coefficient，不单独增加 MAC。

---

# 14. Neuron AC 的定义

`Neuron ACs` 只统计 temporal spiking neuron 自身 membrane / reset 的 accumulation 类 operation。

不要把普通 Transformer residual / bias addition 塞入这里。

## 14.1 Phase neuron

当前逻辑：

```text
membrane = abs(x) + v0
for t:
    compare membrane vs amplitude
    spike
    membrane = membrane - amplitude * spike
```

主表 AC accounting：

1. initial membrane preload：每 element 1 AC；
2. 每个 timestep，只有实际 `spike != 0` 时 reset 记 1 AC。

因此：

\[
N_{\mathrm{AC,Phase\ neuron}}
=
N_{\mathrm{elements}}
+
\sum_t\mathrm{nnz}(spike_t).
\]

Comparison 不计 MAC/AC。固定 amplitude lookup 不额外计 MAC。

## 14.2 MTN neuron

当前：

```text
for t:
    membrane = membrane + incoming[t]
    threshold comparison
    selected level
    membrane = membrane - spike
```

主表：

1. 每个 timestep membrane integration：1 AC / element / timestep；
2. 若实际 `spike != 0`，reset：1 AC。

因此：

\[
\boxed{
N_{\mathrm{AC,MTN\ neuron}}
=
T\cdot N_{\mathrm{elements}}
+
\sum_t\mathrm{nnz}(spike_t)
}
\]

K 个 threshold comparison 不计 MAC/AC。

## 14.3 GIF neuron

GIF 当前 runtime 是 static quantization + two-step decomposition，不使用 Phase/MTN 那种 membrane accumulation。

本轮 `Neuron ACs` 对 GIF 只统计**显式存在的 temporal recursive/additive state update**。

当前 `StaticGIF.temporal()` 主要是：

```text
incoming.sum(dim=0)
quantize
integer chunk decomposition
emit two temporal chunks
```

为了不人为把 quantizer arithmetic 当作 neuron membrane AC：

```text
GIF quantization / round / clamp / scale arithmetic
不进入 Neuron ACs
```

因此当前 GIF `Neuron ACs` 可以为 0，除非 profiler 在当前实现中发现明确的 recurrent membrane/reset AC。

metadata 写：

```text
gif_neuron_ac_policy = "no_recurrent_membrane_ac_in_current_static_gif_temporal_impl"
```

GIF 的主要 event cost体现在 downstream `Synaptic ACs`。

---

# 15. Final RMSNorm global neuron

当前：

- Phase SNN：global final RMSNorm 后有 temporal Phase neuron；
- MTN SNN：global final RMSNorm 后有 temporal MTN neuron；
- GIF SNN：final norm neuron identity；
- ANN phase-aware：static PhaseSurrogate；
- ANN gif-aware：按当前 static rules。

Energy Profiler 必须覆盖 `_global/final_rmsnorm` 的 Phase / MTN neuron dynamics。

Phase/MTN SNN 时，其 neuron AC 加入 `Neuron ACs`；GIF 不增加 temporal final-norm neuron AC。

---

# 16. ANN 行的定义

对于 `--neuron ann`，必须 profile 当前 config 对应的 **Final ANN actual forward**，不是统一 identity ANN。

| ann_mode | `--neuron ann` forward |
|---|---|
| vanilla | identity ANN |
| unaware | identity ANN（保留相应 Rotation/Prefix protocol） |
| phase_aware | `PhaseSurrogate.forward()` + 当前 common Clip 语义 |
| gif_aware | 当前 static GIF forward + common Clip 语义 |

但 `Neuron ACs` 是 temporal spiking neuron dynamics 指标，因此：

```text
--neuron ann -> Neuron ACs = 0
```

static surrogate / quantizer 的 round/clamp/comparison 等不伪装成 temporal neuron AC。

ANN dense Linear / dense attention / dense MLP product按 MAC 统计。

---

# 17. 本轮主 Energy 范围不包含的项目

为了与当前表结构和 MAC/AC 理论模型保持一致，下列成本不进入主 Energy(J)：

```text
memory read/write
HBM/DRAM
cache movement
communication
kernel launch
comparison
round
clamp
exp
rsqrt
division
attention mask
indexing / reshape / transpose
wall-clock power
GPU idle/static power
```

普通 dense residual/bias additions 本轮也不纳入 `Synaptic ACs` / `Neuron ACs`。

Hadamard rotation 的实现成本本轮不额外塞入 neuron/synaptic AC，避免改变既定表结构。

必须在 `energy_metadata.json` 写清楚：

```json
{
  "energy_scope": "mac_synaptic_ac_neuron_ac_only",
  "memory_energy_included": false,
  "dense_residual_bias_ac_included": false,
  "special_function_energy_included": false,
  "hadamard_rotation_energy_included": false
}
```

论文中以后应描述为 `theoretical computational energy`，而不是完整芯片/整机能耗。

---

# 18. Fixed coefficient 统一假设

Phase/MTN/GIF 中的 `tau / thresholds / base-dependent amplitudes / GIF scale` 在 calibration/conversion 后是 frozen 的。

主 Energy Analysis 采用：

```text
fixed neuron coefficients are pre-folded, routed by event type,
or implemented by fixed lookup
```

因此 event-driven static-weight Linear 不再为这些 fixed amplitude 单独加 runtime MAC。

但是 `dynamic × dynamic` 不能用“pre-fold”免费掉。

将该 assumption 写入 metadata：

```text
fixed_neuron_coefficient_policy = "prefold_or_fixed_event_lookup"
```

---

# 19. 推荐的 profiler 实现架构

建议新增：

```text
scripts/profile_energy.py
snn2/energy_profiler.py
tests/test_energy_profiler.py
```

可按需要再拆：

```text
snn2/energy_accounting.py
snn2/energy_data.py
```

但不要过度拆分。

## 19.1 `scripts/profile_energy.py`

职责：

1. parse args；
2. setup config/layout；
3. apply deployment overrides；
4. validate artifacts；
5. 调用 `profile_energy(...)`；
6. 写结果/日志；
7. 打印最终单行摘要。

CLI：

```python
parser(
    "Profile theoretical MAC/AC energy on fixed held-out 512-token sequences",
    neuron=True,
    allow_ann=True,
)
```

必须支持：

```text
--config
--neuron ann|phase|gif|mtn
--phase-T
--phase-base
--mtn-T
--mtn-K
```

不要新增 `--num-samples` / `--sequence-length`。

## 19.2 `snn2/energy_profiler.py`

建议包含：

```python
@dataclass
class EnergyCounts:
    mac_raw: int = 0
    synaptic_ac_raw: int = 0
    neuron_ac_raw: int = 0

    @property
    def total_ac_raw(self):
        return self.synaptic_ac_raw + self.neuron_ac_raw
```

并提供：

```python
def profile_energy(cfg, layout, *, neuron: str) -> dict:
    ...

def energy_j_from_raw_counts(mac_raw: int, ac_raw: int) -> float:
    ...

def raw_to_g(value: int) -> float:
    ...
```

---

# 20. 不修改 evaluation 数值路径的 instrumentation 方式

优先采用 profiler process-local temporary hooks / wrappers，而不是修改 existing evaluation forward。

推荐：

```python
with EnergyInstrumentation(model, controller, cfg) as profiler:
    ...
```

退出后必须恢复所有临时 monkey patch / hook。即使 profiling 中途抛异常，也要通过 `try/finally` 恢复。

## 20.1 可临时包装 controller.apply

在 profiler 当前进程中保存：

```python
original_apply = controller.apply
```

包装函数：

1. 保留原始 input；
2. 调原始 `controller.apply()`；
3. 只读地计算 site / role / temporal representation / event multiplicity；
4. 缓存给后续 consumer accounting；
5. 原样返回 original output。

不要把 wrapper 写进 `snn2/controller.py` 的默认路径。

## 20.2 GIF integer multiplicity

对于 deploy GIF，包装 `controller.apply()` 时可通过：

```python
modules = controller._load(layer_index, site_index)
gif_module = modules["gif"]
```

只读访问当前实际 GIF module/state。

在 profiler module 中提供：

```python
gif_temporal_event_multiplicity(
    gif_module,
    incoming_temporal,
    role,
)
```

该函数必须严格镜像当前 GIF integer code/chunk policy，但不能改变 forward。

## 20.3 Linear

给所有 `torch.nn.Linear` 注册 profiler-only `forward_pre_hook`。

需要知道：

```text
输入 representation
weight 固定
in_features/out_features
实际 logical tensor shape
```

然后按第 11 节累计 MAC 或 Synaptic AC。

注意 temporal deployment 中 PyTorch batch 是 `[T*B, ...]`，但 event multiplicity 已包含 timestep；不要因为 shape 有 `T*B` 再把 event count额外乘一次 T。

## 20.4 Attention temporal_seq_matmul

不要修改 `snn2/temporal_ops.temporal_seq_matmul()`。

Profiler 进程内临时包装 `snn2.temporal_model` 当前引用的：

```python
temporal_seq_matmul
```

调用 original function 得到**完全相同的数值输出**，同时依据 operand representation/multiplicity 统计三个实际 term 的 MAC/AC。

必须 patch `snn2.temporal_model.temporal_seq_matmul`，因为该模块通过直接 import 持有引用。

退出 profiler 后恢复 original reference。

## 20.5 temporal_symmetric_hadamard

同理，不修改 `snn2/temporal_ops.py`。

Profiler 进程内临时包装 `snn2.model_integration` 当前引用的：

```text
temporal_symmetric_hadamard
```

调用 original 得到相同输出，同时按第 13 节统计。

---

# 21. Representations 的传播不要依赖脆弱的 `id(tensor)`

不要只用 `id(tensor)` 做 representation registry，因为 `reshape / transpose / contiguous / from_temporal / to_temporal` 可能创建新 object。

优先按**已知 topology / consumer hook**建立 counting：

```text
Site 1 -> q/k/v Linear
Site 2/3 -> QK
Site 4 -> PV value side
Site 5 -> PV attention side
Site 6 -> o_proj
Site 7 -> gate_proj/up_proj
Site 8/9 -> temporal MLP product
Site 10 -> down_proj
global final norm neuron -> lm_head input
```

对于 known consumer，把 event multiplicity直接缓存到：

```text
(layer, site, role, sample, forward)
```

而不是给 arbitrary tensor 全局打标签。

需要特别处理：

```text
Site 1 q/k/v role
Site 7 gate/up role
```

不要把 q branch 的 GIF mask/multiplicity误用于 k/v。

---

# 22. LM head

LM head 是 Linear consumer。

Final ANN：`dense hidden -> lm_head` 按 MAC。

SNN：

- Phase：global final RMSNorm neuron 后为 event representation，因此 lm_head 可按 event × fixed weight -> Synaptic AC；
- MTN：同理；
- GIF：global final norm GIF identity，因此 lm_head input 不是新增 GIF event，按 actual representation 规则处理；不能因为 `deploy_gif` 就自动算 AC。

必须覆盖 lm_head，否则大 vocabulary projection 会被漏掉。

---

# 23. 512-token padding 的 operation accounting

输入必须 pad 到 512，但 padding token 不应被 profiler 手工删除。

本实验定义：

```text
fixed tensor length = 512
```

因此 Transformer dense shape-based operations按 512 的实际 tensor length统计。

Attention mask只是限制语义注意力，不意味着 GEMM shape 自动缩短。

对于 event activity，padding positions经过模型后的实际 event activity以 actual forward 为准。

---

# 24. Per-sample mean

对 `N_profile` 条序列逐条得到 raw counts：

```text
counts_1
counts_2
...
counts_N
```

最终报告：

\[
\overline N_{\mathrm{MAC}}=\frac1N\sum_iN_{\mathrm{MAC},i}
\]

\[
\overline N_{\mathrm{synAC}}=\frac1N\sum_iN_{\mathrm{synAC},i}
\]

\[
\overline N_{\mathrm{neuronAC}}=\frac1N\sum_iN_{\mathrm{neuronAC},i}
\]

\[
\overline N_{\mathrm{AC}}=\overline N_{\mathrm{synAC}}+\overline N_{\mathrm{neuronAC}}.
\]

Energy：

\[
\boxed{
\overline E
=
4.6\times10^{-12}\overline N_{\mathrm{MAC}}
+
0.9\times10^{-12}\overline N_{\mathrm{AC}}
}
\]

主结果报告 per-sequence mean，不是 N 条总和。

metadata 同时保存 raw totals：

```text
total_mac_raw
total_synaptic_ac_raw
total_neuron_ac_raw
profile_num_samples
```

---

# 25. 输出路径

结果目录与 `./ann`、`./snn` 同级。

在 `ArtifactLayout` 增加：

```python
@property
def energy_root(self) -> Path:
    return self.root / "energy"
```

最终：

```text
<run_root>/
├── ann/
├── snn/
├── energy/
└── ...
```

## 25.1 ANN 路径

```text
<run_root>/energy/ann/
```

## 25.2 SNN 路径

为防止 selector / Phase override / MTN override / GIF variant 相互覆盖，SNN energy 路径必须镜像当前 `layout.snn_dir(neuron)` 的 deployment identity。

推荐：

```python
relative = layout.snn_dir(neuron).relative_to(layout.root / "snn")
energy_dir = layout.energy_root / "snn" / relative
```

即：

```text
<run_root>/energy/snn/<与现有 snn deployment identity 相同的相对路径>/
```

这样 `phase T=4/8`、`mtn T/K`、selector true/false 不会互相覆盖。

---

# 26. 每次调用写出的主结果文件

每个 `--config + --neuron` 目录至少写：

```text
energy_results.csv
energy_results.json
energy_metadata.json
energy_sample_manifest.json
```

## 26.1 `energy_results.csv`

必须只有一行结果，列严格为：

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

**不要包含 Reduction。**

示例：

```csv
Model,Neuron,T,MACs (G),Synaptic ACs (G),Neuron ACs (G),Total ACs (G),Energy (J)
Qwen/Qwen3-8B-Base,phase,4,....,....,....,....,....
```

## 26.2 `energy_results.json`

包含与 CSV 相同字段。

ANN：`T=null`。

GIF：`T=controller.temporal_steps`，当前应为 2，而不是读 `phase.T`。

Phase / MTN：写实际 deployment T。

## 26.3 `energy_metadata.json`

至少保存：

```text
experiment id
task
model_name
ann_mode
neuron
actual deployment T
phase base if applicable
mtn K if applicable
conversion.use_post_finetuning_artifacts
evaluation.prefix_enabled
actual prefix artifact stage
prefix length
profile sequence length = 512
profile num samples
profile seed = calibration.seed
selected validation positions
validation manifest path
validation manifest sha256
checkpoint source
controller mode
MAC energy = 4.6 pJ
AC energy = 0.9 pJ
all counting-scope flags
raw total counts
per-sample raw mean
fixed coefficient policy
GIF unit-event expansion policy
```

不要把 Reduction 写入主结果。

---

# 27. Model 字段

`Model` 使用：

```python
cfg["experiment"]["model_name"]
```

例如：

```text
Qwen/Qwen3-1.7B-Base
Qwen/Qwen3-8B-Base
meta-llama/Meta-Llama-3-8B
```

---

# 28. 配置文件不增加 Energy 专属参数

本轮不要向 `configs/experiment_matrix.yaml` 增加：

```yaml
energy:
  num_samples:
  sequence_length:
```

原因：

- 样本数复用 `calibration.num_samples`；
- sequence length 固定 512；
- 避免 materializer / config validation 扩大修改面；
- 不改变当前 12 个 generated config 的主 identity。

`materialize_configs.py` 不需要改变。

---

# 29. ArtifactLayout helper

增加：

```python
@property
def energy_root(self) -> Path:
    return self.root / "energy"

def energy_dir(self, neuron: str) -> Path:
    ...
```

规则：

```text
ann -> root/energy/ann
phase/gif/mtn -> root/energy/snn/<mirror current snn relative path>
```

`ensure()` 可以创建 `energy_root`，但不要枚举所有 neuron/T leaf 目录。

---

# 30. 日志

Profiler 可继续使用 `StageRun`，stage：

```text
profile_energy_ann
profile_energy_phase
profile_energy_gif
profile_energy_mtn
```

日志优先继续写 `layout.logs_dir`，或写 profiler leaf 下 `logs/`。无论采用哪种，不能覆盖其他 neuron 的日志。

---

# 31. 更新 `实验执行总结.md`

在 Step 10 后新增：

```markdown
## Step 11：Energy profiling
```

示例：

```bash
for CFG in "${ALL_CFGS[@]}"; do
  for NEURON in ann phase gif mtn; do
    python scripts/profile_energy.py \
      --config "$CFG" \
      --neuron "$NEURON"
  done
done
```

说明：

```text
ann 不依赖 SNN conversion descriptor
phase/gif/mtn 要求对应 conversion descriptor 已存在
```

并写清：

```text
calibration.num_samples 同时作为 Energy profiling sample-count hyperparameter；
Energy profiling 自身使用 held-out validation，并不复用 Stage-A calibration rows。
```

---

# 32. 单元测试要求

至少新增：

```text
tests/test_energy_profiler.py
```

覆盖以下内容。

## 32.1 Energy unit conversion

```python
energy_j_from_raw_counts(1_000_000_000, 0) == approx(0.0046)
energy_j_from_raw_counts(0, 1_000_000_000) == approx(0.0009)
```

以及 `100 G MAC + 500 G AC -> 0.91 J`。

## 32.2 Phase event multiplicity

nonzero positive / negative 都计 1，不按 amplitude 大小计数。

## 32.3 MTN event multiplicity

nonzero signed threshold output = 1 event，K 不直接乘 event count。

## 32.4 GIF unit-event expansion

测试：

```text
q_high=0 -> 0
q_high=1 -> 1
q_high=15 -> 15
q_high=16 -> 15+1 -> 16
q_high=30 -> 15+15 -> 30
```

并测试：

```text
AllLow: t1 multiplicity = 0
IdentityGIF: dense_dynamic，不伪造 event
SoftmaxIdentityGIF: dense_dynamic
```

## 32.5 Dense Linear MAC

toy：`B=1,L=2,Din=3,Dout=4`，断言 24 MAC。

## 32.6 Event Linear AC

若 unit-event total=5、Dout=4，断言 20 Synaptic AC、0 MAC。

## 32.7 temporal_seq_matmul

构造 tiny event multiplicity tensor，手工计算三个 term，断言 profiler count 一致。

另测：

```text
dynamic × dynamic -> MAC
event × event -> AC
```

## 32.8 held-out selection

相同 validation / seed / num_samples 重复调用必须一致；超出 validation length 必须报错。

## 32.9 exactly 512 tokens

测试：

- 短样本 pad 到 512；
- 长样本 truncate 到 512；
- input_ids / attention_mask shape `[1,512]`；
- prefix token 不手工 prepend。

## 32.10 Prefix generation 不得发生

Profiler 只允许：

```text
prefix_key_values_for_stage
install_prefix_kv_forward
```

不允许重新计算 `prefixed_key_values.pt`。

## 32.11 Output path isolation

断言：

```text
ann
phase T4
phase T8
mtn T4K6
mtn T8K12
selector true
selector false
```

结果路径互不覆盖。

---

# 33. 回归测试

完成后至少运行：

```bash
pytest -q
```

必须保持全部旧测试通过。

不要通过修改旧测试期望值来“适配”本轮修改，现有 evaluation numeric expectation 不得更改。

---

# 34. 数值路径回归保护

新增轻量 regression test：

1. 构造 toy model/controller；
2. profiler instrumentation 关闭时运行 existing forward；
3. 开启 profiler instrumentation 运行相同 forward；
4. 断言 logits/output 完全一致或在原 dtype 合理严格 tolerance 下 allclose；
5. profiler context 退出后再运行一次；
6. 再次与原始 output 一致。

重点：

```text
instrumentation 只观察/count，不修改 tensor 数值。
```

---

# 35. 运行时内存要求

Profiler 逐样本执行 512-token forward。

不要为了保存 N 个样本的 event tensors 而把所有中间 activation 留在 GPU。

每个 sample：

1. reset per-sample counters；
2. forward；
3. 将 raw integer counts 汇总到 CPU/Python；
4. 释放临时 profiler tensor/multiplicity；
5. 下一 sample。

不要保存完整 activation trace。

---

# 36. no_grad / eval

整个 profiler forward：

```python
model.eval()

with torch.no_grad():
    ...
```

不构建 backward graph，不启用 training checkpointing。

---

# 37. Position IDs

固定 512-token input：

```python
position_ids = position_ids_from_attention_mask(attention_mask)
```

复用现有 helper。

Prefix wrapper 会根据 cached prefix length 对 position ids 加 offset；不要自己重复 offset。

---

# 38. ANN forward 与 temporal SNN forward

ANN：

```python
logits = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=False,
).logits
```

SNN：

```python
logits = temporal_forward(
    model,
    controller,
    input_ids,
    attention_mask,
    position_ids=position_ids,
)
```

只做一次 forward。

不要调用：

```text
greedy_generate()
lm_eval
ROUGE
```

Energy profiler 不计算任务指标。

---

# 39. 不需要输出预测结果

本轮不保存：

```text
decoded text
predictions
references
accuracy
ROUGE
PPL
```

输入内容只用于产生 realistic event activity。

---

# 40. 一行结果的数值定义

| Column | 定义 |
|---|---|
| Model | `cfg["experiment"]["model_name"]` |
| Neuron | `ann / phase / gif / mtn` |
| T | ANN=`null`；SNN=`controller.temporal_steps` |
| MACs (G) | N 条 held-out 512-token sequences 的平均 MAC / 1e9 |
| Synaptic ACs (G) | 平均 event-driven synaptic/temporal product AC / 1e9 |
| Neuron ACs (G) | 平均 temporal neuron AC / 1e9 |
| Total ACs (G) | Synaptic + Neuron |
| Energy (J) | `4.6e-12*mean_MAC_raw + 0.9e-12*mean_TotalAC_raw` |

不要输出 `Reduction`。

---

# 41. Deterministic 要求

相同：

```text
config
neuron
deployment overrides
artifacts
validation manifest
calibration.seed
calibration.num_samples
```

重复运行，operation count 应完全一致。不同即视为 bug。

---

# 42. Counting policy version

定义：

```python
ENERGY_PROFILER_VERSION = 1
ENERGY_ACCOUNTING_POLICY = "sat_llm_mac_ac_v1"
```

写入 metadata。

以后若统计边界变化，必须升级 version，不得静默改变旧结果语义。

---

# 43. GIF 特别回归要求

由于 GIF 当前有：

```text
Site 5 identity
Site 8 identity
Site 9 identity
Site 2 all-low
Site 1/7 multi-role
ordinary high qmax=30
two chunks of max 15
```

必须至少对 synthetic layer 做 end-to-end profiler test，确认：

```text
identity sites不会被计为 GIF unit events
all-low site第二个 timestep不会产生低分支 events
role-aware mask按照 q/k/v 或 gate/up 分开读取
high code 16..30 的 unit events不会因为只有两个 temporal frames而被误计成 1 或 2
```

---

# 44. Phase 特别要求

Phase：

```text
T 可 deployment override
base 可 deployment override
```

Profiler：

- actual T 取 override 后 cfg；
- 结果路径必须包含当前 Phase deployment identity；
- event count来自 actual current Phase temporal forward；
- 不允许用 calibration 时旧 T 猜测。

---

# 45. MTN 特别要求

MTN：

```text
T
K
threshold_factor
```

Profiler：

- T/K 按实际 deployment；
- `K` 不作为 event 数乘数；
- threshold comparison 不计 MAC/AC；
- actual nonzero selected spike 才计 reset；
- result path 必须隔离 T/K override。

---

# 46. 不改变现有 `calibration.num_samples` 的含义

虽然 Energy Profiling 样本数复用 `calibration.num_samples`，但不要改变当前 calibration artifact identity / manifest 的任何逻辑。

Energy profiler 只是“读取这个数值作为自己的 sample-count hyperparameter”。

不要把 energy held-out indices 写进 `calibration_manifest.json`，也不要覆盖 Stage-A calibration manifest。

Energy held-out selection provenance 只能写进 energy 自己的 metadata / manifest。

---

# 47. Energy sample manifest

每个 energy leaf dir 写：

```text
energy_sample_manifest.json
```

建议内容：

```json
{
  "source": "validation_manifest",
  "sequence_length": 512,
  "num_samples": 128,
  "selection_seed_source": "calibration.seed",
  "selection_seed": 42,
  "sampling": "seeded_random_without_replacement",
  "positions_in_validation": [],
  "validation_manifest_sha256": "..."
}
```

不要硬编码示例中的 128/42，实际写 config 值。

---

# 48. 文件修改范围总结

## 必须新增

```text
scripts/profile_energy.py
snn2/energy_profiler.py
tests/test_energy_profiler.py
```

## 推荐修改

```text
snn2/artifacts.py
实验执行总结.md
```

## 如确有必要可新增

```text
snn2/energy_accounting.py
snn2/energy_data.py
```

## 原则上不要修改数值实现

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
snn2/evaluation.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
snn2/temporal_ops.py
snn2/neurons.py
```

若仅为导出无副作用 helper 必须动公共文件，需保证旧路径无行为变化，并增加 regression test。

---

# 49. 完成后的使用示例

先确保已有：

```text
Final ANN checkpoint
对应 Prefix artifacts
对应 Stage A
对应 SNN conversion descriptor（SNN only）
```

然后：

```bash
CFG=configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml

python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron ann

python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron phase

python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron gif

python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron mtn
```

Deployment override：

```bash
python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron phase \
  --phase-T 8

python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron mtn \
  --mtn-T 8 \
  --mtn-K 12
```

---

# 50. 验收标准

### A. 功能

- [ ] `profile_energy.py --config ... --neuron ann` 可运行；
- [ ] Phase/GIF/MTN 均可运行；
- [ ] 一次调用生成一行结果；
- [ ] 结果含 8 列，不含 Reduction；
- [ ] `calibration.num_samples` 控制 profiling sample count；
- [ ] 输入来自同一 held-out validation sampling protocol；
- [ ] 每条序列固定 512 token；
- [ ] 不进行 generation；
- [ ] Prefix KV cache 只加载、不生成；
- [ ] Prefix 对 K/V length 的 operation cost 被统计；
- [ ] 输出为 per-sequence mean。

### B. 公平统计

- [ ] Phase nonzero timestep output = 1 unit event；
- [ ] MTN nonzero timestep output = 1 unit event；
- [ ] GIF multi-level code 按 unit events 展开；
- [ ] GIF identity site 不伪造 event；
- [ ] dense Linear = MAC；
- [ ] event × fixed weight Linear = AC；
- [ ] temporal QK/PV 按实际 three-term temporal algorithm accounting；
- [ ] dynamic operand 不被误算成 pure event-event AC；
- [ ] Phase/MTN neuron membrane/reset 计入 Neuron AC；
- [ ] GIF 当前 static temporal quantizer 不伪造 membrane AC；
- [ ] final RMSNorm neuron 被统计；
- [ ] lm_head 被统计。

### C. 路径

- [ ] `<run_root>/energy/` 与 `ann/`、`snn/` 同级；
- [ ] ANN/SNN 结果不互相覆盖；
- [ ] Phase T/base override 不覆盖默认结果；
- [ ] MTN T/K override 不覆盖默认结果；
- [ ] selector true/false 不覆盖。

### D. 数值路径保护

- [ ] 不改变现有 evaluation CLI；
- [ ] 不改变 existing evaluation output；
- [ ] profiler instrumentation 退出后所有 hooks/wrappers 恢复；
- [ ] `pytest -q` 全部通过；
- [ ] 新增 regression test 证明 profiler on/off logits 一致。

---

# 51. 最终原则

本轮 Energy Analysis 的目的不是让 SNN 得到“更好看的”数字，而是建立一个对 ANN / Phase / GIF / MTN 均使用相同规则的可复现实验框架。

实现时始终遵守：

```text
1. 先保持 actual numerical forward 完全不变；
2. 再观察 actual runtime representation；
3. 最后根据统一 accounting policy 分类 MAC / AC；
4. 不因 neuron 名称而特殊优惠；
5. 不把 GIF multi-level code 当成单个 1-bit spike；
6. 不把 GIF identity site 当成 spike；
7. 不把 Prefix preprocessing cost重复摊到每个 sample；
8. 但 Prefix 对当前 sample attention 的计算必须计入；
9. 所有主结果都是 512-token held-out sequence 的 per-sample mean；
10. Reduction 本轮不计算。
```

完成代码修改后，最终报告必须列出：

```text
修改/新增的文件
核心 accounting 规则
新增 CLI
输出路径
测试命令
pytest -q 结果
一个 smoke-test 命令
```

不要在未实际运行 profiler 的情况下伪造任何 MAC/AC/Energy 数值。
