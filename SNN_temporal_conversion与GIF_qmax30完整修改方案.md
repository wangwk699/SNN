# SNN 项目：Full-Temporal Deployment 与 GIF qmax=30 完整修改方案

> **目标读者**：部署在服务器上的 Codex / 代码修改智能体  
> **目标仓库**：`wangwk699/SNN` 的 `main` 分支代码  
> **任务性质**：修正 Phase / GIF / MTN SNN conversion 的 full-temporal Transformer 算子、Prefix temporal 注入、GIF 两步整数分解、common Clip 部署语义，并补齐工件版本校验和 temporal correctness 测试。  
> **重要要求**：本文件是本次修改任务的完整上下文。服务器端执行者不应依赖本次对话、`SNN.pdf` 的 Eq. (15)–(17)，也不应依赖本地 `SNN3/build/` 下的参考仓库。本文已完整写出需要实现的算法语义、文件级改动、测试和重跑范围。

---

## 0. 执行前要求

开始修改前：

1. 进入服务器上的 `SNN` 仓库根目录。
2. 确认代码与 `https://github.com/wangwk699/SNN/tree/main` 一致，并记录当前 commit：

   ```bash
   git rev-parse HEAD
   git status --short
   ```

3. 保留用户已有改动，不得使用 `git reset --hard`、`git checkout -- .` 或其它会丢失工作区内容的命令。
4. 完整阅读以下文件后再修改：

   ```text
   snn2/model_integration.py
   snn2/controller.py
   snn2/neurons.py
   snn2/prefix_cache.py
   snn2/calibration.py
   snn2/conversion.py
   snn2/config.py
   snn2/evaluation.py
   snn2/sites.py
   snn2/rotation.py
   scripts/materialize_configs.py
   scripts/verify_artifacts.py
   configs/experiment_matrix.yaml
   tests/
   实验执行总结.md
   代码结构总结.md
   ```

5. 全仓库搜索当前 temporal、GIF 和 Prefix 实现，避免只修改一个调用点：

   ```bash
   rg -n \
     'temporal_forward|deploy_phase|deploy_gif|deploy_mtn|temporal_steps|_fresh_dynamic_cache|StaticGIF|_qparams|2\*\*bits|add_bits|format_version|full_temporal_steps' \
     snn2 scripts tests configs *.md
   ```

6. 本任务必须先修改代码并完成测试，再更新仓库说明文档。不要先运行完整 GPU 实验。
7. `evaluation.tldr_test_samples: 128` 是用户有意设置的快速实验配置，不得以此为问题，也不得擅自改回 `null`。
8. 当前结果均来自 `prefix_enabled=true`；本任务必须保护这一条件下的 temporal Prefix 正确性。

---

# 1. 本任务的事实来源与明确边界

## 1.1 算法事实来源

本次修改按以下优先级执行：

```text
1. 本文写出的、从 SparseLLM / SpikingLLM 基础实现抽取出的完整算法语义
2. 当前 SNN 仓库的训练、checkpoint、数据与工件协议
3. 仓库现有说明文档中与以上两项不冲突的内容
```

以下内容**不作为实现依据**：

```text
SNN.pdf Eq. (15)–(17) 中的 deployment Transformer 算子
SNN.pdf 中已经过时的训练协议
服务器上不存在的 SNN3/build/SpikingLLM
服务器上不存在的 SNN3/build/SparseLLM
服务器上不存在的 SNN3/build/SpikeLLM
```

不得要求服务器端用户补充上述 `build/` 目录。本文已经给出全部必要公式和伪代码。

## 1.2 本次必须修改

```text
1. temporal RMSNorm
2. temporal Softmax
3. temporal SiLU
4. Attention QK activation-activation product
5. Attention PV activation-activation product
6. MLP gate-up temporal Hadamard product
7. final RMSNorm
8. temporal linear bias 只累计一份
9. Prefix K/V 每步注入原 cache 的 1/T
10. GIF high integer qmax 从 31 改为 30
11. GIF 两步分解每步整数 code 严格限制在 0～15
12. deployment common Clip
13. temporal 实现与 GIF policy 的 artifact 版本和旧工件拒绝逻辑
14. 能保护真实 temporal 行为的单元、集成和回归测试
15. 与以上行为有关的配置、生成配置和 Markdown 文档
```

其中 `temporal RMSNorm` 不只指每层 residual branch 前的两个 norm，还包括：

```text
Qwen3 Attention 内部的 q_norm
Qwen3 Attention 内部的 k_norm
backbone final_norm
```

Llama 若没有 `q_norm/k_norm`，应按可选属性兼容，不能因此报错。

## 1.3 本次明确不修改

```text
Phase：T=4、base=2、当前 tau/v0 初始化、固定 scalar/group_size=-1
Phase：不训练 neuron parameter，不引入 multi-granularity
GIF：继续使用完整 calibration set 得到的 site-specific 固定 scalar scale/zero
GIF：不改为 per-token dynamic activation scale
GIF：low/high channel mask 与 operator-aware saliency 规则不变
MTN：T=4、K=6、threshold_factor=0.75，继续使用单一 scalar scale
每层 10 个 activation replacement site 及其 rotation 坐标
Rotation / Prefix discovery / ANN checkpoint / 数据选择的现行协议
训练 epoch、learning rate、batch、DeepSpeed 等现行训练参数
128 条 calibration samples
用户设置的 128 条 TL;DR test samples 快速评估配置
```

不得借本任务引入第五种 ANN mode、修改数据 split、调整 greedy decoding、改变 ROUGE 计算或重新设计神经元参数共享粒度。

---

# 2. 当前实现中的问题与修正目标

## 2.1 普通逐 timestep 算子不是 full-temporal 等价算子

当前 deployment 把 `[T,B,...]` 展平为 `[T*B,...]` 后直接调用普通 Transformer。这样会使以下非线性或双线性运算各时间步独立执行：

```text
RMSNorm(x_t)
Softmax(score_t)
SiLU(gate_t)
Q_t @ K_t^T
A_t @ V_t
gate_t * up_t
```

这会遗漏跨 timestep 项。例如：

```text
(Σ_t Q_t) @ (Σ_t K_t)^T
```

不仅包含 `Q_t @ K_t^T`，还包含所有 `Q_i @ K_j^T, i != j`。当前实现没有这些项。

本次修正后的基本不变量为：

```text
temporal 输出沿 T 求和
    =
对应累计 ANN 激活执行一次目标算子的输出
```

对于 differential 非线性，还必须满足每个累计前缀：

```text
Σ_{i=0..t} y_i = op(Σ_{i=0..t} x_i)
```

## 2.2 普通 forward hook 无法修复已错误计算的算子

当前 RMSNorm 等位置使用输出 hook 后再交给 controller。若普通 RMSNorm 已经分别计算 `RMSNorm(x_t)`，hook 只能看到错误结果，无法恢复 `RMSNorm(Σx_t)` 的时间差分。

因此：

> 必须替换 deployment 模式下算子本身的 forward 行为；不能在普通算子输出之后用 hook “补救”。

静态 ANN、collect、phase-aware 和 gif-aware training 路径仍使用当前普通算术，不进入新 temporal 分支。

Qwen3 还必须特别注意：HF 的 `q_norm/k_norm` 在调用注册的 AttentionInterface backend **之前**执行。只重写 `snn2_eager_attention_forward()` 无法修复已经逐 timestep 执行过的 Q/K RMSNorm；必须在 Attention 模块进入 backend 前替换这两个 norm 的 deployment forward。

## 2.3 Prefix 当前被每个 timestep 完整复制

当前 `_fresh_dynamic_cache(prefix_key_values, batch_size)` 会给 `[T*B,...]` 的每一行注入完整 Prefix K/V。总时间量因此相当于 `T` 份 Prefix。

正确目标为：

```text
每一步 Prefix K = original_prefix_K / T
每一步 Prefix V = original_prefix_V / T
沿时间求和后恰好等于一份原始 Prefix cache
```

这一规则同时适用于 Phase、GIF 和 MTN deployment。

## 2.4 GIF 5-bit high code 的 31 无法由两个 4-bit timestep 表示

当前：

```text
high qmax = 2^5 - 1 = 31
temporal steps = 2
每步最大 code = 2^4 - 1 = 15
两步最大总 code = 15 + 15 = 30
```

因此 code 31 会留下 residual 1。正确策略已经确定：

```text
low qmax  = 15
high qmax = 30
step qmax = 15
steps     = 2
```

不得通过第三个 timestep、允许某步 code=16、静默丢弃 residual 或 saturate residual 的方式处理。

## 2.5 deployment 漏掉 common Clip

仓库当前 ANN replacement 是：

```text
Phase output → common Clip
GIF output   → common Clip
```

项目说明也要求：

```text
MTN deployment output → common Clip
```

但当前 `SiteController.apply()` 的 `deploy_*` 分支只执行 neuron temporal，没有执行 `clip_state.pt`。

本次必须补齐，并以 temporal differential Clip 实现，保证：

```text
Σ temporal_clip(y)_t = clip(Σ y_t)
```

不得简单地对每个 `y_t` 独立 clamp，因为 `Σ clip(y_t)` 通常不等于 `clip(Σy_t)`。

---

# 3. 全项目统一 temporal 约定

## 3.1 唯一布局约定

模型边界和 Hugging Face 模块边界保持：

```text
[T * B, ...]
```

所有 temporal helper 内部统一转换为：

```text
[T, B, ...]
```

其中展平顺序必须为 time-major：

```text
t0: batch 0 ... B-1
t1: batch 0 ... B-1
...
t(T-1): batch 0 ... B-1
```

当前 `input_ids.repeat(T, 1)` 符合这一顺序。不得改为会产生 batch-major 交错顺序的 `repeat_interleave(T, dim=0)`。

新增统一 helper，并在所有 temporal 算子和 Prefix 注入中复用：

```python
def to_temporal(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0 or x.shape[0] % steps != 0:
        raise ValueError(...)
    batch = x.shape[0] // steps
    return x.reshape(steps, batch, *x.shape[1:])


def from_temporal(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
```

建议新建：

```text
snn2/temporal_ops.py
```

作为 temporal 算子、布局和版本常量的单一事实来源。

## 3.2 部署模式判定

所有新 temporal 行为只能在：

```python
controller.mode in {"deploy_phase", "deploy_gif", "deploy_mtn"}
```

或等价的 `controller.mode.startswith("deploy_")` 且 neuron 合法时生效。

以下路径不得改变数值行为：

```text
identity / none
collect
phase ANN replacement
gif ANN replacement
普通 ANN evaluation
Prefix discovery
calibration statistics collection
rotation regression
```

## 3.3 数值精度

以下累计和非线性计算使用 `float32`：

```text
RMSNorm 累计与均方
Softmax 累计 logits 与 softmax
SiLU 累计与激活
QK/PV temporal cross-term 的累计量
MLP temporal Hadamard product 的累计量
temporal Clip 的累计量
```

输出再转换回输入 dtype。测试容差应区分 float32 与 bf16，不得要求 bf16 bitwise equality。

## 3.4 线性层 bias 只允许累计一次

若普通 `nn.Linear` 在每个 timestep 都加入 bias，则最终时间和为 `T*bias`。本次统一采用：

```text
bias 只加在 timestep 0
timestep 1..T-1 不加 bias
```

即：

```python
y_t = W @ x_t
y_0 = y_0 + bias
```

这样严格满足：

```text
Σ y_t = W @ (Σ x_t) + bias
```

实现时可以给 deployment 中执行的 Linear 安装受 controller 控制的 bias-correction hook：普通 Linear 先对每步加 bias，再将 `t>0` 的 bias 减掉。应覆盖：

```text
q_proj / k_proj / v_proj / o_proj
gate_proj / up_proj / down_proj
lm_head
模型中其它确实参与 decoder deployment forward 且 bias 非空的 Linear
```

必须按模块对象去重，避免 tied/shared module 重复注册 hook。bias 为空时不得复制 tensor 或产生额外计算。

---

# 4. `snn2/temporal_ops.py` 的完整算法定义

## 4.1 版本与策略常量

至少定义：

```python
TEMPORAL_IMPLEMENTATION_VERSION = 2
TEMPORAL_IMPLEMENTATION = "sparse_llm_temporal_v2"
TEMPORAL_LAYOUT = "time_major_flattened_TB"
TEMPORAL_LINEAR_BIAS_POLICY = "first_timestep_once"
PREFIX_TEMPORAL_POLICY = "uniform_kv_divide_by_T"

GIF_BASE_BITS = 4
GIF_ADD_BITS = 1
GIF_LOCAL_STEPS = 2
GIF_LOW_QMAX = 15
GIF_HIGH_QMAX = 30
GIF_STEP_QMAX = 15
GIF_INTEGER_DECOMPOSITION = "two_unsigned_chunks_each_0_to_15_high_qmax_30"
```

常量名称可以根据仓库风格微调，但语义必须只有一个来源。`neurons.py`、`calibration.py`、`config.py`、`conversion.py` 和 verifier 不得各自重新硬编码另一套值。

## 4.2 Differential unary operator

RMSNorm、Softmax、SiLU 和 Clip 共用以下数学结构：

```python
def temporal_difference(x, op):
    # x: [T, B, ...]
    cumulative = x.float().cumsum(dim=0)
    current = op(cumulative)
    zero = torch.zeros_like(current[:1])
    previous = torch.cat((zero, current[:-1]), dim=0)
    return (current - previous).to(x.dtype)
```

实现可以使用循环以降低峰值显存，但必须满足：

```text
prefix_sum(output, t) = op(prefix_sum(input, t))
```

## 4.3 Temporal RMSNorm

对输入增量 `x[t]`：

```text
X_t = Σ_{i=0..t} x[i]
Y_t = RMSNorm(X_t)
y[0] = Y_0
y[t] = Y_t - Y_(t-1)
```

RMSNorm 必须调用原模块的实际 epsilon 和 weight，兼容 Qwen3 与 Llama：

```python
def rms_op(cumulative):
    variance = cumulative.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = cumulative.float() * torch.rsqrt(variance + eps)
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized
```

优先从模块读取：

```text
variance_epsilon
eps
```

若两者都不存在，应显式报错，而不是静默采用任意默认值。

必须覆盖：

```text
每层 input_layernorm
每层 post_attention_layernorm
Qwen3 每层 self_attn.q_norm（若存在）
Qwen3 每层 self_attn.k_norm（若存在）
backbone final_norm
```

其中：

```text
input_layernorm 输出继续进入 Site 1
post_attention_layernorm 输出继续进入 Site 7
q_norm/k_norm 不新增 activation replacement site
final_norm 不新增 activation replacement site
```

`q_norm/k_norm` 和 final norm 虽然没有 activation replacement site，仍必须 temporal 化。对没有 `q_norm/k_norm` 的 Llama 层直接跳过。

## 4.4 Temporal SiLU

对 `gate_proj` 的 temporal 增量使用：

```text
G_t = Σ gate[i]
S_t = SiLU(G_t)
s[0] = S_0
s[t] = S_t - S_(t-1)
```

不得继续使用逐步 `SiLU(gate_t)`。

## 4.5 Temporal QK / PV activation-activation product

定义 `A`、`B` 的形状为 `[T,B,...]`，SparseLLM 基础语义为：

```python
sum_a = A.float().cumsum(dim=0)
sum_b = B.float().cumsum(dim=0)

out = (
    torch.matmul(sum_a, B.float())
    + torch.matmul(A.float(), sum_b)
    - torch.matmul(A.float(), B.float())
)
```

其累计性质为：

```text
Σ_{i=0..t} out[i]
    =
(Σ_{i=0..t} A[i]) @ (Σ_{i=0..t} B[i])
```

Attention 中：

```text
QK: A = query
    B = key.transpose(-2, -1)

PV: A = temporal attention weight increments
    B = value
```

QK scaling 只应用一次。推荐先计算 temporal QK increment，再整体乘 `scaling`。

GQA/MQA 中必须先按每个 temporal frame 执行 `repeat_kv`，不得把 T 误当成 batch 或 KV group 维度。

## 4.6 Temporal Softmax

QK temporal product 返回的是 score increment。Softmax 必须在累计 score 上计算并输出时间差：

```text
S_t = Σ score_increment[i]
P_t = softmax(softcap(S_t * scale) + attention_mask)
p[0] = P_0
p[t] = P_t - P_(t-1)
```

Attention mask 是每个累计状态都要满足的**固定约束**，不是 temporal activation：

```text
正确：softmax(cumulative_score + mask)
错误：先把 mask 加到每个 score increment，再 cumsum
错误：把 mask 除以 T
错误：把 mask 交给 seq_matmul
```

要求：

1. `[T*B,...]` 的每个 temporal frame 必须具有相同逻辑 mask。
2. 实现可使用第 0 帧 mask，并在 debug/test 中检查其它帧一致。
3. causal mask、padding mask 和 Prefix 扩展 mask 均要覆盖。
4. `softcap` 若存在，应用在累计、已缩放 score 上，再做 Softmax。
5. deployment evaluation 必须处于 `model.eval()`；若 attention dropout 非零且模块处于 training，应立即报错，不得在 temporal Softmax 后随机 dropout。

Site 5 的现有拓扑语义保持：

```text
Prefix 对应的 Softmax weight 不进入 Site 5 neuron replacement
当前 token 对应的 Softmax weight 进入 Site 5
```

因此应先对完整 `Prefix + current` score 做 temporal Softmax，再切分 Prefix/current weights；只将 current 部分传给 `controller.apply(layer, 5, ...)`，然后拼回。

## 4.7 MLP temporal Hadamard product

以 SparseLLM 当前主路径的对称 cross-time 分配为准，不使用简单逐 timestep `gate_t * up_t`。

设 `A=temporal_silu(gate)`、`B=up`，定义：

```text
C_t = A_t * B_t
      + 1/2 Σ_{j != t} (A_t * B_j + A_j * B_t)
```

等价的低显存实现为：

```python
sum_a = A.float().sum(dim=0, keepdim=True)
sum_b = B.float().sum(dim=0, keepdim=True)
C = 0.5 * (A.float() * sum_b + sum_a * B.float())
```

必须满足：

```text
Σ_t C_t = (Σ_t A_t) * (Σ_t B_t)
```

这一分配是对称的，不要求每个中间 timestep 是 causal prefix reconstruction；测试必须验证公式本身与最终总和，不能错误地对它套用 unary differential 的 prefix-sum 判据。

R4 rotation 继续作用在 temporal product 的每个输出 timestep 上。之后进入 Site 10 neuron，再进入 `down_proj`。

## 4.8 Temporal Clip

神经元输出 `z[t]` 后执行：

```text
Z_t = Σ z[i]
C_t = clip(Z_t, lower, upper)
c[0] = C_0
c[t] = C_t - C_(t-1)
```

`lower/upper` 使用当前 site 的 `clip_state.pt`，广播规则继续沿用 `Clipper` 的 `group_size`。

对任意 Phase/GIF/MTN site，controller deployment 输出必须是 `c[t]`，而不是未经 clip 的 neuron output。

---

# 5. 重写 `snn2/model_integration.py`

## 5.1 保留静态路径，新增明确的 deployment 路径

`snn2_eager_attention_forward()` 必须按 controller mode 分支：

```text
非 deploy：保留当前 collect / ANN 逻辑与 saliency 统计
deploy：执行本节完整 temporal attention
```

不要把 collect saliency 逻辑硬塞进 temporal deployment 分支。deployment 下只需要正确 forward 和必要计数。

## 5.2 Deployment Attention 的固定顺序

每层 Attention 必须按以下顺序实现：

```text
1. q_proj/k_proj/v_proj 的 bias 在整个时间轴只保留一份
2. Qwen3 q_norm/k_norm（若存在）已经由 temporal RMSNorm wrapper 产生时序差分
3. RoPE 对每个 timestep 的增量使用相同逻辑 position；RoPE 是线性变换，不需要新增 cross-time 算子
4. 接收 HF 已完成 projection / QK norm / RoPE / cache 拼接后的 query, key, value
5. 对 Q/K 执行当前在线 R3（若启用）
6. Q 进入 Site 2
7. current K 进入 Site 3；Prefix K 不进入 Site 3
8. current V 进入 Site 4；Prefix V 不进入 Site 4
9. Prefix K/V 保持 temporal K/T、V/T
10. 按 temporal frame 执行 repeat_kv，兼容 GQA/MQA
11. 用 temporal seq_matmul 计算 QK increments
12. 乘一次 attention scaling
13. 用累计 score + 固定 mask 执行 temporal Softmax difference
14. 仅 current weight increments 进入 Site 5，Prefix weights 旁路
15. 拼回完整 Prefix/current temporal weights
16. 用 temporal seq_matmul 计算 PV increments
17. 输出进入 Site 6
18. 转回 HF 需要的 `[T*B, query_len, heads*dim]` 对应布局
```

不得对 prefix/current 分别做两个 Softmax。Softmax 的归一化域必须仍是完整 key length。

## 5.3 Deployment RMSNorm

不要再让 `norm1_hook` / `norm2_hook` 尝试修正普通 RMSNorm 输出。

推荐做法：安装受 controller mode 控制的 forward wrapper：

```text
deploy_* → temporal_rmsnorm(original_input)
其它模式 → original_forward(original_input)
```

随后原 site hook 仍可将 temporal RMSNorm 输出送入：

```text
Site 1
Site 7
```

每层还必须检测并包装：

```python
getattr(layer.self_attn, "q_norm", None)
getattr(layer.self_attn, "k_norm", None)
```

二者存在时执行 temporal RMSNorm，但不调用 controller site。这样保证 Qwen3 在进入 `snn2_eager_attention_forward()` 前得到的是正确 Q/K temporal increments。Llama 没有这两个属性时跳过。

必须防止重复安装 wrapper，并在模型释放/测试隔离时保存原 forward 或可靠移除。

## 5.4 Deployment MLP

重写 `_make_mlp_forward()`：

```text
非 deploy：完全保留当前 gate_proj → SiLU → Site 8、up_proj → Site 9、逐元素 product → R4 → Site 10 → down_proj

deploy：
  gate_increment = gate_proj(x)
  gate_increment = temporal_silu(gate_increment)
  gate_increment = Site 8 neuron + temporal Clip

  up_increment = up_proj(x)
  up_increment = Site 9 neuron + temporal Clip

  product_increment = sparse_llm_symmetric_temporal_hadamard(
      gate_increment,
      up_increment,
  )
  product_increment = R4(product_increment)
  product_increment = Site 10 neuron + temporal Clip
  output_increment = down_proj(product_increment)
```

Site 8 和 Site 9 之后的 temporal Hadamard 输入必须是 site replacement 后的时序输出，不能绕过 neuron。

## 5.5 Embedding、final norm 和 lm_head

保留当前 embedding 语义：

```text
timestep 0 = ordinary embedding
timestep 1..T-1 = 0
```

必须补充：

```text
backbone final_norm → temporal RMSNorm difference
lm_head bias        → 只在 timestep 0 保留
logits              → 最后沿 T 求和
```

`temporal_forward()` 最终输出仍是普通 `[B,seq,vocab]` logits，不得把 temporal 维泄漏到 evaluator。

## 5.6 模型集成幂等性

当前已有 `_snn2_handles` 防止重复安装。新增 wrapper/hook 后必须继续满足：

```text
同一 model 不允许重复 install_model_integration
所有新增 handles 或 original_forward 引用可追踪
测试创建多个 model 时互不污染
AttentionInterface 的全局注册保持幂等
```

---

# 6. 修正 `snn2/prefix_cache.py`

## 6.1 普通 ANN Prefix 行为不得改变

普通 ANN、训练、calibration、Prefix discovery 使用原始一份 fixed KV cache：

```text
batch B → 每个样本一份完整 Prefix K/V
```

只有 controller 处于 `deploy_*` 且模型输入 batch 是 `[T*B,...]` 时，才使用 temporal Prefix。

## 6.2 新的 cache 构造接口

不要继续只传无法区分 logical batch 和 temporal steps 的 `batch_size=T*B`。建议改为显式接口：

```python
def _fresh_dynamic_cache(
    prefix_key_values,
    *,
    logical_batch_size: int,
    temporal_steps: int | None = None,
):
    ...
```

普通 ANN：

```text
temporal_steps=None
输出 batch=B，每份是原 K/V
```

SNN deployment：

```text
temporal_steps=T
先构造 [T,B,H,P,D]
每个 frame = original K/V / T
再按 time-major reshape 为 [T*B,H,P,D]
```

参考伪代码：

```python
base_k = key.expand(B, *key.shape[1:])
base_v = value.expand(B, *value.shape[1:])

temporal_k = (
    base_k.unsqueeze(0)
    .expand(T, B, *base_k.shape[1:])
    .div(T)
    .reshape(T * B, *base_k.shape[1:])
)

temporal_v = (
    base_v.unsqueeze(0)
    .expand(T, B, *base_v.shape[1:])
    .div(T)
    .reshape(T * B, *base_v.shape[1:])
)
```

若保存的 Prefix cache batch 不是 1，应显式验证它等于 logical `B`；不要依靠不明确的 `repeat_interleave` 猜测。

## 6.3 mask 与 position

Prefix attention mask 继续对每个 temporal frame 添加一列逻辑 1：

```text
Prefix mask 是固定可见性约束，不除以 T
```

`position_ids` 和 `cache_position` 继续只按 Prefix 长度偏移一次：

```text
position += prefix_length
```

不得乘 T，也不得对每个 timestep 递增不同 offset。

## 6.4 wrapper 如何得到 controller/T

`install_prefix_kv_forward()` 当前不知道 controller。修改接口时应显式传入 controller，或从已安装到 model 的受控属性读取；不得通过全局变量猜测。

推荐：

```python
install_prefix_kv_forward(model, prefix_key_values, controller=controller)
```

并同步修改 training、calibration、evaluation 中调用方。controller 非 deployment 时仍走普通 ANN Prefix。

---

# 7. 修正 GIF calibration、训练和 deployment

## 7.1 `_qparams` 必须接收明确 qmax

将：

```python
def _qparams(minimum, maximum, bits):
    qmax = 2**bits - 1
```

改为类似：

```python
def _qparams(minimum, maximum, *, qmin: int, qmax: int):
    if qmin != 0 or qmax <= qmin:
        raise ValueError(...)
    scale = ((maximum - minimum) / (qmax - qmin)).clamp_min(1e-8)
    zero = torch.round(qmin - minimum / scale).clamp(qmin, qmax)
    ...
```

调用固定为：

```text
low  qmin=0, qmax=15
high qmin=0, qmax=30
```

因此 high scale 的分母是 30，不是 31。

## 7.2 `StaticGIF._quantize` 不得再从 bits 隐式得到 high qmax

修改接口使 low/high 分支显式传入合法范围：

```text
forward low  → clamp 0..15
forward high → clamp 0..30
temporal low → clamp 0..15
temporal high→ clamp 0..30
```

`bits=5` 可以保留为描述 high precision 的名义位宽，但不能再用 `2**bits-1` 决定 high 最大 code。

## 7.3 两步分解

对 high code `q_high ∈ [0,30]`：

```python
chunk0 = q_high.clamp(0, 15)
chunk1 = q_high - chunk0

assert 0 <= chunk0 <= 15
assert 0 <= chunk1 <= 15
assert chunk0 + chunk1 == q_high
```

zero-point correction 只减一次，保持当前方案：

```text
t=0: high_output = chunk0 * scale - high_zero * scale
t=1: high_output = chunk1 * scale
```

low branch 继续只在 `t=0` 输出：

```text
t=0: (low_q - low_zero) * low_scale
t=1: 0
```

最终：

```text
Σ high_output = (q_high - high_zero) * high_scale
Σ low_output  = (low_q  - low_zero)  * low_scale
```

## 7.4 GIF state metadata

将 `gif_state.pt` 的 `format_version` 提升，并保存至少：

```text
base_bits: 4
add_bits: 1
low_qmin: 0
low_qmax: 15
high_qmin: 0
high_qmax: 30
temporal_steps: 2
per_step_qmin: 0
per_step_qmax: 15
integer_decomposition: two_unsigned_chunks_each_0_to_15_high_qmax_30
```

`StaticGIF.__init__()` 必须验证这些字段，拒绝旧的 qmax=31 state。不得为旧 state 静默填默认值 30，否则旧工件会被误认为新工件。

## 7.5 common clipping interval

`gif_high_range` 必须由 qmax=30 的 high qparams 计算。随后仍按当前规则构造：

```text
intersection(
    phase range,
    mtn range,
    intersection(gif low range, gif high range)
)
```

因为 qmax 改变可能改变 high representable range 和最终 common Clip，所以共享 ANN-training calibration 必须重新生成，`phase_aware` 也必须重训，不能只重训 `gif_aware`。

## 7.6 四类 state 都必须严格校验

不要只在 `StaticGIF` 中校验版本。以下构造器都必须拒绝旧格式和错误 `state_kind`：

```text
PhaseSurrogate 只接受 state_kind=phase、format_version=2
StaticGIF 只接受 state_kind=gif、format_version=2
MultiThresholdNeuron 只接受 state_kind=mtn、format_version=2
Clipper 只接受 state_kind=clip、format_version=2
```

四者还必须核对 `temporal_implementation_version=2`。GIF 再额外核对 qmax/chunk policy；Clip 再核对合法区间和 qmax=30 生成的 GIF range metadata。不得在 constructor 中用新默认值补齐旧 state。

---

# 8. 修正 `snn2/controller.py` 与 state 一致性校验

## 8.1 deployment site 的执行顺序

`SiteController.apply()` 的 `deploy_*` 分支改为：

```text
1. `[T*B,...] → [T,B,...]`
2. 执行对应 neuron.temporal
3. 对 neuron temporal output 执行 differential common Clip
4. `[T,B,...] → [T*B,...]`
```

ANN replacement 分支保持：

```text
phase → clip(phase(x))
gif   → clip(gif(x))
```

## 8.2 不得只读取第一个 site 推断 T

当前 `set_deployment()` 只读取第一个 site state。改为验证全部 layer/site：

```text
每层恰好 10 sites
所有 Phase state 的 T 一致
所有 MTN state 的 T 一致
所有 GIF state 的 temporal_steps 一致且等于 2
所有 GIF state 的 high_qmax=30、per_step_qmax=15
所有 state format version 合法
calibration manifest 的 temporal/GIF policy 与 state 一致
```

任一不一致立即报错，并输出具体 site 路径。

建议建立一个共享的 `validate_site_state_bundle(directory, manifest)`，由以下入口复用：

```text
SiteController._load / set_deployment
conversion.validate_calibration
scripts/verify_artifacts.py
```

不得在三个入口各写一套逐渐漂移的版本判断。conversion descriptor 必须在完整验证全部 layer/site state 后才能创建，不能等到 SNN evaluation 首次 forward 才发现旧工件。

## 8.3 shape 防御

所有 deployment site 继续验证：

```text
batch dimension % T == 0
incoming temporal length == neuron T
输出 shape 与输入 shape 完全一致
输出 dtype/device 与输入兼容
```

---

# 9. 配置、工件版本与旧工件拒绝

## 9.1 `configs/experiment_matrix.yaml`

在 defaults 中增加明确策略，例如：

```yaml
deployment:
  temporal_implementation: sparse_llm_temporal_v2
  temporal_layout: time_major_flattened_TB
  linear_bias_policy: first_timestep_once
  prefix_temporal_policy: uniform_kv_divide_by_T
  common_clip_temporal_policy: cumulative_then_difference

gif:
  base_bits: 4
  add_bits: 1
  high_qmax: 30
  temporal_steps: 2
  per_step_qmax: 15
  low_ratio: 0.90
  ...
```

字段名可按仓库风格调整，但生成的 12 个 resolved config 必须都显式带有这些值。

## 9.2 `snn2/config.py`

新增严格校验：

```text
deployment.temporal_implementation == sparse_llm_temporal_v2
deployment.temporal_layout == time_major_flattened_TB
deployment.linear_bias_policy == first_timestep_once
deployment.prefix_temporal_policy == uniform_kv_divide_by_T
deployment.common_clip_temporal_policy == cumulative_then_difference
gif.base_bits == 4
gif.add_bits == 1
gif.high_qmax == 30
gif.temporal_steps == 2
gif.per_step_qmax == 15
2 * gif.per_step_qmax == gif.high_qmax
```

这不是把 GIF 泛化成任意 bit 配置的任务。当前实验方案只支持上述固定组合；不支持的配置应尽早报错。

## 9.3 工件格式版本

必须提升以下版本，不保留“按旧 state 字段猜测新语义”的兼容分支：

```text
phase_state.pt：format_version 1 → 2
gif_state.pt：format_version 1 → 2
mtn_state.pt：format_version 1 → 2
clip_state.pt：format_version 1 → 2
calibration_state_manifest.json：format_version 1 → 2
conversion_metadata.json：format_version 1 → 2
```

四类 site state 都至少记录 `state_kind`、`format_version=2` 和 `temporal_implementation_version=2`；GIF/Clip 再记录自己的 qmax/range policy。manifest 与 conversion descriptor 统一记录：

```text
temporal_implementation_version: 2
temporal_implementation: sparse_llm_temporal_v2
temporal_layout: time_major_flattened_TB
temporal_linear_bias_policy: first_timestep_once
prefix_temporal_policy: uniform_kv_divide_by_T
common_clip_temporal_policy: cumulative_then_difference
gif_high_qmax: 30
gif_local_decomposition_steps: 2
gif_per_step_qmax: 15
```

`conversion_metadata.json` 还必须继续保存现有 checkpoint、calibration、rotation、Prefix 哈希与 topology metadata。

## 9.4 旧工件必须 fail closed

以下情况必须拒绝 conversion/evaluation：

```text
缺少 temporal implementation version
temporal implementation version 不是 2
Prefix temporal policy 不是 uniform_kv_divide_by_T
GIF state 缺少 high_qmax
GIF high_qmax=31
GIF temporal_steps != 2
GIF per_step_qmax != 15
conversion metadata 与 calibration manifest policy 不一致
任一 site 的 state 与其它 site 不一致
任一 phase/gif/mtn/clip state 的 format_version != 2
任一 state 的 state_kind 与文件名不一致
```

错误信息必须明确提示：

```text
旧工件不兼容；需要重新 materialize calibration states / conversion descriptor，
不能继续使用旧 SNN evaluation 结果。
```

不得通过 `dict.get(..., new_default)` 让旧工件通过。

## 9.5 `scripts/verify_artifacts.py`

扩展最终验证，至少检查：

```text
12 个 resolved config 的 temporal/GIF policy
所有 post-finetuning calibration manifests 的新版本
全部 site 的 GIF qmax/chunk policy
36 个 conversion descriptors 的 temporal implementation version
conversion 与 calibration manifest 哈希和 policy 一致
SNN metrics 中 neuron/T 与 descriptor 一致
不存在被当前 run 引用的旧 qmax=31 state
```

如果 metrics 增加 temporal policy 字段，verifier 也应核对；推荐增加，便于结果审计。

---

# 10. 必须新增的 temporal correctness 测试

现有只检查 shape、调用次数或简单 neuron sum 的测试不足以保护真实 temporal 算术。本节测试是本任务的验收条件，不是可选项。

## 10.1 新建 `tests/test_temporal_ops.py`

### A. 布局 round trip

覆盖：

```text
T=2、T=4
B=1、B=3
普通 token tensor
attention head tensor
```

验证 `from_temporal(to_temporal(x,T)) == x`，并用带唯一编号的样本证明 time-major 顺序正确。

### B. QK/PV seq_matmul

随机小 tensor，验证每个 `t`：

```text
out[:t+1].sum(0)
≈
A[:t+1].sum(0) @ B[:t+1].sum(0)
```

覆盖：

```text
T=2/4
B=1/2
多 head
query_len != key_len
float32 与 bf16
```

### C. Temporal RMSNorm

对每个 `t` 验证：

```text
temporal_output[:t+1].sum(0)
≈
ordinary_rmsnorm(input[:t+1].sum(0))
```

覆盖非单位 weight 和不同 epsilon。

同一个 helper 必须分别用 residual hidden shape `[T,B,L,H]` 和 Q/K head shape `[T,B,num_heads,L,head_dim]` 测试，防止实现错误地假定只有三维 hidden tensor。

### D. Temporal Softmax

对每个 `t` 验证：

```text
temporal_probability[:t+1].sum(0)
≈
softmax(score_increment[:t+1].sum(0) + fixed_mask)
```

覆盖：

```text
causal mask
padding mask
Prefix key positions
完全屏蔽位置始终为 0
mask 不被累计 T 次
可选 softcap
```

### E. Temporal SiLU

对每个 `t` 验证累计输出等于 `SiLU(累计输入)`。

### F. MLP temporal Hadamard

验证：

```text
实现输出等于显式双循环的对称 cross-time 公式
Σ output == (Σ A) * (Σ B)
```

不要给该算子添加不适用的 causal prefix-sum 断言。

### G. Temporal Clip

构造跨 timestep 正负抵消、越界后回落等输入，验证每个累计前缀等于对累计输入做 clip。该测试必须能让“逐 timestep 独立 clip”的错误实现失败。

### H. Linear bias

使用非零 bias 验证：

```text
Σ temporal_linear(x) == ordinary_linear(Σx)
```

并检查只有 timestep 0 保留 bias。

## 10.2 扩展 `tests/test_neurons.py`

GIF 至少覆盖整数边界：

```text
q_high = 0
q_high = 15
q_high = 16
q_high = 29
q_high = 30
```

逐项验证：

```text
每步 raw chunk ∈ [0,15]
两个 chunk 之和等于 q_high
q=30 → [15,15]
不存在 q=31
temporal dequant sum 等于 static high dequant
zero-point correction 只减一次
low channel 第二步为 0
```

还要验证旧 `high_qmax=31` state 在加载时被拒绝。

Phase/GIF/MTN 都新增 `neuron temporal → differential Clip` 测试，证明最终时间和符合约定。

## 10.3 新建 `tests/test_temporal_prefix.py`

使用人工 KV cache，不依赖下载模型，覆盖：

```text
T=2 与 T=4
B=1 与 B=3
多层、多 KV head、prefix_len>1
```

验证：

```text
cache shape = [T*B,H,P,D]
布局为 time-major
每个 frame == original/T
reshape 为 [T,B,...] 后沿 T 求和 == 每个 batch 的 original
mask 每帧都添加完整 Prefix 可见位
position/cache_position 只偏移 prefix_len 一次
ANN 非 deploy cache 仍为完整 original，不除以 T
```

必须包含一个测试，能让当前“每步完整 Prefix”的实现失败。

## 10.4 新建或扩展 `tests/test_temporal_model_integration.py`

使用 tiny fake decoder 或本地构造的小模块，不下载 Qwen/Llama 权重。

至少覆盖：

```text
1. bypass neuron（时间序列原样返回）时：
   temporal decoder block logits sum
   ≈
   ordinary decoder block 对 summed input 的 logits

2. B>1，防止 T/B 布局混淆
3. GQA：num_heads > num_key_value_heads
4. 有 Prefix 与无 Prefix
5. causal/padding mask
6. final RMSNorm 确实走 temporal wrapper
7. lm_head 非零 bias 只累计一次
8. embedding 只在 t=0 注入
9. deploy 之外的普通 ANN forward 数值不变
10. 带 q_norm/k_norm 的 Qwen3-like attention 在进入 backend 前已 temporal 化
11. 不带 q_norm/k_norm 的 Llama-like attention 正常跳过且数值正确
```

建议对一个 tiny attention + MLP block 做逐模块 reference，不只检查最终 shape。

## 10.5 配置、工件与 verifier 测试

扩展：

```text
tests/test_generated_configs.py
tests/test_calibration_topology.py
tests/test_post_finetuning_protocol.py
tests/test_evaluation_paths.py
```

或新增专门 artifact tests，验证：

```text
12 个 generated config 都带新 policy
旧 calibration manifest 被拒绝
任一 phase/gif/mtn/clip state format_version=1 均被拒绝
旧 GIF state 被拒绝
qmax=31 conversion metadata 被拒绝
policy mismatch 被拒绝
完整新工件通过 verify
```

## 10.6 真实模型 smoke test

在单元测试通过后，对服务器已有的一个最小本地模型 checkpoint 做 smoke test；若没有本地权重，可跳过下载，但必须在交付说明中标记未执行原因。

推荐对 Qwen3-1.7B 的一个短序列执行：

```text
B=2
short sequence
Prefix enabled
Phase T=4 / MTN T=4 / GIF T=2
max_new_tokens=1
```

检查：

```text
无 shape/device/dtype 错误
无 GIF residual
输出 logits finite
temporal counters 与 T 一致
```

---

# 11. 文件级修改清单

## 11.1 必须修改或新增

| 文件 | 修改内容 |
|---|---|
| `snn2/temporal_ops.py` | 新增布局、版本常量、differential unary、RMSNorm、Softmax、SiLU、seq_matmul、对称 Hadamard、Clip helper |
| `snn2/model_integration.py` | 重写 deploy attention/MLP/norm/final norm；加入 bias-once；保留静态路径 |
| `snn2/prefix_cache.py` | 按 logical B/T 构造 Prefix/T cache；ANN 路径不变 |
| `snn2/controller.py` | 全 site state 一致性校验；neuron 后 temporal Clip |
| `snn2/neurons.py` | GIF qmax=30、两步 0..15 分解；Phase/GIF/MTN/Clip 四类 state 严格校验 |
| `snn2/calibration.py` | 显式 qmax；high scale 分母 30；manifest/state metadata |
| `snn2/conversion.py` | 新 conversion format 与 temporal/GIF policy；拒绝旧 calibration |
| `snn2/config.py` | 新 deployment/GIF 配置严格校验 |
| `snn2/evaluation.py` | 将实现版本/policy 写入 SNN metrics；保持 counters 正确 |
| `scripts/verify_artifacts.py` | 校验新版本、全部 site 和 conversion/evaluation policy |
| `configs/experiment_matrix.yaml` | 增加明确 deployment 与 GIF qmax/chunk policy |
| `configs/generated/*.yaml` | 通过 materialize 脚本重新生成 12 个配置，不手工逐份漂移 |
| `tests/test_temporal_ops.py` | 新增真实 temporal 算术测试 |
| `tests/test_temporal_prefix.py` | 新增 Prefix/T 与布局测试 |
| `tests/test_temporal_model_integration.py` | 新增 tiny block 集成测试 |
| `tests/test_neurons.py` | GIF 边界和 temporal Clip 测试 |
| 相关 artifact/config tests | 旧工件拒绝与 policy 一致性测试 |
| `实验执行总结.md` | 按现有风格更新 temporal 算子、GIF qmax、工件版本和重跑说明 |
| `代码结构总结.md` | 更新 deployment 数据流和 temporal operator 说明 |

## 11.2 可能需要同步的调用方

如果 `install_prefix_kv_forward()` 增加 controller 参数，全仓库用 `rg` 查找并逐一修改：

```text
snn2/training.py
snn2/calibration.py
snn2/evaluation.py
其它测试或脚本调用点
```

普通 ANN 路径可传 controller，也可显式传 `None`；行为必须保持原 cache。

## 11.3 不应修改

除非测试证明是本任务的直接依赖，不要修改：

```text
snn2/sites.py 中 10-site 拓扑
snn2/data.py 的数据选择
snn2/rotation.py 的 R1/R2/R3/R4 定义和 fusion
Prefix token discovery 算法
训练 optimizer/scheduler/checkpoint 逻辑
ROUGE/lm-eval 指标实现
```

---

# 12. 推荐实施顺序

必须按以下顺序推进，每一步通过对应测试后再进入下一步：

```text
1. 新增 temporal_ops.py 与纯函数单元测试
        ↓
2. 修正 GIF qmax=30、两步分解与 calibration qparams
        ↓
3. 修正 controller deployment common Clip 和全 site state validation
        ↓
4. 修正 Prefix/T cache 与 Prefix 单元测试
        ↓
5. 重写 deployment RMSNorm / Attention / Softmax / MLP / final norm / bias
        ↓
6. 增加 tiny decoder integration tests
        ↓
7. 增加 config、manifest、conversion、verifier 的版本保护
        ↓
8. 重新生成 12 个 resolved configs
        ↓
9. 运行完整 CPU 单元测试和静态检查
        ↓
10. 可用时运行单模型 GPU smoke test
        ↓
11. 更新 实验执行总结.md 与 代码结构总结.md
        ↓
12. 最终 rg 审计 qmax=31、旧版本与逐 timestep 错误算子
```

---

# 13. 测试与验收命令

先重新生成配置：

```bash
python scripts/materialize_configs.py
```

先运行定向测试：

```bash
pytest -q \
  tests/test_temporal_ops.py \
  tests/test_temporal_prefix.py \
  tests/test_temporal_model_integration.py \
  tests/test_neurons.py \
  tests/test_generated_configs.py \
  tests/test_calibration_topology.py \
  tests/test_post_finetuning_protocol.py \
  tests/test_evaluation_paths.py
```

再运行完整测试：

```bash
pytest -q
```

根据仓库当前工具配置运行已有 lint/type check；若仓库没有配置，不得擅自引入大型新工具链。

最终搜索：

```bash
rg -n '2\s*\*\*\s*bits\s*-\s*1|qmax.?31|high_qmax|per_step_qmax|TEMPORAL_IMPLEMENTATION_VERSION|uniform_kv_divide_by_T' \
  snn2 scripts tests configs *.md
```

允许出现 `31` 的唯一情况是：

```text
测试旧工件必须被拒绝
迁移说明中解释旧行为
```

生产代码、当前配置和新工件不得把 high qmax 设为 31。

---

# 14. 修改后 `实验执行总结.md` 必须写清的内容

保持当前文档的编号、解释风格、命令块和 artifact tree 风格，不要把它改成简短 changelog。至少更新以下内容：

## 14.1 Full-temporal deployment 定义

明确写出：

```text
embedding：只在 t=0 注入
linear bias：只在 t=0 注入
RMSNorm / Softmax / SiLU / Clip：累计输入执行算子，再输出 temporal difference
Qwen3 q_norm/k_norm：进入 attention backend 前执行 temporal RMSNorm；Llama 无此模块时跳过
QK / PV：SparseLLM seq_matmul cross-time 算法
MLP gate-up：SparseLLM 对称 cross-time Hadamard
final norm：temporal RMSNorm
lm_head 后：沿 T 求和
```

## 14.2 Prefix

明确区分：

```text
ANN：每个样本注入一份完整 Prefix KV
SNN：每步注入 Prefix KV/T，时间和为一份完整 Prefix KV
mask 与 position offset 仍是固定约束，不除以 T
```

## 14.3 GIF

写明：

```text
high precision 名义 5-bit
有效整数范围 0..30
两个 timestep
每步 0..15
30 分解为 15+15
zero point 只减一次
```

## 14.4 Artifact compatibility

写明旧 temporal implementation / qmax=31 calibration 和 conversion 工件不兼容，运行 conversion 或 verifier 会直接拒绝。

## 14.5 重跑范围

按下一节写入，不得只写“重新运行 SNN evaluation”。

---

# 15. 修改后的实验重跑范围

## 15.1 为什么 Phase-aware 也需要重训

`phase_aware` 与 `gif_aware` ANN training 共用 ANN-training calibration 中的 common Clip。GIF high qmax 从 31 改为 30 后：

```text
GIF high scale / representable range 可能改变
        ↓
common clipping interval 可能改变
        ↓
Phase + Clip 的训练 forward 也可能改变
```

因此必须从共享 ANN-training calibration 起重跑两种 aware training。

## 15.2 可以复用与必须重跑

| 工件 / 实验 | 是否必须重跑 | 原因 |
|---|---:|---|
| 固定数据 manifest | 否 | 数据协议未变 |
| Rotation / fused Base / rotation regression | 否 | rotation 算法未变 |
| ANN-training Prefix discovery/cache | 否 | analog Prefix token/cache 本身未变；Prefix/T 只影响 SNN deployment |
| Vanilla analysis statistics | 否 | 仅分析且不 materialize conversion states |
| 共享 ANN-training calibration states | 是 | GIF qmax 与 common Clip 改变 |
| Vanilla ANN training | 否 | 不使用 replacement |
| Unaware ANN training | 否 | 不使用 replacement calibration |
| Phase-aware ANN training | 是 | common Clip 可能改变 |
| GIF-aware ANN training | 是 | GIF quantizer 与 common Clip 改变 |
| Vanilla/Unaware final ANN checkpoint | 可复用 | 训练路径未变 |
| Phase-aware/GIF-aware final ANN checkpoint | 不可复用 | 必须重训 |
| Phase/GIF-aware post-FT Prefix | 是 | final checkpoint 已改变 |
| Vanilla/Unaware post-FT Prefix | 可复用 | final checkpoint 与 analog Prefix discovery 未变 |
| 四种 mode 的 post-FT conversion calibration states | 是 | GIF qmax、common Clip、manifest version 改变 |
| Vanilla/Unaware ANN metrics | 数值上可复用 | ANN forward 未变；完整新表建议重跑 |
| Phase/GIF-aware ANN metrics | 是 | checkpoint 改变 |
| 全部 36 个 conversion descriptor | 是 | temporal implementation/version 改变 |
| 全部 36 个 SNN evaluation | 是 | deployment 算子与 Prefix temporal 注入改变 |

## 15.3 推荐的安全重跑顺序

沿用 `实验执行总结.md` 中的既有命令和 stage，不创建新实验协议：

```text
1. 备份旧 aware checkpoint、旧 calibration、旧 conversion、旧 SNN metrics
2. 重新 materialize 12 个 resolved config
3. 重新生成每个 model-task pair 的 shared ANN-training calibration
4. 重训 phase_aware 与 gif_aware
5. 对新的 phase_aware/gif_aware checkpoint 重新 discover post-finetuning Prefix
6. 对四个 mode 的每个 final checkpoint 重新做 post-finetuning conversion calibration
7. 重跑 phase_aware/gif_aware ANN evaluation
8. 为四个 mode × 三种 neuron 重新创建 conversion descriptor
9. 重跑全部 Phase/GIF/MTN SNN evaluation
10. 运行 verify_artifacts
```

为了得到来源一致的新 Table 2，推荐最终把四种 ANN mode 的 ANN evaluation 也全部重跑一次，但这不是 vanilla/unaware checkpoint 数值有效性的硬要求。

## 15.4 旧目录处理

不要让脚本混用旧 state 和新 manifest。推荐把旧工件移动到带日期/commit 的备份目录，而不是直接删除。移动前应确认目标路径位于当前实验 artifact 根目录内。

例如概念上保存：

```text
artifacts_backup_before_temporal_v2_<date>/
```

具体路径必须由服务器端执行者根据当前 `ArtifactLayout` 解析后决定，不得对模糊变量或仓库根执行递归删除。

---

# 16. 最终验收标准

只有同时满足以下条件，任务才算完成：

## 16.1 代码行为

```text
deploy_* 使用 temporal RMSNorm/Softmax/SiLU/QK/PV/Hadamard/final norm
Qwen3 q_norm/k_norm 已 temporal 化，Llama optional-absence 路径正常
Prefix 在每步为 K/T、V/T
Attention mask 未除 T、未累计 T 次
linear bias 最终只累计一份
GIF high q 永远不超过 30
GIF 两步 chunk 各自永远在 0..15
Phase/GIF/MTN deployment 均应用 differential common Clip
ANN/collect/training 非 deployment 路径保持原行为
```

## 16.2 测试

```text
所有新增 temporal correctness 测试通过
完整 pytest 通过
B>1、GQA、Prefix、mask、GIF boundary、final norm、bias 均有覆盖
测试能明确杀死当前旧实现，而不只是检查 shape
```

## 16.3 工件

```text
新 calibration state 和 conversion metadata 带 temporal v2 / qmax30 policy
旧 qmax31 或无版本工件 fail closed
12 个 generated config 一致
verify_artifacts 能发现跨 site 或跨阶段不一致
```

## 16.4 文档

```text
实验执行总结.md 的完整流程、命令、工件说明仍然保留
新增 temporal 算法、Prefix/T、GIF qmax30、重跑范围和旧工件失效说明
代码结构总结.md 与实际代码一致
文档不再把 SNN.pdf Eq. (15)–(17) 当作 deployment 算子依据
```

---

# 17. 禁止的错误实现

以下实现即使能运行，也视为任务失败：

```text
1. 继续逐 timestep 普通 RMSNorm/Softmax/SiLU
2. QK/PV 只计算同 timestep 对角项
3. MLP 只计算 gate_t * up_t
4. 用普通 forward hook 修补已经错误计算的非线性
5. Prefix 每步注入完整 K/V
6. Prefix mask 除以 T 或累计 T 次
7. high q 仍允许 31
8. 用第三步、code=16 或静默 residual 处理 q=31
9. 每步独立 clamp neuron output
10. deployment 完全遗漏 common Clip
11. linear bias 每步重复
12. 只修 decoder residual layer norm，遗漏 Qwen3 q_norm/k_norm 或 final norm
13. 只用 B=1 测试，掩盖 T/B reshape 错误
14. 只检查输出 shape 和调用次数，不检查时间累计数学性质
15. 静默接受旧 state / manifest / conversion metadata
16. 因本任务擅自改动训练协议、采样数量或 neuron 参数共享粒度
17. 要求服务器提供 build/ 参考代码后才实施
```

---

# 18. 服务器端 Codex 的最终交付格式

完成代码修改后，向用户报告：

```text
1. 修改结果概述
2. 关键文件与关键函数
3. temporal 数学不变量如何被实现
4. GIF qmax=30 与两步边界如何被保证
5. Prefix/T 与 mask/position 如何处理
6. common Clip 如何 temporal 化
7. 新旧 artifact 兼容策略
8. 实际运行的测试命令与结果
9. 未运行的 GPU smoke/full experiment 及原因
10. 用户接下来按 实验执行总结.md 应执行的第一条命令
```

不要声称 Table 2 指标已经改善，除非确实完成对应实验并有新 metrics。代码和单元测试完成后，应把“实现正确性已验证”与“任务指标已经恢复”明确区分。
