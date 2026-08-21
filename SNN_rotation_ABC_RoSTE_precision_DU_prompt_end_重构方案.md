# SNN Rotation 三路回归、RoSTE 数值精度对齐、Prompt-End 诊断与 Hadamard 符号顺序重构方案

> **用途**：本文件用于指导部署在服务器上的 Codex 在**没有任何先验上下文**的情况下，一次性完成本轮 SNN 项目 Rotation 相关重构。  
> **目标仓库**：`https://github.com/wangwk699/SNN`  
> **参考仓库**：`https://github.com/OptimAI-Lab/RoSTE`  
> **默认项目根目录**：`~/SNN`  
> **目标分支**：以执行时 `main` 最新代码为基线。  
> **原则**：本轮允许对 Rotation/Hadamard/regression 代码进行较大重构，但不要顺带改变 Prefix、Calibration、10-site、ANN training、SNN conversion、TL;DR ROUGE 评估协议，也不要修改用户明确要求暂时保留的 regression hard-gate 阈值。

---

# 0. 本轮必须一次性完成的四个任务

本轮必须同时完成以下四项，不允许只完成其中一部分。

## 任务 1：把 Rotation regression 改成 A / B / C 三路对照

定义：

```text
A = Original HF Base
    原始 Hugging Face Base model
    不安装 SNN2 model integration
    不做 Rotation
    不做 Prefix
    不做 activation replacement

B = Base + SNN2 identity integration, no rotation
    与 A 使用完全相同的 Base checkpoint
    安装 SNN2 model integration
    SiteController(mode="identity")
    rotation_state=None
    不做 Rotation
    不做 Prefix
    不做 activation replacement

C = Rotated + SNN2 identity integration
    Base checkpoint 经完整 fused/offline Rotation
    + online R3
    + online R4
    + SNN2 model integration
    + SiteController(mode="identity")
    不做 Prefix
    不做 activation replacement
```

必须分别计算：

```text
A ↔ B
B ↔ C
A ↔ C
```

并在同一个 `rotation_regression.json` 中保存三组结果和自动诊断结论。

---

## 任务 2：本项目 Rotation 数值精度向 RoSTE 对齐

当前 SNN 的 offline Hadamard 大量通过 FP32 FHT 完成，而 RoSTE 的关键 Rotation 路径不是统一 FP32。

必须按 RoSTE **实际 active implementation** 对齐各阶段的 compute dtype，而不是简单地把所有东西全部改成 FP64。

本轮目标精度策略：

```text
RMSNorm fusion:
    FP64 compute
    → cast 回原权重 dtype

R1 offline:
    FP64 explicit matrix multiplication
    → cast 回原权重 dtype

R2 value/output-side block transform:
    FP64 matrix multiplication
    → cast 回原权重 dtype

R2 o_proj/input-side FHT transform:
    FP32 FHT
    → cast 回原权重 dtype

R3 online:
    保持输入 activation dtype
    不再无条件 `.float()`
    Hadamard 后保持原 dtype

R4 offline:
    FP32 FHT
    → cast 回原权重 dtype

R4 online:
    FP32 FHT
    → cast 回原 activation dtype
```

这与 RoSTE 中以下 active 路径一致：

```text
rotation_utils/fuse_norm.py
rotation_utils/rotation.py
rotation_utils/hadamard.py
```

不要把“向 RoSTE 对齐”误实现成“所有 Rotation 全部 FP64”。

---

## 任务 3：在 all-token regression 之外增加真正的 prompt-end decision 检查

当前 regression 使用训练式 calibration tokenization：

```text
prompt + completion
```

并在所有有效 token 上统计 logits regression。

这不能准确回答：

> “真正开始 greedy generation 时，第一个 next-token decision 是否一致？”

因此必须额外使用同一套 128 calibration raw rows，只取 **TL;DR prompt 本身**，并严格按 TL;DR evaluation 的输入构造方式：

```text
tokenizer.encode(
    prompt,
    add_special_tokens=True,
    truncation=True,
    max_length=cfg["evaluation"]["tldr_input_length"],
)
```

比较 prompt 最后一个有效 token 位置对应的 next-token logits。

该 prompt-end diagnostic 必须对：

```text
A ↔ B
B ↔ C
A ↔ C
```

全部计算。

### 本轮非常重要的 hard-gate 规则

用户已经明确：

> `0.05 L2 + 95% Top-1 对 greedy generation 过宽` 这一点暂时不改。

因此：

```text
原有 all-token hard gate：
relative_l2_error <= 0.05
top1_agreement > 0.95
```

继续保留，不改阈值、不改比较符号。

**新增 prompt-end metrics 本轮只作为 mandatory diagnostic，不加入 hard gate。**

也就是说：

```text
pair.passed
```

仍然只由原 all-token 两个 hard gate 决定。

不要擅自新增：

```text
prompt_end_top1 == 1.0 才 pass
```

或其它 prompt-end hard gate。

---

## 任务 4：随机符号顺序从 `Q = U D` 改成 RoSTE 的 `Q = D U`

定义：

```text
D = diag(random_signs)
U = normalized structured Hadamard transform
```

RoSTE：

\[
Q_{\mathrm{RoSTE}} = D U
\]

当前 SNN：

\[
Q_{\mathrm{SNN,old}} = U D
\]

本轮必须改成：

\[
Q_{\mathrm{SNN,new}} = D U
\]

对于 row-vector convention：

```text
x Q
= x D U
```

因此 forward transform 必须是：

```text
1. 先逐元素乘 random signs
2. 再执行 Hadamard
```

而 transpose/inverse：

\[
Q^T = U^T D
\]

因此：

```text
x Q^T
```

必须是：

```text
1. 先执行 transpose Hadamard
2. 再逐元素乘 random signs
```

本轮只改变 **D 与 U 的乘法顺序**。

不要额外改变：

```text
R1/R2/R3/R4 的 seed 方案
每个 spec 的 signs 样本生成方式
R1/R2/R3/R4 的共享策略
Rotation 是否启用
```

除非为实现 fail-fast artifact versioning 所必需。

---

# 1. 修改前必须检查当前仓库状态

Codex 开始工作前执行：

```bash
cd ~/SNN

git status
git rev-parse HEAD
git branch --show-current

git submodule status
git -C fast-hadamard-transform rev-parse HEAD

python - <<'PY'
import fast_hadamard_transform
print(fast_hadamard_transform.__file__)
PY
```

必须确认当前使用的是项目内 pinned submodule 对应的 `fast-hadamard-transform`。

同时检查本轮关键文件：

```bash
git status --short -- \
  scripts/prepare_rotation.py \
  snn2/hadamard.py \
  snn2/rotation.py \
  snn2/model_integration.py \
  snn2/data.py \
  scripts/evaluate_tldr.py \
  scripts/verify_artifacts.py \
  tests/test_hadamard.py \
  tests/test_rotation_regression.py \
  实验执行总结.md \
  代码结构总结.md
```

如果这些文件已有未提交修改：

- 不要覆盖用户未提交工作；
- 先阅读 diff；
- 在保留用户已有修改的基础上实施本方案。

不要执行：

```bash
git reset --hard
git checkout -- .
git clean -fd
```

---

# 2. 修改前先理解当前问题

当前 `scripts/prepare_rotation.py` 的 regression 实际比较的是：

```text
B = Base + SNN2 identity integration
vs
C = Rotated + SNN2 identity integration
```

因为当前代码在 Base regression model 上执行：

```python
base_controller = SiteController(mode="identity")
install_model_integration(
    base_model,
    base_controller,
    None,
)
```

但真正 Step 7 Base baseline：

```bash
scripts/evaluate_tldr.py ... --neuron ann --base
```

不会安装 SNN2 integration。

因此当前 regression 即使通过，也只证明：

```text
B ≈ C
```

没有证明：

```text
A ≈ B
A ≈ C
```

本轮必须彻底修复这一参照路径问题。

---

# 3. RoSTE 参考实现：本轮必须遵守的数值精度事实

Codex 必须在修改前直接阅读：

```text
OptimAI-Lab/RoSTE/rotation_utils/fuse_norm.py
OptimAI-Lab/RoSTE/rotation_utils/rotation.py
OptimAI-Lab/RoSTE/rotation_utils/hadamard.py
```

不要仅凭本文档记忆修改。

当前 RoSTE active implementation 的关键行为如下。

---

## 3.1 RMSNorm fusion：FP64

RoSTE：

```python
W_ = linear.weight.data.double()
linear.weight.data = (
    W_ * layernorm.weight.double()
).to(linear_dtype)
```

因此 SNN 的 RMSNorm scale fusion 必须继续使用 FP64 compute。

当前 `snn2/rotation.py::fuse_rmsnorm_scale()` 已经基本如此。

本轮需要保留并补强测试。

如果支持的 norm 将来存在 bias，应参考 RoSTE 同步处理 bias，但不要改变当前 Qwen3/Llama RMSNorm 的正常行为。

---

## 3.2 R1：FP64 explicit matmul

RoSTE R1 对 embedding、lm_head、q/k/v、gate/up、o/down 都使用：

```python
.to(device="cuda", dtype=torch.float64)
torch.matmul(...)
```

再 cast 回原 dtype。

因此 SNN 的 R1 不再允许通过：

```text
weight -> FP32 -> fast FHT -> original dtype
```

实现。

必须改为：

```text
materialize Q_R1 in FP64
weight -> FP64
explicit torch.matmul in FP64
-> original dtype
```

---

## 3.3 R2：active RoSTE 是混合精度

RoSTE `rotate_R2_offline()`：

### v_proj output-side

`apply_exact_had_to_linear(..., had_dim=head_dim, output=True)`：

```text
FP32 load
→ reshape
→ FP64 matrix multiply with hadK
→ original dtype
```

真正核心 transform 是 FP64 matmul。

### o_proj input-side

`apply_exact_had_to_linear(..., had_dim=-1, output=False)`：

```text
FP32
→ matmul_hadU_cuda
→ original dtype
```

因此不要把 R2 两侧统一强制成同一个 dtype。

在 SNN 当前 R2 结构不改变的前提下，对**对应的数学角色**采用：

```text
value/output-side: FP64
o_proj/input-side: FP32
```

---

## 3.4 R3 online：保持 activation dtype

RoSTE active R3：

```python
x = HadamardTransform.apply(x.contiguous()) \
    / torch.tensor(x.shape[-1]).sqrt().to(x_dtype)
```

当前 SNN 是：

```python
query = random_hadamard(query.float(), r3).to(query_dtype)
key = random_hadamard(key.float(), r3).to(key_dtype)
```

这会强制使用 FP32。

本轮必须去掉这一无条件 `.float()`。

新逻辑必须保持：

```text
输入 BF16 → Hadamard 在 BF16-compatible 路径上执行 → 输出 BF16
输入 FP16 → 对应 FP16
输入 FP32 → FP32
```

如果 CUDA backend 对某个实际训练 dtype 不支持，必须明确报错，不允许 silent fallback 到另一种 precision。

---

## 3.5 R4 offline / online：FP32

RoSTE active R4：

```python
x = matmul_hadU_cuda(x.float(), ...).to(x_dtype)
```

因此：

```text
R4 offline weight transform: FP32
R4 online activation transform: FP32
```

然后 cast 回原 dtype。

当前 SNN R4 基本采用这一策略，但在修改 Hadamard orientation 时必须保持这个 precision contract。

---

# 4. 重构 `snn2/hadamard.py`

这是本轮核心文件之一。

---

## 4.1 给 Hadamard 语义增加明确的 orientation metadata

当前 `HadamardSpec`：

```python
@dataclass
class HadamardSpec:
    name: str
    dimension: int
    seed: int
    signs: torch.Tensor
    factor_k: int
    generator: str = "paley_or_sylvester"
```

建议升级为类似：

```python
@dataclass
class HadamardSpec:
    name: str
    dimension: int
    seed: int
    signs: torch.Tensor
    factor_k: int
    generator: str = "paley_or_sylvester"
    orientation: str = "DU"
```

要求：

```text
orientation == "DU"
```

是新代码唯一接受的当前格式。

不要让新代码把旧的无 orientation rotation state 静默解释成 `DU`。

---

## 4.2 `random_hadamard()` 改成 `Q = D U`

当前旧实现等价于：

```python
# old: Q = U D
if transpose:
    return structured_hadamard(x * signs, ..., transpose=True)
return structured_hadamard(x, ..., transpose=False) * signs
```

必须改为：

```python
# new: Q = D U
if transpose:
    # x Q^T = x U^T D
    return (
        structured_hadamard(
            x,
            spec.factor_k,
            transpose=True,
        )
        * signs
    )

# x Q = x D U
return structured_hadamard(
    x * signs,
    spec.factor_k,
    transpose=False,
)
```

必须更新注释，明确写：

```text
Q = D U
D = diag(signs)
```

禁止继续保留任何：

```text
Q = H diag(sign)
Q = U D
```

的旧注释。

---

## 4.3 保证 transpose/inverse 与新的 Q 完全一致

必须满足：

\[
Q Q^T = I
\]

以及：

```python
reconstructed = random_hadamard(
    random_hadamard(x, spec),
    spec,
    transpose=True,
)
```

恢复原始 `x`。

注意：

```text
round-trip test 通过
```

本身不足以证明符号顺序正确，因为 `UD` 也可以正交。

所以必须增加显式 orientation test，见测试章节。

---

## 4.4 新增 materialize FP64 random Hadamard matrix 的 helper

R1 需要 RoSTE 风格 FP64 explicit matmul。

建议新增：

```python
def materialize_random_hadamard_matrix(
    spec: HadamardSpec,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    ...
```

数学结果必须是：

\[
Q = D U
\]

推荐实现方式：

1. 在 CPU FP64 上构造 identity；
2. 使用现有 `structured_hadamard()` 得到 normalized `U`；
3. 左乘 `D`，即对 **U 的行**乘 signs：

```python
Q = signs[:, None] * U
```

4. 最后移动到目标 device/dtype。

即：

```python
eye = torch.eye(
    spec.dimension,
    dtype=torch.float64,
    device="cpu",
)

U = structured_hadamard(
    eye,
    spec.factor_k,
    transpose=False,
)

signs = spec.signs.to(
    dtype=torch.float64,
    device=U.device,
)

Q = signs[:, None] * U
```

然后：

```python
return Q.to(device=device, dtype=dtype)
```

### 重要限制

这个 dense matrix helper主要用于 R1 hidden dimension。

不要拿它 materialize 12288/11008 等 intermediate-size R4 dense matrix，否则会产生巨大显存/内存占用。

---

## 4.5 不要破坏 fast-hadamard-transform 的 fail-fast 策略

保留当前规则：

```text
CPU:
    可使用 pure Torch fallback

CUDA:
    必须使用项目 pinned fast-hadamard-transform
    import/runtime 失败直接报错
    不允许 silent fallback
```

---

## 4.6 建议把 precision-specific helper 分开，不要继续用一个模糊的 `transform_weight_right()`

当前：

```python
transform_weight_right(...)
```

内部固定：

```python
weight.to(..., dtype=torch.float32)
```

这个 API 已经不能表达本轮 R1/R2/R4 不同 precision policy。

建议重构成语义清晰的 helper，例如：

```python
transform_weight_right_fp64_dense(...)
transform_weight_left_transpose_fp64_dense(...)

transform_weight_right_fp32_fht(...)
transform_weight_left_transpose_fp32_fht(...)
```

或其它同等清晰的命名。

要求是：

> 调用点一眼就能看出该 Rotation path 使用什么 compute precision。

不要继续保留一个内部偷偷固定 FP32、但调用者不知道的通用函数。

---

# 5. 重构 `snn2/rotation.py` 的 offline fusion

---

## 5.1 `fuse_rmsnorm_scale()` 保持 FP64

核心逻辑保持：

```python
scale = norm.weight.detach().to(dtype=torch.float64)

weight_fp64 = weight.to(torch.float64)
new_weight = weight_fp64 * scale_fp64
linear.weight.data = new_weight.to(original_dtype)
```

若线性层和 norm 在不同 device，继续支持 cross-device。

增加测试保证：

```text
实际乘法 compute dtype = float64
结果 cast 回原 weight dtype
```

---

## 5.2 R1 必须统一改为 FP64 dense Q

在 `fuse_rotations()` 中创建：

```python
r1_matrix_fp64 = materialize_random_hadamard_matrix(
    r1,
    device=device,
    dtype=torch.float64,
)
```

然后 R1 所有 offline transform 都使用这个同一个 matrix。

### embedding

目标：

\[
W'_{\mathrm{embed}} = W_{\mathrm{embed}} Q
\]

FP64：

```python
W64 = W.to(device=device, dtype=torch.float64)
W_new = W64 @ Q64
```

cast 回 original device + dtype。

---

### lm_head

同样：

\[
W'_{\mathrm{head}} = W_{\mathrm{head}} Q
\]

FP64。

---

### q_proj / k_proj / v_proj / gate_proj / up_proj

同样：

\[
W' = W Q
\]

FP64。

---

### o_proj / down_proj

目标：

\[
W' = Q^T W
\]

必须直接使用：

```python
Q64.T @ W64
```

FP64。

不要通过 FP32 FHT 间接实现。

---

### o_proj / down_proj bias

如果存在 bias，按 RoSTE：

\[
b' = b Q
\]

FP64 compute 后 cast 回原 dtype。

---

## 5.3 R2 value/output-side 改为 FP64

当前 `_rotate_value_projection()`：

- reshape 每个 KV head；
- 使用 `transform_weight_left_transpose()`；
- bias 通过 `random_hadamard(... float32 ...)`。

本轮修改为 FP64。

建议对 R2 head_dim materialize 一个 FP64 dense Q：

```python
Q2_64 = materialize_random_hadamard_matrix(
    r2,
    device=device,
    dtype=torch.float64,
)
```

head_dim 很小，因此没有 R4 那种内存风险。

对于每个 output head block，保持当前数学方向：

\[
W' = Q_2^T W
\]

通过 FP64：

```python
Q2_64.T @ W_block_64
```

bias：

\[
b' = b Q_2
\]

也使用 FP64。

最后 cast 回原 dtype。

---

## 5.4 R2 o_proj/input-side 保持 FP32 FHT

当前 `_rotate_o_projection_input()` 已经基本是：

```python
weight -> FP32
random_hadamard(...)
-> original dtype
```

保留这一 precision policy。

但由于 `random_hadamard()` orientation 改成 `DU`，新的 transform 自动变成：

```text
先 signs
再 U
```

必须加注释：

```text
R2 o_proj/input-side follows RoSTE active FP32 FHT precision policy.
```

---

## 5.5 R4 offline 保持 FP32 FHT

`mlp.down_proj.weight` 的 R4 fusion：

\[
W' = W Q_4
\]

继续：

```text
original weight
→ FP32
→ new random_hadamard(Q = DU)
→ original dtype
```

不要改成 dense FP64。

---

## 5.6 Rotation state format 必须 bump version

因为：

```text
旧 artifacts:
Q = U D

新代码:
Q = D U
```

两者不能兼容。

如果只改 Python 函数、不升级 state version，新代码会把旧 `rotation_state.pt` 的 signs 当成 `DU` 使用，而旧 `fused_base` 权重是按 `UD` 融合的，最终 R3/R4 会严重错配。

因此 `fuse_rotations()` 输出 state 必须升级，例如：

```python
{
    "format_version": 2,
    "seed": ...,
    "random_hadamard_orientation": "DU",
    "precision_policy": "roste_aligned_v1",
    ...
}
```

同时建议写入 model config：

```python
model.config.snn2_rotation_fused = True
model.config.snn2_rotation_seed = ...
model.config.snn2_online_rotations = ["R3", "R4"]

model.config.snn2_rotation_format_version = 2
model.config.snn2_random_hadamard_orientation = "DU"
model.config.snn2_rotation_precision_policy = "roste_aligned_v1"
```

---

## 5.7 `load_rotation_state()` / `load_specs()` 必须 fail-fast 拒绝旧 state

不要：

```python
state.get("random_hadamard_orientation", "DU")
```

因为这会把旧 artifact 静默解释成新 artifact。

必须显式检查：

```text
format_version == 2
random_hadamard_orientation == "DU"
precision_policy == "roste_aligned_v1"
```

若缺失或不匹配：

```text
raise RuntimeError(
    "Legacy/incompatible rotation artifact detected. "
    "Re-run scripts/prepare_rotation.py with the current code."
)
```

这条保护对：

```text
train_ann.py
discover_prefix.py
calibrate_sites.py
evaluate_tldr.py
evaluate_lm_harness.py
```

后续任何通过 `rotation_state()` 加载旧工件的流程都必须生效。

---

# 6. 修改 `snn2/model_integration.py` 的 online precision

---

## 6.1 R3 去掉无条件 FP32 cast

当前：

```python
query_dtype = query.dtype
key_dtype = key.dtype

query = random_hadamard(
    query.float(),
    r3,
).to(query_dtype)

key = random_hadamard(
    key.float(),
    r3,
).to(key_dtype)
```

改为：

```python
query = random_hadamard(
    query.contiguous(),
    r3,
)

key = random_hadamard(
    key.contiguous(),
    r3,
)
```

或等价实现。

必须保证输出 dtype 与输入一致。

由于新的 `random_hadamard()` 是 `Q = DU`：

```text
query/key:
先乘 signs
再 Hadamard
```

---

## 6.2 R4 online 继续 FP32

当前：

```python
product_dtype = product.dtype
product = random_hadamard(
    product.float(),
    r4,
).to(product_dtype)
```

这个 precision policy与 RoSTE active R4 一致。

保留：

```text
FP32 compute
→ original activation dtype
```

但语义会随 `random_hadamard()` 改为 `DU`。

建议把代码写得更明确，例如：

```python
product_dtype = product.dtype
product = random_hadamard(
    product.to(torch.float32),
    r4,
).to(product_dtype)
```

并注释：

```text
RoSTE-aligned R4 online precision: FP32 compute, then cast back.
```

---

## 6.3 不改变 attention / MLP 其它数值路径

本轮不要顺带修改：

```text
softmax dtype
attention qk accumulation
controller identity behavior
activation replacement
SNN temporal execution
```

本轮只改 online Rotation precision 与 Hadamard orientation。

---

# 7. 为 TL;DR prompt-end diagnostic 提取共享 prompt helper

当前 `scripts/evaluate_tldr.py` 自己定义：

```python
_prompt_and_reference(row)
```

为了保证 regression 的 prompt-end 输入与正式 TL;DR evaluation 不漂移，建议把 TL;DR prompt parsing/encoding 下沉为共享 helper。

推荐位置：

```text
snn2/data.py
```

---

## 7.1 新增共享解析 helper

例如：

```python
def tldr_prompt_and_reference(
    row: dict[str, Any],
) -> tuple[str, str]:
    ...
```

逻辑必须与当前 `evaluate_tldr.py::_prompt_and_reference()` 完全一致：

```python
prompt = _as_text(
    row.get(
        "prompt",
        row.get(
            "pompt",
            row.get(
                "article",
                row.get("text", ""),
            ),
        ),
    )
)

reference = _as_text(
    row.get(
        "completion",
        row.get(
            "summary",
            row.get(
                "label",
                row.get("response", ""),
            ),
        ),
    )
)
```

---

## 7.2 新增 evaluation-prompt encoding helper

例如：

```python
def encode_tldr_generation_prompt(
    row: dict[str, Any],
    tokenizer: Any,
    cfg: dict[str, Any],
) -> list[int]:
    prompt, _ = tldr_prompt_and_reference(row)

    return tokenizer.encode(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=int(
            cfg["evaluation"].get(
                "tldr_input_length",
                512,
            )
        ),
    )
```

注意：

**这里必须使用 `evaluation.tldr_input_length`，不是 `data.max_seq_length`。**

这是 prompt-end 与真实 TL;DR generation 对齐的关键。

---

## 7.3 `evaluate_tldr.py` 改为复用共享 helper

删除脚本内重复的 `_prompt_and_reference()`，改为 import：

```python
from snn2.data import (
    ...,
    tldr_prompt_and_reference,
    encode_tldr_generation_prompt,
)
```

生成 batch 时也调用共享 `encode_tldr_generation_prompt()`。

这样以后 evaluation 和 regression 不会出现两套 prompt tokenization。

---

# 8. 重构 `snn2/rotation.py` 的 regression 架构

不要继续保留一个只接受：

```python
original_model,
rotated_model
```

的 `validate_rotation_logits()` 作为核心架构。

建议重构为：

```text
低层 pair metric
+
三模型 suite orchestration
```

---

# 9. 保留并复用现有 all-token metric accumulator

当前 `_LogitsErrorAccumulator` 已经统计：

```text
relative_l2_error
max_abs_error
mean_abs_error
p99_abs_error
p999_abs_error
top1_agreement
Top-1 counts
margin-aware diagnostics
```

这些 all-token metrics 继续保留。

不要因为新增 A/B/C 和 prompt-end 把原有诊断删掉。

---

# 10. 把 hard-gate assessment 与 raise 解耦

当前：

```python
enforce_rotation_regression(...)
```

会在某一 pair fail 时立即 raise。

三路 regression 不能这样做，否则：

```text
A↔B 一 fail
→ 直接退出
→ B↔C、A↔C 根本没有结果
```

无法完成自动定位。

必须重构。

---

## 10.1 新增纯 assessment helper

例如：

```python
def assess_rotation_regression(
    result: dict[str, Any],
    relative_l2_threshold: float,
    top1_agreement_threshold: float,
) -> dict[str, Any]:
    ...
```

它：

- 添加 threshold；
- 计算 `passed`；
- **不 raise**。

逻辑保持完全不变：

```python
relative_l2_passed = (
    result["relative_l2_error"]
    <= relative_l2_threshold
)

top1_passed = (
    result["top1_agreement"]
    > top1_agreement_threshold
)
```

注意：

```text
<= 0.05
> 0.95
```

比较符号不要改。

---

## 10.2 保留兼容 wrapper

可以继续保留：

```python
enforce_rotation_regression(...)
```

作为：

```python
checked = assess_rotation_regression(...)

if not checked["passed"]:
    raise RotationRegressionError(checked)

return checked
```

从而减少对现有 tests/API 的不必要破坏。

---

# 11. 新增 Prompt-End Decision Accumulator

建议新增独立类：

```python
class _PromptEndDecisionAccumulator:
    ...
```

不要把 prompt-end 逻辑硬塞进 `_LogitsErrorAccumulator`，因为两者统计粒度不同：

```text
all-token:
    每个有效 token 一个 decision

prompt-end:
    每个 prompt 只取 1 个 decision
```

---

## 11.1 Prompt-end 位置必须由 attention_mask 推导

不要假设 padding 一定在右边。

通用计算：

```python
positions = torch.arange(
    attention_mask.shape[1],
    device=attention_mask.device,
).unsqueeze(0)

end_positions = positions.masked_fill(
    ~attention_mask.bool(),
    -1,
).amax(dim=1)
```

然后 gather：

```python
batch_index = torch.arange(batch_size)

end_logits = logits[
    batch_index,
    end_positions,
    :
]
```

这样 left/right padding 都安全。

---

## 11.2 Prompt-end 至少记录以下 metrics

每个 pair：

```json
{
  "num_prompts_compared": 128,

  "relative_l2_error": ...,
  "max_abs_error": ...,
  "mean_abs_error": ...,

  "top1_agreement": ...,
  "top1_agreement_count": ...,
  "top1_disagreement_count": ...,

  "reference_top1_margin": {
    "mean": ...,
    "p50": ...,
    "p90": ...,
    "p99": ...,
    "max": ...
  },

  "per_prompt_max_abs_error": {
    "mean": ...,
    "p50": ...,
    "p90": ...,
    "p99": ...,
    "max": ...
  },

  "gating": false
}
```

推荐额外保存有限数量 mismatch examples，例如最多 32 个：

```json
"mismatch_examples": [
  {
    "calibration_position": 17,
    "dataset_index": 12345,
    "reference_top1_token_id": 123,
    "candidate_top1_token_id": 456
  }
]
```

不要保存完整 vocab logits 到 JSON。

---

## 11.3 prompt-end 不参与本轮 `passed`

必须明确：

```python
prompt_end["gating"] = False
```

pair：

```python
pair["passed"]
```

仍由：

```text
pair["all_tokens"]["relative_l2_error"]
pair["all_tokens"]["top1_agreement"]
```

决定。

---

# 12. 新增单个 pair regression helper

建议接口：

```python
def validate_model_pair_logits(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    reference_label: str,
    candidate_label: str,
    tokenizer: Any,
    calibration_dataset: Any,
    cfg: dict[str, Any],
    calibration_manifest_path: str | Path,
    calibration_manifest_sha256: str,
) -> dict[str, Any]:
    ...
```

但为了避免 A/B/C 各跑三遍整套模型，**更推荐**在 suite 中一次 forward A/B/C，然后同时更新三个 pair accumulators。

---

# 13. 推荐的 A/B/C suite forward 实现

新增：

```python
def validate_rotation_regression_suite(
    model_a: nn.Module,
    model_b: nn.Module,
    model_c: nn.Module,
    tokenizer: Any,
    calibration_dataset: Any,
    cfg: dict[str, Any],
    *,
    calibration_manifest_path: str | Path,
    calibration_manifest_sha256: str,
) -> dict[str, Any]:
    ...
```

---

## 13.1 入口 assertions

必须验证：

### A

```text
不要求 snn2_site_integration
不得带 snn2_online_rotations
```

### B

```text
snn2_site_integration == True
controller.mode == "identity"
rotation_state == None
没有 R3/R4
```

建议给 model B 保存 runtime marker：

```python
model_b.config.snn2_regression_variant = "B_identity_no_rotation"
```

或通过函数参数/controller 显式检查。

### C

```text
snn2_rotation_fused == True
snn2_site_integration == True
online rotations == {"R3", "R4"}
rotation artifact orientation == "DU"
precision policy == "roste_aligned_v1"
controller.mode == "identity"
```

---

## 13.2 all-token pass

继续使用同一套：

```text
128 calibration samples
training-style tokenization
no Prefix
```

即当前：

```python
tokenize_dataset(
    calibration_dataset,
    tokenizer,
    cfg,
    prefix_ids=None,
)
```

不要删掉原 all-token regression。

---

## 13.3 每个 batch 的推荐执行顺序

为了避免同时在 GPU 上长期保留三份 logits：

```text
A forward
→ logits_A detach → CPU

B forward
→ logits_B detach → CPU
→ update A↔B all-token accumulator

C forward
→ logits_C
→ update A↔C
→ update B↔C

delete batch logits
```

三模型本身可以同时 resident。

当前目标硬件是大显存 GPU；如果 `device_map="auto"` 因三模型产生 CPU offload，也允许正确执行，但不要为了速度牺牲三路 regression 正确性。

禁止因为显存问题重新回到只测 B↔C。

如果 Codex 实测三模型无法 resident，可以把 suite 实现成 memory-safe 两模型分阶段跑，但最终 JSON 必须仍包含完整三组结果。

---

# 14. Prompt-end pass 必须使用同样 128 raw calibration rows

在 all-token pass 后，再执行 prompt-only pass。

对每个 calibration row：

```python
prompt_ids = encode_tldr_generation_prompt(
    row,
    tokenizer,
    cfg,
)
```

构造 prompt batch。

### Padding

正式 `evaluate_tldr.py` 设置：

```python
tokenizer.padding_side = "left"
```

因此 prompt-end diagnostic 也应显式使用 left padding，避免未来 batch_size > 1 时发生位置差异。

推荐：

```python
old_padding_side = tokenizer.padding_side

try:
    tokenizer.padding_side = "left"
    ...
finally:
    tokenizer.padding_side = old_padding_side
```

不要永久改变 tokenizer 全局状态。

---

## 14.1 Prompt-only forward

A/B/C 都：

```python
model(
    input_ids=...,
    attention_mask=...,
    use_cache=False,
).logits
```

然后只 gather 每个 prompt 最后有效位置的 logits。

不需要生成 32 token。

本轮第 3 项只要求检查：

```text
真正启动 generation 的 prompt-end decision
```

不要擅自把 regression 扩展成完整 32-step greedy generation hard gate。

---

# 15. 三组 comparison 的 JSON 结构

建议：

```json
"comparisons": {
  "A_vs_B": {
    "reference": "A_original_hf_base",
    "candidate": "B_base_snn2_identity_no_rotation",

    "all_tokens": {
      "... existing metrics ...": "...",
      "threshold": {
        "relative_l2_error": 0.05,
        "top1_agreement": 0.95
      },
      "passed": true
    },

    "prompt_end": {
      "... prompt-end metrics ...": "...",
      "gating": false
    },

    "passed": true
  },

  "B_vs_C": {
    ...
  },

  "A_vs_C": {
    ...
  }
}
```

pair root `passed` 应与：

```python
pair["all_tokens"]["passed"]
```

一致。

---

# 16. 自动 diagnosis 规则

必须在 `rotation_regression.json` 中生成结构化 diagnosis，而不是只写日志字符串。

建议：

```json
"diagnosis": {
  "code": "...",
  "summary": "...",
  "pair_pass": {
    "A_vs_B": true,
    "B_vs_C": true,
    "A_vs_C": true
  },
  "prompt_end_top1_agreement": {
    "A_vs_B": ...,
    "B_vs_C": ...,
    "A_vs_C": ...
  }
}
```

---

## 16.1 至少实现以下 diagnosis mapping

### Case 1：三组都 pass

```text
A↔B = pass
B↔C = pass
A↔C = pass
```

```text
code = "no_regression_detected"
```

说明：

```text
HF Base → SNN2 identity integration
以及
SNN2 identity → Rotation
在当前 hard gates 下都通过。
```

---

### Case 2：A↔B fail，B↔C pass，A↔C fail

```text
code = "snn2_integration_mismatch"
```

优先定位：

```text
model_integration.py / custom attention or MLP integration
```

而不是 Hadamard。

---

### Case 3：A↔B pass，B↔C fail，A↔C fail

```text
code = "rotation_mismatch"
```

优先定位：

```text
Hadamard orientation
offline fusion
R2/R3/R4
precision
```

---

### Case 4：A↔B fail，B↔C fail

```text
code = "integration_and_rotation_mismatch"
```

无论 A↔C 是否 pass，都说明至少两个阶段分别存在问题。

如果 A↔C 恰好 pass，要在 summary 中明确：

```text
possible cancellation; do not interpret A↔C pass as correctness
```

---

### Case 5：A↔B pass，B↔C pass，但 A↔C fail

由于 hard thresholds 并不具备传递性，允许发生：

```text
A≈B
B≈C
但累计误差导致 A↔C 超阈值
```

使用：

```text
code = "end_to_end_accumulation_mismatch"
```

不要错误地 raise “impossible state”。

---

### Case 6：其它组合

使用：

```text
code = "mixed_regression_failure"
```

并在 `summary` 中列出三个 pair 状态。

---

# 17. 整体 `passed` 与 hard-fail 语义

新的 suite：

```python
overall_passed = (
    comparisons["A_vs_B"]["passed"]
    and comparisons["B_vs_C"]["passed"]
    and comparisons["A_vs_C"]["passed"]
)
```

即：

```text
三组 all-token hard gate 全部 pass
→ suite passed

任何一组 fail
→ suite failed
```

prompt-end 不参与这一布尔值。

---

## 17.1 不能在第一组 fail 时立即退出

必须：

```text
先完成 A↔B
再完成 B↔C
再完成 A↔C
生成 diagnosis
写完整 JSON
最后才 hard-fail
```

---

## 17.2 建议新增 suite-level exception

例如：

```python
class RotationRegressionSuiteError(RuntimeError):
    def __init__(self, result):
        self.result = result
        ...
```

不要让旧 `RotationRegressionError` 假设 root 上存在：

```text
relative_l2_error
top1_agreement
```

因为新版 root 是 suite。

---

# 18. `rotation_regression.json` 升级为 format version 4

建议完整顶层结构：

```json
{
  "format_version": 4,

  "purpose": "three_way_rotation_regression",

  "status": "passed",

  "model_name": "Qwen/Qwen3-1.7B-Base",
  "model_revision": "...",
  "rotation_seed": 42,
  "dtype": "bfloat16",

  "num_samples": 128,

  "calibration_manifest_path": "...",
  "calibration_manifest_sha256": "...",

  "hard_gate": {
    "scope": "all_tokens_only",
    "relative_l2_error": 0.05,
    "top1_agreement": 0.95,
    "prompt_end_is_gating": false
  },

  "rotation_implementation": {
    "random_hadamard_orientation": "DU",
    "precision_policy": "roste_aligned_v1"
  },

  "models": {
    "A": {
      "label": "original_hf_base",
      "snn2_integration": false,
      "rotation": false
    },
    "B": {
      "label": "base_snn2_identity_no_rotation",
      "snn2_integration": true,
      "controller": "identity",
      "rotation": false
    },
    "C": {
      "label": "rotated_snn2_identity",
      "snn2_integration": true,
      "controller": "identity",
      "rotation": true,
      "online_rotations": ["R3", "R4"]
    }
  },

  "comparisons": {
    "A_vs_B": { "...": "..." },
    "B_vs_C": { "...": "..." },
    "A_vs_C": { "...": "..." }
  },

  "diagnosis": {
    "...": "..."
  },

  "passed": true
}
```

---

# 19. 重构 `scripts/prepare_rotation.py`

---

## 19.1 模型实例必须明确为 A/B/C

不要继续使用模糊变量：

```python
base_model
model
```

建议：

```python
model_a
model_b
model_c
```

或更具语义：

```python
original_hf_base
integrated_base
rotated_model
```

---

## 19.2 加载三份同源模型

三者均从：

```python
cfg["experiment"]["model_name"]
```

以及相同：

```text
model_revision
dtype
attn_implementation=eager
device_map
```

加载。

---

## 19.3 A 不安装 integration

A：

```python
model_a = load_model(...)
model_a.eval()
```

**不要调用**：

```python
install_model_integration(model_a, ...)
```

这是本轮任务 1 的核心。

---

## 19.4 B 安装 identity integration，但 no rotation

```python
controller_b = SiteController(
    mode="identity",
)

install_model_integration(
    model_b,
    controller_b,
    None,
)
```

必须确认：

```text
R3 = None
R4 = None
```

---

## 19.5 C 先 fuse，再安装完整 integration

```python
state = fuse_rotations(
    model_c,
    seed=...,
    device=...,
)

controller_c = SiteController(
    mode="identity",
)

install_model_integration(
    model_c,
    controller_c,
    state,
)
```

C 必须同时包含：

```text
fused/offline R1
fused/offline R2
fused/offline R4 inverse/weight transform
online R3
online R4
```

---

## 19.6 调用三路 suite

替换旧：

```python
validate_rotation_logits(
    base_model,
    model,
    ...
)
```

为：

```python
validate_rotation_regression_suite(
    model_a,
    model_b,
    model_c,
    tokenizer,
    calibration,
    cfg,
    calibration_manifest_path=...,
    calibration_manifest_sha256=...,
)
```

---

## 19.7 即使 fail 也必须写完整 JSON

流程：

```python
try:
    regression = validate_rotation_regression_suite(...)
except RotationRegressionSuiteError as exc:
    write_json(
        regression_path,
        exc.result,
    )
    run.event(
        "rotation_regression_failed",
        ...
    )
    raise
```

不要写只有：

```text
num_samples = 0
```

的最终失败文件。

`in_progress` 可以在开始时写，但 suite fail 后必须覆盖为完整结果。

---

## 19.8 只有 suite 全 pass 才允许保存 reusable rotation artifacts

只有：

```text
A↔B all-token pass
B↔C all-token pass
A↔C all-token pass
```

全部满足后，才：

```text
save rotation_state.pt
save fused_base/
write rotation_summary.json
```

如果 suite fail：

```text
不要产生“看起来可用”的新 fused_base / rotation_state
```

若目录中已经存在旧 artifact，建议在运行开始时不要直接删除；但新代码必须通过 artifact version fail-fast 防止下游误用旧 state。

文档中要提醒用户主动清理/重建旧 rotation artifacts。

---

## 19.9 回收 A/B 显存，保留 C 用于保存

suite pass 后：

```python
del model_a
del model_b
gc.collect()
torch.cuda.empty_cache()
```

保留 `model_c`。

保存 C 前恢复仅与 Python runtime integration 有关的 config metadata，例如当前已有：

```python
model_c.config._attn_implementation = "eager"
model_c.config._attn_implementation_internal = "eager"

if hasattr(
    model_c.config,
    "snn2_site_integration",
):
    delattr(
        model_c.config,
        "snn2_site_integration",
    )
```

但必须保留：

```text
snn2_rotation_fused
snn2_rotation_seed
snn2_online_rotations
snn2_rotation_format_version
snn2_random_hadamard_orientation
snn2_rotation_precision_policy
```

因为这些属于 checkpoint provenance。

---

# 20. `rotation_summary.json` 同步升级

至少增加：

```json
{
  "random_hadamard_orientation": "DU",
  "precision_policy": "roste_aligned_v1",

  "precision_details": {
    "rmsnorm_fusion": "float64",
    "R1_offline": "float64_explicit_matmul",
    "R2_value_output_side": "float64_matmul",
    "R2_o_proj_input_side": "float32_fht",
    "R3_online": "preserve_input_dtype",
    "R4_offline": "float32_fht",
    "R4_online": "float32_fht_then_cast_back"
  },

  "rotation_regression_format_version": 4,
  "rotation_regression_passed": true
}
```

---

# 21. 更新 `scripts/verify_artifacts.py`

当前 verifier 假定：

```text
rotation_regression.format_version == 3
purpose == "base_vs_rotated_logits_regression"
root 直接包含一组 metrics
```

本轮必须全部升级。

---

## 21.1 新 verifier 顶层要求

要求：

```text
format_version == 4
purpose == "three_way_rotation_regression"
num_samples == 128
passed == true
status == "passed"
```

---

## 21.2 校验三个 pair 都存在

必须有：

```text
A_vs_B
B_vs_C
A_vs_C
```

每个 pair：

```text
all_tokens
prompt_end
passed
```

---

## 21.3 复用原 `_verify_rotation_regression_metrics()`

建议把现有 verifier 改名或保持功能，让它验证：

```python
comparison["all_tokens"]
```

不要删掉现有：

```text
histogram consistency
Top-1 counts
margin-aware diagnostic
threshold consistency
passed consistency
```

---

## 21.4 新增 prompt-end verifier

至少检查：

```text
num_prompts_compared == 128
top1_agreement in [0,1]
agreement_count + disagreement_count == 128
relative_l2_error finite >= 0
max_abs_error finite >= 0
mean_abs_error finite >= 0
gating == false
```

以及分布字段合法。

---

## 21.5 校验 pair passed

```python
pair["passed"]
==
pair["all_tokens"]["passed"]
```

---

## 21.6 校验 suite passed

```python
regression["passed"]
==
(
    A_vs_B.passed
    and B_vs_C.passed
    and A_vs_C.passed
)
```

---

## 21.7 校验 orientation / precision metadata

要求：

```text
rotation_implementation.random_hadamard_orientation == "DU"
rotation_implementation.precision_policy == "roste_aligned_v1"
```

若读取 `rotation_summary.json`，也要核对一致。

---

# 22. 更新 `tests/test_hadamard.py`

现有 round-trip test 保留，但必须新增 orientation test。

---

## 22.1 显式证明新 Q = D U

测试：

```python
spec = make_spec(
    "test",
    dimension,
    seed,
)

U = structured_hadamard(
    torch.eye(
        dimension,
        dtype=torch.float64,
    ),
    spec.factor_k,
)

D = torch.diag(
    spec.signs.to(
        dtype=torch.float64,
    )
)

Q_expected = D @ U

x = torch.randn(
    ...,
    dimension,
    dtype=torch.float64,
)

actual = random_hadamard(
    x,
    spec,
)

expected = x @ Q_expected

torch.testing.assert_close(
    actual,
    expected,
    ...
)
```

---

## 22.2 显式证明 transpose

```python
actual_t = random_hadamard(
    x,
    spec,
    transpose=True,
)

expected_t = x @ Q_expected.T
```

必须一致。

---

## 22.3 加一个“不是旧 UD”的 regression test

选取：

```text
非全相同 signs
非特殊对称 x
```

计算：

```python
Q_old = U @ D
Q_new = D @ U
```

断言：

```python
x @ Q_new
```

与：

```python
x @ Q_old
```

不是同一个结果。

这样可以防止以后有人把 sign 顺序又改回去但 round-trip test 仍通过。

---

## 22.4 dense matrix helper test

测试：

```python
materialize_random_hadamard_matrix(spec)
```

等于：

```python
D @ U
```

并验证 dtype：

```text
float64
```

---

# 23. 新增/扩展 Rotation precision tests

建议在：

```text
tests/test_hadamard.py
tests/test_rotation_regression.py
```

之间合理拆分。

至少覆盖：

---

## 23.1 R1 FP64 helper

构造小 linear weight，验证结果等于：

```python
W.double() @ Q.double()
```

以及：

```python
Q.double().T @ W.double()
```

然后 cast 回原 dtype。

---

## 23.2 R2 value-side FP64

小 head_dim 测试：

```text
output block transform
```

与显式 FP64 matrix multiplication 一致。

---

## 23.3 R2 o-side FP32 contract

测试 helper 输入到 FHT 的 compute tensor 是 FP32。

可以：

- monkeypatch lower-level helper；
- 或暴露一个足够小的 precision-specific function 后直接检查输出/内部 contract。

不要用只比较最终 BF16 输出的测试来“推测”内部 compute dtype。

---

## 23.4 R3 preserve dtype

至少测试：

```text
float32 CPU
```

如果测试环境支持 CUDA，再测试：

```text
bfloat16 CUDA
```

要求：

```python
output.dtype == input.dtype
```

并确保代码没有 `.float()`。

可以通过 monkeypatch `random_hadamard` / lower helper 记录输入 dtype。

---

## 23.5 R4 online FP32

测试 R4 wrapper 确实将 activation 转为 FP32 compute，再 cast 回原 dtype。

---

# 24. 重构 `tests/test_rotation_regression.py`

现有 all-token tests 保留。

---

## 24.1 `assess_rotation_regression()` 测试

保留原 hard gate case：

```text
0.04 / 0.96 → pass
0.06 / 0.99 → fail
0.01 / 0.94 → fail
0.01 / 0.95 → fail
0.05 / 0.96 → pass
```

这证明阈值完全没变。

---

## 24.2 suite 不得 first-failure short-circuit

构造 fake/synthetic metrics：

```text
A↔B fail
B↔C pass
A↔C fail
```

验证最终 suite 同时保存三个 pair，而不是第一个 fail 就中断。

---

## 24.3 diagnosis mapping tests

至少覆盖：

```text
all pass
integration mismatch
rotation mismatch
integration + rotation mismatch
end-to-end accumulation mismatch
mixed
```

---

## 24.4 Prompt-end 位置测试

构造：

```text
right padded
left padded
```

attention mask。

确保选择的都是最后一个有效位置，而不是：

```text
tensor[:, -1]
```

或：

```text
attention_mask.sum()-1
```

这种只适用于特定 padding 的实现。

---

## 24.5 Prompt-end Top-1 测试

构造：

```text
all-token top1 很高
但 prompt-end top1 不一致
```

验证：

```text
all_tokens.passed 可以为 true
prompt_end.top1_agreement < 1
pair.passed 仍按 all-token gate
```

这正是本轮要求。

---

## 24.6 Prompt-only tokenization 测试

确保：

```text
completion 不进入 prompt-end input
```

并确保最大长度取：

```text
evaluation.tldr_input_length
```

不是：

```text
data.max_seq_length
```

---

# 25. 更新 `tests/test_rotated_pre_finetuning_protocol.py`

检查现有 tests 是否 mock：

```text
rotation_state format
rotation_regression format
fused Base metadata
```

如有，升级为：

```text
orientation = DU
precision_policy = roste_aligned_v1
regression format_version = 4
```

不要破坏：

```text
rotated_pre_finetuning.prefix_enabled
```

协议。

---

# 26. 更新其它受影响测试

执行：

```bash
grep -RIn \
  --exclude-dir=.git \
  --exclude-dir=artifacts \
  --exclude-dir=fast-hadamard-transform \
  -e 'format_version.*3' \
  -e 'base_vs_rotated_logits_regression' \
  -e 'rotation_regression' \
  -e 'Q = H diag' \
  -e 'Q = U D' \
  tests snn2 scripts *.md docs 2>/dev/null
```

逐项判断。

所有 active test 如果依赖旧 regression schema 都必须更新。

不要机械改 `docs/history/` 中作为历史记录保留的旧方案，除非文件明确声明自己描述“当前实现”。

---

# 27. 更新 `实验执行总结.md`

这是用户实际执行实验的主文档，必须同步。

---

## 27.1 总流程中 Rotation regression 描述

原来类似：

```text
Base ↔ Rotated logits regression
```

改为：

```text
A = Original HF Base
B = Base + SNN2 identity integration, no rotation
C = Rotated + SNN2 identity integration

prepare_rotation.py 自动执行：
A ↔ B
B ↔ C
A ↔ C
```

---

## 27.2 明确两类 regression signal

写清：

### all-token regression

使用：

```text
固定 128 calibration samples
训练式 prompt + completion tokenization
```

hard gate 保持：

```text
relative_l2_error <= 0.05
top1_agreement > 0.95
```

### prompt-end diagnostic

使用：

```text
同样 128 calibration raw rows
只编码 TL;DR prompt
evaluation.tldr_input_length
检查 generation 第一个 next-token decision
```

本轮：

```text
diagnostic only
not hard-gating
```

---

## 27.3 写清 diagnosis 含义

至少写：

```text
A↔B fail，B↔C pass：
    优先排查 SNN2 integration

A↔B pass，B↔C fail：
    优先排查 Rotation/Hadamard

A↔B、B↔C 都 fail：
    两阶段均存在问题

三者 all-token 都 pass：
    prepare_rotation 才允许输出 reusable rotation artifacts
```

---

## 27.4 写清新 Hadamard orientation

加入：

\[
Q = D U
\]

其中：

```text
D = diag(random_signs)
U = normalized structured Hadamard
```

明确：

```text
先 sign，再 Hadamard
```

---

## 27.5 写清 RoSTE-aligned precision policy

写入：

```text
RMSNorm fusion: FP64
R1 offline: FP64
R2 value-side: FP64
R2 o-side: FP32
R3 online: preserve activation dtype
R4 offline/online: FP32
```

---

## 27.6 明确旧 artifacts 全部失效

由于本轮：

```text
Q: UD → DU
```

且 precision 改变，因此所有旧 rotation-derived artifacts 不再可复用。

文档必须写明：

```text
必须重新运行 prepare_rotation.py
```

以及下游依赖关系。

---

# 28. 更新 `代码结构总结.md`

Rotation 一节必须更新。

至少改成：

```text
Random Hadamard:
Q = D U

R1:
FP64 offline fusion

R2:
value-side FP64
o-side FP32

R3:
online, activation dtype preserved

R4:
FP32 offline/online
```

并补充：

```text
prepare_rotation.py 现在执行三路 A/B/C regression
```

---

# 29. 旧 artifacts 的兼容性与重跑范围

这是本轮非常重要的实验完整性要求。

由于 Rotation 本身发生实质改变：

```text
旧 Q = U D
新 Q = D U
```

并且 R1/R2/R3 precision 改变，所以旧的：

```text
rotation/fused_base
rotation_state.pt
```

全部不能继续使用。

---

## 29.1 必须重新生成的 model-level shared artifacts

对所有 rotation-enabled model-task pair：

```text
rotation/
fused_base/
rotation_state.pt
rotation_regression.json
rotation_summary.json

rotated_prefix/ann_training_prefix/
rotated_prefix/ann_training_calibration/

rotated_prefix/rotated_pre_finetuning/
```

都必须重新生成。

---

## 29.2 已经基于旧 Rotation 训练的 ANN checkpoint

如果以下 mode 已经训练：

```text
unaware
phase_aware
gif_aware
```

这些 checkpoint 的初始 rotated Base 已改变，因此必须视为旧实验结果，不能与新 Rotation pipeline 混用。

需要重新：

```text
ANN training
post-finetuning Prefix
post-finetuning conversion calibration
ANN evaluation
SNN conversion/evaluation
```

---

## 29.3 Vanilla

Vanilla training 本身：

```text
no rotation
no training Prefix
```

因此其 checkpoint 不因 `UD → DU` 直接失效。

但如果用户要重建完整统一实验目录，可以按完整主流程重新跑。

不要在代码中自动删除用户 artifacts。

---

# 30. `prepare_rotation.py` 真实运行后的验收

完成代码修改和全部 unit tests 后，用 Qwen3-1.7B TL;DR 做真实 smoke/regression。

---

## 30.1 先确保数据 manifest 已存在

如果已经按主流程准备过：

```text
artifacts/snn2_main_v1/tldr/_shared/seed42/data/calibration_manifest.json
```

不需要因为本轮 Rotation 修改重新抽样。

若不存在：

```bash
python scripts/prepare_data.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
```

---

## 30.2 清理/隔离旧 Rotation artifacts

不要让旧 `UD` artifact 混进验证。

可以先备份：

```bash
ROT_ROOT="artifacts/snn2_main_v1/tldr/Qwen_Qwen3-1.7B-Base/_shared/seed42/rotation"

if [ -d "$ROT_ROOT" ]; then
  mv "$ROT_ROOT" "${ROT_ROOT}.pre_DU_backup"
fi
```

具体真实路径以 `ArtifactLayout` 为准，不要盲目复制上述路径。

---

## 30.3 运行

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"
```

---

# 31. 检查新版 `rotation_regression.json`

必须确认：

```text
format_version = 4
purpose = three_way_rotation_regression
num_samples = 128

comparisons.A_vs_B 存在
comparisons.B_vs_C 存在
comparisons.A_vs_C 存在

每组：
    all_tokens 存在
    prompt_end 存在

prompt_end.gating = false

rotation_implementation.random_hadamard_orientation = DU
rotation_implementation.precision_policy = roste_aligned_v1
```

---

## 31.1 查看核心结果

推荐：

```bash
python - <<'PY'
import json
from pathlib import Path

paths = list(Path("artifacts").rglob("rotation_regression.json"))
if not paths:
    raise SystemExit("rotation_regression.json not found")

path = max(paths, key=lambda p: p.stat().st_mtime)
data = json.loads(path.read_text())

print("path:", path)
print("overall passed:", data["passed"])
print("diagnosis:", data["diagnosis"])

for name, pair in data["comparisons"].items():
    all_tokens = pair["all_tokens"]
    prompt_end = pair["prompt_end"]

    print()
    print(name)
    print(
        "  all-token:",
        "L2=", all_tokens["relative_l2_error"],
        "top1=", all_tokens["top1_agreement"],
        "passed=", all_tokens["passed"],
    )
    print(
        "  prompt-end:",
        "L2=", prompt_end["relative_l2_error"],
        "top1=", prompt_end["top1_agreement"],
        "gating=", prompt_end["gating"],
    )
PY
```

---

# 32. 根据 A/B/C 结果定位

真实运行后按以下规则解释。

---

## 32.1 A↔B 明显差

说明问题与 Rotation 无关或不只与 Rotation 有关。

优先检查：

```text
snn2/model_integration.py
custom eager attention
MLP forward replacement
attention mask/sliding attention
Qwen3 q_norm/k_norm compatibility
```

---

## 32.2 A↔B 好，B↔C 差

说明：

```text
SNN2 identity integration 本身正常
Rotation 引入错误
```

继续按：

```text
R1
R2
R3
R4
Hadamard orientation
precision
```

逐项定位。

---

## 32.3 A↔B、B↔C 都好，但 prompt-end 已出现 mismatch

说明当前：

```text
all-token average metrics
```

虽然通过，但 generation 起点仍存在 decision divergence。

本轮先记录，不改变 hard gate。

不要偷偷把 prompt-end 加入 pass/fail。

---

# 33. 完成修改后必须运行的 tests

先跑定向 tests：

```bash
cd ~/SNN

python -m pytest -q \
  tests/test_hadamard.py \
  tests/test_rotation_regression.py \
  tests/test_rotated_pre_finetuning_protocol.py \
  tests/test_evaluation_paths.py
```

然后完整：

```bash
python -m pytest -q
```

必须全部通过。

---

# 34. 必须增加的测试覆盖清单

Codex 完成后逐项确认。

```text
[ ] Q = D U 显式矩阵测试
[ ] transpose = Q^T 测试
[ ] round-trip 测试
[ ] new DU != old UD 防回退测试

[ ] R1 FP64 explicit matmul
[ ] R2 value-side FP64
[ ] R2 o-side FP32
[ ] R3 preserve input dtype
[ ] R4 FP32

[ ] old rotation_state fail-fast
[ ] new rotation metadata accepted

[ ] A/B/C 三组 comparison 都生成
[ ] first failure 不会 short-circuit
[ ] diagnosis mapping tests

[ ] prompt-end left padding
[ ] prompt-end right padding
[ ] prompt-only 不包含 completion
[ ] prompt-end 使用 evaluation.tldr_input_length
[ ] prompt-end diagnostic 不参与 hard gate

[ ] verify_artifacts 支持 format_version 4
[ ] verify_artifacts 校验 DU / roste_aligned_v1
```

---

# 35. 不允许做的事情

本轮明确禁止：

1. **不要修改 regression thresholds**
   ```text
   relative_l2_threshold = 0.05
   top1_agreement_threshold = 0.95
   ```

2. 不要把 prompt-end 加成新 hard gate。

3. 不要删除现有 all-token regression metrics。

4. 不要只测 A↔C 而删除 B。

5. 不要只测 B↔C 保持旧行为。

6. 不要把所有 Rotation 一律改成 FP64。

7. 不要把所有 Rotation 一律改成 FP32。

8. 不要改变 R1/R2/R3/R4 seed 方案。

9. 不要改变 Prefix discovery 逻辑。

10. 不要改变 Prefix KV cache 逻辑。

11. 不要改变 10 activation replacement sites。

12. 不要改变 ANN training modes。

13. 不要改变 TL;DR test sample selection。

14. 不要改变 greedy generation 逻辑。

15. 不要改变 ROUGE metric 逻辑。

16. 不要把旧 `rotation_state.pt` 静默当成新 `DU` state。

17. 不要在 CUDA FHT 失败时 silent fallback。

---

# 36. 建议的最终代码结构

修改后 Rotation 相关职责建议变成：

```text
snn2/hadamard.py
├── HadamardSpec + orientation metadata
├── structured U
├── Q = D U semantic transform
├── materialize Q
├── FP64 dense transform helpers
└── FP32 FHT transform helpers

snn2/rotation.py
├── regression metric accumulators
│   ├── all-token
│   └── prompt-end
├── regression assessment
├── A/B/C suite
├── diagnosis
├── RMSNorm fusion
├── R1 FP64 offline
├── R2 mixed precision offline
├── R4 FP32 offline
├── state save/load/version validation
└── fused rotation provenance

snn2/model_integration.py
├── R3 preserve dtype online
├── R4 FP32 online
└── identity / SNN site integration

snn2/data.py
├── TL;DR train tokenization
├── shared TL;DR prompt/reference parser
└── TL;DR generation-prompt encoding

scripts/prepare_rotation.py
├── construct A
├── construct B
├── construct C
├── run three-way suite
├── write full diagnosis
├── hard fail only after all comparisons
└── save C only after suite passes
```

---

# 37. 建议的 implementation 顺序

Codex 按下面顺序实施，减少“同时改很多文件后无法定位”的风险。

### Phase 1：Hadamard semantic

```text
1. HadamardSpec orientation
2. random_hadamard: UD → DU
3. dense Q materialization
4. orientation tests
```

先跑：

```bash
python -m pytest -q tests/test_hadamard.py
```

---

### Phase 2：RoSTE precision alignment

```text
1. R1 FP64
2. R2 value FP64
3. R2 o-side FP32
4. R3 preserve dtype
5. R4 FP32
6. precision tests
```

再跑相关 tests。

---

### Phase 3：artifact versioning

```text
rotation state format v2
DU metadata
precision policy metadata
legacy fail-fast
```

---

### Phase 4：TL;DR shared prompt helper

```text
data.py helper
evaluate_tldr.py reuse
prompt-only tests
```

必须确认正式 evaluation 的输出语义没有变化。

---

### Phase 5：A/B/C regression suite

```text
all-token 3-way
prompt-end 3-way
diagnosis
suite exception
```

---

### Phase 6：prepare_rotation orchestration

```text
A original
B integrated
C rotated
write full JSON
hard-fail
save artifacts
```

---

### Phase 7：verifier + docs

```text
verify_artifacts.py
实验执行总结.md
代码结构总结.md
```

---

### Phase 8：all tests + real Qwen3-1.7B prepare_rotation

---

# 38. Definition of Done

只有同时满足以下条件，本轮才算完成。

## 代码

```text
A = real Original HF Base
B = SNN2 identity Base
C = Rotated SNN2 identity

A↔B / B↔C / A↔C 全部实际运行
```

---

## Hadamard

数学语义：

\[
Q = D U
\]

并有显式测试证明，不只是 round-trip。

---

## Precision

实际代码满足：

```text
RMSNorm fusion        FP64
R1 offline            FP64
R2 value-side         FP64
R2 o-side             FP32
R3 online             preserve input dtype
R4 offline            FP32
R4 online             FP32
```

---

## Prompt-end

使用：

```text
同一 calibration manifest 的 128 raw rows
prompt only
TL;DR evaluation input length
last valid prompt position
```

三组 pair 均产生 metrics。

---

## Hard gate

仍为：

```text
relative_l2_error <= 0.05
top1_agreement > 0.95
```

且：

```text
prompt-end 不 gating
```

---

## Artifact safety

旧：

```text
Q = UD
```

rotation state 被明确拒绝。

新 state 记录：

```text
orientation = DU
precision_policy = roste_aligned_v1
```

---

## Regression artifact

`rotation_regression.json`：

```text
format_version = 4
three comparisons
all-token metrics
prompt-end metrics
diagnosis
overall passed
```

---

## Tests

```bash
python -m pytest -q
```

全部通过。

---

## Real model

Qwen3-1.7B：

```bash
python scripts/prepare_rotation.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
```

成功生成新版三路 regression；若失败，也必须生成完整三路结果和 diagnosis，不能只留下半截工件。

---

# 39. Codex 最终汇报格式

修改完成后，Codex 最后只需汇报：

```text
1. 修改了哪些文件
2. Q=DU 如何实现
3. 各 R1/R2/R3/R4 precision 最终是什么
4. A/B/C regression 最终 JSON schema
5. prompt-end 如何构造
6. legacy artifact 如何 fail-fast
7. pytest 结果
8. 真实 Qwen3-1.7B prepare_rotation 是否运行；若运行，三组 A/B/C 的 all-token 与 prompt-end 核心数值
9. 是否存在仍未解决的 blocker
```

不要在完成后再自行改变实验协议。
