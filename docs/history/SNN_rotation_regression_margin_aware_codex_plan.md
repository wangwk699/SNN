# SNN Rotation Regression：增加 Margin-Aware Diagnostic 的完整代码修改规范

> **用途**：本文件用于指导部署在服务器上的 Codex 在**没有任何上下文**的情况下，对 `wangwk699/SNN` 项目完成本轮代码修改。  
> **项目根目录**：默认 `~/SNN`。  
> **当前状态**：Rotation regression 已升级到 `format_version=2`，并记录 `relative_l2_error`、P99/P99.9 absolute error 与 Top-1 agreement。当前 Qwen3-1.7B-Base 在固定 128 个 calibration samples 上得到：
>
> ```text
> relative_l2_error       = 0.007705130877581711
> p99_abs_error           = 0.2509765625
> p999_abs_error          = 0.4384765625
> max_abs_error           = 6.6875
> top1_agreement          = 0.9622168875900219
> top1_agreement_count    = 46095
> top1_disagreement_count = 1810
> num_tokens_compared     = 47905
> passed                  = true
> ```
>
> **本轮目标**：增加 margin-aware regression diagnostic，用理论充分条件 `m_t > 2*delta_t` 判断 token position 的 Top-1 是否具有严格扰动稳定性，并分析 1810 个 Top-1 disagreement 是否集中在 Base 本来就接近决策边界的位置。

---

# 0. 本轮修改原则

本轮只扩展 Rotation regression diagnostics。必须保持以下实验协议完全不变：

```text
Base regression side:
    Original Base
    + SNN2 identity integration
    + no rotation_state
    + no Prefix
    + no activation replacement

Rotated regression side:
    same Base checkpoint
    + fused/offline R1/R2/R4-inverse
    + online R3/R4
    + SNN2 identity integration
    + no Prefix
    + no activation replacement

data:
    fixed existing 128 calibration samples

forward:
    eval()
    torch.inference_mode()
    use_cache=False
```

不要修改 R1/R2/R3/R4 placement、Hadamard sign order、RMSNorm fusion、fast-hadamard-transform backend、R3/R4 FP32 Hadamard、Prefix、Calibration、10 activation sites、ANN training、SNN conversion 或评估协议。

当前 quality hard gate 仍保持：

```text
relative_l2_error <= 0.01
```

本轮不要增加经验性的 Top-1 或 margin-safe-fraction 阈值。

---

# 1. 修改前确认当前代码基线

```bash
cd ~/SNN
git status
git rev-parse HEAD
python -m pytest -q tests/test_rotation_regression.py
```

当前 `snn2/rotation.py` 应已有：

```python
class _StreamingAbsErrorHistogram:
    ...
```

以及 `_LogitsErrorAccumulator` 中的 Top-1 counts 与 8192-bin histogram。当前 `rotation_regression.json` 为 `format_version=2`。

---

# 2. Margin-aware diagnostic 的理论定义

对每一个有效 token position `t`，Base logits 记为 `L_B[t, :]`，Rotated logits 记为 `L_R[t, :]`。

## 2.1 Base Top-1 margin

令 Base 最大与第二大 logit 分别为 `L_B(t,1)` 与 `L_B(t,2)`，定义：

```text
m_t = L_B(t,1) - L_B(t,2)
```

必有 `m_t >= 0`。`m_t` 很小表示 Base top-1/top-2 本来就接近打平。

## 2.2 Per-token 最大 Rotation perturbation

定义：

```text
delta_t = max_v |L_B[t,v] - L_R[t,v]|
```

注意它与 root `max_abs_error` 不同；root `max_abs_error = max_t delta_t`。

## 2.3 Margin-safe sufficient condition

严格使用：

```text
m_t > 2 * delta_t
```

若成立，则该 token 的 argmax 在当前 Base→Rotated 扰动下必然不变。原因是 Base winner 最坏下降 `delta_t`，竞争 token 最坏上升 `delta_t`，margin 最多缩小 `2*delta_t`。

必须用严格 `>`，不能用 `>=`。

定义：

```text
margin-safe:   m_t > 2*delta_t
margin-unsafe: m_t <= 2*delta_t
```

`margin-unsafe` 只表示没有这个充分条件保证，不代表 Top-1 一定改变。

---

# 3. Margin-safe disagreement 是 correctness invariant

若 `m_t > 2*delta_t`，理论上不可能出现 Base 与 Rotated Top-1 不同。因此：

```text
margin_safe_disagreement_count == 0
```

应作为 metric/alignment correctness invariant，而不是模型 quality threshold。

若非 0，应优先排查 metric implementation、mask/token 对齐或 logits 来源不一致。

---

# 4. `rotation_regression.json` 升级到 format_version=3

因为新增 required schema，将：

```text
format_version: 2 -> 3
```

涉及：

```text
snn2/rotation.py
scripts/prepare_rotation.py
scripts/verify_artifacts.py
tests/test_rotation_regression.py
实验执行总结.md
```

旧 `format_version=2` artifact 视为过期，重新运行 `prepare_rotation.py` 生成 v3。

---

# 5. 新增 JSON：`margin_aware_diagnostic`

建议所有新字段放在根层下的单独对象：

```json
"margin_aware_diagnostic": {
  "definition": "base_top1_margin_gt_2x_per_token_max_abs_error",
  "margin_safe_token_count": 0,
  "margin_unsafe_token_count": 0,
  "margin_safe_fraction": 0.0,
  "margin_safe_agreement_count": 0,
  "margin_safe_disagreement_count": 0,
  "margin_unsafe_agreement_count": 0,
  "margin_unsafe_disagreement_count": 0,
  "disagreement_margin_unsafe_fraction": 0.0
}
```

必须满足：

```text
safe + unsafe = num_tokens_compared
safe_agree + safe_disagree = safe
unsafe_agree + unsafe_disagree = unsafe
safe_agree + unsafe_agree = top1_agreement_count
safe_disagree + unsafe_disagree = top1_disagreement_count
```

`margin_safe_fraction = safe / num_tokens_compared`。

若有 disagreement：

```text
disagreement_margin_unsafe_fraction = unsafe_disagree / top1_disagreement_count
```

若 `top1_disagreement_count == 0`，约定为 `1.0`。

---

# 6. 必须增加四组 token-level exact distribution

由于这里只保存约 `num_tokens_compared` 个 scalar，而不是 `num_tokens × vocab`，允许全部存 CPU 并做 exact quantile。不要为这些 token-level 指标再用 histogram。

新增：

```json
"base_top1_margin_all_tokens": {
  "mean": 0.0,
  "p50": 0.0,
  "p90": 0.0,
  "p99": 0.0,
  "max": 0.0
},
"per_token_max_abs_error_all_tokens": {
  "mean": 0.0,
  "p50": 0.0,
  "p90": 0.0,
  "p99": 0.0,
  "max": 0.0
},
"base_top1_margin_disagreement_tokens": {
  "count": 0,
  "mean": 0.0,
  "p50": 0.0,
  "p90": 0.0,
  "p99": 0.0,
  "max": 0.0
},
"per_token_max_abs_error_disagreement_tokens": {
  "count": 0,
  "mean": 0.0,
  "p50": 0.0,
  "p90": 0.0,
  "p99": 0.0,
  "max": 0.0
}
```

若 disagreement 为 0，disagreement distribution 使用 `count=0` 且统计字段为 JSON `null`，禁止 NaN/Infinity。

---

# 7. 建议增加 disagreement stability ratio

定义：

```text
r_t = 2*delta_t / (m_t + 1e-12)
```

只对 disagreement tokens 记录：

```json
"stability_ratio_disagreement_tokens": {
  "definition": "2_delta_over_base_top1_margin_plus_1e-12",
  "count": 0,
  "mean": 0.0,
  "p50": 0.0,
  "p90": 0.0,
  "p99": 0.0
}
```

解释：

```text
r_t < 1  -> margin-safe
r_t = 1  -> sufficient-condition boundary
r_t > 1  -> margin-unsafe
```

用 FP64 计算 ratio，并检查 finite。

---

# 8. `_LogitsErrorAccumulator.__init__()` 新增 state

保留当前 state，新增：

```python
self.margin_safe_token_count = 0
self.margin_unsafe_token_count = 0
self.margin_safe_agreement_count = 0
self.margin_safe_disagreement_count = 0
self.margin_unsafe_agreement_count = 0
self.margin_unsafe_disagreement_count = 0

self.base_top1_margin_chunks: list[torch.Tensor] = []
self.per_token_max_abs_error_chunks: list[torch.Tensor] = []
self.disagreement_base_top1_margin_chunks: list[torch.Tensor] = []
self.disagreement_per_token_max_abs_error_chunks: list[torch.Tensor] = []
self.disagreement_stability_ratio_chunks: list[torch.Tensor] = []
```

这些 list 只能保存 `O(num_tokens)` CPU scalar tensors，不得保存 vocab-level errors。

---

# 9. 在现有 `update()` chunk loop 中直接计算 margin/delta

当前每个 chunk 已有：

```python
base      # [n_valid_tokens, vocab]
rotated   # [n_valid_tokens, vocab]
difference = base - rotated
absolute_error = difference.abs()
```

直接复用，不重新 forward，不保存完整 logits。

首先检查：

```python
if vocab_size < 2:
    raise RuntimeError("Margin-aware regression requires vocabulary size >= 2")
```

Base top-2：

```python
top2 = torch.topk(
    base,
    k=2,
    dim=-1,
    largest=True,
    sorted=True,
)

base_top1_margin = (
    top2.values[:, 0]
    - top2.values[:, 1]
)
```

检查 margin finite 且 nonnegative。

每-token delta：

```python
per_token_max_abs_error = absolute_error.amax(dim=-1)
```

检查 finite 且 nonnegative。

---

# 10. Top-1 disagreement mask 必须复用当前 exact Top-1 indexing

当前 batch 级已有：

```python
mask_cpu = attention_mask.to(device="cpu", dtype=torch.bool)
base_top1 = base_logits.argmax(dim=-1)
rotated_top1 = rotated_logits.argmax(dim=-1).detach().to("cpu")
```

不要建立另一套可能错位的 token indexing。

每个 chunk：

```python
chunk_mask_cpu = mask_cpu[:, start:stop]

base_top1_valid = (
    base_top1[:, start:stop][chunk_mask_cpu]
)

rotated_top1_valid = (
    rotated_top1[:, start:stop][chunk_mask_cpu]
)

disagreement_mask_cpu = (
    base_top1_valid != rotated_top1_valid
)

disagreement_mask = disagreement_mask_cpu.to(device=device)
```

检查：

```python
disagreement_mask.numel() == base_top1_margin.numel()
disagreement_mask.numel() == per_token_max_abs_error.numel()
```

---

# 11. safe/unsafe 计数

```python
margin_safe = (
    base_top1_margin
    > 2.0 * per_token_max_abs_error
)
margin_unsafe = ~margin_safe
agreement_mask = ~disagreement_mask
```

累计：

```python
self.margin_safe_token_count += int(margin_safe.sum().item())
self.margin_unsafe_token_count += int(margin_unsafe.sum().item())

self.margin_safe_agreement_count += int(
    (margin_safe & agreement_mask).sum().item()
)
self.margin_safe_disagreement_count += int(
    (margin_safe & disagreement_mask).sum().item()
)
self.margin_unsafe_agreement_count += int(
    (margin_unsafe & agreement_mask).sum().item()
)
self.margin_unsafe_disagreement_count += int(
    (margin_unsafe & disagreement_mask).sum().item()
)
```

---

# 12. 保存 token-level scalar

所有有效 token：

```python
self.base_top1_margin_chunks.append(
    base_top1_margin.detach().float().cpu()
)
self.per_token_max_abs_error_chunks.append(
    per_token_max_abs_error.detach().float().cpu()
)
```

只对 disagreement：

```python
if bool(disagreement_mask.any()):
    disagreement_margin = base_top1_margin[disagreement_mask]
    disagreement_delta = per_token_max_abs_error[disagreement_mask]

    disagreement_ratio = (
        2.0 * disagreement_delta.double()
        / (disagreement_margin.double() + 1e-12)
    )

    if not bool(torch.isfinite(disagreement_ratio).all()):
        raise RuntimeError("Non-finite margin stability ratio")

    self.disagreement_base_top1_margin_chunks.append(
        disagreement_margin.detach().float().cpu()
    )
    self.disagreement_per_token_max_abs_error_chunks.append(
        disagreement_delta.detach().float().cpu()
    )
    self.disagreement_stability_ratio_chunks.append(
        disagreement_ratio.detach().cpu()
    )
```

---

# 13. 新增 exact distribution helper

在 `snn2/rotation.py` 增加内部 helper，例如：

```python
def _exact_scalar_distribution(
    values: torch.Tensor,
    *,
    include_max: bool = True,
) -> dict[str, float]:
    values = values.detach().double().cpu().flatten()
    if values.numel() == 0:
        raise ValueError("Expected non-empty values")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("Non-finite scalar diagnostic")

    result = {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50, interpolation="linear").item()),
        "p90": float(torch.quantile(values, 0.90, interpolation="linear").item()),
        "p99": float(torch.quantile(values, 0.99, interpolation="linear").item()),
    }
    if include_max:
        result["max"] = float(values.max().item())
    return result
```

再增加 optional helper，用于 disagreement=0 时返回 `null` fields。

---

# 14. `metrics()` consistency invariants

在返回前必须检查：

```python
safe + unsafe == self.num_tokens
safe_agree + safe_disagree == safe
unsafe_agree + unsafe_disagree == unsafe
safe_agree + unsafe_agree == self.top1_agreement_count
safe_disagree + unsafe_disagree == self.top1_disagreement_count
```

并强制：

```python
if self.margin_safe_disagreement_count != 0:
    raise RuntimeError(
        "Margin-safe token changed Top-1 despite m_t > 2*delta_t; "
        "metric alignment is inconsistent"
    )
```

这不是新 quality threshold，而是数学 correctness invariant。

拼接：

```python
all_margin = torch.cat(self.base_top1_margin_chunks)
all_delta = torch.cat(self.per_token_max_abs_error_chunks)
```

检查：

```python
all_margin.numel() == self.num_tokens
all_delta.numel() == self.num_tokens
```

若有 disagreement，再拼接 disagreement margin/delta/ratio，并检查 numel 全部等于 `top1_disagreement_count`。

---

# 15. `metrics()` 目标输出

在当前 metrics 根层继续保留所有 v2 字段，同时增加：

```python
"margin_aware_diagnostic": {
    "definition": "base_top1_margin_gt_2x_per_token_max_abs_error",
    "margin_safe_token_count": self.margin_safe_token_count,
    "margin_unsafe_token_count": self.margin_unsafe_token_count,
    "margin_safe_fraction": self.margin_safe_token_count / self.num_tokens,
    "margin_safe_agreement_count": self.margin_safe_agreement_count,
    "margin_safe_disagreement_count": self.margin_safe_disagreement_count,
    "margin_unsafe_agreement_count": self.margin_unsafe_agreement_count,
    "margin_unsafe_disagreement_count": self.margin_unsafe_disagreement_count,
    "disagreement_margin_unsafe_fraction": (
        1.0
        if self.top1_disagreement_count == 0
        else self.margin_unsafe_disagreement_count / self.top1_disagreement_count
    ),
    "base_top1_margin_all_tokens": ...,
    "per_token_max_abs_error_all_tokens": ...,
    "base_top1_margin_disagreement_tokens": ...,
    "per_token_max_abs_error_disagreement_tokens": ...,
    "stability_ratio_disagreement_tokens": ...,
}
```

---

# 16. `scripts/prepare_rotation.py`

仅将 in-progress JSON：

```python
"format_version": 2
```

改为：

```python
"format_version": 3
```

保留当前双-controller 结构与 `rotated_controller` 参数，不要改回旧代码。

---

# 17. `validate_rotation_logits()`

最终 result：

```python
result = {
    "format_version": 3,
    ...,
    **accumulator.metrics(),
}
```

其余 128 calibration、`prefix_ids=None`、`shuffle=False`、`use_cache=False`、`eval()`、`torch.inference_mode()` 全部保持。

---

# 18. `scripts/verify_artifacts.py` 必须扩展

Rotation regression manifest 要求 `format_version == 3`，并把 `margin_aware_diagnostic` 加入 required fields。

读取 counts 后验证：

```python
safe + unsafe == num_tokens_compared
safe_agree + safe_disagree == safe
unsafe_agree + unsafe_disagree == unsafe
safe_agree + unsafe_agree == top1_agreement_count
safe_disagree + unsafe_disagree == top1_disagreement_count
```

强制：

```python
safe_disagree == 0
```

验证：

```python
margin_safe_fraction == safe / num_tokens_compared
```

容差 `1e-12`。

若有 disagreement：

```python
disagreement_margin_unsafe_fraction == unsafe_disagree / top1_disagreement_count
```

否则必须等于 `1.0`。

---

# 19. verify distribution schema

建议增加 helper：

```python
def _verify_scalar_distribution(...):
    ...
```

非空分布检查：

```text
all finite
all nonnegative
p50 <= p90 <= p99 <= max
```

允许 `1e-12` 浮点容差。

All-token 两个分布必须非空。

额外要求：

```python
margin_diag[
    "per_token_max_abs_error_all_tokens"
]["max"]
```

与 root：

```python
regression["max_abs_error"]
```

在 `1e-6` 内一致。

若 disagreement > 0，三个 disagreement distribution 的 `count` 必须都等于 `top1_disagreement_count`；若 disagreement=0，要求 count=0 且统计值为 `null`。

---

# 20. Tests：`tests/test_rotation_regression.py`

必须新增以下测试，禁止加载真实大模型。

## Test A：margin-safe + agreement

```python
base = torch.tensor([[[10.0, 0.0, -1.0]]])
rotated = torch.tensor([[[9.5, 0.5, -1.0]]])
mask = torch.tensor([[1]])
```

此时 margin=10，delta=0.5，`10 > 1`，应：

```text
safe=1
unsafe=0
safe_agree=1
safe_disagree=0
top1_agreement=1
```

## Test B：margin-unsafe + disagreement

```python
base = torch.tensor([[[10.0, 9.9, 0.0]]])
rotated = torch.tensor([[[9.9, 10.0, 0.0]]])
mask = torch.tensor([[1]])
```

应：

```text
unsafe=1
unsafe_disagree=1
safe_disagree=0
disagreement_margin_unsafe_fraction=1.0
```

## Test C：margin-unsafe 但仍 agreement

构造 `m_t <= 2*delta_t` 但 argmax 不变，证明 unsafe 只表示“无保证”，不是“必然翻转”。

## Test D：padding 不影响 margin-aware metrics

第二个 padding token 刻意制造巨大 disagreement/delta，最终 counts/distributions 只能反映第一个 valid token。

## Test E：token delta max 与 root max 一致

```python
metrics["margin_aware_diagnostic"]["per_token_max_abs_error_all_tokens"]["max"]
== pytest.approx(metrics["max_abs_error"])
```

## Test F：disagreement distribution count

构造多 token，确保 disagreement margin/delta/ratio 的 count 都等于 `top1_disagreement_count`。

## Test G：zero-disagreement JSON

检查 disagreement distribution `count=0` 且统计字段 `None`，`json.dumps(metrics)` 通过。

## Test H：quantile monotonicity

检查 all-token 和 disagreement distributions 的 `p50 <= p90 <= p99`，有 max 的再检查 `p99 <= max`。

## Test I：Top-1 partition consistency

检查 safe/unsafe agreement/disagreement counts 与现有 Top-1 counts 完全对齐，并 `safe_disagree == 0`。

---

# 21. `实验执行总结.md`

同步加入以下定义：

```text
m_t = Base top1 logit - Base top2 logit

delta_t = max_v |Base_logit(t,v) - Rotated_logit(t,v)|

若 m_t > 2 delta_t，
则该 token 的 Top-1 在当前观测扰动下具有严格稳定性保证。
```

明确：

```text
margin-safe   -> 有充分条件保证 argmax 不变
margin-unsafe -> 只是没有该充分条件保证，不代表一定改变
```

并明确当前 `passed=true` 仍由 `relative_l2_error <= 0.01` 决定；margin-aware 目前用于解释 Top-1 disagreement 和检查 metric 对齐。

---

# 22. 不新增 config threshold

本轮不修改 `configs/experiment_matrix.yaml`，不新增：

```yaml
regression_top1_agreement_threshold:
regression_margin_safe_fraction_threshold:
```

旧 generated config 无需因 config key 改变而 materialize；但旧 `rotation_regression.json` 必须重新生成，因为 schema 升级到了 v3。

---

# 23. 修改完成后的测试

```bash
cd ~/SNN
python -m compileall -q snn2 scripts tests
python -m pytest -q tests/test_rotation_regression.py
python -m pytest -q
```

全部必须通过。

---

# 24. 真实 Qwen3-1.7B regression 重跑

```bash
cd ~/SNN
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
python scripts/prepare_rotation.py --config "$ROT_CFG"
```

必须生成 `format_version=3` 的 `rotation_regression.json`。

---

# 25. 重跑后重点报告

必须报告：

```text
relative_l2_error
top1_agreement
top1_disagreement_count

margin_safe_token_count
margin_unsafe_token_count
margin_safe_fraction

margin_safe_agreement_count
margin_safe_disagreement_count
margin_unsafe_agreement_count
margin_unsafe_disagreement_count

disagreement_margin_unsafe_fraction

base_top1_margin_disagreement_tokens:
    mean / p50 / p90 / p99 / max

per_token_max_abs_error_disagreement_tokens:
    mean / p50 / p90 / p99 / max

stability_ratio_disagreement_tokens:
    mean / p50 / p90 / p99
```

---

# 26. 如何解释结果

优先看：

```text
margin_safe_disagreement_count
```

理论上必须为 0；若非 0，停止后续实验并排查 metric alignment。

如果：

```text
margin_safe_disagreement_count = 0
disagreement_margin_unsafe_fraction = 1.0
```

再观察 disagreement Base margin 的 p50/p90/p99。如果普遍很小，可支持“整体 Rotation 数值误差较小，但 BF16/FHT/fused-weight 数值扰动会在原本接近决策边界的位置改变 argmax”的解释。

如果 disagreement margin 并不小且 delta 也很大，则下一步才考虑 RMSNorm-fusion-only regression、R1/R2/R3/R4 ablation、FP32 diagnostic 或 layerwise divergence。**本轮不要提前实现这些实验。**

---

# 27. 性能与显存要求

禁止保存：

```text
全部 logits
全部 vocab-level absolute differences
```

允许保存每个有效 token 一个 margin、一个 delta，以及 disagreement token 一个 ratio。对于当前 47,905 token，即使 FP64，每个 vector 也只有约 0.38 MB。

现有 vocab-level P99/P99.9 的 streaming histogram 保持不变。

---

# 28. Codex 最终交付要求

Codex 完成后必须报告：

1. 修改文件列表；
2. `rotation_regression.json` 是否升级到 v3；
3. `m_t` 与 `delta_t` 精确定义；
4. 是否严格使用 `m_t > 2*delta_t`；
5. padding 是否完全排除；
6. margin/delta 是否来自与现有 regression 相同 logits；
7. 是否没有保存 vocab-level 全量 errors；
8. token-level quantile 是否 exact；
9. `margin_safe_disagreement_count==0` 是否作为 correctness invariant；
10. quality hard gate 是否仍只有 `relative_l2_error<=0.01`；
11. `verify_artifacts.py` 新增了哪些 consistency check；
12. 单元测试新增了哪些场景；
13. `pytest -q` 结果；
14. 若真实重跑，报告第 25 节全部指标；
15. `git status`。

不要只回复“修改完成”。

---

# 29. 最终 checklist

```text
[ ] format_version = 3

[ ] m_t = Base top1 - Base top2
[ ] delta_t = max_v |Base - Rotated|
[ ] strict safe condition: m_t > 2 delta_t

[ ] safe/unsafe token counts added
[ ] safe/unsafe agreement counts added
[ ] safe/unsafe disagreement counts added
[ ] margin_safe_disagreement_count == 0 invariant

[ ] margin_safe_fraction added
[ ] disagreement_margin_unsafe_fraction added

[ ] all-token Base margin exact distribution added
[ ] all-token per-token-delta exact distribution added
[ ] disagreement Base margin exact distribution added
[ ] disagreement delta exact distribution added
[ ] disagreement stability ratio distribution added

[ ] token-level exact quantiles use all valid tokens
[ ] no token-level histogram approximation
[ ] existing vocab-level P99/P99.9 histogram unchanged

[ ] padding excluded everywhere
[ ] counts reconcile with existing Top-1 counts
[ ] token delta max matches root max_abs_error

[ ] prepare_rotation schema updated
[ ] verify_artifacts updated
[ ] tests updated
[ ] 实验执行总结.md updated

[ ] relative-L2 1% quality gate unchanged
[ ] no arbitrary Top-1 threshold added
[ ] no arbitrary margin-safe-fraction threshold added

[ ] all tests pass
[ ] real Qwen3-1.7B regression regenerates cleanly
```
