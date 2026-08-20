# SNN Rotation Regression：增加 Top-1 Agreement 与 P99 / P99.9 Absolute Error 的修改规范

> **用途**：本文件用于指导部署在服务器上的 Codex 在**没有任何上下文**的情况下，对 `wangwk699/SNN` 项目完成本轮代码修改。  
> **项目根目录**：默认 `~/SNN`。  
> **当前协议背景**：`scripts/prepare_rotation.py` 已经在固定的 128 个 calibration samples 上执行 Base ↔ Rotated logits regression，并以 `relative_l2_error <= 0.01` 作为当前 hard gate。  
> **本轮目标**：在不改变现有 Rotation、Prefix、Calibration、10-site、训练/评估协议的前提下，为 Rotation regression 增加：
>
> 1. **Top-1 agreement**
> 2. **P99 absolute error**
> 3. **P99.9 absolute error**
>
> 并同步更新 regression JSON、tests、artifact verification 与 `实验执行总结.md`。

---

# 0. 本轮修改原则

本轮只修改 Rotation regression 的**诊断指标与验证工件**。

必须保持以下现有逻辑不变：

```text
Base regression side:
    Original Base
    + SNN2 identity integration
    + no rotation_state
    + no Prefix
    + no activation replacement

Rotated regression side:
    same Original Base revision
    + fused/offline R1/R2/R4-inverse
    + online R3/R4
    + SNN2 identity integration
    + no Prefix
    + no activation replacement

data:
    existing fixed 128 calibration samples

forward:
    eval()
    torch.inference_mode()
    use_cache=False
```

不要恢复旧的 “native HF Base vs SNN2-integrated Rotated” 比较方式。

不要改变：

```yaml
rotation:
  regression_relative_l2_threshold: 0.01
```

本轮**不要擅自为 top1_agreement 拍脑袋设置 hard threshold**。

原因是当前只已经得到 Qwen3-1.7B 的自然误差；应先让 Qwen3-1.7B、Qwen3-8B、Llama3-8B 都记录 `top1_agreement` 后，再决定是否把它升级为第二个 hard gate。

因此本轮：

```text
relative_l2_error:
    hard gate

top1_agreement:
    required diagnostic

p99_abs_error:
    required diagnostic

p999_abs_error:
    required diagnostic
```

---

# 1. 修改前检查

开始前：

```bash
cd ~/SNN

git status
git rev-parse HEAD

python -m pytest -q tests/test_rotation_regression.py
```

确认当前 `scripts/prepare_rotation.py` 已经是双 controller 结构：

```python
base_controller = SiteController(mode="identity")
install_model_integration(
    base_model,
    base_controller,
    None,
)

rotated_controller = SiteController(mode="identity")
install_model_integration(
    model,
    rotated_controller,
    state,
)
```

并且：

```python
validate_rotation_logits(
    ...,
    rotated_controller,
    ...
)
```

不要把这里重新改回单一 `controller`。

---

# 2. 当前 Rotation regression 的基线定义

当前 `snn2/rotation.py` 中 `_LogitsErrorAccumulator` 已经累计：

```text
num_samples
num_tokens_compared
num_elements
max_abs_error
sum_abs_error
sum_squared_error
sum_squared_base
sum_squared_rotated
```

并输出：

```text
max_abs_error
mean_abs_error
relative_l2_error
base_logits_l2
rotated_logits_l2
```

当前 hard gate：

\[
\text{relative\_l2\_error}
=
\frac{
\sqrt{\sum_i (\ell^B_i-\ell^R_i)^2}
}{
\sqrt{\sum_i (\ell^B_i)^2}+10^{-12}
}
\]

且：

```python
passed = relative_l2_error <= 0.01
```

其中 \(i\) 遍历所有：

```text
128 calibration samples
× valid tokens
× vocabulary logits
```

padding token 不参与。

本轮保留这个定义和 hard gate。

---

# 3. 新指标 1：Top-1 Agreement

## 3.1 数学定义

对每一个有效 token position \(t\)：

\[
a_t^B = \arg\max_v \ell^B_{t,v}
\]

\[
a_t^R = \arg\max_v \ell^R_{t,v}
\]

Top-1 agreement 定义：

\[
\text{top1\_agreement}
=
\frac{
\sum_t \mathbf{1}[a_t^B=a_t^R]
}{
N_{\text{valid tokens}}
}
\]

只统计 `attention_mask == 1` 的 token position。

---

## 3.2 必须输出的字段

在 `rotation_regression.json` 增加：

```json
{
  "top1_agreement": 0.9995,
  "top1_agreement_count": 47881,
  "top1_disagreement_count": 24
}
```

其中：

```text
top1_agreement_count
+
top1_disagreement_count
=
num_tokens_compared
```

`top1_agreement` 必须满足：

```text
0.0 <= top1_agreement <= 1.0
```

---

## 3.3 实现要求

在 `_LogitsErrorAccumulator.__init__()` 增加：

```python
self.top1_agreement_count = 0
self.top1_disagreement_count = 0
```

在 `update()` 中对整个 batch 计算：

```python
base_top1 = base_logits.argmax(dim=-1)
rotated_top1 = rotated_logits.argmax(dim=-1)
```

注意当前：

```text
base_logits:
    CPU

rotated_logits:
    GPU
```

所以不要为了 top1 把完整 rotated logits 搬 CPU。

推荐：

```python
base_top1 = base_logits.argmax(dim=-1)

rotated_top1 = (
    rotated_logits.argmax(dim=-1)
    .detach()
    .to("cpu")
)

mask_cpu = attention_mask.to(
    device="cpu",
    dtype=torch.bool,
)

matches = (
    base_top1[mask_cpu]
    ==
    rotated_top1[mask_cpu]
)

agreement = int(matches.sum().item())
total = int(matches.numel())

self.top1_agreement_count += agreement
self.top1_disagreement_count += total - agreement
```

必须确认：

```python
total == valid_tokens
```

否则抛异常。

不要计算：

```text
labels 上的 top1
只看最后一个 token
只看 completion token
```

这里检查的是模型函数等价性，所以使用 regression 中的**所有有效 token positions**。

---

# 4. 新指标 2 / 3：P99 与 P99.9 Absolute Error

## 4.1 定义

首先定义每一个逐元素 logit absolute error：

\[
d_i
=
|\ell_i^B-\ell_i^R|
\]

其中 \(i\) 遍历：

```text
所有有效 token
× vocabulary dimension
```

要统计：

\[
P99(d)
\]

以及：

\[
P99.9(d)
\]

JSON 字段：

```text
p99_abs_error
p999_abs_error
```

这里 `p999` 表示 percentile `99.9%`，不是 999%。

---

# 5. 重要：不能保存所有逐元素误差

以 Qwen3 为例：

```text
~47,905 valid tokens
× ~150k vocab
```

逐元素 logit difference 可达到数十亿个。

因此禁止以下实现：

```python
all_errors.append(difference.abs().cpu())
...
torch.cat(all_errors)
torch.quantile(...)
```

这会占用巨量内存，8B 模型上更危险。

也不要把每个 chunk 的 P99 再平均：

```python
mean(chunk_p99)
```

因为：

```text
平均 chunk percentile
≠
全局 percentile
```

---

# 6. P99/P99.9 推荐实现：Streaming Linear Histogram

为了：

```text
确定性
低内存
不保存全部 logits difference
覆盖所有逐元素 logits
```

本项目采用 **streaming linear histogram** 近似全局 percentile。

必须在 JSON 中明确记录 percentile 是 histogram approximation，而不是假装 exact quantile。

---

## 6.1 新增 helper

推荐在 `snn2/rotation.py` 中新增内部类：

```python
class _StreamingAbsErrorHistogram:
    ...
```

默认：

```python
num_bins = 8192
initial_max = 1.0
```

内部状态：

```python
self.num_bins
self.range_max
self.counts
self.total_count
```

其中：

```python
self.counts = torch.zeros(
    num_bins,
    dtype=torch.int64,
    device="cpu",
)
```

---

## 6.2 动态 range 扩张

由于运行前不知道 `max_abs_error`，histogram 范围不能写死。

初始：

```text
[0, 1.0]
```

如果当前 chunk：

```python
local_max > self.range_max
```

则不断：

```python
self.range_max *= 2.0
```

直到：

```python
local_max <= self.range_max
```

每次 range 从 \(R\) 扩张到 \(2R\) 时，旧 histogram 必须正确 rebin。

因为：

```text
old bin width = R / B
new bin width = 2R / B
              = 2 × old bin width
```

所以两个相邻旧 bin 可以精确合并到一个新 bin。

实现：

```python
old = self.counts

merged = old.reshape(
    self.num_bins // 2,
    2,
).sum(dim=1)

new_counts = torch.zeros_like(old)
new_counts[: self.num_bins // 2] = merged

self.counts = new_counts
self.range_max *= 2.0
```

要求：

```python
num_bins
```

必须为偶数；推荐直接固定 `8192`。

如果一次扩张仍不够，则循环执行。

---

## 6.3 当前 chunk histogram

在 `_LogitsErrorAccumulator.update()` 当前已经得到：

```python
difference = base - rotated
```

随后：

```python
abs_difference = difference.abs()
```

将：

```python
abs_difference
```

传入 histogram。

推荐：

```python
hist = torch.histc(
    abs_difference.float(),
    bins=self.num_bins,
    min=0.0,
    max=self.range_max,
)
```

然后：

```python
self.counts += (
    hist
    .to(dtype=torch.int64)
    .cpu()
)

self.total_count += abs_difference.numel()
```

在调用 `torch.histc()` 前必须先确保：

```python
abs_difference.max() <= self.range_max
```

即先执行动态扩张。

---

## 6.4 Histogram consistency

每个完整 regression 结束后必须满足：

```python
histogram.total_count
==
self.num_elements
```

如果不一致，抛：

```python
RuntimeError
```

原因是 percentile 必须覆盖与 relative L2 完全相同的逐元素 logits 集合。

---

# 7. 从 histogram 估计 percentile

新增：

```python
def percentile(self, q: float) -> float:
    ...
```

要求：

```text
q=0.99
q=0.999
```

算法：

```python
target = math.ceil(q * self.total_count)

cumulative = torch.cumsum(
    self.counts,
    dim=0,
)

index = int(
    torch.searchsorted(
        cumulative,
        torch.tensor(target, dtype=cumulative.dtype),
    ).item()
)
```

然后返回该 bin 的**上边界**：

```python
bin_width = self.range_max / self.num_bins

value = (index + 1) * bin_width
```

使用上边界而不是 bin 中心，是一个保守估计。

必须 clamp：

```python
index = min(index, self.num_bins - 1)
```

---

# 8. Histogram provenance

`rotation_regression.json` 增加：

```json
{
  "absolute_error_percentile_estimator": {
    "method": "streaming_linear_histogram",
    "num_bins": 8192,
    "final_range_max": 8.0,
    "reported_value": "bin_upper_edge",
    "exact": false
  }
}
```

具体 `final_range_max` 根据实际 run 自动生成。

必须明确：

```text
p99_abs_error / p999_abs_error
是对全部逐元素 logits absolute error 的 histogram approximation
```

而不是抽样 percentile。

---

# 9. `_LogitsErrorAccumulator` 的完整目标状态

`__init__()` 至少包含：

```python
class _LogitsErrorAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_tokens = 0
        self.num_elements = 0

        self.max_abs_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.sum_squared_base = 0.0
        self.sum_squared_rotated = 0.0

        self.top1_agreement_count = 0
        self.top1_disagreement_count = 0

        self.abs_error_histogram = _StreamingAbsErrorHistogram(
            num_bins=8192,
            initial_max=1.0,
        )
```

---

# 10. `update()` 的目标逻辑

保留现有：

```text
shape validation
mask validation
chunk_tokens
num_samples
num_tokens
num_elements
relative-L2 accumulators
```

新增：

```text
top1 exact agreement
streaming histogram
```

推荐结构：

```python
def update(...):
    ...

    mask_cpu = attention_mask.to(
        device="cpu",
        dtype=torch.bool,
    )

    base_top1 = base_logits.argmax(dim=-1)
    rotated_top1 = (
        rotated_logits
        .argmax(dim=-1)
        .detach()
        .to("cpu")
    )

    top1_matches = (
        base_top1[mask_cpu]
        ==
        rotated_top1[mask_cpu]
    )

    agreement = int(top1_matches.sum().item())
    top1_total = int(top1_matches.numel())

    if top1_total != valid_tokens:
        raise RuntimeError(...)

    self.top1_agreement_count += agreement
    self.top1_disagreement_count += (
        top1_total - agreement
    )

    for start in ...:
        ...
        difference = base - rotated
        abs_difference = difference.abs()

        self.max_abs_error = max(
            self.max_abs_error,
            float(abs_difference.max().item()),
        )

        self.sum_abs_error += ...
        self.sum_squared_error += ...
        self.sum_squared_base += ...
        self.sum_squared_rotated += ...

        self.abs_error_histogram.update(
            abs_difference
        )
```

避免重复：

```python
difference.abs()
```

多次构造临时大 tensor；尽量复用 `abs_difference`。

---

# 11. `metrics()` 的目标输出

在现有结果中增加：

```python
if (
    self.abs_error_histogram.total_count
    != self.num_elements
):
    raise RuntimeError(...)

top1_total = (
    self.top1_agreement_count
    +
    self.top1_disagreement_count
)

if top1_total != self.num_tokens:
    raise RuntimeError(...)

top1_agreement = (
    self.top1_agreement_count
    / top1_total
)
```

返回：

```python
return {
    "num_samples": self.num_samples,
    "num_tokens_compared": self.num_tokens,

    "max_abs_error": self.max_abs_error,
    "mean_abs_error": (
        self.sum_abs_error
        / self.num_elements
    ),

    "p99_abs_error": (
        self.abs_error_histogram.percentile(0.99)
    ),

    "p999_abs_error": (
        self.abs_error_histogram.percentile(0.999)
    ),

    "top1_agreement": top1_agreement,
    "top1_agreement_count": (
        self.top1_agreement_count
    ),
    "top1_disagreement_count": (
        self.top1_disagreement_count
    ),

    "relative_l2_error": (
        self.sum_squared_error**0.5
        /
        (
            self.sum_squared_base**0.5
            + 1e-12
        )
    ),

    "base_logits_l2": (
        self.sum_squared_base**0.5
    ),

    "rotated_logits_l2": (
        self.sum_squared_rotated**0.5
    ),

    "absolute_error_percentile_estimator": {
        "method": "streaming_linear_histogram",
        "num_bins": (
            self.abs_error_histogram.num_bins
        ),
        "final_range_max": (
            self.abs_error_histogram.range_max
        ),
        "reported_value": "bin_upper_edge",
        "exact": False,
    },
}
```

---

# 12. Pass/fail 规则本轮保持不变

当前：

```python
def enforce_rotation_regression(
    result,
    relative_l2_threshold,
):
```

本轮继续：

```python
passed = (
    relative_l2_error
    <=
    regression_relative_l2_threshold
)
```

不要因为新增：

```text
top1_agreement
p99_abs_error
p999_abs_error
```

就擅自增加未知阈值。

也就是说：

```json
"threshold": {
  "relative_l2_error": 0.01
}
```

暂时保持。

---

# 13. 为什么本轮不立即把 Top-1 设为 hard gate

当前已经得到 Qwen3-1.7B：

```text
relative_l2_error ≈ 0.007705
```

但还不知道：

```text
Qwen3-1.7B top1_agreement
Qwen3-8B top1_agreement
Llama3-8B top1_agreement
```

因此不应现在随便设：

```text
99%
99.9%
99.99%
```

任何一个数字。

正确实验流程：

```text
先在三个 model-task pair 上收集自然的
Base ↔ Rotated regression metrics

        ↓

比较：
relative_l2_error
top1_agreement
p99_abs_error
p999_abs_error

        ↓

再决定是否增加统一的 top1 hard gate
```

本轮只确保 metrics 被可靠记录并进入 artifact verification。

---

# 14. `rotation_regression.json` schema

建议将：

```text
format_version
```

从：

```text
1
```

升级为：

```text
2
```

因为 regression JSON schema 已经增加新的 required metrics。

完整结构示例：

```json
{
  "format_version": 2,
  "purpose": "base_vs_rotated_logits_regression",

  "model_name": "Qwen/Qwen3-1.7B-Base",
  "rotation_seed": 42,
  "dtype": "bfloat16",

  "calibration_manifest_path": ".../calibration_manifest.json",
  "calibration_manifest_sha256": "...",

  "num_samples": 128,
  "num_tokens_compared": 47905,

  "max_abs_error": 6.6875,
  "mean_abs_error": 0.0634,

  "p99_abs_error": 0.0,
  "p999_abs_error": 0.0,

  "top1_agreement": 0.0,
  "top1_agreement_count": 0,
  "top1_disagreement_count": 0,

  "relative_l2_error": 0.007705,

  "base_logits_l2": 978731.7,
  "rotated_logits_l2": 978510.2,

  "absolute_error_percentile_estimator": {
    "method": "streaming_linear_histogram",
    "num_bins": 8192,
    "final_range_max": 8.0,
    "reported_value": "bin_upper_edge",
    "exact": false
  },

  "threshold": {
    "relative_l2_error": 0.01
  },

  "passed": true
}
```

示例中的：

```text
p99
p999
top1
```

数值只是 schema placeholder，不得写死。

---

# 15. `scripts/prepare_rotation.py`

当前主要流程已经正确。

本轮不要重新设计 regression。

只需要：

1. 将初始 `rotation_regression.json`：
   ```python
   "format_version": 1
   ```
   改成：
   ```python
   "format_version": 2
   ```

2. `validate_rotation_logits()` 成功后自然写入新的 metrics。

3. 保留：
   ```python
   base_controller = SiteController(mode="identity")
   install_model_integration(
       base_model,
       base_controller,
       None,
   )

   rotated_controller = SiteController(mode="identity")
   install_model_integration(
       model,
       rotated_controller,
       state,
   )
   ```

4. 保留：
   ```python
   validate_rotation_logits(
       ...,
       rotated_controller,
       ...
   )
   ```

不要再出现旧的未定义：

```python
controller
```

---

# 16. `scripts/verify_artifacts.py`

当前 Rotation regression verification 已检查：

```text
purpose
num_samples == 128
passed == true
calibration manifest path
calibration manifest SHA256
```

本轮必须扩展。

---

## 16.1 Required flags

Rotation regression 至少要求：

```python
{
    "format_version": 2,
    "purpose": (
        "base_vs_rotated_logits_regression"
    ),
    "num_samples": 128,
    "passed": True,
}
```

---

## 16.2 Required metric fields

检查以下字段存在：

```text
num_tokens_compared

relative_l2_error
max_abs_error
mean_abs_error

p99_abs_error
p999_abs_error

top1_agreement
top1_agreement_count
top1_disagreement_count

absolute_error_percentile_estimator
```

缺失则：

```python
raise ValueError(...)
```

---

## 16.3 数值一致性检查

必须验证：

```python
0.0 <= top1_agreement <= 1.0
```

必须：

```python
top1_agreement_count >= 0
top1_disagreement_count >= 0
```

且：

```python
top1_agreement_count
+
top1_disagreement_count
==
num_tokens_compared
```

并验证：

```python
expected_agreement = (
    top1_agreement_count
    /
    num_tokens_compared
)
```

与记录的：

```python
top1_agreement
```

在浮点容差内一致，例如：

```python
abs(
    expected_agreement
    -
    top1_agreement
) <= 1e-12
```

---

## 16.4 Percentile 单调性

必须：

```text
0 <= mean_abs_error
0 <= p99_abs_error
0 <= p999_abs_error
0 <= max_abs_error
```

以及：

```python
p99_abs_error
<=
p999_abs_error
<=
max_abs_error + histogram_bin_width
```

注意由于 percentile 返回 histogram bin 的**上边界**，最后一个 percentile upper edge 理论上可能略高于真实 observed `max_abs_error`。

因此不要强制：

```python
p999_abs_error <= max_abs_error
```

应允许最多一个 histogram bin width。

获取：

```python
estimator = regression[
    "absolute_error_percentile_estimator"
]

bin_width = (
    float(estimator["final_range_max"])
    /
    int(estimator["num_bins"])
)
```

然后：

```python
p999_abs_error
<=
max_abs_error + bin_width + 1e-12
```

---

## 16.5 Percentile estimator metadata

要求：

```python
estimator["method"]
==
"streaming_linear_histogram"
```

```python
estimator["num_bins"]
==
8192
```

```python
estimator["reported_value"]
==
"bin_upper_edge"
```

```python
estimator["exact"]
is False
```

以及：

```python
estimator["final_range_max"] > 0
```

---

## 16.6 Hard gate consistency

即使 JSON 已有：

```text
passed=true
```

verify_artifacts 还应额外检查：

```python
relative_l2_error
<=
threshold["relative_l2_error"]
```

避免 `passed` 字段被手工改成 true。

如果不满足，报错。

---

# 17. Tests：`tests/test_rotation_regression.py`

必须扩展现有 test。

不要加载真实 1.7B / 8B 模型。

---

## 17.1 更新现有 metrics test

当前测试：

```python
base = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0]]]
)

rotated = torch.tensor(
    [[[0.0, 2.0], [30.0, 40.0]]]
)

attention_mask = torch.tensor(
    [[1, 0]]
)
```

只有第一个 token 有效。

有效 token：

```text
Base logits:
[1, 2]

Rotated logits:
[0, 2]
```

所以：

```text
base top1 = index 1
rotated top1 = index 1
top1 agreement = 1.0
```

增加断言：

```python
assert metrics[
    "top1_agreement"
] == pytest.approx(1.0)

assert metrics[
    "top1_agreement_count"
] == 1

assert metrics[
    "top1_disagreement_count"
] == 0
```

并检查：

```python
assert (
    metrics["p99_abs_error"]
    <=
    metrics["p999_abs_error"]
)
```

以及：

```python
assert (
    metrics["p999_abs_error"]
    <=
    metrics["max_abs_error"]
    +
    bin_width
    +
    1e-12
)
```

---

## 17.2 新增 Top-1 disagreement test

例如：

```python
base = torch.tensor(
    [[[10.0, 0.0]]]
)

rotated = torch.tensor(
    [[[0.0, 10.0]]]
)

mask = torch.tensor(
    [[1]]
)
```

应：

```text
top1_agreement = 0.0
top1_agreement_count = 0
top1_disagreement_count = 1
```

---

## 17.3 Padding 必须不影响 Top-1

构造：

```text
token 1:
    valid
    top1 same

token 2:
    padding
    top1 intentionally different
```

结果必须仍：

```text
top1_agreement = 1.0
```

---

## 17.4 Histogram expansion test

直接测试：

```python
hist = _StreamingAbsErrorHistogram(
    num_bins=8,
    initial_max=1.0,
)
```

先输入：

```text
[0.1, 0.5, 0.9]
```

再输入：

```text
[2.5]
```

确认：

```text
range_max
```

至少扩到：

```text
4.0
```

且：

```python
hist.total_count == 4
```

并确认 percentile 返回非负、单调。

---

## 17.5 Histogram count preservation

测试多次 range doubling 后：

```python
hist.counts.sum().item()
==
hist.total_count
```

如果 rebin 写错，这个测试必须失败。

---

## 17.6 JSON serializable

保留：

```python
json.dumps(metrics)
```

确保：

```text
torch.Tensor
torch.dtype
numpy scalar
```

没有混入最终 JSON。

---

# 18. `compute_logits_error_metrics()` 行为

这个 helper 必须自然返回新增字段：

```text
p99_abs_error
p999_abs_error
top1_agreement
top1_agreement_count
top1_disagreement_count
absolute_error_percentile_estimator
```

不要建立两套不同 metric 逻辑。

测试和真实 regression 必须共用同一个 accumulator。

---

# 19. `RotationRegressionError`

本轮错误信息仍以 hard gate：

```text
relative_l2_error
```

为主。

现有：

```text
Rotation logits regression failed:
relative_l2_error=...
exceeds threshold=...
```

可以保持不变。

不要让：

```text
P99
P99.9
Top1
```

在没有 threshold 的情况下触发 failure。

---

# 20. `实验执行总结.md`

必须同步更新 Rotation regression 描述。

当前 `prepare_rotation.py` 部分应补充：

```text
Rotation regression 在固定 128 个 calibration samples 上记录：

1. relative L2 error
   - 当前 hard gate
   - 必须 <= 0.01

2. mean / max absolute error
   - diagnostics

3. P99 / P99.9 absolute error
   - 对所有有效 token × vocab logits 的逐元素 absolute error
   - 使用 deterministic streaming histogram 近似计算
   - diagnostics

4. Top-1 agreement
   - 对每一个有效 token 比较 Base 与 Rotated 的 argmax token id
   - exact metric
   - 当前先记录，不设置 hard threshold
```

明确写：

```text
本阶段 passed=true 当前仍由
relative_l2_error <= 0.01
决定。
```

不要写成：

```text
P99/P99.9 必须小于某个尚未确定的阈值
```

也不要写：

```text
top1 必须 >= 99.9%
```

---

# 21. `rotation_regression.json` 文档示例

在 `实验执行总结.md` 可增加简化示例：

```json
{
  "relative_l2_error": 0.0077,
  "top1_agreement": 0.999,
  "p99_abs_error": 0.1,
  "p999_abs_error": 0.5,
  "max_abs_error": 6.6875,
  "threshold": {
    "relative_l2_error": 0.01
  },
  "passed": true
}
```

所有数值只作为格式示例，不能声称是实际结果。

---

# 22. 是否修改 `configs/experiment_matrix.yaml`

本轮：

```text
不需要
```

新增 top1 threshold。

继续保持：

```yaml
rotation:
  regression_relative_l2_threshold: 0.01
```

不要新增：

```yaml
regression_top1_agreement_threshold:
```

除非用户之后根据三个模型结果明确决定 threshold。

---

# 23. Generated config

由于本轮不增加 config key：

```text
不需要重新 materialize config
```

只是因为 schema version / metrics 改变，旧的：

```text
rotation_regression.json
```

应视为过期，需要重新运行：

```bash
python scripts/prepare_rotation.py \
  --config "$ROT_CFG"
```

生成 `format_version=2` regression artifact。

---

# 24. 旧 regression 工件兼容策略

不要让 `verify_artifacts.py` 静默接受：

```text
format_version=1
```

因为 version 1 不包含：

```text
top1
p99
p999
```

应要求：

```text
format_version == 2
```

这样已有旧 Rotation 工件必须重新跑一次 `prepare_rotation.py`。

这正符合项目 reproducibility 目标。

---

# 25. 性能 / 显存要求

新增指标不能显著增加 GPU peak memory。

禁止：

```text
保存所有 Base logits
保存所有 Rotated logits
保存所有 absolute differences
```

现有流程：

```text
batch forward
→ compare
→ accumulate
→ delete
```

保持不变。

Top-1 只保存：

```text
batch × sequence
```

的 token ids，开销很小。

Histogram 只保存：

```text
8192 int64 counts
```

CPU 内存开销极小。

额外主要开销是：

```text
每个 logits chunk 一次 histogram scan
```

这是可接受的 correctness regression 开销。

---

# 26. Numerical semantics

当前 regression logits 比较仍使用：

```text
Base BF16 outputs
Rotated BF16 outputs
```

而误差计算：

```python
base.float()
rotated.float()
```

即：

```text
BF16 model output
→ FP32 metric accumulation
```

保持现有逻辑。

Histogram 也基于相同的：

```python
difference.float().abs()
```

不要基于原始 BF16 直接做 histogram。

---

# 27. P99/P99.9 的语义必须写清楚

必须明确它们不是：

```text
每 token mean error 的 percentile
每 sample error 的 percentile
每 batch percentile 的平均
top1 logit error percentile
```

而是：

\[
d_{t,v}
=
|
L^B_{t,v}
-
L^R_{t,v}
|
\]

对全部：

```text
valid token t
× vocabulary index v
```

的 \(d_{t,v}\) 做全局 percentile。

---

# 28. Top-1 Agreement 的语义必须写清楚

不是比较生成序列。

当前 regression 使用 teacher-forced forward：

```text
input_ids
→ full logits
```

Top1 agreement 是：

```text
对每一个已有输入 token position 的 next-token logits
比较 argmax token id 是否一致
```

不运行：

```text
model.generate()
greedy generation
sampling
beam search
```

---

# 29. 修改完成后的测试

执行：

```bash
cd ~/SNN

python -m compileall -q snn2 scripts tests

python -m pytest -q \
  tests/test_rotation_regression.py

python -m pytest -q
```

全部通过。

---

# 30. 真实 Qwen3-1.7B regression 重跑

已有：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
```

重新运行：

```bash
python scripts/prepare_rotation.py \
  --config "$ROT_CFG"
```

然后检查：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/snn2_main_v1/tldr/"
    "Qwen_Qwen3-1.7B-Base/"
    "_shared/seed42/rotated_prefix/"
    "rotation/rotation_regression.json"
)

data = json.loads(
    path.read_text()
)

for key in (
    "format_version",
    "relative_l2_error",
    "top1_agreement",
    "top1_agreement_count",
    "top1_disagreement_count",
    "p99_abs_error",
    "p999_abs_error",
    "max_abs_error",
    "passed",
):
    print(
        f"{key}: {data[key]}"
    )

assert data["format_version"] == 2
assert data["num_samples"] == 128

assert (
    data["top1_agreement_count"]
    +
    data["top1_disagreement_count"]
    ==
    data["num_tokens_compared"]
)

assert (
    0.0
    <=
    data["top1_agreement"]
    <=
    1.0
)

assert (
    data["p99_abs_error"]
    <=
    data["p999_abs_error"]
)

assert data["passed"] is True
PY
```

---

# 31. 本轮结果应如何解读

假设新 JSON 为：

```text
relative_l2_error:
    0.0077

top1_agreement:
    0.999x

p99_abs_error:
    ...

p999_abs_error:
    ...

max_abs_error:
    6.6875
```

解释：

```text
relative_l2:
    整体 logits 差异

top1_agreement:
    模型在多少 token positions 上保持相同 argmax 决策

P99:
    约 99% 的逐元素 logits absolute errors
    不超过该值

P99.9:
    约 99.9% 的逐元素 logits absolute errors
    不超过该值

max:
    最极端单个 logit error
```

这比只看：

```text
relative_l2
```

更完整。

---

# 32. 后续三个模型都要记录

修改完成后：

```text
Qwen3-1.7B
Qwen3-8B
Llama3-8B
```

分别运行：

```text
prepare_rotation.py
```

每个模型都必须保存自己的：

```text
rotation_regression.json
```

然后对比：

```text
relative_l2_error
top1_agreement
p99_abs_error
p999_abs_error
max_abs_error
```

只有在获得三个模型的自然结果后，才讨论：

```text
是否增加 top1 hard gate
以及 threshold 取多少
```

---

# 33. 不得修改的内容

本轮不要修改：

```text
Hadamard Q 定义
random signs 顺序
R1 / R2 / R3 / R4 placement
RMSNorm fusion
fast-hadamard-transform backend
R3/R4 FP32 Hadamard
Prefix protocol
Calibration protocol
10 activation sites
ANN training
SNN conversion
TL;DR evaluation
lm-eval evaluation
learning-rate layout
```

本轮只扩展 regression metrics。

---

# 34. Codex 最终交付说明

修改完成后 Codex 最终必须报告：

1. 修改文件列表；
2. `rotation_regression.json` 是否升级到 format version 2；
3. `top1_agreement` 的精确定义；
4. `p99_abs_error / p999_abs_error` 的定义；
5. percentile estimator 是否为：
   ```text
   streaming_linear_histogram
   num_bins=8192
   ```
6. 是否确认没有保存全部 logits errors；
7. 当前 hard gate 是否仍只有：
   ```text
   relative_l2_error <= 0.01
   ```
8. `tests/test_rotation_regression.py` 新增了哪些测试；
9. `verify_artifacts.py` 新增了哪些 consistency check；
10. `pytest -q` 结果；
11. 如果实际重跑 Qwen3-1.7B，报告：
    ```text
    relative_l2_error
    top1_agreement
    p99_abs_error
    p999_abs_error
    max_abs_error
    passed
    ```
12. `git status`。

不要只回复“已修改完成”。

---

# 35. 最终完成标准

```text
[ ] rotation_regression format_version = 2

[ ] exact top1_agreement added
[ ] top1_agreement_count added
[ ] top1_disagreement_count added

[ ] p99_abs_error added
[ ] p999_abs_error added

[ ] P99/P99.9 cover all valid token × vocab logits
[ ] no all-errors materialization
[ ] deterministic streaming histogram used
[ ] histogram metadata recorded in JSON
[ ] histogram count == num_elements

[ ] padding excluded from top1 and percentile metrics

[ ] current relative-L2 hard gate unchanged
[ ] no arbitrary top1 threshold introduced

[ ] prepare_rotation initial JSON updated
[ ] verify_artifacts validates new schema
[ ] tests cover top1 agreement/disagreement
[ ] tests cover histogram range expansion
[ ] tests cover histogram count preservation
[ ] tests cover percentile monotonicity

[ ] 实验执行总结.md updated

[ ] all tests pass
[ ] existing experiment protocol unchanged
```
