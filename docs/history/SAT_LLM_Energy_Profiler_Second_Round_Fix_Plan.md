# SAT-LLM Energy Profiler 修正方案（第二轮）

> 适用仓库：`https://github.com/wangwk699/SNN`  
> 基线：当前 `main` 分支已完成第一轮 Energy Profiler，实现位于 `scripts/profile_energy.py`、`snn2/energy_profiler.py` 等文件。  
> 本轮目标：修复 mixed dynamic/event temporal matmul 计数错误，降低 profiler 的额外计算开销，并完善 Energy 结果路径隔离与 GIF zero-point accounting metadata。  
> **最重要约束：不得改变现有 ANN/SNN evaluation 数值路径。**

---

## 1. 本轮必须修正的问题

当前实现整体方向正确，但存在以下四个问题：

1. `count_temporal_matmul()` 在 `dynamic × event` / `event × dynamic` 时错误地对 dense operand 的全 1 multiplicity 做 `cumsum()`，会高估 mixed temporal matmul 的 MAC 数。
2. `_pair()` 当前通过真正执行 FP64 dense matmul 来统计 pair count；对于 8B、512-token、128 samples 的 profiling，这会引入非常大的无必要额外开销。
3. Energy 输出路径未完整隔离：
   - `calibration.num_samples`
   - 实际 `Prefix enabled` 状态  
   后续做 sample-count sweep 或 Prefix ablation 时可能覆盖旧结果。
4. GIF asymmetric zero-point 的硬件 accounting assumption 目前 metadata 不够明确，也缺少 non-zero zero-point 测试。

以上四项全部纳入本轮修改。

---

# 2. mixed dynamic/event temporal matmul 计数必须修正

当前 `snn2/energy_profiler.py` 中的逻辑大致为：

```python
def count_temporal_matmul(a, b, ma, mb):
    if ma is None and mb is None:
        ...
    aa = ma if ma is not None else torch.ones_like(a, dtype=torch.int32)
    bb = mb if mb is not None else torch.ones_like(b, dtype=torch.int32)
    count = (
        _pair(aa.cumsum(0), bb)
        + _pair(aa, bb.cumsum(0))
        + _pair(aa, bb)
    )
```

这个公式仅对：

```text
event × event
```

正确。

对于：

```text
dynamic × event
event × dynamic
```

不正确。

原因是 `temporal_seq_matmul()` 的实际数值计算为：

\[
\operatorname{cumsum}(A)_t B_t
+
A_t\operatorname{cumsum}(B)_t
-
A_tB_t.
\]

如果 `A` 是 dynamic dense operand，则：

```text
cumsum(A)_t
```

是已经形成的一个动态数值，不应因为它由前面多个 temporal increments 累积而把乘法次数乘 `t+1`。

---

# 3. 正确的四类 temporal matmul accounting

请将：

```python
count_temporal_matmul(a, b, ma, mb)
```

明确拆成四种情况。

## 3.1 dynamic × dynamic

若：

```python
ma is None
mb is None
```

三个实际 temporal matmul term 都是 dense MAC。

若单个 temporal matmul 的 shape 为：

```text
A [..., I, K]
B [..., K, J]
```

则单项 operation count 为：

\[
\prod(\text{batch/head dims})\cdot I K J.
\]

总计：

\[
\boxed{
N_{\mathrm{MAC}}
=
3\times
\prod(\text{batch/head dims})\cdot I K J
}
\]

不要真正执行 matmul 来数 operation。

---

## 3.2 event × event

若：

```python
ma is not None
mb is not None
```

继续使用：

\[
\mathrm{Pair}(\widehat M_A,M_B)
+
\mathrm{Pair}(M_A,\widehat M_B)
+
\mathrm{Pair}(M_A,M_B)
\]

其中：

\[
\widehat M_A(t)=\sum_{\tau\le t}M_A(\tau),
\qquad
\widehat M_B(t)=\sum_{\tau\le t}M_B(\tau).
\]

总数计入：

```text
Synaptic ACs
```

---

## 3.3 dynamic × event

若：

```python
ma is None
mb is not None
```

第一项：

\[
\operatorname{cumsum}(A)_t B_t
\]

由于 `A` 是 dynamic，`cumsum(A)_t` 每个位置仍只是一个动态 operand，因此：

\[
\mathrm{Pair}(1,M_B)
\]

而不是：

\[
\mathrm{Pair}(\operatorname{cumsum}(1),M_B).
\]

第二项：

\[
A_t\operatorname{cumsum}(B)_t
\]

为：

\[
\mathrm{Pair}(1,\widehat M_B).
\]

第三项：

\[
A_tB_t
\]

为：

\[
\mathrm{Pair}(1,M_B).
\]

因此：

\[
\boxed{
N_{\mathrm{MAC}}
=
\mathrm{Pair}(1,M_B)
+
\mathrm{Pair}(1,\widehat M_B)
+
\mathrm{Pair}(1,M_B)
}
\]

计入 `MACs`，不是 AC。

---

## 3.4 event × dynamic

若：

```python
ma is not None
mb is None
```

第一项：

\[
\operatorname{cumsum}(A)_t B_t
\]

为：

\[
\mathrm{Pair}(\widehat M_A,1).
\]

第二项：

\[
A_t\operatorname{cumsum}(B)_t
\]

dynamic `cumsum(B)` 仍只算一个动态 operand：

\[
\mathrm{Pair}(M_A,1).
\]

第三项：

\[
A_tB_t
\]

为：

\[
\mathrm{Pair}(M_A,1).
\]

因此：

\[
\boxed{
N_{\mathrm{MAC}}
=
\mathrm{Pair}(\widehat M_A,1)
+
\mathrm{Pair}(M_A,1)
+
\mathrm{Pair}(M_A,1)
}
\]

---

# 4. 必须修改现有错误测试

当前测试中有：

```python
assert count_temporal_matmul(a, b, None, m).mac_raw == 8
```

这个期望值是错误的。

对于最小例子：

```text
T = 2
dynamic A
event B，每 timestep 1 event
```

正确计数：

- `cumsum(A) * B`：2 MAC
- `A * cumsum(B)`：3 MAC
- `A * B`：2 MAC

总计：

\[
\boxed{7}
\]

所以应改为：

```python
assert count_temporal_matmul(a, b, None, m).mac_raw == 7
```

同时增加对称测试：

```python
assert count_temporal_matmul(a, b, m, None).mac_raw == 7
```

---

# 5. 增加真实 GIF PV regression test

必须新增一个明确覆盖真实 GIF attention 语义的测试：

```text
GIF Site 5 = dense dynamic
GIF Site 4 = event
```

即：

\[
P V
\]

中：

```text
P = SoftmaxIdentityGIF -> dense_dynamic
V = GIF event
```

这是：

```text
dynamic × event
```

的真实主路径。

建议测试名：

```python
def test_gif_pv_dynamic_attention_times_event_value_uses_mixed_mac_rule():
    ...
```

测试步骤：

1. 构造 tiny temporal attention weights；
2. 构造 tiny GIF event multiplicity；
3. 调用 `count_temporal_matmul()`；
4. 手工计算三个 temporal terms；
5. 断言 MAC count 完全一致。

不要只测试抽象 helper，需要在测试注释或命名中明确这是 GIF PV 路径。

---

# 6. `_pair()` 不得再通过 FP64 dense matmul 实现

当前 `_pair()` 类似：

```python
value = torch.matmul(
    a.to(torch.float64),
    b.to(torch.float64),
).sum().item()
```

必须删除。

原因：

- 只是为了统计 operation count，却真的执行一次 dense matmul；
- 8B × 512 token × 128 samples 时额外开销非常大；
- 可能创建 `[T,B,H,L,L+P]` 级别 FP64 中间 tensor；
- profiler 的统计逻辑不应比模型 forward 本身更重。

---

# 7. `_pair()` 改为 reduction 公式

若：

```text
A shape = [..., I, K]
B shape = [..., K, J]
```

则：

\[
\mathrm{Pair}(A,B)
=
\sum_k
\left(\sum_i A_{ik}\right)
\left(\sum_j B_{kj}\right).
\]

推荐实现：

```python
def _pair(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape[:-2] != b.shape[:-2]:
        raise ValueError(...)
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(...)

    left = a.to(torch.int64).sum(dim=-2)
    right = b.to(torch.int64).sum(dim=-1)
    value = int((left * right).sum().item())

    if value < 0:
        raise OverflowError("Event pair count overflowed int64")
    return value
```

当前 multiplicity 很小：

- Phase / MTN：0/1；
- GIF：单 timestep 每 element 最大 15；
- T 很小；
- sequence length 固定 512。

int64 足够。

生产 profiler 不再使用 FP64 matmul 来统计 pair count。

---

# 8. mixed case 不要构造巨大 `ones_like()`

mixed case 中不要继续：

```python
torch.ones_like(a)
```

或：

```python
torch.ones_like(b)
```

尤其 attention shape 较大。

直接利用 shape 与 event multiplicity：

对于：

```text
dynamic × event
```

若：

```text
A [..., I, K]
B [..., K, J]
```

则：

\[
\mathrm{Pair}(1,M_B)
=
I\sum M_B.
\]

所以：

```python
pair_dense_left_event_right = (
    int(a.shape[-2]) * int(mb.sum().item())
)
```

若右侧为 cumulative event：

```python
int(a.shape[-2]) * int(mb.cumsum(0).sum().item())
```

对于：

```text
event × dynamic
```

有：

\[
\mathrm{Pair}(M_A,1)
=
J\sum M_A
\]

因此：

```python
int(b.shape[-1]) * int(ma.sum().item())
```

不要为 mixed accounting 构造全 1 tensor。

---

# 9. 推荐重构 `count_temporal_matmul()`

直接写成清晰的四分支：

```python
def count_temporal_matmul(a, b, ma, mb):
    if ma is None and mb is None:
        ...
    if ma is not None and mb is not None:
        ...
    if ma is None:
        ...
    else:
        ...
```

不要为了复用一个公式而强行用 `aa/bb` 统一。

本轮优先保证：

```text
accounting correctness
```

而不是代码最短。

---

# 10. `temporal_product_counts()` mixed case 一并修正

当前 `temporal_symmetric_hadamard()`：

\[
0.5
\left(
A_t\sum_\tau B_\tau
+
\sum_\tau A_\tau B_t
\right).
\]

现有 helper 也存在 mixed dynamic/event 把 dense operand 当作 temporal event-history 的同类风险。

请明确拆成四类：

## dynamic × dynamic

两个实际 product branches：

\[
2\times \mathrm{numel}(A)
\]

计 MAC。

## event × event

按实际两项 event multiplicity 计 AC。

## dynamic × event

dynamic side 每个位置仍是一个动态 operand；只根据 event side 的 multiplicity决定有效 MAC 数。

## event × dynamic

对称处理。

虽然当前主实验中：

- Phase / MTN 多数是 event-event；
- GIF Site 8/9 是 identity，通常是 dynamic-dynamic；

但 helper 本身必须修正确，避免未来 topology 改变后出现错误。

---

# 11. Energy 结果路径增加 profiling sample count 隔离

当前 Energy ANN 路径类似：

```text
<run_root>/energy/ann/
```

对于 vanilla / unaware，仅修改：

```yaml
calibration.num_samples
```

时 ANN run root 不一定变化，因此：

```text
128 samples
256 samples
```

可能覆盖同一目录。

本轮必须在 Energy leaf path 中加入：

```text
profile_num_samples_<N>
```

---

# 12. Energy 路径增加实际 Prefix enabled 隔离

`evaluation.prefix_enabled` 会改变 profiling graph 和 Prefix length。

因此 Energy 结果还必须区分：

```text
prefix_enabled_true
prefix_enabled_false
```

优先复用已有：

```python
prefix_enabled_dirname(...)
```

不要再手写字符串规则。

---

# 13. 推荐最终 Energy 路径

## ANN

推荐：

```text
<run_root>/
  energy/
    ann/
      prefix_enabled_<actual_bool>/
        profile_num_samples_<N>/
```

对于 vanilla：

即使 config 里 `evaluation.prefix_enabled=true`，如果 Final ANN evaluation 的实际语义是禁用 Prefix，则路径必须反映**实际 Final ANN Prefix 状态**。

不要直接用：

```python
cfg["evaluation"]["prefix_enabled"]
```

应复用：

```python
final_ann_evaluation_prefix_enabled(cfg)
```

或等价已有 helper。

---

## SNN

先镜像当前：

```python
layout.snn_dir(neuron)
```

的 deployment identity，再增加：

```text
prefix_enabled_<actual_bool>/
profile_num_samples_<N>/
```

例如：

```text
.../energy/snn/post_finetuning/phase/phase_base_2_T_4/
    prefix_enabled_true/
    profile_num_samples_128/
```

必须保证以下结果互不覆盖：

- Phase T/base override；
- MTN T/K override；
- `conversion.use_post_finetuning_artifacts=true/false`；
- Prefix on/off；
- 128/256 profiling samples。

---

# 14. `ArtifactLayout.energy_dir()` 推荐修改

现有：

```python
@property
def energy_root(self):
    return self.root / "energy"
```

可以保留。

修改 `energy_dir(neuron)`，使其 leaf suffix 至少包含：

```text
actual prefix enabled
profile_num_samples
```

建议：

```python
num_samples = int(self._cfg["calibration"]["num_samples"])
```

ANN Prefix 状态：

```python
final_ann_evaluation_prefix_enabled(self._cfg)
```

SNN Prefix 状态：

```python
evaluation_prefix_enabled(self._cfg)
```

或复用项目中更精确的 final SNN helper。

不要自行重写 mode-specific Prefix 判断。

---

# 15. 更新 Energy path tests

扩展现有：

```python
test_energy_paths_isolate_deployments
```

至少验证：

1. Phase T4 / T8；
2. Phase base 2 / 3；
3. MTN T/K；
4. selector true / false；
5. Prefix true / false；
6. `calibration.num_samples = 128 / 256`；
7. ANN 128 / 256；
8. ANN Prefix on / off。

所有 path 必须 distinct。

---

# 16. GIF zero-point accounting assumption 写入 metadata

当前 GIF 使用 asymmetric quantization：

\[
q=\operatorname{round}(x/s)+z
\]

并：

\[
x_{\mathrm{dequant}}=(q-z)s.
\]

所以：

```text
low_zero
high_zero
```

通常可以非零。

本轮**不改变**既定 GIF 主 Energy accounting：

> GIF integer code 按 unit events 展开；固定 zero-point correction 视为 frozen implementation compensation，可通过预折叠、bias-like correction 或固定 event routing 处理，不单独作为 runtime MAC/AC 计入主表。

但必须在 `energy_metadata.json` 中明确写：

```json
"gif_zero_point_compensation_policy":
"fixed_prefolded_or_bias_like_compensation_not_counted"
```

并增加：

```json
"gif_asymmetric_zero_point_energy_included": false
```

---

# 17. 增加 non-zero zero-point 单元测试

现有 GIF test 多数 zero point 为 0。

新增：

```python
def test_gif_nonzero_zero_point_uses_unsigned_code_event_policy():
    ...
```

要求：

1. 构造 `low_zero != 0`；
2. 构造 `high_zero != 0`；
3. `gif_integer_multiplicity()` 仍返回 unsigned integer code / chunks；
4. 不因为 `(q-zero)` 的数值幅度改变 unit-event multiplicity；
5. 不修改真实 `StaticGIF.temporal()` 数值 forward。

这个测试是 accounting policy test，不是修改 GIF neuron 数学。

---

# 18. 不得修改 GIF 数值 forward

本轮不要修改：

```text
StaticGIF.temporal()
AllLowStaticGIF.temporal()
IdentityGIF
SoftmaxIdentityGIF
```

zero-point 只是补充 Energy accounting assumption。

---

# 19. 更新 `energy_metadata.json`

在现有 metadata 基础上增加：

```json
{
  "gif_zero_point_compensation_policy":
    "fixed_prefolded_or_bias_like_compensation_not_counted",

  "gif_asymmetric_zero_point_energy_included": false,

  "energy_path_profile_num_samples": 128,

  "energy_path_prefix_enabled": true
}
```

其中：

```text
energy_path_prefix_enabled
```

必须写实际 profiler 使用的 Prefix enabled 状态。

不要仅记录 raw：

```python
cfg["evaluation"]["prefix_enabled"]
```

---

# 20. 升级 Energy accounting policy version

因为 mixed dynamic/event 的计数语义发生修正，旧结果与新结果可能不同。

将：

```python
ENERGY_PROFILER_VERSION = 1
ENERGY_ACCOUNTING_POLICY = "sat_llm_mac_ac_v1"
```

升级为：

```python
ENERGY_PROFILER_VERSION = 2
ENERGY_ACCOUNTING_POLICY = "sat_llm_mac_ac_v2_mixed_temporal_fix"
```

不要静默沿用 v1。

---

# 21. 旧 Energy 结果

若此前已经用 v1 profiler 生成过：

```text
energy_results.csv
energy_results.json
energy_metadata.json
```

不要继续作为最终论文 Energy 结果。

修复后重新运行。

无需自动 migration。

---

# 22. 推荐修改文件

本轮主要修改：

```text
snn2/energy_profiler.py
snn2/artifacts.py
tests/test_energy_profiler.py
实验执行总结.md
```

可选同步更新：

```text
代码结构总结.md
```

原则上不要修改：

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

除非只是复用/导入已有 helper。

---

# 23. `实验执行总结.md` 更新

Step 11 主命令不变：

```bash
python scripts/profile_energy.py \
  --config "$CFG" \
  --neuron "$NEURON"
```

只需补充：

1. Energy 路径现在包含实际 Prefix enabled 状态；
2. Energy 路径包含 `profile_num_samples_<N>`；
3. 修改 `calibration.num_samples` 后，新的 profiling 不会覆盖旧结果。

---

# 24. 必须新增/修改的测试

至少完成以下测试。

## 24.1 mixed temporal matmul

修改：

```python
assert count_temporal_matmul(a, b, None, m).mac_raw == 7
```

新增：

```python
assert count_temporal_matmul(a, b, m, None).mac_raw == 7
```

保留：

```text
event-event
dynamic-dynamic
```

测试。

## 24.2 GIF PV

新增真实语义 regression：

```python
test_gif_pv_dynamic_attention_times_event_value_uses_mixed_mac_rule
```

## 24.3 GIF zero point

新增：

```python
test_gif_nonzero_zero_point_uses_unsigned_code_event_policy
```

## 24.4 Energy path

扩展 path isolation：

```text
num_samples
Prefix enabled
Phase deployment overrides
MTN deployment overrides
selector
```

## 24.5 `_pair()` 不调用 dense matmul

新增：

```python
def test_pair_counter_does_not_call_dense_matmul(monkeypatch):
    ...
```

仅测试 `_pair()` helper。

可以 monkeypatch `torch.matmul` 为抛异常，确认 `_pair()` 不调用它。

## 24.6 `_pair()` correctness

用 tiny tensor：

```python
efficient = _pair(a, b)
reference = int(torch.matmul(a.float(), b.float()).sum().item())
```

仅在 test 中用 tiny dense matmul 作为 reference。

---

# 25. 性能实现要求

正式 profiler 中：

```text
_pair()
mixed temporal counting
```

不得为了计数执行与原 attention 同规模的 dense matmul。

也不要创建巨大 `ones_like()` 仅用于 accounting。

event pair 统计应主要是：

```text
sum
cumsum
shape multiplication
small reduction
```

---

# 26. 结果变化预期

修复后：

- ANN 通常基本不受 mixed fix 影响；
- Phase / MTN 主要 QK/PV 为 event-event，通常变化较小；
- GIF 的：
  ```text
  Site 5 dense attention × Site 4 event value
  ```
  属于真实 mixed path，因此：
  ```text
  MACs (G)
  Energy (J)
  ```
  可能比 v1 降低。

这是正确修正，不是 regression。

---

# 27. 以下既定规则保持不变

不要改变：

```text
held-out validation
calibration.seed 随机无放回
profile sample count = calibration.num_samples
exactly 512-token full samples
logical batch size = 1
one forward only
no generation
Prefix KV cache 只加载、不生成
Prefix preprocessing cost 不计
Prefix 对当前 attention 的额外运算要计
ANN/Phase/GIF/MTN 一次命令一行
内部累计 raw count
最后转 G
4.6 pJ/MAC
0.9 pJ/AC
Reduction 不计算
Phase nonzero = 1 event
MTN nonzero = 1 event
GIF multi-level code 按 unit events
GIF identity sites 不算 event
Phase/MTN neuron dynamics 计 Neuron AC
GIF static quantizer 不伪造 membrane AC
memory/special function/residual/bias 不进入主 Energy
fixed coefficient 采用 prefold/fixed lookup assumption
```

---

# 28. 重新运行测试

修改完成后必须运行：

```bash
pytest -q
```

全部通过。

不要为了本轮 Energy profiler 修改旧 evaluation 数值测试的期望值。

---

# 29. 数值路径回归保护

继续保留并通过已有：

```text
profiler instrumentation on/off forward output一致
hook/wrapper退出后恢复
```

测试。

本轮修改不得影响现有 evaluation CLI 和输出。

---

# 30. 最终验收标准

### Correctness

- [ ] `dynamic × event` 不再使用 `cumsum(ones)`；
- [ ] `event × dynamic` 对称正确；
- [ ] tiny `None,event` case = 7；
- [ ] tiny `event,None` case = 7；
- [ ] GIF PV mixed regression 通过；
- [ ] event-event 逻辑保持正确；
- [ ] dynamic-dynamic 逻辑保持正确。

### Performance

- [ ] `_pair()` 不调用 dense `torch.matmul`；
- [ ] `_pair()` 使用 reduction；
- [ ] mixed accounting 不构造巨大 `ones_like()`。

### Path

- [ ] 128/256 不覆盖；
- [ ] Prefix true/false 不覆盖；
- [ ] ANN path 同样隔离 sample count / Prefix；
- [ ] Phase T/base、MTN T/K、selector isolation 保持有效。

### GIF zero point

- [ ] metadata 新增 zero-point compensation policy；
- [ ] non-zero zero-point test 通过；
- [ ] GIF 数值 forward 不变。

### Safety

- [ ] existing evaluation 路径不变；
- [ ] profiler hooks 退出后恢复；
- [ ] `pytest -q` 全部通过。

---

# 31. Codex 完成后需汇报

完成后返回：

```text
1. 修改的文件
2. mixed temporal counting 的修正方式
3. 新 `_pair()` 的 reduction 公式
4. Energy 路径的新结构
5. GIF zero-point metadata policy
6. 新增/修改的测试
7. pytest -q 结果
8. 一个可直接运行的 profile_energy.py 示例命令
```

不要在未实际运行 profiler 的情况下伪造任何 MAC/AC/Energy 数值。
