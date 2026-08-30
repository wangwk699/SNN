# Phase-aware Site 1/7 Role-specific Clip 运行错误修正方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 本文档只处理本次 Phase-aware ANN training 首个 forward 中出现的 Site 1/7 role-specific Clip 错误。  
> 不修改 DeepSpeed、NCCL、Rotation、Prefix、Calibration A/B、SNN deployment 等其他逻辑。

---

# 1. 问题现象

运行：

```bash
CFG_17_P=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml
export CUDA_VISIBLE_DEVICES=6,7
NGPU=2

torchrun \
  --standalone \
  --nproc_per_node="$NGPU" \
  scripts/train_ann.py \
  --config "$CFG_17_P"
```

训练已经完成数据准备并进入第一个 model forward，随后在 Site 1 报错：

```text
input_layernorm
→ norm1_hook
→ controller.apply(index, 1, output)
→ modules["clip"](output)
→ Clipper.forward(role=None)
→ ValueError
```

最终：

```text
ValueError: Clip role must be one of ('q', 'k', 'v'), got None
```

这不是 DeepSpeed、双卡、NCCL 或显存问题。

---

# 2. 根因

Site 1 和 Site 7 的 Stage B Clip 是 role-specific：

```text
Site 1:
roles = ("q", "k", "v")

Site 7:
roles = ("gate", "up")
```

因此它们不能在共享 RMSNorm 输出上执行：

```python
clip(output)
```

因为共享输出没有唯一 role。

正确位置应该是：

```text
Site 1:
input_layernorm
→ Phase
→ q_proj pre-hook   → Clip(role="q")
→ k_proj pre-hook   → Clip(role="k")
→ v_proj pre-hook   → Clip(role="v")

Site 7:
post_attention_layernorm
→ Phase
→ gate_proj pre-hook → Clip(role="gate")
→ up_proj pre-hook   → Clip(role="up")
```

当前 `model_integration.py` 中上述 branch hook 已经存在，方向是正确的。

真正的问题在 `SiteController` 的 module cache。

---

# 3. 当前代码为什么仍然会失败

当前 `snn2/controller.py::_load()` 已有类似防护：

```python
clip_enabled = (
    self.common_clip_enabled
    and site_supports_clip_for_mode(site_index, self.mode)
    and not (self.mode == "phase" and site_index in {1, 7})
)
```

这只能保证第一次 `controller.apply(Site 1/7)` 时不主动加载通用 Clip。

但是 `apply_role_clip(...)` 使用同一个缓存：

```python
modules = self._modules.setdefault(key, {})
```

并会：

```python
modules["clip"] = Clipper(state)
```

因此可能出现：

```text
第一次 shared Norm:
controller.apply(Site 1)
→ cache = {"phase": ...}

q_proj pre-hook:
apply_role_clip(role="q")
→ cache = {"phase": ..., "clip": role-specific Clipper}

下一次 forward / gradient recomputation:
controller.apply(Site 1)
→ _load() 返回已有 cache
→ cache 中已经存在 "clip"
→ modules["clip"](output)
→ role=None
→ ValueError
```

所以不能只依赖 `_load()` 阶段“不加载 Clip”的条件。必须在 Phase `apply()` 本身建立硬性规则：Site 1/7 永远禁止 generic Clip。

---

# 4. 必须修改 `snn2/controller.py`

找到：

```python
if self.mode == "phase":
    output = modules["phase"](x)
    output = modules["clip"](output) if "clip" in modules else output
    if recorder is not None:
        self.record_regression(f"{checkpoint}/post", output)
    return output
```

修改为：

```python
if self.mode == "phase":
    output = modules["phase"](x)

    # Site 1/7 use role-specific Clip only in branch pre-hooks.
    # Never apply their cached role-specific Clipper to the
    # shared RMSNorm output, because the shared tensor has no role.
    if site_index not in {1, 7} and "clip" in modules:
        output = modules["clip"](output)

    if recorder is not None:
        self.record_regression(f"{checkpoint}/post", output)

    return output
```

核心规则：

```text
Phase mode:

Site 1/7:
    shared activation
    → Phase only
    → NO generic Clip

Other Clip-eligible sites:
    activation
    → Phase
    → generic Clip
```

---

# 5. `model_integration.py` 的 branch hook 保持不变

当前已有类似：

```python
if (
    controller.mode == "phase"
    and controller.common_clip_enabled
    and site_index in {1, 7}
):
    replaced = controller.apply_role_clip(
        index,
        site_index,
        inputs[0],
        role=role,
    )
    return (replaced, *inputs[1:])
```

Site 1 分支：

```text
q_proj   → role="q"
k_proj   → role="k"
v_proj   → role="v"
```

Site 7 分支：

```text
gate_proj → role="gate"
up_proj   → role="up"
```

这些 hook 不要移动，也不要删除。

---

# 6. 不允许的修复方式

## 6.1 不要给 `norm1_hook` / `norm2_hook` 人工指定 role

错误示例：

```python
controller.apply(index, 1, output, gif_role="q")
```

共享 RMSNorm 输出不是 q-only、k-only 或 v-only tensor，因此不能人为指定某个 role。Site 7 同理。

## 6.2 不要关闭整个 Phase-aware common Clip

错误方向：

```yaml
common_clip_enabled: false
```

这会改变实验定义，只是绕开异常，不是正确修复。

## 6.3 不要删除 `apply_role_clip()`

Site 1/7 的 Stage B profile 本来就是 role-specific，它们必须继续在 branch hook 中执行。

## 6.4 不要修改 GIF-aware 逻辑

本次问题只针对：

```text
Phase-aware
+ Site 1/7
+ role-specific Clip
+ shared Phase apply
```

不要顺带改变 GIF-aware 的 role-specific GIF / Clip 行为。

---

# 7. 必须补 regression test

当前已有类似：

```python
def test_phase_site1_shared_apply_does_not_clip_then_branch_clips(...):
```

但它只测试：

```text
shared apply
→ branch role Clip
```

没有再次执行 shared apply，因此没有覆盖：

```text
role-specific Clip 已进入 self._modules cache
→ shared Phase apply 再次执行
```

这一真实失败场景。

建议新增参数化测试。

---

# 8. 推荐测试代码

```python
@pytest.mark.parametrize(
    ("site_index", "roles", "role"),
    [
        (1, ("q", "k", "v"), "q"),
        (7, ("gate", "up"), "gate"),
    ],
)
def test_phase_multirole_shared_apply_never_reuses_role_clip(
    tmp_path,
    site_index,
    roles,
    role,
):
    stage_a = tmp_path / "sites"
    stage_b = tmp_path / "profile"

    _write(
        stage_a,
        site_index,
        "phase",
        _phase_state(),
    )

    _write(
        stage_b,
        site_index,
        "clip",
        _clip_state(roles=roles),
        clip=True,
    )

    controller = SiteController(
        mode="phase",
        site_root=stage_a,
        clip_root=stage_b,
        common_clip_enabled=True,
        phase_T=4,
        phase_surrogate_slope=1.0,
    )

    x = torch.full((1, 1, 4), 3.0)

    # First shared Norm path: Phase only.
    shared_before = controller.apply(
        0,
        site_index,
        x,
    )

    # Branch-specific Clip is loaded into controller cache.
    branch_output = controller.apply_role_clip(
        0,
        site_index,
        shared_before,
        role=role,
    )

    key = site_key(0, site_index)

    # Important regression condition:
    # role-specific Clipper is now present in the shared module cache.
    assert "clip" in controller._modules[key]

    # Shared Norm path must STILL run Phase only.
    # It must not call Clipper(role=None),
    # and must not apply the cached branch Clip.
    shared_after = controller.apply(
        0,
        site_index,
        x,
    )

    torch.testing.assert_close(
        shared_after,
        shared_before,
    )
```

最关键的是：

```python
assert "clip" in controller._modules[key]
```

之后再次调用：

```python
controller.apply(...)
```

这一步精确覆盖本次真实 bug。

---

# 9. Site 1 和 Site 7 都必须覆盖

不要只测试 Site 1。

至少覆盖：

```text
Site 1 + q
Site 7 + gate
```

如果希望更严格，也可以覆盖全部 role：

```python
[
    (1, ("q", "k", "v"), "q"),
    (1, ("q", "k", "v"), "k"),
    (1, ("q", "k", "v"), "v"),
    (7, ("gate", "up"), "gate"),
    (7, ("gate", "up"), "up"),
]
```

---

# 10. 保留普通 Site generic Clip test

现有 Site 6 测试应继续保留，验证：

```text
Site 6:
Phase
→ generic Clip
```

例如：

```python
output = controller.apply(
    0,
    6,
    torch.full((1, 1, 4), 3.0),
)

assert torch.all(output <= 0.25)
```

这样可以证明本次修复没有错误关闭所有 Phase Clip。

---

# 11. 最终 Phase-aware forward 语义

## Site 1

```text
input_layernorm
      ↓
shared RMSNorm output
      ↓
PhaseSurrogate
      ↓
      ├── q_proj pre-hook → Clip(q)
      ├── k_proj pre-hook → Clip(k)
      └── v_proj pre-hook → Clip(v)
```

共享位置禁止：

```text
Clip(role=None)
```

## Site 7

```text
post_attention_layernorm
      ↓
shared RMSNorm output
      ↓
PhaseSurrogate
      ↓
      ├── gate_proj pre-hook → Clip(gate)
      └── up_proj pre-hook   → Clip(up)
```

共享位置禁止：

```text
Clip(role=None)
```

## 其他普通 Clip site

保持：

```text
activation
→ PhaseSurrogate
→ generic Clip
```

---

# 12. DeepSpeed / NCCL 日志无需修改

以下不是本次失败根因：

```text
Gradient accumulation steps mismatch:
GradientAccumulationPlugin has 1,
DeepSpeed config has 16.
Using DeepSpeed's value.
```

以及：

```text
using GPU ... to perform barrier as devices used by this process are currently unknown
```

实际训练已经进入 Trainer 的 model forward，然后才在 Site 1 role-specific Clip 处失败。

本轮不要修改：

```text
DeepSpeed config
torchrun GPU mapping
gradient accumulation
NCCL initialization
```

---

# 13. 推荐修改文件

必须修改：

```text
snn2/controller.py
```

必须新增/加强测试：

```text
tests/test_controller_state_loading.py
```

原则上不需要修改：

```text
snn2/model_integration.py
```

因为 branch-specific role hook 已经在正确位置。

---

# 14. 验收标准

完成修改后必须满足：

1. `pytest -q` 全部通过。
2. Phase Site 1 首次 shared apply 不执行 Clip。
3. Phase Site 1 role Clip 被 cache 后，再次 shared apply 仍不执行 Clip。
4. Phase Site 7 同样满足上述规则。
5. Site 1 q/k/v branch Clip 正常执行。
6. Site 7 gate/up branch Clip 正常执行。
7. 普通 Site（例如 Site 6）继续执行 `Phase → generic Clip`。
8. GIF-aware 行为不变。
9. SNN deployment 行为不变。
10. 不再出现：
   ```text
   Clip role must be one of ('q', 'k', 'v'), got None
   ```
11. 不再出现 Site 7 对应：
   ```text
   Clip role must be one of ('gate', 'up'), got None
   ```

---

# 15. 修改后执行

先运行：

```bash
pytest -q
```

全部通过后重新执行：

```bash
CFG_17_P=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml
export CUDA_VISIBLE_DEVICES=6,7
NGPU=2

torchrun \
  --standalone \
  --nproc_per_node="$NGPU" \
  scripts/train_ann.py \
  --config "$CFG_17_P"
```

预期：

```text
训练能够越过第一个 forward，
不再在 Site 1 input_layernorm 处因为 role=None 的 Clipper 调用退出。
```

如果随后出现新的错误，应基于新的第一条真实 traceback 继续定位，不要把当前 DeepSpeed/NCCL warning 当作根因。
