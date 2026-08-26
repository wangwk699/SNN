# SNN `create_conversion()` P0 修复方案

> 基线：当前 `main` 提交 `c28394f1504923dbe55a81f638c2b6924be935ac`
>
> 目标：修复 `snn2/conversion.py::create_conversion()` 中局部变量 `reused` 在赋值前被使用的问题，并补充真正覆盖 `create_conversion()` 的回归测试。
>
> 本次只修复本文描述的问题，不改动已经完成的 Clip 三态语义、per-head grouped calibration、Site 5 特殊策略、artifact 路径设计。

---

## 1. 当前问题

当前 `snn2/conversion.py` 中 `create_conversion()` 的逻辑顺序类似：

```python
def create_conversion(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    neuron: str,
) -> dict[str, Any]:
    ann_checkpoint = layout.ann_checkpoint_dir
    ann_config = ann_checkpoint / "config.json"
    expected_num_hidden_layers = _ann_num_hidden_layers(ann_config)

    prefix, validation, manifest_path, training_provenance = _source_bundle(
        cfg,
        layout,
    )

    controller = SiteController(
        site_root=layout.conversion_site_dir
    )

    steps = controller.set_deployment(
        neuron,
        clip_bundle_policy=(
            "allow_eligible"
            if reused
            else "forbid_all"
        ),
    )

    ...

    reused = conversion_reuses_ann_training_artifacts(cfg)
```

这里：

```python
reused
```

在执行：

```python
"allow_eligible" if reused else "forbid_all"
```

时尚未赋值。

因此实际运行：

```bash
python scripts/convert_snn.py \
  --config <CONFIG> \
  --neuron phase
```

会触发：

```text
UnboundLocalError:
local variable 'reused' referenced before assignment
```

这是必须修复的 P0 运行时错误。

---

# 2. 修改 `snn2/conversion.py`

目标函数：

```python
create_conversion()
```

必须将：

```python
reused = conversion_reuses_ann_training_artifacts(cfg)
```

移动到第一次使用 `reused` 之前。

推荐改成：

```python
def create_conversion(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    neuron: str,
) -> dict[str, Any]:
    ann_checkpoint = layout.ann_checkpoint_dir
    ann_config = ann_checkpoint / "config.json"

    expected_num_hidden_layers = _ann_num_hidden_layers(
        ann_config
    )

    reused = conversion_reuses_ann_training_artifacts(cfg)

    (
        prefix,
        validation,
        manifest_path,
        training_provenance,
    ) = _source_bundle(
        cfg,
        layout,
    )

    controller = SiteController(
        site_root=layout.conversion_site_dir
    )

    steps = controller.set_deployment(
        neuron,
        clip_bundle_policy=(
            "allow_eligible"
            if reused
            else "forbid_all"
        ),
    )

    output = layout.snn_conversion_dir(neuron)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rotation_enabled = bool(
        cfg["rotation"]["enabled"]
    )

    rotation_path = (
        layout.rotation_dir
        / "rotation_state.pt"
    )

    if (
        rotation_enabled
        and not rotation_path.exists()
    ):
        raise FileNotFoundError(
            rotation_path
        )

    metadata = {
        ...
        "reused_ann_training_artifacts": reused,
        ...
    }

    write_json(
        output / "conversion_metadata.json",
        metadata,
    )

    return metadata
```

然后删除函数后半段原来的重复赋值：

```python
reused = conversion_reuses_ann_training_artifacts(cfg)
```

避免同一变量在函数中出现两次来源。

---

# 3. 保持 Clip 三态语义不变

本次修复不能改变已经确定的 Clip bundle 规则。

继续保持：

```python
clip_bundle_policy = (
    "allow_eligible"
    if reused
    else "forbid_all"
)
```

其含义：

## `reused == True`

即：

```text
phase_aware
gif_aware
```

conversion 复用：

```text
ANN-training calibration bundle
```

该 bundle：

```text
Site 1/2/3/4/6/7/8/9/10
    可以保留 clip_state.pt

Site 5
    永远不存在 clip_state.pt
```

因此：

```python
clip_bundle_policy="allow_eligible"
```

SNN deployment 只加载：

```text
Phase
GIF
MTN
```

不得加载或执行 Clip。

---

## `reused == False`

即：

```text
vanilla
unaware
```

conversion 使用：

```text
Post-finetuning conversion calibration
```

该 bundle 必须完全 clip-free。

因此：

```python
clip_bundle_policy="forbid_all"
```

---

# 4. 补充 `create_conversion()` 回归测试

当前已有测试主要覆盖：

```python
validate_conversion_metadata()
```

但没有真正走：

```python
create_conversion()
```

因此这次 `reused` 未定义错误没有被发现。

必须新增实际调用 `create_conversion()` 的测试。

建议修改：

```text
tests/test_conversion_metadata.py
```

---

# 5. 测试一：至少验证 `create_conversion()` 不再触发 `UnboundLocalError`

构造最小合法 conversion bundle，然后：

```python
metadata = create_conversion(
    cfg,
    layout,
    "phase",
)
```

断言：

```python
assert metadata["deployment_neuron"] == "phase"
```

并：

```python
assert (
    layout.snn_conversion_dir("phase")
    / "conversion_metadata.json"
).exists()
```

这个测试的核心目的：

> 真正执行 `create_conversion()`，防止以后再出现函数内部局部变量顺序错误。

---

# 6. 测试二：覆盖非 aware 的 `forbid_all`

构造：

```text
ann_mode = vanilla 或 unaware
```

确保：

```python
conversion_reuses_ann_training_artifacts(cfg) is False
```

调用：

```python
create_conversion(
    cfg,
    layout,
    "phase",
)
```

应成功。

可以使用 monkeypatch 捕获：

```python
SiteController.set_deployment
```

并断言收到：

```python
clip_bundle_policy == "forbid_all"
```

示意：

```python
captured = {}

def fake_set_deployment(
    self,
    neuron,
    *,
    clip_bundle_policy,
):
    captured["neuron"] = neuron
    captured["clip_bundle_policy"] = (
        clip_bundle_policy
    )
    return 4

monkeypatch.setattr(
    SiteController,
    "set_deployment",
    fake_set_deployment,
)

create_conversion(
    cfg,
    layout,
    "phase",
)

assert (
    captured["clip_bundle_policy"]
    == "forbid_all"
)
```

---

# 7. 测试三：覆盖 aware 的 `allow_eligible`

再构造：

```text
ann_mode = phase_aware
```

或：

```text
ann_mode = gif_aware
```

确保：

```python
conversion_reuses_ann_training_artifacts(cfg) is True
```

调用：

```python
create_conversion(...)
```

并断言：

```python
clip_bundle_policy == "allow_eligible"
```

推荐参数化：

```python
@pytest.mark.parametrize(
    (
        "ann_mode",
        "expected_clip_policy",
    ),
    [
        (
            "phase_aware",
            "allow_eligible",
        ),
        (
            "gif_aware",
            "allow_eligible",
        ),
        (
            "vanilla",
            "forbid_all",
        ),
        (
            "unaware",
            "forbid_all",
        ),
    ],
)
def test_create_conversion_selects_correct_clip_bundle_policy(
    ...
):
    ...
```

如果构造完整 aware provenance 成本过高，可以：

- monkeypatch `_source_bundle()`
- monkeypatch `_ann_num_hidden_layers()`
- monkeypatch `SiteController.set_deployment()`
- 使用最小 fake `ArtifactLayout`

重点必须真正调用：

```python
create_conversion()
```

不能只单独测试 helper。

---

# 8. 测试中必须确认 metadata 使用的是同一个 `reused`

除了检查 `clip_bundle_policy`，还应断言：

```python
metadata[
    "reused_ann_training_artifacts"
]
```

与 mode 一致：

```text
phase_aware -> True
gif_aware   -> True
vanilla     -> False
unaware     -> False
```

这样能确保：

```text
deployment policy
conversion metadata
```

使用的是同一个 `reused` 结果，不会以后再次发生逻辑分叉。

---

# 9. 最低测试命令

先运行：

```bash
pytest -q \
  tests/test_conversion_metadata.py \
  tests/test_controller_state_loading.py \
  tests/test_post_finetuning_protocol.py
```

然后：

```bash
pytest -q
```

---

# 10. 实际 smoke test

修复后至少对一个 aware mode 实际执行：

```bash
python scripts/convert_snn.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml \
  --neuron phase
```

要求：

1. 不再出现：

```text
UnboundLocalError
```

2. aware ANN-training bundle 中即使存在 9 个合法：

```text
clip_state.pt
```

也能通过 conversion。

3. 生成：

```text
conversion_metadata.json
```

4. metadata 中：

```json
{
  "reused_ann_training_artifacts": true,
  "snn_clip_applied": false
}
```

5. deployment validator 使用：

```text
allow_eligible
```

---

# 11. 建议再测一个非 aware mode

如果已有 vanilla/unaware Post-finetuning conversion calibration，可再运行：

```bash
python scripts/convert_snn.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml \
  --neuron phase
```

要求：

```text
reused_ann_training_artifacts = false
```

且使用：

```text
forbid_all
```

任何 stale：

```text
clip_state.pt
```

仍必须报错。

---

# 12. 最终验收标准

只有以下全部满足才算完成：

- [ ] `reused` 在 `create_conversion()` 中第一次使用前赋值；
- [ ] 删除后面重复的 `reused = ...`；
- [ ] aware conversion 仍使用 `allow_eligible`；
- [ ] vanilla/unaware conversion 仍使用 `forbid_all`；
- [ ] conversion metadata 的 `reused_ann_training_artifacts` 与实际 mode 一致；
- [ ] 新测试真正调用 `create_conversion()`；
- [ ] 测试覆盖 aware 和 non-aware 两类 mode；
- [ ] `tests/test_conversion_metadata.py` 通过；
- [ ] `pytest -q` 全通过；
- [ ] `phase_aware convert_snn.py --neuron phase` smoke test 通过。

---

# 13. Codex 最终回复要求

完成后请明确报告：

1. 修改了 `snn2/conversion.py` 哪一处；
2. `reused` 现在在哪里赋值；
3. 删除了哪一个重复赋值；
4. 新增了哪个 `create_conversion()` 回归测试；
5. aware/non-aware 分别使用哪个 Clip bundle policy；
6. `pytest -q` 结果；
7. 实际 `convert_snn.py` smoke test 结果。

不要只回复“修复完成”。
