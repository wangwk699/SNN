# SNN 项目修改方案：移除 Post-finetuning SNN Conversion 对 Common Clipping 的生成与依赖

## 1. 修改目标

当前项目中，`Step 8：生成 Post-finetuning conversion calibration` 复用了 ANN-training calibration 的 state materialization 逻辑，因此会和 Step 5 一样生成 `clip_state.pt`。同时，SNN conversion 的校验逻辑和 `SiteController` 的 state 加载逻辑也把 `clip_state.pt` 当成必需工件。

这与项目算法定义不一致。

正确语义必须是：

- **ANN `phase_aware` 微调**：`Phase local encoder -> common clip -> 后续静态 ANN`。
- **ANN `gif_aware` 微调**：`Static GIF fake quantization -> common clip -> 后续静态 ANN`。
- **SNN conversion / deployment**：只使用所选 `Phase / GIF / MTN` 的 full-temporal neuron realization，**不执行 common clipping，也不需要 `clip_state.pt`**。
- GIF 内部量化公式中的 `clamp(q, qmin, qmax)` 属于 GIF 量化器自身定义，**必须保留**，不要与 common clipping 混淆。

本次修改不得改变 Rotation、Prefix、10 个 activation replacement sites、Phase/GIF/MTN neuron 数学定义以及 ANN 微调时已有的 common clipping 行为。

---

## 2. 当前错误位置

当前主要问题在以下文件：

1. `snn2/calibration.py`
   - `build_site_states()` 无条件构造 `phase_state`、`gif_state`、`mtn_state`、`clip_state`。
   - `materialize_calibration_states()` 无条件保存四种 state。
   - 因此 `post_finetuning_conversion_calibration` 也会生成 `clip_state.pt`。

2. `snn2/controller.py`
   - `SiteController._load()` 当前无条件读取：
     - `phase_state.pt`
     - `gif_state.pt`
     - `mtn_state.pt`
     - `clip_state.pt`
   - 虽然 `deploy_phase/deploy_gif/deploy_mtn` 的实际 `apply()` 路径没有调用 `Clipper`，但 deployment 仍然被迫依赖并读取 `clip_state.pt`。

3. `snn2/conversion.py`
   - `validate_calibration()` 当前要求每个 site 必须存在 `clip_state.pt`。
   - 还会读取并校验 common clipping interval。
   - 这使 conversion artifact protocol 错误地依赖 ANN fine-tuning 专用 clipping state。

4. `tests/test_calibration_topology.py` 等测试目前也把 `clip_state.pt` 视作 conversion calibration 的必需文件。

5. `实验执行总结.md` 的 Step 8 没有明确区分 Step 5 和 conversion calibration 的 state 集合，需要同步修正。

---

## 3. 修改后的 calibration 工件语义

三个 calibration stage 必须严格区分：

| Stage | `statistics.pt` | `phase_state.pt` | `gif_state.pt` | `mtn_state.pt` | `clip_state.pt` |
|---|---:|---:|---:|---:|---:|
| `vanilla_analysis` | 是 | 否 | 否 | 否 | 否 |
| `ann_training` | 是 | 是 | 是 | 是 | **是** |
| `post_finetuning` / conversion calibration | 是 | 是 | 是 | 是 | **否** |

其中：

- `ann_training` calibration 的 `clip_state.pt` 继续用于 `phase_aware` / `gif_aware` ANN fine-tuning。
- `post_finetuning` calibration 只为 Phase/GIF/MTN SNN conversion 重新统计并生成 neuron states，不生成 common clipping interval。

---

# 4. 具体代码修改

## 4.1 修改 `snn2/calibration.py`

### 4.1.1 让 `build_site_states()` 支持是否生成 common clip

当前函数：

```python
build_site_states(statistics, cfg)
```

修改为显式参数，例如：

```python
build_site_states(
    statistics: dict[str, Any],
    cfg: dict[str, Any],
    *,
    include_clip: bool,
) -> dict[str, dict[str, Any]]
```

共同逻辑仍然生成：

```python
states = {
    "phase": phase_state,
    "gif": gif_state,
    "mtn": mtn_state,
}
```

只有 `include_clip=True` 时才计算 common clipping interval：

```python
phase_bound = ...
mtn_bound = ...
gif_lower = ...
gif_upper = ...
lower = ...
upper = ...
```

并进行：

```python
if torch.any(lower >= upper):
    ...

states["clip"] = clip_state
```

`include_clip=False` 时：

- 不计算 common clipping intersection；
- 不做 `lower >= upper` 的 common-clip 合法性检查；
- 返回结果中不存在 `"clip"` key。

注意：

- Phase、GIF、MTN state 的计算必须保持现有实现不变。
- GIF 的 `low_min/low_max/high_min/high_max` 等量化参数仍然需要正常计算，因为这些属于 GIF neuron state，不是 common clipping。

### 4.1.2 让 `materialize_calibration_states()` 区分两种 state profile

修改函数，使其接收 `include_clip`，例如：

```python
def materialize_calibration_states(
    site_root,
    cfg,
    metadata=None,
    *,
    include_clip: bool,
):
```

内部调用：

```python
states = build_site_states(
    statistics,
    cfg,
    include_clip=include_clip,
)
```

然后只保存 `states` 中真实存在的 state。

### 4.1.3 必须清理旧的 stale `clip_state.pt`

这是必须实现的兼容处理。

服务器上可能已经存在旧版 Step 8 结果。如果新的 post-finetuning calibration 只是停止写 `clip_state.pt`，旧文件会继续残留，从结果目录看仍然像 conversion calibration 使用 clipping。

因此，当 `include_clip=False` 时，对每个 site 显式执行等价逻辑：

```python
clip_path = directory / "clip_state.pt"
if clip_path.exists():
    clip_path.unlink()
```

也可以使用兼容当前 Python 版本的 `unlink(missing_ok=True)`。

最终必须保证重新执行：

```bash
python scripts/calibrate_sites.py \
  --config <CFG> \
  --stage post_finetuning
```

以后，`post_finetuning_site_dir/layer_*/site_*/` 下**绝对不存在** `clip_state.pt`。

### 4.1.4 修改 calibration summary / manifest

不要再让 conversion calibration 的 summary 暗示 clipping 有效。

建议在 metadata / manifest 中加入明确字段：

```json
{
  "state_profile": "ann_training_with_common_clip",
  "common_clip_required": true
}
```

对于 post-finetuning conversion calibration：

```json
{
  "state_profile": "snn_conversion_without_common_clip",
  "common_clip_required": false
}
```

对于 vanilla analysis：

```json
{
  "state_profile": "analysis_statistics_only",
  "common_clip_required": false
}
```

每个 site 的 `calibration_summary.json`：

- ANN-training 可以继续记录 `clip_valid: true`；
- Post-finetuning conversion 不应再计算 `clip_valid`，建议写：

```json
"clip_state_present": false
```

或者直接省略 `clip_valid`，但必须有一个明确字段能证明该 calibration 不包含 common clipping state。

不要仅靠文件不存在来表达语义，manifest 也应记录该 protocol。

### 4.1.5 在 `collect_site_statistics()` 中按 purpose 决定 `include_clip`

当前已经有：

```python
eligible_ann = purpose == "ann_training_calibration"
eligible_conversion = purpose == "post_finetuning_conversion_calibration"
```

直接基于该语义决定：

```python
include_clip = eligible_ann
```

即：

```text
ann_training_calibration                  -> include_clip=True
post_finetuning_conversion_calibration    -> include_clip=False
vanilla_analysis_calibration              -> 不 materialize states
```

调用 `materialize_calibration_states()` 时传入该值。

不要通过 `cfg["experiment"]["ann_mode"]` 判断是否生成 clip。是否生成 clip 是 **calibration stage 的属性**，不是 `vanilla/unaware/phase_aware/gif_aware` mode 的属性。

Post-finetuning calibration 无论来源 ANN checkpoint 是：

- vanilla
- unaware
- phase_aware
- gif_aware

都不能生成 `clip_state.pt`。

---

## 4.2 修改 `snn2/controller.py`

这是本次修改的关键项，不能遗漏。

### 4.2.1 `_load()` 不得再无条件加载四种 state

当前逻辑类似：

```python
states = {
    name: torch.load(...)
    for name in ("phase", "gif", "mtn", "clip")
}
```

必须改成**按 controller mode 惰性加载所需 state**。

目标依赖关系：

```text
mode == "phase"       -> phase_state + clip_state
mode == "gif"         -> gif_state + clip_state
mode == "deploy_phase"-> phase_state only
mode == "deploy_gif"  -> gif_state only
mode == "deploy_mtn"  -> mtn_state only
```

`identity`、`none`、`collect` 不应该调用 state loader。

推荐实现为按 site 维护增量 cache：

```python
modules = self._modules.setdefault(key, {})
```

然后根据当前 mode 计算 required state names，只加载 cache 中缺失的 module。

不要继续一次性加载所有 neuron states。

### 4.2.2 ANN training common clipping 行为必须保持原样

以下行为禁止改变：

```python
if self.mode == "phase":
    return modules["clip"](modules["phase"](x))

if self.mode == "gif":
    return modules["clip"](modules["gif"](x))
```

也就是说 ANN `phase_aware` / `gif_aware` 的 common clipping 必须继续存在。

### 4.2.3 SNN deployment 不得加载或调用 `Clipper`

以下 deployment 语义保持：

```python
if self.mode.startswith("deploy_"):
    neuron = self.mode.removeprefix("deploy_")
    output = modules[neuron].temporal(temporal)
```

但必须保证此时 `modules` 中只要求对应 neuron state，不要求 `clip`。

最终 Phase/GIF/MTN SNN evaluation 在整个运行过程中均不应访问 `clip_state.pt`。

### 4.2.4 修正 `set_deployment()` 的 state 查找

当前 `set_deployment()` 先通过查找 `phase_state.pt` 获取第一个 site directory，即使实际 neuron 是 GIF 或 MTN。

顺手修正为 neuron-specific state：

```python
state_name = f"{neuron}_state.pt"
first = next(self.site_root.glob(f"layer_*/site_*/{state_name}"), None)
```

然后直接加载该 neuron 对应 state 读取 temporal steps：

- phase：`phase_state.pt -> T`
- mtn：`mtn_state.pt -> T`
- gif：`gif_state.pt -> 2 ** add_bits`

这样 deployment dependency 与实际 neuron 一致。

---

## 4.3 修改 `snn2/conversion.py`

### 4.3.1 `validate_calibration()` 改为真正的 conversion calibration validation

当前每个 site 要求：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

修改为只要求：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

删除：

- 对 `clip_state.pt` 的 required check；
- `torch.load(clip_state.pt)`；
- `lower < upper` 的 conversion-side common clipping validation。

### 4.3.2 Conversion calibration 中发现 `clip_state.pt` 时必须报错

为了防止旧 Step 8 工件被误用，建议不要简单 ignore 多余 `clip_state.pt`，而是将其视为 stale artifact。

每个 site 做：

```python
clip_path = directory / "clip_state.pt"
if clip_path.exists():
    raise ValueError(
        "Post-finetuning conversion calibration must not contain clip_state.pt; "
        "re-run post_finetuning calibration with the current code."
    )
```

这样可以确保：

- 旧 calibration 不会静默通过；
- 用户必须重新执行 Step 8；
- conversion protocol 从 artifact 层面也明确没有 common clipping。

### 4.3.3 校验 manifest 的 state profile

在 `create_conversion()` 使用 post-finetuning manifest 时，除现有：

```text
purpose == post_finetuning_conversion_calibration
eligible_for_conversion == true
post_finetuning_recalibration == true
```

之外，再要求：

```text
state_profile == snn_conversion_without_common_clip
common_clip_required == false
```

### 4.3.4 Conversion metadata 显式记录无 common clip

建议在 `conversion_metadata.json` 中增加：

```json
"common_clip_applied": false
```

不要记录 `clip_state` 路径或 hash。

---

## 4.4 `snn2/neurons.py`：不要误删 GIF 自身 clamp

本文件原则上**不需要为了本次修复删除 clipping 代码**。

特别注意 `StaticGIF._quantize()` 中：

```python
q = (
    self.round_ste(x.float() / scale.float())
    + zero.float()
).clamp(qmin, qmax)
```

必须保留。

它实现 GIF affine quantization 的合法整数范围：

```text
q in [0, 2^N - 1]
```

这不是本次要删除的 common clipping：

```text
clip(encoded_activation; a_i, b_i)
```

`Clipper` 类本身也不能删除，因为 ANN `phase_aware` / `gif_aware` training 仍然需要它。

---

## 4.5 `scripts/calibrate_sites.py`

命令行接口不要改变：

```bash
python scripts/calibrate_sites.py --config ... --stage ann_training
python scripts/calibrate_sites.py --config ... --stage vanilla_analysis
python scripts/calibrate_sites.py --config ... --stage post_finetuning
```

如果 `collect_site_statistics()` 已经根据 `purpose` 决定 `include_clip`，这里无需增加新的 CLI 参数。

不要增加诸如 `--disable-clip` 的人工开关，因为是否生成 common clip 应由 stage 固定决定，避免用户生成语义错误的 calibration。

---

## 4.6 修改 `scripts/verify_artifacts.py`

保留现有 post-finetuning provenance 校验，同时加入 state profile 校验。

### ANN-training manifest

应要求：

```text
purpose = ann_training_calibration
eligible_for_ann_training = true
eligible_for_conversion = false
state_profile = ann_training_with_common_clip
common_clip_required = true
```

### Post-finetuning manifest

应要求：

```text
purpose = post_finetuning_conversion_calibration
eligible_for_ann_training = false
eligible_for_conversion = true
state_profile = snn_conversion_without_common_clip
common_clip_required = false
```

继续调用 conversion calibration validator，使其确认：

- 10-site topology 正确；
- `statistics/phase/gif/mtn` states 齐全；
- `clip_state.pt` 不存在。

如果 conversion descriptor 新增：

```json
"common_clip_applied": false
```

artifact verifier 也应断言该字段严格为 `false`。

---

# 5. 测试修改

## 5.1 修改 `tests/test_calibration_topology.py`

当前 `_REQUIRED` 包含 `clip_state.pt`，这是旧语义，必须修改。

Conversion calibration topology test 的 required files 改为：

```python
_REQUIRED = (
    "statistics.pt",
    "phase_state.pt",
    "gif_state.pt",
    "mtn_state.pt",
)
```

增加两个测试：

### 测试 A：conversion calibration 没有 clip 可以通过

构造完整 10-site topology，每个 site 只有：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

`validate_calibration()` 必须通过。

### 测试 B：conversion calibration 残留 clip 必须失败

在任意 site 加入：

```text
clip_state.pt
```

`validate_calibration()` 必须报错，并提示这是 stale / invalid post-finetuning conversion calibration。

---

## 5.2 增加 calibration profile 单元测试

建议新增：

```text
tests/test_calibration_profiles.py
```

至少覆盖：

1. `include_clip=True`
   - `build_site_states()` 返回 `phase/gif/mtn/clip`。

2. `include_clip=False`
   - 返回 `phase/gif/mtn`；
   - 不包含 `clip`。

3. post-finetuning materialization 会删除目录中预先存在的旧 `clip_state.pt`。

4. ann-training materialization 仍然保存合法 `clip_state.pt`。

---

## 5.3 增加 `SiteController` dependency 测试

建议新增：

```text
tests/test_controller_state_loading.py
```

必须覆盖：

### Deployment Phase

目录只提供：

```text
phase_state.pt
```

不提供 `clip_state.pt`。

设置：

```python
controller.set_deployment("phase")
```

执行 site replacement 时必须成功，证明 deployment 不读取 clip。

### Deployment GIF

只要求 `gif_state.pt`，不能要求 clip。

### Deployment MTN

只要求 `mtn_state.pt`，不能要求 clip。

### ANN Phase mode

`SiteController(mode="phase")` 若只有 `phase_state.pt` 而没有 `clip_state.pt`，应失败，证明 ANN-training common clipping 仍然是硬依赖。

### ANN GIF mode

同理，`gif_state.pt + clip_state.pt` 才是完整依赖。

---

## 5.4 保留已有 neuron tests

已有 `tests/test_neurons.py` 中 GIF temporal decomposition 等行为不得因本次修改失效。

尤其确保：

```python
temporal.sum(dim=0) == static GIF fake-quant output
```

相关测试继续通过。

---

# 6. 更新 `实验执行总结.md`

## Step 5：ANN-training calibration

在现有说明中明确增加：

```text
ANN-training calibration 会基于训练前 ANN 的 site statistics 生成
Phase、GIF、MTN 参数以及 neuron-independent common clipping interval。
其中 clip_state.pt 仅供 phase_aware / gif_aware ANN fine-tuning 使用。
```

不要改变 Step 5 命令。

## Step 8：Post-finetuning conversion calibration

将说明改成明确与 Step 5 区分：

```text
每个 final ANN checkpoint 独立重新进行 conversion calibration。
该阶段只重新统计 final ANN activation，并生成 Phase、GIF、MTN full-temporal
SNN conversion 所需的 neuron states。

该阶段不计算、不生成、也不使用 ANN fine-tuning 专用的 common clipping interval，
因此 conversion calibration 目录中不存在 clip_state.pt。
```

命令仍然是：

```bash
for CFG in "${ALL_CFGS[@]}"; do
  python scripts/calibrate_sites.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

## Step 10：SNN conversion / evaluation

明确写：

```text
Phase/GIF/MTN SNN conversion 只读取对应的 post-finetuning neuron calibration state。
SNN temporal deployment 不执行 common clipping，也不读取 clip_state.pt。
```

同时说明 GIF neuron 内部的合法量化整数范围 clamp 属于 GIF 算法自身，不属于 common clipping。

---

# 7. 其他文档一致性检查

修改完成后搜索整个仓库：

```bash
rg -n "clip_state|common clip|common clipping|conversion calibration|post_finetuning" \
  . \
  --glob '*.py' \
  --glob '*.md'
```

若 `代码结构总结.md`、`docs/` 或其他项目 Markdown 仍描述：

```text
post-finetuning conversion calibration 必须生成/使用 clip_state
```

需要同步改正。

不要删除 ANN-training common clipping 的文档说明。

---

# 8. 不允许修改的行为

本次任务只修复 calibration / conversion 中错误的 common clipping 依赖。以下内容禁止改变：

1. 10 个 activation replacement site 的位置和编号。
2. Rotation / Hadamard 实现与精度策略。
3. Prefix discovery、fixed KV cache 和四个 `prefix_enabled` 开关语义。
4. ANN `phase_aware` 的 `PhaseSurrogate -> Clipper` 行为。
5. ANN `gif_aware` 的 `StaticGIF -> Clipper` 行为。
6. Phase neuron 的 temporal 算法。
7. GIF neuron 的 temporal decomposition。
8. GIF `_quantize()` 内部的 `[qmin, qmax]` clamp。
9. MTN temporal 算法。
10. ANN/SNN evaluation 的数据选择和指标计算逻辑。
11. `calibrate_sites.py` 的现有三个 `--stage` CLI 名称。

---

# 9. 旧工件迁移要求

修改完成后，旧版 Post-finetuning conversion calibration 不再合法，因为里面存在 `clip_state.pt`。

代码本身必须做到：重新执行 Step 8 时自动删除旧 `clip_state.pt`。

对已有实验 run，至少重新执行：

```bash
python scripts/calibrate_sites.py \
  --config <CFG> \
  --stage post_finetuning
```

然后重新创建 conversion descriptor：

```bash
for NEURON in phase gif mtn; do
  python scripts/convert_snn.py \
    --config <CFG> \
    --neuron "$NEURON"
done
```

因为 calibration manifest 内容发生改变，conversion metadata 中记录的 calibration manifest hash 也必须更新。

当前 SNN runtime 原本没有实际调用 common `Clipper`，因此在 Phase/GIF/MTN neuron states 不变的前提下，本次修复不应改变 SNN 数值前向；它修复的是错误的 artifact 生成、加载和依赖关系。

---

# 10. 验收标准

代码修改完成后必须同时满足以下条件。

### ANN-training calibration

执行：

```bash
python scripts/calibrate_sites.py \
  --config <phase-aware-or-shared-rotated-config> \
  --stage ann_training
```

每个 site 应存在：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

### Post-finetuning conversion calibration

执行：

```bash
python scripts/calibrate_sites.py \
  --config <CFG> \
  --stage post_finetuning
```

每个 site 应只存在：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

且：

```bash
find <post_finetuning_conversion_calibration_root> -name 'clip_state.pt'
```

必须无输出。

### Conversion

以下三条都必须在没有任何 `clip_state.pt` 的 post-finetuning calibration 上成功：

```bash
python scripts/convert_snn.py --config <CFG> --neuron phase
python scripts/convert_snn.py --config <CFG> --neuron gif
python scripts/convert_snn.py --config <CFG> --neuron mtn
```

### SNN evaluation

Phase/GIF/MTN evaluation 不得尝试打开 `clip_state.pt`。

### ANN training regression

`phase_aware` / `gif_aware` training 仍必须使用 common clip，不能因为 deployment 修改而绕过 `Clipper`。

### Tests

至少执行：

```bash
pytest -q
```

全部通过。

---

# 11. 最终实现原则

最终代码必须形成以下清晰边界：

```text
ANN fine-tuning calibration
        |
        +-- Phase state
        +-- GIF state
        +-- MTN state
        +-- Common Clip state
                    |
                    +--> phase_aware ANN: Phase -> Clip
                    +--> gif_aware ANN:   GIF   -> Clip

Final ANN checkpoint
        |
        v
Post-finetuning conversion calibration
        |
        +-- Phase state
        +-- GIF state
        +-- MTN state
        +-- NO clip_state
              |
              +--> Phase full-temporal SNN
              +--> GIF full-temporal SNN
              +--> MTN full-temporal SNN
```

**核心约束：Common clipping 是 conversion-aware ANN fine-tuning 的训练侧约束，不是 Phase/GIF/MTN full-temporal SNN conversion 的 deployment operator。**
