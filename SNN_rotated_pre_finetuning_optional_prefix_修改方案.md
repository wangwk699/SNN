# Rotated Pre-finetuning ANN Evaluation：可选 Prefix 修改方案

## 1. 修改目标

仅针对：

```bash
scripts/evaluate_tldr.py \
  --config "$ROT_CFG" \
  --neuron ann \
  --rotated-pre-finetuning
```

增加一个**独立的 Prefix 开关**，使 Rotation 后、ANN 微调前的 TL;DR ANN evaluation 可以分别测试：

```text
Rotation + Prefix
Rotation only（不使用 Prefix）
```

该开关不得复用现有：

```yaml
prefix:
  enabled: true
```

因为 `unaware / phase_aware / gif_aware` 的 `prefix.enabled` 属于 ANN-training 协议，`resolve_config()` 会按 `ann_mode` 强制设置，不能用它控制本次独立的 rotated-pre-finetuning evaluation。

新增独立配置：

```yaml
rotated_pre_finetuning:
  prefix_enabled: true
```

之后只需修改：

```yaml
rotated_pre_finetuning:
  prefix_enabled: true   # 使用 Prefix
```

或：

```yaml
rotated_pre_finetuning:
  prefix_enabled: false  # 不使用 Prefix
```

即可控制 `--rotated-pre-finetuning` 是否使用 Prefix。

---

## 2. `configs/experiment_matrix.yaml`

在 `defaults` 中新增：

```yaml
rotated_pre_finetuning:
  prefix_enabled: true
```

默认保持当前行为，即 rotated-pre-finetuning evaluation 使用 Prefix。

重新执行：

```bash
python scripts/materialize_configs.py
```

后，所有 generated config 都应包含这一字段。

**不要修改现有 `prefix.enabled`、`post_finetuning.prefix_enabled` 的语义。**

---

## 3. `snn2/config.py`

新增 helper：

```python
def rotated_pre_finetuning_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("rotated_pre_finetuning", {}).get(
            "prefix_enabled",
            True,
        )
    )
```

为兼容旧 config，在 `resolve_config()` 中补默认值：

```python
cfg.setdefault("rotated_pre_finetuning", {})
cfg["rotated_pre_finetuning"].setdefault(
    "prefix_enabled",
    True,
)
```

不要在 `ann_mode == "unaware"` 等分支中覆盖这个值。

因此：

```yaml
experiment:
  ann_mode: unaware

prefix:
  enabled: true

rotated_pre_finetuning:
  prefix_enabled: false
```

必须是合法配置，并表示：

```text
ANN-training Prefix：enabled
rotated-pre-finetuning evaluation Prefix：disabled
```

---

## 4. `snn2/modeling.py`

修改：

```python
prefix_ids_for_stage(...)
```

中：

```python
stage == "rotated_pre_finetuning"
```

的逻辑。

当前逻辑只检查 `rotation.enabled`。改为同时检查新的独立开关：

```python
elif stage == "rotated_pre_finetuning":
    if (
        not bool(cfg["rotation"]["enabled"])
        or not rotated_pre_finetuning_prefix_enabled(cfg)
    ):
        return []
    path = (
        layout.rotated_pre_finetuning_prefix_dir
        / "prefix_state.json"
    )
```

这样：

```text
prefix_enabled=true
    → 读取 rotated_pre_finetuning/prefix/prefix_state.json
    → 非空 Prefix 时读取 prefixed_key_values.pt

prefix_enabled=false
    → prefix_ids_for_stage() 返回 []
    → prefix_key_values_for_stage() 返回 None
    → 不安装 Prefix KV cache
```

即使目录中残留以前生成的：

```text
prefix_state.json
prefixed_key_values.pt
```

当 `prefix_enabled=false` 时也必须**完全忽略**，不能读取或注入。

---

## 5. `scripts/discover_prefix.py`

对于：

```bash
--stage rotated_pre_finetuning
```

增加对：

```python
rotated_pre_finetuning_prefix_enabled(cfg)
```

的判断。

要求：

- `prefix_enabled=true`：保持当前行为，正常 discover Prefix 并生成 fixed KV cache。
- `prefix_enabled=false`：该阶段不需要 Prefix discovery；不得因为缺少 Prefix 工件影响后续 evaluation。

建议在显式执行：

```bash
python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage rotated_pre_finetuning
```

且 `prefix_enabled=false` 时直接记录 `prefix_disabled` event 后正常退出，不删除已有 Prefix 工件。

**不要删除已有 Prefix 工件。**  
因为同一个 `fused_base` 可能需要分别执行 enabled / disabled 两组对照实验；disabled 模式只需要忽略这些工件。

---

## 6. `scripts/evaluate_tldr.py`

### 6.1 Dependency validation

修改：

```python
_validate_rotated_pre_finetuning_dependencies(...)
```

逻辑。

无论 Prefix 是否开启，始终要求：

```text
rotation/fused_base/config.json
rotation/rotation_state.pt
```

当：

```yaml
rotated_pre_finetuning:
  prefix_enabled: true
```

时，额外要求：

```text
rotated_pre_finetuning/prefix/prefix_state.json
```

如果 `prefix_token_ids` 非空，再要求：

```text
rotated_pre_finetuning/prefix/prefixed_key_values.pt
```

当：

```yaml
rotated_pre_finetuning:
  prefix_enabled: false
```

时：

```text
不检查 prefix_state.json
不检查 prefixed_key_values.pt
```

因此关闭 Prefix 后，只执行：

```bash
python scripts/prepare_rotation.py \
  --config "$ROT_CFG"

CUDA_VISIBLE_DEVICES=6 accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$ROT_CFG" \
  --neuron ann \
  --rotated-pre-finetuning
```

即可，不需要先运行 `discover_prefix.py --stage rotated_pre_finetuning`。

### 6.2 Prefix 注入

`--rotated-pre-finetuning` evaluation 继续调用：

```python
prefix_ids_for_stage(...)
prefix_key_values_for_stage(...)
install_prefix_kv_forward(...)
```

但由新的 config helper 保证：

```text
prefix_enabled=true
    → 正常使用 fixed Prefix KV cache

prefix_enabled=false
    → prefix ids = []
    → prefix KV = None
    → install_prefix_kv_forward() 不进行任何 Prefix 注入
```

Rotation 部分保持不变：

```text
fused_base
+
rotation_state.pt
+
online R3/R4
```

所以 disabled 实验必须严格表示：

```text
Rotated Pre-finetuning ANN
= fused Rotation + online R3/R4 + no Prefix
```

---

## 7. TL;DR 输出路径必须区分 Prefix 开关

当前：

```text
.../evaluation/tldr/test_samples_128/
```

或：

```text
.../evaluation/tldr/test_samples_6553_full/
```

改为仅对 `--rotated-pre-finetuning` 追加一级目录：

### Prefix 开启

```text
.../evaluation/tldr/test_samples_128/prefix_enabled_ture/
```

完整 test：

```text
.../evaluation/tldr/test_samples_6553_full/prefix_enabled_ture/
```

### Prefix 关闭

```text
.../evaluation/tldr/test_samples_128/prefix_enabled_false/
```

完整 test：

```text
.../evaluation/tldr/test_samples_6553_full/prefix_enabled_false/
```

> 按本方案要求，目录名保留 `prefix_enabled_ture` 这一拼写，不改成 `prefix_enabled_true`。

目录内仍然保存：

```text
predictions.jsonl
selection.json
metrics.json
```

只修改 `--rotated-pre-finetuning` 的输出路径。

以下评估路径**全部保持原样**：

```text
--base
final ANN
Phase SNN
GIF SNN
MTN SNN
```

建议在 `snn2/evaluation.py` 增加一个简单 helper，避免字符串散落：

```python
def rotated_pre_finetuning_prefix_dirname(
    enabled: bool,
) -> str:
    return (
        "prefix_enabled_ture"
        if enabled
        else "prefix_enabled_false"
    )
```

然后在 `evaluate_tldr.py` 构造：

```python
output_dir = (
    model_output_dir
    / "evaluation"
    / "tldr"
    / test_samples_dirname
)

if args.rotated_pre_finetuning:
    output_dir = (
        output_dir
        / rotated_pre_finetuning_prefix_dirname(
            rotated_pre_finetuning_prefix_enabled(cfg)
        )
    )
```

---

## 8. `metrics.json` 元数据

对于 `--rotated-pre-finetuning`，增加：

```json
"prefix_enabled": true
```

或：

```json
"prefix_enabled": false
```

并调整 `prefix_root`：

```text
prefix_enabled=true
    → rotated_pre_finetuning_prefix_dir

prefix_enabled=false
    → null
```

保持：

```json
"prefix_stage": "rotated_pre_finetuning",
"checkpoint_stage": "rotated_pre_finetuning",
"post_finetuning_recalibration": false,
"calibration_root": null
```

这样即使单独拿到 `metrics.json`，也可以明确判断这次结果是否使用 Prefix。

---

## 9. 测试

至少补充以下测试。

### `tests/test_evaluation_paths.py`

保留现有：

```text
test_samples_128
test_samples_6553_full
```

的基础 layout 测试，同时新增：

```text
True  → prefix_enabled_ture
False → prefix_enabled_false
```

并验证最终 rotated-pre-finetuning 路径分别为：

```text
test_samples_128/prefix_enabled_ture
test_samples_128/prefix_enabled_false
test_samples_6553_full/prefix_enabled_ture
test_samples_6553_full/prefix_enabled_false
```

### Prefix stage 测试

验证：

```text
rotated_pre_finetuning.prefix_enabled=true
    → prefix_ids_for_stage() 读取对应 prefix_state

rotated_pre_finetuning.prefix_enabled=false
    → 即使 prefix_state / prefixed_key_values.pt 已存在
      prefix_ids_for_stage() 仍返回 []
      prefix_key_values_for_stage() 仍返回 None
```

### Config 测试

验证 `unaware`：

```yaml
prefix:
  enabled: true

rotated_pre_finetuning:
  prefix_enabled: false
```

经过 `resolve_config()` 后仍然保持：

```text
prefix.enabled == true
rotated_pre_finetuning.prefix_enabled == false
```

---

## 10. 文档同步

更新：

```text
代码结构总结.md
实验执行总结.md
```

将原先“rotated-pre-finetuning 必须使用独立 Prefix”的表述改为：

```text
rotated-pre-finetuning Prefix 可通过
rotated_pre_finetuning.prefix_enabled
独立开启或关闭。

enabled=true：
使用该 fused_base 独立 discover 的 Prefix / fixed KV cache。

enabled=false：
只评估 fused Rotation + online R3/R4，不读取或注入任何 Prefix。
```

并补充新的结果路径：

```text
evaluation/tldr/test_samples_<N>[_full]/
├── prefix_enabled_ture/
└── prefix_enabled_false/
```

---

## 11. 修改后的使用方式

### Rotation + Prefix

配置：

```yaml
rotated_pre_finetuning:
  prefix_enabled: true
```

执行：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"

python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage rotated_pre_finetuning

CUDA_VISIBLE_DEVICES=6 accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$ROT_CFG" \
  --neuron ann \
  --rotated-pre-finetuning
```

输出：

```text
.../test_samples_128/prefix_enabled_ture/
```

或：

```text
.../test_samples_6553_full/prefix_enabled_ture/
```

### Rotation only，不使用 Prefix

只修改：

```yaml
rotated_pre_finetuning:
  prefix_enabled: false
```

然后执行：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"

CUDA_VISIBLE_DEVICES=6 accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$ROT_CFG" \
  --neuron ann \
  --rotated-pre-finetuning
```

不需要执行：

```bash
discover_prefix.py --stage rotated_pre_finetuning
```

输出：

```text
.../test_samples_128/prefix_enabled_false/
```

或：

```text
.../test_samples_6553_full/prefix_enabled_false/
```

---

## 12. 不允许改动的行为

本次修改只作用于：

```text
--rotated-pre-finetuning
```

不得改变：

```text
vanilla ANN training
unaware ANN training
phase_aware ANN training
gif_aware ANN training

ANN-training Prefix
post-finetuning Prefix
post-finetuning conversion calibration

Base evaluation
final ANN evaluation
SNN evaluation

Rotation fusion
online R3/R4 Hadamard implementation
```

尤其不要通过修改 `resolve_config()` 中 `unaware` 的：

```python
cfg["prefix"]["enabled"] = True
```

来实现本功能。
