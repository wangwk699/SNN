# SNN：Rotated Pre-finetuning ANN 独立 Prefix 与 TL;DR Evaluation 修改方案

## 1. 目标

在当前项目中，执行：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"
```

后会得到旋转后、ANN 微调前的 checkpoint：

```text
artifacts/<experiment>/<task>/<model>/_shared/seed42/rotated_prefix/rotation/fused_base/
```

现在增加一条**独立评估流程**：

```text
Original Base
    ↓
prepare_rotation.py
    ↓
rotation/fused_base
    ↓
针对 fused_base 独立重新发现 Prefix
    ↓
生成该 checkpoint 专属 fixed Prefix KV cache
    ↓
TL;DR ANN greedy evaluation
    ↓
独立保存 evaluation 结果
```

这条流程用于评估：

> **Rotation 已完成，但尚未进行任何 ANN fine-tuning 的 ANN 模型性能。**

必须保证它与以下现有工件完全隔离：

```text
ann_training_prefix/
post_finetuning/prefix/
<ann_mode>/<lr>/seed42/ann/final/
<ann_mode>/<lr>/seed42/post_finetuning/
```

---

## 2. 核心语义

### 2.1 待评估模型

待评估模型必须是：

```text
layout.rotation_dir / "fused_base"
```

也就是 `prepare_rotation.py` 保存的旋转后、微调前 checkpoint。

不要：

- 加载原始 Hugging Face Base；
- 加载 `ann/final/`；
- 再复制一份新的 checkpoint；
- 对该 checkpoint 做任何训练。

### 2.2 Rotation 执行语义

`fused_base` 只保存已经融合进权重的 rotation 部分。

当前项目中的 online `R3/R4` 仍然依赖：

```text
rotation/rotation_state.pt
```

因此，无论是 Prefix discovery 还是 TL;DR evaluation，都必须像当前 rotated ANN 路径一样：

```python
install_model_integration(
    model,
    controller,
    rotation_state(cfg, layout),
)
```

不能只加载 `fused_base` 后直接 forward，否则不是完整的 rotated model。

### 2.3 Prefix 语义

这里需要的是：

> 使用与当前 post-finetuning Prefix 相同的“针对当前待评估 checkpoint 重新发现 Prefix + 构造 fixed KV cache”的方法，但 checkpoint 来源改为 `rotation/fused_base`。

不要直接复用：

```text
ann_training_prefix/
```

即使二者当前理论上可能得到相同 token，也必须**独立生成、独立保存**，从而让该评估具有自己的 provenance。

也不要写入：

```text
<ann_mode>/<lr>/seed42/post_finetuning/prefix/
```

因为该目录只属于真正的 final fine-tuned ANN。

Prefix 仍然采用项目当前实现：

```text
prefix token discovery
    ↓
build_prefix_key_values(...)
    ↓
prefixed_key_values.pt
    ↓
evaluation 时通过 past_key_values 注入
```

不要把 Prefix token 真正拼接到 TL;DR `input_ids` 前面。

---

## 3. 新增工件目录

在模型级 `_shared` 下新增：

```text
artifacts/<experiment>/<task>/<model>/_shared/seed42/rotated_prefix/
├── rotation/
│   ├── fused_base/
│   ├── rotation_state.pt
│   ├── rotation_regression.json
│   └── rotation_summary.json
│
├── ann_training_prefix/
│
└── rotated_pre_finetuning/
    ├── config/
    │   └── resolved_config.yaml
    ├── logs/
    ├── prefix/
    │   ├── prefix_state.json
    │   └── prefixed_key_values.pt
    └── evaluation/
        └── tldr/
            └── test_samples_<N>[_full]/
                ├── predictions.jsonl
                ├── selection.json
                └── metrics.json
```

这一级必须：

- 不包含 `ann_mode`；
- 不包含 learning rate；
- 只依赖 model / task / seed / rotation policy。

原因是 `rotation/fused_base` 本身就是 model-level shared pre-finetuning checkpoint。

---

# 4. 代码修改

## 4.1 `snn2/artifacts.py`

在 `ArtifactLayout` 中增加独立目录属性。

建议增加：

```python
@property
def rotated_pre_finetuning_dir(self) -> Path:
    return self.shared_model_root / "rotated_prefix" / "rotated_pre_finetuning"

@property
def rotated_pre_finetuning_config_dir(self) -> Path:
    return self.rotated_pre_finetuning_dir / "config"

@property
def rotated_pre_finetuning_logs_dir(self) -> Path:
    return self.rotated_pre_finetuning_dir / "logs"

@property
def rotated_pre_finetuning_prefix_dir(self) -> Path:
    return self.rotated_pre_finetuning_dir / "prefix"

@property
def rotated_pre_finetuning_evaluation_dir(self) -> Path:
    return self.rotated_pre_finetuning_dir / "evaluation"
```

在 `ensure()` 中创建这些目录。

注意：

```text
rotated_pre_finetuning_dir
```

必须从：

```python
self.shared_model_root
```

构造，不能从：

```python
self.root
```

构造。

否则会错误地进入：

```text
<ann_mode>/<lr>/seed42/
```

---

## 4.2 `scripts/_common.py`

给 `setup()` 增加一个新的 config scope：

```python
elif config_scope == "rotated_pre_finetuning":
    config_dir = layout.rotated_pre_finetuning_config_dir
```

这样该流程自己的：

```text
resolved_config.yaml
```

写入：

```text
_shared/.../rotated_prefix/rotated_pre_finetuning/config/
```

不要覆盖现有 run-level config。

---

## 4.3 `snn2/modeling.py`

新增 stage：

```text
rotated_pre_finetuning
```

### `model_source_for_stage()`

增加：

```python
if stage == "rotated_pre_finetuning":
    return str(layout.rotation_dir / "fused_base")
```

最终 stage 语义应至少保持：

```text
ann_training              -> rotation/fused_base
vanilla_analysis          -> original HF Base
base_evaluation           -> original HF Base
rotated_pre_finetuning    -> rotation/fused_base
post_finetuning           -> ann/final
```

### `prefix_ids_for_stage()`

增加：

```python
elif stage == "rotated_pre_finetuning":
    if not bool(cfg["rotation"]["enabled"]):
        return []
    path = (
        layout.rotated_pre_finetuning_prefix_dir
        / "prefix_state.json"
    )
```

然后继续使用现有逻辑读取：

```python
state["prefix_token_ids"]
```

### `prefix_key_values_for_stage()`

增加对应目录选择：

```python
elif stage == "rotated_pre_finetuning":
    directory = layout.rotated_pre_finetuning_prefix_dir
```

并继续强制：

```text
非空 prefix_token_ids
    =>
prefixed_key_values.pt 必须存在
```

不能 fallback 到 `ann_training_prefix_dir`。

---

## 4.4 `scripts/discover_prefix.py`

扩展：

```python
--stage
```

choices，从：

```python
("ann_training", "post_finetuning")
```

改为：

```python
(
    "ann_training",
    "rotated_pre_finetuning",
    "post_finetuning",
)
```

### setup scope

规则：

```text
ann_training
    -> policy_shared

rotated_pre_finetuning
    -> rotated_pre_finetuning

post_finetuning
    -> run
```

### 新 stage 的前置检查

`rotated_pre_finetuning` 必须检查：

```python
cfg["rotation"]["enabled"] is True
```

并且至少确认：

```text
layout.rotation_dir / "fused_base" / "config.json"
layout.rotation_dir / "rotation_state.pt"
```

存在。

不存在时直接报错，并提示先执行：

```bash
python scripts/prepare_rotation.py --config ...
```

不要自动运行 rotation preparation。

### 输出目录

新 stage 使用：

```python
output_dir = layout.rotated_pre_finetuning_prefix_dir
```

### 模型来源

通过：

```python
source = model_source_for_stage(
    cfg,
    layout,
    stage="rotated_pre_finetuning",
)
```

获得：

```text
rotation/fused_base
```

### Prefix discovery

保持现有 discovery 算法完全不变：

```python
model = load_model(...)
tokenizer = load_tokenizer(...)

install_model_integration(
    model,
    SiteController(mode="identity"),
    rotation_state(cfg, layout),
)

state = discover_prefix_tokens(
    model,
    tokenizer,
    load_selected_raw(cfg, layout).calibration,
    cfg,
    output_dir / "prefix_state.json",
)

values = build_prefix_key_values(
    model,
    state["prefix_token_ids"],
)
```

并保存：

```text
rotated_pre_finetuning/prefix/prefix_state.json
rotated_pre_finetuning/prefix/prefixed_key_values.pt
```

如果 Prefix 为空，则沿用当前逻辑删除旧的：

```text
prefixed_key_values.pt
```

### 日志

使用：

```python
layout.rotated_pre_finetuning_logs_dir
```

StageRun 名称建议：

```text
discover_prefix_rotated_pre_finetuning
```

---

## 4.5 `scripts/evaluate_tldr.py`

增加 CLI：

```bash
--rotated-pre-finetuning
```

含义：

> Evaluate the rotated fused Base checkpoint before ANN fine-tuning.

### CLI 互斥

`--base` 与 `--rotated-pre-finetuning` 必须互斥。

建议直接使用 argparse mutually exclusive group。

两者都只能：

```text
--neuron ann
```

因此：

```bash
--rotated-pre-finetuning --neuron phase
```

必须报错。

### Rotation 条件

当：

```text
--rotated-pre-finetuning
```

时必须要求：

```python
cfg["rotation"]["enabled"] is True
```

vanilla config 必须拒绝。

### setup scope

选择：

```text
--base
    -> base

--rotated-pre-finetuning
    -> rotated_pre_finetuning

normal ANN/SNN evaluation
    -> run
```

### model source

改为三路：

```python
if args.base:
    source = model_source_for_stage(
        cfg,
        layout,
        stage="base_evaluation",
    )
elif args.rotated_pre_finetuning:
    source = model_source_for_stage(
        cfg,
        layout,
        stage="rotated_pre_finetuning",
    )
else:
    source = model_source_for_stage(
        cfg,
        layout,
        stage="post_finetuning",
    )
```

不要在这里硬编码 `layout.rotation_dir / "fused_base"`；统一走 stage resolver。

### Controller

rotated pre-finetuning ANN 不使用 conversion calibration：

```text
site_root = None
mode = identity
```

即不能读取：

```text
layout.post_finetuning_site_dir
```

因为这时根本不存在 final ANN post-finetuning conversion calibration。

### Model integration

rotated pre-finetuning 必须安装：

```python
install_model_integration(
    model,
    controller,
    rotation_state(cfg, layout),
)
```

这样 online R3/R4 才存在。

### Prefix

stage 选择：

```python
if args.base:
    prefix_stage = "base_evaluation"
elif args.rotated_pre_finetuning:
    prefix_stage = "rotated_pre_finetuning"
else:
    prefix_stage = "post_finetuning"
```

然后统一：

```python
prefixes = prefix_ids_for_stage(
    cfg,
    layout,
    stage=prefix_stage,
)

install_prefix_kv_forward(
    model,
    prefix_key_values_for_stage(
        cfg,
        layout,
        stage=prefix_stage,
    ),
)
```

对 rotated pre-finetuning 来说，必须从：

```text
rotated_pre_finetuning/prefix/
```

读取。

不能读取：

```text
ann_training_prefix/
```

也不能读取：

```text
post_finetuning/prefix/
```

---

## 4.6 TL;DR evaluation 输出目录

当前代码最后根据 model variant 选择 output root。

增加第三类：

```python
if args.base:
    model_output_dir = layout.base_dir

elif args.rotated_pre_finetuning:
    model_output_dir = layout.rotated_pre_finetuning_dir

elif args.neuron == "ann":
    model_output_dir = layout.ann_dir

else:
    model_output_dir = layout.snn_dir(args.neuron)
```

最终结果应写到：

```text
_shared/seed42/rotated_prefix/rotated_pre_finetuning/
└── evaluation/
    └── tldr/
        └── test_samples_<N>[_full]/
            ├── predictions.jsonl
            ├── selection.json
            └── metrics.json
```

继续复用当前：

```python
resolve_tldr_evaluation_layout(...)
```

以及当前的：

```text
tldr_test_samples
tldr_test_seed
batch_size
greedy generation
ROUGE
```

逻辑，不要复制出第二套 TL;DR evaluator。

---

# 5. `metrics.json` 必须正确标记模型身份

rotated pre-finetuning evaluation 至少写入：

```json
{
  "neuron": "ann",
  "model_variant": "rotated_pre_finetuning_ann",
  "checkpoint_stage": "rotated_pre_finetuning",
  "prefix_stage": "rotated_pre_finetuning",
  "post_finetuning_recalibration": false,
  "calibration_root": null,
  "rotation_enabled": true,
  "model_source": ".../_shared/seed42/rotated_prefix/rotation/fused_base",
  "prefix_root": ".../_shared/seed42/rotated_prefix/rotated_pre_finetuning/prefix"
}
```

同时保留当前已有：

```text
ROUGE
samples
test sampling metadata
batch_size
world_size
decode
input_length
max_new_tokens
forward counters
site topology metadata
```

注意：

```text
post_finetuning_recalibration = false
calibration_root = null
```

是正确语义，因为本次只评估 ANN，不做 SNN conversion。

不要为了这次 ANN evaluation 生成 conversion calibration。

---

# 6. 前置依赖检查

rotated pre-finetuning Prefix discovery 至少检查：

```text
rotation/fused_base/config.json
rotation/rotation_state.pt
```

rotated pre-finetuning TL;DR evaluation 至少检查：

```text
rotation/fused_base/config.json
rotation/rotation_state.pt
rotated_pre_finetuning/prefix/prefix_state.json
```

如果 `prefix_state.json` 中：

```text
prefix_token_ids != []
```

则还必须检查：

```text
rotated_pre_finetuning/prefix/prefixed_key_values.pt
```

缺失时直接报错，提示先运行：

```bash
python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage rotated_pre_finetuning
```

不要静默退回无 Prefix evaluation。

---

# 7. 测试

## 7.1 Artifact path 测试

扩展现有测试或新建：

```text
tests/test_rotated_pre_finetuning_protocol.py
```

必须验证：

```python
layout.rotated_pre_finetuning_dir
layout.rotated_pre_finetuning_prefix_dir
layout.rotated_pre_finetuning_logs_dir
layout.rotated_pre_finetuning_config_dir
```

全部位于：

```text
_shared/.../rotated_prefix/rotated_pre_finetuning/
```

### 必须验证共享性

分别构造：

```text
unaware + lr=5e-6
phase_aware + lr=1e-6
gif_aware + 其它 lr
```

只要：

```text
experiment
task
model
seed
rotation policy
```

相同，则：

```python
rotated_pre_finetuning_dir
```

必须完全相同。

也就是说该路径不能受：

```text
ann_mode
learning_rate
```

影响。

---

## 7.2 Stage resolver 测试

验证：

```python
model_source_for_stage(
    cfg,
    layout,
    stage="rotated_pre_finetuning",
)
```

严格等于：

```text
layout.rotation_dir / "fused_base"
```

验证 rotated pre-finetuning Prefix resolver 只读取：

```text
layout.rotated_pre_finetuning_prefix_dir
```

并且与：

```text
ann_training_prefix_dir
post_finetuning_prefix_dir
```

完全分离。

---

## 7.3 Prefix KV 完整性测试

覆盖：

1. `prefix_state.json` 不存在 -> 报错；
2. Prefix 为空 -> 不要求 KV；
3. Prefix 非空且 KV 缺失 -> 报错；
4. Prefix 非空且 KV 存在 -> 正常加载。

不要允许 fallback 到其它 Prefix stage。

---

## 7.4 CLI 语义测试

至少覆盖：

```text
--base + --rotated-pre-finetuning
    -> 拒绝

--rotated-pre-finetuning + --neuron phase/gif/mtn
    -> 拒绝

vanilla config + --rotated-pre-finetuning
    -> 拒绝

rotation/fused_base 缺失
    -> 清晰报错
```

---

## 7.5 回归测试

必须保证原有行为不变：

```text
TL;DR Base evaluation
final ANN evaluation
Phase/GIF/MTN SNN evaluation
ann_training Prefix
post_finetuning Prefix
post_finetuning conversion calibration
```

执行：

```bash
pytest -q
```

全部通过。

---

# 8. 文档同步

更新：

```text
代码结构总结.md
实验执行总结.md
```

加入一条新的**可选分析评估流程**：

```text
prepare_rotation
    ↓
rotated pre-finetuning fused_base
    ↓
independent rotated-pre-finetuning Prefix
    ↓
TL;DR ANN evaluation
```

明确说明：

- 这不是四种 ANN fine-tuning 中的一种新 mode；
- 不增加第五种 `ann_mode`；
- 不进入 SNN conversion；
- 不产生 conversion calibration；
- 主要用于量化“仅 Rotation + Prefix、尚未 fine-tuning”时的 TL;DR 性能；
- 结果属于 model-level shared artifact。

---

# 9. 修改完成后的标准执行命令

## Step 1：生成 rotated pre-finetuning checkpoint

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"
```

得到：

```text
_shared/seed42/rotated_prefix/rotation/fused_base/
```

## Step 2：为该 checkpoint 独立生成 Prefix

```bash
python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage rotated_pre_finetuning
```

得到：

```text
_shared/seed42/rotated_prefix/rotated_pre_finetuning/prefix/
├── prefix_state.json
└── prefixed_key_values.pt
```

如果检测到空 Prefix，则可以只有：

```text
prefix_state.json
```

## Step 3：执行 TL;DR ANN evaluation

单 GPU 示例：

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$ROT_CFG" \
  --neuron ann \
  --rotated-pre-finetuning
```

评估结果写到：

```text
_shared/seed42/rotated_prefix/rotated_pre_finetuning/
└── evaluation/
    └── tldr/
        └── test_samples_<N>[_full]/
            ├── predictions.jsonl
            ├── selection.json
            └── metrics.json
```

`evaluation.tldr_test_samples`、`evaluation.tldr_test_seed`、`evaluation.batch_size` 等继续直接使用 YAML 中现有配置。

---

# 10. 明确禁止的实现

Codex 修改代码时不要采用以下方案：

1. **不要直接使用 `ann_training_prefix/` 做本次 evaluation。**
2. **不要把新 Prefix 写入 run-specific `post_finetuning/prefix/`。**
3. **不要把 evaluation 结果写入 `unaware/lr.../seed42/ann/`。**
4. **不要增加新的 ANN training mode。**
5. **不要重新训练或复制 `fused_base`。**
6. **不要为此次 ANN evaluation 生成 conversion calibration。**
7. **不要修改现有 final ANN/SNN evaluation 的 checkpoint 或 Prefix 选择语义。**
8. **不要只加载 `fused_base` 而漏掉 online R3/R4 的 `rotation_state.pt`。**
9. **不要把 Prefix token 拼接进 TL;DR `input_ids`；继续使用 fixed KV injection。**
10. **不要让缺失 rotated-pre-finetuning Prefix 时静默退回无 Prefix evaluation。**

---

# 11. 最终验收标准

修改完成后，以下流程必须可以从头独立执行：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"

python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage rotated_pre_finetuning

CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config "$ROT_CFG" \
  --neuron ann \
  --rotated-pre-finetuning
```

并满足：

```text
模型：
rotation/fused_base

Rotation：
fused weights + online R3/R4

Prefix：
针对 rotation/fused_base 独立重新发现
并使用独立 fixed KV cache

Fine-tuning：
无

Activation replacement：
无，ANN identity

Conversion calibration：
无

Evaluation：
TL;DR greedy generation + ROUGE

工件归属：
model-level _shared

与 ann_training / final ANN / post_finetuning：
完全隔离
```

最后运行：

```bash
pytest -q
```

现有测试与新增测试全部通过后，修改才算完成。
