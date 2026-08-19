# SNN 项目本轮修改方案：分离 ANN-training calibration 与 Post-finetuning conversion calibration

## 0. 本文档目的

本文档用于指导部署在服务器上的 Codex 在**没有其他上下文**的情况下完成本轮代码与文档修改。

本轮修改的核心目标是：

1. 将当前项目中“ANN 微调前 calibration”和“SNN 转换前 calibration”彻底分离。
2. ANN 微调阶段继续使用 pre-finetuning 的 Prefix / calibration。
3. ANN 评估、SNN 转换与 SNN 评估阶段统一使用 **post-finetuning Prefix + post-finetuning conversion calibration**。
4. 四条 ANN 分支 `vanilla / unaware / phase_aware / gif_aware` 在得到 final ANN checkpoint 之后进入完全一致的 post-finetuning pipeline。
5. 修改代码后，必须同步更新项目 Markdown 文档，尤其是 `实验执行总结.md`，使其明确说明新的运行流程。

---

## 1. 当前项目背景

当前项目中主要实验流程大致为：

```text
prepare_data
  -> prepare_rotation
  -> discover_prefix
  -> calibrate_sites
  -> train_ann
  -> convert_snn
  -> evaluate_tldr / evaluate_lm_harness
```

当前主要问题是：

```text
Base 阶段生成的一套 prefix / calibration
        ↓
同时用于 ANN training、SNN conversion、SNN evaluation
```

这会导致：

1. SNN conversion 使用的是 **fine-tuning 前 Base 模型** 的 activation statistics。
2. `conversion_metadata.json` 中记录 `"post_finetuning_recalibration": false`。
3. `unaware / phase_aware / gif_aware` 三条 rotated 分支共享同一套 calibration。
4. `vanilla_original/calibration/` 在语义上容易被误解为 Vanilla → SNN conversion 使用的 calibration。

本轮修改后，必须变为：

```text
ANN 微调前：
  使用 pre-finetuning prefix / calibration
  仅服务于 ANN training

ANN 微调后：
  每个 final ANN checkpoint 独立 rediscover prefix
  每个 final ANN checkpoint 独立 recalibrate 10 个 site
  ANN evaluation / SNN conversion / SNN evaluation 全部使用 post-finetuning prefix
```

---

## 2. 新实验协议的核心定义

本轮修改后，必须严格区分三类 calibration。

### 2.1 Vanilla analysis calibration

英文名称：

```text
Vanilla analysis calibration
```

中文含义：

```text
Vanilla 原始坐标分析用 calibration
```

对象：

```text
Original pretrained Base model
```

条件：

```text
no rotation
no prefix
no activation replacement
```

目的：

```text
仅用于分析 rotation + prefix 技术前后，相同 10 个 activation replacement site 的统计分布变化。
```

它不用于：

```text
ANN 微调
SNN conversion
SNN evaluation
```

建议 manifest 中明确写入：

```yaml
purpose: vanilla_analysis_calibration
analysis_only: true
eligible_for_ann_training: false
eligible_for_conversion: false
```

如果代码中仍然为了复用函数生成了 `phase_state.pt / gif_state.pt / mtn_state.pt / clip_state.pt`，也必须在 manifest 中标记它们 **not eligible for conversion**。更推荐只保存 statistics，不生成 neuron states。

---

### 2.2 ANN-training calibration

英文名称：

```text
ANN-training calibration
```

中文含义：

```text
ANN 微调用 calibration
```

对象：

```text
Rotated / fused Base model
```

条件：

```text
rotation enabled
ANN-training Prefix enabled
fixed ANN-training Prefix KV cache
```

目的：

```text
仅用于 phase_aware / gif_aware 的 ANN 微调阶段。
```

使用方式：

```text
phase_aware:
  phase_state.pt + clip_state.pt

gif_aware:
  gif_state.pt + clip_state.pt

unaware:
  不使用 calibration state

vanilla:
  不使用 ANN-training calibration
```

注意：

```text
unaware 训练阶段使用 rotation + ANN-training Prefix，但不使用 activation replacement，因此不读取 ANN-training calibration state。
```

---

### 2.3 Post-finetuning conversion calibration

英文名称：

```text
Post-finetuning conversion calibration
```

中文含义：

```text
微调后 SNN 转换用 calibration
```

对象：

```text
每一个 final ANN checkpoint
```

覆盖四条分支：

```text
vanilla
unaware
phase_aware
gif_aware
```

条件：

```text
每个 final ANN checkpoint 独立 rediscover Prefix
每个 final ANN checkpoint 独立构造 fixed Prefix KV cache
每个 final ANN checkpoint 独立统计 10 个 site
```

目的：

```text
用于 ANN evaluation、Phase/GIF/MTN SNN conversion、Phase/GIF/MTN SNN evaluation。
```

最重要的语义变化：

```text
ANN evaluation 也必须使用 post-finetuning Prefix。
SNN evaluation 必须使用同一个 post-finetuning Prefix。
```

即：

```text
Final ANN checkpoint
        ↓
rediscover post-finetuning Prefix
        ↓
fixed post-finetuning Prefix KV
        ├── ANN evaluation
        └── post-finetuning conversion calibration
                ↓
            SNN conversion
                ↓
            SNN evaluation
```

---

## 3. Vanilla 分支的新定义

训练阶段：

```text
vanilla ANN training
= no rotation
+ no prefix
+ no activation replacement
```

但得到 final checkpoint 之后，Vanilla 分支必须和其他三条分支保持相同行为：

```text
Vanilla final ANN checkpoint
        ↓
rediscover post-finetuning Prefix
        ↓
fixed post-finetuning Prefix KV
        ↓
post-finetuning conversion calibration
        ↓
ANN evaluation
        ↓
Phase/GIF/MTN SNN conversion and evaluation
```

唯一不同点：

```text
Vanilla post-finetuning pipeline 不使用 rotation。
```

本轮修改中 **不要额外添加不带 Prefix 的 Vanilla final ANN evaluation**。

---

## 4. 新目录结构

必须将共享状态和 run-specific 状态严格分离。

### 4.1 共享目录

建议目录结构：

```text
artifacts/<experiment>/<task>/<model>/
└── _shared/seed42/
    ├── vanilla_original/
    │   └── vanilla_analysis_calibration/
    │       └── sites/
    │
    └── rotated_prefix/
        ├── rotation/
        │   ├── rotation_state.pt
        │   └── fused_base/
        │
        ├── ann_training_prefix/
        │   ├── prefix_state.json
        │   └── prefixed_key_values.pt
        │
        └── ann_training_calibration/
            └── sites/
```

含义：

```text
vanilla_analysis_calibration:
  原始 Base、无 rotation、无 prefix，仅用于分析。

ann_training_prefix:
  在 rotated/fused Base 上 discover 的 Prefix，用于 rotated ANN training。

ann_training_calibration:
  在 rotated/fused Base + ANN-training Prefix 条件下统计的 calibration，
  用于 phase_aware / gif_aware ANN training。
```

---

### 4.2 Run-specific 目录

每个 final ANN checkpoint 独立拥有 post-finetuning prefix 与 conversion calibration：

```text
artifacts/<experiment>/<task>/<model>/<ann_mode>/lr<learning_rate>/seed42/
├── config/
├── logs/
├── ann/
│   └── final/
│
├── post_finetuning/
│   ├── prefix/
│   │   ├── prefix_state.json
│   │   └── prefixed_key_values.pt
│   │
│   └── conversion_calibration/
│       └── sites/
│           ├── layer_000/
│           │   ├── site_01_post_input_rmsnorm/
│           │   │   ├── statistics.pt
│           │   │   ├── phase_state.pt
│           │   │   ├── gif_state.pt
│           │   │   ├── mtn_state.pt
│           │   │   ├── clip_state.pt
│           │   │   └── calibration_summary.json
│           │   └── ...
│           ├── statistics_manifest.json
│           └── calibration_state_manifest.json
│
└── snn/
    ├── phase/
    ├── gif/
    └── mtn/
```

注意：

```text
post_finetuning/ 目录必须依赖具体 ann_mode、learning_rate、seed 和 final ANN checkpoint。
不能放到 _shared 中。
```

---

## 5. 需要修改的核心代码

### 5.1 `snn2/artifacts.py`

当前 `ArtifactLayout` 中 `prefix_dir`、`calibration_dir`、`site_dir` 语义过于模糊，需要改为阶段明确的属性。

新增或替换为以下属性：

```python
@property
def rotation_dir(self) -> Path:
    return self.shared_model_root / "rotated_prefix" / "rotation"

@property
def ann_training_prefix_dir(self) -> Path:
    return self.shared_model_root / "rotated_prefix" / "ann_training_prefix"

@property
def ann_training_calibration_dir(self) -> Path:
    return self.shared_model_root / "rotated_prefix" / "ann_training_calibration"

@property
def ann_training_site_dir(self) -> Path:
    return self.ann_training_calibration_dir / "sites"

@property
def vanilla_analysis_calibration_dir(self) -> Path:
    return self.shared_model_root / "vanilla_original" / "vanilla_analysis_calibration"

@property
def vanilla_analysis_site_dir(self) -> Path:
    return self.vanilla_analysis_calibration_dir / "sites"

@property
def post_finetuning_dir(self) -> Path:
    return self.root / "post_finetuning"

@property
def post_finetuning_prefix_dir(self) -> Path:
    return self.post_finetuning_dir / "prefix"

@property
def post_finetuning_conversion_calibration_dir(self) -> Path:
    return self.post_finetuning_dir / "conversion_calibration"

@property
def post_finetuning_site_dir(self) -> Path:
    return self.post_finetuning_conversion_calibration_dir / "sites"
```

不要再让训练、评估、conversion 直接使用旧的：

```python
layout.prefix_dir
layout.site_dir
```

如果为了兼容旧代码暂时保留，也必须禁止新逻辑继续使用它们。

`ensure()` 中需要创建新目录。

---

### 5.2 `snn2/config.py`

新增配置语义，建议在 `defaults` 中加入：

```yaml
post_finetuning:
  rediscover_prefix: true
  recalibrate_sites: true
  prefix_enabled: true
  post_finetuning_recalibration: true
```

保留现有 `prefix.enabled`，但重新定义其语义：

```text
prefix.enabled:
  仅表示 ANN-training Prefix 是否用于 rotated ANN training。
```

对于 `vanilla`：

```python
if mode == "vanilla":
    cfg["rotation"]["enabled"] = False
    cfg["prefix"]["enabled"] = False
    cfg["replacement"]["train_mode"] = "none"
```

但不要把：

```python
cfg["post_finetuning"]["prefix_enabled"]
```

改成 false。

也就是说：

```text
vanilla:
  training prefix disabled
  post-finetuning prefix enabled
```

需要新增 helper 函数，避免到处直接读 `cfg["prefix"]["enabled"]`：

```python
def training_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return (
        cfg["experiment"]["ann_mode"] != "vanilla"
        and bool(cfg["prefix"].get("enabled", False))
    )

def post_finetuning_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("post_finetuning", {}).get("prefix_enabled", True))
```

也建议新增：

```python
def post_finetuning_recalibration_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("post_finetuning", {}).get("post_finetuning_recalibration", True))
```

并在 `validate_config()` 中强制：

```text
post_finetuning.post_finetuning_recalibration must be true
post_finetuning.rediscover_prefix must be true
post_finetuning.recalibrate_sites must be true
```

本轮修改后不再支持旧的 `post_finetuning_recalibration=false` 主实验协议。

---

### 5.3 `snn2/modeling.py`

当前函数：

```python
model_source(cfg, layout, ann=False)
prefix_ids(cfg, layout)
prefix_key_values(cfg, layout)
```

语义不足，需要改为 stage-specific。

建议新增：

```python
def model_source_for_stage(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    *,
    stage: str,
) -> str:
    ...
```

stage 取值建议：

```text
pre_finetuning
ann_training
vanilla_analysis
post_finetuning
base_evaluation
```

语义：

```python
if stage in {"ann_training", "pre_finetuning"}:
    if cfg["rotation"]["enabled"]:
        return str(layout.rotation_dir / "fused_base")
    return cfg["experiment"]["model_name"]

if stage == "vanilla_analysis":
    return cfg["experiment"]["model_name"]

if stage == "post_finetuning":
    return str(layout.ann_checkpoint_dir)

if stage == "base_evaluation":
    return cfg["experiment"]["model_name"]
```

新增 Prefix stage-specific 函数：

```python
def prefix_ids_for_stage(cfg, layout, *, stage: str) -> list[int]:
    if stage == "ann_training":
        if not training_prefix_enabled(cfg):
            return []
        path = layout.ann_training_prefix_dir / "prefix_state.json"

    elif stage == "post_finetuning":
        if not post_finetuning_prefix_enabled(cfg):
            return []
        path = layout.post_finetuning_prefix_dir / "prefix_state.json"

    elif stage in {"vanilla_analysis", "base_evaluation"}:
        return []

    else:
        raise ValueError(stage)

    ...
```

```python
def prefix_key_values_for_stage(cfg, layout, *, stage: str):
    ids = prefix_ids_for_stage(...)
    if not ids:
        return None

    if stage == "ann_training":
        path = layout.ann_training_prefix_dir / "prefixed_key_values.pt"
    elif stage == "post_finetuning":
        path = layout.post_finetuning_prefix_dir / "prefixed_key_values.pt"
    ...
```

所有训练、评估、calibration、conversion 必须改用这些 stage-specific 函数。

---

### 5.4 `scripts/discover_prefix.py`

当前脚本只支持 shared policy prefix。需要改成支持 stage。

新增参数：

```bash
--stage {ann_training,post_finetuning}
```

行为：

#### `--stage ann_training`

只允许用于 rotated config：

```text
cfg["rotation"]["enabled"] == true
training_prefix_enabled(cfg) == true
```

加载模型：

```text
layout.rotation_dir / fused_base
```

安装：

```text
rotation integration
```

输出：

```text
layout.ann_training_prefix_dir/prefix_state.json
layout.ann_training_prefix_dir/prefixed_key_values.pt
```

#### `--stage post_finetuning`

允许四种 ANN mode：

```text
vanilla
unaware
phase_aware
gif_aware
```

加载模型：

```text
layout.ann_checkpoint_dir
```

如果该 config rotation enabled，则安装 rotation integration；Vanilla 不安装 rotation。

输出：

```text
layout.post_finetuning_prefix_dir/prefix_state.json
layout.post_finetuning_prefix_dir/prefixed_key_values.pt
```

注意：

```text
Vanilla post-finetuning prefix 也必须 discover。
不能因为 cfg["prefix"]["enabled"] == false 就跳过。
```

`discover_prefix.py` 中原先的：

```python
if not cfg["prefix"]["enabled"]:
    ...
    return
```

必须改为根据 stage 判断。

---

### 5.5 `scripts/calibrate_sites.py`

当前脚本只支持 shared policy calibration。需要改成支持 stage。

新增参数：

```bash
--stage {ann_training,vanilla_analysis,post_finetuning}
```

#### `--stage ann_training`

加载：

```text
layout.rotation_dir / fused_base
```

使用：

```text
layout.ann_training_prefix_dir/prefixed_key_values.pt
```

输出：

```text
layout.ann_training_site_dir
```

生成：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
calibration_state_manifest.json
```

manifest 写入：

```yaml
purpose: ann_training_calibration
eligible_for_ann_training: true
eligible_for_conversion: false
post_finetuning_recalibration: false
source_model_stage: rotated_fused_base
```

#### `--stage vanilla_analysis`

加载：

```text
cfg["experiment"]["model_name"]
```

不使用：

```text
rotation
prefix
```

输出：

```text
layout.vanilla_analysis_site_dir
```

manifest 写入：

```yaml
purpose: vanilla_analysis_calibration
analysis_only: true
eligible_for_ann_training: false
eligible_for_conversion: false
post_finetuning_recalibration: false
source_model_stage: original_pretrained_base
```

建议只保存 statistics，不生成 neuron states。若为了复用代码生成了 neuron states，也必须禁止 conversion 使用。

#### `--stage post_finetuning`

加载：

```text
layout.ann_checkpoint_dir
```

使用：

```text
layout.post_finetuning_prefix_dir/prefixed_key_values.pt
```

输出：

```text
layout.post_finetuning_site_dir
```

生成 conversion 使用的：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
calibration_state_manifest.json
```

manifest 写入：

```yaml
purpose: post_finetuning_conversion_calibration
eligible_for_ann_training: false
eligible_for_conversion: true
post_finetuning_recalibration: true
source_ann_checkpoint: <absolute path to ann/final>
source_ann_mode: <ann_mode>
learning_rate: <lr>
```

注意：

```text
post_finetuning calibration 统计的是 final ANN 的原始 activation。
即使 final checkpoint 来自 phase_aware / gif_aware，calibration 时也只使用 collect mode，不要再次套 phase/gif replacement。
```

---

### 5.6 `snn2/calibration.py`

需要支持不同 purpose 的 manifest。

建议修改：

```python
collect_site_statistics(...)
```

新增参数：

```python
purpose: str
materialize_states: bool = True
extra_metadata: dict[str, Any] | None = None
```

行为：

1. `purpose == "vanilla_analysis_calibration"` 时可以设置 `materialize_states=False`。
2. `purpose == "ann_training_calibration"` 时生成 states，但 manifest 标记 `eligible_for_conversion=false`。
3. `purpose == "post_finetuning_conversion_calibration"` 时生成 states，并标记 `eligible_for_conversion=true`。

`calibration_state_manifest.json` 至少记录：

```yaml
format_version: 1
purpose: ...
site_topology_version: ...
site_count: ...
eligible_for_ann_training: ...
eligible_for_conversion: ...
post_finetuning_recalibration: ...
source_model_stage: ...
source_ann_checkpoint: ...
source_ann_config_sha256: ...
calibration_data_manifest_sha256: ...
prefix_enabled: ...
prefix_token_ids: ...
prefix_state_sha256: ...
prefix_kv_sha256: ...
rotation_enabled: ...
rotation_state_sha256: ...
```

如果某项不适用，写 `null`，不要省略。

---

### 5.7 `snn2/training.py`

当前训练需要改成：

#### Vanilla training

```text
source = original pretrained Base
rotation disabled
training prefix disabled
replacement none
```

不要安装 prefix KV。

#### Unaware training

```text
source = rotated fused Base
rotation enabled
ANN-training Prefix enabled
replacement none
```

使用：

```python
prefix_key_values_for_stage(cfg, layout, stage="ann_training")
```

#### Phase-aware training

```text
source = rotated fused Base
rotation enabled
ANN-training Prefix enabled
replacement phase
site_root = layout.ann_training_site_dir
```

#### GIF-aware training

```text
source = rotated fused Base
rotation enabled
ANN-training Prefix enabled
replacement gif
site_root = layout.ann_training_site_dir
```

当前：

```python
controller = SiteController(mode=mode, site_root=layout.site_dir)
install_prefix_kv_forward(model, prefix_key_values(cfg, layout))
```

必须改为：

```python
controller = SiteController(mode=mode, site_root=layout.ann_training_site_dir)
install_prefix_kv_forward(
    model,
    prefix_key_values_for_stage(cfg, layout, stage="ann_training"),
)
```

但对于 `vanilla`，prefix 返回 `None`。

训练 metadata 中建议写入：

```json
{
  "training_prefix_stage": "ann_training" or "disabled",
  "ann_training_calibration_root": "...",
  "post_finetuning_prefix_used_for_training": false
}
```

---

### 5.8 `snn2/conversion.py`

当前 conversion 直接读取 shared calibration，并写：

```json
"post_finetuning_recalibration": false
```

必须改为：

```text
conversion 只允许读取 layout.post_finetuning_site_dir
```

校验：

```text
layout.post_finetuning_site_dir/calibration_state_manifest.json 必须存在
manifest.purpose == "post_finetuning_conversion_calibration"
manifest.eligible_for_conversion == true
manifest.post_finetuning_recalibration == true
```

如果发现：

```text
ann_training_calibration
vanilla_analysis_calibration
legacy shared calibration
```

必须报错。

metadata 必须改为：

```json
"post_finetuning_recalibration": true
"calibration_root": "<run-specific>/post_finetuning/conversion_calibration/sites"
"prefix_root": "<run-specific>/post_finetuning/prefix"
"prefix_enabled": true
```

对 Vanilla：

```json
"rotation_enabled": false
"prefix_enabled": true
```

对 rotated 三分支：

```json
"rotation_enabled": true
"prefix_enabled": true
```

---

### 5.9 `scripts/convert_snn.py`

不一定需要改 CLI，但其内部必须走新的 `create_conversion()`。

运行方式仍建议保持：

```bash
python scripts/convert_snn.py --config "$CFG" --neuron phase
python scripts/convert_snn.py --config "$CFG" --neuron gif
python scripts/convert_snn.py --config "$CFG" --neuron mtn
```

前置条件变为：

```text
ann/final 已存在
post_finetuning/prefix 已存在
post_finetuning/conversion_calibration/sites 已存在
```

---

### 5.10 `scripts/evaluate_tldr.py`

修改 evaluation 行为。

#### Base evaluation

保留当前语义：

```bash
python scripts/evaluate_tldr.py --config <vanilla cfg> --neuron ann --base
```

使用：

```text
original pretrained Base
no prefix
no rotation
```

#### Final ANN evaluation

所有四种 mode：

```bash
python scripts/evaluate_tldr.py --config "$CFG" --neuron ann
```

必须使用：

```text
layout.ann_checkpoint_dir
layout.post_finetuning_prefix_dir/prefixed_key_values.pt
```

即 Vanilla final ANN evaluation 也使用 post-finetuning Prefix。

#### SNN evaluation

```bash
python scripts/evaluate_tldr.py --config "$CFG" --neuron phase/gif/mtn
```

必须使用：

```text
layout.ann_checkpoint_dir
layout.post_finetuning_prefix_dir/prefixed_key_values.pt
layout.post_finetuning_site_dir
```

当前：

```python
controller = SiteController(mode="identity", site_root=layout.site_dir)
install_prefix_kv_forward(model, prefix_key_values(cfg, layout))
```

必须改为：

```python
controller = SiteController(mode="identity", site_root=layout.post_finetuning_site_dir)
install_prefix_kv_forward(
    model,
    prefix_key_values_for_stage(cfg, layout, stage="post_finetuning"),
)
```

但 `--base` 时 prefix 必须为 `None`。

输出 metrics 中建议新增：

```json
"prefix_stage": "post_finetuning"
"post_finetuning_recalibration": true
"post_finetuning_prefix_enabled": true
"calibration_root": "..."
```

---

### 5.11 `scripts/evaluate_lm_harness.py`

与 `evaluate_tldr.py` 做同样修改。

Final ANN 和 SNN evaluation 必须使用：

```text
post_finetuning Prefix
post_finetuning conversion calibration
```

Base evaluation 保持：

```text
original Base
no prefix
```

metadata 中记录：

```json
"prefix_stage": "post_finetuning"
"post_finetuning_recalibration": true
"calibration_root": "..."
```

---

### 5.12 `scripts/verify_artifacts.py`

必须修改校验逻辑。

新的校验项：

#### Shared

```text
data manifest exists
rotation exists for rotated modes
ann_training_prefix exists for rotated modes
ann_training_calibration exists for rotated modes
vanilla_analysis_calibration exists for model-task pair
```

#### Run-specific

每个 config 必须校验：

```text
ann/final exists
post_finetuning/prefix/prefix_state.json exists
post_finetuning/prefix/prefixed_key_values.pt exists
post_finetuning/conversion_calibration/sites exists
post_finetuning/conversion_calibration/sites/calibration_state_manifest.json exists
manifest.purpose == post_finetuning_conversion_calibration
manifest.post_finetuning_recalibration == true
manifest.eligible_for_conversion == true
```

SNN conversion descriptor 必须校验：

```text
snn/phase/conversion_metadata.json
snn/gif/conversion_metadata.json
snn/mtn/conversion_metadata.json
```

并检查：

```text
conversion_metadata.post_finetuning_recalibration == true
conversion_metadata.calibration_root points to post_finetuning/conversion_calibration/sites
```

---

## 6. 新运行流程：必须写入 `实验执行总结.md`

`实验执行总结.md` 必须重写相关章节。建议采用以下内容结构。

---

### 6.1 生成 12 个配置

```bash
python scripts/materialize_configs.py
```

输出仍为：

```text
configs/generated/exp1_qwen3_1_7b_tldr__{vanilla,unaware,phase_aware,gif_aware}.yaml
configs/generated/exp1_qwen3_8b_tldr__{vanilla,unaware,phase_aware,gif_aware}.yaml
configs/generated/exp2_llama3_8b_tulu3__{vanilla,unaware,phase_aware,gif_aware}.yaml
```

---

### 6.2 准备数据 manifest

同一任务只需准备一次：

```bash
python scripts/prepare_data.py --config configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
python scripts/prepare_data.py --config configs/generated/exp2_llama3_8b_tulu3__unaware.yaml
```

---

### 6.3 每个模型准备 rotation、ANN-training Prefix、ANN-training calibration、Vanilla analysis calibration

对每个 model-task pair 执行。

以 Qwen3-1.7B TL;DR 为例：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
VAN_CFG=configs/generated/exp1_qwen3_1_7b_tldr__vanilla.yaml

python scripts/prepare_rotation.py --config "$ROT_CFG"

python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage ann_training

python scripts/calibrate_sites.py \
  --config "$ROT_CFG" \
  --stage ann_training

python scripts/calibrate_sites.py \
  --config "$VAN_CFG" \
  --stage vanilla_analysis
```

含义：

```text
prepare_rotation:
  生成 rotated/fused Base checkpoint。

discover_prefix --stage ann_training:
  在 rotated/fused Base 上 discover ANN-training Prefix，
  并构造 fixed ANN-training Prefix KV cache。

calibrate_sites --stage ann_training:
  在 rotated/fused Base + ANN-training Prefix 条件下统计 10 个 site，
  生成用于 phase_aware/gif_aware ANN 微调的 frozen states。

calibrate_sites --stage vanilla_analysis:
  在 original Base、无 rotation、无 prefix 条件下统计 10 个 site，
  仅用于分析 rotation+prefix 前后的 activation distribution 变化。
```

必须明确写：

```text
Vanilla analysis calibration 不用于 SNN conversion。
```

---

### 6.4 训练 12 个 ANN checkpoint

训练命令基本保持不变。

示例：

```bash
export CUDA_VISIBLE_DEVICES=0,1
NGPU=2

for MODE in vanilla unaware phase_aware gif_aware; do
  torchrun --standalone --nproc_per_node="$NGPU" \
    scripts/train_ann.py \
    --config "configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"
done
```

说明：

```text
vanilla:
  no rotation, no prefix, no activation replacement

unaware:
  rotation + ANN-training Prefix, no activation replacement

phase_aware:
  rotation + ANN-training Prefix + Phase replacement using ANN-training calibration

gif_aware:
  rotation + ANN-training Prefix + GIF replacement using ANN-training calibration
```

---

### 6.5 每个 final ANN checkpoint 执行 post-finetuning Prefix discovery 与 conversion calibration

这是本轮新增的关键阶段。

每个 config 训练完成后都必须执行：

```bash
CFG=configs/generated/exp1_qwen3_1_7b_tldr__vanilla.yaml

python scripts/discover_prefix.py \
  --config "$CFG" \
  --stage post_finetuning

python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning
```

对四种 mode 全部执行：

```bash
for MODE in vanilla unaware phase_aware gif_aware; do
  CFG="configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"

  python scripts/discover_prefix.py \
    --config "$CFG" \
    --stage post_finetuning

  python scripts/calibrate_sites.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

必须在文档中明确说明：

```text
post_finetuning Prefix / calibration 是 run-specific 的。
每个 final ANN checkpoint 独立生成。
不允许 unaware、phase_aware、gif_aware 共享同一套 post-finetuning calibration。
Vanilla final ANN checkpoint 也必须执行同样的 post-finetuning Prefix discovery 和 calibration，只是不使用 rotation。
```

---

### 6.6 Base evaluation

Base evaluation 保持原始 pretrained Base baseline：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 scripts/evaluate_tldr.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__vanilla.yaml \
  --neuron ann \
  --base
```

说明：

```text
--base 评估的是 original pretrained Base。
不使用 post-finetuning Prefix。
不使用 final ANN checkpoint。
```

---

### 6.7 Final ANN evaluation

每个 final ANN checkpoint 的 ANN evaluation 必须在 post-finetuning Prefix discovery 后执行。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 scripts/evaluate_tldr.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__vanilla.yaml \
  --neuron ann
```

说明：

```text
普通 --neuron ann 评估的是 ann/final。
四种 mode 都使用各自 run-specific post-finetuning Prefix。
Vanilla final ANN evaluation 也使用 post-finetuning Prefix。
本轮不额外添加 no-prefix Vanilla final ANN evaluation。
```

---

### 6.8 SNN conversion

每个 ANN config 分别转换三种 neuron：

```bash
CFG=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml

for NEURON in phase gif mtn; do
  python scripts/convert_snn.py \
    --config "$CFG" \
    --neuron "$NEURON"
done
```

说明：

```text
convert_snn.py 使用 run-specific post_finetuning/conversion_calibration。
conversion_metadata.json 中 post_finetuning_recalibration 必须为 true。
```

---

### 6.9 SNN evaluation

TL;DR：

```bash
CFG=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml

for NEURON in phase gif mtn; do
  accelerate launch --num_processes 4 scripts/evaluate_tldr.py \
    --config "$CFG" \
    --neuron "$NEURON"
done
```

Tulu / lm-eval：

```bash
CFG=configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml

for NEURON in phase gif mtn; do
  accelerate launch --num_processes 4 scripts/evaluate_lm_harness.py \
    --config "$CFG" \
    --neuron "$NEURON"
done
```

说明：

```text
SNN evaluation 使用与 Final ANN evaluation 相同的 post-finetuning Prefix。
SNN activation replacement 使用 post-finetuning conversion calibration 生成的 states。
```

---

## 7. 其他 Markdown 文档需要同步修改

至少修改：

```text
代码结构总结.md
实验执行总结.md
环境配置.md 如有相关路径说明
docs/history/ 中如果存在旧 9-site 或旧 calibration 表述，也需要更新或标记为历史方案
```

重点删除或改写以下旧表述：

```text
unaware、phase-aware、GIF-aware 共用同一 model-task 的 rotation、prefix 和 frozen calibration
```

应改成：

```text
unaware、phase-aware、GIF-aware 在 ANN training 前共用 rotation 和 ANN-training Prefix；
phase-aware、GIF-aware 在 ANN training 阶段使用共享 ANN-training calibration；
但 SNN conversion 前，每个 final ANN checkpoint 都会独立执行 post-finetuning Prefix discovery 和 post-finetuning conversion calibration，不能共享。
```

旧表述：

```text
ANN 微调后禁止重新校准
post_finetuning_recalibration: false
```

必须改成：

```text
ANN 微调后必须重新执行 post-finetuning Prefix discovery 和 post-finetuning conversion calibration。
post_finetuning_recalibration: true
```

旧表述：

```text
vanilla_original/calibration 用于 Vanilla SNN conversion
```

必须改成：

```text
vanilla_original/vanilla_analysis_calibration 仅用于分析，不用于 SNN conversion。
Vanilla final checkpoint 的 SNN conversion 使用 run-specific post_finetuning/conversion_calibration。
```

---

## 8. 测试要求

必须更新或新增测试。

建议新增测试文件：

```text
tests/test_post_finetuning_protocol.py
```

至少测试：

1. `ArtifactLayout` 路径：
   ```text
   ann_training_prefix_dir
   ann_training_site_dir
   vanilla_analysis_site_dir
   post_finetuning_prefix_dir
   post_finetuning_site_dir
   ```

2. Vanilla config：
   ```text
   training_prefix_enabled == false
   post_finetuning_prefix_enabled == true
   ```

3. Rotated config：
   ```text
   training_prefix_enabled == true
   post_finetuning_prefix_enabled == true
   ```

4. `convert_snn.py` 只能接受：
   ```text
   purpose == post_finetuning_conversion_calibration
   eligible_for_conversion == true
   post_finetuning_recalibration == true
   ```

5. ANN evaluation metadata：
   ```text
   prefix_stage == post_finetuning
   ```

6. SNN evaluation metadata：
   ```text
   prefix_stage == post_finetuning
   calibration_root contains post_finetuning/conversion_calibration/sites
   ```

7. Vanilla analysis calibration：
   ```text
   eligible_for_conversion == false
   ```

保留并更新现有 10-site tests，确保：

```text
expected_sites_per_layer = 10
site_topology_version 正确
旧 9-site calibration 仍然会被拒绝
```

运行：

```bash
pytest -q
```

---

## 9. 兼容性与旧 artifacts

本轮修改后，旧 artifacts 不应继续被新流程消费。

建议在文档中写明：

```text
本轮修改改变了 prefix/calibration 目录结构。
旧的 _shared/rotated_prefix/prefix/ 和 _shared/rotated_prefix/calibration/ 不再作为主流程输入。
为了避免误用，建议将旧 artifacts 目录整体移动或删除后重新运行。
```

代码层面也应避免 silent fallback。

如果 post-finetuning prefix 或 calibration 不存在，evaluation / conversion 必须报错，而不是回退到 shared old prefix/calibration。

---

## 10. 最终验收标准

完成修改后，必须满足：

1. `materialize_configs.py` 能正常生成 12 个配置。
2. `prepare_rotation.py` 仍能生成 fused Base。
3. `discover_prefix.py --stage ann_training` 输出到：
   ```text
   _shared/seed42/rotated_prefix/ann_training_prefix/
   ```
4. `calibrate_sites.py --stage ann_training` 输出到：
   ```text
   _shared/seed42/rotated_prefix/ann_training_calibration/sites/
   ```
5. `calibrate_sites.py --stage vanilla_analysis` 输出到：
   ```text
   _shared/seed42/vanilla_original/vanilla_analysis_calibration/sites/
   ```
6. `discover_prefix.py --stage post_finetuning` 输出到：
   ```text
   <ann_mode>/lr.../seed42/post_finetuning/prefix/
   ```
7. `calibrate_sites.py --stage post_finetuning` 输出到：
   ```text
   <ann_mode>/lr.../seed42/post_finetuning/conversion_calibration/sites/
   ```
8. `train_ann.py`：
   ```text
   vanilla 不使用 prefix/calibration
   unaware 使用 ANN-training Prefix，不使用 calibration
   phase_aware 使用 ANN-training Prefix + Phase calibration
   gif_aware 使用 ANN-training Prefix + GIF calibration
   ```
9. `evaluate_tldr.py --neuron ann` 和 `evaluate_lm_harness.py --neuron ann`：
   ```text
   final ANN evaluation 使用 post-finetuning Prefix
   ```
10. `convert_snn.py`：
    ```text
    只使用 post-finetuning conversion calibration
    conversion_metadata.json 中 post_finetuning_recalibration == true
    ```
11. `evaluate_tldr.py --neuron phase/gif/mtn` 和 `evaluate_lm_harness.py --neuron phase/gif/mtn`：
    ```text
    SNN evaluation 使用 post-finetuning Prefix
    SNN states 来自 post-finetuning conversion calibration
    ```
12. `实验执行总结.md` 清楚说明新的完整运行流程。
13. `pytest -q` 通过。

---

## 11. 修改后推荐的完整命令顺序

以 Qwen3-1.7B TL;DR 为例。

```bash
# 1. 生成配置
python scripts/materialize_configs.py

# 2. 准备数据
python scripts/prepare_data.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

# 3. 准备 pre-finetuning shared states
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
VAN_CFG=configs/generated/exp1_qwen3_1_7b_tldr__vanilla.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"

python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage ann_training

python scripts/calibrate_sites.py \
  --config "$ROT_CFG" \
  --stage ann_training

python scripts/calibrate_sites.py \
  --config "$VAN_CFG" \
  --stage vanilla_analysis

# 4. 训练四个 ANN
export CUDA_VISIBLE_DEVICES=0,1
NGPU=2

for MODE in vanilla unaware phase_aware gif_aware; do
  torchrun --standalone --nproc_per_node="$NGPU" \
    scripts/train_ann.py \
    --config "configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"
done

# 5. 每个 final ANN checkpoint 重新 discover prefix + recalibrate
for MODE in vanilla unaware phase_aware gif_aware; do
  CFG="configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"

  python scripts/discover_prefix.py \
    --config "$CFG" \
    --stage post_finetuning

  python scripts/calibrate_sites.py \
    --config "$CFG" \
    --stage post_finetuning
done

# 6. Final ANN evaluation
for MODE in vanilla unaware phase_aware gif_aware; do
  CFG="configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"

  accelerate launch --num_processes 1 scripts/evaluate_tldr.py \
    --config "$CFG" \
    --neuron ann
done

# 7. SNN conversion
for MODE in vanilla unaware phase_aware gif_aware; do
  CFG="configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"

  for NEURON in phase gif mtn; do
    python scripts/convert_snn.py \
      --config "$CFG" \
      --neuron "$NEURON"
  done
done

# 8. SNN evaluation
for MODE in vanilla unaware phase_aware gif_aware; do
  CFG="configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"

  for NEURON in phase gif mtn; do
    accelerate launch --num_processes 1 scripts/evaluate_tldr.py \
      --config "$CFG" \
      --neuron "$NEURON"
  done
done

# 9. 校验
for MODE in vanilla unaware phase_aware gif_aware; do
  python scripts/verify_artifacts.py \
    --config "configs/generated/exp1_qwen3_1_7b_tldr__${MODE}.yaml"
done
```

---

## 12. 一句话总结

本轮修改后，项目实验协议应变为：

```text
训练所需的 Prefix / calibration 放在 _shared；
任何依赖 final ANN checkpoint 的 Prefix、calibration、conversion state 和 evaluation state 一律放在对应 ann_mode/lr/seed 的 run-specific 目录中；
ANN evaluation 和 SNN evaluation 使用同一个 post-finetuning Prefix；
SNN conversion 使用 post-finetuning conversion calibration；
conversion_metadata.json 中 post_finetuning_recalibration 必须为 true。
```