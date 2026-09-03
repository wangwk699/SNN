# SNN 双套 Artifact Source 收尾修复方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 适用状态：仓库已完成 `conversion.use_post_finetuning_artifacts` 双套 Prefix / Calibration artifact source 主体改造。  
> 本文面向**没有本次对话上下文的服务器 Codex**；请仅依据本文完成本轮收尾修改。
>
> 本轮不重新设计 selector 语义。核心 dual-artifact source 逻辑已经基本正确，本轮只修复剩余代码 bug、协议不一致、Markdown 旧表述和测试覆盖缺口。

---

# 1. 当前已确认的正确语义

当前已有：

```yaml
conversion:
  use_post_finetuning_artifacts: true
```

其含义必须保持：

## `true`

SNN conversion / SNN evaluation 使用：

```text
Post-finetuning Prefix
+
Post-finetuning conversion Stage A
```

## `false`

仅允许：

```text
unaware
phase_aware
gif_aware
```

使用：

```text
shared Pre-finetuning Prefix
+
shared ANN-training Calibration Stage A
```

## vanilla

```text
vanilla + use_post_finetuning_artifacts=false
```

必须在 config validation 阶段报错，不允许静默改成 `true`。

两套 artifact 可以同时存在：

```text
Step 4 + Step 5 = Pre-finetuning bundle
Step 7 + Step 8 = Post-finetuning bundle
```

selector 只决定：

```text
convert_snn.py
SNN Evaluation
```

最终使用哪一套。

**不要改变以上设计。**

---

# 2. 本轮必须完成的修改

本轮处理以下所有问题：

1. `scripts/evaluate_lm_harness.py` 缺少 `validate_conversion_metadata` import，SNN lm-eval 会 `NameError`
2. `scripts/calibrate_sites.py` 仍允许 `post_finetuning + Stage B`
3. `README.md` 仍保留旧的 mode-fixed conversion source 描述
4. `AGENTS.md` 仍保留旧的 aware 固定复用、旧 SNN 路径和旧 Clip bundle 描述
5. `实验执行总结.md` 附录 A、TL;DR 样本数和 Temporal 版本存在旧表述
6. `代码结构总结.md` 没有同步 selector 与当前 schema/version
7. selector=false、aware Post-finetuning、verify、lm-harness SNN 等测试覆盖不足

---

# 3. 修复 `scripts/evaluate_lm_harness.py`

## 3.1 问题

当前文件实际执行：

```python
if args.neuron != "ann" and not args.base and not args.rotated_pre_finetuning:
    validate_conversion_metadata(cfg, layout, args.neuron)
```

但顶部没有导入：

```python
validate_conversion_metadata
```

因此：

```bash
accelerate launch --num_processes 1 \
  scripts/evaluate_lm_harness.py \
  --config "$CFG" \
  --neuron phase
```

以及：

```text
gif
mtn
```

都会在真正进入 lm-eval 前出现：

```text
NameError: name 'validate_conversion_metadata' is not defined
```

## 3.2 修改

在 import 区新增：

```python
from snn2.conversion import validate_conversion_metadata
```

不要删除现有调用。

SNN lm-harness 必须继续在真正执行 temporal evaluation 前校验 conversion descriptor。

## 3.3 测试

必须补一个 regression / smoke test，至少保证：

```python
evaluate_lm_harness.py
```

模块加载后：

```python
validate_conversion_metadata
```

存在并可调用。

推荐两种方式任选一种。

### 方式 A：最小 smoke test

动态 import：

```python
scripts/evaluate_lm_harness.py
```

然后：

```python
assert callable(module.validate_conversion_metadata)
```

### 方式 B：更强

monkeypatch：

```text
parser
setup
validate_conversion_metadata
```

模拟：

```text
--neuron phase
```

进入 SNN validation 分支，验证：

```text
validate_conversion_metadata(cfg, layout, "phase")
```

实际被调用一次，且不出现 `NameError`。

---

# 4. 真正禁止 Post-finetuning Calibration Stage B

涉及：

```text
scripts/calibrate_sites.py
```

## 4.1 当前问题

当前脚本仍包含：

```python
("post_finetuning", "B"): "post_finetuning_clip_profile"
```

并且：

```python
profile_root = {
    "ann_training": layout.ann_training_clip_profile_dir,
    "post_finetuning": layout.post_finetuning_clip_profile_dir,
}.get(args.stage)
```

因此下面命令仍能进入 `materialize_clip_profile()`：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning \
  --calibration-phase B
```

这与当前协议冲突。

当前协议必须是：

```text
Post-finetuning conversion calibration = Stage A only
Post-finetuning 不生成 Stage B
Post-finetuning bundle 完全 clip-free
SNN conversion/deployment 不加载或执行 common Clip
```

## 4.2 最终合法组合

只允许：

```text
ann_training + A
ann_training + B
vanilla_analysis + A
post_finetuning + A
```

禁止：

```text
vanilla_analysis + B
post_finetuning + B
```

## 4.3 推荐实现

scope mapping 改成：

```python
scope = {
    ("ann_training", "A"): "ann_training_calibration",
    ("ann_training", "B"): "ann_training_clip_profile",
    ("vanilla_analysis", "A"): "vanilla_analysis_calibration",
    ("post_finetuning", "A"): "post_finetuning_calibration",
}.get((args.stage, args.calibration_phase))
```

然后显式拒绝：

```python
if scope is None:
    if args.stage == "post_finetuning" and args.calibration_phase == "B":
        raise ValueError(
            "Post-finetuning conversion calibration is Stage A only; "
            "Stage B Clip profiles are forbidden."
        )
    if args.stage == "vanilla_analysis" and args.calibration_phase == "B":
        raise ValueError(
            "vanilla_analysis is analysis-only and supports calibration phase A only"
        )
    raise ValueError(
        f"Unsupported calibration stage/phase combination: "
        f"{args.stage} + {args.calibration_phase}"
    )
```

## 4.4 `profile_root`

改成只允许 ANN-training 使用 Stage B：

```python
profile_root = (
    layout.ann_training_clip_profile_dir
    if args.stage == "ann_training"
    else None
)
```

或等价实现。

不要再从实际执行路径访问：

```python
layout.post_finetuning_clip_profile_dir
```

`ArtifactLayout` 中历史 property 如仍有兼容原因，可以暂时保留，但当前主实验脚本不得创建 Post-finetuning Clip profile。

## 4.5 修正文案

当前类似：

```text
ANN-training calibration is only used by phase_aware/gif_aware modes
```

不够准确。

改成类似：

```text
ANN-training calibration generation is only performed with
phase_aware/gif_aware configs; the resulting shared Stage A may also
be consumed by unaware SNN conversion.
```

注意：

```python
requires_ann_training_calibration(cfg)
```

继续保持 aware-only。

不要改成：

```text
unaware -> True
```

因为 `unaware` 只是消费 Step 5 shared Stage A，不单独运行 Step 5。

## 4.6 测试

增加：

```text
post_finetuning + B -> ValueError
```

错误消息至少包含：

```text
Stage A only
```

并确认：

```text
ann_training + B
```

仍合法。

如果当前 `main()` 难以单测，建议抽一个小 helper：

```python
def calibration_scope(stage: str, phase: str) -> str:
    ...
```

让 main 和 tests 共用，避免以后合法组合再次漂移。

---

# 5. 同步 `README.md`

## 5.1 删除旧协议

删除或重写所有类似：

```text
aware 模式在转换时复用 ANN-training calibration，
vanilla/unaware 使用 Post-finetuning conversion calibration。
```

以及：

```text
vanilla/unaware -> Post-finetuning
phase_aware/gif_aware -> 固定复用 Pre-finetuning Prefix + ANN-training Stage A
```

这些已经失效。

## 5.2 改成 dual-artifact source

明确写：

对于：

```text
unaware
phase_aware
gif_aware
```

可以同时拥有：

### Pre bundle

```text
shared Pre-finetuning Prefix
+
shared ANN-training Calibration Stage A
```

### Post bundle

```text
per-run Post-finetuning Prefix
+
per-run Post-finetuning conversion Stage A
```

由：

```yaml
conversion:
  use_post_finetuning_artifacts: true|false
```

唯一决定 SNN conversion / SNN evaluation 使用哪一套。

加入 canonical 表：

| ANN mode | selector=false | selector=true |
|---|---|---|
| `vanilla` | 非法 | Post-finetuning Prefix + Post-finetuning Stage A |
| `unaware` | shared Pre Prefix + shared ANN-training Stage A | Post Prefix + Post Stage A |
| `phase_aware` | shared Pre Prefix + shared ANN-training Stage A | Post Prefix + Post Stage A |
| `gif_aware` | shared Pre Prefix + shared ANN-training Stage A | Post Prefix + Post Stage A |

并明确：

- selector 不改变 ANN checkpoint
- selector 不改变 Final ANN Evaluation
- aware Post-finetuning artifacts 来自各自 Final ANN
- aware Post-finetuning artifacts 不进入 `_shared`
- SNN conversion/deployment 只读取 Stage A
- SNN 不读取 Stage B Clip

## 5.3 Final ANN Evaluation

README 中继续明确：

```text
vanilla final ANN
    -> no Prefix

unaware final ANN
    -> Post-finetuning Prefix
    -> 是否实际加载由 evaluation.prefix_enabled 控制

phase_aware / gif_aware final ANN
    -> Pre-finetuning Prefix
    -> 是否实际加载由 evaluation.prefix_enabled 控制
```

并强调：

```text
conversion.use_post_finetuning_artifacts
```

不影响 Final ANN Evaluation。

---

# 6. 同步 `AGENTS.md`

这是高优先级文件，因为后续 Codex 会直接遵循它。

---

## 6.1 更新 SNN path 规则

旧规则如果还是：

```text
aware -> snn/<neuron>
vanilla/unaware -> snn/calibration_group_size_<G>/<neuron>
```

必须删除。

### aware

正确：

```text
.../<aware-run>/
  snn/
    use_post_finetuning_artifacts_<bool>/
      phase/
        phase_T_<P>/
      gif/
      mtn/
        mtn_T_<M>_mtn_K_<K>/
```

aware run root 已包含：

```text
num_samples_<N>
calibration_group_size_<G>
```

因此 SNN 子树不要重复 G/N。

### vanilla / unaware

正确：

```text
.../<run>/
  snn/
    use_post_finetuning_artifacts_<bool>/
      calibration_group_size_<G>_num_samples_<N>/
        phase/
          phase_T_<P>/
        gif/
        mtn/
          mtn_T_<M>_mtn_K_<K>/
```

必须明确：

```text
selector segment 恰好出现一次
```

并且：

```text
ANN run root 不受 selector 影响
vanilla + false 非法
```

---

## 6.2 新增 source invariant

在 AGENTS 中明确写：

```text
SNN conversion artifact source must never be inferred from ann_mode alone.
It is selected only by conversion.use_post_finetuning_artifacts.
```

即：

```text
true
    -> Post-finetuning Prefix
    -> Post-finetuning Stage A

false
    -> Pre-finetuning Prefix
    -> ANN-training Stage A
```

仅 vanilla 禁止 false。

---

## 6.3 修正 aware provenance 规则

写清：

### phase/gif aware + false

必须使用：

```text
ANN training 当时使用的同一 shared Pre Prefix
+
同一 shared ANN-training Stage A
```

并校验 frozen ANN-training provenance。

### phase/gif aware + true

必须使用：

```text
当前 Final ANN
    -> Post-finetuning Prefix
    -> Post-finetuning Stage A
```

不要求：

```text
Post Stage A hash == ANN-training Stage A hash
```

也不要求 Post bundle 与 training bundle 内容完全一致。

### unaware + false

使用：

```text
shared Pre Prefix
+
shared ANN-training Stage A
```

但不得要求：

```text
aware ANN training Clip provenance
aware ANN training calibration provenance record
```

因为 unaware ANN training 本身没有 aware replacement / Clip。

---

## 6.4 修正 Stage A / B / Clip 描述

如果仍有类似：

```text
ANN-training calibration site 目录直接生成 9 个 clip_state.pt
```

必须改掉。

当前准确语义：

### ANN-training Stage A

每个 site 生成：

```text
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

Stage A site 目录：

```text
不得包含 clip_state.pt
```

### ANN-training Stage B

位于：

```text
clip_profiles/
  phase_T_<P>_mtn_T_<M>/
```

只为 eligible sites 生成 Clip。

当前：

```text
9 个 eligible sites
Site 5 永久无 Clip
```

### Post-finetuning

```text
Stage A only
完全 clip-free
禁止 Stage B
```

---

## 6.5 不要改其它已经正确的 invariant

不要顺带改变：

- 10 activation sites
- Site 2/3/4/6 topology/grouping
- Site 5 GIF identity
- Site 5 no-Clip
- Final RMSNorm Phase/MTN/GIF 规则
- Phase/MTN EMA + clamp
- Prefix temporal policy
- ANN static vs SNN temporal 边界
- role-specific GIF / Clip 规则

---

# 7. 修正 `实验执行总结.md`

主体 Step 1 / 4 / 5 / 7 / 8 / 9 / 10 已基本正确，不需要全文重写。

重点修以下残留。

---

## 7.1 附录 A：依赖树重写

旧 aware 分支仍类似：

```text
Aware
└── Rotation
    ├── shared Pre Prefix
    ├── shared ANN-training calibration
    └── aware ANN training
        └── final ANN
            └── conversion / evaluation
                （复用训练前 Prefix + calibration）
```

这是旧协议。

改成：

```text
固定数据 manifest
├── Base baseline
│
├── Vanilla
│   └── ANN training
│       └── Final ANN
│           ├── Step 7 Post-finetuning Prefix
│           ├── Step 8 Post-finetuning Stage A
│           └── SNN conversion / evaluation
│               └── selector 必须 true
│
└── Rotated modes
    ├── unaware
    ├── phase_aware
    └── gif_aware
        └── Rotation
            ├── Step 4 shared Pre-finetuning Prefix
            ├── Step 5 shared ANN-training Stage A
            │   └── phase/gif aware 另外生成 Stage B 供 ANN fine-tuning
            └── ANN training
                └── Final ANN
                    ├── Step 7 per-run Post-finetuning Prefix
                    ├── Step 8 per-run Post-finetuning Stage A
                    └── SNN conversion / evaluation
                        ├── selector=false
                        │   └── Step 4 + Step 5 Stage A
                        └── selector=true
                            └── Step 7 + Step 8
```

注明：

```text
两套 artifacts 可以同时存在。
切换 selector 不重新训练 ANN。
切换 selector 不重新生成另一套已存在的 artifacts。
只需重新 convert / evaluate SNN。
```

---

## 7.2 修正当前 TL;DR Evaluation 样本数

当前 `configs/experiment_matrix.yaml`：

```yaml
evaluation:
  tldr_test_samples: 1000
```

Qwen3-1.7B 与 Qwen3-8B 当前都使用 1000。

删除旧表述：

```text
Qwen3-1.7B 快速评估 128
Qwen3-8B full test split
```

改成：

```text
当前 Qwen3-1.7B 与 Qwen3-8B TL;DR generated configs 均使用：
evaluation.tldr_test_samples: 1000

如果后续 matrix 改为 null，则表示完整 evaluation split。
```

不要在文档里硬编码未来可能变化的 test split 总数。

---

## 7.3 Temporal v7 -> Temporal v8

当前：

```python
TEMPORAL_IMPLEMENTATION_VERSION = 8
```

因此当前入口文档里所有：

```text
Temporal v7
temporal v7
```

改为：

```text
Temporal v8
temporal v8
```

特别修正附录 G 标题与正文。

---

## 7.4 Post-finetuning Stage B

保留并强化：

```text
Post-finetuning 禁止运行 Stage B。
```

修改代码后确保文档与脚本严格一致。

---

# 8. 修正 `代码结构总结.md`

遵守现有 `AGENTS.md`：

```text
只保留 2. 目录结构
每个文件一句话描述职责
```

不要新增说明章节。

---

## 8.1 `snn2/artifacts.py`

更新为类似：

```text
组织 shared Pre/ANN-training bundle、per-run Post-finetuning bundle、
selector-aware SNN 路径以及 G/N/T/K 隔离。
```

---

## 8.2 `snn2/config.py`

更新为类似：

```text
验证 dual-artifact selector、vanilla selector 限制、
ANN/SNN Prefix stage 语义以及 temporal/calibration 配置。
```

---

## 8.3 `snn2/conversion.py`

更新为类似：

```text
按 conversion.use_post_finetuning_artifacts 选择 Prefix/Stage A，
校验 source provenance，并生成/验证当前 conversion descriptor。
```

不得继续描述成：

```text
aware 固定复用 ANN-training artifacts
```

---

## 8.4 `snn2/modeling.py`

更新为类似：

```text
按 stage 解析模型与 Prefix，
并将 Final ANN Prefix source 与 Final SNN selector source 解耦。
```

---

## 8.5 `snn2/evaluation.py`

更新为类似：

```text
构造 mode-aware static ANN 与 selector-aware temporal SNN controller，
并记录 calibration/forward provenance。
```

---

## 8.6 `scripts/calibrate_sites.py`

更新为：

```text
支持 ANN-training A/B、vanilla-analysis A 和 Post-finetuning A；
Post-finetuning Stage B 明确禁止。
```

---

## 8.7 `scripts/evaluate_lm_harness.py`

更新为：

```text
执行 Base/Final ANN/Temporal SNN lm-eval，
并在 SNN evaluation 前验证 selector 对应 conversion descriptor。
```

---

## 8.8 schema/version 描述

当前代码常量：

```text
TEMPORAL_IMPLEMENTATION_VERSION = 8
STATISTICS_FORMAT_VERSION = 4
CONVERSION_METADATA_FORMAT_VERSION = 14
```

因此至少修：

### `test_conversion_metadata.py`

旧：

```text
v12
```

改：

```text
v14
```

### `test_generated_configs.py`

旧：

```text
temporal v7
```

改：

```text
temporal v8
```

### `test_temporal_ops.py`

旧：

```text
temporal v7
```

改：

```text
temporal v8
```

### `test_statistics.py`

旧：

```text
statistics v3
```

改：

```text
statistics v4
```

如果该文件中还有其他已过时 schema/version，一并按实际常量修正。

---

## 8.9 history 文件列表

在：

```text
docs/history/
```

列表中加入：

```text
SNN_conversion_dual_artifact_source_modification_plan.md
```

一句话标明：

```text
保存 SNN conversion 双套 artifact source selector 的实施方案。
```

---

# 9. 测试补强

本轮不重新测试整个 neuron 数学，只补 dual-artifact protocol 关键缺口。

---

## 9.1 aware Post-finetuning artifact path

当前 Post-finetuning path 测试若只覆盖：

```text
vanilla
unaware
```

扩展到：

```python
@pytest.mark.parametrize(
    "mode",
    ["vanilla", "unaware", "phase_aware", "gif_aware"],
)
```

验证：

```python
layout.post_finetuning_prefix_dir
layout.post_finetuning_site_dir
```

都属于：

```text
layout.root / "post_finetuning"
```

对于 aware 额外验证：

```python
layout.post_finetuning_site_dir != layout.ann_training_site_dir
layout.post_finetuning_prefix_dir != layout.ann_training_prefix_dir
```

并确认路径不在：

```text
_shared/seed42/rotated_prefix/
```

---

## 9.2 selector 不影响 ANN identity

对：

```text
unaware
phase_aware
gif_aware
```

分别构造：

```python
cfg_true
cfg_false
```

验证：

```python
ArtifactLayout(cfg_true).ann_checkpoint_dir
==
ArtifactLayout(cfg_false).ann_checkpoint_dir
```

同时：

```python
ArtifactLayout(cfg_true).snn_dir("phase")
!=
ArtifactLayout(cfg_false).snn_dir("phase")
```

以及：

```python
conversion_prefix_dir(true) != conversion_prefix_dir(false)
conversion_site_dir(true) != conversion_site_dir(false)
```

Final ANN Prefix source 必须在 true/false 下完全一致。

---

## 9.3 selector=false 完整 conversion regression

当前 `test_conversion_metadata.py` 不要只测 `use_post=true`。

新增以下完整路径。

### A. unaware + false

构造：

```text
shared Pre Prefix
shared ANN-training Stage A
Final unaware ANN
```

调用：

```python
create_conversion(...)
validate_conversion_metadata(...)
```

验证 metadata：

```json
{
  "use_post_finetuning_artifacts": false,
  "prefix_source_stage": "pre_finetuning",
  "calibration_source_stage": "ann_training",
  "reused_ann_training_artifacts": true,
  "post_finetuning_recalibration": false
}
```

最重要：

```text
unaware + false
```

不得调用 aware training provenance validator。

可 monkeypatch：

```python
_validate_aware_training_provenance
```

使其一旦调用就：

```python
raise AssertionError(...)
```

测试必须仍通过。

---

### B. phase_aware + false

准备有效：

```text
training_result.json
ANN-training Prefix provenance
ANN-training Stage A provenance
Stage B Clip profile provenance
```

然后：

```python
create_conversion(...)
validate_conversion_metadata(...)
```

必须通过。

之后篡改：

```text
ANN-training calibration manifest/hash
```

确认 validation fail-fast。

---

### C. phase_aware + true

使用：

```text
Post-finetuning Prefix
Post-finetuning Stage A
```

确认：

```text
不调用 aware shared-bundle training provenance validator
```

并确认：

```text
Post Stage A hash 可以与 ANN-training Stage A 不同
```

仍合法。

source model 必须是：

```text
Final phase-aware ANN
```

gif-aware 至少补一个等价 source-path test。

---

# 10. `verify_artifacts.py` selector regression

至少覆盖：

---

## 10.1 unaware + false

验证：

```text
prefix source = shared Pre
calibration source = shared ANN-training Stage A
```

同时不要求：

```text
aware ANN Stage B Clip provenance
```

---

## 10.2 aware + false

验证：

```text
source = shared Pre bundle
```

且必须要求：

```text
training_result.json
ANN-training Prefix provenance
ANN-training Stage A provenance
Stage B provenance
```

---

## 10.3 aware + true

验证：

```text
source = per-run Post bundle
post_finetuning_recalibration = true
```

不要求：

```text
Post Stage A hash == ANN-training Stage A hash
```

---

## 10.4 vanilla + false

保持：

```text
config validation 直接拒绝
```

不要让 verify 再承担这个错误分支。

---

# 11. lm-harness SNN smoke test

必须新增一个测试专门捕获本轮缺 import 问题。

最低要求：

```python
module = load evaluate_lm_harness.py
assert callable(module.validate_conversion_metadata)
```

更推荐：

```text
monkeypatch setup/parser/validation
```

让：

```text
--neuron phase
```

实际走到：

```python
validate_conversion_metadata(cfg, layout, "phase")
```

并验证调用一次。

---

# 12. Post-finetuning Stage B rejection test

增加明确测试：

```text
post_finetuning + B
    -> ValueError
```

错误消息包含：

```text
Stage A only
```

并确认：

```text
ann_training + B
```

仍正常。

---

# 13. 核对 `scripts/verify_artifacts.py` 当前 source matrix

主体 selector 逻辑目前已经基本正确，不需要重构。

修改/补测试时必须保持以下 invariant。

---

## selector=true

必须：

```text
layout.conversion_prefix_dir
    == layout.post_finetuning_prefix_dir

layout.conversion_site_dir
    == layout.post_finetuning_site_dir

prefix_source_stage
    == post_finetuning

calibration_source_stage
    == post_finetuning

reused_ann_training_artifacts
    == false

post_finetuning_recalibration
    == true
```

source calibration manifest：

```text
source_model_stage
    = final_ann_checkpoint

source_ann_mode
    = 当前 ann_mode

source_ann_checkpoint
    = 当前 layout.ann_checkpoint_dir
```

---

## selector=false

必须：

```text
layout.conversion_prefix_dir
    == layout.ann_training_prefix_dir

layout.conversion_site_dir
    == layout.ann_training_site_dir

prefix_source_stage
    == pre_finetuning

calibration_source_stage
    == ann_training

reused_ann_training_artifacts
    == true

post_finetuning_recalibration
    == false
```

source calibration manifest：

```text
source_model_stage
    = rotated_fused_base

source_ann_mode
    = None

source_ann_checkpoint
    = None
```

### aware false

必须：

```text
validate ANN-training frozen provenance
```

### unaware false

不得：

```text
validate aware ANN-training provenance
```

---

# 14. 不要改动当前已经正确的实现

本轮禁止破坏以下行为。

---

## 14.1 Config selector

继续由：

```python
use_post_finetuning_artifacts(cfg)
```

作为 SNN source 的唯一决定条件。

不要重新使用：

```python
is_aware_ann_mode(cfg)
```

直接决定 conversion source。

---

## 14.2 Final ANN Evaluation Prefix

保持：

```text
vanilla -> no Prefix
unaware -> Post-finetuning Prefix
phase_aware/gif_aware -> Pre-finetuning Prefix
```

是否实际加载继续由：

```yaml
evaluation:
  prefix_enabled: true|false
```

控制。

selector 不影响 Final ANN Evaluation。

---

## 14.3 Prefix helper 分离

继续保留：

```python
final_ann_evaluation_prefix_artifact_stage()
final_snn_evaluation_prefix_artifact_stage()
```

不要重新合并成一个 mode-aware final helper。

---

## 14.4 SNN 输出路径

继续：

```text
snn/use_post_finetuning_artifacts_true/...
snn/use_post_finetuning_artifacts_false/...
```

---

## 14.5 shared false bundle

保持：

```text
unaware
phase_aware
gif_aware
```

在 selector=false 时共享：

```text
_shared/seed42/rotated_prefix/pre_finetuning_prefix
_shared/seed42/rotated_prefix/ann_training_calibration
```

unaware 不单独运行 Step 5。

---

## 14.6 Post bundle

四种 mode 都允许：

```bash
python scripts/discover_prefix.py \
  --config "$CFG" \
  --stage post_finetuning
```

以及：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage post_finetuning \
  --calibration-phase A
```

source 都是各自：

```text
Final ANN checkpoint
```

---

# 15. 当前入口 Markdown 全仓清理

修改结束前，对仓库执行搜索。

重点搜索：

```text
aware 模式在转换时复用
phase_aware 和 gif_aware 复用
aware conversion 可以复用
vanilla/unaware 使用 Post-finetuning
snn/<neuron>
snn/calibration_group_size_
Temporal v7
temporal v7
statistics v3
conversion v12
Post-finetuning Stage B
post_finetuning_clip_profile
```

注意：

```text
docs/history/
```

属于历史方案，可以保留历史语义。

不要把历史文档强行重写成当前规范。

当前规范必须以以下文件和实际代码一致：

```text
README.md
AGENTS.md
实验执行总结.md
代码结构总结.md
```

---

# 16. 推荐保留一份 canonical source matrix

在 `README.md` 或 `实验执行总结.md` 至少保留一份：

| ANN mode | selector | SNN Prefix | SNN Calibration | Calibration source model |
|---|---:|---|---|---|
| vanilla | true | Post-finetuning | Post-finetuning Stage A | Final vanilla ANN |
| vanilla | false | 非法 | 非法 | - |
| unaware | true | Post-finetuning | Post-finetuning Stage A | Final unaware ANN |
| unaware | false | shared Pre-finetuning | shared ANN-training Stage A | rotated fused Base |
| phase_aware | true | Post-finetuning | Post-finetuning Stage A | Final phase-aware ANN |
| phase_aware | false | shared Pre-finetuning | shared ANN-training Stage A | rotated fused Base |
| gif_aware | true | Post-finetuning | Post-finetuning Stage A | Final gif-aware ANN |
| gif_aware | false | shared Pre-finetuning | shared ANN-training Stage A | rotated fused Base |

表后必须注明：

```text
conversion.use_post_finetuning_artifacts
只影响 SNN conversion / SNN evaluation，
不影响 ANN checkpoint 或 Final ANN Evaluation。
```

---

# 17. 最终验证

修改完成后执行：

```bash
cd /home/wangwenkang/SNN

python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```

服务器环境可用：

```bash
conda run -n snn2 python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

conda run -n snn2 pytest -q
```

并执行静态语法检查：

```bash
python -m py_compile \
  scripts/evaluate_lm_harness.py \
  scripts/evaluate_tldr.py \
  scripts/calibrate_sites.py \
  scripts/verify_artifacts.py \
  snn2/config.py \
  snn2/artifacts.py \
  snn2/modeling.py \
  snn2/conversion.py \
  snn2/evaluation.py
```

---

# 18. 完成标准

以下全部满足才算本轮完成：

1. `evaluate_lm_harness.py` SNN 分支不再缺 `validate_conversion_metadata` import
2. `post_finetuning + calibration-phase B` 被代码明确拒绝
3. `ann_training + B` 仍合法
4. Post-finetuning Stage A 四种 mode 全部仍可生成
5. selector true/false 不改变 ANN checkpoint path
6. unaware false 继续复用 Step 4/5 shared bundle
7. unaware false 不要求 aware training provenance
8. aware false 继续校验 ANN-training frozen provenance
9. aware true 使用自身 Final ANN 的 Step 7/8 bundle
10. aware true 不要求 Post Stage A hash 等于 ANN-training Stage A
11. vanilla false 继续由 config validation 拒绝
12. `README.md` 不再写固定 mode-aware SNN conversion source
13. `AGENTS.md` 明确 selector 是 SNN source 唯一决定条件
14. `AGENTS.md` 的 SNN path 包含 `use_post_finetuning_artifacts_<bool>`
15. `AGENTS.md` 的 Stage A/B/Clip 语义与当前代码一致
16. `实验执行总结.md` 附录 A 显示双 bundle 与 selector 分叉
17. `实验执行总结.md` 不再写 Qwen3-1.7B=128 / Qwen3-8B=full 的旧评估规模
18. 当前入口文档全部从 Temporal v7 更新为 Temporal v8
19. `代码结构总结.md` 的 conversion v14 / statistics v4 / temporal v8 与代码一致
20. aware Post-finetuning artifact path 有明确 test
21. selector=false conversion metadata 有完整 test
22. selector true/false verify 有明确 regression
23. lm-harness SNN validation/import 有 smoke test
24. `pytest -q` 全部通过

---

# 19. 本轮禁止事项

- 不要重新改变 `conversion.use_post_finetuning_artifacts` 的含义
- 不要让 selector 进入 ANN run root
- 不要让 selector 改变 Final ANN Evaluation Prefix source
- 不要让 unaware 单独生成 ANN-training Stage A
- 不要重新禁止 aware Post-finetuning Prefix / Stage A
- 不要把 aware Post-finetuning artifact 放进 `_shared`
- 不要允许 Post-finetuning Stage B
- 不要恢复“aware 固定 reuse / unaware 固定 post”的旧逻辑
- 不要要求 aware true 的 Post Stage A 与 ANN-training Stage A hash 相同
- 不要让 unaware false 校验 aware Clip/training provenance
- 不要修改已经正确的 Site topology、Phase/MTN/GIF 数学、Final RMSNorm 和 Prefix temporal policy
- 不要将 `docs/history/` 中历史方案误改成当前规范
