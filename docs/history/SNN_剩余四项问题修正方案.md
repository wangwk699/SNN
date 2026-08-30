# SNN Calibration A/B 重构后剩余四项问题修正方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 适用基线：已经完成此前 Calibration A/B、`num_samples` sweep、Prefix 按 `num_samples` 隔离、Stage B Clip profile、runtime T/K 解耦等重构后的当前 `main`。  
> 本文档只处理本轮检查后确认的 4 个剩余问题，不重复已有总体设计。  
> 要求：服务器端 Codex 在无对话上下文的情况下，仅根据本文档即可完成本轮修正。

---

# 1. 本轮需要修正的四个问题

本轮只处理以下 4 项：

1. Rotation regression 内部仍然错误读取 `cfg.calibration.num_samples`，尚未真正固定为 canonical 128。
2. `scripts/regress_phase_conversion.py` 使用 `apply_deployment_overrides()` 但漏掉 import。
3. Evaluation output path 对 `num_samples` 的隔离规则还不完整，需要按“evaluation 是否实际使用随 `num_samples` 变化的 Prefix”条件化处理。
4. Prefix discovery 的 `resolved_config.yaml` 和 logs 仍未按 `num_samples` 隔离。

其中：

- 1、2 是直接运行错误 / blocker；
- 3、4 会导致 sweep 结果覆盖或 provenance 不清晰。

完成后必须：

```bash
pytest -q
```

全部通过。

---

# 2. 修正 1：Rotation regression 永久固定 canonical 128

## 2.1 最终原则

Rotation regression 的数据来源永久固定为：

```text
canonical preprocessing calibration
num_samples = 128
without replacement
```

并且：

> `calibration.num_samples` 只控制 Stage A calibration 以及随 Stage A sweep 的 Prefix discovery，绝不能重新定义 Rotation regression 的数据来源。

最终关系：

```text
Rotation regression samples = fixed canonical 128

Prefix discovery samples
    = cfg.calibration.num_samples

Stage A calibration samples
    = cfg.calibration.num_samples
```

---

## 2.2 当前已正确的部分

`prepare_rotation.py` 已经改成：

```python
calibration = load_canonical_preprocessing_raw(cfg, layout)
manifest_path = layout.canonical_preprocessing_calibration_manifest_path
```

这部分保持不变。

`verify_artifacts.py` 也已经使用：

```python
layout.canonical_preprocessing_calibration_manifest_path
```

验证 Rotation regression provenance。

这部分也保持不变。

---

## 2.3 当前仍错误的部分

在：

```text
snn2/rotation.py
```

中的：

```python
validate_rotation_regression_suite(...)
```

仍然存在类似：

```python
expected_samples = int(cfg["calibration"]["num_samples"])

if expected_samples != 128 or len(calibration_dataset) != expected_samples:
    raise RuntimeError(...)
```

这是错误的。

例如：

```yaml
calibration:
  num_samples: 64
```

即使传入的已经是正确的 canonical 128 dataset：

```text
len(calibration_dataset) == 128
```

这里仍会因为：

```text
expected_samples = 64
```

而失败。

---

# 3. Rotation regression 的正确实现

建议直接复用：

```python
CANONICAL_PREPROCESSING_NUM_SAMPLES = 128
```

不要再次硬编码另一个 128。

例如：

```python
from snn2.data import CANONICAL_PREPROCESSING_NUM_SAMPLES
```

然后：

```python
expected_samples = CANONICAL_PREPROCESSING_NUM_SAMPLES

if len(calibration_dataset) != expected_samples:
    raise RuntimeError(
        "Three-way rotation regression requires the canonical "
        f"{expected_samples}-sample preprocessing selection; "
        f"dataset={len(calibration_dataset)}"
    )
```

删除：

```python
int(cfg["calibration"]["num_samples"])
```

对 Rotation sample count 的任何控制。

---

## 3.1 Manifest validation

仍然读取传入：

```python
calibration_manifest_path
```

并检查：

```text
manifest_role == canonical_preprocessing_calibration
num_samples == 128
sampling == seeded_without_replacement
duplicates_preserved == false
```

如果目前 `load_canonical_preprocessing_raw()` 已做这一层检查，可以继续保留；但 Rotation regression 自身至少还应校验：

```python
manifest["num_samples"] == CANONICAL_PREPROCESSING_NUM_SAMPLES
```

以及：

```python
len(manifest["indices"]) == CANONICAL_PREPROCESSING_NUM_SAMPLES
```

---

## 3.2 Rotation metadata

输出：

```json
{
  "num_samples": 128,
  "calibration_manifest_path": ".../canonical_preprocessing/num_samples_128/calibration_manifest.json",
  "calibration_manifest_sha256": "..."
}
```

这里的：

```text
num_samples
```

必须始终为 canonical 128。

不要增加：

```text
stage_a_num_samples
cfg_calibration_num_samples
```

之类会让 Rotation regression 看起来依赖 Stage A sweep 的字段。

---

# 4. Rotation 必须新增的测试

至少新增以下测试。

## 4.1 config num_samples != 128 仍允许 Rotation canonical 128

构造：

```python
cfg["calibration"]["num_samples"] = 64
```

同时：

```text
len(calibration_dataset) = 128
```

断言：

```text
validate_rotation_regression_suite()
```

不会因为 `cfg.calibration.num_samples=64` 而失败。

---

## 4.2 canonical dataset 非 128 必须失败

例如：

```text
len(calibration_dataset)=64
```

无论：

```text
cfg.calibration.num_samples
```

是多少，都必须 fail。

---

## 4.3 Rotation result num_samples 永远 128

config：

```text
num_samples=64
```

最后 regression metadata：

```json
"num_samples": 128
```

---

# 5. 修正 2：`regress_phase_conversion.py` 缺少 import

## 5.1 当前问题

当前：

```text
scripts/regress_phase_conversion.py
```

顶部仍然类似：

```python
from _common import parser, setup
```

但是 main 中已经调用：

```python
cfg = apply_deployment_overrides(args, cfg)
```

因此脚本会直接：

```text
NameError: name 'apply_deployment_overrides' is not defined
```

---

# 6. 正确修正

改成：

```python
from _common import apply_deployment_overrides, parser, setup
```

其他 deployment override 实现保持不变。

---

## 6.1 保持现有正确顺序

必须继续：

```python
cfg, layout = setup(args.config)

source_phase_T = int(cfg["phase"]["T"])
source_mtn_T = int(cfg["mtn"]["T"])

cfg = apply_deployment_overrides(args, cfg)
```

不要改成：

```python
apply override
→ 再创建 ArtifactLayout
```

因为：

> ArtifactLayout 必须根据 ANN training 时的 source T 构造；deployment T 只能在 layout 已固定后覆盖 runtime config。

例如：

```text
ANN training phase.T = 4
deployment --phase-T 8
```

必须仍然找到：

```text
phase_T_4_.../ann/final
```

而不是错误寻找：

```text
phase_T_8_.../ann/final
```

---

# 7. Phase regression metadata

继续同时记录：

```text
source_phase_T
source_mtn_T
deployment_phase_T
deployment_mtn_T
```

Static Phase oracle 和 temporal Phase 均使用：

```text
deployment_phase_T
```

这一语义保持不变。

---

# 8. 修正 3：Evaluation output path 的 `num_samples` 条件化隔离

这是本轮需要特别严格实现的部分。

核心原则：

> 只有当当前 evaluation 实际使用了一个随 `calibration.num_samples` 改变的 Prefix artifact 时，evaluation output path 才增加 `num_samples_N`。

不能机械地给所有 evaluation 加 `num_samples_N`。

---

# 9. Vanilla 的最终语义

## 9.1 Vanilla 仍然需要 post-finetuning artifacts

Vanilla final ANN checkpoint 之后仍必须生成：

```text
Post-finetuning Prefix
Post-finetuning Calibration Stage A
```

它们用于后续：

```text
Vanilla ANN
→ Phase/GIF/MTN SNN conversion
→ SNN evaluation
```

因此：

```python
requires_post_finetuning_artifacts(vanilla)
```

必须继续为：

```text
True
```

不能因为 Vanilla final ANN evaluation 不使用 Prefix，就关闭 post-finetuning preprocessing。

---

## 9.2 但 Vanilla final ANN evaluation 不使用 Prefix

Vanilla final ANN evaluation 的 forward 必须是：

```text
final vanilla ANN checkpoint
+ identity forward
+ no Prefix
+ no replacement calibration
```

因此：

```text
prefix_token_ids = []
prefix KV = None
evaluation prefix_enabled = false
```

无论已经生成了哪一套：

```text
post_finetuning/prefix/num_samples_N/
```

Vanilla final ANN evaluation 都不能使用它。

---

## 9.3 Vanilla final ANN evaluation 不依赖 `num_samples`

因此 Vanilla final ANN evaluation output path：

```text
vanilla/.../ann/evaluation/...
```

不需要增加：

```text
num_samples_N
```

例如 TL;DR：

```text
ann/evaluation/tldr/
└── test_samples_256/
    └── prefix_enabled_false/
        ├── metrics.json
        └── predictions.jsonl
```

保持无 `num_samples_N`。

---

# 10. Unaware final ANN evaluation

Unaware final ANN 本身是 identity replacement，但 evaluation 可按 config 选择是否使用 post-finetuning Prefix。

---

## 10.1 Unaware + `evaluation.prefix_enabled=false`

实际 forward：

```text
final unaware ANN
+ no Prefix
```

因此不依赖：

```text
calibration.num_samples
```

结果路径不加：

```text
num_samples_N
```

例如：

```text
ann/evaluation/tldr/
└── test_samples_256/
    └── prefix_enabled_false/
```

---

## 10.2 Unaware + `evaluation.prefix_enabled=true`

Final ANN evaluation 使用：

```text
post_finetuning_prefix/num_samples_N
```

因此：

```text
calibration.num_samples
→ Prefix_N
→ ANN evaluation result
```

结果必须按 N 隔离。

建议路径：

```text
ann/evaluation/tldr/
└── test_samples_256/
    └── prefix_enabled_ture/
        └── num_samples_64/
            ├── metrics.json
            └── predictions.jsonl
```

注意：

> 保留历史 typo：`prefix_enabled_ture`，除非项目统一另行清理。

---

# 11. Rotated pre-finetuning evaluation

Rotated pre-finetuning evaluation 使用固定 Rotation：

```text
Rotation = canonical 128
```

Rotation 自身不依赖：

```text
calibration.num_samples
```

因此是否需要 `num_samples_N` 只取决于是否启用 Prefix。

---

## 11.1 rotated pre-finetuning + `prefix_enabled=false`

Forward：

```text
fixed rotated Base
+ identity integration
+ no Prefix
```

结果不依赖 Stage A num_samples。

路径不加：

```text
num_samples_N
```

例如：

```text
rotated_pre_finetuning/evaluation/tldr/
└── test_samples_256/
    └── prefix_enabled_false/
```

---

## 11.2 rotated pre-finetuning + `prefix_enabled=true`

使用：

```text
pre_finetuning_prefix/num_samples_N
```

因此必须按 N 隔离。

建议：

```text
rotated_pre_finetuning/evaluation/tldr/
└── test_samples_256/
    └── prefix_enabled_ture/
        └── num_samples_64/
```

LM Harness 同理：

```text
rotated_pre_finetuning/evaluation/lm_harness/
└── prefix_enabled_ture/
    └── num_samples_64/
        └── results.json
```

---

# 12. Aware final ANN evaluation

`phase_aware` / `gif_aware` 的 ANN run root 本身已经包含：

```text
num_samples_N
```

例如：

```text
num_samples_128_lr.../
```

因此其：

```text
ann/evaluation/
```

天然已经按 N 隔离。

不要再在 evaluation path 里额外重复加一层：

```text
num_samples_N
```

否则路径冗余。

---

# 13. SNN evaluation

## Aware

Aware SNN output 已位于带 `num_samples_N` 的 ANN run root 下，因此已经隔离。

## Vanilla / Unaware

Non-aware：

```python
layout.snn_dir()
```

已经包含：

```text
calibration_group_size_G_num_samples_N
```

因此也已经隔离。

本轮不要重复修改 SNN output path。

---

# 14. 建议新增统一 helper

避免：

```text
evaluate_tldr.py
evaluate_lm_harness.py
```

各自实现一套条件判断。

建议在：

```text
snn2/evaluation.py
```

或：

```text
snn2/artifacts.py
```

新增纯 helper，例如：

```python
def evaluation_depends_on_prefix_num_samples(
    cfg: dict[str, Any],
    *,
    base: bool = False,
    rotated_pre_finetuning: bool = False,
    neuron: str = "ann",
) -> bool:
```

逻辑：

```python
if base:
    return False

if neuron != "ann":
    return False

mode = cfg["experiment"]["ann_mode"]

if rotated_pre_finetuning:
    return rotated_pre_finetuning_prefix_enabled(cfg)

if mode == "vanilla":
    return False

if mode == "unaware":
    return evaluation_prefix_enabled(cfg)

# aware run root 本身已经按 num_samples 隔离
return False
```

然后新增：

```python
def append_evaluation_num_samples_if_needed(
    output_dir: Path,
    cfg: dict[str, Any],
    **context,
) -> Path:
```

如：

```python
if evaluation_depends_on_prefix_num_samples(...):
    output_dir = output_dir / f"num_samples_{int(cfg['calibration']['num_samples'])}"
```

---

# 15. `evaluate_tldr.py` 路径修改

当前结构大致：

```python
output_dir = (
    model_output_dir
    / "evaluation"
    / "tldr"
    / test_samples_dirname
)

if not args.base:
    output_dir = output_dir / prefix_enabled_dirname(...)
```

修改为：

```python
output_dir = (
    model_output_dir
    / "evaluation"
    / "tldr"
    / test_samples_dirname
)

if not args.base:
    output_dir = output_dir / prefix_enabled_dirname(active_prefix_enabled)

if evaluation_depends_on_prefix_num_samples(
    cfg,
    base=args.base,
    rotated_pre_finetuning=args.rotated_pre_finetuning,
    neuron=args.neuron,
):
    output_dir = output_dir / (
        f"num_samples_{int(cfg['calibration']['num_samples'])}"
    )
```

---

# 16. `evaluate_lm_harness.py` 同步

当前：

```python
output_dir = model_output_dir / "evaluation" / "lm_harness"

if not args.base:
    output_dir = output_dir / prefix_enabled_dirname(active_prefix_enabled)
```

后面增加完全相同 helper：

```python
if evaluation_depends_on_prefix_num_samples(...):
    output_dir = output_dir / f"num_samples_{...}"
```

保证 TL;DR 与 lm-eval 规则完全一致。

---

# 17. Vanilla final ANN evaluation 的 Prefix policy 必须显式固定

需要审计：

```text
snn2/config.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
snn2/modeling.py
snn2/evaluation.py
scripts/verify_artifacts.py
```

确保：

```text
vanilla final ANN evaluation
```

无论：

```yaml
post_finetuning:
  prefix_enabled: true

evaluation:
  prefix_enabled: true
```

原始 matrix/config 中如何写，最终语义都不能误加载 post-finetuning Prefix。

推荐明确建立 helper，例如：

```python
def final_ann_evaluation_prefix_enabled(cfg):
    mode = cfg["experiment"]["ann_mode"]

    if mode == "vanilla":
        return False

    return evaluation_prefix_enabled(cfg)
```

避免继续直接调用通用：

```python
evaluation_prefix_enabled(cfg)
```

来决定 Vanilla final ANN forward。

---

# 18. 注意：不要关闭 Vanilla post-finetuning Prefix

必须区分：

```text
final_ann_evaluation_prefix_enabled(vanilla) = false
```

与：

```text
post_finetuning_prefix_enabled(vanilla)
```

后者仍可为 true，因为它服务于 SNN conversion。

也就是说：

```text
Vanilla:
    生成 post-finetuning Prefix      -> YES
    生成 post-finetuning Stage A     -> YES
    final ANN evaluation 使用 Prefix -> NO
    SNN conversion 使用 Prefix       -> YES（按 conversion policy）
```

不要因为 ANN evaluation 关闭 Prefix 而删除：

```text
post_finetuning/prefix/num_samples_N
```

---

# 19. 修正 4：Prefix discovery config/log 按 `num_samples` 隔离

## 19.1 当前问题

Prefix state/KV 已经正确放到：

```text
pre_finetuning_prefix/num_samples_N/
post_finetuning/prefix/num_samples_N/
```

但是 `discover_prefix.py` 当前 pre-finetuning 仍使用：

```python
config_scope = "policy_shared"
logs_dir = layout.policy_logs_dir
```

post-finetuning 仍使用：

```python
config_scope = "run"
logs_dir = layout.logs_dir
```

因此不同：

```text
num_samples=64
num_samples=128
num_samples=256
```

会共享：

```text
resolved_config.yaml
*.log
*.jsonl
*_result.json
```

特别是：

```text
*_result.json
```

会被后一次直接覆盖。

---

# 20. Pre-finetuning Prefix 新目录结构

改成：

```text
pre_finetuning_prefix/
└── num_samples_64/
    ├── config/
    │   └── resolved_config.yaml
    ├── logs/
    │   ├── discover_prefix_ann_training.log
    │   ├── discover_prefix_ann_training.jsonl
    │   └── discover_prefix_ann_training_result.json
    ├── prefix_state.json
    └── prefixed_key_values.pt
```

`num_samples_128`、`256` 分别独立。

---

# 21. Post-finetuning Prefix 新目录结构

```text
post_finetuning/
└── prefix/
    └── num_samples_64/
        ├── config/
        │   └── resolved_config.yaml
        ├── logs/
        │   ├── discover_prefix_post_finetuning.log
        │   ├── discover_prefix_post_finetuning.jsonl
        │   └── discover_prefix_post_finetuning_result.json
        ├── prefix_state.json
        └── prefixed_key_values.pt
```

---

# 22. ArtifactLayout 新属性

新增：

```python
@property
def ann_training_prefix_config_dir(self) -> Path:
    return self.ann_training_prefix_dir / "config"

@property
def ann_training_prefix_logs_dir(self) -> Path:
    return self.ann_training_prefix_dir / "logs"

@property
def post_finetuning_prefix_config_dir(self) -> Path:
    return self.post_finetuning_prefix_dir / "config"

@property
def post_finetuning_prefix_logs_dir(self) -> Path:
    return self.post_finetuning_prefix_dir / "logs"
```

`ensure()` 同步创建。

---

# 23. `_common.setup()` 增加 Prefix scopes

增加：

```python
elif config_scope == "ann_training_prefix":
    config_dir = layout.ann_training_prefix_config_dir

elif config_scope == "post_finetuning_prefix":
    config_dir = layout.post_finetuning_prefix_config_dir
```

---

# 24. `discover_prefix.py` 修改

原先：

```python
config_scope = (
    "policy_shared"
    if canonical_stage == "pre_finetuning"
    else "run"
)
```

改成：

```python
config_scope = (
    "ann_training_prefix"
    if canonical_stage == "pre_finetuning"
    else "post_finetuning_prefix"
)
```

logs：

```python
logs_dir = (
    layout.ann_training_prefix_logs_dir
    if canonical_stage == "pre_finetuning"
    else layout.post_finetuning_prefix_logs_dir
)
```

output_dir 保持：

```python
layout.ann_training_prefix_dir
layout.post_finetuning_prefix_dir
```

---

# 25. Rotation config/log 继续独立

Rotation 继续：

```text
policy_shared/config
policy_shared/logs
rotation/
```

Prefix discovery 不再写：

```text
policy_shared/config/resolved_config.yaml
policy_logs_dir
```

这样可以避免：

```text
prepare_rotation canonical 128
```

之后又被：

```text
Prefix num_samples=256
```

的 resolved config/log 覆盖或混淆。

---

# 26. 需要修改/审计的文件

至少：

```text
snn2/rotation.py
snn2/artifacts.py
snn2/config.py
snn2/evaluation.py
snn2/modeling.py

scripts/_common.py
scripts/regress_phase_conversion.py
scripts/discover_prefix.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
```

Tests 至少：

```text
tests/test_rotation_regression.py
tests/test_evaluation_paths.py
tests/test_post_finetuning_protocol.py
tests/test_rotated_pre_finetuning_protocol.py
tests/test_phase_conversion_regression.py
tests/test_verify_artifacts.py
```

如有 Prefix path 专门测试文件，同步修改。

---

# 27. 必须新增/修改的 tests

## 27.1 Rotation 与 Stage A num_samples 解耦

```text
cfg.calibration.num_samples = 64
canonical dataset length = 128
```

Rotation validator 不得因 cfg=64 失败。

---

## 27.2 Rotation canonical 非 128 失败

```text
canonical dataset length = 64
```

必须 fail。

---

## 27.3 Phase regression import/CLI smoke

至少直接 import：

```text
scripts/regress_phase_conversion.py
```

或测试 CLI parser/deployment override path，防止再次出现 `NameError`。

---

## 27.4 Vanilla final ANN path 不含 num_samples

即使：

```text
calibration.num_samples=64
post_finetuning.prefix_enabled=true
```

Vanilla final ANN evaluation：

```text
prefix=false
output path 不含 num_samples_64
```

---

## 27.5 Vanilla 仍需要 post-finetuning artifacts

断言：

```text
requires_post_finetuning_artifacts(vanilla) == True
```

同时：

```text
final ANN evaluation Prefix == False
```

这是两个独立语义。

---

## 27.6 Unaware final ANN Prefix=false

```text
evaluation.prefix_enabled=false
```

path 不含：

```text
num_samples_N
```

---

## 27.7 Unaware final ANN Prefix=true

64 和 128：

```text
output path 不同
```

分别包含：

```text
prefix_enabled_ture/num_samples_64
prefix_enabled_ture/num_samples_128
```

---

## 27.8 Rotated pre-finetuning Prefix=false

64/128：

```text
output path 相同
```

且不含：

```text
num_samples_N
```

---

## 27.9 Rotated pre-finetuning Prefix=true

64/128：

```text
output path 不同
```

分别包含对应：

```text
num_samples_N
```

---

## 27.10 Aware final ANN 不重复 num_samples

Aware ANN root 本身已包含：

```text
num_samples_N
```

evaluation 子目录不要再额外重复一层。

---

## 27.11 Prefix config/log isolation

64/128：

```text
ann_training_prefix_config_dir 不同
ann_training_prefix_logs_dir 不同

post_finetuning_prefix_config_dir 不同
post_finetuning_prefix_logs_dir 不同
```

并且均位于：

```text
.../num_samples_N/
```

下。

---

## 27.12 Prefix sweep 不覆盖 Rotation config

模拟：

```text
prepare_rotation
discover_prefix num_samples=64
discover_prefix num_samples=128
```

断言 Rotation 的：

```text
policy_shared/config/resolved_config.yaml
```

未被 Prefix discovery 修改。

---

# 28. `verify_artifacts.py` 需要同步的 evaluation 规则

验证 evaluation path 时，必须使用与 runtime 相同 helper。

不要自己重新手写另一套规则。

应验证：

### Vanilla final ANN

```text
prefix_enabled_false
no num_samples_N
```

### Unaware final ANN

```text
prefix=false -> no N
prefix=true  -> num_samples_N
```

### rotated pre-finetuning

```text
prefix=false -> no N
prefix=true  -> num_samples_N
```

### aware final ANN

```text
run root 已含 N
evaluation path 不再重复 N
```

---

# 29. 不允许的实现方式

## 29.1 不要重新让 Rotation 读取 Stage A num_samples

禁止：

```python
expected_samples = cfg["calibration"]["num_samples"]
```

出现在 Rotation regression sample-count 逻辑中。

---

## 29.2 不要关闭 Vanilla post-finetuning preprocessing

禁止因为：

```text
Vanilla final ANN 不用 Prefix
```

而改成：

```text
Vanilla 不生成 post-finetuning Prefix/calibration
```

Vanilla 后续 SNN conversion 仍需要这些 artifact。

---

## 29.3 不要给所有 ANN evaluation 统一加 `num_samples_N`

特别是：

```text
Vanilla final ANN
Unaware Prefix=false
rotated pre-finetuning Prefix=false
```

都不应因为 config 中存在 `calibration.num_samples` 就被强行分叉。

---

## 29.4 不要给 aware evaluation 重复加 N

Aware run root 已经有：

```text
num_samples_N
```

不要再次：

```text
.../evaluation/.../num_samples_N
```

重复表达。

---

## 29.5 不要继续让 Prefix discovery 使用 `policy_shared` / `run` config scope

Prefix 已经成为：

```text
per-num_samples artifact
```

其 config/log 也必须跟随 Prefix root。

---

# 30. 最终验收标准

全部满足才算本轮完成。

1. `pytest -q` 全部通过。
2. Rotation regression 使用固定 canonical 128。
3. `cfg.calibration.num_samples=64/256` 不影响 Rotation expected sample count。
4. Rotation regression metadata 永远记录 `num_samples=128`。
5. Rotation provenance 始终绑定 canonical preprocessing manifest。
6. `regress_phase_conversion.py` 不再出现 `NameError`。
7. Phase regression 仍支持 training T != deployment T。
8. Vanilla 仍生成 post-finetuning Prefix。
9. Vanilla 仍生成 post-finetuning Stage A calibration。
10. Vanilla final ANN evaluation 不使用 Prefix。
11. Vanilla final ANN output path 不含 `num_samples_N`。
12. Unaware final ANN Prefix=false 时 path 不含 N。
13. Unaware final ANN Prefix=true 时 path 含对应 N。
14. Rotated pre-finetuning Prefix=false 时 path 不含 N。
15. Rotated pre-finetuning Prefix=true 时 path 含对应 N。
16. Aware final ANN evaluation 不重复增加 N。
17. Non-aware SNN 路径现有 `calibration_group_size_G_num_samples_N` 规则保持不变。
18. Pre-finetuning Prefix config/log 按 N 隔离。
19. Post-finetuning Prefix config/log 按 N 隔离。
20. Prefix discovery 不再覆盖/污染 Rotation 的 shared resolved config 与 logs。

---

# 31. 推荐实施顺序

建议 Codex 按以下顺序实施：

1. 修 `snn2/rotation.py`：Rotation expected samples 固定 canonical 128。
2. 补 `regress_phase_conversion.py` import。
3. 在 `snn2/config.py` 明确区分：
   ```text
   post-finetuning Prefix 是否生成
   final ANN evaluation Prefix 是否使用
   ```
4. 新增统一 evaluation `num_samples` dependency helper。
5. 修改 TL;DR evaluation path。
6. 修改 lm-eval path。
7. 新增 Prefix config/log ArtifactLayout。
8. 新增 Prefix-specific config scopes。
9. 修改 `discover_prefix.py` 的 config/log destination。
10. 同步 `verify_artifacts.py`。
11. 更新 tests。
12. `pytest -q`。
13. 最后做最小 smoke test：

```text
Rotation with cfg num_samples=64
    -> still canonical 128

Vanilla:
    post-FT Prefix A exists
    post-FT Stage A exists
    final ANN Prefix=false
    eval path no num_samples

Unaware:
    Prefix=false -> no num_samples
    Prefix=true, N=64/128 -> distinct paths

Rotated pre-finetuning:
    Prefix=false -> same path across N
    Prefix=true, N=64/128 -> distinct paths

Phase conversion regression:
    source T=4
    deployment T=8
    runs without NameError
```
