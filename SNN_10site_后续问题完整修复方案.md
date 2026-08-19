# SNN：10-Site 修改后的问题完整修复方案

> **目标读者**：服务器端 Codex / 编程智能体  
> **目标仓库**：`/home/wangwenkang/SNN`，GitHub `wangwk699/SNN`  
> **目标分支**：`main`  
> **参考提交**：10-site 修改前 `24b5efa70413c4d05e121400b513294c26c8d3f5`；已知 10-site 修改后提交 `dd9c839c8f8b41b9b191f593c17d374a82c11acf`。  
> **目标**：保留已经正确的 10-site activation replacement，不重做该算法；修复配置回退、`ann/final`/`ann/best` 不一致、TL;DR verification 路径错误、测试覆盖不足以及 Markdown 与代码不一致等问题，使从 config materialization → calibration → ANN training → ANN evaluation → SNN conversion → SNN evaluation → artifact verification 的整条链路一致。

---

## 0. 先确认当前代码状态

```bash
cd /home/wangwenkang/SNN
git status
git branch --show-current
git rev-parse HEAD
```

不要覆盖用户未提交改动，不要 `git reset --hard`，不要 `git clean`。

先确认 10-site 核心仍然正确：

```bash
python - <<'PY'
from snn2.sites import (
    SITE_COUNT, SITE_IDS, SITE_NAMES,
    SITE_COORDINATES, SITE_TOPOLOGY_VERSION,
)

assert SITE_TOPOLOGY_VERSION == 2
assert SITE_COUNT == 10
assert SITE_IDS == tuple(range(1, 11))
assert SITE_NAMES[9] == "post_mlp_up_proj"
assert SITE_COORDINATES[9] == "I"
assert SITE_NAMES[10] == "post_mlp_product_r4"
assert SITE_COORDINATES[10] == "R4"
print("10-site topology OK")
PY
```

必须继续保持 MLP：

```text
gate_proj → SiLU → Site 8 ───┐
                             ⊙ → R4 → Site 10 → down_proj
up_proj        → Site 9 ─────┘
```

Site 8/9 的 product-aware GIF saliency 继续使用 `gate² × up²`；Site 10 继续使用 `down_proj` linear-consumer saliency。不要新增 R5，不要修改 Prefix/Rotation 算法。

---

# 1. 问题一：`materialize_configs.py` 覆盖了 Qwen3-1.7B 的正式超参数

10-site 修改前的提交 `24b5efa...` 中，Qwen3-1.7B 正式 generated config 为：

| mode | `training.learning_rate` | `evaluation.batch_size` |
|---|---:|---:|
| vanilla | `1.0e-6` | `32` |
| unaware | **`5.0e-6`** | `32` |
| phase_aware | `1.0e-6` | `32` |
| gif_aware | `1.0e-6` | `32` |

10-site 修改后重新 materialize 时，由于 `experiment_matrix.yaml` 默认值是：

```yaml
training:
  learning_rate: 1.0e-6

evaluation:
  batch_size: 8
```

导致 Qwen3-1.7B 的特殊值被覆盖。

必须恢复上述四个 config 的正式参数，同时保持：

```yaml
calibration:
  expected_sites_per_layer: 10
```

Qwen3-8B、Llama3-8B 不要因此被改成 Qwen3-1.7B 的 batch size 或 learning rate。

---

# 2. 修复配置源：`experiment_matrix.yaml` 成为唯一事实来源

不要直接手改 generated YAML 作为最终解决方案。

推荐在 Qwen3-1.7B 的 `model_runs` 项中加入 model-level 和 mode-level override：

```yaml
model_runs:
  - name: exp1_qwen3_1_7b_tldr
    config:
      experiment:
        task: tldr
        model_name: Qwen/Qwen3-1.7B-Base
        model_revision: ea980cb
      data:
        dataset_name: trl-lib/tldr
        dataset_revision: 21233da376667088e6eb1ce4ce19ed832c2935d3
        train_split: train
        validation_split: validation
        evaluation_split: test
      evaluation:
        batch_size: 32

    mode_overrides:
      unaware:
        training:
          learning_rate: 5.0e-6
```

这样：

```text
Qwen3-1.7B all modes: evaluation.batch_size = 32
Qwen3-1.7B unaware:    learning_rate = 5e-6
其它 Qwen3-1.7B mode:  learning_rate = default 1e-6
```

---

# 3. 修改 `scripts/materialize_configs.py`

合并顺序必须为：

```text
defaults
  ↓
model_run["config"]
  ↓
model_run["mode_overrides"].get(mode, {})
  ↓
experiment.ann_mode = mode
  ↓
resolve_config()
  ↓
validate_config()
```

推荐实现：

```python
for model_run in matrix["model_runs"]:
    for mode in matrix["ann_modes"]:
        cfg = deep_merge(
            matrix["defaults"],
            model_run["config"],
        )

        mode_override = (
            model_run
            .get("mode_overrides", {})
            .get(mode, {})
        )

        cfg = deep_merge(cfg, mode_override)

        cfg.setdefault(
            "experiment", {}
        )["ann_mode"] = mode

        cfg = resolve_config(cfg)
        validate_config(cfg)
        ...
```

mode override 必须在 `resolve_config()` 前生效；`resolve_config()` 仍负责四种 mode 的 rotation/prefix/replacement 语义。

---

# 4. 增加 generated config 回归测试

新增例如：

全部 12 个 generated config 都必须：

```python
cfg["calibration"]["expected_sites_per_layer"] == SITE_COUNT == 10
```

修改后运行：

```bash
python scripts/materialize_configs.py
```

并确认二次运行幂等，不再产生额外 config diff。

---

# 5. 问题二：ANN checkpoint `final` / `best` 语义冲突

当前真实代码行为：

```text
training.py  → 保存 ann/final
modeling.py  → ANN evaluation 加载 ann/final
```

但：

```text
conversion.py      → 仍要求 ann/best
verify_artifacts.py → 仍要求 ann/best
```

当前训练 config 又是：

```yaml
eval_strategy: "no"
save_strategy: "no"
load_best_model_at_end: false
```

因此本项目当前根本没有 validation-selected `ann/best` 的正式语义。

**统一使用 `ann/final`。**

不要为了兼容旧代码伪造 `ann/best`、复制 final 为 best 或创建 symlink。

---

# 6. 推荐加入 canonical ANN checkpoint property

在 `snn2/artifacts.py` 的 `ArtifactLayout` 增加：

```python
@property
def ann_checkpoint_dir(self) -> Path:
    return self.ann_dir / "final"
```

然后统一修改：

### `snn2/training.py`

```python
final_dir = layout.ann_checkpoint_dir
trainer.save_model(str(final_dir))
```

### `snn2/modeling.py`

```python
if ann:
    return str(layout.ann_checkpoint_dir)
```

### `snn2/conversion.py`

旧：

```python
ann_checkpoint = layout.ann_dir / "best"
```

新：

```python
ann_checkpoint = layout.ann_checkpoint_dir
```

错误信息改成：

```text
The final fine-tuned ANN checkpoint is required before conversion
```

`conversion_metadata.json` 的：

```text
source_ann_checkpoint
source_ann_config_sha256
```

必须指向：

```text
ann/final
ann/final/config.json
```

### `scripts/verify_artifacts.py`

旧：

```python
layout.ann_dir / "best" / "config.json"
```

新：

```python
layout.ann_checkpoint_dir / "config.json"
```

---

# 7. `training_result.json` 建议明确记录 final checkpoint

训练结束的 metadata 增加：

```python
"final_model_checkpoint": str(
    layout.ann_checkpoint_dir.resolve()
)
```

Trainer 自带的：

```text
best_model_checkpoint
best_metric
```

可保留作为原始 Trainer state，但项目任何下游流程不得依赖它们。

---

# 8. 清除当前态中的旧 `ann/best`

执行：

```bash
rg -n \
  'ann_dir\s*/\s*"best"|ann/best|validation-best ANN|ANN best checkpoint' \
  snn2 scripts configs tests 代码结构总结.md 实验执行总结.md
```

当前项目代码和当前态 Markdown 中不应再要求 `ann/best`。

---

# 9. 问题三：TL;DR evaluation 与 verification 的路径不一致

当前 `evaluate_tldr.py` 实际保存：

```text
<model_output_dir>/
└── evaluation/
    └── tldr/
        └── test_samples_<N>[_full]/
            ├── predictions.jsonl
            ├── selection.json
            └── metrics.json
```

规则：

```text
tldr_test_samples = null
→ test_samples_<total>_full/

tldr_test_samples = N 且 N < total
→ test_samples_<N>/

tldr_test_samples >= total
→ test_samples_<total>_full/
```

但当前 `verify_artifacts.py` 仍检查：

```text
evaluation/tldr/metrics.json
```

必须修复。

---

# 10. 抽出共享 TL;DR evaluation dirname helper

建议在：

```text
snn2/evaluation.py
```

增加纯函数，例如：

```python
def resolve_tldr_evaluation_layout(
    total_test_samples: int,
    configured_test_samples: int | None,
) -> dict[str, object]:

    if total_test_samples <= 0:
        raise ValueError(
            "total_test_samples must be positive"
        )

    if configured_test_samples is None:
        selected = total_test_samples
        is_full = True
    else:
        requested = int(configured_test_samples)

        if requested <= 0:
            raise ValueError(
                "evaluation.tldr_test_samples must be "
                "a positive integer or null"
            )

        if requested >= total_test_samples:
            selected = total_test_samples
            is_full = True
        else:
            selected = requested
            is_full = False

    dirname = (
        f"test_samples_{selected}_full"
        if is_full
        else f"test_samples_{selected}"
    )

    return {
        "selected_test_samples": selected,
        "is_full_test": is_full,
        "dirname": dirname,
    }
```

函数名可调整，但目录规则只能有一个 authoritative implementation。

---

# 11. `scripts/evaluate_tldr.py` 使用共享 helper

原随机抽样逻辑继续保留，但：

```text
selected_test_samples
is_full_test
test_samples_dirname
```

必须由共享 helper 确定。

subset 时仍：

```python
rng = random.Random(tldr_test_seed)
selected_indices = rng.sample(...)
selected_indices.sort()
```

full 时：

```python
selected_indices = list(range(total_test_samples))
```

---

# 12. `verify_artifacts.py` 使用同一个 helper

TL;DR verification：

1. 读取：
   ```text
   layout.data_dir / "evaluation_manifest.json"
   ```
2. 得到：
   ```python
   total_test_samples = len(
       evaluation_manifest["indices"]
   )
   ```
3. 读取：
   ```python
   cfg["evaluation"].get("tldr_test_samples")
   ```
4. 调用共享 helper；
5. 构造准确路径。

ANN：

```text
ann/evaluation/tldr/test_samples_<N>[_full]/metrics.json
ann/evaluation/tldr/test_samples_<N>[_full]/selection.json
```

SNN：

```text
snn/<neuron>/evaluation/tldr/test_samples_<N>[_full]/metrics.json
snn/<neuron>/evaluation/tldr/test_samples_<N>[_full]/selection.json
```

Tulu / lm-eval 保持：

```text
evaluation/lm_harness/results.json
```

不要给 lm-harness 增加 TL;DR 风格的 sample 子目录。

---

# 13. Verification 进一步校验 `selection.json`

TL;DR 应至少验证：

```python
len(selection["indices"])
    == expected_selected_samples
```

full split：

```text
sampling == "full_split"
```

subset：

```text
sampling == "seeded_random_without_replacement"
seed == cfg["evaluation"]["tldr_test_seed"]
```

这样 config 改为 128 时不会误验证旧 full result，反之亦然。

---

# 14. 增加 TL;DR path 测试

新增例如：

```text
tests/test_evaluation_paths.py
```

测试：

```text
total=6553, configured=None
→ selected=6553
→ full=True
→ test_samples_6553_full

total=6553, configured=128
→ selected=128
→ full=False
→ test_samples_128

total=6553, configured=9999
→ selected=6553
→ full=True
→ test_samples_6553_full

configured=0 或 -1
→ ValueError
```

还应有一个 tmp-path verification 测试，确保 verify 不再检查旧的：

```text
evaluation/tldr/metrics.json
```

---

# 15. 问题四：补齐 10-site 回归测试

保留现有：

```text
tests/test_sites.py
tests/test_calibration_topology.py
```

它们已经覆盖：

- `SITE_COUNT == 10`
- Site 9/10 名称和 coordinate
- MLP `[8,9,10]` apply 顺序
- identity parity
- Site 8/9 symmetric product saliency
- legacy Site 9 topology rejection
- exact 10-site calibration topology

另外补三类测试。

---

## 15.1 R4 顺序测试

monkeypatch：

```python
snn2.model_integration.random_hadamard
```

记录事件。

预期顺序：

```text
Site 8
Site 9
R4
Site 10
```

明确断言：

```text
Site 9 < R4 < Site 10
```

避免以后误把 Site 9 放到 R4 后。

---

## 15.2 Site 10 `down_proj` saliency 测试

构造最小 fake layer / MLP 或 monkeypatch `get_model_parts()`。

执行 integrated MLP forward 后，确认：

```text
saliency Site 10 存在
```

且来自：

```python
_linear_score(
    down_proj_input,
    down_proj_output,
    down_proj.weight,
)
```

不能让 Site 10 saliency 回到 Site 9。

---

## 15.3 deploy 模式下 Site 9 replacement 测试

至少证明：

```python
controller.apply(layer_index, 9, up)
```

在：

```text
deploy_phase
deploy_gif
deploy_mtn
```

语义下不会绕过。

可以用 fake controller：

```text
mode="deploy_phase"
```

让 Site 9 返回可识别修改后的 tensor，断言最终 MLP output 相应变化。

更完整可 monkeypatch `SiteController._load()` 返回 fake temporal neuron。

---

# 16. 更新 `实验执行总结.md`

必须删除/修正旧描述：

```text
每50个 optimizer step 验证并保存 validation-loss 最优 checkpoint
```

改成与代码一致：

```text
当前主实验不启用训练过程中的周期性 validation checkpoint selection：
eval_strategy="no"、save_strategy="no"、
load_best_model_at_end=false。

每个 ANN run 训练完成后将最终 fine-tuned checkpoint 保存到
ann/final/，后续 ANN evaluation、SNN conversion 和
artifact verification 均使用 ann/final/。
```

同时 Qwen3-1.7B 示例中的：

```text
batch_size=32
```

必须与新 materialized config 一致。

---

# 17. 更新 `代码结构总结.md`

当前 run-specific artifact 实际路径包含 learning rate：

```text
<model>/<ann_mode>/lr<learning_rate>/seed42/
```

如果文档仍写：

```text
<ann_mode>/seed42/
```

改成：

```text
<ann_mode>/lr<learning_rate>/seed42/
```

建议目录树写成：

```text
artifacts/<experiment>/<task>/
├── _shared/seed42/data/
└── <model>/
    ├── _shared/seed42/
    │   ├── rotated_prefix/rotation/
    │   ├── rotated_prefix/prefix/
    │   ├── rotated_prefix/calibration/
    │   └── vanilla_original/calibration/
    └── <ann_mode>/lr<learning_rate>/seed42/
        ├── config/
        ├── ann/
        │   └── final/
        ├── snn/{phase,gif,mtn}/
        └── logs/
```

明确：

```text
ann/final = canonical fine-tuned ANN checkpoint
```

---

# 18. 处理历史迁移方案 Markdown

当前仓库根目录的：

```text
SNN_10_site_activation_replacement_实施方案.md
```

属于已经完成的历史迁移说明，包含大量“旧 9-site → 新 10-site”内容，会干扰后续搜索并可能被误认为当前状态。

推荐：

```bash
git rm SNN_10_site_activation_replacement_实施方案.md
```

当前 authoritative 文档保持：

```text
代码结构总结.md
实验执行总结.md
```

如果必须保留历史记录，则移动到显式：

```text
docs/history/
```

并在第一屏标注：

```text
HISTORICAL MIGRATION PLAN — COMPLETED
当前代码已经是 10-site，本文件不描述当前实现。
```

本次这个“后续问题修复方案 Markdown”只是 Codex 输入，不要自动 commit 到仓库，除非用户另行要求。

---

# 19. 全仓库扫描

### 当前态旧 9-site 描述

```bash
rg -n \
  '九个|nine sites|9 sites|9 replacement sites|Expected nine' \
  --glob '*.py' \
  --glob '*.yaml' \
  --glob '*.md' \
  .
```

历史测试中专门检测 legacy 目录：

```text
site_09_post_mlp_product_r4
```

属于合法命中；其它当前态描述应清理。

### 旧 ANN best 语义

```bash
rg -n \
  'ann_dir\s*/\s*"best"|ann/best|validation-best ANN|ANN best checkpoint|best checkpoint' \
  snn2 scripts configs tests 代码结构总结.md 实验执行总结.md
```

当前态不应再依赖 `ann/best`。

### TL;DR 路径

```bash
rg -n \
  'evaluation.*tldr|tldr.*metrics\.json|metrics\.json' \
  scripts snn2 tests
```

确认 evaluation 与 verification 使用同一个 shared helper。

---

# 20. 不允许修改的内容

本任务禁止：

- 把 10-site 回退成 9-site；
- 修改 Site 9 位置；
- 修改 Site 9 coordinate；
- 修改 Site 10 coordinate；
- 新增 R5；
- 修改 R1/R2/R3/R4 数学；
- 修改 Prefix KV cache 为 input_ids prepend；
- 修改 Prefix discovery；
- 修改 Softmax Site 5 variable-length calibration；
- 自动删除用户已有 `artifacts/`；
- 为了让 conversion 通过而伪造 `ann/best`；
- 只手改 generated YAML 而不修 matrix/materializer；
- 让 verify 用模糊 glob 随便选一个旧 TL;DR result。

---

# 21. 最终验证

先运行：

```bash
python -m compileall -q snn2 scripts tests
```

再：

```bash
python scripts/materialize_configs.py
```

配置断言：

```bash
python - <<'PY'
from pathlib import Path
import yaml
from snn2.sites import SITE_COUNT

assert SITE_COUNT == 10

root = Path("configs/generated")

expected = {
    "exp1_qwen3_1_7b_tldr__vanilla.yaml": (1e-6, 32),
    "exp1_qwen3_1_7b_tldr__unaware.yaml": (5e-6, 32),
    "exp1_qwen3_1_7b_tldr__phase_aware.yaml": (1e-6, 32),
    "exp1_qwen3_1_7b_tldr__gif_aware.yaml": (1e-6, 32),
}

for name, (lr, bs) in expected.items():
    cfg = yaml.safe_load((root / name).read_text())
    assert float(cfg["training"]["learning_rate"]) == lr, name
    assert int(cfg["evaluation"]["batch_size"]) == bs, name
    assert int(cfg["calibration"]["expected_sites_per_layer"]) == SITE_COUNT, name

configs = sorted(root.glob("*.yaml"))
assert len(configs) == 12

for path in configs:
    cfg = yaml.safe_load(path.read_text())
    assert int(cfg["calibration"]["expected_sites_per_layer"]) == SITE_COUNT, path

print("generated configs OK")
PY
```

运行完整测试：

```bash
pytest -q
```

再做最终 diff：

```bash
git diff --stat
git diff
```

---

# 22. 最终验收标准

必须全部满足：

- [ ] 10-site topology 仍正确；
- [ ] Site 9 = `post_mlp_up_proj`，coordinate=`I`；
- [ ] Site 10 = `post_mlp_product_r4`，coordinate=`R4`；
- [ ] Site 8/9 product saliency 未回退；
- [ ] Site 10 down_proj saliency 未回退；
- [ ] Qwen3-1.7B 四种 mode 的 evaluation batch size 全为 32；
- [ ] Qwen3-1.7B unaware learning rate = `5e-6`；
- [ ] Qwen3-1.7B 其它三个 mode learning rate = `1e-6`；
- [ ] 特殊参数已进入 matrix / mode override，不再只存在于 generated YAML；
- [ ] materialization 再运行不会覆盖正式参数；
- [ ] 12 个 generated config 均为 10 sites；
- [ ] canonical ANN checkpoint 统一为 `ann/final`；
- [ ] training / ANN evaluation / conversion / verify 全部使用 `ann/final`；
- [ ] conversion 不再要求 `ann/best`；
- [ ] verification 不再要求 `ann/best`；
- [ ] TL;DR dirname 规则只有一个 shared implementation；
- [ ] verify 能正确检查 full/subset TL;DR 结果；
- [ ] TL;DR verify 检查 `selection.json`；
- [ ] lm-harness 路径未破坏；
- [ ] 已增加 R4/Site9/Site10 顺序测试；
- [ ] 已增加 Site10 saliency 测试；
- [ ] 已增加 deploy Site9 测试；
- [ ] 已增加 generated config regression 测试；
- [ ] 已增加 TL;DR path 测试；
- [ ] `实验执行总结.md` 不再描述不存在的 best-checkpoint 流程；
- [ ] `代码结构总结.md` run path 包含 `lr<learning_rate>`；
- [ ] 当前态 Markdown 统一为 10-site；
- [ ] 历史 migration plan 已删除或明确归档；
- [ ] Prefix / Rotation 行为未改变；
- [ ] `python -m compileall` 通过；
- [ ] `python scripts/materialize_configs.py` 成功；
- [ ] `pytest -q` 通过，或明确报告唯一环境阻塞；
- [ ] `git diff` 不含无关改动。

---

# 23. Codex 完成后必须汇报

最终回复至少报告：

1. 当前 HEAD；
2. 修改/新增/删除哪些文件；
3. Qwen3-1.7B 四种 mode 最终 `learning_rate` 和 `evaluation.batch_size`；
4. matrix 如何表达 model-level / mode-level override；
5. materializer 如何防止参数再次回退；
6. canonical ANN checkpoint 是否已统一为 `ann/final`；
7. conversion / verify 是否均已切换到 final；
8. TL;DR shared path helper 在哪个文件；
9. full/subset 结果目录分别是什么；
10. 新增了哪些测试；
11. `compileall` 结果；
12. `materialize_configs.py` 结果；
13. `pytest -q` 的 passed/failed 数量；
14. `rg ann/best` 是否还有当前态命中；
15. 旧 9-site 描述是否还有当前态命中；
16. 历史 migration plan 如何处理；
17. 是否发现其它会阻塞正式实验的新问题。

如果任何关键测试失败，不要声称项目已经可以正式重新跑实验。

---

## 最终目标流程

```text
experiment_matrix.yaml
       │
       ↓
materialize_configs.py
       │
       ├── Qwen3-1.7B: eval batch 32
       └── Qwen3-1.7B unaware: lr 5e-6
       │
       ↓
configs/generated/*.yaml
       │
       ↓
shared data / rotation / prefix / 10-site calibration
       │
       ↓
ANN training
       │
       ↓
ann/final
       │
       ├── ANN evaluation
       └── SNN conversion
               │
               ↓
        Phase / GIF / MTN
               │
               ↓
          SNN evaluation
               │
               ↓
       verify_artifacts.py
```

TL;DR：

```text
evaluation/tldr/test_samples_<N>[_full]/
```

Tulu/lm-eval：

```text
evaluation/lm_harness/
```

项目最终不应再存在当前态的：

```text
ann/best
9-site topology
verify 指向旧 TL;DR metrics 路径
materialization 覆盖正式实验超参数
```
