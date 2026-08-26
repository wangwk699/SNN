# SNN 三项收尾修复方案：Aware SNN Clip Bundle、Group-Size 路径去重与 Calibration Config/Logs 隔离

> 目标：修复当前 `main`（基于提交 `1d87c715d493f443deb3835727a040933fdc5937`）中剩余的 3 个问题。
>
> 本文档可直接交给部署在服务器上的 Codex 实施。
>
> 本次只修改本文列出的三项，不改变已经完成的 per-head grouped calibration、Site 3/4 KV-head、Site 5 Q16/no-Clip、Phase/GIF/MTN 数学规则。

---

# 1. 修复项一：Aware ANN-training bundle 在 SNN deployment 时被错误判定为非法

## 1.1 当前问题

`phase_aware` / `gif_aware` 的 SNN conversion 按当前协议：

```text
复用 ANN-training calibration bundle
```

该 bundle 合法包含：

```text
Site 1/2/3/4/6/7/8/9/10:
    phase_state.pt
    gif_state.pt
    mtn_state.pt
    clip_state.pt

Site 5:
    phase_state.pt
    gif_state.pt
    mtn_state.pt
    no clip_state.pt
```

SNN deployment 的要求是：

```text
Clip 可以存在于 reused ANN-training bundle 中，
但 SNN controller 不得加载、实例化或执行 Clip。
```

当前 `snn2/state_validation.py::validate_site_state_bundle()` 将：

```python
require_clip=False
```

错误地解释为：

```text
目录中绝对不能存在 clip_state.pt
```

因此 `SiteController.set_deployment()`：

```python
validate_site_state_bundle(
    self.site_root,
    require_clip=False,
)
```

会拒绝合法的 aware ANN-training bundle。

---

## 1.2 必须采用三态 Clip bundle 语义

不要继续用单一 `require_clip: bool` 同时承担：

```text
需要 Clip
允许 Clip
禁止 Clip
```

建议显式改成：

```python
clip_policy: Literal[
    "require_eligible",
    "allow_eligible",
    "forbid_all",
]
```

如果不想引入 `Literal`，也可以用字符串常量。

### 三种语义

#### `require_eligible`

用于：

```text
aware ANN training
aware final ANN evaluation
```

要求：

```text
Site 1/2/3/4/6/7/8/9/10 必须存在 clip_state.pt
Site 5 必须不存在 clip_state.pt
```

#### `allow_eligible`

用于：

```text
aware SNN conversion
aware SNN deployment
aware SNN evaluation
```

要求：

```text
Site 1/2/3/4/6/7/8/9/10:
    clip_state.pt 可以存在，也可以不作为 required state 加载

Site 5:
    clip_state.pt 永远禁止存在
```

同时：

```text
SNN controller 只能加载 selected neuron state
不得加载 Clip
```

#### `forbid_all`

用于：

```text
vanilla/unaware 的 Post-finetuning conversion calibration
clip-free conversion bundle
```

要求：

```text
10 个 Site 全部不允许 clip_state.pt
```

---

## 1.3 修改 `snn2/state_validation.py`

重构：

```python
validate_site_state_bundle(...)
```

推荐接口：

```python
def validate_site_state_bundle(
    site_root,
    manifest=None,
    *,
    clip_policy: str,
    expected_num_hidden_layers=None,
):
```

或保留兼容 wrapper，但 active code 必须最终使用三态语义。

核心逻辑：

```python
clip_path = directory / "clip_state.pt"

if is_softmax_site(site_index):
    if clip_path.exists():
        raise ValueError(...)
else:
    if clip_policy == "require_eligible":
        if not clip_path.exists():
            raise FileNotFoundError(...)
    elif clip_policy == "allow_eligible":
        pass
    elif clip_policy == "forbid_all":
        if clip_path.exists():
            raise ValueError(...)
    else:
        raise ValueError(...)
```

required state：

```python
required = ["phase", "gif", "mtn"]

if (
    clip_policy == "require_eligible"
    and site_supports_clip(site_index)
):
    required.append("clip")
```

`allow_eligible` 绝不能因为 Clip 文件存在就构造 `Clipper`。

---

## 1.4 修改所有调用点

### `snn2/training.py`

aware training：

```python
validate_site_state_bundle(
    layout.ann_training_site_dir,
    clip_policy="require_eligible",
)
```

### `snn2/evaluation.py`

aware final ANN：

```python
clip_policy="require_eligible"
```

SNN deployment 前的 controller validation：

```python
clip_policy=(
    "allow_eligible"
    if conversion_reuses_ann_training_artifacts(cfg)
    else "forbid_all"
)
```

注意：`SiteController` 当前不知道 cfg 是否 aware reuse。

因此不要让 `SiteController.set_deployment()` 自己猜。

推荐修改接口：

```python
controller.set_deployment(
    neuron,
    clip_bundle_policy=...
)
```

或者在创建 controller 时注入：

```python
SiteController(
    ...,
    deployment_clip_bundle_policy=...
)
```

必须避免硬编码：

```python
require_clip=False
```

### `snn2/conversion.py`

`validate_calibration()` 已经有：

```python
allow_clip_bundle=reused
```

建议统一为新三态命名，避免两套 Clip 语义长期共存。

推荐：

```python
clip_policy = (
    "allow_eligible"
    if reused
    else "forbid_all"
)
```

然后统一调用 validator。

### `scripts/verify_artifacts.py`

aware reused calibration：

```text
clip_policy = allow_eligible
```

并继续单独验证：

```text
实际 Clip 数量 = num_layers * 9
```

post-finetuning calibration：

```text
clip_policy = forbid_all
Clip 数量 = 0
```

---

## 1.5 必须新增测试

在：

```text
tests/test_controller_state_loading.py
```

新增：

### Test A：aware bundle 含 9 个 Clip 时 deployment 必须成功

```python
_write_bundle(tmp_path, include_clip=True)

controller = SiteController(site_root=tmp_path)

controller.set_deployment(
    "phase",
    clip_bundle_policy="allow_eligible",
)
```

分别覆盖：

```text
phase
gif
mtn
```

必须成功。

并验证：

```python
controller.apply(...)
```

后：

```text
controller._modules[site_key(...)]
```

只包含 selected neuron：

```text
{"phase"}
{"gif"}
{"mtn"}
```

不得包含 `"clip"`。

### Test B：Site 5 即使 allow_eligible 也禁止 Clip

人为创建：

```text
site_05/clip_state.pt
```

必须报错。

### Test C：post-finetuning bundle forbid_all

普通 Site 人为加入：

```text
clip_state.pt
```

必须报错。

### Test D：ANN aware require_eligible

删除普通 Site 的：

```text
clip_state.pt
```

必须报错。

---

# 2. 修复项二：Aware SNN 路径中重复出现 `calibration_group_size_<G>`

## 2.1 当前问题

当前 aware `ArtifactLayout.root` 已包含：

```text
.../
calibration_group_size_<G>/
seed42/
```

但：

```python
ArtifactLayout.snn_dir()
```

又追加：

```text
snn/
calibration_group_size_<G>/
<neuron>
```

导致：

```text
.../
calibration_group_size_-1/
seed42/
snn/
calibration_group_size_-1/
phase/
```

重复。

---

## 2.2 目标路径语义

### phase_aware / gif_aware

aware ANN run root 已按 G 隔离，因此：

```text
.../
calibration_group_size_<G>/
seed42/
snn/
phase/
```

不要再追加第二次 G。

### vanilla / unaware

identity ANN checkpoint 不依赖 G，因此 run root 不包含 G。

SNN artifact 必须分 G：

```text
.../
seed42/
snn/
calibration_group_size_<G>/
phase/
```

---

## 2.3 修改 `snn2/artifacts.py`

修改：

```python
def snn_dir(self, neuron: str) -> Path:
```

推荐：

```python
def snn_dir(self, neuron: str) -> Path:
    base = self.root / "snn"

    if is_aware_ann_mode(self._cfg):
        return base / neuron

    return (
        base
        / calibration_group_dirname(
            self._cfg["calibration"]["group_size"]
        )
        / neuron
    )
```

不要修改：

```text
ann_training_calibration_dir
post_finetuning_conversion_calibration_dir
vanilla_analysis_calibration_dir
```

它们仍必须显式按 G 隔离。

---

## 2.4 检查所有 SNN 路径消费者

至少检查：

```text
snn2/conversion.py
snn2/evaluation.py
scripts/convert_snn.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/verify_artifacts.py
snn2/phase_conversion_regression.py
```

原则：

```text
所有 SNN conversion/evaluation 路径统一通过
layout.snn_dir(...)
layout.snn_conversion_dir(...)
生成。
```

不要在这些文件里手工拼：

```text
calibration_group_size_<G>
```

---

## 2.5 更新路径测试

修改：

```text
tests/test_post_finetuning_protocol.py
tests/test_evaluation_paths.py
tests/test_conversion_metadata.py
```

新增显式断言。

### aware

例如：

```python
layout = ArtifactLayout(phase_aware_cfg)
parts = layout.snn_dir("phase").parts

assert parts.count("calibration_group_size_-1") == 1
```

并断言：

```text
.../calibration_group_size_-1/seed42/snn/phase
```

### vanilla / unaware

断言：

```text
.../seed42/snn/calibration_group_size_-1/phase
```

### 不同 G

继续验证：

```text
G=-1
G=32
```

SNN 路径不同。

---

# 3. 修复项三：Calibration 的 resolved_config / logs 必须按 Group Size 隔离

## 3.1 当前问题

当前 calibration state 已按 G 正确隔离：

```text
ann_training_calibration/.../calibration_group_size_<G>/
vanilla_analysis_calibration/calibration_group_size_<G>/
post_finetuning/.../calibration_group_size_<G>/
```

但是：

```python
scripts/calibrate_sites.py
```

对 shared calibration 使用：

```python
setup(..., config_scope="policy_shared")
```

使：

```text
resolved_config.yaml
logs
```

仍落到与 G 无关的 shared policy 目录。

结果：

```text
G=-1
G=32
```

两次 calibration 会：

- 覆盖同一个 `resolved_config.yaml`
- 写入同一个 stage log/jsonl/result 文件
- provenance 目录不再一一对应具体 calibration state

---

# 3.2 新目录设计

Calibration 自己的：

```text
config/
logs/
```

必须放在对应 calibration root 下。

## ANN-training calibration

```text
.../
ann_training_calibration/
prefix_enabled_.../
calibration_group_size_<G>/
config/
resolved_config.yaml

.../
calibration_group_size_<G>/
logs/
```

state：

```text
.../
calibration_group_size_<G>/
sites/
```

## Vanilla analysis calibration

```text
.../
vanilla_analysis_calibration/
calibration_group_size_<G>/
config/
resolved_config.yaml

.../
calibration_group_size_<G>/
logs/
sites/
```

## Post-finetuning conversion calibration

```text
.../
post_finetuning/
conversion_calibration/
prefix_enabled_.../
calibration_group_size_<G>/
config/
resolved_config.yaml

.../
calibration_group_size_<G>/
logs/
sites/
```

---

# 3.3 `ArtifactLayout` 新增 calibration config/log properties

在：

```text
snn2/artifacts.py
```

新增：

```python
@property
def ann_training_calibration_config_dir(self):
    return self.ann_training_calibration_dir / "config"

@property
def ann_training_calibration_logs_dir(self):
    return self.ann_training_calibration_dir / "logs"

@property
def vanilla_analysis_calibration_config_dir(self):
    return self.vanilla_analysis_calibration_dir / "config"

@property
def vanilla_analysis_calibration_logs_dir(self):
    return self.vanilla_analysis_calibration_dir / "logs"

@property
def post_finetuning_conversion_calibration_config_dir(self):
    return self.post_finetuning_conversion_calibration_dir / "config"

@property
def post_finetuning_conversion_calibration_logs_dir(self):
    return self.post_finetuning_conversion_calibration_dir / "logs"
```

并加入 `ensure()`。

---

# 3.4 修改 `scripts/_common.py`

不要把 calibration 强行塞进现有：

```text
policy_shared
run
```

scope。

推荐增加三个显式 scope：

```text
ann_training_calibration
vanilla_analysis_calibration
post_finetuning_calibration
```

例如：

```python
elif config_scope == "ann_training_calibration":
    config_dir = layout.ann_training_calibration_config_dir

elif config_scope == "vanilla_analysis_calibration":
    config_dir = layout.vanilla_analysis_calibration_config_dir

elif config_scope == "post_finetuning_calibration":
    config_dir = layout.post_finetuning_conversion_calibration_config_dir
```

---

# 3.5 修改 `scripts/calibrate_sites.py`

根据 stage：

```python
if args.stage == "ann_training":
    config_scope = "ann_training_calibration"
    logs_dir = layout.ann_training_calibration_logs_dir

elif args.stage == "vanilla_analysis":
    config_scope = "vanilla_analysis_calibration"
    logs_dir = layout.vanilla_analysis_calibration_logs_dir

elif args.stage == "post_finetuning":
    config_scope = "post_finetuning_calibration"
    logs_dir = layout.post_finetuning_conversion_calibration_logs_dir
```

然后：

```python
cfg, layout = setup(
    args.config,
    config_scope=config_scope,
)
```

以及：

```python
StageRun(
    f"calibrate_sites_{args.stage}",
    logs_dir,
    cfg["experiment"],
)
```

不要再使用：

```python
layout.policy_logs_dir
```

保存 calibration stage 日志。

---

# 3.6 不要错误分叉其他 shared artifact

本项只针对 calibration 自身：

```text
resolved_config
logs
statistics
states
manifest
```

以下仍保持 shared：

```text
data manifests
rotation
pre-finetuning Prefix
Prefix KV
rotation logs/config
Prefix discovery logs/config
```

不要因为 G 改变而重新复制这些 artifact。

---

# 3.7 测试

建议新增到：

```text
tests/test_post_finetuning_protocol.py
```

或单独新建：

```text
tests/test_artifact_grouping_paths.py
```

验证：

### ANN-training calibration

```python
G=-1
G=32
```

必须：

```text
config dir 不同
logs dir 不同
site dir 不同
prefix dir 相同
rotation dir 相同
```

### Vanilla analysis

不同 G：

```text
calibration config/log/site 不同
base/shared data 不重复
```

### Post-finetuning

不同 G：

```text
conversion calibration config/log/site 不同
ANN checkpoint相同
post-finetuning Prefix可按既有协议保持不因 G 重算
```

如果 post-finetuning Prefix 当前 path 位于：

```text
self.post_finetuning_dir
```

而 vanilla/unaware `self.root` 不依赖 G，则它自然保持 shared，这是正确的。

---

# 4. 顺手修正错误/模糊的错误信息

当前 calibration existing-artifact check 可能在：

```text
statistics format stale
manifest format stale
```

时报：

```text
Existing calibration artifact uses a stale site topology
```

这不准确。

建议根据实际 mismatch 区分：

```text
stale site topology
stale statistics schema
stale calibration manifest schema
```

这不是核心功能变化，但本次修改 validator 时应顺手修正，便于后续实验排错。

---

# 5. 必须更新的测试集合

至少检查并更新：

```text
tests/test_controller_state_loading.py
tests/test_calibration_profiles.py
tests/test_post_finetuning_protocol.py
tests/test_conversion_metadata.py
tests/test_evaluation_paths.py
tests/test_training.py
```

建议新增覆盖：

```text
aware reused bundle + 9 Clip + deploy_phase success
aware reused bundle + 9 Clip + deploy_gif success
aware reused bundle + 9 Clip + deploy_mtn success
Site 5 Clip always rejected
post-finetuning stale Clip rejected
aware snn path contains group_size exactly once
vanilla/unaware snn path contains group_size once under snn/
calibration config/log path differs by G
Prefix/rotation shared path remains identical across G
```

---

# 6. 最低测试命令

先运行：

```bash
pytest -q \
  tests/test_controller_state_loading.py \
  tests/test_calibration_profiles.py \
  tests/test_post_finetuning_protocol.py \
  tests/test_conversion_metadata.py \
  tests/test_evaluation_paths.py \
  tests/test_training.py
```

然后：

```bash
pytest -q
```

---

# 7. 建议额外做一个实际 smoke test

使用：

```text
phase_aware
calibration.group_size=-1
```

已有 ANN-training calibration bundle 后，检查：

```bash
python scripts/convert_snn.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml \
  --neuron phase
```

必须不再因为普通 Site 存在 `clip_state.pt` 报错。

随后：

```bash
CUDA_VISIBLE_DEVICES=<GPU> accelerate launch \
  --num_processes 1 \
  scripts/evaluate_tldr.py \
  --config configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml \
  --neuron phase
```

必须能够进入实际 SNN forward。

同时确认日志/metadata 中：

```text
snn_clip_applied = false
```

并确认 runtime 没有实例化 `Clipper`。

---

# 8. 文档同步

同步修改：

```text
README.md
AGENTS.md
实验执行总结.md
代码结构总结.md
```

重点写清：

1. aware ANN-training calibration bundle 可保留 9 个 Clip state；
2. SNN deployment 允许这些文件存在，但绝不加载或执行；
3. post-finetuning conversion calibration 必须完全 clip-free；
4. Site 5 在任何 bundle 中永远 no-Clip；
5. aware SNN path 不重复 group-size；
6. calibration config/log/statistics/state 均按 group-size 隔离。

`代码结构总结.md` 仍遵守当前项目规则，只保留 `2. 目录结构`，每个文件一句说明。

---

# 9. 最终验收标准

只有以下全部满足才算完成：

- [ ] aware ANN-training bundle 中 9 个 Clip state 合法存在；
- [ ] aware SNN conversion 不因这些 Clip state 报错；
- [ ] aware SNN deployment 不加载任何 Clip；
- [ ] Site 5 Clip 仍永久禁止；
- [ ] post-finetuning conversion bundle 仍严格 clip-free；
- [ ] `require / allow / forbid` 三种 Clip bundle 语义清晰且不混用；
- [ ] aware SNN 路径中的 `calibration_group_size_<G>` 只出现一次；
- [ ] vanilla/unaware SNN 仍按 G 隔离；
- [ ] calibration resolved_config 按 G 隔离；
- [ ] calibration logs 按 G 隔离；
- [ ] calibration statistics/state/manifest 按 G 隔离；
- [ ] rotation/data/Prefix 等 G-independent artifact 不被复制；
- [ ] 新增回归测试覆盖 aware reused bundle deployment；
- [ ] `pytest -q` 全通过；
- [ ] phase_aware 实际 `convert_snn.py --neuron phase` smoke test 通过。

---

# 10. Codex 最终回复要求

完成后请报告：

1. 修改的文件；
2. Clip bundle 三态语义最终如何实现；
3. aware SNN 为什么现在可以合法复用含 Clip 的 ANN-training bundle；
4. SNN runtime 如何确保仍不加载 Clip；
5. aware/vanilla/unaware 的最终 SNN 路径示例；
6. calibration config/log 的新路径；
7. 新增/修改了哪些测试；
8. `pytest -q` 结果；
9. phase_aware conversion smoke test 结果。

不要只回复“已修改完成”。
