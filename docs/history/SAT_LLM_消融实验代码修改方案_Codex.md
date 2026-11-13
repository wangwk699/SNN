# SAT-LLM：Weaknesses 5 消融实验代码修改与执行文档编写任务（交给 Codex）

> **这是可独立执行的任务说明，不是已完成的代码变更报告。** Codex 必须依据本文件在服务器上的 `SNN` 仓库中实际修改代码、添加配置/测试、创建 `实验执行总结_消融.md`，并完成可执行的测试。不要假定与用户另有聊天上下文。
>
> **目标基准 commit（必须对照）：** `3801185e29f85c62e191841c03b6a757bca63365`  
> **仓库：** https://github.com/wangwk699/SNN  
> **论文：** SAT-LLM: Conversion-Aware Fine-Tuning of Large Language Models for Low-Latency Spiking Deployment（ICLR 2027 在投）  
> **审稿意见：** “The paper lacks a component-wise ablation that disentangles the contributions of Hadamard rotation and conversion-aware fine-tuning to the overall performance.”

## 0. 最重要的决策（用户已确认，不要再询问）

1. 本次将 **Hadamard Rotation + PrefixQuant Prefix** 定义为一个**联合组件 A**，将 **Conversion-aware Fine-Tuning（Phase 或 GIF）** 定义为组件 B。做 A/B 两因素、每种 aware 类型各一套对照：已有 Vanilla（A=0、B=0）、Unaware（A=1、B=0）、Phase-/GIF-aware（A=1、B=1），新增 **Phase-aware w/o Rotation & Prefix（A=0、B=Phase）** 和 **GIF-aware w/o Rotation & Prefix（A=0、B=GIF）**。
2. **新增两类实验均不进行 Hadamard Rotation，也完全不启用 Prefix**：不仅 ANN training，连 pre-finetuning、post-finetuning、final ANN evaluation 和 SNN conversion/evaluation 都不加载或生成有效的 Prefix token/KV 工件。不能把“无 Rotation”实现成“Rotation=false、Prefix=true”。不改变已有四种实验的配置、结果或历史产物。
3. 新增 6 份配置（3 个模型任务 × 2 个 aware 模式）置于 `configs/ablations/`，不能覆盖 `configs/generated/` 的 12 个现有配置，不能将新增模式塞入原 12 配置 materializer 的输出集合。
4. 新实验的 `experiment.id` **必须逐字使用**：`snn2_ablation_no_rotation_and_aware_FT_v1`（大小写与下划线不变）。现有实验仍为 `snn2_main_v1`。依靠 `ArtifactLayout` 现有的 `experiment.id` 根目录隔离，**不更改**既有四种实验的 artifact 路径规则。
5. 在仓库根目录**新建** `实验执行总结_消融.md`，参考既有 `实验执行总结.md` 的结构和命令风格，写出本次独立消融实验从数据清单到 Stage A/B、训练、ANN/SNN 评测、验证的完整执行步骤，说明哪些产物必须创建、哪些步骤绝对跳过、何时重跑。**保留原 `实验执行总结.md`，不要以新文档替换它。**
6. 本次为**代码及文档修改**任务。除必要的轻量测试/本地 smoke 外，不要擅自启动 1.7B/8B 的完整 GPU 训练或大规模评测；运行这些实验由用户按新增执行总结控制。

## 1. 对照设计与方法学边界

| 实验组 | 组件 A：Rotation + Prefix | 组件 B：Conversion-aware FT | 是否现有 |
| --- | --- | --- | --- |
| Vanilla | 否（ANN training） | 否 | 已有 |
| Unaware | 是 | 否 | 已有 |
| Phase-aware w/o Rotation & Prefix | **否** | Phase | **新增** |
| Phase-aware | 是 | Phase | 已有 |
| GIF-aware w/o Rotation & Prefix | **否** | GIF | **新增** |
| GIF-aware | 是 | GIF | 已有 |

本设计比较的是联合组件 A 与 aware FT 组件 B；**不要求也不应声称单独辨析 Rotation 与 Prefix 各自的贡献**。相同 B 条件下比较 aware w/o A 与 aware + A，可估计联合组件的贡献；在 A 已启用时比较 Unaware 与 aware，可估计 awareness 的贡献。

**论文表述注意：** 现有 `vanilla` 配置虽然 ANN training 不使用 Prefix，但在指定 commit 的 generated YAML 中 `post_finetuning.prefix_enabled`/`evaluation.prefix_enabled` 仍可能为 `true`（其 SNN evaluation 可能使用 Post-finetuning Prefix）。因此复用 Vanilla 已有 SNN 结果时，不能不加限制地声称六行在**所有推理/转换阶段**形成完全无混杂的 2×2。**不得为了消除此问题而擅自改动、重跑现有 Vanilla/Unaware；应在新执行总结和最终修改报告中清楚标注各行实际的训练与 SNN Prefix 语义。** 同时，复用已有超参数时，Unaware 与 aware 的最优学习率等可能不同，不能宣称所有超参数完全匹配。此次优先保证**每个新增 aware 实验与其对应的已有 aware 实验**除联合组件 A 外，训练、校准轨迹、超参数、数据、Clip 与 SNN source selector 均一致。

## 2. 代码基线、修改前检查与仓库约束

- 首先执行 `git rev-parse HEAD`、`git status --short`，确认服务器仓库相对基准 commit 的状态；若服务器已有用户未提交的修改，**不能覆盖或还原**。基准 commit 仅用于识别实现；若 HEAD 已前进，应对照真实当前代码并兼容修改。
- 阅读仓库 `AGENTS.md`、`实验执行总结.md`、`代码结构总结.md`、`configs/generated/` 的配置，以及 `tests/` 现有相关测试。`AGENTS.md` 的 site 布局、GIF Site 5 恒等映射、Stage A/B 分层、clipping、Final RMSNorm、provenance、SNN source selector 约束必须保留。
- `configs/generated/` 在该 commit **已经包含并追踪 12 个 YAML**；`scripts/materialize_configs.py` 固定生成 12 个配置并校验数量，**不要因为本次消融修改这 12 个主配置或改为 18 个**。
- `代码结构总结.md` 按 `AGENTS.md` 第 13 条只能保留“2. 目录结构”，每个文件后一行职责说明。新增 `configs/ablations/`、`实验执行总结_消融.md` 或改变现有文件职责时，同步更新该目录树的相应单行；不要重写成介绍性长文。

## 3. 必须创建的六个独立 YAML

将每个文件**复制相应现有 generated 完整配置**后最小化修改，不要从通用默认模板重建，更不要将 Phase/GIF 在不同模型中的已调学习率、slope、GIF 校准参数或 `*_previous_layers_snn` 设置统一化：

```text
configs/ablations/
├── exp1_qwen3_1_7b_tldr__phase_aware_no_rotation.yaml
├── exp1_qwen3_1_7b_tldr__gif_aware_no_rotation.yaml
├── exp1_qwen3_8b_tldr__phase_aware_no_rotation.yaml
├── exp1_qwen3_8b_tldr__gif_aware_no_rotation.yaml
├── exp2_llama3_8b_tulu3__phase_aware_no_rotation.yaml
└── exp2_llama3_8b_tulu3__gif_aware_no_rotation.yaml
```

**新增 YAML 对应源文件：** 同名 `__phase_aware.yaml` / `__gif_aware.yaml`，均位于 `configs/generated/`；保留 `experiment.ann_mode` 原值（`phase_aware` / `gif_aware`），**不新增 `ann_mode` 枚举**。

对每个新增文件修改或明确设置以下字段（将它们全部作为显式的 YAML 布尔值）：

```yaml
experiment:
  id: snn2_ablation_no_rotation_and_aware_FT_v1
  # ann_mode: phase_aware 或 gif_aware；其余 experiment 字段沿用对应原配置

rotation:
  enabled: false
  fused_weights_are_finetuned: false
  # seed / implementation 等仅作为关闭后的静态占位参数，不能触发融合

prefix:
  enabled: false
  # 其他阈值字段可保留，但关闭状态下不得调用前缀发现

ann_training:
  prefix_enabled: false

rotated_pre_finetuning:
  prefix_enabled: false

post_finetuning:
  prefix_enabled: false
  rediscover_prefix: false
  # 其他 Post Stage A 相关字段继承原配置

evaluation:
  prefix_enabled: false

# conversion.use_post_finetuning_artifacts：必须沿用该原配置的值，不得统一改 true/false。
# replacement.common_clip_enabled：必须沿用该原配置的值。
# 训练学习率/seed/slope、校准 group_size/num_samples/trajectory、Phase/GIF/MTN、评价设置均沿用原文件。
```

以上是**需修改字段摘录，不是完整配置**。输出的六份 YAML 必须包含各自原始配置的全部其他字段，且经 `load_config()` → `resolve_config()` → `validate_config()` 正常解析。新 YAML 不依赖 `materialize_configs.py` 再生成；按本任务以 Git 追踪。

## 4. 源码修改清单（先做这些，再跑任何实验）

### 4.1 `snn2/config.py`：解除 aware 与 Rotation/Prefix 强制开启的绑定，同时增加“新消融联合关闭”约束

**现状：** `resolve_config()` 根据 `phase_aware` / `gif_aware` 将 `cfg['rotation']['enabled'] = True`，使 YAML `false` 无效；`validate_config()` 对所有非 Vanilla 强制 `ann_training.prefix_enabled = true` 且 `fused_weights_are_finetuned = true`；`requires_pre_finetuning_prefix()` 仅按 `ann_mode != vanilla` 判断。

**必须实现：**

1. `vanilla` 仍强制 Rotation=false、ANN-training Prefix=false、无 aware replacement；`unaware` 仍是 Rotation=true、ANN-training Prefix=true、无 aware replacement；**原有 12 配置的 resolved 配置和行为不变**。
2. 对 `phase_aware` / `gif_aware` 不再无条件覆盖 `rotation.enabled` 为 true，而应保留 YAML 显式布尔值；`replacement.train_mode` 仍分别为 `phase` / `gif`。
3. 为 aware 模式允许且只允许两种本次定义的组件 A 状态：
   - `rotation.enabled=true` 且 `ann_training.prefix_enabled=true`（原路径）；
   - `rotation.enabled=false` 且 `ann_training.prefix_enabled=false`（消融路径）。
   - 一真一假必须以清晰 `ValueError` 拒绝，避免误把 Rotation 或 Prefix 单独消融。保持 `prefix.enabled` 兼容字段与训练阶段开关一致，不要在解析中**悄悄覆盖**明显矛盾的用户输入。
4. 新增 aware + A=false 还须验证其 `rotated_pre_finetuning.prefix_enabled=false`、`post_finetuning.prefix_enabled=false`、`evaluation.prefix_enabled=false`，防止 Final ANN/SNN 意外使用 Post Prefix；`post_finetuning.rediscover_prefix=false` 应当保持真实行为。该强限制只针对 aware + A=false，**不能全局用于 vanilla**（因为现有 vanilla 的 Post Prefix 配置不同）。
5. `rotation.fused_weights_are_finetuned` 仅在真正启用 Rotation 时要求为 true；新 aware+A=false 必须允许并建议明确要求 false；不能破坏现有 vanilla 配置中的历史占位值。显式检查 `rotation.enabled` 等是布尔值。
6. `requires_pre_finetuning_prefix(cfg)` 改为根据**是否实际启用 ANN-training Prefix**判断（或等效策略），以便无 Prefix aware 不执行 pre Prefix discovery、不在 artifact verifier 中错误要求 pre Prefix 文件；原有三种非 Vanilla 的 Prefix 行为不变。
7. 保留 `is_aware_ann_mode()`、`requires_ann_training_calibration()` 与 SNN 模式枚举：**无 Rotation/Prefix 的 aware 仍必须 Stage A/B 校准与 Phase/GIF replacement**。

### 4.2 `snn2/modeling.py` + `snn2/training.py` + `snn2/model_integration.py`：利用已有无 Rotation 路径，只修复确实不兼容的地方

已知现有能力：

- `model_source_for_stage(..., stage='ann_training')` 在 `rotation.enabled=false` 时返回 `experiment.model_name`（原始 Base），而非 `rotation/fused_base`；
- `rotation_state(...)` 在关闭 Rotation 时返回 `None`；
- `install_model_integration(model, controller, None)` 已支持无 R3/R4 的 Attention/MLP 集成；
- `train_full_parameters()` 中条件 `rotation.enabled or mode != 'none'` 可确保 Phase/GIF-aware 即使无 Rotation 仍安装替换逻辑；
- `prefix_ids_for_stage()` / `prefix_key_values_for_stage()` 对已关闭的 Prefix 阶段返回 `[]` / `None`。

**不要**重写 Hadamard、神经元、10 sites、surrogate、全参数 Trainer 或 Prefix KV 模块。检查新配置的实际初始化链：原始权重 + **无 R1/R2/R3/R4 变换** + 保留 `SiteController(mode='phase'|'gif')`、Common Clip 和 Final RMSNorm 逻辑 + 无 Prefix KV。必要时修复仅属于这条分支的问题。检查保存的 `snn2_ann_mode`、`snn2_fused_weights_are_trainable` 与 Rotation metadata 值真实反映关闭状态。

### 4.3 `scripts/evaluate_tldr.py`、`scripts/evaluate_lm_harness.py`：无 Rotation 的 aware Final ANN 必须安装 SNN2 integration

**现状存在明确 bug 风险：** 两个评测入口使用类似条件：

```python
if args.neuron != 'ann' or cfg['rotation']['enabled']:
    install_model_integration(model, controller, rotation_state(cfg, layout))
```

因此 `--neuron ann` + `phase_aware/gif_aware` + `rotation.enabled=false` 时会跳过训练时的 site-local Phase/GIF replacement，使评测图与训练图不同。

**必须改为：** SNN evaluation 一律安装；Final ANN 非 `--base`、非 `--rotated-pre-finetuning` 且属于 aware 模式时**不论 Rotation 开关**均安装；原始 Base 与 vanilla Final ANN 仍维持旧 identity 语义。推荐抽成可测试的谓词，避免两个脚本逻辑漂移。保留 `controller` 的 ANN `phase`/`gif` 与 SNN `deploy_*` 的区分。Final ANN 的 `evaluation.prefix_enabled=false` 保证不读 Prefix。

### 4.4 `snn2/calibration.py`：Stage A provenance 不可硬编码为 rotated_fused_base

**现状：** `calibration_provenance()` 对 `stage='ann_training'` 固定写入 `source_model_stage='rotated_fused_base'`。新消融实际用原始预训练 Base 收集激活。

**必须改为：**

- ANN-training Stage A 在 Rotation=true 时继续写 **`rotated_fused_base`**（保证已有工件与旧检验行为一致）；Rotation=false 时写已有语义 **`original_pretrained_base`**；
- `vanilla_analysis` 仍是 `original_pretrained_base`，Post-finetuning Stage A 仍是 `final_ann_checkpoint`；
- `rotation_enabled=false`、rotation state 路径/hash=null、`prefix_enabled=false`、`prefix_token_ids=[]`、Prefix state/KV 路径和 hash=null；使用当前 schema 可表达空 Prefix，不得伪造 Prefix 文件或弱化其他 provenance；
- Stage A 的 site statistics、Phase/GIF/MTN states、global Final RMSNorm state，和 Stage B Clip profile 的规范不变。

### 4.5 `snn2/conversion.py`：允许 aware + 无 Prefix + 复用无 Rotation Stage A

**现状：** `_source_bundle()` 在 `conversion.use_post_finetuning_artifacts=false` 时固定期望 `source_model_stage='rotated_fused_base'`；`_validate_aware_training_provenance()` 总是期待 `ann_training_prefix_*` 一组元数据，而当前无 Prefix training 的 `capture_training_artifact_provenance()` 不会写这些字段。

**必须改为：**

1. Stage A 复用（selector=false）时按 Rotation 状态验证来源：true→`rotated_fused_base`；false→`original_pretrained_base`；同时保留 Stage A/Stage B manifest hash、grouping、trajectory、site states 等严格校验。
2. Prefix enabled 时沿用原本对 `ann_training_prefix_*` 根目录、SHA256、token ids、manifest 的严格冻结校验；Prefix disabled 时将“没有 Prefix”视为**合法且明确的协议状态**，验证训练结果与当前所选 SNN Prefix 都没有有效 Prefix，允许字段缺席或按统一 schema 为 `None`/`[]`，**不要**要求虚构 prefix_state.json / KV；禁止在此分支“为通过校验”全局跳过其他来源检查。
3. `validate_conversion_prefix()` 已在 `conversion_prefix_enabled=false` 时返回 `prefix_root=None`、`prefix_token_ids=[]`、相关 hash=null；沿用这种语义，descriptor 中 `rotation_enabled=false`、`prefix_enabled=false`、no Prefix hashes/ids，`snn_clip_applied=false`。
4. `conversion.use_post_finetuning_artifacts=true` 时仍须能使用该 Final ANN 的 **no-Prefix Post Stage A**，不要因没有 Post Prefix 而拒绝；但这不是本次默认流程，默认仍继承原 generated 配置的 selector 值。

### 4.6 `scripts/discover_prefix.py` / `scripts/calibrate_sites.py` / `scripts/verify_artifacts.py`：检查空 Prefix 分支，不允许偷偷生成或依赖 Prefix

- 新实验**不执行** `scripts/prepare_rotation.py`、`scripts/discover_prefix.py --stage pre_finetuning`、`scripts/discover_prefix.py --stage post_finetuning` 或 `--rotated-pre-finetuning` 诊断；如果脚本在关闭 Prefix 时被误调用，建议主动给出清晰的拒绝提示（或干净 no-op，**不得输出非空 Prefix 工件**），且不能影响现有启用 Prefix 的流程。
- `scripts/calibrate_sites.py --stage ann_training --calibration-phase A` 必须从原始 Base 获取数据、调用 `SiteController(mode='collect')`，集成 10 sites 统计；`prefix_key_values_for_stage()` 应为 `None`；无 Rotation 不能使 Stage A 被误认为 vanilla analysis。
- Stage B **照常执行**：即使 `replacement.common_clip_enabled=false`，现有 aware 训练仍需要校验 Stage B profile 的存在和 hash。本次不重构这个协议。
- `verify_artifacts.py` 必须支持无 Rotation/无 Prefix 的 aware 模式：不要求 fused Base / rotation state / Rotation regression / Prefix state/KV，但要求数据清单、ANN-training Stage A/B、ANN checkpoint、对应评估与 SNN conversion 的全部应有产物；检查被禁用状态的 metadata 为 `false`、`[]`、`None`。
- `ArtifactLayout.ensure()` 可能会创建一些**空的** Prefix/rotation 子目录：这不代表 Prefix 或 Rotation 被实际启用。验收重点是无 `prefix_state.json`、`prefixed_key_values.pt`、`rotation_state.pt`、`fused_base` 有效产物、且模型运行时无 Prefix/Rotation；不要为避免空目录而大幅修改现有路径基础设施。

### 4.7 `snn2/artifacts.py`：优先零修改，完全靠 experiment.id 隔离

`ArtifactLayout` 已使用：

```python
experiment_root = Path(exp['output_root']) / exp['id']
```

因此所有 run、`_shared` data/policy、ANN-training Stage A/B、checkpoint、Post Stage A 和 SNN conversion/evaluation 都自然隔离到新根目录。**优先不修改 `artifacts.py` 的路径结构和命名函数**：保留 `prefix_enabled_ture`（现有历史拼写）与 `prefix_enabled_false`、`calibration_group_size_*`、`use_post_finetuning_artifacts_*` 等旧目录协议。只在检测到实际冲突时做向后兼容的小修复，不能把新 Rotation flag 插入所有旧实验的根路径，导致旧工件失联。

### 4.8 测试与文档

- 新增/扩展 `tests/test_generated_configs.py` 或单独 `tests/test_ablation_no_rotation.py`，并补充有必要的 evaluation/calibration/conversion/verify 测试。保留原有 12 配置测试的预期。
- 创建 `实验执行总结_消融.md`；按 `AGENTS.md` 更新 `代码结构总结.md` 的目录树相关条目，其他文档可小幅补充但**不改动原 `实验执行总结.md` 的既有实验命令和定义**。

## 5. 新增产物与旧产物的绝对隔离

新实验所有持久化路径必须以：

```text
artifacts/snn2_ablation_no_rotation_and_aware_FT_v1/
```

为顶层（此后还有 `<task>/<model>/...`）。旧的：

```text
artifacts/snn2_main_v1/
```

**所有旧产物均保留原路径、格式和内容**，包括已有四种 ANN checkpoint、shared Rotation/Pre Prefix、Stage A/B、Post-finetuning 工件与 SNN conversion/evaluation。**旧的四种实验不需要重新校准、重新训练或重新评测。** 新 `experiment.id` 下不得从旧 `_shared/rotated_prefix` 或旧 `_shared/vanilla_original` 目录“借用”依赖 Rotation/Prefix 的工件。

示意（省略 run identity、calibration trajectory、group_size、LR、seed 等路径中间层）：

```text
artifacts/
├── snn2_main_v1/                           # 现有 4 种模式，不变
└── snn2_ablation_no_rotation_and_aware_FT_v1/
    ├── tldr/
    │   ├── Qwen_Qwen3-1.7B-Base/
    │   └── Qwen_Qwen3-8B-Base/
    └── tulu3/
        └── meta-llama_Meta-Llama-3-8B/
            ├── _shared/.../vanilla_original/.../ann_training_calibration/
            │   ├── .../sites/               # Stage A，所有所需 states/manifest
            │   └── .../clip_profiles/.../   # Stage B
            ├── phase_aware/.../ann/final/     # 仅对应本模型的 phase-aware 消融
            ├── phase_aware/.../snn/.../      # phase/gif/mtn SNN 转换与评测
            ├── gif_aware/.../ann/final/
            └── gif_aware/.../snn/.../
```

**共享策略：** 仅当 model、数据选择 identity、group_size、GIF calibration 策略、以及 `calibration.*_previous_layers_snn` 等形成的完整 Stage A 轨迹/路径签名完全一致时，新 Phase 和 GIF 才能共享新实验下的 Stage A/B（Stage B 还须 `phase.T/mtn.T` 等匹配）。**不要假定所有 Phase/GIF 可以共享**：指定 commit 的部分 generated configs 在 `phase_previous_layers_snn`、`gif_previous_layers_snn` 上不同。若不同，各自校准。任何新校准都不得复用旧 Rotation=true 的统计值/state。

## 6. `实验执行总结_消融.md` 必须写入的内容与完整命令

应参考原 `实验执行总结.md` 的“第一部分：实验执行命令总结 / 第二部分：附录”格式，**新文档独立可运行，不依赖用户读旧文档才能得知命令**。至少包含下列 Step、变量、具体文件清单、产物矩阵、失败排查与重跑边界。

### Step 1：配置确认（本次不运行 materializer 覆盖消融 YAML）

```bash
cd /home/wangwenkang/SNN   # 若服务器仓库实际路径不同，以实际路径为准

CFG_17_P=configs/ablations/exp1_qwen3_1_7b_tldr__phase_aware_no_rotation.yaml
CFG_17_G=configs/ablations/exp1_qwen3_1_7b_tldr__gif_aware_no_rotation.yaml
CFG_8_P=configs/ablations/exp1_qwen3_8b_tldr__phase_aware_no_rotation.yaml
CFG_8_G=configs/ablations/exp1_qwen3_8b_tldr__gif_aware_no_rotation.yaml
CFG_L_P=configs/ablations/exp2_llama3_8b_tulu3__phase_aware_no_rotation.yaml
CFG_L_G=configs/ablations/exp2_llama3_8b_tulu3__gif_aware_no_rotation.yaml
TLDR_ABLATIONS=("$CFG_17_P" "$CFG_17_G" "$CFG_8_P" "$CFG_8_G")
TULU_ABLATIONS=("$CFG_L_P" "$CFG_L_G")
ALL_ABLATIONS=("${TLDR_ABLATIONS[@]}" "${TULU_ABLATIONS[@]}")
```

新文档提供一个 Python/YAML 预检命令，逐一加载完整 resolved config 并**assert**：`id` 精确匹配、6 个必要 Prefix/Rotation 开关均 false、aware `train_mode` 为 phase/gif、source selector/超参数与对应原 generated config 相同（明确允许 ID/开关/`fused_weights_are_finetuned`/`rediscover_prefix` 的差别）。列出三个 model-task pair 和六份配置的文件名。

### Step 2：准备新的数据 manifests

```bash
python scripts/prepare_data.py --config "$CFG_17_P"
python scripts/prepare_data.py --config "$CFG_8_P"
python scripts/prepare_data.py --config "$CFG_L_P"
```

数据选择目录位于**新的 experiment.id** 下；Phase/GIF 使用相同模型/seed/训练样本设置时共用相同数据选择 manifest。不可仅借用旧 id 的 manifest 或重新采样来代替此阶段。

### Step 3：Rotation 与 Prefix —— **全部跳过**

**不要执行** `prepare_rotation.py`、Pre/Post `discover_prefix.py`、rotated-pre-finetuning evaluation。无需 Rotation regression/fused checkpoint/fixed Prefix token/KV；其缺失是预期结果，不是需要“补齐”的错误。Stage A/B 的输入应使用原始模型和无 Prefix 运行时。

### Step 4：ANN-training Stage A 与 Stage B（必须生成）

```bash
for CFG in "${ALL_ABLATIONS[@]}"; do
  python scripts/calibrate_sites.py --config "$CFG" --stage ann_training --calibration-phase A
  python scripts/calibrate_sites.py --config "$CFG" --stage ann_training --calibration-phase B
done
```

如果两份新配置确实映射同一 Stage A/B 输出路径，按签名只执行一次并复用，避免两次无意义写入；新文档应解释这一点。**不论 common Clip 运行期开关为真或假，当前训练验证仍需要 Stage B。** Stage A 生成 statistics、Phase/GIF/MTN state、global Final RMSNorm 与 manifest，不能含 Clip；Stage B 仅为 eligible sites 生成 Clip profiles，Site 5 例外。

### Step 5：训练 6 个新 ANN checkpoint

```bash
export CUDA_VISIBLE_DEVICES=0,1  # 服务器实际可用 GPU 由使用者修改
NGPU=2
for CFG in "${ALL_ABLATIONS[@]}"; do
  torchrun --standalone --nproc_per_node="$NGPU" \
    scripts/train_ann.py --config "$CFG"
done
```

每份配置沿用原始 generated 的 learning rate、epoch、batch、slope、gradient accumulation 和对应 DeepSpeed JSON；仅组件 A 关闭。全部是 full-parameter fine-tuning，无 Rotation/fused weights，但**有 Phase/GIF-aware replacement 和既定 Clip 策略**。

### Step 6：Final ANN evaluation（必须）

```bash
export CUDA_VISIBLE_DEVICES=0
for CFG in "${TLDR_ABLATIONS[@]}"; do
  accelerate launch --num_processes 1 scripts/evaluate_tldr.py --config "$CFG" --neuron ann
done
for CFG in "${TULU_ABLATIONS[@]}"; do
  accelerate launch --num_processes 1 scripts/evaluate_lm_harness.py --config "$CFG" --neuron ann
done
```

`--neuron ann` 必须反映相应 Phase/GIF-aware static replacement 和当前 Stage A/B，不能误成 identity，且不得加载 Prefix。Tulu-3 的 lm-eval 还可以沿用原执行总结的多卡文档并行命令，但不能更换评测任务设置。

### Step 7：Post-finetuning artifacts —— 按 selector 决定

- **无论 selector 值如何，本次都不创建 Post-finetuning Prefix**（`post_finetuning.prefix_enabled=false` / `rediscover_prefix=false`）。
- 如 `conversion.use_post_finetuning_artifacts=false`（从原 generated 配置继承的默认值），SNN 使用本次 **ANN-training Stage A**，无需另做 Post-finetuning Stage A；Final ANN eval 同样无需 Post Stage A。
- 如某份继承的原配置 selector 为 `true`，则必须从该新 Final ANN 生成**无 Prefix** Post-finetuning Stage A：

```bash
python scripts/calibrate_sites.py --config "$CFG" --stage post_finetuning --calibration-phase A
```

Post-finetuning **禁止 Stage B**。Selector 只能选择相应 Pre/Post Stage A 来源，不能隐式重新启用 Prefix；**绝不能为节省步骤而改变原配置 selector**。

### Step 8：创建三类 SNN conversion descriptors（必须）

```bash
for CFG in "${ALL_ABLATIONS[@]}"; do
  for NEURON in phase gif mtn; do
    python scripts/convert_snn.py --config "$CFG" --neuron "$NEURON"
  done
done
```

SNN 转换读取实际 selector 选定的 Stage A，并使用 Phase/GIF/MTN full-temporal spiking；**既没有 Prefix，也不在 SNN 运行 common Clip**。校验 descriptor 的 `source_ann_checkpoint` 属于新 id、`rotation_enabled=false`、`prefix_enabled=false`、`prefix_token_ids=[]`、各 hash 为 None、校准来源与 selector 一致。

### Step 9：SNN evaluation（必须）

```bash
export CUDA_VISIBLE_DEVICES=0
for CFG in "${TLDR_ABLATIONS[@]}"; do
  for NEURON in phase gif mtn; do
    accelerate launch --num_processes 1 scripts/evaluate_tldr.py --config "$CFG" --neuron "$NEURON"
  done
done
for CFG in "${TULU_ABLATIONS[@]}"; do
  for NEURON in phase gif mtn; do
    accelerate launch --num_processes 1 scripts/evaluate_lm_harness.py --config "$CFG" --neuron "$NEURON"
  done
done
```

每份 ANN checkpoint 对应 Final ANN 指标 + 三种 SNN 指标。收集每个任务原表需要的 ROUGE、lm-eval metrics 与平均值；新指标与原对应 aware 指标进行**相同 evaluator、数据、神经元类型**的对比，汇总联合组件 A 的作用。

### Step 10：验证（必须）

```bash
for CFG in "${ALL_ABLATIONS[@]}"; do
  python scripts/verify_artifacts.py --config "$CFG"
done
```

如果因用户尚未执行完整训练或评测导致缺少工件，验证报错是正常的；Codex 的代码任务阶段应通过**mock/临时 artifact/单元测试**验证逻辑，不应假装真实 GPU 实验已经通过。新文档增加简洁的“缺什么补哪个 Step”排查表。

### 新文档必须包含的产物清单

| 产物 | 新消融实验 | 备注 |
| --- | --- | --- |
| data train/validation/calibration/evaluation manifests | 需要 | 新 `experiment.id`；同 model-task/seed 可共享 |
| Rotation state / regression / fused_base | **不需要** | 不运行 prepare_rotation |
| Pre-finetuning Prefix state / KV | **不需要** | 关闭并跳过发现 |
| ANN-training Stage A statistics/neuron states/manifest | **需要** | 无 Rotation、无 Prefix；新生成 |
| ANN-training Stage B Clip profile | **需要** | 保持既有 provenance/校验协议 |
| ANN trained final checkpoint + training_result | **需要** | 六份独立 checkpoint |
| Final ANN evaluation | **需要** | Phase/GIF static semantics，无 Prefix |
| Post-finetuning Prefix state / KV | **不需要** | 所有新实验均关闭 |
| Post-finetuning Stage A | selector=true 才需要 | 无 Prefix；不能生成 Stage B |
| Phase/GIF/MTN conversion descriptors | **需要** | 每份 checkpoint 三类 |
| Phase/GIF/MTN SNN evaluations | **需要** | 每份 checkpoint 三类 |
| verify artifact report/logs | **需要** | 校验 no Rotation/no Prefix 与 artifact lineage |

新文档还需解释：`prefix_enabled_false` 出现在目录名中是正确的；空的 Prefix 目录可由 layout 初始化创建，但**不能存在有效 Prefix state/KV**；更改 training seed、train_samples、calibration seed/group_size/trajectory、Clip profile T 等时分别从何处重跑，必须遵守原 `AGENTS.md` 与原执行总结的重跑边界。

## 7. 必须添加的自动化测试与验收标准

请尽量使用项目现有 pytest fixture 与结构，**新增代码仅为兼容无 Rotation/Prefix 的 aware 分支，不修改既有算法**。

**配置测试**

- [ ] 六份 `configs/ablations/*.yaml` 均可 `load_config` + `validate_config`；`experiment.id` 精确等于指定值；其 `ann_mode` 为 Phase/GIF，`train_mode` 正确，Rotation 和各阶段 Prefix 都是 false。
- [ ] 对应原 `configs/generated/` 12 配置的解析结果、枚举、输出数量不变；`scripts/materialize_configs.py` 仍产生 12 份，不覆盖 ablations。
- [ ] aware 模式 `rotation=true, training_prefix=false` 或反之必须报错；无 Rotation-aware 但 `post_finetuning.prefix_enabled=true` / `evaluation.prefix_enabled=true` 必须报错（防止引入 Prefix）。
- [ ] 不把 `vanilla` 原有 Post Prefix 行为当作非法配置；不破坏 Unaware 原有行为。

**路径隔离测试**

- [ ] 新的 6 个 run 及其 `_shared`、data、Stage A/B、ANN checkpoint、SNN dir 均位于 `artifacts/snn2_ablation_no_rotation_and_aware_FT_v1/` 内；旧 12 配置仍位于 `artifacts/snn2_main_v1/`。
- [ ] 原有 `phase_aware`、`gif_aware`、`unaware`、`vanilla` 的路径、suffix、artifact source selector、G 字段全部保持一致；禁用 Rotation/Prefix 不导致新/旧相互引用。
- [ ] 对 Phase/GIF Stage A 相同/不同 calibration trajectory 的目录实际共享/分叉行为符合现有 `ArtifactLayout` 签名。

**功能与元数据测试**

- [ ] aware+无 Rotation 的训练安装 SNN2 integration、保持对应 SiteController；`--neuron ann` 评测也必须安装 integration，而 vanilla ANN/原始 Base 仍保持 identity。
- [ ] `model_source_for_stage` 加载原始 Base；`rotation_state()` 为 None；不开启 R3/R4；训练/校准/ANN/SNN 都不加载 Prefix KV。
- [ ] 无 Prefix Stage A manifest 的 `prefix_enabled=false`、ids=[]、Prefix hashes=null、`rotation_enabled=false`、`source_model_stage='original_pretrained_base'`；Rotation=true 的旧 Stage A 仍为 `rotated_fused_base`。
- [ ] Stage A 只含规定的统计和 neuron state，Stage B 仍可生成/验证；缺失 Stage B 时 aware training 有清晰错误，而不是静默跳过。
- [ ] selector=false 的 aware+no Prefix SNN conversion 可通过**严格** provenance/Stage A/B 冻结校验；selector=true 的无 Prefix Post Stage A 分支同样可工作（可用 fixture/mock 测试）。
- [ ] `validate_conversion_metadata()` / `verify_artifacts.py` 能验证 no Prefix 的 `None`/空值；不能为通过测试而禁用其他 hash、source、trajectory、grouping 校验。
- [ ] negative tests：缺 Stage A、Stage B 或篡改校准 SHA，仍应失败；误混入来自旧 id/Rotation=true 的 artifact 应失败。

**工程质量**

- [ ] 运行与本次变更有关的 pytest，以及环境许可时的 `pytest -q`；逐项汇报通过/失败/未运行原因。单元测试不需要下载大模型或实际 GPU 全量训练。
- [ ] `git diff -- configs/generated/ configs/experiment_matrix.yaml` 应为空（除非事先有用户更改，绝不覆盖）；不修改历史 `artifacts/snn2_main_v1`。
- [ ] 不改 Phase/GIF neuron 数学、GIF Bit Width、Site 5 identity、RMSNorm、SNN temporal operators、Clip 禁止进入 SNN 的协议；不得全局放宽原有错误校验。
- [ ] 最终 `实验执行总结_消融.md` 有完整可复制命令、六份文件路径、所需产物矩阵、配置字段、失败处理与旧实验保留声明；`代码结构总结.md` 的目录树已同步。

## 8. 推荐实现次序及 Codex 最终交付

按以下次序实现，减少反复调试：

1. 检查基准与工作区 → 阅读 AGENTS / 源码 / 原执行文档。
2. 改 `snn2/config.py` 并增配置校验测试；添加 6 份 ablations YAML。
3. 检查原始 Base + no Prefix 的数据/校准/训练链，修复 `calibration.py` Stage A 来源。
4. 修复两个评测脚本 aware+no Rotation 时的集成安装；增加对应 forward 测试。
5. 修复 `conversion.py` 的来源校验和无 Prefix frozen provenance；审计 `verify_artifacts.py`。
6. 证明 `ArtifactLayout` 依靠新 ID 全面隔离，尽量不改路径代码；增加路径对照测试。
7. 创建 `实验执行总结_消融.md`，同步 `代码结构总结.md`，必要时小幅更新 AGENTS 相关的协议描述（不得删改其他约束）。
8. 运行单元测试与轻量 smoke，不做完整 GPU 训练；报告修改文件、原因、验收结果、尚待用户运行的 6 个实验与指标收集步骤。

**最终回答用户的格式：** 简短报告实际修改的源文件、新增配置与文档、测试命令及结果、是否影响旧路径/旧产物、是否存在未解决问题；如未运行真实训练与评测，必须明确标记“尚未执行”，不可编造实验精度。

---

**再次确认本任务不可变更的决策：** 
- 联合组件 = `Hadamard Rotation + Prefix`；
- 仅新增两种 aware w/o Rotation **且 w/o Prefix**（每个模型任务各一份，共 6 份 YAML / 6 次新增微调）；
- `experiment.id = snn2_ablation_no_rotation_and_aware_FT_v1`；
- `configs/ablations/` 与 `实验执行总结_消融.md` 必须创建；
- 原 4 种模式的 12 份配置、路径、Stage A/B、checkpoint、SNN 工件不动、不重算；
- 不需要再向用户询问设计选择，按本文完成代码修改与测试。
