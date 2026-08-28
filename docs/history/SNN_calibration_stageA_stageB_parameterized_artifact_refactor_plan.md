# SNN calibration Stage A / Stage B 拆分、参数化工件路径与 Phase max_spikes 删除完整修改方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`
>
> 本文档面向部署在服务器上的 Codex。实施时不得依赖本次聊天上下文；仅依据本文档和仓库当前 `main` 分支完成修改。
>
> 本次修改的核心目标：
>
> 1. 将现有一次性 calibration 拆成：
>    - **Stage A：collect statistics**
>    - **Stage B：materialize states**
> 2. Stage A 的昂贵模型前向只做一次；之后修改 `phase.T / mtn.T / mtn.K / gif.low_ratio / gif.salient_ratio` 时，只重复 Stage B。
> 3. calibration、ANN Fine-tuning、SNN conversion/evaluation 的保存路径完整体现本轮需要扫描的关键超参数。
> 4. 必须完整支持 `replacement.common_clip_enabled: true` 和 `false` 两种情况；`true` 不是临时场景，而是正式实验路径。
> 5. 彻底删除 `phase.max_spikes` 及所有 spike-count 限制逻辑。
> 6. provenance/validation 必须按“实际使用的 state”绑定，不能继续把 ANN 或某一种 SNN 与整个 Stage-B bundle 的 hash 强耦合。

---

## 1. 当前实现背景与本次必须保留的语义

当前仓库的 calibration 一次执行会同时完成：

1. 加载对应 stage 的模型、Prefix/KV、rotation；
2. 在 calibration dataset 上进行前向；
3. 收集每层 10 个 replacement site 以及 `_global/final_rmsnorm` 的统计量；
4. 保存 `statistics.pt / statistics_summary.json`；
5. 立即从 statistics 构造：
   - `phase_state.pt`
   - `gif_state.pt`
   - `mtn_state.pt`
   - ANN-training 时的 `clip_state.pt`
   - `calibration_summary.json`
   - `calibration_state_manifest.json`

现有 `scripts/calibrate_sites.py` 因此把昂贵的 statistics collection 和轻量的 state materialization 耦合在了一起。

本次修改后必须保持以下既有实验语义：

- `ann_training` calibration：
  - 仅用于 `phase_aware / gif_aware`；
  - 使用 rotated fused base；
  - 使用 pre-finetuning / ANN-training Prefix/KV；
  - **Stage B 始终生成 eligible sites 的 `clip_state.pt`**；
  - `replacement.common_clip_enabled` 只控制 ANN forward 是否实际应用 Clip，不控制 Clip state 是否生成。
- `post_finetuning` calibration：
  - 用于 `vanilla / unaware`；
  - statistics 来源是各自 final ANN checkpoint；
  - 使用 post-finetuning Prefix/KV；
  - Stage B 不生成任何 `clip_state.pt`。
- `vanilla_analysis`：
  - 保持“只做 statistics analysis”的语义；
  - 不需要 Stage B，不生成 Phase/GIF/MTN/Clip state。
- Site 5 (`post_spiking_softmax`)：
  - GIF 继续使用 Softmax identity 特殊策略；
  - 永远不生成 `clip_state.pt`。
- 10-site topology、per-head/grouped calibration、Prefix/rotation 行为均不得因本次重构改变。

---

# 2. 最终确定的参数依赖

## 2.1 Stage A statistics 不依赖以下参数

Stage A 收集的原始 statistics 不依赖：

```yaml
phase.T
mtn.T
mtn.K
gif.low_ratio
gif.salient_ratio
```

Stage A 继续保存的底层统计包括但不限于：

- `value_min`
- `value_max`
- `abs_max`
- `sum_abs`
- `sum_sq`
- `saliency_sum`
- `saliency_row_count`
- `phase_ema_abs_max`
- `phase_ema_updates`

因此只修改上述 Stage-B 参数时，禁止重新进行模型 calibration forward；应直接复用 Stage A。

## 2.2 Stage B 参数依赖

### `phase.T`

影响：

- 每个 site 的 `phase_state.pt`
  - `T`
  - `v0`
- ANN-training 的 `clip_state.pt`
- `_global/final_rmsnorm/phase_state.pt`
- 对应 summary / manifest

`tau` 的底层 EMA statistics 不依赖 `phase.T`。

### `mtn.T`

影响：

- `mtn_state.pt`
- ANN-training 的 `clip_state.pt`
- 对应 summary / manifest

### `mtn.K`

影响：

- `mtn_state.pt`
- 对应 summary / manifest

当前 Clip 公式不使用 `mtn.K`，因此不得错误地让 `mtn.K` 改变 `clip_state.pt`。

### `gif.low_ratio`

影响普通 Site 1/2/3/4/6/7/8/9/10 的：

- `gif_state.pt`
  - `low_ratio`
  - `mask_low`
- 对应 summary / manifest

不影响 Site 5 的 GIF identity state。

当前 common Clip 使用 low/high quantization representable range，而不是 `mask_low`，因此 `gif.low_ratio` 不应改变 `clip_state.pt`。

### `gif.salient_ratio`

当前实现中它不是独立 state 计算变量，而是用于保证：

```text
gif.low_ratio + gif.salient_ratio == 1
```

但本次要求将它显式写入 Stage-B、ANN、SNN 路径和 manifest，使实验配置可直接从目录名读取。

---

# 3. calibration 根目录命名修改

当前：

```text
calibration_group_size_-1
```

统一改为：

```text
calibration_group_size_<group_size>_num_samples_<num_samples>
```

例如：

```text
calibration_group_size_-1_num_samples_128
calibration_group_size_32_num_samples_128
calibration_group_size_-1_num_samples_256
```

实现一个唯一 helper，禁止在不同文件中手写字符串拼接，例如：

```python
def calibration_scope_dirname(*, group_size, num_samples) -> str:
    ...
```

要求：

- `group_size == -1` 或正整数；
- `num_samples` 为正整数；
- 全项目所有 calibration root、aware ANN calibration suffix、non-aware SNN calibration scope 均通过同一个 helper 生成。

---

# 4. Stage A / Stage B 最终目录结构

## 4.1 ANN-training calibration

示例配置：

```yaml
calibration:
  group_size: -1
  num_samples: 128

phase:
  T: 4

mtn:
  T: 4
  K: 6

gif:
  low_ratio: 0.9
  salient_ratio: 0.1
```

目录必须为：

```text
artifacts/snn2_main_v1/tldr/Qwen_Qwen3-1.7B-Base/
└── _shared/
    └── seed42/
        └── rotated_prefix/
            └── ann_training_calibration/
                └── prefix_enabled_ture/
                    └── calibration_group_size_-1_num_samples_128/
                        ├── statistics/
                        │   ├── layer_000/
                        │   │   ├── site_01_post_input_rmsnorm/
                        │   │   │   ├── statistics.pt
                        │   │   │   └── statistics_summary.json
                        │   │   ├── ...
                        │   │   └── site_10_post_mlp_product_r4/
                        │   ├── ...
                        │   ├── _global/
                        │   │   └── final_rmsnorm/
                        │   │       ├── statistics.pt
                        │   │       └── statistics_summary.json
                        │   ├── statistics_manifest.json
                        │   ├── config/
                        │   │   └── resolved_config.yaml
                        │   └── logs/
                        │
                        └── states/
                            └── phase_T_4_mtn_T_4_mtn_K_6_gif_low_ratio_0.9_gif_salient_ratio_0.1/
                                ├── layer_000/
                                │   ├── site_01_post_input_rmsnorm/
                                │   │   ├── phase_state.pt
                                │   │   ├── gif_state.pt
                                │   │   ├── mtn_state.pt
                                │   │   ├── clip_state.pt
                                │   │   └── calibration_summary.json
                                │   ├── ...
                                │   └── site_05_post_spiking_softmax/
                                │       ├── phase_state.pt
                                │       ├── gif_state.pt
                                │       ├── mtn_state.pt
                                │       └── calibration_summary.json
                                ├── ...
                                ├── _global/
                                │   └── final_rmsnorm/
                                │       └── phase_state.pt
                                ├── calibration_state_manifest.json
                                ├── config/
                                │   └── resolved_config.yaml
                                └── logs/
```

注意：

- **Stage A 与 Stage B 不再共用 `layer_xxx/site_xx` 目录。**
- Stage-B state 目录中不再复制 `statistics.pt`。
- Stage-A statistics 目录中不允许出现 `phase_state.pt / gif_state.pt / mtn_state.pt / clip_state.pt`。
- Stage-B variant 目录必须显式包含：
  - `phase.T`
  - `mtn.T`
  - `mtn.K`
  - `gif.low_ratio`
  - `gif.salient_ratio`

Stage-B variant helper建议为：

```python
def calibration_state_variant_dirname(cfg) -> str:
    return (
        f"phase_T_{...}"
        f"_mtn_T_{...}"
        f"_mtn_K_{...}"
        f"_gif_low_ratio_{...}"
        f"_gif_salient_ratio_{...}"
    )
```

## 4.2 Post-finetuning conversion calibration

`vanilla / unaware` 使用同样的二阶段布局：

```text
.../<run-root>/post_finetuning/
└── conversion_calibration/
    └── prefix_enabled_<...>/
        └── calibration_group_size_-1_num_samples_128/
            ├── statistics/
            │   ├── layer_xxx/site_xx/statistics.pt
            │   ├── layer_xxx/site_xx/statistics_summary.json
            │   ├── _global/final_rmsnorm/statistics.pt
            │   ├── _global/final_rmsnorm/statistics_summary.json
            │   ├── statistics_manifest.json
            │   ├── config/resolved_config.yaml
            │   └── logs/
            │
            └── states/
                └── phase_T_4_mtn_T_4_mtn_K_6_gif_low_ratio_0.9_gif_salient_ratio_0.1/
                    ├── layer_xxx/site_xx/phase_state.pt
                    ├── layer_xxx/site_xx/gif_state.pt
                    ├── layer_xxx/site_xx/mtn_state.pt
                    ├── layer_xxx/site_xx/calibration_summary.json
                    ├── _global/final_rmsnorm/phase_state.pt
                    ├── calibration_state_manifest.json
                    ├── config/resolved_config.yaml
                    └── logs/
```

Post-finetuning Stage B：

- 不生成 `clip_state.pt`；
- 如果 state 目标目录中发现任何旧 `clip_state.pt`，必须删除或直接拒绝 stale artifact；
- validation 必须保证 `clip_policy="forbid_all"`。

## 4.3 Vanilla analysis

保留 statistics-only：

```text
.../vanilla_original/vanilla_analysis_calibration/
└── calibration_group_size_-1_num_samples_128/
    └── statistics/
        ├── layer_xxx/site_xx/statistics.pt
        ├── ...
        └── statistics_manifest.json
```

不创建 `states/`。

---

# 5. 新的 Stage A / Stage B CLI

新增两个明确脚本：

```text
scripts/collect_calibration_statistics.py
scripts/materialize_calibration_states.py
```

旧的 `scripts/calibrate_sites.py` 不再允许执行旧的一次性流程。

推荐保留该文件作为 compatibility guard：运行时直接报错并提示新命令，防止旧 shell script 静默重新跑完整 calibration，例如：

```text
One-shot calibration has been removed.
Run:
  python scripts/collect_calibration_statistics.py ...
then:
  python scripts/materialize_calibration_states.py ...
```

不要继续保留任何“单命令自动先 A 后 B”的入口。

---

# 6. 新命令

## 6.1 ANN-training calibration

原命令：

```bash
CFG_17_P=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml
python scripts/calibrate_sites.py \
  --config "$CFG_17_P" \
  --stage ann_training
```

改成：

### Stage A：只收集 statistics

```bash
CFG_17_P=configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml

python scripts/collect_calibration_statistics.py \
  --config "$CFG_17_P" \
  --stage ann_training
```

### Stage B：只根据已有 statistics 生成 states

```bash
python scripts/materialize_calibration_states.py \
  --config "$CFG_17_P" \
  --stage ann_training
```

如果只修改：

```yaml
phase.T
mtn.T
mtn.K
gif.low_ratio
gif.salient_ratio
```

则**不要再运行 Stage A**，只运行：

```bash
python scripts/materialize_calibration_states.py \
  --config "$CFG_17_P" \
  --stage ann_training
```

新的 Stage-B variant 自动进入新的 sibling directory。

## 6.2 Post-finetuning conversion calibration

原：

```bash
for CFG in "$CFG_17_V" "$CFG_17_U"; do
  python scripts/calibrate_sites.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

改成：

### Stage A

```bash
for CFG in "$CFG_17_V" "$CFG_17_U"; do
  python scripts/collect_calibration_statistics.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

### Stage B

```bash
for CFG in "$CFG_17_V" "$CFG_17_U"; do
  python scripts/materialize_calibration_states.py \
    --config "$CFG" \
    --stage post_finetuning
done
```

只改变 Stage-B 参数时，只执行第二段。

## 6.3 Vanilla analysis

```bash
python scripts/collect_calibration_statistics.py \
  --config "$VAN_CFG" \
  --stage vanilla_analysis
```

`materialize_calibration_states.py --stage vanilla_analysis` 必须拒绝并明确说明 vanilla analysis 是 statistics-only。

---

# 7. Stage A 实现要求

重构 `snn2/calibration.py`：

现有 `collect_site_statistics(...)` 必须变成真正的 statistics-only 函数。

Stage A 应：

1. 加载对应 stage 模型；
2. 加载 tokenizer；
3. 安装正确 rotation；
4. 安装正确 Prefix fixed KV；
5. 遍历 calibration data；
6. `StatisticsStore.reduce_and_save(statistics_root)`；
7. 保存 Stage-A manifest；
8. 结束。

Stage A **不得调用** `materialize_calibration_states()`。

删除类似：

```python
materialize_states=True
```

这种让 collect 函数决定是否继续 materialize 的旧接口，避免未来重新耦合。

---

# 8. Stage A manifest 必须增强

`statistics_manifest.json` 至少记录：

```json
{
  "purpose": "...",
  "statistics_format_version": 2,
  "statistics_manifest_format_version": 1,
  "calibration_group_size": -1,
  "calibration_num_samples": 128,
  "calibration_seed": 42,

  "source_model_stage": "...",
  "source_ann_mode": "...",
  "source_ann_checkpoint": "...",
  "source_ann_config_sha256": "...",

  "prefix_enabled": true,
  "prefix_state_path": "...",
  "prefix_state_sha256": "...",
  "prefix_kv_path": "...",
  "prefix_kv_sha256": "...",

  "rotation_enabled": true,
  "rotation_state_path": "...",
  "rotation_state_sha256": "...",

  "calibration_data_manifest_path": "...",
  "calibration_data_manifest_sha256": "...",

  "expected_num_hidden_layers": ...,
  "expected_layer_names": [...],
  "site_topology_version": ...,
  "site_count": 10,

  "sites": {
    "layer_000/site_01_...": {
      "...existing summary...": "...",
      "statistics_sha256": "..."
    }
  },

  "global_states": {
    "final_rmsnorm": {
      "...existing statistics summary...": "...",
      "statistics_sha256": "..."
    }
  }
}
```

新增 per-site/global `statistics_sha256`。

Stage B 必须依赖这些 hashes，而不是仅依赖目录存在。

建议新增独立：

```python
STATISTICS_MANIFEST_FORMAT_VERSION = 1
```

不要因为 manifest schema 改变而强行修改 `statistics.pt` 的 `STATISTICS_FORMAT_VERSION`；底层统计 tensor schema 没有因为本次重构改变。

如果旧 `statistics_manifest.json` 没有新 manifest version 或 statistics hash，Stage B 应拒绝并提示重新运行 Stage A。

---

# 9. Stage B 实现要求

把现有：

```python
materialize_calibration_states(site_root, cfg, ...)
```

改成显式双 root：

```python
materialize_calibration_states(
    statistics_root=...,
    state_root=...,
    cfg=cfg,
    metadata=...,
    include_clip=...,
    expected_num_hidden_layers=...,
)
```

核心流程：

```text
statistics_root/layer_*/site_*/statistics.pt
                 ↓
          build_site_states()
                 ↓
state_root/layer_*/site_*/
    phase_state.pt
    gif_state.pt
    mtn_state.pt
    [clip_state.pt]
    calibration_summary.json
```

必须：

1. 从 `statistics_root` 遍历 statistics；
2. 在 `state_root` 创建镜像 layer/site topology；
3. 不往 statistics_root 写任何 state；
4. 不往 state_root 复制 statistics；
5. 读取 Stage-A `statistics_manifest.json`；
6. 校验：
   - manifest schema；
   - statistics format；
   - group size；
   - num samples；
   - topology；
   - statistics SHA-256；
   - stage/purpose；
   - Prefix/rotation/source model provenance；
7. Stage B 不加载 model/tokenizer/dataset，不进行任何 GPU calibration forward；
8. Stage B 的运行成本应只来自读取 `.pt` + CPU state construction。

---

# 10. Stage-B manifest 必须记录 Stage-A 来源

`calibration_state_manifest.json` 新增并严格验证：

```json
{
  "source_statistics_root": ".../statistics",
  "source_statistics_manifest_path": ".../statistics/statistics_manifest.json",
  "source_statistics_manifest_sha256": "...",

  "calibration_group_size": -1,
  "calibration_num_samples": 128,

  "materialization_parameters": {
    "phase_T": 4,
    "mtn_T": 4,
    "mtn_K": 6,
    "gif_low_ratio": 0.9,
    "gif_salient_ratio": 0.1
  }
}
```

每个 site summary 继续保存 state hashes，例如：

```json
"state_sha256": {
  "phase": "...",
  "gif": "...",
  "mtn": "...",
  "clip": "..."
}
```

Post-finetuning 不含 `"clip"`。

global final RMSNorm Phase state 继续写入 manifest 并包含 SHA-256。

---

# 11. ArtifactLayout 重构

重点修改 `snn2/artifacts.py`。

至少新增/重构以下概念。

## 11.1 calibration scope

```python
calibration_scope_dirname(
    group_size=cfg["calibration"]["group_size"],
    num_samples=cfg["calibration"]["num_samples"],
)
```

结果：

```text
calibration_group_size_-1_num_samples_128
```

## 11.2 ANN-training

建议提供明确 property：

```text
ann_training_calibration_dir
ann_training_statistics_dir
ann_training_statistics_config_dir
ann_training_statistics_logs_dir

ann_training_states_root
ann_training_state_dir
ann_training_state_config_dir
ann_training_state_logs_dir
```

其中：

```text
ann_training_state_dir
=
ann_training_states_root
/
phase_T_..._mtn_T_..._mtn_K_..._gif_low_ratio_..._gif_salient_ratio_...
```

下游原先使用 `layout.ann_training_site_dir` 的地方可以：

- 最好改成语义明确的 `layout.ann_training_state_dir`；
- 如为了减少迁移成本保留 legacy alias，也必须只指向当前 Stage-B variant，禁止再指向 mixed statistics/state root。

## 11.3 Post-finetuning

同理新增：

```text
post_finetuning_statistics_dir
post_finetuning_state_dir
```

`layout.conversion_site_dir` 应最终解析到当前配置对应的 **Stage-B state variant**。

## 11.4 config/log scope

`scripts/_common.py::setup()` 增加明确 config scopes，例如：

```text
ann_training_statistics
ann_training_states
vanilla_analysis_statistics
post_finetuning_statistics
post_finetuning_states
```

Stage A/B 各自在自己的目录保存：

```text
config/resolved_config.yaml
logs/
```

不要让不同 Stage-B variant 共享同一个 resolved config/log 目录。

---

# 12. 浮点目录值必须统一格式化

新增唯一 helper，例如：

```python
def format_path_scalar(value) -> str:
    ...
```

要求：

- `4 -> "4"`
- `1.0 -> "1.0"`
- `0.9 -> "0.9"`
- `0.0 -> "0.0"`
- 不允许同一个数有时输出 `0.9`、有时输出 `0.900000`。
- 处理 `-0.0` 时规范成 `0.0`。
- 所有：
  - `low_ratio`
  - `salient_ratio`
  - `surrogate_slope`
  - `warmup_ratio`
  - 以及未来复用该 helper 的浮点 path token
  必须走同一实现。

---

# 13. calibration.num_samples 改为真正可配置

当前 `validate_config()` 强制：

```text
calibration.num_samples == 128
```

删除这个固定约束。

改为：

```text
calibration.num_samples 必须是正整数
```

实际无放回采样能否完成仍由 data preparation 根据 train population 检查：

```text
num_samples <= available training samples
```

保留：

```yaml
with_replacement: false
```

主实验无放回策略。

---

# 14. Calibration data manifest 也必须避免不同 num_samples 相互覆盖

当前共享：

```text
_shared/seed42/data/calibration_manifest.json
```

如果 `num_samples` 从 128 改 256，会覆盖旧 calibration selection。

必须参数化 calibration manifest。

推荐：

```text
_shared/seed42/data/
├── train_manifest.json
├── validation_manifest.json
├── evaluation_manifest.json
└── calibration/
    ├── calibration_seed_42_num_samples_128/
    │   └── calibration_manifest.json
    └── calibration_seed_42_num_samples_256/
        └── calibration_manifest.json
```

至少必须包含 `num_samples`；建议同时包含 `calibration.seed`，因为 calibration seed 同样决定样本集合。

修改：

- `ArtifactLayout`
- `prepare_manifests()`
- `load_manifests()`
- `load_selected_raw()`
- calibration provenance

确保不同 `num_samples` 的 calibration manifests 可同时存在。

当 `calibration.num_samples` 改变时，工作流应为：

```bash
python scripts/prepare_data.py --config "$CFG"

python scripts/collect_calibration_statistics.py \
  --config "$CFG" \
  --stage <...>

python scripts/materialize_calibration_states.py \
  --config "$CFG" \
  --stage <...>
```

只改 Stage-B 参数时不需要重新 prepare data 或 Stage A。

---

# 15. ANN Fine-tuning 路径最终规则

## 15.1 Vanilla / unaware

**保持原 ANN Fine-tuning 路径规则不变。**

不要因为本次新增 Phase/GIF 参数目录而改变 vanilla/unaware ANN checkpoint 路径。

## 15.2 Aware run 的 calibration scope

当前 aware ANN run 路径已有：

```text
..._calibration_group_size_-1
```

扩展为：

```text
..._calibration_group_size_-1_num_samples_128
```

这样不同 Stage-A sample count 不会落到同一个 aware ANN run root。

## 15.3 Phase-aware

最终必须是：

```text
.../
prefix_enabled_ture_common_clip_enabled_<true|false>/
phase_T_<phase.T>_mtn_T_<mtn.T>_low_ratio_<gif.low_ratio>_salient_ratio_<gif.salient_ratio>_surrogate_slope_<phase.surrogate_slope>_warmup_ratio_<training.warmup_ratio>/
seed42/
```

例：

```text
.../
prefix_enabled_ture_common_clip_enabled_true/
phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_surrogate_slope_1.0_warmup_ratio_0.0/
seed42/
```

`common_clip_enabled=false` 时也使用完全相同的参数子目录，只改变上一层：

```text
prefix_enabled_ture_common_clip_enabled_false/
```

**不要把 `mtn.K` 放入 Phase-aware ANN Fine-tuning 路径。**

虽然 Phase-aware 本身不需要 low/salient ratio，但用户已明确决定为了 Phase/GIF aware 实验管理一致性，Phase-aware 路径也必须显式包含：

```text
low_ratio
salient_ratio
```

不得省略。

## 15.4 GIF-aware

最终：

```text
.../
prefix_enabled_ture_common_clip_enabled_<true|false>/
phase_T_<phase.T>_mtn_T_<mtn.T>_low_ratio_<gif.low_ratio>_salient_ratio_<gif.salient_ratio>_warmup_ratio_<training.warmup_ratio>/
seed42/
```

例：

```text
.../
prefix_enabled_ture_common_clip_enabled_true/
phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_warmup_ratio_0.0/
seed42/
```

GIF-aware 不加入 `surrogate_slope`。

GIF-aware ANN path 同样不加入 `mtn.K`。

---

# 16. common_clip_enabled=true 必须作为正式路径完整支持

本次实现绝不能仅针对 `false`。

ANN-training Stage B 始终生成 Clip state：

```text
Site 1/2/3/4/6/7/8/9/10 -> clip_state.pt
Site 5 -> 无 clip_state.pt
```

不论：

```yaml
replacement:
  common_clip_enabled: true
```

还是：

```yaml
replacement:
  common_clip_enabled: false
```

Stage-B state 内容应相同。

`common_clip_enabled` 只决定 ANN `SiteController` 是否加载/应用 Clip。

Clip 当前依赖：

- `phase.T`
- `mtn.T`
- statistics-derived GIF low/high representable range

不依赖：

- `mtn.K`
- `gif.low_ratio`
- `gif.salient_ratio`

路径仍按照用户确定的统一规则显式记录 low/salient ratio。

必须保留/补测试证明：

```text
同一 Stage A + 同一 Stage-B 参数
common_clip_enabled true/false
=> materialized state 文件完全相同
```

---

# 17. ANN training provenance 必须改为 dependency-scoped fingerprint

这是本次重构的关键要求。

当前 aware ANN training 将整个：

```text
calibration_state_manifest.json
```

hash 作为训练 provenance。

这在 Stage-B 多 variant 后是错误的。

例如只改：

```yaml
mtn.K: 6 -> 8
```

Stage-B 目录和整个 manifest 会变化，但：

- Phase-aware ANN 不使用 MTN neuron；
- GIF-aware ANN 不使用 MTN neuron；
- common Clip 也不使用 `mtn.K`。

因此同一个 ANN checkpoint 不应仅因为整个 bundle manifest 不同而失效。

## 17.1 新增 state dependency fingerprint helper

建议在 `snn2/state_validation.py` 或独立 provenance module 实现：

```python
compute_state_fingerprint(
    state_root,
    state_kinds=(...),
    include_global_phase=False,
)
```

fingerprint 输入必须为：

```text
相对路径 + 文件 SHA-256
```

按确定性排序后计算一个 aggregate SHA-256。

同时可以保留逐文件 hash map 便于调试。

## 17.2 Phase-aware ANN

### common_clip=false

训练实际依赖：

```text
所有 layer/site 的 phase_state.pt
```

fingerprint 只包含 `phase`。

### common_clip=true

训练实际依赖：

```text
所有 layer/site 的 phase_state.pt
eligible sites 的 clip_state.pt
```

fingerprint 包含：

```text
phase + clip
```

注意：

`_global/final_rmsnorm/phase_state.pt` 只用于 temporal Phase SNN deployment，不用于静态 Phase-aware ANN training，因此不要错误加入 ANN training fingerprint。

## 17.3 GIF-aware ANN

### common_clip=false

只绑定：

```text
gif_state.pt
```

### common_clip=true

绑定：

```text
gif_state.pt + eligible clip_state.pt
```

## 17.4 training_result.json

建议记录：

```json
{
  "ann_training_state_dependency_kinds": ["phase", "clip"],
  "ann_training_state_fingerprint_sha256": "...",
  "ann_training_state_file_hashes": {
    "layer_000/site_01_.../phase_state.pt": "...",
    "...": "..."
  },

  "ann_training_state_root_at_training_time": "...",
  "ann_training_statistics_manifest_sha256": "...",
  "ann_training_calibration_group_size": -1,
  "ann_training_calibration_num_samples": 128
}
```

`state_root_at_training_time` 可用于信息展示，但**不能作为语义一致性判断的唯一条件**。

尤其是 `mtn.K` 变化会切换 Stage-B bundle root，但只要 ANN 实际依赖的 state fingerprint 未变化，就不能因为 root 字符串不同而判定 checkpoint 无效。

更新：

- `capture_training_artifact_provenance`
- `verify_training_artifact_provenance_unchanged`
- `validate_recorded_training_artifact_provenance`

---

# 18. SNN conversion provenance 也必须 neuron-scoped

这同样是必须项。

用户要求 SNN 输出路径：

```text
phase/T_<T>
mtn/T_<T>_K_<K>
gif/low_ratio_<...>_salient_ratio_<...>
```

因此 conversion identity 必须与这些 neuron 实际使用的 state 一致，而不能与整个 Stage-B bundle hash 强绑定。

## 18.1 Phase SNN

严格绑定：

```text
所有 per-site phase_state.pt
+
_global/final_rmsnorm/phase_state.pt
```

输出路径：

```text
.../snn/phase/T_<phase.T>/
```

例：

```text
.../snn/phase/T_4/
```

`mtn.K`、GIF ratio 等无关参数变化不得让已经等价的 Phase SNN 被误判为不同语义。

## 18.2 MTN SNN

只绑定：

```text
所有 per-site mtn_state.pt
```

输出：

```text
.../snn/mtn/T_<mtn.T>_K_<mtn.K>/
```

例：

```text
.../snn/mtn/T_4_K_6/
```

## 18.3 GIF SNN

只绑定：

```text
所有 per-site gif_state.pt
```

输出：

```text
.../snn/gif/low_ratio_<gif.low_ratio>_salient_ratio_<gif.salient_ratio>/
```

例：

```text
.../snn/gif/low_ratio_0.9_salient_ratio_0.1/
```

## 18.4 conversion_metadata.json

建议新增：

```json
{
  "deployment_neuron": "phase",
  "deployment_state_kinds": ["phase"],
  "deployment_state_fingerprint_sha256": "...",
  "deployment_state_file_hashes": {...},
  "global_final_norm_phase_state_sha256": "...",

  "source_statistics_manifest_sha256": "...",

  "deployment_parameters": {
    "phase_T": 4
  }
}
```

MTN：

```json
"deployment_parameters": {
  "mtn_T": 4,
  "mtn_K": 6
}
```

GIF：

```json
"deployment_parameters": {
  "gif_low_ratio": 0.9,
  "gif_salient_ratio": 0.1
}
```

可以继续记录完整 Stage-B manifest path/hash作为 informational provenance，但：

**`validate_conversion_metadata()` 不得再把整个 Stage-B manifest SHA-256 作为该 neuron conversion 的唯一语义身份。**

必须严格校验 neuron-scoped fingerprint。

---

# 19. aware ANN -> SNN conversion 的 training provenance 校验

对于 `phase_aware / gif_aware`，conversion 仍然必须保证：

```text
当前加载的 final ANN checkpoint
确实对应训练时实际使用的 ANN replacement state
```

但校验应使用第 17 节定义的 ANN-training dependency fingerprint，而不是整个 Stage-B manifest。

例如：

- Phase-aware + clip=true：
  - 比较训练记录的 `phase+clip` fingerprint；
- GIF-aware + clip=true：
  - 比较训练记录的 `gif+clip` fingerprint。

然后 SNN conversion 再独立绑定当前要部署的 neuron state fingerprint。

这样可允许：

```text
同一个 Phase-aware ANN checkpoint
+
不同 mtn.K Stage-B variant
→ 不同 MTN SNN conversion
```

因为 `mtn.K` 不影响 Phase-aware ANN training，但会影响 MTN deployment。

---

# 20. SNN ArtifactLayout 修改

当前 `snn_dir(neuron)`：

- aware 和 non-aware 路径结构不同；
- non-aware 额外包含 calibration group scope。

修改后统一在 neuron 后追加 neuron-specific variant。

## aware

```text
<run-root>/
└── snn/
    ├── phase/
    │   └── T_4/
    ├── mtn/
    │   └── T_4_K_6/
    └── gif/
        └── low_ratio_0.9_salient_ratio_0.1/
```

## vanilla/unaware

保留现有 calibration scope 层，但 scope 升级为 group + num samples：

```text
<run-root>/
└── snn/
    └── calibration_group_size_-1_num_samples_128/
        ├── phase/
        │   └── T_4/
        ├── mtn/
        │   └── T_4_K_6/
        └── gif/
            └── low_ratio_0.9_salient_ratio_0.1/
```

建议 helper：

```python
def snn_neuron_variant_dirname(cfg, neuron: str) -> str:
    if neuron == "phase":
        return f"T_{...}"
    if neuron == "mtn":
        return f"T_{...}_K_{...}"
    if neuron == "gif":
        return f"low_ratio_{...}_salient_ratio_{...}"
```

然后：

```python
snn_dir(neuron)
```

直接返回包含该 variant 的最终 SNN root。

这样 `convert_snn.py`、`evaluate_tldr.py`、`evaluate_lm_harness.py` 只要继续通过 ArtifactLayout 获取路径，就可自动进入新目录。

---

# 21. 删除 phase.max_spikes：配置

从：

```text
configs/experiment_matrix.yaml
```

删除所有：

```yaml
phase:
  max_spikes: ...
```

重新运行：

```bash
python scripts/materialize_configs.py
```

确保所有：

```text
configs/generated/*.yaml
```

也不再包含 `max_spikes`。

`resolve_config()` 不要再补 default。

`validate_config()` 不要再校验该字段。

测试 fixture 中也全部删除。

---

# 22. 删除 phase.max_spikes：state

当前 `build_phase_state()` 中类似：

```python
"max_spikes": int(phase_cfg.get("max_spikes", steps))
```

完整删除。

新生成的：

```text
phase_state.pt
```

不得包含：

```text
max_spikes
```

旧 state 若仍包含旧 schema，应通过 format/version 拒绝使用，不做兼容读取。

---

# 23. 删除 phase.max_spikes：PhaseSurrogate runtime

从 `snn2/neurons.py::PhaseSurrogate` 完整删除：

```python
self.max_spikes = ...
```

以及：

```python
spike_count = torch.zeros_like(x)
```

删除：

```python
if self.max_spikes > 0:
    spike = spike * (spike_count < self.max_spikes).to(spike.dtype)
```

删除：

```python
spike_count = spike_count + spike.detach()
```

最终每个 timestep 是否发放只由：

```text
membrane
threshold/amplitude
surrogate/hard spike
```

决定。

不再有任何跨 timestep 的 spike-count cap。

---

# 24. Phase runtime 新语义回归测试

新增一个明确测试：

- 构造 `T > 4`（例如 `T=6`）的 Phase state；
- 输入足够大，使每一个 timestep 都满足发放条件；
- 证明 6 个 timestep 均允许发放；
- 不存在历史 `max_spikes=4` 截断。

同时断言：

```python
assert "max_spikes" not in phase_state
assert not hasattr(module, "max_spikes")
```

仓库执行：

```bash
git grep -n "max_spikes"
git grep -n "spike_count"
```

最终应不存在与旧 spike-count cap 有关的生产代码或配置引用。

若测试名称/文档中是在描述“已删除的历史行为”，可保留文字说明；生产配置和 runtime 不得残留。

---

# 25. 版本升级

本次既改变：

- calibration manifest schema；
- Stage A/B artifact schema；
- conversion metadata fingerprint schema；
- Phase temporal runtime 语义；

因此不要让旧 artifact 被新代码静默使用。

当前 `snn2/temporal_ops.py` 中存在：

```text
TEMPORAL_IMPLEMENTATION_VERSION
TEMPORAL_IMPLEMENTATION
SITE_STATE_FORMAT_VERSION
CALIBRATION_MANIFEST_FORMAT_VERSION
CONVERSION_METADATA_FORMAT_VERSION
```

本次至少进行一次整体版本升级，例如：

```text
TEMPORAL_IMPLEMENTATION_VERSION: 5 -> 6
TEMPORAL_IMPLEMENTATION: sparse_llm_temporal_v5 -> sparse_llm_temporal_v6

SITE_STATE_FORMAT_VERSION: 7 -> 8
CALIBRATION_MANIFEST_FORMAT_VERSION: 8 -> 9
CONVERSION_METADATA_FORMAT_VERSION: 9 -> 10
```

具体新整数必须全项目一致。

`STATISTICS_FORMAT_VERSION` 不因为 `max_spikes` 删除而改变，因为底层 `statistics.pt` schema 不依赖 max_spikes。

另外新增：

```text
STATISTICS_MANIFEST_FORMAT_VERSION
```

用于 Stage-A manifest schema。

同步更新：

```text
configs/experiment_matrix.yaml
deployment.temporal_implementation
```

重新 materialize generated configs。

所有 validator/test fixture 同步新版本。

---

# 26. state validation 重构

修改 `snn2/state_validation.py`。

当前逻辑默认：

```text
同一个 site directory 同时存在
statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

新结构下该假设必须完全删除。

Stage-B state bundle validator 只要求：

普通 state bundle：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
[clip_state.pt depending policy]
calibration_summary.json
```

不要求 `statistics.pt`。

statistics 的完整性通过：

```text
calibration_state_manifest
    -> source_statistics_manifest_sha256
    -> Stage-A statistics hashes
```

绑定。

保留：

- topology 完整性；
- per-site state hash；
- Site 5 clip 禁止；
- phase/gif/mtn temporal step consistency；
- global final RMSNorm Phase state；
- grouping policy；
- softmax Site 5 policy。

新增 fingerprint helper 和相应 validator。

---

# 27. conversion.py 必须删除“state root 必须有 statistics.pt”的旧要求

当前 `validate_calibration()` 中类似：

```python
required = [
    "statistics.pt",
    "phase_state.pt",
    "gif_state.pt",
    "mtn_state.pt",
]
```

改为不要求：

```text
statistics.pt
```

State root 中 statistics 已不应该存在。

所有需要 statistics provenance 的地方改为读取 Stage-B manifest 中的：

```text
source_statistics_root
source_statistics_manifest_sha256
```

---

# 28. training.py 调整

修改：

- `capture_training_artifact_provenance`
- `verify_training_artifact_provenance_unchanged`
- `validate_recorded_training_artifact_provenance`
- `train_full_parameters`

`SiteController` 的：

```text
site_root
```

必须指向当前配置对应的：

```text
ann_training Stage-B state variant
```

而不是 Stage-A statistics root。

记录：

- calibration `group_size`
- calibration `num_samples`
- Stage-A statistics manifest hash
- 当前 ANN dependency fingerprint
- 当前 Stage-B root（informational）
- Prefix provenance

不再用整个 Stage-B manifest hash 决定 ANN checkpoint 是否一致。

---

# 29. evaluation.py 调整

`build_evaluation_controller()`：

- aware ANN evaluation 使用当前对应 Stage-B state variant；
- SNN evaluation 使用当前 `conversion_site_dir` / current Stage-B variant；
- 校验 ANN dependency fingerprint；
- SNN 依赖 conversion metadata 中 neuron-scoped fingerprint。

`evaluation_calibration_metadata()` / `evaluation_forward_metadata()` 增加必要字段：

```text
calibration_num_samples
state_variant
deployment_parameters
deployment_state_fingerprint_sha256
```

不要让 metrics 中只记录 `group_size` 而漏掉 `num_samples`。

---

# 30. TL;DR / lm-eval 输出路径

`evaluate_tldr.py` 和 `evaluate_lm_harness.py` 若已统一使用：

```python
layout.snn_dir(neuron)
```

则不要重复拼 neuron variant。

只保证 ArtifactLayout 返回的新 `snn_dir()` 已经包含：

```text
phase/T_x
mtn/T_x_K_x
gif/low_ratio_x_salient_ratio_x
```

TL;DR 后续：

```text
evaluation/tldr/test_samples_x/prefix_enabled_xxx/
```

继续接在新的 neuron-specific SNN root 之后。

---

# 31. common Clip provenance 的明确规则

ANN-training Stage B bundle 始终含 Clip state。

ANN training fingerprint：

| ANN mode | common_clip=false | common_clip=true |
|---|---|---|
| phase_aware | phase | phase + clip |
| gif_aware | gif | gif + clip |
| vanilla | 无 replacement state | 不适用 |
| unaware | 无 replacement state | 不适用 |

这张表必须体现在代码 helper 中，禁止散落多个 if/else 实现不同规则。

建议：

```python
def ann_training_state_kinds(cfg) -> tuple[str, ...]:
    ...
```

---

# 32. Stage-B variant 与 ANN path 的关系

Stage-B variant 包含：

```text
phase.T
mtn.T
mtn.K
gif.low_ratio
gif.salient_ratio
```

ANN path 不包含 `mtn.K`。

因此出现：

```text
mtn.K=6 Stage-B root
mtn.K=8 Stage-B root
```

但可能对应同一个 Phase-aware/GIF-aware ANN output path。

这是有意设计。

代码必须依靠 **dependency-scoped fingerprint** 判断 ANN 所依赖的 state 是否真正变化，不能依赖 Stage-B root string。

例如 Phase-aware + common_clip=true：

```text
mtn.K 6 -> 8
```

应该：

- Stage-B `mtn_state.pt` 变化；
- Stage-B variant 目录变化；
- ANN `phase+clip` fingerprint 不变；
- ANN checkpoint 可以继续有效；
- MTN SNN conversion 使用新的 MTN state 和新的 `T_x_K_x` SNN 目录。

---

# 33. Stage-B variant 与 SNN path 的关系

同理，Stage-B bundle 是全 neuron bundle，而 SNN path 是 neuron-specific。

例如只改：

```text
mtn.K
```

则：

- Phase SNN path 不变；
- GIF SNN path 不变；
- MTN SNN path 改变；
- conversion validation 必须看 neuron state fingerprint，不能看整个 bundle hash。

这一点必须用测试覆盖。

---

# 34. scripts/verify_artifacts.py

该文件当前包含大量 artifact path/provenance 检查，本次必须系统更新。

重点删除所有旧假设：

```text
statistics 与 states 共目录
calibration root 下直接是 layer_xxx
整个 calibration manifest hash == ANN/SNN semantic identity
calibration_group_size_x 是唯一 calibration path token
num_samples 固定 128
phase state 包含 max_spikes
SNN neuron root 没有 neuron-specific parameter directory
```

新增检查：

1. Stage-A statistics root 完整；
2. Stage-B state root 完整；
3. state manifest -> statistics manifest hash 能正确追溯；
4. group_size + num_samples 与路径/manifest/config 一致；
5. Stage-B variant 名与 manifest materialization parameters 一致；
6. ANN path 参数与 config 一致；
7. SNN path neuron 参数与 config 一致；
8. dependency-scoped fingerprints 正确；
9. no stale Clip in post-finetuning；
10. no `max_spikes` in Phase state。

---

# 35. 需要重点修改的文件

Codex 必须检查并按实际引用关系修改至少以下文件：

```text
configs/experiment_matrix.yaml
configs/generated/*.yaml

scripts/_common.py
scripts/calibrate_sites.py
scripts/collect_calibration_statistics.py          # new
scripts/materialize_calibration_states.py          # new
scripts/materialize_configs.py
scripts/verify_artifacts.py
scripts/convert_snn.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
scripts/train_ann.py

snn2/artifacts.py
snn2/calibration.py
snn2/config.py
snn2/data.py
snn2/stats.py
snn2/neurons.py
snn2/state_validation.py
snn2/training.py
snn2/conversion.py
snn2/evaluation.py
snn2/temporal_ops.py
snn2/controller.py
```

以及所有 repository shell scripts / docs / tests 中对以下旧接口的引用。

实施前执行：

```bash
git grep -n "calibrate_sites.py"
git grep -n "ann_training_site_dir"
git grep -n "post_finetuning_site_dir"
git grep -n "conversion_site_dir"
git grep -n "calibration_group_dirname"
git grep -n "calibration_group_size_"
git grep -n "calibration_state_manifest"
git grep -n "statistics_manifest"
git grep -n "max_spikes"
git grep -n "spike_count"
git grep -n "snn_dir("
```

必须逐处判断并更新，不要仅修改本文列出的文件而遗漏其它调用方。

---

# 36. Shell scripts / 项目文档

仓库中现有 ANN training shell scripts 可能仍调用：

```bash
python scripts/calibrate_sites.py ...
```

全部改为 Stage A + Stage B 两条明确命令。

注意不要在每次参数 sweep 时都调用 Stage A。

参数 sweep shell 应遵循：

```text
如果变化的是：
phase.T / mtn.T / mtn.K / gif.low_ratio / gif.salient_ratio
→ 只运行 Stage B

如果变化的是：
calibration.num_samples
→ prepare_data + Stage A + Stage B

如果变化的是：
calibration.group_size
→ 按本次目录设计视为新的 Stage-A scope：
   Stage A + Stage B

如果模型 checkpoint / Prefix / rotation / calibration sample selection 改变
→ Stage A + Stage B
```

项目 markdown 文档中所有旧“一次 calibration”表述统一更新成 Stage A / Stage B。

---

# 37. 测试修改/新增

重点更新现有：

```text
tests/test_calibration_gif.py
tests/test_calibration_profiles.py
tests/test_calibration_topology.py
tests/test_controller_state_loading.py
tests/test_conversion_metadata.py
tests/test_evaluation_paths.py
tests/test_generated_configs.py
tests/test_neurons.py
tests/test_phase_conversion_regression.py
tests/test_post_finetuning_protocol.py
```

以及其它受路径/manifest影响的测试。

至少新增以下覆盖。

## 37.1 Stage A/B 分离

断言 Stage A 后：

```text
statistics.pt 存在
statistics_summary.json 存在
statistics_manifest.json 存在

phase_state.pt 不存在
gif_state.pt 不存在
mtn_state.pt 不存在
clip_state.pt 不存在
```

Stage B 后：

```text
state_root 中 state 文件存在
state_root 中 statistics.pt 不存在
statistics_root 未被写入 state
```

## 37.2 Stage B 复用 statistics

同一个 Stage-A root：

先：

```text
phase.T=4, mtn.T=4, mtn.K=6, low=0.9, salient=0.1
```

materialize。

再改：

```text
mtn.K=8
```

只运行 materialize。

断言：

- 出现两个 sibling Stage-B variant；
- Stage-A `statistics_manifest.json` hash 不变；
- Stage-A per-site `statistics.pt` hash 不变。

## 37.3 参数影响正确性

- phase.T 改变：
  - phase state 变；
  - ANN-training clip 可能变；
  - global final Phase state 变。
- mtn.T 改变：
  - mtn state 变；
  - ANN-training clip 可能变。
- mtn.K 改变：
  - mtn state 变；
  - clip state 不变。
- low_ratio 改变：
  - ordinary GIF mask/state 变；
  - Site 5 identity GIF 不变；
  - clip state 不因 ratio 改变。

## 37.4 common_clip true/false

同 Stage-A + 同 Stage-B params：

- materialized state hashes相同；
- Phase/GIF ANN controller 在 false 时不加载 Clip；
- true 时加载 eligible Clip；
- Site 5 永远不 Clip。

## 37.5 ANN path

精确断言：

Phase-aware：

```text
phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_surrogate_slope_1.0_warmup_ratio_0.0
```

GIF-aware：

```text
phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_warmup_ratio_0.0
```

vanilla/unaware 不新增该参数层。

## 37.6 SNN path

断言：

```text
snn/phase/T_4
snn/mtn/T_4_K_6
snn/gif/low_ratio_0.9_salient_ratio_0.1
```

non-aware 前面保留：

```text
calibration_group_size_-1_num_samples_128
```

## 37.7 provenance false-positive 回归

### ANN

Phase-aware + clip=true：

只改 `mtn.K`，新 Stage-B bundle root 不同，但：

```text
phase+clip fingerprint 相同
```

因此旧 ANN checkpoint provenance validation 必须仍通过。

### SNN

只改 `mtn.K`：

- Phase deployment fingerprint 不变；
- GIF deployment fingerprint 不变；
- MTN deployment fingerprint 改变。

## 37.8 max_spikes 删除

- config 无该 key；
- phase state 无该 key；
- runtime 无该 attribute；
- T=6 时不再受 4-spike cap；
- legacy old state version 被拒绝。

## 37.9 num_samples

验证：

```text
num_samples=128
num_samples=256
```

可产生不同 calibration data manifest 和不同 calibration root，互不覆盖。

`num_samples <= 0` 被拒绝。

---

# 38. 旧 artifact 处理策略

不要尝试让新代码静默兼容当前旧 mixed calibration 目录：

```text
.../sites/layer_xxx/site_xx/
    statistics.pt
    phase_state.pt
    ...
```

新 Stage A/B schema 应通过版本和 manifest 校验明确拒绝旧 state bundle。

建议错误信息明确提示：

```text
Legacy one-shot calibration artifact detected.
Re-run Stage A statistics collection and Stage B state materialization.
```

因为本次还删除了 `max_spikes`，旧 Phase state 本身也不应继续使用。

---

# 39. 不要做的事情

1. 不要在 Stage B 加载模型或重新跑 calibration dataset。
2. 不要在 Stage-B state 目录复制 `statistics.pt` 以“兼容旧 validator”。
3. 不要继续使用整个 Stage-B manifest SHA 作为 ANN training 唯一 identity。
4. 不要继续使用整个 Stage-B manifest SHA 作为单 neuron SNN conversion 唯一 identity。
5. 不要让 `common_clip_enabled=false` 导致 Stage B 不生成 Clip。
6. 不要让 `mtn.K` 进入 Phase-aware/GIF-aware ANN output path。
7. 不要从 Phase-aware ANN 路径中省略用户明确要求的 `low_ratio` / `salient_ratio`。
8. 不要在 GIF-aware ANN 路径加入 `surrogate_slope`。
9. 不要修改 vanilla/unaware ANN Fine-tuning path 的参数层规则。
10. 不要保留任何 runtime `max_spikes` / `spike_count` cap。
11. 不要让 Site 5 生成 Clip。
12. 不要让不同 `num_samples` 覆盖同一个 calibration data manifest。

---

# 40. 最终示例

假定：

```yaml
calibration:
  group_size: -1
  num_samples: 128

phase:
  T: 4
  surrogate_slope: 1.0

mtn:
  T: 4
  K: 6

gif:
  low_ratio: 0.9
  salient_ratio: 0.1

training:
  warmup_ratio: 0.0

replacement:
  common_clip_enabled: true
```

## Stage A

```text
.../ann_training_calibration/
prefix_enabled_ture/
calibration_group_size_-1_num_samples_128/
statistics/
```

## Stage B

```text
.../ann_training_calibration/
prefix_enabled_ture/
calibration_group_size_-1_num_samples_128/
states/
phase_T_4_mtn_T_4_mtn_K_6_gif_low_ratio_0.9_gif_salient_ratio_0.1/
```

## Phase-aware ANN

```text
.../
prefix_enabled_ture_common_clip_enabled_true/
phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_surrogate_slope_1.0_warmup_ratio_0.0/
seed42/
```

## GIF-aware ANN

```text
.../
prefix_enabled_ture_common_clip_enabled_true/
phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_warmup_ratio_0.0/
seed42/
```

## Phase SNN

```text
.../snn/phase/T_4/
```

## MTN SNN

```text
.../snn/mtn/T_4_K_6/
```

## GIF SNN

```text
.../snn/gif/low_ratio_0.9_salient_ratio_0.1/
```

---

# 41. 推荐实施顺序

按以下顺序实施，减少中间状态混乱：

1. `artifacts.py`
   - path scalar helper
   - calibration scope helper
   - Stage-B variant helper
   - ANN aware run variant
   - SNN neuron variant
   - Stage A/B directories
2. `data.py`
   - parameterized calibration data manifest
3. `config.py`
   - num_samples positive
   - remove max_spikes assumptions
4. `stats.py`
   - statistics manifest hash/schema
5. `calibration.py`
   - collect-only
   - dual-root materialization
6. 新增两个 scripts + retire old one-shot script
7. `neurons.py`
   - remove max_spikes/spike_count
8. `temporal_ops.py`
   - version bump
9. `state_validation.py`
   - new state-only validation
   - state fingerprint
10. `training.py`
    - dependency-scoped ANN provenance
11. `conversion.py`
    - neuron-scoped conversion fingerprint
12. `evaluation.py` + evaluate scripts
13. `verify_artifacts.py`
14. configs + generated configs
15. shell scripts/docs
16. tests

---

# 42. 验收命令

修改后至少执行：

```bash
python scripts/materialize_configs.py
```

然后：

```bash
python -m compileall snn2 scripts tests
```

针对性测试：

```bash
pytest -q \
  tests/test_calibration_gif.py \
  tests/test_calibration_profiles.py \
  tests/test_calibration_topology.py \
  tests/test_controller_state_loading.py \
  tests/test_conversion_metadata.py \
  tests/test_evaluation_paths.py \
  tests/test_generated_configs.py \
  tests/test_neurons.py \
  tests/test_phase_conversion_regression.py \
  tests/test_post_finetuning_protocol.py
```

最后：

```bash
pytest -q
```

并执行：

```bash
git grep -n "max_spikes"
git grep -n "spike_count"
git grep -n "python scripts/calibrate_sites.py"
```

生产代码/配置/正式执行脚本中不应残留旧逻辑或旧 one-shot 命令。

---

# 43. 最终验收标准

只有以下全部满足才算完成：

- [ ] Stage A 只进行 statistics collection。
- [ ] Stage B 不加载模型、不跑 calibration forward。
- [ ] 修改 Stage-B 参数时只需重复 Stage B。
- [ ] calibration root 为 `calibration_group_size_<g>_num_samples_<n>`。
- [ ] Stage-B variant 显式包含 `phase.T / mtn.T / mtn.K / gif.low_ratio / gif.salient_ratio`。
- [ ] Phase-aware ANN path 精确包含 `phase_T / mtn_T / low_ratio / salient_ratio / surrogate_slope / warmup_ratio`。
- [ ] GIF-aware ANN path 精确包含 `phase_T / mtn_T / low_ratio / salient_ratio / warmup_ratio`。
- [ ] vanilla/unaware ANN Fine-tuning path 的参数层规则不变。
- [ ] `common_clip_enabled=true/false` 均完整可运行。
- [ ] ANN-training Stage B 始终生成 eligible Clip state。
- [ ] ANN provenance 按实际 Phase/GIF/Clip dependency fingerprint。
- [ ] SNN conversion provenance 按 deployment neuron fingerprint。
- [ ] SNN 路径分别为 `phase/T_x`、`mtn/T_x_K_x`、`gif/low_ratio_x_salient_ratio_x`。
- [ ] calibration data manifest 不会因不同 num_samples 相互覆盖。
- [ ] `phase.max_spikes` 从配置、state、runtime、测试 fixture 中完整删除。
- [ ] `spike_count < self.max_spikes` 及 spike-count cap 完整删除。
- [ ] 新旧 artifact 通过 schema/version 明确隔离。
- [ ] `verify_artifacts.py` 能验证新的 Stage A/B + fingerprint provenance。
- [ ] 全部测试通过。

