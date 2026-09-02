# SNN

本项目用于研究 **Transformer 大语言模型从 ANN fine-tuning 到 SNN deployment 的转换流程**。核心目标是在 Transformer 的多个 activation site 上引入脉冲神经元相关的 activation replacement，并比较不同 ANN 微调策略与最终 SNN 推理行为。

## 当前实现

项目当前主要支持：

- Qwen3 Base 系列上的 TL;DR summarization 实验。
- Llama 3 Base 上的 Tulu 3 / lm-eval 实验。
- 四种 ANN 设置：`vanilla`、`unaware`、`phase_aware`、`gif_aware`。
- 每个 Transformer layer 固定 10 个 activation replacement sites。
- Hadamard Rotation，包括离线权重融合与在线 R3/R4 变换。
- Prefix 技术：Prefix 预先转换为固定 KV cache，并在训练、calibration 或 evaluation 的指定阶段注入。
- 模式感知 calibration：aware 模式在转换时复用 ANN-training calibration，vanilla/unaware 使用 Post-finetuning conversion calibration。
- Phase、GIF、MTN 三类 SNN conversion 与 temporal deployment。
- TL;DR ROUGE evaluation 与 lm-eval evaluation。

## ANN Fine-tuning

`vanilla` 和 `unaware` 不执行 activation replacement。

`phase_aware` 与 `gif_aware` 在 ANN fine-tuning 阶段分别使用 Phase 或 GIF replacement。ANN-training calibration 拆成 T/K-independent Stage A 和按 `phase.T/mtn.T` 隔离的 Stage B Clip profile；`replacement.common_clip_enabled` 只决定是否应用已选 profile。Site 5 永久 no-Clip，SNN conversion/deployment 始终不执行 common Clip。

## ANN Training / Final ANN Evaluation / SNN Evaluation Semantics

`--neuron ann` 表示 non-temporal final ANN execution，并不等价于 identity。Final ANN evaluation 会按 checkpoint 的 `ann_mode` 自动恢复训练期 static activation semantics。

| Mode | ANN training | Final ANN `--neuron ann` | SNN `--neuron phase/gif/mtn` |
|---|---|---|---|
| `vanilla` | identity | identity | temporal selected neuron |
| `unaware` | identity | identity | temporal selected neuron |
| `phase_aware` | `PhaseSurrogate.forward()` | 同一 ANN-training states 的 `PhaseSurrogate.forward()` | temporal selected neuron |
| `gif_aware` | Site 1/7 使用 role-aware `StaticGIF`，Site 2 使用 all-low 4-bit GIF，Site 5/8/9 为 identity，其余 salient sites 使用 `StaticGIF` | 同一 ANN-training states 的 static GIF forward | temporal selected neuron |

Aware final ANN evaluation 同时镜像训练期 common Clip 开关；SNN evaluation 永远不执行 common Clip。`phase_aware`/`gif_aware` 的 site-local surrogate 会立即聚合回 static tensor，时间维度不跨层传播，因此仍是 ANN fine-tuning；只有 `deploy_phase/gif/mtn` 属于 full-temporal SNN。Base 与 rotated-pre-finetuning diagnostic 始终保持 identity semantics。

## Mode-aware SNN Conversion

`vanilla` 和 `unaware` 在 ANN 微调后生成 Post-finetuning Prefix，并只运行 Post-finetuning Stage A。`phase_aware` 和 `gif_aware` 复用 ANN 微调前固定的 Prefix、Stage A 与训练所选 Stage B profile；conversion/deployment 始终只读取 Stage A，绝不加载 Clip。

`vanilla` 的 final ANN `--neuron ann` 始终是无 Prefix 的 identity 评估；即使已生成 Post-finetuning Prefix，运行时也不会加载它。该 Prefix 用于 vanilla 后续的 Post-finetuning conversion calibration、SNN conversion，以及 Phase/GIF/MTN SNN evaluation。`unaware` 的 final ANN evaluation 则仍由 `evaluation.prefix_enabled` 决定是否加载其 Post-finetuning Prefix。

Stage A 的 site 目录只包含 statistics 与 T/K-independent Phase/GIF/MTN state。Stage B 位于独立的 `clip_profiles/phase_T_<P>_mtn_T_<M>/`，并通过 Stage A manifest 哈希绑定来源。训练和 final aware ANN evaluation 分别传入 Stage A root 与 Stage B Clip root；两者都冻结并校验 provenance。

其中 GIF 内部量化使用的整数范围 clamp 属于 GIF 自身算法，不属于 ANN-aware training 中的 common Clip。

Prefix K/V 在 ANN-aware replacement 与 SNN deployment runtime 中都经过 Site 3/4 neuron；calibration statistics 仍可排除 Prefix positions。Phase EMA 固定为 FP32、factor `0.99`。Phase state 只保存 T-independent `tau`，运行时按 `v0 = 0.5 * tau * 2^-T` 构造；MTN state 也不保存 T/K。`phase.base` 固定为 `2.0`，旧 `max_spikes` 配置已删除。

`calibration.group_size` 同时控制 Phase/GIF/MTN/Clip 和 final RMSNorm Phase。Site 2/3/4 的参数保持 logical per-head grouping，Site 6 使用 merged last-dim。Stage B Clip 对每个 group 做 mask-aware all-low/all-high/mixed 分类；Site 1 保存 q/k/v role-specific interval，Site 7 保存 gate/up interval，Site 5 永久 identity/no-Clip。

Calibration data、Stage A、aware run 与 non-aware SNN 路径同时按 `calibration.group_size` 和 `calibration.num_samples` 隔离。Aware run 还包含训练期 `phase.T/mtn.T`；Phase SNN 使用 `phase/phase_T_<P>`，MTN SNN 使用 `mtn/mtn_T_<M>_mtn_K_<K>`，GIF 路径不变。改变部署 T/K 不修改 Stage A；改变 `num_samples` 或 G 必须重建对应数据 manifest 和 Stage A。

## 主要入口

- `实验执行总结.md`：当前实验执行顺序与命令。
- `代码结构总结.md`：当前仓库目录结构及各文件功能。
- `环境配置.md`：项目运行环境与依赖配置。
- `AGENTS.md`：后续修改代码时必须遵守的项目规则。
