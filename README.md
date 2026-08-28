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

`phase_aware` 与 `gif_aware` 在 ANN fine-tuning 阶段分别使用 Phase 或 GIF replacement。ANN-training calibration 为 9 个 clip-eligible sites 生成 `clip_state.pt`；Site 5 永久 no-Clip。`replacement.common_clip_enabled` 决定 ANN forward 是否在其余 sites 执行 Clip，SNN conversion/deployment 始终不执行 common Clip。

## ANN Training / Final ANN Evaluation / SNN Evaluation Semantics

`--neuron ann` 表示 non-temporal final ANN execution，并不等价于 identity。Final ANN evaluation 会按 checkpoint 的 `ann_mode` 自动恢复训练期 static activation semantics。

| Mode | ANN training | Final ANN `--neuron ann` | SNN `--neuron phase/gif/mtn` |
|---|---|---|---|
| `vanilla` | identity | identity | temporal selected neuron |
| `unaware` | identity | identity | temporal selected neuron |
| `phase_aware` | `PhaseSurrogate.forward()` | 同一 ANN-training states 的 `PhaseSurrogate.forward()` | temporal selected neuron |
| `gif_aware` | ordinary sites 使用 `StaticGIF`，Site 5 使用 SpikeLLM 16-bit sentinel identity `SoftmaxIdentityGIF` | 同一 ANN-training states 的 static GIF forward | temporal selected neuron |

Aware final ANN evaluation 同时镜像训练期 common Clip 开关；SNN evaluation 永远不执行 common Clip。`phase_aware`/`gif_aware` 的 site-local surrogate 会立即聚合回 static tensor，时间维度不跨层传播，因此仍是 ANN fine-tuning；只有 `deploy_phase/gif/mtn` 属于 full-temporal SNN。Base 与 rotated-pre-finetuning diagnostic 始终保持 identity semantics。

## Mode-aware SNN Conversion

`vanilla` 不使用 Pre-finetuning Prefix；`vanilla` 和 `unaware` 在 ANN 微调后生成 Post-finetuning Prefix 与 clip-free conversion calibration。`phase_aware` 和 `gif_aware` 则跳过这两个 Post-finetuning 步骤，最终 Phase/GIF/MTN conversion 必须复用 ANN 微调前固定的 Pre-finetuning Prefix 与 ANN-training calibration，并用 SHA-256 校验它们与训练记录完全一致。

Aware ANN-training bundle 在 Site 1/2/3/4/6/7/8/9/10 保留 `clip_state.pt`，Site 5 不存在该文件；bundle 校验使用 `require_eligible`（aware ANN）、`allow_eligible`（aware SNN reuse）和 `forbid_all`（post-finetuning）三态协议。SNN controller 即使复用含 9 个 Clip 的 bundle 也只加载选定的 Phase/GIF/MTN state，conversion、deployment 和 evaluation 均不实例化或执行 Clip。

其中 GIF 内部量化使用的整数范围 clamp 属于 GIF 自身算法，不属于 ANN-aware training 中的 common Clip。

Prefix K/V 在 ANN-aware replacement 与 SNN deployment runtime 中都经过 Site 3/4 neuron；calibration statistics 仍可排除 Prefix positions。Phase `surrogate_slope` 接受正有限值，phase-aware run 按 slope 和 `training.warmup_ratio` 联合隔离，但 shared calibration/state 与这两个训练参数无关；训练和 final ANN 评估从当前 YAML 显式注入 slope。Phase EMA 固定为 FP32、factor `0.99`。

`calibration.group_size` 同时控制 ordinary Phase/GIF/MTN/Clip 和 final RMSNorm Phase。`G=-1` 对非 attention 表示整个最后维度一组，对 Site 2/3/4/6 表示每个 head 各自一组；`G>0` 只在每个 head 的 `D` 内分组，绝不跨 head。Site 2/6 使用 query heads，Site 3/4 保留 `repeat_kv()` 前的原生 KV heads，并把 repeat 后 saliency 累加回 KV heads。Site 5 忽略 G：Phase/MTN 为 per-head `[H,1]`，GIF 严格按 SpikeLLM `n_bits=16` sentinel 执行 identity，不做 GIF calibration、Q16 fake quantization 或 Site 5 temporal encoding，且永远 no-Clip。ordinary GIF 的 `high_qmax=30`、`per_step_qmax=15` 仅适用于其余九个 Site。完整 Softmax（含 Prefix columns）在 runtime 经过 Site 5；final RMSNorm Phase 也按 G 分组。

所有 G-dependent calibration（包括 resolved config、logs、statistics、states、manifest）、aware ANN run 和 SNN conversion/evaluation 路径均包含且只包含一次 `calibration_group_size_<G>_num_samples_<N>`，metadata 同时记录 G 与 `per_head_within_head_groups_v1` policy。aware ANN 将 G 写入学习率目录后缀（如 `lr5e-05_train_samples_2048_calibration_group_size_-1_num_samples_128`），其 SNN 路径为该 run root 下的 `snn/<neuron>/<neuron-variant>`；vanilla/unaware 为 `snn/calibration_group_size_<G>_num_samples_<N>/<neuron>/<neuron-variant>`。identity ANN checkpoint 可跨 G 共享，但改变 G 后仍必须重做其 post-finetuning calibration 与 SNN 工件；Rotation、数据和 Prefix 不随 G 复制。

## 主要入口

- `实验执行总结.md`：当前实验执行顺序与命令。
- `代码结构总结.md`：当前仓库目录结构及各文件功能。
- `环境配置.md`：项目运行环境与依赖配置。
- `AGENTS.md`：后续修改代码时必须遵守的项目规则。
