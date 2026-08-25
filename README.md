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

`phase_aware` 与 `gif_aware` 在 ANN fine-tuning 阶段分别使用 Phase 或 GIF replacement。ANN-training calibration 始终为 aware mode 生成 `clip_state.pt`；`replacement.common_clip_enabled` 决定 ANN forward 是否实际在 replacement 后执行该 Clip。SNN conversion/deployment 始终不使用 common Clip。

## ANN Training / Final ANN Evaluation / SNN Evaluation Semantics

`--neuron ann` 表示 non-temporal final ANN execution，并不等价于 identity。Final ANN evaluation 会按 checkpoint 的 `ann_mode` 自动恢复训练期 static activation semantics。

| Mode | ANN training | Final ANN `--neuron ann` | SNN `--neuron phase/gif/mtn` |
|---|---|---|---|
| `vanilla` | identity | identity | temporal selected neuron |
| `unaware` | identity | identity | temporal selected neuron |
| `phase_aware` | `PhaseSurrogate.forward()` | 同一 ANN-training states 的 `PhaseSurrogate.forward()` | temporal selected neuron |
| `gif_aware` | `StaticGIF.forward()` | 同一 ANN-training states 的 `StaticGIF.forward()` | temporal selected neuron |

Aware final ANN evaluation 同时镜像训练期 common Clip 开关；SNN evaluation 永远不执行 common Clip。`phase_aware`/`gif_aware` 的 site-local surrogate 会立即聚合回 static tensor，时间维度不跨层传播，因此仍是 ANN fine-tuning；只有 `deploy_phase/gif/mtn` 属于 full-temporal SNN。Base 与 rotated-pre-finetuning diagnostic 始终保持 identity semantics。

## Mode-aware SNN Conversion

`vanilla` 不使用 Pre-finetuning Prefix；`vanilla` 和 `unaware` 在 ANN 微调后生成 Post-finetuning Prefix 与 clip-free conversion calibration。`phase_aware` 和 `gif_aware` 则跳过这两个 Post-finetuning 步骤，最终 Phase/GIF/MTN conversion 必须复用 ANN 微调前固定的 Pre-finetuning Prefix 与 ANN-training calibration，并用 SHA-256 校验它们与训练记录完全一致。

Aware ANN-training bundle 为局部静态 replacement 保留 `clip_state.pt`，但 SNN controller 只加载选定的 Phase/GIF/MTN state，conversion、deployment 和 evaluation 均不实例化或执行 Clip。

其中 GIF 内部量化使用的整数范围 clamp 属于 GIF 自身算法，不属于 ANN-aware training 中的 common Clip。

Prefix K/V 在 ANN-aware replacement 与 SNN deployment runtime 中都经过 Site 3/4 neuron；calibration statistics 仍排除 Prefix positions。Phase `surrogate_slope` 接受正有限值，phase-aware run 按 slope 隔离，但 shared ANN-training calibration 与 `phase_state.pt` 不按 slope 分叉；训练和 final ANN 评估由 controller 从当前配置注入实际 slope。Phase `tau` 按 SpikingLLM 的逐 forward、逐 channel absolute-max EMA（factor `0.99`）校准且 accumulator 固定为 FP32。Aware ANN-training calibration manifest 明确允许 `aware_modes_only` conversion reuse。Temporal deployment 对全部 Softmax（含 Prefix columns）执行 Site 5，embedding 均匀分配为 `x/T`，且仅 Phase deployment 在最终 RMSNorm 后执行独立的 global Phase neuron。

Phase τ 使用 SpikingLLM-aligned channel view：attention heads 只在 Phase statistics 中按参考实现 reshape，每个 channel 完成 FP32 EMA 后再取 global max 得到 scalar τ；该 statistical view 不改变 generic GIF/MTN/Clip statistics 或 runtime neuron tensor layout。

## 主要入口

- `实验执行总结.md`：当前实验执行顺序与命令。
- `代码结构总结.md`：当前仓库目录结构及各文件功能。
- `环境配置.md`：项目运行环境与依赖配置。
- `AGENTS.md`：后续修改代码时必须遵守的项目规则。
