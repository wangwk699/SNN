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

`phase_aware` 与 `gif_aware` 在 ANN fine-tuning 阶段分别使用 Phase 或 GIF replacement。**Clip 只属于这两个 ANN-aware training 路径**：ANN-training calibration 会生成 `clip_state.pt`，用于约束 Phase/GIF replacement 的输出范围。

## Mode-aware SNN Conversion

`vanilla` 不使用 Pre-finetuning Prefix；`vanilla` 和 `unaware` 在 ANN 微调后生成 Post-finetuning Prefix 与 clip-free conversion calibration。`phase_aware` 和 `gif_aware` 则跳过这两个 Post-finetuning 步骤，最终 Phase/GIF/MTN conversion 必须复用 ANN 微调前固定的 Pre-finetuning Prefix 与 ANN-training calibration，并用 SHA-256 校验它们与训练记录完全一致。

Aware ANN-training bundle 为局部静态 replacement 保留 `clip_state.pt`，但 SNN controller 只加载选定的 Phase/GIF/MTN state，conversion、deployment 和 evaluation 均不实例化或执行 Clip。

其中 GIF 内部量化使用的整数范围 clamp 属于 GIF 自身算法，不属于 ANN-aware training 中的 common Clip。

Phase `tau` 按 SpikingLLM 的逐 forward、逐 channel absolute-max EMA（factor `0.99`）校准；temporal deployment 对全部 Softmax（含 Prefix columns）执行 Site 5，embedding 均匀分配为 `x/T`，且仅 Phase deployment 在最终 RMSNorm 后执行独立的 global Phase neuron。

## 主要入口

- `实验执行总结.md`：当前实验执行顺序与命令。
- `代码结构总结.md`：当前仓库目录结构及各文件功能。
- `环境配置.md`：项目运行环境与依赖配置。
- `AGENTS.md`：后续修改代码时必须遵守的项目规则。
