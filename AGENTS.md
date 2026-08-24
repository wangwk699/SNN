# AGENTS.md

1. Clip 只用于 ANN 的 `phase_aware` / `gif_aware` 微调：ANN-training calibration 可以生成并使用 `clip_state.pt`；Post-finetuning conversion calibration、SNN conversion、Phase/GIF/MTN SNN deployment 与 SNN evaluation 均不得生成、加载或执行任何 Clip。 `phase_aware`/`gif_aware` 的 SNN conversion 可以复用含 `clip_state.pt` 的 ANN-training calibration bundle，但该文件只因 ANN-aware training 而存在，SNN controller 仍只能读取选定 neuron state。

2. `代码结构总结.md` 只允许保留 `2. 目录结构`，用于记录当前仓库目录结构；每个文件后只用一句话描述该文件实现的功能，任何代码修改导致目录或文件功能变化时都必须同步更新该文件。

3. ANN 与 SNN 必须根据模型前向传播是否按真实 SNN 时间动态运行来区分：只有时间维度 `T` 在层间持续传播、模型以真实脉冲动态执行前向并据此微调时，才属于 SNN 微调。`phase_aware` 与 `gif_aware` 微调只在各 activation replacement site 内进行局部 SNN 动态模拟并立即聚合为静态 tensor，用于模拟转换误差；时间维度 `T` 不得在层间传播，因此二者仍属于 ANN 微调，不得称为完整 SNN 微调，也不得在其训练路径中引入跨层 temporal deployment。

4. Prefix K/V 在实际 ANN-aware replacement 与 SNN deployment 中必须经过 Site 3/4 neuron；calibration statistic 可排除 Prefix positions，但不得因此让 Prefix runtime bypass neuron。

5. 普通 Phase main experiment 固定 `surrogate_slope=1.0`，Phase `tau` EMA accumulator 固定 FP32。
