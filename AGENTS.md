# AGENTS.md

1. Clip 只用于 ANN 的 `phase_aware` / `gif_aware` 微调：ANN-training calibration 可以生成并使用 `clip_state.pt`；Post-finetuning conversion calibration、SNN conversion、Phase/GIF/MTN SNN deployment 与 SNN evaluation 均不得生成、加载或执行任何 Clip。 `phase_aware`/`gif_aware` 的 SNN conversion 可以复用含 `clip_state.pt` 的 ANN-training calibration bundle，但该文件只因 ANN-aware training 而存在，SNN controller 仍只能读取选定 neuron state。

2. `代码结构总结.md` 只允许保留 `2. 目录结构`，用于记录当前仓库目录结构；每个文件后只用一句话描述该文件实现的功能，任何代码修改导致目录或文件功能变化时都必须同步更新该文件。

3. ANN 与 SNN 必须根据模型前向传播是否按真实 SNN 时间动态运行来区分：只有时间维度 `T` 在层间持续传播、模型以真实脉冲动态执行前向并据此微调时，才属于 SNN 微调。`phase_aware` 与 `gif_aware` 微调只在各 activation replacement site 内进行局部 SNN 动态模拟并立即聚合为静态 tensor，用于模拟转换误差；时间维度 `T` 不得在层间传播，因此二者仍属于 ANN 微调，不得称为完整 SNN 微调，也不得在其训练路径中引入跨层 temporal deployment。

4. Prefix K/V 在实际 ANN-aware replacement 与 SNN deployment 中必须经过 Site 3/4 neuron；calibration statistic 可排除 Prefix positions，但不得因此让 Prefix runtime bypass neuron。

5. Phase `surrogate_slope` 允许使用任意正有限值进行对比实验；`phase_aware` run root 与所有 aware ANN-training calibration root 必须包含 `surrogate_slope_<value>`，同一 slope 的 phase-aware/gif-aware 继续共享 calibration，不同 slope 必须隔离，防止复用错误的 `phase_state.pt`。Phase `tau` EMA accumulator 固定 FP32。

6. Phase calibration statistics 与 generic site statistics 必须解耦。Site 2/3/4/5/6 的 Phase EMA 必须使用 SpikingLLM-aligned statistical view；不得为了 Phase τ 对齐而改变 GIF/MTN/Clip statistics 或 runtime neuron layout。

7. ANN-training calibration 对 `phase_aware` / `gif_aware` 始终生成 `clip_state.pt`；`replacement.common_clip_enabled` 只控制 ANN training forward 是否应用 Clip，不得改变 shared calibration 内容。

8. aware run root 必须包含 `common_clip_enabled_true` 或 `common_clip_enabled_false`；shared Pre-finetuning Prefix 和 ANN-training calibration 不得因为该开关拆成两套。

9. Final ANN evaluation (`--neuron ann`) 必须复现对应 ANN training 的 static activation semantics：`vanilla`/`unaware` 为 `identity(x)`，`phase_aware` 为 `PhaseSurrogate.forward()`，`gif_aware` 为当前 `StaticGIF.forward()`。

10. `--neuron ann` 只表示 non-temporal ANN execution，不等价于 identity；`--neuron phase|gif|mtn` 在正式 evaluation 中始终表示 full-temporal `deploy_*` SNN deployment。

11. `phase_aware`/`gif_aware` final ANN evaluation 必须读取对应 ANN training 的同一 calibration states、验证训练 provenance，并镜像 `replacement.common_clip_enabled`；不得改用 Post-finetuning conversion calibration。

12. 任何 `ann_mode` 进入 SNN evaluation 后都必须使用选定 neuron 的 `deploy_*` temporal path；不得用 static surrogate 代替 Temporal SNN neuron。

13. Base baseline 与 rotated-pre-finetuning ANN diagnostic 保持 identity activation semantics；mode-aware static surrogate 只作用于 final ANN checkpoint evaluation。
