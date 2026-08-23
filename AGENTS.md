# AGENTS.md

1. Clip 只用于 ANN 的 `phase_aware` / `gif_aware` 微调：ANN-training calibration 可以生成并使用 `clip_state.pt`；Post-finetuning conversion calibration、SNN conversion、Phase/GIF/MTN SNN deployment 与 SNN evaluation 均不得生成、加载或执行任何 Clip。

2. `代码结构总结.md` 只允许保留 `2. 目录结构`，用于记录当前仓库目录结构；每个文件后只用一句话描述该文件实现的功能，任何代码修改导致目录或文件功能变化时都必须同步更新该文件。
