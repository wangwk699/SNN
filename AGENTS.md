# AGENTS.md

1. `calibration.group_size` 必须为 `-1` 或正整数，并同时控制普通 Site 与 final RMSNorm 的 Phase/GIF/MTN/Clip grouping；改变 G 后必须重新 calibration，禁止复用另一 G 的 statistics/state/manifest/conversion/SNN 工件。

2. Site 2/3/4/6 必须保留原生 `[B,H,L,D]` layout，只允许在每个 head 的 `D` 内 grouping，禁止 flatten heads 或重新引入跨 head global τ；其中 Site 2/6 使用 query heads，GQA/MQA Site 3/4 使用 `repeat_kv()` 前的原生 KV heads，repeat 后 saliency 必须按 query groups 求和回 KV heads。

3. Site 5 忽略全局 G：Phase `tau` 与 MTN `base_scale` 固定为 `[H,1]`，GIF 固定执行 `[0,1]` Q16 fake quantization，temporal GIF 使用 quantized cumulative difference；Site 5 永远不得生成、加载或执行 Clip。

4. ANN-training calibration 对 Site 1/2/3/4/6/7/8/9/10 全部生成 `clip_state.pt`，Site 5 是永久例外；`replacement.common_clip_enabled` 只控制 phase-aware/gif-aware ANN forward 是否在 clip-eligible site 执行 Clip，不得改变 shared calibration 内容。Post-finetuning conversion calibration 与所有 SNN deployment 必须完全无 Clip。

5. Prefix K/V 在实际 ANN-aware replacement 与 SNN deployment 中必须经过 Site 3/4 neuron，完整 Softmax（含 Prefix key columns）必须经过 Site 5；calibration statistics 可以排除 Prefix positions/columns，但不得让 Prefix runtime bypass Site 3/4/5。

6. Phase calibration 必须在原生 site layout 上逐 channel/head-channel 执行 FP32、factor `0.99` 的有序 EMA，再只在 group 内取 max；calibration 固定 `batch_size=1` 且单进程。final RMSNorm Phase 同样受 G 控制。

7. Phase `surrogate_slope` 允许任意正有限值，且仅作为 phase-aware ANN training/final ANN evaluation 的运行时反向传播参数；Phase state 不得保存 slope，ANN controller 必须从当前 YAML 显式接收，SNN deployment 使用硬阈值。phase-aware run root 必须包含 `surrogate_slope_<value>_warmup_ratio_<value>`。

8. G-dependent calibration、aware ANN run、SNN conversion/evaluation 路径必须包含 `calibration_group_size_<G>`，metadata 还必须显式保存 G 与 grouping policy；vanilla/unaware identity ANN checkpoint 不因 G 分叉，但其 post-finetuning calibration 和 SNN 工件必须分叉。Rotation、数据与 Prefix 等 G-independent shared 工件不得复制。

9. `phase_aware` 与 `gif_aware` 仍是 site-local static replacement 的 ANN fine-tuning，时间维度不跨层传播；只有 `deploy_phase/gif/mtn` 属于 full-temporal SNN。`--neuron ann` 按 ann_mode 恢复 identity/Phase/GIF static semantics，`--neuron phase|gif|mtn` 始终使用 temporal deployment。

10. Final aware ANN evaluation 必须读取训练时同一 G 的 ANN-training states、镜像 common Clip 开关并验证 frozen provenance；aware conversion 可以复用含 9-site Clip 的 bundle，但 SNN controller 只能加载选定 neuron state。

11. statistics/state/manifest/conversion/temporal schema 必须严格拒绝旧版本，不得保留旧 statistical-view、scalar τ、mask padding/truncation或其他兼容 fallback。

12. Base baseline 与 rotated-pre-finetuning ANN diagnostic 始终保持 identity activation semantics。

13. `代码结构总结.md` 只允许保留 `2. 目录结构`；每个文件后只用一句话描述功能，任何职责变化都必须同步更新。
