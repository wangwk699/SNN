# AGENTS.md

1. `calibration.group_size` 必须为 `-1` 或正整数，并控制普通 Site 和 global Final RMSNorm 的 Phase/MTN grouping；Final RMSNorm 永远不生成或执行 GIF/Clip，改变 G 后必须重新 calibration，禁止复用另一 G 的 statistics/state/manifest/conversion/SNN 工件。

2. Site 2 保留原生 `[B,H,L,D]` layout，并只允许在每个 head 的 `D` 内 grouping；Site 3/4 必须在 `repeat_kv()` 后以 query heads 为逻辑 per-head 参数坐标，实际 replacement tensor 为 merged `[B,L,HD]`，saliency 不得折叠回原生 KV heads；Site 6 必须在 attention head merge 后以普通 `[B,L,HD]` last-dim layout replacement，不得保留 per-head 参数。

3. Site 5 忽略全局 G：Phase `tau` 与 MTN `base_scale` 固定为 `[H,1]`；GIF 必须严格遵循 SpikeLLM `n_bits=16` sentinel 的真实行为，static 与 temporal 均 exact identity，不得执行 Q16、round/clamp、scale/zero-point calibration、qmax30/[0,15] chunk 或 cumulative-difference quantization；Site 5 永远不得生成、加载或执行 Clip。

3a. ordinary GIF 的 qmax30/T=2/[0,15]×2 仅允许用于 Site 1/2/3/4/6/7/8/9/10；禁止将 ordinary GIF metadata 解释为 Site 5 策略。

4. ANN-training calibration 对 Site 1/2/3/4/6/7/8/9/10 全部生成 `clip_state.pt`，Site 5 是永久例外；Clip bundle 必须使用 `require_eligible`（aware ANN）、`allow_eligible`（aware SNN 复用）与 `forbid_all`（post-finetuning）三态语义。`replacement.common_clip_enabled` 只控制 aware ANN forward；aware SNN bundle 可保留 9 个 Clip 文件，但 SNN controller 永不加载、实例化或执行 Clip，post-finetuning bundle 则必须完全 clip-free。

5. Prefix K/V 在实际 ANN-aware replacement 与 SNN deployment 中必须经过 Site 3/4 neuron，完整 Softmax（含 Prefix key columns）必须经过 Site 5；calibration statistics 可以排除 Prefix positions/columns，但不得让 Prefix runtime bypass Site 3/4/5。

6. Phase `tau` 与 MTN `base_scale` 必须共享原生 site layout 的 FP32、factor `0.99` 有序 EMA，再按既有 group policy 取 group max；`tau=clamp(EMA_group_max,5e-4,1e4)`，`base_scale=clamp(2*EMA_group_max,5e-4,1e4)`，并直接保存 clamp 后 tensor，禁止旧 extrema-based MTN 初始化；calibration 固定 `batch_size=1` 且单进程，Final RMSNorm 同样受 G 控制。

7. Phase `surrogate_slope` 允许任意正有限值，且仅作为 phase-aware ANN training/final ANN evaluation 的运行时反向传播参数；Phase state 不得保存 slope，ANN controller 必须从当前 YAML 显式接收，SNN deployment 使用硬阈值。phase-aware run root 必须包含 `surrogate_slope_<value>_warmup_ratio_<value>`。

8. G-dependent calibration 的 config/log/statistics/state/manifest、aware ANN run、SNN conversion/evaluation 路径必须包含且只包含一次 `calibration_group_size_<G>`，metadata 还必须显式保存 G 与 grouping policy；aware ANN run 将其写入学习率目录后缀 `lr..._calibration_group_size_<G>`，并在该 run root 下使用 `snn/<neuron>`，vanilla/unaware 在 `snn/calibration_group_size_<G>/<neuron>` 下分叉。identity ANN checkpoint 不因 G 分叉，Rotation、数据与 Prefix 等 G-independent shared 工件不得复制。

9. `phase_aware` 与 `gif_aware` 仍是 site-local static replacement 的 ANN fine-tuning，时间维度不跨层传播；只有 `deploy_phase/gif/mtn` 属于 full-temporal SNN。`--neuron ann` 按 ann_mode 恢复 identity/Phase/GIF static semantics，`--neuron phase|gif|mtn` 始终使用 temporal deployment。
14. Final RMSNorm 是 global replacement position，不计入每层 10 sites：Phase-aware ANN 在 ordinary Final RMSNorm 后执行无 Clip 的 `PhaseSurrogate`；GIF-aware ANN 与 GIF SNN 均 identity；Phase/MTN SNN 在 Temporal Final RMSNorm 后分别执行 temporal Phase/MTN neuron。global 目录只允许 `phase_state.pt` 与 `mtn_state.pt`，禁止 GIF/Clip state。

15. Embedding temporal encoding 固定 `uniform_embedding_divide_by_T`，Prefix K/V temporal decomposition 固定 `uniform_kv_divide_by_T`；不得为了对齐 SparseLLM 改写这些策略或本项目的 site-specific grouping granularity。


10. Final aware ANN evaluation 必须读取训练时同一 G 的 ANN-training states、镜像 common Clip 开关并验证 frozen provenance；aware conversion 可以复用含 9-site Clip 的 bundle，但 SNN controller 只能加载选定 neuron state。

11. statistics/state/manifest/conversion/temporal schema 必须严格拒绝旧版本，不得保留旧 statistical-view、scalar τ、mask padding/truncation或其他兼容 fallback。

12. Base baseline 与 rotated-pre-finetuning ANN diagnostic 始终保持 identity activation semantics。

13. `代码结构总结.md` 只允许保留 `2. 目录结构`；每个文件后只用一句话描述功能，任何职责变化都必须同步更新。
