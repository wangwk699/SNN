# AGENTS.md

1. `calibration.group_size` 必须为 `-1` 或正整数，并控制普通 Site 和 global Final RMSNorm 的 Phase/MTN grouping；Final RMSNorm 永远不生成或执行 GIF/Clip，改变 G 后必须重新 calibration，禁止复用另一 G 的 statistics/state/manifest/conversion/SNN 工件。

2. Site 2 保留原生 `[B,H,L,D]` layout，并只允许在每个 head 的 `D` 内 grouping；Site 3/4 必须在 `repeat_kv()` 后以 query heads 为逻辑 per-head 参数坐标，实际 replacement tensor 为 merged `[B,L,HD]`，saliency 不得折叠回原生 KV heads；Site 6 必须在 attention head merge 后以普通 `[B,L,HD]` last-dim layout replacement，不得保留 per-head 参数。

3. Site 5 忽略全局 G：Phase `tau` 与 MTN `base_scale` 固定为 `[H,1]`；GIF 必须严格遵循 SpikeLLM `n_bits=16` sentinel 的真实行为，static 与 temporal 均 exact identity，不得执行 Q16、round/clamp、scale/zero-point calibration、qmax30/[0,15] chunk 或 cumulative-difference quantization；Site 5 永远不得生成、加载或执行 Clip。

3a. ordinary GIF 的 qmax30/T=2/[0,15]×2 仅允许用于 Site 1/2/3/4/6/7/8/9/10；禁止将 ordinary GIF metadata 解释为 Site 5 策略。

4. ANN-training Stage A 的 site 目录只生成 `statistics.pt`、`phase_state.pt`、`gif_state.pt` 与 `mtn_state.pt`，不得含 `clip_state.pt`；Stage B 才在 `clip_profiles/phase_T_<P>_mtn_T_<M>/` 为 9 个 eligible sites 生成 Clip，Site 5 永久无 Clip。`replacement.common_clip_enabled` 只控制 aware ANN forward；SNN conversion/deployment 永不加载、实例化或执行 Clip，Post-finetuning 只允许完全 clip-free 的 Stage A，禁止 Stage B。

5. Prefix K/V 在实际 ANN-aware replacement 与 SNN deployment 中必须经过 Site 3/4 neuron，完整 Softmax（含 Prefix key columns）必须经过 Site 5；calibration statistics 可以排除 Prefix positions/columns，但不得让 Prefix runtime bypass Site 3/4/5。

6. Phase `tau` 与 MTN `base_scale` 必须共享原生 site layout 的 FP32、factor `0.99` 有序 EMA，再按既有 group policy 取 group max；`tau=clamp(EMA_group_max,5e-4,1e4)`，`base_scale=clamp(2*EMA_group_max,5e-4,1e4)`，并直接保存 clamp 后 tensor，禁止旧 extrema-based MTN 初始化；calibration 固定 `batch_size=1` 且单进程，Final RMSNorm 同样受 G 控制。

7. Phase `surrogate_slope` 允许任意正有限值，且仅作为 phase-aware ANN training/final ANN evaluation 的运行时反向传播参数；Phase state 不得保存 slope，ANN controller 必须从当前 YAML 显式接收，SNN deployment 使用硬阈值。phase-aware run root 必须包含 `surrogate_slope_<value>_warmup_ratio_<value>`。

8. G-dependent calibration 的 config/log/statistics/state/manifest、aware ANN run、SNN conversion/evaluation 路径必须包含且只包含一次 `calibration_group_size_<G>`，metadata 还必须显式保存 G 与 grouping policy；aware ANN run 将其写入学习率目录后缀 `lr..._calibration_group_size_<G>`。所有 SNN 路径在 `snn/` 后必须恰好包含一次 `use_post_finetuning_artifacts_<bool>`：aware 接 `phase|gif|mtn`，vanilla/unaware 再接 `calibration_group_size_<G>_num_samples_<N>` 与 neuron。identity ANN checkpoint 不因 selector 或 G 分叉，Rotation、数据与 shared Pre bundle 不得复制。

8a. SNN conversion artifact source 只能由 `conversion.use_post_finetuning_artifacts` 决定，不得从 `ann_mode` 推断：true 使用 Post-finetuning Prefix + Post-finetuning Stage A；false 使用 Pre-finetuning Prefix + ANN-training Stage A，且 vanilla 禁止 false。phase/gif aware + false 必须校验训练时冻结的 shared Prefix/Stage A/Stage B provenance；phase/gif aware + true 必须使用当前 Final ANN 的 Post bundle，不能要求其 hash 等于 ANN-training Stage A；unaware + false 不得要求 aware training provenance。

9. `phase_aware` 与 `gif_aware` 仍是 site-local static replacement 的 ANN fine-tuning，时间维度不跨层传播；只有 `deploy_phase/gif/mtn` 属于 full-temporal SNN。`--neuron ann` 按 ann_mode 恢复 identity/Phase/GIF static semantics，`--neuron phase|gif|mtn` 始终使用 temporal deployment。
14. Final RMSNorm 是 global replacement position，不计入每层 10 sites：Phase-aware ANN 在 ordinary Final RMSNorm 后执行无 Clip 的 `PhaseSurrogate`；GIF-aware ANN 与 GIF SNN 均 identity；Phase/MTN SNN 在 Temporal Final RMSNorm 后分别执行 temporal Phase/MTN neuron。global 目录只允许 `phase_state.pt` 与 `mtn_state.pt`，禁止 GIF/Clip state。

15. Embedding temporal encoding 固定 `uniform_embedding_divide_by_T`，Prefix K/V temporal decomposition 固定 `uniform_kv_divide_by_T`；不得为了对齐 SparseLLM 改写这些策略或本项目的 site-specific grouping granularity。


10. Final aware ANN evaluation 必须读取训练时同一 G 的 ANN-training states、镜像 common Clip 开关并验证 frozen provenance；Final ANN Prefix source 独立于 selector：vanilla 不加载、unaware 用 Post Prefix、aware 用 Pre Prefix（均受 evaluation.prefix_enabled 约束）。

11. statistics/state/manifest/conversion/temporal schema 必须严格拒绝旧版本，不得保留旧 statistical-view、scalar τ、mask padding/truncation或其他兼容 fallback。

12. Base baseline 与 rotated-pre-finetuning ANN diagnostic 始终保持 identity activation semantics。

13. `代码结构总结.md` 只允许保留 `2. 目录结构`；每个文件后只用一句话描述功能，任何职责变化都必须同步更新。
