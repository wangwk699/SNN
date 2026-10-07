# SNN

本项目用于研究 **Transformer 大语言模型从 ANN fine-tuning 到 SNN deployment 的转换流程**。当前实现围绕 **SAT-LLM** 展开：在 ANN fine-tuning 阶段显式引入后续脉冲部署相关的神经元行为，使优化后的 ANN 更适合低 timestep 的 SNN conversion 与 temporal deployment。

## SAT-LLM 概览

<p align="center">
  <img src="assets/SAT_LLM_architecture.svg" alt="SAT-LLM architecture" width="100%">
</p>


SAT-LLM 首先对 Transformer 进行 Hadamard rotation，以缓解 activation outlier；随后在 Self-attention 与 MLP 中设置 10 个 layer-local spiking neuron replacement sites。`phase_aware` 与 `gif_aware` 在 ANN fine-tuning 阶段分别局部实现 Phase 或 GIF 神经元行为，并结合 neuron-independent common clipping；时间维度不会在 ANN layer 之间传播。Fine-tuning 完成后，模型可进一步转换为采用 Phase、GIF 或 MTN 神经元的 full-temporal SNN。

## 主要入口

- [`实验执行总结.md`](实验执行总结.md)：当前实验执行顺序、artifact 依赖与常用命令。
- [`代码结构总结.md`](代码结构总结.md)：仓库目录结构及主要模块说明。
- [`环境配置.md`](环境配置.md)：运行环境与依赖配置。
- [`AGENTS.md`](AGENTS.md)：后续修改代码时需要遵守的项目规则。

## 当前实现

项目当前主要支持：

- Qwen3 Base 系列上的 Reddit TL;DR summarization 实验。
- Llama 3 Base 上的 Tulu 3 / lm-eval 实验。
- 四种 ANN 设置：`vanilla`、`unaware`、`phase_aware`、`gif_aware`。
- 每个 Transformer layer 固定 10 个 activation replacement sites。
- Hadamard Rotation，包括离线权重融合与在线 R3/R4 变换。
- Prefix 技术：Prefix 预先转换为固定 KV cache，并在 training、calibration 或 evaluation 的指定阶段注入。
- 双套 SNN artifact source：non-vanilla 可选择 shared Pre-finetuning Prefix + ANN-training Stage A，或 per-run Post-finetuning Prefix + Stage A。
- Phase、GIF、MTN 三类 SNN conversion 与 temporal deployment。
- TL;DR ROUGE evaluation 与 lm-eval evaluation。

## 实验结果

### Qwen3-8B：Reddit TL;DR

**Table 2. Performance of Qwen3-8B on the Reddit TL;DR dataset under different ANN fine-tuning and SNN conversion settings.**

| Training Mode | Evaluated Model |    R-1 |    R-2 |    R-L | R-LSum | ROUGE (Avg.) |
| ------------- | --------------- | -----: | -----: | -----: | -----: | -----------: |
| Base          | Pre-trained ANN | 0.3216 | 0.1186 | 0.2508 | 0.2550 |       0.2365 |
| Vanilla       | Fine-tuned ANN  | 0.3526 | 0.1349 | 0.2781 | 0.2784 |       0.2610 |
| Vanilla       | Phase SNN       | 0.1295 | 0.0156 | 0.1019 | 0.1015 |       0.0871 |
| Vanilla       | GIF SNN         | 0.0569 | 0.0004 | 0.0498 | 0.0488 |       0.0390 |
| Vanilla       | MTN SNN         | 0.1038 | 0.0069 | 0.0834 | 0.0845 |       0.0697 |
| Unaware       | Fine-tuned ANN  | 0.3531 | 0.1343 | 0.2785 | 0.2785 |       0.2611 |
| Unaware       | Phase SNN       | 0.2398 | 0.0690 | 0.1832 | 0.1827 |       0.1687 |
| Unaware       | GIF SNN         | 0.1382 | 0.0179 | 0.1096 | 0.1096 |       0.0938 |
| Unaware       | MTN SNN         | 0.1734 | 0.0398 | 0.1353 | 0.1348 |       0.1208 |
| Phase-aware   | Fine-tuned ANN  | 0.3212 | 0.1075 | 0.2464 | 0.2466 |       0.2305 |
| Phase-aware   | Phase SNN       | 0.3337 | 0.1071 | 0.2562 | 0.2547 |       0.2379 |
| Phase-aware   | GIF SNN         | 0.2285 | 0.0645 | 0.1865 | 0.1865 |       0.1665 |
| Phase-aware   | MTN SNN         | 0.2499 | 0.0715 | 0.1985 | 0.1976 |       0.1794 |
| GIF-aware     | Fine-tuned ANN  | 0.3384 | 0.1181 | 0.2596 | 0.2600 |       0.2440 |
| GIF-aware     | Phase SNN       | 0.3279 | 0.1088 | 0.2461 | 0.2458 |       0.2321 |
| GIF-aware     | GIF SNN         | 0.3407 | 0.1203 | 0.2639 | 0.2631 |       0.2470 |
| GIF-aware     | MTN SNN         | 0.3221 | 0.1071 | 0.2450 | 0.2445 |       0.2297 |

### Llama3-8B：Tulu 3 SFT mixture

**Table 3. Downstream performance of Llama3-8B fine-tuned on the Tulu 3 SFT mixture under different ANN fine-tuning and SNN conversion settings.**

| Training Mode | Evaluated Model | AGIEval |  ARC-C |  BoolQ |   PIQA |    TQA |  WinoG |   Avg. |
| ------------- | --------------- | ------: | -----: | -----: | -----: | -----: | -----: | -----: |
| Base          | Pre-trained ANN |  0.2610 | 0.3882 | 0.7012 | 0.7633 | 0.2632 | 0.6235 | 0.5001 |
| Vanilla       | Fine-tuned ANN  |  0.3420 | 0.4812 | 0.8376 | 0.8063 | 0.3586 | 0.6851 | 0.5851 |
| Vanilla       | Phase SNN       |  0.2640 | 0.2927 | 0.5376 | 0.5968 | 0.2326 | 0.4909 | 0.4024 |
| Vanilla       | GIF SNN         |  0.2500 | 0.2722 | 0.4367 | 0.5103 | 0.2375 | 0.5162 | 0.3705 |
| Vanilla       | MTN SNN         |  0.2600 | 0.3114 | 0.4988 | 0.6273 | 0.2632 | 0.4980 | 0.4098 |
| Unaware       | Fine-tuned ANN  |  0.3120 | 0.4940 | 0.8483 | 0.8101 | 0.3672 | 0.7127 | 0.5907 |
| Unaware       | Phase SNN       |  0.2450 | 0.2995 | 0.6260 | 0.6317 | 0.3023 | 0.5312 | 0.4393 |
| Unaware       | GIF SNN         |  0.2470 | 0.2312 | 0.4266 | 0.5234 | 0.2570 | 0.4964 | 0.3636 |
| Unaware       | MTN SNN         |  0.2670 | 0.2577 | 0.5691 | 0.5941 | 0.2448 | 0.5122 | 0.4075 |
| Phase-aware   | Fine-tuned ANN  |  0.2660 | 0.3729 | 0.7446 | 0.7258 | 0.3219 | 0.5675 | 0.4998 |
| Phase-aware   | Phase SNN       |  0.2600 | 0.3643 | 0.7456 | 0.7073 | 0.3121 | 0.5556 | 0.4908 |
| Phase-aware   | GIF SNN         |  0.2710 | 0.2423 | 0.4193 | 0.5604 | 0.2472 | 0.4783 | 0.3698 |
| Phase-aware   | MTN SNN         |  0.2690 | 0.2927 | 0.5911 | 0.6425 | 0.3072 | 0.5304 | 0.4388 |
| GIF-aware     | Fine-tuned ANN  |  0.2580 | 0.3558 | 0.7446 | 0.7133 | 0.2938 | 0.5343 | 0.4833 |
| GIF-aware     | Phase SNN       |  0.2790 | 0.3754 | 0.7031 | 0.7018 | 0.3207 | 0.5485 | 0.4881 |
| GIF-aware     | GIF SNN         |  0.2600 | 0.3490 | 0.7453 | 0.6997 | 0.2889 | 0.5257 | 0.4781 |
| GIF-aware     | MTN SNN         |  0.2710 | 0.3370 | 0.7569 | 0.6741 | 0.3452 | 0.5359 | 0.4867 |

从结果可以看到，常规 ANN fine-tuning 在转换为低 timestep SNN 后存在明显性能下降；`phase_aware` 与 `gif_aware` 通过在 ANN 优化阶段显式考虑脉冲神经元行为，能够显著缩小 ANN→SNN 的 conversion gap。两种 aware 模式都具备一定的 cross-neuron generalization，其中 `gif_aware` 在不同部署神经元之间表现出更稳定的泛化。

## ANN Fine-tuning

`vanilla` 和 `unaware` 不执行 activation replacement。其中 `vanilla` 使用原始 ANN 进行常规 fine-tuning，`unaware` 在 Hadamard-rotated ANN 上进行 fine-tuning，但仍不显式建模后续 spiking deployment。

`phase_aware` 与 `gif_aware` 在 ANN fine-tuning 阶段分别使用 Phase 或 GIF replacement。ANN-training calibration 拆成 T/K-independent Stage A 和按 `phase.T/mtn.T` 隔离的 Stage B Clip profile；`replacement.common_clip_enabled` 只决定是否应用已选 profile。Site 5 永久 no-Clip，SNN conversion/deployment 始终不执行 common Clip。

## ANN Training / Final ANN Evaluation / SNN Evaluation Semantics

`--neuron ann` 表示 non-temporal final ANN execution，并不等价于 identity。Final ANN evaluation 会按 checkpoint 的 `ann_mode` 自动恢复训练期 static activation semantics。

| Mode          | ANN training                                                 | Final ANN `--neuron ann`                               | SNN `--neuron phase/gif/mtn` |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------- |
| `vanilla`     | identity                                                     | identity                                               | temporal selected neuron     |
| `unaware`     | identity                                                     | identity                                               | temporal selected neuron     |
| `phase_aware` | `PhaseSurrogate.forward()`                                   | 同一 ANN-training states 的 `PhaseSurrogate.forward()` | temporal selected neuron     |
| `gif_aware`   | Site 1/7 使用 role-aware `StaticGIF`，Site 2 使用 all-low 4-bit GIF，Site 5/8/9 为 identity，其余 salient sites 使用 `StaticGIF` | 同一 ANN-training states 的 static GIF forward         | temporal selected neuron     |

Aware final ANN evaluation 同时镜像训练期 common Clip 开关；SNN evaluation 永远不执行 common Clip。`phase_aware`/`gif_aware` 的 site-local surrogate 会立即聚合回 static tensor，时间维度不跨层传播，因此仍属于 ANN fine-tuning；只有 `deploy_phase/gif/mtn` 属于 full-temporal SNN。Base 与 rotated-pre-finetuning diagnostic 始终保持 identity semantics。

## SNN Conversion Artifact Source

`conversion.use_post_finetuning_artifacts` 是 SNN conversion 与 SNN evaluation 的唯一 artifact source selector：`true` 选择每个 Final ANN run 的 Post-finetuning Prefix + Post-finetuning Stage A；`false` 选择 shared Pre-finetuning Prefix + shared ANN-training Stage A。`vanilla` 只能选择 `true`。

| ANN mode      | selector=false                                  | selector=true                                    |
| ------------- | ----------------------------------------------- | ------------------------------------------------ |
| `vanilla`     | 非法                                            | Post-finetuning Prefix + Post-finetuning Stage A |
| `unaware`     | shared Pre Prefix + shared ANN-training Stage A | Post Prefix + Post Stage A                       |
| `phase_aware` | shared Pre Prefix + shared ANN-training Stage A | Post Prefix + Post Stage A                       |
| `gif_aware`   | shared Pre Prefix + shared ANN-training Stage A | Post Prefix + Post Stage A                       |

`unaware`、`phase_aware` 与 `gif_aware` 可以同时拥有两套 artifact。selector 不改变 ANN checkpoint，也不改变 Final ANN evaluation。aware 的 Post-finetuning artifacts 从各自 Final ANN 生成，并保存在各自 run 的 `post_finetuning/` 下，不进入 `_shared`。

SNN conversion/deployment 只读取 Stage A，永不读取 ANN-training Stage B Clip；Post-finetuning calibration 仅有完全 clip-free 的 Stage A。

Final ANN Prefix 规则独立保持：vanilla 不加载 Prefix；unaware 按 `evaluation.prefix_enabled` 从 Post-finetuning Prefix 加载；phase-aware/gif-aware 按同一开关从 Pre-finetuning Prefix 加载。