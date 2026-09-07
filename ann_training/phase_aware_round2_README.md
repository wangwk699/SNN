# 第二轮受控 Phase-aware 实验

运行（自行启动 GPU 实验）：

```bash
cd /home/wangwenkang/SNN
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/sweep_qwen3_8b_phase_aware_round2.sh
```

默认 snn2 Python，14组实验依次使用4张GPU。固定 phase.T=4、mtn.T=4、mtn.K=6、G4、seed42、max_grad_norm=1、训练1024条/评估128条、1 epoch、全局batch64。不自动启动10000/1000正式训练。

矩阵：slope={3,4,5,6}×warmup={0.1,0.2}、lr1e-6、Clip=true共8组；slope={4,5}×lr={5e-7,2e-6}、warmup0.2、Clip=true共4组；slope={4,5}、warmup0.2、lr1e-6、Clip=false共2组。每个新训练均从既有共享旋转后预训练模型开始，复用同G校准与冻结Prefix，不从p03继续训练。

预演：同命令追加 `--dry-run`。汇总：同命令追加 `--summarize-only`。重新执行原命令会复用已完成且校验通过的结果；中断训练会从头重启该未完成项，部分最终检查点会拒绝覆盖。可用 SWEEP_DIR 设置新的计划/报告目录，但它不会复制或隔离底层模型工件，因此同配置已完成模型仍会复用。

输出目录：`artifacts/phase_aware_round2_v1/`。

- `plan.json`、`configs/`：冻结的实验计划和配置。
- `summary.csv/json/md`、`paired_bootstrap.json`：ROUGE、训练损失、状态及已有配对比较。
- `training_audit_status.json`：每组训练探针的记录数和路径，旧结果无探针时标记 unavailable。
- `training_audit_layers.csv`：每个优化器步、每层的完整分片归约preclip梯度范数、全局范数与DeepSpeed范数交叉检查、估算裁剪系数，以及采样FP32主权重实际更新范数和相对更新幅度。
- `logs/`、`status/`：运行日志和失败原因；单个实验失败后继续其余实验，最终以非零退出码报告存在失败。

探针只在新脚本配置 audit_gradients=true 时启用，在ZeRO-3优化器step前后读取数据，不更改梯度、权重或优化器算法。梯度统计覆盖完整有效分片并排除padding，更新统计每参数每rank最多4096个确定性采样坐标，归约所有rank后按层输出；FP32主权重更新包含AdamW自适应更新和weight decay，不等于BF16权重每坐标可见的变化。支持GPU驻留和当前CPU optimizer offload的ZeRO-3；CPU路径直接读取同步后的CPU分片，只将小型统计表搬到GPU用于NCCL归约，不支持NVMe optimizer swapping。完整8B训练的新增探针尚未GPU实测，建议关注第一组日志。

已有相同配置的旧实验不会为补探针而覆盖重训，其指标仍可作对照，但不可声称其已有训练更新统计。三组原有vanilla/unaware/GIF参考模型训练样本为10000，不能与1024训练候选直接作公平结论；候选之间比较后再选少数正式验证。

最多14个新8B检查点约224GB，实际因复用减少；另需评估和日志空间。运行时间取决于探针开销及复用数量，尚无本轮实测估计。运行完成后保留整个报告目录及其引用的模型结果，通知助手读取分析即可。
