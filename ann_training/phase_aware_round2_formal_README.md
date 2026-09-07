# 第二轮正式实验

```bash
cd /home/wangwenkang/SNN
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/run_qwen3_8b_phase_aware_round2_formal.sh
```

默认使用snn2环境，4张GPU依次运行一组新的10000训练/1000测试实验：

| 新ID | 快速实验来源 | slope | lr | warmup | Clip |
|---|---|---:|---:|---:|---|
| p01 | 第二轮p10 | 4 | 2e-6 | 0.2 | true |

固定phase.T=4、mtn.T=4、mtn.K=6、G4、seed42、max_grad_norm=1、训练1 epoch、全局batch64。该组从共享旋转后预训练模型开始，不继续训练1024样本检查点。保持原CPU optimizer offload，并记录训练梯度和采样FP32主权重实际更新。

脚本包含4项只评估参考：vanilla、unaware、GIF，以及此前正式Phase对照（slope4/lr1e-6/warmup0.2/Clip=true）。均为10000训练、1000测试；校验通过的既有评估直接复用。Phase参考缺失训练结果时会报错，不自动重训参考。

结果：`artifacts/phase_aware_round2_formal_cliptrue_v3/`，包含冻结plan/configs、summary.csv/json/md、paired_bootstrap.json、training_audit_layers.csv、training_audit_status.json和logs/status。paired_bootstrap默认比较当前最高ROUGE-L候选与4个参考；候选预测文件保留供后续逐样本分析，新p01对应快速实验p10。

预演：原命令追加 `--dry-run`；仅重建汇总：追加 `--summarize-only`。中断后重跑原命令，已完成且验证通过的实验复用，未完成训练从头开始；不要删除旧结果。SWEEP_DIR可以指定新的报告目录，但不隔离底层模型路径。

预计一个新8B检查点约16GB，另需日志和评估空间。运行时长尚未实测；脚本不会启动其他调参训练。实验结束后通知助手读取结果分析即可。

用户最新约束：今后新Phase-aware实验固定Clip=true，入口启用--require-clip-true拒绝false候选。已删除第二组，仅运行slope4/lr2e-6/warmup0.2/Clip=true；旧v1/v2计划和结果保留，新计划使用cliptrue_v3目录避免冻结配置冲突。
