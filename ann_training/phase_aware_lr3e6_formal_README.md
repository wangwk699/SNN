# lr=3e-6 正式实验

```bash
cd /home/wangwenkang/SNN
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/run_qwen3_8b_phase_aware_lr3e6_formal.sh
```

仅启动1组新训练：lr=3e-6，slope=4，warmup=0.2，Clip=true，G4，phase.T=4，mtn.T=4，mtn.K=6，seed42，max_grad_norm=1。训练10000条、测试1000条、1 epoch、全局batch64，使用snn2及既有CPU optimizer offload。从共享旋转后预训练初始化开始，不从lr2e-6检查点续训。不会自动运行lr4e-6。

4项参考仅评估/复用：vanilla、unaware、GIF和最新lr2e-6 Phase正式对照（ROUGE-L=0.248093）。新候选与参考采用相同1000条测试样本，已存在的兼容评估复用。

结果目录 `artifacts/phase_aware_lr3e6_formal_v1/`：

- plan.json与configs：冻结计划和配置。
- summary.csv/json/md：指标、训练损失、完成状态和工件路径。
- paired_bootstrap.json：新候选相对4项参考的逐样本ROUGE-L配对比较。
- training_audit_layers.csv与training_audit_status.json：新训练的逐步逐层梯度、采样FP32主权重实际更新及记录完整性；原始记录在summary引用的training_audit_path。
- logs与status：日志及失败原因。

预演追加 `--dry-run`；仅汇总追加 `--summarize-only`。中断后重跑原命令，完成工件校验复用，未完成训练从头开始；部分最终模型不会自动覆盖。默认输出目录独立于之前实验，旧结果保留。

新检查点约16GB，另需评估和日志空间。完成后通知助手读取上述目录，比较lr3e-6相对lr2e-6的收益及训练变化，再决定下一步。
