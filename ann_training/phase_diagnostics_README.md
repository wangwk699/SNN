# Qwen3-8B Phase诊断

从项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/diagnose_qwen3_8b_phase_aware.sh
```

自动使用snn2环境，**单进程、模型均衡分布在四张GPU**，不是torchrun/DDP。没有optimizer、optimizer.step、微调checkpoint保存，也不改动正式评估结果。保留BF16、eager、无gradient checkpointing；模型设为eval以消除dropout，反向时启用autograd。梯度是每条validation样本的诊断梯度，不是训练时64条global batch聚合后的梯度。

## 流程

1. 校验正式p01、p03的frozen Prefix、Stage A、Stage B provenance；验证两者共享同一套工件。对模型分片、配置和诊断相关代码计算哈希，避免恢复时混用不同输入。
2. 从原固定revision的TL;DR validation split，以seed42生成固定排列，选择前128条在原2048长度截断后仍含completion标签的记录；排除项、原始内容哈希、tokenized输入和标签均保存。使用现有tokenize_row规则，completion loss使用shifted labels且屏蔽prompt。无padding，因为batch_size=1。128条的前16条用于冒烟检查。
3. 三个checkpoint依次检查前16条，再扩展到同一128条：未训练rotated权重、正式p01、正式p03。已经完成的前16条逐样本结果直接复用。
4. 各checkpoint分别运行同权重identity参考和原生slope的Phase探针；p03额外覆盖slope=0.5/1/2/4/8。每个pass都做反向但不更新权重，同时采集量化与梯度信息。不同slope核对同一checkpoint的抽样logits与全completion loss一致，否则停止并保存invariance_failure.json。
5. 全部128条完成后，在p03上按局部抽样NMSE选出最可疑的3个Phase位置，再加入global Final RMSNorm和误差最大的Site5（去重）；这些位置若存在Clip，再加入Clip旁路。每次只旁路一个算子，在同一128条上比较loss和logits KL变化。

未训练rotated模型的identity结果仍是参考路径；给该权重加Phase的结果明确标记为人工量化探针，不是项目正式的rotated baseline成绩。所有旁路只存在于诊断进程，不导出模型，不改变正式定义；不会给Site5或Final RMSNorm添加Clip。Phase旁路仍保留该位置后续Clip，Clip旁路仍保留Phase。角色Clip位置的旁路会同时旁路该位置各分支的Clip。

## 统计口径

- 现有实现是**Phase后再Clip**；挂钩观察实际模块输入/输出，分别统计两者，不重写算子算法。
- 每个算子每次调用按flatten坐标等间距最多抽样4096个值；Site2/3/4/5另外逐逻辑query head抽样256个值。抽样包含有效prompt和completion位置，Site3/4包含实际Prefix K/V。它们不是全张量精确统计，也不是仅completion位置统计。
- 输出相对L2、NMSE、余弦、偏移、零输出/非零变零比例，以及Phase输入/τ分位数、近似饱和比例。分位数汇总是每次调用抽样分位数的平均，不是全数据集分位数。
- Site5每head最多采样32个query行，但保留全部key列：统计行和、有效注意力项被清零比例、零行率、Prefix/正文质量，并用相同V比较量化权重前后的attention×V输出。因果mask的原始零项排除在“有效项清零比例”的分母之外，运行时mask不变。
- 每层输出与Final RMSNorm输出最多抽样8192个值，与同权重identity路径的同坐标输出比较；KL使用至多32个有效completion预测位置的完整词表概率，completion NLL则使用全部有效completion预测位置。
- 参数梯度精确L2范数（FP32累积）、逐层范数、梯度/权重范数比；分布、零梯度比例、与identity的方向相似度使用确定性抽样。Clip另记录被改变输入位置的零梯度比例。全局范数为全参数范数，报告假想clip系数而不实际裁剪。
- identity梯度是参考，不是量化模型的“正确梯度”。不使用有限差分验证不连续硬阈值的代理梯度。
- `training_clip_audit.json`从已有正式训练stdout中的裁剪前范数估计触发比例，与validation逐样本的假想clip比例分开记录，不能把两者或梯度范数直接解释为优化器实际更新幅度。
- 高误差位置排名用于定位，不直接证明bug。旁路会改变下游分布，只作诊断证据。

## 运行、恢复与资源

预览配置，不读取数据、不加载模型：

```bash
./ann_training/diagnose_qwen3_8b_phase_aware.sh --dry-run
```

可先只运行16条冒烟检查（仍冻结完整128条manifest）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/diagnose_qwen3_8b_phase_aware.sh --smoke-only
```

随后运行默认命令即可继续128条和自动旁路，不重复已完成样本。也可直接一次执行默认命令。

后台运行：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/diagnose_qwen3_8b_phase_aware.sh > phase_diagnostics.log 2>&1 &
```

同一输出目录有进程锁。Ctrl-C/SIGTERM停止子进程组；每条样本完成后原子保存，重新运行跳过完成样本，中断的那一条从头重做。某个任务失败时整条诊断流水线停止，避免继续分析不完整的冒烟结果。已完成结果保留。

初始模型哈希需要读取约三份模型的数据；反向诊断和各模块统计比纯评估慢，建议根据16条实际耗时估算整批时长，可能需要数小时或更久。本脚本不会申请CPU/disk模型offload，不会自动修改序列长度来规避OOM；若OOM，保留日志后分析。输出保存统计而非完整激活或新模型，但128条多个pass的逐参数JSON仍可能占用数GB。

环境覆盖项：`SNN2_PYTHON`、`FORMAL_DIR`、`DIAGNOSTIC_DIR`。改代码、样本数或slopes等选项后需使用新的DIAGNOSTIC_DIR，旧结果的fingerprint禁止混用。

## 结果

默认目录：`artifacts/phase_diagnostics_v1/calibration_group_size_4/`。

- `specification.json`、`run_fingerprint.json`：输入模型、工件、代码哈希及诊断协议。
- `validation_manifest.json`：固定样本、排除项、输入token和completion标签。
- `status/`、`logs/`：各任务状态及每次尝试的完整日志。
- `checkpoints/<checkpoint>/sample_*.json`：每条样本各pass完整统计，可做配对分析。
- `summary.md`、`summary.json`：loss、KL、梯度及全部位置汇总；未完成时只汇总已有样本并显示样本数。
- `local_sampled.csv`、`attention_sampled_rows.csv`、`cumulative_hidden_sampled.csv`、`gradient_statistics.csv`：逐位置、head和参数的可筛选表。
- `layer_gradient_norms.csv/json`：逐层参数梯度范数和权重比。
- `training_clip_audit.json`：既有训练日志的裁剪触发估计。
- `ablation_targets.json`、`ablations/sample_*.json`、`ablation_summary.json`：自动选择的诊断旁路及loss/KL恢复量，loss_delta_vs_phase为负表示旁路后loss降低。

重新汇总，不加载模型：

```bash
./ann_training/diagnose_qwen3_8b_phase_aware.sh --summarize-only
```

完成后告知运行结束，我将读取上述文件，结合量化误差、梯度方向与旁路恢复量定位下一步工作。本脚本不会自动安排后续训练。
