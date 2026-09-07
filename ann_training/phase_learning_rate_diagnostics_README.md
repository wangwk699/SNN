# 学习率检查点与Site5累积误差诊断

```bash
cd /home/wangwenkang/SNN
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/diagnose_qwen3_8b_phase_learning_rate.sh
```

默认snn2环境，单进程在4张GPU上分配8B模型，依次加载lr2e-6和lr3e-6的正式检查点。保持各自冻结的Prefix/Stage A/Stage B，校验hash和tokenizer；固定T4/T4/K6、G4、slope4、warmup0.2，正式Phase语义的Clip保持true。无优化器、无反向传播、不写模型检查点。

使用validation按seed42选择相同128条具有有效completion标签的样本，先各跑16条再补齐128条。每个检查点每样本6次前向：identity诊断、完整Phase、仅旁路第0–11层Site5、仅旁路第12–23层Site5、仅旁路第24–35层Site5、旁路全部36层Site5。分块均单独相对完整Phase，其他位置保持原样；Site5本身从来没有Clip。identity和旁路均仅用于诊断，不能用来替代正式评分模型，不能把不同块收益相加。

NLL覆盖所有有效completion token，prompt不计loss。完整Phase收集局部量化误差、层间隐藏误差以及注意力统计；注意力最多选32个预测completion的query行，保留全部Prefix/正文key，输出Prefix和正文质量、行和、非零项归零比例和同V下局部PV误差。其他激活统计仍是全序列确定性采样，不能视为全张量或completion专属。KL为最多32个completion预测位置的采样统计，identity参照属于同一检查点。

输出：`artifacts/phase_lr_diagnostics_v1/calibration_group_size_4/`。

- comparison.md/json：两个检查点的token加权NLL、lr3-minus-lr2配对差、4种旁路NLL收益及5000次配对bootstrap逐项95%区间（未多重比较校正），以及局部/注意力/隐藏状态统计。
- records/p01和records/p03：分别对应lr2e-6和lr3e-6，包含每条样本原始统计。
- validation_manifest.json、specification.json、extension.json、run_fingerprint.json：固定数据、模型/代码hash与分块定义。
- logs/和status.json：执行日志及完成/失败状态。

预演加 `--dry-run`；仅先完成16条加 `--smoke-only`，随后重跑默认命令会复用已完成样本；仅重建汇总加 `--summarize-only`。中断后原命令续跑，已完成样本保留，未完成单条重新计算。汇总的样本数可能是部分结果，以status和两模型样本数为准。变更代码、模型或选项后需新的DIAGNOSTIC_DIR，避免混合结果。

完整流程1536次前向（首轮16条被复用），另含模型加载与hash读取。没有本轮GPU实测时长；可先smoke-only观察。保留原始数据及统计，完成后通知助手读取分析。修改仅扩展旁路集合和completion query选择，原诊断脚本默认行为保持；修改代码hash后不要试图续写旧诊断目录。
