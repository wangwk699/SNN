# Qwen3-8B Phase-aware 批量调参

在项目根目录执行（自动使用 `/home/wangwenkang/miniconda3/envs/snn2/bin/python`）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/sweep_qwen3_8b_phase_aware.sh
```

如需后台执行：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/sweep_qwen3_8b_phase_aware.sh > phase_sweep.log 2>&1 &
```

默认生成 `artifacts/phase_aware_sweep_v1/`，50 组候选全部使用 `training.tldr_train_samples=1024`、`evaluation.tldr_test_samples=128`，固定 `phase.T=4`、`mtn.T=4`、`mtn.K=6`、seed42、校准128条、Prefix开启、训练1 epoch、四卡全参数训练和原项目训练语义。本批不自动启动 10000/1000 正式实验，等全部结果分析后再选择正式候选。

## 实验矩阵

参数矩阵直接写在 Bash 中，Python 负责验证、调度与汇总；下面各行之间没有重复配置。

| 实验族 | 参数组合 | 数量 |
| --- | --- | ---: |
| 分组 × 斜率 | G=1/4/16/32/128，slope=0.5/1/4，lr=1e-6，warmup=0.1，Clip=true | 15 |
| 学习率 | G=1/4/16，slope=1/4，lr=2.5e-7/5e-7/2e-6，warmup=0.1，Clip=true | 18 |
| Clip 对照 | G=1/4/16，slope=1/4，lr=1e-6，warmup=0.1，Clip=false | 6 |
| warmup 对照 | G=1/4，slope=1/4，lr=1e-6，warmup=0/0.2，Clip=true | 8 |
| 大斜率对照 | G=1，slope=8/16，lr=1e-6，warmup=0.1，Clip=true | 2 |
| G64 对照 | G=64，slope=1，lr=1e-6，warmup=0.1，Clip=true | 1 |

此前 `G=1、slope=1、lr=1e-5` 的 loss 连续恶化，故未将该学习率纳入本批。slope=16 虽梯度大但已有完整结果，保留作对照并验证后复用。

此外先对用户指定的 vanilla lr1e-6、unaware lr1e-6、gif_aware lr2e-6/T4 的 10000 条训练 checkpoint 各评估同一 seed42 的 128 条测试样本，**不重新训练基线**。汇总会标明候选与基线训练样本数不同；历史1000条评估单独保留，不能和128条结果直接配对。测试子集上的排名与 bootstrap 都是探索性结果。

## 运行与恢复

- 逐组使用全部4张指定GPU运行；切换 G 时，在该 G 的独立目录重新采集 Stage A，随后物化 Stage B；已存在且通过校验的同 G 工件会复用，Clip=false 也保留项目要求的 Stage B provenance。
- 需要项目既有的 shared Rotation、数据 manifest、Pre Prefix 和三个基线 checkpoint；不会复制或重新生成这些共享工件。现有工件损坏、缺失或配置冲突会记录明确错误。
- 每组训练/校准/评估日志带唯一时间戳；失败后继续其他组，最终只要有失败就返回非零退出码。
- 再运行同一命令：验证配置、训练样本数、GPU数量、checkpoint分片和 frozen provenance 后跳过已完成结果；训练已完成但评估缺失时只补评估。
- Ctrl-C 或 SIGTERM 会终止当前实验进程组、记录 interrupted 状态并汇总；不会继续下一组。`save_strategy=no`，中断的训练下次从头开始，已完成训练可复用；不承诺恢复优化器中间状态。
- 若存在不完整的最终 checkpoint 或校准文件，脚本会拒绝覆盖；先检查对应错误和日志，再由使用者处理这些文件。
- 本程序通过锁防止自身的两个实例同时修改共享工件；旧实验脚本不遵守此锁，运行本批时不要同时用旧脚本写相同的模型、校准或数据目录。
- 50 组总预算粗估12–16小时、模型空间最多约0.85TB（不计原有模型），依负载和校准速度变化；保留所有模型和预测，不自动删除。前面已完成且配置一致的两组会复用。

配置一经计划生成便冻结；修改 Bash 矩阵或源配置时请设置新的 `SWEEP_DIR`，不要覆盖旧计划。默认源配置为 `configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml`，可通过 `SOURCE_CFG` 指定；不同主机可通过 `SNN2_PYTHON` 指定 snn2 Python 路径。即使换 SWEEP_DIR，若某组目标模型目录已存在但配置不同，也会拒绝覆盖。

## 预览和汇总

只生成计划与配置，不启动任何训练/校准/生成评估：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/sweep_qwen3_8b_phase_aware.sh --dry-run
```

每组结束自动更新汇总；全部结束后也可以单独重建：

```bash
./ann_training/sweep_qwen3_8b_phase_aware.sh --summarize-only
# 或直接运行辅助脚本（不需要 GPU）：
/home/wangwenkang/miniconda3/envs/snn2/bin/python scripts/phase_aware_sweep.py \
  --output artifacts/phase_aware_sweep_v1 --summarize-only
```

输出文件：

- `plan.json`、`configs/*.yaml`：完整矩阵、配置及训练设置签名。
- `status/*.json`：每组成功/复用/失败/中断状态、时间和错误。
- `logs/<实验ID>/*.log`：每次尝试的完整子进程日志。
- `summary.csv`、`summary.json`：每组参数、状态、ROUGE-1/2/L/Lsum、训练loss/耗时、首末loss、最大记录梯度范数及原始工件路径。
- `summary.md`：已验证完成的候选按ROUGE-L降序排列、基线比较以及未完成清单。
- `historical_metrics.json`：所有历史 ANN 评估，保留真实测试样本数和路径。
- `paired_bootstrap.json`：当前最佳候选对三个基线的逐样本ROUGE-L差值及5000次配对bootstrap区间，要求样本ID和参考摘要完全匹配；第一个候选完成后生成。

原始模型、`metrics.json` 和 `predictions.jsonl` 仍在项目原有 `artifacts/snn2_main_v1/...` 目录。汇总不会将失败实验、待运行实验或仅发现但未核验的历史指标计入成功排名。实验完成后提供 `summary.md` 或告知输出目录即可继续分析。


## 三组正式实验（快速扫描之后）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/run_qwen3_8b_phase_aware_formal.sh
```

运行顺序如下，均训练10000条、测试1000条，phase.T=4、mtn.T=4、mtn.K=6；其余设置保持对应快速实验一致：

| 正式ID | 原快速ID | G | slope | lr | warmup | Clip |
|---|---|---:|---:|---:|---:|---|
| p01 | p05 | 4 | 1 | 1e-6 | 0.1 | true |
| p02 | p19 | 1 | 1 | 1e-6 | 0.1 | false |
| p03 | p47 | 4 | 4 | 1e-6 | 0.2 | true |

默认输出目录为 `artifacts/phase_aware_formal_v1/`。三个指定基线的1000条评估经过验证后复用；若缺失则只补评估，不重训基线。新组沿用同G的已验证校准工件，重新从项目原始训练初始化做一轮10000条训练，不从1024条微调checkpoint继续训练。排名、CSV/JSON与bootstrap都使用1000条口径，不会误用128条结果。

预计四卡运行4–6小时，三份新checkpoint约50GB；实际依负载变化。断点重跑、错误记录和信号中断行为与快速脚本相同。首次运行前可加 `--dry-run` 预览；完成后重新汇总：

```bash
./ann_training/run_qwen3_8b_phase_aware_formal.sh --summarize-only
```

也可后台运行：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 ./ann_training/run_qwen3_8b_phase_aware_formal.sh > phase_formal.log 2>&1 &
```

完成后告知正式实验已完成即可读取 `artifacts/phase_aware_formal_v1/summary.md`、`summary.json` 和 `paired_bootstrap.json` 继续分析。不同规模测试集可能有重叠，正式结果也不是完全独立于快速筛选的新验证集。
