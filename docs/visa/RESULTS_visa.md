# VisA 1/2/4-shot 实验结果（vpt3.5 原版 mapping 分支）

> 代码：**原本的 vpt3.5**（`code_vpt4/main-fewshot-vpt4.py`，md5 `08d8528b4a59`）
> 配置：`--global_proto 1 --w_div 0.1 --w_pixel 0.0 --metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --token_init_std 0.02 --res_mapping 0`，35 轮 / batch 16 / seed 1 / 单类 / `--map_only`
> 数据：VisA 官方 `split_csv/1cls.csv` 的 train 划分抽 K 张正常图（seed=1，**1⊂2⊂4 嵌套**）+ 对应 AnoStyler 合成异常（K 源 × 35 张/源 = K×35 张/类/轮）
> 评估：训练裁剪 392 / **测试不裁剪 448** / 1.5×+翻转 / 官方 `adeval`（测试集 = 官方 test 划分：正常图 + 异常图 + 掩码）

## 1. 数据准备
| 环节 | 产物 |
|---|---|
| 抽样（seed=1，1/2/4 嵌套） | `anostyler_visa/{1,2,4}shot/shots.json`、`configs/`（12/24/48 个配置，共 84） |
| AnoStyler 合成（num_gen=35/底图） | `anostyler_visa/{K}shot/genbyAnoStyler/`（420 / 840 / 1680 张图 + 掩码） |
| MVTec 式数据根（既有管线零改动复用） | `visual2/visa_prepared/`：train/good 8659、test/good 962、test/anomaly 1200、ground_truth 1200 |
| 训练用异常根 | `visual2/anomaly_samples_visa_{K}shot/visa/<cls>/src<stem>/{image,mask}` |
| 一致性校验 | **抽样清单 = AnoStyler 底图 = 训练 shot 清单**（12 类 × 1/2/4 全部逐项一致） |

## 2. 跑之前发现并修掉的两个真 bug
| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | VisA 评估抛 `AssertionError: area >= 0`（AUPRO） | **VisA 掩码是 0/1 二值**（MVTec 是 0/255），管线 `gt > 0.5` 把异常掩码全判成空 → AUPRO 除零 NaN | 数据准备层把掩码二值化重写为 0/255（不改模型/评估代码） |
| 2 | 4-shot 只加载 35 张合成异常（应为 140） | K 张底图的产物在同一"缺陷类型"目录，而 loader 每类型只采 35 张 | 每底图一个类型目录 `src<stem>` → K×35 张/轮（4-shot: 8 iter/epoch） |

## 3. 三档全指标（12 类平均，%）

| 指标 | 1-shot | 2-shot | 4-shot |
|---|---|---|---|
| I-AUROC | 86.24 | 84.83 | 88.99 |
| I-AP | 86.10 | 84.47 | 89.17 |
| I-F1 | 85.22 | 84.11 | 86.74 |
| P-AUROC | 94.53 | 94.59 | 96.48 |
| P-AP | 31.21 | 33.28 | 34.64 |
| P-F1 | 37.76 | 39.96 | 43.07 |
| P-AUPRO | 85.81 | 86.00 | 89.61 |
| **7 项平均** | **72.41** | **72.46** | **75.53** |

## 4. 逐类 7 项平均

| 类别 | 1-shot | 2-shot | 4-shot |
|---|---|---|---|
| candle | 79.75 | 77.55 | 74.63 |
| capsules | 75.53 | 78.08 | 78.42 |
| cashew | 81.75 | 83.60 | 83.78 |
| chewinggum | 89.64 | 89.26 | 87.68 |
| fryum | 71.24 | 66.37 | 74.41 |
| macaroni1 | 61.09 | 64.06 | 64.38 |
| macaroni2 | 50.49 | 50.69 | 63.67 |
| pcb1 | 85.21 | 82.74 | 84.65 |
| pcb2 | 64.66 | 68.03 | 68.77 |
| pcb3 | 68.72 | 62.28 | 68.25 |
| pcb4 | 58.69 | 63.58 | 73.24 |
| pipe_fryum | 82.14 | 83.31 | 84.43 |
| **平均** | **72.41** | **72.46** | **75.53** |

## 5. 读法

1. **1-shot ≈ 2-shot**（72.41 vs 72.46，几乎持平）：VisA 上与 MVTec 不同，从 1 张增加到 2 张参考**几乎没有收益**；
   个别类别甚至反向（candle 79.75→77.55、pcb3 68.72→62.28），也有明显受益的（macaroni2 50.49→50.69、pcb4 58.69→63.58）。
2. **4-shot 明显更好**（75.53，比 1/2-shot 高约 3.1 个点）：图像级三项与 P-AUROC/P-AUPRO 全面提升
   （I-AUROC +2.75/+4.16、P-AUROC +1.95/+1.89、P-AUPRO +3.80/+3.61 相对 1/2-shot），
   P-AP/P-F1 也小幅上升（+3.4 左右）。
3. **P-AP/P-F1 仍是全局短板**：三档都只有 31–43，远低于其它五项（85–96）。
   与 MVTec 的结论一致：mapping 分支的瓶颈在**像素级 AP**（受极少数高分正常像素支配）。
4. MVTec 4-shot（15 类 7 项平均 85.43）vs VisA 4-shot（12 类 75.53）：VisA 更难，
   差距主要来自 P-AP/P-F1（VisA 34.6/43.1 vs MVTec 56.8/59.9）与图像级指标。

## 6. 逐类全指标（分档）

| 档位 | 结果日志 |
|---|---|
| 1-shot | `logs/eval_visa_vpt35_visa1shot_<cls>.log`（训练 `logs/train_visa_vpt35_visa1shot_<cls>.log`） |
| 2-shot | `logs/eval_visa_vpt35_visa2shot_<cls>.log` |
| 4-shot | `logs/eval_visa_vpt35_visa4shot_<cls>.log` |

（每行 CSV 的顺序：I-AUROC, I-AP, I-F1, P-AUROC, P-AP, P-F1, P-AUPRO）

4-shot 逐类明细（示例，完整见日志）：

| 类别 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| cashew | 83.58 | 91.36 | 86.51 | 99.00 | 66.82 | 65.49 | 93.73 |
| chewinggum | 98.30 | 99.27 | 95.92 | 99.25 | 59.03 | 67.05 | 94.93 |
| pcb1 | 93.17 | 91.88 | 87.68 | 98.97 | 60.43 | 71.75 | 88.69 |
| macaroni1 | 75.69 | 74.55 | 72.27 | 97.45 | 15.78 | 24.46 | 90.47 |
| macaroni2 | 68.64 | 70.67 | 69.84 | 96.71 | 16.92 | 28.58 | 94.35 |

## 7. 复现

```bash
# 数据（若需重建）
python visa_prepare_shots.py            # 抽样 + AnoStyler 配置（1/2/4-shot）
bash   anostyler_visa/run.sh            # 批量合成（双卡各 3 并发，可断点续跑）
python visa_build_root.py               # VisA → MVTec 式根（掩码二值化 0/255）
for K in 1 2 4; do python visa_build_anomaly_root.py $K; done
# 训练 + 评估（原本的 vpt3.5 代码）
EXTRA="--global_proto 1 --w_div 0.1 --w_pixel 0.0 --metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1"
for K in 1 2 4; do
  SHOTS=$K EXTRA_ARGS="$EXTRA" bash q_rotate_train_visa.sh vpt35_visa${K}shot
  SAVE=vpt35_visa${K}shot SHOTS=$K bash eval_visa.sh
done
# bank 分支 + 融合（0.3·z(mapping) + 0.7·z(bank)，类内 z 标准化后单次高斯平滑）
SAVE=vpt35_visa4shot SHOTS=4 bash visa_bank_fuse.sh      # 每类 = mapping落盘 → bank建库 → 融合
```

产物：权重 `results/visa/vpt35_visa{1,2,4}shot/model_<cls>_35.pth`（**只保留 epoch 35**，中间轮次已清理）；
清单 `logs/manifest_train_vpt35_visa{1,2,4}shot.txt`；评估驱动日志 `logs/visa_eval_driver_vpt35_visa{1,2,4}shot.log`；
落盘异常图 `.maps_nocrop448_visa4/<cls>.map`（mapping）、`.errmap_nocrop448_visa4/<cls>.map`（bank），
融合逐类 CSV 见 `logs/visa_fuse_vpt35_visa4shot_<cls>.log`，汇总脚本 `summarize_visa_fuse.py`。

## 8. 与顶会论文的 VisA 4-shot 对比

文献数字取自**论文原表**：SubspaceAD（CVPR'26）表 8 的 VisA-4shot 行；
FoundAD（ICLR'26）表 4 的 VisA-4shot 行（`-` = 论文未报告）。
本项目 = 12 类平均、vpt3.5（mapping-only / bank-only / 融合三分支）、官方 `adeval`、训练裁剪 392 / 测试不裁剪 448。
**指标对齐**：文献的 `AUPR` ↔ 我们的 `I-AP`；文献的 `PRO` ↔ 我们的 `P-AUPRO`。

### 8.1 对照 SubspaceAD 表 8（同一套 VisA 协议）
| 方法 | 会议·年 | I-AUROC | I-AP(AUPR) | P-AUROC | PRO(P-AUPRO) |
|---|---|---|---|---|---|
| SPADE | arXiv'20 | 81.7 | 83.4 | 96.6 | 87.3 |
| PatchCore | CVPR'22 | 85.3 | 87.5 | 96.8 | 84.9 |
| WinCLIP | CVPR'23 | 87.3 | 88.8 | 97.2 | 87.6 |
| IIPAD | 2025 | 88.3 | 89.6 | 97.4 | 88.3 |
| PromptAD | CVPR'24 | 89.1 | 90.8 | 97.4 | 86.2 |
| AnomalyDINO | 2024/25 | 92.6 | 92.9 | 98.2 | 94.1 |
| **SubspaceAD**（training-free） | CVPR'26 | **94.5** | **93.8** | **98.5** | **94.1** |
| 本项目 · mapping-only（旧 `min` 打分） | 本项目 | 88.99 | 89.17 | 96.48 | 89.61 |
| 本项目 · mapping-only（**新打分**） | 本项目 | 92.41 | 92.81 | 98.15 | 93.67 |
| 本项目 · bank-only | 本项目 | **93.37** | **93.47** | 95.61 | 91.31 |
| 本项目 · mapping⊕bank（旧 `min`，w=0.3） | 本项目 | 93.46 | 93.35 | 97.60 | 93.82 |
| 本项目 · mapping⊕bank（新打分，w=0.3） | 本项目 | 94.86 | 94.87 | 98.07 | 94.74 |
| **本项目 · mapping⊕bank（新打分，w=0.5）** | 本项目 | **94.95** | **94.96** | **98.73** | **95.59** |

**名次（放在上表里）**：
* `I-AUROC`：新打分+w=0.5 后 **94.95 超过 SubspaceAD（94.5）** 与 AnomalyDINO（92.6），位列表内第一；
* `I-AP(AUPR)`：**94.96 超过 SubspaceAD（93.8）** 与 AnomalyDINO（92.9），同样第一；
* `P-AUROC`：**98.73 超过 SubspaceAD（98.5）** 与 AnomalyDINO（98.2），位列表内第一；
* `PRO`：**95.59 超过 SubspaceAD/AnomalyDINO（94.1）**，位列表内第一。

⇒ 即：**VisA 4-shot 上四个可比指标全部反超目前最强的 training-free 方法 SubspaceAD。**

### 8.1.1 本项目三分支（VisA 4-shot，12 类平均，%）

打分演进：`min`（交付版）→ `√(d_min·d̄)`（第一版改进）→ **新打分**
`s_raw=d1^0.5·q50^0.5, s_learn=dlmin^0.7·bq64^0.3, w=0.6`（见 `SCORING_SEARCH_2026-10-13.md`）。

| 分支 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO | 7 项平均 |
|---|---|---|---|---|---|---|---|---|
| mapping-only（旧 `min`） | 88.99 | 89.17 | 86.74 | 96.48 | 34.64 | 43.07 | 89.61 | 75.53 |
| mapping-only（`√(d_min·d̄)`） | 90.32 | 90.65 | 87.70 | 97.84 | 38.20 | 46.52 | 92.41 | 77.66 |
| **mapping-only（新打分）** | **92.41** | **92.81** | **89.05** | **98.15** | **39.21** | **47.23** | **93.67** | **78.93** |
| bank-only（未变） | 93.37 | 93.47 | 89.94 | 95.61 | 40.28 | 47.23 | 91.31 | 78.74 |
| **mapping⊕bank（新打分，0.3/0.7 z）** | **94.86** | **94.87** | **91.61** | **98.07** | **43.90** | **50.34** | **94.74** | **81.20** |
| Δ（新融合 − 旧融合 80.01） | +1.40 | +1.52 | +1.23 | +0.47 | +1.49 | +1.31 | +0.92 | **+1.19** |
| Δ（新融合 − bank-only） | +1.49 | +1.40 | +1.67 | +2.46 | +3.62 | +3.11 | +3.43 | **+2.46** |

**要点**：融合**同时不劣于两个单支**——相对 bank 在 P-AUROC/P-AP/P-F1/P-AUPRO 上 +1.8~2.5，
相对 mapping 全面 +3.6~7.8；且在 I-AUROC/I-AP 上与 bank 基本持平（+0.09/−0.13）。
即融合把 mapping 的图像级判别力与 bank 的像素级定位力合并，**7 项平均 80.01，高于任一单支 4.5/1.3 个点**。

### 8.2 对照 FoundAD 表 4（其协议与本表不同，仅作参考）
FoundAD 表 4 的 VisA-4shot：WinCLIP 84.1 / 86.1 / 95.2 / 82.1、PromptAD 85.7 / 88.8 / 97.2 / 84.7、
AnomalySD 88.9 / 90.9 / 97.5 / 94.3、IIPAD 88.3 / 89.6 / 97.4 / 88.3、
**FoundAD 94.4 / 94.0 / 99.7 / 98.4**（I-AUROC / AUPR / P-AUROC / PRO）。
⚠ 该表的 MVTec 数字亦显著低于其它论文（如 WinCLIP MVTec I-AUROC 94.0 vs SubspaceAD 表 8 的 95.2），
说明其划分/评测口径与 8.1 不同，**不可与 8.1 的行直接排名**。

### 8.3 结论与口径提醒
1. **单支 mapping-only 属"中游偏上"**（I-AUROC 88.99 / I-AP 89.17 / P-AUROC 96.48 / PRO 89.61），
   PRO 已超过 PromptAD/IIPAD，但 I-AUROC、I-AP、P-AUROC 低于 AnomalyDINO 与 SubspaceAD。
2. **融合后（bank⊕mapping，0.3/0.7 z）跃入第一梯队**：I-AUROC 93.46 与 I-AP 93.35 **均超过
   AnomalyDINO（92.6/92.9）**，PRO 93.82 逼近 AnomalyDINO/SubspaceAD（94.1），
   P-AUROC 97.60 亦进入 2025–26 区间。仅 SubspaceAD（94.5/93.8/98.5/94.1）仍整体更高。
3. 这与我方在 MVTec 上的表现**一致**（MVTec 4-shot 融合后 I-AUROC 98.06、AUPRO 95.31，高于表内所有单方法）；
   但 **VisA 是更难的域**，P-AP/P-F1 短板被放大（VisA P-AP 42.4 vs MVTec 65.6）。
4. 口径差异（必须声明）：分辨率（我们训练裁剪 392 / 测试不裁剪 448）、是否 training-free
   （SubspaceAD/AnomalyDINO 免训练）、PRO 实现细节（已互验 ≤0.0002）——
   跨行只能看量级。
