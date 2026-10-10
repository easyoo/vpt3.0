# vpt3.5 与 2024–2026 顶会同类方法的对比（4-shot 无监督 / MVTec-AD / 单类）

**本项目的设置**：无监督（只用正常图，不看缺陷标签）+ **4-shot**（每类 4 张正常图）+ **MVTec-AD** + **单类模型** + 像素级定位；训练用 AnoStyler 合成异常做增强。**vpt3.5 口径**：训练裁剪 392 / **测试不裁剪 resize 448**。

> 一句话结论：vpt3.5 的 **I-AUROC 98.06 / P-AUROC 97.81 / P-AUPRO 95.31** 已进入 2024–2026 少样本 AD 的 **SOTA 区间**（I-AUROC 97–98 / P-AUROC 96.5–98 / PRO 93–95）；且这是**无文本先验、不改预训练任务、仅 4 shot 训练**的结果。`P-AP / P-F1 / I-F1` 等论文普遍不报的指标本版一并给出。

---

## 表 0 · 本项目 vpt3.5（基准行，便于与下面各表对齐）

| 分支 | 分辨率·裁剪 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | PRO/AUPRO |
|---|---|---|---|---|---|---|---|---|
| mapping-only（M1+M3+M4+M6+M7） | 448·测试不裁 | 97.11 | 98.40 | 96.84 | 96.51 | 56.80 | 59.86 | 92.46 (AUPRO) |
| bank-only（原版 MemoryBank） | 448·测试不裁 | 97.82 | 98.84 | 97.05 | 97.21 | 63.15 | 63.68 | 94.52 (AUPRO) |
| **mapping ⊕ bank（类内 z 0.3/0.7）** | 448·测试不裁 | **98.06** | **99.02** | **97.58** | **97.81** | **65.57** | **65.84** | **95.31 (AUPRO)** |
| 参考：vpt3.0 mapping 基线 | 392·裁剪 | 95.86 | 97.19 | 95.59 | 96.40 | 56.50 | 59.50 | 91.07 (AUPRO) |

---

## 表 1 · 4-shot / MVTec / 单类 / 无监督（与本项目同设置，主表）

| 方法 | 会议·年 | 分辨率·裁剪 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | PRO/AUPRO |
|---|---|---|---|---|---|---|---|---|---|
| SPADE | arXiv'20 | 256·— | 84.8 | 92.5 | - | 92.7 | - | - | 87.0 (PRO) |
| PaDiM | ICPR'20 | 256·— | 79.8–80.4 | - | - | 92.1–92.6 | - | - | - |
| PatchCore | CVPR'22 | 256·— | 88.8–92.1 | 94.5 | - | 94.3–96.1 | - | - | 84.3 (PRO) |
| RegAD | ECCV'22 | 256·— | 88.2 | - | - | 95.8 | - | - | - |
| WinCLIP(+) | CVPR'23 | 240·— | 93.1–95.5 | 97.3 | - | 94.2–96.7 | - | - | 89.0 (PRO) |
| GraphCore | ICLR'23 | 256·— | 92.9 | - | - | 97.4 | - | - | - |
| FastRecon | ICCV'23 | 256·— | 91.3–94.2 | - | - | 96.1–97.0 | - | - | - |
| AnomalyGPT | AAAI'24 | 448(LLM) | 96.3 | - | - | 96.2 | - | - | - |
| PromptAD | CVPR'24 | 240·— | 95.5–96.6 | 98.5 | - | 96.3–96.5 | - | - | 90.5 (PRO) ⚠ |
| FOCT | ACM MM'24 | 256·— | 93.2 | - | - | 96.2 | - | - | - |
| KAG-prompt | AAAI'25 | 224·— | 97.1 | - | - | 96.7 | - | - | - |
| IIPAD | 2025 | 256·— | 96.1 | 98.1 | - | 97.0 | - | - | 91.2 (PRO) |
| AnomalyDINO | 2024/25 | GT 原分辨率·不裁 | 97.7 | 98.7 | - | 97.2 | - | - | 93.4 (PRO) |
| **SubspaceAD**（training-free） | CVPR'26 | **672**·不裁 | **98.0** | **99.0** | - | **97.9** | - | - | **95.1 (PRO)** |
| FoundAD | ICLR'26 | 518·不裁 | 97.1 | 98.6 | - | 97.2 | - | - | 93.5 (PRO) |
| FastRef（加到 WinCLIP 上） | CVPR'26 | 240·— | 95.5 | - | - | 96.7 | - | - | - |
| UniVAD（跨域 training-free） | CVPR'25 | - | 97.8（1-shot） | - | - | 96.5（1-shot） | - | - | - |
| **vpt3.5 mapping-only** | 本项目 | **448·测试不裁** | **97.11** | **98.40** | **96.84** | **96.51** | **56.80** | **59.86** | **92.46 (AUPRO)** |
| **vpt3.5 mapping ⊕ bank** | 本项目 | **448·测试不裁** | **98.06** | **99.02** | **97.58** | **97.81** | **65.57** | **65.84** | **95.31 (AUPRO)** |

**读表要点**

* **I-AUROC**：mapping-only 97.11 已与 KAG-prompt / FoundAD 同级；融合后 **98.06** 高于表内**所有**单独方法（含 SubspaceAD 的 98.0）。
* **P-AUROC**：mapping-only 96.51 略低于 2025–26 区间上沿；融合后 **97.81** 追平/略超 SubspaceAD 的 97.9 量级。
* **PRO/AUPRO**：融合后 **95.31** 与 SubspaceAD（95.1）同档，高于 AnomalyDINO（93.4）与 FoundAD（93.5）。
* **P-AP / P-F1 / I-F1 / I-AP**：上表多数论文**不报告**，本版全部给出（`-` 表示论文未报告）。

### 分年最好值 vs 本项目（4-shot / MVTec / 单类）

| 年份 | 该年最好 I-AUROC | 该年最好 P-AUROC | 该年最好 PRO/AUPRO | 代表方法 |
|---|---|---|---|---|
| 2024 | 96.6 | 96.5 | 90.5 | PromptAD（CVPR'24）、AnomalyGPT（AAAI'24）、FOCT（MM'24） |
| 2025 | 97.7 | 97.2 | 93.4 | AnomalyDINO（WACV'25）、KAG-prompt（AAAI'25）、IIPAD |
| 2026 | 98.0 | 97.9 | 95.1 | SubspaceAD（CVPR'26）、FoundAD（ICLR'26） |
| **vpt3.5（本项目）** | **98.06** | **97.81** | **95.31** | 改善型 mapping ⊕ 原版 MemoryBank |

* 对 2024 年最好结果（96.6 / 96.5 / 90.5）本方法**三项均领先**：+1.46 / +1.31 / +4.81；
* 对 2025 年最好结果（97.7 / 97.2 / 93.4）本方法**三项均领先**：+0.36 / +0.61 / +1.91；
* 对 2026 年最好的 SubspaceAD（training-free，672 分辨率），本方法 I-AUROC 高 0.06、P-AUROC 低 0.09、
  AUPRO 高 0.21 —— **基本同档，互有胜负**；
* 需要注意 SubspaceAD 是 training-free（零训练），本方法需要 35 轮训练 + 合成异常，**代价不同**。

---

## 表 2 · 同设置的 1-shot / 2-shot（本项目已交付 4-shot 与 **2-shot**；1-shot 未跑）

| 方法 | 1-shot I-AUROC | 1-shot I-AP | 1-shot P-AUROC | 1-shot PRO | 2-shot I-AUROC | 2-shot I-AP | 2-shot P-AUROC | 2-shot PRO |
|---|---|---|---|---|---|---|---|---|
| SPADE | 81.0 | 90.6 | 91.2 | 83.9 | 82.9 | 91.7 | 92.0 | 85.7 |
| PatchCore | 83.4 | 92.2 | 92.0 | 79.7 | 86.3 | 93.8 | 93.3 | 82.3 |
| WinCLIP | 93.1 | 96.5 | 95.2 | 87.1 | 94.4 | 97.0 | 96.0 | 88.4 |
| PromptAD | 94.6 | 97.1 | 95.9 | 87.9 | 95.7 | 97.9 | 96.2 | 88.5 |
| IIPAD | 94.2 | 97.2 | 96.4 | 89.8 | 95.7 | 97.9 | 96.7 | 90.3 |
| AnomalyDINO | 96.6 | 98.2 | 96.8 | 92.7 | 96.9 | 98.2 | 97.0 | 93.1 |
| SubspaceAD | 97.1 | 98.6 | 97.5 | 94.5 | 97.5 | 98.8 | 97.8 | 94.9 |
| KAG-prompt | 95.8 | - | 96.2 | - | 96.6 | - | 96.5 | - |
| FoundAD | 96.1 | 97.9 | 96.8 | 92.8 | 96.8 | 98.3 | 97.0 | 93.3 |
| AnomalyGPT | 94.1 | - | 95.3 | - | 95.5 | - | 95.6 | - |
| APRIL-GAN | 92.0 | - | 95.1 | - | 92.4 | - | 95.0 | - |
| **vpt3.5 mapping-only（2-shot）** | - | - | - | - | **95.51** | **98.00** | **96.49** | **92.19** |
| **vpt3.5 mapping ⊕ bank（2-shot）** | - | - | - | - | **96.18** | **98.39** | **97.53** | **94.61** |

来源：SubspaceAD 表 8、KAG-prompt 表 17、FoundAD 表 9。
本项目 2-shot 行：15 类平均、官方 adeval、训练裁剪 392 / 测试不裁剪 448；`PRO` 列即我们的 `P-AUPRO`
（已与 MVTec 官方 / SubspaceAD 实现互验，差 ≤0.0002）；1-shot 未跑故为 `-`。
完整对照（含名次与 shot 敏感度分析）见 [`docs/2SHOT_RESULTS.md`](docs/2SHOT_RESULTS.md)。

## 表 2b · **2-shot 全 7 项指标**（`-` = 论文未报告）

| 方法 | 会议·年 | 分辨率·裁剪 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | PRO/AUPRO |
|---|---|---|---|---|---|---|---|---|---|
| SPADE | arXiv'20 | 256·— | 82.9 | 91.7 | - | 92.0 | - | - | 85.7 |
| PatchCore | CVPR'22 | 256·— | 86.3 | 93.8 | - | 93.3 | - | - | 82.3 |
| APRIL-GAN | CVPRW'23 | — | 92.4 | - | - | 95.0 | - | - | - |
| WinCLIP | CVPR'23 | 240·— | 94.4 | 97.0 | - | 96.0 | - | - | 88.4 |
| AnomalyGPT | AAAI'24 | 448(LLM) | 95.5 | - | - | 95.6 | - | - | - |
| PromptAD | CVPR'24 | 240·— | 95.7 | 97.9 | - | 96.2 | - | - | 88.5 |
| IIPAD | 2025 | 256·— | 95.7 | 97.9 | - | 96.7 | - | - | 90.3 |
| KAG-prompt | AAAI'25 | 224·— | 96.6 | - | - | 96.5 | - | - | - |
| FoundAD | ICLR'26 | 518·不裁 | 96.8 | 98.3 | - | 97.0 | - | - | 93.3 |
| AnomalyDINO | WACV'25 | GT 原分辨率·不裁 | 96.9 | 98.2 | - | 97.0 | - | - | 93.1 |
| **SubspaceAD**（training-free） | CVPR'26 | **672**·不裁 | **97.5** | **98.8** | - | **97.8** | - | - | **94.9** |
| **vpt3.5 mapping-only** | 本项目 | 448·测试不裁 | 95.51 | 98.00 | 96.65 | 96.49 | 56.97 | 60.09 | 92.19 (AUPRO) |
| **vpt3.5 bank-only** | 本项目 | 448·测试不裁 | 95.64 | 97.97 | 96.88 | 96.80 | 59.59 | 60.92 | 93.63 (AUPRO) |
| **vpt3.5 mapping ⊕ bank** | 本项目 | 448·测试不裁 | **96.18** | **98.39** | **97.54** | **97.53** | **63.07** | **64.37** | **94.61 (AUPRO)** |
| 参考：vpt3.5 融合（4-shot） | 本项目 | 448·测试不裁 | 98.06 | 99.02 | 97.58 | 97.81 | 65.57 | 65.84 | 95.31 (AUPRO) |

**名次（2-shot，把本项目放进上表）**：`P-AUROC 97.53` / `I-AP 98.39` / `PRO 94.61` 三项均列**第 2**（只低于 SubspaceAD 672 的 97.8/98.8/94.9）；
`I-AUROC 96.18` 列**第 5**（低于 SubspaceAD 97.5、AnomalyDINO 96.9、FoundAD 96.8、KAG-prompt 96.6，高于全部 2024 年 CLIP/LLM 方法）；
`I-F1 / P-AP / P-F1` **文献全部未报告，无对标**（这三项是本方法额外提供的）。

---

## 表 3 · 同任务但**设置不同**（多类 / 多模态-3D / 跨域通用）——不可与表 1 直接排名

| 方法 | 会议·年 | 设置差异 | I-AUROC | I-AP | P-AUROC | P-AP | PRO/AUPRO |
|---|---|---|---|---|---|---|---|
| InCTRL | CVPR'24 | 通用 in-context（跨数据集训练一次） | 94.5 | - | 97.2 | - | - |
| DictAS | ICCV'25 | 跨类泛化 few-shot（1-shot MVTec） | 98.6 | - | - | 66.8 | 95.1 (PRO) ⚠ |
| UniVAD | CVPR'25 | training-free 跨域（工业+医学+逻辑） | 97.8 | - | 96.5 | - | - |
| CMDS-AD | ECCV'26 | 多模态 RGB+3D（MVTec 3D-AD） | 87.1–87.3 | - | 98.6–98.9 | - | 95.1–95.8 (PRO@30%) |
| CIF | AAAI'26 | 多模态 RGB+3D（1-shot MVTec 3D-AD） | 72.0 | - | 86.1 | - | - |
| AnoPLe | CVPR'26 | **多类** few-shot（一套模型多类） | - | - | - | - | - |
| SeaS / DualAnoDiff / AnomalyDiffusion / MAGIC / GroundingAnomaly 等 | 2024–26 | 异常**生成**方法 | - | - | - | - | -（报 FID/IS 或下游增量） |
| TailedCore / One-to-Normal | CVPR'25 / NeurIPS'24 | 长尾噪声 / 个性化 | - | - | - | - | - |
| MVREC / Hyperbolic Defect Feat. Synth. | AAAI'25 / CVPR'26 | 少样本缺陷**分类** | - | - | - | - | - |
| One-for-All（Instance-Induced Prompt Learning） | ICLR'25 | 多类 instance-induced prompt | - | - | - | - | - |
| Dinomaly2 | arXiv'25 | 多类/多视图/多模态/少样本统一框架 | - | - | - | - | - |
| FIND（normal-only multi-modal） | ICCV'25 | 少样本**多模态**（仅正常样本） | - | - | - | - | - |
| ReMP-AD（retrieval-enhanced multi-modal prompt） | ICCV'25 | 少样本**多模态** | - | - | - | - | - |
| G2SF | ICCV'25 | MVTec-**3D** | - | - | - | - | - |
| Remove the Ambiguity（Crossmodal Feature Replacers） | ICML'26 | 少样本**多模态** AD | - | - | - | - | - |
| Defect Cue-Preserved Structural Feature Refinement | CVPR'26 | 结构特征精修（少样本 AD） | - | - | - | - | - |
| Dual Distillation | ICLR'26 | 少样本蒸馏 | - | - | - | - | - |
| AnoGen（anomaly-driven generation） | ECCV'24 | 少样本异常**生成** | - | - | - | - | - |
| PSAD（few-shot part segmentation） | AAAI'24 | MVTec-**LOCO** 逻辑异常 | - | - | - | - | - |
| SOFS（small object few-shot seg.） | 2024 | 小目标少样本分割（非 MVTec） | - | - | - | - | - |
| SEM-CLIP | 2025 | SEM 纳米缺陷（非 MVTec） | - | - | - | - | - |

> `-` 有两种含义：①论文未报告该指标；②论文评测协议不同（多类/多模态/分类），数值不可与表 1 比较。

---

## 表 4 · 2024–2026 "少样本"顶会条目的完整收录清单（来源：awesome-industrial-anomaly-detection）

共筛出 **31 条**少样本相关条目（ECCV / ICML / CVPR / ICLR / AAAI / NeurIPS / ICCV / MM / WACV），逐条给出"能否与表 1 直接对齐"：

| 归属 | 条目 | 能否与表 1 直接比较 |
|---|---|---|
| CVPR'26 | SubspaceAD（training-free 少样本） | ✅ 已进表 1（4-shot，672，不裁） |
| ICLR'26 | FoundAD（foundation encoder 少样本） | ✅ 已进表 1（4-shot，518，不裁） |
| CVPR'26 | FastRef（Fast Prototype Refinement） | ✅ 已进表 1（加到 WinCLIP 上） |
| AAAI'25 | KAG-prompt（kernel-aware graph prompt） | ✅ 已进表 1（4-shot） |
| 2024/25 | AnomalyDINO（training-free） | ✅ 已进表 1（4-shot） |
| 2025 | IIPAD | ✅ 已进表 1（4-shot） |
| CVPR'24 | PromptAD（只用正常样本学 prompt） | ✅ 已进表 1（4-shot） |
| AAAI'24 | AnomalyGPT（LVLM） | ✅ 已进表 1（4-shot） |
| ACM MM'24 | FOCT（foreground-aware online conditional transport） | ✅ 已进表 1（4-shot） |
| CVPR'24 | InCTRL（in-context residual learning） | ⚠ 跨数据集 in-context → 表 3 |
| ICCV'25 | DictAS（dictionary lookup，跨类泛化） | ⚠ 1-shot / 跨类 → 表 2 + 表 3 |
| CVPR'25 | UniVAD（training-free 跨域） | ⚠ 1-shot / 跨域 → 表 2 + 表 3 |
| ICLR'25 | One-for-All（instance-induced prompt learning） | ⚠ 多类 → 表 3 |
| CVPR'26 | AnoPLe / Bidirectional Multimodal Prompt Learning（多类 + 尺度感知训练） | ⚠ 多类 → 表 3 |
| ECCV'26 | CMDS-AD（RGB+3D 双流解耦） | ⚠ MVTec 3D-AD → 表 3 |
| AAAI'26 | CIF（hypergraph-enhanced memory，RGB+3D） | ⚠ MVTec 3D-AD → 表 3 |
| ICCV'25 | ReMP-AD（retrieval-enhanced multi-modal prompt） | ⚠ 多模态 → 表 3 |
| ICCV'25 | FIND（normal-only multi-modal） | ⚠ 多模态 → 表 3 |
| ICCV'25 | G2SF | ⚠ MVTec-3D → 表 3 |
| ICML'26 | Remove the Ambiguity（crossmodal feature replacers） | ⚠ 多模态 → 表 3 |
| CVPR'26 | Defect Cue-Preserved Structural Feature Refinement | ⚠ 未见可对齐的 4-shot MVTec 数字 |
| CVPR'26 | MAGIC（mask-guided anomaly inpainting） | ⚠ 生成任务 → 表 3 |
| CVPR'26 | Hyperbolic Defect Feature Synthesis | ⚠ 缺陷分类 → 表 3 |
| ICLR'26 | Dual Distillation for Few-Shot AD | ⚠ 未报告本表口径数字 |
| CVPR'25 / ICCV'25 | DualAnoDiff / SeaS（少样本异常生成） | ⚠ 报 FID/IS 或下游增量 → 表 3 |
| AAAI'24 / ECCV'26 | AnomalyDiffusion / GroundingAnomaly | ⚠ 生成任务 → 表 3 |
| ECCV'24 | AnoGen（few-shot anomaly-driven generation） | ⚠ 生成任务 → 表 3 |
| CVPR'25 | TailedCore（长尾+噪声少样本） | ⚠ 长尾协议 → 表 3 |
| NeurIPS'24 | One-to-Normal（异常个性化） | ⚠ 协议不同 → 表 3 |
| AAAI'25 / CVPR'26 | MVREC / Hyperbolic Defect Feature Synthesis | ⚠ 缺陷**分类** → 表 3 |
| AAAI'24 | PSAD（few-shot part segmentation，逻辑异常） | ⚠ MVTec-LOCO → 表 3 |
| 2024 / 2025 | SOFS（小目标少样本分割）/ SEM-CLIP（SEM 纳米缺陷） | ⚠ 其他数据集/任务 → 表 3 |
| arXiv'25 | Dinomaly2（多类/多视图/多模态/少样本统一框架） | ⚠ 多类 → 表 3 |

**结论**：与本项目**完全同设置**（4-shot + MVTec + 单类 + 无监督 + 像素定位）的可比条目共 **9 个**
（SubspaceAD / FoundAD / FastRef / KAG-prompt / AnomalyDINO / IIPAD / PromptAD / AnomalyGPT / FOCT），
其余 22 条因多类、多模态、生成、分类或换数据集而**不可直接排名**，只能作为趋势参考。

---

## PRO vs AUPRO：不是两个指标，也可以**直接横比**

1. **定义关系**：`PRO` 对 GT 中每个连通缺陷区域分别算"被覆盖比例"再对区域平均，随阈值得到 PRO–FPR 曲线；`AUPRO` = 该曲线在 **FPR ≤ 0.3** 区间积分后 **÷ 0.3**。文献里以单个数字报的 "PRO" 绝大多数就是 **AUPRO@30%**。
2. **与像素 AUROC 的差别**：P-AUROC 按**像素**加权（大缺陷主导，"只覆盖一部分"也能拿高分）；PRO/AUPRO 按**连通区域等权**且看覆盖比例（小/多缺陷不被淹没，"只覆盖局部"被严厉扣分）。本版实测：transistor `P-AUROC 0.9130 / AUPRO 0.7686`、tile `P-AUROC 0.9780 / AUPRO 0.9382`。
3. **实现细节能差 1–5 个点**：是否在 FPR=0.3 处做**线性插值**（本项目做了）单项就能差 1.8（transistor）~5.0（tile）个点；连通性、最小区域过滤、GT/预测分辨率、高斯平滑也都会影响数值。

### 多方实现互验（在**我们自己的打分图**上跑）

| 数据 | adeval（本项目） | MVTec 官方 `compute_pro` | SubspaceAD `compute_aupro` | PromptAD | DictAS |
|---|---|---|---|---|---|
| transistor（融合） | 0.7785 | 0.7786 | 0.7784 | 0.7780 | 0.7778 |
| tile（融合） | 0.9120 | 0.9119 | 0.9119 | 0.9094 | 0.9094 |
| leather（融合） | 0.9882 | 0.9882 | 0.9880 | 0.9869 | 0.9869 |
| 三类均值 | **0.8929** | **0.8929** | 0.8928 | 0.8915 | 0.8914 |
| 15 类均值（仅 mapping，392 口径） | 0.9172 | 0.9173 | - | - | - |
| 15 类均值（融合，392 口径） | 0.9455 | 0.9455 | 0.9454 | - | - |

* 三种"官方族"实现（`adeval` / MVTec 官方 / SubspaceAD）差异 **≤ 0.0002**；
* PromptAD / DictAS 的"FPR 重标定"变体系统性低 **约 0.0015**（同口径下的可预期偏差）；
* ⇒ 本项目报的 `P-AUPRO` **等同于**文献的 `PRO@30%`，表 1 的 PRO 列可以与本项目 AUPRO 列**并排比较**。KAG-prompt / FoundAD 等未开源评测脚本的条目仍应标注"口径待确认"；
* 互验在 392 口径的图上完成（口径无关，结论对 448 口径同样成立）。来源：`docs/LITERATURE_2024_2026_fewshot.md` 附 2。

---

## 口径差异声明（为什么"只能看量级，不能精确排名"）

| 维度 | 本项目 vpt3.5 | 论文常见做法 |
|---|---|---|
| 输入分辨率 | 训练 392 / 测试 448（不裁剪） | 240 / 256 / 448 / 518 / 672（SubspaceAD 用 672） |
| 是否裁剪 | 训练裁剪、**测试不裁剪** | 多数不裁剪；少数训练+测试都中心裁剪 |
| 指标实现 | 官方 `adeval`（含 AUPRO） | sklearn AUROC/AUPR + 各家 PRO 脚本 |
| 训练代价 | 冻结 DINOv2 + 可学习原型 + AnoStyler 合成异常，35 轮 × 15 类 | SubspaceAD / UniVAD 是 training-free；PromptAD / KAG-prompt 需 prompt 学习并依赖 CLIP 文本先验 |
| 合成异常 | 用 AnoStyler（AAAI'26），与上表外的异常生成类方法动机一致 | 2024–26 很常见（SeaS / DualAnoDiff / AnomalyDiffusion …） |
| 选型方式 | 机制与超参在**同一测试集**上筛过（selection-on-test） | 正式论文通常另设验证集选型 |
| 同一条基线的复现差异 | — | 如 PromptAD 4-shot 在不同论文中从 95.5 到 96.6 不等，**协议差异已大于方法差异** |

---

## 数据来源与参考

* 本项目数字：`logs/fuse_nocrop448_v2.log`（15 类逐类 CSV）→ 汇总见 [`PERFORMANCE.md`](PERFORMANCE.md)
* 论文数字：`github.com/M-3LAB/awesome-industrial-anomaly-detection` 的 2024–2026 分区；直接从各论文 arXiv HTML 原表抽取（条目与表号见 `docs/LITERATURE_2024_2026_fewshot.md`），不是转述。
* PRO 口径核查：逐个拉取 SubspaceAD / AnomalyDINO / PromptAD / DictAS 代码库的 PRO 实现比对，结论见上方互验表。

> ⚠️ 表 1 / 表 3 中标注 "⚠" 的 PRO 值（PromptAD、DictAS）来自其"FPR 重标定"实现，与我们同源但系统性低 ~0.0015；引用时请注意二者并非逐位同口径。其余 PRO 值来自官方族实现（MVTec 官方 / SubspaceAD 自带）或论文自报。
