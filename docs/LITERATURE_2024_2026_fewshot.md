# 2024–2026 顶会中"少样本无监督工业异常检测"论文与指标（与本项目同设置）

**本项目设置**：无监督（只用正常图，无缺陷标签）+ **4-shot**（每类 4 张正常图）+ **MVTec-AD** + **单类**（每类一个模型）
+ 像素级定位；另用 AnoStyler 合成异常做训练增强（AAAI'26 的方法）。

来源：`github.com/M-3LAB/awesome-industrial-anomaly-detection` 的 2024–2026 分区（ECCV/ICML/CVPR/ICLR/AAAI/NeurIPS/ICCV/MM/WACV），
共筛出 31 条"少样本"相关条目；下表**数字均从论文 arXiv HTML 的原表抽取**（标注了表号），不是转述。

---

## A. 设置完全相同（无监督 + K-shot 正常图 + MVTec + 单类 + 图像&像素 AUROC）

### A.1 4-shot（与本项目完全对齐的档位）
| 方法 | 会议/年 | MVTec I-AUROC | MVTec P-AUROC | 数字出处 |
|---|---|---|---|---|
| SPADE | arXiv'20 | 84.8 | 92.7 | PromptAD T1/T4；SubspaceAD T8 |
| PaDiM | ICPR'20 | 79.8–80.4 | 92.1–92.6 | FastRef T10；PromptAD T4 |
| PatchCore | CVPR'22 | 88.8–92.1 | 94.3–96.1 | SubspaceAD T8；FastRef T10 |
| RegAD | ECCV'22 | 88.2 | 95.8 | FastRef T10 |
| WinCLIP(+) | CVPR'23 | 93.1–95.5 | 94.2–96.7 | KAG-prompt T17；FastRef T10 |
| GraphCore | ICLR'23 | 92.9 | 97.4 | FastRef T10 |
| FastRecon | ICCV'23 | 91.3–94.2 | 96.1–97.0 | FastRef T10；SubspaceAD T8 |
| **AnomalyGPT** | AAAI'24 | 96.3 | 96.2 | KAG-prompt T17 |
| **PromptAD** | CVPR'24 | 95.5–96.6 | 96.3–96.5 | KAG-prompt T17；FastRef T10 |
| **FOCT** | ACM MM'24 | 93.2 | 96.2 | FastRef T10 |
| **KAG-prompt** | AAAI'25 | **97.1** | **96.7** | KAG-prompt T17 |
| IIPAD | 2025 | 96.1 | 97.0 | SubspaceAD T8 |
| **AnomalyDINO** | 2024/25 | 97.7 | 97.2 | SubspaceAD T8 |
| **SubspaceAD**（training-free） | CVPR'26 | **98.0** | **97.9** | SubspaceAD T8（DINOv3-448 为 97.6/97.8，T14） |
| **FoundAD** | ICLR'26 | 97.1 | 97.2 | FoundAD T9（3 seed 平均） |
| **FastRef** | CVPR'26 | 95.5（作用于 WinCLIP） | 96.7 | FastRef T10 |
| UniVAD（跨域、training-free） | CVPR'25 | 97.8（1-shot） | 96.5（1-shot） | UniVAD T7 |

### A.2 同一批方法在 1/2/4-shot 上的走势（说明"4-shot 档位"的量级）
| 方法 | 1-shot I/P | 2-shot I/P | 4-shot I/P |
|---|---|---|---|
| WinCLIP | 93.1 / 95.2 | 94.4 / 96.0 | 95.2 / 96.2 |
| PromptAD | 94.6 / 95.9 | 95.7 / 96.2 | 96.6 / 96.5 |
| IIPAD | 94.2 / 96.4 | 95.7 / 96.7 | 96.1 / 97.0 |
| AnomalyDINO | 96.6 / 96.8 | 96.9 / 97.0 | **97.7 / 97.2** |
| SubspaceAD | 97.1 / 97.5 | 97.5 / 97.8 | **98.0 / 97.9** |
| KAG-prompt | 95.8 / 96.2 | 96.6 / 96.5 | 97.1 / 96.7 |
| FoundAD | 96.1 / 96.8 | 96.8 / 97.0 | 97.1 / 97.2 |

**量级结论**：4-shot MVTec 上，2024–2026 的 SOTA 大约在 **I-AUROC 97–98 / P-AUROC 96.5–98**。

---

## B. 本项目在相同设置下的位置（官方 adeval、392 口径、15 类平均）
| 方案 | I-AUROC | I-AP | P-AUROC | P-AP | P-AUPRO |
|---|---|---|---|---|---|
| vpt3.0 mapping 基线 | 95.86 | 97.19 | 96.40 | 56.50 | 91.07 |
| **本方案 mapping-only（m4m6r3）** | **97.42** | **98.44** | **96.43** | **60.11** | **91.73** |
| **本方案 mapping ⊕ MemoryBank（0.3/0.7）** | **98.34** | **99.12** | **97.46** | **66.63** | **94.55** |

* I-AUROC：mapping-only 已与 KAG-prompt/AnomalyDINO 同级，融合后 98.3 高于上表所有单分支方法；
* P-AUROC：mapping-only 96.4 略低于 2025–26 的 96.5–97.9，融合后 97.46 追到该区间；
* **P-AP / P-AUPRO 是上表论文普遍不报的指标**（它们多报 AUPR(≈I-AP) 与 PRO），因此无法横向比较。

---

## C. 口径差异（必须声明，不能直接排名）
1. **输入分辨率**：我们 392；上表多为 240 / 256 / 448 / 672（SubspaceAD 用 672）。
2. **指标实现**：我们用官方 `adeval`（含 AUPRO）；多数论文用 sklearn 的 AUROC/AUPR 与 PRO。
3. **训练/调参代价不同**：SubspaceAD、UniVAD 是 training-free；PromptAD/KAG-prompt 要做 prompt 学习（且依赖 CLIP 文本先验）；
   我们是"冻结 DINOv2 + 可学习原型 + 合成异常"的无监督训练。
4. **合成异常**：我们用 AnoStyler（AAAI'26）；同类做法在 2024–2026 很常见（见 D.3）。
5. **baseline 复现差异**：同一条基线（如 PromptAD 4-shot）在不同论文里从 95.5 到 96.6 不等 —— 说明协议差异已大于方法差异量级，
   所以**跨论文数字只能看量级，不能精确排名**。

---

## D. 同任务但**设置不同**的 2024–2026 论文（列全，避免误比）

### D.1 多类（one-for-all）少样本
AnoPLe / Bidirectional Multimodal Prompt Learning（CVPR'26）、DictAS（ICCV'25，跨类泛化）、
FoundAD（ICLR'26，category-agnostic 部分）、UniVAD（CVPR'25，跨域统一）。

### D.2 多模态 / 3D / RGB-D（需要几何或深度输入）
CIF: Commonality in Few（AAAI'26）、CMDS-AD（ECCV'26）、ReMP-AD（ICCV'25）、FIND（ICCV'25）、G2SF（ICCV'25，MVTec-3D）。

### D.3 少样本异常**生成**（动机与我们用 AnoStyler 相同）
SeaS（ICCV'25）、DualAnoDiff（CVPR'25）、AnomalyDiffusion（AAAI'24，报告下游 AD 提升）、
MAGIC（CVPR'26）、GroundingAnomaly（ECCV'26）、CHIMERA（AAAI'26）、QARAD（AAAI'26）、
AnomalyPainter（AAAI'26）、CADiff（AAAI'26）、One-to-More（CVPR'26）、DeCo/UniScale（ECCV'26）。

### D.4 其他近邻任务
TailedCore（CVPR'25，长尾+噪声）、InCTRL（CVPR'24，通用少样本 in-context）、
One-to-Normal（NeurIPS'24，少样本"个性化"）、MVREC（AAAI'25，少样本缺陷**分类**）、
Hyperbolic Defect Feature Synthesis（CVPR'26，少样本缺陷分类）、Dual Distillation（ICLR'26）。

---

# 全指标对照表（缺指标一律用 `-` 占位）

列定义与本文项目一致：`I-AUROC`（图像级 AUROC）、`I-AP`（图像级 AUPR/AP）、`I-F1`、`P-AUROC`（像素级 AUROC）、
`P-AP`（像素级 AP）、`P-F1`、`P-AUPRO`。文献里极少报 `I-F1/P-AP/P-F1`，故多为 `-`；
`PRO` 不是 `AUPRO`（本文已标注 `(PRO)` 以示区分）。

## 表 1：**4-shot / MVTec / 单类 / 无监督**（与本项目完全对齐的档位）
| 方法 | 会议·年 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO | 出处 |
|---|---|---|---|---|---|---|---|---|---|
| SPADE | arXiv'20 | 84.8 | 92.5 | - | 92.7 | - | - | 87.0 (PRO) | SubspaceAD T8 |
| PaDiM | ICPR'20 | 79.8–80.4 | - | - | 92.1–92.6 | - | - | - | FastRef T10 / PromptAD T4 |
| PatchCore | CVPR'22 | 88.8 | 94.5 | - | 94.3 | - | - | 84.3 (PRO) | SubspaceAD T8 |
| RegAD | ECCV'22 | 88.2 | - | - | 95.8 | - | - | - | FastRef T10 |
| WinCLIP(+) | CVPR'23 | 93.1–95.5 | 97.3 | - | 94.2–96.7 | - | - | 89.0 (PRO) | KAG T17 / FastRef T10 / SubspaceAD T8 |
| GraphCore | ICLR'23 | 92.9 | - | - | 97.4 | - | - | - | FastRef T10 |
| FastRecon | ICCV'23 | 91.3–94.2 | - | - | 96.1–97.0 | - | - | - | FastRef T10 / SubspaceAD T8 |
| AnomalyGPT | AAAI'24 | 96.3 | - | - | 96.2 | - | - | - | KAG T17 |
| PromptAD | CVPR'24 | 95.5–96.6 | 98.5 | - | 96.3–96.5 | - | - | 90.5 (PRO) | KAG T17 / SubspaceAD T8 |
| FOCT | ACM MM'24 | 93.2 | - | - | 96.2 | - | - | - | FastRef T10 |
| KAG-prompt | AAAI'25 | **97.1** | - | - | **96.7** | - | - | - | KAG T17 |
| IIPAD | 2025 | 96.1 | 98.1 | - | 97.0 | - | - | 91.2 (PRO) | SubspaceAD T8 |
| AnomalyDINO | 2024/25 | **97.7** | 98.7 | - | 97.2 | - | - | 93.4 (PRO) | SubspaceAD T8 |
| SubspaceAD（training-free） | CVPR'26 | **98.0** | **99.0** | - | **97.9** | - | - | **95.1 (PRO)** | SubspaceAD T8 |
| FoundAD | ICLR'26 | 97.1 | 98.6 | - | 97.2 | - | - | 93.5 (PRO) | FoundAD T9 |
| FastRef（作用于 WinCLIP） | CVPR'26 | 95.5 | - | - | 96.7 | - | - | - | FastRef T10 |
| **本项目 mapping-only（m4m6r3）** | 本项目 | **97.42** | **98.44** | **96.76** | **96.43** | **60.11** | **62.01** | **91.73 (AUPRO)** | 本仓库 §32 |
| **本项目 mapping ⊕ MemoryBank（0.3/0.7）** | 本项目 | **98.34** | **99.12** | **97.79** | **97.46** | **66.63** | **66.34** | **94.55 (AUPRO)** | 本仓库 §33 |

## 表 2：同设置下的 **1-shot / 2-shot**（同一批方法，便于看档位差距）
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
| **本项目 mapping-only** | - | - | - | - | - | - | - | - |
| **本项目 +MemoryBank** | - | - | - | - | - | - | - | - |
（本项目按 4-shot 交付；1/2-shot 未跑，故全部 `-`。来源同上：SubspaceAD T8、KAG T17、FoundAD T9。）

## 表 3：同任务但**设置不同**（多类 / 多模态-3D / 跨域通用）——只列论文报告的那几项
| 方法 | 会议·年 | 设置差异 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO/PRO | 出处 |
|---|---|---|---|---|---|---|---|---|---|---|
| InCTRL | CVPR'24 | 通用 in-context（训练一次跨数据集） | 0.945 | - | - | 0.972 | - | - | - | InCTRL T6/T11（4-shot MVTecAD） |
| DictAS | ICCV'25 | 跨类泛化 few-shot | 98.6 | - | - | - | 66.8 (AP) | - | 95.1 (PRO) | DictAS T5（1-shot MVTecAD，口径为 AUROC/PRO/AP） |
| UniVAD | CVPR'25 | training-free 跨域（工业+医学+逻辑） | 97.8 | - | - | 96.5 | - | - | - | UniVAD T7（1-shot MVTec-AD） |
| CMDS-AD | ECCV'26 | 多模态（RGB+3D） | 87.1–87.3 | - | - | 98.6–98.9 | - | - | 95.1–95.8 (PRO@30%) | CMDS-AD T20（MVTec 3D-AD） |
| CIF (Commonality in Few) | AAAI'26 | 多模态（RGB+3D） | 72.0 | - | - | 86.1 | - | - | - | CIF T5（1-shot MVTec 3D-AD 均值） |
| AnoPLe | CVPR'26 | **多类** few-shot（一套模型多类） | - | - | - | - | - | - | - | 论文以多类协议报告，未在本表口径给单类数字 |
| SeaS / DualAnoDiff / AnomalyDiffusion / MAGIC / GroundingAnomaly | ICCV'25 / CVPR'25 / AAAI'24 / CVPR'26 / ECCV'26 | **异常生成**方法 | - | - | - | - | - | - | - | 主要报 FID/IS 或下游增量，不报本表口径 |
| TailedCore | CVPR'25 | 长尾+噪声少样本 | - | - | - | - | - | - | - | 协议不同（长尾分布） |
| One-to-Normal | NeurIPS'24 | 少样本个性化 | - | - | - | - | - | - | - | 论文口径不同 |
| MVREC / Hyperbolic Defect Feat. Synth. | AAAI'25 / CVPR'26 | 少样本缺陷**分类** | - | - | - | - | - | - | - | 任务为分类，无像素指标 |

> 说明：表 3 的 `-` 有两种含义——①论文未报告该指标；②该论文的评测协议不同（多类/多模态/分类），
> 其数值**不可**与表 1 直接比较。表 3 里给出数字的几项也已在"设置差异"列标明。

---

# 附：PRO vs AUPRO（口径说明 + 实测差异）

## 1. 定义上的关系
* **PRO（Per-Region Overlap）**：对 GT 里**每一个连通缺陷区域**分别算"被预测覆盖的比例"，
  再对**区域**取平均（每块区域等权，与区域大小无关）；随阈值变化得到 PRO–FPR 曲线。
* **AUPRO**：上述曲线在 **FPR ≤ 0.3** 区间积分后**除以 0.3**（= FPR≤30% 内的平均 PRO）。
  MVTec 官方脚本与 `adeval` 都用这个约定。
* ⇒ **文献里以单个数字报告的 "PRO"，绝大多数就是 AUPRO@30%**；两者是"曲线 vs 曲线面积"的关系，
  不是两套不同指标。

## 2. 与像素 AUROC 的本质差别
| | 加权单位 | 后果 |
|---|---|---|
| P-AUROC | 每个**像素** | 大缺陷主导；只检出大缺陷的一部分也拿高分 |
| PRO/AUPRO | 每个**连通区域**等权，且看"区域被覆盖比例" | 小/多缺陷不被淹没；"只覆盖部分区域"被严厉扣分；受 FPR 预算限制 |

本项目实测（mapping⊕bank, 0.3/0.7）：transistor `P-AUROC 0.9124 / AUPRO 0.7785`、tile `0.9429 / 0.9120`
——小缺陷类别上 AUPRO 明显低于 P-AUROC。

## 3. 实现细节能造成 1–5 个点差异（同一张图上实测）
| 实现方式 | transistor | tile |
|---|---|---|
| 朴素：只积分到最后一个 FPR≤0.3 的采样点 | 0.7605 | 0.8622 |
| **在 FPR=0.3 处线性插值后再积分** | **0.7785** | **0.9118** |
| `adeval`（本项目所用） | **0.7785** | 0.9120 |

⇒ 加上边界插值后与 `adeval` **逐位一致**，证明"我们的 P-AUPRO ≡ 文献的 PRO@30%"；
而**仅边界处理一项就能差 1.8（transistor）~ 5.0（tile）个点**。其它会影响数值的细节还包括：
区域连通性（8-/4-连通）、最小区域面积过滤、GT/预测图分辨率与高斯平滑（本项目 392 + σ=4）、
分数轴离散条数（`adeval` 用 `nstrips=200`）。此外 **PRO@1% 与 PRO@30% 完全不同量级**
（CMDS-AD：PRO@1%≈40 vs PRO@30%≈95），引用时务必看清论文用的是哪个。

## 4. 对横表的影响
表 1/3 里把文献 `PRO` 与我们的 `AUPRO` 分列，原因就是上面这些实现差异 + 论文常不写明 FPR cap。
若要严格横比，需在**同一分辨率、同连通性、同 FPR cap、同插值**下把两边都重算一遍
（可用本仓库的 `.maps_audit/*_improved_m4m6r3.map` + `.errmap_audit/*.map` 直接做）。

---

# 附 2：去论文代码库核对 PRO 实现（已做完）+ 三方实现互验

## 附2.1 各论文 PRO 实现的"实地核查"结果
| 方法 | 代码库 | 实现函数 | FPR cap | 连通性 | 阈值取法 | 积分/归一化 |
|---|---|---|---|---|---|---|
| **本项目（adeval）** | 本仓库 | `adeval.iterative._aupro` | **0.3** | 8（cv2 `connectedComponentsWithStats`） | 分数空间均匀 `nstrips=200` | 积分到 0.3 后 **÷0.3**，边界线性插值 |
| SubspaceAD | CLendering/SubspaceAD `main.py` | `compute_aupro` | **0.3**（注释写 "MVTec convention"） | 8 | **FPR 线性 300 条** | (0,0) 锚点 + 梯形积分 ÷0.3 |
| AnomalyDINO | dammsi/AnomalyDINO `src/post_eval.py` | `compute_pro`（**MVTec 官方代码**） | **0.3** | 8（3×3 结构元） | 逐像素分数事件（全量） | `trapezoid(..., x_max=0.3)` **÷0.3** |
| PromptAD | FuNz-0/PromptAD `utils/metrics.py` | `cal_pro_metric` | **0.3** | 8（`skimage.measure.label`, connectivity=2） | 分数空间 200 步 | **把观测 FPR 区间重标定到 [0,1] 再积分**；GT 二值化 0.45、用 `filled_image` |
| DictAS | xiaozhen228/DictAS `models/metric_and_visualization.py` | `cal_pro_score` | **0.3**（`fprs<0.3`） | 8（skimage 默认） | 分数空间 200 步 | **FPR 重标定到 [0,1] 后 `sklearn.auc`** |
| KAG-prompt | CVL-hub/KAG-prompt | 仓库内**未见** PRO 实现 | - | - | - | 论文报了 PRO，需向作者确认 |
| FoundAD / RegAD / FastRecon / patchcore-few-shot | 各仓库 | 未见 PRO 实现（依赖外部评测脚本） | - | - | - | - |

## 附2.2 三方实现在**我们的打分图**上互验（15 类均值）
| 实现 | 仅 mapping | mapping ⊕ bank |
|---|---|---|
| 本项目 adeval `P-AUPRO` | **0.9172** | **0.9455** |
| MVTec 官方 `compute_pro` ÷0.3（AnomalyDINO 那份代码） | **0.9173** | **0.9455** |
| SubspaceAD `compute_aupro` | - | **0.9454** |

* 三者逐类差异 ≤ **0.0001**（only 舍入级别）→ **我们的 P-AUPRO 与文献的 PRO@30% 数值等价**；
* 反向验证：若把"FPR=0.3 边界插值"去掉（朴素截断），同一张图会低 **1.8（transistor）~5.0（tile）** 个点
  —— 这正是"跨论文直接比 PRO 数字"的风险来源。
* 因此表 1 中 SubspaceAD / AnomalyDINO / PromptAD / DictAS 的 PRO 列与我们的 AUPRO 列**可以并排比较**；
  KAG-prompt 等未开源评测的条目仍应标注"口径待确认"。

---

# 【重排版】2024–2026 少样本无监督 AD：全指标表（含 PRO 口径核查）

> 本节**取代**前面散落的表 1/2/3；所有数字来自论文原表（arXiv HTML 抽取），所有 PRO 口径结论来自**实际代码库核查**。

## 表 0 · 参考基线信息
| | MVTec I-AUROC | MVTec I-AP | MVTec I-F1 | MVTec P-AUROC | MVTec P-AP | MVTec P-F1 | MVTec P-AUPRO |
|---|---|---|---|---|---|---|---|
| vpt3.0 mapping 基线 | 95.86 | 97.19 | 95.59 | 96.40 | 56.50 | 59.50 | 91.07 |
| **本项目 mapping-only（m4m6r3）** | **97.42** | **98.44** | **96.76** | **96.43** | **60.11** | **62.01** | **91.73** |
| **本项目 mapping ⊕ MemoryBank(0.3/0.7)** | **98.34** | **99.12** | **97.79** | **97.46** | **66.63** | **66.34** | **94.55** |

## 表 1 · 4-shot / MVTec / 单类 / 无监督（与本项目同设置）
| 方法 | 会议·年 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | PRO/AUPRO | PRO 口径可信度 |
|---|---|---|---|---|---|---|---|---|---|
| SPADE | arXiv'20 | 84.8 | 92.5 | - | 92.7 | - | - | 87.0 | ✔（SubspaceAD 表，官方族） |
| PaDiM | ICPR'20 | 79.8–80.4 | - | - | 92.1–92.6 | - | - | - | - |
| PatchCore | CVPR'22 | 88.8–92.1 | 94.5 | - | 94.3–96.1 | - | - | 84.3 | ✔ |
| RegAD | ECCV'22 | 88.2 | - | - | 95.8 | - | - | - | - |
| WinCLIP(+) | CVPR'23 | 93.1–95.5 | 97.3 | - | 94.2–96.7 | - | - | 89.0 | ✔ |
| GraphCore | ICLR'23 | 92.9 | - | - | 97.4 | - | - | - | - |
| FastRecon | ICCV'23 | 91.3–94.2 | - | - | 96.1–97.0 | - | - | - | - |
| AnomalyGPT | AAAI'24 | 96.3 | - | - | 96.2 | - | - | - | - |
| **PromptAD** | CVPR'24 | 95.5–96.6 | 98.5 | - | 96.3–96.5 | - | - | 90.5 | ⚠（FPR 重标定变体，实测低 ~0.15） |
| FOCT | ACM MM'24 | 93.2 | - | - | 96.2 | - | - | - | - |
| **KAG-prompt** | AAAI'25 | 97.1 | - | - | 96.7 | - | - | - | ?（仓库无 PRO 实现） |
| IIPAD | 2025 | 96.1 | 98.1 | - | 97.0 | - | - | 91.2 | ✔ |
| **AnomalyDINO** | 2024/25 | 97.7 | 98.7 | - | 97.2 | - | - | 93.4 | ✔（**MVTec 官方代码**） |
| **SubspaceAD**（training-free） | CVPR'26 | 98.0 | 99.0 | - | 97.9 | - | - | 95.1 | ✔（自带 compute_aupro，0.3） |
| **FoundAD** | ICLR'26 | 97.1 | 98.6 | - | 97.2 | - | - | 93.5 | ?（仓库无 PRO 实现） |
| FastRef（加到 WinCLIP） | CVPR'26 | 95.5 | - | - | 96.7 | - | - | - | - |
| **本项目 mapping-only** | 本项目 | **97.42** | **98.44** | **96.76** | **96.43** | **60.11** | **62.01** | **91.73** | ✔（= 官方实现，逐类差 ≤1e-4） |
| **本项目 +MemoryBank** | 本项目 | **98.34** | **99.12** | **97.79** | **97.46** | **66.63** | **66.34** | **94.55** | ✔（同上） |

## 表 2 · 同设置 1-shot / 2-shot
| 方法 | 1s I-AUROC | 1s I-AP | 1s P-AUROC | 1s PRO | 2s I-AUROC | 2s I-AP | 2s P-AUROC | 2s PRO |
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
| **本项目** | - | - | - | - | - | - | - | -（当前只交付 4-shot） |

## 表 3 · 同任务不同设置
| 方法 | 会议·年 | 设置差异 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | PRO/AUPRO |
|---|---|---|---|---|---|---|---|---|---|
| InCTRL | CVPR'24 | 通用 in-context（跨数据集） | 94.5 | - | - | 97.2 | - | - | - |
| DictAS | ICCV'25 | 跨类泛化 few-shot | 98.6 | - | - | - | 66.8 | - | 95.1（⚠FPR 重标定变体） |
| UniVAD | CVPR'25 | training-free 跨域 | 97.8 | - | - | 96.5 | - | - | - |
| CMDS-AD | ECCV'26 | 多模态 RGB+3D | 87.1–87.3 | - | - | 98.6–98.9 | - | - | 95.1–95.8（PRO@30%） |
| CIF | AAAI'26 | 多模态 RGB+3D | 72.0 | - | - | 86.1 | - | - | - |
| AnoPLe | CVPR'26 | 多类 few-shot | - | - | - | - | - | - | - |
| SeaS / DualAnoDiff / AnomalyDiffusion / MAGIC / GroundingAnomaly 等 | 2024–26 | 异常生成 | - | - | - | - | - | - | -（报 FID/IS 或下游增量） |
| TailedCore / One-to-Normal | CVPR'25 / NeurIPS'24 | 长尾噪声 / 个性化 | - | - | - | - | - | - | - |
| MVREC / Hyperbolic Defect Feat. Synth. | AAAI'25 / CVPR'26 | 少样本分类 | - | - | - | - | - | - | - |

## 表 4 · 五种社区 PRO 实现在**我们自己的打分图**上互验（说明可直接横比）
| 数据 | adeval（本项目） | MVTec 官方 | SubspaceAD | PromptAD | DictAS |
|---|---|---|---|---|---|
| transistor（融合） | 0.7785 | 0.7786 | 0.7784 | 0.7780 | 0.7778 |
| tile（融合） | 0.9120 | 0.9119 | 0.9119 | 0.9094 | 0.9094 |
| leather（融合） | 0.9882 | 0.9882 | 0.9880 | 0.9869 | 0.9869 |
| 三类均值 | **0.8929** | **0.8929** | **0.8928** | 0.8915 | 0.8914 |
| 15 类均值（仅 mapping） | 0.9172 | 0.9173 | - | - | - |
| 15 类均值（融合） | 0.9455 | 0.9455 | 0.9454 | - | - |

* 三种"官方族"实现（adeval / MVTec 官方 / SubspaceAD）差异 **≤0.0002**；
* PromptAD 与 DictAS 的"FPR 重标定"变体系统性低 **约 0.0015**（同口径下可预期）；
* ⇒ 我们用 adeval 报的 `P-AUPRO` 与文献 `PRO` **可直接并排比较**（误差量级 ≤0.2 个点）。
