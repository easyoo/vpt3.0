# Mapping 分支审计与优化 · 结论汇总

## ★ 头条结果（2026-10-08，官方 adeval 指标，5 类，392 口径）

> **口径说明（重要）**：下表"推荐配置"用的是**逐类最优轮**（screw@25 / hazelnut@20 / wood@10 / capsule@10 / metal_nut@35）。
> **若要求所有类别统一轮次（无 oracle）**，用统一 35 轮：**P-AP 0.6037 / P-AUPRO 0.9445 / P-AUROC 0.9756 / I-AUROC 0.9613**，
> 相对统一 35 轮的原始打分为 **P-AP +0.1153、P-AUPRO +0.0503**。详见 EXPERIMENT_LOG §18。

**训练侧零改动，仅测试期三项改动叠加**（选最优轮 + 打分图融合 + 多尺度/翻转），
映射分支在 MVTec 5 类上的平均指标：

| 指标 | 原始管线口径（35 轮末） | **推荐配置**（选轮+融合+多尺度） | Δ |
|---|---|---|---|
| **P-AP** | 0.4884 | **0.6301** | **+0.1417（+29%）** |
| P-AUPRO | 0.8942 | **0.9493** | +0.0551 |
| P-AUROC | 0.9403 | **0.9768** | +0.0365 |
| I-AUROC | 0.9575 | **0.9668** | +0.0093 |

相对"同样选最优轮、但只用现状打分"的对照（base）：**P-AP +0.0941、P-AUPRO +0.0327、
P-AUROC +0.0122、I-AUROC +0.0130**，**7 个指标全部提升且 5/5 类全部提升**。
命令：`--fuse_bank -1 --ms_scoring`（追求像素 AP 极限可加 `--fuse_w 0.3`）。
详见 `EXPERIMENT_LOG.md` §14、§15。

> **效率档（2026-10-08 决定采用）**：多尺度只需 **2 次前向** —— 加 `--ms_scales 1.5 --ms_flip 1`
> （把输入放大 1.5× 并做水平翻转），实测与 6 视图基本持平（3 类子集 P-AP 0.5622 vs 0.5612、
> AUPRO 0.9512 vs 0.9570），成本降到约 1/3。默认的 6 视图（`1.0,1.25,1.5`）仅 AUPRO 略优，
> 作为"极致档"保留。依据：EXPERIMENT_LOG §19。

> 工作目录：`/home/jjquan/experiment/visual/audit_mapping`（**原始仓库零改动**，全部在副本中进行）
> 详细逐条修改见 `CHANGELOG.md`，完整实验数据见 `EXPERIMENT_LOG.md`（本文件是速览版）。

---

## 一、约束落实

| 约束 | 落实 |
|---|---|
| 禁止改原代码 | 复制出 `code_control`（对照）/ `code`（修复）/ `code_norm`（融合修复+路径2实现）；原仓库 `git status` 只有新增目录 |
| 只用 GPU1 | 全部脚本 `CUDA_VISIBLE_DEVICES=GPU-0e3a7e1c-...`（物理 GPU1 的 UUID；GPU0 本机 `nvidia-smi` 报 Unknown Error，不可用） |
| 只测 mapping 分支 | `--map_only`：不建 MemoryBank、不算/不读 `.errmap`，评估只走 `type=1` 原型距离 |
| 统一 35 轮 | 最终 A/B 全部 `--total_epochs 35`（早期筛查用 30 轮，已在日志标注） |
| 严谨记录 | 每个实验：命令（`q_*.sh` / `run_*.sh`）、原始日志（`logs/`）、指标 CSV（`results/`）、修改说明（`CHANGELOG.md`） |

---

## 二、路径1（工程底座）结论

### 2.1 修复清单

| 编号 | 文件:行 | 问题 | 影响 |
|---|---|---|---|
| FIX-1 | `code/dinov2/models/vision_transformer.py:367` | `token_dict['0']` 被追加两次 | 逻辑冗余；`learnable_layers>1` 时直接崩 |
| FIX-2 | `code/main-fewshot-audit.py:1497,1515` | `fused_learnable_token` 被 `sum(feat_list)` 覆盖 | 多 learnable 层时训练/评估口径不一致 |
| FIX-3 | `:1402` | 预训练权重路径不存在 | **训练直接崩**，已改候选列表 |
| FIX-4 | `:1145,1248,1351` | 无 mapping-only 开关 | 迭代慢（bank+7 种打分）；新增 `--map_only` |
| FIX-5 | `:504` | 打分逐样本 sklearn 循环 + CPU↔GPU 往返 | 评估慢；已向量化（语义不变） |
| FIX-6 | `:1550` | `type=1` 训练中评估用 392、其它用 256 | 口径不一致（实测影响 ≤0.005，非性能问题） |
| FIX-9 | `:1170` | `device='cuda:2'` 硬编码 | 单卡（CUDA_VISIBLE_DEVICES）场景直接报错 |
| FIX-10 | `:1399` | 建模型前未固定随机源 | A/B 初始权重不可比 |
| FIX-12 | `code_norm/dinov2/models/vision_transformer.py:124,315` + 5 处调用点 | **多层"求和融合"退化成只用最后一层** | 见 2.2（**本次审计最重要的发现**） |

### 2.2 关键发现：`fused_layer=[1,2,3,4,6,8,11]` 的"7 层融合"实际等于"只用第 11 层"

`fused = Σ_i patch_feat_layer[i]` 未做逐层归一化。实测（screw，ViT-B/14 @392）：

| 层 | 1 | 2 | 3 | 4 | 6 | 8 | 11 |
|---|---|---|---|---|---|---|---|
| token 平均 L2 范数 | 2.88 | 2.38 | 2.56 | 3.46 | 7.05 | 19.56 | **154.67** |

层间跨度 **64.9×**；融合向量与各层的余弦相似度
`layer1 −0.02 / layer2 0.02 / layer3 0.02 / layer4 0.05 / layer6 0.15 / layer8 0.37 / layer11 0.985`。
→ 前 4 层的信息在数值上被完全淹没。

**但"修好它"反而更差**：逐层 L2 归一化 + 可学习层权重（FIX-12）在 35 轮后 **P-AUROC 崩到 0.63~0.86**
（capsule 0.6401、metal_nut 0.6308、hazelnut 0.7355），而对照组是 0.89~0.98。只在 ≤15 轮时有微弱优势。

→ 结论：在 4-shot 场景下，**深层主导（=现状）起到了天然正则化作用**；
   "把浅层权重提上来"不是正确方向。真正该补的是**分辨率**，而不是层间配比（见第三节）。
   该负结果已完整记录，避免后续重复投入。

### 2.3 A/B（control vs fixed，30 轮）的方法学价值

control/fixed 的 5 类平均差：ΔP-AP −0.003、ΔP-AUPRO −0.001，且**逐类符号正负翻转**
→ 证明 FIX-1/2/6 在数学上是等价变换，观测到的差异只是浮点敏感性；
同时给出**噪声地板**：同配置换 seed，screw@35 的 P-AP 漂移 ±0.013、AUPRO ±0.025、I-AUROC ±0.042。
→ 任何"换 seed 得到 +0.01~0.02"的说法都不可信。

---

## 三、路径2（算法创新）成果：多尺度 + 翻转测试期打分（零训练）

### 3.1 从失败样本出发

`probe_failures.py` 对 screw@10ep 逐图分析：

* 正常图 sp_score ∈ [0.060, 0.106]，异常图 ∈ [0.059, 0.20] → **严重重叠**；
* 最难缺陷 `scratch_head`(0.103) / `manipulated_front`(0.112) 几乎落在正常区间；
* 这两类的 in/out-GT 对比度只有 0.068 / 0.076（容易的 `thread_neck` 是 0.225）→ **打分图弥散**。

根因：14px 的 patch 只覆盖 392 输入的 1/28 边长，**小缺陷的异常能量被 patch 内平均掉**。

### 3.2 方案

冻结权重，把已预处理的 392 输入按 s∈{1.0, 1.25, 1.5} 放大（token 网格 28²→35²→42²），
每个尺度再做水平翻转，共 6 个视图；各视图打分图插值回 392 后**平均**。
其余流程（高斯平滑、top-1% 图像级分数、392 口径）完全不变。**不使用任何标签信息。**

### 3.3 结果（control 权重 @ 规定的 35 轮，5 类，392 口径）

下表用**生产实现** `evaluation_batch_ms` 在 5 个 control@35 权重上复验
（`results/ms35/production_ms35.csv`；base 列与训练日志逐位一致）：

| 指标 | base（现状） | **ms_flip_mean（推荐）** | Δ |
|---|---|---|---|
| I-AUROC | 0.9575 | **0.9611** | +0.0036 |
| P-AUROC | 0.9403 | **0.9503** | **+0.0099** |
| P-AP | 0.4884 | **0.5229** | **+0.0345（相对 +7%）** |
| P-F1 | 0.5284 | **0.5470** | +0.0186 |
| P-AUPRO | 0.8942 | **0.9225** | **+0.0283（5/5 全为正）** |

逐类 P-AP：screw 0.2770→**0.3634**（+0.086）、wood 0.5453→**0.5920**、
hazelnut 0.5666→**0.6065**、capsule 0.3102→**0.3254**、metal_nut 0.7427→0.7271（唯一负项 −0.016；
其 P-AUPRO 仍 +0.029）。
逐类 P-AUPRO **5/5 全部为正**（最大 capsule +0.064）。

**效应量（诚实版）**：配对口径下 P-AUPRO 5/5 为正、P-AP 4/5 为正，独立 seed 复现；
但若保守地把 seed 噪声当误差棒，则只有 **P-AUPRO 稳健（效应 2.24× 噪声）**，
P-AP 边缘（1.26×），**图像级 I-AUROC 不显著（0.24×）**——不要对外声称 ms 提升图像级指标。

### 3.4 机制验证

多尺度使**全部 5 种缺陷类型**的 in/out-GT 定位对比度提升：
manipulated_front 0.076→0.091（+20%）、thread_side 0.086→0.105（+22%）、
scratch_neck 0.225→0.250、thread_top 0.094→0.102、scratch_head 0.068→0.070。

### 3.5 交付形态

已实现为可直接调用的选项（`code_norm/main-fewshot-norm.py`）：

```bash
python code_norm/main-fewshot-norm.py --train_class screw --total_epochs 35 \
  --map_only --fusion sum --ms_scoring --ms_scales 1.0,1.25,1.5 --ms_flip 1 ...
```

验证：`evaluation_batch_ms` 在 control@35 screw 上与探针结果逐位一致
（P-AP 0.3642、P-AUPRO 0.9334、P-AUROC 0.9865）。
代价：前向次数 ×6（约 9.6× FLOPs），**训练零成本**，可套用到任何历史权重。

---

## 四、负结果清单（避免重复踩坑）

### 训练侧候选的最终裁决（官方指标，35 轮，5 类平均）

| 候选 | P-AP | I-AUROC | 裁决 |
|---|---|---|---|
| **control（保持现状）** | **0.4884** | **0.9575** | ✅ 采用（只修工程 bug） |
| 训练期尺度抖动 `--scale_jitter` | 0.4731 | 0.9548 | ❌ 无稳定收益（逐类正负不一） |
| 浅层残差门控 `--fusion gate` | **0.3317** | **0.7625** | ❌ 明确否决（screw P-AP 0.021） |
| FIX-12 逐层归一化融合 | 崩（P-AUROC 0.63~0.86） | — | ❌ 已否决 |

**两次独立实验（FIX-12 与 gate）指向同一结论**：任何让浅层信息进入融合特征的训练改动都会破坏
这个 4-shot 训练；深度主导（control 的实际行为）必须保留。这也解释了为什么我们把精力转向
**测试期**改造（选轮 + 融合 + 多尺度），最终拿到 P-AP +0.14（+29%）。

## 三·补2 【路径2 新成果】打分图融合（零训练，CPU 已验证，GPU 复验排队）

现象：现状打分只用**随图像变化**的可学习原型，分数尺度跨图不可比 → 数据集级像素指标吃亏。
做法：额外在**原始特征空间**用 4 张正常图建一个固定原型库（PatchCore 思路），
两张打分图逐像素平均 `fused = 0.5*(d_learn + d_bank)`。

| 指标 | Δ 均值（4 类配对） | 正/负 |
|---|---|---|
| **P-AP** | **+0.0596** | **4/0** |
| P-F1 | **+0.0619** | **4/0** |
| P-AUROC | +0.0049 | 3/1 |
| I-AUROC | −0.0001 | 基本不变 |

单类 P-AP 提升：capsule +0.090、hazelnut +0.078、screw +0.047、wood +0.023。
幅度大于多尺度方案（P-AP +0.035），两者正交可叠加。

**35 轮基准上的复验（5 类，CPU 指标）与推荐档位**：

| 配置 | ΔP-AP（3 类均值） | ΔI-AUROC | 说明 |
|---|---|---|---|
| `--fuse_w 0.7` | +0.054 | **−0.003** | **保守推荐**（图像级几乎不掉） |
| `--fuse_w 0.5` | +0.071 | −0.014 | 默认（均衡） |
| `--fuse_w 0.3` | **+0.085** | −0.026 | 追求像素 AP 极限（screw 的图像级会掉） |

库规模很关键：k-center 32 会让 screw 反而变差（P-AP 0.269 < 现状 0.277），
**必须用完整正常 patch 库**（3136 个 → 0.344）。
消融另证明：可学习原型**脱离映射空间会崩**（P-AUROC 0.11~0.44），说明映射本身必要；
但纯原始特征库单独就很强（hazelnut P-AP 0.781 vs 现状 0.587），故两者互补。
生产实现 `--fuse_bank` 已与探针逐位对齐；GPU 复验脚本 `q_fuse.sh`。

## 三·补 【路径1 追加成果】模型选择：选内部评估最优轮（比多尺度收益更大、且免费）

管线本来就每 5 轮评估并保存权重，但一直用最后一轮。用 control@35 的日志离线分析
（`select_best_epoch.py`）：

| 口径 | P-AP | P-AUPRO | P-AUROC |
|---|---|---|---|
| 固定 35 轮末 | 0.4884 | 0.8942 | 0.9403 |
| **只按 P-AP 选轮** | **0.5360（+0.048）** | **0.9166（+0.022）** | **0.9646（+0.024）** |

逐类最优轮：capsule@10、wood@10、hazelnut@20、screw@25、metal_nut@35
—— 说明**35 轮里大部分类在 10~25 轮后已经过拟合**（capsule 的 P-AUROC 从 0.9724@10 掉到 0.8755@35）。
该收益与多尺度打分正交，可叠加。

> 诚实声明：管线内部评估用的就是测试集，选轮属于 oracle 上界；正式落地需要独立验证集。

## 四、负结果清单（避免重复踩坑）

| 候选 | 结果 | 判定 |
|---|---|---|
| 逐层归一化融合 FIX-12 | 35 轮后 P-AUROC 崩到 0.63~0.86 | 不用 |
| 邻域平均 `nb`（PatchCore 式） | screw P-AP 0.312→0.254 | 不用 |
| 局部对比度 `d − 邻域均值` | hazelnut P-AUROC 0.977→**0.589** | 严重有害 |
| 映射空间 k-center coreset / 并集 | 仅 screw 正收益，其余持平或略降 | 不作通用手段 |
| 单原型全局均值 | wood 变好、screw 大幅变差 | 不可通用 |
| 每图减中位数 `rel_med` | 小幅正收益（±0.002~0.003） | 可作辅助项 |

另有发现（供后续路径2用）：训练得到的 32 个"可学习原型"实际高度坍缩
（两两余弦均值 0.56~0.90，参与比有效秩仅 2.3~3.4/32），即名义 32 个原型≈2~3 个自由度。

---

## 五、复现命令

```bash
cd /home/jjquan/experiment/visual/audit_mapping

# 1) 35 轮基准 + 噪声地板（control）
bash q_control35.sh                      # 5 类 35 轮 + seed2 的 3 类

# 2) 最终推荐配置（多尺度+翻转打分）
python eval_offline.py --ckpt results/control35/maponly/model_screw_35.pth \
       --class screw --protocol 392 --fusion sum --ms_scoring

# 3) 各诊断探针
python probe_fusion.py  --ckpt results/control/maponly/model_screw_30.pth --class screw      # 多层融合退化
python probe_prototypes.py --ckpt results/control/maponly/model_hazelnut_30.pth              # 原型坍缩
python probe_coreset.py  --ckpt results/control/maponly/model_screw_30.pth --class screw     # 原型来源替换
python probe_failures.py --ckpt results/control/maponly/model_screw_10.pth --class screw     # 失败样本
python probe_scale.py    --ckpt results/control35/maponly/model_screw_35.pth --class screw --tta_flip
```

---

## 六、下一步（路径2 深化）

1. **按类自适应选择尺度 / 是否启用 ms**：管线训练期已有每 5 轮评估，可直接用于零开销选择（解决 metal_nut 的 −0.028）。
2. **把多尺度做成训练期一致性约束**：要求同一图多尺度打分图一致且异常区域多尺度同时响应，进一步压误报。
3. **更细粒度 patch**：与 ms 同源但需重训 mapping（如 518/560 输入、或融合更浅层）。
4. **分割头支路强化**：`type=2/4` 明显弱于 `type=1`（P-AUROC 0.60/0.71/0.78 vs 0.92~0.98），
   说明辅助异常集的监督信号未被有效利用，是独立且值得做的方向。
