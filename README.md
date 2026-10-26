# vpt3.9 使用说明（2026-10-26）

> **v3.9 = VPT3.8 主干与训练配方 + 数据集通用 SS-MSF 推理融合**

VPT3.9 删除 VPT3.8 中依赖数据集阈值的 Support-Adaptive 规则，改为统一的
support-only 连续门控：

```text
c = cosine(中心区域平均特征, 边缘区域平均特征)
r = 中心区域平均特征范数 / 边缘区域平均特征范数
w_m = clip((1-c) * r, 0, 1)
w_b = 1-w_m
```

异常图在 9、25、41 三个池化尺度上融合，三个尺度共享同一张异常图，不增加
网络前向次数。融合不使用任何测试集统计量。

| 数据集 | 1-shot | 2-shot | 4-shot |
|---|---:|---:|---:|
| MVTec-AD | 87.6806 | 88.3969 | 89.3952 |
| VisA | 82.0201 | 82.1743 | 82.9413 |

详细说明见 `docs/SUPPORT_STRUCTURE_MULTISCALE_FUSION.md` 和
`docs/VPT39_124SHOT_UNIFIED_FUSION_COMPARISON.md`。

---

# vpt3.8 使用说明（2026-10-20）

> **v3.8 = 严格 4-shot 数据协议 + raw 高分辨率融合**

本版不是新的网络主干，而是在 v3.7 的 learnable mapping 基础上加入：

- 严格 MVTec 4-shot 数据集视图；
- 严格 4-source AnoStyler 数据；
- 训练/mapping/bank 共用同一 shot 清单；
- raw 无测试集统计量融合；
- 672 bank 不降采样；
- 可重复的 MemoryBank 22× 增强；
- 六组归一化对照与完整指标审计。

## v3.8 默认流程

```text
严格 4 张正常 shot
├─ 已有 AnoStyler 4-source 异常 → 训练 mapping
└─ 4 × 22 增强视图 → 原始 MemoryBank

测试
├─ mapping: 448 → 1.5× → 672
└─ bank:    672

mapping 输出 448 → 672
bank 保持 672
raw:
  0.5 · box25(mapping_672)
+ 0.5 · box25(bank_672)
→ Gaussian(k=5, sigma=4)
→ top-0.5% mean
```

当前不执行单图 min-max、逐图 z-score 或全类别测试集 z-score。

## v3.8 目录

```text
vpt3.8/
├── code/          当前 mapping 训练、模型与 MemoryBank 源码
├── tools/         推理、建库、融合、严格数据构造、审计和运行脚本
├── data_prep/
│   ├── mvtec4_strict_existing_manifest.json
│   └── shots_mvtec4_strict/
├── docs/          当前方案、代码一致性审计、归一化与高分辨率对照
├── logs/          不包含异常图/权重的关键指标 CSV
├── VERSION
├── ENVIRONMENT.md
└── PERFORMANCE.md
```

## 严格 4-shot 运行

```bash
cd <repo>
bash tools/run_mvtec4_strict_existing_pipeline.sh
```

该脚本依次执行：

1. 从已有 4-source AnoStyler 数据构造严格数据集视图和异常训练根；
2. 35 epochs 训练 15 类 mapping；
3. 生成 448 mapping 异常图和 672 MemoryBank 异常图；
4. 在 672 上执行 raw 融合并计算七项指标。

严格数据要求：

- 每类 `train/good` 恰好 4 张；
- AnoStyler source 与这 4 张逐字节一致；
- 不使用 1/2-shot 生成数据；
- 测试图不进入训练；
- bank 增强 seed 固定。

## 当前方案文档

```text
docs/current_strict_mvtec4_scheme_zh.md
```

代码与方案逐项核对：

```text
docs/严格4shot方案与代码一致性核对_20261020.md
```

## 本版重要说明

- v3.8 代码已包含严格 4-shot 流水线；
- MVTec 严格 4-shot 完整指标以运行后生成的 CSV 为准；
- VisA 2/4-shot 的 672 bank raw 对照已写入 `PERFORMANCE.md`；
- checkpoint、异常图、数据集和 723GB 实验目录均未提交到 Git；
- 仓库只保存源码、脚本、配置、文本日志和小型指标 CSV。

下面保留 v3.7 历史说明和更早版本记录作为参考。

# vpt3.7 使用说明（2026-10-13 冻结）

> **本版 = 把"简化版"固化为默认**：相对 vpt3.6 只改推理端打分与默认值，**训练代码与权重完全一致**。
> 默认已关闭 learn 侧全局项（`--l_global 0`，连带不再对 bank 做 `token_mapping`）与翻转视图（`--flip 0`）→ **推理前向次数减半**。
> 要恢复 v3.6 行为：`--l_global 64 --flip 1`。

## 零、v3.7 一句话

```
s_raw   = d1^0.5 · q50^0.5        # 原始库侧：最近邻 × 全库距离中位数
s_learn = dlmin                    # 可学习侧：仅最近原型（无全局项）
m       = 0.6·s_learn + 0.4·s_raw  # 单视图（1.5×，无翻转）
```

三数据集（官方 adeval，7 项平均）：

| 数据集 | mapping-only | bank-only | **融合** |
|---|---|---|---|
| VisA 4-shot（12 类） | 78.85 | 78.74 | **81.64** |
| MVTec 2-shot（15 类） | 85.09 | 85.92 | **87.51** |
| MVTec 4-shot（15 类） | 85.59 | 87.47 | **88.57** |

（与 v3.6 的融合差：+0.03 / −0.14 / −0.07；mapping 单支差：−0.08 / −0.50 / −0.65）

> **本版与 vpt3.5 的训练代码完全一致**（`code/main-fewshot-vpt4.py` md5 `08d8528b4a59`，未改动、未重训）。
> 本版所有改动都在**推理端**（打分算子 + 融合权重）与**数据准备/实验框架**上。

## 一、本版改了什么（相对 vpt3.5）

> 下表为 v3.6 的改动记录；**v3.7 在此基础上再把"learn 侧全局项"与"翻转视图"两项移除并设为默认**（见 §零）。

| 组件 | 状态 |
|---|---|
| 训练 | ✅ **不变**：35 轮 / K-shot / aux 合成异常；`Resize(448) → CenterCrop(392)` |
| 测试预处理 | ✅ 不变：`Resize(448)` **不裁剪**；1.5× + 翻转共 2 视图；高斯平滑 k=5, σ=4 |
| **mapping 原始库侧打分** | 🆕 `s_raw = d1^0.5 · q50^0.5`（旧版为 `d1 = 1 − max cos`） |
| **mapping 可学习侧打分** | 🆕 `s_learn = dlmin^0.7 · bq64^0.3`（旧版为 `dlmin`） |
| **视图内融合** | 🆕 `m = 0.6·s_learn + 0.4·s_raw`（旧版 `0.5/0.5`） |
| **mapping ⊕ bank 融合** | 🆕 `0.5·z(mapping) + 0.5·z(bank)`（旧版 `0.3/0.7`） |
| bank 分支 | ✅ 不变：`facebook/dinov2-base` 双 stage + faiss，K shot × 22 增强 |
| **VisA 全流程** | 🆕 官方 VisA → MVTec 式数据根、1/2/4-shot 嵌套抽样、AnoStyler 合成异常、训练/评估脚本 |
| **打分搜索框架** | 🆕 分量网格落盘 → 离线枚举任意 (局部项, 全局项, 指数, 融合权重) → 生产重推与官方复核 |
| 诊断/审计脚本 | 🆕 分位数信息量诊断、指标单调不变性校验、按类配对 bootstrap、bank 侧打分扫描 |

### 为什么这样改（一句话机制）

异常分数不再只看"离最近的**一个**正常 patch 多远"，而是取
**`(到最近正常 patch 的距离)^a × (到整个正常库距离分布的分位数)^(1−a)`**。
它同时压掉两类假高分：侥幸贴近某个原型的正常像素、整幅纹理偏高的正常图。
可学习侧的全局项必须取自"映射到同一度量空间的正常 patch 库"，否则几何平均的两个因子不在同一尺子上。

## 二、三数据集结果（官方 adeval，7 项平均）

| 数据集 | 分支 | 旧（vpt3.5） | **新（vpt3.6）** | Δ |
|---|---|---|---|---|
| VisA 4-shot（12 类） | mapping-only | 75.53 | **78.93** | **+3.40** |
| VisA 4-shot | mapping ⊕ bank | 80.01 | **81.61** | **+1.60** |
| MVTec 2-shot（15 类） | mapping-only | 85.13 | **85.59** | +0.46 |
| MVTec 2-shot | mapping ⊕ bank | 87.38 | **87.65** | +0.27 |
| MVTec 4-shot（15 类） | mapping-only | 85.43 | **86.24** | +0.81 |
| MVTec 4-shot | mapping ⊕ bank | 88.45 | **88.64** | +0.19 |

⚠ **统计与口径提醒（务必连同数字一起引用）**：按类配对 bootstrap 显示，
**VisA 的提升显著**（mapping +3.41，95%CI [+2.02,+4.84]，12/12 类为正）；
**MVTec 的 +0.27/+0.19 的 95% CI 包含 0（不显著）**；且 +1.60 里有约 0.4 来自
"融合权重 0.3→0.5"这一在测试集上选出的改动（单独改权重对旧打分是 −0.51）。
详见 `docs/SCORING_SEARCH_2026-10-13.md`、`docs/COMPARE_three_datasets.md`。

## 三、目录

```
vpt3.6/
├── code/        主训练/评估代码（与 vpt3.5 逐字节一致，未改动）
├── tools/       推理（新打分）、融合、建库、搜索框架、诊断与审计脚本
├── data_prep/   VisA 数据准备脚本 + AnoStyler 配置与 shot 清单（不含生成图）
├── docs/        报告：打分搜索、推理细节、全流程、论文对比、流程图（png+svg）
├── logs/        关键实验日志（训练/评估/建库/融合/搜索过程 CSV 与驱动日志）
├── VERSION      版本说明与已知限制
├── README.md    本文件
├── ENVIRONMENT.md  环境与依赖
└── PERFORMANCE.md  历史性能记录（vpt3.5 及更早）
```

## 四、快速开始

```bash
# 0) 环境
source tools/common_env.sh

# 1) 训练（与 vpt3.5 相同；以 VisA 4-shot 为例）
SHOTS=4 EXTRA_ARGS="--global_proto 1 --w_div 0.1 --w_pixel 0.0 --metric_head 1 \
  --w_rank 0.02 --rank_margin 0.25 --rank_norm 1" \
  bash tools/q_rotate_train_visa.sh vpt35_visa4shot

# 2) 推理（vpt3.6 新打分）：落盘 mapping 异常图 + 出 mapping-only 指标
python tools/eval_score_v3.py --save vpt35_visa4shot --class candle --shots 4 --size 448 \
  --data_path /home/jjquan/experiment/visual2/visa_prepared \
  --ckpt_dir results/visa/vpt35_visa4shot \
  --raw_alpha 0.5 --raw_global 512 --l_alpha 0.7 --l_global 64 --w 0.6 \
  --save_maps_dir .maps_v3_nocrop448_visa4

# 3) bank 分支
python tools/cache_bank_nocrop.py --class candle --shots 4 --size 448 \
  --data_path /home/jjquan/experiment/visual2/visa_prepared --out_dir .errmap_nocrop448_visa4

# 4) 融合 + 评测（输出 mapping / bank / fused 三行）
python tools/fuse_nocrop.py --class candle --size 448 --w 0.5 \
  --map_dir .maps_v3_nocrop448_visa4 --bank_dir .errmap_nocrop448_visa4 \
  --data_path /home/jjquan/experiment/visual2/visa_prepared
```

## 五、打分搜索（可复现的探索路径）

```bash
# ① 一次推理，落盘"分量"网格（raw: d1,q8..q1024；learn: dlmin,dlmed,bmin,bq8..bq1024）
python tools/probe_score_grid3.py --save vpt35_visa4shot --class candle --shots 4 \
  --data_path <visa_prepared> --ckpt_dir results/visa/vpt35_visa4shot --out_dir .grids3_visa4
# ② 离线枚举任意组合（CPU，几秒/组合）
python tools/eval_comp.py --grids .grids3_visa4 --specs "d1:q512:0.5|dlmin:bq64:0.7|0.6"
# ③ 对最终配方用官方 adeval 复核（需 CUDA）
CUDA_VISIBLE_DEVICES=<uuid> python tools/eval_comp.py --grids .grids3_visa4 --official 1 \
  --specs "d1:q512:0.5|dlmin:bq64:0.7|0.6"
```

## 六、已知限制（review 时请一并阅读）

## 六之补：【2026-10-13 更新】消融结论与简化版

**训练侧 8 组消融全部变差 ⇒ 没有可删的组件**（详见 `docs/ABLATION_2026-10-13.md`）：

| 去掉 | 5 类筛选 Δ（新打分） | 判定 |
|---|---|---|
| `L_push` | **−2.68** | 必要（否定了"可能冗余"的假设） |
| `L_seg` | **−1.45** | 必要（代价集中在 screw） |
| `L_rank` / `L_div` | −2.82 / −0.79 | 必要 |
| 度量头 `W`（占 53% 参数） | −1.07 | 必要（旧打分下曾看似可删 +0.44） |
| 32 个全局锚点 | −3.19 | 必要 |

**按指示移除的两项（简化版，`eval_simple.sh`）**：`--l_global 0`（learn 侧全局项，连带去掉对 bank 的
`token_mapping`）+ `--flip 0`（翻转视图，推理前向减半）。

| 数据集 | 分支 | v3.6 | **简化版** | Δ |
|---|---|---|---|---|
| VisA 4-shot | mapping-only / **融合** | 78.93 / 81.61 | **78.85 / 81.64** | −0.08 / **+0.03** |
| MVTec 2-shot | mapping-only / **融合** | 85.59 / 87.65 | **85.09 / 87.51** | −0.50 / **−0.14** |
| MVTec 4-shot | mapping-only / **融合** | 86.24 / 88.64 | **85.59 / 88.57** | −0.65 / **−0.07** |

⇒ **交付口径（mapping ⊕ bank）几乎不受影响**，而推理前向次数减半、代码路径更短。
若要恢复：去掉 `--l_global 0` 与 `--flip 0` 即可（两个开关默认值即 v3.6 原行为）。

1. **训练/推理打分不一致**：训练损失仍用 `proto_distance_scores`（min），新公式只在推理端生效。
2. **超参在测试集上选过**：打分指数（0.5 / 0.7）、分位点（512 / 64）、融合权重（0.5）均未见独立验证集。
3. **融合是 transductive**：类内 z 的均值/方差取自该类全部测试像素。
4. **`fuse_nocrop.py` 只校验两张图的形状**，不校验图像身份；换数据根/换顺序时需人工确认。
5. 代码遗留：前向中 `token_dict['0']` 被追加两次（原型集因此为 96 而非 64），训练/推理两侧一致，
   但容易被误读为 bug —— 详见 `docs/PIPELINE_full.md` §3.6 与 §9。

---

## 附：vpt3.5 使用说明（历史）

## 这一版改了什么（相对 vpt3.0）
| 组件 | 状态 |
|---|---|
| **测试预处理** | ✅ **改为不裁剪**：`Resize(448) → ToTensor → Normalize`（vpt3.0 是 `Resize(448)→中心裁392`） |
| 训练预处理 | 保持 vpt3.0 口径不变：`Resize(448) → CenterCrop(392)`，35 轮 / 4-shot / aux 合成异常 |
| 改善后 mapping 分支 | ✅ M1 原型初始化(std=0.02) + M3 原型多样性正则 + M4 全局原型锚点 + **M6 可学习度量头** + **M7 跨图排序一致性损失** |
| 原版 MemoryBank 分支 | ✅ 不变：`facebook/dinov2-base` 双 stage + faiss，4 shot × 22 增强 = 88 张 |
| 融合规则 | ✅ 不变：类内 z 标准化后 `0.3·z(mapping) + 0.7·z(bank)`，再高斯平滑(k=5,σ=4) |
| **数据源统一** | ✅ 新增 `tools/shot_utils.py`：训练 / mapping 特征库 / bank 建库 **共用同一份 shot 清单**（落盘 `.shots/<cls>_shotK_seedS.json`），换 shot 数仍一致 |
| 代码副本隔离 | ✅ `produce_both.py` 两次加载不同代码副本时有断言：mapping 侧必须用 `code_vpt4` 的 `dinov2`，`MemoryBank` 实现必须与 `code_vpt2` 副本逐字节一致 |

## 目录
```
vpt3.5/
├── code/       主训练/评估代码（原仓库零改动，本目录为副本）
├── tools/      运行、评估、建库、融合、可视化与自检脚本
├── docs/       实验日志 / 变更记录 / 风险审计 / 文献对比
├── logs/       关键实验日志（训练、评估、融合、清单）
├── PERFORMANCE.md   本版本性能（15 类 × 7 指标）
├── COMPARISON.md    与 2024–2026 同类论文的对比
└── ENVIRONMENT.md   环境与依赖
```

## 运行（当前口径）
```bash
# 1) 训练（训练裁剪 392，35 轮；shot 清单由 tools/shot_utils.py 统一给出）
bash tools/q_rotate_train.sh <save_name> <class...> -- --global_proto 1 --w_div 0.1 \
     --w_pixel 0.0 --metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1

# 2) 测试不裁剪口径下重推 mapping 分支（并落盘未平滑异常图）
python tools/eval_nocrop.py --save <save_name> --class <class> --size 448 \
     --save_maps_dir .maps_nocrop448_<save>

# 3) 测试不裁剪口径下重建 MemoryBank 异常图
python tools/cache_bank_nocrop.py --class <class> --size 448 --out_dir .errmap_nocrop448

# 4) 类内 z 融合（0.3/0.7）+ 官方指标
python tools/fuse_nocrop.py --class <class> --size 448          # 逐类
#   或一次性：tools/produce_both.py（单 loader 同时出两分支）
```
关键开关：`--global_proto 1`（M4）、`--metric_head 1`（M6）、`--w_rank 0.02 --rank_margin 0.25 --rank_norm 1`（M7）、
`--w_div 0.1`（M3）、`--token_init_std 0.02`（M1）；评估端必须与训练端保持一致
（`--num_tokens 32 --res_mapping 0 --global_proto 1 --metric_head 1`），否则会被 `strict=False` 静默丢权重。

## 性能（15 类平均，官方 adeval，测试不裁剪 448）
| 分支 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| mapping-only | 0.9711 | 0.9840 | 0.9684 | 0.9651 | 0.5680 | 0.5986 | 0.9246 |
| bank-only | 0.9782 | 0.9884 | 0.9705 | 0.9721 | 0.6315 | 0.6368 | 0.9452 |
| **mapping ⊕ bank** | **0.9806** | **0.9902** | **0.9758** | **0.9781** | **0.6557** | **0.6584** | **0.9531** |

逐类明细见 `PERFORMANCE.md`，与论文的对比见 `COMPARISON.md`。

## 与顶会论文的对比（摘要）
同设置口径 **4-shot / MVTec / 单类 / 无监督 / 像素定位**：

| 方法 | 会议·年 | I-AUROC | P-AUROC | PRO/AUPRO |
|---|---|---|---|---|
| PromptAD | CVPR'24 | 95.5–96.6 | 96.3–96.5 | 90.5 |
| AnomalyGPT | AAAI'24 | 96.3 | 96.2 | - |
| KAG-prompt | AAAI'25 | 97.1 | 96.7 | - |
| IIPAD | 2025 | 96.1 | 97.0 | 91.2 |
| AnomalyDINO | 2024/25 | 97.7 | 97.2 | 93.4 |
| SubspaceAD（training-free, 672） | CVPR'26 | 98.0 | 97.9 | 95.1 |
| FoundAD | ICLR'26 | 97.1 | 97.2 | 93.5 |
| **vpt3.5 mapping-only（本版）** | 本项目 | 97.11 | 96.51 | 92.46 (AUPRO) |
| **vpt3.5 mapping ⊕ bank（本版）** | 本项目 | **98.06** | **97.81** | **95.31 (AUPRO)** |

融合后在 I-AUROC 与 PRO/AUPRO 上高于表内所有单独方法（PRO 已与 MVTec 官方/SubspaceAD/PromptAD/DictAS 四种实现互验，差 ≤0.0002）。
完整表（含 1/2-shot、多类/多模态/跨域等其他设置、PRO vs AUPRO 口径核查、2024–2026 全部 31 条少样本条目收录清单）见 `COMPARISON.md`。
