# 当前严格 MVTec 4-shot 方案

更新日期：2026-10-20  
对应实现目录：`/home/jjquan/experiment/visual/audit_mapping`

## 0. 总体流程

```text
每类 4 张严格 shot
├─ 4 张 shot 对应的 AnoStyler 异常 → 训练 learnable mapping
└─ 4 张 shot × 22 增强 = 88 张正常视图 → 构建原版 MemoryBank

测试图
├─ Resize 448 → 放大 1.5× → mapping 推理 → 448 mapping 异常图
└─ Resize 672 → MemoryBank 推理 → 672 bank 异常图

mapping 448 → 672
bank 672 保持
→ raw 0.5/0.5 融合
→ Gaussian
→ 最高 0.5% 像素均值作为图像异常分数
→ 计算 I/P 两级七项指标
```

每个类别独立训练和评估，不做跨类别模型。

## 1. 严格 4-shot 数据

使用已有的 AnoStyler 原始 4-shot 数据，不引用任何 1-shot 或 2-shot 数据。

每类有：

```text
4 张正常 source image
× 每张 35 对 AnoStyler 异常图/mask
= 库存 140 对异常样本
```

MVTec 共：

```text
15 类 × 4 source × 35 对 = 2100 对
```

这 4 个 source image 同时也是该实验唯一的 `train/good` 正常图。已核对：

- 每类 `train/good` 恰好只有 4 张；
- 4 张正常图与 AnoStyler 使用的 source image 逐字节一致；
- 训练正常图与 MVTec 测试图不存在精确哈希重叠；
- 训练、mapping 和 MemoryBank 使用同一份严格 shot 清单。

需要注意：140 对是库存量，不代表全部进入训练。训练器采用：

```python
samples_per_anomaly_type = 35
```

因此实际异常训练数为：

```text
35 × defect 目录数
```

已有日志中：

- bottle/cable/capsule/carpet/grid/pill/screw/tile：70；
- hazelnut/transistor/wood：105；
- toothbrush：35。

这仍是严格的“最多 4 个正常 source”设置，但各 source 的实际使用次数不完全均衡。

## 2. 训练预处理

正常图和 AnoStyler 异常图都使用：

```text
Resize(448,448)
→ ToTensor
→ CenterCrop(392)
→ ImageNet Normalize
```

因此 mapping 训练时模型实际看到的是中心 `392×392` 区域。

GT mask 转换为 patch 网格时使用：

```text
adaptive_max_pool2d
→ 28×28 patch-level mask
```

不是平均池化。

每张正常 shot 使用 22 个增强视图：

```text
原图
+ 8 个任意角度旋转
+ 8 个平移
+ 1 个水平翻转
+ 1 个灰度图
+ 3 个 90°旋转
= 22
```

严格 4-shot 共：

```text
4 × 22 = 88 个正常视图
```

## 3. 特征主干

主 mapping 分支使用冻结的：

```text
DINOv2 ViT-B/14 with registers
```

融合 7 个 patch 特征层：

```text
[1, 2, 3, 4, 6, 8, 11]
```

当前实现直接逐元素相加：

```text
f_fused = f_1 + f_2 + f_3 + f_4 + f_6 + f_8 + f_11
```

没有逐层 L2 归一化或 LayerNorm。各层范数差异很大，实际接近由深层特征主导。

## 4. 可学习模块

只训练：

- 32 个 layer-0 可学习 token；
- 特征映射模块；
- 768×768 可学习度量矩阵；
- 辅助分割头。

DINOv2 主干保持冻结。

### 4.1 可学习 token

每类初始化：

```text
32 个 token
shape = [32,768]
init std = 0.02
```

代码保留原项目的重复追加行为：

```text
layer-0 的 32 个 token 被追加两次
```

经过 ViT block 后得到 64 个图像条件 token。推理时再加上 32 个全局原型锚点：

```text
64 个图像条件 token
+ 32 个全局原型锚点
= 96 个可学习侧 token
```

因此准确说法是：

> 64 个图像条件层 token + 32 个全局原型锚点。

不是 96 个完全固定、与测试图无关的全局参数原型。

### 4.2 特征映射

当前 `res_mapping=0`，没有残差旁路：

```text
LayerNorm(768)
→ Linear(768,128)
→ GELU
→ LayerNorm(128)
→ Linear(128,768)
→ 768×768 可学习度量矩阵
```

度量矩阵初始化为单位阵，之后随训练更新。它同时作用于 patch 和可学习侧 token。

## 5. 训练前向

每个 iteration 使用：

```text
正常 batch = 16 // 2 = 8
异常 batch = 16
```

正常分支产生：

- fused patch feature；
- 64 个图像条件 token；
- 32 个全局 token；
- 共 96 个可学习侧 token。

异常分支产生：

- fused patch feature；
- 96 个可学习侧 token；
- 28×28 patch-level GT mask。

Patch 与 token 均经过 `token_mapping` 和可学习度量矩阵。

## 6. 训练损失

当前实际启用的总损失：

```text
L =
  0.5 · L_scaled_cos
+ 1.0 · L_mse
+ 1.0 · L_push_top10
+ 0.1 · L_div
+ 0.02 · L_rank
+ 1.0 · L_seg
```

其他 cluster、distill、EMA、pixel BCE、peak、prod、barlow 等损失权重均为 0。

### 6.1 正常 patch 软原型混合

正常 patch 与 96 个 token 计算余弦相似度，温度 0.07：

```text
weights = softmax(similarity / 0.07)
soft_token = Σ weights · prototype_token
```

计算：

```text
L_mse = MSE(soft_token, normal_patch)
cos_dist = 1 - cos(normal_patch, soft_token)
```

难 patch 动态加权：

```text
ratio = cos_dist / mean(cos_dist)
ratio = min(ratio, 5)
weight = ratio³
L_scaled_cos = mean(weight · cos_dist)
```

### 6.2 合成异常 patch 推离

异常 patch 与正常 token 比较，取最近 top-10 token 的平均相似度：

```text
sim_top10 = mean(top10 similarity)
```

异常 mask 内的 hinge：

```text
L_push =
  mean(mask · relu(sim_top10 - 0.5))
```

### 6.3 原型多样性

96 个 token 两两计算余弦相似度，惩罚超过 0.5 的部分：

```text
L_div =
  mean(relu(cos(token_i, token_j) - 0.5))
```

### 6.4 跨图排序一致性

从正常像素取最高 256 个分数，从异常像素取最低 256 个：

```text
L_rank =
  mean(relu(
    (normal_score - abnormal_score) / IQR(normal_score)
    + 0.25
  ))
```

其中：

```text
IQR = p75(normal_score) - p25(normal_score)
```

正常像素包括：

- 正常图的全部 patch；
- 异常图中 mask=0 的 patch。

### 6.5 辅助分割损失

分割头：

```text
Linear(768,384)
→ ReLU
→ Linear(384,1)
→ Sigmoid
```

使用 Focal + Dice：

```text
L_seg = 0.5 · L_focal + 0.5 · L_dice
Focal alpha = 0.25
Focal gamma = 2
```

分割头只在训练中使用，不进入最终异常图。

## 7. 优化配置

```text
epochs       = 35
seed         = 1
optimizer    = StableAdamW
lr           = 1e-3
final lr     = 5e-5
warmup       = 100 iterations
weight decay = 1e-3
grad clip    = 0.5
```

固定使用 epoch 35 checkpoint，不按测试集选择最优轮。

## 8. mapping 推理

测试图预处理：

```text
Resize(448,448)
不裁剪
```

mapping 只跑一个视图：

```text
448 → bilinear 1.5× → 672
不翻转
```

存在明确的训练/推理几何差异：

```text
训练：392 输入 → 28×28 patch
推理：672 输入 → 48×48 patch
```

### 8.1 raw 正常 patch 库

使用严格 4-shot 的原始正常图，在 448 分辨率提取 7 层特征并相加：

```text
4 张图 × 32×32 patch
= 4096 个正常 patch
```

每个 patch 做 L2 归一化，不做增强，不使用测试数据。

### 8.2 raw 分数

测试图在 672 上产生：

```text
48×48 = 2304 个测试 patch
```

每个测试 patch 与全部 4096 个正常 patch 计算余弦距离：

```text
d1 = min distance to raw bank
```

另外固定随机抽取 1024 个 bank patch，计算各测试 patch 到该子库距离的第 512 小值：

```text
q512
```

raw 分数：

```text
s_raw = sqrt(d1 · q512)
```

注意：测试 patch 是 672，而 raw bank 是 448，原方案保留了这一尺度差。

### 8.3 learnable 分数

测试 patch 与 96 个可学习侧 token 都经过 `token_mapping`：

```text
s_learn =
  min_k [1 - cos(mapped_patch, mapped_token_k)]
```

当前 `l_global=0`，没有 learnable 侧全局分位项。

### 8.4 mapping 总分数

```text
mapping_score =
  0.6 · s_learn
+ 0.4 · s_raw
```

672 上的分数图随后双线性插值回 448，落盘为 mapping 异常图。

## 9. MemoryBank 分支

MemoryBank 使用：

```text
facebook/dinov2-base
```

输入为同样 4 张 shot，每张使用 22 个增强视图：

```text
4 × 22 = 88 个正常视图
```

两个特征阶段：

```text
stage 1 = hidden layers [2,3,4,5] 相加
stage 2 = hidden layers [6,7,8,9] 相加
```

每个 patch 做 L2 归一化，建立两个 FAISS inner-product 最近邻库。

测试图直接 Resize 672。每个测试 patch 分别查询两个库：

```text
d_stage = 1 - max_cosine
```

两个阶段图插值到 672 后取平均：

```text
bank_score =
  0.5 · d_stage1
+ 0.5 · d_stage2
```

建库增强已固定随机种子，保证相同 shot 和 seed 可复现。

## 10. 最终融合

mapping 图为 448，bank 图为 672：

```text
mapping_672 = bilinear_resize(mapping_448, 672)
bank_672    = 原 672 bank 图
```

分别使用 25×25 平均池化：

```text
F =
  0.5 · box25(mapping_672)
+ 0.5 · box25(bank_672)
```

当前方案没有：

- 单图 min-max；
- 单图 z-score；
- 全类别测试集 z-score；
- 测试集统计量归一化。

随后执行：

```text
Gaussian(k=5, sigma=4)
```

## 11. 图像分数与指标

图像异常分数：

```text
score_image =
  mean(top 0.5% pixels of Gaussian anomaly map)
```

计算：

- I-AUROC
- I-AP
- I-F1
- P-AUROC
- P-AP
- P-F1
- P-AUPRO

每类的七项指标独立计算，再对类别做宏平均。

其他口径：

- 融合时 GT 先 bilinear Resize 到 672，再以 `>0.5` 二值化；
- AUPRO 使用 `adeval`，`nstrips=200`；
- AUPRO 积分上限为 FPR=0.3。

## 12. 已知风险与限制
1 名义上使用 7 层特征，但直接相加使深层 layer 11 数值主导。
2 96 个可学习侧 token 中有 32 个重复 token。
3 当前结果仍是单 seed，尚未报告训练随机性方差。

## 13. 具体代码与实验路径

以下路径均对应当前严格 MVTec 4-shot 实验。

### 13.1 训练入口与训练损失

主训练文件：

```text
/home/jjquan/experiment/visual/audit_mapping/code_vpt4/main-fewshot-vpt4.py
```

关键位置：

- 训练和测试预处理：
  `main-fewshot-vpt4.py:141-153`
- 22 倍正常图增强：
  `main-fewshot-vpt4.py:107-130`
- raw 正常 patch 库：
  `main-fewshot-vpt4.py:730-740`
- 正常/异常 batch 的原型构造：
  `main-fewshot-vpt4.py:2077-2122`
- `L_div`：
  `main-fewshot-vpt4.py:2164-2174`
- 辅助分割损失：
  `main-fewshot-vpt4.py:2175-2178`
- `L_rank`：
  `main-fewshot-vpt4.py:2196-2207`
- 完整训练循环：
  `main-fewshot-vpt4.py:2031-2524`
- 模型、优化器和 35 epoch 配置：
  `main-fewshot-vpt4.py:1923-2004`

### 13.2 可学习原型与映射结构

文件：

```text
/home/jjquan/experiment/visual/audit_mapping/code_vpt4/dinov2/models/vision_transformer.py
```

关键位置：

- `LearnableTokens`，32 个可学习 token 初始化：
  `vision_transformer.py:49-70`
- `TokenMapping`，低秩映射和可学习度量矩阵：
  `vision_transformer.py:73-118`
- 可学习 token 与 `token_mapping` 的实例化：
  `vision_transformer.py:320-337`
- ViT block 中 token 的重复追加和特征收集：
  `vision_transformer.py:424-465`

### 13.3 训练损失的数学实现

文件：

```text
/home/jjquan/experiment/visual/audit_mapping/code_vpt4/main-fewshot-vpt4.py
```

- `ScaledPatchCosineLoss3`：
  `main-fewshot-vpt4.py:1123-1224`
- `proto_distance_scores`：
  `main-fewshot-vpt4.py:1274-1286`
- 像素级跨图核化排序损失：
  `main-fewshot-vpt4.py:1289-1349`
- `FocalDiceLoss`：
  `main-fewshot-vpt4.py:1403-1442`

### 13.4 mapping 推理与异常图生成

文件：

```text
/home/jjquan/experiment/visual/audit_mapping/eval_score_v3.py
```

关键位置：

- 推理参数与默认值：
  `eval_score_v3.py:74-165`
- 4-shot raw 正常 patch 库：
  `eval_score_v3.py:227-303`
- raw 分数 `sqrt(d1*q512)`：
  `eval_score_v3.py:401-438`
- 672 单视图 mapping 推理：
  `eval_score_v3.py:440-589`
- mapping 异常图保存：
  `eval_score_v3.py:596-609`
- mapping 分支训练内指标：
  `eval_score_v3.py:611-621`

### 13.5 MemoryBank 分支

建库和查询入口：

```text
/home/jjquan/experiment/visual/audit_mapping/cache_bank_nocrop.py
```

关键位置：

- 672 正常图预处理：
  `cache_bank_nocrop.py:158-170`
- 4-shot × 22 增强和建库：
  `cache_bank_nocrop.py:165-191`
- 672 测试图查询：
  `cache_bank_nocrop.py:195-227`
- bank 异常图保存：
  `cache_bank_nocrop.py:228-240`

MemoryBank 核心实现：

```text
/home/jjquan/experiment/visual/audit_mapping/code_vpt4/util/memorybank.py
```

关键位置：

- 两组 hidden layers 的特征融合：
  `memorybank.py:221-238`
- `MemoryBank` 类：
  `memorybank.py:240-423`
- 两阶段正常特征建库：
  `memorybank.py:394-423`
- 两阶段最近邻查询与平均：
  `memorybank.py:471-506`

### 13.6 raw 672 融合和指标

文件：

```text
/home/jjquan/experiment/visual/audit_mapping/sweep_fusion_advanced.py
```

关键位置：

- 类内/逐图 z-score 与 identity：
  `sweep_fusion_advanced.py:32-43`
- 两分支融合候选与 `box25`：
  `sweep_fusion_advanced.py:106-120`
- 672 mapping/bank 加载与插值：
  `sweep_fusion_advanced.py:289-317`
- map/bank sidecar 样本重排：
  `sweep_fusion_advanced.py:318-332`
- Gaussian、top-0.5% 和指标调用：
  `sweep_fusion_advanced.py:333-364`

指标定义：

```text
/home/jjquan/experiment/visual/audit_mapping/code_vpt4/main-fewshot-vpt4.py
```

- Gaussian kernel：
  `main-fewshot-vpt4.py:429-462`
- F1 与 `ader_evaluator`：
  `main-fewshot-vpt4.py:464-511`

### 13.7 严格 shot 与数据构造

训练、mapping 和 bank 共用的 shot 工具：

```text
/home/jjquan/experiment/visual/audit_mapping/shot_utils.py
```

严格 4-shot 数据集与异常数据构造：

```text
/home/jjquan/experiment/visual/audit_mapping/build_mvtec4_strict_existing.py
```

严格 shot 清单：

```text
/home/jjquan/experiment/visual/audit_mapping/.shots_mvtec4_strict/
```

严格数据集视图：

```text
/home/jjquan/experiment/visual2/MVTec-AD_strict4/
```

严格异常训练数据：

```text
/home/jjquan/experiment/visual2/anomaly_samples_mvtec_4shot_strict/
```

原始 4-source AnoStyler 数据：

```text
/home/jjquan/AnoStyler/genbyAnoStyler/
```

严格 4-shot source 清单：

```text
/home/jjquan/experiment/visual/audit_mapping/logs/mvtec4_strict_existing_manifest.json
```

### 13.8 本实验运行脚本

完整流水线：

```text
/home/jjquan/experiment/visual/audit_mapping/run_mvtec4_strict_existing_pipeline.sh
```

35 epoch 训练：

```text
/home/jjquan/experiment/visual/audit_mapping/run_mvtec4_strict_35e.sh
```

mapping 与 672 bank 推理：

```text
/home/jjquan/experiment/visual/audit_mapping/run_mvtec4_strict_eval.sh
```

raw 672 融合：

```text
/home/jjquan/experiment/visual/audit_mapping/run_mvtec4_strict_fusion.sh
```

### 13.9 checkpoints、异常图与日志

严格 4-shot mapping checkpoints：

```text
/home/jjquan/experiment/visual/audit_mapping/results/vpt4/vpt37_mvtec4shot_strict_promptv2_seed1/
```

448 mapping 异常图：

```text
/home/jjquan/experiment/visual/audit_mapping/.maps_mvtec4_strict/
```

672 MemoryBank 异常图：

```text
/home/jjquan/experiment/visual/audit_mapping/.errmap672_mvtec4_strict/
```

逐类训练日志：

```text
/home/jjquan/experiment/visual/audit_mapping/logs/train_mvtec4strict_vpt37_mvtec4shot_strict_promptv2_seed1_*.log
```

mapping/bank 推理日志：

```text
/home/jjquan/experiment/visual/audit_mapping/logs/mvtec4_strict_eval/
```

raw 672 最终指标：

```text
/home/jjquan/experiment/visual/audit_mapping/logs/mvtec4_strict_raw672.csv
```

流水线总日志：

```text
/home/jjquan/experiment/visual/audit_mapping/logs/mvtec4_strict_pipeline_driver.log
```

### 13.10 相关审计记录

方案与代码的一致性核对：

```text
/home/jjquan/experiment/visual/audit_mapping/logs/严格4shot方案与代码一致性核对_20261020.md
```

严格数据源、shot 顺序和缓存一致性审计：

```text
/home/jjquan/experiment/visual/audit_mapping/logs/pipeline_end_to_end_audit.json
/home/jjquan/experiment/visual/audit_mapping/logs/synthetic_sources_all_shots.json
/home/jjquan/experiment/visual/audit_mapping/logs/mapping672_cache_audit.json
```
