# Support-Structure Multi-Scale Fusion

日期：2026-10-26

## 方法

该方法替代原有硬阈值 Support-Adaptive，只使用 K-shot 正常样本估计门控，
不使用测试图像、测试标签或测试集统计量进行融合权重选择。

### 1. Support structure

在正常 support patch 特征上计算：

```text
c = cosine(中心区域平均特征, 边缘区域平均特征)
r = 中心区域平均特征范数 / 边缘区域平均特征范数
```

两个数据集统一使用连续权重：

```text
w_m = clip((1 - c) * r, 0, 1)
w_b = 1 - w_m
```

与原始 Support-Adaptive 的固定阈值 `c<0.25`、`r>=0.88` 和硬编码
`0.75/0.25` 相比，新方法移除了人工阈值、数据集专用门控和离散权重档位。

### 2. Multi-scale fusion

Mapping 和外部 Bank 分数图分别使用三个固定区域池化尺度：

```text
S = 1/3 * sum_{k in {9,25,41}}
        [ w_m * box_k(Mapping)
        + w_b * box_k(Bank) ]
```

随后执行 Gaussian(k=5, sigma=4)，图像异常分数取 top-0.5% 像素均值。
没有使用 z-score、min-max 或其他测试集归一化。

## 性能

| 设置 | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO | 七项均值 | Support-Adaptive | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MVTec 1-shot | 95.5680 | 98.1435 | 97.2800 | 97.9251 | 65.6588 | 64.6056 | 94.5836 | **87.6806** | 87.5583 | +0.1223 |
| MVTec 2-shot | 96.0196 | 98.3823 | 97.5838 | 98.0225 | 67.4277 | 66.4043 | 94.9384 | **88.3969** | 88.2868 | +0.1101 |
| MVTec 4-shot strict | 98.2852 | 99.0949 | 97.5924 | 97.9678 | 69.4419 | 68.0102 | 95.3743 | **89.3952** | 89.3671 | +0.0281 |
| VisA 1-shot | 94.4370 | 94.7150 | 90.9300 | 98.7470 | 47.4860 | 52.3180 | 95.5090 | **82.0201** | 81.8453 | +0.1748 |
| VisA 2-shot | 93.8040 | 94.1600 | 90.4790 | 98.8770 | 49.0360 | 53.1440 | 95.7200 | **82.1743** | 81.9533 | +0.2210 |
| VisA 4-shot | 95.0740 | 95.1100 | 91.6380 | 99.0160 | 49.5040 | 53.9280 | 96.3200 | **82.9413** | 82.8104 | +0.1309 |

六个设置的七项均值平均提升：

```text
(0.1223 + 0.1101 + 0.0281 + 0.1748 + 0.2210 + 0.1309) / 6
= +0.13120
```

所有六个设置均高于原 Support-Adaptive，平均提升 `+0.13120`。
统一公式后性能低于数据集专用门控版本，但不再需要任何数据集专用规则。

## 无测试集参数审计

- `c` 和 `r` 只读取 K-shot 正常 support 图像。
- 融合公式、池化尺寸和输出聚合在最终评测前固定。
- 权重计算不读取测试图像、测试标签、测试异常图统计或测试指标。
- 测试异常图只作为推理输入参与固定融合，不参与任何参数估计。

## 协议

- 单 seed，固定 epoch 35。
- Mapping 使用 1.5x 单视图，无翻转。
- MVTec 4-shot 使用 strict 4-shot，最终在 672 图空间评测。
- MVTec 2-shot 和 VisA 4-shot 的 checkpoint 为 VPT3.8-equivalent。

## 复现代码与结果

- 评测实现：`eval_support_structural_continuous.py`
- MVTec 1-shot：`logs/crossfit_normal_reliability_20261026/mvtec1_eval_continuous_multiscale.csv`
- MVTec 2-shot：`logs/crossfit_normal_reliability_20261026/mvtec2_eval_multiscale.csv`
- MVTec 4-shot：`logs/crossfit_normal_reliability_20261026/mvtec4_eval_continuous_multiscale.csv`
- VisA 1-shot：`logs/crossfit_normal_reliability_20261026/visa1_eval_unified_multiscale.csv`
- VisA 2-shot：`logs/crossfit_normal_reliability_20261026/visa2_eval_unified_multiscale.csv`
- VisA 4-shot：`logs/crossfit_normal_reliability_20261026/visa4_eval_unified_multiscale.csv`

## 限制

1. 当前为单 seed，未估计训练随机性和置信区间。
2. 统一连续公式相对数据集专用门控损失了部分平均增益。
3. 多尺度池化尺寸是固定算子设计，没有在测试集上逐类搜索。
4. MVTec 4-shot 单项 P-F1 略低于旧方法；总体提升来自其余指标和类别。
