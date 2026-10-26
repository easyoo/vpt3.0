# VPT3.9 ChangeLog

日期：2026-10-26

## 定位

VPT3.9 保持 VPT3.8 的主干、训练流程、损失和 checkpoint 结构，只替换推理端
Support-Adaptive 融合规则。

## 核心改动

1. 删除数据集专用阈值 `c<0.25`、`r>=0.88`。
2. 删除固定离散权重 `0.50/0.75`。
3. 统一使用 support-only 连续权重：

```text
w_m = clip((1-c) * r, 0, 1)
w_b = 1-w_m
```

4. 使用固定 9/25/41 多尺度池化：

```text
S = 1/3 * sum_k [w_m * box_k(Mapping) + w_b * box_k(Bank)]
```

5. 三个尺度共享同一异常图，不增加模型前向次数。
6. 融合参数不读取测试图像、测试标签或测试集统计量。

## 性能

| 设置 | VPT3.9 七项均值 | Support-Adaptive | 提升 |
|---|---:|---:|---:|
| MVTec 1-shot | 87.6806 | 87.5583 | +0.1223 |
| MVTec 2-shot | 88.3969 | 88.2868 | +0.1101 |
| MVTec 4-shot strict | 89.3952 | 89.3671 | +0.0281 |
| VisA 1-shot | 82.0201 | 81.8453 | +0.1748 |
| VisA 2-shot | 82.1743 | 81.9533 | +0.2210 |
| VisA 4-shot | 82.9413 | 82.8104 | +0.1309 |

六组全部高于原 Support-Adaptive，平均提升 `+0.1312`。

## 吞吐量

RTX 3090，`transistor`，100 张测试图，纯推理阶段：

| 设置 | 学习分支 | 完整推理 |
|---|---:|---:|
| 2-shot batch=1 | 16.33 img/s | 5.82 img/s |
| 2-shot batch=4 | 18.47 img/s | 6.42 img/s |
| 4-shot batch=1 | - | 4.66 img/s |
| 4-shot batch=4 | - | 5.12 img/s |

## 审计限制

- 当前仍为单 seed。
- MVTec 2-shot 与 VisA 4-shot checkpoint 为 VPT3.8-equivalent。
- 文档性能来自论文结果与本项目单点评测的混合比较，不是严格同实现复现。
