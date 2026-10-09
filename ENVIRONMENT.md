# 环境与依赖（vpt3.5 复现所需）

## 硬件
| | |
|---|---|
| GPU | 2 × NVIDIA GeForce RTX 3090（24 GB）× 可用（另有 1 张掉总线的故障卡） |
| GPU UUID | `GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822`（3B:00.0）、`GPU-3bf0407c-aa78-8aab-cd2f-1c7b3356c5be`（5E:00.0） |
| 驱动 / CUDA | 驱动 580.178.04，系统 CUDA 13.0（PyTorch 自带 cu121 运行时） |

> ⚠️ 本机按**数字索引**设置 `CUDA_VISIBLE_DEVICES` 会错位（故障卡导致枚举异常），必须用 **UUID** 绑定；
> 仓库内 `tools/gpu_guard.py` 会在进程内断言当前卡 UUID 与预期一致。

## 软件
| 组件 | 版本 |
|---|---|
| Python | 3.10.20（`/home/jjquan/miniconda3/envs/vpt`） |
| PyTorch / torchvision | 2.5.0+cu121 / 0.20.0+cu121 |
| transformers | 5.17.0 |
| faiss | 1.13.2 |
| adeval | 官方包（指标实现，路径 `site-packages/adeval`） |
| timm / opencv / scikit-learn / numpy | 1.0.25 / 4.13.0 / 1.7.2 / 1.26.4 |

## 数据
| 用途 | 路径 |
|---|---|
| MVTec-AD（15 类，单类评估） | `/home/jjquan/MVTec-AD` |
| 辅助合成异常集（AnoStyler，4-shot 训练用） | `/home/jjquan/experiment/visual2/anomaly_samples/mvtec` |
| DINOv2 预训练权重 | `/home/jjquan/Downloads/dinov2_vitb14_reg4_pretrain.pth`（或列表内其它同名权重） |
| MemoryBank 主干 | `facebook/dinov2-base`（HuggingFace `AutoModel`，离线需本地缓存） |

## 关键预处理口径（v3.5）
| 阶段 | 变换 |
|---|---|
| 训练图 / GT | `Resize(448) → ToTensor → CenterCrop(392) → Normalize(ImageNet)` |
| 测试图（本版改动） | `Resize(448) → ToTensor → Normalize(ImageNet)`，**不做 CenterCrop** |
| 测试 GT | `Resize(448) → ToTensor`（与预测图逐像素对齐） |
| 指标 | 官方 `adeval`（含 P-AUPRO，FPR≤0.3 的 per-region overlap 平均）；图像级分数 = 最热 1% 像素均值 |

## 数据一致性保证（v3.5 新增）
* `tools/shot_utils.py` 是 shot 清单的**唯一来源**：`get_shots(data_path, cls, shots=K, seed=S)`，
  首次生成后落盘 `.shots/<cls>_shotK_seedS.json`；训练、mapping 特征库、bank 建库三处共用。
* 改 shot 数（4→8/16）时三处自动取同一批图；清单哈希会打进训练日志（`[shots] cls shotK hash=...`）。
