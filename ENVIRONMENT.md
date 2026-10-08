# vpt3.0 运行环境（2026-10-08 冻结记录）

## 1. 硬件
| 项目 | 值 |
|---|---|
| 机型 | SuperMicro **X11DPG-QT**（BIOS 4.0 / 2023-07-11） |
| GPU | **NVIDIA RTX 3090 ×4**（24 GB 显存/卡；`nvidia-smi` 编号 0/1/2 与 UUID 见下） |
| 驱动 | **580.178.04**，CUDA 12.2（`nvidia-smi` 报告） |
| GPU UUID | `GPU-0e3a7e1c-...`(3B:00.0) / `GPU-3bf0407c-...`(5E:00.0) / `GPU-8e8231eb-...`(86:00.0) |
| 已知硬件问题 | **`86:00.0` 那张卡在高负载下会 `Xid 79 GPU has fallen off the bus`**，并触发整组 `Xid 154 / Node Reboot Required` → 必须冷启动（AC power cycle）恢复。已发生 5 次（10-04 / 10-06 / 10-07 / 10-08×2）。**建议拔卡或换槽后再做长跑**；单卡训练时限制功耗（`nvidia-smi -pl 280`）可降低触发概率 |
| 推荐用法 | 用 UUID 绑定目标卡（本仓库 `tools/gpu_guard.py` + `tools/common_env.sh` 会在启动前断言 UUID，防止误用同事的卡） |

## 2. 系统
| 项目 | 值 |
|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS |
| 内核 | `6.17.0-40-generic` |
| Python | 3.10.20（conda 环境 **`vpt`**：`/home/jjquan/miniconda3/envs/vpt/bin/python`） |
| 其它 conda 环境 | `anostyler`（生成辅助异常集用） |

## 3. 关键依赖版本（`pip` 元数据实测）
```
torch            2.5.0+cu121      torchvision 0.20.0+cu121     torchaudio 2.5.0+cu121
(torch.cuda 12.1, cudnn 90100)
numpy            1.26.4            scipy       1.15.3          scikit-learn 1.7.2
faiss-cpu        1.13.2            faiss-gpu   1.7.2
transformers     5.17.0            timm        1.0.25          einops 0.8.2
kornia           0.8.2             opencv-python 4.13.0.92     Pillow 12.1.1
tqdm             4.67.3            adeval      1.1.0
```
* `adeval` 提供官方指标累加器（**P-AUPRO 依赖它，且必须跑在 CUDA 上**；CPU 评估时用 `tools/fuse_sweep_cpu.py` 的 numpy 实现，无 P-AUPRO）。
* MemoryBank 分支依赖 **本地缓存的 `facebook/dinov2-base`**（HuggingFace `AutoModel`，离线加载）+ **faiss**；实测离线可用。

## 4. 数据路径
| 用途 | 路径 |
|---|---|
| 测试/正常图（MVTec-AD 15 类） | `/home/jjquan/MVTec-AD/<cls>/{train/good, test, ground_truth}` |
| 辅助合成异常集（仅训练 mapping 分支用） | `/home/jjquan/experiment/visual2/anomaly_samples/mvtec/<cls>/<anomaly_type>/{image,mask}` |
| 预训练骨干权重 | `/home/jjquan/Downloads/dinov2_vitb14_reg4_pretrain.pth`（原路径 `/home/jjquan/codebase/vpt/checkpoint/...` 已失效，vpt3.0 按候选列表回退） |
| 参考环境文件 | `docs/environment.yaml`（复制自原仓库，供重建 conda 环境参考） |

## 5. 复现提示
```bash
# 训练 + vpt3.0 口径评估（35 轮，4-shot；bank 分支默认启用）
CUDA_VISIBLE_DEVICES=<目标卡的UUID> \
python code/main-fewshot-vpt3.py --train_class screw --total_epochs 35 --internal 5 \
  --save_dir results/vpt3 --save_name run1 \
  --data_path /home/jjquan/MVTec-AD \
  --anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples
```
* 训练脚本默认 `--vpt3 1`（融合评估）、`--vpt3_w 0.3`、`--vpt3_scales 1.5`、`--vpt3_flip 1`、`--map_only` **关闭**。
