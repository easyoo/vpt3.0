# vpt3.5 使用说明（2026-10-09 冻结）

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
