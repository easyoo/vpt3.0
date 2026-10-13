# vpt3.0 使用说明（2026-10-08 冻结）

## 这一版启用了什么
| 组件 | 状态 |
|---|---|
| 训练配方 | 原版 control：35 轮、4-shot、aux 合成异常集，**`--map_only` 默认关闭**（bank 分支启用） |
| 改善后 mapping 分支 | ✅ 打分图 = `0.5·可学习原型距离图 + 0.5·原始特征空间正常 patch 库距离图`，多尺度 `1.5×` + 水平翻转（2 次前向） |
| 原版 MemoryBank 分支 | ✅ `facebook/dinov2-base`（AutoModel）`hidden_states[2:6]`/`[6:10]` 双 stage + faiss，输入为 4 张 shot × 原版 22 倍增强 = 88 张 |
| 融合规则 | ✅ **类内 z 标准化后加权**：`0.3·z(mapping) + 0.7·z(bank)`（z 用整类全部像素的 μ/σ，非单图） |
| 结果落盘 | ✅ 两个分支的逐图异常图会缓存（`--vpt3_cache_dir`），便于换融合规则复算 |

## 运行
```bash
# 训练 + 每 5 轮用 vpt3.0 口径评估（默认行为）
python code_vpt3/main-fewshot-vpt3.py --train_class screw --total_epochs 35 --internal 5 \
  --save_dir results/vpt3 --save_name run1 --data_path /home/jjquan/MVTec-AD \
  --anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples \
  --vpt3_cache_dir /path/to/cache        # 可选：缓存异常图
```
关键开关：`--vpt3 1`（默认开）、`--vpt3_w 0.3`、`--vpt3_scales 1.5`、`--vpt3_flip 1`。

## 15 类官方指标（统一 epoch=35，MVTec，392 口径）
| 指标 | vpt3.0（0.3/0.7） | 仅 bank | 仅改善 mapping | 原管线口径 |
|---|---|---|---|---|
| I-AUROC | 0.9802 | 0.9803 | 0.9586 | 0.9417 |
| I-AP | 0.9855 | 0.9898 | 0.9719 | 0.9580 |
| I-F1 | 0.9708 | 0.9721 | 0.9559 | 0.9424 |
| P-AUROC | 0.9734 | 0.9666 | 0.9640 | 0.9115 |
| **P-AP** | **0.6420** | 0.6368 | 0.5650 | 0.4296 |
| P-F1 | **0.6503** | 0.6392 | 0.5950 | 0.4732 |
| P-AUPRO | **0.9433** | 0.9327 | 0.9107 | 0.8526 |

（"原管线口径" = 35 轮末 + 原始 type=1 打分；vpt3.0 相对它 **P-AP +0.2124 / AUPRO +0.0907**）

## 已冻结的依赖与环境
- 仅使用 **GPU**（默认 `cuda:0`，建议用 `CUDA_VISIBLE_DEVICES` 或 `AUDIT_GPU_UUID` 绑定；工程副本另含 UUID 断言脚本 `gpu_guard.py`）
- bank 依赖本地缓存的 `facebook/dinov2-base` 与 `faiss`（均已验证可用）
- 数据：MVTec-AD + `visual2/anomaly_samples/mvtec/<cls>`

## 验证
`test_vpt3.py`（carpet@35）与参考实现 `fuse_eval.py --rule zwsum:0.3` **逐位一致**：
`I-AUROC=0.9992 I-AP=0.9998 I-F1=0.9944 P-AUROC=0.9939 P-AP=0.7090 P-F1=0.7187 P-AUPRO=0.9840`
