# vpt3.0 仓库文件说明

## 顶层
| 文件 | 说明 |
|---|---|
| `VERSION` | 版本号与一句话构成说明（`vpt3.0`） |
| `README.md` | 使用说明：启用内容、运行命令、15 类官方指标、验证方式 |
| `ENVIRONMENT.md` | 硬件/系统/依赖/数据路径/复现提示（含本机 GPU 已知故障说明） |
| `.gitignore` | 排除大文件与产物（`results/`、`.maps_audit/`、`.errmap_audit/`、`*.pth`、`*.map` 等）——**本仓库只保留代码、文档与日志** |

## `code/`（可运行代码，vpt3.0 主体）
| 文件 | 说明 |
|---|---|
| `main-fewshot-vpt3.py` | 主脚本：训练（control 配方 35 轮 / 4-shot）+ 评估。**vpt3.0 新增**：`build_raw_bank` / `compute_fused_map` / `evaluation_batch_vpt3`，开关 `--vpt3 / --vpt3_w / --vpt3_scales / --vpt3_flip / --vpt3_cache_dir`；默认启用 bank 分支与类内 z 加权融合（0.3/0.7） |
| `dinov2/models/vision_transformer.py` | DINOv2 ViT-B/14 骨干 + 本项目改造：`LearnableTokens`（32 个可学习 token 注入 block-0）、`token_mapping`（768→128→768 低秩映射） |
| `dinov2/…`、`util/…`、`ucc/…`、`dataset1.py` | 从原仓库复制的依赖（`util/memorybank.py` 为原版 MemoryBank，`util/dataset.py` 含辅助异常集 Dataset） |

## `tools/`（评估器与探针，均为本次审计产出）
| 文件 | 说明 |
|---|---|
| `eval_offline.py` | 离线评估已保存 checkpoint（`--type` 1/2/4、`--fuse_bank/--fuse_z/--fuse_w/--ms_scoring`、`--cpu_metrics`） |
| `fuse_eval.py` | **只读异常图缓存**做任意融合规则评估（GPU/官方指标含 P-AUPRO） |
| `fuse_sweep_cpu.py` | 同上但走 CPU numpy 指标（无 P-AUPRO），一次遍历算完 19 条规则 |
| `cache_maps.py` | 缓存 mapping 分支异常图 + GT + meta（含 bank 图复用） |
| `cache_improved.py` | 缓存**改善后 mapping** 的异常图（融合 + 1.5× + 翻转） |
| `eval_bank_combo.py` | 原版 MemoryBank × mapping 的组合评估（type=1/3/5/7） |
| `test_vpt3.py` | **vpt3.0 端到端验证**：给定 checkpoint 跑一遍融合评估（与 `fuse_eval --rule zwsum:0.3` 对齐） |
| `gpu_guard.py` + `common_env.sh` | GPU UUID 断言与统一环境变量（防止误用同事的卡） |
| `select_best_epoch.py` | 按内部评估选最优轮（并给出 oracle 上界） |
| `probe_fusion.py` | 诊断：多层求和融合是否退化成只用最后一层 |
| `probe_prototypes.py` / `probe_coreset.py` | 诊断：可学习原型坍缩、原型来源替换实验 |
| `probe_failures.py` | 失败样本分析（逐缺陷类型的 GT 内外对比度、最严重漏检/误报） |
| `probe_scale.py` / `probe_scoring.py` / `probe_tta.py` | 多尺度/TTA/打分变体的对照实验 |
| `probe_identity.py` / `probe_rawbank.py` | CPU 指标实现与"原始特征库 vs 可学习原型"消融 |
| `verify_dice_loss.py` | 分割头 Dice 口径（整 batch vs 逐样本）的 CPU 验证 |

## `docs/`（完整记录）
| 文件 | 说明 |
|---|---|
| `EXPERIMENT_LOG.md` | 全部实验日志（§1–§21）：A/B、多尺度、选轮、融合、vpt3.0 版本冻结与验证 |
| `CHANGELOG.md` | 逐条修改说明（文件/行号/原因/预期），FIX-1…FIX-14 与各候选 |
| `README_审计结论.md` | 结论速览与头条指标 |
| `RUN_STATUS.md` | 运维记录：GPU 故障时间线、磁盘清理、待跑队列 |
| `environment.yaml` | 原仓库 conda 环境文件（复制，供重建参考） |

## `logs/`（训练与评估日志）
146 个文本日志（约 2.8 MB）：`control35_<cls>.log`（35 轮训练）、`eval_*`（各口径评估）、`q_*_driver.log`（队列调度）、`bank15_*`/`cacheimproved_*`（异常图缓存过程）等。
**大体积产物不入库**：`results/`（权重与 CSV）、`.maps_audit/`、`.errmap_audit/`（异常图缓存）、`visualization/` 均在 `.gitignore` 中；它们可由上述脚本重新生成。
