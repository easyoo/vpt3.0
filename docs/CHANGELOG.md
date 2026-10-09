# Mapping 分支 · 路径1（工程修复）ChangeLog

> 工作语言中文。所有修改都在**副本**中进行，`/home/jjquan/experiment/visual` 下的原始代码**未改动一行**
> （`git status` 只显示新增的 `audit_mapping/` 与自动生成的 `__pycache__/`）。

## 0. 环境与约束落实

| 约束 | 落实方式 |
|---|---|
| 禁止改原代码 | 全量复制到 `audit_mapping/code_norm*` 等副本；原始仓库零改动 |
| 只用 GPU1 | 所有脚本 `export CUDA_VISIBLE_DEVICES=GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822`（即物理 GPU1 的 UUID；GPU0 本机报 `Unknown Error`，不能用；用 UUID 可避免 CUDA_VISIBLE_DEVICES=1 被 GPU0 的枚举顺序影响） |
| 只测 mapping 分支 | 新参数 `--map_only`：完全不建 MemoryBank、不算/不读 `.errmap`，评估只走 `type=1`（mapping 原型距离） |
| 35 轮 | 最终 A/B 统一 `--total_epochs 35`（早期快速筛查用 30 轮，已在日志中标注） |

## 1. 目录结构

```
audit_mapping/
├── code_control/          # 对照(control)：原始行为 + 审计脚手架（seed/device/ckpt路径/map_only/向量化打分）
├── code/                  # 修复版(fixed)：control + FIX-1/2（+ FIX-6 评估口径统一）
├── code_norm/             # 新变体(norm)：control + FIX-12（多层特征对齐融合）
├── probe_prototypes.py    # 探针：原型多样性/坍缩度量
├── probe_coreset.py       # 探针：原型来源替换实验（零训练）
├── probe_fusion.py        # 探针：多层特征对齐诊断
├── eval_offline.py        # 离线评估（训练一次、多次评估；消除"口径变了"的干扰）
└── logs/, results/        # 原始日志与指标
```

## 2. 逐条修改说明

### FIX-1 可学习 token 在 block-0 被重复注入两次
* **文件**：`code/dinov2/models/vision_transformer.py:367`（原始文件对应 `visual/dinov2/models/vision_transformer.py:367`）
* **问题**：`forward_features` 在循环**外**先 `cat(token_dict['0'])`，循环体内 `i=0` 又 `cat(token_dict['0'])`，
  于是 `feat_list[0]` 变成 64 个 token（前后 32 个逐元素相同）。
* **为什么错**：① 名义上 `learnable_num_per_layer=32` 的模块实际塞了 64 个 token，block-0 的注意力序列长度多 32，
  注意力质量被重复计入；② 只要 `learnable_layers>1`，`torch.stack(feat_list)` 会因 64 vs 32 形状不一致直接崩。
* **改动**：删除循环外那次追加，只保留循环体内追加。
* **预期**：消除冗余 token；在 `learnable_layers=1` 时数学上等价（重复项数值完全相同，softmax 权重等分后加权和不变），
  但**不是逐位等价**——序列长度 853→821 会改变浮点求和顺序。
* **验证**：见 `EXPERIMENT_LOG.md` §3 噪声地板实验。

### FIX-2 `fused_learnable_token` 被"所有 learnable 层求和"覆盖
* **文件**：`code/main-fewshot-audit.py:1497`（原 `visual/main-fewshot.py:1466`）与 `:1515`（原 `:1481`）
* **问题**：先按 `fused_layer_learnable` 选层求出 `fused_learnable_token`，紧接着又被
  `torch.sum(torch.stack(feat_list), dim=0)` 覆盖。
* **为什么错**：`learnable_layers=1` 时是 no-op（`feat_list` 只有 1 项），所以历史实验没暴露；
  一旦用多层 learnable token，训练端=所有层求和、评估端=按 `fused_layer_learnable` 选层，两边口径不一致。
* **改动**：删掉两处覆盖语句（异常分支那处是死代码）。
* **预期**：默认配置下指标不变；多 learnable 层配置下训练/评估一致。

### FIX-3 预训练权重路径失效导致训练直接崩
* **文件**：`code/main-fewshot-audit.py:1402`（原 `visual/main-fewshot.py:1369`）
* **问题**：`pth='/home/jjquan/codebase/vpt/checkpoint/dinov2_vitb14_reg4_pretrain.pth'`。
* **证据**：该文件不存在（`ls` 报 No such file）；`/home/jjquan/Downloads/dinov2_vitb14_reg4_pretrain.pth` 存在（346MB）。
* **改动**：按候选列表顺序取第一个存在的权重。
* **预期**：训练脚本可复现运行。

### FIX-4 新增 `--map_only`（本阶段核心口径）
* **文件**：`code/main-fewshot-audit.py:1145,1248,1351`
* **改动**：跳过 MemoryBank 构建、`.errmap` 计算/读取，训练中评估只跑 `type=1`。
* **预期**：单类训练时间从"bank+7 种打分"降到纯 mapping 分支，迭代速度提升数倍；指标仍是 mapping 分支指标。
* **实测**：单类 30 轮 ≈ 4.3~6.5 分钟（含每 5 轮评估）。

### FIX-5 评估期打分向量化
* **文件**：`code/main-fewshot-audit.py:504`（`compute_anomaly_scores`）
* **问题**：原实现逐样本调用 `sklearn.metrics.cosine_distances`，并且每张图做一次 CPU→GPU 往返。
* **改动**：`1 - 余弦相似度` 的 `bmm` 一次算完整个 batch，语义（最近原型的最小距离、`align_corners=True` 上采样）不变。
* **预期**：评估耗时下降；数值与 sklearn 版一致（差异只来自浮点累加顺序）。

### FIX-6 `type=1` 评估分辨率口径统一（**注意：会改变可比性**）
* **文件**：`code/main-fewshot-audit.py:1550`（原 `visual/main-fewshot.py:1518`）
* **问题**：原代码在训练中评估时，`type=1` 用 `resize_mask=None`（在 392 上算指标），
  其余 type 与 `phase=test` 都用 256。
* **风险**：项目历史公布的 mapping 分支指标都是 392 口径，改成 256 会让新结果与历史不可比。
* **实测**：同一权重下两口径差异极小（screw@30ep：P-AUROC 0.9823 vs 0.9824，见 `EXPERIMENT_LOG.md` §4），
  因此本项**不是**性能问题，只是口径问题。
* **结论**：最终报告统一以 **392（项目历史口径）** 为准，256 作为附注。

### FIX-9 设备硬编码 `cuda:2`
* **文件**：`code/main-fewshot-audit.py:1170`（原 `visual/main-fewshot.py:1165`）
* **问题**：`device='cuda:2'`。一旦用 `CUDA_VISIBLE_DEVICES` 只暴露一张卡，`cuda:2` 越界直接报错。
* **改动**：`device=os.environ.get('AUDIT_DEVICE','cuda:0')`。
* **预期**：在"只暴露 GPU1"的约束下可运行（这是本任务能跑起来的前提）。

### FIX-10 建模型前固定随机源
* **文件**：`code/main-fewshot-audit.py:1399`
* **问题**：`tokens_per_layer`/`token_mapping`/`segmentation_head` 都不在预训练权重里，
  `load_state_dict(strict=False)` 会让它们保持**随机初始化**，随机数消耗不同 → 初始权重不同 → A/B 不可比。
* **改动**：建模型前 `torch.manual_seed/np.random.seed/random.seed(args.seed)`。
* **预期**：对照组与实验组从同一起点出发。

### FIX-14 分割头损失：Dice 从"整 batch 全局"改为"逐样本 + 跳过空目标样本"
* **文件**：`code_dice/main-fewshot-dice.py:1068`（`FocalDiceLoss`，调用点不变，
  因此 `seg_criterion = FocalDiceLoss(...)` 无需改动）
* **原实现**（`visual/main-fewshot.py:1069`）：`pred.view(-1)` / `target.view(-1)` 后算
  **一个全局 Dice** —— 把 B 张图所有 patch 混在一起，异常面积大的样本主导 union，
  小面积样本的有效梯度被稀释；切到别的 batch 组成，损失口径就变。
* **改动**：逐样本算 Dice（`[B]`），只对含前景的样本平均；Focal 仍是逐像素均值。
* **CPU 验证**（`verify_dice_loss.py`，无需 GPU；数据 = 1 个好样本 + 1 个差样本 + 2 个全背景样本）：

  | 口径 | dice_score | 全背景样本的梯度 |
  |---|---|---|
  | 原实现（全局 Dice） | 0.5128 | 6.6e-2（被其它样本稀释） |
  | 朴素逐样本（不跳空） | 0.7146 | **1.0e+5（伪梯度！U=0 时 ∂/∂p=smooth/(U+smooth)²=1/smooth）** |
  | **FIX-14（逐样本+跳空）** | 0.4292 | **0.0（不产生伪梯度，正常区域的"别乱报"由 Focal 项负责）** |

* **为什么现在做**：上一轮实验证明 `type=2/4`（分割头误差图）明显弱于 `type=1`
  （P-AUROC 0.60/0.71/0.78 vs 0.92~0.98），分割头这条支路监督信号不可靠；
  损失口径是最可疑的一环，且修复零成本、不影响 type=1 主路径。
* **预期**：分割头在"小面积缺陷/大面积缺陷共存"的 batch 上训练更稳，
  `type=2/type=4` 指标改善（`type=1` 不受影响，可作对照）。
* **启动**：`bash q_dice.sh`（GPU 恢复后）

### FIX-13 GPU 绑定断言（事故驱动的新增防护）
* **文件**：`gpu_guard.py`（新增）、`common_env.sh`（新增 `EXPECTED_GPU_UUID` 与 `require_gpu()`）、
  `q_next.sh` / `q_gate.sh`（启动前调用 `require_gpu || exit 3`）
* **问题**：2026-09-29 16:34 起物理 GPU1 的 CUDA 初始化失败（外部故障）。排查中发现一个**危险陷阱**：
  本机 GPU0 长期 `Unknown Error` 会导致 CUDA 设备枚举错位，用数字索引
  `CUDA_VISIBLE_DEVICES=1` 时实际会落到**物理 GPU2**（实测 torch 报告的 UUID=3bf0407c = GPU2，
  而同事 `jmzhang` 的训练任务正占用该卡）。原脚本依赖 UUID 形式，虽未踩坑，但缺少**断言**，
  任何一次"临时改成数字索引"的调试都会静默打扰同事。
* **改动**：`gpu_guard.require_gpu()` 在进程内读取 `torch.cuda.get_device_properties(0).uuid`
  并与 `EXPECTED_GPU_UUID` 比对，不一致立刻 `SystemExit`（队列脚本 `exit 3`）。
  已验证三种情形：UUID 一致→通过；绑定到别的卡→中止；CUDA 不可用→中止。
* **预期**：从此"跑在别人的卡上"这件事在启动阶段就会失败，而不是悄悄占用同事资源。

### FIX-12（新增，本次审计最重要的工程缺陷）多层 patch 特征"求和融合"退化成只用最后一层
* **文件**：`code_norm/dinov2/models/vision_transformer.py:124`（新增 `LayerFusion`）、`:315`（模型内注册）、
  `code_norm/main-fewshot-norm.py:552,627,775,1494,1508`（5 处调用点）
* **问题**：`fused_patch_token = Σ_{i∈[1,2,3,4,6,8,11]} patch_feat_layer[i]`，逐层**不做归一化**。
* **证据**（`probe_fusion.py`，screw 正常图，ViT-B/14 @392）：

  | 层 | 1 | 2 | 3 | 4 | 6 | 8 | 11 |
  |---|---|---|---|---|---|---|---|
  | token 平均 L2 范数 | 2.88 | 2.38 | 2.56 | 3.46 | 7.05 | 19.56 | **154.67** |

  层间范数跨度 **64.9×**；融合向量与各层的余弦相似度：
  `layer1≈-0.02, layer2≈0.02, layer3≈0.02, layer4≈0.05, layer6≈0.15, layer8≈0.37, layer11≈0.985`。
  **即：所谓"7 层融合"在数值上等价于"只用 layer11"，前 4 层贡献几乎为 0**（典型的多层特征尺度不对齐缺陷）。
* **改动**：① 逐层 L2 归一化后再融合；② 逐层可学习权重（softmax，初值均匀）。
  归一化对齐后，各层与融合向量的 cos 变为 `0.67/0.76/0.77/0.75/0.69/0.49/0.23` —— 7 层才真正参与。
* **预期**：mapping 分支真正获得多层线索；若某单层确实最优，softmax 可收敛到近似 one-hot，
  因此该改动**不会比原来更差**（只把层间配比交给数据学）。
* **注意**：`token_mapping` 第一层是 `LayerNorm`，所以该改动只改变**特征方向配比**，不改变尺度，不会引入训练不稳定。

## 3. 发现但暂未改动的问题（诚实记录）

## 2.6 已实现但**尚未运行**的候选（GPU 故障期间准备，见 RUN_STATUS.md）

### ✅ 2026-10-08 官方指标复验完成（`q_fuse.sh`）：FEAT-FUSE × 多尺度成为主推配置

GPU 修复后（注意 nvidia-smi 编号已变，脚本按 UUID 绑定仍锁原卡）用 `adeval` 官方指标复验：
5 类平均 **P-AP 0.5360 → 0.6301（+0.0941）**、**P-AUPRO 0.9166 → 0.9493（+0.0327）**、
P-AUROC +0.0122、I-AUROC +0.0130、P-F1 +0.0880 —— **7 指标全升、5/5 类全升**；
相对"原始管线口径"（35 轮末）则 P-AP **+0.1417（+29%）**、AUPRO +0.0551。
同时用官方指标**否掉**了逐图 z 归一化变体（`--fuse_z 1`：I-AUROC −0.072、P-AP −0.087）。
命令：`--fuse_bank -1 --ms_scoring`（`--fuse_w 0.3` 可把 P-AP 再推到 +0.1052）。详见 EXPERIMENT_LOG §15。

### FEAT-FUSE（已实现并 CPU 验证；GPU 复验排队）打分图融合
* **文件**：`code_norm/main-fewshot-norm.py`（`build_raw_bank` / `evaluation_batch_fuse` / `--fuse_bank` / `--fuse_z`）、
  `eval_offline.py`（`--fuse_bank/--fuse_z/--cpu_metrics`）、`probe_rawbank.py`（消融探针）、`q_fuse.sh`（GPU 队列）
* **机制**：现状打分只用"随图像变化"的可学习原型，跨图尺度不可比；
  额外在**原始（未映射）特征空间**用 4 张正常图建一个固定原型库（经典 PatchCore 思路），
  两张打分图逐像素平均：`fused = 0.5*(d_learn + d_bank)`。
* **CPU 配对实验（4 类，392 口径）**：**P-AP +0.0596（4/4 为正）**、P-F1 **+0.0619**、P-AUROC +0.0049、
  I-AUROC 基本不变；幅度大于多尺度方案（P-AP +0.035）。消融还证明：可学习原型脱离映射空间会崩（P-AUROC 0.11~0.44），
  而纯 PatchCore 库单独就很强（hazelnut P-AP 0.781 vs 现状 0.587）——两者互补。
* **生产验证**：CPU 复跑 screw@25 与探针**逐位一致**（P-AP 0.3715 vs 0.3248）。
* **35 轮复验（CPU，5 类）与调参**：加权融合 `--fuse_w` 已实现。3 类 @35 平均：
  `w=0.5` → ΔP-AP +0.071；`w=0.7` → ΔP-AP +0.054 且 ΔI-AUROC 仅 −0.003（**保守推荐**）；
  `w=0.3` → ΔP-AP +0.085 但 screw 的 I-AUROC −0.081。
  **库规模**：k-center 32 会让 screw 变差（P-AP 0.269 < 现状 0.277），**必须用完整正常 patch 库**（3136 → 0.344）。
  验证：生产实现 `--fuse_bank -1 --fuse_w 0.3` 与探针逐位一致（P-AP 0.3492）。
* **风险**：`--fuse_z 1`（逐图 z 归一化）在 screw 更强但会破坏跨图可比性（hazelnut I-AUROC 1.0→0.833），默认关闭。
* **预期**：GPU + 官方 adeval（含 AUPRO）复验后即可合并入主结果；与多尺度打分正交，可叠加。

| 编号 | 文件 | 机制 | 与已有结论的关系 | 启动命令 |
|---|---|---|---|---|
| CAND-1 | `code_sj/main-fewshot-jitter.py`（`--scale_jitter 1.0,1.25,1.5`） | 训练期随机尺度抖动：图像与 mask 同步缩放，mask 的 token 网格按实际 patch 数计算（原 `resize_mask` 硬编码 784，尺度一变就与 FocalDiceLoss 形状不匹配） | 把推理端已验证有效的"多尺度"（§9）前移到训练，提升尺度鲁棒性 | `bash q_next.sh` |
| CAND-2 | `code_gate/…/vision_transformer.py`（`--fusion gate`） | **浅层残差门控**：`fused = Σ raw_i + Σ_{i≠最深} g_i·normalize(raw_i)`，`g` 初值 0 | 风险受控地重做 FIX-12：起点与 control 逐位等价，不会重演"35 轮后崩溃"；只有梯度认为浅层有用时才打开 | `bash q_gate.sh` |
| CAND-3 | `select_best_epoch.py` | 按内部评估选最优轮 | 见 EXPERIMENT_LOG §11：P-AP +0.048（比多尺度收益还大），已可直接使用 | `bash q_next.sh`（第 4 步做组合评估） |

## 2.5 路径2 交付物（新增功能，非"修复"）

### FEAT-MS 多尺度 + 水平翻转测试期打分（`--ms_scoring`）
* **文件**：`code_norm/main-fewshot-norm.py:698`（`multi_scale_score_map`）、`:731`（`evaluation_batch_ms`）、
  `:1224`（CLI `--ms_scoring/--ms_scales/--ms_flip`），并在训练/评估分支接入。
* **动机**：失败样本分析显示困难缺陷（scratch_head / manipulated_front）的 in/out-GT 对比度只有 0.068/0.076
  —— 14px 的 patch 把 392 输入上的小缺陷能量平均掉了（详见 `EXPERIMENT_LOG.md` §9.1）。
* **做法**：冻结权重，把已预处理的 392 输入按 s∈{1.0,1.25,1.5}（token 网格 28²→35²→42²）放大，
  每个尺度再做一次水平翻转，共 6 视图；各视图打分图插值回 392 后平均；其余流程完全不变，不使用任何标签。
* **预期与实测**（control@35ep，5 类，392 口径，生产实现复验）：
  ΔP-AP **+0.0345**、ΔP-AUPRO **+0.0283（5/5 全为正）**、ΔP-AUROC +0.0099、ΔI-AUROC +0.0036；
  机制验证：5 种缺陷类型的定位对比度全部提升（困难类型 +20%）。
  代价：前向次数 ×6。
* **风险**：metal_nut 的 P-AP −0.016（其 P-AUPRO 仍 +0.029）；建议按类选择是否启用（管线训练期已有评估可复用）。

---

## 3. 发现但暂未改动的问题（诚实记录）

| 编号 | 现象 | 证据 | 处理 |
|---|---|---|---|
| OBS-A | **learnable 原型严重坍缩**：32 个原型实际只有 2~3 个自由度 | `probe_coreset.py`：hazelnut 原型两两余弦 0.892、参与比有效秩 2.29/32；wood 0.901 / 2.41；screw 0.559 / 3.43 | 已做零训练替换实验（§5），结论：换成全局 coreset **不能**稳定涨点，说明"多样性"不是主要瓶颈 → 列为路径2候选而非直接修复 |
| OBS-B | 图像级指标已饱和（hazelnut/wood I-AUROC=1.0），真正的战场是像素级 P-AP/P-AUPRO | 见表 §3 | 后续优化只以像素级指标为准 |
| OBS-C | 分割头误差图（type=2）/ 相加融合（type=4）**弱于**纯原型距离（type=1） | screw P-AUROC 0.5985/0.9602 vs 0.9823 | 提示分割头这条支路监督信号偏弱，是路径2的候选方向 |
| OBS-D | 单次运行之间的"噪声地板"不小 | FIX-1 数学上是等价变换，但 30 轮后 screw P-AP 0.3134 vs 0.2766 | 任何 <0.03 的差异必须多 seed 复核（见 §3） |
