# 运行状态 / 待办（运维记录）

## 2026-09-29 17:56 复核：**所有卡都不可用**（阻塞加重）

四次轮询（17:54 / 17:55 / 17:55:47 / 17:56:34）结果一致：

```
nvidia-smi -L : Unable to determine the device handle for gpu 0000:18:00.0: Unknown Error
              （GPU1/2/3 仍被列出，但全部 10 MiB / 0%）
UUID 绑定实测 : GPU1 → 不可用 ；GPU2 → 不可用（17:53 起从"被占用"变成"不可用"）；GPU3 → 不可用
torch         : cuda_available=False, device_count=0
```

即从"只有 GPU2 可用（且被同事占用）"进一步退化为**零张卡可用**，需要管理员重置驱动
（GPU0 自始即 `Unknown Error`，现在 GPU1/2/3 也全部无法初始化）。

**已做的准备**：`common_env.sh` 支持一行切换目标卡且仍带 UUID 断言（默认仍是用户指定的 GPU1）：

```bash
AUDIT_GPU_UUID=GPU-xxxx bash q_next.sh      # 获授权换卡时
bash q_next.sh                              # 默认 GPU1
```


## 2026-09-29 17:20 磁盘清理（用户授权）
### 17:50 第三、四轮：其它目录的冗余权重（用户授权"保留最优轮次 + 删成对重复项"）

**A. `/home/jjquan/vpt_memorybank_run/saved_results/4-32token_70epoch_only_normal`（104G → 7.4G）**

该目录是 2026-09-05 的一次 70 轮实验（15 类 × 每 5 轮存一次 = 210 个 checkpoint）。
从 `log.txt` 按 **mapping 分支（每类第 1 行 = type1）** 统计"跨 15 类的平均"：

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | **45** | 50 | 55 | 60 | 65 | 70 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 平均 P-AUROC | .9589 | **.9802** | .9615 | .8105 | .9390 | .9358 | .9399 | .9477 | .9734 | .9541 | .9702 | .9531 | .7206 | .8532 |
| 平均 P-AP | .4293 | .2713 | .2531 | .2892 | .6385 | .6112 | .4894 | .5686 | **.7647** | .6967 | .4276 | .2862 | .1629 | .2430 |

**保留 epoch 45**（P-AP 跨类最优 0.7647，且 P-AUROC 0.9734 仅比最优的 0.9802 低 0.007，
若按"两指标之和"也是最优）；删除其余 13 个轮次 **195 个 checkpoint，释放 95.8G**。
另把解析结果写入同目录 `epoch_summary_type1.csv`（原始 `log.txt` 保留）。
校验：保留的 15 个权重 **15/15 可加载**（含 `dino`/`seg_head`）。

**B. `/home/jjquan/pretrained_models/stable-diffusion-v1-5`（45G → 26G）**

先确认用法：`experiment/visual2_v2/**` 全部通过 `StableDiffusionPipeline.from_pretrained(目录)`
加载（diffusers 布局），**没有任何脚本引用单文件 `.ckpt` 或 `.bin`**（grep 仅命中 diffusers 自身文档）。
据此按"成对重复保留 `.safetensors`"删除 5 个副本，**释放 19.15G**：

| 删除 | 保留 |
|---|---|
| `v1-5-pruned.ckpt` (7.17G) | `v1-5-pruned.safetensors` |
| `v1-5-pruned-emaonly.ckpt` (3.97G) | `v1-5-pruned-emaonly.safetensors` |
| `unet/diffusion_pytorch_model.bin` (3.20G) | `unet/diffusion_pytorch_model.safetensors` |
| `unet/…non_ema.bin` (3.20G) | `unet/…non_ema.safetensors` |
| `unet/…fp16.bin` (1.60G) | `unet/…fp16.safetensors` |

校验：`model_index.json` 与各子目录完好；保留的 safetensors 可读（unet 686 / 全量 1831 / emaonly 1145 个张量）。

### 本轮小结（四次清理合计）

| 轮次 | 对象 | 释放 |
|---|---|---|
| 1 | audit_mapping 被否决变体 | 22.0G |
| 2 | audit_mapping 中间轮次 | 22.1G |
| 3 | vpt_memorybank_run 非最优轮次 | 95.8G |
| 4 | SD1.5 成对重复副本 | 19.15G |
| **合计** | | **≈159G** |

磁盘：**225G → 384G 可用**（94% → 89%）。所有指标 CSV、训练日志、最优轮次权重均已保留并逐一校验可加载。

### 17:35 第二轮：中间轮次 checkpoint

按"**是否被待跑队列/文档验证点引用**"判定，删除冗余的中间轮次：

| 目录 | 保留 | 删除 | 释放 |
|---|---|---|---|
| `results/control35` | 9 个（5 类的 35 轮 + 选轮组合用的 capsule_10/hazelnut_20/screw_25/wood_10） | 26 个 | 8.4G |
| `results/control35_seed2` | 3 个（seed2 的 35 轮，噪声地板引用） | 18 个 | 5.8G |
| `results/control` | 6 个（文档里探针验证点：screw_10/30、hazelnut_30、wood_30、capsule_10、metal_nut_30） | 24 个 | 7.8G |
| **合计** | **18 个（5.8G）** | **68 个** | **22.1G** |

删除前已核对：所有待删路径均在 `audit_mapping/results/` 下且匹配 `model_*.pth`；
删除后校验 `q_next.sh` 引用的 9 个权重、以及 `_metrics_archive/`、各 `metrics_maponly.csv`、
`results/probe/`、`results/ms35/` 全部完好。

**本地合计**：`audit_mapping/` 50G → **5.9G**（两轮共释放 ≈44G）。

### 17:20 第一轮：被否决变体

删除**被否决变体**的 checkpoint（只删 `.pth`，指标 CSV 与训练日志一律保留）：

| 目录 | 删除内容 | 释放 |
|---|---|---|
| `results/norm35` | 35 个 `model_*.pth`（FIX-12，已判定崩溃） | ~12G |
| `results/fixed` | 30 个 `model_*.pth`（FIX-1/2/6，已判定等价无性能影响） | ~9.8G |
| `results/{audit_smoke,smoke_norm,smoke_sum}` | 冒烟测试 checkpoint | ~1G |
| **合计** | 68 个文件 | **≈22G** |

证据保全：所有指标 CSV 与训练日志已复制到 `results/_metrics_archive/`
（`norm35_metrics.csv`、`fixed_metrics.csv`、`norm35_train.log`、`fixed_train.log`、冒烟 CSV），
原目录下的 CSV/log 也仍在原地未动。保留的关键权重：`results/control35`（35 个，最终基准）、
`results/control35_seed2`（21 个，噪声地板）、`results/control`（30 个，30 轮对照）。

磁盘：**225G → 247G 可用**（占用 94% → 93%）；`audit_mapping/` 50G → 28G。
如需进一步释放，`results/control`（30 轮对照，9.8G）及其探针权重也可清理（探针结果 CSV 已保存）。

## 2026-09-29 17:04 第三次复核：阻塞条件持续（**已标记 goal=blocked**）

```
GPU0: Unknown Error（历史故障）
GPU1: 10 MiB / 0 %   ← 用户指定唯一可用卡，CUDA 初始化仍然 FAIL
GPU2: 6449 MiB / 98 % ← 可初始化，但被同事 jmzhang 的训练任务占用
GPU3: 10 MiB / 0 %   ← CUDA 初始化 FAIL
UUID 绑定实测: GPU-0e3a7e1c → FAIL ; GPU-3bf0407c → ok ; GPU-8e8231eb → FAIL
sudo: 需要密码（无免密 sudo）→ 无法自行重置驱动
```

**结论**：在"只能使用 GPU1"的约束下，本目标无法继续推进（任何候选方案——尺度抖动、FIX-14 分割头、
残差门控、选轮×多尺度组合——都需要 GPU 训练才能验证）。
需要**外部状态改变**：管理员重置 GPU1/GPU3 驱动，或用户明确授权改用其它空闲卡
（当前除被同事占用的 GPU2 外没有既空闲又能初始化 CUDA 的卡）。

所有已完成工作与待跑队列均已固化，GPU 恢复后可直接执行，无需重新阅读上下文：
`bash q_next.sh` → `bash q_dice.sh` → `bash q_gate.sh`（均带 UUID 断言守卫）。

## 2026-09-29 16:34 起：**GPU1 本身** CUDA 不可用（外部环境故障）

### 精确诊断（16:50 复核，四张卡逐一用 UUID 绑定实测）

| 物理卡 (nvidia-smi 编号) | UUID | 显存/负载 | CUDA 初始化（UUID 绑定） |
|---|---|---|---|
| GPU0 | (驱动枚举即报 Unknown Error) | — | 不可用（任务开始前就是坏的） |
| **GPU1**（**用户指定只能用这张**） | GPU-0e3a7e1c-… | 10 MiB / 0 % | **FAIL（CUDA driver initialization failed）** |
| GPU2 | GPU-3bf0407c-… | 6449 MiB / 99 % | **ok，但正被同事 `jmzhang` 的训练任务占用** |
| GPU3 | GPU-8e8231eb-… | 10 MiB / 0 % | FAIL（同样初始化失败） |

**结论：用户允许使用的 GPU1 目前无法初始化 CUDA**；唯一能初始化的 GPU2 是同事在跑的卡，
因此**在本轮约束下无法继续任何 GPU 实验**（即使获得授权也无法在不打扰同事的前提下跑）。
需要管理员对 GPU1（可能还有 GPU0/GPU3）做 `nvidia-smi -r` 重置或重启驱动。

### ⚠️ 本次事故暴露的一个**危险陷阱**（已修复，见 FIX-13）

`CUDA_VISIBLE_DEVICES` 用**数字索引**时，本机因为 GPU0 枚举异常会**静默错位**：

```
CUDA_VISIBLE_DEVICES=1  →  torch 报告的 UUID 是 3bf0407c-… = 物理 GPU2（同事正在用！）
CUDA_VISIBLE_DEVICES=0  →  初始化失败
CUDA_VISIBLE_DEVICES=2  →  初始化失败
```

即"以为在用 GPU1，实际落在同事的 GPU2 上"。所以：**必须用 UUID 绑定，并在进程内断言 UUID**。
已新增 `gpu_guard.py`（并在 `common_env.sh` 提供 `require_gpu`），任何训练/评估在启动前
都会校验"当前 cuda:0 的 UUID == 预期物理卡 UUID"，不一致就立刻中止。
其单元测试（不占用 GPU）：UUID 一致→通过；绑定到别的卡→中止；CUDA 不可用→中止。

## 历史记录：首次发现（16:34）

**现象**（`nvidia-smi` 与 torch 两路证据）：

```
$ nvidia-smi            → Unable to determine the device handle for GPU0000:18:00.0: Unknown Error
$ nvidia-smi -i 1       → 10 MiB / 24576 MiB, 0 % util, 46 °C   （GPU1 空转，没有任何进程）
$ python -c "import torch; print(torch.cuda.is_available())"
  Can't initialize NVML
  → cuda False, device_count = 0
```

* 触发时间：完成 `q_ms_eval35.sh`（16:30:36 结束）后，16:34 起所有 CUDA 初始化失败。
* 影响面：**不只是本任务**——GPU1 上其它同事的训练进程（`jmzhang` 的 `train.py`、`jjquan` 的
  `main-fewshot-fullshot.py`）同时从 GPU 上消失（GPU1 0% 利用率、10MiB），
  说明是驱动/硬件层面的故障，而非本任务的显存或代码问题。
* 本任务进程状态：所有训练/评估队列均已正常结束（无残留进程），无数据损坏；
  已完成的 50+ 组实验日志与指标 CSV 全部完好。
* 权限：当前账号 `jjquan` **非 root**，无法执行 `nvidia-smi -r` / 重载驱动 → 需要管理员介入恢复。

## 恢复后待跑队列（均已写好、语法与（可在 CPU 上做的）单测通过、且都带 GPU 守卫）

全部脚本启动前会先执行 `require_gpu`（UUID 断言），若仍绑在坏卡/别人的卡上会**直接拒绝启动**。

> 新增（2026-09-29 18:20，GPU 故障期间用 **CPU 推理**完成）：**FEAT-FUSE 打分融合**
> （`--fuse_bank`）——CPU 配对实验 4 类 P-AP 平均 **+0.0596（4/4 为正）**，生产实现已与探针逐位对齐。
> GPU 恢复后优先跑 `q_fuse.sh`（官方 adeval 含 AUPRO）。
>
> 18:53 继续（仍无 GPU）：补充 **35 轮 5 类复验**与调参 ——
> 3 类 @35 平均 ΔP-AP：`w=0.5` +0.071 / **`w=0.7` +0.054 且 I-AUROC 仅 −0.003（保守推荐）** / `w=0.3` +0.085；
> **库规模必须用完整正常 patch**（k-center 32 反而比现状差）。`q_fuse.sh` 已加入 w=0.3/0.5 与「融合×多尺度」组合共 6 种配置。

**`bash q_next.sh`**（约 1.5 小时）

1. **打分变体补齐**：`ms_flip_min` / `ms_flip_med` / `ms_flip_mean_med`
   —— 即"多尺度一致性"变体，目标是压掉 metal_nut 的 P-AP 退化（−0.016）。
2. **训练期尺度抖动**（`code_sj/main-fewshot-jitter.py`，已实现并通过语法检查）：
   `--scale_jitter 1.0,1.25,1.5`，让 mapping 在 28²/35²/42² 三种 token 网格上都有监督。
   关键实现：图像与 mask 同步缩放，且 mask 的 token 网格按实际 patch 数计算
   （原 `resize_mask` 硬编码 784 token，尺度变化后会导致 FocalDiceLoss 形状不匹配）。
3. 用生产实现 `evaluation_batch_ms` 评估抖动模型的 base / ms 两组指标。

预计耗时：约 25 分钟（评分变体）+ 40 分钟（5 类训练）+ 20 分钟（评估）。

**`bash q_dice.sh`**（约 1 小时）：FIX-14 分割头损失（逐样本 Dice + 跳过空背景样本）× 35 轮，
评估 type=1/2/4 三种口径，与 control35 做配对比较。

**`bash q_gate.sh`**（约 1 小时）：浅层残差门控 `--fusion gate`（初值等价 control）× 35 轮 + base/ms 评估。

优先级建议：q_next 的"选轮 × 多尺度组合"（最可能立刻刷新最好成绩）→ q_dice → q_gate。
