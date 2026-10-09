# code/ 说明（vpt3.5）

本目录是**主训练/评估代码**（原仓库 `/home/jjquan/experiment/visual` 的副本，原仓库保持零改动）。

| 文件/目录 | 说明 |
|---|---|
| `main-fewshot-vpt4.py` | 训练与在线评估主脚本。vpt3.5 的模型/算法改动都在这里 + `dinov2/`：M1 原型初始化、M3 多样性正则、M4 全局锚点、M5 像素 BCE（本版关闭）、M6 可学习度量头、M7 跨图排序一致性损失；另有 `--nocrop_size` 开关（训练/测试都用 Resize 不裁剪） |
| `dinov2/` | DINOv2 骨干与 `TokenMapping`（M2 残差、M6 度量头在此定义）、`LearnableTokens`（M1 初始化在此） |
| `util/` | 数据与 MemoryBank 实现（`util/memorybank.py` 为原版实现） |
| `dataset1.py` / `ucc/` | 原始数据与工具代码（未改动） |
| `main-fewshot-audit.py` | 审计用副本（仅作对照，不参与 vpt3.5 主流程） |

## 本版新增/修改的要点（对应 `docs/EXPERIMENT_LOG.md` §22–§37）
1. **M1** 可学习原型初始化为 `std=0.02`（原 1e-6，导致 32 个原型几乎全同）；
2. **M3** 原型多样性正则（`--w_div 0.1 --div_margin 0.5`）；
3. **M4** 原型集拼接全局锚点（`--global_proto 1`）；
4. **M6** 可学习度量头 `W`（初值单位阵，`--metric_head 1`）；
5. **M7** 跨图排序一致性损失（`--w_rank 0.02 --rank_margin 0.25 --rank_norm 1`，尺度无关）；
6. **FIX-INIT-REPLAY** 原型初始化改回与原实现同一算子（`nn.init.normal_`），保证随机流对齐；
7. **FIX-M4** 训练端异常分支补齐全局锚点，使 M5/M7 的打分口径与推理一致；
8. **FIX-METRIC-ROBUST** 指标库偶发断言不再中断整类训练；
9. `--nocrop_size` 支持"训练/测试同一不裁剪几何"的对照实验；
10. shot 抽样交给 `tools/shot_utils.py` 统一（训练与两个分支共用同一清单）。
