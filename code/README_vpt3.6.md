# code/ 说明（vpt3.6）

**本目录与 vpt3.5 逐字节相同，本轮未做任何修改。**

| 文件 | md5 | 说明 |
|---|---|---|
| `main-fewshot-vpt4.py` | `08d8528b4a590fcefad8ce0ae57c5e2d` | 训练 + 内置评估（v3.5 交付版） |
| `dinov2/models/vision_transformer.py` | — | 主干 + `LearnableTokens` / `TokenMapping`（含 M1/M2/M6 与自定义前向） |
| `util/dataset.py` | — | `AnoStylerGeneratedDataset`（合成异常加载） |

## 为什么 vpt3.6 的代码目录没有变化

vpt3.6 的改动**全部在推理端**（异常打分算子与融合权重），因此只需要替换评估脚本、不需要重训：

* 新打分：`tools/eval_score_v3.py`
  ```
  s_raw   = d1^0.5 · q50^0.5          # 原始特征库侧（旧版 = d1）
  s_learn = dlmin^0.7 · bq64^0.3      # 可学习原型侧（旧版 = dlmin）
  m       = 0.6·s_learn + 0.4·s_raw   # 视图内融合（旧版 0.5/0.5）
  ```
* 旧打分（对照基线）：`tools/eval_nocrop.py`

权重文件与 vpt3.5 完全相同（`model_<cls>_35.pth`），因此新/旧打分的对比是**配对比较**，
收益不受训练随机性影响。

## 若要继续推进（训练端对齐）

当前训练损失里取分仍走 `proto_distance_scores`（`main-fewshot-vpt4.py:1224`，等价 `1 − max cos`）。
把它换成与推理一致的公式即可让训练/推理口径统一，但需要重训 35 轮：

```python
# main-fewshot-vpt4.py 的 proto_distance_scores
def proto_distance_scores(patch_mapping, proto_mapping):
    p = F.normalize(patch_mapping, dim=-1)
    q = F.normalize(proto_mapping, dim=-1)
    d = (1.0 - torch.bmm(p, q.transpose(1, 2))).clamp_min(0)     # [B,N,K]
    d1 = d.min(dim=-1).values                                    # 最近原型
    # 全局项需要"到正常库"的距离分布分位 —— 训练时可用 batch 内全部正常 patch 近似
    ...
    return d1.pow(a) * q_global.pow(1 - a)
```

注意：训练端没有现成的"正常 patch 库"（库是推理时才由 K 张 shot 构建的），
若要对齐，需要在训练时也构建一次（K×1024 个 patch，代价很小），或改用 batch 内的正常 patch 近似。
