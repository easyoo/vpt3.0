# VPT3.9 Code Snapshot

VPT3.9 的训练代码默认行为保持 VPT3.8，新增用于推理融合的实现与工具：

- `../tools/eval_support_structural_continuous.py`：SS-MSF 评测。
- `../tools/probe_support_structure.py`：计算 support 的 `c/r`。
- `../tools/benchmark_learning_branch.py`：学习分支吞吐量。
- `../tools/benchmark_unified_inference.py`：完整推理吞吐量。

核心推理公式：

```text
w_m = clip((1-c) * r, 0, 1)
w_b = 1-w_m
S = mean_{k in {9,25,41}}
      [w_m * box_k(Mapping) + w_b * box_k(Bank)]
```

所有 support 统计仅来自 K-shot 正常图像。
