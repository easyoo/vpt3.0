"""[AUDIT-FIX-5] 数值等价性校验：新的向量化 compute_anomaly_scores 与旧的 sklearn 版是否一致。

做法：import 审计副本里的函数（脚本有 __main__ 守卫，import 不会跑训练），
再用 sklearn 原实现算一遍，比对最大绝对误差。
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
sys.path.insert(0, CODE)
spec = importlib.util.spec_from_file_location("mfa", os.path.join(CODE, "main-fewshot-audit.py"))
M = importlib.util.module_from_spec(spec)
sys.modules["mfa"] = M
spec.loader.exec_module(M)
M.device = "cpu"          # 该函数内部用模块级 device


def old_impl(patches, prototypes, out_size):
    from sklearn.metrics.pairwise import cosine_distances
    out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
    b, n, _ = patches.shape
    maps = []
    for i in range(b):
        d = cosine_distances(patches[i], prototypes[i])
        s = np.min(d, axis=1)
        s = torch.from_numpy(s).float()
        s = s.reshape(1, 1, int(n ** 0.5), int(n ** 0.5)).contiguous()
        s = F.interpolate(s, size=out_size, mode="bilinear", align_corners=True).squeeze(0)
        maps.append(s)
    return torch.stack(maps, 0)


rng = np.random.default_rng(0)
B, N, D, K = 3, 784, 768, 32
p = rng.standard_normal((B, N, D)).astype(np.float32)
q = rng.standard_normal((B, K, D)).astype(np.float32)
a = old_impl(p, q, 392)
b = M.compute_anomaly_scores(p, q, 392)
print("old shape", tuple(a.shape), "new shape", tuple(b.shape))
print(f"max|diff| = {float((a - b).abs().max()):.3e}   mean|diff| = {float((a - b).abs().mean()):.3e}")
print("一致（<1e-5）:", bool(torch.allclose(a, b, atol=1e-5)))

# 额外检查：重复 prototype 不影响 min（验证 AUDIT-FIX-1 对打分是良性的）
q2 = np.concatenate([q, q], axis=1)
c = old_impl(p, q2, 392)
print(f"prototype 数量翻倍后的 max|diff| = {float((a - c).abs().max()):.3e}（应为 0，说明重复 token 不改变 min 打分）")
