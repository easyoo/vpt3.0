"""探针：量出 mapping 分支的张量形状与 learnable token 的实际行为。

默认跑审计副本（audit_mapping/code），可用 --code_root 指向原项目做对照。
CPU 即可，不需要任何预训练权重（只看形状与流程，不看精度）。
"""
import argparse
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--code_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
args = ap.parse_args()
sys.path.insert(0, args.code_root)
print("[probe] code_root =", args.code_root)

from dinov2.models import vision_transformer  # noqa: E402


def build(ns):
    return vision_transformer.__dict__["vit_base"](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=ns)


x = torch.randn(1, 3, 392, 392)
for LL in [1, 4]:
    ns = argparse.Namespace(learnable_layers=LL, learnable_num_per_layer=32)
    m = build(ns).eval()
    with torch.no_grad():
        out = m(x, is_training=True)
    f, p = out["feat_list"], out["patch_feat_layer"]
    print(f"learnable_layers={LL}: len(feat_list)={len(f)} shapes={[tuple(t.shape) for t in f]}")
    print(f"   len(patch_feat_layer)={len(p)} shape[0]={tuple(p[0].shape)}")
    if LL == 1:
        a, b = f[0][:, :32], f[0][:, 32:]
        same = torch.allclose(a, b) if f[0].shape[1] == 64 else "n/a(只有32个token)"
        print(f"   feat_list[0] 是否含重复的 32 个 token: {same}")
    try:
        print("   fused_layer_learnable=[0] ->", tuple(torch.stack([f[0]], 0).sum(0).shape))
    except Exception as e:
        print("   fused learnable error:", type(e).__name__, e)
    try:
        print("   overwrite sum(all feat_list) ->", tuple(torch.stack(f, 0).sum(0).shape))
    except Exception as e:
        print("   overwrite error:", type(e).__name__, str(e)[:120])
