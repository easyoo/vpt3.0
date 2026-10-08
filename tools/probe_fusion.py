# -*- coding: utf-8 -*-
"""[审计探针] 多层特征"相加融合"是否被单一层主导？（特征对齐诊断）

代码事实：patch_feat_layer 共 12 项（每层一个），fused_layer=[1,2,3,4,6,8,11]，
融合方式是 fused = sum_i patch_feat_layer[i]（未做逐层归一化），
随后 token_mapping 的第一层是 LayerNorm。
若各层 token 的 L2 范数差异很大，则"求和"实际等价于"只取范数最大的那一层"，
其余 6 层的信息被数值淹没——这是典型的多层特征对齐缺陷。
"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(script_path):
    spec = importlib.util.spec_from_file_location('audit_main', script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code', 'main-fewshot-audit.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, 'code'))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(args.device).eval()

    tf, _ = mod.get_data_transforms(448, 392)
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    f = sorted(os.listdir(root))[0]
    img = tf(Image.open(os.path.join(root, f)).convert('RGB')).unsqueeze(0).to(args.device)

    fused_layer = [1, 2, 3, 4, 6, 8, 11]
    with torch.no_grad():
        ret = model(img, is_training=True)
        pfl = ret['patch_feat_layer']
        print(f"[{args.cls}] checkpoint={os.path.basename(args.ckpt)}  image={f}")
        print(f"  patch_feat_layer 共 {len(pfl)} 项，每项 shape={tuple(pfl[0].shape)}")
        norms = []
        for i, t in enumerate(pfl):
            n = t.norm(dim=-1)
            norms.append(n.mean().item())
        print("  各层 patch token 平均 L2 范数：")
        for i, n in enumerate(norms):
            mark = ' ← 被融合' if i in fused_layer else ''
            print(f"    layer[{i:2d}] mean|f|={n:8.3f}  逐样本std={pfl[i].norm(dim=-1).std():6.3f}{mark}")
        sel = [norms[i] for i in fused_layer]
        print(f"  被融合的 7 层范数：min={min(sel):.3f} max={max(sel):.3f} 比值={max(sel)/min(sel):.2f}x")

        fused = torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
        print(f"  融合后 token 平均范数={fused.norm(dim=-1).mean():.3f}")
        print("  各层与『融合向量』的余弦相似度（越接近 1 说明它越主导这个和）：")
        fn = F.normalize(fused, dim=-1)
        for i in fused_layer:
            cn = F.normalize(pfl[i], dim=-1)
            cos = (fn * cn).sum(-1).mean().item()
            print(f"    layer[{i:2d}] cos(融合, 本层)={cos:.4f}")

        # 逐层 L2 归一化后求和（一种"对齐"修复）的对照
        fused_n = torch.sum(torch.stack([F.normalize(pfl[i], dim=-1) for i in fused_layer], dim=0), dim=0)
        fn2 = F.normalize(fused_n, dim=-1)
        print("  若改成『逐层 L2 归一化后再求和』：")
        for i in fused_layer:
            cn = F.normalize(pfl[i], dim=-1)
            cos = (fn2 * cn).sum(-1).mean().item()
            print(f"    layer[{i:2d}] cos(新融合, 本层)={cos:.4f}")


if __name__ == '__main__':
    main()
