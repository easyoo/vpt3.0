# -*- coding: utf-8 -*-
"""诊断：可学习原型空间里，正常 patch 与原型的相似度分布 / 分配熵 / 簇使用是否均衡。
用于判断"聚类式损失"是否真的有信号，以及损失形式该怎么设计。
用法： CUDA_VISIBLE_DEVICES=1 python diag_cluster.py --cls bottle --shots 4 --ckpt results/vpt4/m4m6r3_15
"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(p):
    spec = importlib.util.spec_from_file_location('diag_cluster_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['diag_cluster_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', default='bottle')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--ckpt', default='results/vpt4/m4m6r3_15')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--ano', default='/home/jjquan/experiment/visual2/anomaly_samples')
    ap.add_argument('--size', type=int, default=392)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    dev = 'cuda'
    model = mod.build_backbone(argparse.Namespace(backbone='base')) if False else None

    class A:
        pass
    a = A()
    for k, v in dict(fused_layer=[1, 2, 3, 4, 6, 8, 11], fused_layer_learnable=[0], learnable_layers=1,
                     learnable_num_per_layer=32, res_mapping=0, global_proto=1, metric_head=1,
                     token_init_std=0.02, temperature=0.07, scaled_factor=3).items():
        setattr(a, k, v)
    mod.args = a
    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8, num_register_tokens=4,
        interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = os.path.join(args.ckpt, f'model_{args.cls}_35.pth')
    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['dino'], strict=False)
    model.to(dev).eval()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((448, 448)), T.CenterCrop(args.size), T.ToTensor(), T.Normalize(mean, std)])
    from shot_utils import get_shots
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=1)
    imgs = torch.stack([tf(Image.open(os.path.join(root, p)).convert('RGB')) for p in picks], 0).to(dev)
    with torch.no_grad():
        ret = model(imgs, is_training=True)
        fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
        flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
        g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
        flr = torch.cat([flr, g], dim=1)
        X = F.normalize(model.token_mapping(fp), dim=-1)
        P = F.normalize(model.token_mapping(flr), dim=-1)
        sim = torch.bmm(X, P.transpose(1, 2))                     # [B,N,K]
        p = F.softmax(sim / 0.1, dim=-1)
        ent = -(p * torch.log(p + 1e-12)).sum(-1)                 # [B,N] nats
        print(f'协议：{args.shots}-shot {args.cls}，patch {tuple(X.shape)}，原型 {tuple(P.shape)}')
        print(f'  相似度 sim: mean={sim.mean():.4f} std={sim.std():.4f} max={sim.max():.4f}')
        print(f'  分配熵(温度0.1): mean={ent.mean():.3f} nats   (均匀=log K={torch.log(torch.tensor(float(P.shape[1]))):.2f})')
        # 簇使用均衡度：每个原型被"最近邻"选中的次数占比
        nn = sim.argmax(-1).reshape(-1)
        cnt = torch.bincount(nn, minlength=P.shape[1]).float()
        used = (cnt > 0).sum().item()
        share = cnt / cnt.sum()
        print(f'  最近邻命中：{used}/{P.shape[1]} 个原型被用到；最大簇占比={share.max():.3f}（均匀={1/P.shape[1]:.3f}）')
        # 原型冗余：两两余弦
        Q = F.normalize(P.reshape(-1, P.shape[-1]), dim=-1)
        C = Q @ Q.t()
        off = C - torch.eye(C.shape[0], device=C.device)
        print(f'  原型两两余弦: mean={off.sum()/(C.shape[0]*(C.shape[0]-1)):.3f} max={off.max():.3f}')


if __name__ == '__main__':
    main()
