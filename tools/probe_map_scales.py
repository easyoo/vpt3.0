# -*- coding: utf-8 -*-
"""[CPU 探针] 量最终打分图的两个分量各自的尺度：
     m_learn = 1 - max_k cos( token_mapping(patch), token_mapping(proto) )      ← 我们所有模型级改进作用的对象
     m_bank  = 1 - max_j cos( raw_patch, raw_bank_j )                          ← 4 张 shot 的原始特征库，训练完全不碰
   最终评估用的是 fused = w_learn*m_learn + (1-w_learn)*m_bank（默认 w_learn=0.5）。

如果两者尺度相差一个数量级，那么 0.5/0.5 只是**名义权重**：小尺度那一项几乎不参与最终打分，
模型级改进就会被稀释掉——这必须先用数据确认，再决定是否做尺度对齐。

用法：python probe_map_scales.py --save m4m6 --class transistor --n_images 3
"""
import argparse
import importlib.util
import os
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def load_audit_module(script_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(script_path)))
    spec = importlib.util.spec_from_file_location('audit_main', script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='m4m6')
    ap.add_argument('--class', dest='cls', default='transistor')
    ap.add_argument('--n_images', type=int, default=3)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--res_mapping', type=int, default=0)
    ap.add_argument('--global_proto', type=int, default=1)
    ap.add_argument('--metric_head', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    mod = load_audit_module(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.learnable_num_per_layer = args.num_tokens
    a.res_mapping = args.res_mapping
    a.global_proto = args.global_proto
    a.metric_head = args.metric_head
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    mod.args = a
    mod.device = 'cpu'

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ckpt = os.path.join('results', 'vpt4', args.save, f'model_{args.cls}_35.pth')
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.eval()
    print(f'[probe] ckpt={ckpt}')

    data_transform, gt_transform = mod.get_data_transforms(448, 392)
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    files = sorted(os.listdir(root))
    random.seed(args.seed)
    picks = random.sample(files, args.shots)
    normal = torch.stack([data_transform(Image.open(os.path.join(root, f)).convert('RGB'))
                          for f in picks], dim=0)
    with torch.no_grad():
        bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)
    print(f'[probe] 原始特征库: {tuple(bank.shape)}（4 张 shot，k=0 全 patch，训练过程完全不参与）')
    q_raw = F.normalize(bank, dim=-1)

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=data_transform, gt_transform=gt_transform, phase='test', sort_items=False)
    rows = []
    with torch.no_grad():
        for i in range(min(args.n_images, len(ds))):
            img, gt, label, _ = ds[i]
            inp = img.unsqueeze(0)
            for s in (1.0, 1.5):
                v = inp if s == 1.0 else F.interpolate(inp, scale_factor=s, mode='bilinear', align_corners=False)
                ret = model(v, is_training=True)
                fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
                if args.global_proto:
                    _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                    flr = torch.cat([flr, _g], dim=1)
                m_learn = mod.proto_distance_scores(model.token_mapping(fp), model.token_mapping(flr)).reshape(-1)
                p_raw = F.normalize(fp, dim=-1)
                m_bank = (1.0 - torch.bmm(p_raw, q_raw.T.unsqueeze(0).expand(p_raw.shape[0], -1, -1))
                          ).clamp_min(0).min(dim=-1).values.reshape(-1)
                rows.append((i, s, m_learn, m_bank))
                print(f'  图{i} scale={s}: m_learn 均值={m_learn.mean():.4f} p95={torch.quantile(m_learn,0.95):.4f} '
                      f'| m_bank 均值={m_bank.mean():.4f} p95={torch.quantile(m_bank,0.95):.4f} '
                      f'| 尺度比(m_bank/m_learn)={float(m_bank.mean()/m_learn.mean()):.1f}x')
    ml = torch.cat([r[2] for r in rows])
    mb = torch.cat([r[3] for r in rows])
    print(f'\n汇总：m_learn 均值={ml.mean():.4f} 标准差={ml.std():.4f}')
    print(f'      m_bank  均值={mb.mean():.4f} 标准差={mb.std():.4f}')
    print(f'      尺度比 = {float(mb.mean()/ml.mean()):.1f}×  → 名义 0.5/0.5 融合里，'
          f'可学习图的**实际**权重 ≈ {float(ml.mean()/(ml.mean()+mb.mean()))*100:.1f}%')


if __name__ == '__main__':
    main()
