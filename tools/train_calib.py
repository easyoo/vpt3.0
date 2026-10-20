# -*- coding: utf-8 -*-
"""训练侧超参标定（**不使用任何测试集数据**）。

数据来源只允许：K 张 shot 正常图 + 其 AnoStyler 合成异常（均属训练侧）。
对每个候选打分配置计算：图像级 AUROC（正常 shot vs 合成异常）与像素级 AUROC（对合成掩码），
并输出候选排序 —— 供"无测试集调参"的模型选择使用。

用法：
  python train_calib.py --save m4m6r3_2shot --class transistor --shots 2 \
      --data_path /home/jjquan/MVTec-AD --anom_root /home/jjquan/experiment/visual2/anomaly_samples_2shot/mvtec \
      --ckpt_dir results/vpt4/m4m6r3_2shot
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots          # noqa: E402
SUBN = 1024


def load_mod(p, name='tc'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def collect_anomalies(root):
    """<root>/<cls>/<type>/{image,mask}/*.jpg → [(img_path, mask_path), ...]"""
    out = []
    for t in sorted(os.listdir(root)):
        d = os.path.join(root, t)
        if not os.path.isdir(d):
            continue
        idir, mdir = os.path.join(d, 'image'), os.path.join(d, 'mask')
        if not (os.path.isdir(idir) and os.path.isdir(mdir)):
            continue
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith(('.jpg', '.png')):
                continue
            mf = f.replace('gen_ano', 'gen_mask')
            mp = os.path.join(mdir, mf)
            if os.path.exists(mp):
                out.append((os.path.join(idir, f), mp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='m4m6r3_2shot')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=2)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--anom_root', required=True)
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit_anom', type=int, default=0)
    ap.add_argument('--seg_w_list', default='0,0.25,0.5,1.0')
    ap.add_argument('--alpha_list', default='0.5,0.7')
    args = ap.parse_args()

    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.res_mapping = 0
    a.global_proto = 1
    a.metric_head = 1
    mod.args = a
    mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(os.path.join(args.ckpt_dir or f'results/vpt4/{args.save}',
                                 f'model_{args.cls}_35.pth'), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(args.device).eval()
    seg_head = torch.nn.Sequential(torch.nn.Linear(768, 384), torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(384, 1), torch.nn.Sigmoid()).to(args.device)
    seg_head.load_state_dict(ck['seg_head'])
    seg_head.eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=1)
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)
    M = bank.shape[0]
    q_raw = F.normalize(bank, dim=-1)
    sub = torch.randperm(M, generator=torch.Generator().manual_seed(1234))[:min(SUBN, M)].to(args.device)

    anom = collect_anomalies(os.path.join(args.anom_root, args.cls))
    if args.limit_anom:
        anom = anom[:args.limit_anom]
    print(f'[{args.cls}] 训练侧标定：正常 shot {len(picks)} 张；合成异常 {len(anom)} 张（均为训练数据）')

    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    rec = {'d1': [], 'q512': [], 'dlmin': [], 'bq64': [], 'seg': [], 'is_norm': [], 'mask': []}
    CH = 1024
    plan = [(i, min(i + CH, M), sub[(sub >= i) & (sub < min(i + CH, M))] - i) for i in range(0, M, CH)]

    def comps(img_batch, is_norm, mask_batch=None):
        with torch.no_grad():
            v = F.interpolate(img_batch, scale_factor=1.5, mode='bilinear', align_corners=False)
            ret = model(v, is_training=True)
            fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
            flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
            flr = torch.cat([flr, model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)], 1)
            g = int(round(fp.shape[1] ** 0.5))
            p = F.normalize(fp, dim=-1)
            d1, cols = None, []
            for i, j, sel in plan:
                sim = torch.bmm(p, q_raw[i:j].t().unsqueeze(0).expand(p.shape[0], -1, -1))
                dch = (1.0 - sim).clamp_min(0)
                mch = dch.min(-1).values
                d1 = mch if d1 is None else torch.minimum(d1, mch)
                if sel.numel():
                    cols.append(dch.index_select(-1, sel))
            q512 = torch.sort(torch.cat(cols, -1), -1).values[..., 511]
            pl = F.normalize(model.token_mapping(fp), dim=-1)
            ql = F.normalize(model.token_mapping(flr), dim=-1)
            dlmin = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0).min(-1).values
            seg = seg_head(pl)[..., 0]
            for k, t in (('d1', d1), ('q512', q512), ('dlmin', dlmin), ('bq64', q512 * 0), ('seg', seg)):
                rec[k].append(t.reshape(-1, g, g).float().cpu())
            rec['is_norm'].append(torch.full((img_batch.shape[0],), float(is_norm)))
            if mask_batch is not None:
                mm = F.interpolate(mask_batch, size=(g, g), mode='nearest')[:, 0]
                rec['mask'].append((mm > 0.5).float().cpu())
            else:
                rec['mask'].append(torch.zeros(img_batch.shape[0], g, g))

    bs = 8
    for i in range(0, normal.shape[0], bs):
        comps(normal[i:i + bs].to(args.device), 1)
    for i in range(0, len(anom), bs):
        chunk = anom[i:i + bs]
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p, _ in chunk], 0)
        msks = torch.stack([gtf(Image.open(m).convert('L')) for _, m in chunk], 0)
        comps(imgs.to(args.device), 0, msks.to(args.device))

    cat = {k: torch.cat(v, 0) for k, v in rec.items()}
    is_norm = cat['is_norm'].numpy()
    print(f"  正常 {int(is_norm.sum())} 张 / 合成异常 {int((1 - is_norm).sum())} 张")

    def image_score(m):
        am = gk(m.to(args.device)).cpu()
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        return torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(1).numpy()

    print(f"\n{'alpha':<6}{'seg_w':<7}{'I-AUROC':>9}{'P-AUROC':>9}{'均值':>8}")
    for al in [float(x) for x in args.alpha_list.split(',')]:
        for sw in [float(x) for x in args.seg_w_list.split(',')]:
            m = 0.6 * cat['dlmin'].unsqueeze(1) + 0.4 * (cat['d1'].clamp_min(1e-6).pow(al)
                                                         * cat['q512'].clamp_min(1e-6).pow(1 - al)).unsqueeze(1)
            if sw > 0:
                sd = cat['seg'].std() + 1e-8
                m = m + sw * ((cat['seg'].unsqueeze(1) - cat['seg'].mean()) / sd)
            isp = image_score(m)
            iauc = roc_auc_score(1 - is_norm, isp)
            # 像素级：只在异常图上取像素（正常图全 0，会稀释） → 用全部像素池更接近线上
            px = m[:, 0].reshape(-1).numpy()
            gy = cat['mask'].reshape(-1).numpy()
            if gy.sum() > 0 and gy.sum() < gy.size:
                pauc = roc_auc_score(gy, px)
            else:
                pauc = float('nan')
            print(f"{al:<6}{sw:<7}{iauc*100:>9.2f}{pauc*100:>9.2f}{(iauc+pauc)/2*100:>8.2f}")


if __name__ == '__main__':
    main()
