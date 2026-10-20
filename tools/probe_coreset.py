# -*- coding: utf-8 -*-
"""[审计/机制验证探针] mapping 分支"原型集"的替换实验（零训练成本）。

背景（已度量的事实）：
  token_mapping(fused_learnable_token) 这组"原型"在训练后几乎共线
  （hazelnut@30ep：两两余弦均值 0.87，softmax 有效原型数 ≈1.75/32）。
  也就是说 mapping 分支名义上有 32 个原型，实际接近 1 个全局模板。

本脚本在"冻结已训练权重、只改原型来源"的前提下做对照，回答：
  Q1 把原型换成"映射空间里真正有区分度的原型集"（k-center coreset），指标会不会变好？
  Q2 如果只用 1 个原型（全局均值），会掉多少？→ 用于定位"多样性"的价值。

注意：所有变体共享同一套打分流程（最近原型余弦距离 → 双线性上采样 → 高斯平滑
      → top-1% 像素均值作图像级分数 → ader_evaluator），只有原型来源不同。
"""
import argparse
import importlib.util
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(script_path):
    spec = importlib.util.spec_from_file_location('audit_main', script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(mod, ckpt, device):
    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(device).eval()
    return model, a


@torch.no_grad()
def mapped_tokens(model, imgs, fused_layer):
    """返回 (mapped_patch [B,N,D], mapped_proto [B,K,D])"""
    ret = model(imgs, is_training=True)
    pfl, fl = ret['patch_feat_layer'], ret['feat_list']
    fp = torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
    flr = torch.sum(torch.stack(fl, dim=0), dim=0)
    return model.token_mapping(fp), model.token_mapping(flr)


@torch.no_grad()
def kcenter_coreset(feats, k, seed=0):
    """feats: [N,D] 已 L2 归一化；贪心 k-center（PatchCore 做法）。"""
    g = torch.Generator(device=feats.device).manual_seed(seed)
    n = feats.shape[0]
    first = torch.randint(0, n, (1,), generator=g, device=feats.device).item()
    centers = [first]
    d = 1.0 - feats @ feats[first]
    for _ in range(k - 1):
        idx = int(torch.argmax(d).item())
        centers.append(idx)
        d = torch.minimum(d, 1.0 - feats @ feats[idx])
    return torch.stack([feats[i] for i in centers], dim=0)


@torch.no_grad()
def score_maps(patch_map, protos, out_size, chunk=4096):
    """patch_map [N,D], protos [K,D] → [1,1,out,out] 最近原型余弦距离图"""
    p = F.normalize(patch_map, dim=-1)
    q = F.normalize(protos, dim=-1)
    outs = []
    for i in range(0, p.shape[0], chunk):
        sim = p[i:i + chunk] @ q.T
        outs.append((1.0 - sim).clamp_min(0).min(dim=-1).values)
    s = torch.cat(outs, dim=0)
    gsz = int(round(s.shape[0] ** 0.5))
    am = s.reshape(1, 1, gsz, gsz).contiguous()
    return F.interpolate(am, size=out_size, mode='bilinear', align_corners=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--protocol', type=int, default=392)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code', 'main-fewshot-audit.py'))
    ap.add_argument('--out', default=None, help='结果 csv 追加路径')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))  # 用 --script 所在目录的 dinov2 副本
    mod = load_mod(args.script)
    device = args.device
    fused_layer = [1, 2, 3, 4, 6, 8, 11]
    model, _ = build_model(mod, args.ckpt, device)

    data_transform, gt_transform = mod.get_data_transforms(448, 392)
    test_data = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                                 transform=data_transform, gt_transform=gt_transform, phase='test', sort_items=False)
    dl = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ---- 1) 收集正常训练样本的映射 patch（few-shot 的同一批 4 张，含增强，与训练口径一致）
    random.seed(args.seed)
    tr_root = os.path.join(args.data_path, args.cls, 'train', 'good')
    files = sorted(os.listdir(tr_root))
    files = random.sample(files, args.shots)
    normal_imgs = []
    for f in files:
        im = Image.open(os.path.join(tr_root, f)).convert('RGB')
        normal_imgs.append(data_transform(im))
    normal_imgs = torch.stack(normal_imgs, dim=0).to(device)
    mp, mproto = mapped_tokens(model, normal_imgs, fused_layer)
    bank = mp.reshape(-1, mp.shape[-1])                      # [shots*784, D]
    bank_n = F.normalize(bank, dim=-1)
    coreset = kcenter_coreset(bank_n, args.k)                # [K, D]
    glob_mean = F.normalize(mproto.reshape(-1, mproto.shape[-1]), dim=-1).mean(0, keepdim=True)
    glob_mean = F.normalize(glob_mean, dim=-1)
    pn = F.normalize(mproto[0], dim=-1)          # 单张正常图的 K 个原型
    C = pn @ pn.T
    K = C.shape[0]
    cos_mean = C[~torch.eye(K, dtype=bool)].mean().item()
    sv = torch.linalg.svdvals(mproto[0] - mproto[0].mean(0, keepdim=True))
    sv = sv / sv.sum()
    eff_rank = float(1.0 / (sv ** 2).sum())
    # 与训练损失同口径：patch→原型的 softmax 权重熵（temperature=0.07）
    x = F.normalize(mp[0], dim=-1)
    psoft = F.softmax(x @ pn.T / 0.07, dim=-1)
    eff_k = float(torch.exp(-(psoft * torch.log(psoft + 1e-12)).sum(-1)).mean())
    print(f"[{args.cls}] 正常样本 {args.shots} 张 → bank {tuple(bank.shape)}，coreset {tuple(coreset.shape)}")
    print(f"[{args.cls}] 【原型坍缩度量·真实前向】两两余弦均值={cos_mean:.4f}；"
          f"patch→原型 softmax 有效原型数≈{eff_k:.2f}/{mproto.shape[1]}；"
          f"参与比有效秩={eff_rank:.2f}/{mproto.shape[1]}")

    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rm = None if args.protocol == 392 else args.protocol
    variants = ['learnable_image', 'learnable_global_mean', f'coreset{args.k}', f'coreset{args.k}+learnable']

    rows = {}
    for v in variants:
        gt_px, pr_px, gt_sp, pr_sp = [], [], [], []
        for img, gt, label, _ in tqdm(dl, ncols=60, desc=v):
            img = img.to(device)
            p_map, p_proto = mapped_tokens(model, img, fused_layer)  # [B,N,D],[B,K,D]
            for b in range(img.shape[0]):
                if v == 'learnable_image':
                    protos = p_proto[b]
                elif v == 'learnable_global_mean':
                    protos = glob_mean
                elif v.startswith('coreset') and '+' not in v:
                    protos = coreset
                else:
                    protos = torch.cat([coreset, F.normalize(p_proto[b], dim=-1)], dim=0)
                am = score_maps(p_map[b], protos, img.shape[-1])
                if rm is not None:
                    am = F.interpolate(am, size=rm, mode='bilinear', align_corners=False)
                    gtb = F.interpolate(gt[b:b + 1], size=rm, mode='nearest')
                else:
                    gtb = gt[b:b + 1]
                am = gk(am)
                gtb = (gtb > 0.5).float()
                gt_px.append(gtb[0, 0].cpu().numpy())
                pr_px.append(am[0, 0].cpu().numpy())
                gt_sp.append(int(label[b]))
                flat = am.flatten(1)
                k_top = max(1, int(flat.shape[1] * 0.01))
                pr_sp.append(torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean().item())
        r = mod.ader_evaluator(np.stack(pr_px), np.array(pr_sp),
                               np.stack(gt_px), np.array(gt_sp))
        rows[v] = r

    names = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    print(f"\n[{args.cls}] ckpt={os.path.basename(args.ckpt)} protocol={args.protocol}")
    print("  " + " " * 22 + " ".join(f"{n:>8}" for n in names))
    for v in variants:
        print(f"  {v:<22}" + " ".join(f"{x:8.4f}" for x in rows[v]))
    if args.out:
        with open(args.out, 'a') as fh:
            for v in variants:
                fh.write(f"{args.cls},{os.path.basename(args.ckpt)},{args.protocol},{v}," +
                         ",".join(f"{x:.4f}" for x in rows[v]) + "\n")


if __name__ == '__main__':
    main()
