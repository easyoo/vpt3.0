# -*- coding: utf-8 -*-
"""[路径2 候选 · 零训练] 打分几何的鲁棒化：白化(Mahalanobis)最近邻 vs 现状的余弦最近邻。

背景（已排除项，见 EXPERIMENT_LOG.md §10 / RESULTS_v5.md §15）：
  * softmin（把 min 软化）、子空间残差、k-NN 邻域聚合、局部对比度 d−邻域均值
    —— **全部为负**。它们的共同点是"只换聚合算子，不动距离度量本身"。
本脚本换的是**度量**：先用正常 patch 库估计协方差，做白化（去相关 + 各向异性重标定），
再在新的几何里取最近邻。直觉：现状的余弦距离把 768 个方向一视同仁；而"正常流形"在某些
方向上本来就抖动很大、在另一些方向上极其稳定——后者的偏离才应该被判为异常。

变体：
  base       : 现状 = 1 − max cos（必须复现官方 2-shot mapping 基线）
  knn        : 余弦口径下取 k 个最近邻的**均值**（对照：只换聚合）
  white      : 白化后的余弦最近邻（Mahalanobis 风格）
  white_knn  : 白化后的 k 近邻均值
只改 Q_raw（原始特征库）这一侧的空间；可学习原型侧保持不变，便于归因。

用法: python probe_whiten.py --save m4m6r3_2shot --class transistor --shots 2 \
                             --size 448 --lam 0.1 --k 3 [--variants base,white,...]
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
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash         # noqa: E402
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load_mod(path):
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def whitener(bank, lam, dim=0):
    """由正常 patch 库估计白化矩阵 W ≈ Σ^(-1/2)（含收缩正则）。

    lam: 收缩系数（0=纯经验协方差，1=各向同性→退化成现状）。
    dim: >0 时只对**最大的 dim 个主方向**做白化，其余方向用剩余特征值均值做各向同性缩放
         —— 2-shot 下 M=1568 个 patch 估 768 维协方差是欠定的，尾部特征值纯噪声，
         截断即"部分白化"。
    """
    B = bank.float()
    Bc = B - B.mean(dim=0, keepdim=True)
    D = B.shape[1]
    n = max(1, Bc.shape[0] - 1)
    cov = (Bc.t() @ Bc) / n
    eye = torch.eye(D, device=cov.device, dtype=cov.dtype)
    cov = (1.0 - lam) * cov + lam * (torch.trace(cov) / D) * eye
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(1e-8)
    if dim and dim < D:
        dd = D - dim                                   # eigh 升序 → 尾部是最大的 dim 个
        tail = evals[dd:]
        head = torch.cat([evals[:dd].mean().expand(dd), tail])
        scale = torch.cat([head[:dd].rsqrt(), tail.rsqrt()])
    else:
        scale = evals.rsqrt()
    W = evecs @ torch.diag(scale) @ evecs.t()
    return W


def local_scale(bank_norm, k=8):
    """密度归一化用：每个正常 patch 到它第 k 近的**其它正常 patch**的距离。bank_norm: [M,D] 已归一化。"""
    S = bank_norm @ bank_norm.t()
    S.fill_diagonal_(-1e9)
    kk = min(k, S.shape[0] - 1)
    sim_k = torch.topk(S, kk, dim=-1).values[:, -1]     # 第 k 大相似度 = 第 k 近
    return (1.0 - sim_k).clamp_min(1e-6)


def min_or_knn(sim, k):
    """sim: [B, N, M] 相似度；k=1 取 max（现状），k>1 取 top-k 的均值。"""
    if k <= 1:
        return sim.max(dim=-1).values
    kk = min(k, sim.shape[-1])
    return torch.topk(sim, kk, dim=-1).values.mean(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='m4m6r3_2shot')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--shots', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--scales', default='1.5')
    ap.add_argument('--flip', type=int, default=1)
    ap.add_argument('--w_learn', type=float, default=0.5)
    ap.add_argument('--lam', type=float, default=0.1, help='协方差收缩系数')
    ap.add_argument('--k', type=int, default=3, help='kNN 的 k')
    ap.add_argument('--variants', default='base,knn,white,white_knn')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.softmin = 0
    a.learnable_num_per_layer = args.num_tokens
    a.res_mapping = 0
    a.global_proto = 1
    a.metric_head = 1
    a.subspace = 0
    a.layer_scale_learn = 0
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    a.ms_bank_proto = 0
    mod.args = a
    mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ckpt = os.path.join(args.ckpt_dir or os.path.join('results', 'vpt4', args.save),
                        f'model_{args.cls}_35.pth')
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(args.device).eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)      # [M, D] 未归一化
    q_raw = F.normalize(bank, dim=-1)
    lof_scale = local_scale(q_raw, k=8)                              # [M]

    # ---- 打分变体注册表：name -> (库向量[已归一化], kNN 的 k, 密度归一化标志) ----
    scorers = {'base': (q_raw, 1, False)}
    for name, lam, dim in (('w10', 0.1, 0), ('w30', 0.3, 0), ('w60', 0.6, 0),
                           ('w90', 0.9, 0), ('pca128', 0.3, 128), ('pca256', 0.3, 256)):
        W = whitener(bank, lam, dim)
        scorers[name] = (F.normalize(bank @ W.t(), dim=-1), 1, False)
    scorers['knn3'] = (q_raw, args.k, False)
    scorers['lof8'] = (q_raw, 1, True)
    scorers['lof8_pca256'] = (scorers['pca256'][0], 1, True)

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test', sort_items=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    scales = tuple(float(x) for x in args.scales.split(','))
    want = [v.strip() for v in args.variants.split(',') if v.strip()]
    bad = [v for v in want if v not in scorers]
    if bad:
        raise SystemExit(f'未知变体 {bad}；可选：{sorted(scorers)}')

    # 落盘策略：只存"未平滑"的图（可学习侧 1 份 + 每个变体 1 份），
    # 平滑与融合放在评估阶段，保证与官方口径（先融合后单次高斯）一致。
    raw_store = {v: [] for v in want}
    learn_store = []
    gt_px_all, gt_sp_all = [], []

    def _map28(patch_feat, proto_feat):
        p = F.normalize(model.token_mapping(patch_feat), dim=-1)
        q = F.normalize(model.token_mapping(proto_feat), dim=-1)
        s = (1.0 - torch.bmm(p, q.transpose(1, 2))).clamp_min(0).min(dim=-1).values
        g = int(round(s.shape[1] ** 0.5))
        return s.reshape(-1, 1, g, g)

    def _raw_map(fp, name):
        q_vec, k, use_lof = scorers[name]
        p = F.normalize(fp, dim=-1)
        sim = torch.bmm(p, q_vec.t().unsqueeze(0).expand(p.shape[0], -1, -1))
        if use_lof:                       # 密度归一化：除以最近正常 patch 自身的局部尺度
            best = sim.max(dim=-1)
            s = (1.0 - best.values).clamp_min(0) / lof_scale[best.indices].clamp_min(1e-6)
        else:
            s = (1.0 - min_or_knn(sim, k)).clamp_min(0)
        g = int(round(s.shape[1] ** 0.5))
        return s.reshape(-1, 1, g, g)

    base_hw = (args.size, args.size)
    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(args.device)
            sum_raw = {v: None for v in want}
            sum_learn = None
            n_view = 0
            for s in scales:
                v_img = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear',
                                                           align_corners=False)
                for vi, view in enumerate([v_img, v_img.flip(3)] if args.flip else [v_img]):
                    ret = model(view, is_training=True)
                    fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                    flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
                    _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                    flr = torch.cat([flr, _g], dim=1)
                    m_learn = _map28(fp, flr)
                    l_am = F.interpolate(m_learn, size=base_hw, mode='bilinear', align_corners=True)
                    if vi == 1:
                        l_am = l_am.flip(3)
                    sum_learn = l_am if sum_learn is None else sum_learn + l_am
                    for v in want:
                        am = F.interpolate(_raw_map(fp, v), size=base_hw, mode='bilinear',
                                           align_corners=True)
                        if vi == 1:
                            am = am.flip(3)
                        sum_raw[v] = am if sum_raw[v] is None else sum_raw[v] + am
                    n_view += 1
            gt_px = (gt.to(args.device) > 0.5).float()
            if gt_px.shape[1] > 1:
                gt_px = torch.max(gt_px, dim=1, keepdim=True)[0]
            learn_store.append(sum_learn / max(1, n_view))
            for v in want:
                raw_store[v].append(sum_raw[v] / max(1, n_view))
            gt_px_all.append(gt_px)
            gt_sp_all.append(label)

    gt_px_all = torch.cat(gt_px_all, 0)[:, 0].cpu().numpy()
    gt_sp_all = torch.cat(gt_sp_all).flatten().cpu().numpy()
    learn_map = torch.cat(learn_store, 0)

    def _score(am):
        flat = am.flatten(1)
        kk = max(1, int(flat.shape[1] * 0.01))
        return torch.sort(flat, dim=1, descending=True)[0][:, :kk].mean(dim=1)

    def _eval(am):
        am = gk(am)
        return list(mod.ader_evaluator(am[:, 0].cpu().numpy(),
                                       _score(am).flatten().cpu().numpy(),
                                       gt_px_all, gt_sp_all))

    print(f'=== {args.cls} (save={args.save} shot={args.shots}) ===')
    for v in want:
        raw = torch.cat(raw_store[v], 0)
        # 变体图与余弦图量纲对齐：用**类内一个标量**（只影响 0.5/0.5 融合，不影响 raw-only 的排序指标）
        c = (torch.median(torch.cat(raw_store['base'], 0)) /
             torch.median(raw).clamp_min(1e-8)).item()
        for tag, am in (('raw', raw), ('blend', args.w_learn * learn_map
                                       + (1.0 - args.w_learn) * (c * raw))):
            r = _eval(am)
            print(f"  {v+'-'+tag:<16}" + "  ".join(f"{n}={x:.4f}" for n, x in zip(NAMES, r)))
            print("  CSV," + f"{args.cls},{v}_{tag}," + ",".join(f"{x:.4f}" for x in r))


if __name__ == '__main__':
    main()
