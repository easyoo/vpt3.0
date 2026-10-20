# -*- coding: utf-8 -*-
"""[路径2 候选 · 零训练] 异常分数聚合算子族：把"最近邻距离"换成
"综合多个原型距离"的各种鲁棒形式，**只加载已有权重做推理**。

动机：现状 = `s = 1 − max_k cos(patch, proto_k)`，等价于只信任**一个**最近原型。
单个离群/坏原型就能把整片区域压低（漏检），单个侥幸靠近的正常 patch 也会把分数压低。
本脚本一次性把一批聚合算子在同一遍前向里算出来（无需重训），逐类给出官方 adeval 指标。

算子（d_1 ≤ d_2 ≤ … 为到各原型的距离，d̄ 为全体均值）：
  min       : d_1                                  （现状基线）
  kn2/3/5/8 : mean(d_1..d_k)                       （k 近邻平均）
  med3      : median(d_1..d_3)
  har3      : 3 / Σ 1/d_i  (i≤3)                   （调和平均，偏向更近的原型）
  trim      : mean(d_2..d_6)                       （丢掉最近的一个 → 抗"坏原型"）
  minmean   : 0.5·d_1 + 0.5·d̄                       （局部 + 全局混合）
  geom      : sqrt(d_1 · d̄)                         （几何混合）
  psoft     : 相对温度的 softmin over d_1..d_8      （尺度自适应软化）

侧别：`raw` = 原始特征库（PatchCore 式）；`learn` = 可学习原型集（32 局部 + 32 全局锚点）。
用法: python probe_score_agg.py --save m4m6r3_2shot --class transistor --shots 2 --size 448
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
KMAX = 64


def load_mod(path):
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def ops_from(ds, dbar, name):
    """ds: [..., KMAX] 升序距离；dbar: [...] 全体原型距离均值。返回聚合后的分数图向量。"""
    e = 1e-6
    if name == 'min':
        return ds[..., 0]
    if name.startswith('kn'):
        k = int(name[2:])
        return ds[..., :k].mean(dim=-1)
    if name == 'med3':
        return ds[..., :3].median(dim=-1).values
    if name == 'har3':
        return 3.0 / (1.0 / ds[..., :3].clamp_min(e)).sum(dim=-1)
    if name == 'trim':
        return ds[..., 1:6].mean(dim=-1)
    if name == 'minmean':
        return 0.5 * ds[..., 0] + 0.5 * dbar
    if name == 'geom':
        return (ds[..., 0].clamp_min(e) * dbar.clamp_min(e)).sqrt()
    if name == 'psoft':                      # 相对温度的 softmin：τ 由"最近 8 个的散布"决定
        span = (ds[..., 7] - ds[..., 0]).clamp_min(e)
        tau = 0.25 * span
        return -tau * torch.logsumexp(-ds[..., :8] / tau.unsqueeze(-1), dim=-1)
    raise ValueError(name)


RAWOPS = ['min', 'kn2', 'kn3', 'kn5', 'kn8', 'med3', 'har3', 'trim',
          'minmean', 'geom', 'psoft']
LEARNOPS = ['min', 'kn2', 'kn3', 'kn5', 'har3', 'minmean', 'geom']
# 可学习侧的"全局参照"变体：可学习原型只有 64 个且高度冗余（实测 capsules 上各种聚合到小数点后 4 位都一样），
# 所以 `learn_geom` 里的 d̄ 信息量很小。这两个变体改用**过了一遍 token_mapping 的原始库**
# （M = K×784 个真实正常 patch）作为学到的度量空间里的全局参照，与原始库侧 geom 同构。
LEARNBANKOPS = ['geom_bank', 'minmean_bank']


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
    ap.add_argument('--variants', default='',
                    help='只跑指定算子（逗号分隔，如 raw_min,raw_geom,learn_min）；空=全跑')
    ap.add_argument('--no_align', type=int, default=0,
                    help='1 = 关掉类内中位数对齐（c 恒为 1），用于排除"缩放/权重"混淆')
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
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)
    q_raw = F.normalize(bank, dim=-1)

    ds_test = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                               transform=tf, gt_transform=gtf, phase='test', sort_items=False)
    dl = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    scales = tuple(float(x) for x in args.scales.split(','))

    variants = ([('raw_' + o) for o in RAWOPS] + [('learn_' + o) for o in LEARNOPS]
                + [('learn_' + o) for o in LEARNBANKOPS])
    if args.variants:
        keep = [v.strip() for v in args.variants.split(',') if v.strip()]
        for need in ('raw_min', 'learn_min'):
            if need not in keep:
                keep.append(need)
        bad = [v for v in keep if v not in variants]
        if bad:
            raise SystemExit(f'未知算子 {bad}；可选 {variants}')
        variants = keep
        RAWOPS_K = [v.split('_', 1)[1] for v in variants if v.startswith('raw_')]
        LEARNOPS_K = [v.split('_', 1)[1] for v in variants
                      if v.startswith('learn_') and v.split('_', 1)[1] not in LEARNBANKOPS]
    else:
        RAWOPS_K, LEARNOPS_K = RAWOPS, LEARNOPS
    BANKOPS_K = [v.split('_', 1)[1] for v in variants
                 if v.startswith('learn_') and v.split('_', 1)[1] in LEARNBANKOPS]
    if BANKOPS_K:                       # 学到度量空间里的原始库（全局参照），只在需要时算一次
        with torch.no_grad():
            q_bank_l = F.normalize(model.token_mapping(bank), dim=-1)      # [M, D]
    else:
        q_bank_l = None
    store = {v: [] for v in variants}
    base_hw = (args.size, args.size)

    def _agg(sim, side):
        """sim: [B,N,M] 相似度 → 该侧所有算子的距离图 dict{op: [B,1,g,g]}"""
        d = (1.0 - sim).clamp_min(0)
        kk = min(KMAX, d.shape[-1])
        ds, _ = torch.topk(d, kk, dim=-1, largest=False)
        dbar = d.mean(dim=-1)
        out = {}
        for o in (RAWOPS_K if side == 'raw' else LEARNOPS_K):
            s = ops_from(ds, dbar, o)
            g = int(round(s.shape[1] ** 0.5))
            out[o] = s.reshape(-1, 1, g, g)
        return out

    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(args.device)
            accum = {v: None for v in variants}
            n_view = 0
            for s in scales:
                v_img = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear',
                                                           align_corners=False)
                for vi, view in enumerate([v_img, v_img.flip(3)] if args.flip else [v_img]):
                    ret = model(view, is_training=True)
                    fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                    flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
                    if getattr(a, 'global_proto', 0):
                        _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                        flr = torch.cat([flr, _g], dim=1)
                    sim_raw = torch.bmm(F.normalize(fp, dim=-1), q_raw.t().unsqueeze(0)
                                        .expand(fp.shape[0], -1, -1))
                    pl = F.normalize(model.token_mapping(fp), dim=-1)
                    ql = F.normalize(model.token_mapping(flr), dim=-1)
                    sim_learn = torch.bmm(pl, ql.transpose(1, 2))
                    maps = {}
                    for o, m in _agg(sim_raw, 'raw').items():
                        maps['raw_' + o] = m
                    for o, m in _agg(sim_learn, 'learn').items():
                        maps['learn_' + o] = m
                    if BANKOPS_K:
                        # 可学习侧的"全局参照"换成映射到同一度量空间的真实正常 patch（M=K×784），
                        # 与原始库侧 geom 同构，用来区分"公式不适用" vs "参考集太小没法用"。
                        d_l = (1.0 - sim_learn).clamp_min(0).min(dim=-1).values
                        d_b = (1.0 - torch.bmm(pl, q_bank_l.t().unsqueeze(0)
                                               .expand(pl.shape[0], -1, -1))).clamp_min(0).mean(dim=-1)
                        for o in BANKOPS_K:
                            if o == 'geom_bank':
                                s = (d_l.clamp_min(1e-6) * d_b.clamp_min(1e-6)).sqrt()
                            else:                                  # minmean_bank
                                s = 0.5 * d_l + 0.5 * d_b
                            g = int(round(s.shape[1] ** 0.5))
                            maps['learn_' + o] = s.reshape(-1, 1, g, g)
                    for v in variants:
                        am = F.interpolate(maps[v], size=base_hw, mode='bilinear', align_corners=True)
                        if vi == 1:
                            am = am.flip(3)
                        accum[v] = am if accum[v] is None else accum[v] + am
                    n_view += 1
            gt_px = (gt.to(args.device) > 0.5).float()
            if gt_px.shape[1] > 1:
                gt_px = torch.max(gt_px, dim=1, keepdim=True)[0]
            for v in variants:
                store[v].append(accum[v] / max(1, n_view))
            store['__gt_px'] = store.get('__gt_px', [])
            store['__gt_px'].append(gt_px)
            store['__gt_sp'] = store.get('__gt_sp', [])
            store['__gt_sp'].append(label)

    gt_px_all = torch.cat(store.pop('__gt_px'), 0)[:, 0].cpu().numpy()
    gt_sp_all = torch.cat(store.pop('__gt_sp')).flatten().cpu().numpy()
    maps_all = {v: torch.cat(store[v], 0) for v in variants}
    # 尺度对齐：每个算子的图量纲不同（kNN 均值/几何混合都会整体偏移），
    # 用**同类内一个标量**把变体图的中位数对齐到该侧 min 图的中位数，
    # 这样 `raw_min` / `learn_min` 与官方基线逐位相同，比较的纯粹是"排序形状"。
    ref_raw = torch.median(maps_all['raw_min'])
    ref_learn = torch.median(maps_all['learn_min'])

    def _score(am):
        flat = am.flatten(1)
        kk = max(1, int(flat.shape[1] * 0.01))
        return torch.sort(flat, dim=1, descending=True)[0][:, :kk].mean(dim=1)

    def _eval(am):
        am = gk(am)
        return list(mod.ader_evaluator(am[:, 0].cpu().numpy(),
                                       _score(am).flatten().cpu().numpy(), gt_px_all, gt_sp_all))

    print(f'=== {args.cls} (save={args.save} shot={args.shots}) ===')
    for v in variants:
        am = maps_all[v]
        side, op = v.split('_', 1)
        if side == 'raw':
            c = 1.0 if args.no_align else (ref_raw / torch.median(am).clamp_min(1e-8)).item()
            fused = args.w_learn * maps_all['learn_min'] + (1.0 - args.w_learn) * (c * am)
        else:
            c = 1.0 if args.no_align else (ref_learn / torch.median(am).clamp_min(1e-8)).item()
            fused = args.w_learn * (c * am) + (1.0 - args.w_learn) * maps_all['raw_min']
        r = _eval(fused)
        print(f"  {v:<14}" + "  ".join(f"{n}={x:.4f}" for n, x in zip(NAMES, r)))
        print("  CSV," + f"{args.cls},{v}," + ",".join(f"{x:.4f}" for x in r))
        if side == 'raw':      # 裸分支：排序指标对缩放不变 → 完全隔离"打分形状"的贡献
            r2 = _eval(am)
            print(f"  {v+'_rawonly':<14}" + "  ".join(f"{n}={x:.4f}" for n, x in zip(NAMES, r2)))
            print("  CSV," + f"{args.cls},{v}_rawonly," + ",".join(f"{x:.4f}" for x in r2))
        if side == 'learn':
            r2 = _eval(am)
            print(f"  {v+'_learnonly':<14}" + "  ".join(f"{n}={x:.4f}" for n, x in zip(NAMES, r2)))
            print("  CSV," + f"{args.cls},{v}_learnonly," + ",".join(f"{x:.4f}" for x in r2))


if __name__ == '__main__':
    main()
