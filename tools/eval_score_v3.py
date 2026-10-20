# -*- coding: utf-8 -*-
"""[vpt3.7 生产版] 重推 mapping 异常图并落盘，用于与 bank 融合。

**v3.7 默认配置 = 简化版**（相对 v3.6 去掉两处组件，见 docs/ABLATION_2026-10-13.md §5）：
  raw_score   = d1^0.5 * q50^0.5                    # 原始库侧：最近邻 × 全库距离中位数
  learn_score = dlmin                                 # 可学习侧：仅"到最近原型"，**不再有全局项**
  final       = w*learn_score + (1-w)*raw_score       # w=0.6；**单视图**（1.5×，无翻转）
默认值：--l_global 0（关闭 learn 侧全局项，同时不再对 bank 做 token_mapping）
        --flip 0      （关闭翻转视图 → 推理前向次数减半）
要恢复 v3.6 行为：`--l_global 64 --flip 1`。

用法:
  python eval_score_v3.py --save vpt35_visa4shot --class candle --shots 4 \
    --data_path /home/jjquan/experiment/visual2/visa_prepared \
    --ckpt_dir results/visa/vpt35_visa4shot \
    --save_maps_dir .maps_simple_nocrop448_visa4        # 其余参数用默认即 v3.7 配方
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash         # noqa: E402
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
SUBN = 1024


def _default_script():
    """代码目录自动探测：兼容"仓库包布局"（../code）与"工作目录布局"（./code_vpt4）。"""
    for c in (os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'),
              os.path.join(HERE, '..', 'code', 'main-fewshot-vpt4.py'),
              os.path.join(HERE, 'main-fewshot-vpt4.py')):
        if os.path.exists(c):
            return os.path.normpath(c)
    return os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')


def load_mod(path):
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def _pow(local, glob, a):
    e = 1e-6
    return local.clamp_min(e).pow(a) * glob.clamp_min(e).pow(1.0 - a)


def write_sidecar(map_dir, cls, ds, size, extra=None):
    """[FIX-4] 落盘样本顺序侧车：让下游（fuse 等）能把 GT/标签对齐到生成异常图时的顺序。"""
    if not map_dir:
        return
    rec = {'cls': cls, 'n': len(ds), 'size': int(size),
           'list_hash': getattr(ds, 'list_hash', None),
           'keys': list(getattr(ds, 'keys', []) or []),
           'config': extra or {}}
    p = os.path.join(map_dir, f'{cls}.json')
    with open(p, 'w') as fh:
        json.dump(rec, fh, ensure_ascii=False)
    print(f'  [sidecar] 已写 {p}（n={rec["n"]} hash={rec["list_hash"]}）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='vpt35_visa4shot')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/experiment/visual2/visa_prepared')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--pth_epoch', type=int, default=35,
                    help='[训练侧变量] 使用第几个 epoch 的权重（默认 35 = 历史行为）')
    ap.add_argument('--backbone', type=str, default='base',
                    help="[主干实验] 'base'=DINOv2 ViT-B/14（默认）；'large'=DINOv2 ViT-L/14")
    ap.add_argument('--fused_layers', type=str, default='1,2,3,4,6,8,11',
                    help='[主干实验] 参与相加的 patch 特征层（ViT-L 建议用先验·同相对深度 "2,4,6,8,12,16,22"）')
    ap.add_argument('--res_mapping', type=int, default=0, help='须与训练一致（0=低秩瓶颈，1=残差）')
    ap.add_argument('--global_proto', type=int, default=1, help='须与训练一致（1=⊕32 全局锚点）')
    ap.add_argument('--proto_set', type=str, default='all',
                    help='[CLUSTER] 打分用的原型集：all=v3.7 原行为(96，含 32 个重复)；'
                         'unique=去重(64)；anchor=仅 32 锚点；layer=仅 32 layer token')
    ap.add_argument('--metric_head', type=int, default=1, help='须与训练一致（1=可学习度量头 W）')
    ap.add_argument('--scales', default='1.5')
    ap.add_argument('--flip', type=int, default=0,
                    help='[v3.7 默认 0] 关闭翻转视图 → 单视图（1.5×），推理前向次数减半；置 1 恢复 v3.6 双视图')
    ap.add_argument('--raw_alpha', type=float, default=0.5)
    ap.add_argument('--raw_global', type=int, default=512,
                    help='raw 全局项分位（1024 子采样中第 k 小）；**-1 = 用全库精确中位数**（无子采样旋钮）')
    ap.add_argument('--l_alpha', type=float, default=0.7)
    ap.add_argument('--l_global', type=int, default=0,
                    help='[v3.7 默认 0] learn 侧全局项分位；0 = 关闭该项（s_learn 退化为 dlmin，'
                         '且不再对 bank 做 token_mapping）；置 64 恢复 v3.6 行为')
    ap.add_argument('--l_score_mode', type=int, default=0,
                    help='[探索] 可学习侧打分的全局项来源：'
                         '0=dlmin（v3.7 现状，无全局项）；'
                         '1=dlmin^0.5·dlmed^0.5；2=dlmin^0.7·dlmed^0.3'
                         '（dlmed = 到**全部可学习原型**距离的中位数）；'
                         '3=soft-min(-τ·logΣexp(-d/τ)，簇混合似然)；4=最近 k 个簇的平均距离')
    ap.add_argument('--l_softmin_tau', type=float, default=0.05,
                    help='[CLUSTER] l_score_mode=3 的 soft-min 温度 τ')
    ap.add_argument('--l_topk', type=int, default=5,
                    help='[CLUSTER] l_score_mode=4 的平均最近 k 个原型距离')
    ap.add_argument('--whiten_shrink', type=float, default=0.5,
                    help='[CLUSTER] l_score_mode=7 的方差收缩系数 λ（0=不收缩，1=完全各向同性）')
    ap.add_argument('--kmeans_clusters', type=int, default=16,
                    help='[CLUSTER] l_score_mode=10 的簇中心个数')
    ap.add_argument('--kmeans_iters', type=int, default=10,
                    help='[CLUSTER] l_score_mode=10 的 k-means 迭代次数')
    ap.add_argument('--w', type=float, default=0.6,
                    help='[v3.7 默认 0.6] learn 侧权重（v3.6 亦为 0.6；融合 mapping⊕bank 的权重是 fuse_nocrop.py 的 --w）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--script', default=_default_script())
    ap.add_argument('--sub_seed', type=int, default=1234)
    ap.add_argument('--batch', type=int, default=16, help='测试 DataLoader batch（只影响显存，不影响结果）')
    ap.add_argument('--bank_chunk', type=int, default=8, help='建库时的前向分块（只影响显存）')
    ap.add_argument('--raw_bank_augment', action='store_true',
                    help='[去原始bank] mapping raw 侧改用与 MemoryBank 相同的 22× 增强集；'
                         '4-shot 原图作为 88 张增强集的子集，不再单独建库')
    ap.add_argument('--raw_bank_parent_select_originals', action='store_true',
                    help='[bank 层级] 只构建 88 张增强父库，再从父库中按索引取 4 个'
                         'identity 视图作为 raw 子库；输出与单独构建原始库一致')
    ap.add_argument('--raw_bank_parent_random_k', type=int, default=0,
                    help='[仅增强父库] 从 88 张增强父库的 raw patch 特征中按固定种子'
                         '抽取 K 个代表；不包含任何 identity 子库选择逻辑')
    ap.add_argument('--raw_bank_photometric', action='store_true',
                    help='[仅增强父库] raw 参考库使用几何对齐的外观增强视图：'
                         '原图 + 水平翻转 + 灰度（每 shot 3 张，共 12 张）')
    ap.add_argument('--learn_bank_proto', action='store_true',
                    help='[单增强 bank] 把增强 bank 的 patch 特征经 token_mapping 后并入'
                         '可学习原型集；不建立第二份 bank')
    ap.add_argument('--bank_scale', type=float, default=1.0,
                    help='[对齐实验] 建原始特征库时对 shot 图的缩放；1.0=历史行为（库在 448 尺度），'
                         '1.5=与查询视图同一尺度（672）→ 消除"库 1.0× vs 查询 1.5×"的尺度错配')
    ap.add_argument('--raw_feat', type=int, default=0,
                    help='[原始侧特征定义] 0=历史行为（多层特征**直接相加**再归一化，数值上≈只用最后一层）；'
                         '1=**逐层 L2 归一化后相加**（真正用上多层；只改 raw 打分路径，'
                         '不改变 learn 侧输入 fp，训练权重仍然有效）')
    ap.add_argument('--raw_score_mode', type=int, default=0,
                    help='[原始侧打分] 0=历史 (d1^a·q^(1-a))；'
                         '1=**比值检验** d1/d2（d2=第二近的正常 patch 距离，Lowe ratio，抑制"模糊匹配"）；'
                         '2=比值的几何平均版本 sqrt(d1/d2)·sqrt(q)')
    ap.add_argument('--chunk', type=int, default=1024,
                    help='[精简] raw 侧按该宽度分块计算，避免 [B,N,M] 大张量；结果逐位不变')
    ap.add_argument('--segmap_mode', type=int, default=0,
                    help='[探索] 使用训练过但推理未用的**分割头输出**：'
                         '0=关闭（默认，原行为）；1=solo（分割头图直接当可学习侧打分，'
                         '走 --w 加权与 raw 侧融合）；2=与可学习侧相加（需 --w_segmap 调权重）')
    ap.add_argument('--w_segmap', type=float, default=0.0,
                    help='[探索] segmap_mode=2 时分割头图的相加权重')
    ap.add_argument('--segmap_out_dir', default='',
                    help='[探索] 非空时把分割头图单独落盘到该目录（<cls>.map，未平滑），'
                         '供与 mapping / bank 做第三方融合')
    ap.add_argument('--save_maps_dir', default='')
    args = ap.parse_args()
    if args.raw_bank_augment and args.raw_bank_parent_select_originals:
        ap.error('--raw_bank_augment 与 --raw_bank_parent_select_originals 不能同时开启；'
                 '前者把整个父库当 raw bank，后者保持原性能地取 identity 子库')
    if args.raw_bank_parent_random_k and not args.raw_bank_augment:
        ap.error('--raw_bank_parent_random_k 需要同时开启 --raw_bank_augment')
    if args.raw_bank_photometric and args.raw_bank_augment:
        ap.error('--raw_bank_photometric 与 --raw_bank_augment 不能同时开启')

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [int(x) for x in args.fused_layers.split(',') if x != '']
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.softmin = 0
    a.learnable_num_per_layer = args.num_tokens
    a.res_mapping = args.res_mapping
    a.global_proto = args.global_proto
    a.metric_head = args.metric_head
    a.subspace = 0
    a.layer_scale_learn = 0
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    a.ms_bank_proto = 0
    mod.args = a
    mod.device = args.device

    if str(args.backbone).startswith('dinov3'):
        # [主干实验] DINOv3 主干（RoPE）；可学习 token / 映射模块与 v3.7 完全同一份实现
        from dinov3_mapping import build_dinov3
        model = build_dinov3(args.backbone, a, img_size=518)
    elif str(args.backbone).startswith('clip'):
        # [主干实验] CLIP 主干（不同特征家族）；核心创新模块同上
        from clip_mapping import build_clip
        model = build_clip(args.backbone, a, img_size=518)
    else:
        model = mod.vision_transformer.__dict__['vit_' + args.backbone](
            patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
            num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ckpt = os.path.join(args.ckpt_dir or os.path.join('results', 'vpt4', args.save),
                        f'model_{args.cls}_{args.pth_epoch}.pth')
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['dino'],
                          strict=False)
    model.to(args.device).eval()
    # [探索] 分割头：训练时由 L_seg 监督，原推理流程完全没用 → 这里可选地作为第三张异常图
    seg_head = None
    if int(args.segmap_mode) > 0 or args.segmap_out_dir:      # 保存 seg 图也需要构建分割头
        seg_head = torch.nn.Sequential(
            torch.nn.Linear(model.embed_dim, model.embed_dim // 2), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(model.embed_dim // 2, 1), torch.nn.Sigmoid()).to(args.device)
        seg_head.load_state_dict(torch.load(ckpt, map_location='cpu',
                                            weights_only=False)['seg_head'])
        seg_head.eval()
        print(f'[segmap] 启用分割头图：mode={args.segmap_mode} w_segmap={args.w_segmap}')

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    raw_ref = normal
    if abs(args.bank_scale - 1.0) > 1e-8:
        normal = F.interpolate(normal, scale_factor=args.bank_scale, mode='bilinear',
                               align_corners=False)
        raw_ref = normal
        print(f'[bank_scale] 原始库按 {args.bank_scale}× 重建 → {tuple(normal.shape[-2:])}')
    if args.raw_bank_augment:
        raw_ref = torch.cat([mod.augmentation(im) for im in raw_ref], 0)
        print(f'[raw_bank_augment] raw 侧使用增强 bank：{tuple(raw_ref.shape)} '
              f'（4-shot 原图包含在该集合中）')
    if args.raw_bank_photometric:
        raw_ref = torch.stack([
            torch.stack([im, mod.hflip_img(im.unsqueeze(0)).squeeze(0),
                         mod.grey_img(im.unsqueeze(0)).squeeze(0)], 0)
            for im in raw_ref
        ], 0).flatten(0, 1)
        print(f'[raw_bank_photometric] raw 侧使用外观增强父库：{tuple(raw_ref.shape)}')
    parent_origin_indices = None
    if args.raw_bank_parent_select_originals:
        parent_parts, parent_origin_indices = [], []
        for im in raw_ref:
            views = mod.augmentation(im)
            same = (views - im.unsqueeze(0)).abs().amax(dim=(1, 2, 3)) <= 1e-7
            hit = same.nonzero(as_tuple=False).flatten()
            if hit.numel() != 1:
                raise RuntimeError(f'增强父库中 identity 视图数量异常：{hit.numel()}')
            parent_origin_indices.append(len(parent_parts) * views.shape[0] + int(hit[0]))
            parent_parts.append(views)
        raw_ref = torch.cat(parent_parts, 0)
        print(f'[bank-parent] 仅构建增强父库 {tuple(raw_ref.shape)}；'
              f'identity 子库索引={parent_origin_indices}')
    def _raw_feat(model_, imgs_):
        """按 --raw_feat 的定义算原始侧 patch 特征（build_raw_bank 的等价实现）。"""
        outs = []
        for _i in range(0, imgs_.shape[0], max(1, int(args.bank_chunk))):
            _ret = model_(imgs_[_i:_i + max(1, int(args.bank_chunk))].to(args.device),
                          is_training=True)
            _parts = [_ret['patch_feat_layer'][j] for j in a.fused_layer]
            if int(args.raw_feat) == 1:
                _parts = [F.normalize(_p, dim=-1) for _p in _parts]
            _fp = torch.sum(torch.stack(_parts, 0), 0)
            outs.append(F.normalize(_fp, dim=-1).reshape(-1, _fp.shape[-1]))
        return torch.cat(outs, 0)

    if int(args.raw_feat) == 0:
        bank = mod.build_raw_bank(model, raw_ref, a.fused_layer, k=0, chunk=int(args.bank_chunk))
    else:
        bank = _raw_feat(model, raw_ref)
        print(f'[raw_feat] raw 侧改用逐层 L2 归一化后相加 → bank {tuple(bank.shape)}')
    if parent_origin_indices is not None:
        patches_per_image = bank.shape[0] // raw_ref.shape[0]
        child_rows = torch.cat([
            torch.arange(i * patches_per_image, (i + 1) * patches_per_image,
                         device=bank.device, dtype=torch.long)
            for i in parent_origin_indices
        ])
        bank = bank.index_select(0, child_rows)
        print(f'[bank-parent] 从父库 {len(parent_origin_indices)} 个 identity 视图抽出 '
              f'raw 子库 {tuple(bank.shape)}')
    if args.raw_bank_parent_random_k:
        k = min(int(args.raw_bank_parent_random_k), bank.shape[0])
        gen_bank = torch.Generator().manual_seed(args.sub_seed)
        idx = torch.randperm(bank.shape[0], generator=gen_bank)[:k].to(bank.device)
        bank = bank.index_select(0, idx)
        print(f'[bank-parent-random] 从增强父库固定采样 {tuple(bank.shape)}')
    M = bank.shape[0]
    q_raw = F.normalize(bank, dim=-1)
    gen = torch.Generator().manual_seed(args.sub_seed)
    sub_idx = torch.randperm(M, generator=gen)[:min(SUBN, M)].to(args.device)
    _simple_learn = int(args.l_global) <= 0     # [简化] 关掉 learn 侧全局项 → s_learn = dlmin
    if _simple_learn:
        q_bank_l = None
        print('[simple] learn 侧全局项已关闭：s_learn = dlmin（不再对 bank 做 token_mapping）')
    else:
        # [精简] 该全局项只用到 1024 个子采样 patch，因此**只映射这批**（原来的全量映射是冗余的）
        with torch.no_grad():
            q_bank_l = F.normalize(model.token_mapping(bank.index_select(0, sub_idx)), dim=-1)
    # [精简] raw 侧分块：只按块取 min（d1）与收集子采样列（q50），不再物化 [B,N,M] 大张量
    _ch = max(1, int(args.chunk))
    _plan = []
    for _i in range(0, M, _ch):
        _j = min(_i + _ch, M)
        _sel = sub_idx[(sub_idx >= _i) & (sub_idx < _j)] - _i
        _plan.append((_i, _j, _sel))

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    # [FIX-4] 打印并随图落盘样本顺序（排序确定 + 哈希可校验）
    print(f'  [samples] {len(ds)} 个测试样本 list_hash={ds.list_hash}')
    dl = torch.utils.data.DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    scales = tuple(float(x) for x in args.scales.split(','))

    # [CLUSTER · 打分配置] l_score_mode=7：用 K 张 shot 在**测试尺度**上的 patch 特征做类内白化，
    # 再在可学习原型空间里算 soft-min（仍是**一张图、一个模型**，只是距离度量换成白化欧氏）。
    _mu = _sd = _centers = _pscale = None
    if int(args.l_score_mode) == 7:
        _s = float(scales[0]) if len(scales) == 1 else float(args.scales.split(',')[0])
        _nrm = normal if abs(_s - 1.0) < 1e-8 else F.interpolate(
            normal, scale_factor=_s, mode='bilinear', align_corners=False)
        _feats = []
        with torch.no_grad():
            for _i in range(0, _nrm.shape[0], 2):
                _r = model(_nrm[_i:_i + 2].to(args.device), is_training=True)
                _fp = torch.sum(torch.stack([_r['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                _feats.append(model.token_mapping(_fp).reshape(-1, _fp.shape[-1]))
        _allf = torch.cat(_feats, 0)
        _mu = _allf.mean(0, keepdim=True)
        # 方差收缩（Ledoit-Wolf 式）：把每维方差向"平均方差"收缩，避免极小方差维度被放大成噪声主导。
        _var = _allf.var(0, keepdim=True)
        _lam = float(getattr(args, 'whiten_shrink', 0.5))
        _var = (1.0 - _lam) * _var + _lam * _var.mean()
        _sd = _var.clamp_min(1e-8).sqrt()
        print(f'[whiten] {args.cls}: 用 {tuple(_nrm.shape)} 的 {_allf.shape[0]} 个 patch 估计白化统计')

    # [CLUSTER · 打分F] l_score_mode=10：在"学习到的映射空间"里对 K 张 shot 的正常 patch
    # 直接做 spherical k-means，得到若干簇中心（不是 patch 库），再用 soft-min 打分。
    # 这正是"正常 patch 会被聚成若干簇"的直接实现，且**仍然是单模型、单前向、单图**。
    if int(args.l_score_mode) == 10:
        _s = float(scales[0])
        _nrm = normal if abs(_s - 1.0) < 1e-8 else F.interpolate(
            normal, scale_factor=_s, mode='bilinear', align_corners=False)
        _feats = []
        with torch.no_grad():
            for _i in range(0, _nrm.shape[0], 2):
                _r = model(_nrm[_i:_i + 2].to(args.device), is_training=True)
                _fp = torch.sum(torch.stack([_r['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                _feats.append(F.normalize(model.token_mapping(_fp), dim=-1).reshape(-1, _fp.shape[-1]))
        _X = torch.cat(_feats, 0)                                  # [M, C]
        _nc = int(args.kmeans_clusters)
        _gen = torch.Generator(device='cpu').manual_seed(1234)
        _idx = torch.randperm(_X.shape[0], generator=_gen)[:_nc].to(_X.device)
        _C = _X[_idx].clone()                                      # 初始化：随机样本
        for _ in range(int(args.kmeans_iters)):
            _sim = _X @ _C.t()
            _asg = _sim.argmax(-1)
            for _k in range(_nc):
                _m = _asg == _k
                if _m.any():
                    _C[_k] = F.normalize(_X[_m].mean(0, keepdim=True), dim=-1)
        _centers = _C
        print(f'[kmeans] {args.cls}: {_X.shape[0]} 个正常 patch → {_nc} 个簇中心')

    # [CLUSTER · 打分G] l_score_mode=11：**每个原型槽位一个尺度 σ_k**（无需重训）。
    #   用 K 张 shot 的正常 patch 到各自原型的距离分布估计 σ_k（簇越紧 σ 越小 → 该簇在打分里更有话语权），
    #   打分 = d_k/σ_k 的 soft-min（可再与 min 做几何平均）。仍是单模型、单前向、单图。
    if int(args.l_score_mode) == 11:
        _s = float(scales[0])
        _nrm = normal if abs(_s - 1.0) < 1e-8 else F.interpolate(
            normal, scale_factor=_s, mode='bilinear', align_corners=False)
        _dists = []
        with torch.no_grad():
            for _i in range(0, _nrm.shape[0], 2):
                _r = model(_nrm[_i:_i + 2].to(args.device), is_training=True)
                _fp = torch.sum(torch.stack([_r['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                _fl = torch.sum(torch.stack(_r['feat_list'], 0), 0)
                _g = model.tokens_per_layer.token_dict['0'].expand(_fl.shape[0], -1, -1)
                _fl = torch.cat([_fl[:, :_g.shape[1]], _g], dim=1)          # 与 proto_set=unique 对齐
                _p = F.normalize(model.token_mapping(_fp), dim=-1)
                _q = F.normalize(model.token_mapping(_fl), dim=-1)
                _dd = (1.0 - torch.bmm(_p, _q.transpose(1, 2))).clamp_min(0)  # [B,N,K]
                _dists.append(_dd.reshape(-1, _dd.shape[-1]))
        _D = torch.cat(_dists, 0)                                          # [M,K]
        _pscale = _D.std(0, keepdim=True).clamp_min(1e-3)                  # [1,K]
        print(f'[pscale] {args.cls}: 用 {_D.shape[0]} 个正常 patch 估计 {_pscale.shape[-1]} 个原型的尺度 '
              f'(min={_pscale.min():.4f} max={_pscale.max():.4f})')

    def score_of(p):
        """raw 侧打分：逐块求 d1=min，同时收集 1024 个子采样列算 q50（与全量实现逐位等价）。"""
        if int(args.raw_global) < 0:                     # [无调参] 全库精确中位数：收集全部列
            d1, allcols = None, []
            for i, j, _sel in _plan:
                sim = torch.bmm(p, q_raw[i:j].t().unsqueeze(0).expand(p.shape[0], -1, -1))
                dch = (1.0 - sim).clamp_min(0)
                mch = dch.min(-1).values
                d1 = mch if d1 is None else torch.minimum(d1, mch)
                allcols.append(dch)
            dall = torch.cat(allcols, dim=-1)            # [B,N,M]
            qmed = torch.median(dall, dim=-1).values
            del allcols, dall
            return _pow(d1, qmed, args.raw_alpha)
        d1, cols = None, []
        top2 = None
        for i, j, sel in _plan:
            sim = torch.bmm(p, q_raw[i:j].t().unsqueeze(0).expand(p.shape[0], -1, -1))
            dch = (1.0 - sim).clamp_min(0)
            if int(args.raw_score_mode) >= 1:        # 需要 d1 与 d2（第二近）→ 逐块维护 top-2
                v2 = torch.topk(dch, k=2, dim=-1, largest=False).values
                top2 = v2 if top2 is None else torch.topk(
                    torch.cat([top2, v2], dim=-1), k=2, dim=-1, largest=False).values
            else:
                mch = dch.min(-1).values
                d1 = mch if d1 is None else torch.minimum(d1, mch)
            if sel.numel():
                cols.append(dch.index_select(-1, sel))
        d_sub = torch.cat(cols, dim=-1)                 # [B,N,1024]（顺序与 sub_idx 不同，排序后一致）
        ds_ = torch.sort(d_sub, dim=-1).values
        q = ds_[..., min(args.raw_global, ds_.shape[-1]) - 1]
        if int(args.raw_score_mode) >= 1:
            d1, d2 = top2[..., 0], top2[..., 1].clamp_min(1e-6)
            ratio = d1 / d2
            if int(args.raw_score_mode) == 1:
                return ratio
            return _pow(ratio, q, args.raw_alpha)       # mode 2: sqrt(ratio)·sqrt(q)
        return _pow(d1, q, args.raw_alpha)

    raw_store, gt_px_l, gt_sp_l = [], [], []
    seg_store = []
    seg_sum, seg_n = None, 0        # 分割头图（逐视图平均，未平滑）
    base_hw = (args.size, args.size)
    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(args.device)
            acc, n_view = None, 0
            for s in scales:
                v_img = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear',
                                                           align_corners=False)
                for vi, view in enumerate([v_img, v_img.flip(3)] if args.flip else [v_img]):
                    ret = model(view, is_training=True)
                    fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                    flr = torch.cat(ret['feat_list'], dim=1)   # [MULTI-LAYER] 与训练一致：按 token 维拼接各层
                    _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                    # [CLUSTER · 原型集] v3.7 原行为 = all：
                    #   feat_list 里 layer-0 的 32 个 token 因"双重追加"实际出现 **两遍**（64），
                    #   再拼 32 个全局锚点 = 96。soft-min（以及 min）会把重复的那 32 个算两次，
                    #   造成"某些簇权重翻倍"的偏差。这里提供去重/子集选项（默认 all 与 v3.7 逐位一致）。
                    _ps = str(getattr(args, 'proto_set', 'all'))
                    _k = _g.shape[1]                          # 每个复制单元的 token 数（32）
                    if _ps == 'unique':                       # 32 唯一 layer token + 32 锚点 = 64
                        flr = torch.cat([flr[:, :_k], _g], dim=1)
                    elif _ps == 'anchor':                     # 只用 32 个全局锚点
                        flr = _g
                    elif _ps == 'layer':                      # 只用 32 个 layer token（去重）
                        flr = flr[:, :_k]
                    else:                                     # all（v3.7 原行为，96）
                        flr = torch.cat([flr, _g], dim=1)
                    n_el = fp.shape[1]
                    gg = int(round(n_el ** 0.5))

                    if int(args.raw_feat) == 1:      # 只改 raw 打分路径；fp 原样供 learn 侧使用
                        _parts = [F.normalize(ret['patch_feat_layer'][j], dim=-1)
                                  for j in a.fused_layer]
                        fp_raw = torch.sum(torch.stack(_parts, 0), 0)
                    else:
                        fp_raw = fp
                    p = F.normalize(fp_raw, dim=-1)
                    m_raw = score_of(p).reshape(-1, 1, gg, gg)

                    pl = F.normalize(model.token_mapping(fp), dim=-1)
                    ql = F.normalize(model.token_mapping(flr), dim=-1)
                    if args.learn_bank_proto:
                        if q_bank_l is None:
                            raise RuntimeError('--learn_bank_proto 需要 --l_global > 0')
                        ql = torch.cat([ql, q_bank_l.unsqueeze(0).expand(ql.shape[0], -1, -1)],
                                       dim=1)
                    dlmin = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0).min(-1).values
                    if int(args.l_score_mode) == 3:
                        # [CLUSTER · 打分A] soft-min（簇的"混合似然"）：
                        #   -τ·log Σ_k exp(-d_k/τ)  —— 只要某个簇足够近就算近，
                        #   但不会像 min 那样被单个噪声原型决定，适合"多个簇"的情形。
                        _d = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                        _tau = float(args.l_softmin_tau)
                        _sm = -_tau * torch.logsumexp(-_d / _tau, dim=-1)
                        m_learn = _sm.reshape(-1, 1, gg, gg)
                    elif int(args.l_score_mode) == 7:
                        # [CLUSTER · 打分C] 白化空间的 soft-min：
                        #   x'=(x-μ)/σ, p'=(p-μ)/σ（μ,σ 来自 K 张 shot 的 patch），
                        #   d_k = 0.5·mean_c (x'-p'_k)²，score = -τ·log Σ_k exp(-d_k/τ)
                        _xw = (pl - _mu) / _sd
                        _pw = (ql - _mu) / _sd
                        # ||x-p||² 用内积恒等式算，避免物化 [B,N,K,C]（会吃掉数 GB 显存）
                        _xw2 = (_xw ** 2).sum(-1, keepdim=True)                  # [B,N,1]
                        _pw2 = (_pw ** 2).sum(-1).unsqueeze(1)                   # [B,1,K]
                        _dot = torch.bmm(_xw, _pw.transpose(1, 2))               # [B,N,K]
                        _d = 0.5 * (_xw2 + _pw2 - 2.0 * _dot) / float(_xw.shape[-1])
                        _tau = float(args.l_softmin_tau)
                        m_learn = (-_tau * torch.logsumexp(-_d / _tau, dim=-1)).reshape(-1, 1, gg, gg)
                    elif int(args.l_score_mode) == 4:
                        # [CLUSTER · 打分B] 到最近 k 个簇的**平均距离**（top-k 稳健最小）
                        _d = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                        _k = min(int(args.l_topk), _d.shape[-1])
                        _tk = torch.topk(_d, _k, dim=-1, largest=False).values.mean(-1)
                        m_learn = _tk.reshape(-1, 1, gg, gg)
                    elif int(args.l_score_mode) in (8, 9):
                        # [CLUSTER · 打分D/E] 单一模型、单一前向、单一图：
                        #   dlmin 与 soft-min 是同一组距离的两种聚合，互有短板（前者图像级好、后者像素级好）。
                        #   mode 8 = 几何平均 sqrt(dlmin·softmin)（与 raw 侧 d1^0.5·q^0.5 同构）
                        #   mode 9 = 算术平均 0.5·dlmin + 0.5·softmin
                        _d = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                        _dmin = _d.min(-1).values
                        _tau = float(args.l_softmin_tau)
                        _sm = -_tau * torch.logsumexp(-_d / _tau, dim=-1)
                        if int(args.l_score_mode) == 8:
                            _mix = _dmin.clamp_min(1e-8).pow(0.5) * _sm.clamp_min(1e-8).pow(0.5)
                        else:
                            _mix = 0.5 * _dmin + 0.5 * _sm
                        m_learn = _mix.reshape(-1, 1, gg, gg)
                    elif int(args.l_score_mode) == 10:
                        # [CLUSTER · 打分F] 到"学习空间 k-means 簇中心"的 soft-min
                        _d = (1.0 - torch.bmm(pl, _centers.t().unsqueeze(0)
                                              .expand(pl.shape[0], -1, -1))).clamp_min(0)
                        _tau = float(args.l_softmin_tau)
                        m_learn = (-_tau * torch.logsumexp(-_d / _tau, dim=-1)).reshape(-1, 1, gg, gg)
                    elif int(args.l_score_mode) == 11:
                        # [CLUSTER · 打分G] 每原型尺度归一化后的 soft-min ⊕ min（几何平均）
                        _d = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0) / _pscale
                        _tau = float(args.l_softmin_tau)
                        _sm = -_tau * torch.logsumexp(-_d / _tau, dim=-1)
                        _dmin = _d.min(-1).values
                        _mix = _dmin.clamp_min(1e-8).pow(0.5) * _sm.clamp_min(1e-8).pow(0.5)
                        m_learn = _mix.reshape(-1, 1, gg, gg)
                    elif int(args.l_score_mode) == 12:
                        # [CLUSTER · 打分H] 分配不确定性：p_k ∝ exp(-d_k/τ)，异常 patch 应对"归哪个簇"含糊。
                        #   score = 1 - max_k p_k（0=非常确定地属于某簇，1=完全含糊）
                        _d = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                        _tau = float(args.l_softmin_tau)
                        _p = torch.softmax(-_d / _tau, dim=-1)
                        m_learn = (1.0 - _p.max(-1).values).reshape(-1, 1, gg, gg)
                    elif int(args.l_score_mode) > 0:
                        # [探索] 全局项改用"到全部可学习原型距离的中位数 dlmed"
                        d_proto = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                        dlmed = d_proto.median(dim=-1).values
                        _a = 0.5 if int(args.l_score_mode) == 1 else 0.7
                        m_learn = _pow(dlmin, dlmed, _a).reshape(-1, 1, gg, gg)
                    elif _simple_learn:
                        m_learn = dlmin.reshape(-1, 1, gg, gg)
                    else:
                        db = (1.0 - torch.bmm(pl, q_bank_l.t().unsqueeze(0)
                                              .expand(pl.shape[0], -1, -1))).clamp_min(0)
                        dls = torch.sort(db, dim=-1).values
                        ql_ = dls[..., min(args.l_global, dls.shape[-1]) - 1]
                        m_learn = _pow(dlmin, ql_, args.l_alpha).reshape(-1, 1, gg, gg)
                    if seg_head is not None:
                        m_seg = seg_head(pl).reshape(-1, 1, gg, gg)      # [B,1,g,g] ∈ (0,1)
                        if args.segmap_out_dir:
                            _s = F.interpolate(m_seg, size=base_hw, mode='bilinear',
                                               align_corners=True)
                            if vi == 1:
                                _s = _s.flip(3)
                            seg_sum = _s if seg_sum is None else seg_sum + _s
                            seg_n += 1
                        if int(args.segmap_mode) == 1:
                            m_learn = m_seg                              # solo：分割头直接当该侧打分
                        else:
                            m_learn = m_learn + args.w_segmap * m_seg

                    am = F.interpolate(args.w * m_learn + (1.0 - args.w) * m_raw,
                                       size=base_hw, mode='bilinear', align_corners=True)
                    if vi == 1:
                        am = am.flip(3)
                    acc = am if acc is None else acc + am
                    n_view += 1
            gt_px = (gt.to(args.device) > 0.5).float()
            if gt_px.shape[1] > 1:
                gt_px = torch.max(gt_px, dim=1, keepdim=True)[0]
            raw_store.append(acc / max(1, n_view))
            if args.segmap_out_dir and seg_n:
                seg_store.append(seg_sum / seg_n)
                seg_sum, seg_n = None, 0
            gt_px_l.append(gt_px)
            gt_sp_l.append(label)

    raw_map = torch.cat(raw_store, 0)
    if args.segmap_out_dir and seg_store:
        os.makedirs(args.segmap_out_dir, exist_ok=True)
        _sg = torch.cat(seg_store, 0)
        torch.save(_sg.cpu(), os.path.join(args.segmap_out_dir, f'{args.cls}.map'))
        print(f'  [segmap] 已保存 {args.segmap_out_dir}/{args.cls}.map shape={tuple(_sg.shape)}')
    if args.save_maps_dir:
        os.makedirs(args.save_maps_dir, exist_ok=True)
        torch.save(raw_map.cpu(), os.path.join(args.save_maps_dir, f'{args.cls}.map'))
        print(f'  [maps] 保存 {args.save_maps_dir}/{args.cls}.map shape={tuple(raw_map.shape)}')
        write_sidecar(args.save_maps_dir, args.cls, ds, args.size,
                      {'scales': args.scales, 'flip': int(args.flip),
                       'l_global': int(args.l_global), 'w': float(args.w),
                       'raw_global': int(args.raw_global), 'raw_alpha': float(args.raw_alpha)})

    am = gk(raw_map)
    flat = am.flatten(1)
    kk = max(1, int(flat.shape[1] * 0.01))
    pr_sp = torch.sort(flat, dim=1, descending=True)[0][:, :kk].mean(dim=1)
    r = list(mod.ader_evaluator(am[:, 0].cpu().numpy(), pr_sp.flatten().cpu().numpy(),
                                torch.cat(gt_px_l, 0)[:, 0].cpu().numpy(),
                                torch.cat(gt_sp_l).flatten().cpu().numpy()))
    print(f"[v3] {args.cls}  raw(d1^{args.raw_alpha}*q{args.raw_global}) "
          f"learn(dlmin^{args.l_alpha}*bq{args.l_global}) w={args.w}")
    print("  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, r)))
    print("  CSV:" + ",".join(f"{v:.4f}" for v in r))


if __name__ == '__main__':
    main()
