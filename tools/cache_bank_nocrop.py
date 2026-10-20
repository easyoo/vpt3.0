# -*- coding: utf-8 -*-
"""[落盘] 在"测试不裁剪"口径下重建**原版 MemoryBank** 的逐图异常图。

与评估完全同一几何：`Resize(448) → ToTensor → Normalize`（**不做 CenterCrop**），
bank 的 4 张 shot 也用同一几何 + 原版 augmentation（22 张/图）。
输出：<out_dir>/<cls>.map  [N,1,448,448] float32（未平滑，便于后续融合）。

用法：python cache_bank_nocrop.py --class transistor --size 448 --out_dir .errmap_nocrop448
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys

import faiss
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash        # noqa: E402  [统一数据源]


def _default_script():
    """Support both audit-workspace and packaged vpt3.8 layouts."""
    for path in (
        os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'),
        os.path.join(HERE, '..', 'code', 'main-fewshot-vpt4.py'),
        os.path.join(HERE, 'main-fewshot-vpt4.py'),
    ):
        if os.path.exists(path):
            return os.path.normpath(path)
    return os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')


def load_mod(p):
    spec = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['m'] = m
    spec.loader.exec_module(m)
    return m


def write_sidecar(map_dir, cls, ds, size, extra=None):
    """[FIX-4] 落盘样本顺序侧车（与 eval_score_v3 同格式），供 fuse 对齐 GT/标签。"""
    os.makedirs(map_dir, exist_ok=True)
    keys = list(getattr(ds, 'keys', []) or [])
    if not keys:
        # vpt2 的 MVTecDataset 未维护 keys/list_hash；用与 mapping 侧完全相同的
        # "缺陷类型/文件名"格式补齐，便于缓存完成后做逐样本顺序审计。
        img_paths = getattr(ds, 'img_paths', None)
        if img_paths is None:
            img_paths = []
        img_paths = list(img_paths)
        keys = [f'{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}'
                for p in img_paths]
    list_hash = getattr(ds, 'list_hash', None)
    if list_hash is None and keys:
        list_hash = hashlib.sha1('\n'.join(keys).encode()).hexdigest()[:12]
    rec = {'cls': cls, 'n': len(ds), 'size': int(size),
           'list_hash': list_hash,
           'keys': keys,
           'config': extra or {}}
    p = os.path.join(map_dir, f'{cls}.json')
    with open(p, 'w') as fh:
        json.dump(rec, fh, ensure_ascii=False)
    print(f'[bank-nocrop{size}] 侧车已写 {p}（n={rec["n"]} hash={rec["list_hash"]}）')


def _faiss_flat(index, device):
    """Exact vectors stored by IndexFlatIP, as a GPU float32 tensor."""
    n, d = int(index.ntotal), int(index.d)
    arr = faiss.rev_swig_ptr(index.get_xb(), n * d).copy().reshape(n, d)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def query_batch_gpu(bank, imgs, size, bank_tensors, bank_chunk, extractor):
    """Equivalent to faiss IndexFlatIP.search(k=1), evaluated in GPU chunks."""
    bsz = int(imgs.shape[0])
    with torch.no_grad():
        feats = extractor(bank.model_t, imgs)
        feats = [F.normalize(rearrange(x, 'b l c->b c l'), dim=1)
                 for x in feats if bank.use_cos]
        n_tok = int(feats[0].shape[-1])
        grid = int(round(n_tok ** 0.5))
        if grid * grid != n_tok:
            raise RuntimeError((imgs.shape, n_tok))
        stages = []
        for feat, bank_flat in zip(feats, bank_tensors):
            query_flat = rearrange(feat, 'b c l->(b l) c').contiguous()
            best = torch.full((query_flat.shape[0],), -torch.inf,
                              device=query_flat.device, dtype=query_flat.dtype)
            for i in range(0, bank_flat.shape[0], int(bank_chunk)):
                sim = query_flat @ bank_flat[i:i + int(bank_chunk)].T
                best = torch.maximum(best, sim.max(dim=1).values)
            err = (1.0 - best).reshape(bsz, 1, grid, grid)
            stages.append(F.interpolate(
                err, size=size, mode='bilinear', align_corners=True))
    return ((stages[0] + stages[1]) * 0.5)[:, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--test_size', type=int, default=0,
                    help='[同一 bank 多尺度查询] 建库仍用 --size；测试图单独放大到该尺寸后查询，'
                         '结果再缩回 --size 保存。0=与 --size 相同')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_dir', default='.errmap_nocrop448')
    ap.add_argument('--batch', type=int, default=16,
                    help='测试 DataLoader batch（只影响显存；448 默认 16 与历史一致，672 建议 4）')
    ap.add_argument('--bank_chunk', type=int, default=0,
                    help='建库时前向分块大小（0=历史行为：88 张一次性前向，显存峰值 ~17GB@672；'
                         '>0 时分块喂入并增量 add 到 faiss，等价结果、显存大幅下降）')
    ap.add_argument('--script', default=_default_script())
    ap.add_argument('--backbone', type=str, default='base',
                    help="[主干实验] 'base'=原版 HF DINOv2-base（默认）；'dinov3b'/'dinov3l'=DINOv3 主干")
    ap.add_argument('--aug_copies', type=int, default=1,
                    help='[增强 bank] 每张 shot 调用几次增强管线（1=原版 22 张；2=44 张，库容量翻倍，其余不变）')
    ap.add_argument('--stage_out_dir', default='',
                    help='[仅增强 bank 重融合] 额外保存两个 stage 的异常图；'
                         '不设置时行为与原实现完全一致')
    ap.add_argument('--gpu_query', action='store_true',
                    help='用 GPU 分块点积执行与 faiss IndexFlatIP.search(k=1) 等价的查询；'
                         '仅加速，不改变打分公式')
    ap.add_argument('--query_bank_chunk', type=int, default=8192,
                    help='--gpu_query 时 bank 行分块大小，控制显存')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    from util.memorybank import MemoryBank, extract_fused_features
    # [主干实验] 只换 bank 分支的特征主干，MemoryBank 其余逻辑（增强/faiss/打分）一字不改
    bank_model = None
    if str(args.backbone).startswith('dinov3'):
        # 用绝对路径导入，避免 sys.path 顺序把 `util.*` 解析到别的代码副本
        import importlib.util as _ilu
        _p = os.path.join(HERE, 'code_vpt4', 'dinov3_hf_adapter.py')
        _spec = _ilu.spec_from_file_location('dinov3_hf_adapter_abs', _p)
        _m = _ilu.module_from_spec(_spec)
        sys.modules['dinov3_hf_adapter_abs'] = _m
        _spec.loader.exec_module(_m)
        build_adapter = _m.build_adapter
        bank_model = build_adapter(args.backbone)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    mod.args = a
    mod.device = args.device

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    test_size = int(args.test_size) if int(args.test_size) > 0 else int(args.size)
    test_tf = T.Compose([T.Resize((test_size, test_size)), T.ToTensor(),
                         T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((test_size, test_size)), T.ToTensor()])

    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    # [统一数据源] 与训练/其它脚本共用同一份 shot 清单（改 shots 数时两边仍然一致）
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    # [REPRO] augmentation() below uses the global torch RNG.  Fix it here so
    # rebuilding the 22x bank with the same seed is deterministic.
    mod.setup_seed(int(args.seed))
    print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
    imgs = [tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks]
    normal_imgs = torch.cat([mod.augmentation(im) for im in imgs for _ in range(max(1, int(args.aug_copies)))], 0)
    print(f'[bank-nocrop{args.size}] {args.cls}: {len(picks)} shot → 增强后 {tuple(normal_imgs.shape)}')
    if int(args.bank_chunk) > 0:
        _ch = int(args.bank_chunk)
        bank = MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_imgs[:_ch].to(args.device),
                          info=print, device=args.device, model=bank_model)
        for _i in range(_ch, normal_imgs.shape[0], _ch):
            bank.extract_features(normal_imgs[_i:_i + _ch].to(args.device))
        print(f'[bank-nocrop{args.size}] 分块建库完成 chunk={_ch}，共 {normal_imgs.shape[0]} 张')
    else:
        bank = MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_imgs.to(args.device),
                          info=print, device=args.device, model=bank_model)
    bank_tensors = None
    if args.gpu_query:
        if args.stage_out_dir:
            raise SystemExit('--gpu_query 暂不支持与 --stage_out_dir 同时使用')
        torch.backends.cuda.matmul.allow_tf32 = False
        bank_tensors = [
            _faiss_flat(bank.faiss_index_1, args.device),
            _faiss_flat(bank.faiss_index_2, args.device),
        ]
        print(f'[bank-nocrop{args.size}] GPU 查询已启用 '
              f'bank_shapes={[tuple(x.shape) for x in bank_tensors]} '
              f'chunk={int(args.query_bank_chunk)}')

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=test_tf, gt_transform=gtf, phase='test')
    # [FIX] code_vpt2 的 MVTecDataset 没有 list_hash 属性（vpt4 才有）→ 用 getattr 兜底，
    # 避免只是打印失败就把整个建库/推理流程打断。
    print(f'[bank-nocrop{args.size}] {args.cls}: {len(ds)} 个测试样本 '
          f'list_hash={getattr(ds, "list_hash", "n/a")}')
    dl = torch.utils.data.DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=4)
    out, out_s1, out_s2 = [], [], []
    for img, gt, label, _ in tqdm(dl, ncols=60, desc=f'bank{args.size}'):
        if args.gpu_query:
            am = query_batch_gpu(bank, img.to(args.device), test_size,
                                 bank_tensors, args.query_bank_chunk,
                                 extract_fused_features)
        elif args.stage_out_dir:
            s1, s2, am = bank.query_batch(img.to(args.device), return_stages=True)
            out_s1.append(torch.tensor(s1).unsqueeze(1).to(args.device))
            out_s2.append(torch.tensor(s2).unsqueeze(1).to(args.device))
        else:
            am = bank.query_batch(img.to(args.device))
        t = torch.tensor(am)
        # [FIX] bank.query_batch 内部对结果做了 .squeeze()：当**最后一个 batch 只有 1 张图**时，
        # 形状会从 (1,1,H,W) 被压成 (H,W)，导致后面 torch.cat 维度不一致而报错
        # （carpet/tile 等 N≡1 mod batch 的类必现）。这里按维度还原。
        if t.ndim == 2:            # (H,W)          ← batch=1 被 squeeze 两次
            t = t.unsqueeze(0)
        if t.ndim == 3:            # (N,H,W)
            t = t.unsqueeze(1)
        assert t.ndim == 4, (args.cls, t.shape)
        if test_size != args.size:
            t = torch.nn.functional.interpolate(
                t, size=(int(args.size), int(args.size)), mode='bilinear',
                align_corners=True)
        out.append(t.to(args.device))
    m = torch.cat(out, 0).cpu()
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f'{args.cls}.map')
    torch.save(m, p)
    print(f'[bank-nocrop{args.size}] 已保存 {p} shape={tuple(m.shape)}')
    write_sidecar(args.out_dir, args.cls, ds, args.size,
                  {'shots': int(args.shots), 'seed': int(args.seed),
                   'query_size': test_size})
    if args.stage_out_dir:
        os.makedirs(args.stage_out_dir, exist_ok=True)
        for name, chunks in [('s1', out_s1), ('s2', out_s2)]:
            q = os.path.join(args.stage_out_dir, f'{args.cls}_{name}.map')
            torch.save(torch.cat(chunks, 0).cpu(), q)
            print(f'[bank-stage] 已保存 {q} shape={tuple(torch.cat(chunks, 0).shape)}')


if __name__ == '__main__':
    main()
