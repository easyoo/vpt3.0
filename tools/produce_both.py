# -*- coding: utf-8 -*-
"""[统一数据流] **一个 loader 同时产出两个分支**的异常图，从构造上消除顺序/样本不一致。

一次遍历测试集，对同一批张量同时计算：
  * mapping 分支：m4m6r3_15 权重 + 原始特征库（`compute_fused_map`，含 1.5×+翻转 两视图）
  * bank  分支：原版 MemoryBank（4shot×22 增强建库，单视图检索）
两者写入**同一个目录**，并保存测试图路径/标签，供融合脚本逐图校验。

shot 清单来自 `shot_utils.get_shots()`（与训练、评估共用同一份落盘清单）。

用法：
  python produce_both.py --class transistor --size 448 --out_dir .both_nocrop448 \
      --map_save results/vpt4/m4m6r3_15 --shots_dir .shots
"""
import argparse
import importlib.util
import json
import os
import sys

import torch
from torchvision import transforms as T
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash      # noqa: E402


def load_mod(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_dir', default='.both_nocrop448')
    ap.add_argument('--map_save', default='results/vpt4/m4m6r3_15', help='mapping 权重目录')
    ap.add_argument('--shots_dir', default=os.path.join(HERE, '.shots'))
    ap.add_argument('--ours_script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    ap.add_argument('--bank_script', default=os.path.join(HERE, 'code_vpt2', 'main-fewshot-vpt2.py'))
    ap.add_argument('--scales', default='1.5')
    ap.add_argument('--flip', type=int, default=1)
    ap.add_argument('--w_learn', type=float, default=0.5)
    args = ap.parse_args()

    # [关键] 两份代码副本里都有 `dinov2` 包，**同名会互相覆盖**（先插入 sys.path 的目录优先）。
    # 必须：先把 code_vpt4 放在 sys.path 最前 → 建 mapping 模型（此时 dinov2 只能来自 code_vpt4）；
    # 之后再引入 code_vpt2 仅用于 `augmentation` / 原版 `MemoryBank`（它们不使用本地 dinov2）。
    # 并加断言：一旦两份副本交叉，立刻报错，而不是静默算出错误结果。
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.ours_script)))
    ours = load_mod(args.ours_script, 'ours_mod')
    _ours_vt = ours.vision_transformer.__file__
    assert 'code_vpt4' in _ours_vt, f'mapping 侧 dinov2 来源错误：{_ours_vt}（应来自 code_vpt4）'
    print(f'[guard] mapping dinov2 = {_ours_vt}')

    # ---------- 统一 shot 清单（训练/两分支共用）----------
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed,
                      shot_dir=args.shots_dir, verbose=True)
    print(f'[unified] shot hash = {shots_hash(picks)}')

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    # ---------- 模型（mapping）----------
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
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    a.ms_bank_proto = 0
    ours.args = a
    ours.device = args.device
    model = ours.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(os.path.join(args.map_save, f'model_{args.cls}_35.pth'),
                    map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(args.device).eval()

    # ---------- 用同一份 shot 图建"两套参照" ----------
    from PIL import Image
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    shot_imgs = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    raw_bank = ours.build_raw_bank(model, shot_imgs, a.fused_layer, k=0)          # mapping 侧的原始特征库

    # 到这里 mapping 侧所需的 dinov2 已经实例化完毕，再引入 code_vpt2（只取 augmentation / MemoryBank）
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.bank_script)))
    bankmod = load_mod(args.bank_script, 'bank_mod')
    _bank_dir = os.path.dirname(os.path.abspath(bankmod.__file__))
    assert os.path.basename(_bank_dir) == 'code_vpt2', f'bank 侧脚本来源错误：{_bank_dir}'
    from util.memorybank import MemoryBank
    import util.memorybank as _mbmod
    # `util` 包会被先导入的那份副本占用（这里通常是 code_vpt4）；只要**文件内容与 code_vpt2 完全一致**就安全。
    # 若将来两份分叉（内容不同）→ 立刻报错，避免默默用错实现。
    import hashlib
    _p_used = _mbmod.__file__
    _p_ref = os.path.join(os.path.dirname(os.path.abspath(args.bank_script)), 'util', 'memorybank.py')
    _h = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
    assert _h(_p_used) == _h(_p_ref), (
        f'MemoryBank 实现与 code_vpt2 不一致：used={_p_used} ref={_p_ref}（两份副本已分叉，禁止混用）')
    print(f'[guard] bank 侧 = {_bank_dir} | MemoryBank = {_mbmod.__file__}')
    bankmod.args = a
    bankmod.device = args.device
    normal_aug = torch.cat([bankmod.augmentation(im) for im in shot_imgs], 0).to(args.device)
    print(f'[unified] bank 建库：{len(picks)} shot → {tuple(normal_aug.shape)}')
    mem_bank = MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_aug,
                          info=print, device=args.device)

    # ---------- 一次遍历，两个分支同时算 ----------
    ds = ours.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                           transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    gk = ours.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    scales = tuple(float(x) for x in args.scales.split(','))
    mp, bp, gts, labs = [], [], [], []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(dl, ncols=60, desc='both'):
            img = img.to(args.device)
            am = ours.compute_fused_map(model, img, raw_bank, a.fused_layer, scales=scales,
                                        tta_flip=bool(args.flip), w_learn=args.w_learn)
            mp.append(am.float().cpu())                     # 未平滑，落盘供融合
            bm = mem_bank.query_batch(img)                  # 同一批张量 → 逐图严格对齐
            bp.append(torch.tensor(bm).unsqueeze(1).cpu())
            gts.append(gt); labs.append(label)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(torch.cat(mp, 0), os.path.join(args.out_dir, f'{args.cls}_mapping.map'))
    torch.save(torch.cat(bp, 0).float(), os.path.join(args.out_dir, f'{args.cls}_bank.map'))
    torch.save(torch.cat(gts, 0).float(), os.path.join(args.out_dir, f'{args.cls}_gt.map'))
    json.dump({'cls': args.cls, 'size': args.size, 'shots': picks,
               'shots_hash': shots_hash(picks), 'paths': [str(p) for p in ds.img_paths],
               'labels': [int(x) for x in torch.cat(labs).flatten().tolist()],
               'map_save': args.map_save},
              open(os.path.join(args.out_dir, f'{args.cls}_meta.json'), 'w'), indent=1, ensure_ascii=False)
    print(f'[unified] 已保存 {args.out_dir}/{args.cls}_{{mapping,bank,gt}}.map  '
          f'mapping={tuple(torch.cat(mp,0).shape)} bank={tuple(torch.cat(bp,0).shape)}')


if __name__ == '__main__':
    main()
