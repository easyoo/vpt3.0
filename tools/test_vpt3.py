# -*- coding: utf-8 -*-
"""[test_vpt3.0 端到端验证] 用 control35 的 carpet@35 权重跑一次「改善后 mapping ⊕ 原版 MemoryBank」，
并与参考实现（fuse_eval.py --rule zwsum:0.3）对比，确认版本化代码行为一致。"""
import argparse, importlib.util, os, random, sys
import numpy as np, torch, torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec); sys.modules['audit_main'] = m
    spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', default='carpet')
    ap.add_argument('--w', type=float, default=0.3)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, 'code_test_vpt3'))
    mod = load_mod(os.path.join(HERE, 'code_test_vpt3', 'main-fewshot-test_vpt3.py'))
    from util.memorybank import MemoryBank

    class A: pass
    a = A(); a.learnable_layers = 1; a.learnable_num_per_layer = 32
    a.temperature = 0.07; a.scaled_factor = 3; a.fusion = 'sum'
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]; a.fused_layer_learnable = [0]
    mod.args = a; mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=False)['dino'], strict=False)
    model.to(args.device).eval()

    tf, gtf = mod.get_data_transforms(448, 392)
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    files = sorted(os.listdir(root)); random.seed(1); picks = random.sample(files, 4)
    shots = [tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks]
    raw_bank = mod.build_raw_bank(model, torch.stack(shots, 0), a.fused_layer, k=0)
    print(f'[test_vpt3] 原始特征库 {tuple(raw_bank.shape)}')
    bank_imgs = torch.cat([mod.augmentation(s) for s in shots], 0).to(args.device)
    mb = MemoryBank(target_cls=args.cls, use_json=False, imgs=bank_imgs, info=print, device=args.device)

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    err = []
    for img, gt, label, _ in dl:
        err.append(torch.tensor(mb.query_batch(img.to(args.device))).unsqueeze(1).to(args.device))
    errmap = torch.cat(err, 0)
    r = mod.evaluation_batch_test_vpt3(model, dl, args.device, errmap, raw_bank, a.fused_layer,
                                  w=args.w, scales=(1.5,), tta_flip=True)
    names = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    print(f'\n[test_vpt3.0 / {args.cls}] w={args.w}: ' +
          '  '.join(f'{n}={v:.4f}' for n, v in zip(names, r)))


if __name__ == '__main__':
    main()
