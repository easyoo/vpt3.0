# -*- coding: utf-8 -*-
"""[审计工具] 离线评估已保存的 checkpoint（只跑 mapping 分支 type=1）。

目的：训练一次、评估多次。同一份权重可以在不同"评估口径"下打分，
      从而把"训练变好"与"评估口径变了"这两件事严格分开。

用法：
  python eval_offline.py --ckpt results/control/maponly/model_screw_30.pth \
      --class screw --protocol 392
"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash        # noqa: E402  [统一数据源]


def load_audit_module(script_path):
    """把 main-fewshot-audit.py 当模块导入（文件名带 '-'，只能用 spec 方式）。"""
    spec = importlib.util.spec_from_file_location('audit_main', script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--protocol', type=int, default=392,
                    help='392 = 项目历史口径(resize_mask=None)；256 = 统一降采样口径')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    # 统一用 code_norm 的脚本来评估任意 checkpoint：fusion='sum' 时它与 control 逐位一致（已用冒烟测试验证），
    # fusion='norm' 时它按 FIX-12 融合。这样"评估代码路径"对两组完全一致。
    ap.add_argument('--script', default=os.path.join(HERE, 'code_norm', 'main-fewshot-norm.py'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tag', default='offline')
    ap.add_argument('--cpu_metrics', action='store_true',
                    help='用纯 numpy 指标替代 CUDA 的 adeval（无 GPU 时也能验证代码路径；P-AUPRO 不可得）')
    ap.add_argument('--metric_head', type=int, default=1, help='M6: 是否使用可学习度量头（需与训练一致）')
    ap.add_argument('--global_proto', type=int, default=0, help='M4: 原型集是否拼接全局原型（需与训练一致）')
    ap.add_argument('--num_tokens', type=int, default=32, help='可学习原型数量（需与训练一致，vpt4-k64 用 64）')
    ap.add_argument('--res_mapping', type=int, default=1, help='映射是否残差（需与训练一致）')
    ap.add_argument('--type', type=int, default=1,
                    help='打分类型：1=原型距离 2=分割头误差图 4=两者相加（均属于 mapping 分支）')
    ap.add_argument('--fusion', type=str, default='sum', choices=['sum', 'norm', 'gate'],
                    help="多层特征融合：'sum'=原行为（control 权重用这个）；'norm'=FIX-12（code_norm 权重用这个）")
    ap.add_argument('--setup_seed', type=int, default=1,
                    help='复现训练期评估的 cudnn 选核（训练脚本在 main 开头调用了 setup_seed(1)）')
    ap.add_argument('--ms_scoring', action='store_true',
                    help='[路径2] 多尺度+翻转打分（需 --script 指向 code_norm 的脚本）')
    ap.add_argument('--ms_scales', type=str, default='1.0,1.25,1.5')
    ap.add_argument('--ms_flip', type=int, default=1)
    ap.add_argument('--fuse_bank', type=int, default=0,
                    help='>0: 原始空间正常特征库原型数(k-center)；-1: 全部 patch；0: 关闭')
    ap.add_argument('--ms_bank_proto', type=int, default=0,
                    help='M8: 1=把原始特征库过一遍 token_mapping 后并入可学习原型图的原型集（学到的度量里的 PatchCore）')
    ap.add_argument('--fuse_z', type=int, default=0)
    ap.add_argument('--fuse_w', type=float, default=0.5,
                    help='w*可学习原型图 + (1-w)*原始特征库图；实测 0.3 在 35 轮上最稳')
    ap.add_argument('--shots', type=int, default=4, help='建库用的正常图数量（与训练 shot 一致）')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))  # 用 --script 所在目录的 dinov2 副本
    mod = load_audit_module(args.script)

    # evaluation_batch 依赖模块级全局：args / device / segmentation_head
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
    a.ms_bank_proto = args.ms_bank_proto
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = args.fusion
    mod.args = a
    if args.cpu_metrics:
        # 用纯 numpy 实现替换 CUDA 累加器，使整条评估代码路径可在 CPU 上验证
        sys.path.insert(0, HERE)
        from probe_identity import cpu_metrics
        mod.ader_evaluator = lambda pr_px, pr_sp, gt_px, gt_sp, **kw: cpu_metrics(
            pr_px, pr_sp, gt_px, gt_sp)
    mod.device = args.device
    mod.setup_seed(args.setup_seed)

    device = args.device
    data_transform, gt_transform = mod.get_data_transforms(448, 392)

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    head = nn.Sequential(
        nn.Linear(768, 768 // 2), nn.ReLU(inplace=True),
        nn.Linear(768 // 2, 1), nn.Sigmoid()).to(device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    # [防呆] 权重里的模块与命令行开关必须一致，否则会"静默用错架构"算出错的指标
    # （strict=False 会忽略不匹配的键，不报错）。历史教训：m4m6 的权重带 token_mapping.metric，
    # 若评估时忘记 --metric_head 1，就会用单位阵打分，看起来"正常"但其实是错的。
    _has_metric = any('token_mapping.metric' in k for k in ck['dino'].keys())
    if _has_metric and not args.metric_head:
        print('  [WARN] 权重含 token_mapping.metric，但 --metric_head 0 → 训练好的度量头会被丢弃！')
    if (not _has_metric) and args.metric_head:
        print('  [note] 权重不含 metric 键（旧版本），--metric_head 1 时等价于单位阵，无影响')
    _has_gp = any('tokens_per_layer' in k for k in ck['dino'].keys())
    if not _has_gp:
        print('  [WARN] 权重里找不到可学习原型，请检查 ckpt 是否正确')
    model.load_state_dict(ck['dino'], strict=False)
    head.load_state_dict(ck['seg_head'])
    model.to(device).eval()
    head.eval()
    mod.segmentation_head = head

    test_data = mod.MVTecDataset(
        root=os.path.join(args.data_path, args.cls),
        transform=data_transform, gt_transform=gt_transform, phase='test')
    dl = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    rm = None if args.protocol == 392 else args.protocol
    if args.fuse_bank != 0:
        import random as _random
        from PIL import Image as _Image
        root = os.path.join(args.data_path, args.cls, 'train', 'good')
        # [统一数据源] 与训练/bank 分支共用同一份 shot 清单
        picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
        print(f'  [shots] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
        normal = torch.stack([data_transform(_Image.open(os.path.join(root, f)).convert('RGB'))
                              for f in picks], dim=0)
        bank = mod.build_raw_bank(model, normal, [1, 2, 3, 4, 6, 8, 11],
                                  k=(0 if args.fuse_bank < 0 else args.fuse_bank))
        print(f'  [FEAT-FUSE] 原始空间原型库: {tuple(bank.shape)}')
        r = mod.evaluation_batch_fuse(model, dl, device, bank, max_ratio=0.01,
                                      resize_mask=rm, use_z=bool(args.fuse_z), w_learn=args.fuse_w,
                                      scales=(tuple(float(x) for x in args.ms_scales.split(','))
                                              if args.ms_scoring else (1.0,)),
                                      tta_flip=bool(args.ms_flip) and args.ms_scoring)
    elif args.ms_scoring:
        _sc = tuple(float(x) for x in args.ms_scales.split(','))
        r = mod.evaluation_batch_ms(model, dl, device, max_ratio=0.01,
                                    resize_mask=rm, scales=_sc, tta_flip=bool(args.ms_flip))
    else:
        r = mod.evaluation_batch(model, dl, device, max_ratio=0.01,
                                 resize_mask=rm, visual_bank_map=None, type=args.type)
    names = ["I-Auroc", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    print(f"[{args.tag}] {args.cls} type={args.type} protocol={args.protocol} "
          f"ckpt={os.path.basename(args.ckpt)}")
    print("  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, r)))
    print("  CSV:" + f"{args.cls},{os.path.basename(args.ckpt)},{args.protocol},type{args.type}," +
          ",".join(f"{v:.4f}" for v in r))


if __name__ == '__main__':
    main()
