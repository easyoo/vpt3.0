# -*- coding: utf-8 -*-
"""[CPU 单元测试] vpt4 的像素打分口径与 M7 跨图排序损失。

为什么必须测（用户要求"代码不要出现 bug，符合预设方案"）：
  * M5/M7 的分数必须与**线上推理**（compute_fused_map 里的 `1 - max cos`）逐位一致，
    否则训练在优化一个线上不存在的打分函数；
  * M7 的 hinge / 困难挖掘语义必须可验证（顺序对了 → 0；顺序反了 → margin + 分差）。

运行：/home/jjquan/miniconda3/envs/vpt/bin/python test_vpt4_m7.py
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')

OK = 0
FAIL = 0


def check(name, cond, extra=''):
    global OK, FAIL
    if cond:
        OK += 1
        print(f'  PASS {name} {extra}')
    else:
        FAIL += 1
        print(f'  FAIL {name} {extra}')


def load_module():
    sys.path.insert(0, os.path.dirname(SCRIPT))      # 让脚本内的 `from util.dataset import *` 可解析
    spec = importlib.util.spec_from_file_location('vpt4_main', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['vpt4_main'] = mod
    spec.loader.exec_module(mod)                     # 只执行模块级定义，不进入 __main__ 训练流程
    return mod


def main():
    torch.manual_seed(0)
    mod = load_module()
    print('[1] 模块导入（不执行训练主流程）')
    check('已加载两个新函数',
          hasattr(mod, 'cross_image_rank_loss') and hasattr(mod, 'proto_distance_scores'))

    s_fn = mod.proto_distance_scores
    r_fn = mod.cross_image_rank_loss

    print('[2] proto_distance_scores 与"训练/推理打分口径"逐位一致')
    B, N, K, D = 3, 7, 5, 16
    patch = torch.randn(B, N, D)
    proto = torch.randn(B, K, D)
    got = s_fn(patch, proto)
    # 参考实现 = 评估端 compute_fused_map._map28 的写法（clamp_min(0) 后取 min）
    p = torch.nn.functional.normalize(patch, dim=-1)
    q = torch.nn.functional.normalize(proto, dim=-1)
    ref = (1.0 - torch.bmm(p, q.transpose(1, 2))).clamp_min(0).min(dim=-1).values
    check('形状为 [B,N]', tuple(got.shape) == (B, N), str(tuple(got.shape)))
    check('与评估端公式逐位一致', torch.allclose(got, ref, atol=1e-6),
          f'max_abs_delta={float((got - ref).abs().max()):.2e}')
    same = torch.randn(1, 3, D)
    d0 = s_fn(same, same.clone())
    check('patch==proto 时距离为 0', bool((d0.abs() < 1e-5).all()),
          f'max={float(d0.abs().max()):.2e}')
    perm = torch.randperm(K)
    check('对原型排列不变', torch.allclose(got, s_fn(patch, proto[:, perm]), atol=1e-6))

    print('[3] M7 语义：顺序正确 -> 0；顺序反了 -> margin + 分差')
    nrm = torch.full((40,), 0.20)
    anm = torch.full((40,), 0.90)
    check('完全可分时损失为 0', float(r_fn(nrm, anm, margin=0.1, n_sample=64, normalize=False)) == 0.0)
    rev = float(r_fn(anm, nrm, margin=0.1, n_sample=64, normalize=False))
    check('完全反向时损失 = 分差 + margin', abs(rev - 0.8) < 1e-6, f'got={rev:.4f}')

    print('[4] M7 数学正确性：与手工 all-pairs hinge 一致（含困难挖掘、损失必须非零才算有效测试）')
    # 关键：让两类分布**重叠**，否则 hinge 恒为 0，测试会"假通过"
    nrm = 0.55 + 0.12 * torch.randn(500)         # 正常像素分数
    anm = 0.45 + 0.12 * torch.randn(500)         # 异常像素分数（真实场景下两者常有重叠区）
    for n_s in (500, 64, 16):
        got = float(r_fn(nrm, anm, margin=0.10, n_sample=n_s, normalize=False))
        ns = torch.topk(nrm, min(n_s, nrm.numel())).values
        as_ = torch.topk(anm, min(n_s, anm.numel()), largest=False).values
        ref = float(torch.relu(ns.unsqueeze(1) - as_.unsqueeze(0) + 0.10).mean())
        check(f'原始模式 n_sample={n_s} 与手工计算一致且非零', abs(got - ref) < 1e-6 and got > 0,
              f'got={got:.6f} ref={ref:.6f}')

    print('[4b] M7 归一化模式（默认）：与手工 tau=IQR 公式一致 + 尺度无关')
    nrm2 = 0.012 + 0.004 * torch.randn(2000)
    anm2 = 0.014 + 0.004 * torch.randn(200)
    got = float(r_fn(nrm2, anm2, margin=0.5, n_sample=256, normalize=True))
    tau = float(torch.clamp(torch.quantile(nrm2, 0.75) - torch.quantile(nrm2, 0.25), min=1e-6))
    ns = torch.topk(nrm2, 256).values
    as_ = torch.topk(anm2, 200, largest=False).values
    ref = float(torch.relu(((ns.unsqueeze(1) - as_.unsqueeze(0)) / tau).clamp(-20, 20) + 0.5).mean())
    check('与手工 tau=IQR 公式一致', abs(got - ref) < 1e-6, f'got={got:.6f} ref={ref:.6f}')
    for k in (0.01, 100.0):
        got_k = float(r_fn(nrm2 * k, anm2 * k, margin=0.5, n_sample=256, normalize=True))
        check(f'尺度不变（分数整体 ×{k:g}）', abs(got_k - got) < 1e-4, f'{got_k:.6f} vs {got:.6f}')

    print('[5] M7 边界：空集合 / 梯度可回传、有限、非零且方向正确')
    check('异常集为空 -> 0 且不报错', float(r_fn(torch.rand(10), torch.zeros(0), margin=0.1)) == 0.0)
    check('正常集为空 -> 0 且不报错', float(r_fn(torch.zeros(0), torch.rand(10), margin=0.1)) == 0.0)
    a = (0.55 + 0.12 * torch.randn(64)).requires_grad_(True)     # 正常像素分数
    b = (0.45 + 0.12 * torch.randn(64)).requires_grad_(True)     # 异常像素分数
    loss = r_fn(a, b, margin=0.1, n_sample=64)
    check('重叠分布下损失非零', float(loss) > 0, f'loss={float(loss):.4f}')
    loss.backward()
    check('梯度存在且有限', a.grad is not None and b.grad is not None
          and bool(torch.isfinite(a.grad).all()) and bool(torch.isfinite(b.grad).all()))
    check('存在非零梯度（确认真的有梯度信号，而非恒 0）',
          float(a.grad.abs().sum()) > 0 and float(b.grad.abs().sum()) > 0,
          f'|g_norm|={float(a.grad.abs().sum()):.3f} |g_anom|={float(b.grad.abs().sum()):.3f}')
    check('异常分数被往下压（dL/ds_anom <= 0）', bool((b.grad <= 0).all()),
          f'max={float(b.grad.max()):.2e}')
    check('正常分数被往上抬（dL/ds_norm >= 0）', bool((a.grad >= 0).all()),
          f'min={float(a.grad.min()):.2e}')

    print('[6] Batch 形状鲁棒性（模拟 [B,N] 掩码展平 + 拼接）')
    nrm2 = torch.rand(4, 30)
    anm2 = torch.rand(4, 30)
    m = torch.zeros(4, 30, dtype=torch.bool)
    m[:, :10] = True
    flat_n = nrm2.reshape(-1)
    flat_a = anm2.reshape(-1)
    mf = m.reshape(-1)
    got = float(r_fn(torch.cat([flat_n, flat_a[~mf]], 0), flat_a[mf], margin=0.1, n_sample=256))
    check('展平拼接不报错且有限', got == got and got >= 0, f'got={got:.4f}')

    print('[7] 集成形状测试：用训练循环真实的张量形状跑 M7 封装函数')
    f_fn = mod.cross_image_rank_loss_from_features
    Bn, Ba, N, D = 8, 16, 1024, 768          # 正常 batch=8 / 异常 batch=16 / patch=1024 / dim=768
    K = 96                                    # 原型数（sum(feat_list)=64 ⊕ 全局锚点 32，见 FIX-M4）
    n_patch = torch.randn(Bn, N, D)
    n_proto = torch.randn(Bn, K, D)
    a_patch = torch.randn(Ba, N, D)
    a_proto = torch.randn(Ba, K, D)
    mask = torch.zeros(Ba, N)
    mask[:, :100] = 1.0                       # 每张辅助图 100 个"异常"patch
    n_patch.requires_grad_(True)
    a_patch.requires_grad_(True)
    l = f_fn(n_patch, n_proto, a_patch, a_proto, mask, margin=0.1, n_sample=256)
    l.backward()
    check('真实形状下可前向且损失非负有限', bool(torch.isfinite(l)) and float(l) >= 0, f'loss={float(l):.4f}')
    check('梯度可回传到 patch 特征（两个分支都有信号）',
          n_patch.grad is not None and a_patch.grad is not None
          and bool(torch.isfinite(n_patch.grad).all()) and bool(torch.isfinite(a_patch.grad).all())
          and float(n_patch.grad.abs().sum()) > 0 and float(a_patch.grad.abs().sum()) > 0)
    # 掩码长度不匹配时必须显式报错（而不是静默算错）
    bad = False
    try:
        f_fn(n_patch.detach(), n_proto, a_patch.detach(), a_proto, torch.zeros(Ba, 512), margin=0.1)
    except RuntimeError:
        bad = True
    check('掩码长度不匹配时显式报错', bad)

    print('[8] M5 重构等价性：新写法(1 - proto_distance_scores) 必须与旧的 bmm 写法逐位相同')
    B2, N2, K2, D2 = 16, 1024, 96, 768
    ap = torch.randn(B2, N2, D2)
    aq = torch.randn(B2, K2, D2)
    # 旧写法（重构前训练循环里的原始代码）
    _Pm = torch.nn.functional.normalize(ap, dim=-1)
    _Qm = torch.nn.functional.normalize(aq, dim=-1)
    old_sim = torch.bmm(_Pm, _Qm.transpose(1, 2)).max(dim=-1).values
    old_prob = ((1.0 - old_sim) / 2.0).clamp(1e-6, 1.0 - 1e-6)
    # 新写法（现在训练循环实际调用）
    new_score = s_fn(ap, aq)
    new_sim = 1.0 - new_score
    new_prob = ((1.0 - new_sim) / 2.0).clamp(1e-6, 1.0 - 1e-6)
    check('M5 概率逐位一致', bool(torch.allclose(old_prob, new_prob, atol=1e-6)),
          f'max_abs_delta={float((old_prob-new_prob).abs().max()):.2e}')
    tgt = (torch.rand(B2, N2) > 0.7).float()
    old_bce = (-(tgt * torch.log(old_prob) + (1 - tgt) * torch.log(1 - old_prob))).mean()
    new_bce = (-(tgt * torch.log(new_prob) + (1 - tgt) * torch.log(1 - new_prob))).mean()
    check('M5 的 BCE 逐位一致', bool(torch.allclose(old_bce, new_bce, atol=1e-6)),
          f'old={float(old_bce):.6f} new={float(new_bce):.6f}')

    print('[9] M10 图像级跨图排序：语义 + 尺度不变性 + 与评估口径的 top-1% 一致')
    ir = mod.image_level_rank_loss
    trm = mod.top_ratio_mean
    # top-1% 均值必须等于评估里"排序后取前 1%"的写法
    s = torch.rand(4, 1000)
    k = max(1, int(1000 * 0.01))
    ref = torch.sort(s, dim=-1, descending=True)[0][:, :k].mean(dim=-1)
    check('top_ratio_mean 与评估口径一致', bool(torch.allclose(trm(s), ref)), f'k={k}')
    check('M10 顺序正确时为 0', float(ir(torch.full((8,), 0.1), torch.full((16,), 0.9))) == 0.0)
    rev = float(ir(torch.full((8,), 0.9), torch.full((16,), 0.1)))
    check('M10 顺序反了 > 0', rev > 0, f'got={rev:.4f}')
    torch.manual_seed(0)
    n2 = 0.5 + 0.1 * torch.randn(8)
    a2 = 0.45 + 0.1 * torch.randn(16)
    v1 = float(ir(n2, a2, margin=0.5))
    v2 = float(ir(n2 * 1000, a2 * 1000, margin=0.5))
    check('M10 尺度不变', abs(v1 - v2) < 1e-4, f'{v1:.6f} vs {v2:.6f}')
    n2g = n2.clone().requires_grad_(True)
    a2g = a2.clone().requires_grad_(True)
    ir(n2g, a2g, margin=0.5).backward()
    check('M10 梯度有限且方向正确（正常↑ 异常↓）',
          bool(torch.isfinite(n2g.grad).all()) and bool(torch.isfinite(a2g.grad).all())
          and float(n2g.grad.min()) >= 0 and float(a2g.grad.max()) <= 0)

    print(f'\n结果：{OK} 项通过，{FAIL} 项失败')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
