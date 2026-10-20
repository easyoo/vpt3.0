# -*- coding: utf-8 -*-
# [vpt5.0 · M11] 尾部排序损失的语义 / 梯度 / 形状自检（CPU，秒级）。
# 用法: python test_vpt5_m11.py
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'code_vpt5', 'main-fewshot-vpt4.py')


def load_mod(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location('m5', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['m5'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ok = 0
    m = load_mod(SCRIPT)

    # 1) 完美排序（异常分数整体高出 margin 个 τ）→ 损失应为 0
    torch.manual_seed(0)
    ns = torch.rand(2000) * 0.01                      # 正常：小分
    as_ = ns.max() + 10 * 0.001                       # 异常：全在大分侧
    l0 = m.tail_rank_loss(ns, as_)
    assert l0.item() == 0.0, l0
    print('1) 完美可分 → loss = %.4f ✔' % l0.item()); ok += 1

    # 2) 有正常像素压过异常像素 → 损失 > 0，且随 margin 增大
    as_bad = ns.max() - 0.001
    l1 = m.tail_rank_loss(ns, as_bad, margin=0.5)
    l2 = m.tail_rank_loss(ns, as_bad, margin=1.0)
    assert l1.item() > 0 and l2.item() > l1.item(), (l1, l2)
    print('2) 尾部被压 → loss=%.3f，margin 1.0 → %.3f（单调）✔' % (l1.item(), l2.item())); ok += 1

    # 3) 梯度能回传到两侧分数
    ns_g = ns.clone().requires_grad_(True); as_g = as_bad.clone().requires_grad_(True)
    m.tail_rank_loss(ns_g, as_g).backward()
    assert ns_g.grad is not None and as_g.grad is not None
    assert torch.isfinite(ns_g.grad).all() and torch.isfinite(as_g.grad).all()
    print('3) 梯度有限且两侧都有 ✔  |grad_ns|_1=%.4f  sum(grad_as)=%.4f'
          % (ns_g.grad.abs().sum(), as_g.grad.sum())); ok += 1

    # 4) 只挖 top-k：把 k 个正常像素之外的分数改小，损失不变（normalize=False 时纯看排序）
    k = 64
    ns2 = ns.clone(); ns2[ns2.argsort()[:-k]] = -10.0
    assert torch.allclose(m.tail_rank_loss(ns2, as_bad, topk=k, normalize=False),
                          m.tail_rank_loss(ns, as_bad, topk=k, normalize=False), atol=1e-6)
    print('4) 只依赖 top-k 正常像素 ✔'); ok += 1

    # 5) 真实形状：B=8 正常图 / 16 辅助图，N=784 patch，D=16，K=64 原型
    Bn, Ba, N, D, K = 8, 16, 784, 16, 64
    np_ = torch.randn(Bn, N, D, requires_grad=True); nl = torch.randn(Bn, K, D, requires_grad=True)
    ap_ = torch.randn(Ba, N, D, requires_grad=True); al = torch.randn(Ba, K, D, requires_grad=True)
    mask = torch.zeros(Ba, N); mask[:, N // 2:] = 1.0
    lt = m.tail_rank_loss_from_features(np_, nl, ap_, al, mask, topk=64)
    assert torch.isfinite(lt) and lt.item() >= 0
    lt.backward()
    assert torch.isfinite(np_.grad).all()
    print('5) 真实形状集成（B=8/16, N=784, K=64）→ loss=%.4f，梯度 OK ✔' % lt.item()); ok += 1

    # 6) mask 全 0（batch 内没有异常像素）→ 可导的 0，不报错
    lz = m.tail_rank_loss_from_features(np_, nl, ap_, al, torch.zeros(Ba, N))
    lz.backward()
    print('6) 无异常像素 → 可导 0 ✔'); ok += 1

    print(f'\n{ok}/6 通过')


if __name__ == '__main__':
    main()
