# -*- coding: utf-8 -*-
"""[FIX-14 验证] 分割头损失：Dice 应为"逐样本计算后再平均"，而不是整 batch 一个全局 Dice。

原实现（`visual/main-fewshot.py:1069` `FocalDiceLoss.forward`）：
    pred = pred_score_map.view(-1); target = gt_mask.view(-1)
    dice_score = (2*Σ(p*t) + s) / (Σp + Σt + s)      # ← 整个 batch 合成一个大 Dice
问题：把 B 张图的所有 patch 混在一起算一个 Dice，梯度被 batch 组成稀释/耦合——
      异常面积大的样本主导全局 union，异常面积小的样本几乎拿不到有效梯度信号。
标准做法：每张图各算 Dice，再对 B 取平均。

本脚本在 **CPU** 上做三件事（无需 GPU）：
  1) 数值对照：构造"一张图预测很好、另一张图预测很差"的 batch，展示两种实现的差异；
  2) 梯度检查：分别回传，检查各样本是否都能拿到梯度；
  3) 形状/边界情形（全背景样本、全前景样本）不产生 NaN。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLossGlobal(nn.Module):
    """原实现（逐位复刻，仅用于对照）。"""

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.dice_weight, self.focal_weight = dice_weight, focal_weight

    def forward(self, pred_score_map, gt_mask):
        pred = pred_score_map.view(-1)
        target = gt_mask.view(-1)
        s = 1e-5
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        loss_dice = 1 - (2. * inter + s) / (union + s)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        pt = torch.where(target == 1, pred, 1 - pred)
        focal = (1 - pt).pow(self.gamma)
        alpha = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss_focal = (alpha * focal * bce).mean()
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal, loss_dice


class FocalDiceLossPerSample(nn.Module):
    """[FIX-14] Dice 逐样本计算再平均；Focal 仍是逐像素均值（等价于逐样本均值再平均）。

    注意（由 §3 的梯度检查实测暴露）：如果直接对**全背景样本**也算 Dice，
    当该样本预测也恰好全为 0 时，U=0，`1 - s/(U+s)` 虽然损失≈0，
    但梯度 ∂/∂p = s/(U+s)² = 1/s = **1e5** —— 一个"完美预测"却产生爆炸性梯度。
    因此对 `target.sum()==0` 的样本**跳过 Dice 项**（这些样本的"别乱报"信号由 Focal/BCE 提供），
    只对有前景的样本平均。`skip_empty=False` 可复现上述危险行为，供对照。
    """

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5,
                 skip_empty=True):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.dice_weight, self.focal_weight = dice_weight, focal_weight
        self.skip_empty = skip_empty

    def forward(self, pred_score_map, gt_mask):
        B = pred_score_map.shape[0]
        pred = pred_score_map.reshape(B, -1)
        target = gt_mask.reshape(B, -1).float()
        s = 1e-5
        inter = (pred * target).sum(dim=1)                 # [B]
        union = pred.sum(dim=1) + target.sum(dim=1)        # [B]
        per_dice = 1 - (2. * inter + s) / (union + s)      # [B]
        if self.skip_empty:
            valid = (target.sum(dim=1) > 0).float()
            loss_dice = (per_dice * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            loss_dice = per_dice.mean()
        p = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        pt = torch.where(target == 1, p, 1 - p)
        alpha = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss_focal = (alpha * (1 - pt).pow(self.gamma) * bce).mean()
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal, loss_dice


def main():
    torch.manual_seed(0)
    B, N = 4, 64
    # 构造：样本0/1 是"异常图"（有前景像素，且样本0的预测很好、样本1很差），样本2/3 是"正常图"（全背景）
    t = torch.zeros(B, N)
    t[0, :8] = 1.0
    t[1, :8] = 1.0
    p = torch.zeros(B, N, 1)
    p[0, :8, 0] = 0.95                      # 好样本
    p[1, :8, 0] = 0.05                      # 差样本
    p[0, 8:20, 0] = 0.30                    # 一些误报
    p[1, 8:20, 0] = 0.30

    g = FocalDiceLossGlobal(); ps = FocalDiceLossPerSample(skip_empty=True)
    ps_bad = FocalDiceLossPerSample(skip_empty=False)
    print('=== 1) 数值对照（同一 batch，两种 Dice 口径）===')
    for name, crit in [('global(原实现)', g),
                       ('per-sample(不跳空)', ps_bad),
                       ('per-sample(FIX-14)', ps)]:
        loss, dice = crit(p, t)
        print(f'  {name:<22} total={loss.item():.4f}  dice_loss={dice.item():.4f}'
              f'  → dice_score={1-dice.item():.4f}')

    print('\n=== 2) 逐样本 Dice（只可能由 FIX-14 给出）===')
    s = 1e-5
    for i in range(B):
        pi, ti = p[i].reshape(-1), t[i].reshape(-1)
        inter = (pi * ti).sum(); union = pi.sum() + ti.sum()
        print(f'  sample{i}: dice_score={(2*inter+s)/(union+s):.4f}  (前景像素={int(ti.sum())})')

    print('\n=== 3) 梯度检查（每个样本是否都能影响分割头）===')
    for name, crit in [('global', g), ('per-sample(不跳空)', ps_bad), ('per-sample(FIX-14)', ps)]:
        pp = p.clone().requires_grad_(True)
        loss, _ = crit(pp, t)
        loss.backward()
        gn = pp.grad.reshape(B, -1).norm(dim=1)
        print(f'  {name:<11} 各样本梯度范数: ' + ' '.join(f'{x:.2e}' for x in gn)
              + f'  (min={gn.min():.2e})')

    print('\n=== 4) 边界情形（全背景 / 全前景）不产生 NaN ===')
    for tag, tt in [('全背景', torch.zeros(B, N)), ('全前景', torch.ones(B, N))]:
        o = ps(p, tt)[0]
        print(f'  {tag}: loss={o.item():.4f}  finite={torch.isfinite(o).item()}')


if __name__ == '__main__':
    main()
