#!/usr/bin/env python3
"""Unsupervised support-image structure statistics for fusion weighting.

Only the strict 4-shot normal images are used.  No test images or labels are
read.  The statistics measure how strongly DINO patch features separate into
foreground/background-like clusters and how different the border is from the
image center.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel

from shot_utils import get_shots


def two_cluster_ratio(x, iterations=20):
    x = x - x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=1, center=False)
    z = x @ v[:, 0]
    lo, hi = z.min(), z.max()
    centers = torch.stack([lo, hi])
    for _ in range(iterations):
        assign = (z - centers[0]).abs() <= (z - centers[1]).abs()
        new = centers.clone()
        if assign.any():
            new[0] = z[assign].mean()
        if (~assign).any():
            new[1] = z[~assign].mean()
        centers = new
    between = 0.0
    total = (z - z.mean()).pow(2).sum()
    for flag in (False, True):
        cur = z[assign == flag]
        if cur.numel():
            between = between + cur.numel() * (cur.mean() - z.mean()).pow(2)
    minority = min(float(assign.float().mean()), 1.0 - float(assign.float().mean()))
    return float(between / total.clamp_min(1e-8)), minority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes",
        default="bottle cable capsule carpet grid hazelnut leather metal_nut "
                "pill screw tile toothbrush transistor wood zipper",
    )
    parser.add_argument(
        "--data-path",
        default="/home/jjquan/experiment/visual2/MVTec-AD_strict4",
    )
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size", type=int, default=672)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model", default="facebook/dinov2-base"
    )
    args = parser.parse_args()

    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    model = AutoModel.from_pretrained(
        args.model, output_hidden_states=True
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    patch = int(getattr(model.config, "patch_size", 14))
    side = args.size // patch
    yy, xx = torch.meshgrid(
        torch.arange(side), torch.arange(side), indexing="ij"
    )
    center_mask = (
        (yy >= side * 0.25) & (yy < side * 0.75) &
        (xx >= side * 0.25) & (xx < side * 0.75)
    ).flatten()
    border_mask = ~center_mask

    print(
        "class,cluster_ratio,cluster_minority,center_border_cos,"
        "center_var,border_var,center_norm,border_norm,norm_ratio"
    )
    for class_name in args.classes.split():
        picks = get_shots(
            args.data_path, class_name,
            shots=args.shots, seed=args.seed,
        )
        root = os.path.join(
            args.data_path, class_name, "train", "good"
        )
        images = torch.stack([
            transform(Image.open(os.path.join(root, name)).convert("RGB"))
            for name in picks
        ]).to(args.device)
        with torch.no_grad():
            hidden = model(images).hidden_states
            raw_feat = torch.stack(hidden[7:11], 0).sum(0)[:, 1:, :]
            feat = F.normalize(raw_feat, dim=-1)
        cluster_values = [
            two_cluster_ratio(cur) for cur in feat
        ]
        cluster = np.mean([value[0] for value in cluster_values])
        minority = np.mean([value[1] for value in cluster_values])
        center = feat[:, center_mask].mean(1)
        border = feat[:, border_mask].mean(1)
        center_border = float(
            (center * border).sum(-1).mean())
        center_var = float(
            (1.0 - feat[:, center_mask] @ center.unsqueeze(-1))
            .mean())
        border_var = float(
            (1.0 - feat[:, border_mask] @ border.unsqueeze(-1))
            .mean())
        center_norm = float(
            raw_feat[:, center_mask].norm(dim=-1).mean())
        border_norm = float(
            raw_feat[:, border_mask].norm(dim=-1).mean())
        print(
            f"{class_name},{cluster:.6f},{minority:.6f},"
            f"{center_border:.6f},{center_var:.6f},{border_var:.6f},"
            f"{center_norm:.6f},{border_norm:.6f},"
            f"{center_norm / max(border_norm, 1e-8):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
