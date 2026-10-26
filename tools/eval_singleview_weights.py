#!/usr/bin/env python3
"""Single-view mapping/bank weight sweep with exact box25 fusion."""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [
    "I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def zscore(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def box(x, k=25):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2)


def load_map(root, class_name, frame, size):
    base = os.path.join(HERE, root)
    mapping = torch.load(
        os.path.join(base, f"{class_name}.map"),
        map_location="cpu", weights_only=False,
    ).float()
    sidecar_path = os.path.join(base, f"{class_name}.json")
    keys = []
    if os.path.exists(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as handle:
            keys = json.load(handle).get("keys") or []
    if keys and list(keys) != list(frame):
        if set(keys) != set(frame):
            raise RuntimeError("mapping and test frames differ")
        mapping = mapping[
            torch.tensor([keys.index(key) for key in frame])
        ]
    if mapping.shape[-2:] != (size, size):
        mapping = F.interpolate(
            mapping, size=(size, size),
            mode="bilinear", align_corners=True,
        )
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True)
    parser.add_argument("--mapping-dir", default=".maps_mvtec4_strict")
    parser.add_argument("--bank-dir", default=".errmap672_mvtec4_strict")
    parser.add_argument("--data-path", default="/home/jjquan/MVTec-AD")
    parser.add_argument("--weights", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    script = os.path.join(HERE, "code_vpt4", "main-fewshot-vpt4.py")
    if not os.path.exists(script):
        script = os.path.join(
            HERE, "..", "code", "main-fewshot-vpt4.py"
        )
    sys.path.insert(0, os.path.dirname(script))
    module = load_module(script, "weight_sweep_main")
    gaussian = module.get_gaussian_kernel(
        kernel_size=5, sigma=4
    ).to(args.device)
    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    gt_transform = T.Compose([
        T.Resize((args.size, args.size)), T.ToTensor()
    ])
    weights = [float(value) for value in args.weights.split(",")]
    rows = []
    for class_name in args.classes.split():
        dataset = module.MVTecDataset(
            root=os.path.join(args.data_path, class_name),
            transform=transform,
            gt_transform=gt_transform,
            phase="test",
            sort_items=False,
        )
        gt = torch.stack(
            [dataset[index][1] for index in range(len(dataset))],
            dim=0,
        ).float()
        labels = np.asarray(
            [int(dataset[index][2]) for index in range(len(dataset))]
        )
        frame = list(dataset.keys)
        mapping = load_map(
            args.mapping_dir, class_name, frame, args.size)
        bank = load_map(args.bank_dir, class_name, frame, args.size)
        map_box = box(mapping)
        bank_box = box(bank)
        for weight in weights:
            fused = zscore(
                weight * map_box + (1.0 - weight) * bank_box)
            with torch.no_grad():
                anomaly_map = gaussian(fused.to(args.device)).cpu()
            flat = anomaly_map.flatten(1)
            count = max(1, int(flat.shape[1] * 0.005))
            image_score = torch.sort(
                flat, dim=1, descending=True
            )[0][:, :count].mean(dim=1).numpy()
            metrics = list(module.ader_evaluator(
                anomaly_map[:, 0].numpy(),
                image_score,
                gt[:, 0].numpy() > 0.5,
                labels,
            ))
            mean = float(np.mean(metrics) * 100)
            rows.append([class_name, weight, *metrics, mean])
            print(
                f"{class_name:14s} wm={weight:.2f} "
                + " ".join(
                    f"{name}={value * 100:.3f}"
                    for name, value in zip(NAMES, metrics)
                )
                + f" mean={mean:.4f}",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as handle:
        handle.write(
            ",".join(["class", "mapping_weight", *NAMES, "mean"])
            + "\n"
        )
        for row in rows:
            handle.write(
                ",".join(str(value) for value in row) + "\n"
            )
    for weight in weights:
        selected = [row for row in rows if row[1] == weight]
        print(
            f"MACRO wm={weight:.2f} " + " ".join(
                f"{name}={np.mean([row[2 + i] for row in selected]) * 100:.3f}"
                for i, name in enumerate(NAMES)
            ) + f" mean={np.mean([row[-1] for row in selected]):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
