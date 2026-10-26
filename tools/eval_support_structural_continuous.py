#!/usr/bin/env python3
"""Evaluate continuous support-structure fusion without test-set tuning."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_singleview_weights import NAMES, box, load_map, load_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--calibration-json", default="")
    parser.add_argument("--mapping-residual", action="store_true")
    parser.add_argument("--normalized-max", action="store_true")
    parser.add_argument("--mapping-dir", required=True)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--data-path", default="/home/jjquan/MVTec-AD")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--use-norm-ratio", action="store_true")
    parser.add_argument("--dataset-relative-gate", action="store_true")
    parser.add_argument("--median-norm-gate", action="store_true")
    parser.add_argument("--ensemble-support-prior", action="store_true")
    parser.add_argument("--visa-support-rule", action="store_true")
    parser.add_argument("--geometric-consensus", action="store_true")
    parser.add_argument("--weighted-geometric", action="store_true")
    parser.add_argument("--agreement-residual", action="store_true")
    parser.add_argument("--scaled-agreement-residual", action="store_true")
    parser.add_argument("--multiscale-box", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--mask-root",
        default="",
        help="Training anomaly root; use median mask area as image top ratio.",
    )
    args = parser.parse_args()

    if args.support_json.endswith(".json"):
        records = json.load(open(args.support_json, encoding="utf-8"))
    else:
        records = list(csv.DictReader(
            open(args.support_json, encoding="utf-8")
        ))
    support = {record["class"]: record for record in records}
    calibration = {}
    if (args.mapping_residual or args.normalized_max
            or args.agreement_residual
            or args.scaled_agreement_residual):
        if not args.calibration_json:
            raise ValueError("--mapping-residual requires --calibration-json")
        calibration = {
            record["class"]: record
            for record in json.load(
                open(args.calibration_json, encoding="utf-8")
            )
        }
    class_names = args.classes.split()
    norm_ratios = np.asarray([
        float(support[name]["norm_ratio"]) for name in class_names
    ])
    norm_median = float(np.median(norm_ratios))
    norm_scale = float(
        (np.quantile(norm_ratios, 0.75) -
         np.quantile(norm_ratios, 0.25)) / 1.349
    )
    norm_scale = max(norm_scale, 1e-4)
    script = os.path.join(HERE, "code_vpt4", "main-fewshot-vpt4.py")
    if not os.path.exists(script):
        script = os.path.join(
            HERE, "..", "code", "main-fewshot-vpt4.py"
        )
    sys.path.insert(0, os.path.dirname(script))
    module = load_module(script, "continuous_support_eval")
    gaussian = module.get_gaussian_kernel(5, 4).to(args.device)
    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gt_transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
    ])
    rows = []
    for class_name in class_names:
        dataset = module.MVTecDataset(
            root=os.path.join(args.data_path, class_name),
            transform=transform,
            gt_transform=gt_transform,
            phase="test",
            sort_items=False,
        )
        gt = torch.stack([
            dataset[index][1] for index in range(len(dataset))
        ], 0).float()
        labels = np.asarray([
            int(dataset[index][2]) for index in range(len(dataset))
        ])
        frame = list(dataset.keys)
        mapping = load_map(
            args.mapping_dir, class_name, frame, args.size
        )
        bank = load_map(
            args.bank_dir, class_name, frame, args.size
        )
        record = support[class_name]
        center_border = float(record["center_border_cos"])
        norm_ratio = float(record["norm_ratio"])
        mapping_weight = 1.0 - center_border
        if args.median_norm_gate:
            mapping_weight = (
                mapping_weight
                if norm_ratio >= norm_median else 0.5
            )
        elif args.dataset_relative_gate:
            gate = 1.0 / (
                1.0 + np.exp(-(norm_ratio - norm_median) / norm_scale)
            )
            mapping_weight = (
                0.5 + (mapping_weight - 0.5) * gate
            )
        elif args.use_norm_ratio:
            mapping_weight *= norm_ratio
        mapping_weight = max(0.0, min(1.0, mapping_weight))
        fused = (
            mapping_weight * box(mapping)
            + (1.0 - mapping_weight) * box(bank)
        )
        if args.multiscale_box:
            fused = sum(
                mapping_weight * box(mapping, kernel)
                + (1.0 - mapping_weight) * box(bank, kernel)
                for kernel in (9, 25, 41)
            ) / 3.0
        if args.ensemble_support_prior:
            prior_trigger = center_border < 0.25
            if args.visa_support_rule:
                prior_trigger = prior_trigger and norm_ratio >= 0.88
            prior_weight = 0.75 if prior_trigger else 0.5
            prior_fused = (
                prior_weight * box(mapping)
                + (1.0 - prior_weight) * box(bank)
            )
            fused = 0.5 * (fused + prior_fused)
        if args.geometric_consensus:
            fused = torch.sqrt(
                (box(mapping) + 1e-6) * (box(bank) + 1e-6)
            )
        elif args.weighted_geometric:
            fused = (
                (box(mapping) + 1e-6).pow(mapping_weight)
                * (box(bank) + 1e-6).pow(1.0 - mapping_weight)
            )
        if args.mapping_residual:
            record_cal = calibration[class_name]
            median_m = float(record_cal["mapping_median"])
            median_b = float(record_cal["bank_median"])
            scale_m = float(record_cal["mapping_scale"])
            scale_b = float(record_cal["bank_scale"])
            z_m = (mapping - median_m) / scale_m
            z_b = (bank - median_b) / scale_b
            residual = torch.tanh(torch.relu(z_m - z_b))
            fused = fused + mapping_weight * residual
        elif args.agreement_residual:
            record_cal = calibration[class_name]
            z_m = (
                mapping - float(record_cal["mapping_median"])
            ) / float(record_cal["mapping_scale"])
            z_b = (
                bank - float(record_cal["bank_median"])
            ) / float(record_cal["bank_scale"])
            mean_m = box(z_m)
            mean_b = box(z_b)
            covariance = box(z_m * z_b) - mean_m * mean_b
            agreement = torch.sigmoid(covariance)
            residual = (
                mapping_weight * (1.0 - mapping_weight)
                * agreement * torch.tanh(torch.relu(z_m - z_b))
            )
            fused = fused + residual
        elif args.scaled_agreement_residual:
            record_cal = calibration[class_name]
            median_m = float(record_cal["mapping_median"])
            median_b = float(record_cal["bank_median"])
            scale_m = float(record_cal["mapping_scale"])
            scale_b = float(record_cal["bank_scale"])
            z_m = (mapping - median_m) / scale_m
            z_b = (bank - median_b) / scale_b
            mean_m = box(z_m)
            mean_b = box(z_b)
            covariance = box(z_m * z_b) - mean_m * mean_b
            agreement = torch.sigmoid(covariance)
            support_base_level = (
                mapping_weight * median_m
                + (1.0 - mapping_weight) * median_b
            )
            residual = (
                mapping_weight * (1.0 - mapping_weight)
                * support_base_level * agreement
                * torch.tanh(torch.relu(z_m - z_b))
            )
            fused = fused + residual
        elif args.normalized_max:
            record_cal = calibration[class_name]
            median_m = float(record_cal["mapping_median"])
            median_b = float(record_cal["bank_median"])
            scale_m = float(record_cal["mapping_scale"])
            scale_b = float(record_cal["bank_scale"])
            z_m = (mapping - median_m) / scale_m
            z_b = (bank - median_b) / scale_b
            fused = torch.maximum(
                mapping_weight * z_m,
                (1.0 - mapping_weight) * z_b,
            )
        with torch.no_grad():
            anomaly_map = gaussian(fused.to(args.device)).cpu()
        flat = anomaly_map.flatten(1)
        top_ratio = 0.005
        if args.mask_root:
            from PIL import Image
            mask_ratios = []
            class_root = os.path.join(args.mask_root, class_name)
            for defect in sorted(os.listdir(class_root)):
                mask_dir = os.path.join(class_root, defect, "mask")
                if not os.path.isdir(mask_dir):
                    continue
                for name in sorted(os.listdir(mask_dir)):
                    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    mask = Image.open(
                        os.path.join(mask_dir, name)
                    ).convert("L")
                    mask_ratios.append(
                        float((np.asarray(mask) > 0).mean())
                    )
            if not mask_ratios:
                raise RuntimeError(f"no training masks for {class_name}")
            top_ratio = float(np.median(mask_ratios))
        count = max(1, int(flat.shape[1] * top_ratio))
        image_score = torch.sort(
            flat, dim=1, descending=True
        )[0][:, :count].mean(dim=1).numpy()
        metric = list(module.ader_evaluator(
            anomaly_map[:, 0].numpy(),
            image_score,
            gt[:, 0].numpy() > 0.5,
            labels,
        ))
        mean = float(np.mean(metric) * 100)
        rows.append([class_name, mapping_weight, *metric, mean])
        print(
            f"{class_name:14s} wm={mapping_weight:.4f} "
            f"top={top_ratio:.4f} "
            + " ".join(
                f"{name}={value * 100:.3f}"
                for name, value in zip(NAMES, metric)
            )
            + f" mean={mean:.4f}",
            flush=True,
        )
    os.makedirs(
        os.path.dirname(os.path.abspath(args.out_csv)),
        exist_ok=True,
    )
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "class", "mapping_weight", *NAMES, "mean"
        ])
        writer.writerows(rows)
    print(
        "MACRO " + " ".join(
            f"{name}={np.mean([row[2 + index] for row in rows]) * 100:.3f}"
            for index, name in enumerate(NAMES)
        )
        + f" mean={np.mean([row[-1] for row in rows]):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
