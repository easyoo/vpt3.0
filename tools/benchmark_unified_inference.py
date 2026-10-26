#!/usr/bin/env python3
"""Benchmark the unified VPT3.8 inference pipeline after setup."""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cache_bank_nocrop import (  # noqa: E402
    _faiss_flat,
    query_batch_gpu,
)
from shot_utils import get_shots  # noqa: E402


def load_module(path, name):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name", default="transistor")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-path", default="/home/jjquan/MVTec-AD")
    parser.add_argument("--checkpoint-dir", default="results/vpt4/m4m6r3_2shot")
    parser.add_argument(
        "--support-csv",
        default="logs/mvtec2_support_adaptive_20261025/support_structure.csv",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--out-json", required=True)
    default_script = os.path.join(
        HERE, "code_vpt4", "main-fewshot-vpt4.py"
    )
    if not os.path.exists(default_script):
        default_script = os.path.join(
            HERE, "..", "code", "main-fewshot-vpt4.py"
        )
    parser.add_argument("--script", default=default_script)
    args = parser.parse_args()
    module = load_module(args.script, "benchmark_main")

    class Config:
        pass

    config = Config()
    config.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    config.fused_layer_learnable = [0]
    config.learnable_layers = 1
    config.learnable_insert_layers = ""
    config.learnable_num_per_layer = 32
    config.res_mapping = 0
    config.global_proto = 1
    config.metric_head = 1
    config.shot = args.shots
    module.args = config
    module.device = args.device

    model = module.vision_transformer.__dict__["vit_base"](
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-8,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        arg=config,
    )
    state = torch.load(
        os.path.join(
            args.checkpoint_dir,
            f"model_{args.class_name}_35.pth",
        ),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state["dino"], strict=False)
    model.to(args.device).eval()

    with open(args.support_csv, encoding="utf-8") as handle:
        support_row = next(
            row for row in csv.DictReader(handle)
            if row["class"] == args.class_name
        )
    center_border = float(support_row["center_border_cos"])
    norm_ratio = float(support_row["norm_ratio"])
    mapping_weight = max(
        0.0, min(1.0, (1.0 - center_border) * norm_ratio)
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf448 = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    tf672 = T.Compose([
        T.Resize((672, 672)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_root = os.path.join(
        args.data_path, args.class_name, "train", "good"
    )
    picks = get_shots(
        args.data_path,
        args.class_name,
        shots=args.shots,
        seed=args.seed,
    )
    support448 = torch.stack([
        tf448(Image.open(os.path.join(train_root, name)).convert("RGB"))
        for name in picks
    ], 0)
    support672 = torch.stack([
        tf672(Image.open(os.path.join(train_root, name)).convert("RGB"))
        for name in picks
    ], 0)

    raw_bank = F.normalize(
        module.build_raw_bank(
            model, support448, config.fused_layer, k=0
        ),
        dim=-1,
    )
    bank_patches = raw_bank.shape[0]
    generator = torch.Generator().manual_seed(1234)
    sub_idx = torch.randperm(
        bank_patches, generator=generator
    )[:min(1024, bank_patches)].to(raw_bank.device)

    from transformers import AutoModel
    from util.memorybank import MemoryBank, extract_fused_features

    bank_model = AutoModel.from_pretrained(
        "facebook/dinov2-base", output_hidden_states=True
    ).to(args.device).eval()
    module.setup_seed(args.seed)
    augmented = torch.cat([
        module.augmentation(image) for image in support672
    ], 0)
    external_bank = MemoryBank(
        target_cls=args.class_name,
        use_json=False,
        imgs=augmented.to(args.device),
        info=lambda *_: None,
        device=args.device,
        model=bank_model,
        dim=int(bank_model.config.hidden_size),
    )
    bank_tensors = [
        _faiss_flat(external_bank.faiss_index_1, args.device),
        _faiss_flat(external_bank.faiss_index_2, args.device),
    ]

    test_dataset = module.MVTecDataset(
        root=os.path.join(args.data_path, args.class_name),
        transform=tf448,
        gt_transform=tf448,
        phase="test",
        sort_items=False,
    )
    paths = list(test_dataset.img_paths)
    images448 = torch.stack([
        tf448(Image.open(path).convert("RGB")) for path in paths
    ], 0)
    images672 = torch.stack([
        tf672(Image.open(path).convert("RGB")) for path in paths
    ], 0)
    gaussian = module.get_gaussian_kernel(5, 4).to(args.device)

    @torch.no_grad()
    def mapping_score(images):
        images672_gpu = F.interpolate(
            images.to(args.device),
            scale_factor=1.5,
            mode="bilinear",
            align_corners=False,
        )
        result = model(images672_gpu, is_training=True)
        patch = torch.sum(torch.stack([
            result["patch_feat_layer"][layer]
            for layer in config.fused_layer
        ], 0), 0)
        prototype = torch.cat(result["feat_list"], dim=1)
        anchor = model.tokens_per_layer.token_dict["0"].expand(
            prototype.shape[0], -1, -1
        )
        prototype = torch.cat([prototype, anchor], dim=1)
        mapped_patch = F.normalize(
            model.token_mapping(patch), dim=-1
        )
        mapped_proto = F.normalize(
            model.token_mapping(prototype), dim=-1
        )
        learn = (
            1.0 - torch.bmm(
                mapped_patch, mapped_proto.transpose(1, 2)
            )
        ).clamp_min(0).min(dim=-1).values
        query = F.normalize(patch, dim=-1)
        nearest = None
        selected = []
        for start in range(0, bank_patches, 1024):
            end = min(start + 1024, bank_patches)
            distance = (
                1.0 - torch.bmm(
                    query,
                    raw_bank[start:end].t().unsqueeze(0).expand(
                        query.shape[0], -1, -1
                    ),
                )
            ).clamp_min(0)
            current = distance.min(dim=-1).values
            nearest = (
                current if nearest is None
                else torch.minimum(nearest, current)
            )
            local = sub_idx[
                (sub_idx >= start) & (sub_idx < end)
            ] - start
            if local.numel():
                selected.append(distance.index_select(-1, local))
        selected = torch.sort(
            torch.cat(selected, -1), -1
        ).values
        q512 = selected[..., min(512, selected.shape[-1]) - 1]
        raw = nearest.clamp_min(1e-6).sqrt() * \
            q512.clamp_min(1e-6).sqrt()
        score = 0.6 * learn + 0.4 * raw
        side = int(round(score.shape[1] ** 0.5))
        return F.interpolate(
            score.reshape(-1, 1, side, side),
            size=(448, 448),
            mode="bilinear",
            align_corners=True,
        )

    results = []
    for batch_size in [
        int(value) for value in args.batches.split(",")
    ]:
        for index in range(min(
            args.warmup_batches * batch_size, len(images448)
        ) // batch_size):
            start = index * batch_size
            end = start + batch_size
            mapping_score(images448[start:end])
        synchronize(args.device)
        total_images = 0
        total_map = total_bank = total_fusion = total_all = 0.0
        for start in range(0, len(images448), batch_size):
            end = min(start + batch_size, len(images448))
            batch_n = end - start

            synchronize(args.device)
            t0 = time.perf_counter()
            mapping = mapping_score(images448[start:end])
            synchronize(args.device)
            t1 = time.perf_counter()

            bank = query_batch_gpu(
                external_bank,
                images672[start:end].to(args.device),
                672,
                bank_tensors,
                8192,
                extract_fused_features,
            )
            bank = torch.as_tensor(
                bank, dtype=torch.float32, device=args.device
            )
            if bank.ndim == 2:
                bank = bank.unsqueeze(0)
            bank = bank.unsqueeze(1)
            synchronize(args.device)
            t2 = time.perf_counter()

            bank448 = F.interpolate(
                bank,
                size=(448, 448),
                mode="bilinear",
                align_corners=True,
            )[:, :1]
            fused = sum(
                mapping_weight
                * F.avg_pool2d(mapping, k, 1, k // 2)
                + (1.0 - mapping_weight)
                * F.avg_pool2d(bank448, k, 1, k // 2)
                for k in (9, 25, 41)
            ) / 3.0
            final_map = gaussian(fused)
            synchronize(args.device)
            t3 = time.perf_counter()

            total_map += t1 - t0
            total_bank += t2 - t1
            total_fusion += t3 - t2
            total_all += t3 - t0
            total_images += batch_n
        result = {
            "device": torch.cuda.get_device_name(args.device),
            "class": args.class_name,
            "shots": args.shots,
            "images": int(len(images448)),
            "batch_size": int(batch_size),
            "mapping_images_per_s": total_images / total_map,
            "bank_images_per_s": total_images / total_bank,
            "fusion_images_per_s": total_images / total_fusion,
            "end_to_end_images_per_s": total_images / total_all,
            "end_to_end_ms_per_image": 1000.0 * total_all / total_images,
            "mapping_share_pct": 100.0 * total_map / total_all,
            "bank_share_pct": 100.0 * total_bank / total_all,
            "fusion_share_pct": 100.0 * total_fusion / total_all,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.out_json)),
        exist_ok=True,
    )
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
