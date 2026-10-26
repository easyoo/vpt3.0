#!/usr/bin/env python3
"""Benchmark the learnable anomaly branch after model setup."""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(path, name):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name", default="transistor")
    parser.add_argument(
        "--data-path", default="/home/jjquan/MVTec-AD"
    )
    parser.add_argument(
        "--checkpoint",
        default="results/vpt4/m4m6r3_2shot/model_transistor_35.pth",
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
    module = load_module(args.script, "learn_benchmark_main")

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
    config.shot = 2
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
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["dino"], strict=False)
    model.to(args.device).eval()

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    dataset = module.MVTecDataset(
        root=os.path.join(args.data_path, args.class_name),
        transform=transform,
        gt_transform=transform,
        phase="test",
        sort_items=False,
    )
    images = torch.stack([
        transform(Image.open(path).convert("RGB"))
        for path in dataset.img_paths
    ], 0)

    @torch.no_grad()
    def learning_score(batch):
        image = F.interpolate(
            batch.to(args.device),
            scale_factor=1.5,
            mode="bilinear",
            align_corners=False,
        )
        result = model(image, is_training=True)
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
        score = (
            1.0 - torch.bmm(
                mapped_patch, mapped_proto.transpose(1, 2)
            )
        ).clamp_min(0).min(dim=-1).values
        side = int(round(score.shape[1] ** 0.5))
        score = score.reshape(-1, 1, side, side)
        return F.interpolate(
            score,
            size=(448, 448),
            mode="bilinear",
            align_corners=True,
        )

    results = []
    for batch_size in [
        int(value) for value in args.batches.split(",")
    ]:
        warmup = min(
            args.warmup_batches * batch_size, len(images)
        )
        for start in range(0, warmup, batch_size):
            learning_score(images[start:min(start + batch_size, warmup)])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = 0.0
        count = 0
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            begin = time.perf_counter()
            learning_score(batch)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - begin
            count += len(batch)
        result = {
            "device": torch.cuda.get_device_name(args.device),
            "class": args.class_name,
            "images": count,
            "batch_size": batch_size,
            "learning_branch_images_per_s": count / elapsed,
            "learning_branch_ms_per_image": 1000.0 * elapsed / count,
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
