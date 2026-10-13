import numpy as np
import torch
import os
import matplotlib.cm as cm
import torch.nn as nn
import cv2
from PIL import Image
# [AUDIT-FIX-7] accimage 是可选加速后端，原代码在模块级硬 import，
# 当前环境（vpt/vpt2/vpt3/vpt4）都没有装，导致 `from util.vis import *` 直接 ModuleNotFoundError、
# 整个脚本无法启动。改成可选依赖：缺了就退回 PIL 后端（torchvision 默认行为），不影响任何数值。
try:
    import accimage  # noqa: F401
    _HAS_ACCIMAGE = True
except ImportError:
    _HAS_ACCIMAGE = False
import torchvision
import torchvision.transforms as transforms
# [AUDIT-FIX-8] 同上：`from skimage import color` 在本文件里**从未被使用**（全文件搜不到 color. 的调用），
# 却因为当前环境没装 scikit-image 而阻断整个脚本启动。做成可选 import。
try:
    from skimage import color  # noqa: F401
except ImportError:
    color = None
import torch.nn.functional as F

def vis_rgb_gt_amp(img_paths, imgs, img_masks, anomaly_maps, method, root_out, dataset_name):
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)
    for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}/{specific_root}'
        os.makedirs(out_dir, exist_ok=True)
        img_path = f'{out_dir}/{img_num}_img.png'
        img_ano_path = f'{out_dir}/{img_num}_amp.png'
        mask_path = f'{out_dir}/{img_num}_mask.png'

        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        img_rec = img * std[:, None, None] + mean[:, None, None]
        # RGB image
        img_rec = Image.fromarray((img_rec * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        img_rec.save(img_path)
        # RGB image with anomaly map
        anomaly_map = anomaly_map / anomaly_map.max()
        anomaly_map = cm.jet(anomaly_map)
        # anomaly_map = cm.rainbow(anomaly_map)
        anomaly_map = (anomaly_map[:, :, :3] * 255).astype('uint8')
        anomaly_map = Image.fromarray(anomaly_map)
        img_rec_anomaly_map = Image.blend(img_rec, anomaly_map, alpha=0.4)
        img_rec_anomaly_map.save(img_ano_path)
        # mask
        img_mask = Image.fromarray((img_mask * 255).astype(np.uint8).transpose(1, 2, 0).repeat(3, axis=2))
        img_mask.save(mask_path)
        


import torch
import numpy as np

def denormalize(tensor, mean=None, std=None, to_numpy=True):
    """
    Denormalize a normalized tensor image and scale to [0, 255].
    
    Args:
        tensor (Tensor): Tensor of shape (C, H, W) or (B, C, H, W), values in normalized range.
        mean (list or None): Mean used in normalization. Default: ImageNet mean.
        std (list or None): Std used in normalization. Default: ImageNet std.
        to_numpy (bool): If True, return numpy array of dtype uint8; else return float tensor in [0,255].
        
    Returns:
        Tensor or np.ndarray: Denormalized image in range [0, 255].
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # Move to CPU for safety if converting to numpy
    if to_numpy:
        tensor = tensor.cpu()

    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Handle both (C, H, W) and (B, C, H, W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        unsqueezed = True
    elif tensor.dim() == 4:
        unsqueezed = False
    else:
        raise ValueError("Tensor must be 3D (C, H, W) or 4D (B, C, H, W)")

    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    # Denormalize: x = x * std + mean
    tensor = tensor * std + mean

    # Scale to [0, 255]
    tensor = tensor * 255.0

    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 255)

    if unsqueezed:
        tensor = tensor.squeeze(0)

    if to_numpy:
        # Convert to numpy uint8 (H, W, C) or (B, H, W, C)
        if tensor.dim() == 3:
            # (C, H, W) -> (H, W, C)
            arr = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            # (B, C, H, W) -> (B, H, W, C)
            arr = tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        return arr
    else:
        return tensor  # still float32/float64, but in [0, 255]
    

def global_min_max_norm(tensor):
    # tensor: (B, 1, H, W)
    t_min = tensor.min()
    t_max = tensor.max()
    normalized = (tensor - t_min) / (t_max - t_min)
    return torch.clamp(normalized, 0.0, 1.0)




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path

    
def save_img(img, gt, anomaly_map, path):
    """
    Save a visualization with a compact 2x2 layout:
    Row 1: [Original Image] [Ground Truth]
    Row 2: [Overlay]        [Anomaly Map]
    
    - Single vertical colorbar on the far right spanning both rows.
    - Compact spacing with no extra whitespace.
    """
    # 自动创建目录
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # --- 数据预处理 (与之前保持一致) ---
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    if gt.ndim == 3 and gt.shape[0] == 1: gt = gt[0]
    gt = gt.astype(np.uint8)

    if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1: anomaly_map = anomaly_map[0]
    
   
    display_map = anomaly_map.copy()
   

    # 生成热力图
    jet_heatmap = plt.cm.jet(display_map)
    jet_heatmap = (jet_heatmap[:, :, :3] * 255).astype(np.uint8)

    # 生成叠加图
    alpha = 0.5
    overlay = (1 - alpha) * img + alpha * jet_heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # --- 绘图核心：2x2 布局 ---
    
    # figsize=(10, 8) 保证每个子图接近正方形，constrained_layout 自动压缩间距
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # 1. 左上: Original Image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis("off")

    # 2. 右上: Ground Truth
    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title("Ground Truth", fontsize=12)
    axes[0, 1].axis("off")

    # 3. 左下: Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Overlay", fontsize=12)
    axes[1, 0].axis("off")

    # 4. 右下: Anomaly Map (含统计信息)
    # 使用 jet colormap，并保存 im 对象用于生成色条
    im = axes[1, 1].imshow(display_map, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title(f"Anomaly Map\nMax: {anomaly_map.max():.3f} | Mean: {anomaly_map.mean():.3f}", fontsize=11)
    axes[1, 1].axis("off")

    # --- 统一色条设置 ---
    # ax=axes 表示色条依附于整个 2x2 网格，放在最右侧
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label('Anomaly Score', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=10)

    # 保存
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

