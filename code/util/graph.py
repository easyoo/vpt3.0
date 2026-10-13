import os
import cv2
import numpy as np
from pathlib import Path

def split_image_to_9_patches(image_path, output_dir):
    """
    将图片切分为9个相同大小的子图
    
    Args:
        image_path (str): 输入图片路径
        output_dir (str): 输出目录路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 计算每个子图的尺寸
    patch_height = height // 3
    patch_width = width // 3
    
    # 切分图像并保存
    for i in range(3):
        for j in range(3):
            # 计算裁剪区域
            y1 = i * patch_height
            y2 = (i + 1) * patch_height if i < 2 else height
            x1 = j * patch_width
            x2 = (j + 1) * patch_width if j < 2 else width
            
            # 裁剪子图
            patch = img[y1:y2, x1:x2]
            
            # 生成输出文件名
            base_name = Path(image_path).stem
            patch_name = f"{base_name}_patch_{i}_{j}.jpg"
            output_path = os.path.join(output_dir, patch_name)
            
            # 保存子图
            cv2.imwrite(output_path, patch)
            
    print(f"成功将图片 {image_path} 切分为9个子图并保存到 {output_dir}")

if __name__ == "__main__":
    # 示例用法
    # 请根据需要修改以下路径
    input_image_path = "/home/jjquan/datasets/mvtec/bottle/test/good/000.png"
    output_directory = "./split_images"
    
    split_image_to_9_patches(input_image_path, output_directory)