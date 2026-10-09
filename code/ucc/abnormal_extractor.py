import sys

# 1. 从 sys.path 中移除包含 'dinov2' 的本地路径（可选，但建议）
sys.path = [p for p in sys.path if 'dinov2' not in p]

# 2. 关键！强制删除已缓存的 dinov2 模块
if 'dinov2' in sys.modules:
    del sys.modules['dinov2']

# 3. 如果还有子模块，也一并清除（更彻底）
modules_to_remove = [mod for mod in sys.modules if mod.startswith('dinov2')]
for mod in modules_to_remove:
    del sys.modules[mod]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn.functional as F

import sys
import os

anomaly_dict = {
    'bottle':"/home/jjquan/datasets/mvtec/bottle/test/broken_large/000.png",
    'cable':'/home/jjquan/datasets/mvtec/cable/test/bent_wire/000.png',
    'capsule':'/home/jjquan/datasets/mvtec/capsule/test/crack/000.png',
    'carpet':'/home/jjquan/datasets/mvtec/carpet/test/color/000.png',
    'grid':'/home/jjquan/datasets/mvtec/grid/test/bent/000.png',
    'hazelnut':'/home/jjquan/datasets/mvtec/hazelnut/test/crack/000.png',
    'leather':'/home/jjquan/datasets/mvtec/leather/test/cut/000.png',
    'metal_nut':'/home/jjquan/datasets/mvtec/metal_nut/test/scratch/000.png',
    'pill':'/home/jjquan/datasets/mvtec/pill/test/crack/000.png',
    'screw':'/home/jjquan/datasets/mvtec/screw/test/scratch_head/000.png',
    'tile':'/home/jjquan/datasets/mvtec/tile/test/crack/000.png',
    'toothbrush':'/home/jjquan/datasets/mvtec/toothbrush/test/defective/000.png',
    'transistor':'/home/jjquan/datasets/mvtec/transistor/test/bent_lead/000.png',
    'wood':'/home/jjquan/datasets/mvtec/wood/test/color/000.png',
    'zipper':'/home/jjquan/datasets/mvtec/zipper/test/broken_teeth/000.png'
}


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms




def dinov2_extractor(img_path):
    '''
    x: 输入图片张量，尺寸为(1, 3, 392, 392)
    返回patch token:BLC
    
    '''
    
    data_transform, gt_transform = get_data_transforms(448,392)
    
    mask_path = img_path.replace('.png', '_mask.png').replace('test', 'ground_truth')
    
    img = Image.open(img_path).convert('RGB')
    x = data_transform(img)
    x.unsqueeze_(0)  # 增加batch维度
    x_mask = gt_transform(Image.open(mask_path))
    
    # 加载模型
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    # model = torch.hub.load('/home/jjquan/codebase/dinov2', 'dinov2_vitb14_reg',source='local')
    model.eval()
    
    pth = '/home/jjquan/codebase/vpt/checkpoint/dinov2_vitb14_reg4_pretrain.pth'
    state_dict = torch.load(pth, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    

    # Hook 存储
    # features_after_block0 = {}

    # def hook_fn(module, input, output):
    #     features_after_block0['tokens'] = output.detach()

    # handle = model.blocks[0].register_forward_hook(hook_fn)

    # 输入：392x392
   
   
    features_after_blocks = {}

    def make_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            features_after_blocks[f'{layer_idx}'] = output.detach()
        return hook_fn

    # 注册多个层的hook
    hook_handles = []
    layers_to_register = [0, 1, 2, 3, 4, 5, 6, 8, 11]
    for layer_idx in layers_to_register:
        handle = model.blocks[layer_idx].register_forward_hook(make_hook_fn(layer_idx))
        hook_handles.append(handle)

    with torch.no_grad():
        _ = model.forward_features(x)  # 自动处理任意尺寸（只要能被14整除）

    for handle in hook_handles:
        handle.remove()
    features_after_blocks = {k: v[:,5:,] for k, v in features_after_blocks.items()}
    
    layers_to_sum = ['0', '1', '2', '3', '4', '5', '6', '8', '11']
    selected_features = [features_after_blocks[layer] for layer in layers_to_sum]
    
    stacked_features = torch.stack(selected_features, dim=0)  # 形状变为 [num_layers, B, L, C]

    # 对第 0 维度进行求和
    summed_features = torch.sum(stacked_features, dim=0)  # 形状变为 [B, L, C]
    
    tokens = summed_features
    B,L,C = tokens.shape
    tokens = tokens.reshape(B, C, 28, 28)
    pixel_features = F.interpolate(
        tokens, 
        size=(392, 392), 
        mode='bilinear', 
        align_corners=False
    )

    return pixel_features,x_mask


def extract_features_with_mask(features, mask):
    """
    从BCHW格式的特征图中提取mask为1位置的特征
    
    Args:
        features: BCHW格式的特征图
        mask: 1HW格式的mask
        
    Returns:
        masked_features: mask为1位置的特征 (N, C) 其中N是mask为1的像素数
    """
    # 确保mask是二值化的
    binary_mask = (mask > 0).float()
    # 将mask扩展到和features相同的batch和channel维度
    # features: (B, C, H, W)
    # mask: (1, H, W) -> (B, C, H, W)
    B, C, H, W = features.shape
    expanded_mask = binary_mask.expand(B, C, H, W)
    
    # 使用mask选择特征
    # 将特征和mask都展平为(B, C, H*W)
    features_flat = features.view(B, C, H * W)
    mask_flat = expanded_mask.view(B, C, H * W)
    
    # 提取mask为1位置的特征
    masked_features = features_flat[mask_flat.bool()]
    
    # 重塑为(N, C)形状，其中N是选中的特征数
    if masked_features.numel() > 0:
        masked_features = masked_features.view(-1, C)
    
    return masked_features,torch.mean(masked_features,dim=0)



def anomaly_feature(cls_name):
    feat,mask = dinov2_extractor(anomaly_dict[cls_name])
    return extract_features_with_mask(feat, mask)

if __name__ == '__main__':
    print(anomaly_feature('bottle')[1].shape)
    