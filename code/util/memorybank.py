# import model
import yaml
import torch
# from model import get_model
from PIL import Image
from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F
import torch.nn.functional as F1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# from putil.logging import setup_logger
import faiss
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from timm.models import create_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
# from data import get_loader
# from util.metric import get_evaluator
# from configs.mambaad.mambaad_mvtec import cfg
from tqdm import tqdm
from transformers import AutoModel
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image
import os
import matplotlib.cm as cm
import random
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as T
import numpy as np
import json
import argparse
import glob
# from test_dino_finetune import AnomalyDINO

def augment_images(image_paths, n, m, save_dir="./logs/augments"):
    """
    对每个样本进行多种数据增强，并保存到指定目录，同时返回扩充后样本的绝对路径列表。
    
    参数：
        image_paths (list): 图片路径列表。
        n (int): 每种几何变换的增强次数。
        m (int): 每种颜色和噪声变换的增强次数。
        save_dir (str): 增强后的文件保存目录。
    
    返回：
        list: 扩充后样本的绝对路径列表。
    """
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录


    # 定义增强操作
    def rotation(image,angle):
        return F.rotate(image, angle,expand=True)

    def flip(image,Horizontal=True):
        if Horizontal:
            return F.hflip(image)
        return F.vflip(image)


    def random_crop(image):
        max_crop_size = min(image.height, image.width) // 2  # 设置最大裁剪尺寸为最小边的一半
        top = random.randint(0, max(0, image.height - max_crop_size))
        left = random.randint(0, max(0, image.width - max_crop_size))
        height = random.randint(image.height // 2, image.height - top)
        width = random.randint(image.width // 2, image.width - left)
        
        return F.resized_crop(image, top=top, left=left, height=height, width=width, size=(image.height, image.width))
    
    def random_translation(image):
        translate = (random.uniform(-0.03, 0.03) * image.width, random.uniform(-0.03, 0.03) * image.height)
        return F.affine(image, angle=0, translate=translate, scale=1.0, shear=0)

    def random_perspective(image):
        transform = T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=T.InterpolationMode.BILINEAR)
        return transform(image)

    def color_jitter(image):
        transform = transform = T.ColorJitter(
        brightness=(0.7, 1.4),  # 随机从0.5到1.5之间选择亮度因子
        contrast=(0.8, 1.2),    # 对比度因子范围
        saturation=(0.8, 1.2),  # 饱和度因子范围
        hue=(-0.2, 0.2)       # 色调变化范围，注意hue的取值范围是-0.5到0.5
    )
        return transform(image)

    def add_noise(image):
        np_image = np.array(image)
        noise = np.random.normal(0, 25, np_image.shape).astype(np.int16)
        noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def blur_image(image):
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))

    # def random_erase(image):
    #     np_image = np.array(image)
    #     h, w, _ = np_image.shape
    #     x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
    #     x2, y2 = random.randint(x1, w), random.randint(y1, h)
    #     np_image[y1:y2, x1:x2, :] = 0
    #     return Image.fromarray(np_image)

    # 存储扩充后样本的路径
    augmented_paths = []

    # 遍历每张图片
    for img_path in tqdm(image_paths, desc='Data augmenting...'):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        cls_name = next((cls for cls in clses if cls in img_path), None)
        assert cls_name is not None, f"Error: Could not extract class name from path: {img_path}"
        img_save_dir = os.path.join(save_dir, cls_name, img_name)
        os.makedirs(img_save_dir, exist_ok=True)  # 为每张图片创建单独的目录

        # 加载图片
        image = Image.open(img_path).convert("RGB")
        for i in range(360//base_angle):
            rotation_path = os.path.join(img_save_dir, f"rotation_{i+1}.png")
            angle = (i+1) * base_angle
            rotation(image,angle).save(rotation_path) 
            augmented_paths.append(os.path.abspath(rotation_path))
        
        flip_path = os.path.join(img_save_dir, f"flip_h.png")
        flip(image,True).save(flip_path)
        augmented_paths.append(os.path.abspath(flip_path))
        
        flip_path = os.path.join(img_save_dir, f"flip_v.png")
        flip(image,False).save(flip_path)
        augmented_paths.append(os.path.abspath(flip_path))
        # 几何变换增强
        for i in range(n):
            crop_path = os.path.join(img_save_dir, f"crop_{i}.png")
            random_crop(image).save(crop_path)
            augmented_paths.append(os.path.abspath(crop_path))

            translation_path = os.path.join(img_save_dir, f"translation_{i}.png")
            random_translation(image).save(translation_path)
            augmented_paths.append(os.path.abspath(translation_path))

            # perspective_path = os.path.join(img_save_dir, f"perspective_{i}.png")
            # random_perspective(image).save(perspective_path)
            # augmented_paths.append(os.path.abspath(perspective_path))
       
        # 颜色和噪声增强
        for i in range(m):
            jitter_path = os.path.join(img_save_dir, f"color_jitter_{i}.png")
            color_jitter(image).save(jitter_path)
            augmented_paths.append(os.path.abspath(jitter_path))

            # noise_path = os.path.join(img_save_dir, f"noise_{i}.png")
            # add_noise(image).save(noise_path)
            # augmented_paths.append(os.path.abspath(noise_path))

            # blur_path = os.path.join(img_save_dir, f"blur_{i}.png")
            # blur_image(image).save(blur_path)
            # augmented_paths.append(os.path.abspath(blur_path))

            # erase_path = os.path.join(img_save_dir, f"erase_{i}.png")
            # random_erase(image).save(erase_path)
            # augmented_paths.append(os.path.abspath(erase_path))

    info(f"Augmented images saved to {save_dir}")
    return augmented_paths

def create_single_class_dataloader(dataloader, target_class='bottle'):
    """
    将多类 dataloader 转换为单类 dataloader。
    
    参数：
        dataloader (DataLoader): 原始多类 dataloader。
        target_class (str): 目标类别名称。
        
    返回：
        DataLoader: 仅包含目标类别的 dataloader。
    """
    data = {
    "carpet": (0, 116),
    "grid": (117, 194),
    "leather": (195, 318),
    "tile": (319, 435),
    "wood": (436, 514),
    "bottle": (515, 597),
    "cable": (598, 747),
    "capsule": (748, 879),
    "hazelnut": (880, 989),
    "metal_nut": (990, 1104),
    "pill": (1105, 1271),
    "screw": (1272, 1431),
    "toothbrush": (1432, 1473),
    "transistor": (1474, 1573),
    "zipper": (1574, 1724)}

    # 构建字典
    cls_indices = {class_name: list(range(begin, end + 1)) for class_name, (begin, end) in data.items()}
    # 读取所有数据并筛选目标类别的索引
    # cls_indices = {}
    # for idx, data in enumerate(tqdm(dataloader.dataset,desc='Creating single class dataloader')):
    #     if data['cls_name'] not in cls_indices:
    #         cls_indices[data['cls_name']] = []
    #     cls_indices[data['cls_name']].append(idx)
    # for name in cls_indices.keys():
    #     info(f"class-name:{name} == begin_index:{cls_indices[name][0]} == end_index:{cls_indices[name][-1]}")
    # 创建仅包含目标类别的子集
    single_class_dataset = Subset(dataloader.dataset, cls_indices[target_class])
    
    # 创建新的 dataloader
    single_class_dataloader = DataLoader(
        single_class_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers
    )
    
    return single_class_dataloader

def extract_fused_features(model,images,fuse_layer_encoder=[[2,3,4,5], [6,7,8,9]]):
    # 提取特征
    with torch.no_grad():
        if isinstance(model,nn.Linear):
            # 如果模型是 AnomalyDINO，则提取两个阶段的特征
            in1, out1, in2, out2, class_out = model(images)
            fused_features = [out1, out2]
        else: 
            outputs = model(images)
            hidden_states = outputs.hidden_states  # 获取所有隐藏层的输出
            # 融合指定层的特征
            fused_features = []
            for layer_group in fuse_layer_encoder:
                # 将子列表内的特征相加
                fused_feature = sum(hidden_states[layer] for layer in layer_group)
                fused_features.append(fused_feature[:,1:,:])

        return fused_features

class MemoryBank():
    def __init__(self,
                 target_cls,
                 use_json,shot=1,
                 model = None,
                 dim=768,
                 use_augment=True,
                 augment_cache_dir=None,
                 dataset_json_path='/home/jjquan/datasets/mvtec/meta.json',
                 yaml_path='/home/jjquan/codebase/OmniADBank/reference/ref.yml',
                 resize=448,
                 crop_size=392,
                 dataset='mvtec',
                 use_cos = True,
                 faster_index=(False,768),
                 gussian_sigma=4,
                 imgs=None,
                 info=None,
                 device=None):
        self.dataset = dataset
        self.target_cls = target_cls
        self.use_augment = use_augment
        self.augment_cache_dir = augment_cache_dir
        self.use_json = use_json
        self.shot = shot
        self.yaml_path = yaml_path
        self.dataset_json_path = dataset_json_path
        self.dataset = dataset
        self.use_cos = use_cos
        self.faster_index = faster_index
        self.resize = resize
        self.gussian_sigma = gussian_sigma
        
        
        self.feature_bank_1 = []
        self.feature_bank_2 = []
          
        self.model_t = AutoModel.from_pretrained("facebook/dinov2-base", output_hidden_states=True) if model is None else model
        self.model_t.eval()
        self.model_t.to(device)
        
        self.train_transforms = self.test_transforms = Compose([
            Resize((self.resize, self.resize), interpolation=F.InterpolationMode.BILINEAR),
            CenterCrop((crop_size, crop_size)),
            ToTensor(),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ])

        self.target_transforms = Compose([
            Resize((self.resize, self.resize), interpolation=F.InterpolationMode.BILINEAR),
            CenterCrop((crop_size, crop_size)),
            ToTensor(),
        ])
        
        self.info = info
       
        # 初始化 faiss 索引
        use_gpu = False
        if use_gpu:
            res = faiss.StandardGpuResources()  # 初始化 GPU 资源
            if use_cos:
                self.faiss_index_1 = faiss.GpuIndexFlatIP(res,dim) 
                self.faiss_index_2 = faiss.GpuIndexFlatIP(res,dim) 
            else:
                self.faiss_index_1 = faiss.GpuIndexFlatL2(res,dim) 
                self.faiss_index_2 = faiss.GpuIndexFlatL2(res,dim)  
        else:
            if use_cos:
                
                if faster_index[0]:
                    quantizer = faiss.IndexFlatIP(dim) 

                    # 使用上述 quantizer 创建 IVF flat 索引
                    self.faiss_index_1 = faiss.IndexIVFFlat(quantizer, dim, faster_index[1], faiss.METRIC_INNER_PRODUCT)
                    self.faiss_index_2 = faiss.IndexIVFFlat(quantizer, dim, faster_index[1], faiss.METRIC_INNER_PRODUCT)
                else:
                    self.faiss_index_1 = faiss.IndexFlatIP(dim) 
                    self.faiss_index_2 = faiss.IndexFlatIP(dim) 
                
            else:
                self.faiss_index_1 = faiss.IndexFlatL2(dim) 
                self.faiss_index_2 = faiss.IndexFlatL2(dim)  
        
        # 构建特征库
        self.extract_features(self.load_and_stack_images() if imgs is None else imgs)
        
   
        
        
        
        
    def load_and_stack_images(self):
        """
        加载所有图片并转换为张量，然后堆叠为一个批量张量
        """
        if self.use_json:
            self.info(f"Dataset:{self.dataset}, Loading images for category: {self.target_cls}, use_json = {True}, shot = {self.shot}")        
            # 读取 JSON 文件
            with open(self.dataset_json_path, 'r') as f:
                cls_datas = json.load(f)['train'][self.target_cls]
                cls_datas = sorted(cls_datas, key=lambda x: x['img_path'])
            # 随机选择指定数量的图片
            random.seed(args.seed)
            image_paths = [os.path.join(dataset_path,dic['img_path']) for dic in random.sample(cls_datas, self.shot)]
        else:
            # yaml版本                  
            with open(self.yaml_path, 'r') as f:
                image_paths = yaml.safe_load(f)[self.dataset][self.target_cls]
            
            self.info(f"Dataset:{self.dataset}, Loading images for category: {self.target_cls}, use_yaml = {True}, shot = {len(image_paths)}") 
        
        all_images = []

        for path in image_paths:
            # 加载图片并转换
            self.info(f"Loading image: {path}")
            img = Image.open(path).convert("RGB")
            img_tensor = self.test_transforms(img)  # 转换为张量
            all_images.append(img_tensor)
        
        # 应用数据增强
        if self.use_augment:
            if self.augment_cache_dir:
                info(f"Using cached augmenting images for category: {self.target_cls},cache dir: {self.augment_cache_dir}")     
                augmented_paths = glob.glob(os.path.join(self.augment_cache_dir,"*.png"))
            else:
                info(f"Augmenting images for category: {self.target_cls},n={n},m={m},save_dir={args.augment_save_dir}")
                augmented_paths = augment_images(image_paths, n=n, m=m, save_dir=args.augment_save_dir)
            for path in augmented_paths:
                img = Image.open(path).convert("RGB")
                img_tensor = self.test_transforms(img)
                all_images.append(img_tensor)
        # 堆叠所有图片张量为一个批量张量
        stacked_images = torch.stack(all_images, dim=0)  # 形状：[N, C, H, W]
        self.info(f"loading images finshed, shape: {stacked_images.shape}")
        return stacked_images
    
    def avg_pool(self, feature_map, kernel_size=3):
        """
        使用自适应平均池化对特征图的每个位置进行 3x3 patch 平均处理，并确保输出尺寸与输入尺寸一致
        :param feature_map: [B, C, H, W] 的特征张量
        :param kernel_size: 滑动窗口的大小
        :return: 处理后的特征张量
        """
        # 计算填充大小
        padding = kernel_size // 2

        # 对特征图进行填充，避免边界问题
        feature_map_padded = F1.pad(feature_map, (padding, padding, padding, padding), mode='reflect')

        # 使用自适应平均池化，输出尺寸与输入特征图相同
        pool = nn.AdaptiveAvgPool2d(output_size=feature_map.shape[2:])  # 输出大小与输入特征图相同
        return pool(feature_map_padded)
    
    def extract_features(self, stacked_images):
        """
        从堆叠的图片张量中提取特征
        """
        self.info("Extracting pretrained features from images...")
        # 提取特征
        with torch.no_grad():
            features = extract_fused_features(self.model_t,stacked_images)  # 提取2个阶段的特征
            # features = [self.avg_pool(feature) for feature in features]
            features = [F1.normalize(rearrange(feature,'b l c->b c l')) for feature in features if self.use_cos]
            

        # 展平每个阶段的特征为 # [B, C, H, W] -> [B*H*W, D]，并添加到 faiss 索引
        features_1_flat = rearrange(features[0],'b c l->(b l) c').cpu().contiguous().numpy()  
        features_2_flat = rearrange(features[1],'b c l->(b l) c').cpu().contiguous().numpy()
        
        # 倒排时先训练索引
        if self.faster_index[0]:
            self.info(f"Training faiss index 1 with {self.faster_index[1]} clusters...")
            self.faiss_index_1.train(features_1_flat)
            self.info(f"Training faiss index 2 with {self.faster_index[1]} clusters...")
            self.faiss_index_2.train(features_2_flat)
        
        self.faiss_index_1.add(features_1_flat)
        self.faiss_index_2.add(features_2_flat)
       
        # 将特征分别存储到对应的特征库
        self.info(f'feature stage-1\'s shape：{features_1_flat.shape}')
        self.info(f'feature stage-2\'s shape：{features_2_flat.shape}')
        self.info(f"Building memory bank finished.")
        
    def query_image(self, query_image_path):
        """
        给定查询图片，提取其特征并从特征库中找到最接近的特征
        """
        # 加载并转换查询图片
        query_img = Image.open(query_image_path).convert("RGB")
        query_tensor = self.test_transforms(query_img).unsqueeze(0)  # 添加 batch 维度
        #1chw
        size = query_tensor.shape[-2:]
        
        # 提取查询图片的特征
        with torch.no_grad():
            query_features = self.model_t(query_tensor)
            # query_features = [self.avg_pool(feature) for feature in query_features]
            query_features = [F1.normalize(feature) for feature in query_features if self.use_cos]
            

        # 计算与特征库中每个特征的相似度
        closest_features = []
        for i, (query_feature, faiss_index) in enumerate(zip(query_features, 
                                                              [self.faiss_index_1, self.faiss_index_2, self.faiss_index_3])):
            query_feature_flat = query_feature.permute(0, 2, 3, 1).reshape(-1, query_feature.shape[1]).cpu().contiguous().numpy()
            distances, indices = faiss_index.search(query_feature_flat, k=1)  # 检索每个 patch 的最近邻
            if self.use_cos:
                distances = 1 - distances
            closest_features.append((distances, indices))
        
        # [FIX-3] 同 query_batch：改用 [0,0] 切片，并按真实 patch 网格 reshape（原实现硬编码 64/32/16 且用 .squeeze()）
        _g = size[0] // 14
        stage1 = F1.interpolate(torch.tensor(closest_features[0][0]).reshape(1,1,_g,_g),size,mode='bilinear', align_corners=True)[0,0].numpy()
        stage2 = F1.interpolate(torch.tensor(closest_features[1][0]).reshape(1,1,_g,_g),size,mode='bilinear', align_corners=True)[0,0].numpy()
        stage3 = F1.interpolate(torch.tensor(closest_features[2][0]).reshape(1,1,_g,_g),size,mode='bilinear', align_corners=True)[0,0].numpy()
        stage4 = (stage1 + stage2 + stage3) / 9
        stage1 /= stage1.max()
        stage2 /= stage2.max() 
        stage3 /= stage3.max() 
        stage4 /= stage4.max() 
        if self.gussian_sigma:
            stage1 = gaussian_filter(stage1, sigma=self.gussian_sigma)
            stage2 = gaussian_filter(stage2, sigma=self.gussian_sigma)
            stage3 = gaussian_filter(stage3, sigma=self.gussian_sigma)
            stage4 = gaussian_filter(stage4, sigma=self.gussian_sigma)
    
        
        return [stage1,stage2,stage3,stage4]
        
    def query_batch(self, test_data, return_stages=False):
        size = test_data.shape[-2:]
        batch_size = test_data.shape[0]
        # 提取查询图片的特征
        with torch.no_grad():
            query_features = extract_fused_features(self.model_t,test_data)
            # query_features = [self.avg_pool(feature) for feature in query_features]
            query_features = [F1.normalize(rearrange(feature,'b l c->b c l')) for feature in query_features if self.use_cos]

        # [主干实验] 同 code_vpt2 副本：patch 数不再硬编码 size//14，改从特征长度反推网格
        n_tok = int(query_features[0].shape[-1])
        patch_size = int(round(n_tok ** 0.5))
        assert patch_size * patch_size == n_tok, (n_tok, patch_size)
            
            

        # 计算与特征库中每个特征的相似度
        closest_features = []
        for i, (query_feature, faiss_index) in enumerate(zip(query_features, 
                                                              [self.faiss_index_1, self.faiss_index_2])):
            query_feature_flat = rearrange(query_feature,'b c l->(b l) c').cpu().contiguous().numpy()
            distances, indices = faiss_index.search(query_feature_flat, k=1)  # 检索每个 patch 的最近邻
            if self.use_cos:
                distances = 1 - distances
            closest_features.append((distances, indices))
        err1 = torch.tensor(closest_features[0][0].reshape(batch_size,patch_size**2,1).reshape(batch_size,1,patch_size,patch_size))
        err2 = torch.tensor(closest_features[1][0].reshape(batch_size,patch_size**2,1).reshape(batch_size,1,patch_size,patch_size))
        # [FIX-3] 不能用 .squeeze()：最后一个 batch 只有 1 张图时 (1,1,H,W) 会被压成 (H,W)，
        # 下游 cat/unsqueeze 维度不一致直接崩（MVTec carpet/tile N=117、batch=4 时必现）。
        stage1 = F1.interpolate(err1, size, mode='bilinear', align_corners=True)[:, 0].numpy()
        stage2 = F1.interpolate(err2, size, mode='bilinear', align_corners=True)[:, 0].numpy()
        err_map = (stage1 + stage2 ) / 2
        
        if return_stages:
            return stage1, stage2, err_map
        return err_map
    
    def evaluation(self,tar_cls_name='bottle'):
        config = cfg()
        
        evaluator = get_evaluator(config.evaluator)
        train_loader, test_loader = get_loader(config)
        imgs_masks, anomaly_maps, cls_names, anomalys, img_list, sample_predicts = [], [], [], [], [], []
        
        loader = create_single_class_dataloader(test_loader, target_class=tar_cls_name)
        for test_data in tqdm(loader,desc='Evaluating'):
            imgs = test_data['img']
            imgs_mask = test_data['img_mask']
            cls_name = test_data['cls_name']
            anomaly = test_data['anomaly']
            imgs_mask[imgs_mask > 0.5], imgs_mask[imgs_mask <= 0.5] = 1, 0
            anomaly_map = self.query_batch(imgs)            
            
            img_list.append(imgs.cpu().numpy().astype(float))
            imgs_masks.append(imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            cls_names.append(np.array(cls_name))
            anomalys.append(anomaly.cpu().numpy().astype(int))
        
        results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys, img_list=img_list)
        results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
       
            
        metric_results = evaluator.run(results, tar_cls_name)
        return metric_results,results


def generate_markdown_table(result_cls):
    """
    将结果字典转换为 Markdown 表格格式，并添加每个指标的平均值。
    
    参数：
        result_cls (dict): 类别结果字典，格式为 {cls: {metric: value, ...}, ...}
    
    返回：
        str: Markdown 表格字符串
    """
    # 获取所有评估指标
    metrics = list(next(iter(result_cls.values())).keys())
    
    # 构建表头
    header = "| Class | " + " | ".join(metrics) + " |"
    separator = "|:----:" + "|:----:" * len(metrics) + "|"
    
    # 构建表格内容
    rows = []
    metric_sums = {metric: 0 for metric in metrics}  # 用于计算每个指标的总和
    for cls, metrics_values in result_cls.items():
        row = f"| {cls} | " + " | ".join(f"{100 * metrics_values[metric]:.1f}" for metric in metrics) + " |"
        rows.append(row)
        for metric in metrics:
            metric_sums[metric] += metrics_values[metric]
    
    # 计算每个指标的平均值
    num_classes = len(result_cls)
    avg_row = "| Average | " + " | ".join(f"{100 * (metric_sums[metric] / num_classes):.1f}" for metric in metrics) + " |"
    
    # 合并表头、内容和平均值
    markdown_table = "\n".join([header, separator] + rows + [avg_row])
    return markdown_table

def denormalize(img):
    """
    将标准化的张量还原为原始图像。
    
    参数：
        img (Tensor): 标准化的图像张量，形状为 [C, H, W]。
        mean (list): 标准化时使用的均值。
        std (list): 标准化时使用的标准差。
    
    返回：
        Tensor: 还原后的图像张量。
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return img * std + mean

def visualize_and_save(imgs, masks, anomaly_maps,cls_name, save_dir='logs/vis'):
    """
    可视化图片、mask 和异常图，并保存到指定路径。
    
    参数：
        imgs (Tensor): 原始图片张量，形状为 [N, C, H, W]。
        masks (numpy.ndarray): Mask 数组，形状为 [N, H, W]。
        anomaly_maps (numpy.ndarray): 异常图数组，形状为 [N, H, W]。
        save_dir (str): 保存路径。
        cls_name (str): 当前类别名称。
    """
    cls_dir = os.path.join(save_dir, cls_name)
    os.makedirs(cls_dir, exist_ok=True)  

    for i in tqdm(range(len(imgs)),desc='Visualizing'):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 还原图片为 RGB 格式
        img_denormalized = denormalize(torch.tensor(imgs[i]))
        img_rgb = to_pil_image(img_denormalized)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Image")
        axes[0].axis("off")
        
        # Mask
        axes[1].imshow(masks[i].squeeze(), cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")
        
        # 异常图
        anomaly_map = cm.jet(anomaly_maps[i]/anomaly_maps[i].max())
        anomaly_map = (anomaly_map[:, :, :3] * 255).astype(np.uint8)  # 转换为 RGB 格式
        anomaly_map = Image.fromarray(anomaly_map)  # 转换为 PIL 图像
        anomaly_map = Image.blend(img_rgb, anomaly_map, alpha=0.4)
        axes[2].imshow(anomaly_map)
        axes[2].set_title("Anomaly Map")
        axes[2].axis("off")
        
        # 保存图片
        save_path = os.path.join(cls_dir, f"{i}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    info(f"Visualization saved to {cls_dir}")

def set_seed(seed):
    """
    设置随机种子以确保结果的可重复性。
    
    参数：
        seed (int): 随机种子值。
    """
    random.seed(seed)  # Python 内置随机数生成器
    np.random.seed(seed)  # NumPy 随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU 随机数生成器
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数生成器
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True  # 确保每次运行的结果一致
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 的自动优化

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, default='/home/jjquan/datasets/mvtec')
    parser.add_argument("--cls", type=str, default='toothbrush')
    parser.add_argument("--use_json", type=bool, default=True)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--use_augment", type=bool, default=True)
    parser.add_argument("--augment_save_dir", type=str, default='./logs_5.15_finetune/augments')
    parser.add_argument("--augment_cache_dir", type=str, default=None)
    #'/home/jjquan/codebase/OmniADBank/logs/mvtec/augments/capsule/148'
    parser.add_argument("--use_finetune", action='store_true')
    parser.add_argument("--pth_path", type=str, default='/home/jjquan/codebase/OmniADBank/logs5.15_w0.3/pths/epoch_3.pth')
    
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--base_angle", type=int, default=15)
    parser.add_argument("--vis", type=bool, default=False)
    parser.add_argument("--vis_dir", type=str, default='logs_5.15_finetune/vis')
    parser.add_argument("--cls_list", type=list, default=['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'])
    parser.add_argument("--log_path", type=str, default='logs_5.15_finetune/memory_bank.log')
    parser.add_argument("--result_path", type=str, default='logs_5.15_finetune/results.md')
   
    args = parser.parse_args()
    
    info = setup_logger(log_file=args.log_path).info
    
    dataset_path = args.dataset_path
    use_json,shot = args.use_json,args.shot
    # 数据增强参数
    n,m,seed,base_angle = args.n,args.m,args.seed,args.base_angle
    set_seed(seed)
    info(f"random seed: {seed}")
    # 可视化参数
    vis,vis_dir = args.vis, args.vis_dir
    clses = args.cls_list
    
    result_cls = {}
    for cls in clses:
        if  args.cls is not None and cls != args.cls:
            continue
        model1 = None
        augment_cache_dir = args.augment_cache_dir
        augment_cache_dir = os.path.join('/home/jjquan/codebase/OmniADBank/logs_5.15_finetune/augments', cls)
        # 匹配所有子目录
        augment_cache_dirs = [d for d in glob.glob(os.path.join(augment_cache_dir, '*')) if os.path.isdir(d)]
        assert len(augment_cache_dirs) == 1, f"Expected exactly one subdirectory in {augment_cache_dir}, but found {len(augment_cache_dirs)}"
        augment_cache_dir = augment_cache_dirs[0]
        pth_path = args.pth_path
        if args.use_finetune:
            model1 = AnomalyDINO()
            # pth_path = os.path.join('/home/jjquan/codebase/OmniADBank/logs_5.23_finetune/finetune_logs', cls, 'epoch_1.pth')
            pth_path = '/home/jjquan/codebase/OmniADBank/logs5.23_w0.1/pths/epoch_20.pth'
            pth = torch.load(pth_path)
            model1.load_state_dict(pth)
            model1.eval()
            augment_cache_dir = os.path.join('/home/jjquan/codebase/OmniADBank/logs_5.15_finetune/augments', cls)
            # 匹配所有子目录
            augment_cache_dirs = [d for d in glob.glob(os.path.join(augment_cache_dir, '*')) if os.path.isdir(d)]
            assert len(augment_cache_dirs) == 1, f"Expected exactly one subdirectory in {augment_cache_dir}, but found {len(augment_cache_dirs)}"
            augment_cache_dir = augment_cache_dirs[0]
            info(f"Using finetune model for {cls} from {pth_path}")
        memory_bank = MemoryBank(cls,use_json=use_json,shot=shot,model=model1,augment_cache_dir=augment_cache_dir)
        metric_results,data_results = memory_bank.evaluation(cls)
        result_cls[cls] = metric_results
        info(f"Evaluation {cls}: {metric_results}")
        if vis:
            visualize_and_save(data_results['img_list'], data_results['imgs_masks'], data_results['anomaly_maps'], cls, save_dir=vis_dir)
           
        info(f"====================================================================================================================")

    
    markdown_table = generate_markdown_table(result_cls)
    # 保存到文件
    markdown_file_path = args.result_path
    with open(markdown_file_path, "a") as f:
        f.write("\n\n"+markdown_table)

    info(f"Markdown table saved to {markdown_file_path}")
