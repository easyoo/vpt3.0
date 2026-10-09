import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    
        
        
        

# # --- renet使用示例 ---
# if __name__ == "__main__":
#     # 定义变换
#     data_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     # 初始化 Dataset
#     # 假设 root='data', cls_name='hazelnut', 每个异常类抽 5 个
#     dataset = AnomalySyntheticDataset(
#         root='/home/jjquan/datasets/mvtec_sys_anomaly/sdas', 
#         cls_name='hazelnut', 
#         num_per_rank=5, 
#         transform=data_transform
#     )

#     # 配合 DataLoader 使用
#     from torch.utils.data import DataLoader
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#     for imgs, labels in dataloader:
#         print(imgs.shape, labels)
#         break






import os
import random
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class GeneratedAnomalyDataset(Dataset):
    """
    PyTorch Dataset for the generated anomaly dataset.
    
    The dataset structure is:
    - dataset_root/
        - category1/ (e.g., bottle/)
            - anomaly_type1/ (e.g., broken_large/)
                - image/ (anomalous images)
                - mask/ (anomaly masks)
                - ori/ (original normal images)
            - anomaly_type2/...
        - category2/...
    """
    
    def __init__(
        self,
        root_dir: str,
        category: str = None,  # Parameter to specify category
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_original: bool = False,
        image_suffix: str = ".jpg",
        samples_per_anomaly_type: int = None  # Parameter for random sampling
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            category: Specific category to load (e.g., 'bottle', 'cable', etc.). 
                     If None, loads all categories.
            transform: Transform applied to images
            target_transform: Transform applied to masks
            return_original: Whether to return original normal images as well
            image_suffix: Suffix of image files (default .jpg)
            samples_per_anomaly_type: Number of samples to randomly select from each anomaly type.
                                     If None, loads all samples.
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.transform = transform
        self.target_transform = target_transform
        self.return_original = return_original
        self.image_suffix = image_suffix
        self.samples_per_anomaly_type = samples_per_anomaly_type
        
        # Collect all data samples
        self.samples = []
        self._build_samples()
        
    def _build_samples(self):
        """Build the list of samples by traversing the dataset structure."""
        # If category is specified, only process that category directory
        if self.category is not None:
            category_dir = self.root_dir / self.category
            if category_dir.is_dir():
                category_dirs = [category_dir]
            else:
                raise ValueError(f"Category '{self.category}' not found in root directory '{self.root_dir}'")
        else:
            category_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        for category_dir in category_dirs:
            if not category_dir.is_dir():
                continue
                
            for anomaly_type_dir in category_dir.iterdir():
                if not anomaly_type_dir.is_dir():
                    continue
                
                image_dir = anomaly_type_dir / "image"
                mask_dir = anomaly_type_dir / "mask"
                
                # Check if required directories exist
                if not image_dir.exists() or not mask_dir.exists():
                    continue
                
                # Get all image files and match with masks
                anomaly_samples = []
                for img_file in image_dir.glob(f"*{self.image_suffix}"):
                    mask_file = mask_dir / img_file.name
                    
                    if mask_file.exists():
                        sample = {
                            "image_path": str(img_file),
                            "mask_path": str(mask_file),
                        }
                        
                        if self.return_original:
                            ori_dir = anomaly_type_dir / "ori"
                            if ori_dir.exists():
                                # Use same filename for original image
                                ori_file = ori_dir / img_file.name
                                if ori_file.exists():
                                    sample["ori_path"] = str(ori_file)
                        
                        anomaly_samples.append(sample)
                
                # If sampling is enabled and there are more samples than required, randomly select
                if self.samples_per_anomaly_type and len(anomaly_samples) > self.samples_per_anomaly_type:
                    selected_samples = random.sample(anomaly_samples, self.samples_per_anomaly_type)
                    self.samples.extend(selected_samples)
                    print(f"Selected {len(selected_samples)} samples from {anomaly_type_dir.name} in {category_dir.name}")
                else:
                    self.samples.extend(anomaly_samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            A dictionary containing:
            - 'image': The anomalous image (PIL Image or transformed tensor)
            - 'mask': The anomaly mask (PIL Image or transformed tensor)
            - 'original': The original normal image (if return_original=True)
            - 'category': The category name (e.g., 'bottle')
            - 'anomaly_type': The anomaly type (e.g., 'broken_large')
        """
        sample = self.samples[idx]
        
        # Load image and mask
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")  # Grayscale from actual mask file
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        result = {
            "img": image,
            "img_mask": mask,
            "category": Path(sample["image_path"]).parent.parent.parent.name,
            "anomaly_type": Path(sample["image_path"]).parent.parent.name,
        }
        
        # Add original image if requested
        if self.return_original and "ori_path" in sample:
            original = Image.open(sample["ori_path"]).convert("RGB")
            if self.transform:
                original = self.transform(original)
            result["original"] = original
        
        return result


def get_anomaly_categories(root_dir: str) -> list:
    """
    Helper function to get all anomaly categories in the dataset.
    
    Args:
        root_dir: Root directory of the dataset
        
    Returns:
        List of category names
    """
    root_path = Path(root_dir)
    categories = []
    for item in root_path.iterdir():
        if item.is_dir():
            categories.append(item.name)
    return sorted(categories)


def get_anomaly_types(root_dir: str, category: str) -> list:
    """
    Helper function to get all anomaly types for a specific category.
    
    Args:
        root_dir: Root directory of the dataset
        category: Category name (e.g., 'bottle')
        
    Returns:
        List of anomaly type names for the category
    """
    category_path = Path(root_dir) / category
    anomaly_types = []
    for item in category_path.iterdir():
        if item.is_dir():
            anomaly_types.append(item.name)
    return sorted(anomaly_types)





class AnoStylerGeneratedDataset(Dataset):
    """
    PyTorch Dataset for AnoStyler generated anomaly dataset.
    
    The dataset structure is:
    - dataset_root/
        - category1/ (e.g., carpet/)
            - anomaly_type1/ (e.g., hole、cut、thread、defect0/)
                - image/ (anomalous images)
                - mask/ (anomaly masks)
            - anomaly_type2/...
        - category2/...
    """
    
    def __init__(
        self,
        root_dir: str,
        category: str = None,  # Parameter to specify category
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_original: bool = False,
        image_suffix: str = ".png",
        samples_per_anomaly_type: int = None,  # Parameter for random sampling
        shuffle_samples: bool = True  # Whether to shuffle samples when loading
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            category: Specific category to load (e.g., 'carpet', 'screw', etc.). 
                     If None, loads all categories.
            transform: Transform applied to images
            target_transform: Transform applied to masks
            return_original: Whether to return original normal images as well (not applicable for this dataset)
            image_suffix: Suffix of image files (default .jpg)
            samples_per_anomaly_type: Number of samples to randomly select from each anomaly type.
                                     If None, loads all samples.
            shuffle_samples: Whether to shuffle the dataset samples
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.transform = transform
        self.target_transform = target_transform
        self.return_original = return_original
        self.image_suffix = image_suffix
        self.samples_per_anomaly_type = samples_per_anomaly_type
        self.shuffle_samples = shuffle_samples
        
        # Collect all data samples
        self.samples = []
        self._build_samples()
        
        # Shuffle samples if required
        if self.shuffle_samples:
            random.shuffle(self.samples)
        
    def _build_samples(self):
        """Build the list of samples by traversing the dataset structure."""
        # If category is specified, only process that category directory
        if self.category is not None:
            category_dir = self.root_dir / self.category
            if category_dir.is_dir():
                category_dirs = [category_dir]
            else:
                raise ValueError(f"Category '{self.category}' not found in root directory '{self.root_dir}'")
        else:
            # Skip .DS_Store and other non-directory items
            category_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and d.name != '.DS_Store']
        
        for category_dir in category_dirs:
            if not category_dir.is_dir():
                continue
                
            for anomaly_type_dir in category_dir.iterdir():
                if not anomaly_type_dir.is_dir():
                    continue
                
                image_dir = anomaly_type_dir / "image"
                mask_dir = anomaly_type_dir / "mask"
                
                # Check if required directories exist
                if not image_dir.exists() or not mask_dir.exists():
                    continue
                
                # Get all image files and match with masks
                anomaly_samples = []
                for img_file in image_dir.glob(f"*{self.image_suffix}"):
                    # Find corresponding mask file (replace 'gen_ano' with 'gen_mask' in filename)
                    mask_filename = img_file.name.replace('gen_ano', 'gen_mask')
                    mask_file = mask_dir / mask_filename
                    
                    # If the exact replacement doesn't work, look for any mask file with the same base name
                    if not mask_file.exists():
                        # Try to find mask with the same numeric suffix
                        stem = img_file.stem  # e.g., 'gen_ano_0'
                        # Extract number from filename like gen_ano_0 -> 0
                        import re
                        num_match = re.search(r'(\d+)$', stem)
                        if num_match:
                            num = num_match.group(0)
                            # Look for mask files with the same number
                            for potential_mask in mask_dir.glob(f"*{num}{self.image_suffix}"):
                                mask_file = potential_mask
                                break
                    
                    if mask_file.exists():
                        sample = {
                            "image_path": str(img_file),
                            "mask_path": str(mask_file),
                        }
                        
                        anomaly_samples.append(sample)
                
                # If sampling is enabled and there are more samples than required, randomly select
                if self.samples_per_anomaly_type and len(anomaly_samples) > self.samples_per_anomaly_type:
                    selected_samples = random.sample(anomaly_samples, self.samples_per_anomaly_type)
                    self.samples.extend(selected_samples)
                    print(f"Selected {len(selected_samples)} samples from {anomaly_type_dir.name} in {category_dir.name}")
                else:
                    self.samples.extend(anomaly_samples)
                    
        print(f"Total samples loaded: {len(self.samples)} from {len(category_dirs)} categories")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            A dictionary containing:
            - 'img': The anomalous image (PIL Image or transformed tensor)
            - 'img_mask': The anomaly mask (PIL Image or transformed tensor)
            - 'category': The category name (e.g., 'carpet')
            - 'anomaly_type': The anomaly type (e.g., 'hole、cut、thread、defect0')
        """
        sample = self.samples[idx]
        
        # Load image and mask
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")  # Grayscale
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        result = {
            "img": image,
            "img_mask": mask,
            "category": Path(sample["image_path"]).parent.parent.parent.name,
            "anomaly_type": Path(sample["image_path"]).parent.parent.name,
        }
        
        return result





# anomalydiffusion 使用
if __name__ == "__main__":
    # Define transforms
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # 1. Define transforms for images and masks
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # # 2. Load a specific category with random sampling (e.g., 10 samples per anomaly type)
    # dataset_sampled = GeneratedAnomalyDataset(
    #     root_dir="/home/jjquan/datasets/generated_dataset",
    #     category="bottle",  # Only load data from the 'bottle' directory
    #     transform=transform,
    #     target_transform=target_transform,
    #     return_original=True,  # This will include original images if available
    #     samples_per_anomaly_type=30  # Randomly select 10 samples from each anomaly type
    # )
    
    
    dataset_sampled = AnoStylerGeneratedDataset(
        root_dir="/home/jjquan/experiment/AnoStyler/genbyDTD",
        category="screw",  # Only load data from the 'carpet' directory
        transform=transform,
        target_transform=target_transform,
        samples_per_anomaly_type=35  # Randomly select 10 samples from each anomaly type
    )
    
    print(f"Total samples in sampled bottle dataset: {len(dataset_sampled)}")

    sample=dataset_sampled[10]
    image = sample['img']
    mask = sample['img_mask']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)

    # Convert to numpy for plotting
    image_np = image_denorm.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()  # Remove channel dimension if present

    # Create subplots to show image and mask side by side
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot image
    ax1.imshow(image_np)
    ax1.set_title(f"Image - {sample['category']}/{sample['anomaly_type']}")
    ax1.axis('off')

    # Plot mask
    ax2.imshow(mask_np, cmap='gray')
    ax2.set_title(f"Mask - {sample['category']}/{sample['anomaly_type']}")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()