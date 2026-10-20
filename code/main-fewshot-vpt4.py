import warnings
import hashlib          # [FIX-4] 样本清单哈希（MVTecDataset.list_hash）
warnings.filterwarnings("ignore")

from dinov2.models import vision_transformer
import torch
import random


####################################################
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from torchvision.datasets import ImageFolder
import logging

from util.vis import *
###########################
import numpy as np
import torch
import torch.nn.functional as F
import kornia as K
from torch.utils.data import DataLoader, ConcatDataset

from ucc.abnormal_extractor import anomaly_feature

def embedding_concat(x, y, use_cuda):
    device = torch.device('cuda' if use_cuda else 'cpu')
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def mahalanobis_torch(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x


def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x

def grey_img(x):
    x = K.color.rgb_to_grayscale(x)
    x = x.repeat(1, 3, 1,1)
    return x


def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


###########################

def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
        # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
        # hflip img
    flipped_img = hflip_img(img)
    augment_img = torch.cat([augment_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(img)
    augment_img = torch.cat([augment_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(img, angle)
        augment_img = torch.cat([augment_img, rotate90_img], dim=0)
    augment_img = (augment_img[torch.randperm(augment_img.size(0))])
    return augment_img

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# [FIX-4] 样本顺序确定性：原先缺陷目录用 os.listdir（文件系统顺序，跨机器不可复现）、
# good 分支的 glob 结果也未排序，导致"异常图 ↔ GT"的对齐只依赖同一次 listdir 顺序。
# 现改为可排序（sort_items=True 为新默认），并暴露 keys/list_hash 供落盘与校验。
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, sort_items=True):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.sort_items = bool(sort_items)
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0
        # [FIX-4] 有序样本键（缺陷类型/文件名）与短哈希：落盘到异常图旁，供后续对齐/校验
        self.keys = [f'{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}'
                     for p in self.img_paths]
        self.list_hash = hashlib.sha1('\n'.join(self.keys).encode()).hexdigest()[:12]

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        if self.sort_items:
            defect_types = sorted(defect_types)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                if self.sort_items:
                    img_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger

####################################################
"""
forked from https://github.com/pytorch/pytorch/blob/master/torch/optim/adamw.py
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class StableAdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_threshold: float = 1.0
                 ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, clip_threshold=clip_threshold
                        )
        super(StableAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(StableAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    # lr_scale = torch.rsqrt(max_exp_avg_sq + 1e-16).mul_(grad)

                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    # lr_scale = torch.rsqrt(exp_avg_sq + 1e-16).mul_(grad)

                lr_scale = grad / denom
                lr_scale = max(1.0, self._rms(lr_scale) / group["clip_threshold"])

                step_size = group['lr'] / bias_correction1 / (lr_scale)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

#################################################
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score
from adeval import EvalAccumulatorCuda


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, use_metrics = ['I-AUROC', 'I-AP', 'I-F1_max','P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)

    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()

    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))

    # for i in range(torch.tensor(pr_px).size(0)):
    #     accum.add_image(torch.tensor(pr_sp[i]), torch.tensor(gt_sp[i]))

    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())




def compute_anomaly_scores(patches, prototypes,out_size):
    """[AUDIT-FIX-5] 向量化实现（与 sklearn 版数值等价，但快很多）。

    原实现按样本循环调用 sklearn.metrics.cosine_distances，并反复 numpy<->cuda 拷贝，
    是整个评估阶段的主要瓶颈（每张图一次 Python 循环 + 一次 CPU->GPU 往返）。
    这里改成：余弦距离 = 1 - 余弦相似度，用一次 bmm 算完整个 batch；
    语义保持不变：取"到最近 prototype 的最小距离"，上采样 align_corners=True。
    patches: [B, N, D]；prototypes: [B, K, D]；返回 [B, 1, H, W]
    """
    out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
    p = torch.as_tensor(patches, dtype=torch.float32, device=device)
    q = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
    p = F.normalize(p, p=2, dim=-1)
    q = F.normalize(q, p=2, dim=-1)
    sim = torch.bmm(p, q.transpose(1, 2))                       # [B, N, K]
    score = (1.0 - sim).clamp_min(0.0).min(dim=-1).values       # [B, N]
    n = score.shape[1]
    g = int(round(n ** 0.5))
    am = score.reshape(-1, 1, g, g).contiguous()
    am = F.interpolate(am, size=out_size, mode='bilinear', align_corners=True)
    return am


def min_max_norm_map(anomaly_map):
    # anomaly_map: (b, 1, h, w)
    b, c, h, w = anomaly_map.shape
    anomaly_map = anomaly_map.view(b, -1)  # 展平成 (b, h*w)
    min_val = anomaly_map.min(dim=1, keepdim=True)[0]
    max_val = anomaly_map.max(dim=1, keepdim=True)[0]
    norm_map = (anomaly_map - min_val) / (max_val - min_val + 1e-8)
    norm_map = norm_map.view(b, 1, h, w)
    return norm_map

def evaluation_batch0(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img,is_training=True)
            patch_feat_layer, feat_list = output['patch_feat_layer'], output['feat_list'] # b 4 c || b l c

            fused_layer = args.fused_layer
            fused_learnable_layer = args.fused_layer_learnable

            fused_patch_token_list = [patch_feat_layer[i] for i in fused_layer]
            fused_patch_token = torch.sum(torch.stack(fused_patch_token_list,dim=0), dim=0)

            fused_learnable_token_list = [feat_list[i] for i in fused_learnable_layer]
            fused_learnable_token = torch.sum(torch.stack(fused_learnable_token_list,dim=0), dim=0)

            patch_mapping = model.token_mapping(fused_patch_token)
            learnable_mapping = model.token_mapping(fused_learnable_token)
            error_map = segmentation_head(patch_mapping)
            B,L,C = error_map.shape
            error_map = error_map.reshape(B,1,int(L**0.5),int(L**0.5)).contiguous()
            error_map = F.interpolate(error_map, size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=True)
            # anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # b 1 392 392
            anomaly_map = compute_anomaly_scores(patch_mapping.cpu().numpy(), learnable_mapping.cpu().numpy(), img.shape[-1])
            anomaly_map = anomaly_map + error_map
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy() # B H W
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy() # B
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy() # B

        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]



def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, visual_bank_map=None,type=1):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        ii=-1
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            ii+=1
            img = img.to(device)
            output = model(img,is_training=True)
            patch_feat_layer, feat_list = output['patch_feat_layer'], output['feat_list'] # b 4 c || b l c

            fused_layer = args.fused_layer
            fused_learnable_layer = args.fused_layer_learnable

            fused_patch_token_list = [patch_feat_layer[i] for i in fused_layer]
            fused_patch_token = torch.sum(torch.stack(fused_patch_token_list,dim=0), dim=0)

            fused_learnable_token_list = [feat_list[i] for i in fused_learnable_layer]
            fused_learnable_token = torch.sum(torch.stack(fused_learnable_token_list,dim=0), dim=0)

            patch_mapping = model.token_mapping(fused_patch_token)
            learnable_mapping = model.token_mapping(fused_learnable_token)
            error_map = segmentation_head(patch_mapping)
            B,L,C = error_map.shape
            error_map = error_map.reshape(B,1,int(L**0.5),int(L**0.5)).contiguous()
            error_map = F.interpolate(error_map, size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=True)
            # anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # b 1 392 392
            anomaly_map = compute_anomaly_scores(patch_mapping.cpu().numpy(), learnable_mapping.cpu().numpy(), img.shape[-1])
            if type==1:
                anomaly_map = anomaly_map
            elif type==2:
                anomaly_map = error_map
            elif type==3:
                anomaly_map = visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            elif type==4:
                anomaly_map = anomaly_map + error_map
            elif type==5:
                anomaly_map = anomaly_map + visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            elif type==6:
                anomaly_map = error_map + visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            elif type==7:
                anomaly_map = anomaly_map + error_map + visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            
            
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy() # B H W
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy() # B
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy() # B

        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


# ============================================================================
#  vpt3.0 新增：改善后 mapping 分支 ⊕ 原版 MemoryBank 分支（全部启用）
#  改善后 mapping = 打分图融合(可学习原型 ⊕ 原始特征空间正常 patch 库) + 1.5× + 水平翻转
#  融合规则      = 两支各自按【类别内】z 标准化后加权相加：w·z(mapping) + (1-w)·z(bank)
#  说明：归一化用整类全部像素的 μ/σ（不是单图），w 默认 0.3（EXPERIMENT_LOG §20 实测最优）
# ============================================================================
def _kcenter_torch(feats, k, seed=0):
    g = torch.Generator(device=feats.device).manual_seed(seed)
    first = int(torch.randint(0, feats.shape[0], (1,), generator=g).item())
    d = 1.0 - feats @ feats[first]
    idx = [first]
    for _ in range(k - 1):
        i = int(torch.argmax(d).item()); idx.append(i)
        d = torch.minimum(d, 1.0 - feats @ feats[i])
    return torch.stack([feats[i] for i in idx], dim=0)


@torch.no_grad()
def build_raw_bank(model, imgs, fused_layer, k=0, chunk=8):
    """原始（未过 token_mapping）特征空间下的正常 patch 库；k=0 表示用全部 patch。"""
    feats = []
    for i in range(0, imgs.shape[0], chunk):
        ret = model(imgs[i:i + chunk].to(next(model.parameters()).device), is_training=True)
        fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in fused_layer], 0), 0)
        feats.append(F.normalize(fp, dim=-1).reshape(-1, fp.shape[-1]))
    bank = torch.cat(feats, 0)
    if k and k < bank.shape[0]:
        bank = _kcenter_torch(bank, k)
    return bank


# [FIX-2] 与生产打分一致的幂次算子：local^a · global^(1-a)
def _pow_geo(local, glob, a):
    e = 1e-6
    return local.clamp_min(e).pow(a) * glob.clamp_min(e).pow(1.0 - a)


def compute_fused_map(model, img, bank, fused_layer, scales=(1.5,), tta_flip=False, w_learn=0.6,
                      use_z=False, raw_alpha=0.5, raw_global=512, sub_seed=1234):
    """改善后 mapping 打分图（未高斯平滑），**与生产口径 eval_score_v3.py 对齐**：
      s_raw   = d1^raw_alpha · q^(1-raw_alpha)   # q = 到正常 patch 库距离分布的 raw_global 分位（1024 子采样）
      s_learn = 到最近可学习原型的距离 dlmin
      m       = w_learn·s_learn + (1-w_learn)·s_raw
    默认 w_learn=0.6、无翻转（= vpt3.7 生产配方）。
    [FIX-2] 旧实现 raw 侧只取 d1（缺 ×q 的几何全局项）、默认 w_learn=0.5 且开翻转，
    导致"训练内打印的指标 ≠ 生产指标"，据此选轮/判断都会走偏。"""
    base_hw = img.shape[-2:]
    q_raw = F.normalize(bank, dim=-1)
    # [FIX-2] 全局项所用的固定子采样（与生产同 seed=1234、同 1024 个 patch）
    _M = q_raw.shape[0]
    _gen = torch.Generator().manual_seed(int(sub_seed))
    sub_idx = torch.randperm(_M, generator=_gen)[:min(1024, _M)].to(q_raw.device)
    # [vpt4.0 · M8] 把 4 张 shot 的原始特征库**过一遍 token_mapping**，作为可学习原型图的额外原型。
    # 动机（实测）：在选定协议 w_learn=0.5 下，原始特征库图（w=0 那一侧）单独就比可学习原型图更强；
    # 可学习图只有 96 个"图像自适应"原型，覆盖面远小于 3136 个真实正常 patch。
    # 把库映射到**同一个学到的度量空间**后并入原型集，等于在"学到的度量"里做 PatchCore：
    # 既有覆盖面、又能被 M5/M7 的像素级损失直接训练。默认关闭（0），开启不影响其它实验。
    _bank_proto = None
    if getattr(args, 'ms_bank_proto', 0):
        _bank_proto = F.normalize(model.token_mapping(bank), dim=-1)      # [M,768]

    def _map28(patch_feat, proto_feat):
        p = F.normalize(model.token_mapping(patch_feat), dim=-1)
        q = F.normalize(model.token_mapping(proto_feat), dim=-1)
        if _bank_proto is not None:
            q = torch.cat([q, _bank_proto.unsqueeze(0).expand(q.shape[0], -1, -1)], dim=1)
        s = (1.0 - torch.bmm(p, q.transpose(1, 2))).clamp_min(0).min(dim=-1).values
        g = int(round(s.shape[1] ** 0.5))
        return s.reshape(-1, 1, g, g)

    fused_sum, n_view = None, 0
    for s in scales:
        v_img = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
        for vi, view in enumerate([v_img, v_img.flip(3)] if tta_flip else [v_img]):
            ret = model(view, is_training=True)
            fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in fused_layer], 0), 0)
            flr = torch.cat(ret['feat_list'], dim=1)   # [MULTI-LAYER] 各层 token 数可不同 → 拼接（layers=1 时与 sum 等价）
            if getattr(args, 'global_proto', 0):     # [M4] 评估端与训练端保持同一套原型集
                _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                flr = torch.cat([flr, _g], dim=1)
            m_learn = _map28(fp, flr)
            p_raw = F.normalize(fp, dim=-1)
            # [FIX-2] raw 侧改为生产口径：d1^a · q^(1-a)（q=到库距离的 raw_global 分位）
            d_all = (1.0 - torch.bmm(p_raw, q_raw.T.unsqueeze(0)
                                     .expand(p_raw.shape[0], -1, -1))).clamp_min(0)
            d1 = d_all.min(dim=-1).values
            d_sub = torch.sort(d_all.index_select(-1, sub_idx), dim=-1).values
            q_sel = d_sub[..., min(int(raw_global), d_sub.shape[-1]) - 1]
            s_b = _pow_geo(d1, q_sel, raw_alpha)
            g = int(round(s_b.shape[1] ** 0.5))
            m_bank = s_b.reshape(-1, 1, g, g)
            if use_z:                      # 逐视图 z（仅在显式要求时启用；默认关闭，与基线协议一致）
                def _z(x):
                    return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-8)
                fused28 = 0.5 * (_z(m_learn) + _z(m_bank))
            else:
                fused28 = w_learn * m_learn + (1.0 - w_learn) * m_bank
            am = F.interpolate(fused28, size=base_hw, mode='bilinear', align_corners=True)
            if vi == 1:
                am = am.flip(3)
            fused_sum = am if fused_sum is None else fused_sum + am
            n_view += 1
    return fused_sum / max(1, n_view)


def evaluation_batch_fuse(model, dataloader, device, bank, max_ratio=0.01, resize_mask=None,
                          w_learn=0.6, scales=(1.5,), tta_flip=False, use_z=False):
    """[mapping-only 协议] 只跑映射分支（**不含 MemoryBank**）：
    逐视图 = w·可学习原型距离图 + (1-w)·原始特征空间正常 patch 库距离图，再对多尺度×翻转视图取平均。
    [FIX-2] 默认与生产 vpt3.7 对齐：w_learn=0.6、无翻转、raw 侧用 d1^a·q^(1-a)。"""
    model.eval()
    gt_list_px, pr_list_px, gt_list_sp, pr_list_sp = [], [], [], []
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, _ in tqdm(dataloader, ncols=80):
            img = img.to(device)
            am = compute_fused_map(model, img, bank, args.fused_layer,
                                   scales=scales, tta_flip=tta_flip, w_learn=w_learn, use_z=use_z)
            if resize_mask is not None:
                am = F.interpolate(am, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            am = gk(am)
            gt_b = (gt > 0.5).float()
            if gt_b.shape[1] > 1:
                gt_b = torch.max(gt_b, dim=1, keepdim=True)[0]
            gt_list_px.append(gt_b)
            pr_list_px.append(am)
            gt_list_sp.append(label)
            flat = am.flatten(1)
            k_top = max(1, int(flat.shape[1] * max_ratio))
            pr_list_sp.append(torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean(dim=1))
    gt_list_px = torch.cat(gt_list_px, 0)[:, 0].cpu().numpy()
    pr_list_px = torch.cat(pr_list_px, 0)[:, 0].cpu().numpy()
    gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
    return list(ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp))


def evaluation_batch_vpt3(model, dataloader, device, visual_bank_map, raw_bank, fused_layer,
                          w=0.5, scales=(1.5,), tta_flip=False, max_ratio=0.01,
                          cache_dir=None, class_name=None):
    """两个分支全启用：同一 DataLoader 顺序 → 逐图严格同一样本；类内 z 后加权融合。
    [FIX-2] 默认 0.5/0.5 且无翻转，与生产 vpt3.7 一致（旧默认 0.3/开翻转）。"""
    model.eval()
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    m_list, gt_list, sp_list = [], [], []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(dataloader, ncols=80):
            img = img.to(device)
            m_list.append(compute_fused_map(model, img, raw_bank, fused_layer,
                                            scales=scales, tta_flip=tta_flip).float().cpu())
            gt_b = (gt > 0.5).float()
            if gt_b.shape[1] > 1:
                gt_b = torch.max(gt_b, dim=1, keepdim=True)[0]
            gt_list.append(gt_b.cpu()); sp_list.append(label)
    M = torch.cat(m_list, 0).to(device)
    B = visual_bank_map
    if B.ndim == 3:
        B = B.unsqueeze(1)
    B = B.to(device).float()
    assert M.shape == B.shape, f'分支形状不一致 {M.shape} vs {B.shape}'
    if cache_dir and class_name:                       # 落盘两个分支的异常图，便于后续换融合方式
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(M.cpu(), os.path.join(cache_dir, f'{class_name}_improved.map'))
        torch.save(B.cpu(), os.path.join(cache_dir, f'{class_name}_bank.map'))
    # === 类内 z 标准化（用整类全部像素的 μ/σ）后加权 ===
    zm = (M - M.mean()) / (M.std() + 1e-8)
    zb = (B - B.mean()) / (B.std() + 1e-8)
    fused = gk(w * zm + (1.0 - w) * zb)
    gt_px = torch.cat(gt_list, 0).to(device)
    flat = fused.flatten(1)
    k_top = max(1, int(flat.shape[1] * max_ratio))
    sp = torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean(dim=1)
    gt_sp = torch.cat(sp_list).flatten().cpu().numpy()
    return list(ader_evaluator(fused[:, 0].cpu().numpy(), sp.cpu().numpy(),
                               gt_px[:, 0].cpu().numpy(), gt_sp))


def evaluation_batch_memory(memory_bank, dataloader, device, max_ratio=0, resize_mask=None):
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            anomaly_map = memory_bank.query_batch(img)
            anomaly_map = torch.tensor(anomaly_map)
            anomaly_map = anomaly_map.unsqueeze(1)
            anomaly_map = anomaly_map.to(device)
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy() # B H W
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy() # B
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy() # B

        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def inference(model, seg_head, dataloader, device="cpu", resize_mask=None, max_ratio=0,vis_dir=None, visual_bank_map=None,type=1):
    model.eval()
    seg_head.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
 
    img_list = []
    gt_list = []
    img_path_list = []
    anomaly_map_list = []
    ii = -1
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            ii += 1
            img = img.to(device)
            output = model(img,is_training=True)
            patch_feat_layer, feat_list = output['patch_feat_layer'], output['feat_list'] # b 4 c || b l c

            fused_layer = args.fused_layer
            fused_learnable_layer = args.fused_layer_learnable

            fused_patch_token_list = [patch_feat_layer[i] for i in fused_layer]
            fused_patch_token = torch.sum(torch.stack(fused_patch_token_list,dim=0), dim=0)

            fused_learnable_token_list = [feat_list[i] for i in fused_learnable_layer]
            fused_learnable_token = torch.sum(torch.stack(fused_learnable_token_list,dim=0), dim=0)

            patch_mapping = model.token_mapping(fused_patch_token)
            learnable_mapping = model.token_mapping(fused_learnable_token)
            error_map = segmentation_head(patch_mapping)
            B,L,C = error_map.shape
            error_map = error_map.reshape(B,1,int(L**0.5),int(L**0.5)).contiguous()
            error_map = F.interpolate(error_map, size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=True)
            # anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # b 1 392 392
            anomaly_map = compute_anomaly_scores(patch_mapping.cpu().numpy(), learnable_mapping.cpu().numpy(), img.shape[-1])
            eps = 1e-9
            if type==1:
                anomaly_map = anomaly_map
            elif type==2:
                anomaly_map = error_map
            elif type==3:
                anomaly_map = visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            elif type==4:
                anomaly_map = anomaly_map + error_map
            elif type==5:
                anomaly_map = anomaly_map + visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            elif type==6:
                anomaly_map = error_map + visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            elif type==7:
                anomaly_map = anomaly_map + error_map + visual_bank_map[ii*img.shape[0]:ii*img.shape[0]+img.shape[0],:,:]
            # 调和平均计算分支
            elif type == 8:
                # 假设这是两个图层的调和平均: 2 / (1/map1 + 1/map2)
                anomaly_map = 2 / (1 / (anomaly_map + eps) + 1 / (error_map + eps))

            elif type == 9:
                # anomaly_map 与 visual_bank_map 的调和平均
                v_map = visual_bank_map[ii*img.shape[0] : ii*img.shape[0]+img.shape[0], :, :]
                anomaly_map = 2 / (1 / (anomaly_map + eps) + 1 / (v_map + eps))

            elif type == 10:
                # error_map 与 visual_bank_map 的调和平均
                v_map = visual_bank_map[ii*img.shape[0] : ii*img.shape[0]+img.shape[0], :, :]
                anomaly_map = 2 / (1 / (error_map + eps) + 1 / (v_map + eps))

            elif type == 11:
                # 三者之间的调和平均: 3 / (1/m1 + 1/m2 + 1/m3)
                v_map = visual_bank_map[ii*img.shape[0] : ii*img.shape[0]+img.shape[0], :, :]
                anomaly_map = 3 / (1 / (anomaly_map + eps) + 1 / (error_map + eps) + 1 / (v_map + eps))
            anomaly_map_t = anomaly_map
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map1 = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map1, dim=1, descending=True)[0][:, :int(anomaly_map1.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            
            pr_list_sp.append(sp_score)
            img_list.append(img)
            gt_list.append(gt)
            img_path_list.extend(img_path)
            anomaly_map_list.append(anomaly_map_t)
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy() # B H W
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy() # B
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy() # B

        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)
        
        imgs = denormalize(torch.cat(img_list, dim=0))
        gts = torch.cat(gt_list, dim=0).cpu().numpy()
        anomaly_maps = global_min_max_norm(torch.cat(anomaly_map_list, dim=0)).cpu().numpy()
        
        if vis_dir is not None:
            for i in range(len(img_path_list)):
                names = img_path_list[i].split('/')
                dataset_name = names[-5]
                cls_name = names[-4]
                anomaly_name = names[-2]
                pic_name = names[-1]
                save_path = os.path.join(vis_dir, dataset_name, cls_name, anomaly_name, pic_name)
                save_img(imgs[i], gts[i], anomaly_maps[i], save_path)
        
        return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]
                
            
import torch
import torch.nn.functional as F

def resize_mask2(mask, num_tokens=784):
    """
    Args:
        mask: [B, 1, 392, 392]
        num_tokens: 784
    Returns:
        mask_flat: [B, 784]
    """
    # 1. 计算网格大小 (sqrt(784) = 28)
    grid_size = int(num_tokens ** 0.5)  # 结果是 28

    # 2. 下采样 (Downsampling)
    # 将 392x392 -> 28x28
    # mode='nearest' 保证 mask 依然是 0 或 1 (二值化)，不会出现 0.5 这种模糊值
    mask_small = F.interpolate(mask.float(), size=(grid_size, grid_size), mode='nearest')
    # 此时 mask_small 形状为 [B, 1, 28, 28]

    # 3. 展平 (Flatten)
    # 将 spatial 维度拉平成 token 序列维度
    # [B, 1, 28, 28] -> [B, 784]
    mask_flat = mask_small.flatten(1)

    return mask_flat


def resize_mask(mask, num_tokens=784):
    """
    Args:
        mask: [B, 1, 392, 392]
        num_tokens: 784 (28x28)
    Returns:
        mask_flat: [B, 784]
    """
    # 1. 计算网格大小
    grid_size = int(num_tokens ** 0.5)  # 28

    # 2. 下采样 (关键修改：使用 Adaptive Max Pool)
    # 不使用 interpolate，改用 max_pool
    # 这能保证只要 patch 内有异常像素，mask 就为 1
    mask_small = F.adaptive_max_pool2d(mask.float(), output_size=(grid_size, grid_size))

    # 3. 展平
    mask_flat = mask_small.flatten(1)

    return mask_flat



import torch.nn as nn




class ScaledPatchCosineLoss3(nn.Module):
    def __init__(self, margin=0.5):
        super(ScaledPatchCosineLoss3, self).__init__()
        # 假设 args 在外部定义，或者你可以把这些参数传入 __init__
        self.temperature = nn.Parameter(torch.tensor(args.temperature)) # 示例值，原为 args.temperature
        self.scaled_factor = args.scaled_factor # 示例值，原为 args.scaled_factor
        self.margin = margin  # 新增：负样本的边界阈值 (0.4-0.6 通常效果较好)
        # [主干实验 · 损失配比] 负推项原本硬编码权重 1.0。
        # 实测：换 DINOv3 主干后该项在 epoch35 达到 0.25（ViT-B 只有 0.0085，30×），
        # 而正样本项（cos/mse）反而更小 → 训练被"推开异常"主导，原型几何被扭曲。
        # 默认 1.0 = 与 v3.7 逐位一致；>0 时才改变行为。
        self.w_push = float(getattr(args, 'w_push', 1.0))
        # [PUSH-TOPK] 难负例聚合：默认 10（原实现=top-10 均值）；1=max，3=top-3 均值
        self.push_topk = int(getattr(args, 'push_topk', 10))
        # [PUSH-REF] 难负例的参照原型：'global' = 原实现（正常图 batch 平均，脱离推理）；
        #            'self'   = 异常图自己的原型（与推理端 eval_score_v3 的"每图自参照"一致）
        self.push_ref = str(getattr(args, 'push_ref', 'global'))

    def forward(self, x, y, neg=None, mask=None, neg_proto=None):
        """
        Args:
            x: [B, N, D] 正常样本的 Patch Tokens
            y: [B, K, D] 正常样本生成的 Prototypes (Reference)
            neg: [B_aux, N, D] 辅助集(含异常)的 Patch Tokens
            mask: [B_aux, N] 辅助集的 Patch 级 Mask (1=异常, 0=背景)
        """
        # x = x.detach()
        # neg = neg.detach()
        # === Part 1: 正样本损失 (保持你原有的逻辑不变) ===
        x_normalized = F.normalize(x, p=2, dim=-1)
        y_normalized = F.normalize(y, p=2, dim=-1)

        # 计算相似度 [B, N, K]
        cosine_sim = torch.einsum('bnd,bkd->bnk', x_normalized, y_normalized)

        # Softmax 加权融合原型
        weights = F.softmax(cosine_sim / self.temperature, dim=-1)
        soft_tokens = torch.einsum('bnk,bkd->bnd', weights, y)

        # MSE Loss
        loss_mse = F.mse_loss(soft_tokens, x)

        # Scaled Cosine Distance Loss
        cosine_sim_selected = F.cosine_similarity(x, soft_tokens, dim=-1) # [B, N]
        cosine_dist = 1 - cosine_sim_selected

        # 动态缩放因子 (你的核心逻辑)
        dis = cosine_dist.detach()
        dis_mean = dis.mean() + 1e-6

        # 增加数值稳定性保护
        ratio = dis / dis_mean
        # 核心修复：截断比率，防止 3 次方后数值爆炸
        ratio = torch.clamp(ratio, max=5)
        scaled_factor = ratio ** self.scaled_factor
        loss_pos = (scaled_factor * cosine_dist).mean()

        total_loss = 0.5 * loss_pos + 1.0 * loss_mse
        loss_dict = {
            "pos_cos": loss_pos.item(),
            "pos_mse": loss_mse.item()
        }

        # === Part 2: 负样本损失 (Top-K 优化版) ===
        if neg is not None and mask is not None:
            neg_normalized = F.normalize(neg, p=2, dim=-1)

            if self.push_ref == 'self' and neg_proto is not None:
                # [PUSH-REF] 与推理一致：用**本张异常图自己的原型**做参照（每图自参照）
                y_ref = F.normalize(neg_proto, p=2, dim=-1).detach()          # [B_aux, K, D]
                sim_neg = torch.bmm(neg_normalized, y_ref.transpose(1, 2))    # [B_aux, N, K]
                y_global = y_ref
            else:
                # 原实现：正常图 batch 平均后的原型 [1, K_proto, D]
                y_global = y_normalized.detach().mean(dim=0, keepdim=True)
                # 计算相似度矩阵: [B_aux, N_patch, K_proto]
                sim_neg = torch.einsum('bnd,ckd->bnk', neg_normalized, y_global)

            # --- 修改部分：从 max 改为 top-k ---
            # 这里的 k 不能超过 y_global 的原型总数 K_proto
            topk_val = min(max(1, self.push_topk), y_global.size(1))

            # 取相似度最高（离得最近）的 top-k 个值: [B_aux, N_patch, topk_val]
            topk_sims, _ = torch.topk(sim_neg, k=topk_val, dim=-1)

            # 对这 top-k 个相似度取平均，作为该 patch 离正常区域的“综合接近程度”
            avg_topk_sim = topk_sims.mean(dim=-1) # [B_aux, N_patch]

            # 构造 Hinge Loss: 相似度超过 margin 产生惩罚
            # 借鉴 ResAD 的空间约束思想，确保异常特征位于超球体边界之外 [cite: 78, 179, 187]
            push_loss_map = F.relu(avg_topk_sim - self.margin)

            # 计算有效异常区域的 loss
            valid_anomaly_loss = push_loss_map * mask
            num_anomaly_pixels = mask.sum() + 1e-6
            loss_neg = valid_anomaly_loss.sum() / num_anomaly_pixels


            total_loss += self.w_push * loss_neg
            loss_dict["neg_push"] = loss_neg.item()

        return total_loss, loss_dict




class FocalDiceLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred_score_map, gt_mask):
        """
        Args:
            pred_score_map: [B, N] 或 [B, H*W] 模型的预测异常分数 (0~1之间)
            gt_mask: [B, N] 或 [B, H*W] 真实标签 (0或1)
        """
        # 确保输入平展
        pred = pred_score_map.view(-1)
        target = gt_mask.view(-1)

        # 1. Dice Loss
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice_score = (2. * intersection + smooth) / (union + smooth)
        loss_dice = 1 - dice_score

        # 2. Focal Loss
        # 既然输入已经是 0~1 的分数，我们不经过 sigmoid，直接计算 BCE
        # 为了数值稳定性，限制 pred 范围
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)

        bce = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        # 增加 Focal 权重因子 (1-pt)^gamma
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_term = (1 - pt).pow(self.gamma)

        # 增加 Alpha 平衡因子
        alpha_term = torch.where(target == 1, self.alpha, 1 - self.alpha)

        loss_focal = (alpha_term * focal_term * bce).mean()

        return self.dice_weight * loss_dice + self.focal_weight * loss_focal



def proto_distance_scores(patch_mapping, proto_mapping):
    """[vpt4.0 · 公共打分口径] s = 1 - max_k cos(patch, proto_k)，返回 [B, N]。

    这是 vpt4 **训练与评估唯一**使用的像素打分定义（评估端 `compute_fused_map._map28` 用的是同一式子），
    所以任何像素级损失（M5/M7）都必须通过本函数取分数，才能和线上推理口径严格一致
    —— 之前 M5 手写 bmm 时踩过"正常分支 batch=8 / 异常分支 batch=16 导致原型集错位"的坑。

    patch_mapping: [B, N, D]；proto_mapping: [B, K, D]（**必须同 batch**，即"每张图用自己的原型集"）。
    """
    p = torch.nn.functional.normalize(patch_mapping, dim=-1)
    q = torch.nn.functional.normalize(proto_mapping, dim=-1)
    sim = torch.bmm(p, q.transpose(1, 2)).max(dim=-1).values
    return 1.0 - sim


def cross_image_rank_loss(norm_scores, anom_scores, margin=0.5, n_sample=256, normalize=True):
    """[vpt4.0 · M7] 跨图排序一致性损失 —— P-AUROC 的成对 hinge 代理。

    动机（实测瓶颈）：vpt4 变体在 7 项指标里 6 项已达 +1.5%，**唯一不达标的是 P-AUROC**。
    P-AUROC 把**整类所有图的像素分数放在一起排序**，它惩罚的恰恰是"某张图的正常像素分数
    高于另一张图的异常像素分数"这种**跨图不可比**。而现有损失项全是**图内**的：
      * 原始 cos/MSE 损失：正常图内部，没有异常像素参与；
      * 分割头 Dice/Focal：逐图 mask（图内排序）；
      * M5 像素 BCE：逐像素二分类（图内，且各自图用自己的原型集）。
    因此没有任何一项在优化"跨图可比性"。M7 直接把该需求写成损失：

        L = mean_{i,j} relu( (s_norm_j - s_anom_i)/tau + margin )

    即：任意正常像素的分数都应比任意异常像素低至少 margin。样本用**困难挖掘**取
    （正常里分数最高的 n 个、异常里分数最低的 n 个）——这才是真正卡住 P-AUROC 的那些对，
    同时把复杂度压到 O(n²)（n 默认 256 → 6.5 万对，显存与时间可忽略）。
    分数由 `proto_distance_scores` 给出，与推理完全同口径；**不使用任何测试集统计量**。

    **尺度无关（关键）**：实测本任务的异常分数落在 0.010~0.023 这个很窄的区间
    （见 probe_m7_scores.py：正常 p50=0.012 / p95=0.016，异常 p50=0.016 / p95=0.023），
    若直接用固定 margin，训练中映射输出尺度一变，"有效 margin"就漂了（实测 margin≥0.01 时
    100% 的样本对都被激活，损失退化成常数项）。因此这里把分差除以 tau
    ——正常分数分布的稳健离散度 `p75-p25`（detach）——margin 以 "几个 tau" 计（默认 0.5）。
    除以 tau 后损失与类别/训练阶段无关，w_rank 也才有稳定的物理含义。
    """
    norm_scores = norm_scores.reshape(-1)
    anom_scores = anom_scores.reshape(-1)
    if norm_scores.numel() == 0 or anom_scores.numel() == 0:
        return norm_scores.sum() * 0.0                     # 可导的 0，避免 batch 内无异常像素时报错
    k_n = min(int(n_sample), norm_scores.numel())
    k_a = min(int(n_sample), anom_scores.numel())
    ns = torch.topk(norm_scores, k_n).values                              # 最难的"正常"（分数偏高）
    as_ = torch.topk(anom_scores, k_a, largest=False).values              # 最难的"异常"（分数偏低）
    gap = ns.unsqueeze(1) - as_.unsqueeze(0)
    if normalize:
        q75, q25 = torch.quantile(norm_scores.detach(), 0.75), torch.quantile(norm_scores.detach(), 0.25)
        tau = torch.clamp(q75 - q25, min=1e-6)
        gap = (gap / tau).clamp(-20.0, 20.0)              # 防止 tau 塌缩时数值爆炸
    return torch.relu(gap + margin).mean()


def cross_image_rank_loss_from_features(normal_patch_mapping, normal_learnable_mapping,
                                        abnormal_patch_mapping, abnormal_learnable_mapping,
                                        aux_mask, margin=0.5, n_sample=256, normalize=True):
    """[vpt4.0 · M7] 训练循环实际调用的封装（抽出来是为了能被单元测试逐位覆盖）。

    正常像素 = 正常图全部 patch ∪ 辅助图里 mask=0 的 patch；
    异常像素 = 辅助图里 mask=1 的 patch。
    两侧都用**各自图自己的原型集**算分（与推理时"测试图自参照"完全一致）。
    """
    s_norm = proto_distance_scores(normal_patch_mapping, normal_learnable_mapping)     # [B_n, N]
    s_aux = proto_distance_scores(abnormal_patch_mapping, abnormal_learnable_mapping)  # [B_a, N]
    flat = s_aux.reshape(-1)
    m = aux_mask.reshape(-1) > 0.5
    if int(m.numel()) != int(flat.numel()):            # 配置错误 → 立刻报错，不静默降级
        raise RuntimeError(
            f'[M7] aux_mask 与 patch 分数元素数不一致：{m.numel()} vs {flat.numel()}；'
            '请检查 resize_mask / 输入分辨率设置')
    normal_scores = torch.cat([s_norm.reshape(-1), flat[~m]], dim=0)
    return cross_image_rank_loss(normal_scores, flat[m], margin=margin,
                                 n_sample=n_sample, normalize=normalize)


def image_level_rank_loss(normal_img_scores, anom_img_scores, margin=0.5, normalize=True):
    """[vpt4.0 · M10] **图像级**跨图排序一致性损失 —— M7 的"上一层"版本。

    动机：M7 治好了像素级的跨图漂移（leather/pill/zipper），但 15 类里仍有**图像级**回退
    （screw I-AUROC −0.055、hazelnut I-F1 −0.029 等）。图像级分数在评估里定义为
    "该图全部像素分数中最高的 1% 的均值"，所以这里也用同一规则取分，再做跨图 hinge：

        L = mean_{i,j} relu( (s_img_normal_i - s_img_anom_j)/tau + margin )

    tau = 正常图像级分数的标准差（detach，按 batch 估计），使损失与分数尺度无关。
    每个 batch 只有 8 张正常图 + 16 张异常图，样本量小、不需要困难挖掘，直接全对即可。
    """
    n = normal_img_scores.reshape(-1)
    a = anom_img_scores.reshape(-1)
    if n.numel() == 0 or a.numel() == 0:
        return n.sum() * 0.0
    gap = n.unsqueeze(1) - a.unsqueeze(0)
    if normalize:
        tau = torch.clamp(n.detach().std(unbiased=False), min=1e-6)
        gap = (gap / tau).clamp(-20.0, 20.0)
    return torch.relu(gap + margin).mean()


def top_ratio_mean(scores, ratio=0.01):
    """按评估口径取"图像级分数"：每张图取最高的 ratio 比例的像素分数求均值。"""
    k = max(1, int(scores.shape[-1] * ratio))
    return torch.sort(scores, dim=-1, descending=True)[0][:, :k].mean(dim=-1)


def sinkhorn_assign(scores, eps=0.05, iters=3):
    """[CLUSTER] SwAV 式 Sinkhorn 平衡分配。

    输入 scores [M, K]（M = 一个 batch 的全部正常 patch 数，K = 原型数），
    输出 Q [M, K]：行和 = 1、列和 ≈ 1/K（即每个簇被**大致等量**的正常 patch 使用）。
    这一步正是"让正常 patch 形成若干簇"的机制：既要求每个 patch 明确归属（行归一），
    又禁止所有 patch 挤到少数簇里（列归一 → 均衡使用）。
    """
    # 数值安全：先在列（每个 patch）方向减最大值再取 exp，避免 exp(±200) 溢出/下溢成 0；
    # 归一化分母统一 clamp，避免除零把 NaN 传进损失。
    _S = (scores / eps)
    _S = _S - _S.max(dim=-1, keepdim=True).values
    Q = torch.exp(_S).t()                               # [K, M]，取值 (0, 1]
    Q = Q / Q.sum().clamp_min(1e-30)
    K, M = Q.shape
    for _ in range(int(iters)):
        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-30)               # 每个 patch（列）和为 1
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-30) * (M / K)     # 每个原型（行）和为 M/K
    Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-30)                   # 收尾：每个 patch 和为 1
    return Q.t()                                                    # [M, K]：行和=1、列和≈M/K


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred_score_map, gt_mask):
        """
        pred_score_map: [B, N] 必须在 0~1 之间
        gt_mask: [B, N] 0 或 1
        """
        # Flatten
        pred = pred_score_map.view(-1)
        target = gt_mask.view(-1)

        # === 1. Dice Loss ===
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice_score = (2. * intersection + smooth) / (union + smooth)
        loss_dice = 1 - dice_score

        # === 2. Focal Loss ===
        # 极其重要：确保 pred 在 (0, 1) 范围内，防止 Log(0)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)

        bce = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        # (1-pt)^gamma
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_term = (1 - pt).pow(self.gamma)

        # Alpha 平衡
        alpha_term = torch.where(target == 1, self.alpha, 1 - self.alpha)

        loss_focal = (alpha_term * focal_term * bce).mean()

        return self.dice_weight * loss_dice + self.focal_weight * loss_focal






import argparse
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from util.dataset import *
from util.memorybank import MemoryBank

def build_backbone(args):
    """[主干实验] 统一的主干构建入口（**只换特征主干**，核心创新模块完全不变）。

    base/large → DINOv2 ViT-B/14、ViT-L/14（v3.7 原路径）
    dinov3b/dinov3l → DINOv3 ViT-B/16、ViT-L/16（timm 实现，RoPE；权重在封装内从本地读入）
    """
    if str(args.backbone).startswith('dinov3'):
        from dinov3_mapping import build_dinov3
        return build_dinov3(args.backbone, args, img_size=518)
    if str(args.backbone).startswith('clip'):
        from clip_mapping import build_clip
        return build_clip(args.backbone, args, img_size=518)
    return vision_transformer.__dict__['vit_' + args.backbone](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8, num_register_tokens=4,
        interpolate_antialias=False, interpolate_offset=0.1, arg=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    # model info
    parser.add_argument('--learnable_layers',type=int, default=1)
    parser.add_argument('--learnable_num_per_layer',type=int, default=32)

    parser.add_argument('--fused_layer',type=list,default=[1,2,3,4,6,8,11])
    parser.add_argument('--fused_layer_learnable',type=list,default=[0])
    # [主干实验] 特征主干与层配置（只换主干，核心创新部分保持不变）
    parser.add_argument('--backbone', type=str, default='base',
                        help="[主干实验] 'base'=DINOv2 ViT-B/14（默认）；'large'=DINOv2 ViT-L/14")
    parser.add_argument('--fused_layer_csv', type=str, default='',
                        help='[主干实验] 覆盖 --fused_layer，例如 "2,4,6,8,12,16,22"（ViT-L）')
    parser.add_argument('--train_class',type=str,default='carpet',choices=['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',"candle", "capsules", "cashew",   "chewinggum",  "fryum",  "macaroni1",  "macaroni2",  "pcb1",  "pcb2",  "pcb3","pcb4",  "pipe_fryum"])

    parser.add_argument('--temperature',type=float,default=0.07,help='Temperature for the cosine distance loss')
    parser.add_argument('--scaled_factor',type=int,default=3,help='Scaled factor for the cosine distance loss')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'/home/jjquan/datasets/mvtec')  # Replace it with your path.
    parser.add_argument('--anomaly_data_root', type=str, default=r'/home/jjquan/experiment/AnoStyler/genbyAnoStyler')  # 合成异常路径
   
    parser.add_argument('--shot', type=int, default=4) # Number of samples
    parser.add_argument('--total_epochs', type=int, default=70)
    parser.add_argument('--internal', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
    # [AUDIT-FIX-4] 只评估 mapping 分支（type=1）并跳过 bank 构建/误差图计算，用于快速迭代
    # ===== vpt3.0：改善后 mapping ⊕ 原版 MemoryBank 全启用（默认开启）=====
    parser.add_argument('--vpt3', type=int, default=1, help='1=vpt3.0 融合评估(默认)；0=关闭，退回原 7 类打分')
    parser.add_argument('--vpt3_w', type=float, default=0.5,
                        help='[FIX-2] 融合权重：w*class-z(mapping) + (1-w)*class-z(bank)；默认 0.5 与生产一致（旧默认 0.3）')
    parser.add_argument('--vpt3_scales', type=str, default='1.5', help='改善后 mapping 的多尺度')
    parser.add_argument('--vpt3_flip', type=int, default=0,
                        help='[FIX-2] 改善后 mapping 是否做水平翻转 TTA；默认 0 与生产 vpt3.7 一致（旧默认 1）')
    parser.add_argument('--vpt3_cache_dir', type=str, default='', help='非空则把两个分支的异常图落盘到该目录')
    # ===== vpt4.0：mapping 分支的模型/算法改动（M1 原型初始化 / M2 残差映射 / M3 原型多样性正则）=====
    parser.add_argument('--token_init_std', type=float, default=0.02,
                        help='M1: 可学习原型初始化尺度（0.02=与 patch embedding 同量级；原实现为 1e-6）')
    parser.add_argument('--res_mapping', type=int, default=1,
                        help='M2: 1=残差映射 x+f(LN(x))（最后一层零初始化，起点=不做映射）；0=原低秩映射')
    parser.add_argument('--metric_head', type=int, default=1,
                        help='M6: 1=在映射后加可学习度量头（初值单位阵，等价于现状余弦）；0=关闭')
    parser.add_argument('--w_pixel', type=float, default=0.2,
                        help='M5: 像素级直接监督权重（用 aux 掩码约束"到最近原型距离图"的 BCE）')
    parser.add_argument('--global_proto', type=int, default=0,
                        help='M4: 1=原型集 = 图像自适应原型 ⊕ 全局可学习原型（给"跨图校准"一个固定锚点）')
    parser.add_argument('--w_div', type=float, default=0.1,
                        help='M3: 原型多样性正则权重（惩罚映射后原型之间过高的两两余弦）')
    parser.add_argument('--div_margin', type=float, default=0.5,
                        help='M3: 超过该余弦相似度才惩罚')
    parser.add_argument('--w_rank', type=float, default=0.0,
                        help='M7: 跨图排序一致性损失权重（默认 0=关闭）；该项直接优化 P-AUROC 的跨图可比性')
    parser.add_argument('--push_topk', type=int, default=10,
                        help='[PUSH-TOPK] 难负例聚合的 top-k（10=原实现均值；1=max）')
    parser.add_argument('--push_ref', type=str, default='global',
                        help="[PUSH-REF] 难负例参照原型：'global'=原实现（正常图 batch 平均）；"
                             "'self'=异常图自己的原型（与推理每图自参照一致）")
    parser.add_argument('--w_push', type=float, default=1.0,
                        help='[主干实验 · 损失配比] 负推项权重（默认 1.0 = v3.7 原行为）。'
                             '换 DINOv3 主干后该项比 ViT-B 大 30×，可下调以恢复"正样本项/负推项"平衡')
    parser.add_argument('--save_every', type=int, default=1,
                        help='[工程] 每 N 轮存一次权重（默认 1 = 与历史一致）；长轮数训练时降低落盘频率')
    parser.add_argument('--w_cluster', type=float, default=0.0,
                        help='[CLUSTER] 聚类式损失权重（SwAV 平衡分配；默认 0 = 与 v3.7 逐位一致）')
    parser.add_argument('--cluster_temp', type=float, default=0.1, help='[CLUSTER] 相似度温度')
    parser.add_argument('--cluster_eps', type=float, default=0.05, help='[CLUSTER] Sinkhorn 温度')
    parser.add_argument('--cluster_iters', type=int, default=3, help='[CLUSTER] Sinkhorn 迭代次数')
    parser.add_argument('--w_cluster_cons', type=float, default=0.0,
                        help='[CLUSTER] 跨增强簇一致性损失权重（默认 0 = 关闭）')
    parser.add_argument('--w_distill', type=float, default=0.0,
                        help='[LOSS-DISTILL] 教师=原始库侧打分（训练无关，仅用 K shot 建库），'
                             '学生=可学习侧打分；在"教师分差明显"的样本对上要求排序一致（尺度无关 hinge）。'
                             '默认 0=关闭（与 v3.7 完全一致）。')
    parser.add_argument('--distill_margin', type=float, default=0.5,
                        help='[LOSS-DISTILL] 学生侧 hinge margin，单位为"学生分数的 batch 内标准差"')
    parser.add_argument('--distill_topk', type=int, default=16,
                        help='[LOSS-DISTILL] 每图取教师最高/最低各 k 个像素组成 k×k 对（只在这些极端对上约束）')
    parser.add_argument('--distill_tau_ratio', type=float, default=0.25,
                        help='[LOSS-DISTILL] 只对"教师分差 > ratio×std(T)"的对生效，过滤噪声对')
    parser.add_argument('--distill_global', type=int, default=512,
                        help='[LOSS-DISTILL] 教师分数里的全局项分位（与推理 raw_global 同义，512=中位数）')
    parser.add_argument('--distill_alpha', type=float, default=0.5,
                        help='[LOSS-DISTILL] 教师分数指数：d1^alpha·q^(1-alpha)（与推理 raw_alpha 一致）')
    parser.add_argument('--distill_sub_seed', type=int, default=1234,
                        help='[LOSS-DISTILL] 教师全局项子采样种子（与推理 sub_seed 一致）')
    parser.add_argument('--w_flip_cons', type=float, default=0.0,
                        help='[LOSS-FLIP] 翻转一致性损失权重（默认 0=关闭）：要求可学习侧分数图在'
                             '水平翻转下保持一致（逐图 z 后 L1）；推理端仍为单视图。')
    parser.add_argument('--w_scale_cons', type=float, default=0.0,
                        help='[LOSS-SCALE] 跨尺度一致性损失权重（默认 0=关闭）：要求可学习侧分数图在'
                             '训练尺度与 scale_cons_scale 倍尺度下一致（逐图 z 后 L1）；推理仍单视图。')
    parser.add_argument('--scale_cons_scale', type=float, default=1.5,
                        help='[LOSS-SCALE] 一致性第二视图的放缩倍数（默认 1.5，与推理 1.5× 对齐）')
    parser.add_argument('--w_peak', type=float, default=0.0,
                        help='[LOSS-PEAK] 分数图峰值/稀疏正则权重（默认 0=关闭）：异常图要求 top-1% '
                             '远高于自身中位数、正常图要求两者接近（对准 P-AP/P-F1 的高分段精度）。')
    parser.add_argument('--peak_margin', type=float, default=0.5,
                        help='[LOSS-PEAK] 峰-中位差距的 hinge margin（单位：batch 内分数标准差）')
    parser.add_argument('--peak_top_ratio', type=float, default=0.01,
                        help='[LOSS-PEAK] 峰值取多少比例的像素（默认 1%，与评估的图像级分数口径一致）')
    parser.add_argument('--w_ema', type=float, default=0.0,
                        help='[LOSS-EMA] 均值教师(Mean-Teacher)一致性权重（默认 0=关闭）：维护可训练参数的 '
                             'EMA 副本，要求可学习侧分数图与之逐图 z 后一致。用于压低种子方差。')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='[LOSS-EMA] EMA 衰减系数')
    parser.add_argument('--w_balance', type=float, default=0.0,
                        help='[LOSS-BAL] 原型使用均衡正则权重（默认 0=关闭）：把"正常 patch→原型"的软分配'
                             '平均分布拉向均匀（KL(avg||uniform)），抑制少数原型包办、其余闲置。')
    parser.add_argument('--balance_temp', type=float, default=0.07,
                        help='[LOSS-BAL] 软分配 softmax 温度（与训练 temperature 一致）')
    parser.add_argument('--w_prox', type=float, default=0.0,
                        help='[LOSS-PROX] 参数空间邻近项权重（默认 0=关闭）：||θ−θ_ema||²/参数量，'
                             '抑制 epoch 间质量摆动（实测最好轮常在训练中断）。')
    parser.add_argument('--prox_decay', type=float, default=0.99,
                        help='[LOSS-PROX] 参数 EMA 锚点的衰减系数')
    parser.add_argument('--w_prod', type=float, default=0.0,
                        help='[LOSS-PROD] **训推一致性**损失权重（默认 0=关闭）：用合成掩码直接监督'
                             '推理端真正使用的分数 m = w_learn·s_learn + (1-w_learn)·s_raw（s_raw 无参数）。')
    parser.add_argument('--prod_w_learn', type=float, default=0.6,
                        help='[LOSS-PROD] m 里可学习侧的权重（与推理端 eval_score_v3 --w 一致，默认 0.6）')
    parser.add_argument('--prod_znorm', type=int, default=0,
                        help='[LOSS-PROD][修复] 1=两支各自"图内 z"后再加权（贴近推理端"类内 z 后加权"），'
                             '0=按原始距离加权（原实现）')
    parser.add_argument('--w_near', type=float, default=0.0,
                        help='[LOSS-NEAR] 最近原型量化损失权重（默认 0=关闭）：对正常 patch 直接最小化'
                             '"到最近原型的距离 dlmin"（推理端真正使用的可学习侧分数），hinge 防塌缩。')
    parser.add_argument('--near_margin', type=float, default=0.05,
                        help='[LOSS-NEAR] dlmin 的 hinge margin（超过才惩罚，避免过度收缩）')
    parser.add_argument('--w_barlow', type=float, default=0.0,
                        help='[LOSS-BARLOW] 映射特征去冗余正则权重（默认 0=关闭）：Barlow 式，'
                             '惩罚映射后正常 patch 特征维间的相关（去冗余、保方差）。')
    parser.add_argument('--barlow_lambda', type=float, default=0.005,
                        help='[LOSS-BARLOW] 对角项（方差保持）的权重')
    parser.add_argument('--w_comp', type=float, default=0.0,
                        help='[LOSS-COMP] "补短板"损失权重（默认 0=关闭）：用原始库侧的错向响应加权，'
                             '让可学习侧专门学习"原始库侧判错的地方"，提升互补性（诊断驱动）。')
    parser.add_argument('--comp_lam', type=float, default=1.0,
                        help='[LOSS-COMP] 教师错误权重的放大系数（w = 1 + λ·err）')
    parser.add_argument('--rank_margin', type=float, default=0.5,
                        help='M7: 成对 hinge 的 margin；rank_norm=1 时单位为"正常分数离散度的倍数"')
    parser.add_argument('--rank_norm', type=int, default=1,
                        help='M7: 1=分差按正常分数稳健离散度(p75-p25)归一化（尺度无关，推荐）；0=用原始分差')
    parser.add_argument('--rank_samples', type=int, default=256,
                        help='M7: 困难样本挖掘时，正常/异常各取的像素数（复杂度 O(n^2)）')
    parser.add_argument('--w_rank_final', type=float, default=-1.0,
                        help='[LOSS-CURRIC] w_rank 的课程终值（<0 = 保持常数，行为与 v3.7 一致）；'
                             '给定时 w_rank 从 --w_rank 线性变到该值（例如 0.02→0.04）。')
    parser.add_argument('--w_img_rank', type=float, default=0.0,
                        help='M10: 图像级跨图排序一致性损失权重（默认 0=关闭）；对准 I-AUROC/I-AP 的跨图可比性')
    parser.add_argument('--img_rank_margin', type=float, default=0.5,
                        help='M10: 图像级 hinge 的 margin（单位为正常图像级分数的标准差）')
    parser.add_argument('--map_only', action='store_true',
                        help='只跑 mapping 分支（type=1）：不建 bank、不读 errmap、评估只算 type=1')
    parser.add_argument('--seed', type=int, default=1,
                        help='模型里 tokens_per_layer / token_mapping 是随机初始化的（预训练权重不含这两个模块），'
                             '所以固定这个种子才能保证 A/B 的初始权重一致')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    # [vpt4.0 · 2026-10-09] 测试不裁剪口径：>0 时训练/GT 都用 Resize(N) 且**不做 CenterCrop**
    # （与评估脚本 eval_nocrop.py --size N 完全同一几何；默认 0 = 保持原 448→裁392）
    parser.add_argument('--nocrop_size', type=int, default=0,
                        help='>0: 训练与 GT 预处理改为 Resize(N) 无裁剪（例如 392）')
    parser.add_argument('--eval_size', type=int, default=0,
                        help='[FIX-2] >0 时**训练内评估**的测试集改用 Resize(N) 无裁剪（生产口径，例如 448）；'
                             '0 = 沿用训练几何（裁剪 392）。仅影响训练期打印的指标，不影响训练本身。')
    # save info
    parser.add_argument('--save_dir', type=str, default='./mvtec_saved_results0')
    parser.add_argument('--save_name', type=str, default='4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add')
    
    parser.add_argument('--pth_epoch', type=str, default='5')
    parser.add_argument('--vis_dir', type=str, default='visualization')
    
    args = parser.parse_args()
    if getattr(args, 'fused_layer_csv', ''):
        args.fused_layer = [int(x) for x in args.fused_layer_csv.split(',') if x != '']
        print(f'[主干实验] fused_layer 覆盖为 {args.fused_layer}')
    if args.dataset == 'MVTec-AD':
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    if args.dataset == 'VisA':
        args.item_list = ["candle", "capsules", "cashew",   "chewinggum",  "fryum",  "macaroni1",  "macaroni2",  "pcb1",  "pcb2",  "pcb3","pcb4",  "pipe_fryum"]


    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    # [AUDIT-FIX-9] 原代码把设备硬编码成 'cuda:2'：一旦用 CUDA_VISIBLE_DEVICES 只暴露一张卡
    # （本机常用做法），cuda:2 就越界，脚本直接跑不起来。这里改成环境变量可覆盖，默认 cuda:0。
    device = os.environ.get('AUDIT_DEVICE', 'cuda:0') if torch.cuda.is_available() else 'cpu'
    print(f'[audit] device = {device}')

    setup_seed(1)

    
    if getattr(args, 'nocrop_size', 0) and args.nocrop_size > 0:
        _S = int(args.nocrop_size)
        _mean, _std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        data_transform = transforms.Compose([transforms.Resize((_S, _S)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=_mean, std=_std)])
        gt_transform = transforms.Compose([transforms.Resize((_S, _S)), transforms.ToTensor()])
        print_fn(f'[nocrop] 预处理 = Resize({_S}) 无裁剪（训练与 GT 同几何）')
    else:
        data_transform, gt_transform = get_data_transforms(args.input_size,args.crop_size)
    # [FIX-2] 训练内评估可单独指定测试几何：--eval_size 448 → Resize(448) 无裁剪（与生产 eval_score_v3 一致）
    if getattr(args, 'eval_size', 0) and args.eval_size > 0:
        _ES = int(args.eval_size)
        _em, _es = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        eval_transform = transforms.Compose([transforms.Resize((_ES, _ES)), transforms.ToTensor(),
                                             transforms.Normalize(mean=_em, std=_es)])
        eval_gt_transform = transforms.Compose([transforms.Resize((_ES, _ES)), transforms.ToTensor()])
        print_fn(f'[FIX-2] 训练内评估几何 = Resize({_ES}) 无裁剪（生产口径）')
    else:
        eval_transform, eval_gt_transform = data_transform, gt_transform
    train_data_list = []
    test_data_list = []

    
    
    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):

            test_path = os.path.join(args.data_path, item)
            test_data = MVTecDataset(root=test_path, transform=eval_transform,
                                     gt_transform=eval_gt_transform, phase="test")
            test_data_list.append(test_data)

            if args.train_class is not None and args.train_class != item:
                continue
            
            train_path = os.path.join(args.data_path, item, 'train')
            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            # [统一数据源 2026-10-09] shot 清单改由 `shot_utils.get_shots()` 统一给出：
            #   训练 / mapping 分支评估 / bank 建库 三处**共用同一份清单**（首次生成后落盘在
            #   `.shots/<cls>_shot{K}_seed{S}.json`），因此把 shot 数改成 8/16 时三处依然取同一批图，
            #   不会再出现"两个分支抽到不同正常图"的问题；同时把清单哈希打进日志便于追溯。
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from shot_utils import get_shots, shots_hash
            _picks = get_shots(args.data_path, item, shots=args.shot, seed=args.seed)
            _keep = set(_picks)
            _sel = [(p, _i) for (p, _i) in train_data.samples if os.path.basename(p) in _keep]
            assert len(_sel) == args.shot, f'[统一数据源] {item}: 期望 {args.shot} 张，实际 {len(_sel)}'
            train_data.samples = _sel
            print_fn(f'[shots] {item}: shot={args.shot} seed={args.seed} '
                     f'hash={shots_hash(_picks)} picks={_picks}')
            
            if False:
                # 拷贝正常样本到指定目录下
                import shutil

                # 保存路径根目录
                dest_root = '/home/jjquan/experiment/AnoStyler/normal_imgs/visa'
                os.makedirs(dest_root, exist_ok=True)

                # 将每个选中的样本拷贝到按类别的子目录下
                for src_path, _ in train_data.samples:
                
                    cls_name = args.train_class
                    
                    dest_dir = os.path.join(dest_root, cls_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(src_path, os.path.join(dest_dir, os.path.basename(src_path)))
                
                
                continue
            # print_fn(train_data.samples)
            # Data augmentation
            augmented_data = []
            img_counter=0
            # [vpt3.0] 保存未增强的 shot 张正常图，用于在原始特征空间建 patch 库
            normal_imgs_for_bank = torch.stack([img for img, _ in train_data], dim=0)
            for img, label in train_data:
                augmented_imgs = augmentation(img)
                for aug_img in augmented_imgs:
                    augmented_data.append((aug_img, label))
                #     # 保存增强图像到磁盘
                #     augment_save_dir = os.path.join('augment')
                #     os.makedirs(augment_save_dir, exist_ok=True)
                #     save_path = os.path.join(augment_save_dir, f"{img_counter}.png")
                #     # 将tensor转换为PIL图像并保存
                #     aug_img_pil = transforms.ToPILImage()(denorm(aug_img))
                #     aug_img_pil.save(save_path)
                #     img_counter += 1
            augmented_dataset = torch.utils.data.TensorDataset(
                torch.stack([item[0] for item in augmented_data]),
                torch.tensor([item[1] for item in augmented_data])
            )

            train_data_list.append(augmented_dataset)

        train_data = ConcatDataset(train_data_list)
        from itertools import cycle
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size//2, shuffle=True, num_workers=4, drop_last=True)
        # [AUDIT-FIX-4] --map_only：完全不建 bank、不算/不读 errmap。
        # 原因：bank 的构建 + 误差图计算是整条流程里最慢的一段（每类 10~20 分钟，且与 mapping 分支无关），
        # 本阶段只评估 mapping 分支（type=1），所以整段跳过。
        visual_bank_map = None
        if not args.map_only:
            imgs = []
            for ba in train_dataloader:
                imgs.append(ba[0])
            imgs = torch.cat(imgs,dim=0)
            imgs = imgs.to(device)
            visual_bank = MemoryBank(target_cls=args.train_class,use_json=False,imgs=imgs,info=print_fn,device=device)
            for item, test_data in zip(args.item_list, test_data_list):
                if args.train_class is not None and args.train_class != item:
                    continue
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
                errmap_list = []
                print_fn(f"computing visual bank error map...")
                path = f'.errmap/{args.train_class}.map'
                if os.path.exists(path):
                    print_fn(f"load pre-computed visual bank error map from {path}")
                    visual_bank_map = torch.load(path).to(device)
                else:
                    for img, gt, label, img_path in tqdm(test_dataloader, ncols=80):
                        img = img.to(device)
                        anomaly_map = visual_bank.query_batch(img)
                        anomaly_map = torch.tensor(anomaly_map)
                        anomaly_map = anomaly_map.unsqueeze(1)
                        anomaly_map = anomaly_map.to(device)
                        errmap_list.append(anomaly_map)
                    visual_bank_map = torch.cat(errmap_list, dim=0)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    torch.save(visual_bank_map, path)
        train_dataloader = cycle(train_dataloader)
        # visa辅助训练集
        # visa_test_set = Dataset('/home/jjquan/datasets/visa', data_transform, gt_transform,'visa','test')
        # abnormal_train_dataloader = torch.utils.data.DataLoader(visa_test_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

        # realnet辅助训练集

        # realnet_dataset = AnomalySyntheticDataset(
        #     root='/home/jjquan/datasets/mvtec_sys_anomaly/sdas',
        #     cls_name=args.train_class,
        #     num_per_rank=60,
        #     transform=data_transform,
        #     tar_tansform=gt_transform
        # )
        
        # anomaly diffusion数据集
        # anomaly_diffusion_dataset = GeneratedAnomalyDataset(
        #     root_dir="/home/jjquan/datasets/generated_dataset",
        #     category=args.train_class,
        #     transform=data_transform,
        #     target_transform=gt_transform,
        #     return_original=False ,
        #     samples_per_anomaly_type=35 
        # )
        
        # anomaly styler数据集
        name = 'mvtec' if args.dataset == 'MVTec-AD' else 'visa'
        anomaly_styler_dataset = AnoStylerGeneratedDataset(
            root_dir=os.path.join(args.anomaly_data_root, name),
            category=args.train_class,  # Only load data from the 'carpet' directory
            transform=data_transform,
            image_suffix='.jpg',
            target_transform=gt_transform,
            samples_per_anomaly_type=35  # Randomly select 10 samples from each anomaly type
        )
        
         # anomaly Draem数据集
        # anomaly_DTD_dataset = AnoStylerGeneratedDataset(
        #     root_dir="/home/jjquan/experiment/AnoStyler/genbyDTD",
        #     category=args.train_class,  # Only load data from the 'carpet' directory
        #     transform=data_transform,
        #     target_transform=gt_transform,
        #     samples_per_anomaly_type=35  # Randomly select 10 samples from each anomaly type
        # )
        abnormal_train_dataloader = torch.utils.data.DataLoader(anomaly_styler_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    if args.phase == 'test':
        print(f'Testing :\n--vis_dir:{args.save_name}/{args.vis_dir}\n--test_class:{args.train_class}\n--pth_epoch:{args.pth_epoch}\n\n')
        checkpoint = torch.load(os.path.join(args.save_dir, args.save_name, f'model_{args.train_class}_{args.pth_epoch}.pth'))
        
        model = build_backbone(args)
        model.to(device)
        segmentation_head = nn.Sequential(
            nn.Linear(model.embed_dim, model.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(model.embed_dim // 2, 1), # 输出单通道，用于二分类
            nn.Sigmoid() 
        ).to(device)
        
        model.load_state_dict(checkpoint['dino'])
        segmentation_head.load_state_dict(checkpoint['seg_head'])
        
        for item, test_data in zip(args.item_list, test_data_list):
            if args.train_class is not None and args.train_class != item:
                continue
            
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
            metric_names = ["I-Auroc", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
            header = "{:^12}".format("Class") + "".join(["{:^12}".format(name) for name in metric_names])
            print_fn(header)
            if args.map_only:
                # [AUDIT-FIX-4] 只评估 mapping 分支，跳过 bank 相关融合
                r1 = inference(model, segmentation_head, test_dataloader, device=device, resize_mask=256,
                               max_ratio=0.01, vis_dir=None, visual_bank_map=None, type=1)
                print_fn("{:^12}".format(item) + "".join(["{:^12.4f}".format(v) for v in r1]))
                continue
            results1 = inference(model, segmentation_head, test_dataloader, device=device, resize_mask=256, max_ratio=0.01, vis_dir='/home/jjquan/experiment/vpt4/saved_results_VISA/4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add/vis-10epoch',visual_bank_map=visual_bank_map,type=5)
            results2 = inference(model, segmentation_head, test_dataloader, device=device, resize_mask=256, max_ratio=0.01, vis_dir=None, visual_bank_map=visual_bank_map,type=8)
            results3 = inference(model, segmentation_head, test_dataloader, device=device, resize_mask=256, max_ratio=0.01, vis_dir=None,visual_bank_map=visual_bank_map,type=9)
            results4 = inference(model, segmentation_head, test_dataloader, device=device, resize_mask=256, max_ratio=0.01, vis_dir=None,visual_bank_map=visual_bank_map,type=10)
            results5 = inference(model, segmentation_head, test_dataloader, device=device, resize_mask=256, max_ratio=0.01, vis_dir=None,visual_bank_map=visual_bank_map,type=11)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results1
            auroc_sp2, ap_sp2, f1_sp2, auroc_px2, ap_px2, f1_px2, aupro_px2 = results2
            auroc_sp3, ap_sp3, f1_sp3, auroc_px3, ap_px3, f1_px3, aupro_px3 = results3
            auroc_sp4, ap_sp4, f1_sp4, auroc_px4, ap_px4, f1_px4, aupro_px4 = results4
            auroc_sp5, ap_sp5, f1_sp5, auroc_px5, ap_px5, f1_px5, aupro_px5 = results5

            row = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]])
            print_fn(row)
            row = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp2, ap_sp2, f1_sp2, auroc_px2, ap_px2, f1_px2, aupro_px2]])
            print_fn(row)
            row = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp3, ap_sp3, f1_sp3, auroc_px3, ap_px3, f1_px3, aupro_px3]])
            print_fn(row)
            row = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp4, ap_sp4, f1_sp4, auroc_px4, ap_px4, f1_px4, aupro_px4]])
            print_fn(row)
            row = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp5, ap_sp5, f1_sp5, auroc_px5, ap_px5, f1_px5, aupro_px5]])
            print_fn(row)
            
            print(f"Visualization Done!")
        
    if args.phase == 'train':
        train_args_str = (
            f"\n\n\n\n\n=======================================☆☆☆ Configuration Information ☆☆☆========================================\n"
            f"--Running Time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"--train_class {args.train_class}\n"
            f"--learnable_layers {args.learnable_layers}\n"
            f"--learnable_num_per_layer {args.learnable_num_per_layer}\n"
            f"--fused_layer {args.fused_layer}\n"
            f"--fused_layer_learnable {args.fused_layer_learnable}\n"
            f'--total_epochs {args.total_epochs}\n'
            f'--internal {args.internal}\n'
            f'--batch_size {args.batch_size}\n'
            f'--shot {args.shot}\n'
            f'--temperature {args.temperature}\n'
            f'--scaled_factor {args.scaled_factor}'
            f"\n========================================☆☆☆☆☆ End ☆☆☆☆☆=======================================================\n"
        )
        print_fn(train_args_str)
        # Model Initialization
        # [CONTROL] 与审计版保持一致：建模型前固定随机源（否则 tokens_per_layer/token_mapping
        # 的随机初始化会因 RNG 消耗不同而不同，A/B 不可比）。
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        # [AUDIT-FIX-3] 原路径 /home/jjquan/codebase/vpt/checkpoint/... 已不存在，训练阶段会直接崩。
        # 改成按顺序找可用权重（Downloads 下那份是同名同源权重）。
        if str(args.backbone).startswith('dinov3'):
            # [主干实验] DINOv3：预训练权重已在封装内从本地 safetensors 载入，这里不再重复加载
            _pth_candidates = ['<dinov3-internal>']
        elif str(args.backbone).startswith('clip'):
            # [主干实验] CLIP：同上，权重在 clip_mapping 封装内载入
            _pth_candidates = ['<clip-internal>']
        elif args.backbone == 'large':      # [主干实验] ViT-L/14 预训练权重
            _pth_candidates = [
                '/home/jjquan/Downloads/dinov2_vitl14_reg4_pretrain_converted.pth',
                '/home/jjquan/Downloads/timm_dinov2_vitl14.pth',
            ]
        else:
            _pth_candidates = [
                '/home/jjquan/codebase/vpt/checkpoint/dinov2_vitb14_reg4_pretrain.pth',
                '/home/jjquan/Downloads/dinov2_vitb14_reg4_pretrain.pth',
                '/home/jjquan/experiment/visual2_v2/dinov2_vitb14_reg4_pretrain.pth',
            ]
        pth = next((p for p in _pth_candidates if os.path.exists(p)), _pth_candidates[0])
        print_fn(f'[audit] 预训练权重: {pth}')
        model = build_backbone(args)
        model.to(device)
        if not (str(args.backbone).startswith('dinov3') or str(args.backbone).startswith('clip')):
            state_dict = torch.load(pth, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        model.eval()

        # 先全部冻结再解冻子模块
        for param in model.parameters():
            param.requires_grad = False

        for param in model.tokens_per_layer.parameters():
            param.requires_grad = True

        for param in model.token_mapping.parameters():
            param.requires_grad = True

        
        fused_layer = args.fused_layer
        fused_layer_learnable = args.fused_layer_learnable
       
        # 轻量级分割头示例
        segmentation_head = nn.Sequential(
            nn.Linear(model.embed_dim, model.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(model.embed_dim // 2, 1), # 输出单通道，用于二分类
            nn.Sigmoid() 
        ).to(device)
        trainable = nn.ModuleList([model.tokens_per_layer,model.token_mapping,segmentation_head])

        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=5e-5, total_iters=args.total_epochs*len(abnormal_train_dataloader),
                                           warmup_iters=100)
        
        # 1. 针对不同模块设置不同的学习率
        # 让 mapping 层学习更谨慎，seg_head 可以稍微快一点
        # optimizer = StableAdamW([
        #     {'params': model.tokens_per_layer.parameters(), 'lr': 1e-4},
        #     {'params': model.token_mapping.parameters(), 'lr': 1e-4},
        #     {'params': segmentation_head.parameters(), 'lr': 5e-4} 
        # ], betas=(0.9, 0.999), weight_decay=1e-2, amsgrad=True, eps=1e-10)

        # # 2. 调整 Scheduler，让它在早期就完成大部分衰减
        # # 将 total_iters 设为总轮数的 1/2 或 1/3，强制模型在后期进入极低学习率的微调状态
        # total_steps = args.total_epochs * len(abnormal_train_dataloader)

        # lr_scheduler = WarmCosineScheduler(
        #     optimizer, 
        #     base_value=1e-4, 
        #     final_value=1e-6, # 最终值设小，锁定特征
        #     total_iters=total_steps,
        #     warmup_iters=100
        # )

        print_fn('train image number:{}'.format(len(train_data)))

        # [LOSS-DISTILL] 训练用"原始库侧"教师库：只用该类的 K 张 shot（与推理同一个库，不扩充、不用测试集）
        _teacher_bank = None
        _teacher_idx = None
        if ((getattr(args, 'w_distill', 0.0) and float(args.w_distill) > 0) or
                (getattr(args, 'w_prod', 0.0) and float(args.w_prod) > 0)):
            with torch.no_grad():
                # 只用该类 **K 张原始 shot**（与推理同一个库；**不做增强、不扩充**），
                # 几何用训练几何（Resize(448)→CenterCrop(392)），与该 batch 的特征同尺度。
                import sys as _sys_
                _sys_.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from shot_utils import get_shots as _get_shots
                _picks = _get_shots(args.data_path, args.train_class, shots=args.shot, seed=args.seed)
                _root = os.path.join(args.data_path, args.train_class, 'train', 'good')
                _shots = torch.stack([data_transform(Image.open(os.path.join(_root, _f)).convert('RGB'))
                                      for _f in _picks], 0)
                _ret = model(_shots.to(device), is_training=True)
                _fp = torch.sum(torch.stack([_ret['patch_feat_layer'][j] for j in fused_layer], 0), 0)
                _teacher_bank = F.normalize(_fp, dim=-1).reshape(-1, _fp.shape[-1])   # [K*N, D]
                _M = _teacher_bank.shape[0]
                _g = torch.Generator().manual_seed(int(getattr(args, 'distill_sub_seed', 1234)))
                _teacher_idx = torch.randperm(_M, generator=_g)[:min(1024, _M)].to(device)
            print_fn(f'[LOSS-DISTILL] 教师原始库：{tuple(_teacher_bank.shape)}，'
                     f'子采样 {int(_teacher_idx.numel())}（仅用 K={args.shot} 张 shot：{_picks}）')


        cos_loss = ScaledPatchCosineLoss3()
        # [LOSS-EMA] 均值教师（Mean-Teacher）一致性：维护一份可训练参数的 EMA 副本，
        # 要求"学生"与"教师"的可学习侧分数图一致（逐图 z 后 L1）。
        # 动机：实测种子方差很大（15 类 mapping 极差 1.74、融合 1.18），
        # 一致性平滑是标准做法，且属于**纯损失侧**改动（不加数据、不动 bank）。
        _ema_model = None
        if float(getattr(args, 'w_ema', 0.0)) > 0:
            import copy as _copy
            _ema_model = _copy.deepcopy(model)
            for _p in _ema_model.parameters():
                _p.requires_grad_(False)
            _ema_model.eval()
            print_fn(f'[LOSS-EMA] 已建立 EMA 教师（decay={float(getattr(args,"ema_decay",0.999))}）')
        _ema_step = 0
        # [LOSS-PROX] 参数空间邻近项：把可训练参数拉向其"参数 EMA"（θ_ref），
        # 抑制 epoch 间质量摆动（实测内部 P-AUROC 极差 0.27~0.53、71~100% 的类最好轮≠35）。
        # 与 [LOSS-EMA]（输出一致）不同：本项作用在**参数空间**、不需要额外前向。
        _prox_ref = None
        if float(getattr(args, 'w_prox', 0.0)) > 0:
            _prox_ref = [p.detach().clone() for p in trainable.parameters()]
            print_fn(f'[LOSS-PROX] 已建立参数 EMA 锚点（decay={float(getattr(args,"prox_decay",0.99))}）')
        seg_criterion = FocalDiceLoss(dice_weight=0.5, focal_weight=0.5)
        for epoch in range(args.total_epochs):
            # [LOSS-CURRIC] 损失权重课程：w_rank 在训练中从 w_rank 线性变到 w_rank_final
            # （动机：M7 的"跨图可比性"约束在表征成熟后更有意义；稳定性诊断显示后期质量会摆动）
            if float(getattr(args, 'w_rank_final', -1.0)) >= 0 and args.total_epochs > 1:
                _t = epoch / float(args.total_epochs - 1)
                _w_rank_eff = args.w_rank + (float(args.w_rank_final) - args.w_rank) * _t
            else:
                _w_rank_eff = args.w_rank
            model.token_mapping.train()
            loss_list = []
            loss_cos_list = []
            loss_mse_list = []
            loss_tri_list = []
            loss_seg_list = []
            # for img, _ in tqdm(abnormal_train_dataloader, ncols=80):
            for datas in tqdm(abnormal_train_dataloader, ncols=80):
               target_imgs, _ = next(train_dataloader)

               target_imgs = target_imgs.to(device)
               aux_imgs = datas['img'].to(device)
               # [FIX-GRID] aux_mask 的格数必须跟随主干实际的 patch 格数，不能硬编码 784(28×28)。
               # DINOv2/patch14 在 392 裁剪下恰好是 28×28；DINOv3/patch16 不是。
               # 这里推迟到 aux 前向之后，用真实的 patch token 数下采样（见下方）。

               # 正常部分
               ret = model(target_imgs,is_training=True)
               patch_feat_layer, feat_list = ret['patch_feat_layer'], ret['feat_list']

               fused_patch_token_list = [patch_feat_layer[i] for i in fused_layer]
               fused_patch_token = torch.sum(torch.stack(fused_patch_token_list,dim=0), dim=0)
               # [FIX-GRID] 用真实 patch 格数下采样 GT 掩码（DINOv2=784，DINOv3/patch16=625）
               aux_mask = resize_mask(
                   datas['img_mask'], num_tokens=fused_patch_token.shape[1]).to(device)  # B N

               fused_learnable_token_list = [feat_list[i] for i in fused_layer_learnable]
               fused_learnable_token = torch.sum(torch.stack(fused_learnable_token_list,dim=0), dim=0)

               # [CONTROL] 保留原行为：用"所有 learnable 层求和"覆盖上面的选层结果
               fused_learnable_token = torch.cat(feat_list, dim=1)   # [MULTI-LAYER] 拼接各层 token
               # [vpt4.0 · M4] 双原型集：再拼上"全局（与图像无关）的可学习原型"，
               # 给原型集一个跨图固定的锚点 → 改善像素级 AUROC 的跨图校准
               if args.global_proto:
                   _g = model.tokens_per_layer.token_dict['0'].expand(fused_learnable_token.shape[0], -1, -1)
                   fused_learnable_token = torch.cat([fused_learnable_token, _g], dim=1)
               normal_patch_mapping = model.token_mapping(fused_patch_token)
               normal_learnable_mapping = model.token_mapping(fused_learnable_token)
               # 异常部分
               ret = model(aux_imgs,is_training=True)
               patch_feat_layer, feat_list = ret['patch_feat_layer'], ret['feat_list']

               fused_patch_token_list = [patch_feat_layer[i] for i in fused_layer]
               fused_patch_token = torch.sum(torch.stack(fused_patch_token_list,dim=0), dim=0)

               fused_learnable_token_list = [feat_list[i] for i in fused_layer_learnable]
               fused_learnable_token = torch.sum(torch.stack(fused_learnable_token_list,dim=0), dim=0)

               # [CONTROL] 同上，保留原行为
               fused_learnable_token = torch.cat(feat_list, dim=1)   # [MULTI-LAYER] 拼接各层 token
               # [vpt4.0 · FIX-M4] 异常分支也要拼接全局原型锚点：推理时（compute_fused_map）对
               # **测试图**是 `sum(feat_list) ⊕ token_dict['0']`（K=96），而这里若只留 K=64，
               # M5/M7 用的原型集就与线上推理不一致 → 损失在优化一个"线上不存在的"打分函数。
               if args.global_proto:
                   _ga = model.tokens_per_layer.token_dict['0'].expand(fused_learnable_token.shape[0], -1, -1)
                   fused_learnable_token = torch.cat([fused_learnable_token, _ga], dim=1)
               abnormal_patch_mapping = model.token_mapping(fused_patch_token)
               # [PUSH-REF] 异常分支自己的原型（供"与推理一致的每图自参照"使用）
               abnormal_learnable_mapping = model.token_mapping(fused_learnable_token)
               loss,loss_dict = cos_loss(normal_patch_mapping, normal_learnable_mapping,
                                         abnormal_patch_mapping, aux_mask,
                                         neg_proto=abnormal_learnable_mapping)
               # [CLUSTER · 机制1] 让正常 patch 在"可学习原型空间"里形成若干个簇：
               #   SimCLR 式的 softmax 目标会把每个 patch 拉向原型的**凸组合**（趋向均值），
               #   但正常 patch 本身是多样的（不同材质/纹理/部件），应当各自归属到某个簇。
               #   这里用 SwAV 的 Sinkhorn 平衡分配 Q 作为"硬一点的目标"：
               #     Q = sinkhorn(sim/eps)  → 行和 1（patch 明确归属）、列均衡（簇被等量使用）
               #     loss = CE(log_softmax(sim), Q)  → 让 patch 去预测这个平衡归属。
               #   --w_cluster 默认 0 → 与 v3.7 逐位一致。
               if float(getattr(args, 'w_cluster', 0.0)) > 0:
                   _Xc = torch.nn.functional.normalize(normal_patch_mapping, dim=-1)
                   _Pc = torch.nn.functional.normalize(normal_learnable_mapping, dim=-1)
                   _sim = torch.bmm(_Xc, _Pc.transpose(1, 2)) / float(args.cluster_temp)   # [B,N,K]
                   _Bc, _Nc, _Kc = _sim.shape
                   with torch.no_grad():
                       _Q = sinkhorn_assign(_sim.reshape(-1, _Kc),
                                            eps=float(args.cluster_eps),
                                            iters=int(args.cluster_iters))
                   _logp = torch.nn.functional.log_softmax(_sim.reshape(-1, _Kc), dim=-1)
                   loss_cluster = -(_Q * _logp).sum(-1).mean()
                   loss = loss + float(args.w_cluster) * loss_cluster
                   loss_dict['cluster'] = float(loss_cluster)
                   # [CLUSTER · 机制2] 跨增强的簇一致性：同一正常图的水平翻转视图，
                   # 其 patch 应当落到**同一簇**。目标 = 主视图的平衡分配 Q（detach），
                   # 学生 = 翻转视图的 log_softmax；空间上把 Q 按列翻转对齐 patch 顺序。
                   if float(getattr(args, 'w_cluster_cons', 0.0)) > 0:
                       _xf = torch.flip(target_imgs, dims=[3])
                       _rf = model(_xf, is_training=True)
                       _fpf = torch.sum(torch.stack([_rf['patch_feat_layer'][j] for j in fused_layer], 0), 0)
                       _flrf = torch.sum(torch.stack(_rf['feat_list'], 0), 0)
                       if args.global_proto:
                           _gf = model.tokens_per_layer.token_dict['0'].expand(_flrf.shape[0], -1, -1)
                           _flrf = torch.cat([_flrf, _gf], dim=1)
                       _Xf = torch.nn.functional.normalize(model.token_mapping(_fpf), dim=-1)
                       _Pf = torch.nn.functional.normalize(model.token_mapping(_flrf), dim=-1)
                       _simf = torch.bmm(_Xf, _Pf.transpose(1, 2)) / float(args.cluster_temp)
                       _gg = int(round(_sim.shape[1] ** 0.5))
                       _Qt = _Q.detach().reshape(_Bc, _gg, _gg, _Kc).flip(2).reshape(-1, _Kc)
                       _logpf = torch.nn.functional.log_softmax(_simf.reshape(-1, _Kc), dim=-1)
                       loss_cons = -(_Qt * _logpf).sum(-1).mean()
                       loss = loss + float(args.w_cluster_cons) * loss_cons
                       loss_dict['cluster_cons'] = float(loss_cons)

               # [vpt4.0 · M3] 原型多样性正则：把映射后的 K 个原型两两余弦中"过高"的部分拉低，
               # 让 32 个原型真正覆盖多个正常模式（实测原始版本它们两两余弦 0.87、有效自由度仅 2~3）
               if args.w_div > 0:
                   P = torch.nn.functional.normalize(normal_learnable_mapping, dim=-1)
                   C = torch.bmm(P, P.transpose(1, 2))                 # [B,K,K]
                   K = C.shape[1]
                   eye = torch.eye(K, device=C.device, dtype=torch.bool).unsqueeze(0)
                   off = C.masked_fill(eye, 0.0)
                   loss_div = torch.nn.functional.relu(off - args.div_margin).sum(dim=(1, 2)) / (K * (K - 1))
                   loss = loss + args.w_div * loss_div.mean()
                   loss_dict['div'] = float(loss_div.mean())
               abnor_pred = segmentation_head(abnormal_patch_mapping)       
                
               loss_seg = seg_criterion(abnor_pred,aux_mask)
             
            
               # [vpt4.0 · M5] 像素级直接监督：把"patch 到最近原型的余弦"直接当成异常分数，
               # 用辅助异常掩码做 BCE —— 让原型集在**像素级**上把正常/异常分开
               # （原配方只有 patch 级 hinge 与分割头，像素级排序指标 P-AP/P-AUROC 缺少直接梯度）
               if args.w_pixel > 0:
                   # 原型集必须与异常分支同 batch（正常分支 batch=batch_size//2=8，异常分支=16），
                   # 且要与推理时一致：用**该异常图自身**的可学习原型做自参照
                   # [vpt4.0 · REFACTOR] 改用公共打分口径函数（与 M7、与线上推理同一份实现）
                   _score = proto_distance_scores(abnormal_patch_mapping,
                                                  model.token_mapping(fused_learnable_token))  # [B,N]
                   _sim = 1.0 - _score                                                # 与最近原型相似度
                   _prob = ((1.0 - _sim) / 2.0).clamp(1e-6, 1.0 - 1e-6)               # 映射到 (0,1)
                   _tgt = aux_mask.float()
                   _bce = -(_tgt * torch.log(_prob) + (1 - _tgt) * torch.log(1 - _prob))
                   loss_px = _bce.mean()
                   loss = loss + args.w_pixel * loss_px
                   loss_dict['px'] = float(loss_px)
               # [vpt4.0 · M7] 跨图排序一致性损失：把"整类所有像素放一起排序"（P-AUROC 的定义）
               # 直接写进损失。正常像素 = 正常图全部像素 ∪ 辅助图里 mask=0 的像素；
               # 异常像素 = 辅助图里 mask=1 的像素。两侧都用**各自图自己的原型集**算分，
               # 因而优化的正是"不同图之间分数可比"这件事。
               if args.w_rank > 0:
                   loss_rank = cross_image_rank_loss_from_features(
                       normal_patch_mapping, normal_learnable_mapping,
                       abnormal_patch_mapping, model.token_mapping(fused_learnable_token),
                       aux_mask, margin=args.rank_margin, n_sample=args.rank_samples,
                       normalize=bool(args.rank_norm))
                   loss = loss + _w_rank_eff * loss_rank       # [LOSS-CURRIC] 支持权重课程
                   loss_dict['rank'] = float(loss_rank)
               # [LOSS-DISTILL] 教师=原始库侧打分（训练无关、确定性、只用 K shot），学生=可学习侧打分。
               # 只在"教师认为差异明显"的样本对上要求学生的排序一致（尺度无关的成对 hinge）：
               #   若 T_i − T_j 足够大，则要求 (S_j − S_i)/τ_S + margin ≤ 0
               # 目的：把"大而多样的真实正常 patch 库"携带的信息蒸馏进可学习度量空间，
               # 使 v3.7 的简化打分（s_learn = dlmin，无全局项）本身变得更有信息量。
               if _teacher_bank is not None:
                   with torch.no_grad():
                       _tn = F.normalize(fused_patch_token, dim=-1)
                       _d = (1.0 - torch.bmm(_tn, _teacher_bank.t().unsqueeze(0)
                                             .expand(_tn.shape[0], -1, -1))).clamp_min(0)
                       _d1 = _d.min(dim=-1).values
                       _ds = torch.sort(_d.index_select(-1, _teacher_idx), dim=-1).values
                       _qq = _ds[..., min(int(args.distill_global), _ds.shape[-1]) - 1]
                       T = _pow_geo(_d1, _qq, float(args.distill_alpha))          # [B,N]
                   S = proto_distance_scores(abnormal_patch_mapping,
                                             model.token_mapping(fused_learnable_token))  # [B,N]
                   _k = int(getattr(args, 'distill_topk', 16))
                   _k = max(1, min(_k, S.shape[1] // 2))
                   _t_hi = torch.topk(T, _k, dim=-1).values                    # 教师最高分
                   _t_lo = torch.topk(T, _k, dim=-1, largest=False).values     # 教师最低分
                   _i_hi = torch.topk(T, _k, dim=-1).indices
                   _i_lo = torch.topk(T, _k, dim=-1, largest=False).indices
                   _s_hi = torch.gather(S, 1, _i_hi)                           # [B,k]
                   _s_lo = torch.gather(S, 1, _i_lo)
                   _tau = torch.clamp(S.detach().std(unbiased=False), min=1e-8)
                   _gap = (_s_lo.unsqueeze(-1) - _s_hi.unsqueeze(-2)) / _tau \
                          + float(getattr(args, 'distill_margin', 0.5))
                   # 只在教师把两者分得开的对上生效（否则是噪声对）
                   _valid = (_t_hi.unsqueeze(-1) - _t_lo.unsqueeze(-2)) > \
                            float(getattr(args, 'distill_tau_ratio', 0.25)) * \
                            torch.clamp(T.detach().std(unbiased=False), min=1e-8)
                   _hinge = torch.relu(_gap) * _valid
                   loss_distill = _hinge.sum() / torch.clamp(_valid.sum(), min=1.0)
                   loss = loss + float(args.w_distill) * loss_distill
                   loss_dict['distill'] = float(loss_distill)
               # [LOSS-FLIP] 翻转一致性：同一张图与其水平翻转图的"可学习侧分数图"应当一致
               # （历史上测试期翻转 TTA 有 +0.5 量级的 mapping 收益；本项把该收益烧进权重，
               #   推理端仍然只用单视图，不违反"不多视图"约束）。
               if float(getattr(args, 'w_flip_cons', 0.0)) > 0:
                   _xf = torch.flip(aux_imgs, dims=[3])
                   _rf = model(_xf, is_training=True)
                   _fp_f = torch.sum(torch.stack([_rf['patch_feat_layer'][j] for j in fused_layer], 0), 0)
                   _flr_f = torch.sum(torch.stack(_rf['feat_list'], 0), 0)
                   if args.global_proto:
                       _gf = model.tokens_per_layer.token_dict['0'].expand(_flr_f.shape[0], -1, -1)
                       _flr_f = torch.cat([_flr_f, _gf], dim=1)
                   _S_f = proto_distance_scores(model.token_mapping(_fp_f),
                                                model.token_mapping(_flr_f))       # [B,N]
                   # 学生侧分数（与 M5/M7 同一口径）；不依赖蒸馏分支是否开启
                   _S = proto_distance_scores(abnormal_patch_mapping,
                                              model.token_mapping(fused_learnable_token))
                   _B, _N = _S.shape
                   _g = int(round(_N ** 0.5))
                   _a_map = _S.reshape(_B, 1, _g, _g)
                   _b_map = torch.flip(_S_f.reshape(_B, 1, _g, _g), dims=[3])       # 空间对齐
                   _z = lambda t: (t - t.mean(dim=(1, 2, 3), keepdim=True)) / \
                                  (t.std(dim=(1, 2, 3), keepdim=True) + 1e-8)
                   loss_flip = (_z(_a_map) - _z(_b_map)).abs().mean()
                   loss = loss + float(args.w_flip_cons) * loss_flip
                   loss_dict['flip_cons'] = float(loss_flip)
               # [LOSS-SCALE] 跨尺度一致性：同一张图在训练尺度(392)与 1.5×(588) 下的
               # "可学习侧分数图"应当一致（推理在 672=448×1.5，与训练尺度存在 1.7× 错配；
               # 测试期多尺度 TTA 实测有 +0.47 mapping 收益，本项把该收益烧进权重，推理仍单视图）。
               if float(getattr(args, 'w_scale_cons', 0.0)) > 0:
                   _su = float(getattr(args, 'scale_cons_scale', 1.5))
                   _xu = F.interpolate(aux_imgs, scale_factor=_su, mode='bilinear', align_corners=False)
                   _ru = model(_xu, is_training=True)
                   _fp_u = torch.sum(torch.stack([_ru['patch_feat_layer'][j] for j in fused_layer], 0), 0)
                   _flr_u = torch.sum(torch.stack(_ru['feat_list'], 0), 0)
                   if args.global_proto:
                       _gu = model.tokens_per_layer.token_dict['0'].expand(_flr_u.shape[0], -1, -1)
                       _flr_u = torch.cat([_flr_u, _gu], dim=1)
                   _S_u = proto_distance_scores(model.token_mapping(_fp_u),
                                                model.token_mapping(_flr_u))
                   _S0 = proto_distance_scores(abnormal_patch_mapping,
                                               model.token_mapping(fused_learnable_token))
                   _B2, _N2 = _S0.shape
                   _g2 = int(round(_N2 ** 0.5))
                   _gu2 = int(round(_S_u.shape[1] ** 0.5))
                   _m0 = _S0.reshape(_B2, 1, _g2, _g2)
                   _mu = F.interpolate(_S_u.reshape(_B2, 1, _gu2, _gu2), size=(_g2, _g2),
                                       mode='bilinear', align_corners=False)
                   _z2 = lambda t: (t - t.mean(dim=(1, 2, 3), keepdim=True)) / \
                                   (t.std(dim=(1, 2, 3), keepdim=True) + 1e-8)
                   loss_scale = (_z2(_m0) - _z2(_mu)).abs().mean()
                   loss = loss + float(args.w_scale_cons) * loss_scale
                   loss_dict['scale_cons'] = float(loss_scale)
               # [LOSS-PEAK] 分数图峰值/稀疏正则：直接对准最弱的 P-AP / P-F1（"高分段精度"）：
               #   异常图 → 要求 top-1% 像素显著高于中位数（尖峰）;
               #   正常图 → 要求两者接近（平）。
               # 只用到"合成异常属于异常、shot 属于正常"这一训练侧已知信息，不涉及测试集。
               if float(getattr(args, 'w_peak', 0.0)) > 0:
                   _S_a2 = proto_distance_scores(abnormal_patch_mapping,
                                                 model.token_mapping(fused_learnable_token))
                   _S_n2 = proto_distance_scores(normal_patch_mapping, normal_learnable_mapping)
                   _r = float(getattr(args, 'peak_top_ratio', 0.01))
                   _k2 = max(1, int(_S_a2.shape[1] * _r))
                   _tau2 = torch.clamp(torch.cat([_S_a2, _S_n2]).detach().std(unbiased=False),
                                       min=1e-8)
                   def _peak_gap(_S):
                       _top = torch.topk(_S, _k2, dim=-1).values.mean(dim=-1)
                       _med = _S.median(dim=-1).values
                       return (_top - _med) / _tau2
                   _mgn = float(getattr(args, 'peak_margin', 0.5))
                   loss_peak = torch.relu(_mgn - _peak_gap(_S_a2)).mean() + \
                               torch.relu(_peak_gap(_S_n2) - _mgn).mean()
                   loss = loss + float(args.w_peak) * loss_peak
                   loss_dict['peak'] = float(loss_peak)
               # [LOSS-EMA] 与 EMA 教师的一致性（仅可学习侧分数图；教师不做梯度）
               if _ema_model is not None:
                   with torch.no_grad():
                       _rt = _ema_model(aux_imgs, is_training=True)
                       _fp_t = torch.sum(torch.stack([_rt['patch_feat_layer'][j] for j in fused_layer], 0), 0)
                       _flr_t = torch.sum(torch.stack(_rt['feat_list'], 0), 0)
                       if args.global_proto:
                           _gt0 = _ema_model.tokens_per_layer.token_dict['0'].expand(_flr_t.shape[0], -1, -1)
                           _flr_t = torch.cat([_flr_t, _gt0], dim=1)
                       _S_t = proto_distance_scores(_ema_model.token_mapping(_fp_t),
                                                    _ema_model.token_mapping(_flr_t))
                   _S_s = proto_distance_scores(abnormal_patch_mapping,
                                                model.token_mapping(fused_learnable_token))
                   _Be, _Ne = _S_s.shape
                   _ge = int(round(_Ne ** 0.5))
                   _ze = lambda t: (t - t.mean(dim=(1, 2, 3), keepdim=True)) / \
                                   (t.std(dim=(1, 2, 3), keepdim=True) + 1e-8)
                   loss_ema = (_ze(_S_s.reshape(_Be, 1, _ge, _ge)) -
                               _ze(_S_t.reshape(_Be, 1, _ge, _ge))).abs().mean()
                   loss = loss + float(args.w_ema) * loss_ema
                   loss_dict['ema'] = float(loss_ema)
               # [LOSS-BAL] 原型使用均衡正则：把"正常 patch → 可学习原型"的软分配平均分布拉向均匀，
               # 避免少数原型被包办、其余闲置（实测 64 个原型高度冗余、有效自由度仅 2~3）。
               # 现有 L_div 只压"原型两两余弦过高"，不约束"使用是否均衡"，两者互补。
               if float(getattr(args, 'w_balance', 0.0)) > 0:
                   _P = F.normalize(normal_learnable_mapping, dim=-1)          # [B,K,D]
                   _Q = F.normalize(normal_patch_mapping, dim=-1)              # [B,N,D]
                   _sim = torch.bmm(_Q, _P.transpose(1, 2)) / float(getattr(args, 'balance_temp', 0.07))
                   _assign = torch.softmax(_sim, dim=-1)                       # [B,N,K]
                   _avg = _assign.mean(dim=(0, 1))                             # [K]
                   _unif = torch.full_like(_avg, 1.0 / _avg.numel())
                   loss_bal = torch.sum(_unif * (torch.log(_unif + 1e-8) -
                                                 torch.log(_avg + 1e-8)))       # KL(avg || uniform)
                   loss = loss + float(args.w_balance) * loss_bal
                   loss_dict['balance'] = float(loss_bal)
               # [LOSS-PROX] 参数邻近项：||θ − θ_ref||²（按参数量归一），θ_ref 为参数 EMA
               if _prox_ref is not None:
                   _ps = list(trainable.parameters())
                   # 按"参数自身尺度"归一 → 相对漂移（否则绝对平方和/参数量太小、梯度可忽略）
                   _num = sum(int(p.numel()) for p in _ps) + 1e-8
                   _scale = (sum((r ** 2).sum() for r in _prox_ref) / _num).detach() + 1e-12
                   loss_prox = sum(((p - r) ** 2).sum() for p, r in zip(_ps, _prox_ref)) / (_num * _scale)
                   loss = loss + float(args.w_prox) * loss_prox
                   loss_dict['prox'] = float(loss_prox)
               # [LOSS-PROD] **训推一致性**：直接监督"推理时真正使用的那个分数"
               #   推理端: m = w_learn·s_learn + (1-w_learn)·s_raw,  s_raw = d1^α·q^(1-α)（K shot 库）
               #   本项:   用合成掩码对 **同一个 m** 做 BCE（s_raw 无参数 → detach）
               # 与已有损失的差别：M5 只监督 s_learn（推理公式的 0.6 那一半），
               # 而推理实际用的是 m；此前训练从未直接优化过 m。
               if float(getattr(args, 'w_prod', 0.0)) > 0 and _teacher_bank is not None:
                   with torch.no_grad():
                       _tpn = F.normalize(fused_patch_token, dim=-1)
                       _dd = (1.0 - torch.bmm(_tpn, _teacher_bank.t().unsqueeze(0)
                                              .expand(_tpn.shape[0], -1, -1))).clamp_min(0)
                       _dd1 = _dd.min(dim=-1).values
                       _dds = torch.sort(_dd.index_select(-1, _teacher_idx), dim=-1).values
                       _dqq = _dds[..., min(int(args.distill_global), _dds.shape[-1]) - 1]
                       _raw = _pow_geo(_dd1, _dqq, float(args.distill_alpha))
                   _s_learn_p = proto_distance_scores(abnormal_patch_mapping,
                                                      model.token_mapping(fused_learnable_token))
                   _wl = float(getattr(args, 'prod_w_learn', 0.6))
                   if int(getattr(args, 'prod_znorm', 0)) > 0:
                       # [修复] 与推理端融合口径一致：两支各自标准化后再加权
                       # （推理是"类内 z 后 0.5/0.5"；训练按图内 z 近似），避免"原始距离直接混合"的尺度不一致
                       def _z_perimg(t):
                           m = t.mean(dim=1, keepdim=True)
                           s = t.std(dim=1, keepdim=True) + 1e-8
                           return (t - m) / s
                       _zs_l = _z_perimg(_s_learn_p)
                       _zs_r = _z_perimg(_raw)
                       _m_prod = _wl * _zs_l + (1.0 - _wl) * _zs_r
                       _prob_p = torch.sigmoid(_m_prod)          # 标准化后落在±几，用 sigmoid 映射到 (0,1)
                   else:
                       _m_prod = _wl * _s_learn_p + (1.0 - _wl) * _raw
                       _prob_p = (_m_prod / 2.0).clamp(1e-6, 1.0 - 1e-6)   # 与 M5 相同的映射
                   _tgt_p = aux_mask.float()
                   _bce_p = -(_tgt_p * torch.log(_prob_p) + (1 - _tgt_p) * torch.log(1 - _prob_p))
                   loss_prod = _bce_p.mean()
                   loss = loss + float(args.w_prod) * loss_prod
                   loss_dict['prod'] = float(loss_prod)
               # [LOSS-NEAR] 最近原型量化损失（训推一致的"正样本项"）：
               #   推理端可学习侧分数 = 到**最近**原型的距离 dlmin；
               #   现有正样本项只把 patch 拉向 softmax 加权原型（soft 目标），**从未直接优化最小值这一步**。
               #   本项对正常 patch 直接最小化 dlmin，并用 hinge(margin) 防止过度收缩（塌缩另有 L_div 抑制）。
               if float(getattr(args, 'w_near', 0.0)) > 0:
                   _Pn = F.normalize(model.token_mapping(fused_patch_token), dim=-1)
                   _Qn = F.normalize(model.token_mapping(fused_learnable_token), dim=-1)
                   _dn = (1.0 - torch.bmm(_Pn, _Qn.transpose(1, 2))).clamp_min(0).min(dim=-1).values
                   loss_near = torch.relu(_dn - float(getattr(args, 'near_margin', 0.05))).mean()
                   loss = loss + float(args.w_near) * loss_near
                   loss_dict['near'] = float(loss_near)
               # [LOSS-BARLOW] 映射特征空间去冗余（Barlow Twins 式）：
               #   实测"32 个可学习原型两两余弦 0.87、有效自由度仅 2~3"——即映射空间高度冗余。
               #   本项把正常 patch 的映射特征逐维标准化后求相关矩阵 C=ZᵀZ/M，
               #   惩罚非对角元（去相关）+ 对角偏离 1（保方差）。属纯表征正则，不改推理路径。
               if float(getattr(args, 'w_barlow', 0.0)) > 0:
                   _Z = normal_patch_mapping.reshape(-1, normal_patch_mapping.shape[-1])
                   _Z = (_Z - _Z.mean(0, keepdim=True)) / (_Z.std(0, keepdim=True) + 1e-6)
                   _C = torch.mm(_Z.t(), _Z) / max(1, _Z.shape[0])
                   _off = (_C ** 2).sum() - (_C.diagonal() ** 2).sum()
                   _on = ((_C.diagonal() - 1.0) ** 2).sum()
                   loss_barlow = _off / (_C.shape[0] * (_C.shape[0] - 1)) + \
                       float(getattr(args, 'barlow_lambda', 0.005)) * _on / _C.shape[0]
                   loss = loss + float(args.w_barlow) * loss_barlow
                   loss_dict['barlow'] = float(loss_barlow)
               # [LOSS-COMP] "补短板"损失（诊断驱动）：诊断发现"可学习侧单独提升 +3.09 也不传导到融合"，
               # 说明需要提升的是**互补性**而非单独质量。本项让可学习侧专门拟合**原始库侧判错的地方**：
               #   教师(原始库侧,无参)的错向响应 err = relu(-z_raw)·mask + relu(z_raw)·(1-mask)
               #   加权 BCE：权重 w = 1 + λ·err（在教师错得越明显的 patch 上学得越多）
               if float(getattr(args, 'w_comp', 0.0)) > 0 and _teacher_bank is not None:
                   with torch.no_grad():
                       _tc = F.normalize(fused_patch_token, dim=-1)
                       _dc = (1.0 - torch.bmm(_tc, _teacher_bank.t().unsqueeze(0)
                                              .expand(_tc.shape[0], -1, -1))).clamp_min(0)
                       _dc1 = _dc.min(dim=-1).values
                       _dcs = torch.sort(_dc.index_select(-1, _teacher_idx), dim=-1).values
                       _dcq = _dcs[..., min(int(args.distill_global), _dcs.shape[-1]) - 1]
                       _rawc = _pow_geo(_dc1, _dcq, float(args.distill_alpha))
                       _zr = (_rawc - _rawc.mean(dim=1, keepdim=True)) / (_rawc.std(dim=1, keepdim=True) + 1e-8)
                       _tg = aux_mask.float()
                       _err = (torch.relu(-_zr) * _tg + torch.relu(_zr) * (1.0 - _tg)).detach()
                   _sc = proto_distance_scores(abnormal_patch_mapping,
                                               model.token_mapping(fused_learnable_token))
                   _zc = (_sc - _sc.mean(dim=1, keepdim=True)) / (_sc.std(dim=1, keepdim=True) + 1e-8)
                   _pc = torch.sigmoid(_zc).clamp(1e-6, 1.0 - 1e-6)
                   _bce_c = -(_tg * torch.log(_pc) + (1 - _tg) * torch.log(1 - _pc))
                   _wgt = 1.0 + float(getattr(args, 'comp_lam', 1.0)) * _err
                   loss_comp = (_wgt * _bce_c).sum() / torch.clamp(_wgt.sum(), min=1.0)
                   if os.environ.get('DEBUG_COMP'):
                       print_fn(f'  [dbg-comp] sc nan={int(torch.isnan(_sc).sum())} '
                                f'zc nan={int(torch.isnan(_zc).sum())} pc nan={int(torch.isnan(_pc).sum())} '
                                f'tg nan={int(torch.isnan(_tg).sum())} bce nan={int(torch.isnan(_bce_c).sum())} '
                                f'w nan={int(torch.isnan(_wgt).sum())} '
                                f'pc_min={float(_pc.min()):.3e} pc_max={float(_pc.max()):.3e}')
                   loss = loss + float(args.w_comp) * loss_comp
                   loss_dict['comp'] = float(loss_comp)
               # [vpt4.0 · M10] 图像级跨图排序一致性：把"正常图的图像级分数 < 异常图的图像级分数"
               # 直接写进损失（图像级分数按评估口径 = 最高的 1% 像素均值），治图像级跨图漂移。
               if args.w_img_rank > 0:
                   _s_n = proto_distance_scores(normal_patch_mapping, normal_learnable_mapping)
                   _s_a = proto_distance_scores(abnormal_patch_mapping,
                                                model.token_mapping(fused_learnable_token))
                   loss_img_rank = image_level_rank_loss(
                       top_ratio_mean(_s_n), top_ratio_mean(_s_a),
                       margin=args.img_rank_margin, normalize=True)
                   loss = loss + args.w_img_rank * loss_img_rank
                   loss_dict['img_rank'] = float(loss_img_rank)
               loss += loss_seg
               
               optimizer.zero_grad()
               loss.backward()
               nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.5)
               optimizer.step()
               if _ema_model is not None:      # [LOSS-EMA] 更新教师（EMA）
                   _decay = float(getattr(args, 'ema_decay', 0.999))
                   with torch.no_grad():
                       for _pe, _ps in zip(_ema_model.parameters(), model.parameters()):
                           _pe.mul_(_decay).add_(_ps.detach(), alpha=1.0 - _decay)
                       for _be, _bs in zip(_ema_model.buffers(), model.buffers()):
                           _be.copy_(_bs)
                   _ema_step += 1
               if _prox_ref is not None:       # [LOSS-PROX] 更新参数 EMA 锚点（每步，慢速）
                   _pd = float(getattr(args, 'prox_decay', 0.99))
                   with torch.no_grad():
                       for _i, (_p, _r) in enumerate(zip(trainable.parameters(), _prox_ref)):
                           _r.mul_(_pd).add_(_p.detach(), alpha=1.0 - _pd)
               loss_list.append(loss.item())
               loss_cos_list.append(loss_dict['pos_cos'])
               loss_mse_list.append(loss_dict['pos_mse'])
               loss_tri_list.append(loss_dict['neg_push'])
               loss_seg_list.append(loss_seg.item())
               lr_scheduler.step()
               print_fn('epoch [{}/{}], total_loss:{:.4f}, cos_loss:{:.4f}, mse_loss:{:.4f}, neg_push_loss:{:.4f}, seg_loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list),np.mean(loss_cos_list),np.mean(loss_mse_list),np.mean(loss_tri_list),np.mean(loss_seg_list)))
               if args.w_pixel > 0 or args.w_rank > 0 or args.w_div > 0 or args.w_img_rank > 0 or float(getattr(args, 'w_cluster', 0.0)) > 0:
                   _aux = []
                   if args.w_div > 0:
                       _aux.append('div:{:.4f}'.format(loss_dict.get('div', float('nan'))))
                   if args.w_pixel > 0:
                       _aux.append('px:{:.4f}'.format(loss_dict.get('px', float('nan'))))
                   if args.w_rank > 0:
                       _aux.append('rank:{:.4f}'.format(loss_dict.get('rank', float('nan'))))
                   if args.w_img_rank > 0:
                       _aux.append('imgrank:{:.4f}'.format(loss_dict.get('img_rank', float('nan'))))
                   if float(getattr(args, 'w_cluster', 0.0)) > 0:
                       _aux.append('cluster:{:.4f}'.format(loss_dict.get('cluster', float('nan'))))
                   if getattr(args, 'w_distill', 0.0) and args.w_distill > 0:
                       _aux.append('distill:{:.4f}'.format(loss_dict.get('distill', float('nan'))))
                   if getattr(args, 'w_flip_cons', 0.0) and args.w_flip_cons > 0:
                       _aux.append('flipc:{:.4f}'.format(loss_dict.get('flip_cons', float('nan'))))
                   if getattr(args, 'w_scale_cons', 0.0) and args.w_scale_cons > 0:
                       _aux.append('scalec:{:.4f}'.format(loss_dict.get('scale_cons', float('nan'))))
                   if getattr(args, 'w_peak', 0.0) and args.w_peak > 0:
                       _aux.append('peak:{:.4f}'.format(loss_dict.get('peak', float('nan'))))
                   if getattr(args, 'w_ema', 0.0) and args.w_ema > 0:
                       _aux.append('ema:{:.4f}'.format(loss_dict.get('ema', float('nan'))))
                   if getattr(args, 'w_balance', 0.0) and args.w_balance > 0:
                       _aux.append('bal:{:.4f}'.format(loss_dict.get('balance', float('nan'))))
                   if getattr(args, 'w_prox', 0.0) and args.w_prox > 0:
                       _aux.append('prox:{:.3e}'.format(loss_dict.get('prox', float('nan'))))
                   if getattr(args, 'w_prod', 0.0) and args.w_prod > 0:
                       _aux.append('prod:{:.4f}'.format(loss_dict.get('prod', float('nan'))))
                   if getattr(args, 'w_near', 0.0) and args.w_near > 0:
                       _aux.append('near:{:.4f}'.format(loss_dict.get('near', float('nan'))))
                   if getattr(args, 'w_barlow', 0.0) and args.w_barlow > 0:
                       _aux.append('barlow:{:.3e}'.format(loss_dict.get('barlow', float('nan'))))
                   if getattr(args, 'w_comp', 0.0) and args.w_comp > 0:
                       _aux.append('comp:{:.4f}'.format(loss_dict.get('comp', float('nan'))))
                   print_fn('  [aux-loss] ' + ' '.join(_aux))
            # print_fn('epoch [{}/{}], total_loss:{:.4f}, cos_loss:{:.4f}, mse_loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list),np.mean(loss_cos_list),np.mean(loss_mse_list)))

            if (epoch + 1) % args.internal == 0 or (epoch + 1) % args.total_epochs == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                metric_names = ["I-Auroc", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
                header = "{:^12}".format("Class") + "".join(["{:^12}".format(name) for name in metric_names])
                print_fn(header)
                for item, test_data in zip(args.item_list, test_data_list):
                    if args.train_class is not None and args.train_class != item:
                        continue
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
                    if args.vpt3 and not args.map_only:
                        # ===== vpt3.0：改善后 mapping ⊕ 原版 MemoryBank（类内 z 加权 0.3/0.7）=====
                        if 'vpt3_raw_bank' not in globals():
                            vpt3_raw_bank = build_raw_bank(model, normal_imgs_for_bank, args.fused_layer, k=0)
                            print_fn(f'[vpt3.0] 原始特征空间 patch 库: {tuple(vpt3_raw_bank.shape)}')
                        _sc = tuple(float(x) for x in args.vpt3_scales.split(','))
                        _cache = args.vpt3_cache_dir or None
                        r1 = evaluation_batch_vpt3(model, test_dataloader, device, visual_bank_map,
                                                   vpt3_raw_bank, args.fused_layer,
                                                   w=args.vpt3_w, scales=_sc,
                                                   tta_flip=bool(args.vpt3_flip),
                                                   max_ratio=0.01, cache_dir=_cache,
                                                   class_name=item)
                        print_fn("{:^12}".format(item) + "".join(["{:^12.4f}".format(v) for v in r1]))
                        os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)
                        with open(os.path.join(args.save_dir, args.save_name, 'metrics_vpt3.csv'), 'a') as fh:
                            fh.write(f"{args.train_class},{epoch+1},vpt3_w{args.vpt3_w}," +
                                     ",".join(f"{v:.4f}" for v in r1) + "\n")
                        continue
                    if args.map_only:
                        # [CONTROL] 保留原行为：type=1 用 resize_mask=None（在 392 上算指标）
                        # [FIX-METRIC-ROBUST] 指标库偶发数值断言（实测 adeval._aupro 的 `assert area >= 0`）
                        # 会直接打断整类训练（m4m6_15/screw 在 epoch20 就被它杀掉了）。
                        # 这类失败只与"这一轮的指标计算"有关，与训练本身无关 → 跳过本轮指标、继续训练。
                        try:
                            r1 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01,
                                                  resize_mask=None, visual_bank_map=None, type=1)
                        except Exception as _me:
                            print_fn(f'  [WARN] {item} 第 {epoch+1} 轮指标评估失败'
                                     f'（{type(_me).__name__}: {_me}）；跳过本轮指标，训练继续')
                            continue
                        print_fn("{:^12}".format(item) + "".join(["{:^12.4f}".format(v) for v in r1]))
                        os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)
                        with open(os.path.join(args.save_dir, args.save_name, 'metrics_maponly.csv'), 'a') as fh:
                            fh.write(f"{args.train_class},{epoch+1},type1," +
                                     ",".join(f"{v:.4f}" for v in r1) + "\n")
                        continue
                    # results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    results1 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=None,visual_bank_map=visual_bank_map,type=1)
                    results2 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256,visual_bank_map=visual_bank_map,type=2)
                    results3 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256,visual_bank_map=visual_bank_map,type=3)
                    results4 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256,visual_bank_map=visual_bank_map,type=4)
                    results5 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256,visual_bank_map=visual_bank_map,type=5)
                    results6 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256,visual_bank_map=visual_bank_map,type=6)
                    results7 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256,visual_bank_map=visual_bank_map,type=7)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results1
                    auroc_sp2, ap_sp2, f1_sp2, auroc_px2, ap_px2, f1_px2, aupro_px2 = results2
                    auroc_sp3, ap_sp3, f1_sp3, auroc_px3, ap_px3, f1_px3, aupro_px3 = results3
                    auroc_sp4, ap_sp4, f1_sp4, auroc_px4, ap_px4, f1_px4, aupro_px4 = results4
                    auroc_sp5, ap_sp5, f1_sp5, auroc_px5, ap_px5, f1_px5, aupro_px5 = results5
                    auroc_sp6, ap_sp6, f1_sp6, auroc_px6, ap_px6, f1_px6, aupro_px6 = results6
                    auroc_sp7, ap_sp7, f1_sp7, auroc_px7, ap_px7, f1_px7, aupro_px7 = results7

                    # auroc_sp_list.append(auroc_sp)
                    # ap_sp_list.append(ap_sp)
                    # f1_sp_list.append(f1_sp)
                    # auroc_px_list.append(auroc_px)
                    # ap_px_list.append(ap_px)
                    # f1_px_list.append(f1_px)
                    # aupro_px_list.append(aupro_px)
                    # print_fn(
                    #     '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    #         item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                    row = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]])
                    print_fn(row)
                    
                    row2 = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp2, ap_sp2, f1_sp2, auroc_px2, ap_px2, f1_px2, aupro_px2]])
                    print_fn(row2)
                    
                    row3 = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp3, ap_sp3, f1_sp3, auroc_px3, ap_px3, f1_px3, aupro_px3]])
                    print_fn(row3)
                    
                    row4 = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp4, ap_sp4, f1_sp4, auroc_px4, ap_px4, f1_px4, aupro_px4]])
                    print_fn(row4)
                    
                    row5 = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp5, ap_sp5, f1_sp5, auroc_px5, ap_px5, f1_px5, aupro_px5]])
                    print_fn(row5)
                    
                    row6 = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp6, ap_sp6, f1_sp6, auroc_px6, ap_px6, f1_px6, aupro_px6]])
                    print_fn(row6)
                    
                    row7 = "{:^12}".format(item) + "".join(["{:^12.4f}".format(val) for val in [auroc_sp7, ap_sp7, f1_sp7, auroc_px7, ap_px7, f1_px7, aupro_px7]])
                    print_fn(row7)
                    

                # print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                #         np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                #         np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                # mean_values = [np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)]
                # mean_row = "{:^12}".format("Mean") + "".join(["{:^12.4f}".format(val) for val in mean_values])
                # print_fn(mean_row)
                # [主干实验] 默认每轮都存（save_every=1，与历史行为一致）；
                # 长轮数训练（如 70 epoch）时用 --save_every 降低落盘频率，避免磁盘被打爆。
                _se = max(1, int(getattr(args, 'save_every', 1)))
                if (epoch + 1) % _se == 0 or (epoch + 1) == args.total_epochs:
                    torch.save({"dino":model.state_dict(),"seg_head":segmentation_head.state_dict()}, os.path.join(args.save_dir, args.save_name, f'model_{args.train_class}_{epoch+1}.pth'))
