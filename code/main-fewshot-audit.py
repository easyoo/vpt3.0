import warnings
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

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
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
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
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

    def forward(self, x, y, neg=None, mask=None):
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

            # y_global 是所有原型的全局参考 [1, K_proto, D]
            y_global = y_normalized.detach().mean(dim=0, keepdim=True)

            # 计算相似度矩阵: [B_aux, N_patch, K_proto]
            sim_neg = torch.einsum('bnd,ckd->bnk', neg_normalized, y_global)

            # --- 修改部分：从 max 改为 top-k ---
            # 这里的 k 不能超过 y_global 的原型总数 K_proto
            topk_val = min(10, y_global.size(1))

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


            total_loss += 1.0 * loss_neg
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    # model info
    parser.add_argument('--learnable_layers',type=int, default=1)
    parser.add_argument('--learnable_num_per_layer',type=int, default=32)

    parser.add_argument('--fused_layer',type=list,default=[1,2,3,4,6,8,11])
    parser.add_argument('--fused_layer_learnable',type=list,default=[0])
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
    parser.add_argument('--map_only', action='store_true',
                        help='只跑 mapping 分支（type=1）：不建 bank、不读 errmap、评估只算 type=1')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    # save info
    parser.add_argument('--save_dir', type=str, default='./mvtec_saved_results0')
    parser.add_argument('--save_name', type=str, default='4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add')
    
    parser.add_argument('--pth_epoch', type=str, default='5')
    parser.add_argument('--vis_dir', type=str, default='visualization')
    
    args = parser.parse_args()
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

    
    data_transform, gt_transform = get_data_transforms(args.input_size,args.crop_size)
    train_data_list = []
    test_data_list = []

    
    
    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):

            test_path = os.path.join(args.data_path, item)
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            test_data_list.append(test_data)

            if args.train_class is not None and args.train_class != item:
                continue
            
            train_path = os.path.join(args.data_path, item, 'train')
            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            # Randomly select shot samples per class
            train_data.samples = [(sample[0], i) for sample in random.sample(train_data.samples, args.shot)]
            
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
        
        model = vision_transformer.__dict__['vit_base'](patch_size=14, img_size=518,block_chunks=0, init_values=1e-8,num_register_tokens=4,interpolate_antialias=False,interpolate_offset=0.1,arg=args)
        model.to(device)
        segmentation_head = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 2, 1), # 输出单通道，用于二分类
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
        # [AUDIT-FIX-3] 原路径 /home/jjquan/codebase/vpt/checkpoint/... 已不存在，训练阶段会直接崩。
        # 改成按顺序找可用权重（Downloads 下那份是同名同源权重）。
        _pth_candidates = [
            '/home/jjquan/codebase/vpt/checkpoint/dinov2_vitb14_reg4_pretrain.pth',
            '/home/jjquan/Downloads/dinov2_vitb14_reg4_pretrain.pth',
            '/home/jjquan/experiment/visual2_v2/dinov2_vitb14_reg4_pretrain.pth',
        ]
        pth = next((p for p in _pth_candidates if os.path.exists(p)), _pth_candidates[1])
        print_fn(f'[audit] 预训练权重: {pth}')
        model = vision_transformer.__dict__['vit_base'](patch_size=14, img_size=518,block_chunks=0, init_values=1e-8,num_register_tokens=4,interpolate_antialias=False,interpolate_offset=0.1,arg=args)
        state_dict = torch.load(pth, map_location='cpu',weights_only=True)
        model.to(device)
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
            nn.Linear(768, 768 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 2, 1), # 输出单通道，用于二分类
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


        cos_loss = ScaledPatchCosineLoss3()
        seg_criterion = FocalDiceLoss(dice_weight=0.5, focal_weight=0.5)
        for epoch in range(args.total_epochs):
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
               aux_mask = resize_mask(datas['img_mask']).to(device) # B N

               # 正常部分
               ret = model(target_imgs,is_training=True)
               patch_feat_layer, feat_list = ret['patch_feat_layer'], ret['feat_list']

               fused_patch_token_list = [patch_feat_layer[i] for i in fused_layer]
               fused_patch_token = torch.sum(torch.stack(fused_patch_token_list,dim=0), dim=0)

               fused_learnable_token_list = [feat_list[i] for i in fused_layer_learnable]
               fused_learnable_token = torch.sum(torch.stack(fused_learnable_token_list,dim=0), dim=0)

               # [AUDIT-FIX-2] 原代码这里有一句 `fused_learnable_token = sum(stack(feat_list))`，
               # 把上面按 fused_layer_learnable 选层的写法覆盖成"所有 learnable 层求和"。
               # learnable_layers=1 时它恰好是 no-op（feat_list 只有 1 项），所以过去没暴露；
               # 一旦用多个 learnable 层，它会与 evaluate 端的取层方式不一致（且 stack 会因形状不同直接崩）。
               # 这里删掉，保证训练与评估用的是同一个 fused_learnable_token。
               normal_patch_mapping = model.token_mapping(fused_patch_token)
               normal_learnable_mapping = model.token_mapping(fused_learnable_token)

               # 异常部分
               ret = model(aux_imgs,is_training=True)
               patch_feat_layer, feat_list = ret['patch_feat_layer'], ret['feat_list']

               fused_patch_token_list = [patch_feat_layer[i] for i in fused_layer]
               fused_patch_token = torch.sum(torch.stack(fused_patch_token_list,dim=0), dim=0)

               fused_learnable_token_list = [feat_list[i] for i in fused_layer_learnable]
               fused_learnable_token = torch.sum(torch.stack(fused_learnable_token_list,dim=0), dim=0)

               # [AUDIT-FIX-2] 同上（异常分支这句其实是死代码，一并删掉以免误导）
               abnormal_patch_mapping = model.token_mapping(fused_patch_token)
               loss,loss_dict = cos_loss(normal_patch_mapping, normal_learnable_mapping,abnormal_patch_mapping,aux_mask)
               abnor_pred = segmentation_head(abnormal_patch_mapping)       
                
               loss_seg = seg_criterion(abnor_pred,aux_mask)
             
            
               loss += loss_seg
               
               optimizer.zero_grad()
               loss.backward()
               nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.5)
               optimizer.step()
               loss_list.append(loss.item())
               loss_cos_list.append(loss_dict['pos_cos'])
               loss_mse_list.append(loss_dict['pos_mse'])
               loss_tri_list.append(loss_dict['neg_push'])
               loss_seg_list.append(loss_seg.item())
               lr_scheduler.step()
            print_fn('epoch [{}/{}], total_loss:{:.4f}, cos_loss:{:.4f}, mse_loss:{:.4f}, neg_push_loss:{:.4f}, seg_loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list),np.mean(loss_cos_list),np.mean(loss_mse_list),np.mean(loss_tri_list),np.mean(loss_seg_list)))
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
                    if args.map_only:
                        # [AUDIT-FIX-4/6] 只评估 mapping 分支；resize_mask 统一成 256，
                        # 原来 type=1 那行用的是 resize_mask=None（在 392 上算），
                        # 与其它 type 以及测试阶段（都用 256）口径不一致，这里统一。
                        r1 = evaluation_batch(model, test_dataloader, device, max_ratio=0.01,
                                              resize_mask=256, visual_bank_map=None, type=1)
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
                torch.save({"dino":model.state_dict(),"seg_head":segmentation_head.state_dict()}, os.path.join(args.save_dir, args.save_name, f'model_{args.train_class}_{epoch+1}.pth'))
