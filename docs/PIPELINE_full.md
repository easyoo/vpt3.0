# 全流程文档：从数据准备到推理评测（基于当前代码逐行核对）

> 生成时间：2026-10-13
> 依据：`code_vpt4/main-fewshot-vpt4.py`、`code_vpt4/dinov2/models/vision_transformer.py`、
> `code_vpt4/util/dataset.py`、`code_vpt2/util/memorybank.py`、`shot_utils.py`、
> `eval_score_v3.py`（当前推理）、`eval_nocrop.py`（旧口径）、`cache_bank_nocrop.py`、`fuse_nocrop.py`。
> 所有行号对应当前磁盘上的文件；**本文只描述现状，不做评价**（风险另见 §9）。

---

## 0. 全流程总览

```
[数据准备]
  MVTec-AD（官方目录）                      VisA（官方 → 转成 MVTec 式根）
        └── <cls>/train/good/*.png               └── /home/jjquan/experiment/visual2/visa_prepared/
                                                       <cls>/{train/good, test/good, test/anomaly, ground_truth}
  shot 清单：shot_utils.get_shots(data_path, cls, K, seed=1)  → .shots/<cls>_shotK_seed1.json
  合成异常：AnoStyler 用**同一批 K 张正常图**生成 35 张/图 → anomaly_samples_<...>/<mvtec|visa>/<cls>/src<stem>/{image,mask}
        │
        ▼
[训练] 每类一个模型（单类无监督 few-shot）
  正常分支：K 张 shot × 22 倍增强（旋转/平移/翻转/灰度/90°） → batch 8
  异常分支：K 个 src<stem> 目录 × 35 张合成图 = K×35 张        → batch 16
  冻结 DINOv2-B/14(reg4)，只训 3 个模块：tokens_per_layer / token_mapping / segmentation_head
  损失 = 0.5·L_cos + 1.0·L_mse + 1.0·L_push + 1.0·L_seg + 0.1·L_div + 0.02·L_rank
  35 轮；每 5 轮存一次权重 + 在测试集上算一次指标（392 口径）；交付用第 35 轮
        │
        ▼
[推理 · mapping 分支]  （eval_score_v3.py，当前生产版）
  测试图 Resize(448) 不裁剪 → 2 视图：1.5× 放大 与 水平翻转
  每个视图算 3 处分数：
    ① 可学习原型侧  dlmin → s_learn = dlmin^0.7 · bq64^0.3
    ② 原始特征库侧  d1     → s_raw   = d1^0.5   · q50^0.5
    （③ bank 分支另算，见 §5）
  融合：s = 0.6·s_learn + 0.4·s_raw → 插值回 448 → 2 视图平均 → 高斯平滑(k5,σ4)
        │
        ▼
[推理 · bank 分支]  （cache_bank_nocrop.py + code_vpt2/util/memorybank.py）
  K 张 shot × 22 倍增强 = 4-shot 88 张 → DINOv2-base 双 stage 特征 → faiss 库(90112/stage)
  测试图同特征 → 每 stage 取 1-NN 距离 → 32×32 图插值 448 → 两 stage 平均
        │
        ▼
[融合 + 评测]  （fuse_nocrop.py）
  类内 z：0.5·z(mapping) + 0.5·z(bank)（旧交付为 0.3/0.7）→ 高斯平滑 → 官方 adeval 7 指标
```

---

## 1. 代码地图

| 文件 | 职责 |
|---|---|
| `code_vpt4/main-fewshot-vpt4.py` | **全部训练 + 内置评估**；数据变换、数据集、损失、优化器、存档 |
| `code_vpt4/dinov2/models/vision_transformer.py` | 主干与三个可训练模块（`LearnableTokens` / `TokenMapping`）与自定义前向 |
| `code_vpt4/util/dataset.py` | `AnoStylerGeneratedDataset`（合成异常加载） |
| `code_vpt2/util/memorybank.py` | 原版 MemoryBank（faiss 建库与查询） |
| `shot_utils.py` | **三处共用的 shot 清单**（落盘 + 哈希校验） |
| `eval_score_v3.py` | **当前推理脚本**：新打分公式，落盘 mapping 异常图并出 mapping-only 指标 |
| `eval_nocrop.py` | 旧推理脚本（`1−max cos` 基线打分），用于对照 |
| `cache_bank_nocrop.py` | 建 bank 并落盘 bank 异常图 |
| `fuse_nocrop.py` | 融合两支异常图（类内 z 加权）并出三支指标 |
| `q_rotate_train_visa.sh` / `q_rotate_train_2shot.sh` / `q_rotate_train_v5.sh` | 训练驱动（轮换挑卡、写 manifest） |
| `visa_build_root.py` / `visa_build_anomaly_root.py` / `visa_prepare_shots.py` | VisA 数据准备 |

---

## 2. 数据准备

### 2.1 数据集根

* **MVTec-AD**：`/home/jjquan/MVTec-AD/<cls>/{train/good, test/<defect>, ground_truth/<defect>}`（官方结构，15 类）。
* **VisA**：官方 `/mnt/data/datasets/VisA`，由 `visa_build_root.py` 转成 MVTec 式根
  `/home/jjquan/experiment/visual2/visa_prepared/<cls>/{train/good, test/good, test/anomaly, ground_truth/anomaly}`（12 类）。
  * VisA 官方掩码是 `uint8 [0,1]`，该脚本**把掩码二值化重写为 0/255**（`write_mask_bin`：`(a>0)*255`），
    因为管线判定用 `(gt > 0.5)`，若保持 0/1 则 GT 全为空（会导致 `adeval` 的 `assert area >= 0` 崩溃）。
    已核实：原始掩码确为 0/1，掩码与原图几何一致（无转置）。
  * 图像用**软链接**，只有掩码是真实文件。

### 2.2 shot 清单（训练 / mapping 评估 / bank 建库三处共用）

`shot_utils.get_shots(data_path, cls, shots=K, seed=1)`：

```python
files = sorted(os.listdir(<data_path>/<cls>/train/good))     # 只取文件
fp = .shots/<cls>_shot{K}_seed{1}.json                        # 首次生成后落盘复用
random.seed(seed); picks = random.sample(files, shots)        # 首次
# 之后读盘，并校验 train/good 文件列表的 sha1 未变
```
落盘内容含 `picks / n_files / files_hash / data_path`，保证换 shot 数或换数据根时不会静默用错清单。

### 2.3 合成异常（AnoStyler）

* 配置：`anostyler_visa/{K}shot/configs/<idx>_<cls>_<stem>.yaml`，`num_gen: 35`、`defect: defect`。
* 产物：`anostyler_visa/{K}shot/genbyAnoStyler/<cls>/defect/<stem>/{image/gen_ano_i.jpg, mask/gen_mask_i.jpg}`。
* 训练用根（软链接）：`/home/jjquan/experiment/visual2/anomaly_samples_visa_<K>shot/visa/<cls>/src<stem>/{image,mask}`，
  MVTec 侧为 `/home/jjquan/experiment/visual2/anomaly_samples_2shot/mvtec/<cls>/<defect_type>/{image,mask}`。
* **合成异常的源图 = 同一份 shot 清单**（已逐项校验，见 `logs/visa/RESULTS_visa.md` §1）。

---

## 3. 训练（`code_vpt4/main-fewshot-vpt4.py`）

### 3.1 入口与超参（VisA 4-shot 的实测命令）

```bash
python code_vpt4/main-fewshot-vpt4.py --dataset VisA \
  --data_path /home/jjquan/experiment/visual2/visa_prepared \
  --anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples_visa_4shot \
  --train_class <cls> --shot 4 --total_epochs 35 --internal 5 --map_only \
  --batch_size 16 --seed 1 --token_init_std 0.02 --res_mapping 0 \
  --save_dir results/visa --save_name <save> \
  --global_proto 1 --w_div 0.1 --w_pixel 0.0 --metric_head 1 \
  --w_rank 0.02 --rank_margin 0.25 --rank_norm 1
```
MVTec 2-shot 用 `--shot 2` + `--anomaly_data_root .../anomaly_samples_2shot`，其余相同（`q_rotate_train_2shot.sh`）。

相关默认值（`--help` 未给的一律取代码默认）：`input_size=448`、`crop_size=392`、
`fused_layer=[1,2,3,4,6,8,11]`、`learnable_layers=1`、`learnable_num_per_layer=32`、
`temperature=0.07`、`scaled_factor=3`、`div_margin=0.5`、`rank_samples` 默认（见 §3.7）。

### 3.2 预处理（训练）

`get_data_transforms(input_size=448, crop_size=392)`：

```
图像：Resize((448,448)) → ToTensor → CenterCrop(392) → Normalize(ImageNet mean/std)
掩码：Resize((448,448)) → CenterCrop(392) → ToTensor        # 无 Normalize
```
⇒ **训练几何 = 448 缩放后中心裁 392**（保留面积 76.6%）。

### 3.3 正常分支数据

1. `ImageFolder(<data_path>/<cls>/train)`，再用 shot 清单过滤：`train_data.samples = [(p,i) for ... if basename(p) in picks]`，
   并 `assert len(_sel) == args.shot`（`main-fewshot-vpt4.py:1506-1522`）。
2. `augmentation(img)`（`main-fewshot-vpt4.py:106-129`）产生 **22 个变体**：
   原图 1 + 旋转 8（±π/16…±π/4）+ 平移 8（±0.1/±0.2）+ 水平翻转 1 + 灰度 1 + 旋转 90°×3 = 22。
   最后 `torch.randperm` 打乱顺序。
3. 打包成 `TensorDataset` → `ConcatDataset` → `DataLoader(batch_size=8, shuffle=True, drop_last=True)`，
   再用 `itertools.cycle` 无限循环（`main-fewshot-vpt4.py:1568-1570`）。
4. 同时保留 `normal_imgs_for_bank = stack(K 张未增强 shot)`，仅在不带 `--map_only` 时用于建 bank（`1571-1575`）。

### 3.4 异常分支数据

`AnoStylerGeneratedDataset(root_dir=<anomaly_root>/{mvtec|visa}, category=<cls>,
transform=data_transform, target_transform=gt_transform, image_suffix='.jpg',
samples_per_anomaly_type=35)`（`main-fewshot-vpt4.py:1630-1637`）：

* 遍历 `<category>/<anomaly_type>/{image,mask}`；图片名 `gen_ano_i.jpg` 对应掩码 `gen_mask_i.jpg`；
* 若某类型下样本数 > 35 则 `random.sample(..., 35)`（本例每类型恰好 35 张 → 全取）；
* 构造时 `random.shuffle(self.samples)` 一次；
* `DataLoader(batch_size=16, shuffle=True, drop_last=True)`。

⇒ 4-shot：4 个 `src<stem>` 类型 × 35 = **140 张/轮**，`drop_last` 后 8 个 iteration/epoch（与训练日志一致）。

### 3.5 模型结构

```
backbone: vision_transformer.vit_base(patch_size=14, img_size=518, block_chunks=0,
          init_values=1e-8, num_register_tokens=4, interpolate_antialias=False,
          interpolate_offset=0.1, arg=args)
预训练权重: /home/jjquan/Downloads/dinov2_vitb14_reg4_pretrain.pth   (strict=False)
```

三个可训练模块（其余全部冻结）：

| 模块 | 结构 | 初始化 |
|---|---|---|
| `tokens_per_layer` (`LearnableTokens`) | `learnable_layers=1` 层 × 32 个 token，`token_dict['0']`：[32,768] | `std=0.02`（`--token_init_std`） |
| `token_mapping` (`TokenMapping`) | `LayerNorm(768) → Linear(768→128) → GELU → LayerNorm(128) → Linear(128→768)`；`residual=0`、`metric=1` | 默认初始化 |
| `segmentation_head` | `Linear(768→384) → ReLU → Linear(384→1) → Sigmoid` | 默认（从头训） |

### 3.6 自定义前向的 token 组织（含一处既有的重复追加行为）

`DinoVisionTransformer.forward`（`vision_transformer.py:427-490`）：

1. `x = prepare_tokens_with_masks(x)` → `[B, L, 768]`，`L = 1(cls) + 4(reg) + N(patch)`；
2. **循环外**先 `x = cat([x, token_dict['0']])`（+32）——注释里明确写了这是**刻意保留的原项目行为**；
3. 逐 block（12 层）：
   * 若 `i < len(token_dict)`（本配置只有 `'0'`，即 `i==0`）：**再追加一次** `token_dict['0']`（再 +32）、跑 block、
     记录 `feat_list.append(x[:, -length:])`（本层被追加的 64 个 token）与 `patch_feat_layer.append(x[:, 5:l])`（patch tokens），
     然后把追加的 token 切掉（`x = x[:, :l]`）；
   * 否则：跑 block，只记录 patch tokens。

⇒ 因此 `feat_list` 只有 1 项、形状 `[B,64,768]`（`'0'` 被追加两次，前后各 32 个逐元素相同）；
`patch_feat_layer` 有 12 项，第 j 项是第 j 层输出的 patch tokens `[B,N,768]`。

### 3.7 损失函数（逐项）

设 `fp = Σ_{j∈[1,2,3,4,6,8,11]} patch_feat_layer[j]`，`flr = Σ_i feat_list[i]`，再拼上 `token_dict['0']`（+32）
⇒ 原型集 `K=96`（64 图像自适应 + 32 全局锚点）。

1) **`ScaledPatchCosineLoss3`**（`main-fewshot-vpt4.py:1088-1175`，权重见下）
   ```
   cosine_sim = cos(fp_norm, proto_norm)        # [B,N,96]
   weights    = softmax(cosine_sim / 0.07)      # 温度 0.07（可学习参数）
   soft_tokens= Σ weights · proto               # softmax 加权融合原型
   L_mse      = MSE(soft_tokens, fp)
   cos_dist   = 1 − cos(fp, soft_tokens)
   ratio      = clamp(cos_dist / mean(cos_dist), max=5)
   L_pos      = mean( ratio^3 · cos_dist )      # 动态缩放因子 3
   total      = 0.5·L_pos + 1.0·L_mse
   ```
2) **负样本 push**（同函数后半）：对**异常图**、
   `y_global = mean(proto over batch).detach()`，`sim = cos(f_aux, y_global)`，
   取 top-10 相似度求均值 → `relu(avg_topk_sim − 0.5)`，只在 `mask=1` 的像素上求均值 →
   权重 `1.0`。（= 要求异常像素与"全局平均原型"的相似度不超过 0.5）
3) **`L_div`**（`args.w_div=0.1`，`:1843-1852`）：对映射后的 96 个原型，惩罚两两余弦 > `div_margin=0.5` 的部分，
   按 `K(K−1)` 归一化 → 防止原型塌缩。
4) **`L_seg`**（`FocalDiceLoss`，权重 1.0，`loss += loss_seg`）：分割头输出（`abnormal_patch_mapping` 过 sigmoid）
   vs `resize_mask(datas['img_mask'])`；`dice_weight=0.5, focal_weight=0.5, alpha=0.25, gamma=2.0`。
   `resize_mask` 把 392×392 掩码用 `nearest` 下采样到 28×28 再展平（`main-fewshot-vpt4.py:1060+`）。
5) **`L_pixel`**（`--w_pixel`，**当前为 0.0 → 关闭**）。
6) **`L_rank`**（`args.w_rank=0.02`，`:1884-1891`）：`cross_image_rank_loss_from_features(...)`，
   正常像素 = 正常图全部 patch ∪ 辅助图 mask=0 的 patch；异常像素 = 辅助图 mask=1 的 patch；
   强制"最差的异常像素分数 > 最好的正常像素分数"的 pairwise hinge，`margin=0.25`，`n_sample=rank_samples=256`（默认），
   分数按 `rank_norm=1` 做图内归一化。
7) **`L_img_rank`**（`--w_img_rank`，**当前为 0.0 → 关闭**）。

**总损失**（本配置）：

```
L = 1.0·L_seg + 1.0·[0.5·L_pos + 1.0·L_mse] + 1.0·L_push + 0.1·L_div + 0.02·L_rank
    （w_pixel = 0, w_img_rank = 0）
```

### 3.8 优化器 / 调度器 / 梯度

```python
optimizer = StableAdamW(trainable.parameters(), lr=1e-3, betas=(0.9,0.999),
                        weight_decay=1e-3, amsgrad=True, eps=1e-10)
lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=5e-5,
                        total_iters=35 × len(abnormal_train_dataloader), warmup_iters=100)
# 每步：zero_grad → backward → clip_grad_norm_(trainable, max_norm=0.5) → step → scheduler.step()
```
`total_iters = 35 × 8 = 280`（4-shot）。`trainable = [tokens_per_layer, token_mapping, segmentation_head]`。

### 3.9 训练内评估与存档

* 每 `internal=5` 轮（5/10/…/35）：对该类测试集跑 `evaluation_batch(type=1)`，写入
  `results/<save>/metrics_maponly.csv`；异常时捕获跳过（不影响训练）。
  **注意该内部指标与最终报告指标不是同一个量**（`:609-643`）：
  * 打分函数是 `compute_anomaly_scores(patch_mapping, learnable_mapping)` = **min over 64 个原型**
    （`feat_list[[0]]`，**不含** `⊕ token_dict['0']` 的 32 个全局锚点）；
  * 几何是 **392 裁剪**（`resize_mask=None`，在 392 上算指标）；
  * **单视图**（无 1.5×/翻转 TTA），也**不含原始特征库项**。
  ⇒ 它只用于训练过程监控与"选哪一轮"，不能与 §6 的报告数字混用。
* 存档：`torch.save({"dino": model.state_dict(), "seg_head": segmentation_head.state_dict()},
  results/<save>/model_<cls>_<epoch>.pth)`（`:2032`），每 5 轮一个，共 7 个。
* **交付与本次实验统一使用第 35 轮**（`model_<cls>_35.pth`）。

---

## 4. 推理 · mapping 分支（当前生产版 `eval_score_v3.py`）

### 4.1 预处理（测试，**不裁剪**）

```
图像：Resize((448,448)) → ToTensor → Normalize(ImageNet)
GT  ：Resize((448,448)) → ToTensor      → (gt > 0.5) 二值化
```
DataLoader：`batch_size=16, shuffle=False, num_workers=4`。

### 4.2 双视图 TTA

`--scales 1.5 --flip 1` ⇒ 2 个视图：`F.interpolate(img, scale_factor=1.5)` (672×672) 与其水平翻转。
每个视图独立过 backbone，得到 48×48 的 patch 网格；每个视图的打分图插值回 448（`bilinear, align_corners=True`），
翻转视图再翻回来，最后两视图**算术平均**。

### 4.3 三处分数（逐视图）

设 `fp = Σ_{j∈fused_layer} patch_feat_layer[j]`，`flr = Σ feat_list ⊕ token_dict['0']`（96 原型），
`bank` = 该类 K 张 shot 的全部 patch 特征（**K×1024**：448 输入 → 32×32 patch 网格；4-shot = 4096 个），L2 归一化。

| # | 位置 | 距离定义 | 当前公式 |
|---|---|---|---|
| ① | **可学习原型侧** | `dlmin = min_{q∈96} (1 − cos( token_mapping(fp), token_mapping(flr) ))`；`db = 1 − cos(pl, normalize(token_mapping(bank)))`，取 `db` 的 1024-子采样第 64 小值 `bq64` | **`s_learn = dlmin^0.7 · bq64^0.3`** |
| ② | **原始特征库侧** | `d_j = 1 − cos(fp, bank_j)`；`d1 = min_j d_j`；`q50` = `d` 的 1024-子采样第 512 小值（中位数） | **`s_raw = d1^0.5 · q50^0.5`** |
| ③ | bank 分支 | 见 §5（独立网络与库） | `1 − sim(1-NN)`，两 stage 平均 |

* 子采样索引固定：`torch.Generator().manual_seed(1234); randperm(K*1024)[:1024]`（可用 `--sub_seed` 改）。
* 融合：`s = 0.6·s_learn + 0.4·s_raw`（`--w 0.6` 表示 **learn 侧权重**）。
* 旧版对照（`eval_nocrop.py`）：`s_learn = dlmin`（min），`s_raw = d1`（min），`s = 0.5·s_learn + 0.5·s_raw`。

### 4.4 后处理与图像级分数

```
gk = 5×5 高斯核，σ=4，padding=2（get_gaussian_kernel）
anomaly_map = gk( 平均后的 448 图 )                     # 平滑
image_score = mean( 该图 anomaly_map 中最高的 1% 像素 )    # 图像级分数
```

---

## 5. 推理 · bank 分支（原版 MemoryBank）

### 5.1 建库（`cache_bank_nocrop.py` + `code_vpt2/util/memorybank.py`）

1. 取该类的 K 张 shot（同一份 `shot_utils` 清单），每张过 `augmentation()` 得 22 个变体 ⇒ **4-shot = 88 张**，
   直接以 448 缩放（不裁剪）的 tensor 传入 `MemoryBank`。
2. `MemoryBank` 内部模型：`AutoModel.from_pretrained("facebook/dinov2-base", output_hidden_states=True)`，
   **不是** mapping 用的那个 finetune 主干。
3. `extract_fused_features(model, images, fuse_layer_encoder=[[2,3,4,5],[6,7,8,9]])` → 两段融合特征（stage-1 / stage-2），
   L2 归一化后各建一个 faiss `IndexFlatIP`（GPU）索引；4-shot 时每索引 **90112 = 88×32×32** 个向量。

### 5.2 查询（测试）

```python
qf = extract_fused_features(bank.model_t, test_batch)      # 2 个 stage
for feat, idx in zip(qf, [index1, index2]):
    d, _ = idx.search(feat.reshape(-1,C), k=1)             # 1-NN（硬编码 k=1）
    dist = 1 - d                                           # use_cos=True
    err_map = dist.reshape(B, 32, 32)
stage1, stage2 = interpolate(err_map, size=448, bilinear)
bank_map = (stage1 + stage2) / 2
```
落盘为 `.errmap_nocrop448*/<cls>.map`（未平滑，`[N,1,448,448]`）。

> 注：`query_image()`（按路径查询）里还有一段 `stage /= stage.max()` 的**逐图归一化**，
> 但**批量推理走的是 `query_batch()`，没有这一步**，所以落盘图是原始距离。

---

## 6. 融合与评测（`fuse_nocrop.py` + `adeval`）

### 6.1 融合

```python
M = load(<map_dir>/<cls>.map)          # mapping 图（未平滑）
B = load(<bank_dir>/<cls>.map)         # bank 图（未平滑）
z(x) = (x - x.mean()) / (x.std() + 1e-8)      # ★ 统计量取自**该类全部测试像素**
fused = 0.5*z(M) + 0.5*z(B)            # 旧交付为 0.3/0.7（w 是 mapping 权重）
然后 gk(fused) → 图像分数 top-1% → adeval
```
GT 与标签重新用 `MVTecDataset(phase='test')` 以**同一顺序**加载（脚本内有 `assert M.shape == B.shape`）。

### 6.2 指标定义（`main-fewshot-vpt4.py:459-495`）

| 指标 | 实现 |
|---|---|
| I-AUROC | `sklearn.roc_auc_score(gt_sp, pr_sp)` |
| I-AP | `sklearn.average_precision_score` |
| I-F1 | `f1_score_max`：按分数排序扫描取最大 F1 |
| P-AUROC | `adeval.EvalAccumulatorCuda` 的 `p_auroc`，**200 条阈值带**覆盖 `[anomap_min, anomap_max]` |
| P-AP | 同上 `p_aupr` |
| P-F1 | 在所有像素上做 `f1_score_max` |
| P-AUPRO | `p_aupro`：PRO 曲线在 **FPR ≤ 0.3** 区间的积分（`adeval/iterative.py:136-147`） |

**7 项平均** = 上述 7 个数的**未加权算术平均**（本项目自定义的汇总口径，非论文标准）。

已实测：对异常图做任何**严格单调**变换（平方/开方/log/×10），7 项指标变化 ≤0.0024
⇒ 该评测在本项目实现下基本等价于排序类指标。

---

## 7. 端到端命令（可复制）

```bash
cd /home/jjquan/experiment/visual/audit_mapping && source ./common_env.sh

# --- 训练（VisA 4-shot 示例）---
SHOTS=4 EXTRA_ARGS="--global_proto 1 --w_div 0.1 --w_pixel 0.0 --metric_head 1 \
 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1" \
 bash q_rotate_train_visa.sh vpt35_visa4shot

# --- 推理：mapping（当前生产版，落盘 + mapping-only 指标）---
$PY eval_score_v3.py --save vpt35_visa4shot --class candle --shots 4 --size 448 \
  --data_path /home/jjquan/experiment/visual2/visa_prepared \
  --ckpt_dir results/visa/vpt35_visa4shot \
  --raw_alpha 0.5 --raw_global 512 --l_alpha 0.7 --l_global 64 --w 0.6 \
  --save_maps_dir .maps_v3_nocrop448_visa4

# --- 推理：bank ---
$PY cache_bank_nocrop.py --class candle --shots 4 --size 448 \
  --data_path /home/jjquan/experiment/visual2/visa_prepared --out_dir .errmap_nocrop448_visa4

# --- 融合 + 评测（三支指标）---
$PY fuse_nocrop.py --class candle --size 448 --w 0.5 \
  --map_dir .maps_v3_nocrop448_visa4 --bank_dir .errmap_nocrop448_visa4 \
  --data_path /home/jjquan/experiment/visual2/visa_prepared
```

---

## 8. 与论文对比时的口径差异（必须声明）

1. **分辨率/裁剪**：训练 `Resize(448)+CenterCrop(392)`；测试 `Resize(448)` **不裁剪**。
   多数论文用 256×256（或 448）裁剪口径，**不可直接排名**。
2. **训练数据**：本项目在 K 张正常图之外，**额外使用 AnoStyler 合成异常**；标注为 few-shot 的论文多数只用正常图。
3. **PRO**：本项目用 `adeval` 的 `p_aupro`（FPR ≤ 0.3、200 条带），与论文的自实现 PRO 可能有细微差别。
4. **7 项平均**是本项目自定义汇总，论文无此列。
5. **bank 分支**是"K shot × 22 倍增强"的密集库，属于本项目的工程选择，论文基线多为原版 PatchCore/MemoryBank 设置。

---

## 9. 实现细节风险点（供你把控，非评价）

| # | 位置 | 现象 | 影响面 |
|---|---|---|---|
| R1 | `MVTecDataset.load_dataset`（`:175`） | `defect_types = os.listdir(...)` **未排序**，`glob` 每类内部有序 | 测试集图像顺序 = 文件系统顺序；所有本仓库脚本共用同一类故内部一致，但任何"假定按类别名排序"的第三方脚本会错位 |
| R2 | 训练损失用 `proto_distance_scores`（min） | 推理（v3）在原始库侧改用 `d1^0.5·q50^0.5` | **训练/推理打分不一致**：模型是按 min 训练的，新公式只在推理端生效 |
| R3 | `:431-432` 前向 | `token_dict['0']` 在循环外与 `i=0` 各追加一次（注释称为刻意保留） | `feat_list` 为 64（32 重复），原型集因此是 96 而非 64；影响所有训练与推理一致性（两侧一致） |
| R4 | `fuse_nocrop.py` | 类内 z 的均值/方差取自**测试集**全部像素 | 属 transductive；基线与新方法共用，A/B 公平，但绝对数字非纯归纳式 |
| R5 | `eval_score_v3.py` | `q50/bq64` 由 1024-子采样估计，种子固定 1234 | 换种子实测 4 类均值漂移 ≤0.20（单类最大 0.54） |
| R6 | 训练内评估 | `evaluation_batch(type=1)`：392 裁剪、单视图、**64 原型（无全局锚点）**、只用可学习侧 min；**与最终报告指标（96 原型 / 新公式 / 448 不裁剪 / 2 视图 / 含原始库项）不是同一个量** | 只用于 `metrics_maponly.csv` 监控与"选第几轮"；若按它选轮次，与报告指标并不对齐 |
| R7 | `MemoryBank` | 只用于 bank 分支，其 `query_batch` 硬编码 `k=1` | 若要用分位数类打分需改成 `k>1`（已单独试验，见 `logs/SCORING_SEARCH_2026-10-13.md` §6.1） |
| R8 | 全局随机性 | `shot_utils` 在建清单时调用 `random.seed(seed)`，之后 `AnoStylerGeneratedDataset` 复用同一全局 RNG | shot 清单固定，但合成异常的 35 张"子采样顺序"依赖此后的 RNG 消耗路径 |
