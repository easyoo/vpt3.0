# -*- coding: utf-8 -*-
"""绘制"异常分数计算"流程图（依据 eval_score_v3.py / memorybank.py / fuse_nocrop.py 实际代码）。
横向泳道布局；输出 PNG + SVG。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch    # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
W, H = 116, 178
fig, ax = plt.subplots(figsize=(23.2, 35.6), dpi=100)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
FILL = {'ref': '#e6f4ea', 'trunk': '#e8f0fe', 'raw': '#fff4e5',
        'lrn': '#f3e8fd', 'bank': '#fde8e8', 'post': '#f0f0f0', 'fuse': '#e8f7fa'}


def box(x, y, w, h, text, kind, fs=13.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.2",
                                fc=FILL[kind], ec='#3a3a3a', lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, linespacing=1.6)


def arr(x1, y1, x2, y2, label='', color='#333', ls='-', rad=0.0, fs=11.5, lw=1.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18,
                                 lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 1.0, label, ha='center', va='bottom',
                fontsize=fs, color=color)


def lane(y, title, num):
    ax.text(1.0, y - 2.6, f"{num}  {title}", ha='left', va='center', fontsize=16,
            fontweight='bold', color='#111')
    ax.plot([1, W - 1], [y - 0.6, y - 0.6], color='#bbb', lw=1.0, ls=':')


ax.text(W / 2, 4.5, "异常分数计算全流程（逐算子 · 依据 eval_score_v3.py / memorybank.py / fuse_nocrop.py）",
        ha='center', va='center', fontsize=21, fontweight='bold')
ax.text(W / 2, 9.0, "示例常量（VisA 4-shot）：K=4；视图 672×672 → N=48×48=2304；参考库 4096=4×1024；"
                    "原型集 96=64 图像自适应 + 32 全局锚点",
        ha='center', va='center', fontsize=12, color='#555')

lane(16, "参考库构建（每类一次，训练后固定；推理只读）", "①")
box(1.5, 17, 25, 11, "K 张 shot（shot_utils 清单, seed=1）\nResize(448,448)→ToTensor→Normalize", 'ref')
box(29.5, 17, 25, 11, "过同一个 finetune 主干\nΣ patch_feat_layer[1,2,3,4,6,8,11]\n"
                      "每张 → 32×32 = 1024 个 patch", 'ref')
box(57.5, 17, 25, 11, "bank = [K×1024, 768]\n（4-shot = [4096, 768]）\n"
                      "q_raw = L2norm(bank) → [4096,768]", 'ref')
box(85.5, 17, 29, 11, "q_bankL = L2norm(token_mapping(bank)) → [4096,768]\n"
                      "sub = randperm(4096)[:1024]（torch 种子 1234）", 'ref')
arr(26.5, 22.5, 29.5, 22.5); arr(54.5, 22.5, 57.5, 22.5); arr(82.5, 22.5, 85.5, 22.5)

lane(40, "测试图 → 双视图 → 主干（每视图独立前向）", "②")
box(1.5, 41, 25, 11, "每类全部测试图\nResize(448,448)→Normalize\n（测试不裁剪）→ [B,3,448,448]", 'trunk')
box(29.5, 41, 25, 11, "视图 1：interpolate ×1.5 → [B,3,672,672]\n"
                      "视图 2：视图 1 的 flip(3)\n⇒ 2 视图，patch 网格 48×48", 'trunk')
box(57.5, 41, 25, 11, "每视图独立过主干（model.eval）\npatch_feat_layer : 12 × [B,2304,768]\n"
                      "feat_list : [B,64,768]", 'trunk')
box(85.5, 41, 29, 11, "fp  = Σ patch_feat_layer[1,2,3,4,6,8,11] → [B,N,768]\n"
                      "flr2 = Σ feat_list ⊕ token_dict['0'] → [B,96,768]", 'trunk')
arr(26.5, 46.5, 29.5, 46.5); arr(54.5, 46.5, 57.5, 46.5); arr(82.5, 46.5, 85.5, 46.5)

lane(64, "原始特征库侧（打分 ①）：局部 min × 全库分位数", "③")
box(1.5, 65, 25, 11, "p = L2norm(fp) → [B,N,768]\nd = (1 − p · q_raw^T).clamp_min(0)\n→ [B,N,4096]", 'raw')
box(29.5, 65, 25, 11, "d1 = d.min(-1) → [B,N]\n（对全部 4096 个取最小）", 'raw')
box(57.5, 65, 25, 11, "d_s = sort(d[:,:,sub], -1) → [B,N,1024]\nq50 = d_s[..., 511]\n"
                      "（子采样中位数）", 'raw')
box(85.5, 65, 29, 11, "s_raw = clamp(d1,e)^0.5 · clamp(q50,e)^0.5\ne = 1e-6 → [B,N]\n"
                      "（0.5 / 0.5 几何平均）", 'raw')
arr(26.5, 70.5, 29.5, 70.5); arr(54.5, 70.5, 57.5, 70.5); arr(82.5, 70.5, 85.5, 70.5)

lane(88, "可学习原型侧（打分 ②）：局部 min × 映射库分位数", "④")
box(1.5, 89, 25, 11, "pl = L2norm(token_mapping(fp)) → [B,N,768]\n"
                     "ql = L2norm(token_mapping(flr2)) → [B,96,768]", 'lrn')
box(29.5, 89, 25, 11, "dl = (1 − pl · ql^T).clamp_min(0) → [B,N,96]\ndlmin = dl.min(-1) → [B,N]", 'lrn')
box(57.5, 89, 25, 11, "db = (1 − pl · q_bankL^T) → [B,N,4096]\nd_b = sort(db[:,:,sub]) → [B,N,1024]\n"
                      "bq64 = d_b[..., 63]（≈6.25% 分位）", 'lrn')
box(85.5, 89, 29, 11, "s_learn = clamp(dlmin,e)^0.7 · clamp(bq64,e)^0.3\n→ [B,N]", 'lrn')
arr(26.5, 94.5, 29.5, 94.5); arr(54.5, 94.5, 57.5, 94.5); arr(82.5, 94.5, 85.5, 94.5)

arr(36, 28.5, 14, 64.5, color='#0b6b3a', ls='--', rad=-0.30, lw=1.2)
ax.text(15.0, 34.0, '推理时读取 q_raw / sub', fontsize=10.5, color='#0b6b3a', ha='left')
arr(96, 28.5, 70, 88.5, color='#0b6b3a', ls='--', rad=0.26, lw=1.2)
ax.text(88.0, 33.6, '推理时读取 q_bankL / sub', fontsize=10.5, color='#0b6b3a', ha='left')

lane(112, "视图内融合与空间还原", "⑤")
box(1.5, 113, 53, 12, "输入：③ 的 s_raw + ④ 的 s_learn\n"
                      "m = w · s_learn + (1−w) · s_raw　（w = 0.6 = 可学习侧权重）\n"
                      "reshape → [B,1,48,48]（在 patch 网格上融合，尚未插值）", 'trunk')
box(59.5, 113, 55, 12, "F.interpolate(m, (448,448), bilinear, align_corners=True) → [B,1,448,448]\n"
                       "若该视图是翻转视图 → am = am.flip(3) 翻回原空间\n"
                       "2 个视图算术平均 → raw_map = [N,1,448,448]", 'trunk')
arr(54.5, 119, 59.5, 119)

lane(132, "落盘、平滑与指标", "⑥")
box(1.5, 133, 25, 11, "torch.save → .maps_v3_nocrop448_*/\n<cls>.map\n（未平滑，供融合复用）", 'post')
box(29.5, 133, 25, 11, "gk = get_gaussian_kernel(5, σ=4)\nanomaly_map = gk(raw_map)\n→ [N,1,448,448]", 'post')
box(57.5, 133, 25, 11, "图像级分数 = 最高 1% 像素均值\n（2007 / 200704）→ [N]", 'post')
box(85.5, 133, 29, 11, "adeval：I-AUROC/I-AP/I-F1（sklearn）\nP-AUROC/P-AP/P-F1 + AUPRO(FPR≤0.3)\n"
                       "「7 项平均」= 未加权算术平均", 'post')
arr(26.5, 138.5, 29.5, 138.5); arr(54.5, 138.5, 57.5, 138.5); arr(82.5, 138.5, 85.5, 138.5)

lane(156, "bank 分支（独立网络，非 mapping 主干）", "⑦")
box(1.5, 157, 25, 11, "K 张 shot × augmentation 22 变体/张\n= 88 张（4-shot）→ 448 不裁剪", 'bank')
box(29.5, 157, 25, 11, "facebook/dinov2-base（未训练）\n层组 [[2,3,4,5],[6,7,8,9]]\n→ stage-1 / stage-2", 'bank')
box(57.5, 157, 25, 11, "各 stage L2norm → faiss.IndexFlatIP\n90112 = 88×32×32 / stage\n"
                       "查询 search(q, k=1) → dist = 1 − 内积", 'bank')
box(85.5, 157, 29, 11, "reshape 32×32 → interpolate 448\n两 stage 平均 → "
                       ".errmap_nocrop448*/<cls>.map（未平滑）", 'bank')
arr(26.5, 162.5, 29.5, 162.5); arr(54.5, 162.5, 57.5, 162.5); arr(82.5, 162.5, 85.5, 162.5)

box(1.5, 171, 113, 6.0,
    "融合（fuse_nocrop.py）：z(x) = (x − μ)/σ，μ,σ 取自该类全部测试像素 → "
    "fused = w·z(mapping) + (1−w)·z(bank)，w = 0.5（mapping 权重）\n"
    "→ 再 gk 平滑 → 图像分数 top-1% → adeval → 输出三行 CSV：mapping / bank / fused",
    'fuse', fs=13.0)

fig.savefig('logs/score_flow_mapping.png', bbox_inches='tight', facecolor='white')
fig.savefig('logs/score_flow_mapping.svg', bbox_inches='tight', facecolor='white')
print('saved logs/score_flow_mapping.png / .svg')
