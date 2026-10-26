# VPT3.9 Unified Fusion：MVTec-AD / VisA 1/2/4-shot 完整性能对比

日期：2026-10-26

## 方法

统一融合方法在 MVTec-AD 和 VisA 上使用完全相同的权重公式：

```text
c = cosine(中心区域平均特征, 边缘区域平均特征)
r = 中心区域平均特征范数 / 边缘区域平均特征范数

w_m = clip((1-c) · r, 0, 1)
w_b = 1-w_m
```

最终异常图使用 9、25、41 三个区域池化尺度：

```text
S = 1/3 · Σ_{k∈{9,25,41}}
        [w_m · box_k(Mapping) + w_b · box_k(Bank)]
```

三个尺度只对同一张异常图做不同池化，不增加网络前向次数。`c` 和 `r` 只由
K-shot 正常 support 图像计算，不使用测试集统计量或数据集专用阈值。

## 性能对比

Common-4 为三种方法都报告的指标均值：

```text
(I-AUROC + I-AP + P-AUROC + P-AUPRO) / 4
```

<table>
  <thead>
    <tr>
      <th rowspan="2">Shot</th>
      <th rowspan="2">Method</th>
      <th colspan="8" align="center" style="text-align: center;">MVTec-AD</th>
      <th style="width: 24px;"></th>
      <th colspan="8" align="center" style="text-align: center;">VisA</th>
    </tr>
    <tr>
      <th align="right">I-AUROC</th>
      <th align="right">I-AP</th>
      <th align="right">I-F1</th>
      <th align="right">P-AUROC</th>
      <th align="right">P-AP</th>
      <th align="right">P-F1</th>
      <th align="right">P-AUPRO</th>
      <th align="right">Common-4</th>
      <th style="width: 24px;"></th>
      <th align="right">I-AUROC</th>
      <th align="right">I-AP</th>
      <th align="right">I-F1</th>
      <th align="right">P-AUROC</th>
      <th align="right">P-AP</th>
      <th align="right">P-F1</th>
      <th align="right">P-AUPRO</th>
      <th align="right">Common-4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">1-shot</td>
      <td>SubspaceAD</td>
      <td align="right"><u>97.1</u></td><td align="right"><u>98.6</u></td><td align="right">-</td><td align="right"><u>97.5</u></td><td align="right">-</td><td align="right">-</td><td align="right"><u>94.5</u></td><td align="right"><b>96.9</b></td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right"><u>93.2</u></td><td align="right">93.0</td><td align="right">-</td><td align="right"><u>98.2</u></td><td align="right">-</td><td align="right">-</td><td align="right"><u>95.5</u></td><td align="right"><u>95.0</u></td>
    </tr>
    <tr>
      <td>DCP-SFR</td>
      <td align="right"><b>97.3</b></td><td align="right"><b>98.8</b></td><td align="right"><u>96.7</u></td><td align="right">96.9</td><td align="right"><u>61.2</u></td><td align="right"><u>60.7</u></td><td align="right">92.7</td><td align="right">96.4</td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right">92.0</td><td align="right"><u>93.7</u></td><td align="right"><u>87.8</u></td><td align="right"><u>98.2</u></td><td align="right"><u>40.4</u></td><td align="right"><u>44.9</u></td><td align="right">93.3</td><td align="right">94.3</td>
    </tr>
    <tr>
      <td><b>Unified Fusion</b></td>
      <td align="right">95.6</td><td align="right">98.1</td><td align="right"><b>97.3</b></td><td align="right"><b>97.9</b></td><td align="right"><b>65.7</b></td><td align="right"><b>64.6</b></td><td align="right"><b>94.6</b></td><td align="right"><u>96.6</u></td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right"><b>94.4</b></td><td align="right"><b>94.7</b></td><td align="right"><b>90.9</b></td><td align="right"><b>98.8</b></td><td align="right"><b>47.5</b></td><td align="right"><b>52.3</b></td><td align="right"><b>95.5</b></td><td align="right"><b>95.9</b></td>
    </tr>
    <tr>
      <td rowspan="3">2-shot</td>
      <td>SubspaceAD</td>
      <td align="right"><u>97.5</u></td><td align="right"><u>98.8</u></td><td align="right">-</td><td align="right"><u>97.8</u></td><td align="right">-</td><td align="right">-</td><td align="right"><u>94.9</u></td><td align="right"><b>97.3</b></td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right"><u>93.8</u></td><td align="right">93.4</td><td align="right">-</td><td align="right">98.3</td><td align="right">-</td><td align="right">-</td><td align="right"><u>95.7</u></td><td align="right"><u>95.3</u></td>
    </tr>
    <tr>
      <td>DCP-SFR</td>
      <td align="right"><b>97.7</b></td><td align="right"><b>98.9</b></td><td align="right"><u>97.0</u></td><td align="right">97.1</td><td align="right"><u>61.7</u></td><td align="right"><u>61.6</u></td><td align="right">93.3</td><td align="right">96.8</td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right">92.5</td><td align="right"><b>94.2</b></td><td align="right"><u>88.0</u></td><td align="right"><u>98.4</u></td><td align="right"><u>40.6</u></td><td align="right"><u>46.9</u></td><td align="right">93.8</td><td align="right">94.7</td>
    </tr>
    <tr>
      <td><b>Unified Fusion</b></td>
      <td align="right">96.0</td><td align="right">98.4</td><td align="right"><b>97.6</b></td><td align="right"><b>98.0</b></td><td align="right"><b>67.4</b></td><td align="right"><b>66.4</b></td><td align="right"><b>94.9</b></td><td align="right"><u>96.8</u></td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right"><b>93.8</b></td><td align="right"><u>94.2</u></td><td align="right"><b>90.5</b></td><td align="right"><b>98.9</b></td><td align="right"><b>49.0</b></td><td align="right"><b>53.1</b></td><td align="right"><b>95.7</b></td><td align="right"><b>95.6</b></td>
    </tr>
    <tr>
      <td rowspan="3">4-shot</td>
      <td>SubspaceAD</td>
      <td align="right"><u>98.0</u></td><td align="right"><u>99.0</u></td><td align="right">-</td><td align="right"><u>97.9</u></td><td align="right">-</td><td align="right">-</td><td align="right"><u>95.1</u></td><td align="right"><u>97.5</u></td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right"><u>94.7</u></td><td align="right">94.3</td><td align="right">-</td><td align="right">98.4</td><td align="right">-</td><td align="right">-</td><td align="right"><u>96.0</u></td><td align="right"><u>95.9</u></td>
    </tr>
    <tr>
      <td>DCP-SFR</td>
      <td align="right"><u>98.0</u></td><td align="right">98.9</td><td align="right"><u>96.8</u></td><td align="right">97.5</td><td align="right"><u>61.8</u></td><td align="right"><u>61.9</u></td><td align="right">93.5</td><td align="right">97.0</td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right">92.8</td><td align="right"><u>94.7</u></td><td align="right"><u>88.9</u></td><td align="right"><u>98.6</u></td><td align="right"><u>40.6</u></td><td align="right"><u>47.1</u></td><td align="right">94.2</td><td align="right">95.1</td>
    </tr>
    <tr>
      <td><b>Unified Fusion</b></td>
      <td align="right"><b>98.3</b></td><td align="right"><b>99.1</b></td><td align="right"><b>97.6</b></td><td align="right"><b>98.0</b></td><td align="right"><b>69.4</b></td><td align="right"><b>68.0</b></td><td align="right"><b>95.4</b></td><td align="right"><b>97.7</b></td>
      <td style="width: 24px;">&nbsp;</td>
      <td align="right"><b>95.1</b></td><td align="right"><b>95.1</b></td><td align="right"><b>91.6</b></td><td align="right"><b>99.0</b></td><td align="right"><b>49.5</b></td><td align="right"><b>53.9</b></td><td align="right"><b>96.3</b></td><td align="right"><b>96.4</b></td>
    </tr>
  </tbody>
</table>

## 说明

- 加粗为每个 shot、每个数据集指标列内最高值；下划线为第二高值。
- 缺失值不参与排名；同值时并列使用相同排名。
- SubspaceAD 未报告 I-F1、P-AP、P-F1，因此不能计算完整七项均值。
- SubspaceAD 使用论文五 seed 均值；DCP-SFR 和 Unified Fusion 为论文或本项目的单点结果。
- MVTec 4-shot 使用 strict 4-shot；MVTec 2-shot 与 VisA 4-shot checkpoint 为
  VPT3.8-equivalent。
- Unified Fusion 为单 seed，尚未完成多 seed 方差估计。

## 结果文件

- `logs/crossfit_normal_reliability_20261026/mvtec1_eval_continuous_multiscale.csv`
- `logs/crossfit_normal_reliability_20261026/mvtec2_eval_multiscale.csv`
- `logs/crossfit_normal_reliability_20261026/mvtec4_eval_continuous_multiscale.csv`
- `logs/crossfit_normal_reliability_20261026/visa1_eval_unified_multiscale.csv`
- `logs/crossfit_normal_reliability_20261026/visa2_eval_unified_multiscale.csv`
- `logs/crossfit_normal_reliability_20261026/visa4_eval_unified_multiscale.csv`
