# 严格 MVTec 4-shot 数据说明

本目录只保存 shot 清单和 source manifest，不包含图像、mask 或 checkpoint。

- `shots_mvtec4_strict/`：15 类严格 4-shot 清单及文件列表哈希。
- `mvtec4_strict_existing_manifest.json`：每类 4 个 AnoStyler source、异常类别和 35 对库存记录。

实际图像路径：

```text
/home/jjquan/MVTec-AD
/home/jjquan/AnoStyler/genbyAnoStyler
```

训练数据视图与异常根可通过以下脚本重建：

```bash
python tools/build_mvtec4_strict_existing.py
```
