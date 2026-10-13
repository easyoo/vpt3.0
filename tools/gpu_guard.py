# -*- coding: utf-8 -*-
"""GPU 绑定守卫：确认当前 CUDA 可见的第一张卡就是"指定的那张物理卡"。

为什么需要它（2026-09-29 事故）：
  本机 GPU0 长期处于 "Unknown Error"（驱动枚举异常）。当按**数字索引**设置
  `CUDA_VISIBLE_DEVICES=1` 时，CUDA 的枚举会把断掉的 GPU0 排除在外，
  于是"1"实际落在**物理 GPU2**（`nvidia-smi` 编号）上——实测 torch 报告的 UUID 是
  3bf0407c-...（= GPU2），而不是我们以为的 GPU1。
  而 GPU2 当时正被同事的训练任务占用（6.4GB / 99% util）。
  ⇒ 仅凭 CUDA_VISIBLE_DEVICES 的数字索引**无法**保证不打扰同事；
     必须用 UUID 绑定，并在进程内断言 UUID 与预期一致。

用法：
  EXPECTED_GPU_UUID=GPU-0e3a7e1c-... python gpu_guard.py     # 校验并打印
  from gpu_guard import require_gpu; require_gpu()           # 在训练脚本里调用
"""
import os
import sys


def _norm(u):
    # torch.cuda.get_device_properties(0).uuid 返回的是 torch._C._CUuuid 对象（不是 str），
    # 必须显式转换，否则 .strip() 会 AttributeError（实测踩到）。
    return str(u or '').strip().replace('GPU-', '').lower()


def require_gpu(expected=None, verbose=True):
    expected = expected or os.environ.get('EXPECTED_GPU_UUID', '')
    try:
        import torch
    except Exception as e:                                     # pragma: no cover
        raise SystemExit(f'[gpu_guard] 无法导入 torch: {e}')
    if not torch.cuda.is_available():
        raise SystemExit('[gpu_guard] CUDA 不可用：请检查驱动/是否指定了坏卡（见 RUN_STATUS.md）')
    got = torch.cuda.get_device_properties(0).uuid
    name = torch.cuda.get_device_name(0)
    if verbose:
        print(f'[gpu_guard] 当前 cuda:0 = {name} uuid={got}')
    if expected and _norm(got) != _norm(expected):
        raise SystemExit(
            '[gpu_guard] 中止：绑定的物理卡与预期不一致！\n'
            f'  预期 {expected}\n  实际 {got}\n'
            '  这通常是因为用了数字索引 CUDA_VISIBLE_DEVICES（本机 GPU0 异常会打乱枚举）。\n'
            '  请改用 UUID 形式，例如 CUDA_VISIBLE_DEVICES=GPU-0e3a7e1c-...')
    return got


if __name__ == '__main__':
    require_gpu()
    print('[gpu_guard] 校验通过')
