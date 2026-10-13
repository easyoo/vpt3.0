# -*- coding: utf-8 -*-
"""[审计工具] 挑一张**空闲 GPU**并输出其 UUID。

设计要点（2026-10-08 修订；用户要求：每个任务跑完换卡，别长时间占同一张）：
  1. 空闲判据：显存占用 < --max_used_mib（默认 3000 MiB）且利用率 < --max_util（默认 20%）；
  2. **轮换**：同一个调度脚本（--owner $$）连续挑卡时，若存在两张及以上空闲卡，
     优先换一张（上次用过的卡记在 .gpu_claims/last_<owner>），避免长期压在一张物理卡上；
  3. **原子占用声明**：用 O_CREAT|O_EXCL 在 .gpu_claims/<uuid>.json 建声明，内容含 owner
     （调度脚本 pid）。多进程同时挑卡时只有一个能抢到，杜绝"两个任务挤同一张卡"；
  4. **存活检测**：声明仅在"owner 进程仍存活 且 未超过 --hold 秒"时算数；owner 已退出
     或超时的陈旧声明会被自动清理，不会卡住后续任务；
  5. 成功换卡时自动释放本 owner 的旧声明；
  6. 无空闲卡 → 休眠 --sleep 秒（默认 120 = 2 分钟）后重试；--once 则立即退出码 1。

用法：
  U=$(python pick_gpu.py --owner $$)          # 阻塞直到拿到一张卡，打印 UUID
  CUDA_VISIBLE_DEVICES=$U python train.py
  python pick_gpu.py --release $U             # 任务结束释放（owner 退出后也会自动失效）
  python pick_gpu.py --status                 # 查看各卡占用 / 声明情况
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLAIM_DIR = os.environ.get('PICK_GPU_CLAIM_DIR') or os.path.join(HERE, '.gpu_claims')


def _claim_path(uuid):
    return os.path.join(CLAIM_DIR, uuid + '.json')


def _last_path(owner):
    return os.path.join(CLAIM_DIR, f'last_{owner}')


def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_claim(uuid):
    try:
        with open(_claim_path(uuid)) as fh:
            rec = json.load(fh)
    except (FileNotFoundError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def claim_alive(rec, hold, grace=300):
    """声明是否仍然有效。

    有效条件（满足其一）：
      * owner 进程仍存活，且声明未超过 hold 秒；
      * owner 已退出，但声明还很新（距创建不超过 grace 秒）——用于保护
        "U=$(pick_gpu); python train.py" 这种一次性启动：挑卡的 shell 可能先退出。
    """
    if not rec:
        return False
    age = time.time() - float(rec.get('ts', 0))
    if age > hold:
        return False
    if pid_alive(rec.get('owner')):
        return True
    return age <= grace


def occupied_by_others(uuid, hold, owner, grace=300):
    """该卡是否被别人占用（顺手清理陈旧声明）。"""
    rec = read_claim(uuid)
    if not rec:
        return False
    if not claim_alive(rec, hold, grace):
        try:
            os.unlink(_claim_path(uuid))
        except OSError:
            pass
        return False
    return str(rec.get('owner')) != str(owner)


def try_claim(uuid, owner, tag, hold, grace=300):
    """原子抢占该卡，成功返回 True。"""
    os.makedirs(CLAIM_DIR, exist_ok=True)
    payload = json.dumps({'owner': str(owner), 'uuid': uuid, 'tag': tag,
                          'pid': os.getpid(), 'ts': time.time()})
    for _ in range(6):
        try:
            fd = os.open(_claim_path(uuid), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, 'w') as fh:
                fh.write(payload)
            return True
        except FileExistsError:
            if occupied_by_others(uuid, hold, owner, grace):
                return False                      # 真被别人占着 → 放弃这张
            try:                                  # 自己的旧声明 / 陈旧声明 → 删除后重抢
                os.unlink(_claim_path(uuid))
            except OSError:
                pass
    return False


def release_claim(uuid):
    try:
        os.unlink(_claim_path(uuid))
        return True
    except OSError:
        return False


def release_own_except(owner, keep):
    """释放本 owner 除 keep 以外的所有声明（换卡时调用）。"""
    if not os.path.isdir(CLAIM_DIR):
        return
    for fn in os.listdir(CLAIM_DIR):
        if not fn.endswith('.json'):
            continue
        uuid = fn[:-5]
        if uuid == keep:
            continue
        rec = read_claim(uuid)
        if rec and str(rec.get('owner')) == str(owner):
            release_claim(uuid)


def _prio_path(owner):
    return os.path.join(CLAIM_DIR, f'prio_{owner}')


def set_priority_request(owner, tag, on):
    """声明/撤销"高优先级请求"（评估这类短任务用）。

    机制：高优先级任务在**等待期间**留下 prio 标记；普通任务（训练）在下次挑卡时看到标记就主动让路
    （继续休眠），于是下一次有人释放卡时，卡会落到高优先级任务手里，而不是被长任务立刻抢回去。
    标记同样带 owner 存活/宽限判定，高优先级进程挂了不会永久挡住训练。
    """
    os.makedirs(CLAIM_DIR, exist_ok=True)
    p = _prio_path(owner)
    if on:
        try:
            with open(p, 'w') as fh:
                fh.write(json.dumps({'owner': str(owner), 'tag': tag, 'ts': time.time(),
                                     'pid': os.getpid()}))
        except OSError:
            pass
    else:
        try:
            os.unlink(p)
        except OSError:
            pass


def active_priority_requests(hold=1800, grace=120):
    """当前待处理的 prio 标记（陈旧自动清理）。"""
    out = []
    if not os.path.isdir(CLAIM_DIR):
        return out
    for fn in os.listdir(CLAIM_DIR):
        if not fn.startswith('prio_'):
            continue
        p = os.path.join(CLAIM_DIR, fn)
        try:
            with open(p) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if claim_alive(rec, hold, grace):
            out.append(rec)
        else:
            try:
                os.unlink(p)
            except OSError:
                pass
    return out


def read_last(owner):
    try:
        with open(_last_path(owner)) as fh:
            return fh.read().strip()
    except OSError:
        return ''


def write_last(owner, uuid):
    os.makedirs(CLAIM_DIR, exist_ok=True)
    try:
        with open(_last_path(owner), 'w') as fh:
            fh.write(uuid)
    except OSError:
        pass


def list_gpus():
    """返回 [{'idx','uuid','used','util'}]；故障卡（Unknown Error）不会出现在结果里。"""
    fake = os.environ.get('PICK_GPU_FAKE_LIST')      # 仅供自测：uuid,usedMiB,util;uuid,usedMiB,util
    if fake:
        out = []
        for i, item in enumerate(fake.split(';')):
            u, used, total, util = [x.strip() for x in item.split(',')]
            out.append(dict(idx=str(i), uuid=u, used=int(used), total=int(total),
                            util=int(util), shared=False))
        return out
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=60).stdout
    except Exception as e:                                     # pragma: no cover
        print(f'[pick_gpu] nvidia-smi 调用失败: {e}', file=sys.stderr)
        return []
    gpus = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) < 4:
            continue
        if len(parts) < 5:
            continue
        idx, uuid, used, total, util = parts[:5]
        if not uuid.startswith('GPU-'):
            continue
        try:
            gpus.append(dict(idx=idx, uuid=uuid, used=int(used), total=int(total),
                             util=int(util), shared=False))
        except ValueError:
            continue
    return gpus


def show_status(hold, grace=300):
    gpus = list_gpus()
    print(f"{'idx':>3} {'uuid':<40} {'used':>7} {'total':>7} {'free':>7} {'util%':>6}  占用声明")
    for g in gpus:
        rec = read_claim(g['uuid'])
        if rec and claim_alive(rec, hold, grace):
            note = (f"owner={rec.get('owner')} tag={rec.get('tag')} "
                    f"age={int(time.time() - float(rec.get('ts', 0)))}s")
        else:
            note = '-'
        print(f"{g['idx']:>3} {g['uuid']:<40} {g['used']:>7} {g['total']:>7} "
              f"{g['total']-g['used']:>7} {g['util']:>6}  {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_used_mib', type=int, default=3000, help='显存占用不超过该值才算空闲')
    ap.add_argument('--max_util', type=int, default=20, help='利用率不超过该值才算空闲')
    ap.add_argument('--allow_shared', dest='allow_shared', action='store_true', default=True,
                    help='没有整卡空闲时，允许在"剩余显存 >= min_free_mib"的卡上与同事共存（默认开）')
    ap.add_argument('--no-shared', dest='allow_shared', action='store_false',
                    help='严格模式：只使用整卡空闲的卡')
    ap.add_argument('--min_free_mib', type=int, default=12000,
                    help='共存模式要求的最小剩余显存（默认 12GB，远大于我们单任务约 3GB 的需求）')
    ap.add_argument('--sleep', type=int, default=120, help='无空闲卡时的休眠秒数（默认 120 = 2 分钟）')
    ap.add_argument('--hold', type=int, default=1800, help='占用声明的最长有效秒数')
    ap.add_argument('--grace', type=int, default=300,
                    help='owner 进程已退出时，声明仍被视为占用的宽限秒数（保护一次性启动）')
    ap.add_argument('--owner', default=None, help='声明归属（一般传调度脚本的 $$），默认取父进程 pid')
    ap.add_argument('--tag', default='', help='任务标签（写进声明，便于排查）')
    ap.add_argument('--once', action='store_true', help='只尝试一次，没有空闲卡就退出码 1')
    ap.add_argument('--priority', action='store_true',
                    help='高优先级（评估等短任务）：等待期间插 prio 标记，让普通任务在下次挑卡时让路')
    ap.add_argument('--release', default=None, metavar='UUID', help='释放指定卡的声明')
    ap.add_argument('--status', action='store_true', help='只打印各卡状态后退出')
    args = ap.parse_args()

    if args.status:
        show_status(args.hold, args.grace)
        return
    if args.release:
        ok = release_claim(args.release)
        print(f'[pick_gpu] release {args.release}: {"ok" if ok else "无此声明"}', file=sys.stderr)
        return

    owner = str(args.owner if args.owner is not None else os.getppid())
    if args.priority:
        set_priority_request(owner, args.tag, True)
    sleep_s = min(args.sleep, 15) if args.priority else args.sleep

    while True:
        gpus = list_gpus()
        if (not args.priority) and active_priority_requests():
            msg = f'[pick_gpu] 有高优先级任务在等卡（评估），本轮让路：休眠 {sleep_s}s'
            print(msg, file=sys.stderr)
            if args.once:
                raise SystemExit(1)
            time.sleep(sleep_s)
            continue
        cands = []
        for g in gpus:
            if occupied_by_others(g['uuid'], args.hold, owner, args.grace):
                continue
            idle = g['used'] < args.max_used_mib and g['util'] < args.max_util
            free = g['total'] - g['used']
            # 没有整卡空闲时，允许到"余量足够大"的卡上与同事共存（避免为了等整卡而空转几小时）
            shared = (not idle) and args.allow_shared and free >= args.min_free_mib
            if idle or shared:
                g = dict(g, shared=bool(shared))
                cands.append(g)
        cands.sort(key=lambda g: (1 if g['shared'] else 0, -g['total'] + g['used']))
        if cands:
            last = read_last(owner)
            prefer = [g for g in cands if g['uuid'] != last] or cands   # 多卡空闲时优先换卡
            for g in prefer:
                if try_claim(g['uuid'], owner, args.tag, args.hold, args.grace):
                    release_own_except(owner, g['uuid'])
                    write_last(owner, g['uuid'])
                    kind = (f"共享卡（已用 {g['used']}MiB / 共 {g['total']}MiB，"
                            f"剩余 {g['total']-g['used']}MiB）") if g['shared'] else '独占空闲卡'
                    print(f"[pick_gpu] 选中 {kind}: idx={g['idx']} uuid={g['uuid']} "
                          f"util={g['util']}% tag={args.tag}", file=sys.stderr)
                    if args.priority:
                        set_priority_request(owner, args.tag, False)
                    print(g['uuid'])
                    return
        msg = (f"[pick_gpu] 当前无空闲卡（可见 {len(gpus)} 张，"
               f"{[(g['idx'], g['used'], g['util']) for g in gpus]}），休眠 {args.sleep}s 后重试")
        print(msg, file=sys.stderr)
        if args.once:
            raise SystemExit(1)
        time.sleep(sleep_s)


if __name__ == '__main__':
    main()
