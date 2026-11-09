"""Wait for unused GPUs; launch paired v93 arms without touching other jobs."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from worker_unseen_calibration import DATA, PYTHON, check_inputs, environment, write
from summarize_unseen_calibration import summarize as calibration_summary


def gpu_state():
    def query(args):
        result = subprocess.run(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=20)
        return list(csv.reader(result.stdout.splitlines()))
    devices = query(['--query-gpu=index,uuid,memory.used,utilization.gpu'])
    processes = query(['--query-compute-apps=gpu_uuid,pid'])
    busy = {}
    for row in processes:
        if len(row) >= 2:
            busy.setdefault(row[0].strip(), []).append(int(row[1].strip()))
    result = {}
    for row in devices:
        index, uuid, memory, util = [s.strip() for s in row]
        result[int(index)] = dict(uuid=uuid, memory_mib=int(memory), utilization=int(util),
                                  compute_pids=busy.get(uuid, []))
    return result


def free(device):
    return not device['compute_pids'] and device['memory_mib'] < 1024 and device['utilization'] < 10


def summarize(root):
    """Write exact per-seed outputs, preserving incomplete and failed arms."""
    records = []
    statuses = {}
    for arm in ['base', 'identity']:
        status = root / arm / 'worker_status.json'
        statuses[arm] = json.loads(status.read_text()) if status.exists() else {'state': 'waiting'}
        for path in sorted((root / arm / 'calibration').glob('*/result.json')):
            result = json.loads(path.read_text())
            records.append(dict(arm=arm, path=str(path), result=result))
    write(root / 'summary.json', dict(updated_at=time.time(), statuses=statuses, results=records))
    lines = ['# v93：新受试者校准迁移实验', '',
             '主模型只在 S3—S10 上训练和选权重。校准模块在 S2 上训练后测试 S1，另一个模块在 S1 上训练后测试 S2。',
             '无身份监督与身份监督模型共享样本、划分和种子。测试时只输入当前 EEG，不利用目标受试者标签或整体统计量。', '',
             '| 模型 | 当前状态 | 运行日志 |', '| --- | --- | --- |']
    for arm, status in statuses.items():
        lines.append(f'| {arm} | {status["state"]} | [{arm}]({arm}/) |')
    lines += ['', '完整对照表见 [校准实验报告](calibration_report/REPORT.md)。',
              '原始逐种子指标和完整运行状态见 [summary.json](summary.json)。',
              '方法来源及实验边界见 [reference.md](reference.md)，固定配置见 [calibration_protocol.json](calibration_protocol.json)。',
              '', 'seen 对照使用八位源受试者的混合数据，novel 校准使用一位新人；两者的差异也可能来自受试者多样性。主要对照是同池、同种子下身份模型与无身份模型的绝对检索成绩。',
              '这不是 MLDG 或 Episodic-DG 的完整复现，也不把身份聚类更清楚当作适配能力更强的证据。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    try:
        calibration_summary(root)
    except Exception as exc:
        # A reporting problem must not orphan an active training worker.
        write(root / 'report_failure.json', dict(error=repr(exc), updated_at=time.time()))
        print('REPORT_FAILED', repr(exc), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.is_relative_to(DATA / 'experiments')
    check_inputs(root)
    if args.inspect:
        print(json.dumps(gpu_state(), indent=2))
        return
    lock = (root / 'dispatcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = []
    for arm in ['base', 'identity']:
        statuspath = root / arm / 'worker_status.json'
        if statuspath.exists():
            previous = json.loads(statuspath.read_text())
            if previous['state'] not in ['complete', 'failed']:
                raise RuntimeError(f'{arm}: existing worker needs explicit recovery; refusing duplicate launch')
        else:
            pending.append(arm)
    active = {}
    last_status = None
    locks = DATA / 'experiments/.gpu_reservations'
    locks.mkdir(exist_ok=True)
    while pending or active:
        for arm, entry in list(active.items()):
            code = entry['process'].poll()
            if code is not None:
                entry['log'].close()
                entry['lock'].close()
                statuspath = root / arm / 'worker_status.json'
                previous = json.loads(statuspath.read_text()) if statuspath.exists() else {}
                if code or previous.get('state') != 'complete':
                    # SIGKILL and early exits do not reach the worker's handler.
                    # Preserve its last stage but never leave a dead worker running.
                    if previous.get('state') != 'failed':
                        write(statuspath, dict(state='failed', exit_code=code,
                              error='worker exited without a successful completion record',
                              last_status=previous, ended_at=time.time()))
                print('WORKER_EXIT', arm, code, flush=True)
                del active[arm]
                summarize(root)
        try:
            devices = gpu_state()
            query_error = None
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            devices = {}
            query_error = str(exc)
        if shutil.disk_usage(root).free < 100 * 1024 ** 3:
            query_error = 'Less than 100 GiB free; waiting without launching more workers'
            devices = {}
        occupied = {entry['gpu'] for entry in active.values()}
        for gpu in [1, 2, 3, 0]:
            if not pending or gpu in occupied or gpu not in devices or not free(devices[gpu]):
                continue
            reservation = (locks / f'gpu{gpu}.lock').open('a')
            try:
                fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                reservation.close()
                continue
            try:
                fresh = gpu_state()
                available = gpu in fresh and free(fresh[gpu])
            except (OSError, subprocess.SubprocessError, ValueError):
                available = False
            if not available:
                reservation.close()
                continue
            arm = pending.pop(0)
            log = (root / arm / 'worker.log').open('a')
            command = [str(PYTHON), '-B', '-u', str(root / 'source/worker_unseen_calibration.py'),
                       '--root', str(root), '--arm', arm, '--gpu', str(gpu)]
            env = environment(root, gpu)
            env['V93_GPU_RESERVATION_FD'] = str(reservation.fileno())
            try:
                process = subprocess.Popen(command, cwd=root / arm, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(reservation.fileno(),))
            except OSError as exc:
                log.close()
                reservation.close()
                write(root / arm / 'worker_status.json', dict(state='failed',
                      error=f'worker launch failed: {exc}', command=command, ended_at=time.time()))
                continue
            active[arm] = dict(process=process, gpu=gpu, lock=reservation, log=log)
            print('LAUNCHED', arm, 'GPU', gpu, 'PID', process.pid, flush=True)
        current = dict(state='running' if active else 'waiting', dispatcher_pid=os.getpid(),
                       updated_at=time.time(), pending=pending,
                       active={k: dict(gpu=v['gpu'], pid=v['process'].pid) for k, v in active.items()},
                       devices=devices, query_error=query_error)
        write(root / 'dispatcher_status.json', current)
        text = json.dumps({key: current[key] for key in ['state', 'pending', 'active', 'query_error']})
        if text != last_status:
            print(text, flush=True)
            last_status = text
        summarize(root)
        if pending or active:
            time.sleep(max(10, args.poll_seconds))
    summarize(root)
    states = [json.loads((root / arm / 'worker_status.json').read_text())['state'] for arm in ['base', 'identity']]
    write(root / 'dispatcher_status.json', dict(state='complete' if all(s == 'complete' for s in states) else 'finished_with_failures',
                                               dispatcher_pid=os.getpid(), ended_at=time.time(), arms=states))


if __name__ == '__main__':
    main()
