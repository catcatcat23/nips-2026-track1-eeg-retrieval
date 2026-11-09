"""Run one immutable v93 arm, in source -> cache -> calibration order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

DATA = Path('/mnt/DATA-4/anteng/eeg-retrieval')
PYTHON = DATA / 'env/bin/python'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def check_inputs(root):
    for filename, expected in json.loads((root / 'source_sha256.json').read_text()).items():
        assert digest(root / 'source' / filename) == expected, filename
    for filename, expected in json.loads((root / 'input_sha256.json').read_text()).items():
        assert digest(root / filename) == expected, filename


def environment(root, gpu):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HOME=str(DATA / 'huggingface'), UV_CACHE_DIR=str(DATA / 'uv-cache'),
               MPLCONFIGDIR=str(DATA / 'matplotlib'), TMPDIR='/mnt/DATA-4/anteng/tmp',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONPATH=f'{root / "source"}:{DATA / "experiments/upstream_dinov2_mean_sub01_seed21/deps"}',
               PATH=f'{DATA / "env/bin"}:{os.environ.get("PATH", "")}')
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['base', 'identity'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.is_relative_to(DATA / 'experiments')
    arm = root / args.arm
    status = arm / 'worker_status.json'
    check_inputs(root)
    if status.exists():
        old = json.loads(status.read_text())
        if old['state'] == 'complete':
            print('ALREADY_COMPLETE', args.arm, flush=True)
            return
        raise RuntimeError('Existing worker state requires explicit recovery; no automatic overwrite')
    env = environment(root, args.gpu)
    record = dict(arm=args.arm, gpu=args.gpu, worker_pid=os.getpid(), started_at=time.time())

    def run(stage, script, extra, logname):
        check_inputs(root)
        command = [str(PYTHON), '-B', '-u', str(root / 'source' / script),
                   '--config', str(arm / 'config.json'), *extra]
        logpath = arm / logname
        assert not logpath.exists(), f'Refusing to overwrite {logpath}'
        with logpath.open('x') as logfile:
            # Keep the cooperative GPU lease alive even if the dispatcher or
            # worker exits while the actual training process is still alive.
            lease = env.get('V93_GPU_RESERVATION_FD')
            inherited = (int(lease),) if lease is not None else ()
            child = subprocess.Popen(command, cwd=arm, env=env, stdout=logfile,
                                     stderr=subprocess.STDOUT, pass_fds=inherited)
            write(status, dict(record, state=stage, child_pid=child.pid, command=command,
                               log=str(logpath), updated_at=time.time()))
            try:
                code = child.wait()
            except BaseException:
                # Only clean up this worker's child, never another GPU user's job.
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                raise
        if code:
            raise RuntimeError(f'{stage} exited with {code}; see {logpath}')

    try:
        write(status, dict(record, state='starting', updated_at=time.time()))
        run('source_preflight', 'run_unseen_source.py', ['--preflight'], 'preflight.log')
        run('source_training', 'run_unseen_source.py', [], 'train.log')
        run('calibration', 'train_unseen_calibration.py', [], 'calibration.log')
        assert json.loads((arm / 'pipeline_status.json').read_text())['state'] == 'complete'
        write(status, dict(record, state='complete', ended_at=time.time()))
    except BaseException as exc:
        write(status, dict(record, state='failed', error=str(exc),
                           traceback=traceback.format_exc(), ended_at=time.time()))
        raise


if __name__ == '__main__':
    main()
