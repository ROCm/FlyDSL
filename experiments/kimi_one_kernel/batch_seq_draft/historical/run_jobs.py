"""Run only on an idle node, gate polling kernels on compiled HIP residency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/tmp/flydsl-kimi-mtp-nativebf16-20260929/python_packages')


def idle(name, wait_seconds=1800):
    first = None
    deadline = time.monotonic() + wait_seconds
    while True:
        p = subprocess.run(['rocm-smi', '--showpids'], capture_output=True, text=True, timeout=30)
        free = p.returncode == 0 and 'No KFD PIDs currently running' in p.stdout
        lease = os.environ.get('KIMI_TOKEN_BATCH_KFD_PID')
        owned_batch = (lease is not None and p.returncode == 0
                       and set(re.findall(r'^\s*(\d+)\s+\S+\s+\d+\s+', p.stdout, re.M)) == {lease})
        with (ROOT / 'idle.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), job=name, idle=free, owned_batch=owned_batch,
                                   stdout=p.stdout, stderr=p.stderr)) + '\n')
        if owned_batch:
            # A single already-admitted batch owns these idle CUDA contexts.
            # Every child has exited; any other KFD PID prevents the next job.
            return
        if not free:
            first = None
        elif first is None:
            first = time.monotonic()
        elif time.monotonic() - first >= 10:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Node remained occupied: ' + name)
        time.sleep(2)


def run(name, source, command, *, gpu=False, timeout=1800, resource=None):
    if gpu:
        gate = json.loads((ROOT / ('resources_' + (resource or source)) / 'occupancy.json').read_text())
        assert gate['pass'] and gate['hip_occupancy_checked'], gate
        actual = hashlib.sha256((ROOT / source / 'kernels/kimi_k3_monokernel/kernel.py').read_bytes()).hexdigest()
        assert actual == gate['source_sha256'], (source, 'source changed after occupancy check')
        idle(name)
    env = os.environ.copy()
    for k in ['COMPILE_ONLY', 'FLYDSL_DUMP_IR', 'ROCPROF_ATT', 'FLYDSL_RUNTIME_ENABLE_CACHE']:
        env.pop(k, None)
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / source), str(ROOT), str(RUNTIME)]),
        LD_LIBRARY_PATH=str(RUNTIME / 'flydsl/_mlir/_mlir_libs') + ':/opt/rocm/lib',
        FLYDSL_RUNTIME_CACHE_DIR=str(ROOT / ('cache_' + (resource or source))),
        OMP_NUM_THREADS='1', ROCM_PATH='/opt/rocm', FLYDSL_GPU_ARCH='gfx950')
    record = dict(cmd=command, cwd=str(ROOT / source), source=source, gpu=gpu,
                  env={k: env[k] for k in ['PYTHONPATH', 'LD_LIBRARY_PATH', 'FLYDSL_RUNTIME_CACHE_DIR',
                                         'OMP_NUM_THREADS', 'ROCM_PATH', 'FLYDSL_GPU_ARCH']})
    (ROOT / (name + '.command.json')).write_text(json.dumps(record, indent=2) + '\n')
    print('START', name, flush=True)
    start = time.monotonic()
    with (ROOT / (name + '.log')).open('w') as log:
        p = subprocess.Popen(command, cwd=ROOT / source, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        (ROOT / (name + '.pid')).write_text(str(p.pid) + '\n')
        try:
            rc = p.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = 124
    (ROOT / (name + '.rc')).write_text(str(rc) + '\n')
    print('DONE', name, 'rc', rc, 'seconds', round(time.monotonic() - start, 2), flush=True)
    if rc:
        print((ROOT / (name + '.log')).read_text()[-6000:], flush=True)
        raise SystemExit(rc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'bench'])
    parser.add_argument('variants', nargs='+')
    args = parser.parse_args()
    seeds = [1234] if args.mode == 'check' else [1234, 2025, 3141]
    for index, seed in enumerate(seeds):
        order = args.variants if index % 2 == 0 else args.variants[::-1]
        for source in order:
            resource = 'base' if source == 'base_checked' else source
            staged = source.startswith('staged_relocate')
            name = f'{args.mode}_{source}_seed{seed}'
            if args.mode == 'bench' and not staged:
                assert (ROOT / f'check_{source}_seed1234.rc').read_text().strip() == '0'
            argv = [sys.executable, '-m', 'kernels.kimi_k3_monokernel.tools.monokernel',
                    '--mtp', '--samples', '4', '--layer-idx', '1', '--check', '--seed', str(seed),
                    '--output', str(ROOT / (name + '.json'))]
            if args.mode == 'check' and not staged:
                argv.append('--full-replay-check')
            if args.mode == 'bench':
                argv += ['--bench', '--layers', '16', '--repeats', '50']
            if staged:
                argv.append('--staged')
            run(name, source, argv, gpu=True, resource=resource)
    print('JOBS_DONE', args.mode, flush=True)
