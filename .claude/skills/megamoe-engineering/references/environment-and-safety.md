# Environment and execution safety

MegaMoE failures are often caused by a mixed source/runtime stack, a stale
cache, an occupied GPU, a rendezvous collision or insufficient symmetric heap.
Prove the environment before interpreting a kernel result.

## Read-only node preflight

On the target host, run the skill's non-initializing helper twice before
starting:

```bash
.claude/skills/megamoe-engineering/scripts/preflight.sh \
  --aiter <aiter-worktree> \
  --flydsl <flydsl-worktree> \
  --runner <runner> \
  --python <python>
```

It uses package metadata and sysfs rather than importing Torch or invoking a
GPU management runtime. That distinction matters on an unhealthy/shared node:
even a nominally read-only `torch.cuda.device_count()` or `rocm-smi` call can
enter an uninterruptible driver wait. Use runtime imports only as a separate,
bounded build smoke after ownership is established.

Map every holder to a container or scheduler job. GPU utilization at zero is
not enough: a process can hold memory or a context while idle. A busy GPU is
not necessarily an unrelated process: verify ownership, comm name and cgroup.
Inspect full command lines only when needed and avoid copying credentials from
process arguments into reports.

Do not inherit a previous conversation's permission to kill processes. Only
stop PIDs started by the current runner or targets explicitly authorized in the
current task. Prefer a runner that records child PIDs and cleans only those.

## Known machine profiles

Historical work used eight-MI355X nodes referred to as host46 and host47, with
containers such as `guoliang_stage1` and `guoliang_eplb_main47`. These names are
discovery hints only. Containers can be recreated, stopped or repurposed.

Before use, record:

```bash
podman ps -a
podman inspect <container> --format '{{.ImageName}} {{.Image}}'
podman top <container> pid,user,etime,args
```

Confirm that the container has all eight `/dev/dri` devices and `/dev/kfd`, uses
host networking/IPC when required by MORI, and mounts the requested code/model
paths. Do not start or reconfigure a shared container without checking its
owner.

## Canonical Python and library composition

One validated customer profile used:

```bash
VERIFY_ROOT=/home/ghu/opus_verify_46
UNIVERSE_ROOT=/home/ghu/FlyDSL_universe
TORCH_LIB=/opt/venv/lib/python3.12/site-packages/torch/lib

export PYTHONPATH="$VERIFY_ROOT:$UNIVERSE_ROOT/build-fly/python_packages:/home/ghu/mori/python:/home/ghu/aiter_universe/aiter"
export LD_LIBRARY_PATH="$UNIVERSE_ROOT/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$TORCH_LIB"
export FLYDSL_RUNTIME_ENABLE_CACHE=0
export MORI_SHMEM_HEAP_SIZE=64G
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8 MASTER_ADDR=127.0.0.1
```

Use `/opt/venv/bin/python` for that profile. The Torch library directory is
required so AITER extensions can resolve symbols such as `c10`. Paths differ on
host47 and newer images; derive them from the actual environment.

Never trust `PYTHONPATH` order without proving imports:

```bash
/opt/venv/bin/python - <<'PY'
import aiter
import flydsl
import torch

print("aiter", aiter.__file__)
print("flydsl", flydsl.__file__)
print("torch", torch.__version__, torch.version.hip)
PY
```

Also print the exact MegaMoE Stage1, dispatch and test modules. A source edit has
no meaning if Python imports a different checkout or generated package.

## Build compatibility

FlyDSL Python sources, generated dialect bindings and shared libraries form one
ABI set. Do not combine an arbitrary source checkout with an unrelated
`build-fly/python_packages` merely because imports succeed. Validate signatures
for any API changed between revisions, then run a real compile/lower smoke.

AITER binary extensions must match the active Torch/ROCm ABI. A historical
`module_quant.so` failure was fixed by rebuilding it against the active venv.
If the container is replaced, reproduce the build rather than copying the old
`.so` blindly.

Check shared-library resolution when symbols are missing:

```bash
ldd <extension.so>
```

Do not dismiss ROCm differences wholesale. A pure generated kernel can have the
same ISA, but runtime, compiler, profiler, MORI and module ABI differences can
change execution or measurement. Compare ISA only after source and compile keys
are controlled.

## JIT cache modes

Use a rank-private cache directory. Concurrent ranks sharing one cache can
create lock contention or make a fresh compile look like a hit.

- For source/debug experiments: set `FLYDSL_RUNTIME_ENABLE_CACHE=0`, or create a
  fresh rank-specific directory.
- For cache-reuse validation: preserve the artifact manifest and mtimes between
  runs.
- For AOT validation: build into an empty directory, then launch a fresh process
  with `FLYDSL_RUNTIME_RUN_ONLY=1`.
- Do not combine run-only with settings that require JIT IR dumping.

An in-memory compiled-function cache survives within a process even when disk
cache is disabled. Use a new process when proving compilation or cache-key
behavior.

## Distinguishing compile, wait and hang

Normal non-fatbin specialization on the known setup is usually seconds, not
minutes. If a run is silent:

1. inspect CPU command lines and child compiler processes;
2. inspect GPU holders, utilization and memory;
3. inspect cache files, lock files and mtimes;
4. inspect rank logs independently;
5. check rendezvous port ownership;
6. determine whether one rank exited while peers wait at a collective;
7. identify the active GPU kernel before changing code.

Evidence of real compilation includes an active compiler/lowering process or a
growing/new artifact. Eight GPUs at 100% with stable logs and no compiler is a
device wait/hang, not compilation.

Use a unique `MASTER_PORT` below the default ephemeral range when practical.
After a failed rank, terminate only the sibling PIDs recorded by the runner so
they do not keep the port and GPU contexts.

## Symmetric heap and memory

MORI symmetric workspace is outside the framework's ordinary model/KV memory
budget. Size it for the exact MTPR, model dimension, top-k, payload dtype and
dedup scheme. Historical profiles required 16-64 GiB depending on path.

Do not blindly raise the heap:

- compute the expected dispatch/combine allocations;
- leave headroom for model weights, AITER workspaces and CUDA Graph capture;
- record the value in every comparison;
- understand that changing it can affect whether the server starts and how
  much HBM remains for scheduling.

## Safe runner pattern

A multi-rank runner should:

1. choose a unique port and rank-private caches;
2. print source/build/import provenance before launch;
3. record each child PID;
4. redirect one log per rank;
5. stop peers immediately if one rank fails before joining collectives;
6. install an EXIT/INT/TERM trap that signals only recorded children;
7. wait for all ranks and propagate a nonzero exit code;
8. print rank0 gates plus enough tails from every failed rank.

Avoid an unconditional `pkill -f` in a reusable runner. It can kill a colleague
whose command contains the same test filename.

## End-of-run audit

Before declaring a task clean:

```text
all child ranks exited
rendezvous port released
no task-owned /dev/kfd holder
no task-owned server/client/watchdog process
cache/log/trace locations recorded
unrelated process list unchanged
```

If material files or processes were removed, state exactly what and whether it
was recoverable.
