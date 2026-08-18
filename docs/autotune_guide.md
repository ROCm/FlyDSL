# Offline autotune configs

FlyDSL can serve a previously tuned config without benchmarking. This extends
the direct-JIT autotuner with one opt-in argument:

```python
@flyc.jit
def launch(x, out, N: fx.Constexpr[int], BLOCK: fx.Constexpr[int]):
    ...


tuned = autotune(
    configs=[Config(BLOCK=128), Config(BLOCK=256)],
    key=["N"],
    default=lambda x, out, N: Config(BLOCK=256),
    artifact_name="my_kernel",
)(launch)
```

`artifact_name` enables lookup when `FLYDSL_AUTOTUNE_CONFIG_DIR` is set. Normal
calls use the first available source:

```text
searched winner cache -> matching artifact -> default -> search
```

`FLYDSL_AUTOTUNE=1` bypasses those serving decisions, searches the existing
configs, updates the scratch winner cache, and atomically writes an artifact.
A normal fallback search updates only the scratch cache.
While artifact lookup is active, scratch winners use the same device descriptor
so a same-architecture product cannot shadow the matching artifact.

## Identity and compatibility

Artifact identity is the stable `artifact_name`, the declared `key` values,
and the call device's product name, target architecture, and compute-unit
count. Use a globally unique name for each kernel/config schema. The JSON is
self-describing; its filename is an identity digest.

The declared `key` owns the portable tuning axes. Include every
shape, dtype, layout, or mode that can change the winner. Keep structural knobs
as JIT `Constexpr` parameters on the existing entry point; offline tuning does
not need a build factory or a second key callback.

Artifacts intentionally do not include a compiler or kernel-source fingerprint.
Treat them as reviewed deployment inputs, and retune after a compiler, kernel,
compile-hint, or search-space change that can affect the winner.

The scratch winner cache has the same blind spot: it fingerprints the device,
toolchain, environment and compile hints, but not the adopter's kernel source or
search space. An adopter that needs stale scratch winners invalidated should
declare an integer schema parameter on its entry point and list it in `key`, then
bump it with any change that can move the winner. Softmax does this with
`tuning_schema`.

## Candidate correctness gate

`validate_hook(sig_args)` runs once per candidate, outside the timed repetitions
but under the same stream, compile hints, reset/restore policy and call arguments
as the timing run. `sig_args` maps every kernel parameter name to its value, so
positional tensor arguments are visible; `pre_hook` and `post_hook` see only the
merged kwargs and both run inside the timed callable.

Raising from the hook rejects that candidate. If every candidate is rejected the
search raises `RuntimeError("All autotune configs failed")` with the last failure
chained, so a numerical rejection stays distinguishable from a compile failure.
Use it wherever a candidate could launch successfully and still compute the wrong
answer, and hold every candidate to the same tolerance as the default.

## Device timing contract

The shared `do_bench` timer queues a GPU-side backlog before batched event
windows. This is required for sub-100 µs kernels: a fresh event pair on an empty
stream can time the host enqueue gap instead of the kernel. Each window averages
several launches, and the reported value is the median across windows. The
callable must enqueue asynchronous work on the current stream and must not
synchronize internally.

For Softmax results within 2% of the measured minimum, selection prefers the
compatibility default, then a config without an explicit occupancy override,
then the candidate packing more rows per block. This prevents event granularity
from turning equivalent 6--10 µs candidates into unstable deployment artifacts;
an improvement outside the band still wins normally. Softmax uses 10 warmup and
100 measured launches, split into five backlogged event windows; the larger
sample stabilized bandwidth-scale rows that moved by more than the tie band with
the generic 25-launch default.

## Adopters

| Kernel | Module | `artifact_name` | Tuned axes |
|---|---|---|---|
| RMSNorm | `kernels/norm/rmsnorm_autotune.py` | `rmsnorm` | `BLOCK_THREADS`, `waves_per_eu` |
| Softmax forward | `kernels/norm/softmax_autotune.py` | `softmax_fwd` | full-row threads, `waves_per_eu`, threads/rows per block for short rows |

Softmax backward is not an adopter yet; its existing kernel and dispatch are
unchanged by `softmax_fwd` artifacts.

## Failure behavior

FlyDSL ignores missing, unreadable, mismatched, or structurally invalid
artifacts and continues normal lookup to the default or search path. Artifact
config values cannot overwrite arguments that the caller supplies or declared
key axes. Values must preserve their types when encoded as JSON. `Config.pre_hook`
is process-local code, so it blocks forced artifact generation.

Once a matching artifact has been accepted, its compile, launch, and runtime
errors propagate normally; FlyDSL does not hide them by retrying the default.
If forced generation cannot establish a device identity or write its artifact,
it fails clearly without caching the winner.

Generate deployment artifacts on the intended GPU under controlled benchmark
conditions. CI should verify deterministic emit-and-load behavior, not commit a
winner selected from noisy shared-runner timing.
