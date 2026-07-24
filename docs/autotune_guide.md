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

The declared `key` is the single owner of portable tuning axes. Include every
shape, dtype, layout, or mode that can change the winner. Keep structural knobs
as JIT `Constexpr` parameters on the existing entry point; offline tuning does
not need a build factory or a second key callback.

Artifacts intentionally do not include a compiler or kernel-source fingerprint.
Treat them as reviewed deployment inputs and retune after a compiler, kernel,
compile-hint, or search-space change that can affect the winner.

## Failure behavior

Missing, unreadable, mismatched, or structurally invalid artifacts are ignored,
and normal lookup continues to the default or search path. Artifact config
values cannot overwrite arguments supplied by the caller or declared key axes.
Values must preserve their types when encoded as JSON. `Config.pre_hook` is
process-local code, so it blocks forced artifact generation.

Once a matching artifact has been accepted, its compile, launch, and runtime
errors propagate normally; FlyDSL does not hide them by retrying the default.
If forced generation cannot establish a device identity or write its artifact,
it fails clearly without caching the winner.

Generate deployment artifacts on the intended GPU under controlled benchmark
conditions. CI should verify deterministic emit-and-load behavior, not commit a
winner selected from noisy shared-runner timing.
