# Autotuning FlyDSL kernels

FlyDSL's autotuner (`flydsl.autotune`) benchmarks candidate kernel configs,
picks the fastest, and remembers it. It has three consumption modes, mirroring
what Triton, quack, aiter, and SGLang do in practice.

## The three paths

1. **Heuristic default (zero search).** A `default(*args) -> Config` analytic
   rule picks a good config with no benchmarking. This is what normal runs use —
   tests and serving never pay tuning cost. (== Triton `@heuristics`,
   quack `get_default`.)
2. **Online JIT search.** With `FLYDSL_AUTOTUNE=1`, the tuner sweeps the config
   space once per shape, benchmarks each, caches the winner in a machine-local
   scratch cache, and reuses it for the rest of the process / future runs.
3. **Offline committed configs.** Tuned configs written to a checked-in tree and
   looked up at serving with no search — the aiter/SGLang model. Portable across
   identical GPUs; the filename *is* the lookup key.

## When to tune

- **Dev / perf work:** `FLYDSL_AUTOTUNE=1` to explore the space for your shape.
- **Committed offline configs:** tune the shapes you serve, commit the JSON, and
  ship. Downstream engines resolve a config by reconstructing the filename.
- **Everything else (CI, normal runs):** do **not** set `FLYDSL_AUTOTUNE`. The
  heuristic default (or a committed offline config) is used, so no benchmark
  loop runs.

## Do not tune in CI

Autotuning benchmarks the same kernel dozens of times per config. That is slow
and noisy on shared CI runners, and the winning config would depend on the
runner's load. CI should exercise the *default* and *offline-lookup* paths (fast,
deterministic) and leave the search off. The GPU-free unit tests
(`tests/unit/test_autotune.py`) cover serialization, cache keys, `restore_value`,
pruning, and the offline emit/lookup logic without any GPU.

## Cache behavior and keys

Two separate stores:

| Store | Env var | Keyed by | Portable? |
|-------|---------|----------|-----------|
| Scratch cache | `FLYDSL_AUTOTUNE_CACHE_DIR` (default `~/.flydsl/autotune`) | full fingerprint: shape, dtype, **stride pattern**, GPU arch, **toolchain fingerprint**, cache-invalidating env | no (machine-local) |
| Offline configs | `FLYDSL_AUTOTUNE_CONFIG_DIR` | filename: `name,<spec axes>,device_name` | yes (commit + share) |

The scratch cache is invalidated automatically when the compiler toolchain, GPU
arch, or a relevant env var changes — a config tuned under one build is never
silently reused under another. The offline tree is deliberately coarser: it keys
only on `name`, the `specialize()` axes, and `device_name` — **not** the
toolchain fingerprint, stride pattern, or non-spec tensor dtypes that the scratch
cache folds in. So a config tuned once is reusable on any matching GPU, but this
puts two requirements on the adopter:

- **`specialize()` must return every axis that changes the best config** (row
  width, dtype, and any layout/mode flag). Anything the kernel's performance
  depends on but `specialize` omits will collide: two calls with the same spec
  but different (say) stride pattern map to one artifact, and the last tune
  wins. The scratch cache would keep them separate; the offline tree cannot.
- **`specialize()` values must be JSON-round-trippable** (str/int/float/bool,
  or tuples/lists of them). The spec is normalized through JSON on both emit and
  lookup, so a tuple (stored as a list) still self-matches; but an `Enum` /
  `torch.dtype` isn't serializable and disables the offline path for that op
  (with a warning). Use its `.name` / string form instead.

A committed artifact from an older toolchain but the same device name **is**
served (no automatic invalidation) — treat re-tuning after a toolchain change as
a review/policy step, and commit artifacts as reviewed inputs.

An offline artifact must be fully self-describing (`name`, `spec`,
`device_name`, `config`) and every field is validated against the call before
use — a mismatch, a missing field, malformed JSON, or a non-dict `spec` is
ignored with a warning rather than silently mis-serving via the heuristic
fallback. Bare/partial configs are not trusted. Filenames are sanitized to ASCII
`[A-Za-z0-9._-]`, so no spec/name/device value can inject a delimiter or escape
the config directory.

## Correctness: `restore_value` / `reset_to_zero`

Because tuning reruns the kernel many times, any kernel that writes its output
in place or accumulates into its inputs (e.g. fused-add rmsnorm, where the
output overlaps the residual buffer) corrupts its own data across reps — and the
timing/selection is then made on garbage. Declare those tensors:

- `restore_value=[...]` — snapshot before the reps, restore before each rep.
- `reset_to_zero=[...]` — zero before each rep (accumulate-into-zero kernels).

## Adopting `@autotune` on a kernel

Most FlyDSL kernels bake structural knobs (block size, tile width) at
module-build time rather than exposing them as jit `Constexpr` params, so use
`autotune_builder(...)`: supply `build` (rebuilds the module per config),
`specialize` (extracts the shape + build-only scalars — these become both the
cache key and the offline filename), `configs`/`default` (the search space and
the zero-search heuristic), and `structural` (which config knobs route into
`build`). The helper owns cache naming, build caching, and the offline path. See
`kernels/rmsnorm_autotune.py` and `kernels/rmsnorm_config.py` for the reference
adoption — it is a single declaration.
