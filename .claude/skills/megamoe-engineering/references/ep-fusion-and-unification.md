# EP full fusion, capacity and path unification

## Evidence labels and source anchor

Source audit: 2026-11-06, AITER main
`4ecfecb2dcc014a471f7d5c376d1fdfeb58cbe00`, which contains merged EP #6881
through `a060781a9`. Inspect current source again before implementation.

This file separates three things:

- **Source observation:** behavior in that audited main.
- **Historical experiment:** the exact earlier candidate and validation below.
- **Proposed target:** a refactor, not implemented or validated by this skill.

Primary source:
[EP wrapper](https://github.com/ROCm/aiter/blob/4ecfecb2dcc014a471f7d5c376d1fdfeb58cbe00/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py),
[small-M](https://github.com/ROCm/aiter/blob/4ecfecb2dcc014a471f7d5c376d1fdfeb58cbe00/aiter/ops/flydsl/kernels/mega_moe/mega_moe_small_m.py),
[large-M](https://github.com/ROCm/aiter/blob/4ecfecb2dcc014a471f7d5c376d1fdfeb58cbe00/aiter/ops/flydsl/kernels/mega_moe/mega_moe_large_m.py).

## What was actually merged

The wrapper always constructs its staged dispatch/Stage1/Stage2 resources.
Optional runtimes are additional owners, not views of those resources:

| Implementation | Entry/selection at audited main | Important distinction |
|---|---|---|
| Staged fixed-slot/compact | Ordinary forward unless large runtime is enabled | Existing bundle, fanout and capacity-dependent indexed payload |
| Small-M fully fused | Explicit forward_small_m with an EP-wide max_tokens | Separate fixed-slot protocol, default runtime banks 256 and 512 |
| Large-M persistent fusion | Ordinary forward when its runtime is enabled | Compact plan/dispatch/G1/G2/combine; input quant is still a separate launch |

With both options enabled there can be four full runtime allocations: staged,
large, small-256 and small-512. Weights share pointers; communication buffers
and protocol state do not automatically share. A single Python parent object
does not make this one workspace.

Eligibility is narrower than the generic staged operator: at this audit the
new fused variants target gfx950, EP8, H7168/I3072, E384/EPR48, top-k6, native
A8W4, SiLU and limit10, without explicit fanout masks. Small-M additionally
checks the 256-CU topology; large-M requires parent MTPR8192. A V4.1 Flash,
EP4, K3 or GLM instance is not made fused merely by removing one MTPR check.

Small-M has a resident-grid/LDS contract and a tagged mixed BF16/FP8 return
transport. Large-M has its own planner, DispatchSlot definition and dispatch
emitters, plus gm_indexed_payload=False. Its private compute/config helper
also duplicates Stage1Config, Stage2Config and MegaMoEConfig. These are real
deduplication opportunities, but the protocols are not byte-identical aliases.

## TPR, MTPR and max_tokens

| Value | Meaning | Can differ across EP ranks? |
|---|---|---|
| Local TPR | Number of physical input rows passed by this rank, including Graph padding when present | Yes |
| MTPR | Declared per-rank capacity and part of the communication/allocation contract | Must be collectively compatible |
| Runtime max_tokens | Common maximum physical input rows over the EP group, or a safe common upper bound used for selection | No |

Use the physical tensor-row/capture contract, not just the number of valid
requests. In the supported capacity contract:

```text
local physical TPR <= EP-wide max_tokens <= declared MTPR
```

An upper bound is sufficient, although a loose one can lose performance. A
parent MTPR32768 instance can select a child small bank512 for a call whose
EP-wide bound is512. Allocation capacity and executed work are different axes.

Never default max_tokens to local x.shape[0] under unequal EP input. Use
framework metadata already synchronized for scheduling/Graph capture; do not
add a per-layer all-reduce just to dispatch this operator. Without such a hint,
a conservative parent-capacity plan preserves old-call functionality. During
Graph replay, captured host selection is fixed; changing a Python scalar does
not dynamically switch the replayed kernel.

The legacy slice_output=False contract also matters: the staged wrapper
returns parent-capacity rows, while the existing small child returns its own
bank capacity. A unified entry must resolve that difference deliberately,
including output-buffer ownership, shape and lifetime.

## MTPR expansion: historical experiment, not current default

On 2026-11-04 an uncommitted/unpushed five-file patch was tested on PR head
`0a85ae56868bc4bd8ddb19271e5b7b9040e809c3`. It removed the8192 whitelist in
the wrapper, LargeMRuntime and private selector, and updated AOT enablement.
It retained the model/hardware contract, positive power-of-two capacity,
source encoding and address-range checks. Geometry remained the8192 geometry.

Stack: 8x gfx950/256CU, Torch2.10.0, ROCm7.2.4, FlyDSL0.4.1, MORI heap64G,
V4-Pro EP8 A8W4. Both fused options were explicitly enabled; a test driver
selected small/large collectively. This was not evidence of automatic ordinary
forward selection in production.

Uniform MTPR=TPR, median of three rounds of ten Graph replays, rank-max us:

| MTPR/TPR | Original PR | Expanded candidate | Speedup original/candidate |
|---:|---:|---:|---:|
| 1024 | 975.4 | 903.6 | 1.079x |
| 2048 | 1681.9 | 1360.5 | 1.236x |
| 4096 | 2691.9 | 2289.9 | 1.176x |
| 8192 | 4162.5 | 4139.8 | 1.005x |
| 16384 | 9797.7 | 8390.0 | 1.168x |
| 32768 | 19149.3 | 21398.0 | 0.895x |

88 paired cases/176 arms passed eight-rank FP32-reference tolerance and20
Graph replays; native capacity, zero-rank and queued-layer checks also passed.
The tolerance was relL2<=0.08, not bitwise equivalence with the staged path.
Cold rank0 AOT16384 plus fresh CPU compile-only run-only lookup passed; full
eight-GPU strict-AOT execution and whole-model serving were not performed.
An extra reverse-order performance confirmation was interrupted by external
occupancy and was excluded, not counted as passed.

32768 regressions included full-load uniform, masked and zero-rank cases and
one1024-token rank-balanced hotspot. Therefore do not claim universal speedup.
Only rx_em allocation per card, excluding all other buffers:

| MTPR | Staged rx GiB | Expanded large rx GiB |
|---:|---:|---:|
| 8192 | 2.668 | 2.668 |
| 16384 | 5.293 | 5.293 |
| 32768 | 1.750 | 10.543 |

These arenas coexist in that implementation. Staged indexed payload begins
at32768 in the audited main, not8192. Porting its storage/dedup capabilities
and consolidating ownership are prerequisites to claiming a memory-efficient
MAX-capacity fused default. The allocation difference is proven; it is not an
ATT-proven attribution of the entire timing regression.

65536 was accepted by the expanded selector but hit the existing signed32-bit
scatter/resource guard for an approximately2.707-GiB FP8 return buffer. Remove
arbitrary capacity whitelists, not real encoding/address safety checks.

Original local evidence, if present:
`/data/ghu/pr6881_mtpr_unrestricted_compare_20261104.md` and
`/data/ghu/pr6881_mtpr_unrestricted_20261104/capacity_expansion.patch`.
The concise facts above remain usable when those machine-local files are absent;
they are still dated results, not acceptance of a future refactor.

## Proposed target, not implemented

One public MegaMoEV2 owns weights and compatible workspace arenas. Extend the
existing BundlePlan/Entry/config machinery rather than adding a parallel
manager/factory hierarchy. One deterministic plan determines:

```text
capacity bank + fixed/compact protocol + payload/scale layout
+ fused/staged execution + compute geometry + artifact identity
```

Keep rank-global protocol choices separate from locally tunable compute
geometry. If the protocol supports destination-specific metadata or tagged
dtypes, preserve that exact schema rather than demanding unnecessary global
equality of every geometry field.

Initial migration policy for the supported V4-Pro EP8 A8W4 combination:

- EP-wide bound<=512: resident fixed-slot fused specialization.
- Larger bound: compact fused specialization at supported capacities,
  without an8192-only rule, once each default is validated.
- Unsupported semantics or a demonstrated regression: thin staged launch
  composition from the same common components, not another copied engine.

Small/large become scheduling specializations, not four independently owned
operators. Keep separate generated kernel variants where LDS, residency,
wave assignment or register liveness require it; one giant runtime branch can
charge tiny inputs for large-input resource requirements.

Refactor in separable stages:

1. Consolidate config types/selection and genuinely equivalent compute
   helpers. Preserve arithmetic, schedules and each variant's before/after
   numerical behavior; do not assume small-M already matches staged bitwise.
2. Unify public forward, prepare, bind/preload and AOT enumeration through
   the same plan. Preserve legacy call signatures via thin wrappers during
   migration, not a second selection implementation.
3. Consolidate compatible heavyweight arenas. Keep counters/epochs separate
   until their ownership and reuse proof permits sharing. Different layouts
   cannot be aliased just because both are called compact.
4. Port required indexed/fanout capabilities, relax capacity gates and fix
   large-capacity regressions before making them default. This is protocol/
   layout work, not mechanical cleanup; report it as a design change.
5. Remove obsolete runtime owners, experimental env gates, duplicated
   selectors and dead variants only after their replacements pass acceptance.

Production does not need user SMALL_M/LARGE_M switches or per-call mutable
enable setters. An explicit test-only fixed plan can support paired diagnosis.
Do not remove unrelated environment controls just because they share envs.py.

## Acceptance scope

Full shape coverage means the supported model/EP/quant combination and valid
TPR<=MTPR pairs, not only the diagonal plus oneMAX column. Include non-power
TPR, selector boundaries511/512/513 and1023/1024/1025, uniform/hot/Zipf,
rank-balanced hotspots, fanout boundaries, uneven/zero ranks and sentinels.

Test repeated transitions between small banks, compact execution and bound
weights/layer slots with production-depth queued Graph chains. The captured
plan cannot change during one replay. Verify source maps, scale interpretation,
release/acquire, epoch wrap, output shape and no forward-path allocation/JIT.

AOT must enumerate every reachable leaf for each deployment profile. A
16384-only build may need small256/512 artifacts even without an8192 parent.
Compile into a fresh cache, check counts, start new strict run-only processes
on all eight ranks, and include an intentional missing-artifact negative test.

Report each stage's actual gates separately: CPU invariants, compile/lower,
GPU eager/ref/Graph/stress, ISA/resources, performance and serving. Functional
coverage does not imply all shapes are fused, memory-efficient or faster.
