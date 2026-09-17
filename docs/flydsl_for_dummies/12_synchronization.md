# Barriers and Synchronization

Every chapter so far has quietly assumed that when one thread writes LDS and
another reads it, the read sees the write. That assumption is exactly what this
chapter is about. In HIP you buy it with `__syncthreads()`, `__threadfence()`, and
a compiler that inserts `s_waitcnt` where you cannot see it. FlyDSL exposes the
same machine, but because the DSL sits on top of an explicit dialect stack, the
pieces are separated more sharply than they are in C++ — and knowing which piece
you need is most of the battle.

## Four different things called "synchronization"

They are routinely conflated, and on AMD hardware they are four genuinely
different mechanisms with four different costs:

| Concern | Question | FlyDSL tool |
|---------|----------|-------------|
| **Execution** | have the other threads *reached* this point? | `fx.barrier()` (§12.3) |
| **Completion** | has my own issued memory operation *finished*? | wait counters (§12.4) |
| **Visibility** | can another agent *see* my write, and in what order? | fences, orderings, scopes (§12.5) |
| **Scheduling** | in what order does the compiler *emit* instructions? | `sched_barrier` & friends (§12.8) |

The fourth is not synchronization at all — it is a compiler hint with no runtime
semantics — but it is spelled with the word "barrier", which causes no end of
confusion, so it is covered here to be dismissed here.

A useful rule of thumb: **execution and completion are per-wave and cheap;
visibility is per-scope and can be expensive** (it may flush or invalidate
caches). Ask for the narrowest scope that is correct.

## The wave: synchronized for free

The lanes of a single wave execute one instruction stream in lockstep. There is
no wave-level barrier in FlyDSL because there is nothing to synchronize: after a
lane writes a register, every lane of that wave is already past that instruction.
This is why a warp reduction needs no barrier at all:

```python
# a full cross-lane reduction with no synchronization whatsoever
for sh in (1, 2, 4, 8, 16, 32):
    x = fx.maxnumf(x, x.shuffle_xor(sh, WAVE))
```

and why kernels that keep a reduction inside one wave say so proudly
(`kernels/attention/qk_norm_rope_quant.py:11` — *"shuffle_xor, no LDS, no
barrier"*).

Two caveats. First, lockstep does **not** cover memory *completion*: a `ds_write`
issued by a lane is not necessarily visible to a `ds_read` by the same wave until
the LDS counter drains — the compiler normally inserts that wait for you (§12.4).
Second, "the wave" is 64 lanes on CDNA and 32 on RDNA, so a reduction that relies
on wave width for its correctness is architecture-specific unless you name the
width explicitly (`width=` on the `fx.coop` collectives, §13.3).

> **HIP/CK-Tile → FlyDSL.** Same as the `__shfl`-based warp reduce you already
> write without a `__syncthreads()`. The one thing that changes is the number 64
> versus 32.

## The workgroup: `fx.barrier()`

```python
fx.barrier()          # or fx.gpu.barrier() — the same function
```

This is `__syncthreads()`. It emits the MLIR `gpu.barrier` op, which lowers to
`rocdl.barrier` and then to the ISA instruction `s_barrier`: every wave in the
workgroup waits until all of them have arrived. The backend also drains the LDS
counter (`s_waitcnt lgkmcnt(0)`) ahead of the barrier, so a barrier delivers both
execution *and* LDS-completion semantics — which is precisely why the
write-barrier-read idiom works with nothing else added.

`fx.rocdl.s_barrier()` emits the ROCDL op directly, bypassing `gpu.barrier`. Use
it when you are hand-composing a wait with a barrier (§12.4); otherwise prefer
`fx.barrier()`.

> **Gotcha — a divergent barrier deadlocks.** `fx.barrier()` requires *every*
> thread of the workgroup to reach it. Putting one inside a runtime `if` that some
> threads skip hangs the kernel, and nothing in FlyDSL diagnoses it — not the
> tracer, not the verifier, not the backend. If a barrier must be conditional,
> the condition has to be workgroup-uniform (the same for all waves, typically an
> SGPR value derived from block indices or a `Constexpr`), never lane-divergent.

### Split barriers (gfx1250)

Newer hardware can separate the two halves of a barrier, so a wave can announce
its arrival, do useful work, and only then block. FlyDSL surfaces the ROCDL ops
`s_barrier_signal` / `s_barrier_wait`, wrapped per use case:

```python
# kernels/gemm/gemm_common_gfx1250.py:63 — arrive, compute, then wait
pipeline_fence_signal(outstanding=0)   # s_wait_tensorcnt + s_barrier_signal -1
# ... MFMA/WMMA work that does not touch the staged tile ...
pipeline_fence_wait()                  # s_barrier_wait -1
```

The same ops with a different barrier ID implement a **cluster** barrier — a
synchronization across the workgroups of a gfx1250 cluster, signalled once per
workgroup by its leader wave:

```python
from flydsl.expr.rocdl import cluster
cluster.cluster_barrier()    # gpu.barrier + signal(once per WG) + wait
```

> **HIP/CK-Tile → FlyDSL.** `fx.barrier()` is `__syncthreads()`;
> `pipeline_fence_signal` / `pipeline_fence_wait` are
> `__builtin_amdgcn_s_barrier_signal` / `_wait`, the AMD analogue of splitting an
> arrive from a wait the way `cuda::barrier` does.

## Completion: the wait counters

A vector-memory or LDS instruction on AMD hardware is *asynchronous by
construction*: the instruction issues, and a hardware counter tracks how many are
outstanding. `s_waitcnt` blocks the wave until a counter falls to a given value.
There are three classic counters — `vmcnt` (vector memory: global/buffer),
`lgkmcnt` (LDS, GDS, constant, message), and `expcnt` (export/parameter) — plus
newer ones on gfx1250 (`tensorcnt` for TDM, `asynccnt` for async loads).

**In normal kernel code you never write one.** The LLVM AMDGPU backend's wait-count
insertion pass tracks every dependence and emits the waits for you. That is what
makes `fx.copy` composable: the copy is a data dependence, and the backend closes
it.

There are exactly three situations where the compiler steps back and hands you the
job:

- **Async LDS DMA on CDNA4** — the `BufferLoadAsyncLDS*` / `GlobalLoadAsyncLDS*`
  atoms (§8.6). Close a group with `fx.rocdl.asyncmark()`, drain it with
  `fx.rocdl.wait_asyncmark(n)`.
- **TDM copies on gfx1250** — wait on the tensor counter with
  `tdm_ops.tensor_wait(n)`, which emits `s_wait_tensorcnt`.
- **Cluster async loads on gfx1250** — wait on the async counter with
  `fx.rocdl.s_wait_asynccnt(n)`.

These exist because the whole point of an *async* copy is that the data
dependence is deliberately hidden from the backend — if it could see it, it would
insert the wait and destroy the overlap you were buying.

### Writing a wait by hand

```python
fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)     # drain both
fx.rocdl.s_waitcnt(vmcnt=2)                # allow 2 loads still in flight
```

`None` for a counter means "do not wait on this one". The named form is encoded
per architecture (the bit positions differ between CDNA3/4, RDNA3, and RDNA4) and
raises on an architecture it does not know — notably gfx1250, which uses the
newer counters instead. A legacy positional form, `fx.rocdl.s_waitcnt(bitfield)`,
passes a pre-encoded 16-bit immediate straight through; you will see it in older
kernels.

The idiom you will actually meet is a wait fused with a barrier, because "all my
loads have landed" and "all waves have arrived" are usually wanted together:

```python
# kernels/gemm/fp4_gemm_4wave.py:108
def wait_barrier(count):
    """``s_waitcnt vmcnt(count) lgkmcnt(0)`` + ``s_barrier``"""
    fx.rocdl.s_waitcnt(vmcnt=count, lgkmcnt=0)
    fx.rocdl.s_barrier()
```

The `vmcnt=count` — rather than `0` — is the interesting part: it drains only the
*older* loads, deliberately leaving the prefetch for the next tile in flight
across the barrier.

> **HIP/CK-Tile → FlyDSL.** Identical to `__builtin_amdgcn_s_waitcnt` in a
> hand-tuned HIP kernel, and used for the same reason: the compiler's automatic
> wait is correct but conservative, and a pipelined loop wants a weaker one.

## Visibility: fences, orderings, and scopes

Execution barriers and wait counters say nothing about what *other agents* can
see. For that, FlyDSL exposes the LLVM memory model directly (§10.7):

```python
fx.memory_fence(ordering=fx.AtomicOrdering.Release,
                syncscope=fx.rocdl.SyncScope.Agent)
```

Two knobs, and they are orthogonal.

**Ordering** (`fx.AtomicOrdering`) says *what may not be reordered across this
point*: `Monotonic` (atomicity only), `Acquire` (later reads may not move up),
`Release` (earlier writes may not move down), `AcqRel`, `SeqCst`. A fence needs
`Acquire` or stronger.

**Scope** (`syncscope=`) says *who has to see it*, and it is where the cost lives.
FlyDSL has the two target-neutral scopes and a full set of AMD-specific ones:

| Scope | Who participates | Typical cost |
|-------|------------------|--------------|
| `fx.SyncScope.SingleThread` | just this lane | none |
| `fx.rocdl.SyncScope.Wavefront` | the wave | none (lockstep) |
| `fx.rocdl.SyncScope.Workgroup` | the workgroup | LDS/L1-level |
| `fx.rocdl.SyncScope.Agent` | this GPU | L2-level |
| `fx.SyncScope.System` (`""`) | all agents, incl. host & peer GPUs | cache flush/invalidate |

Each also has a `…OneAs` variant (`Agent`, `Workgroup`, `Wavefront`,
`SingleThread` → `AgentOneAs`, …, plus the bare `OneAs`). "One address space"
tells the backend the fence only has to order accesses in the *same* address space
as the operation it guards — so a global-memory handshake does not pay to order
LDS as well. Production communication kernels use the `OneAs` forms almost
exclusively:

```python
# kernels/comm/communication_ops_utils.py:59
def fence_system_acquire():
    """System-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.OneAs)

def fence_agent_acquire():
    """Agent-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.AgentOneAs)
```

The same two knobs appear on the load/store and atomic primitives —
`fx.generic_load(..., memory_order=, syncscope=)`, `fx.atomic_add(..., ordering=,
syncscope=)` — so a release store and an acquire load can be expressed without a
standalone fence.

> **HIP/CK-Tile → FlyDSL.** This is `__threadfence_block()` /
> `__threadfence()` / `__threadfence_system()`, except that the scope is a named
> argument instead of three differently-spelled functions, and the ordering is
> separately selectable rather than always sequentially consistent.

## Across workgroups and devices

There is **no grid-wide barrier** — no cooperative launch, no `grid.sync()`. Two
workgroups synchronize the way they do in any HIP kernel: through global memory,
with an atomic and a spin.

The canonical shape, from the multi-GPU dispatch kernel
(`kernels/comm/flydsl_dispatch_combine_intranode_kernel.py:253`), is
**arrive-and-wait on a ticket counter**:

```python
fx.barrier()                                # 1. this block is internally consistent
if tid == 0:
    atomic_add_global_at(addr_disp_bar, 1)  # 2. publish arrival (release side)
    mori_shmem.int32_wait_until_equals(addr_disp_bar, block_num)   # 3. spin
    fence_system_acquire()                  # 4. peers' writes are now visible
```

Read it as the three-part contract it is: the *atomic* makes the counter update
indivisible, the *spin* is the execution wait, and the *acquire fence* is the
visibility half — without step 4, step 3 proves only that the counter reached its
value, not that the data those blocks wrote is readable.

The peer-to-peer all-reduce (`kernels/comm/custom_all_reduce_kernel.py:185`) uses
the same skeleton with uncached loads and stores, plus one extra tool: an L1
invalidation inside the polling loop, so each poll re-reads from L2 rather than
spinning forever on a cached line.

```python
while _u(current_wait_value) < flag:
    current_wait_value = _load_i32_uncached(wait_rsrc)
    _invalidate_l1()          # inline asm: buffer_inv sc1
```

Note that `buffer_inv` has no first-class FlyDSL API — it is kernel-local inline
assembly (§13.4). Cache control below the fence level is not part of the DSL
surface today.

> **Gotcha — spin loops need a real dependence.** The load inside the wait loop
> must stay inside the loop body. A "hoisted" load, or one the compiler can prove
> loop-invariant, turns the spin into an infinite loop. The kernels above use
> uncached/volatile loads and re-issue them explicitly for exactly this reason.

## Where all this lives in the dialect stack

There is **no `fly.barrier` op**. The Fly dialect is about layouts, tiles, and
atoms; synchronization is orthogonal to all three, so the DSL emits the standard
`gpu.*`, `llvm.*`, and `rocdl.*` ops directly. That has a practical consequence:
sync primitives pass through the Fly→ROCDL lowering untouched, so what you write
is what the backend sees.

```
Execution
  fx.barrier()                → gpu.barrier            → s_barrier
  fx.rocdl.s_barrier()        → rocdl.s_barrier        → s_barrier
  rocdl.s_barrier_signal(id)  → rocdl.s_barrier_signal → s_barrier_signal
  rocdl.s_barrier_wait(id)    → rocdl.s_barrier_wait   → s_barrier_wait

Completion
  fx.rocdl.s_waitcnt(...)     → rocdl.s_waitcnt        → s_waitcnt imm16
  fx.rocdl.asyncmark()        → rocdl.asyncmark        → group marker
  fx.rocdl.wait_asyncmark(n)  → rocdl.wait.asyncmark   → group wait
  tdm_ops.tensor_wait(n)      → rocdl.s_wait_tensorcnt → s_wait_tensorcnt

Visibility
  fx.memory_fence(...)        → llvm.fence             → wait + cache op
  fx.atomic_add(ptr, v)       → llvm.atomicrmw         → global_atomic_add
  fx.generic_load(ptr, ...)   → llvm.load              → global_load_*

Scheduling — a compiler hint, not synchronization
  fx.rocdl.sched_barrier(m)   → rocdl.sched_barrier    → s_sched_barrier
```

(The split-barrier and `s_wait_tensorcnt` rows are gfx1250-only; everything else
is available on every supported target.)

The one place the *atom* layer interacts with this is the synchronous/asynchronous
distinction of §8.6: a `BufferCopyLDS` atom is "synchronous" only in the sense
that the backend is still allowed to see the dependence and insert the wait; an
`…AsyncLDS` atom deliberately hides it.

## Patterns and gotchas

### One barrier is often not enough

The write-barrier-read pattern is only half of a reusable buffer. If the loop
writes the same LDS region on the next iteration, you need a **second** barrier —
after the reads — or the fastest wave will overwrite data a slower wave has not
read yet:

```python
for k in range(...):
    fx.copy(g2s_atom, src_part, lds_part)
    fx.barrier()          # (1) writes are visible to readers
    fx.copy(s2r_atom, lds_part, frag)
    fx.barrier()          # (2) reads are done; safe to overwrite next iteration
```

Two barriers per iteration is a real cost, and the standard way to buy one back is
**double buffering**: write buffer `k & 1` while reading buffer `(k+1) & 1`, so
the write region and the read region are disjoint and barrier (2) disappears. The
all-reduce kernel documents the trade in one line
(`kernels/comm/custom_all_reduce_kernel.py:473`): *"Single-buffer (large tensor):
8KB LDS, 2 barriers/iter, higher occupancy. Double-buffer (small tensor): 16KB
LDS, 1 barrier/iter."* That is the whole design space — LDS footprint (and thus
occupancy) against barrier count.

The same reasoning applies to a `@fx.union` LDS layout (§8.6): the union
guarantees two phases share bytes, and a barrier between the phases is what makes
that safe.

### Scheduling hints are not synchronization

`fx.rocdl.sched_barrier(mask)`, `sched_group_barrier(...)`, and the
`sched_mfma` / `sched_vmem` / `sched_dsrd` / `sched_dswr` wrappers (§13.5)
constrain what the *compiler's instruction scheduler* may move across a point.
`fx.rocdl.s_setprio(n)` changes a wave's scheduling priority. None of them makes
any thread wait for any other thread, and none of them makes any write visible to
anyone. Removing one can change performance dramatically and cannot change
correctness; removing a barrier can do the opposite. Keep the two categories
separate in your head — and note that `sched_barrier` and `s_barrier` differ by
two characters.

Similarly, `fx.rocdl.readfirstlane` and `fx.rocdl.ballot` move data across lanes
but synchronize nothing: within a wave there is nothing to synchronize (§12.2).

### A checklist for "my kernel returns garbage sometimes"

Intermittent wrong results are almost always one of these, in rough order of
frequency:

1. A missing second barrier around a reused LDS buffer.
2. An async copy (`…AsyncLDS`, TDM) consumed without its `wait_asyncmark` /
   `tensor_wait`.
3. A cross-workgroup handshake with the atomic but no acquire fence.
4. A barrier inside divergent control flow (this usually hangs rather than
   corrupts — but a *skipped* barrier corrupts).
5. A scope that is too narrow: `Workgroup` where the reader is another workgroup,
   or `Agent` where the reader is another GPU.

Chapter 15 covers how to isolate these once you suspect them; the fastest
discriminator is that a synchronization bug changes with block size, wave count,
or occupancy, while an indexing bug does not.

With ordering understood, Chapter 13 turns to the escape hatches — what to do when
the high-level atom API cannot express the instruction you need at all.
