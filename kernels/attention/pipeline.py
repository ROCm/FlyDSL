# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Stage-based composable software pipeline for flex attention kernels.

A ``PipelineStage`` declares *what* to compute (resource cost, execute logic,
optional multi-slot decomposition). A ``Wire`` declares *how stages connect*
(which shared registers flow between them). A ``PipelineScheduler`` takes the
wire list and emits a fully pipelined kernel loop — cluster assignment, depth,
stagger, prologue/epilogue, vmcnt, and scheduling hints are all derived
automatically.

Internal state for decomposed stages (e.g. Softmax's partial exp2 registers
between its two sub-stages) is held on the stage object itself, invisible to
the wiring.

All stages use the CuTe-style layout API internally (``fx.gemm``, ``fx.copy``,
``make_fragment``, ``partition_C``). The scheduling layer (``rocdl.*``) sits
alongside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Sequence


# ---------------------------------------------------------------------------
# Enums and value types
# ---------------------------------------------------------------------------


class StageKind(Enum):
    """Hardware resource category for cluster assignment."""
    MEMORY = auto()
    COMPUTE = auto()


@dataclass(frozen=True)
class ResourceDecl:
    """Compile-time resource estimate for depth computation and sched hints.

    Counts are Python ints resolved at kernel build time. They only need
    to be accurate enough to get the depth integer right (2 vs 3).
    """
    dma_count: int = 0
    lds_read_count: int = 0
    mfma_count: int = 0
    valu_count: int = 0
    exp_count: int = 0
    lds_bytes: int = 0


# ---------------------------------------------------------------------------
# InfraContext — infrastructure references shared by all stages
# ---------------------------------------------------------------------------


@dataclass
class InfraContext:
    """Read-only infrastructure references set once by the kernel.

    Stages receive this alongside their ``**kwargs`` inputs. It provides
    the MMA atoms, copy atoms, LDS views, thread slices, and constants
    that stages need to emit layout-API operations. It does NOT carry
    inter-stage data — that flows through ``**kwargs``.
    """
    traits: object = None
    tiled_mma_qk: object = None
    tiled_mma_pv: object = None
    thr_qk: object = None
    thr_pv: object = None
    ca: object = None
    uca: object = None
    dma_atom: object = None
    elem_dtype: object = None
    block_m: int = 0
    block_n: int = 0
    head_dim: int = 0
    scale_log2e: object = None
    n_kv_tiles: int = 0
    stagger_i32: object = None
    # LDS views for the K/V ring buffers (list of per-buffer views)
    sK: object = None
    sVt: object = None
    sP: object = None
    # Tile/buffer state (updated by the kernel loop each iteration)
    tile_idx: object = None
    buf_slot: int = 0
    read_buf: int = 0
    write_buf: int = 0
    prefetch_tile_idx: object = None  # None = no prefetch this iteration
    is_prologue: bool = False
    is_epilogue: bool = False


# ---------------------------------------------------------------------------
# PipelineStage — minimal base class
# ---------------------------------------------------------------------------


class PipelineStage(ABC):
    """A compile-time unit of work within one KV tile iteration.

    Stages declare *what* to compute but not *what they connect to*.
    Shared register names are declared externally via ``Wire``.

    ``execute()`` receives shared registers as ``**kwargs`` (keyed by
    the names declared in the Wire's ``inp``) plus an ``InfraContext``
    as the first positional argument. It returns a dict of outputs keyed
    by the names declared in the Wire's ``out``.

    For decomposed stages, internal intermediate registers (e.g. partial
    exp2 results between Softmax sub-stages) are held on ``self`` — they
    never appear in the wiring.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def kind(self) -> StageKind:
        ...

    @property
    @abstractmethod
    def resources(self) -> ResourceDecl:
        ...

    @property
    def min_slots(self) -> int:
        return 1

    @property
    def max_slots(self) -> int:
        return 1

    @abstractmethod
    def execute(self, infra: InfraContext, **kwargs) -> dict:
        """Emit IR. Receives shared regs as kwargs, returns output dict.

        Example::

            def execute(self, infra, *, frag_K, frag_Q, **_):
                frag_S = fx.gemm(infra.tiled_mma_qk, ...)
                return {"frag_S": frag_S}
        """
        ...

    def decompose(self, allocated_slots: int) -> list["SubStage"]:
        """Split into ``allocated_slots`` sub-stages.

        Each sub-stage callable has the same signature as ``execute()``:
        ``(infra, **shared_regs) -> dict``. Internal intermediates live
        on ``self`` — they don't appear in the returned dict unless they
        are declared outputs in the Wire.

        Default: single sub-stage wrapping ``self.execute``.
        """
        assert self.min_slots <= allocated_slots <= self.max_slots
        return [SubStage(self.name, self.execute, self.resources)]

    def resources_for_slots(self, n_slots: int) -> list[ResourceDecl]:
        r = self.resources
        return [ResourceDecl(
            dma_count=r.dma_count // n_slots,
            lds_read_count=r.lds_read_count // n_slots,
            mfma_count=r.mfma_count // n_slots,
            valu_count=r.valu_count // n_slots,
            exp_count=r.exp_count // n_slots,
            lds_bytes=r.lds_bytes,
        )] * n_slots

    @property
    def sched_hints(self) -> Optional[Callable]:
        return None


# ---------------------------------------------------------------------------
# SubStage — one piece of a decomposed stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubStage:
    """A callable returned by ``decompose()``, placed in one cluster."""
    name: str
    fn: Callable  # (infra, **kwargs) -> dict
    resources: ResourceDecl


# ---------------------------------------------------------------------------
# Wire — declares inter-stage connections at the assembly site
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Wire:
    """Connects a stage to the pipeline's shared register namespace.

    Declared by the pipeline author when assembling the stage list::

        PipelineScheduler([
            Wire(LoadK(),            out=("frag_K",)),
            Wire(LoadV(),            out=("frag_Vt",)),
            Wire(Gemm1_QK(),         inp=("frag_K", "frag_Q"),
                                     out=("frag_S",)),
            Wire(ScoreMod(alibi),    inp=("frag_S",),
                                     out=("frag_S",)),
            Wire(Softmax(),          inp=("frag_S",),
                                     out=("frag_P",),
                                     carry=("m_i", "l_i")),
            Wire(BridgeP(),          inp=("frag_P",),
                                     out=("frag_P_a",)),
            Wire(RescaleAndGemm2(),  inp=("frag_P_a", "frag_Vt"),
                                     out=("frag_O",),
                                     carry=("frag_O",)),
        ])

    Parameters
    ----------
    stage : PipelineStage
    inp : tuple[str, ...]
        Shared register names this stage reads.
    out : tuple[str, ...]
        Shared register names this stage writes.
    carry : tuple[str, ...]
        Shared registers that persist across tile iterations (loop-carried).
        The scheduler includes these in ``scf.for`` init/yield.
    """
    stage: PipelineStage
    inp: tuple = ()
    out: tuple = ()
    carry: tuple = ()


# ---------------------------------------------------------------------------
# ClusterSpec + PipelineConfig — scheduler output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterEntry:
    """One callable within a cluster, with its I/O contract."""
    name: str
    fn: Callable  # (infra, **kwargs) -> dict
    inp: tuple
    out: tuple
    resources: ResourceDecl


@dataclass(frozen=True)
class ClusterSpec:
    """A group of entries executed between two sync barriers."""
    index: int
    kind: StageKind
    entries: tuple[ClusterEntry, ...]

    @property
    def total_dma(self) -> int:
        return sum(e.resources.dma_count for e in self.entries)

    @property
    def total_mfma(self) -> int:
        return sum(e.resources.mfma_count for e in self.entries)

    @property
    def total_valu(self) -> int:
        return sum(e.resources.valu_count for e in self.entries)

    @property
    def total_exp(self) -> int:
        return sum(e.resources.exp_count for e in self.entries)


@dataclass(frozen=True)
class PipelineConfig:
    """Compile-time pipeline configuration."""
    depth: int
    clusters: tuple[ClusterSpec, ...]
    lds_ring_slots: int
    lds_bytes_per_slot: int
    vmcnt_targets: dict
    loop_carried: tuple[str, ...]


# ---------------------------------------------------------------------------
# PipelineScheduler
# ---------------------------------------------------------------------------


class PipelineScheduler:
    """Takes ``Wire``-wrapped stages, validates dataflow, and emits a
    pipelined kernel loop.

    The scheduler:
      1. Validates every wire's inputs are produced by a prior wire or carry
      2. Iteratively assigns clusters + computes depth (converges in 1-2 rounds)
      3. Emits prologue, main loop, epilogue with barriers and vmcnt
      4. Manages wave-group stagger (barrier asymmetry)

    Stages are called as ``stage.execute(infra, **shared_regs_subset)`` where
    the subset is exactly the wire's declared ``inp``. The returned dict is
    merged into ``shared_regs``.
    """

    MFMA_LATENCY_CYCLES = 16
    VALU_LATENCY_CYCLES = 1
    MEMORY_LATENCY_CYCLES = 400

    def __init__(
        self,
        wires: Sequence[Wire],
        *,
        max_depth: int = 3,
        force_depth: Optional[int] = None,
        enable_stagger: bool = True,
        lds_capacity_bytes: int = 160 * 1024,
    ):
        self.wires = list(wires)
        self.max_depth = max_depth
        self.force_depth = force_depth
        self.enable_stagger = enable_stagger
        self.lds_capacity_bytes = lds_capacity_bytes

        self._validate_dataflow()
        self._config = self._build_config()
        # The prime stage is the first entry of the first cluster — the async
        # DMA that must run ahead of everything else in the pipeline.
        self._prime_fn = self._config.clusters[0].entries[0].fn
        # For the epilogue drain: collect the fn of the LAST sub-stage of any
        # decomposed stage. The last sub-stage already consumed the carry in
        # the main loop's final iteration; running it again would double-count.
        self._decomposed_last_fns = set()
        for w in self.wires:
            if w.stage.max_slots > 1 and self._config.depth > 1:
                subs = w.stage.decompose(min(w.stage.max_slots, self._config.depth))
                if len(subs) > 1:
                    self._decomposed_last_fns.add(subs[-1].fn)
        # self._dump_config()  # uncomment for pipeline debug

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def _dump_config(self) -> None:
        """Print the pipeline structure for debugging."""
        cfg = self._config
        print(f"\n{'='*70}")
        print(f"PIPELINE CONFIG: depth={cfg.depth}, lds_ring_slots={cfg.lds_ring_slots}")
        print(f"  loop_carried: {cfg.loop_carried}")
        print(f"  enable_stagger: {self.enable_stagger}")
        print(f"\n  WIRES ({len(self.wires)}):")
        for i, w in enumerate(self.wires):
            print(f"    [{i}] {w.stage.name} ({w.stage.kind.name})"
                  f"  inp={w.inp}  out={w.out}  carry={w.carry}"
                  f"  max_slots={w.stage.max_slots}")
        print(f"\n  CLUSTERS ({len(cfg.clusters)}):")
        for c in cfg.clusters:
            entry_names = [e.name for e in c.entries]
            entry_ios = [(e.inp, e.out) for e in c.entries]
            print(f"    C{c.index} ({c.kind.name}): {entry_names}")
            for e in c.entries:
                print(f"      {e.name}: inp={e.inp} out={e.out}")
        print(f"{'='*70}\n")

    # ── Dataflow validation ──────────────────────────────────────────────

    def _validate_dataflow(self) -> None:
        available: set[str] = set()
        for w in self.wires:
            available.update(w.carry)

        for w in self.wires:
            missing = set(w.inp) - available
            if missing:
                raise ValueError(
                    f"Stage '{w.stage.name}' reads {sorted(missing)}, "
                    f"but no prior stage or carry produces them. "
                    f"Available: {sorted(available)}"
                )
            available.update(w.out)

    # ── Config build (iterative depth ↔ slot allocation) ─────────────────

    def _build_config(self) -> PipelineConfig:
        depth = self._initial_depth_estimate()
        prev_depth = -1

        for _ in range(5):
            if depth == prev_depth:
                break
            prev_depth = depth
            clusters = self._assign_clusters(depth)
            depth = self._refine_depth(clusters)

        clusters = self._assign_clusters(depth)

        lds_per_slot = sum(
            w.stage.resources.lds_bytes for w in self.wires
            if w.stage.kind == StageKind.MEMORY
        )
        while depth > 2 and depth * lds_per_slot > self.lds_capacity_bytes:
            depth -= 1

        vmcnt_targets = {}
        for c in clusters:
            if c.kind == StageKind.MEMORY:
                vmcnt_targets[c.index] = c.total_dma

        loop_carried: set[str] = set()
        for w in self.wires:
            loop_carried.update(w.carry)

        return PipelineConfig(
            depth=depth,
            clusters=tuple(clusters),
            lds_ring_slots=max(depth, 2),  # always >=2 for double-buffered DMA
            lds_bytes_per_slot=lds_per_slot,
            vmcnt_targets=vmcnt_targets,
            loop_carried=tuple(sorted(loop_carried)),
        )

    def _dump_config(self) -> None:
        """Print the pipeline configuration for debugging."""
        cfg = self._config
        print(f"\n{'='*70}")
        print(f"PIPELINE CONFIG: depth={cfg.depth}, lds_ring_slots={cfg.lds_ring_slots}, "
              f"stagger={self.enable_stagger}")
        print(f"  loop_carried: {cfg.loop_carried}")
        print(f"  wires ({len(self.wires)}):")
        for i, w in enumerate(self.wires):
            decomposed = w.stage.max_slots > 1 and cfg.depth > 1
            print(f"    W{i}: {w.stage.name} ({w.stage.kind.name})"
                  f"  inp={w.inp}  out={w.out}  carry={w.carry}"
                  f"  {'DECOMPOSED' if decomposed else 'monolithic'}")
        print(f"  clusters ({len(cfg.clusters)}):")
        for c in cfg.clusters:
            entry_names = [e.name for e in c.entries]
            entry_io = [(e.name, e.inp, e.out) for e in c.entries]
            print(f"    C{c.index} ({c.kind.name}): {entry_names}")
            for name, inp, out in entry_io:
                print(f"      {name}: inp={inp} out={out}")
        print(f"{'='*70}\n")

    def _initial_depth_estimate(self) -> int:
        if self.force_depth is not None:
            return self.force_depth
        total = sum(
            w.stage.resources.mfma_count * self.MFMA_LATENCY_CYCLES
            + w.stage.resources.valu_count * self.VALU_LATENCY_CYCLES
            + w.stage.resources.exp_count * self.VALU_LATENCY_CYCLES
            for w in self.wires if w.stage.kind == StageKind.COMPUTE
        )
        if total <= 0:
            return 2
        raw = (self.MEMORY_LATENCY_CYCLES + total - 1) // total + 1
        return min(self.max_depth, max(2, raw))

    def _assign_clusters(self, depth: int) -> list[ClusterSpec]:
        # Expand wires with decomposition. When a stage decomposes into N
        # sub-stages, the FIRST sub-stage is placed at the wire's position
        # in the list; subsequent sub-stages are DEFERRED and inserted at
        # the next occurrence of a cluster of the same StageKind. This
        # interleaves decomposed sub-stages with other stages, producing
        # the target cluster structure:
        #
        #   Wire order: LoadKV(mem) ReadKV(mem,2) Gemm1(comp) Softmax(comp,2) BridgeP(mem) Gemm2(comp)
        #   Expanded:   LoadKV ReadK | Gemm1 SoftmaxFinish | BridgeP ReadV | SoftmaxStart Gemm2
        #   Clusters:   C0(mem)        C1(comp)              C2(mem)          C3(comp)

        # Phase 1: expand wires, collecting deferred sub-stages per kind.
        expanded: list[tuple[StageKind, ClusterEntry]] = []
        deferred: dict[StageKind, list[ClusterEntry]] = {
            StageKind.MEMORY: [], StageKind.COMPUTE: [],
        }

        for w in self.wires:
            stage = w.stage
            # Before placing this wire, flush any deferred sub-stages of
            # the SAME kind into the expanded list (they attach to the
            # current cluster of their kind).
            if deferred[stage.kind]:
                for d in deferred[stage.kind]:
                    expanded.append((stage.kind, d))
                deferred[stage.kind] = []

            if stage.max_slots > 1:
                avail = max(stage.min_slots, min(stage.max_slots, depth))
                if avail > 1:
                    subs = stage.decompose(avail)
                    sub_res = stage.resources_for_slots(avail)
                    for i, (ss, sr) in enumerate(zip(subs, sub_res)):
                        s_inp = w.inp if i == 0 else ()
                        s_out = w.out if i == len(subs) - 1 else ()
                        entry = ClusterEntry(
                            name=ss.name, fn=ss.fn,
                            inp=s_inp, out=s_out, resources=sr,
                        )
                        if i == 0:
                            expanded.append((stage.kind, entry))
                        else:
                            deferred[stage.kind].append(entry)
                    continue

            expanded.append((stage.kind, ClusterEntry(
                name=stage.name, fn=stage.execute,
                inp=w.inp, out=w.out, resources=stage.resources,
            )))

        # Flush any remaining deferred sub-stages at the end.
        for kind in (StageKind.MEMORY, StageKind.COMPUTE):
            for d in deferred[kind]:
                expanded.append((kind, d))

        # Phase 2: group into clusters on kind change.
        clusters: list[ClusterSpec] = []
        cur_kind = StageKind.MEMORY
        cur_entries: list[ClusterEntry] = []
        idx = 0

        for kind, entry in expanded:
            if kind != cur_kind:
                if cur_entries:
                    clusters.append(ClusterSpec(idx, cur_kind, tuple(cur_entries)))
                    idx += 1
                    cur_entries = []
                cur_kind = kind
            cur_entries.append(entry)
        if cur_entries:
            clusters.append(ClusterSpec(idx, cur_kind, tuple(cur_entries)))

        return clusters

    def _refine_depth(self, clusters: list[ClusterSpec]) -> int:
        if self.force_depth is not None:
            return self.force_depth
        total = sum(
            c.total_mfma * self.MFMA_LATENCY_CYCLES
            + c.total_valu * self.VALU_LATENCY_CYCLES
            + c.total_exp * self.VALU_LATENCY_CYCLES
            for c in clusters if c.kind == StageKind.COMPUTE
        )
        if total <= 0:
            return 2
        raw = (self.MEMORY_LATENCY_CYCLES + total - 1) // total + 1
        return min(self.max_depth, max(2, raw))

    # ── Cluster execution helper ─────────────────────────────────────────

    def _execute_cluster(self, cluster: ClusterSpec, infra: InfraContext,
                         shared_regs: dict) -> None:
        """Run all entries in a cluster, passing/collecting shared regs.

        All shared_regs are passed as kwargs (not just entry.inp) so that
        decomposed sub-stages can access carry values that aren't in the
        Wire's declared inp. Each stage's ``**_`` absorbs unused keys.
        """
        for entry in cluster.entries:
            result = entry.fn(infra, **shared_regs)
            if result:
                shared_regs.update(result)

    # ── Sync barrier helpers ────────────────────────────────────────────

    @staticmethod
    def _dualwave_sync_barrier():
        """Cluster boundary: sched_barrier(0) + s_barrier + sched_barrier(0).

        Reproduces flash_attn_utils.py:69-72 locally (no import from
        existing kernels). sched_barrier(0) prevents the compiler from
        reordering any instruction across the barrier.
        """
        from flydsl.expr import rocdl
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    @staticmethod
    def _stagger_open(stagger_i32):
        """Open wave-group phase shift: group B gets one extra s_barrier."""
        from flydsl.expr import rocdl
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith
        from flydsl.expr.typing import T
        is_group_b = arith.cmpi(arith.CmpIPredicate.ne, stagger_i32, arith.constant(0, T.i32))
        _if = scf.IfOp(is_group_b)
        with _if.then_block:
            rocdl.s_barrier()
            scf.YieldOp([])
        with _if.else_block:
            scf.YieldOp([])

    @staticmethod
    def _stagger_close(stagger_i32):
        """Close wave-group phase shift: group A gets one extra s_barrier."""
        from flydsl.expr import rocdl
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith
        from flydsl.expr.typing import T
        is_group_a = arith.cmpi(arith.CmpIPredicate.eq, stagger_i32, arith.constant(0, T.i32))
        _if = scf.IfOp(is_group_a)
        with _if.then_block:
            rocdl.s_barrier()
            scf.YieldOp([])
        with _if.else_block:
            scf.YieldOp([])

    # ── Code emission ────────────────────────────────────────────────────

    def _is_prime(self, entry: ClusterEntry) -> bool:
        """True if this entry is the pipeline's prime stage (first entry of
        first cluster — the async DMA that must run ahead of everything else)."""
        return entry.fn is self._prime_fn

    def _run_non_prime(self, cluster: ClusterSpec, infra: InfraContext,
                       shared_regs: dict, label: str = "") -> None:
        """Run all entries in a cluster EXCEPT the prime stage."""
        for entry in cluster.entries:
            if not self._is_prime(entry):
                if label:
                    print(f"  {label} C{cluster.index}({cluster.kind.name}): {entry.name}")
                result = entry.fn(infra, **shared_regs)
                if result:
                    shared_regs.update(result)

    def emit_prologue(self, infra: InfraContext, shared_regs: dict) -> None:
        """Prime the pipeline by running the first stage ``depth-1`` tiles ahead.

        - depth=1: no priming needed (all stages run in lockstep per tile).
        - depth>=2: run the prime stage for the first tile(s), wait, barrier.
        """
        from flydsl.expr import rocdl

        depth = self._config.depth

        print(f"[PROLOGUE] depth={depth}")
        if depth >= 2:
            for t in range(depth - 1):
                infra.tile_idx = t
                infra.buf_slot = t % self._config.lds_ring_slots
                print(f"  [PRIME] tile={t} buf={infra.buf_slot}")
                self._prime_fn(infra)
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()

        if self.enable_stagger and infra.stagger_i32 is not None:
            self._stagger_open(infra.stagger_i32)
        elif depth >= 2:
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

    def emit_main_loop(self, infra: InfraContext, shared_regs: dict,
                       n_kv_tiles: int) -> None:
        """Emit the main KV-tile loop.

        - depth=1: each iteration runs the prime stage synchronously (issue +
          wait + barrier), then all remaining stages.
        - depth>=2: prefetch the prime stage for the next tile (async), then
          run all non-prime stages on the current tile, drain at the bottom.
        """
        from flydsl.expr import const_expr, range_constexpr, rocdl

        depth = self._config.depth
        clusters = self._config.clusters
        staggered = self.enable_stagger and infra.stagger_i32 is not None
        print(f"[MAIN_LOOP] n_kv_tiles={n_kv_tiles}, staggered={staggered}, "
              f"num_clusters={len(clusters)}")
        for kv in range_constexpr(n_kv_tiles):
            read_buf = kv % self._config.lds_ring_slots
            write_buf = (kv + 1) % self._config.lds_ring_slots
            infra.buf_slot = read_buf

            if staggered:
                for ci, cluster in enumerate(clusters):
                    infra.cluster_id = ci
                    if cluster.kind == StageKind.MEMORY:
                        if const_expr(kv + depth - 1 < n_kv_tiles):
                            infra.tile_idx = kv + depth - 1
                            infra.buf_slot = write_buf
                            self._execute_cluster(cluster, infra, shared_regs)
                            infra.buf_slot = read_buf
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    else:
                        self._execute_cluster(cluster, infra, shared_regs)
                    self._dualwave_sync_barrier()

            elif depth == 1:
                # Synchronous: prime stage → wait → all other stages.
                infra.tile_idx = kv
                print(f"[LOOP d1 kv={kv}] prime tile={kv} buf={infra.buf_slot}")
                self._prime_fn(infra)
                rocdl.s_waitcnt(vmcnt=0)
                rocdl.s_barrier()
                for c in clusters:
                    self._run_non_prime(c, infra, shared_regs, label=f"[LOOP d1 kv={kv}]")

            else:
                if const_expr(kv + 1 < n_kv_tiles):
                    infra.tile_idx = kv + 1
                    infra.buf_slot = write_buf
                    print(f"[LOOP d2 kv={kv}] prefetch prime tile={kv+1} buf={write_buf}")
                    self._prime_fn(infra)
                    infra.buf_slot = read_buf

                for c in clusters:
                    self._run_non_prime(c, infra, shared_regs, label=f"[LOOP d2 kv={kv}]")

                rocdl.s_waitcnt(vmcnt=0)
                rocdl.s_barrier()

    def emit_epilogue(self, infra: InfraContext, shared_regs: dict) -> None:
        """Drain the pipeline after the main loop.

        - depth=1: nothing to drain.
        - depth>=2: run ``depth-1`` drain iterations through all non-prime
          stages (no new DMA, just process remaining loop-carried state).
        """
        from flydsl.expr import rocdl

        depth = self._config.depth
        clusters = self._config.clusters

        if depth <= 1:
            return

        if self.enable_stagger and infra.stagger_i32 is not None:
            for drain in range(depth - 1):
                for cluster in clusters:
                    infra.cluster_id = cluster.index
                    if cluster.kind == StageKind.MEMORY:
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    else:
                        self._execute_cluster(cluster, infra, shared_regs)
                    self._dualwave_sync_barrier()
            self._stagger_close(infra.stagger_i32)
        else:
            # Drain: run all non-prime stages, skipping the LAST sub-stage
            # of any decomposed stage (it already consumed the carry in the
            # main loop's final iteration — running it again would double-count).
            print(f"[EPILOGUE] depth={depth}, drains={depth-1}")
            for drain in range(depth - 1):
                for c in clusters:
                    for entry in c.entries:
                        if self._is_prime(entry):
                            continue
                        if entry.fn in self._decomposed_last_fns:
                            print(f"  [DRAIN {drain}] SKIP C{c.index}: {entry.name} (decomposed last)")
                            continue
                        print(f"  [DRAIN {drain}] C{c.index}({c.kind.name}): {entry.name}")
                        result = entry.fn(infra, **shared_regs)
                        if result:
                            shared_regs.update(result)

