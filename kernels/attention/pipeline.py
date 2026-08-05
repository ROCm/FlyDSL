# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Stage-based composable software pipeline for flex attention kernels.

A ``PipelineStage`` declares *what* to compute (resource cost, execute logic,
optional multi-slot decomposition). A ``Wire`` declares *how stages connect*
(which shared registers flow between them). A ``PipelineScheduler`` takes the
wire list and emits a fully pipelined kernel loop — cluster assignment, depth,
stagger, prologue/epilogue, vmcnt, and scheduling hints are all derived
automatically.

**Pipeline depth (``depth`` / ``force_depth``):** depth 1 runs all stages
lockstep per KV tile. Depth >= 2 is *software* double-buffering: two LDS ring
slots, prologue DMA ahead, per-iteration prefetch of the next tile's K/V, and
loop-carried softmax state (e.g. partial P between tiles). The main loop still
advances one ``tile_idx`` per iteration; overlap comes from prefetch + carried
regs, not from processing two tile indices in one loop trip.

**Prefetch-only substages** (``SubStage.prefetch_only``): skipped during normal
cluster execution but run during prologue tail and memory-cluster prefetch
(``infra.prefetch_pass``).

**Stagger:** when ``pipeline_stagger_enabled`` (depth >= 2, ``num_groups`` and
``m_waves`` >= 2), wave groups get asymmetric barriers at pipeline open/close.

**Manual cluster emit (stagger main loop):** ``emit_prologue`` / ``emit_epilogue``
are unchanged. For hand scheduling between clusters, call after prologue::

    for kv in range_constexpr(n_kv_tiles):
        pipeline.emit_tile_stagger_kv(kv, n_kv_tiles, infra, shared_regs)
    # or per cluster:
        pipeline.emit_tile_memory_cluster(kv, 0, infra, shared_regs, n_kv_tiles, ...)
        pipeline.emit_cluster_boundary_sync(0)
        pipeline.emit_tile_compute_cluster(kv, 1, ...)
        ...

Requires ``enable_stagger`` and ``infra.stagger_i32`` (flex layout pd>=2).
See ``emit_tile_memory_cluster``, ``emit_tile_compute_cluster``,
``emit_cluster_boundary_sync``, ``emit_tile_stagger_kv``.

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

_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
_LGKMCNT_0_ONLY_FALLBACK = 0xC07F


def _rocdl_waitcnt_vm_n(n: int) -> None:
    from flydsl.expr import rocdl

    val = (
        (n & _VMCNT_LO_MASK)
        | _LGKMCNT_EXPCNT_BASE
        | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    )
    rocdl.s_waitcnt(val)


def _rocdl_waitcnt_vmcnt0() -> None:
    from flydsl.expr import rocdl

    rocdl.s_waitcnt(0)


# ---------------------------------------------------------------------------
# Enums and value types
# ---------------------------------------------------------------------------


class StageKind(Enum):
    """Hardware resource category for cluster assignment."""
    MEMORY = auto()
    COMPUTE = auto()


@dataclass(frozen=True)
class WaitFull:
    """Drain VM and LDS/GDS (``s_waitcnt vmcnt=0 lgkmcnt=0``)."""


@dataclass(frozen=True)
class WaitLgkmOnly:
    """Drain LDS/GDS only; VM loads may remain in flight."""


@dataclass(frozen=True)
class WaitVmcntAtMost:
    """Wait until at most ``n`` global loads are outstanding (``vmcnt(n)``).

    If ``n`` is ``None``, the emitter uses ``PipelineConfig.vmcnt_targets`` for
    this cluster index, or an explicit ``vmcnt_at_most`` override at emit time.
    """
    n: Optional[int] = None


WaitPolicy = WaitFull | WaitLgkmOnly | WaitVmcntAtMost


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
    prefetch_pass: bool = False  # True while emitting prefetch-only DMA substages


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

    @property
    def epilogue_drain(self) -> bool:
        """If True, this stage's cluster entry runs in the depth>=2 epilogue tail."""
        return False

    @property
    def epilogue_fn(self) -> Optional[Callable]:
        """Optional callable for epilogue tail; None means use ``execute``."""
        return None

    @property
    def sync_before(self) -> bool:
        """If True, emit a pipeline sync barrier immediately before ``execute``."""
        return False

    @property
    def sync_after(self) -> bool:
        """If True, emit a pipeline sync barrier after this entry when the next runs."""
        return False

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
        return [SubStage(
            self.name, self.execute, self.resources,
            epilogue_drain=self.epilogue_drain,
            epilogue_fn=self.epilogue_fn,
            sync_before=self.sync_before,
            sync_after=self.sync_after,
            sched_after=self.sched_hints,
        )]

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
        """Optional scheduler hook invoked after this stage's cluster entry runs.

        Signature: ``(infra: InfraContext, cluster_index: int, entry_index: int) -> None``.

        Emit ``sched_group_barrier`` (or similar) here — not in pipeline sync
        barriers. Wired automatically for non-decomposed stages; decomposed
        stages may set ``SubStage.sched_after`` per sub-stage instead.
        """
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
    epilogue_drain: bool = False
    epilogue_fn: Optional[Callable] = None
    sync_before: bool = False
    sync_after: bool = False
    defer_to_next_cluster: bool = True
    sched_after: Optional[Callable] = None
    prefetch_only: bool = False


# ---------------------------------------------------------------------------
# Wire — declares inter-stage connections at the assembly site
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Wire:
    """Connects a stage to the pipeline's shared register namespace.

    Declared by the pipeline author when assembling the stage list::

        PipelineScheduler([
            Wire(StageA(),            out=("buf_a",)),
            Wire(StageB(),            out=("buf_b",)),
            Wire(StageC(),            inp=("buf_a", "buf_b"),
                                     out=("acc",)),
            Wire(StageD(),            inp=("acc",),
                                     out=("out",),
                                     carry=("state",)),
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
    epilogue_drain: bool = False
    epilogue_fn: Optional[Callable] = None
    sync_before: bool = False
    sync_after: bool = False
    sched_after: Optional[Callable] = None
    prefetch_only: bool = False


@dataclass(frozen=True)
class ClusterSpec:
    """A group of entries executed between two sync barriers."""
    index: int
    kind: StageKind
    entries: tuple[ClusterEntry, ...]
    wait_policies: tuple[WaitPolicy, ...] = (WaitFull(),)

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


def pipeline_stagger_enabled(*, depth: int, num_groups: int, m_waves: int) -> bool:
    """True when wave-group stagger + multi-group Strategy A layout is valid."""
    return depth >= 2 and num_groups >= 2 and m_waves >= 2


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
        num_groups: int = 1,
        m_waves: int = 2,
        light_c1_c2_boundary: bool = False,
        inter_tile_readahead: bool = False,
    ):
        self.wires = list(wires)
        self.max_depth = max_depth
        self.force_depth = force_depth
        self.lds_capacity_bytes = lds_capacity_bytes
        self._light_c1_c2_boundary = light_c1_c2_boundary
        self._inter_tile_readahead = inter_tile_readahead

        self._validate_dataflow()
        self._config = self._build_config()
        self.enable_stagger = enable_stagger and pipeline_stagger_enabled(
            depth=self._config.depth,
            num_groups=num_groups,
            m_waves=m_waves,
        )
        if force_depth is not None and force_depth >= 2 and not self.enable_stagger:
            raise ValueError(
                "pipeline depth>=2 requires stagger (num_groups>=2 and m_waves>=2)"
            )
        # The prime stage is the first entry of the first cluster — the async
        # DMA that must run ahead of everything else in the pipeline.
        self._prime_fn = self._config.clusters[0].entries[0].fn
        self._prime_vm_dma = self._config.clusters[0].entries[0].resources.dma_count
        # depth>=2 epilogue tail: entries flagged on Wire.stage / SubStage.
        self._epilogue_tail_entries: list[ClusterEntry] = []
        if self._config.depth > 1:
            for c in self._config.clusters:
                for e in c.entries:
                    if e.epilogue_drain:
                        self._epilogue_tail_entries.append(e)
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
            print(f"    C{c.index} ({c.kind.name}): {entry_names}"
                  f"  wait={c.wait_policies}")
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
        clusters = list(self._clusters_with_wait_policies(clusters))

        lds_per_slot = sum(
            w.stage.resources.lds_bytes for w in self.wires
            if w.stage.kind == StageKind.MEMORY
        )
        while depth > 2 and depth * lds_per_slot > self.lds_capacity_bytes:
            depth -= 1
            clusters = list(self._clusters_with_wait_policies(
                self._assign_clusters(depth),
            ))

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

    def _clusters_with_wait_policies(
        self, clusters: list[ClusterSpec],
    ) -> tuple[ClusterSpec, ...]:
        """Attach per-cluster wait policies (default full drain)."""
        out: list[ClusterSpec] = []
        for c in clusters:
            if c.kind == StageKind.MEMORY:
                # After cluster body + optional prefetch: drain LDS (lgkm) then
                # bound in-flight VM DMA (partial vmcnt), matching flash mem clusters.
                if c.total_dma > 0:
                    policies: tuple[WaitPolicy, ...] = (
                        WaitLgkmOnly(),
                        WaitVmcntAtMost(None),
                    )
                else:
                    policies = (WaitLgkmOnly(),)
            else:
                policies = (WaitFull(),)
            out.append(ClusterSpec(
                c.index, c.kind, c.entries, wait_policies=policies,
            ))
        return tuple(out)

    def _kv_dma_in_flight_bound(self) -> int:
        """Sum of per-cluster DMA ops (LoadK + LoadV) for flash-style partial VM waits.

        Used after steady-state memory-cluster prefetch only; prologue still drains fully.
        """
        return sum(
            c.total_dma for c in self._config.clusters
            if c.kind == StageKind.MEMORY
        )

    def _memory_cluster_vmcnt_after_prefetch(
        self,
        cluster: ClusterSpec,
        prefetch_vmcnt: int,
    ) -> int:
        """VM wait count after issuing this cluster's prefetch DMA."""
        if prefetch_vmcnt > 0:
            kv = self._kv_dma_in_flight_bound()
            return kv if kv > 0 else prefetch_vmcnt
        return self._config.vmcnt_targets.get(cluster.index, 0)

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
        #   Wire order: LoadKV(mem,2) ReadKV(mem,2) Gemm1(comp) Softmax(comp,2) BridgeP(mem) Gemm2(comp)
        #   Expanded:   LoadK ReadK | Gemm1 SoftmaxFinish | LoadV ReadV BridgeP | SoftmaxStart Gemm2
        #   Clusters:   C0(mem)        C1(comp)              C2(mem)              C3(comp)

        # Phase 1: expand wires, collecting deferred sub-stages per kind.
        expanded: list[tuple[StageKind, ClusterEntry]] = []
        deferred: dict[StageKind, list[ClusterEntry]] = {
            StageKind.MEMORY: [], StageKind.COMPUTE: [],
        }
        prev_wire_kind: Optional[StageKind] = None

        for w in self.wires:
            stage = w.stage
            # Flush deferred sub-stages when entering a new kind region (e.g. after
            # compute), not on every consecutive wire of the same kind — so LoadV
            # can stay deferred past ReadKV and land in C2 with ReadV.
            if deferred[stage.kind] and (
                prev_wire_kind is None or prev_wire_kind != stage.kind
            ):
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
                            epilogue_drain=ss.epilogue_drain,
                            epilogue_fn=ss.epilogue_fn,
                            sync_before=ss.sync_before,
                            sync_after=ss.sync_after,
                            sched_after=ss.sched_after,
                            prefetch_only=ss.prefetch_only,
                        )
                        if i == 0 or not ss.defer_to_next_cluster:
                            expanded.append((stage.kind, entry))
                        else:
                            deferred[stage.kind].append(entry)
                    prev_wire_kind = stage.kind
                    continue

            expanded.append((stage.kind, ClusterEntry(
                name=stage.name, fn=stage.execute,
                inp=w.inp, out=w.out, resources=stage.resources,
                epilogue_drain=stage.epilogue_drain,
                epilogue_fn=stage.epilogue_fn,
                sync_before=stage.sync_before,
                sync_after=stage.sync_after,
                sched_after=stage.sched_hints,
            )))

            prev_wire_kind = stage.kind

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

    def _pipeline_sync_barrier(self) -> None:
        """Shared full workgroup sync (cluster boundary and entry handoffs)."""
        self._dualwave_sync_barrier()

    def _run_cluster_entries(
        self,
        cluster: ClusterSpec,
        infra: InfraContext,
        shared_regs: dict,
        *,
        skip_prime: bool = False,
        skip_read_k: bool = False,
        skip_read_v: bool = False,
    ) -> None:
        """Run cluster entries with pipeline sync at declared handoff points."""
        entries = cluster.entries
        prev_idx = -1
        for i, entry in enumerate(entries):
            if skip_prime and self._is_prime(entry):
                continue
            if skip_read_k and entry.name == "ReadK":
                continue
            if skip_read_v and entry.name == "ReadV":
                continue
            if entry.prefetch_only and not infra.prefetch_pass:
                continue
            if prev_idx >= 0:
                prev = entries[prev_idx]
                if prev.sync_after or entry.sync_before:
                    self._pipeline_sync_barrier()
            result = entry.fn(infra, **shared_regs)
            if result:
                shared_regs.update(result)
            if entry.sched_after is not None:
                entry.sched_after(infra, cluster.index, i)
            prev_idx = i

    def _execute_cluster(self, cluster: ClusterSpec, infra: InfraContext,
                         shared_regs: dict) -> None:
        """Run all entries in a cluster, passing/collecting shared regs."""
        self._run_cluster_entries(cluster, infra, shared_regs, skip_prime=False)

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
    def _sched_only_cluster_sync():
        """Compiler scheduling fence only (no WG ``s_barrier``)."""
        from flydsl.expr import rocdl
        rocdl.sched_barrier(0)

    def _emit_stagger_cluster_boundary_sync(self, cluster_index: int) -> None:
        """After cluster ``cluster_index`` completes in the stagger main loop."""
        if (
            self._light_c1_c2_boundary
            and cluster_index == 1
            and len(self._config.clusters) == 4
        ):
            self._sched_only_cluster_sync()
        else:
            self._dualwave_sync_barrier()

    @staticmethod
    def _stagger_open(stagger_i32):
        """Open wave-group phase shift: group B gets one extra s_barrier."""
        from flydsl._mlir import ir
        from flydsl.expr import rocdl
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith
        from flydsl.expr.typing import T
        is_group_b = arith.cmpi(arith.CmpIPredicate.ne, stagger_i32, arith.constant(0, type=T.i32))
        _if = scf.IfOp(is_group_b, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            rocdl.s_barrier()
            scf.YieldOp([])

    @staticmethod
    def _stagger_close(stagger_i32):
        """Close wave-group phase shift: group A gets one extra s_barrier."""
        from flydsl._mlir import ir
        from flydsl.expr import rocdl
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith
        from flydsl.expr.typing import T
        is_group_a = arith.cmpi(arith.CmpIPredicate.eq, stagger_i32, arith.constant(0, type=T.i32))
        _if = scf.IfOp(is_group_a, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            rocdl.s_barrier()
            scf.YieldOp([])

    # ── Code emission ────────────────────────────────────────────────────

    def _is_prime(self, entry: ClusterEntry) -> bool:
        """True if this entry is the pipeline's prime stage (first entry of
        first cluster — the async DMA that must run ahead of everything else)."""
        return entry.fn is self._prime_fn

    def _cluster_prefetch_entry(self, cluster: ClusterSpec) -> Optional[ClusterEntry]:
        """First DMA substage in this memory cluster (LoadK at C0, LoadV at C2, …)."""
        for entry in cluster.entries:
            if entry.resources.dma_count > 0:
                return entry
        return None

    def _emit_cluster_prefetch(self, cluster: ClusterSpec, infra: InfraContext) -> int:
        """Issue async DMA for the next tile from this memory cluster; return vmcnt."""
        entry = self._cluster_prefetch_entry(cluster)
        if entry is not None:
            infra.prefetch_pass = True
            entry.fn(infra)
            infra.prefetch_pass = False
            return entry.resources.dma_count
        self._prime_fn(infra)
        return self._prime_vm_dma

    def _emit_cluster_waitcnt(
        self,
        cluster: ClusterSpec,
        infra: InfraContext,
        *,
        vmcnt_at_most: Optional[int] = None,
    ) -> None:
        """Emit ``s_waitcnt`` sequence for a cluster per ``wait_policies``."""
        from flydsl.expr import rocdl

        for policy in cluster.wait_policies:
            if isinstance(policy, WaitFull):
                _rocdl_waitcnt_vmcnt0()
            elif isinstance(policy, WaitLgkmOnly):
                traits = infra.traits
                lgkm_only = getattr(traits, "LGKMCNT_0_ONLY", None) if traits else None
                if lgkm_only is not None:
                    rocdl.s_waitcnt(lgkm_only)
                else:
                    rocdl.s_waitcnt(_LGKMCNT_0_ONLY_FALLBACK)
            elif isinstance(policy, WaitVmcntAtMost):
                n = vmcnt_at_most
                if n is None:
                    n = policy.n
                if n is None:
                    n = self._config.vmcnt_targets.get(cluster.index, 0)
                _rocdl_waitcnt_vm_n(n)
            else:
                raise TypeError(f"unknown wait policy: {policy!r}")

    def _run_non_prime(
        self,
        cluster: ClusterSpec,
        infra: InfraContext,
        shared_regs: dict,
        label: str = "",
        *,
        skip_read_k: bool = False,
        skip_read_v: bool = False,
    ) -> None:
        """Run all entries in a cluster EXCEPT the prime stage."""
        self._run_cluster_entries(
            cluster, infra, shared_regs,
            skip_prime=True,
            skip_read_k=skip_read_k,
            skip_read_v=skip_read_v,
        )

    @staticmethod
    def _cluster_entry(cluster: ClusterSpec, name: str) -> ClusterEntry:
        for entry in cluster.entries:
            if entry.name == name:
                return entry
        raise KeyError(f"cluster C{cluster.index} has no entry {name!r}")

    def _emit_intertile_reads_for_next(
        self,
        next_kv: int,
        infra: InfraContext,
        shared_regs: dict,
        clusters: tuple,
    ) -> None:
        """ReadK for ``next_kv`` after C3 on the previous tile (hoist from next C0).

        ReadV stays in C2 — ``frag_Vt_next`` must remain tile-current through C3 Gemm2.
        """
        c0 = clusters[0]
        slot = next_kv % self._config.lds_ring_slots
        infra.tile_idx = next_kv
        infra.buf_slot = slot
        entry = self._cluster_entry(c0, "ReadK")
        result = entry.fn(infra, **shared_regs)
        if result:
            shared_regs.update(result)

    def _emit_stagger_memory_cluster(
        self,
        cluster: ClusterSpec,
        kv: int,
        n_kv_tiles: int,
        depth: int,
        infra: InfraContext,
        shared_regs: dict,
        *,
        skip_read_k: bool,
        skip_read_v: bool,
    ) -> None:
        """C0 or C2 body: optional skipped reads, prefetch next tile, waitcnt."""
        from flydsl.expr import const_expr

        read_buf = kv % self._config.lds_ring_slots
        write_buf = (kv + 1) % self._config.lds_ring_slots
        infra.tile_idx = kv
        infra.buf_slot = read_buf
        self._run_non_prime(
            cluster, infra, shared_regs,
            skip_read_k=skip_read_k,
            skip_read_v=skip_read_v,
        )
        if const_expr(kv + depth - 1 < n_kv_tiles):
            infra.tile_idx = kv + depth - 1
            infra.buf_slot = write_buf
            vm_prefetch = self._emit_cluster_prefetch(cluster, infra)
            infra.buf_slot = read_buf
            vm_wait = self._memory_cluster_vmcnt_after_prefetch(
                cluster, vm_prefetch,
            )
            self._emit_cluster_waitcnt(
                cluster, infra, vmcnt_at_most=vm_wait,
            )
        else:
            self._emit_cluster_waitcnt(
                cluster, infra, vmcnt_at_most=0,
            )

    def _require_cluster(self, cluster_index: int, kind: StageKind) -> ClusterSpec:
        clusters = self._config.clusters
        if cluster_index < 0 or cluster_index >= len(clusters):
            raise ValueError(
                f"cluster_index {cluster_index} out of range "
                f"(pipeline has {len(clusters)} clusters)"
            )
        cluster = clusters[cluster_index]
        if cluster.kind != kind:
            raise ValueError(
                f"cluster C{cluster_index} is {cluster.kind.name}, expected {kind.name}"
            )
        return cluster

    def emit_cluster_boundary_sync(self, cluster_index: int) -> None:
        """After cluster ``cluster_index`` in a manual stagger tile (dualwave or light C1→C2)."""
        self._emit_stagger_cluster_boundary_sync(cluster_index)

    def emit_tile_memory_cluster(
        self,
        kv: int,
        cluster_index: int,
        infra: InfraContext,
        shared_regs: dict,
        n_kv_tiles: int,
        *,
        skip_read_k: bool = False,
        skip_read_v: bool = False,
    ) -> None:
        """One memory cluster for tile ``kv``: reads, prefetch, partial vmcnt (stagger path)."""
        cluster = self._require_cluster(cluster_index, StageKind.MEMORY)
        depth = self._config.depth
        infra.cluster_id = cluster_index
        self._emit_stagger_memory_cluster(
            cluster, kv, n_kv_tiles, depth, infra, shared_regs,
            skip_read_k=skip_read_k,
            skip_read_v=skip_read_v,
        )

    def emit_tile_compute_cluster(
        self,
        kv: int,
        cluster_index: int,
        infra: InfraContext,
        shared_regs: dict,
        n_kv_tiles: int,
        *,
        inter_tile_readahead_after: bool = False,
    ) -> None:
        """One compute cluster for tile ``kv``; optional ReadK hoisted after C3."""
        from flydsl.expr import const_expr

        cluster = self._require_cluster(cluster_index, StageKind.COMPUTE)
        infra.cluster_id = cluster_index
        clusters = self._config.clusters
        self._execute_cluster(cluster, infra, shared_regs)
        if inter_tile_readahead_after and self._inter_tile_readahead:
            if const_expr(kv + 1 < n_kv_tiles):
                self._emit_intertile_reads_for_next(
                    kv + 1, infra, shared_regs, tuple(clusters),
                )

    def emit_tile_stagger_kv(
        self,
        kv: int,
        n_kv_tiles: int,
        infra: InfraContext,
        shared_regs: dict,
    ) -> None:
        """One KV tile: C0→sync→C1→sync→C2→sync→C3→sync (same rules as ``emit_main_loop`` stagger)."""
        from flydsl.expr import const_expr

        clusters = self._config.clusters
        read_buf = kv % self._config.lds_ring_slots
        infra.buf_slot = read_buf
        for ci, cluster in enumerate(clusters):
            infra.cluster_id = ci
            if cluster.kind == StageKind.MEMORY:
                if self._inter_tile_readahead and cluster.index == 0:
                    if const_expr(kv > 0):
                        self.emit_tile_memory_cluster(
                            kv, cluster.index, infra, shared_regs, n_kv_tiles,
                            skip_read_k=True, skip_read_v=False,
                        )
                    else:
                        self.emit_tile_memory_cluster(
                            kv, cluster.index, infra, shared_regs, n_kv_tiles,
                            skip_read_k=False, skip_read_v=False,
                        )
                elif self._inter_tile_readahead and cluster.index == 2:
                    self.emit_tile_memory_cluster(
                        kv, cluster.index, infra, shared_regs, n_kv_tiles,
                        skip_read_k=False, skip_read_v=False,
                    )
                else:
                    self.emit_tile_memory_cluster(
                        kv, cluster.index, infra, shared_regs, n_kv_tiles,
                        skip_read_k=False, skip_read_v=False,
                    )
            elif ci == 3 and self._inter_tile_readahead:
                self.emit_tile_compute_cluster(
                    kv, cluster.index, infra, shared_regs, n_kv_tiles,
                    inter_tile_readahead_after=True,
                )
            else:
                self.emit_tile_compute_cluster(
                    kv, cluster.index, infra, shared_regs, n_kv_tiles,
                )
            self.emit_cluster_boundary_sync(ci)

    def emit_prologue(self, infra: InfraContext, shared_regs: dict) -> None:
        """Prime the pipeline by running the first stage ``depth-1`` tiles ahead.

        - depth=1: no priming needed (all stages run in lockstep per tile).
        - depth>=2: run the prime stage for the first tile(s); full ``s_waitcnt(0)``
          + barrier (tile-0 K/V must land in LDS — not partial K+V vmcnt).
        """
        from flydsl.expr import rocdl

        depth = self._config.depth

        if depth >= 2:
            for t in range(depth - 1):
                infra.tile_idx = t
                infra.buf_slot = t % self._config.lds_ring_slots
                self._prime_fn(infra)
            # When K/V DMA are split, prime V for tile 0 (later tiles get V from C2 prefetch).
            clusters = self._config.clusters
            if len(clusters) >= 3 and clusters[2].kind == StageKind.MEMORY:
                v_entry = self._cluster_prefetch_entry(clusters[2])
                if v_entry is not None and v_entry.fn is not self._prime_fn:
                    infra.tile_idx = 0
                    infra.buf_slot = 0
                    infra.prefetch_pass = True
                    v_entry.fn(infra)
                    infra.prefetch_pass = False
                    _rocdl_waitcnt_vm_n(v_entry.resources.dma_count)
            rocdl.s_waitcnt(0)
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
        for kv in range_constexpr(n_kv_tiles):
            read_buf = kv % self._config.lds_ring_slots
            write_buf = (kv + 1) % self._config.lds_ring_slots
            infra.buf_slot = read_buf

            if staggered:
                self.emit_tile_stagger_kv(kv, n_kv_tiles, infra, shared_regs)

            elif depth == 1:
                # Synchronous: prime stage → wait → all other stages.
                infra.tile_idx = kv
                self._prime_fn(infra)
                _rocdl_waitcnt_vmcnt0()
                rocdl.s_barrier()
                for c in clusters:
                    self._run_non_prime(c, infra, shared_regs)

            else:
                if const_expr(kv + 1 < n_kv_tiles):
                    infra.tile_idx = kv + 1
                    infra.buf_slot = write_buf
                    self._prime_fn(infra)
                    infra.buf_slot = read_buf

                for c in clusters:
                    self._run_non_prime(c, infra, shared_regs)
                    self._pipeline_sync_barrier()

                _rocdl_waitcnt_vmcnt0()
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
                        self._emit_cluster_waitcnt(
                            cluster, infra, vmcnt_at_most=0,
                        )
                    else:
                        self._execute_cluster(cluster, infra, shared_regs)
                    self._dualwave_sync_barrier()
            self._stagger_close(infra.stagger_i32)
        else:
            # Main loop leaves one tile of softmax mass and lagged P@V in
            # loop-carried regs; run only the tail stages (no new GEMM1 / DMA).
            tail = self._epilogue_tail_entries
            for i, entry in enumerate(tail):
                if i > 0 and (tail[i - 1].sync_after or entry.sync_before):
                    self._pipeline_sync_barrier()
                fn = entry.epilogue_fn if entry.epilogue_fn is not None else entry.fn
                result = fn(infra, **shared_regs)
                if result:
                    shared_regs.update(result)
            rocdl.s_barrier()
