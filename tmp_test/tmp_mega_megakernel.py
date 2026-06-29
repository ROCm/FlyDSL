# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Single-launch fused **dispatch ⊕ GEMM** megakernel for MoE stage-1 (gate/up), CDNA4 / MI355X.

Implements docs/moe_stage1_mega.md — dispatch and the persistent sparse-tile
group-GEMM run in ONE kernel launch (``scheme="fixedslot"``):

  * fixed-slot dispatch prologue → grid barrier → persistent sparse-tile GEMM (raw
    activation-scale fold).  Validated bit-exact.  For large batches the dispatch
    auto-switches to the COMPACT count-first layout (dodges the 4GB buffer wrap).

Reuses ``FlyDSLDispatchGroupMajorOp`` ONLY for its symmetric (mori-shmem / P2P) buffers and tables;
the dispatch *kernel* is NOT launched — the dispatch logic lives inside the GEMM kernel's prologue
(``compile_fused_moe_gemm1(fuse_dispatch=...)``).
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import torch
import torch.distributed as dist

import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

import flydsl.expr as fx

from .tmp_mega_ep_dispatch import FlyDSLDispatchGroupMajorOp
from .tmp_mega_gemm_2stage import compile_fused_moe_gemm1


def _is_fp4(dt):
    return dt == torch.float4_e2m1fn_x2


# ── Megastage1 default tile fallback ──────────────────────────────────────────────────────────────
# Tuned tiles live in the FlyDSL JSON table below.  When a shape is intentionally
# untuned, keep only the generic validated default instead of a second
# hand-maintained host-side tune table.
def _mega_default_tile(inter_dim):
    return 64, (256 if (int(inter_dim) % 256 == 0) else 128)


# ── Precise megastage1 tile, from the FlyDSL tune JSON (copied from aiter) ─────────────────────────
# The single tuned source for megastage1: the best (tile_m, tile_n, tile_k,
# waves_per_eu, use_async_copy) per per-rank token bucket, keyed by the per-rank
# GEMM1 shape (dtype, model_dim, inter_dim, experts_per_rank, topk).
#
# WHY ``experts_per_rank`` (NOT total ``expert``) is the key
# ---------------------------------------------------------
# Stage-1's GEMM1 runs PER RANK over this rank's ``epr = experts // world_size``
# local experts; per-rank received tokens are ~``mtpr*topk`` (independent of
# world_size, since each rank owns ``epr/experts`` of all routed slots), so the
# per-rank problem -- and thus the best tile -- is fully determined by
# ``(model_dim, inter_dim, epr, topk, mtpr)``.  ``world_size`` only enters through
# ``epr``.  Keying by ``experts_per_rank`` therefore makes a single tuned table
# valid across every ``(world_size, total_experts)`` combo that yields the same
# ``epr`` (and matches stage-2 gemm2's existing EP-local convention in
# ``mega_stage1_stage2._resolve_gemm2_tile``).  The legacy total-``expert`` /
# ``_ep{n}``-in-filename keying silently mis-selected the tile whenever
# world_size changed; it has been removed.
#
# Mirrors the gemm2+combine ``GeometryTuningTable`` bucket round-up
# (tuning_configs/flydsl_{arch}_{model}_MegaStage1.json).  Auto-selected only when
# the caller leaves the tile AUTO; an explicit tile always wins.
_MEGA_TUNING_DIR = Path(__file__).resolve().parent / "tuning_configs"
# Megastage1 weights are always w4(fp4); the JSON ``dtype`` is the ACTIVATION quant.
_MEGA_QUANT_TO_DTYPE = {"a4w4": "fp4", "a8w4": "fp8_ocp"}


def _detect_gpu_model_name(device_index=0):
    """GPU model substring (e.g. ``"mi355x"``) for tuning-file selection."""
    try:
        name = torch.cuda.get_device_properties(device_index).name.lower()
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"\bmi\d+\w*", name)
    return m.group(0) if m else None


@functools.lru_cache(maxsize=8)
def _load_mega_tuning_rows(gpu_model):
    """All ``megastage1`` rows from the best-matching MegaStage1 tune JSON
    (``flydsl_*_MegaStage1.json``), preferring a ``gpu_model`` name match.
    Returns a tuple of dicts (hashable cache value); ``()`` on any miss. Call
    ``_load_mega_tuning_rows.cache_clear()`` after editing the JSON to reload.

    The table is world_size-agnostic: rows are keyed by ``experts_per_rank`` so a
    single file serves every ``world_size`` (no ``_ep{n}`` filename split)."""
    if not _MEGA_TUNING_DIR.is_dir():
        return ()
    cands = [p for p in _MEGA_TUNING_DIR.glob("flydsl_*_MegaStage1.json") if p.is_file()]
    if not cands:
        return ()
    cands.sort(key=lambda p: (1 if (gpu_model and gpu_model in p.name) else 0, p.name),
               reverse=True)
    try:
        with open(cands[0], "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return ()
    return tuple(raw.get("megastage1", []))


def _mega_tuned_tile(model_dim, inter_dim, experts, topk, quant, mtpr, world_size, gpu_model):
    """Precise megastage1 config from the FlyDSL tune JSON, keyed by the per-rank
    GEMM1 shape (``experts_per_rank = experts // world_size``); the token bucket
    rounds up to the smallest ``num_tokens >= mtpr`` (largest on overflow),
    mirroring the gemm2+combine ``GeometryTuningTable.lookup``. Returns
    ``{tile_m, tile_n, tile_k, waves_per_eu, use_async_copy}`` or ``None`` on a
    miss (caller then falls back to the generic default tile)."""
    dtype = _MEGA_QUANT_TO_DTYPE.get(quant)
    if dtype is None:
        return None
    if int(world_size) <= 0 or int(experts) % int(world_size) != 0:
        return None
    epr = int(experts) // int(world_size)
    rows = _load_mega_tuning_rows(gpu_model)
    if not rows:
        return None

    def _match(r):
        try:
            return (r.get("dtype") == dtype
                    and int(r["model_dim"]) == int(model_dim)
                    and int(r["inter_dim"]) == int(inter_dim)
                    and int(r["experts_per_rank"]) == epr
                    and int(r["topk"]) == int(topk))
        except (KeyError, ValueError, TypeError):
            return False

    buckets = {int(r["num_tokens"]): r for r in rows if _match(r)}
    if not buckets:
        return None
    if int(mtpr) in buckets:
        chosen = buckets[int(mtpr)]
    else:
        ge = [k for k in buckets if k >= int(mtpr)]
        chosen = buckets[min(ge)] if ge else buckets[max(buckets)]
    return dict(
        tile_m=int(chosen["tile_m"]),
        tile_n=int(chosen["tile_n"]),
        tile_k=int(chosen["tile_k"]),
        waves_per_eu=int(chosen.get("waves_per_eu", 4)),
        use_async_copy=bool(chosen.get("use_async_copy", True)),
    )


class FusedMoEMegaStage1:
    """Single-launch fused dispatch⊕GEMM megakernel (fixedslot decode strict-phase)."""

    def __init__(self, *, rank, world_size, model_dim, inter_dim, experts, topk,
                 quant, w1, w1_scale, max_tok_per_rank, tune_tokens=None,
                 network=None, scheme="fixedslot",
                 unit_size=-1, tile_n=-1, tile_k=256, warp_num_per_block=4,
                 waves_per_eu=4, use_async_copy=True, out_dtype="auto",
                 total_recv_buf=None):
        assert quant in ("a4w4", "a8w4")
        assert scheme == "fixedslot", f"scheme={scheme!r}: only 'fixedslot' supported (handshake removed)"
        assert out_dtype in ("auto", "f16", "fp8", "fp4"), out_dtype
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must divide world_size={world_size}")
        self.scheme = scheme
        # The megakernel ALWAYS emits a2 in the ATOM contract (value@logical token*topk+s via
        # srcmap, scale@sorted row) so stage2 gemm2+combine consume it zero-adapt; the GEMM is
        # always compiled with atom_contract=True (an implementation detail, not user-configurable).
        self.atom_contract = True
        # Plan A: stage-2 combine op's total_recv buffer; when given, the megakernel writes distinct
        # recv-count DIRECTLY into it (disp idx 21) so the unified op needs no host/device copy bridge.
        self._trecv_buf = total_recv_buf
        self.rank = rank
        self.world_size = world_size
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.epr = experts // world_size
        self.topk = topk
        self.quant = quant
        self.network = network
        self.dev = torch.device("cuda", rank)
        self.data_type = torch.float8_e4m3fn if quant == "a8w4" else torch.float4_e2m1fn_x2
        self.a_dtype = "fp8" if quant == "a8w4" else "fp4"
        # 默认输出 dtype = 输入 dtype（a8w4→fp8 / a4w4→fp4），生产即量化 a2 直接接 stage2；
        # 'f16' 仅 debug。显式传入优先。
        if out_dtype == "auto":
            out_dtype = self.a_dtype
        self.scale_dim = model_dim // 32
        # 朴素实现 (naive=fixedslot) auto-detects INTERNALLY (no env): the plain fixed-slot layout
        # reserves cap=npes*mtpr rows PER expert => epr*npes*mtpr rows total, and its A/out buffer
        # resources use a 32-bit num_records/voffset (4GB hardware cap).  When the largest such buffer
        # would reach ~4GB it WRAPS to 0 records => every load returns 0 => corrupt output (root cause
        # of the bs4096 mismatch_rows=194560).  Below ~3GB plain fixed-slot is bit-exact AND FASTER
        # (single pass; ~1.1x vs compact ~0.8x), so we keep it there and switch to the COMPACT dense
        # count-first layout (rows ~topk/cap smaller -> stays well under 4GB, OOM-free) only once the
        # fixed-slot buffer would approach the 4GB wrap.  Buffer-size threshold (robust across shapes):
        _row_b = int(model_dim) if quant == "a8w4" else (int(model_dim) // 2)
        _max_buf = self.epr * world_size * int(max_tok_per_rank) * max(_row_b, int(inter_dim) * 2)
        # Two layouts, switched purely by buffer size (no env): fixedslot below the wrap, compact
        # above it.  Compact dispatch (dense rx) + GEMM1 atom-logical a2 output -> stage2 unchanged.
        self.compact = _max_buf >= 3_000_000_000
        # TMP-COPY overlap experiment: force the compact_allgather path (which already computes
        # global per-expert counts EARLY via cross-PE#1) so the barrier-free per-expert overlap can
        # derive loop bounds without the post-barrier post-pass.  Compact is also the large-bs path,
        # matching where compute is big enough to hide the xGMI floor.
        import os as _os_c
        if _os_c.environ.get("FLYDSL_TMP_FORCE_COMPACT", "0") == "1":
            self.compact = True
        # tile AUTO (default -1): an explicit caller tile_m(unit_size)/tile_n always wins.
        # Otherwise this is the single megastage1 tune entrypoint: read the precise
        # FlyDSL JSON table keyed by shape + token bucket, then fall back to the
        # generic validated default for intentionally untuned shapes.
        tune_tokens = int(max_tok_per_rank if tune_tokens is None else tune_tokens)
        _tuned = None
        if int(unit_size) <= 0:
            _tuned = _mega_tuned_tile(
                model_dim, inter_dim, experts, topk, quant, tune_tokens,
                world_size, _detect_gpu_model_name(rank))  # keyed by experts_per_rank
        if _tuned is not None:
            unit_size = _tuned["tile_m"]
            if int(tile_n) <= 0:
                tile_n = _tuned["tile_n"]
            # the bucket also carries the tuned tile_k / waves / async-copy
            tile_k = _tuned["tile_k"]
            waves_per_eu = _tuned["waves_per_eu"]
            use_async_copy = _tuned["use_async_copy"]
        else:
            _tm_t, _tn_t = _mega_default_tile(int(inter_dim))
            unit_size = int(unit_size) if int(unit_size) > 0 else _tm_t
            tile_n = int(tile_n) if int(tile_n) > 0 else _tn_t
        self.unit_size = int(unit_size)
        # Expert row-padding / sorting granularity (== the GEMM1 M-tile, floored at the 32-row
        # MFMA atom).  This is ALSO the granularity at which stage1 emits sorted_expert_ids
        # (_se_atom): one entry per sort_block_m rows.  The facade threads this into stage2 gemm2
        # (gemm2 reads expert_ids[bx_m // sort_block_m]) and asserts gemm2_tile_m | sort_block_m,
        # so any gemm2 tile_m dividing it consumes the layout correctly.
        self.sort_block_m = max(32, int(self.unit_size))
        self.tile_n = int(tile_n)
        self.mtpr = int(max_tok_per_rank)
        self.tune_tokens = int(tune_tokens)
        # ── XCD swizzle (XCD-aware block swizzle for L2 locality on the MI355X multi-die): validated
        #    L2-reuse win (v4_flash 1.3x, r1_v3 1.7x at gx|8).  The guard below auto-disables it when
        #    cu % gx != 0 (e.g. v4_pro gx=12), which is load-bearing for correctness. ────
        self._xcd = 4
        # ── XCD swizzle CORRECTNESS GUARD ───────────────────────────────────────────────────────
        # The persistent swizzle remaps blocks across 8 XCDs assuming a grid_y of cu_num; it is
        # bit-exact AND fast when gx divides cu (gx in {8,16,32} -> clean 8-way grouping, big L2
        # reuse: v4_flash 1.3x, r1_v3 1.7x).  When cu % gx != 0 (e.g. v4_pro inter=3072,tile_n=256 ->
        # gx=12) the remap mismatches the co-resident grid (gy=cu//gx=21) and CORRUPTS output
        # (verify_keyed: 199 mismatched rows).  v4_pro also gets no L2-reuse benefit (gx=12 doesn't
        # align to 8 XCDs), so we simply DISABLE xcd there -> safe parity.  (Tried snapping gy to make
        # gx*gy%8==0: it makes v4_pro correct but DESTROYS the gx=8 win, so reverted.)
        if self._xcd > 0:
            try:
                from aiter.jit.utils.chip_info import get_cu_num
                _cu_g = get_cu_num()
            except Exception:
                _cu_g = 256
            _gx_g = max(1, (inter_dim * 2 + 2 * tile_n - 1) // tile_n // 2)
            # XCD swizzle is correct ONLY when the WGM group (xcd_swizzle rows x gx N-tiles) fits in a
            # single XCD's block quota (cu/8): 32 % (xcd_swizzle*gx) == 0 -> gx | 8 (gx in {1,2,4,8}).
            # At gx in {12,16,...} the WGM group straddles XCD boundaries and the remap corrupts/loses
            # tiles (docs/moe_stage1_mega.md §5).  cu % gx != 0 (e.g. v4_pro gx=12) is also illegal.
            if _cu_g % _gx_g != 0 or (8 % _gx_g) != 0:
                import warnings
                warnings.warn(
                    f"XCD swizzle DISABLED: gx={_gx_g} is not a divisor of 8 (or cu="
                    f"{_cu_g} % gx != 0) -> XCD WGM group straddles XCD boundaries and would corrupt/"
                    f"lose output tiles; gx={_gx_g} gives no XCD-alignment benefit anyway.")
                self._xcd = 0

        # w1/w1_scale are this rank's `epr` expert rows ONLY (ATOM local convention; the gemm1
        # kernel indexes by the LOCAL expert id).  The global-w1 path is removed -- it truncates
        # at the 4GB buffer num_records cap for >4GB weights.
        self.w1 = w1.contiguous()
        self.w1_scale = w1_scale.contiguous()

        # ---- symmetric buffers + P2P tables via the dispatch op (op.dispatch is NEVER called) ----
        self.op = FlyDSLDispatchGroupMajorOp(
            rank=rank, world_size=world_size, hidden_dim=model_dim,
            max_tok_per_rank=max_tok_per_rank, experts_per_rank=self.epr, topk=topk,
            data_type=self.data_type, unit_size=unit_size, scale_dim=self.scale_dim,
            scale_type_size=1, warp_num_per_block=warp_num_per_block, ll_unified=True,
            fused_scale_swizzle=False, scheme=scheme, compact=self.compact)
        self.nvm = self.op.num_valid_max
        self.max_blocks = self.op.max_blocks
        self.cap = self.op.ll_cap
        # metadata-ready flag (monotonic, NEVER reset -> CUDAGraph-safe): block0 release-stores the
        # launch epoch after the dispatch post-pass; consumer blocks acquire-spin on it instead of a
        # symmetric grid barrier (1 writer / N readers -> no 256-way atomic-RMW storm).
        self._meta = torch.zeros(1, dtype=torch.int32, device=self.dev)

        # TMP-COPY experiment: scheduler per-expert overlap gate.  When enabled, the copied GEMM1
        # is compiled with overlap_gate=True (consumer spins payload_done[le] >= ll_count[le]-1
        # before running the tile's GEMM body) and the dispatch prologue publishes payload_done.
        # The global meta barrier is KEPT (correctness), so single-GPU output stays bit-exact;
        # this validates the scheduler-gate plumbing on copied production code.
        import os as _os
        self._tmp_overlap_gate = _os.environ.get("FLYDSL_TMP_OVERLAP_GATE", "0") == "1"
        self._payload_done = (
            torch.zeros(self.epr, dtype=torch.int32, device=self.dev)
            if self._tmp_overlap_gate else None
        )

        # ---- compile the megakernel (serialize compile across ranks to bound peak memory) ----
        # Always PERSISTENT: the persistent round-robin GEMM covers all occupied tiles regardless
        # of grid size (decode + prefill).
        self.mega = None
        for pe in range(world_size):
            if rank == pe:
                self.mega = compile_fused_moe_gemm1(
                    model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=1,
                    tile_m=unit_size, tile_n=tile_n, tile_k=tile_k, doweight_stage1=False,
                    a_dtype=self.a_dtype, b_dtype="fp4", out_dtype=out_dtype, act="silu",
                    waves_per_eu=waves_per_eu, use_async_copy=use_async_copy,
                    use_cshuffle_epilog=None, contiguous_io=True, dedup_gather=False,
                    atom_contract=self.atom_contract,
                    sparse_tiles=True, persist_m=-1,
                    overlap_gate=self._tmp_overlap_gate,
                    raw_a_scale=True, xcd_swizzle=self._xcd,
                    fuse_dispatch=scheme, fuse_npes=world_size, fuse_topk=topk,
                    fuse_cap=self.cap, fuse_mtpr=self.mtpr,
                    fuse_scale_dim=self.scale_dim, fuse_scale_type_size=1,
                    rank=rank, experts_per_rank=self.epr, compact_dispatch=self.compact,
                    compact_allgather=True)
            if dist.is_initialized():
                dist.barrier()

        # ---- views into op buffers (A-input + raw A-scale + metadata the GEMM consumes) ----
        v = self.op._ll_views()
        self._rx = v["rx_em"]
        self._scale_i32 = v["scale_em_i32"]
        self._trb = self.op.tile_row_base
        self._se = self.op.sorted_expert_ids
        self._nv = self.op.num_valid

        # ---- output / scratch ----
        # out_dtype: "f16" (default, dequantized activation) or "fp8"/"fp4" -> the gemm1 epilogue
        # quantizes silu(gate)*up to MXFP8/MXFP4 + e8m0 scale (a2/a2_scale for stage2), no extra kernel.
        self.out_dtype = out_dtype
        _nf4 = out_dtype == "fp4"
        _nf8 = out_dtype == "fp8"
        self.out_is_quant = _nf4 or _nf8
        # a2 value rows live in the ATOM logical space token*topk+s, t=src_global in
        # [0, npes*mtpr) -> rows = npes*mtpr*topk (compact gemm2 tiles via sorted_token_ids).
        self._a2rows = world_size * self.mtpr * topk
        # Path-2 substep A (FLYDSL_TMP_GROUPA2): a2 written EXPERT-MAJOR -> rows index the sparse/
        # compact slot space (up to num_valid_max), not the logical t*topk+s space.  Size the buffer
        # to num_valid_max so expert-major rows never OOB.  a2 content (valid rows) is unchanged.
        import os as _os_a2
        if _os_a2.environ.get("FLYDSL_TMP_GROUPA2", "0") == "1":
            self._a2rows = max(self._a2rows, int(self.nvm))
        if _nf4:
            self._out = torch.zeros((self._a2rows, inter_dim // 2), dtype=torch.uint8, device=self.dev)
        elif _nf8:
            self._out = torch.zeros((self._a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        else:
            self._out = torch.zeros((self._a2rows, 1, inter_dim), dtype=torch.float16, device=self.dev)
        self._bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        # The GEMM epilogue emits a COMPACT sorted_token_ids[compact_row] = (k<<24)|src_global into a
        # dedicated [nvm] buffer (op.tile_row_base is per-TILE = max_blocks sized, too small for
        # per-ROW); stage2 reads it as its sorted_token_ids input.
        self._sti = torch.zeros(self.nvm, dtype=torch.int32, device=self.dev)
        # stage2 gemm2 tiles at tile_m=32, but stage1 may compute at sort_block_m=64; the kernel
        # emits sorted_expert_ids at 32-row SUB-TILE granularity -> need nvm/32 entries (vs
        # op.sorted_expert_ids = nvm/sort_block_m, too small at tile_m=64).
        self._se_atom = torch.zeros(self.nvm // 32 + 8, dtype=torch.int32, device=self.dev)
        # combine wts_buf (f32, LOGICAL layout wts[t*topk+s], t=src_global via identity tis):
        # the GEMM gathers recv wts_em -> gemm2_combine weights each token natively (no host weight
        # bridge).  zeroed once; only real (t,s) get written.
        self._sw_atom = torch.zeros(world_size * self.mtpr * topk, dtype=torch.float32, device=self.dev)
        # sorted-row routing weights (f32, [nvm], parallel to _sti): the GEMM emits the same recv
        # weight at the compact row so stage2's GEMM2 doweight epilogue (sorted_weights[row])
        # produces the routing-WEIGHTED output that ATOM's production path expects.
        self._wts_sorted = torch.zeros(self.nvm, dtype=torch.float32, device=self.dev)
        # e8m0 sorted-scale buffer (a2_scale): sized as the gemm epilogue's num_records
        # (padded_rows*padded_cols) + a 1-block slack; empty when out_dtype="f16".
        if self.out_is_quant:
            _sbm = max(32, self.unit_size)
            _prows = ((self.max_blocks * _sbm + 255) // 256) * 256
            _pcols = (((inter_dim // 32) + 7) // 8) * 8
            self._osd = torch.zeros(_prows * _pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        else:
            self._osd = torch.empty((0,), dtype=torch.uint8, device=self.dev)
        self.out_scale = self._osd

        # ---- dispatch-arg FIXED-pointer table (op bufs + p2p); built ONCE.  Per-step input
        # pointers are passed as scalar launch args in forward (slots 0-3 stay 0).
        self._build_disp_table()

    def _build_disp_table(self):
        op = self.op
        if self.scheme == "fixedslot":
            tbl = [0, 0, 0, 0,                                   # 0-3 inputs (lazy)
                   op.gb1.data_ptr(), op.running.data_ptr(), op.done2.data_ptr(),
                   op.ll_count.data_ptr(),                       # 4-7
                   op.p2p_rx_em.data_ptr(), op.p2p_scale_em.data_ptr(), op.p2p_idx_em.data_ptr(),
                   op.p2p_wts_em.data_ptr(), op.p2p_srcmap_em.data_ptr(), op.p2p_running.data_ptr(),
                   op.p2p_done2.data_ptr(),                      # 8-14
                   op.sorted_expert_ids.data_ptr(), op.tile_row_base.data_ptr(),
                   op.num_valid.data_ptr(),                      # 15-17
                   self._meta.data_ptr()]                        # 18 metadata-ready flag
            if self.compact:
                # 19-26: compact count-first extras (compact_base, gb_cnt, done2c, meta2, write_cursor)
                tbl += [op.compact_base.data_ptr(), op.p2p_compact_base.data_ptr(),  # 19-20
                        op.gb_cnt.data_ptr(), op.done2c.data_ptr(), op.p2p_done2c.data_ptr(),  # 21-23
                        op.meta2.data_ptr(),                                          # 24 payload-ready flag
                        op.write_cursor.data_ptr(), op.p2p_write_cursor.data_ptr(),   # 25-26
                        op.done2cb.data_ptr(), op.p2p_done2cb.data_ptr(),             # 27-28 cross-PE#1b
                        # all-gather variant (29-35): local_hist, bigcnt(+p2p), cnt_done(+p2p), my_base, local_cursor
                        op.local_hist.data_ptr(), op.bigcnt.data_ptr(), op.p2p_bigcnt.data_ptr(),  # 29-31
                        op.cnt_done.data_ptr(), op.p2p_cnt_done.data_ptr(),           # 32-33
                        op.my_base.data_ptr(), op.local_cursor.data_ptr()]            # 34-35
                # compact+atom: compact_ag leaves idx 19/20 FREE (those are the non-ag compact_base;
                # the combo forces all-gather).  The shared GEMM atom epilogue HARDCODES disp[19]=LOCAL
                # srcmap (k_slot<<24|src_global) and disp[20]=_sw_atom -> override them here.  Plan A
                # (total_recv/dest_ctr/recv_num) goes to NEW idx 36-39 (21/24 are taken by compact_ag's
                # gb_cnt/meta2).
                tbl[19] = op.srcmap_em.data_ptr()                 # 19 LOCAL srcmap (override compact_base)
                tbl[20] = self._sw_atom.data_ptr()               # 20 compact sorted_weights out
                tbl += [(self._trecv_buf if self._trecv_buf is not None else op.total_recv).data_ptr(),
                        op.dest_ctr.data_ptr(),                  # 36 total_recv | 37 dest_ctr(local)
                        op.recv_num.data_ptr(), op.p2p_recv_num.data_ptr(),  # 38 recv_num | 39 p2p_recv_num
                        # 40 _sti out | 41 _se_atom out (SEPARATE from the A-gather _trb/_se args)
                        self._sti.data_ptr(), self._se_atom.data_ptr(),
                        self._wts_sorted.data_ptr(),                     # 42 sorted-row weights out
                        op.p2p_payload_done.data_ptr()]                  # 43 p2p payload_done (overlap gate)
            else:
                # fixedslot+atom.  idx 19: LOCAL srcmap_em (this rank's recv (k_slot<<24)|src_global)
                # -> the GEMM epilogue reads it to write a2 @ logical row.
                tbl += [op.srcmap_em.data_ptr(),                 # 19 LOCAL srcmap (k_slot<<24|src_global)
                        self._sw_atom.data_ptr(),                # 20 compact sorted_weights out
                        # Plan A: native total_recv (distinct recv) so stage2 combine needs no dce.dispatch.
                        # idx 21 points at the EXTERNAL combine-op total_recv when given (zero-copy bridge),
                        # else the facade's own buffer (standalone megav1).
                        (self._trecv_buf if self._trecv_buf is not None else op.total_recv).data_ptr(),
                        op.dest_ctr.data_ptr(),                              # 21 total_recv | 22 dest_ctr(local)
                        op.recv_num.data_ptr(), op.p2p_recv_num.data_ptr(),  # 23 recv_num(local) | 24 p2p_recv_num
                        self._wts_sorted.data_ptr()]                         # 25 sorted-row weights out
                # TMP-COPY scheduler overlap gate: idx 26 = p2p payload_done peer table (sender bumps
                # DEST rank's payload_done[le] cross-PE).  Appended only on the fixedslot+atom path.
                tbl += [op.p2p_payload_done.data_ptr()]              # 26 p2p payload_done
        self._disp = torch.tensor(tbl, dtype=torch.int64, device=self.dev)
        self._disp_host = torch.tensor(tbl, dtype=torch.int64)

    def _agv(self, t):
        return t.view(torch.uint8) if self.a_dtype == "fp4" else t

    def impl_for(self, tokens) -> str:
        return "mega_" + self.scheme

    def reset_counters(self, tokens=None):
        return

    def forward(self, x, wts, scales, topk_ids, stream=None):
        """Runs the WHOLE stage-1 (dispatch + group-GEMM) in ONE launch."""
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        assert cur_tok <= self.mtpr, f"[mega] cur_tok={cur_tok} > max_tok_per_rank={self.mtpr}"
        xc = x.contiguous(); wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous(); sc = scales.contiguous()
        # NOTE: the per-step input pointers are passed as SCALAR launch args (below), NOT written
        # into the device disp table — a per-call H2D copy_ into the table is illegal during
        # CUDAGraph capture (and --from-bf16 re-quantizes => the input ptr changes every call).  The
        # disp table holds ONLY fixed op/p2p pointers, built once in __init__.

        # per-launch resets (in-graph; graph-safe memsets, ordered before the megakernel).
        pd_ptr = 0
        # TMP-COPY experiment: per-phase s_memrealtime timestamps (block0 lane0 writes i64 ticks
        # to phase-ts buffer[k]).  Enabled via FLYDSL_TMP_PHASE_TS=1 (matches the copied GEMM
        # builder's const_expr gate).  Only the fixedslot non-compact path emits all phases.
        import os as _os
        if _os.environ.get("FLYDSL_TMP_PHASE_TS", "0") == "1":
            if getattr(self, "_pts_buf", None) is None:
                self._pts_buf = torch.zeros(16, dtype=torch.int64, device=self.dev)
            pd_ptr = self._pts_buf.data_ptr()
        # TMP-COPY scheduler overlap gate: consumer reads THIS rank's op.payload_done (symmetric
        # buffer local view); senders bump the DEST rank's view via the idx-26 peer table.  Zero
        # per launch (graph-safe memset).
        if self._tmp_overlap_gate:
            self.op.payload_done.zero_()
            pd_ptr = self.op.payload_done.data_ptr()
        # Plan A: total_recv accumulates in-kernel each launch -> zero first (graph-safe).
        # dest_ctr / recv_num self-reset inside the kernel's recv-count signal.  Zero the SAME
        # buffer the kernel writes (external combine-op buffer when bridged, else own).
        (self._trecv_buf if self._trecv_buf is not None else self.op.total_recv).zero_()
        # P-static: the GEMM reads ll_count via addr_expected_real to derive (expert,k) per tile
        # (fixedslot static-tiles only).
        er_ptr = (self.op.ll_count.data_ptr()
                  if (self.scheme == "fixedslot" and not self.compact)
                  else 0)
        # TMP-COPY overlap gate: the consumer reads expected_real=ll_count[le]; compact_ag also
        # populates ll_count (block0 post-CMP), so route er_ptr there when the gate is on.
        if self._tmp_overlap_gate:
            er_ptr = self.op.ll_count.data_ptr()
        if self.compact:
            # compact all-gather: local_hist (count accumulator) + local_cursor (write cursor) must
            # start at 0 each launch (in-graph memset, CUDAGraph-safe; bigcnt is overwritten by the
            # all-gather, the rest are monotonic epochs).
            self.op.local_hist.zero_()
            self.op.local_cursor.zero_()
        # NOTE compact+atom combo: NO per-launch srcmap reset needed -- the GEMM derives the padding
        # mask in-kernel from ll_count (k,count via the prefix scan), exactly like fixslot.  So the
        # stage1->stage2 hookup adds ZERO extra per-forward ops beyond compact dispatch's own resets.
        a_mat = self._agv(self._rx)
        # compact still A-gathers via sparse_tiles (_trb) + expert (_se), so the sorted_token_ids/
        # sorted_expert_ids ARGS MUST stay the compact tile metadata; the atom outputs (_sti/_se_atom)
        # are emitted to SEPARATE buffers via disp idx 40/41.  Fixedslot's static-tiles A-gather leaves
        # these args free -> emits in-place to _sti/_se_atom.
        if self.compact:
            _sorted_arg = self._trb
            _se_arg = self._se
        else:
            _sorted_arg = self._sti
            _se_arg = self._se_atom
        _wt_arg = self.op.wts_em
        self.mega(self._out, a_mat, self.w1, self._scale_i32, self.w1_scale,
                  _sorted_arg, _se_arg, _wt_arg, self._nv, self._bias, self._osd,
                  fx.Int32(self.nvm), fx.Int32(self.inter_dim * 2), fx.Int32(self.model_dim),
                  fx.Int32(self.max_blocks),
                  fx.Int64(pd_ptr), fx.Int64(er_ptr),
                  fx.Int64(self._disp.data_ptr()), fx.Int32(cur_tok),
                  fx.Int64(xc.data_ptr()), fx.Int64(ic.data_ptr()),
                  fx.Int64(wc.data_ptr()), fx.Int64(sc.data_ptr()),
                  stream=stream)
        return dict(out=self._out, srcmap_em=self.op.srcmap_em, num_valid=self._nv,
                    sorted_expert_ids=self._se_atom,
                    sorted_token_ids=self._sti, sorted_weights=self._sw_atom,
                    sorted_weights_row=self._wts_sorted,
                    a2_scale=self._osd, impl="mega_" + self.scheme)
