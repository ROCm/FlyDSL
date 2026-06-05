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

import os

import torch
import torch.distributed as dist

import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

import flydsl.expr as fx

from .ep_dispatch_groupmajor_op import FlyDSLDispatchGroupMajorOp
from .fused_moe_gemm_2stage import compile_fused_moe_gemm1


def _is_fp4(dt):
    return dt == torch.float4_e2m1fn_x2


# ── Mega decode GEMM tile, TUNED at the interface (host side) ────────────────────────────────────
# Keyed by the GEMM shape that's already in the ctor args -- (model_dim, inter_dim, quant) -- NOT a
# network name.  Value: list of (mtpr_le, tile_m, tile_n); the decode persistent GEMM tile is the
# smallest bucket whose mtpr_le >= the runtime max_tok_per_rank (mtpr = max(16, bs)).  Built by
# sweeping bs1..512 x {tile_m}x{tile_n} (tests/kernels/bench ... --mega-tile-m/--mega-tile-n) and
# keeping the lowest-latency tile per bucket.  An explicit caller tile always overrides this table.
# Fallback (no entry): tile_m=64, tile_n=256 if inter%256==0 else 128.
_MEGA_DECODE_TILE = {
    # RE-SWEPT 2026-06-14 on the FIXED kernel (profiler GPU device-time, 8xMI355X, --from-bf16,
    # tile_n=256 + XCD; baseline = aiter-tune / fp8-dispatch).  (The pre-2026-06-14 table was swept on
    # the BROKEN kernel whose XCD remap skipped ~6/7 of the tiles -> invalid; see notes §1.)
    # Lever: at small mtpr, tile_m=32 cuts the pad-row MFMA waste (decode packs only ~10-20 real tokens
    # into a tile_m-row tile).  High-bs buckets were refreshed on 2026-06-18 over bs256..32768 x
    # {32,64}x{128,256} using stage1-sweep profiler time; keep explicit buckets instead of relying on
    # the "reuse largest bucket" fallback so compact/fixedslot both pick the measured winner.
    # Value: list of (mtpr_le, tile_m, tile_n); pick the smallest bucket whose mtpr_le >= runtime mtpr.
    # r1_v3: 64x128 wins bs256..8192; 64x256 narrowly wins bs16384+.
    (7168, 2048, "a4w4"): [
        (64, 32, 256),
        (128, 64, 256),
        (8192, 64, 128),
        (32768, 64, 256),
    ],
    # v4_flash: 32x128 wins bs256; 64x256 wins bs512+ (XCD-enabled gx=8).
    (4096, 2048, "a8w4"): [
        (64, 32, 256),
        (128, 64, 256),
        (256, 32, 128),
        (32768, 64, 256),
    ],
    # v4_pro: 64x128 wins fixedslot bs256..1024; 64x256 wins compact/high-bs.
    # gx=12 keeps XCD disabled by correctness guard.
    (7168, 3072, "a8w4"): [
        (128, 32, 256),
        (1024, 64, 128),
        (32768, 64, 256),
    ],
}


def _mega_decode_tile(model_dim, inter_dim, quant, mtpr):
    """Return (tile_m, tile_n) for the decode persistent GEMM from the host-side tuned table, keyed by
    the GEMM shape (model_dim, inter_dim, quant); fall back to the validated default (64, 256|128)
    when no tuned bucket matches the shape."""
    _tn_fallback = 256 if (int(inter_dim) % 256 == 0) else 128
    buckets = _MEGA_DECODE_TILE.get((int(model_dim), int(inter_dim), quant))
    if buckets:
        for mtpr_le, tm, tn in sorted(buckets):
            if int(mtpr) <= int(mtpr_le):
                return int(tm), int(tn)
        _, tm, tn = sorted(buckets)[-1]   # mtpr above the top bucket -> use the largest bucket's tile
        return int(tm), int(tn)
    return 64, _tn_fallback


class FusedMoEMegaStage1:
    """Single-launch fused dispatch⊕GEMM megakernel (fixedslot decode strict-phase)."""

    def __init__(self, *, rank, world_size, model_dim, inter_dim, experts, topk,
                 quant, w1, w1_scale, max_tok_per_rank, network=None, scheme="fixedslot",
                 unit_size=-1, tile_n=-1, tile_k=256, warp_num_per_block=4,
                 waves_per_eu=4, use_async_copy=True, out_dtype="auto",
                 atom_contract=False, total_recv_buf=None):
        assert quant in ("a4w4", "a8w4")
        assert scheme == "fixedslot", f"scheme={scheme!r}: only 'fixedslot' supported (handshake removed)"
        assert out_dtype in ("auto", "f16", "fp8", "fp4"), out_dtype
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must divide world_size={world_size}")
        self.scheme = scheme
        # atom_contract: emit a2 in the ATOM contract (value@logical token*topk+s via srcmap,
        # scale@sorted) so stage2 gemm2+combine read it zero-adapt.  fixedslot only (A direct-read).
        self.atom_contract = bool(atom_contract)
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
        self.compact = (_max_buf >= 3_000_000_000
                        or os.environ.get("FUSED_MEGA_FORCE_COMPACT", "0") == "1")
        # ── compact+atom combo (large-bs e2e): compact DISPATCH (dense rx, dodges the 4GB input
        # wrap) + GEMM1 writes the ATOM-LOGICAL a2 (stage2 unchanged).  GUARDED by env (DEFAULT OFF
        # -> existing paths byte-identical) because the GEMM1 atom-output / compact-A-gather
        # decoupling is still WIP (docs/moe_stage1_mega_notes.md 2026-06-16 续2).  When OFF, the
        # historical assert stands (atom_contract requires non-compact).
        self._compact_atom = bool(self.atom_contract and self.compact
                                  and os.environ.get("FUSED_MEGA_COMPACT_ATOM", "0") == "1")
        if not self._compact_atom:
            assert not (self.atom_contract and self.compact), \
                ("atom_contract uses the non-compact fixed-slot layout (disp idx 19 = srcmap); "
                 "set FUSED_MEGA_COMPACT_ATOM=1 to enable the compact+atom combo (GEMM WIP)")
        # tile AUTO (default -1): an explicit caller tile_m(unit_size)/tile_n always wins; otherwise the
        # host-side tuned table below is consulted by GEMM shape (model_dim, inter_dim, quant), all of
        # which are ctor args -- no network name needed.  Safe fallback when the shape isn't tuned.
        #   * prefill (handshake/进阶) -> 128x128 (large MFMA tile for the oversubscribed grid).
        #   * decode (fixedslot/朴素)  -> _mega_decode_tile(): tuned (tile_m, tile_n) keyed by shape+quant+mtpr;
        #     fallback tile_m=64, tile_n=256 when inter%256==0 (gx=inter/256 enables XCD swizzle), else 128.
        # NOTE: the NAIVE scheme (fixedslot, incl. compact) ALWAYS uses decode tiles across ALL bs --
        # it is the decode/strict-phase implementation.  The prefill 128x128 tile (gx=16) + the naive
        # grid barrier do NOT co-reside -> deadlock at large bs; decode tiles (gx=8) stay co-resident.
        _tm_t, _tn_t = _mega_decode_tile(int(model_dim), int(inter_dim), quant, int(max_tok_per_rank))
        unit_size = int(unit_size) if int(unit_size) > 0 else _tm_t
        tile_n = int(tile_n) if int(tile_n) > 0 else _tn_t
        self.unit_size = int(unit_size)
        self.tile_n = int(tile_n)
        self.mtpr = int(max_tok_per_rank)
        # ── XCD swizzle (XCD-aware block swizzle for L2 locality on the MI355X multi-die). DEFAULT
        #    ON for decode (fixedslot): validated L2-reuse win (v4_flash 1.3x, r1_v3 1.7x at gx|8);
        #    the guard below auto-disables when cu % gx != 0 (e.g. v4_pro gx=12).  handshake (prefill)
        #    defaults OFF until P4 (xcd on the oversubscribed grid is unverified).  env override. ────
        _xcd_default = 4
        self._xcd = max(0, int(os.environ.get("FUSED_MEGA_XCD", str(_xcd_default))))
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
                    f"FUSED_MEGA_XCD={self._xcd} DISABLED: gx={_gx_g} is not a divisor of 8 (or cu="
                    f"{_cu_g} % gx != 0) -> XCD WGM group straddles XCD boundaries and would corrupt/"
                    f"lose output tiles; gx={_gx_g} gives no XCD-alignment benefit anyway.")
                self._xcd = 0

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
                    raw_a_scale=True, xcd_swizzle=self._xcd,
                    fuse_dispatch=scheme, fuse_npes=world_size, fuse_topk=topk,
                    fuse_cap=self.cap, fuse_mtpr=self.mtpr,
                    fuse_scale_dim=self.scale_dim, fuse_scale_type_size=1,
                    rank=rank, experts_per_rank=self.epr, compact_dispatch=self.compact,
                    compact_allgather=True)
            if dist.is_initialized():
                dist.barrier()

        # Optional VGPR cap: the FUSED kernel carries the PERSISTENT GEMM's VGPR (=156 on gfx950),
        # which lowers occupancy vs the non-persistent GEMM (=104).  rocprofv3 shows the decode loss
        # is almost entirely this GEMM-phase occupancy penalty.  --amdgpu-num-vgpr=N (maxnreg) forces
        # the allocator to fit N VGPR (spilling if needed) -> higher occupancy; sweep to find the
        # spill-vs-occupancy sweet spot.  Env FUSED_MEGA_MAXNREG (0 = compiler default).
        _maxnreg = int(os.environ.get("FUSED_MEGA_MAXNREG", "0"))
        if _maxnreg > 0 and self.mega is not None:
            self.mega.compile_hints["maxnreg"] = _maxnreg

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
        # atom_contract: a2 value rows are the ATOM logical space token*topk+s, t=src_global in
        # [0, npes*mtpr) -> rows = npes*mtpr*topk (compact gemm2 tiles via sorted_token_ids).
        self._a2rows = (world_size * self.mtpr * topk) if self.atom_contract else self.nvm
        if _nf4:
            self._out = torch.zeros((self._a2rows, inter_dim // 2), dtype=torch.uint8, device=self.dev)
        elif _nf8:
            self._out = torch.zeros((self._a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        else:
            self._out = torch.zeros((self._a2rows, 1, inter_dim), dtype=torch.float16, device=self.dev)
        self._wts = torch.ones(self.nvm, dtype=torch.float32, device=self.dev)
        self._bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        # atom_contract: the GEMM epilogue emits a COMPACT sorted_token_ids[compact_row] =
        # (k<<24)|src_global into a dedicated [nvm] buffer (op.tile_row_base is per-TILE = max_blocks
        # sized, too small for per-ROW).  static-tiles never reads this arg, so we pass it purely as
        # the GEMM's sorted_token_ids output + stage2's sorted_token_ids input.
        if self.atom_contract:
            self._sti = torch.zeros(self.nvm, dtype=torch.int32, device=self.dev)
            # stage2 gemm2 tiles at tile_m=32, but stage1 may compute at sort_block_m=64; the kernel
            # emits sorted_expert_ids at 32-row SUB-TILE granularity -> need nvm/32 entries (vs
            # op.sorted_expert_ids = nvm/sort_block_m, too small at tile_m=64).
            self._se_atom = torch.zeros(self.nvm // 32 + 8, dtype=torch.int32, device=self.dev)
            # combine wts_buf (f32, LOGICAL layout wts[t*topk+s], t=src_global via identity tis):
            # the GEMM gathers recv wts_em -> gemm2_combine weights each token natively (atom
            # contract, no host weight bridge).  zeroed once; only real (t,s) get written.
            self._sw_atom = torch.zeros(world_size * self.mtpr * topk, dtype=torch.float32, device=self.dev)
        else:
            self._sti = None
            self._se_atom = None
            self._sw_atom = None
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
                if self._compact_atom:
                    # compact+atom combo: compact_ag leaves idx 19/20 FREE (those are the non-ag
                    # compact_base; combo forces all-gather).  The shared GEMM atom epilogue HARDCODES
                    # disp[19]=LOCAL srcmap (k_slot<<24|src_global) and disp[20]=_sw_atom -> override
                    # them here.  Plan A (total_recv/dest_ctr/recv_num) goes to NEW idx 36-39 (21/24 are
                    # taken by compact_ag's gb_cnt/meta2).  The compact_ag prologue reads 36-39 iff
                    # _atom_contract; the GEMM1 atom-output decoupling is still WIP (guarded OFF default).
                    tbl[19] = op.srcmap_em.data_ptr()                 # 19 LOCAL srcmap (override compact_base)
                    tbl[20] = self._sw_atom.data_ptr()               # 20 compact sorted_weights out
                    tbl += [(self._trecv_buf if self._trecv_buf is not None else op.total_recv).data_ptr(),
                            op.dest_ctr.data_ptr(),                  # 36 total_recv | 37 dest_ctr(local)
                            op.recv_num.data_ptr(), op.p2p_recv_num.data_ptr(),  # 38 recv_num | 39 p2p_recv_num
                            # 40 _sti out | 41 _se_atom out (SEPARATE from the A-gather _trb/_se args)
                            self._sti.data_ptr(), self._se_atom.data_ptr()]
            elif self.atom_contract:
                # 19: LOCAL srcmap_em (this rank's recv (k_slot<<24)|src_global) -> the GEMM
                # epilogue reads it to write a2 @ logical row.  Only on the non-compact path
                # (atom_contract asserts non-compact); the kernel reads disp[19] iff _atom_contract.
                tbl += [op.srcmap_em.data_ptr(),                 # 19 LOCAL srcmap (k_slot<<24|src_global)
                        self._sw_atom.data_ptr(),                # 20 compact sorted_weights out
                        # Plan A: native total_recv (distinct recv) so stage2 combine needs no dce.dispatch.
                        # idx 21 points at the EXTERNAL combine-op total_recv when given (zero-copy bridge),
                        # else the facade's own buffer (standalone megav1).
                        (self._trecv_buf if self._trecv_buf is not None else op.total_recv).data_ptr(),
                        op.dest_ctr.data_ptr(),                              # 21 total_recv | 22 dest_ctr(local)
                        op.recv_num.data_ptr(), op.p2p_recv_num.data_ptr()]  # 23 recv_num(local) | 24 p2p_recv_num
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
        if os.environ.get("FUSED_MEGA_PHASE_TS", "0") == "1":
            # DIAG: in-kernel phase timestamps -> addr_payload_done (i64[16]); host reads + diffs.
            # Enabled for fixedslot AND compact (the write-imbalance probe lives in compact_ag).
            if not hasattr(self, "_phase_ts"):
                self._phase_ts = torch.zeros(16, dtype=torch.int64, device=self.dev)
            self._phase_ts.zero_()
            pd_ptr = self._phase_ts.data_ptr()
        if self.atom_contract:
            # Plan A: total_recv accumulates in-kernel each launch -> zero first (graph-safe).
            # dest_ctr / recv_num self-reset inside the kernel's recv-count signal.  Zero the SAME
            # buffer the kernel writes (external combine-op buffer when bridged, else own).
            (self._trecv_buf if self._trecv_buf is not None else self.op.total_recv).zero_()
        # P-static: the GEMM reads ll_count via addr_expected_real to derive (expert,k) per tile.
        er_ptr = (self.op.ll_count.data_ptr()
                  if (self.scheme == "fixedslot" and not self.compact
                      and os.environ.get("FUSED_MEGA_STATIC_TILES", "1") != "0")
                  else 0)
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
        # compact+atom combo: compact still A-gathers via sparse_tiles (_trb) + expert (_se), so the
        # sorted_token_ids/sorted_expert_ids ARGS MUST stay the compact tile metadata; the atom
        # outputs (_sti/_se_atom) are emitted to SEPARATE buffers via disp idx 40/41.  Non-compact
        # atom (static-tiles A-gather) leaves these args free -> emits in-place to _sti/_se_atom.
        if self._compact_atom:
            _sorted_arg = self._trb
            _se_arg = self._se
            _wt_arg = self.op.wts_em
        elif self.atom_contract:
            _sorted_arg = self._sti
            _se_arg = self._se_atom
            _wt_arg = self.op.wts_em
        else:
            _sorted_arg = self._trb
            _se_arg = self._se
            _wt_arg = self._wts
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
                    sorted_expert_ids=(self._se_atom if self.atom_contract else self._se),
                    sorted_token_ids=self._sti, sorted_weights=self._sw_atom,
                    a2_scale=self._osd, impl="mega_" + self.scheme)
