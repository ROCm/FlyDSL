# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Single-launch fused **dispatch ⊕ GEMM** megakernel for MoE stage-1 (gate/up), CDNA4 / MI355X.

Implements docs/moe_stage1_mega.md — dispatch and the persistent sparse-tile
group-GEMM run in ONE kernel launch.  Two schemes (host-only choice at construction):

  * ``scheme="fixedslot"`` (decode, strict-phase): handshake-free fixed-slot dispatch prologue →
    grid barrier → persistent sparse-tile GEMM (raw activation-scale fold).  Validated bit-exact;
    its ceiling is break-even (no compute to overlap the dispatch latency).

  * ``scheme="handshake"`` (prefill, PRODUCER/CONSUMER OVERLAP): counts-first handshake inline
    (P0 hist → count all-gather → CMP my_base/expected_real/metadata) → PRODUCER blocks (extra
    grid-x columns, ``by>=gx``) write the dense payload and signal ``payload_done[le]`` per token,
    while CONSUMER blocks (``by<gx``) run the persistent GEMM gated per-expert (``overlap_gate``).
    The GEMM's compute hides the dispatch's payload movement — the structure that can net-win.

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


class FusedMoEMegaStage1:
    """Single-launch fused dispatch⊕GEMM megakernel (fixedslot decode | handshake prefill-overlap)."""

    def __init__(self, *, rank, world_size, model_dim, inter_dim, experts, topk,
                 quant, w1, w1_scale, max_tok_per_rank, network=None, scheme="fixedslot",
                 unit_size=64, tile_n=-1, tile_k=256, warp_num_per_block=4,
                 waves_per_eu=4, use_async_copy=True):
        assert quant in ("a4w4", "a8w4")
        assert scheme in ("fixedslot", "handshake")
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must divide world_size={world_size}")
        self.scheme = scheme
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
        self.scale_dim = model_dim // 32
        # prefill (handshake) buckets run the larger MFMA tile
        _prefill_min_mtpr = int(os.environ.get("FUSED_MOE_PREFILL_MIN_MTPR", "4096"))
        self.is_prefill = int(max_tok_per_rank) >= _prefill_min_mtpr or scheme == "handshake"
        # tile_n AUTO (default -1): prefill -> 128; decode -> 256 (validated best: high MFMA intensity,
        # and gx=inter/256 enables XCD swizzle when inter%256==0, e.g. inter=2048/3072 -> gx=8/12).
        # An explicit tile_n from the caller always wins.  tile_m(unit_size): 128 prefill, 64 decode.
        if self.is_prefill:
            unit_size = max(int(unit_size), 128)
            tile_n = 128 if int(tile_n) < 0 else max(int(tile_n), 128)
        else:
            unit_size = int(unit_size) if int(unit_size) > 0 else 64
            if int(tile_n) < 0:
                tile_n = 256 if (int(inter_dim) % 256 == 0) else 128
        self.unit_size = int(unit_size)
        self.tile_n = int(tile_n)
        self.mtpr = int(max_tok_per_rank)
        # ── XCD swizzle (XCD-aware block swizzle for L2 locality on the MI355X multi-die). DEFAULT
        #    ON for decode (fixedslot): validated L2-reuse win (v4_flash 1.3x, r1_v3 1.7x at gx|8);
        #    the guard below auto-disables when cu % gx != 0 (e.g. v4_pro gx=12).  handshake (prefill)
        #    defaults OFF until P4 (xcd on the oversubscribed grid is unverified).  env override. ────
        _xcd_default = 4 if scheme == "fixedslot" else 0
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
            if _cu_g % _gx_g != 0:
                import warnings
                warnings.warn(
                    f"FUSED_MEGA_XCD={self._xcd} DISABLED: cu={_cu_g} % gx={_gx_g} != 0 -> XCD remap "
                    f"would corrupt output (and gx={_gx_g} gives no XCD-alignment benefit anyway).")
                self._xcd = 0

        self.w1 = w1.contiguous()
        self.w1_scale = w1_scale.contiguous()

        # ---- symmetric buffers + P2P tables via the dispatch op (op.dispatch is NEVER called) ----
        self.op = FlyDSLDispatchGroupMajorOp(
            rank=rank, world_size=world_size, hidden_dim=model_dim,
            max_tok_per_rank=max_tok_per_rank, experts_per_rank=self.epr, topk=topk,
            data_type=self.data_type, unit_size=unit_size, scale_dim=self.scale_dim,
            scale_type_size=1, warp_num_per_block=warp_num_per_block, ll_unified=True,
            fused_scale_swizzle=False, scheme=scheme)
        self.nvm = self.op.num_valid_max
        self.max_blocks = self.op.max_blocks
        self.cap = self.op.ll_cap if scheme == "fixedslot" else 0
        # metadata-ready flag (monotonic, NEVER reset -> CUDAGraph-safe): block0 release-stores the
        # launch epoch after the dispatch post-pass; consumer blocks acquire-spin on it instead of a
        # symmetric grid barrier (1 writer / N readers -> no 256-way atomic-RMW storm).
        self._meta = torch.zeros(1, dtype=torch.int32, device=self.dev)

        # ---- handshake overlap: per-expert payload_done counter (symmetric) + P2P table ----
        #   + counting-sort scatter buffers (LOCAL): local_prefix (exclusive prefix-sum of
        #   local_hist over global experts) and inv (dense send-order -> work_idx), so producers
        #   write EXPERT-GROUPED by iterating dense slots (no per-expert scan).
        self._pd = None
        self._p2p_pd = None
        self._inv = None
        self._lprefix = None
        if scheme == "handshake":
            self._pd = mori_shmem_create_tensor((self.epr,), torch.int32)
            self._pd.zero_()
            ms.shmem_barrier_all()
            tbl = torch.zeros(world_size, dtype=torch.int64, device=self.dev)
            for pe in range(world_size):
                tbl[pe] = ms.shmem_ptr_p2p(self._pd.data_ptr(), rank, pe)
            self._p2p_pd = tbl
            self._inv = torch.zeros(self.mtpr * topk, dtype=torch.int32, device=self.dev)
            self._lprefix = torch.zeros(experts, dtype=torch.int32, device=self.dev)

        # ---- compile the megakernel (serialize compile across ranks to bound peak memory) ----
        # Always PERSISTENT: the persistent round-robin GEMM covers all occupied tiles regardless
        # of grid size (decode + prefill).
        self.mega = None
        for pe in range(world_size):
            if rank == pe:
                self.mega = compile_fused_moe_gemm1(
                    model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=1,
                    tile_m=unit_size, tile_n=tile_n, tile_k=tile_k, doweight_stage1=False,
                    a_dtype=self.a_dtype, b_dtype="fp4", out_dtype="f16", act="silu",
                    waves_per_eu=waves_per_eu, use_async_copy=use_async_copy,
                    use_cshuffle_epilog=None, contiguous_io=True, dedup_gather=False,
                    sparse_tiles=True, persist_m=-1,
                    raw_a_scale=True, xcd_swizzle=self._xcd,
                    fuse_dispatch=scheme, fuse_npes=world_size, fuse_topk=topk,
                    fuse_cap=self.cap, fuse_mtpr=self.mtpr,
                    fuse_scale_dim=self.scale_dim, fuse_scale_type_size=1,
                    rank=rank, experts_per_rank=self.epr)
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
        self._out = torch.zeros((self.nvm, 1, inter_dim), dtype=torch.float16, device=self.dev)
        self._wts = torch.ones(self.nvm, dtype=torch.float32, device=self.dev)
        self._bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        self._osd = torch.empty((0,), dtype=torch.uint8, device=self.dev)

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
        else:  # handshake (25 entries; MUST match the prologue's _dph indices 0..24)
            tbl = [0, 0, 0, 0,                                   # 0-3 inputs (lazy)
                   op.gb1.data_ptr(), op.local_hist.data_ptr(), op.off_buf.data_ptr(),
                   op.bigcnt.data_ptr(), op.my_base.data_ptr(), op.cnt_done.data_ptr(),
                   op.ll_count.data_ptr(),                       # 4-10
                   op.sorted_expert_ids.data_ptr(), op.tile_row_base.data_ptr(),
                   op.num_valid.data_ptr(),                      # 11-13
                   op.p2p_rx_em.data_ptr(), op.p2p_scale_em.data_ptr(), op.p2p_idx_em.data_ptr(),
                   op.p2p_wts_em.data_ptr(), op.p2p_srcmap_em.data_ptr(),
                   op.p2p_bigcnt.data_ptr(), op.p2p_cnt_done.data_ptr(),
                   self._p2p_pd.data_ptr(),                      # 14-21
                   self._inv.data_ptr(), self._lprefix.data_ptr(),  # 22-23
                   self._meta.data_ptr()]                        # 24 metadata-ready flag (per-launch reset)
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
        if self.scheme == "handshake":
            # in-graph (graph-safe) per-launch resets: local_hist (P0 accumulates), payload_done
            # (per-expert gate), and meta_flag (block0 sets to 1 => consumers/producers wait >=1;
            # ABSOLUTE threshold makes the oversubscribed grid deadlock-free, see §9.7).
            self.op.local_hist.zero_()
            self._pd.zero_()
            self._meta.zero_()
            pd_ptr = self._pd.data_ptr()
            er_ptr = self.op.ll_count.data_ptr()
        else:
            pd_ptr = 0
            er_ptr = 0
        # DIAG: FUSED_MEGA_TILECOUNT=1 -> repurpose addr_payload_done[0] as a device tile counter.
        if os.environ.get("FUSED_MEGA_TILECOUNT", "0") == "1" and self.scheme != "handshake":
            if not hasattr(self, "_tilecnt"):
                self._tilecnt = torch.zeros(1, dtype=torch.int32, device=self.dev)
            self._tilecnt.zero_()
            pd_ptr = self._tilecnt.data_ptr()

        a_mat = self._agv(self._rx)
        self.mega(self._out, a_mat, self.w1, self._scale_i32, self.w1_scale,
                  self._trb, self._se, self._wts, self._nv, self._bias, self._osd,
                  fx.Int32(self.nvm), fx.Int32(self.inter_dim * 2), fx.Int32(self.model_dim),
                  fx.Int32(self.max_blocks),
                  fx.Int64(pd_ptr), fx.Int64(er_ptr),
                  fx.Int64(self._disp.data_ptr()), fx.Int32(cur_tok),
                  fx.Int64(xc.data_ptr()), fx.Int64(ic.data_ptr()),
                  fx.Int64(wc.data_ptr()), fx.Int64(sc.data_ptr()),
                  stream=stream)
        if os.environ.get("FUSED_MEGA_TILECOUNT", "0") == "1" and self.scheme != "handshake":
            import torch as _t
            _t.cuda.synchronize()
            _nv = int(self._nv[0].item())
            _exp_tiles = (_nv + self.unit_size - 1) // self.unit_size
            _got = int(self._tilecnt[0].item())
            _gx = max(1, (self.inter_dim * 2 + 2 * self.tile_n - 1) // self.tile_n // 2)
            _gy = max(1, self.op._cu // max(1, _gx))
            if self.rank == 0:
                print(f"[TILECOUNT] scheme={self.scheme} cur_tok={cur_tok} num_valid={_nv} "
                      f"expected_tiles={_exp_tiles} gemm_body_execs={_got} "
                      f"gx={_gx} gy={_gy} gx*gy={_gx*_gy} cu={self.op._cu}", flush=True)
        return dict(out=self._out, srcmap_em=self.op.srcmap_em, num_valid=self._nv,
                    sorted_expert_ids=self._se, impl="mega_" + self.scheme)
