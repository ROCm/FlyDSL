# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Symmetric-buffer holder for the MoE stage-1 megakernel's expert-major dispatch.

This op owns the mori-shmem / P2P buffers + sort-metadata tensors that the megakernel reads;
it has NO ``dispatch()`` entry point — the dispatch logic is INLINED in the megakernel GEMM
prologue (``compile_fused_moe_gemm1(fuse_dispatch=...)``).  Layout (cap = npes*mtpr fixed slots,
or compact dense for handshake): expert-major token payload written to the destination expert's
slot region; a tiny in-kernel post-pass emits the occupied tiles compactly.

Owns, per receiver rank:
  rx_em      [num_valid_max, row_bytes]   expert-major token payload (fixed slots)
  scale_em   [num_valid_max, scale_bytes] expert-major raw e8m0 act scale (GEMM folds swizzle)
  idx_em     [num_valid_max]   per-row GLOBAL expert id
  wts_em     [num_valid_max]   per-row routing weight (f32)
  srcmap_em  [num_valid_max]   (k_slot<<24)|(src_pe*max_tok+src_tok)  for combine/restore
  sorted_expert_ids [max_blocks]  per OCCUPIED tile: global expert id (compact)
  tile_row_base     [max_blocks]  per OCCUPIED tile: sparse row base (le*cap + t*tile_m)
  num_valid  [2]    #occupied_tiles * tile_m  (dynamic; bounds the sparse GEMM tile loop)
  ll_count   [epr]  per-expert recv count (downstream/overflow check)

See ep_dispatch_groupmajor_kernel.py and docs/moe_stage1_mega.md.
"""
from __future__ import annotations

import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor


def _is_fp4(dt):
    return dt == torch.float4_e2m1fn_x2


def _row_bytes(dt, hidden):
    return hidden // 2 if _is_fp4(dt) else hidden * torch.tensor([], dtype=dt).element_size()


def _row_view(dt, hidden):
    return hidden // 2 if _is_fp4(dt) else hidden


class FlyDSLDispatchGroupMajorOp:
    """Unified handshake-free fixed-slot (cap=mtpr) expert-major dispatch op."""

    def __init__(self, *, rank, world_size, hidden_dim, max_tok_per_rank, experts_per_rank,
                 topk, data_type, unit_size, scale_dim, scale_type_size=1,
                 warp_num_per_block=4, block_num=None, num_valid_max=None,
                 dedup_payload=False, low_latency=True, ll_unified=True,
                 fused_scale_swizzle=False, scheme="fixedslot", compact=False):
        assert world_size <= 8
        # scheme: "fixedslot" = per-token remote-atomic packing (current); "handshake" =
        # all-gather counts -> dense expert-major direct write (atomic-free).  Same outputs.
        assert scheme in ("fixedslot", "handshake")
        self.scheme = scheme
        # compact (fixedslot/naive only): COUNT-first two-pass -> COMPACT expert-major layout (no
        # per-expert cap reservation).  Phase-0 counts (remote atomic), block0 broadcasts a dense
        # prefix-sum base (compact_base[le]), Phase-2 writes payload precisely.  Cuts num_valid_max
        # from epr*cap (=epr*npes*mtpr, OOMs at large bs) to ~npes*mtpr*topk -> scales to full bs.
        self.compact = bool(compact and scheme == "fixedslot")
        self.rank = rank
        self.npes = world_size
        self.hidden = hidden_dim
        self.mtpr = max_tok_per_rank
        self.epr = experts_per_rank
        self.topk = topk
        self.dtype = data_type
        self.unit = unit_size
        # cap = per-expert fixed-slot capacity, rounded up to a tile_m multiple.  Each expert
        # reserves `cap` rows; the post-pass emits only the occupied tiles, so the sparse GEMM
        # tracks real load (cap only costs reserved address space, NOT GEMM compute).
        # cap = npes*mtpr = the all-to-one worst case (a token hits an expert at most once, so a
        # local expert receives at most npes*mtpr tokens) => PROVABLY overflow-free; no drops.
        self.max_tokens_per_expert = world_size * max_tok_per_rank
        self.ll_cap = ((self.max_tokens_per_expert + unit_size - 1) // unit_size) * unit_size
        self.low_latency = True
        self.ll_unified = True
        self.scale_dim = scale_dim
        self.scale_type_size = scale_type_size
        self.scale_bytes = scale_dim * scale_type_size
        self.scale_n_i32 = (self.scale_bytes + 3) // 4 if self.scale_bytes > 0 else 0
        self.fused_scale_swizzle = bool(fused_scale_swizzle and self.scale_bytes > 0)
        self.warps = warp_num_per_block
        self._dev = torch.device("cuda", rank)
        self.row_bytes = _row_bytes(data_type, hidden_dim)
        self.row_view = _row_view(data_type, hidden_dim)

        try:
            cu = int(torch.cuda.get_device_properties(self._dev).multi_processor_count)
        except Exception:
            cu = 128
        self._cu = cu

        # Payload extent / GEMM grid bound:
        #  * fixedslot SPARSE le*cap: needs epr*cap rows (one cap-region per local expert).
        #  * handshake COMPACT dense: needs only total-recv worst case + per-expert tile_m padding,
        #    NO per-expert cap reservation -> far smaller buffers AND a tight max_blocks so the
        #    non-persistent GEMM does not over-launch ~empty CTAs (the cap-based bound launched
        #    epr*cap/tile_m CTAs vs the ~num_valid actually occupied).
        if num_valid_max is None:
            if self.scheme == "handshake" or self.compact:
                # tr <= npes*mtpr*topk (each of npes ranks' mtpr tokens hits <= topk of my experts);
                # + epr*unit for the per-expert tile_m padding of the compact layout.
                num_valid_max = world_size * max_tok_per_rank * topk + experts_per_rank * unit_size
            else:
                num_valid_max = experts_per_rank * self.ll_cap + 256
        self.num_valid_max = int(num_valid_max)
        self.max_blocks = (self.num_valid_max + unit_size - 1) // unit_size
        # block_num: full parallelism at prefill; scaled down at small bs where the grid-barrier
        # + large launch is pure fixed overhead.
        if block_num is not None:
            self.block_num = int(block_num)
        else:
            self.block_num = min(self._cu, 128, max(8, (max_tok_per_rank * topk) // 4))

        self._alloc()
        ms.shmem_barrier_all()
        self._build_p2p()
        # NOTE: the standalone dispatch KERNEL is no longer compiled/launched here — the dispatch
        # logic is INLINED in the megakernel's GEMM prologue (compile_fused_moe_gemm1(fuse_dispatch=)).
        # This op now only owns the symmetric (mori-shmem / P2P) buffers + sort-metadata tensors that
        # the megakernel reads; there is NO `op.dispatch()` entry point anymore.

    def _sym(self, shape, dtype):
        t = mori_shmem_create_tensor(shape, dtype)
        t.zero_()
        return t

    def _alloc(self):
        npes, epr = self.npes, self.epr
        nvm = self.num_valid_max
        # symmetric (P2P) buffers
        self.done2 = self._sym((npes,), torch.int32)
        # persistent per-expert recv count: peers atomic-add as their tokens land; the post-pass
        # copies running->ll_count then folds running back to 0 (no separate reset launch).
        self.running = self._sym((epr,), torch.int32)
        self.ll_count = self._sym((epr,), torch.int32)
        self.rx_em = self._sym((nvm * self.row_bytes,), torch.int8)
        self.scale_em = self._sym((max(1, nvm * self.scale_n_i32),), torch.int32)
        self.idx_em = self._sym((nvm,), torch.int32)
        self.wts_em = self._sym((nvm,), torch.float32)
        self.srcmap_em = self._sym((nvm,), torch.int32)
        # local-only buffers
        self.gb1 = torch.zeros(1, dtype=torch.int64, device=self._dev)
        self.sorted_expert_ids = torch.zeros(self.max_blocks, dtype=torch.int32, device=self._dev)
        # per-occupied-tile sparse row base (le*cap + t*tile_m); GEMM reads bx_m from here.
        self.tile_row_base = torch.zeros(self.max_blocks, dtype=torch.int32, device=self._dev)
        self.num_valid = torch.zeros(2, dtype=torch.int32, device=self._dev)
        # ---- Plan A: native total_recv (distinct recv count, == standard-dispatch dce.total_recv) ----
        # fixedslot writes per-(token,expert) (running[le] counts copies), so distinct recv is computed
        # by a per-token distinct-dest-PE dedup pass that bumps dest_ctr[dpe], then a cross-PE recv-count
        # signal (recv_num, symmetric) accumulates total_recv -- mirrors the sorted path / standard
        # dispatch.  Lets stage2 combine read total_recv natively (no redundant dce.dispatch).
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=self._dev)   # local; kernel accumulates
        self.dest_ctr = torch.zeros(self.npes, dtype=torch.int32, device=self._dev)  # local send-count/dest
        self.recv_num = self._sym((self.npes,), torch.int32)                    # symmetric: peers signal here
        # TMP-COPY scheduler overlap gate: per-local-expert payload-landed counter, symmetric so a
        # sender bumps the DEST rank's payload_done[le] (cross-PE) after writing the token; the
        # receiver's GEMM gate spins payload_done[le] >= ll_count[le].  Zeroed per launch.
        self.payload_done = self._sym((self.epr,), torch.int32)
        # ---- compact (naive count-first) buffers ----
        # compact_base[le] = dense prefix-sum base (tile_m-padded) for local expert le; symmetric so
        # senders read the DEST's base to place payload precisely.  done2c = the COUNT cross-PE
        # done-barrier (separate epoch buffer from done2 which gates the WRITE round).  gb_cnt =
        # the count-pass grid-arrival counter (separate from gb1 which gates the write-pass).
        self.compact_base = None
        self.done2c = None
        self.gb_cnt = None
        self.meta2 = None
        self.write_cursor = None
        if self.compact:
            self.compact_base = self._sym((epr,), torch.int32)
            self.done2c = self._sym((npes,), torch.int32)
            self.gb_cnt = torch.zeros(1, dtype=torch.int64, device=self._dev)
            self.meta2 = torch.zeros(1, dtype=torch.int32, device=self._dev)   # payload-ready flag
            # phase-2 write cursor (SEPARATE from running, which is the count accumulator) so neither
            # needs a mid-kernel reset (avoids the cross-rank reset race): both start at 0 and are
            # reset to 0 only at kernel END (after cross-PE#2), mirroring the non-compact post-pass.
            self.write_cursor = self._sym((epr,), torch.int32)
            # cross-PE barrier #1b: AFTER each rank's block0 computes compact_base[], BEFORE any sender
            # reads a DEST's compact_base in phase-2.  Without it a sender can read a dest's stale/zero
            # compact_base (dest's block0 not done yet) -> wrong slots -> corrupt output on some ranks.
            self.done2cb = self._sym((npes,), torch.int32)
            # ---- ALL-GATHER compact (2 cross-PE rounds instead of 3): local count -> bigcnt all-gather
            # -> each rank computes my_base[ge] LOCALLY (where its tokens land in each dest) -> strict
            # write with a LOCAL cursor.  Avoids remote count atomics + remote compact_base read.
            _te = npes * epr
            self.local_hist = torch.zeros(_te, dtype=torch.int32, device=self._dev)     # per-launch reset
            self.bigcnt = self._sym((npes * _te,), torch.int32)                          # [src][ge] all-gather
            self.cnt_done = self._sym((npes,), torch.int32)                              # all-gather epoch barrier
            self.my_base = torch.zeros(_te, dtype=torch.int32, device=self._dev)         # my tokens' base in each dest
            self.local_cursor = torch.zeros(_te, dtype=torch.int32, device=self._dev)    # per-launch reset (write cursor)
        self.swizzled_scale = None
        if self.fused_scale_swizzle:
            if self.scale_dim % 8 != 0:
                raise ValueError(f"scale_dim={self.scale_dim} must be a multiple of 8")
            m_tiles = (self.num_valid_max + 31) // 32
            n_tiles = self.scale_dim // 8
            self.swizzled_scale = torch.zeros((m_tiles, n_tiles, 4, 16), dtype=torch.int32, device=self._dev)
        # ---- handshake-scheme buffers (all-gather counts -> dense expert-major) ----
        if self.scheme == "handshake":
            te = self.npes * self.epr
            self.local_hist = torch.zeros(te, dtype=torch.int32, device=self._dev)   # in-graph reset
            self.off_buf = torch.zeros(self.mtpr * self.topk, dtype=torch.int32, device=self._dev)
            self.my_base = torch.zeros(te, dtype=torch.int32, device=self._dev)
            self.bigcnt = self._sym((self.npes * te,), torch.int32)                  # [src][global_expert]
            self.cnt_done = self._sym((self.npes,), torch.int32)                     # count-ready epoch signal
            # W0 parallel counts-first: 2nd grid-arrival counter (post-SCT barrier). Monotonic, NEVER
            # reset (epoch = gb2/nblk), CUDAGraph-safe — mirrors gb1.
            self.gb2 = torch.zeros(1, dtype=torch.int64, device=self._dev)

    def _p2p_table(self, t):
        tbl = torch.zeros(self.npes, dtype=torch.int64, device=self._dev)
        for pe in range(self.npes):
            tbl[pe] = ms.shmem_ptr_p2p(t.data_ptr(), self.rank, pe)
        return tbl

    def _build_p2p(self):
        self.p2p_done2 = self._p2p_table(self.done2)
        self.p2p_running = self._p2p_table(self.running)
        self.p2p_rx_em = self._p2p_table(self.rx_em)
        self.p2p_scale_em = self._p2p_table(self.scale_em)
        self.p2p_idx_em = self._p2p_table(self.idx_em)
        self.p2p_wts_em = self._p2p_table(self.wts_em)
        self.p2p_srcmap_em = self._p2p_table(self.srcmap_em)
        self.p2p_recv_num = self._p2p_table(self.recv_num)   # Plan A: cross-PE recv-count signal target
        self.p2p_payload_done = self._p2p_table(self.payload_done)  # TMP-COPY scheduler overlap gate
        if self.compact:
            self.p2p_compact_base = self._p2p_table(self.compact_base)
            self.p2p_done2c = self._p2p_table(self.done2c)
            self.p2p_write_cursor = self._p2p_table(self.write_cursor)
            self.p2p_done2cb = self._p2p_table(self.done2cb)
            self.p2p_bigcnt = self._p2p_table(self.bigcnt)
            self.p2p_cnt_done = self._p2p_table(self.cnt_done)
        if self.scheme == "handshake":
            self.p2p_bigcnt = self._p2p_table(self.bigcnt)
            self.p2p_cnt_done = self._p2p_table(self.cnt_done)

    def reset_counters(self):
        """No-op: the per-expert count reset is folded into the kernel post-pass."""
        return

    @property
    def ll_cap_used(self):
        """Per-expert slot capacity (cap = npes*mtpr, tile_m-aligned; provably overflow-free)."""
        return self.ll_cap

    def _ll_views(self):
        rx_em_view = self.rx_em.view(self.dtype).view(self.num_valid_max, self.row_view) \
            if not _is_fp4(self.dtype) else self.rx_em.view(torch.float4_e2m1fn_x2).view(self.num_valid_max, self.row_view)
        scale_em_view = self.scale_em.view(torch.uint8).view(self.num_valid_max, max(1, self.scale_n_i32 * 4))[:, :self.scale_bytes]
        scale_em_i32 = self.scale_em.view(self.num_valid_max, max(1, self.scale_n_i32))
        return dict(rx_em=rx_em_view, scale_em=scale_em_view, scale_em_i32=scale_em_i32,
                    idx_em=self.idx_em, wts_em=self.wts_em, srcmap_em=self.srcmap_em,
                    sorted_expert_ids=self.sorted_expert_ids, tile_row_base=self.tile_row_base,
                    num_valid=self.num_valid, dedup_rx=None,
                    swizzled_scale=(None if self.swizzled_scale is None else
                                    self.swizzled_scale
                                    .view(getattr(torch, "float8_e8m0fnu", torch.uint8))
                                    .view(-1, self.scale_dim)
                                    .view(torch.uint8)))

    # NOTE: no `dispatch()` entry point — the dispatch kernel is INLINED in the megakernel GEMM
    # prologue.  The megakernel reads this op's symmetric buffers + metadata directly (see
    # FusedMoEMegaStage1 / compile_fused_moe_gemm1(fuse_dispatch=...)).
