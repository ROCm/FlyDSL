# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused stage1 with low-ID dispatch producers and oversubscribed FP8xFP4 grouped-GEMM1 consumers."""

import fcntl
import functools
import json
import os

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Autotuner, Config, do_bench
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec
from kernels.common.tensor_shim import _run_compiled
from kernels.gemm.fp8_gemm_utils import ceildiv
from kernels.mega_moe.mega_moe_exp.group_gemm.gemm1_util import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    wait_lds_barrier,
)

from .dispatch import emit_dispatch_prologue, wait_dispatch_ready
from .group_gemm.tune_config import get_stage1_autotune_configs

_AUTOTUNE_SCHEMA = 2
_SC0_CACHE = 1


class _LdsF32View:
    """Float32 LDS view (.ptr) over the Int8 A_buf pool, for the epilogue cshuffle staging."""

    def __init__(self, ptr):
        self.ptr = ptr


@functools.lru_cache(maxsize=None)
# fmt: off
def compile_mega_moe_stage1(*, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int,
    fuse_topk: int, fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, sort_block_m: int=32, tile_n: int=256,
    tile_k: int=256, num_waves: int=4, grid_mult: int=8, wgm: int=1, sched_nmajor: bool=False,
    mfma_amajor: bool=True, swizzle_a: bool=True, use_xcd: bool=True, num_cu: int=256,
    num_dispatch_cu: int=32):
# fmt: on
    NUM_WAVES = int(num_waves)
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    assert (2 * inter_dim) % tile_n == 0, "2*inter_dim must tile evenly by tile_n"
    N_TILES = (2 * inter_dim) // tile_n
    assert grid_mult in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32), "grid_mult out of range"
    NUM_XCDS = 8
    grid_x = num_cu * grid_mult  # GEMM-consumer grid; producer blocks are added separately.
    dispatch_blocks = int(num_dispatch_cu)
    assert 0 < dispatch_blocks <= num_cu, "num_dispatch_cu must be in [1, num_cu]"
    assert not use_xcd or dispatch_blocks % NUM_XCDS == 0, "XCD mode requires num_dispatch_cu % 8 == 0"
    launch_grid_x = dispatch_blocks + grid_x
    use_xcd_eff = use_xcd and (grid_x % NUM_XCDS == 0)
    GROUP_SIZE = wgm * N_TILES
    BLOCKS_PER_XCD = grid_x // NUM_XCDS
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k // 2  # fp4 packed K-step bytes
    assert TILE_K_BYTES % 128 == 0
    A_K_STEP_BYTES = tile_k
    assert A_K_STEP_BYTES == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = max(2 * a_lds_size, cs_size * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)

    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    fz_total_experts = fz_npes * fz_epr
    fz_n_i32, fz_nbytes = model_dim // 4, model_dim
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    fz_enable_scales = fz_scale_bytes > 0
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(known_block_size=[64, 1, 1])
    def epoch_bump_kernel(addr_parity: fx.Int64, addr_expected: fx.Int64):
        if fx.thread_idx.x == 0:
            parity_rsrc = _buffer_ops.create_buffer_resource_from_addr(addr_parity)
            expected_rsrc = _buffer_ops.create_buffer_resource_from_addr(addr_expected)
            parity = _buffer_ops.buffer_load(
                parity_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, cache_modifier=_SC0_CACHE
            )
            next_parity = parity ^ fx.Int32(1)
            expected = _buffer_ops.buffer_load(
                expected_rsrc, next_parity, vec_width=1, dtype=fx.Int32, cache_modifier=_SC0_CACHE
            )
            _buffer_ops.buffer_store(
                expected + fx.Int32(1), expected_rsrc, next_parity, cache_modifier=_SC0_CACHE
            )
            _buffer_ops.buffer_store(next_parity, parity_rsrc, fx.Int32(0), cache_modifier=_SC0_CACHE)
            fx.rocdl.s_waitcnt(0)

    @flyc.kernel(known_block_size=[TOTAL_THREADS, 1, 1])
    # fmt: off
    def kernel(out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, n: fx.Int32, k: fx.Int32, size_expert_ids: fx.Int32, addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64, addr_in_idx: fx.Int64, addr_in_wts: fx.Int64,
        addr_in_sc: fx.Int64, addr_parity: fx.Int64, addr_expected: fx.Int64, addr_ready: fx.Int64):
    # fmt: on
        block_index = fx.Int32(fx.block_idx.x)
        gemm_block_index = block_index - fx.Int32(dispatch_blocks)
        parity_rsrc = _buffer_ops.create_buffer_resource_from_addr(addr_parity)
        expected_rsrc = _buffer_ops.create_buffer_resource_from_addr(addr_expected)
        epoch_parity = _buffer_ops.buffer_load(
            parity_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, cache_modifier=_SC0_CACHE
        )
        epoch_expected = _buffer_ops.buffer_load(
            expected_rsrc, epoch_parity, vec_width=1, dtype=fx.Int32, cache_modifier=_SC0_CACHE
        )

        def _run_dispatch():
            # fmt: off
            emit_dispatch_prologue(num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_tile_m=fz_tile_m, fz_total_experts=fz_total_experts,
                fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32, fz_safe_end_i32=fz_safe_end_i32,
                fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales, addr_disp=addr_disp,
                i32_cur_tok=i32_cur_tok, addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx,
                addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc, dispatch_blocks=dispatch_blocks,
                addr_ready=addr_ready, epoch_parity=epoch_parity, epoch_expected=epoch_expected)
            # fmt: on

        if block_index < fx.Int32(dispatch_blocks):
            _run_dispatch()
        else:
            wait_dispatch_ready(addr_ready, epoch_parity, epoch_expected)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))

        wave_id = fx.thread_idx.x // 64

        x_rsrc = _buffer_ops.create_buffer_resource(x, max_size=True)
        w_rsrc = _buffer_ops.create_buffer_resource(w, max_size=True)
        sx_rsrc = _buffer_ops.create_buffer_resource(scale_x, max_size=True)
        sw_rsrc = _buffer_ops.create_buffer_resource(scale_w, max_size=True)
        trb_rsrc = _buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)  # tile_row_base
        expert_rsrc = _buffer_ops.create_buffer_resource(expert_ids, max_size=True)
        nv_rsrc = _buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        _rows = size_expert_ids * fx.Int32(sort_block_m)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        out_nbytes = _rows * fx.Int32(inter_dim)
        os_nbytes = _rows * fx.Int32(scale_cols) + fx.Int32(8192)
        out_rsrc = _buffer_ops.create_buffer_resource(out, max_size=False, num_records_bytes=out_nbytes)
        os_rsrc = _buffer_ops.create_buffer_resource(out_scale, max_size=False, num_records_bytes=os_nbytes)

        num_valid = _buffer_ops.buffer_load(nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)

        sched = TileScheduler(
            expert_rsrc=expert_rsrc,
            inter_dim=inter_dim,
            use_xcd=use_xcd_eff,
            expert_offset=fz_rank * fz_epr,  # GLOBAL sorted_expert_id -> LOCAL w1 index
        )
        n_wave_base = wave_id * fx.Int32(n_per_wave)

        # fmt: off
        a_gather = ATileLoader(x_rsrc=x_rsrc, row_bytes=model_dim, sort_block_m=sort_block_m,
            k_step_bytes=A_K_STEP_BYTES, total_threads=TOTAL_THREADS, swizzle=swizzle_a)
        # fmt: on
        a_s2r = AS2RLoader(m_repeat=M_REPEAT, k_step_bytes=A_K_STEP_BYTES, swizzle=swizzle_a)
        b_loader = BWeightLoader(w_rsrc=w_rsrc, num_acc_n=NUM_ACC_N, k_step_bytes=TILE_K_BYTES, model_dim=model_dim)
        b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=NUM_ACC_N, model_dim=model_dim)
        a_scale = AScaleLoader(
            scale_rsrc=sx_rsrc,
            m_repeat=M_REPEAT,
            model_dim=model_dim,
            sort_block_m=sort_block_m,
            total_threads=TOTAL_THREADS,
        )
        mfma = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N)
        # fmt: off
        epi = SiluQuantEpilogue(out_rsrc=out_rsrc, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0,
            inter_dim=inter_dim, m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, sort_block_m=sort_block_m, tile_n=tile_n,
            num_waves=NUM_WAVES, lds_out=c_tile, always_valid=True)
        # fmt: on

        if const_expr(use_xcd_eff):
            it0 = gemm_block_index // fx.Int32(NUM_XCDS)
            it_stride = fx.Int32(BLOCKS_PER_XCD)
            xcd_id = gemm_block_index % fx.Int32(NUM_XCDS)
        else:
            it0 = gemm_block_index
            it_stride = fx.Int32(grid_x)
            xcd_id = fx.Int32(0)

        def _it_to_flat(it_v):
            if const_expr(use_xcd_eff):
                gsz = fx.Int32(GROUP_SIZE)
                gi = it_v // gsz
                within = it_v - gi * gsz
                sg = xcd_id + gi * fx.Int32(NUM_XCDS)
                return sg * gsz + within
            return it_v

        # fmt: off
        def _do_tile(m_tile, n_tile_base, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, sort_block_m, mfma_amajor,
            trb_rsrc):
        # fmt: on
            N_ACC = M_REPEAT * NUM_ACC_N
            last = fx.Int32(K_ITERS - 1)
            tile_row_base = _buffer_ops.buffer_load(trb_rsrc, m_tile, vec_width=1, dtype=fx.Int32)
            expert = sched.expert_of(m_tile)
            b_row = sched.gate_base_row(expert) + n_tile_base
            a_gather.for_tile(tile_row_base)
            a_gather.store(a_buf, a_gather.load_regs(fx.Int32(0)), fx.Int32(0))
            a_scale.stage(a_scale_lds, tile_row_base)
            wait_lds_barrier()
            b0 = b_loader.load_step(b_row, fx.Int32(0))
            init = [mfma.zero_value for _ in range(N_ACC)]
            init += [h for ni_list in b0 for h in ni_list]
            for sp_i, state in range(0, K_ITERS, 1, init=init):
                sp = fx.Int32(sp_i)
                acc = [Vec(a) for a in state[:N_ACC]]
                b_prev = [[Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)]
                cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
                nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
                spn = (sp + fx.Int32(1) < last).select(sp + fx.Int32(1), last)
                a_regs = a_gather.load_regs(spn * fx.Int32(A_K_STEP_BYTES))

                def a_load(mi, ks, _base=cur_off):
                    return a_s2r.load_operand(a_buf, mi, ks, _base)

                sa = a_scale.load_step(a_scale_lds, sp)
                sb = b_scale.load_step(b_row, sp)

                def load_next(ni, _kn=spn):
                    return b_loader.load_ni(b_row, ni, _kn)

                call_pipe = mfma.call_pipe_am if mfma_amajor else mfma.call_pipe
                acc, b_next = call_pipe(a_load, b_prev, acc, sa, sb, load_next)
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
                yv = list(acc) + [h for ni_list in b_next for h in ni_list]
                state = yield yv
            acc = [Vec(r) for r in state[:N_ACC]]
            epi.store(acc, m_tile, tile_row_base, n_tile_base)

        if gemm_block_index >= fx.Int32(0):
            itv = it0
            while _it_to_flat(itv) < total_work:
                flat = _it_to_flat(itv)
                if const_expr(sched_nmajor):
                    n_tile = flat // num_m_tiles
                    m_tile = flat - n_tile * num_m_tiles
                else:
                    m_tile = flat // fx.Int32(N_TILES)
                    n_tile = flat - m_tile * fx.Int32(N_TILES)
                n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
                # fmt: off
                _do_tile(m_tile, n_tile_base, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi,
                    a_buf, a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, sort_block_m,
                    mfma_amajor, trb_rsrc)
                # fmt: on
                itv = itv + it_stride

    @flyc.jit
    # fmt: off
    def launch(out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, n: fx.Int32, k: fx.Int32, size_expert_ids: fx.Int32, addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64, addr_in_idx: fx.Int64, addr_in_wts: fx.Int64,
        addr_in_sc: fx.Int64, addr_parity: fx.Int64, addr_expected: fx.Int64, addr_ready: fx.Int64,
        stream: fx.Stream):
    # fmt: on
        epoch_bump_kernel(addr_parity, addr_expected).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
        kernel(
            out,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            num_valid_ids,
            out_scale,
            tokens,
            n,
            k,
            size_expert_ids,
            addr_disp,
            i32_cur_tok,
            addr_in_tok,
            addr_in_idx,
            addr_in_wts,
            addr_in_sc,
            addr_parity,
            addr_expected,
            addr_ready,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}"},
        ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch


def _collective_bench(fn, warmup, rep, quantiles=None):
    elapsed = do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    if not dist.is_initialized():
        return elapsed
    value = torch.tensor(float(elapsed), dtype=torch.float32, device=torch.cuda.current_device())
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


class _CollectiveAutotuner(Autotuner):
    @staticmethod
    def _config_signature(config):
        return tuple(sorted(config.to_dict().items()))

    def _is_allowed_config(self, config):
        allowed = getattr(self, "_allowed_config_signatures", None)
        if allowed is None:
            allowed = {self._config_signature(candidate) for candidate in self.configs}
            self._allowed_config_signatures = allowed
        return self._config_signature(config) in allowed

    def _load_disk_cache(self):
        super()._load_disk_cache()
        self.cache = {key: config for key, config in self.cache.items() if self._is_allowed_config(config)}

    def __call__(self, *args, **kwargs):
        if not dist.is_initialized():
            return super().__call__(*args, **kwargs)
        key = self._make_key(args, kwargs)
        ready = getattr(self, "_collective_ready", set())
        if key in ready:
            if key in self.cache and self._is_allowed_config(self.cache[key]):
                return self._run_config(self.cache[key], args, kwargs)
            ready.discard(key)
            self.cache.pop(key, None)
        payload = [self.cache[key].to_dict() if dist.get_rank() == 0 and key in self.cache else None]
        dist.broadcast_object_list(payload, src=0)
        if payload[0] is not None:
            config = Config.from_dict(payload[0])
            if self._is_allowed_config(config):
                self.cache[key] = config
                ready.add(key)
                self._collective_ready = ready
                return self._run_config(config, args, kwargs)
        self.cache.pop(key, None)
        result = super().__call__(*args, **kwargs)
        ready.add(key)
        self._collective_ready = ready
        return result

    def _save_disk_cache(self):
        distributed = dist.is_initialized()
        error = None
        if not distributed or dist.get_rank() == 0:
            try:
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                lock_path = self._cache_file.with_suffix(".lock")
                with lock_path.open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    data = {}
                    if self._cache_file.exists():
                        try:
                            data = json.loads(self._cache_file.read_text())
                        except (OSError, ValueError):
                            data = {}
                    for key, config in self.cache.items():
                        if self._is_allowed_config(config):
                            data[json.dumps(list(key))] = config.to_dict()
                    tmp = self._cache_file.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_text(json.dumps(data, indent=2))
                    os.replace(tmp, self._cache_file)
            except Exception as exc:
                error = exc
        if distributed:
            dist.barrier()
        if error is not None:
            raise error


# fmt: off
def _run_stage1_config(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, n, k, size_expert_ids, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, addr_ready, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, sort_block_m, num_cu, use_xcd, tune_tokens,
    dispatch_constraint, grid_constraint, autotune_schema, tile_n=256, tile_k=256, num_waves=4, wgm=1,
    grid_mult=4, sched_nmajor=False, mfma_amajor=True, swizzle_a=True, num_dispatch_cu=32):
# fmt: on
    del tune_tokens, dispatch_constraint, grid_constraint, autotune_schema
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k,
        num_waves=num_waves, grid_mult=grid_mult, wgm=wgm, sched_nmajor=sched_nmajor,
        mfma_amajor=mfma_amajor, swizzle_a=swizzle_a, use_xcd=use_xcd, num_cu=num_cu,
        num_dispatch_cu=num_dispatch_cu,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        tokens, n, k, size_expert_ids, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts,
        addr_in_sc, addr_parity, addr_expected, addr_ready, stream,
    )


@functools.lru_cache(maxsize=None)
def make_stage1_autotuner(dispatch_cu=None, grid_mult=None):
    configs = get_stage1_autotune_configs(dispatch_cu=dispatch_cu, grid_mult=grid_mult)
    key = [
        "model_dim", "inter_dim", "experts_per_rank", "fuse_npes", "fuse_topk", "fuse_cap", "fuse_mtpr",
        "fuse_scale_dim", "sort_block_m", "num_cu", "use_xcd", "tune_tokens", "dispatch_constraint",
        "grid_constraint", "autotune_schema",
    ]
    tuner = _CollectiveAutotuner(
        _run_stage1_config, configs=configs, key=key, warmup=2, rep=7, do_bench_fn=_collective_bench
    )
    tuner.schema = _AUTOTUNE_SCHEMA
    return tuner
