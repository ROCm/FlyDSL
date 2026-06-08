"""Kimi fp4 MoE FlyDSL kernels for the M=16384 tuned row.

This file intentionally keeps only the CSV-selected fp4 path:

    token=16384, model_dim=7168, inter_dim=512, experts=385, topk=9
    stage1=flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_xcd4
    stage2=flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_xcd4

No CSV dispatch, alternate dtype kernels, split reduction variants, optional
epilogues, or fallback selection is kept here.
"""
from __future__ import annotations
import functools
import os
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl.expr.typing import T
from flydsl.expr import arith, buffer_ops, const_expr, gpu, rocdl, vector
from flydsl._mlir.dialects import llvm, memref, scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from aiter.ops.flydsl.kernels.layout_utils import crd2idx, idx2crd, get as layout_get
from aiter.ops.flydsl.kernels.mfma_epilogues import c_shuffle_epilog
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    swizzle_xor16,
    tile_chunk_coord_i32,
)
from dataclasses import dataclass
from typing import Optional
import torch
import aiter
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter import dtypes
TOKEN = 16384
EXPERTS = 385
MODEL_DIM = 7168
INTER_DIM = 512
TOPK = 9
BLOCK_M = 64
STAGE1_KERNEL = 'flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_xcd4'
STAGE2_KERNEL = 'flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_xcd4'

@dataclass(frozen=True)
class Stage1Config:
    tile_m: int = 64
    tile_n: int = 128
    tile_k: int = 256
    waves_per_eu: int = 4
    b_nt: int = 0
    xcd_swizzle: int = 4

@dataclass(frozen=True)
class Stage2Config:
    tile_m: int = 64
    tile_n: int = 256
    tile_k: int = 256
    b_nt: int = 0
    xcd_swizzle: int = 4
    persist_m: int = -1
S1 = Stage1Config()
S2 = Stage2Config()
_DLPACK_SAFE = (torch.uint8, torch.float16, torch.bfloat16, torch.float32)

def _view_safe(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t is not None and t.numel() > 0 and (t.dtype not in _DLPACK_SAFE) else t

def _ptr_view_safe(t: torch.Tensor):
    view = _view_safe(t)
    type_name = type(view).__name__
    module_name = type(view).__module__
    if type_name == 'FakeTensor' or 'fake_tensor' in module_name:
        return flyc.from_c_void_p(fx.Uint8, 0)
    return flyc.from_c_void_p(fx.Uint8, view.data_ptr())

def _barrier(vmcnt=63, lgkmcnt=63):
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f'vmcnt({vmcnt})')
        if lgkmcnt < 63:
            wc.append(f'lgkmcnt({lgkmcnt})')
        parts.append('s_waitcnt ' + ' '.join(wc))
    parts.append('s_barrier')
    llvm.InlineAsmOp(res=None, operands_=[], asm_string='\n'.join(parts), constraints='', has_side_effects=True, is_align_stack=False)

def _run_compiled(exe, args):
    try:
        exe(*args)
    except Exception:
        try:
            from flydsl._mlir import ir
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:
            pass
        raise

def kimi_moe_sorting_16384(topk_ids: torch.Tensor, topk_weight: torch.Tensor, moebuf_dtype):
    """Run the fixed Opus MoE sorting step used before the two FlyDSL GEMMs."""
    if tuple(topk_ids.shape) != (TOKEN, TOPK):
        raise ValueError(f'expected topk_ids shape {(TOKEN, TOPK)}, got {tuple(topk_ids.shape)}')
    device = topk_ids.device
    max_sorted = int(topk_ids.numel() + EXPERTS * BLOCK_M - TOPK)
    max_m_blocks = int((max_sorted + BLOCK_M - 1) // BLOCK_M)
    sorted_ids = torch.empty(max_sorted, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_sorted, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(max_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_out = torch.empty((TOKEN, MODEL_DIM), dtype=moebuf_dtype, device=device)
    workspace_size = aiter.moe_sorting_opus_get_workspace_size(TOKEN, EXPERTS, TOPK, 0)
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device) if workspace_size > 0 else None
    aiter.moe_sorting_opus_fwd(topk_ids, topk_weight, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out, EXPERTS, BLOCK_M, None, None, workspace, 0, None)
    return (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out)

@functools.lru_cache(maxsize=1)
def compile_kimi_fp4_stage1_16384():
    """Compile the fixed Kimi fp4 stage1 kernel selected for M=16384."""
    model_dim = MODEL_DIM
    inter_dim = INTER_DIM
    experts = EXPERTS
    topk = TOPK
    tile_m = S1.tile_m # 64
    tile_n = S1.tile_n # 128
    tile_k = S1.tile_k # 256
    model_dim_pad = 0
    inter_dim_pad = 0
    persist_m = 1
    use_async_copy = True
    waves_per_eu = S1.waves_per_eu
    k_batch = 1
    b_nt = S1.b_nt
    xcd_swizzle = S1.xcd_swizzle # TODO(zty), 为啥是4?
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name='smem0')
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name='smem1')
    sort_block_m = max(32, tile_m)
    num_waves = min(4, tile_n // 32) # 4
    total_threads = num_waves * 64
    pack_M = 1 if tile_m < 32 else 2
    n_per_wave = tile_n // num_waves
    pack_N = min(2, n_per_wave // 16)
    pack_K = 2
    scale_mn_pack = 2
    elem_bytes = 1
    a_elem_bytes = 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes) # 256
    a_elem_vec_pack = 2
    cbsz = 4
    blgp = 4
    if tile_k_bytes % 64 != 0:
        raise ValueError(f'tile_k_bytes must be divisible by 64, got {tile_k_bytes}')

    def _x_elem_type():
        return T.i8

    def _w_elem_type():
        return T.i8

    def out_elem():
        return T.bf16
    _k_per_batch = model_dim
    _k_dim = model_dim
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(f'tile_m*tile_k*elem_bytes must be divisible by {total_threads}')
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    _use_lds128 = os.environ.get('FLIR_CK_LDS128', '1') in ('1', 'true', 'True', 'YES', 'yes')
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k # 256 + 0 = 256
    _use_cshuffle_epilog = True
    _xcd_tag = f'_xcd{xcd_swizzle}' if xcd_swizzle > 0 else ''
    module_name = f'mfma_moe1_silu_mul_afp4_wfp4_bf16_t{tile_m}x{tile_n}x{tile_k}_pm{persist_m}_async{_xcd_tag}_v32'
    _cshuffle_elem_bytes = 2
    _single_x_bytes = int(tile_m) * int(lds_stride) * int(a_elem_bytes) # 16k = 64 * 256 * 1
    lds_out_bytes = _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0 # 16k = 2 * 64 * 128
    lds_tid_bytes = int(tile_m) * 4 # 256 = 64 * 4
    _input_elems = _single_x_bytes if a_elem_bytes == 1 else _single_x_bytes // 2
    _GLOBAL_ALIGN = 1024
    _std_pong = max(_single_x_bytes, lds_out_bytes) + lds_tid_bytes # 16k + 256
    _std_ping = _single_x_bytes # 16k = 64 * 256 # A tile 放进lds的大小.
    _std_pong_aligned = allocator_pong._align(_std_pong, 128) # 256
    _std_total = allocator_pong._align(_std_pong_aligned, _GLOBAL_ALIGN) + allocator_pong._align(_std_ping, 128) # 
    _lds_limit = {'gfx950': 163840, 'gfx942': 65536}.get(gpu_arch, 0)
    _split_lds_out = _lds_limit > 0 and lds_out_bytes > 0 and (_std_total > _lds_limit) and (num_waves >= 2) # #False = 160k > 0 and 16k > 0 and ()
    print(f"{_split_lds_out=}, {_std_total=}, {_lds_limit=}, {num_waves=}, {_std_pong=}, {_std_ping=}, {_std_pong_aligned=}, {_std_total=}")
    if _split_lds_out: # False
        _half_out_bytes = _cshuffle_elem_bytes * int(tile_m) * (int(tile_n) // 2)
        _pong_buffer_bytes = max(_single_x_bytes, _half_out_bytes)
        _ping_buffer_bytes = max(_single_x_bytes, _half_out_bytes)
    else:
        _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
        _ping_buffer_bytes = _single_x_bytes

    def x_lds_elem():
        return T.f8
    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes
    _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes
    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes
    if waves_per_eu is not None and waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds
    kpack_bytes = 16
    out_elem_bytes = 2
    w_elem_bytes = 1
    w_elem_pack = 2
    w_nbytes = experts * (2 * inter_dim) * model_dim * w_elem_bytes // w_elem_pack
    _e_vec_s1 = min(tile_n // 32, 8)
    _pipe_m_repeat = tile_m // 16
    _pipe_k_unroll = tile_k_bytes // 128
    _pipe_k_unroll_packed = _pipe_k_unroll // pack_K
    _pipe_m_repeat_packed = _pipe_m_repeat // pack_M
    _pipe_num_acc_n = n_per_wave // 16
    _pipe_a_groups = []
    for _mi in range(_pipe_m_repeat):
        _grp = []
        for _k in range(_pipe_k_unroll):
            _grp.append((_k, _mi))
            if len(_grp) == 2:
                _pipe_a_groups.append(_grp)
                _grp = []
        if _grp:
            _pipe_a_groups.append(_grp)
    _pipe_b_loads = []
    for ku in range(_pipe_k_unroll):
        for ni in range(_pipe_num_acc_n):
            _pipe_b_loads.append(('gate', ku, ni))
            _pipe_b_loads.append(('up', ku, ni))
    _pipe_num_acc_n_packed = _pipe_num_acc_n // pack_N
    _pipe_all_mfma = []
    for _ku128 in range(_pipe_k_unroll_packed):
        for _ni_packed in range(_pipe_num_acc_n_packed):
            for _ikxdl in range(pack_K):
                for _inxdl in range(pack_N):
                    _k_idx = _ku128 * pack_K + _ikxdl
                    _ni_idx = _ni_packed * pack_N + _inxdl
                    _pipe_all_mfma.append((_k_idx, _ni_idx, _ikxdl, _inxdl, _ku128))
    _pipe_mfma_per_phase = max(1, len(_pipe_all_mfma) // 4)
    _pipe_n_phases = len(_pipe_all_mfma) // _pipe_mfma_per_phase
    _a_groups_per_phase = (len(_pipe_a_groups) + _pipe_n_phases - 1) // _pipe_n_phases
    _pipe_phases = []
    _mfma_i = 0
    _a_i = 0
    for _p in range(_pipe_n_phases):
        _a_reads = []
        for _ in range(_a_groups_per_phase):
            if _a_i < len(_pipe_a_groups):
                _a_reads.extend(_pipe_a_groups[_a_i])
                _a_i += 1
        _phase = {'mfma': _pipe_all_mfma[_mfma_i:_mfma_i + _pipe_mfma_per_phase], 'a_reads': _a_reads, 'b_loads': [], 'has_scale': _p == 0}
        _mfma_i += _pipe_mfma_per_phase
        _pipe_phases.append(_phase)
    _bi = 0
    for _p in range(1, _pipe_n_phases):
        _rem_b = len(_pipe_b_loads) - _bi
        _rem_p = _pipe_n_phases - _p
        _n_b = (_rem_b + _rem_p - 1) // _rem_p if _rem_p > 0 else 0
        for _ in range(_n_b):
            if _bi < len(_pipe_b_loads):
                _pipe_phases[_p]['b_loads'].append(_pipe_b_loads[_bi])
                _bi += 1
    _pp_mfma = [p['mfma'] for p in _pipe_phases]
    _pp_a_reads = [p['a_reads'] for p in _pipe_phases]
    _pp_b_loads = [p['b_loads'] for p in _pipe_phases]
    _pp_has_scale = [p['has_scale'] for p in _pipe_phases]
    fp4_ratio = 2
    gui_ratio = 2
    _vmcnt_before_barrier = tile_m // 32 // fp4_ratio + tile_n // 32 * gui_ratio

    @flyc.kernel(name=module_name)
    def moe_gemm1(arg_out: fx.Pointer, arg_x: fx.Pointer, arg_w: fx.Pointer, arg_scale_x: fx.Pointer, arg_scale_w: fx.Pointer, arg_sorted_token_ids: fx.Pointer, arg_expert_ids: fx.Pointer, arg_num_valid_ids: fx.Pointer, i32_tokens_in: fx.Int32, i32_n_in: fx.Int32, i32_k_in: fx.Int32, i32_size_expert_ids_in: fx.Int32):
        tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
        size_expert_ids_in = arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
        x_elem = T.f8
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        vec4_f32 = T.vec(4, f32)
        vec16_elems = 16 if a_elem_bytes == 1 else 8
        vec16_x = T.vec(vec16_elems, x_elem)
        vec2_i64 = T.vec(2, i64)

        def _ptr_buffer_resource(ptr, num_records_bytes): # 这种函数是不是能放外面呢.. 创建 buffer resources.
            addr = fx.ptrtoint(ptr)
            addr_i64 = arith.index_cast(T.i64, addr)
            return buffer_ops.create_buffer_resource_from_addr(addr_i64, num_records_bytes=num_records_bytes)
        acc_init = arith.constant_vector(0.0, vec4_f32) # 这个可以优化到mfma第一次直接初始化掉.
        c_n_total = arith.constant(experts * (2 * inter_dim), index=True) # 385 * 2 * 512.
        b_layout = make_preshuffle_b_layout(arith, c_n=c_n_total, c_k=k_in // pack_K, kpack_bytes=kpack_bytes, elem_bytes=b_elem_bytes) # B的layout. 这里我猜测 和我之前想得应该是一样的？TODO(zty): 
        layout_b = b_layout.layout_b
        sorted_m = size_expert_ids_in * arith.constant(sort_block_m, index=True) # 64.
        layout_a_scale = make_preshuffle_scale_layout(arith, c_mn=sorted_m, c_k=arith.constant(model_dim, index=True))
        layout_b_scale = make_preshuffle_scale_layout(arith, c_mn=c_n_total, c_k=arith.constant(model_dim, index=True))
        _eff_lds_stride = lds_stride # 256.
        _eff_tile_k_bytes = tile_k_bytes
        if const_expr(use_async_copy and a_elem_vec_pack > 1): # 2
            _eff_lds_stride = lds_stride // a_elem_vec_pack # 128 = 256 // 2
            _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack # 128 = 256 // 2
        shape_lds = fx.make_shape(tile_m, _eff_lds_stride) # 64, 128
        stride_lds = fx.make_stride(_eff_lds_stride, 1) # 128, 1
        layout_lds = fx.make_layout(shape_lds, stride_lds) # 64, 128
        tx = gpu.thread_id('x')
        by = gpu.block_id('x')
        bx_persist = gpu.block_id('y')
        if const_expr(xcd_swizzle > 0):
            _NUM_XCDS_S1 = 8
            _c1_sw = arith.constant(1, index=True) # 1
            _c_tn_sw = arith.constant(tile_n, index=True) # 128
            _c_idp_sw = arith.constant(2 * inter_dim_pad, index=True) # 0
            _c2_sw = arith.constant(2, index=True)
            _gx = (n_in - _c_idp_sw + _c2_sw * _c_tn_sw - _c1_sw) / _c_tn_sw / _c2_sw
            _c_pm_sw = arith.constant(persist_m, index=True)
            _gy = (size_expert_ids_in + _c_pm_sw - _c1_sw) / _c_pm_sw
            _linear_id = bx_persist * _gx + by
            _num_wgs = _gx * _gy
            _c_xcds = arith.constant(_NUM_XCDS_S1, index=True)
            _wgs_per_xcd = _num_wgs / _c_xcds
            _wgid = _linear_id % _c_xcds * _wgs_per_xcd + _linear_id / _c_xcds
            _WGM_S1 = xcd_swizzle
            _c_wgm = arith.constant(_WGM_S1, index=True)
            _num_wgid_in_group = _c_wgm * _gx
            _group_id = _wgid / _num_wgid_in_group
            _first_pid_m = _group_id * _c_wgm
            _remaining_m = _gy - _first_pid_m
            _cmp_m = arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
            _group_size_m = arith.select(_cmp_m, _remaining_m, _c_wgm)
            _wgid_in_group = _wgid % _num_wgid_in_group
            bx_persist = _first_pid_m + _wgid_in_group % _group_size_m
            by = _wgid_in_group / _group_size_m
        by_n = by * arith.constant(tile_n, index=True) # by * 128
        k_base_idx = arith.index(0)
        k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True) # 8 = 128 / 16. 
        layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1)) # (4,64):(64,1)
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1)) # (4,16):(16,1)
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_x_pong = SmemPtr(base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(_input_elems,)).get()
        lds_x_ping = SmemPtr(base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(_input_elems,)).get()
        _lds_out_elem_type = T.bf16
        if const_expr(_split_lds_out and _use_cshuffle_epilog): # False
            _half_out_elems = int(tile_m) * (int(tile_n) // 2)
            lds_out = SmemPtr(base_ptr_pong, lds_pong_offset, _lds_out_elem_type, shape=(_half_out_elems,)).get()
            lds_out_B = SmemPtr(base_ptr_ping, lds_ping_offset, _lds_out_elem_type, shape=(_half_out_elems,)).get()
        else:
            lds_out = SmemPtr(base_ptr_pong, lds_pong_offset, _lds_out_elem_type, shape=(tile_m * tile_n,)).get() if _use_cshuffle_epilog else None # True (64 * 128)
            lds_out_B = None
        lds_tid = SmemPtr(base_ptr_pong, _lds_tid_offset_pong, T.i32, shape=(tile_m,)).get()
        c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
        c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)
        x_nbytes_idx = tokens_in * k_in * c_elem_bytes / c_a_pack
        x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
        x_rsrc = _ptr_buffer_resource(arg_x, x_nbytes_i32)
        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)
        numids_rsrc = _ptr_buffer_resource(arg_num_valid_ids, arith.constant(4, type=T.i32))
        num_valid_i32 = buffer_ops.buffer_load(numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32)
        sx_rsrc = 1
        sw_rsrc = 1
        c32 = arith.constant(32, index=True)
        kblk = k_in / c32
        sx_nbytes_idx = sorted_m * kblk
        sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
        sx_rsrc = _ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)
        c32 = arith.constant(32, index=True)
        kblk_w = k_in / c32
        mn_w = arith.constant(experts * (2 * inter_dim), index=True)
        sw_nbytes_idx = mn_w * kblk_w
        sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
        sw_rsrc = _ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)
        sorted_nbytes_idx = size_expert_ids_in * arith.constant(sort_block_m * 4, index=True)
        sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
        sorted_rsrc = _ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
        eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
        eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
        expert_rsrc = _ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
        _PERSIST_M = persist_m
        _c0_p = arith.constant(0, index=True)
        _c1_p = arith.constant(1, index=True)
        _c_pm = arith.constant(_PERSIST_M, index=True)
        _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p)
        _for_ip = ir.InsertionPoint(_for_persist.body)
        _for_ip.__enter__()
        _mi_p = _for_persist.induction_variable
        bx = bx_persist * _c_pm + _mi_p
        bx_m = bx * arith.constant(sort_block_m, index=True)
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
        expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
        expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32))

        def _moe_gemm1_body():
            expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)
            x_load_bytes = 16
            num_x_loads = bytes_per_thread_x // x_load_bytes
            chunk_i32 = x_load_bytes // 4
            c_k_div4 = k_in / c_a_pack * arith.constant(int(a_elem_bytes), index=True) / arith.index(4)
            tile_k_dwords = int(tile_k) * int(a_elem_bytes) // (4 * int(a_elem_vec_pack))
            layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            c_chunk_i32 = arith.constant(chunk_i32, index=True)
            tx_i32_base = tx * c_chunk_i32
            topk_i32 = arith.constant(topk)
            mask24 = arith.constant(16777215)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(arith, tx_i32_base=tx_i32_base, i=i, total_threads=total_threads, layout_tile_div4=layout_x_tile_div4, chunk_i32=chunk_i32)

            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)
                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                t_i32 = arith.andi(fused_i, mask24)
                s_i32 = arith.shrui(fused_i, arith.constant(24))
                t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                ts_valid = arith.andi(t_valid, s_valid)
                t_safe = arith.select(ts_valid, t_i32, arith.constant(0))
                t_idx = arith.index_cast(ir.IndexType.get(), t_safe)
                x_row_base_div4.append(t_idx * c_k_div4)
            coord_wl = idx2crd(tx, layout_tx_wave_lane)
            wave_id = layout_get(coord_wl, 0)
            lane_id = layout_get(coord_wl, 1)
            coord_l16 = idx2crd(lane_id, layout_lane16)
            lane_div_16 = layout_get(coord_l16, 0)
            lane_mod_16 = layout_get(coord_l16, 1)
            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * arith.constant(16, index=True)
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_n_id = wave_id % arith.constant(num_waves, index=True)
            n_tile_base = wave_n_id * c_n_per_wave
            gate_n_intra_list = []
            gate_n_blk_list = []
            up_n_intra_list = []
            up_n_blk_list = []
            col_g_list = []
            c_n0_static = experts * (2 * inter_dim) // 16
            layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
            inter_idx = arith.constant(inter_dim, index=True)
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)
                col_g = by_n + n_tile_base + c_offset + lane_mod_16
                col_g_list.append(col_g)
                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                gate_row_w = expert_off_idx + global_n
                gate_coord = idx2crd(gate_row_w, layout_n_blk_intra)
                gate_n_blk_list.append(layout_get(gate_coord, 0))
                gate_n_intra_list.append(layout_get(gate_coord, 1))
                up_row_w = gate_row_w + inter_idx
                up_coord = idx2crd(up_row_w, layout_n_blk_intra)
                up_n_blk_list.append(layout_get(up_coord, 0))
                up_n_intra_list.append(layout_get(up_coord, 1))
            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 128
            k_unroll_packed = k_unroll // pack_K
            m_repeat_packed = m_repeat // pack_M
            num_acc_n_packed = num_acc_n // pack_N
            _K_per_ku = tile_k // k_unroll
            _pad_k_elems = model_dim_pad % tile_k if model_dim_pad > 0 else 0
            _pad_ku_skip = _pad_k_elems // _K_per_ku
            _tail_ku = k_unroll - _pad_ku_skip
            _tail_ku_packed = (_tail_ku + pack_K - 1) // pack_K if _pad_ku_skip > 0 else None

            def load_b_packs_k64(base_k, ku: int, n_blk, n_intra): # B g2r?
                c64 = arith.constant(64, index=True)
                base_k_bytes = base_k * arith.constant(int(b_elem_bytes), index=True)
                k0 = base_k_bytes // c64 + arith.constant(ku, index=True)
                k1 = lane_div_16
                coord_pack = (n_blk, k0, k1, n_intra, arith.constant(0, index=True))
                idx_pack = crd2idx(coord_pack, layout_b)
                vec_elems = kpack_bytes // int(b_elem_bytes)
                b16 = _buffer_load_vec(buffer_ops, vector, w_rsrc, idx_pack, elem_type=_w_elem_type(), vec_elems=vec_elems, elem_bytes=b_elem_bytes, offset_in_bytes=b_elem_bytes == 1, cache_modifier=b_nt)
                b_i64x2 = vector.bitcast(vec2_i64, b16)
                b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                return (b0, b1)

            def load_b_tile(base_k, ku_limit=k_unroll): # gate down & gate up.
                """Load separated gate and up B tiles."""
                gate_b_tile = []
                up_b_tile = []
                for ku in range_constexpr(ku_limit):
                    g_packs0, g_packs1 = ([], [])
                    u_packs0, u_packs1 = ([], [])
                    for ni in range_constexpr(num_acc_n):
                        gb0, gb1 = load_b_packs_k64(base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni])
                        g_packs0.append(gb0)
                        g_packs1.append(gb1)
                        ub0, ub1 = load_b_packs_k64(base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni])
                        u_packs0.append(ub0)
                        u_packs1.append(ub1)
                    gate_b_tile.append((g_packs0, g_packs1))
                    up_b_tile.append((u_packs0, u_packs1))
                return (gate_b_tile, up_b_tile)
            _scale_lane_elem = lane_div_16 * layout_b_scale.stride_klane + lane_mod_16
            _gate_scale_bases = []
            _up_scale_bases = []
            for _ni in range_constexpr(num_acc_n_packed):
                _col_base = by_n + n_tile_base + arith.constant(_ni * 16 * pack_N, index=True)
                _gate_mni = (expert_off_idx + _col_base) // arith.constant(32, index=True)
                _gate_scale_bases.append(_gate_mni * layout_b_scale.stride_n0 + _scale_lane_elem)
                _up_mni = (expert_off_idx + inter_idx + _col_base) // arith.constant(32, index=True)
                _up_scale_bases.append(_up_mni * layout_b_scale.stride_n0 + _scale_lane_elem)
            _a_scale_bases = []
            for _mi in range_constexpr(m_repeat_packed):
                _a_mni = _mi + bx_m // scale_mn_pack // 16
                _a_scale_bases.append(_a_mni * layout_a_scale.stride_n0 + _scale_lane_elem)
            _c16_idx = arith.constant(16, index=True)
            _c2_idx = arith.constant(2, index=True)
            _scale_mask_lo = arith.constant(255, type=T.i32)
            _m_half_idx = arith.constant(0, type=T.i32)
            _m_half_i32 = arith.constant(0, type=T.i32)
            _scale_shift = arith.constant(0, type=T.i32)
            _scale_shift_hi = arith.constant(0, type=T.i32)
            _n_half_idx = arith.constant(0, type=T.i32)
            _n_half_i32 = arith.constant(0, type=T.i32)
            _bscale_shift = arith.constant(0, type=T.i32)
            _bscale_shift_hi = arith.constant(0, type=T.i32)
            if const_expr(pack_M < scale_mn_pack):
                _m_half_idx = bx_m // _c16_idx % _c2_idx
                _m_half_i32 = arith.index_cast(T.i32, _m_half_idx)
                _scale_shift = _m_half_i32 * arith.constant(8, type=T.i32)
                _scale_shift_hi = _scale_shift + arith.constant(16, type=T.i32)
            if const_expr(pack_N < scale_mn_pack):
                _n_half_idx = n_tile_base // _c16_idx % _c2_idx
                _n_half_i32 = arith.index_cast(T.i32, _n_half_idx)
                _bscale_shift = _n_half_i32 * arith.constant(8, type=T.i32)
                _bscale_shift_hi = _bscale_shift + arith.constant(16, type=T.i32)

            def _rearrange_a_scale(raw_i32):
                """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
                if const_expr(pack_M >= scale_mn_pack):
                    return raw_i32
                b_k0 = arith.andi(arith.shrui(raw_i32, _scale_shift), _scale_mask_lo)
                b_k1 = arith.andi(arith.shrui(raw_i32, _scale_shift_hi), _scale_mask_lo)
                return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

            def _rearrange_b_scale(raw_i32):
                """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                if const_expr(pack_N >= scale_mn_pack):
                    return raw_i32
                b_k0 = arith.andi(arith.shrui(raw_i32, _bscale_shift), _scale_mask_lo)
                b_k1 = arith.andi(arith.shrui(raw_i32, _bscale_shift_hi), _scale_mask_lo)
                return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

            def prefetch_ab_scale_tile(base_k, ku_packed_limit=k_unroll_packed):
                a_scale_tile = []
                gate_b_scale = []
                up_b_scale = []
                for ku in range_constexpr(ku_packed_limit):
                    k_off = (ku + base_k) * layout_b_scale.stride_k0
                    for mi in range_constexpr(m_repeat_packed):
                        s = buffer_ops.buffer_load(sx_rsrc, _a_scale_bases[mi] + k_off, vec_width=1, dtype=T.i32, cache_modifier=0)
                        s = _rearrange_a_scale(s)
                        a_scale_tile.append(vector.from_elements(T.vec(1, T.i32), [s]))
                    for ni in range_constexpr(num_acc_n_packed):
                        gs = buffer_ops.buffer_load(sw_rsrc, _gate_scale_bases[ni] + k_off, vec_width=1, dtype=T.i32, cache_modifier=0)
                        gs = _rearrange_b_scale(gs)
                        gate_b_scale.append(vector.from_elements(T.vec(1, T.i32), [gs]))
                        us = buffer_ops.buffer_load(sw_rsrc, _up_scale_bases[ni] + k_off, vec_width=1, dtype=T.i32, cache_modifier=0)
                        us = _rearrange_b_scale(us)
                        up_b_scale.append(vector.from_elements(T.vec(1, T.i32), [us]))
                return [a_scale_tile, gate_b_scale, up_b_scale]
            _dma_bytes = 16
            _wave_size = 64
            _eff_bytes_per_buffer = int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
            _num_dma_loads = max(1, _eff_bytes_per_buffer // (total_threads * _dma_bytes))

            def dma_x_tile_to_lds(base_k, lds_buffer): # A g2s 
                c4_idx = arith.index(4)
                base_k_div4 = base_k / c_a_pack * arith.constant(int(elem_bytes), index=True) / arith.index(4)
                lds_ptr_i64 = None
                for i in range_constexpr(_num_dma_loads):
                    row_local_i = x_row_local[i]
                    col_local_i32_i = x_col_local_i32[i]
                    col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
                    row_k_dw = x_row_base_div4[i] + base_k_div4
                    global_byte_idx = row_k_dw * c4_idx + col_local_sw
                    global_offset = arith.index_cast(T.i32, global_byte_idx)
                    if const_expr(i == 0):
                        lds_addr = memref.extract_aligned_pointer_as_index(lds_buffer) + wave_id * arith.constant(_wave_size * _dma_bytes, index=True)
                        lds_ptr_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                    else:
                        lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * _dma_bytes, type=T.i64)
                    lds_ptr_type = ir.Type.parse('!llvm.ptr<3>')
                    lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)
                    rocdl.raw_ptr_buffer_load_lds(x_rsrc, lds_ptr, arith.constant(_dma_bytes, type=T.i32), global_offset, arith.constant(0, type=T.i32), arith.constant(0, type=T.i32), arith.constant(0, type=T.i32)) # 关键行. g2s loads了多少.

            def prefetch_x_to_lds(base_k, lds_buffer):
                dma_x_tile_to_lds(base_k, lds_buffer)

            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer): # A s2r
                col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = col_base_swz_bytes if elem_bytes == 1 else col_base_swz_bytes / arith.index(2)
                idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16]) # float4 load.
                a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return (a0, a1)

            def prefetch_full_a_from_lds(lds_buffer, ku_limit=k_unroll): # 啥时候用？ for之前？
                """Load entire A tile from LDS into registers before compute."""
                a_regs = []
                for k_idx in range_constexpr(ku_limit):
                    col_base = col_offset_base + k_idx * 128 // a_elem_vec_pack
                    for mi_idx in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi_idx * 16, index=True)
                        curr_row = row_a_lds + mi_val
                        a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                        a_regs.append((a0, a1))
                return a_regs

            def compute_tile(acc_gate_in, acc_up_in, gate_b_tile_in, up_b_tile_in, a_tile_regs, a_scale=None, gate_b_scale=None, up_b_scale=None, *, prefetch_epilogue=False, ku_count=k_unroll):
                gate_list = list(acc_gate_in)
                up_list = list(acc_up_in)
                mfma_res_ty = vec4_f32
                epilogue_pf = None
                c0_i64 = arith.constant(0, type=T.i64)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                    return vector.bitcast(vec8_i32, v4)
                _eff_packed = (ku_count + pack_K - 1) // pack_K
                for ku128 in range_constexpr(_eff_packed):
                    for ni in range_constexpr(num_acc_n_packed):
                        gate_bs_i32 = gate_b_scale[ku128 * num_acc_n_packed + ni]
                        gate_bs_val = vector.extract(gate_bs_i32, static_position=[0], dynamic_position=[])
                        up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                        up_bs_val = vector.extract(up_bs_i32, static_position=[0], dynamic_position=[])
                        for ikxdl in range_constexpr(pack_K):
                            k_idx = ku128 * pack_K + ikxdl
                            if const_expr(k_idx < ku_count):
                                gate_bp0, gate_bp1 = gate_b_tile_in[k_idx]
                                up_bp0, up_bp1 = up_b_tile_in[k_idx]
                                for inxdl in range_constexpr(pack_N):
                                    ni_idx = ni * pack_N + inxdl
                                    gb0 = gate_bp0[ni_idx]
                                    gb1 = gate_bp1[ni_idx]
                                    gb128 = pack_i64x4_to_i32x8(gb0, gb1, c0_i64, c0_i64)
                                    ub0 = up_bp0[ni_idx]
                                    ub1 = up_bp1[ni_idx]
                                    ub128 = pack_i64x4_to_i32x8(ub0, ub1, c0_i64, c0_i64)
                                    for mi in range_constexpr(m_repeat_packed):
                                        a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                                        a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])
                                        for imxdl in range_constexpr(pack_M):
                                            mi_idx = mi * pack_M + imxdl
                                            _a_reg_idx = k_idx * m_repeat + mi_idx
                                            a0, a1 = a_tile_regs[_a_reg_idx]
                                            a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(mfma_res_ty, [a128, gb128, gate_list[acc_idx], cbsz, blgp, ikxdl * pack_M + imxdl, a_scale_val, ikxdl * pack_N + inxdl, gate_bs_val])
                                            up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(mfma_res_ty, [a128, ub128, up_list[acc_idx], cbsz, blgp, ikxdl * pack_M + imxdl, a_scale_val, ikxdl * pack_N + inxdl, up_bs_val])
                return (gate_list, up_list, epilogue_pf)

            def load_a_subtile(k_idx, mi_idx, lds_buffer): 
                """Load a single A sub-tile from LDS (one ds_read)."""
                col_base = col_offset_base + k_idx * 128 // a_elem_vec_pack
                mi_val = arith.constant(mi_idx * 16, index=True)
                curr_row = row_a_lds + mi_val
                a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                return (a0, a1)

            def compute_bmajor_mfma_phase(all_a_tiles, gate_b_single, up_b_single, a_scale_vals, gate_bs_val, up_bs_val, gate_list, up_list, k_idx, ni_idx, ikxdl, inxdl):
                """B-major MFMA: fix one B (ni), cycle all A tiles (mi).

                    Packs B once and reuses across all mi iterations.
                    A tiles come from LDS (already available, no VMEM wait).

                    all_a_tiles: flat list indexed by [k*m_repeat + mi].
                    gate_b_single/up_b_single: (b0, b1) for one specific ni.
                    a_scale_vals: list of A scale scalars indexed by mi_packed.
                    """
                c0_i64 = arith.constant(0, type=T.i64)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def _pack(x0, x1, x2, x3):
                    v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                    return vector.bitcast(vec8_i32, v4)
                mfma_res_ty = vec4_f32
                gb128 = _pack(gate_b_single[0], gate_b_single[1], c0_i64, c0_i64)
                ub128 = _pack(up_b_single[0], up_b_single[1], c0_i64, c0_i64)
                for mi_p in range_constexpr(m_repeat_packed):
                    a_scale_val = a_scale_vals[mi_p]
                    for imxdl in range_constexpr(pack_M):
                        mi_idx = mi_p * pack_M + imxdl
                        a_reg = all_a_tiles[k_idx * m_repeat + mi_idx]
                        a128 = _pack(a_reg[0], a_reg[1], c0_i64, c0_i64)
                        acc_idx = mi_idx * num_acc_n + ni_idx
                        gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(mfma_res_ty, [a128, gb128, gate_list[acc_idx], cbsz, blgp, ikxdl * pack_M + imxdl, a_scale_val, ikxdl * pack_N + inxdl, gate_bs_val])
                        up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(mfma_res_ty, [a128, ub128, up_list[acc_idx], cbsz, blgp, ikxdl * pack_M + imxdl, a_scale_val, ikxdl * pack_N + inxdl, up_bs_val])

            def _interleaved_half(lds_read, lds_write, next_k_dma_py, next_k_load, prev_a_tile, prev_gate_w, prev_up_w, prev_a_scale, prev_gate_bs, prev_up_bs, acc_gate, acc_up): # 核心调度.
                """One flatmm-style interleaved half-iteration (deep pipeline).

                    Generalized for arbitrary m_repeat (block_m=32, 64, ...).
                    DMA targets lds_write (OTHER buffer) while ds_read uses
                    lds_read (already DMA'd in previous half).

                    Interleaving schedule (per half):
                      Phase 0: scale VMEM + 2 ds_read(A) -> 4 MFMA(prev)
                      Phase 1..N: B VMEM(distributed) + 2 ds_read(A, if avail) -> 4 MFMA(prev)
                      Phase N+1..: remaining B VMEM -> 4 MFMA(prev)
                    """
                _abs_k = k_base_idx + arith.constant(next_k_load, index=True)
                _bk = _abs_k // arith.constant(2, index=True)
                _sk = _abs_k // arith.constant(pack_K * 128, index=True)
                _k_off = _sk * layout_b_scale.stride_k0
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_vmcnt_before_barrier)
                _barrier()
                rocdl.sched_barrier(0)
                _abs_k_dma = k_base_idx + arith.constant(next_k_dma_py, index=True)
                if const_expr(use_async_copy and next_k_dma_py < int(_k_dim)):
                    prefetch_x_to_lds(_abs_k_dma, lds_write)
                _prev_asvs = []
                for _mi_p in range_constexpr(m_repeat_packed):
                    _prev_asvs.append(vector.extract(prev_a_scale[_mi_p], static_position=[0], dynamic_position=[]))
                _prev_gsv_list = []
                for _gs_ni in range_constexpr(num_acc_n_packed):
                    _prev_gsv_list.append(vector.extract(prev_gate_bs[_gs_ni], static_position=[0], dynamic_position=[]))
                _prev_usv_list = []
                for _us_ni in range_constexpr(num_acc_n_packed):
                    _prev_usv_list.append(vector.extract(prev_up_bs[_us_ni], static_position=[0], dynamic_position=[]))
                _a_all = {}
                _b_gate_all = {}
                _b_up_all = {}
                for _p in range_constexpr(_pipe_n_phases):
                    if const_expr(_pp_has_scale[_p]):
                        _new_as_list = []
                        for _mi_p in range_constexpr(m_repeat_packed):
                            _raw_as = buffer_ops.buffer_load(sx_rsrc, _a_scale_bases[_mi_p] + _k_off, vec_width=1, dtype=T.i32, cache_modifier=0)
                            _new_as_list.append(_rearrange_a_scale(_raw_as))
                        _new_gs_list = []
                        for _gs_ni in range_constexpr(num_acc_n_packed):
                            _gs_raw = buffer_ops.buffer_load(sw_rsrc, _gate_scale_bases[_gs_ni] + _k_off, vec_width=1, dtype=T.i32, cache_modifier=0)
                            _new_gs_list.append(_rearrange_b_scale(_gs_raw))
                        _new_us_list = []
                        for _us_ni in range_constexpr(num_acc_n_packed):
                            _us_raw = buffer_ops.buffer_load(sw_rsrc, _up_scale_bases[_us_ni] + _k_off, vec_width=1, dtype=T.i32, cache_modifier=0)
                            _new_us_list.append(_rearrange_b_scale(_us_raw))
                    for _b_j in range_constexpr(len(_pp_b_loads[_p])):
                        _b_type, _b_ku, _b_ni = _pp_b_loads[_p][_b_j]
                        if const_expr(_b_type == 'gate'):
                            _b_gate_all[_b_ku, _b_ni] = load_b_packs_k64(_bk, _b_ku, gate_n_blk_list[_b_ni], gate_n_intra_list[_b_ni])
                        else:
                            _b_up_all[_b_ku, _b_ni] = load_b_packs_k64(_bk, _b_ku, up_n_blk_list[_b_ni], up_n_intra_list[_b_ni])
                    rocdl.sched_barrier(0)
                    for _a_j in range_constexpr(len(_pp_a_reads[_p])):
                        _ak, _ami = _pp_a_reads[_p][_a_j]
                        _a_all[_ak, _ami] = load_a_subtile(_ak, _ami, lds_read)
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1) # 4个warp，还用设置这个么... 好像也没有interleave是吧.
                    for _m_j in range_constexpr(len(_pp_mfma[_p])):
                        _k_idx, _ni_idx, _ikxdl, _inxdl, _ku128 = _pp_mfma[_p][_m_j]
                        _ni_packed_idx = _ni_idx // pack_N
                        _up_b_single = (prev_up_w[_k_idx][0][_ni_idx], prev_up_w[_k_idx][1][_ni_idx])
                        compute_bmajor_mfma_phase(prev_a_tile, (prev_gate_w[_k_idx][0][_ni_idx], prev_gate_w[_k_idx][1][_ni_idx]), _up_b_single, _prev_asvs, _prev_gsv_list[_ni_packed_idx], _prev_usv_list[_ni_packed_idx], acc_gate, acc_up, _k_idx, _ni_idx, _ikxdl, _inxdl) # MFMA. 输出到 gate_list 和 up_list 里面来.
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
                cur_a_tile = []
                for _k in range_constexpr(k_unroll):
                    for _mi in range_constexpr(m_repeat):
                        cur_a_tile.append(_a_all[_k, _mi])
                cur_gate_w = []
                cur_up_w = []
                for ku in range_constexpr(k_unroll):
                    g_packs0, g_packs1 = ([], [])
                    u_packs0, u_packs1 = ([], [])
                    for ni in range_constexpr(num_acc_n):
                        g = _b_gate_all[ku, ni]
                        g_packs0.append(g[0])
                        g_packs1.append(g[1])
                        u = _b_up_all[ku, ni]
                        u_packs0.append(u[0])
                        u_packs1.append(u[1])
                    cur_gate_w.append((g_packs0, g_packs1))
                    cur_up_w.append((u_packs0, u_packs1))
                cur_a_scale = []
                for _mi_p in range_constexpr(m_repeat_packed):
                    cur_a_scale.append(vector.from_elements(T.vec(1, T.i32), [_new_as_list[_mi_p]]))
                cur_gate_bs = []
                for _gs_ni in range_constexpr(num_acc_n_packed):
                    cur_gate_bs.append(vector.from_elements(T.vec(1, T.i32), [_new_gs_list[_gs_ni]]))
                cur_up_bs = []
                for _us_ni in range_constexpr(num_acc_n_packed):
                    cur_up_bs.append(vector.from_elements(T.vec(1, T.i32), [_new_us_list[_us_ni]]))
                return (cur_a_tile, cur_gate_w, cur_up_w, cur_a_scale, cur_gate_bs, cur_up_bs, acc_gate, acc_up)
            rocdl.sched_barrier(0)
            k0 = k_base_idx
            prefetch_x_to_lds(k0, lds_x_pong) # TODO(zty) K0是什么意思？ pong? 应该是说的 for开始前吧.
            rocdl.sched_barrier(0)
            _k0_scale = k_base_idx // arith.constant(pack_K * 128, index=True)
            a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(_k0_scale)
            _c_tile_m_idx = arith.constant(tile_m, index=True)
            _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
            _if_tid = scf.IfOp(_tid_in_range)
            with ir.InsertionPoint(_if_tid.then_block):
                _tid_row = bx_m + tx
                _tid_val = buffer_ops.buffer_load(sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32)
                _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                vector.store(_tid_vec1, lds_tid, [tx])
                scf.YieldOp([])
            acc_gate = [acc_init] * num_acc_n * m_repeat
            acc_up = [acc_init] * num_acc_n * m_repeat
            _k1 = k_base_idx + arith.constant(tile_k, index=True)
            rocdl.sched_barrier(0)
            prefetch_x_to_lds(_k1, lds_x_ping) # TODO(zty) K1是什么意思？ ping? 应该是说的 for开始前吧.
            _k0_b = k_base_idx // arith.constant(2, index=True)
            gate_w0, up_w0 = load_b_tile(_k0_b)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            rocdl.sched_barrier(0)
            a_tile_pong = prefetch_full_a_from_lds(lds_x_pong)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(6)
            num_k_tiles_py = int(_k_dim) // int(tile_k)
            odd_k_tiles = num_k_tiles_py % 2 == 1
            tail_tiles = 1 if odd_k_tiles else 2
            k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
            if const_expr(k_main2_py < 0):
                k_main2_py = 0
            gate_w_pong = gate_w0
            up_w_pong = up_w0
            rocdl.sched_barrier(0)
            if const_expr(k_main2_py > 0):
                for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                    next_k_load_1 = k_iv_py + tile_k
                    next_k_load_2 = k_iv_py + tile_k * 2
                    next_k_dma_1 = k_iv_py + tile_k * 2
                    next_k_dma_2 = k_iv_py + tile_k * 3
                    a_tile_ping, gate_w_ping, up_w_ping, a_scale_ping, gate_bs_ping, up_bs_ping, acc_gate, acc_up = _interleaved_half(
                        lds_x_ping,
                        lds_x_pong,
                        next_k_dma_1,
                        next_k_load_1,
                        a_tile_pong,
                        gate_w_pong,
                        up_w_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                        acc_gate,
                        acc_up,
                    )
                    a_tile_pong, gate_w_pong, up_w_pong, a_scale_pong, gate_bs_pong, up_bs_pong, acc_gate, acc_up = _interleaved_half(
                        lds_x_pong,
                        lds_x_ping,
                        next_k_dma_2,
                        next_k_load_2,
                        a_tile_ping,
                        gate_w_ping,
                        up_w_ping,
                        a_scale_ping,
                        gate_bs_ping,
                        up_bs_ping,
                        acc_gate,
                        acc_up,
                    )
            if const_expr(odd_k_tiles):
                acc_gate, acc_up, epilogue_pf = compute_tile(acc_gate, acc_up, gate_w_pong, up_w_pong, a_tile_pong, a_scale_pong, gate_bs_pong, up_bs_pong, prefetch_epilogue=True, ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll)
            else:
                _k_tail_rel = arith.constant(_k_dim - tile_k, index=True)
                k_tail1 = k_base_idx + _k_tail_rel
                x_regs_ping = []
                prefetch_x_to_lds(k_tail1, lds_x_ping)
                if const_expr(_pad_ku_skip > 0):
                    gate_w_ping, up_w_ping = load_b_tile(k_tail1 // arith.constant(2, index=True), ku_limit=_tail_ku)
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(k_tail1 // arith.constant(pack_K * 128, index=True), ku_packed_limit=_tail_ku_packed)
                else:
                    gate_w_ping, up_w_ping = load_b_tile(k_tail1 // arith.constant(2, index=True))
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(k_tail1 // arith.constant(pack_K * 128, index=True))
                acc_gate, acc_up, _ = compute_tile(acc_gate, acc_up, gate_w_pong, up_w_pong, a_tile_pong, a_scale_pong, gate_bs_pong, up_bs_pong)
                rocdl.s_waitcnt(0)
                _barrier()
                if const_expr(_pad_ku_skip > 0):
                    a_tile_ping = prefetch_full_a_from_lds(lds_x_ping, ku_limit=_tail_ku)
                else:
                    a_tile_ping = prefetch_full_a_from_lds(lds_x_ping)
                acc_gate, acc_up, epilogue_pf = compute_tile(acc_gate, acc_up, gate_w_ping, up_w_ping, a_tile_ping, a_scale_ping, gate_bs_ping, up_bs_ping, prefetch_epilogue=True, ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll)
            def _silu_elem(g):
                """silu(x) = x * sigmoid(x); HW fast path: exp2, rcp"""
                neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                t = g * neg_log2e
                emu = llvm.call_intrinsic(f32, 'llvm.amdgcn.exp2.f32', [t], [], [])
                one = arith.constant(1.0, type=f32)
                den = one + emu
                sig = llvm.call_intrinsic(f32, 'llvm.amdgcn.rcp.f32', [den], [], [])
                return g * sig

            def _silu_mul_vec4(gate_v4, up_v4):
                """Element-wise silu(gate) * up on vec4_f32."""
                result_elems = []
                for ei in range_constexpr(4):
                    g = vector.extract(gate_v4, static_position=[ei], dynamic_position=[])
                    u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
                    result_elems.append(_silu_elem(g) * u)
                return vector.from_elements(vec4_f32, result_elems)

            def _act_vec4(gate_v4, up_v4):
                return _silu_mul_vec4(gate_v4, up_v4)
            acc = [None] * (int(num_acc_n) * int(m_repeat))
            for _mi in range_constexpr(m_repeat):
                for _ni in range_constexpr(num_acc_n):
                    _aidx = _mi * num_acc_n + _ni
                    acc[_aidx] = _act_vec4(acc_gate[_aidx], acc_up[_aidx])
            mask24_i32 = arith.constant(16777215)
            topk_i32_v = topk_i32
            tokens_i32_v = tokens_i32
            out_base_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg_out))
            out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)
            if const_expr(lds_out is None):
                raise RuntimeError('CShuffle epilogue requires lds_out')
            def write_row_to_lds(*, mi: int, ii: int, row_in_tile, row, row_base_lds, col_base_local, num_acc_n: int, lds_out):
                for ni in range_constexpr(num_acc_n):
                    col_local = col_base_local + ni * 16
                    acc_idx = mi * num_acc_n + ni
                    v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                    v_out = arith.trunc_f(out_elem(), v)
                    lds_idx = row_base_lds + col_local
                    vec1_out = T.vec(1, out_elem())
                    v1 = vector.from_elements(vec1_out, [v_out])
                    vector.store(v1, lds_out, [lds_idx], alignment=2)
            _out_row_stride = inter_dim * out_elem_bytes

            def precompute_row(*, row_local, row):
                fused2 = memref.load(lds_tid, [row_local])
                row_i32 = arith.index_cast(T.i32, row)
                row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                t = fused2 & mask24_i32
                s = fused2 >> 24
                t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32_v)
                s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                t_idx = arith.index_cast(ir.IndexType.get(), t)
                s_idx = arith.index_cast(ir.IndexType.get(), s)
                ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                row_byte_base = out_base_idx + ts_idx * arith.constant(_out_row_stride, index=True)
                return ((fused2, row_byte_base), row_valid)

            def _idx_to_llvm_ptr(idx_val, addr_space=1):
                idx_v = idx_val._value if hasattr(idx_val, '_value') else idx_val
                i64_v = arith.index_cast(T.i64, idx_v)
                i64_raw = i64_v._value if hasattr(i64_v, '_value') else i64_v
                ptr_ty = ir.Type.parse(f'!llvm.ptr<{addr_space}>')
                return llvm.inttoptr(ptr_ty, i64_raw)
            _e_vec = _e_vec_s1
            _cshuffle_nlane = min(32, tile_n // _e_vec)

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                fused, row_byte_base = row_ctx
                col_idx = col_g0
                byte_off_col = col_idx * arith.constant(out_elem_bytes, index=True)
                ptr_addr_idx = row_byte_base + byte_off_col
                out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                frag_v = frag._value if hasattr(frag, '_value') else frag
                llvm.StoreOp(frag_v, out_ptr_v, alignment=_e_vec * out_elem_bytes, nontemporal=True)
            _frag_elem = ir.BF16Type.get()
            c_shuffle_epilog(arith=arith, vector=vector, gpu=gpu, scf=scf, range_constexpr=range_constexpr, tile_m=tile_m, tile_n=tile_n, e_vec=_e_vec, cshuffle_nlane=_cshuffle_nlane, block_size=total_threads, m_repeat=m_repeat, num_acc_n=num_acc_n, tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16, bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base, lds_out=lds_out, frag_elem_type=_frag_elem, write_row_to_lds=write_row_to_lds, precompute_row=precompute_row, store_pair=store_pair, lds_out_split=lds_out_B)
        _if_blk = scf.IfOp(blk_valid)
        with ir.InsertionPoint(_if_blk.then_block):
            _ifexpert_of = scf.IfOp(exp_valid)
            with ir.InsertionPoint(_ifexpert_of.then_block):
                _moe_gemm1_body()
                scf.YieldOp([])
            scf.YieldOp([])
        gpu.barrier()
        scf.YieldOp([])
        _for_ip.__exit__(None, None, None)
    _cache_tag = (module_name, tile_m, tile_n, tile_k, model_dim_pad, inter_dim_pad, persist_m, use_async_copy, waves_per_eu, k_batch, xcd_swizzle)

    @flyc.jit
    def launch_kimi_fp4_stage1_16384(arg_out: fx.Pointer, arg_x: fx.Pointer, arg_w: fx.Pointer, arg_scale_x: fx.Pointer, arg_scale_w: fx.Pointer, arg_sorted_token_ids: fx.Pointer, arg_expert_ids: fx.Pointer, arg_max_token_ids: fx.Pointer, i32_tokens_in: fx.Int32, i32_inter_in: fx.Int32, i32_k_in: fx.Int32, i32_size_expert_ids_in: fx.Int32, stream: fx.Stream):
        _ = _cache_tag # 让函数和cache_tag绑定，避免重复编译
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body): # 前面算好了怎么切，这个真正挂到kernel里面.
            allocator_pong.finalize()
            allocator_ping.finalize()
        inter_dim_pad_total = arith.constant(2 * inter_dim_pad, index=True) # 0 如果存在，*2, gemm1 
        tile_k_stage2 = tile_k // 2 # 128 = 256 // 2
        tile2_pad = (tile_k_stage2 - (inter_dim - inter_dim_pad) % tile_k_stage2) % tile_k_stage2 # 0 = div_up(pad, tile_k_stage2)
        inter_in = arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value()) # 1024
        tile_n_index = arith.constant(tile_n, index=True) # 128
        gx = (inter_in - inter_dim_pad_total + tile2_pad + 2 * tile_n_index - 1) / tile_n_index / arith.constant(2, index=True) # 4 = div_up(1024, 2*128) # 成对处理tile_n
        _c_pm_l = arith.constant(persist_m, index=True) # 这里是1.
        gy = (arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value()) + _c_pm_l - arith.constant(1, index=True)) / _c_pm_l # div_up(expert_ids_in, 1)
        moe_gemm1(arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids, arg_expert_ids, arg_max_token_ids, i32_tokens_in, i32_inter_in, i32_k_in, i32_size_expert_ids_in).launch(grid=(gx, gy, k_batch), block=(total_threads, 1, 1), stream=stream)
    return launch_kimi_fp4_stage1_16384

@functools.lru_cache(maxsize=1)
def compile_kimi_fp4_stage2_16384():
    """Compile the fixed Kimi fp4 stage2 kernel selected for M=16384."""
    model_dim = MODEL_DIM
    inter_dim = INTER_DIM
    experts = EXPERTS
    topk = TOPK
    tile_m = S2.tile_m
    tile_n = S2.tile_n
    tile_k = S2.tile_k
    model_dim_pad = 0
    inter_dim_pad = 0
    persist_m = S2.persist_m
    b_nt = S2.b_nt
    xcd_swizzle = S2.xcd_swizzle
    _sort_block_m = tile_m
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name='smem0')
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name='smem1')
    _scale_pack_m = 2
    _scale_pack_n = 2
    _scale_pack_k = 2
    pack_M = min(_scale_pack_m, tile_m // 16)
    pack_N = min(_scale_pack_n, tile_n // 64)
    _k_unroll_raw = int(tile_k) // 128
    pack_K = min(_scale_pack_k, _k_unroll_raw)
    elem_bytes = 1
    a_elem_bytes = 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)
    a_elem_vec_pack = 2
    cbsz = 4
    blgp = 4
    _b_kpack_bytes_s = 16
    _b_kpack_elems_s = _b_kpack_bytes_s // b_elem_bytes
    _b_c_k_s = inter_dim // _scale_pack_k
    _b_c_k0_s = _b_c_k_s * b_elem_bytes // 64
    _b_stride_nlane = _b_kpack_elems_s
    _b_stride_klane = 16 * _b_stride_nlane
    _b_stride_k0 = 4 * _b_stride_klane
    _b_stride_n0 = _b_c_k0_s * _b_stride_k0
    assert model_dim % 16 == 0, 'model_dim must be divisible by 16'
    _expert_b_stride = model_dim // 16 * _b_stride_n0
    if tile_k_bytes % 64 != 0:
        raise ValueError(f'tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} (tile_k={tile_k}, elem_bytes={a_elem_bytes})')
    out_s = 'bf16'
    w_elem_bytes = 1
    w_elem_pack = 2
    w_nbytes = experts * model_dim * inter_dim * w_elem_bytes // w_elem_pack

    def _x_elem_type():
        return T.i8

    def _w_elem_type():
        return T.i8

    def _scale_elem_type():
        return T.i32
    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(f'tile_m*tile_k*elem_bytes must be divisible by {total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}')
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    _use_lds128 = os.environ.get('FLIR_CK_LDS128', '1') in ('1', 'true', 'True', 'YES', 'yes')
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k
    if a_elem_vec_pack > 1:
        _eff_lds_stride = lds_stride // a_elem_vec_pack
        _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack
    else:
        _eff_lds_stride = lds_stride
        _eff_tile_k_bytes = tile_k_bytes
    _use_cshuffle_epilog = True

    def out_elem():
        return T.bf16
    epilog_tag = 'cshuffle'
    _persistent = persist_m <= 0
    if _persistent:
        from aiter.jit.utils.chip_info import get_cu_num
        _cu_num = get_cu_num()
    else:
        _cu_num = 0
    _pm_tag = f'_persist_cu{_cu_num}' if _persistent else f'_pm{persist_m}'
    _xcd_tag = f'_xcd{xcd_swizzle}' if xcd_swizzle > 0 else ''
    module_name = f'mfma_moe2_afp4_wfp4_{out_s}_{epilog_tag}_t{tile_m}x{tile_n}x{tile_k}_vscale_fix3{_pm_tag}{_xcd_tag}'
    _single_x_bytes = int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
    _cshuffle_elem_bytes_s2 = 2
    lds_out_bytes = _cshuffle_elem_bytes_s2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    _input_elems = _single_x_bytes if a_elem_bytes == 1 else _single_x_bytes // 2
    _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
    _ping_buffer_bytes = _single_x_bytes

    def x_lds_elem():
        return T.f8
    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes
    _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes
    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    @flyc.kernel(name=module_name)
    def moe_gemm2(arg_out: fx.Pointer, arg_x: fx.Pointer, arg_w: fx.Pointer, arg_scale_x: fx.Pointer, arg_scale_w: fx.Pointer, arg_sorted_token_ids: fx.Pointer, arg_expert_ids: fx.Pointer, arg_sorted_weights: fx.Pointer, arg_num_valid_ids: fx.Pointer, i32_tokens_in: fx.Int32, i32_n_in: fx.Int32, i32_k_in: fx.Int32, i32_size_expert_ids_in: fx.Int32):
        tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
        size_expert_ids_in = arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
        x_elem = T.f8
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        vec4_f32 = T.vec(4, f32)
        vec4_i32 = T.vec(4, i32)
        vec16_elems = 16 if a_elem_bytes == 1 else 8
        vec16_x = T.vec(vec16_elems, x_elem)
        vec2_i64 = T.vec(2, i64)

        def _ptr_buffer_resource(ptr, num_records_bytes):
            addr = fx.ptrtoint(ptr)
            addr_i64 = arith.index_cast(T.i64, addr)
            return buffer_ops.create_buffer_resource_from_addr(addr_i64, num_records_bytes=num_records_bytes)
        acc_init = arith.constant_vector(0.0, vec4_f32)
        topk_idx = arith.constant(topk, index=True)
        m_in = tokens_in * topk_idx
        c_n_total = arith.constant(experts * model_dim, index=True)
        kpack_bytes = 16
        from aiter.ops.flydsl.kernels.layout_utils import _div_pow2, _mod_pow2

        def check_c_n_valid_gate(base_n):
            return arith.cmpi(CmpIPredicate.ult, base_n, model_dim - model_dim_pad)

        def check_c_k_valid_gate(base_k):
            return arith.cmpi(CmpIPredicate.ult, base_k, inter_dim - inter_dim_pad)
        c_k_orig = arith.constant(inter_dim, index=True)
        layout_a_scale = make_preshuffle_scale_layout(arith, c_mn=m_in, c_k=c_k_orig)
        layout_b_scale = make_preshuffle_scale_layout(arith, c_mn=c_n_total, c_k=c_k_orig)
        shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
        stride_lds = fx.make_stride(_eff_lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)
        tx = gpu.thread_id('x')
        by = gpu.block_id('x')
        bx_persist = gpu.block_id('y')
        if const_expr(xcd_swizzle > 0):
            _NUM_XCDS_S = 8
            _c1_sw = arith.constant(1, index=True)
            _c_tn_sw = arith.constant(tile_n, index=True)
            _c_mdp_sw = arith.constant(model_dim_pad, index=True)
            _gx = (n_in - _c_mdp_sw + _c_tn_sw - _c1_sw) / _c_tn_sw
            if const_expr(_persistent):
                _gy = arith.constant(_cu_num, index=True)
            else:
                _c_pm_sw = arith.constant(persist_m, index=True)
                _gy = (size_expert_ids_in + _c_pm_sw - _c1_sw) / _c_pm_sw
            _linear_id = bx_persist * _gx + by
            _num_wgs = _gx * _gy
            _c_xcds = arith.constant(_NUM_XCDS_S, index=True)
            _wgs_per_xcd = _num_wgs / _c_xcds
            _wgid = _linear_id % _c_xcds * _wgs_per_xcd + _linear_id / _c_xcds
            _WGM_S = xcd_swizzle
            _c_wgm = arith.constant(_WGM_S, index=True)
            _num_wgid_in_group = _c_wgm * _gx
            _group_id = _wgid / _num_wgid_in_group
            _first_pid_m = _group_id * _c_wgm
            _remaining_m = _gy - _first_pid_m
            _cmp_m = arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
            _group_size_m = arith.select(_cmp_m, _remaining_m, _c_wgm)
            _wgid_in_group = _wgid % _num_wgid_in_group
            bx_persist = _first_pid_m + _wgid_in_group % _group_size_m
            by = _wgid_in_group / _group_size_m
        k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_x_pong = SmemPtr(base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(_input_elems,)).get()
        lds_x_ping = SmemPtr(base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(_input_elems,)).get()
        lds_out = SmemPtr(base_ptr_pong, lds_pong_offset, T.bf16, shape=(tile_m * tile_n,)).get() if _use_cshuffle_epilog else None
        lds_tid = SmemPtr(base_ptr_pong, _lds_tid_offset_pong, T.i32, shape=(tile_m,)).get()
        c_topk = arith.constant(topk, index=True)
        c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)
        x_nbytes_idx = _div_pow2(tokens_in * c_topk * k_in * c_elem_bytes, int(a_elem_vec_pack))
        x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
        x_rsrc = _ptr_buffer_resource(arg_x, x_nbytes_i32)
        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)
        out_elem_bytes = 2
        out_nbytes_idx = tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
        out_nbytes_idx = tokens_in * arith.index(topk) * n_in * arith.constant(out_elem_bytes, index=True)
        out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
        out_rsrc = _ptr_buffer_resource(arg_out, out_nbytes_i32)
        numids_rsrc = _ptr_buffer_resource(arg_num_valid_ids, arith.constant(4, type=T.i32))
        num_valid_i32 = buffer_ops.buffer_load(numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32)
        num_valid_i32 = rocdl.ReadfirstlaneOp(T.i32, num_valid_i32).res
        num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)
        sx_rsrc = 1
        sw_rsrc = 1
        kblk = _div_pow2(k_in, 32)
        sx_nbytes_idx = num_valid_idx * kblk
        sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
        sx_rsrc = _ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)
        kblk_w = _div_pow2(k_in, 32)
        mn_w = arith.constant(experts * model_dim, index=True)
        sw_nbytes_idx = mn_w * kblk_w
        sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
        sw_rsrc = _ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)
        sorted_nbytes_idx = size_expert_ids_in * arith.constant(tile_m, index=True) * arith.constant(4, index=True)
        sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
        sorted_rsrc = _ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
        sorted_w_rsrc = _ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_i32)
        _c_sbm = arith.constant(_sort_block_m, index=True)
        _c_tm = arith.constant(tile_m, index=True)
        _c1 = arith.constant(1, index=True)
        _sort_blocks_ub = _div_pow2(size_expert_ids_in * _c_tm + _c_sbm - _c1, _sort_block_m)
        eid_nbytes_idx = _sort_blocks_ub * arith.constant(4, index=True)
        eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
        expert_rsrc = _ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
        _c0_p = arith.constant(0, index=True)
        _c1_p = arith.constant(1, index=True)
        if const_expr(_persistent):
            _c_cu = arith.constant(_cu_num, index=True)
            _c_tm_p = arith.constant(tile_m, index=True)
            _num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)
            _total_m_tiles = (_num_valid_idx + _c_tm_p - _c1_p) / _c_tm_p
            _tiles_per_block = (_total_m_tiles + _c_cu - _c1_p) / _c_cu
            _i1 = ir.IntegerType.get_signless(1)
            _init_active = arith.constant(1, type=_i1)
            _for_persist = scf.ForOp(_c0_p, _tiles_per_block, _c1_p, [_init_active])
        else:
            _c_pm = arith.constant(persist_m, index=True)
            _init_prev_expert = arith.constant(0, type=T.i32)
            _init_prev_b_base = arith.constant(0, index=True)
            _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p, [_init_prev_expert, _init_prev_b_base])
        _for_ip = ir.InsertionPoint(_for_persist.body)
        _for_ip.__enter__()
        _mi_p = _for_persist.induction_variable
        if const_expr(_persistent):
            _still_active = _for_persist.inner_iter_args[0]
            bx = bx_persist * _tiles_per_block + _mi_p
        else:
            _prev_expert_i32 = _for_persist.inner_iter_args[0]
            _prev_expert_b_base = _for_persist.inner_iter_args[1]
            bx = bx_persist * arith.constant(persist_m, index=True) + _mi_p
        bx_m = bx * arith.constant(tile_m, index=True)
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
        sort_blk = _div_pow2(bx_m, _sort_block_m)
        expert_i32 = buffer_ops.buffer_load(expert_rsrc, sort_blk, vec_width=1, dtype=T.i32)
        expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32))
        if const_expr(_persistent):
            _expert_b_base = expert_idx * arith.constant(_expert_b_stride, index=True)
        else:
            _delta_expert = arith.subi(expert_i32, _prev_expert_i32)
            _delta_expert_idx = arith.index_cast(ir.IndexType.get(), _delta_expert)
            _delta_b = _delta_expert_idx * arith.constant(_expert_b_stride, index=True)
            _expert_b_base = _prev_expert_b_base + _delta_b
        _first_tok = buffer_ops.buffer_load(sorted_rsrc, bx_m, vec_width=1, dtype=T.i32)
        _first_tid = arith.andi(_first_tok, arith.constant(16777215, type=T.i32))
        _tokens_i32_guard = arith.index_cast(T.i32, tokens_in)
        tile_has_tokens = arith.cmpi(CmpIPredicate.ult, _first_tid, _tokens_i32_guard)
        if const_expr(pack_M < _scale_pack_m):
            _m_off = _mod_pow2(_div_pow2(bx_m, 16), _scale_pack_m)
            _m_scale_shift_i32 = arith.index_cast(T.i32, _m_off * arith.constant(8, index=True))
        else:
            _m_scale_shift_i32 = None

        def _moe_gemm2_then_body():
            n_idx = arith.constant(model_dim, index=True)
            expert_off_idx = expert_idx * n_idx
            if const_expr(bytes_per_thread_x % 16 == 0):
                x_load_bytes = 16
            elif const_expr(bytes_per_thread_x % 8 == 0):
                x_load_bytes = 8
            elif const_expr(bytes_per_thread_x % 4 == 0):
                x_load_bytes = 4
            else:
                raise ValueError(f'bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping.')
            num_x_loads = bytes_per_thread_x // x_load_bytes
            chunk_i32 = x_load_bytes // 4
            vec4_i32 = T.vec(4, i32)
            c_k_div4 = _div_pow2(_div_pow2(k_in, int(a_elem_vec_pack)) * arith.constant(int(a_elem_bytes), index=True), 4)
            tile_k_dwords = int(tile_k) * int(a_elem_bytes) // (4 * int(a_elem_vec_pack))
            layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            c_chunk_i32 = arith.constant(chunk_i32, index=True)
            tx_i32_base = tx * c_chunk_i32
            topk_i32 = arith.constant(topk)
            mask24 = arith.constant(16777215)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(arith, tx_i32_base=tx_i32_base, i=i, total_threads=total_threads, layout_tile_div4=layout_x_tile_div4, chunk_i32=chunk_i32)
            vec1_i32 = T.vec(1, i32)
            vec2_i32 = T.vec(2, i32)
            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)
                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                t_i32 = arith.andi(fused_i, mask24)
                s_i32 = arith.shrui(fused_i, arith.constant(24))
                t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                ts_valid = arith.andi(t_valid, s_valid)
                t_safe = arith.select(ts_valid, t_i32, arith.constant(0))
                s_safe = arith.select(ts_valid, s_i32, arith.constant(0))
                row_ts_i32 = t_safe * topk_i32 + s_safe
                row_ts_idx = arith.index_cast(ir.IndexType.get(), row_ts_i32)
                x_row_base_div4.append(row_ts_idx * c_k_div4)
            coord_wl = idx2crd(tx, layout_tx_wave_lane)
            wave_id = layout_get(coord_wl, 0)
            lane_id = layout_get(coord_wl, 1)
            coord_l16 = idx2crd(lane_id, layout_lane16)
            lane_div_16 = layout_get(coord_l16, 0)
            lane_mod_16 = layout_get(coord_l16, 1)
            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * arith.constant(16, index=True)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = _mod_pow2(wave_id, 4)
            n_tile_base = wave_mod_4 * c_n_per_wave
            by_n = by * arith.constant(tile_n, index=True)
            if const_expr(pack_N < _scale_pack_n):
                _global_n_base = expert_off_idx + by_n + n_tile_base
                _n_off = _mod_pow2(_div_pow2(_global_n_base, 16), _scale_pack_n)
                _n_scale_shift_i32 = arith.index_cast(T.i32, _n_off * arith.constant(8, index=True))
            else:
                _n_scale_shift_i32 = None
            n_intra_list = [None] * num_acc_n
            n_blk_list = [None] * num_acc_n
            col_g_list = [None] * num_acc_n
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                col_g = by_n + n_tile_base
                col_g = _div_pow2(col_g, 2) + offset
                col_g = col_g + lane_mod_16
                col_g_list[i] = col_g
                c_offset = arith.constant(offset, index=True)
                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                n_blk_list[i] = _div_pow2(global_n, 16)
                n_intra_list[i] = _mod_pow2(global_n, 16)
            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 128
            k_unroll_packed = k_unroll // pack_K
            m_repeat_packed = m_repeat // pack_M
            num_acc_n_packed = num_acc_n // pack_N
            _K_per_ku_s2 = tile_k // k_unroll
            _pad_k_elems_s2 = inter_dim_pad % tile_k if inter_dim_pad > 0 else 0
            _pad_ku_skip_s2 = _pad_k_elems_s2 // _K_per_ku_s2
            _tail_ku_s2 = k_unroll - _pad_ku_skip_s2
            _tail_ku_packed_s2 = (_tail_ku_s2 + pack_K - 1) // pack_K if _pad_ku_skip_s2 > 0 else None

            def load_b_packs_k64(base_k, ku: int, ni: int):
                """Load one K64-byte B micro-step: single 16B load, split into 2x i64."""
                base_k_bytes = base_k * arith.constant(int(b_elem_bytes), index=True)
                k0_base = _div_pow2(base_k_bytes, 64)
                k0 = k0_base + arith.constant(ku, index=True)
                k1 = lane_div_16
                idx_pack = _expert_b_base + n_blk_list[ni] * arith.constant(_b_stride_n0, index=True) + k0 * arith.constant(_b_stride_k0, index=True) + k1 * arith.constant(_b_stride_klane, index=True) + n_intra_list[ni] * arith.constant(_b_stride_nlane, index=True)
                vec_elems = kpack_bytes // int(b_elem_bytes)
                b16 = _buffer_load_vec(buffer_ops, vector, w_rsrc, idx_pack, elem_type=_w_elem_type(), vec_elems=vec_elems, elem_bytes=b_elem_bytes, offset_in_bytes=b_elem_bytes == 1, cache_modifier=b_nt)
                b_i64x2 = vector.bitcast(vec2_i64, b16)
                b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                return (b0, b1)

            def load_b_tile(base_k, ku_limit=k_unroll):
                b_tile = []
                for ku in range_constexpr(ku_limit):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        b0, b1 = load_b_packs_k64(base_k, ku, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile
            _b_split_enabled = k_unroll >= 2
            _b_split_ku = k_unroll // 2 if _b_split_enabled else k_unroll

            def load_b_tile_lo(base_k):
                """Load first half of B tile (ku < _b_split_ku)."""
                b_tile = []
                for ku in range_constexpr(_b_split_ku):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        b0, b1 = load_b_packs_k64(base_k, ku, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            def load_b_tile_hi(base_k):
                """Load second half of B tile (ku >= _b_split_ku)."""
                b_tile = []
                for ku in range_constexpr(_b_split_ku, k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        b0, b1 = load_b_packs_k64(base_k, ku, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                k_lane = lane_div_16
                n_lane = lane_mod_16
                idx_pack = mni * scale_info.stride_n0 + ku * scale_info.stride_k0 + k_lane * scale_info.stride_klane + n_lane
                s = buffer_ops.buffer_load(rsrc, idx_pack, vec_width=1, dtype=T.i32)
                return vector.from_elements(T.vec(1, T.i32), [s])

            def _apply_k_shift(scale_vec, k_shift_bits):
                if const_expr(k_shift_bits > 0):
                    val = vector.extract(scale_vec, static_position=[0], dynamic_position=[])
                    val = arith.shrui(val, arith.constant(k_shift_bits, type=T.i32))
                    return vector.from_elements(T.vec(1, T.i32), [val])
                return scale_vec

            def load_b_scale_tile(base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed):
                b_scale_tile = []
                for ku in range_constexpr(ku_packed_limit):
                    for ni in range_constexpr(num_acc_n_packed):
                        scale = load_scale(arg_scale_w, sw_rsrc, layout_b_scale, ku + base_k, ni + _div_pow2(_div_pow2(expert_off_idx + by_n + n_tile_base, _scale_pack_n), 16))
                        scale = _apply_k_shift(scale, k_shift_bits)
                        b_scale_tile.append(scale)
                return b_scale_tile

            def load_a_scale_tile(base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed):
                a_scale_tile = []
                for ku in range_constexpr(ku_packed_limit):
                    for mi in range_constexpr(m_repeat_packed):
                        scale = load_scale(arg_scale_x, sx_rsrc, layout_a_scale, ku + base_k, mi + _div_pow2(_div_pow2(bx_m, _scale_pack_m), 16))
                        scale = _apply_k_shift(scale, k_shift_bits)
                        a_scale_tile.append(scale)
                return a_scale_tile

            def prefetch_ab_scale_tile(base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed):
                return [load_a_scale_tile(base_k, k_shift_bits, ku_packed_limit=ku_packed_limit), load_b_scale_tile(base_k, k_shift_bits, ku_packed_limit=ku_packed_limit)]

            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = col_base_swz_bytes if elem_bytes == 1 else col_base_swz_bytes / arith.index(2)
                idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return (a0, a1)

            def compute_tile(acc_in, b_tile_in, lds_buffer, a_scale=None, b_scale=None, *, prefetch_epilogue: bool=False, a0_prefetch=None, a1_prefetch=None, b_hi_loader=None, ku_count=k_unroll):
                if const_expr(b_hi_loader is not None):
                    b_tile_full = [None] * k_unroll
                    for i in range_constexpr(_b_split_ku):
                        b_tile_full[i] = b_tile_in[i]
                else:
                    b_tile_full = b_tile_in
                acc_list = list(acc_in)
                mfma_res_ty = vec4_f32
                epilogue_pf = None
                if const_expr(prefetch_epilogue):
                    tw_pf = []
                    lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                    ii_idx_list_pf = [arith.constant(ii, index=True) for ii in range(4)]
                    for mi in range_constexpr(m_repeat):
                        mi_base_pf = arith.constant(mi * 16, index=True)
                        for ii in range_constexpr(4):
                            row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                            row_in_tile_pf = mi_base_pf + row_off_pf
                            sorted_row_pf = bx_m + row_in_tile_pf
                            tw_pf.append(buffer_ops.buffer_load(sorted_w_rsrc, sorted_row_pf, vec_width=1, dtype=f32))
                    epilogue_pf = tw_pf
                c0_i64 = arith.constant(0, type=T.i64)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                    return vector.bitcast(vec8_i32, v4)
                _pack_K_shift = (pack_K - 1).bit_length()
                _pack_K_mask = pack_K - 1
                if const_expr(b_hi_loader is not None):
                    _b_hi = b_hi_loader()
                    for _bhi_i in range_constexpr(len(_b_hi)):
                        b_tile_full[_b_split_ku + _bhi_i] = _b_hi[_bhi_i]
                for k_idx in range_constexpr(ku_count):
                    ku128 = k_idx >> _pack_K_shift
                    ikxdl = k_idx & _pack_K_mask
                    b_packs0, b_packs1 = b_tile_full[k_idx]
                    col_base = col_offset_base + k_idx * 128 // a_elem_vec_pack
                    for mi in range_constexpr(m_repeat_packed):
                        a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                        a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])
                        if const_expr(_m_scale_shift_i32 is not None):
                            a_scale_val = arith.shrui(a_scale_val, _m_scale_shift_i32)
                        for ni in range_constexpr(num_acc_n_packed):
                            b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                            b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])
                            if const_expr(_n_scale_shift_i32 is not None):
                                b_scale_val = arith.shrui(b_scale_val, _n_scale_shift_i32)
                            for imxdl in range_constexpr(pack_M):
                                col_base0 = col_base
                                mi_idx = mi * pack_M + imxdl
                                mi_val = arith.constant(mi_idx * 16, index=True)
                                curr_row_a_lds = row_a_lds + mi_val
                                if const_expr(a0_prefetch is not None and k_idx == 0 and (mi_idx == 0)):
                                    a0, a1 = a0_prefetch
                                elif const_expr(a1_prefetch is not None and k_idx == 1 and (mi_idx == 0)):
                                    a0, a1 = a1_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_buffer)
                                a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                                for inxdl in range_constexpr(pack_N):
                                    ni_idx = ni * pack_N + inxdl
                                    b0 = b_packs0[ni_idx]
                                    b1 = b_packs1[ni_idx]
                                    b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                                    acc_idx = mi_idx * num_acc_n + ni_idx
                                    acc_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(mfma_res_ty, [a128, b128, acc_list[acc_idx], cbsz, blgp, ikxdl * _scale_pack_m + imxdl, a_scale_val, ikxdl * _scale_pack_n + inxdl, b_scale_val])
                return (acc_list, epilogue_pf)
            _dma_bytes = 16
            _wave_size = 64
            _eff_bytes_per_buffer = int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
            _num_dma_loads = max(1, _eff_bytes_per_buffer // (total_threads * _dma_bytes))

            def dma_x_tile_to_lds(base_k, lds_buffer):
                c4_idx = arith.index(4)
                base_k_div4 = _div_pow2(_div_pow2(base_k, int(a_elem_vec_pack)) * arith.constant(int(a_elem_bytes), index=True), 4)
                lds_ptr_i64 = None
                for i in range_constexpr(_num_dma_loads):
                    row_local_i = x_row_local[i]
                    col_local_i32_i = x_col_local_i32[i]
                    col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
                    row_k_dw = x_row_base_div4[i] + base_k_div4
                    global_byte_idx = row_k_dw * c4_idx + col_local_sw
                    global_offset = arith.index_cast(T.i32, global_byte_idx)
                    if const_expr(i == 0):
                        lds_addr = memref.extract_aligned_pointer_as_index(lds_buffer) + wave_id * arith.constant(_wave_size * _dma_bytes, index=True)
                        lds_ptr_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                    else:
                        lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * _dma_bytes, type=T.i64)
                    lds_ptr_type = ir.Type.parse('!llvm.ptr<3>')
                    lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)
                    rocdl.raw_ptr_buffer_load_lds(x_rsrc, lds_ptr, arith.constant(_dma_bytes, type=T.i32), global_offset, arith.constant(0, type=T.i32), arith.constant(0, type=T.i32), arith.constant(0, type=T.i32))

            def prefetch_x_to_lds(base_k, lds_buffer):
                dma_x_tile_to_lds(base_k, lds_buffer)
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                rocdl.sched_barrier(0)

            def _k_shift_bits(k_py):
                if const_expr(pack_K >= _scale_pack_k):
                    return 0
                return k_py // 128 % _scale_pack_k * _scale_pack_m * 8

            def _k_base(k_py):
                return k_py // _scale_pack_k // 128
            _c_tile_m_idx = arith.constant(tile_m, index=True)
            _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
            _if_tid = scf.IfOp(_tid_in_range)
            with ir.InsertionPoint(_if_tid.then_block):
                _tid_row = bx_m + tx
                _tid_val = buffer_ops.buffer_load(sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32)
                _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                vector.store(_tid_vec1, lds_tid, [tx])
                scf.YieldOp([])
            gpu.barrier()
            k0 = arith.index(0)
            if const_expr(_b_split_enabled):
                b_cur = load_b_tile_lo(k0)
            else:
                b_cur = load_b_tile(k0)
            a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(_k_base(0), _k_shift_bits(0))
            rocdl.sched_barrier(0)
            prefetch_x_to_lds(k0, lds_x_pong)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            acc = [acc_init] * num_acc_n * m_repeat
            a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base, lds_x_pong)
            _a1_col_base = col_offset_base + 128 // a_elem_vec_pack
            a1_prefetch_pong = lds_load_packs_k64(row_a_lds, _a1_col_base, lds_x_pong) if pack_K >= 2 else None
            num_k_tiles_py = int(inter_dim) // int(tile_k)
            odd_k_tiles = num_k_tiles_py % 2 == 1
            tail_tiles = 1 if odd_k_tiles else 2
            k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
            if const_expr(k_main2_py < 0):
                k_main2_py = 0
            c2_tile_k = arith.constant(tile_k * 2, index=True)
            b_pong = b_cur
            k0_pong_bk = k0

            def _make_b_hi_loader(base_k):
                """Create a b_hi_loader callable for a given base_k."""
                return lambda _bk=base_k: load_b_tile_hi(_bk)
            if const_expr(odd_k_tiles):
                acc, epilogue_pf = compute_tile(acc, b_pong, lds_x_pong, a_scale_pong, b_scale_pong, a0_prefetch=a0_prefetch_pong, a1_prefetch=a1_prefetch_pong, prefetch_epilogue=True, b_hi_loader=_make_b_hi_loader(k0_pong_bk) if _b_split_enabled else None, ku_count=_tail_ku_s2 if _pad_ku_skip_s2 > 0 else k_unroll)
            else:
                k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                k_tail1_py = (int(inter_dim) + tile_k - 1) // tile_k * tile_k - tile_k
                k_tail1_bk = k_tail1 // 2
                prefetch_x_to_lds(k_tail1, lds_x_ping)
                if const_expr(_pad_ku_skip_s2 > 0):
                    b_ping_lo = load_b_tile(k_tail1_bk, ku_limit=_tail_ku_s2)
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(_k_base(k_tail1_py), _k_shift_bits(k_tail1_py), ku_packed_limit=_tail_ku_packed_s2)
                else:
                    b_ping_lo = load_b_tile_lo(k_tail1_bk) if _b_split_enabled else load_b_tile(k_tail1_bk)
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(_k_base(k_tail1_py), _k_shift_bits(k_tail1_py))
                acc, _ = compute_tile(acc, b_pong, lds_x_pong, a_scale_pong, b_scale_pong, a0_prefetch=a0_prefetch_pong, a1_prefetch=a1_prefetch_pong, b_hi_loader=_make_b_hi_loader(k0_pong_bk) if _b_split_enabled else None)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base, lds_x_ping)
                a1_prefetch_ping = lds_load_packs_k64(row_a_lds, _a1_col_base, lds_x_ping) if pack_K >= 2 and (_pad_ku_skip_s2 == 0 or _tail_ku_s2 >= 2) else None
                acc, epilogue_pf = compute_tile(acc, b_ping_lo, lds_x_ping, a_scale_ping, b_scale_ping, a0_prefetch=a0_prefetch_ping, a1_prefetch=a1_prefetch_ping, prefetch_epilogue=True, b_hi_loader=None if _pad_ku_skip_s2 > 0 else _make_b_hi_loader(k_tail1_bk) if _b_split_enabled else None, ku_count=_tail_ku_s2 if _pad_ku_skip_s2 > 0 else k_unroll)
            tw_pf = None
            if const_expr(epilogue_pf is not None):
                tw_pf = epilogue_pf
            mask24_i32 = arith.constant(16777215)
            topk_i32_v = topk_i32
            zero_i32 = arith.constant(0)

            def atomic_add_f16x2(val_f16x2, byte_off_i32):
                rocdl.raw_ptr_buffer_atomic_fadd(val_f16x2, out_rsrc, byte_off_i32, zero_i32, zero_i32)
            if const_expr(lds_out is None):
                raise RuntimeError('FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased.')
            out_base_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg_out))
            out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

            def write_row_to_lds(*, mi: int, ii: int, row_in_tile, row, row_base_lds, col_base_local, num_acc_n: int, lds_out):
                fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                t2 = fused2 & mask24_i32
                s2 = fused2 >> 24
                t_ok = arith.cmpi(CmpIPredicate.ult, t2, tokens_i32)
                s_ok = arith.cmpi(CmpIPredicate.ult, s2, topk_i32_v)
                ts_ok = arith.andi(t_ok, s_ok)
                t2_safe = arith.select(ts_ok, t2, arith.constant(0))
                s2_safe = arith.select(ts_ok, s2, arith.constant(0))
                t2_safe * topk_i32_v + s2_safe
                tw_idx = mi * 4 + ii
                if const_expr(tw_pf is not None):
                    tw = tw_pf[tw_idx]
                else:
                    tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)
                for ni in range_constexpr(num_acc_n):
                    col_local = col_base_local + ni * 16
                    acc_idx = mi * num_acc_n + ni
                    v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                    v = v * tw
                    v_out = arith.trunc_f(out_elem(), v)
                    lds_idx = row_base_lds + col_local
                    vec1_out = T.vec(1, out_elem())
                    v1 = vector.from_elements(vec1_out, [v_out])
                    vector.store(v1, lds_out, [lds_idx], alignment=2)

            def precompute_row(*, row_local, row):
                fused2 = memref.load(lds_tid, [row_local])
                row_i32 = arith.index_cast(T.i32, row)
                row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                t = fused2 & mask24_i32
                s = fused2 >> 24
                t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32)
                s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                t_idx = arith.index_cast(ir.IndexType.get(), t)
                s_idx = arith.index_cast(ir.IndexType.get(), s)
                ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                row_byte_base = out_base_idx + ts_idx * arith.constant(model_dim * out_elem_bytes, index=True)
                return ((fused2, row_byte_base), row_valid)

            def _idx_to_llvm_ptr(idx_val, addr_space=1):
                """Convert an index-typed byte address to !llvm.ptr<addr_space>."""
                idx_v = idx_val._value if hasattr(idx_val, '_value') else idx_val
                i64_v = arith.index_cast(T.i64, idx_v)
                i64_raw = i64_v._value if hasattr(i64_v, '_value') else i64_v
                ptr_ty = ir.Type.parse(f'!llvm.ptr<{addr_space}>')
                return llvm.inttoptr(ptr_ty, i64_raw)

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                fused, row_byte_base = row_ctx
                col_idx = col_g0
                byte_off_col = col_idx * arith.constant(out_elem_bytes, index=True)
                ptr_addr_idx = row_byte_base + byte_off_col
                out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                frag_v = frag._value if hasattr(frag, '_value') else frag
                llvm.StoreOp(frag_v, out_ptr_v, alignment=_e_vec * out_elem_bytes, nontemporal=True)
            _e_vec = min(tile_n // 32, 8)
            c_shuffle_epilog(arith=arith, vector=vector, gpu=gpu, scf=scf, range_constexpr=range_constexpr, tile_m=tile_m, tile_n=tile_n, e_vec=_e_vec, m_repeat=m_repeat, num_acc_n=num_acc_n, tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16, bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base, lds_out=lds_out, frag_elem_type=ir.BF16Type.get(), write_row_to_lds=write_row_to_lds, precompute_row=precompute_row, store_pair=store_pair)
        _all_valid = arith.andi(blk_valid, arith.andi(exp_valid, tile_has_tokens))
        if const_expr(_persistent):
            _cur_active = arith.andi(_still_active, blk_valid)
            _do_gemm = arith.andi(_cur_active, arith.andi(exp_valid, tile_has_tokens))
            _if_valid = scf.IfOp(_do_gemm)
            with ir.InsertionPoint(_if_valid.then_block):
                _moe_gemm2_then_body()
                scf.YieldOp([])
            gpu.barrier()
            scf.YieldOp([_cur_active])
        else:
            _if_valid = scf.IfOp(_all_valid)
            with ir.InsertionPoint(_if_valid.then_block):
                _moe_gemm2_then_body()
                scf.YieldOp([])
            gpu.barrier()
            scf.YieldOp([expert_i32, _expert_b_base])
        _for_ip.__exit__(None, None, None)
    _cache_tag = (module_name, tile_m, tile_n, tile_k, model_dim_pad, inter_dim_pad, persist_m, _sort_block_m, _cu_num if _persistent else 0, xcd_swizzle)

    @flyc.jit
    def launch_kimi_fp4_stage2_16384(arg_out: fx.Pointer, arg_x: fx.Pointer, arg_w: fx.Pointer, arg_scale_x: fx.Pointer, arg_scale_w: fx.Pointer, arg_sorted_token_ids: fx.Pointer, arg_expert_ids: fx.Pointer, arg_sorted_weights: fx.Pointer, arg_num_valid_ids: fx.Pointer, i32_tokens_in: fx.Int32, i32_n_in: fx.Int32, i32_k_in: fx.Int32, i32_size_expert_ids_in: fx.Int32, stream: fx.Stream):
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()
        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        _tile_n_idx = arith.constant(tile_n, index=True)
        _model_dim_pad_idx = arith.constant(model_dim_pad, index=True)
        gx = (n_in - _model_dim_pad_idx + _tile_n_idx - arith.constant(1, index=True)) / _tile_n_idx
        if const_expr(_persistent):
            gy = arith.constant(_cu_num, index=True)
        else:
            _c_pm_l = arith.constant(persist_m, index=True)
            gy = (arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value()) + _c_pm_l - arith.constant(1, index=True)) / _c_pm_l
        moe_gemm2(arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights, arg_num_valid_ids, i32_tokens_in, i32_n_in, i32_k_in, i32_size_expert_ids_in).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)
    return launch_kimi_fp4_stage2_16384

def kimi_fp4_stage1_16384(a: torch.Tensor, w1: torch.Tensor, w1_scale: torch.Tensor, a1_scale: torch.Tensor, sorted_token_ids: torch.Tensor, sorted_expert_ids: torch.Tensor, num_valid_ids: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Launch the fixed fp4 stage1 GEMM.

    Args:
        a: fp4x2 activation, shape [16384, 7168 / 2].
        w1: fp4x2 preshuffled weight, shape [385, 1024, 7168 / 2].
        w1_scale: e8m0 scale, shape [385, 1024, 7168 / 32].
        a1_scale: e8m0 activation scale from fused_dynamic_mxfp4_quant_moe_sort.
    """
    if a.shape[0] != TOKEN:
        raise ValueError(f'expected TOKEN={TOKEN}, got {a.shape[0]}')
    dev = a.device
    if out is None:
        out = torch.empty((TOKEN, TOPK, INTER_DIM), dtype=dtypes.bf16, device=dev)
    sort_block_m = S1.tile_m
    dense_blocks = min(TOKEN * TOPK * sort_block_m, sorted_token_ids.shape[0]) // sort_block_m
    grid_y = min(dense_blocks, sorted_expert_ids.shape[0])
    args = (_ptr_view_safe(out.view(-1)), _ptr_view_safe(a.view(-1)), _ptr_view_safe(w1.view(-1)), _ptr_view_safe(a1_scale.view(-1)), _ptr_view_safe(w1_scale.view(-1)), _ptr_view_safe(sorted_token_ids), _ptr_view_safe(sorted_expert_ids), _ptr_view_safe(num_valid_ids), TOKEN, INTER_DIM * 2, MODEL_DIM, grid_y, torch.cuda.current_stream())
    _run_compiled(compile_kimi_fp4_stage1_16384(), args)
    return out

def kimi_fp4_stage2_16384(inter_states: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor, a2_scale: torch.Tensor, sorted_token_ids: torch.Tensor, sorted_weights: torch.Tensor, sorted_expert_ids: torch.Tensor, num_valid_ids: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Launch the fixed fp4 reduce-mode stage2 GEMM and reduce TOPK."""
    if inter_states.shape[0] != TOKEN:
        raise ValueError(f'expected TOKEN={TOKEN}, got {inter_states.shape[0]}')
    dev = inter_states.device
    if out is None:
        out = torch.empty((TOKEN, MODEL_DIM), dtype=dtypes.bf16, device=dev)
    target = torch.empty((TOKEN * TOPK * MODEL_DIM,), dtype=out.dtype, device=dev)
    m_blocks = min(sorted_expert_ids.shape[0], TOKEN * TOPK)
    args = (_ptr_view_safe(target), _ptr_view_safe(inter_states), _ptr_view_safe(w2), _ptr_view_safe(a2_scale.view(-1)), _ptr_view_safe(w2_scale.view(-1)), _ptr_view_safe(sorted_token_ids), _ptr_view_safe(sorted_expert_ids), _ptr_view_safe(sorted_weights), _ptr_view_safe(num_valid_ids), TOKEN, MODEL_DIM, INTER_DIM, m_blocks, torch.cuda.current_stream())
    _run_compiled(compile_kimi_fp4_stage2_16384(), args)
    torch.sum(target.view(TOKEN, TOPK, MODEL_DIM), dim=1, out=out)
    return out

def run_kimi_fp4_flydsl_moe_16384(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, w1_scale: torch.Tensor, w2_scale: torch.Tensor) -> torch.Tensor:
    """Run the fixed M=16384 fp4 FlyDSL MoE path end-to-end."""
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = kimi_moe_sorting_16384(topk_ids, topk_weight, hidden_states.dtype)
    a1, a1_scale = aiter.fused_dynamic_mxfp4_quant_moe_sort(hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids, token_num=TOKEN, topk=TOPK, block_size=BLOCK_M, num_rows=None)
    a2 = kimi_fp4_stage1_16384(a1, w1, w1_scale.view(dtypes.fp8_e8m0), a1_scale, sorted_ids, sorted_expert_ids, num_valid_ids)
    a2_q, a2_scale = aiter.fused_dynamic_mxfp4_quant_moe_sort(a2.view(-1, INTER_DIM), sorted_ids=sorted_ids, num_valid_ids=num_valid_ids, token_num=TOKEN, topk=TOPK, block_size=BLOCK_M, num_rows=None)
    return kimi_fp4_stage2_16384(a2_q.view(TOKEN, TOPK, -1), w2, w2_scale.view(dtypes.fp8_e8m0), a2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out)
