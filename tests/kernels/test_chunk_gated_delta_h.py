"""
Tests for FlyDSL K5: chunk_gated_delta_rule_fwd_h (GDN hidden-state recurrence)

Correctness: compare FlyDSL kernel against a pure-PyTorch reference.
Performance: compare FlyDSL kernel against Triton opt3 kernel.
Rocprof:     profile with rocprofv3 for accurate GPU kernel timing.

Runtime parameters derived from Qwen3.5-397B-A17B serving config:
  K=128, V=128, Hk=16, Hv=32, BT=64
  TP_LIST=[1,4] -> Hg=Hk/TP, H=Hv/TP (parametrized per test)
  max_num_batched_tokens=32768

Usage:
  cd /workspace/FlyDSL
  python3 -m pytest tests/kernels/test_chunk_gated_delta_h.py -v -s
  python3 -m pytest tests/kernels/test_chunk_gated_delta_h.py -v -s -k "Correct"
  python3 -m pytest tests/kernels/test_chunk_gated_delta_h.py -v -s -k "Perf"
  python3 -m pytest tests/kernels/test_chunk_gated_delta_h.py -v -s -k "Rocprof"

  # Direct rocprofv3 profiling (without pytest):
  python3 tests/kernels/test_chunk_gated_delta_h.py --mode rocprof
  python3 tests/kernels/test_chunk_gated_delta_h.py --mode rocprof --full-prompt-len 1000
"""

import argparse
import csv
import ctypes
import json
import subprocess
import sys
import os
from ctypes.util import find_library
from pathlib import Path

import pytest
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.chunk_gated_delta_h import chunk_gated_delta_rule_fwd_h_flydsl, _autotune_cache as _flydsl_autotune_cache

# ── Triton opt3 kernel (inlined, no external dependency) ────────────────

import functools
import triton.language as tl

def _check_platform():
    try:
        backend = triton.runtime.driver.active.get_current_target().backend
    except (RuntimeError, AttributeError):
        backend = "cpu"
    return {"cuda": "nvidia", "hip": "amd", "xpu": "intel"}.get(backend, backend)

_use_cuda_graph = _check_platform() == "nvidia" and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

def _tensor_cache(fn):
    cache_entries = []
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal cache_entries
        for i, (la, lk, lr) in enumerate(cache_entries):
            if len(args) == len(la) and all(a is b for a, b in zip(args, la)) \
               and len(kwargs) == len(lk) and all(k in lk and v is lk[k] for k, v in kwargs.items()):
                cache_entries = cache_entries[:i] + cache_entries[i+1:] + [(la, lk, lr)]
                return lr
        result = fn(*args, **kwargs)
        if len(cache_entries) >= 8:
            cache_entries.pop(0)
        cache_entries.append((args, kwargs, result))
        return result
    return wrapper

@_tensor_cache
def _prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]

@_tensor_cache
def _prepare_chunk_indices(cu_seqlens, chunk_size):
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(_prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

@_tensor_cache
def _prepare_chunk_offsets(cu_seqlens, chunk_size):
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(_prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)

@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_GK": lambda args: args["gk"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[triton.Config({"BV": BV}, num_warps=nw, num_stages=ns)
             for nw in [2, 4] for ns in [1, 2, 3, 4] for BV in [16, 32, 64]],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
    use_cuda_graph=_use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def _triton_fwd_kernel_h_opt3(
    k, v, w, v_new, g, gk, h, h0, ht,
    cu_seqlens, chunk_offsets, T, T_flat,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr, IS_VARLEN: tl.constexpr,
    WU_CONTIGUOUS: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:  b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128: b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192: b_h4 = tl.zeros([64, BV], dtype=tl.float32)
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    if WU_CONTIGUOUS:
        if IS_VARLEN:
            v += ((i_h * T_flat + bos) * V).to(tl.int64)
            w += ((i_h * T_flat + bos) * K).to(tl.int64)
        else:
            v += (((i_n * H + i_h) * T_flat) * V).to(tl.int64)
            w += (((i_n * H + i_h) * T_flat) * K).to(tl.int64)
        stride_v, stride_w = V, K
    else:
        v += ((bos * H + i_h) * V).to(tl.int64)
        w += ((bos * H + i_h) * K).to(tl.int64)
        stride_v, stride_w = H * V, H * K
    if SAVE_NEW_VALUE:
        if IS_VARLEN: v_new += ((i_h * T_flat + bos) * V).to(tl.int64)
        else:         v_new += (((i_n * H + i_h) * T_flat) * V).to(tl.int64)
    stride_h, stride_k = H * K * V, Hg * K
    if USE_INITIAL_STATE: h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE: ht = ht + i_nh * K * V
    if USE_INITIAL_STATE:
        b_h1 += tl.load(tl.make_block_ptr(h0, (K,V),(V,1),(0,i_v*BV),(64,BV),(1,0)), boundary_check=(0,1)).to(tl.float32)
        if K > 64:  b_h2 += tl.load(tl.make_block_ptr(h0,(K,V),(V,1),(64,i_v*BV),(64,BV),(1,0)), boundary_check=(0,1)).to(tl.float32)
        if K > 128: b_h3 += tl.load(tl.make_block_ptr(h0,(K,V),(V,1),(128,i_v*BV),(64,BV),(1,0)), boundary_check=(0,1)).to(tl.float32)
        if K > 192: b_h4 += tl.load(tl.make_block_ptr(h0,(K,V),(V,1),(192,i_v*BV),(64,BV),(1,0)), boundary_check=(0,1)).to(tl.float32)
    for i_t in range(NT):
        tl.store(tl.make_block_ptr(h+i_t*stride_h,(K,V),(V,1),(0,i_v*BV),(64,BV),(1,0)), b_h1.to(tl.bfloat16), boundary_check=(0,1))
        if K > 64:  tl.store(tl.make_block_ptr(h+i_t*stride_h,(K,V),(V,1),(64,i_v*BV),(64,BV),(1,0)), b_h2.to(tl.bfloat16), boundary_check=(0,1))
        if K > 128: tl.store(tl.make_block_ptr(h+i_t*stride_h,(K,V),(V,1),(128,i_v*BV),(64,BV),(1,0)), b_h3.to(tl.bfloat16), boundary_check=(0,1))
        if K > 192: tl.store(tl.make_block_ptr(h+i_t*stride_h,(K,V),(V,1),(192,i_v*BV),(64,BV),(1,0)), b_h4.to(tl.bfloat16), boundary_check=(0,1))
        p_w = tl.make_block_ptr(w,(T,K),(stride_w,1),(i_t*BT,0),(BT,64),(1,0))
        b_v = tl.dot(tl.load(p_w, boundary_check=(0,1)), b_h1.to(tl.bfloat16))
        if K > 64:  b_v += tl.dot(tl.load(tl.make_block_ptr(w,(T,K),(stride_w,1),(i_t*BT,64),(BT,64),(1,0)), boundary_check=(0,1)), b_h2.to(tl.bfloat16))
        if K > 128: b_v += tl.dot(tl.load(tl.make_block_ptr(w,(T,K),(stride_w,1),(i_t*BT,128),(BT,64),(1,0)), boundary_check=(0,1)), b_h3.to(tl.bfloat16))
        if K > 192: b_v += tl.dot(tl.load(tl.make_block_ptr(w,(T,K),(stride_w,1),(i_t*BT,192),(BT,64),(1,0)), boundary_check=(0,1)), b_h4.to(tl.bfloat16))
        b_v = tl.load(tl.make_block_ptr(v,(T,V),(stride_v,1),(i_t*BT,i_v*BV),(BT,BV),(1,0)), boundary_check=(0,1)) - b_v
        if SAVE_NEW_VALUE:
            tl.store(tl.make_block_ptr(v_new,(T,V),(V,1),(i_t*BT,i_v*BV),(BT,BV),(1,0)), b_v.to(tl.bfloat16), boundary_check=(0,1))
        last_idx = min((i_t+1)*BT, T) - 1
        if USE_G:
            m_t = (i_t*BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos*H + last_idx*H + i_h)
            b_g = tl.load(tl.make_block_ptr(g+bos*H+i_h,(T,),(H,),(i_t*BT,),(BT,),(0,)), boundary_check=(0,))
            b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:  b_h2 *= b_g_last
            if K > 128: b_h3 *= b_g_last
            if K > 192: b_h4 *= b_g_last
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_h1 *= tl.exp(tl.load(gk+(bos+last_idx)*H*K+i_h*K+o_k1, mask=(o_k1<K), other=0.0))[:, None]
            if K > 64:  b_h2 *= tl.exp(tl.load(gk+(bos+last_idx)*H*K+i_h*K+64+o_k1, mask=(64+o_k1<K), other=0.0))[:, None]
            if K > 128: b_h3 *= tl.exp(tl.load(gk+(bos+last_idx)*H*K+i_h*K+128+o_k1, mask=(128+o_k1<K), other=0.0))[:, None]
            if K > 192: b_h4 *= tl.exp(tl.load(gk+(bos+last_idx)*H*K+i_h*K+192+o_k1, mask=(192+o_k1<K), other=0.0))[:, None]
        b_v = b_v.to(k.dtype.element_ty)
        b_k = tl.load(tl.make_block_ptr(k,(K,T),(1,stride_k),(0,i_t*BT),(64,BT),(0,1)), boundary_check=(0,1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:  b_h2 += tl.dot(tl.load(tl.make_block_ptr(k,(K,T),(1,stride_k),(64,i_t*BT),(64,BT),(0,1)), boundary_check=(0,1)), b_v)
        if K > 128: b_h3 += tl.dot(tl.load(tl.make_block_ptr(k,(K,T),(1,stride_k),(128,i_t*BT),(64,BT),(0,1)), boundary_check=(0,1)), b_v)
        if K > 192: b_h4 += tl.dot(tl.load(tl.make_block_ptr(k,(K,T),(1,stride_k),(192,i_t*BT),(64,BT),(0,1)), boundary_check=(0,1)), b_v)
    if STORE_FINAL_STATE:
        tl.store(tl.make_block_ptr(ht,(K,V),(V,1),(0,i_v*BV),(64,BV),(1,0)), b_h1.to(tl.float32), boundary_check=(0,1))
        if K > 64:  tl.store(tl.make_block_ptr(ht,(K,V),(V,1),(64,i_v*BV),(64,BV),(1,0)), b_h2.to(tl.float32), boundary_check=(0,1))
        if K > 128: tl.store(tl.make_block_ptr(ht,(K,V),(V,1),(128,i_v*BV),(64,BV),(1,0)), b_h3.to(tl.float32), boundary_check=(0,1))
        if K > 192: tl.store(tl.make_block_ptr(ht,(K,V),(V,1),(192,i_v*BV),(64,BV),(1,0)), b_h4.to(tl.float32), boundary_check=(0,1))

def _fwd_h_triton_opt3_kv(
    k, w, u, g=None, gk=None, initial_state=None,
    output_final_state=False, chunk_size=64, save_new_value=True,
    cu_seqlens=None, wu_contiguous=False,
):
    """Raw triton opt3 kernel call with KV layout [K, V] for hidden states.

    Used directly by benchmark/rocprof to avoid transpose overhead.
    """
    B, T, Hg, K = k.shape
    BT = chunk_size
    if wu_contiguous:
        H, V, T_flat = w.shape[1], u.shape[-1], w.shape[2]
    else:
        H, V, T_flat = u.shape[-2], u.shape[-1], w.shape[1]
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None
    _triton_fwd_kernel_h_opt3[(lambda meta: (triton.cdiv(V, meta["BV"]), N * H))](
        k=k, v=u, w=w, v_new=v_new, g=g, gk=gk,
        h=h, h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, T_flat=T_flat, H=H, Hg=Hg, K=K, V=V, BT=BT,
        WU_CONTIGUOUS=wu_contiguous,
    )
    return h, v_new, final_state


def fwd_h_triton_opt3(
    k, w, u, g=None, gk=None, initial_state=None,
    output_final_state=False, chunk_size=64, save_new_value=True,
    cu_seqlens=None, wu_contiguous=False,
):
    """VK-layout wrapper for triton opt3 (transposes at boundaries)."""
    h0_kv = initial_state.transpose(-2, -1).contiguous() if initial_state is not None else None
    h_kv, v_new, fs_kv = _fwd_h_triton_opt3_kv(
        k, w, u, g=g, gk=gk, initial_state=h0_kv,
        output_final_state=output_final_state,
        chunk_size=chunk_size, save_new_value=save_new_value,
        cu_seqlens=cu_seqlens, wu_contiguous=wu_contiguous,
    )
    h_vk = h_kv.transpose(-2, -1).contiguous()
    fs_vk = fs_kv.transpose(-2, -1).contiguous() if fs_kv is not None else None
    return h_vk, v_new, fs_vk


# ── Triton opt_vk kernel (inlined from linear_attn_example chunk_delta_h_vllm) ──
# w/u: [B, H, T, K/V] token-major; h / h0 / ht: [V, K]; k: [B, T, Hg, K]
_FLA_CHUNK_SIZE_OPT_VK = 64


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_GK": lambda args: args["gk"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
        for BV in [16, 32, 64]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=_use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_opt_vk(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    T_flat,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    if IS_VARLEN:
        w += ((i_h * T_flat + bos) * K).to(tl.int64)
    else:
        w += (((i_n * H + i_h) * T_flat) * K).to(tl.int64)
    if IS_VARLEN:
        v += ((i_h * T_flat + bos) * V).to(tl.int64)
    else:
        v += (((i_n * H + i_h) * T_flat) * V).to(tl.int64)
    if SAVE_NEW_VALUE:
        if IS_VARLEN:
            v_new += ((i_h * T_flat + bos) * V).to(tl.int64)
        else:
            v_new += (((i_n * H + i_h) * T_flat) * V).to(tl.int64)
    stride_v = V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t.to(tl.int64) * stride_h,
            (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(
                v_new, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K), other=0.0,
            )
            b_h1 *= tl.exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K), other=0.0,
                )
                b_h2 *= tl.exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K), other=0.0,
                )
                b_h3 *= tl.exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K), other=0.0,
                )
                b_h4 *= tl.exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_opt_vk(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = _FLA_CHUNK_SIZE_OPT_VK,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)

    assert K <= 256, "Current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)
    final_state = k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_opt_vk[grid](
        k=k, v=u, w=w, v_new=v_new,
        g=g, gk=gk,
        h=h, h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, T_flat=T_flat,
        H=H, Hg=Hg, K=K, V=V, BT=BT,
    )
    return h, v_new, final_state


def fwd_h_triton_opt_vk(
    k, w, u, g=None, gk=None, initial_state=None,
    output_final_state=False, chunk_size=64, save_new_value=True,
    cu_seqlens=None,
):
    """Wrapper around chunk_gated_delta_rule_fwd_h_opt_vk.

    All hidden state tensors use VK layout [V, K] natively — no transpose needed.
    """
    return chunk_gated_delta_rule_fwd_h_opt_vk(
        k, w, u, g=g, gk=gk, initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size, save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
    )


def fwd_h_flydsl(
    k, w, u, g=None, initial_state=None,
    output_final_state=False, cu_seqlens=None, wu_contiguous=True,
):
    """FlyDSL K5: h / h0 / final_state are VK [V, K] (same convention as Triton opt_vk)."""
    return chunk_gated_delta_rule_fwd_h_flydsl(
        k, w, u, g=g, initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, wu_contiguous=wu_contiguous,
    )


# ── Global test configuration ──────────────────────────────────────────
# # Qwen3 params
# K = 128
# V = 128
# Hk = 16
# Hv = 32
# TP_LIST = [1, 4]
# BT = 64

# Qwen3.5 params
K = 128
V = 128
Hk = 16
Hv = 64
TP_LIST = [8]
BT = 64

MAX_NUM_BATCHED_TOKENS = 32768
# FULL_PROMPT_LENS = [1024, 2048, 4096, 8192]
FULL_PROMPT_LENS = [8192]

NUM_WARMUP = 10
NUM_ITERS = 200


def _build_context_lens(full_prompt_len, max_tokens=MAX_NUM_BATCHED_TOKENS):
    context_lens = []
    remaining = max_tokens
    while remaining > 0:
        cur = min(full_prompt_len, remaining)
        context_lens.append(cur)
        remaining -= cur
    return context_lens


def _build_cu_seqlens(context_lens, device="cuda"):
    scheduled_q_lens = context_lens
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(scheduled_q_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    return scheduled_q_lens, cu_seqlens


def _make_inputs(context_lens, tp=1, dtype=torch.bfloat16, device="cuda",
                 with_initial_state=True):
    Hg = Hk // tp
    H = Hv // tp
    scheduled_q_lens, cu_seqlens = _build_cu_seqlens(context_lens, device=device)
    T_total = int(cu_seqlens[-1].item())
    N = len(scheduled_q_lens)
    B = 1

    k = torch.randn(B, T_total, Hg, K, dtype=dtype, device=device) * 0.1
    w_orig = torch.randn(B, T_total, H, K, dtype=dtype, device=device) * 0.1
    u_orig = torch.randn(B, T_total, H, V, dtype=dtype, device=device) * 0.1
    g = torch.randn(T_total, H, dtype=torch.float32, device=device).abs() * -0.5
    g = g.cumsum(dim=0)

    w_c = w_orig.permute(0, 2, 1, 3).contiguous()
    u_c = u_orig.permute(0, 2, 1, 3).contiguous()

    initial_state = None
    if with_initial_state:
        initial_state = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01

    return k, w_orig, u_orig, w_c, u_c, g, initial_state, cu_seqlens, scheduled_q_lens


# ── Pure-PyTorch reference ──────────────────────────────────────────────

def ref_chunk_gated_delta_rule_fwd_h(
    k, w, u, g,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
):
    """Reference in FP32 for correctness checking."""
    B, T, Hg_dim, K_dim = k.shape
    H_dim, V_dim = u.shape[-2], u.shape[-1]
    BT_dim = chunk_size
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT_dim)
    else:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        NT = sum(triton.cdiv(int(seq_len), BT_dim) for seq_len in seq_lens)
    gqa_ratio = H_dim // Hg_dim

    h_out = k.new_zeros(B, NT, H_dim, V_dim, K_dim, dtype=torch.float32)
    v_new_out = torch.zeros_like(u, dtype=torch.float32)

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    final_state = torch.zeros(N, H_dim, V_dim, K_dim, dtype=torch.float32,
                              device=k.device) if output_final_state else None

    for b_idx in range(B):
        if cu_seqlens is not None:
            seqs = [(s, cu_seqlens[s].item(), cu_seqlens[s + 1].item())
                    for s in range(N)]
        else:
            seqs = [(b_idx, 0, T)]

        chunk_offset = 0
        for seq_idx, bos, eos in seqs:
            seq_len = eos - bos
            seq_nt = triton.cdiv(seq_len, BT_dim)

            for i_h in range(H_dim):
                i_hg = i_h // gqa_ratio
                # h_state in VK layout: [V, K]
                h_state = torch.zeros(V_dim, K_dim, dtype=torch.float32,
                                      device=k.device)
                if initial_state is not None:
                    h_state = initial_state[seq_idx, i_h].float().clone()

                for i_t in range(seq_nt):
                    t_start = i_t * BT_dim
                    t_end = min(t_start + BT_dim, seq_len)
                    actual_bt = t_end - t_start

                    h_out[b_idx, chunk_offset + i_t, i_h] = h_state.clone()

                    w_chunk = w[b_idx, bos + t_start:bos + t_end, i_h].float()
                    u_chunk = u[b_idx, bos + t_start:bos + t_end, i_h].float()
                    # h_state is [V,K], need [K,V] for w @ h: w[T,K] @ h[K,V]
                    b_v = u_chunk - w_chunk @ h_state.T
                    v_new_out[b_idx, bos + t_start:bos + t_end, i_h] = b_v

                    last_idx = bos + t_end - 1
                    g_last = g[last_idx, i_h].float()
                    g_chunk = g[bos + t_start:bos + t_end, i_h].float()

                    mask = torch.zeros(BT_dim, device=k.device)
                    mask[:actual_bt] = 1.0
                    gate = torch.where(
                        mask[:actual_bt].bool(),
                        torch.exp(g_last - g_chunk),
                        torch.zeros_like(g_chunk),
                    )
                    b_v_gated = b_v * gate.unsqueeze(-1)

                    h_state = h_state * torch.exp(g_last)
                    k_chunk = k[b_idx, bos + t_start:bos + t_end, i_hg].float()
                    b_v_gated_cast = b_v_gated.to(k.dtype).float()
                    # h[V,K] += (k^T @ v_new)^T = v_new^T @ k
                    h_state = h_state + b_v_gated_cast.T @ k_chunk

                if output_final_state:
                    final_state[seq_idx, i_h] = h_state

            chunk_offset += seq_nt

    return h_out, v_new_out.to(u.dtype), final_state


def _normalize_opt_v_new(vn_opt):
    """Convert opt v_new layout [B, H, T, V] back to [B, T, H, V]."""
    return vn_opt.permute(0, 2, 1, 3).contiguous()


PERF_SHAPES = [
    pytest.param(tp, fpl, id=f"TP{tp}_full{fpl}")
    for tp in TP_LIST
    for fpl in FULL_PROMPT_LENS
]


# ── Correctness tests ───────────────────────────────────────────────────

class TestCorrectness:
    """Correctness against PyTorch reference."""

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_correctness_flydsl(self, tp, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)

        h_fly, vn_fly, fs_fly = fwd_h_flydsl(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k, w_orig, u_orig, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )

        torch.testing.assert_close(
            h_fly.float(), h_ref.float(), atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            _normalize_opt_v_new(vn_fly).float(), vn_ref.float(),
            atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            fs_fly.float(), fs_ref.float(), atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_correctness_triton_opt3(self, tp, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)

        h_tri, vn_tri, fs_tri = fwd_h_triton_opt3(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k, w_orig, u_orig, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )

        torch.testing.assert_close(
            h_tri.float(), h_ref.float(), atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            _normalize_opt_v_new(vn_tri).float(), vn_ref.float(),
            atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            fs_tri.float(), fs_ref.float(), atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_correctness_triton_opt_vk(self, tp, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)

        h_vk, vn_vk, fs_vk = fwd_h_triton_opt_vk(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k, w_orig, u_orig, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )

        torch.testing.assert_close(
            h_vk.float(), h_ref.float(), atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            _normalize_opt_v_new(vn_vk).float(), vn_ref.float(),
            atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            fs_vk.float(), fs_ref.float(), atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_correctness_flydsl_vs_triton(self, tp, full_prompt_len):
        """Direct comparison between FlyDSL and Triton opt3 kernels."""
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)

        h_fly, vn_fly, fs_fly = fwd_h_flydsl(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )
        h_tri, vn_tri, fs_tri = fwd_h_triton_opt3(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
        )

        h_fly_f, h_tri_f = h_fly.float(), h_tri.float()
        vn_fly_f, vn_tri_f = vn_fly.float(), vn_tri.float()
        fs_fly_f, fs_tri_f = fs_fly.float(), fs_tri.float()

        def _report(name, a, b):
            diff = (a - b).abs()
            diff_flat = diff.flatten()
            sorted_diff, _ = diff_flat.sort()
            n = sorted_diff.numel()
            median_val = sorted_diff[n // 2].item()
            p99_val = sorted_diff[min(int(n * 0.99), n - 1)].item()
            print(f"  {name}:")
            print(f"    FlyDSL  range: [{a.min().item():.6f}, {a.max().item():.6f}]")
            print(f"    Triton  range: [{b.min().item():.6f}, {b.max().item():.6f}]")
            print(f"    abs_err  max={diff.max().item():.6f}  "
                  f"mean={diff.mean().item():.6f}  "
                  f"median={median_val:.6f}  "
                  f"p99={p99_val:.6f}")

        print(f"\n[FlyDSL vs Triton opt3  TP={tp} full_prompt_len={full_prompt_len}]")
        _report("h", h_fly_f, h_tri_f)
        _report("v_new", vn_fly_f, vn_tri_f)
        _report("final_state", fs_fly_f, fs_tri_f)

        torch.testing.assert_close(h_fly_f, h_tri_f, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(vn_fly_f, vn_tri_f, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(fs_fly_f, fs_tri_f, atol=1e-1, rtol=1e-1)


# ── Best-config helpers ────────────────────────────────────────────────

def _get_flydsl_best_config() -> str:
    """Return the last cached FlyDSL autotune result as a short string."""
    if not _flydsl_autotune_cache:
        return "N/A"
    bv = list(_flydsl_autotune_cache.values())[-1]
    return f"BV={bv}"


def _get_triton_best_config() -> str:
    """Return the Triton autotune best config as a short string."""
    try:
        kernel = _triton_fwd_kernel_h_opt3
        autotuner = kernel.fn if hasattr(kernel, "fn") else kernel
        cfg = autotuner.best_config
        kw = cfg.kwargs
        bv = kw.get("BV", "?")
        nw = cfg.num_warps
        ns = cfg.num_stages
        return f"BV={bv},nw={nw},ns={ns}"
    except (AttributeError, TypeError):
        return "N/A"


def _get_triton_opt_vk_best_config() -> str:
    """Return the Triton opt_vk autotune best config as a short string."""
    try:
        kernel = chunk_gated_delta_rule_fwd_kernel_h_opt_vk
        autotuner = kernel.fn if hasattr(kernel, "fn") else kernel
        cfg = autotuner.best_config
        kw = cfg.kwargs
        bv = kw.get("BV", "?")
        nw = cfg.num_warps
        ns = cfg.num_stages
        return f"BV={bv},nw={nw},ns={ns}"
    except (AttributeError, TypeError):
        return "N/A"


# ── Performance tests ───────────────────────────────────────────────────

def _bench_fn(fn, *args, **kwargs):
    """Warmup + measure, return average us."""
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_ITERS):
        fn(*args, **kwargs)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / NUM_ITERS * 1000


class TestPerformance:
    """Performance comparison: FlyDSL vs Triton opt3."""

    _results: list[dict] = []

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_perf_comparison(self, tp, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, scheduled_q_lens = _make_inputs(
            context_lens, tp=tp)
        total_tokens = int(cu[-1].item())
        num_seqs = len(context_lens)
        Hg = Hk // tp
        H = Hv // tp

        # Triton opt3 uses KV hidden-state layout; FlyDSL / opt_vk use VK natively.
        h0_kv = h0.transpose(-2, -1).contiguous() if h0 is not None else None

        us_fly = _bench_fn(
            chunk_gated_delta_rule_fwd_h_flydsl,
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
        )

        print(f"\n[K5 FlyDSL TP={tp} Hg={Hg} H={H} T={total_tokens}]  {us_fly:.2f} us")

        us_triton = _bench_fn(
            _fwd_h_triton_opt3_kv,
            k, w_c, u_c, g=g, initial_state=h0_kv,
            output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
        )
        speedup = us_triton / us_fly if us_fly > 0 else float('inf')
        print(f"[K5 Triton opt3 TP={tp} T={total_tokens}]  {us_triton:.2f} us")
        print(f"[Speedup FlyDSL/Triton opt3]  {speedup:.3f}x")

        us_triton_vk = _bench_fn(
            chunk_gated_delta_rule_fwd_h_opt_vk,
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )
        speedup_vk = us_triton_vk / us_fly if us_fly > 0 else float('inf')
        print(f"[K5 Triton opt_vk TP={tp} T={total_tokens}]  {us_triton_vk:.2f} us")
        print(f"[Speedup FlyDSL/Triton opt_vk]  {speedup_vk:.3f}x")

        TestPerformance._results.append({
            "tp": tp,
            "Hg": Hg,
            "H": H,
            "full_prompt_len": full_prompt_len,
            "num_seqs": num_seqs,
            "flydsl_us": us_fly,
            "flydsl_cfg": _get_flydsl_best_config(),
            "triton_us": us_triton,
            "triton_cfg": _get_triton_best_config(),
            "speedup": speedup,
            "triton_opt_vk_us": us_triton_vk,
            "triton_opt_vk_cfg": _get_triton_opt_vk_best_config(),
            "speedup_vk": speedup_vk,
        })

    @pytest.fixture(autouse=True, scope="class")
    def _print_table(self):
        TestPerformance._results = []
        yield
        _print_summary_table(TestPerformance._results, title="CUDA Event Performance Summary")


# ── rocprofv3 profiling infrastructure ──────────────────────────────────

TARGET_KERNEL_FLYDSL = "chunk_gdn_fwd_h_opt3"
TARGET_KERNEL_TRITON = "_triton_fwd_kernel_h_opt3"
TARGET_KERNEL_TRITON_OPT_VK = "chunk_gated_delta_rule_fwd_kernel_h_opt_vk"


def _load_roctx_library():
    """Load the roctx shared library for profiler pause/resume control."""
    for candidate in ("rocprofiler-sdk-roctx", "roctx64"):
        libname = find_library(candidate)
        if libname is None:
            continue
        lib = ctypes.CDLL(libname)
        lib.roctxGetThreadId.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.roctxGetThreadId.restype = None
        lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
        lib.roctxProfilerPause.restype = None
        lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
        lib.roctxProfilerResume.restype = None
        lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        lib.roctxRangePushA.restype = ctypes.c_int
        lib.roctxRangePop.argtypes = []
        lib.roctxRangePop.restype = ctypes.c_int
        return lib
    return None


def _roctx_thread_id(lib):
    tid = ctypes.c_uint64()
    lib.roctxGetThreadId(ctypes.byref(tid))
    return int(tid.value)


def _rocprof_worker(full_prompt_len, tp=1, config_path: str | None = None):
    """Inner worker: runs under rocprofv3 --selected-regions.

    Profiling starts paused. We warmup both kernels, then
    Resume -> measured iterations -> Pause for each kernel sequentially.
    """
    roctx = _load_roctx_library()
    if roctx is None:
        raise RuntimeError("roctx library not found; cannot run as profiling worker")

    tid = _roctx_thread_id(roctx)

    context_lens = _build_context_lens(full_prompt_len)
    k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)
    total_tokens = int(cu[-1].item())

    # Triton opt3 (KV) needs transposed h0; FlyDSL uses VK like opt_vk.
    h0_kv = h0.transpose(-2, -1).contiguous() if h0 is not None else None

    run_fly = lambda: chunk_gated_delta_rule_fwd_h_flydsl(
        k, w_c, u_c, g=g, initial_state=h0,
        output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
    )

    # Warmup FlyDSL (paused)
    print(f"[rocprof-worker] Warmup FlyDSL (T={total_tokens}) ...", flush=True)
    for _ in range(NUM_WARMUP):
        run_fly()
    torch.cuda.synchronize()

    # Measure FlyDSL
    roctx.roctxProfilerResume(tid)
    roctx.roctxRangePushA(b"flydsl_k5_bench")
    for _ in range(NUM_ITERS):
        run_fly()
    torch.cuda.synchronize()
    roctx.roctxRangePop()
    roctx.roctxProfilerPause(tid)
    print(f"[rocprof-worker] FlyDSL: {NUM_ITERS} iterations done", flush=True)

    # Triton opt3 (KV layout, h0 pre-transposed)
    run_tri = lambda: _fwd_h_triton_opt3_kv(
        k, w_c, u_c, g=g, initial_state=h0_kv,
        output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
    )

    print(f"[rocprof-worker] Warmup Triton opt3 ...", flush=True)
    for _ in range(NUM_WARMUP):
        run_tri()
    torch.cuda.synchronize()

    roctx.roctxProfilerResume(tid)
    roctx.roctxRangePushA(b"triton_k5_bench")
    for _ in range(NUM_ITERS):
        run_tri()
    torch.cuda.synchronize()
    roctx.roctxRangePop()
    roctx.roctxProfilerPause(tid)
    print(f"[rocprof-worker] Triton opt3: {NUM_ITERS} iterations done", flush=True)

    # Triton opt_vk (VK layout, h0 directly)
    run_tri_vk = lambda: chunk_gated_delta_rule_fwd_h_opt_vk(
        k, w_c, u_c, g=g, initial_state=h0,
        output_final_state=True, cu_seqlens=cu,
    )

    print(f"[rocprof-worker] Warmup Triton opt_vk ...", flush=True)
    for _ in range(NUM_WARMUP):
        run_tri_vk()
    torch.cuda.synchronize()

    roctx.roctxProfilerResume(tid)
    roctx.roctxRangePushA(b"triton_k5_opt_vk_bench")
    for _ in range(NUM_ITERS):
        run_tri_vk()
    torch.cuda.synchronize()
    roctx.roctxRangePop()
    roctx.roctxProfilerPause(tid)
    print(f"[rocprof-worker] Triton opt_vk: {NUM_ITERS} iterations done", flush=True)

    if config_path is not None:
        Path(config_path).write_text(json.dumps({
            "flydsl_cfg": _get_flydsl_best_config(),
            "triton_cfg": _get_triton_best_config(),
            "triton_opt_vk_cfg": _get_triton_opt_vk_best_config(),
        }))


def _parse_kernel_stats(stats_path: Path) -> dict[str, dict]:
    """Parse kernel_stats CSV -> {name: {AverageNs, TotalDurationNs, Calls, ...}}."""
    result = {}
    with stats_path.open(newline="") as f:
        for row in csv.DictReader(f):
            result[row["Name"]] = row
    return result


def _extract_kernel_us(stats: dict, kname: str) -> dict | None:
    """Find a kernel entry in parsed stats and return timing dict, or None."""
    entry = stats.get(kname)
    if entry is None:
        for name in stats:
            if kname in name:
                entry = stats[name]
                break
    if entry is None:
        return None
    return {
        "avg_us": float(entry["AverageNs"]) / 1000,
        "min_us": float(entry["MinNs"]) / 1000,
        "max_us": float(entry["MaxNs"]) / 1000,
        "calls": int(entry["Calls"]),
        "total_ms": float(entry["TotalDurationNs"]) / 1e6,
    }


def _print_rocprof_summary(stats_path: Path, total_tokens: int) -> dict | None:
    """Print a formatted summary and return {flydsl_us, triton_us, speedup} or None."""
    stats = _parse_kernel_stats(stats_path)

    targets = [
        ("FlyDSL", TARGET_KERNEL_FLYDSL),
        ("Triton opt3", TARGET_KERNEL_TRITON),
        ("Triton opt_vk", TARGET_KERNEL_TRITON_OPT_VK),
    ]

    results = {}
    for label, kname in targets:
        t = _extract_kernel_us(stats, kname)
        if t is None:
            print(f"  {label}: kernel '{kname}' not found in stats")
            continue
        results[label] = t
        print(f"  {label} ({kname}):")
        print(f"    Calls:   {t['calls']}")
        print(f"    Average: {t['avg_us']:.2f} us")
        print(f"    Min:     {t['min_us']:.2f} us")
        print(f"    Max:     {t['max_us']:.2f} us")
        print(f"    Total:   {t['total_ms']:.2f} ms")

    row = None
    if "FlyDSL" in results and "Triton opt3" in results:
        speedup = results["Triton opt3"]["avg_us"] / results["FlyDSL"]["avg_us"]
        print(f"\n  Speedup (FlyDSL vs Triton opt3): {speedup:.3f}x")
        row = {
            "flydsl_us": results["FlyDSL"]["avg_us"],
            "triton_us": results["Triton opt3"]["avg_us"],
            "speedup": speedup,
        }
        if "Triton opt_vk" in results:
            speedup_vk = results["Triton opt_vk"]["avg_us"] / results["FlyDSL"]["avg_us"]
            print(f"  Speedup (FlyDSL vs Triton opt_vk): {speedup_vk:.3f}x")
            row["triton_opt_vk_us"] = results["Triton opt_vk"]["avg_us"]
            row["speedup_vk"] = speedup_vk

    if not stats:
        print("  WARNING: no kernels found in stats file")
    elif not results:
        print("  Available kernels:")
        for name in sorted(stats.keys()):
            print(f"    {name}")

    return row


def _do_rocprof(full_prompt_len, tp=1) -> tuple[int, dict | None]:
    """Outer driver: launches rocprofv3 wrapping this script in --_rocprof-worker mode.

    Returns (returncode, row_dict_or_None).
    row_dict keys: tp, Hg, H, full_prompt_len, num_seqs, flydsl_us, triton_us, speedup.
    """
    Hg = Hk // tp
    H = Hv // tp
    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = repo_root / "rocprof_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"gdn_k5_tp{tp}_fpl{full_prompt_len}"
    config_path = output_dir / f"{output_stem}_best_cfg.json"
    if config_path.exists():
        config_path.unlink()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    inner_cmd = [
        "python3", "-u", str(Path(__file__).resolve()),
        "--_rocprof-worker",
        "--full-prompt-len", str(full_prompt_len),
        "--tp", str(tp),
        "--rocprof-config-path", str(config_path),
    ]
    rocprof_cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--marker-trace",
        "--output-format", "csv",
        "-d", str(output_dir),
        "-o", output_stem,
        "--stats",
        "--selected-regions",
        "--", *inner_cmd,
    ]

    context_lens = _build_context_lens(full_prompt_len)
    total_tokens = sum(context_lens)
    num_seqs = len(context_lens)

    print(f"\n[rocprof] TP={tp} Hg={Hg} H={H} full_prompt_len={full_prompt_len}, T={total_tokens}")
    print(f"[rocprof] cmd: {' '.join(rocprof_cmd)}", flush=True)
    result = subprocess.run(rocprof_cmd, cwd=repo_root, env=env)

    row = None
    stats_path = output_dir / f"{output_stem}_kernel_stats.csv"
    if stats_path.exists():
        print(f"\n[rocprof] Results (TP={tp} full_prompt_len={full_prompt_len}, T={total_tokens}):")
        perf = _print_rocprof_summary(stats_path, total_tokens)
        if perf is not None:
            row = {
                "tp": tp,
                "Hg": Hg,
                "H": H,
                "full_prompt_len": full_prompt_len,
                "num_seqs": num_seqs,
                **perf,
            }
            if config_path.exists():
                row.update(json.loads(config_path.read_text()))
    else:
        print(f"[rocprof] kernel stats not found: {stats_path}", flush=True)
        trace_path = output_dir / f"{output_stem}_kernel_trace.csv"
        if trace_path.exists():
            print(f"[rocprof] trace file exists: {trace_path}")

    if result.returncode != 0:
        print(f"[rocprof] rocprofv3 exited with code {result.returncode}", flush=True)

    return result.returncode, row


def _print_summary_table(rows: list[dict], title: str = "Performance Summary"):
    """Print a formatted summary table from collected benchmark rows.

    Rows are grouped by TP value, each group gets its own sub-table.
    Supports both 2-kernel (FlyDSL + Triton opt3) and 3-kernel
    (FlyDSL + Triton opt3 + Triton opt_vk) result sets.
    """
    if not rows:
        return

    from itertools import groupby

    has_opt_vk = any("triton_opt_vk_us" in r for r in rows)

    w = 200 if has_opt_vk else 138
    sep = "-" * w
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"  Fixed: K={K}, V={V}, Hk={Hk}, Hv={Hv}, BT={BT}, max_tokens={MAX_NUM_BATCHED_TOKENS}")
    print(f"{'=' * w}")

    sorted_rows = sorted(rows, key=lambda r: (r.get("tp", 1), r["full_prompt_len"]))
    for tp_val, group in groupby(sorted_rows, key=lambda r: r.get("tp", 1)):
        group_rows = list(group)
        Hg_val = group_rows[0].get("Hg", "?")
        H_val = group_rows[0].get("H", "?")
        print(f"\n  TP={tp_val}  (Hg={Hg_val}, H={H_val})")
        hdr = (f"  {'FullPromptLen':>13}  {'NumSeqs':>8}  "
               f"{'FlyDSL(us)':>11}  {'FlyDSL BestCfg':>15}  "
               f"{'Triton(us)':>11}  {'Triton BestCfg':>22}  "
               f"{'Speedup':>8}")
        if has_opt_vk:
            hdr += (f"  {'TritonVK(us)':>13}  {'TritonVK BestCfg':>22}  "
                    f"{'SpeedupVK':>10}")
        print(hdr)
        print(f"  {sep}")
        for r in group_rows:
            fly_cfg = r.get("flydsl_cfg", "N/A")
            tri_cfg = r.get("triton_cfg", "N/A")
            line = (f"  {r['full_prompt_len']:>13}  {r['num_seqs']:>8}  "
                    f"{r['flydsl_us']:>11.2f}  {fly_cfg:>15}  "
                    f"{r['triton_us']:>11.2f}  {tri_cfg:>22}  "
                    f"{r['speedup']:>7.3f}x")
            if has_opt_vk:
                vk_us = r.get("triton_opt_vk_us", 0)
                vk_cfg = r.get("triton_opt_vk_cfg", "N/A")
                vk_speedup = r.get("speedup_vk", 0)
                line += f"  {vk_us:>13.2f}  {vk_cfg:>22}  {vk_speedup:>9.3f}x"
            print(line)

    print(f"{'=' * w}\n")


# ── rocprofv3 pytest tests ─────────────────────────────────────────────

class TestRocprof:
    """Profile FlyDSL and Triton kernels with rocprofv3."""

    _results: list[dict] = []

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_rocprof(self, tp, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)
        fwd_h_flydsl(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )
        fwd_h_triton_opt3(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
        )
        fwd_h_triton_opt_vk(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )
        torch.cuda.synchronize()

        rc, row = _do_rocprof(full_prompt_len, tp=tp)
        if row is not None:
            row.setdefault("flydsl_cfg", _get_flydsl_best_config())
            row.setdefault("triton_cfg", _get_triton_best_config())
            row.setdefault("triton_opt_vk_cfg", _get_triton_opt_vk_best_config())
            TestRocprof._results.append(row)
        assert rc == 0, f"rocprofv3 exited with code {rc}"

    @pytest.fixture(autouse=True, scope="class")
    def _print_table(self):
        TestRocprof._results = []
        yield
        _print_summary_table(TestRocprof._results, title="Rocprof Performance Summary")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDN K5 test / profile")
    parser.add_argument("--mode", choices=["test", "rocprof"], default="test",
                        help="test=pytest (default), rocprof=rocprofv3 profiling")
    parser.add_argument("--full-prompt-len", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--_rocprof-worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--rocprof-config-path", type=str, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._rocprof_worker:
        _rocprof_worker(args.full_prompt_len, tp=args.tp, config_path=args.rocprof_config_path)
    elif args.mode == "rocprof":
        _do_rocprof(args.full_prompt_len, tp=args.tp)
    else:
        pytest.main([__file__, "-v", "-s"])
