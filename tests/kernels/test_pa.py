# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""PA Decode FP8 — unified test for CDNA (MFMA) and RDNA (WMMA).

CDNA path: FlyDSL vs Gluon (requires aiter) — original run_single()
RDNA path: FlyDSL vs torch reference (self-contained) — run_rdna()
"""
import sys, os, torch, math, logging, random, gc
import pytest

sys.path.insert(0, 'build-fly/python_packages'); sys.path.insert(1, '.')
os.environ['FLYDSL_RUNTIME_ENABLE_CACHE'] = '1'
logging.basicConfig(level=logging.INFO, format='%(message)s')

from tests.test_common import run_perftest, verify_output, checkAllclose
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir as _ir
import flydsl.compiler as flyc, flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

# ── Arch detection ──────────────────────────────────────────────
ARCH = str(get_rocm_arch())
IS_RDNA = is_rdna_arch(ARCH)

# ── Conditional aiter ───────────────────────────────────────────
try:
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        pa_decode_gluon, get_recommended_splits,
        _paged_attention_decode_v2_reduce_kernel_wrapper,
    )
    from aiter import per_tensor_quant, dtypes as aiter_dtypes
    HAS_AITER = True
except ImportError:
    HAS_AITER = False

# ── Arch-dependent kernel imports ───────────────────────────────
if IS_RDNA:
    from kernels.rdna_pa_decode_fp8 import build_pa_decode_module, BLOCK_THREADS
    import kernels.rdna_pa_decode_fp8 as _pa
    from kernels.pa_common import QUERY_GROUP_SIZE, HEAD_SIZE, KV_COMPUTE_BLOCK
    fp8 = torch.float8_e4m3fn
else:
    from kernels.pa_decode_fp8 import build_pa_decode_module, BLOCK_THREADS, QUERY_GROUP_SIZE, HEAD_SIZE, KV_COMPUTE_BLOCK
    import kernels.pa_decode_fp8 as _pa
    fp8 = torch.float8_e4m3fnuz

CPSZ = 256; QG = QUERY_GROUP_SIZE
bf16 = torch.bfloat16; dev = 'cuda'
UNIFORM_RANGE = (-1, 1)
SEED = 0


# ============================================================================
# Shared helpers
# ============================================================================

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_kv_caches(num_blocks, block_size, num_kv_heads, head_size):
    x = 16
    key_shape = (num_blocks, num_kv_heads, head_size // x, block_size, x)
    val_shape = (num_blocks, num_kv_heads, head_size, block_size)
    kc = torch.empty(key_shape, dtype=bf16, device=dev).uniform_(*UNIFORM_RANGE)
    vc = torch.empty(val_shape, dtype=bf16, device=dev).uniform_(*UNIFORM_RANGE)
    return kc, vc


def _per_tensor_quant_fp8_native(tensor, fp8_max=None):
    """Torch-native FP8 per-tensor quantization (no aiter dependency)."""
    if fp8_max is None:
        fp8_max = 448.0 if fp8 == torch.float8_e4m3fn else 240.0
    amax = tensor.float().abs().amax().clamp(min=1e-12)
    scale = amax / fp8_max
    return (tensor.float() / scale).clamp(-fp8_max, fp8_max).to(fp8), scale


def torch_ref_attention(query, key_cache, value_cache, block_tables,
                        context_lengths, key_scale, value_scale,
                        query_scale=None):
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    softmax_scale = 1.0 / math.sqrt(head_dim)
    batch_size = query.shape[0]
    num_query_heads = query.shape[1]
    kc_flat = (key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_dim))
    vc_flat = (value_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_dim))
    kv_dtype = key_cache.dtype
    outputs = []
    for b in range(batch_size):
        bt = block_tables[b]
        ctx_len = context_lengths[b].item()
        tok_idx = (bt.repeat_interleave(block_size)[:ctx_len] * block_size
                   + torch.arange(ctx_len, device=dev) % block_size)
        keys = kc_flat.view(torch.int8)[tok_idx].view(kv_dtype).float()
        if key_scale is not None: keys = keys * key_scale.item()
        vals = vc_flat.view(torch.int8)[tok_idx].view(kv_dtype).float()
        if value_scale is not None: vals = vals * value_scale.item()
        q_b = query[b].float()
        if query_scale is not None: q_b = q_b * query_scale.item()
        qg = num_query_heads // num_heads
        q_grouped = q_b.view(num_heads, qg, head_dim)
        scores = torch.einsum('gqd,tgd->gqt', q_grouped, keys) * softmax_scale
        attn = torch.softmax(scores, dim=-1)
        out_b = torch.einsum('gqt,tgd->gqd', attn, vals)
        outputs.append(out_b.reshape(num_query_heads, head_dim))
    return torch.stack(outputs).to(query.dtype)


# ============================================================================
# CDNA path: FlyDSL vs Gluon (UNCHANGED from original — requires aiter)
# ============================================================================

def quantize_kv_per_tensor(key_cache, value_cache):
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    x = 16
    kc_reshaped = (key_cache.permute(0, 1, 3, 2, 4)
                   .reshape(num_blocks, num_heads, block_size, -1).contiguous())
    kc_reshaped = (kc_reshaped.view(num_blocks, num_heads, block_size,
                                     head_dim // x, x)
                   .permute(0, 1, 3, 2, 4).contiguous())
    q_keys, key_scale = per_tensor_quant(kc_reshaped, quant_dtype=aiter_dtypes.fp8)
    q_vals, val_scale = per_tensor_quant(value_cache, quant_dtype=aiter_dtypes.fp8)
    return q_keys, key_scale, q_vals, val_scale


def gluon_reduce(fd_out, fd_es, fd_ml, context_lengths, num_parts, ps_mode=False):
    batch_size, num_kv_heads, _, qg, head_size = fd_out.shape
    output_5d = torch.empty(batch_size, 1, num_kv_heads, qg, head_size,
                            dtype=bf16, device=dev)
    grid = (batch_size, num_kv_heads, 1)
    _paged_attention_decode_v2_reduce_kernel_wrapper(
        grid, output_5d, fd_es, fd_ml, fd_out, context_lengths, None,
        output_5d.stride(0), output_5d.stride(1),
        output_5d.stride(2), output_5d.stride(3),
        fd_es.stride(0), fd_es.stride(1), fd_es.stride(2),
        fd_out.stride(0), fd_out.stride(1), fd_out.stride(2), fd_out.stride(3),
        query_seq_len=1, query_group_size=qg,
        HEAD_SIZE=head_size, CONTEXT_PARTITION_SIZE=CPSZ,
        PS=ps_mode, context_partition_num=num_parts,
    )
    return output_5d.reshape(batch_size, num_kv_heads, qg, head_size)


def run_single(num_query_heads, num_kv_heads, batch_size, context_length,
               block_size=16, trans_v=False, ps=True, quant_q=True,
               num_iters=100, test_graph=False):
    """CDNA path: full Gluon comparison. Unchanged from original."""
    setup_seed(SEED)
    head_size = HEAD_SIZE
    softmax_scale = 1.0 / math.sqrt(head_size)
    qg = num_query_heads // num_kv_heads
    assert qg == QG, f"QG mismatch: {qg} != {QG}"

    max_ctx = max(16384, context_length)
    max_blocks_per_seq = (max_ctx + block_size - 1) // block_size
    total_blocks = max_blocks_per_seq * batch_size
    blocks_per_seq = (context_length + block_size - 1) // block_size
    num_parts = (context_length + CPSZ - 1) // CPSZ
    _one_shot = (num_parts <= 1)

    fd_num_splits = get_recommended_splits(batch_size, num_kv_heads) if ps else 0

    query = torch.empty(batch_size, num_query_heads, head_size, dtype=bf16, device=dev)
    query.uniform_(*UNIFORM_RANGE)
    kc_bf16, vc_bf16 = create_kv_caches(total_blocks, block_size, num_kv_heads, head_size)
    q_keys, key_scale, q_vals, val_scale = quantize_kv_per_tensor(kc_bf16, vc_bf16)

    block_tables = torch.tensor(
        [[random.randint(0, total_blocks - 1) for _ in range(blocks_per_seq)]
         for _ in range(batch_size)],
        dtype=torch.int32, device=dev)
    context_lengths = torch.full((batch_size,), context_length, dtype=torch.int32, device=dev)

    if quant_q:
        quantized_query, q_scale = per_tensor_quant(query, quant_dtype=aiter_dtypes.fp8)
    else:
        quantized_query = query
        q_scale = None

    ref_out = torch_ref_attention(query, q_keys, q_vals, block_tables,
                                  context_lengths, key_scale, val_scale)

    # ── Gluon ───────────────────────────────────────────────────
    gl_mp = get_recommended_splits(batch_size, num_kv_heads) if ps else num_parts
    gl_inter = (batch_size, num_kv_heads, gl_mp, qg)
    gl_es  = torch.empty(gl_inter, dtype=torch.float32, device=dev)
    gl_ml  = torch.empty(gl_inter, dtype=torch.float32, device=dev)
    gl_tmp = torch.empty(*gl_inter, head_size, dtype=bf16, device=dev)
    gl_out = torch.empty(batch_size, num_query_heads, head_size, dtype=bf16, device=dev)

    pa_decode_gluon(gl_out, quantized_query, q_keys, q_vals, context_lengths,
                    block_tables, softmax_scale, 1, gl_mp, CPSZ, fp8,
                    q_scale, key_scale, val_scale,
                    gl_es, gl_ml, gl_tmp, None, ps=ps)
    torch.cuda.synchronize()

    # ── FlyDSL ──────────────────────────────────────────────────
    if quant_q:
        fd_query = quantized_query
        _qs = q_scale.item()
    else:
        fd_query, _fd_q_scale = per_tensor_quant(query, quant_dtype=aiter_dtypes.fp8)
        _qs = _fd_q_scale.item()

    grid_z = fd_num_splits if fd_num_splits > 0 else (1 if _one_shot else num_parts)
    fd_kfn = build_pa_decode_module(batch_size, num_kv_heads, num_parts,
                                     blocks_per_seq, kv_block_size=block_size,
                                     trans_v=trans_v,
                                     softmax_scale=softmax_scale,
                                     query_scale=_qs,
                                     key_scale=key_scale.item(),
                                     value_scale=val_scale.item(),
                                     one_shot=_one_shot,
                                     ps_num_splits=fd_num_splits)
    fd_al = _pa.allocator

    if _one_shot:
        fd_out = torch.zeros(batch_size, num_kv_heads, 1, qg, head_size, dtype=bf16, device=dev)
        fd_es  = torch.zeros(1, dtype=torch.float32, device=dev)
        fd_ml  = torch.zeros(1, dtype=torch.float32, device=dev)
    else:
        fd_out = torch.zeros(batch_size, num_kv_heads, num_parts, qg, head_size, dtype=bf16, device=dev)
        fd_es  = torch.zeros(batch_size, num_kv_heads, num_parts, qg, dtype=torch.float32, device=dev)
        fd_ml  = torch.full((batch_size, num_kv_heads, num_parts, qg), float('-inf'),
                            dtype=torch.float32, device=dev)

    _cache_tag = (batch_size, num_kv_heads, grid_z)

    @flyc.jit
    def fd_launch(out, es, ml, q, kc, vc, bt, cl: fx.Int32,
                  gx: fx.Int32, gy: fx.Int32, gz: fx.Int32,
                  stream: fx.Stream):
        _ = _cache_tag
        fd_al.finalized = False
        ctx = CompilationContext.get_current()
        with _ir.InsertionPoint(ctx.gpu_module_body):
            fd_al.finalize()
        grid_x = arith.index_cast(T.index, gx.ir_value())
        grid_y = arith.index_cast(T.index, gy.ir_value())
        grid_z_val = arith.index_cast(T.index, gz.ir_value())
        fd_kfn(out, es, ml, q, kc, vc, bt, cl).launch(
            grid=(grid_x, grid_y, grid_z_val),
            block=(BLOCK_THREADS, 1, 1), stream=stream)

    fd_launch(fd_out, fd_es, fd_ml, fd_query, q_keys, q_vals,
              block_tables, context_length,
              batch_size, num_kv_heads, grid_z,
              torch.cuda.current_stream())
    torch.cuda.synchronize()

    if _one_shot:
        fd_final = fd_out.squeeze(2)
    else:
        fd_final = gluon_reduce(fd_out, fd_es, fd_ml, context_lengths,
                                num_parts, ps_mode=(fd_num_splits > 0))
        torch.cuda.synchronize()

    # ── Verify ──────────────────────────────────────────────────
    diff_tol = 5e-2
    fd_flat = fd_final.reshape(batch_size, num_query_heads, head_size).float()
    gl_flat = gl_out.float()
    ref_flat = ref_out.float()

    err_gl = checkAllclose(ref_flat, gl_flat, atol=diff_tol, rtol=diff_tol,
                           msg=f"[Ref vs Gluon]", printLog=False)
    err_fd = checkAllclose(ref_flat, fd_flat, atol=diff_tol, rtol=diff_tol,
                           msg=f"[Ref vs FlyDSL]", printLog=False)
    gl_ok = err_gl < 0.05
    fd_ok = err_fd < 0.05

    mode_str = "one_shot" if _one_shot else (f"ps({fd_num_splits})" if fd_num_splits > 0 else "partitioned")

    # ── Perf ────────────────────────────────────────────────────
    if _one_shot:
        def launch_fd(out, es, ml, q, kc, vc, bt):
            fd_launch(out, es, ml, q, kc, vc, bt,
                      context_length, batch_size, num_kv_heads, grid_z,
                      torch.cuda.current_stream())
    else:
        fd_reduce_out = torch.empty(batch_size, 1, num_kv_heads, qg, head_size,
                                    dtype=bf16, device=dev)
        _reduce_grid = (batch_size, num_kv_heads, 1)
        _is_ps = fd_num_splits > 0

        def launch_fd(out, es, ml, q, kc, vc, bt):
            fd_launch(out, es, ml, q, kc, vc, bt,
                      context_length, batch_size, num_kv_heads, grid_z,
                      torch.cuda.current_stream())
            _paged_attention_decode_v2_reduce_kernel_wrapper(
                _reduce_grid, fd_reduce_out,
                es, ml, out, context_lengths, None,
                fd_reduce_out.stride(0), fd_reduce_out.stride(1),
                fd_reduce_out.stride(2), fd_reduce_out.stride(3),
                es.stride(0), es.stride(1), es.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                query_seq_len=1, query_group_size=qg,
                HEAD_SIZE=head_size, CONTEXT_PARTITION_SIZE=CPSZ,
                PS=_is_ps, context_partition_num=num_parts,
            )

    def launch_gl(out, q, kc, vc, bt):
        pa_decode_gluon(out, q, kc, vc, context_lengths,
                        bt, softmax_scale, 1, gl_mp, CPSZ, fp8,
                        q_scale, key_scale, val_scale,
                        gl_es, gl_ml, gl_tmp, None, ps=ps)

    _, fd_us = run_perftest(launch_fd, fd_out, fd_es, fd_ml, fd_query,
                            q_keys, q_vals, block_tables,
                            num_iters=num_iters, num_warmup=5,
                            testGraph=test_graph)
    _, gl_us = run_perftest(launch_gl, gl_out, quantized_query, q_keys,
                            q_vals, block_tables,
                            num_iters=num_iters, num_warmup=5,
                            testGraph=test_graph)

    torch.cuda.empty_cache(); gc.collect()
    return fd_ok, gl_ok, fd_us, gl_us, mode_str, err_fd


# ============================================================================
# RDNA path: FlyDSL vs torch reference (no aiter dependency)
# ============================================================================

def _torch_reduce(fd_out, fd_es, fd_ml):
    gmax = fd_ml.max(dim=2, keepdim=True).values
    w = fd_es * torch.exp(fd_ml - gmax)
    w = w / w.sum(dim=2, keepdim=True).clamp(min=1e-12)
    return (fd_out.float() * w.unsqueeze(-1)).sum(dim=2).to(bf16)


def run_rdna(num_query_heads, num_kv_heads, batch_size, context_length, block_size=16):
    """RDNA path: self-contained, validates against torch reference."""
    setup_seed(SEED)
    head_size = HEAD_SIZE
    softmax_scale = 1.0 / math.sqrt(head_size)
    qg = num_query_heads // num_kv_heads

    max_ctx = max(16384, context_length)
    max_blocks_per_seq = (max_ctx + block_size - 1) // block_size
    total_blocks = max_blocks_per_seq * batch_size
    blocks_per_seq = (context_length + block_size - 1) // block_size
    num_parts = (context_length + CPSZ - 1) // CPSZ
    _one_shot = (num_parts <= 1)

    query = torch.empty(batch_size, num_query_heads, head_size, dtype=bf16, device=dev).uniform_(*UNIFORM_RANGE)
    kc_bf16, vc_bf16 = create_kv_caches(total_blocks, block_size, num_kv_heads, head_size)
    q_keys, key_scale = _per_tensor_quant_fp8_native(kc_bf16.permute(0,1,3,2,4)
        .reshape(-1, num_kv_heads, block_size, head_size // 16 * 16).contiguous()
        .view(-1, num_kv_heads, block_size, head_size // 16, 16)
        .permute(0,1,3,2,4).contiguous())
    q_vals, val_scale = _per_tensor_quant_fp8_native(vc_bf16)

    block_tables = torch.tensor(
        [[random.randint(0, total_blocks - 1) for _ in range(blocks_per_seq)]
         for _ in range(batch_size)], dtype=torch.int32, device=dev)
    context_lengths_t = torch.full((batch_size,), context_length, dtype=torch.int32, device=dev)
    fd_query, q_scale = _per_tensor_quant_fp8_native(query)

    ref_out = torch_ref_attention(fd_query, q_keys, q_vals, block_tables,
                                  context_lengths_t, key_scale, val_scale,
                                  query_scale=q_scale)

    grid_z = 1 if _one_shot else num_parts
    fd_kfn = build_pa_decode_module(
        batch_size, num_kv_heads, num_parts, blocks_per_seq,
        kv_block_size=block_size, softmax_scale=softmax_scale,
        query_scale=q_scale.item(), key_scale=key_scale.item(),
        value_scale=val_scale.item(), one_shot=_one_shot)
    fd_al = _pa.allocator

    if _one_shot:
        fd_out = torch.zeros(batch_size, num_kv_heads, 1, qg, head_size, dtype=bf16, device=dev)
        fd_es = torch.zeros(1, dtype=torch.float32, device=dev)
        fd_ml = torch.zeros(1, dtype=torch.float32, device=dev)
    else:
        fd_out = torch.zeros(batch_size, num_kv_heads, num_parts, qg, head_size, dtype=bf16, device=dev)
        fd_es = torch.zeros(batch_size, num_kv_heads, num_parts, qg, dtype=torch.float32, device=dev)
        fd_ml = torch.full((batch_size, num_kv_heads, num_parts, qg),
                           float('-inf'), dtype=torch.float32, device=dev)

    _cache_tag = (batch_size, num_kv_heads, grid_z, "rdna")

    @flyc.jit
    def fd_launch(out, es, ml, q, kc, vc, bt, cl: fx.Int32,
                  gx: fx.Int32, gy: fx.Int32, gz: fx.Int32,
                  stream: fx.Stream):
        _ = _cache_tag
        fd_al.finalized = False
        ctx = CompilationContext.get_current()
        with _ir.InsertionPoint(ctx.gpu_module_body):
            fd_al.finalize()
        fd_kfn(out, es, ml, q, kc, vc, bt, cl).launch(
            grid=(arith.index_cast(T.index, gx.ir_value()),
                  arith.index_cast(T.index, gy.ir_value()),
                  arith.index_cast(T.index, gz.ir_value())),
            block=(BLOCK_THREADS, 1, 1), stream=stream)

    def run_kernel():
        fd_launch(fd_out, fd_es, fd_ml, fd_query, q_keys, q_vals,
                  block_tables, context_length, batch_size, num_kv_heads, grid_z,
                  torch.cuda.current_stream())

    run_kernel(); torch.cuda.synchronize()
    fd_final = fd_out.squeeze(2) if _one_shot else _torch_reduce(fd_out, fd_es, fd_ml)

    fd_flat = fd_final.reshape(batch_size, num_query_heads, head_size).float()
    ref_flat = ref_out.float()
    ok = verify_output(fd_flat, ref_flat, atol=0.1, rtol=0.1,
                       msg=f"[RDNA b={batch_size} c={context_length}]")
    _, avg_us = run_perftest(run_kernel, num_iters=50, num_warmup=5)
    torch.cuda.empty_cache(); gc.collect()
    return ok, avg_us


# ============================================================================
# Pytest tests
# ============================================================================

@pytest.mark.parametrize("batch,ctx", [
    pytest.param(1, 128, id="1x128"),
    pytest.param(1, 256, id="1x256"),
    pytest.param(2, 256, id="2x256"),
    pytest.param(1, 512, id="1x512"),
])
def test_pa_decode_rdna(batch, ctx):
    if not IS_RDNA:
        pytest.skip(f"Requires RDNA GPU, got {ARCH}")
    ok, _ = run_rdna(16, 1, batch, ctx)
    assert ok

@pytest.mark.parametrize("batch,ctx", [
    pytest.param(1, 1024, id="1x1024"),
    pytest.param(4, 4096, id="4x4096", marks=pytest.mark.large_shape),
])
def test_pa_decode_rdna_bench(batch, ctx):
    if not IS_RDNA:
        pytest.skip(f"Requires RDNA GPU, got {ARCH}")
    ok, us = run_rdna(16, 1, batch, ctx)
    assert ok
    logging.getLogger("flydsl").info(f"[rdna_pa] batch={batch} ctx={ctx}: {us:.1f} us")

@pytest.mark.parametrize("batch,ctx,ps", [
    pytest.param(128, 4096, False, id="128x4096"),
    pytest.param(128, 8192, True, id="128x8192_ps"),
])
def test_pa_decode_cdna(batch, ctx, ps):
    if IS_RDNA:
        pytest.skip(f"Requires CDNA GPU, got {ARCH}")
    if not HAS_AITER:
        pytest.skip("Requires aiter")
    fd_ok, _, _, _, _, err = run_single(
        16, 1, batch, ctx, block_size=1024, trans_v=True, quant_q=False,
        ps=ps, num_iters=20)
    assert fd_ok, f"FlyDSL err={err:.4f}"


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    if IS_RDNA:
        print(f"GPU: {ARCH} (RDNA)  HEAD={HEAD_SIZE}  QG={QG}")
        print("%5s %5s | %10s | %s" % ("batch", "ctx", "kernel_us", "status"))
        print("-" * 45)
        for batch, ctx in [(1,128),(1,256),(2,256),(1,512),(4,256),(1,1024),(4,4096),(32,4096)]:
            try:
                ok, us = run_rdna(16, 1, batch, ctx)
                print("%5d %5d | %8.1fus | %s" % (batch, ctx, us, "PASS" if ok else "FAIL"))
            except Exception:
                import traceback; traceback.print_exc()
                print("%5d %5d | %10s | FAIL" % (batch, ctx, "ERROR"))
        print("-" * 45)
    elif HAS_AITER:
        import argparse
        parser = argparse.ArgumentParser(description="PA Decode FP8 benchmark")
        parser.add_argument("--test_graph", "-tg", action="store_true", default=False)
        parser.add_argument("--num_iters", type=int, default=100)
        args = parser.parse_args()
        NH, NKV, BS = 16, 1, 1024; tv, qq = True, False
        print(f"{'mode':>14} |  NH  NKV |   BS | batch |   CTX | tv qq |   FlyDSL   |   Gluon    | ratio | status")
        print("-" * 100)
        for batch, CTX, use_ps in [(128,8192,False),(128,8192,True),(128,4096,False)]:
            try:
                fd_ok, gl_ok, fd_us, gl_us, mode_str, err_fd = run_single(
                    NH, NKV, batch, CTX, block_size=BS, trans_v=tv, quant_q=qq,
                    ps=use_ps, num_iters=args.num_iters, test_graph=args.test_graph)
                sp = gl_us / fd_us
                print(f"{mode_str:>14} | {NH:>3} {NKV:>4} | {BS:>4} | {batch:>5} | {CTX:>5} |  T  F | {fd_us:>8.1f}us | {gl_us:>8.1f}us | {sp:>5.2f}x | {'PASS' if fd_ok else 'FAIL':>4} {'PASS' if gl_ok else 'FAIL':>4} (err={err_fd:.4f})")
            except Exception:
                import traceback; traceback.print_exc()
        print("-" * 100)
    else:
        print(f"GPU: {ARCH}, aiter not available, no CDNA tests to run")
