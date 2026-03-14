"""PA Decode FP8 — FlyDSL (sliding-window kernel) vs Gluon benchmark."""
import sys, os, torch, math, logging, random, gc
sys.path.insert(0, 'build-fly/python_packages'); sys.path.insert(1, '.')
os.environ['FLYDSL_RUNTIME_ENABLE_CACHE'] = '1'
logging.basicConfig(level=logging.INFO, format='%(message)s')

from tests.test_common import run_perftest, checkAllclose
from kernels.pa_decode_sw_fp8 import (
    pa_decode_sw_launch,
    QUERY_GROUP_SIZE, HEAD_SIZE, KV_COMPUTE_BLOCK,
)
from aiter.ops.triton.gluon.pa_decode_gluon import (
    pa_decode_gluon, get_recommended_splits,
)
from aiter import per_tensor_quant, dtypes as aiter_dtypes

CPSZ = KV_COMPUTE_BLOCK; QG = QUERY_GROUP_SIZE
fp8 = torch.float8_e4m3fnuz; bf16 = torch.bfloat16; dev = 'cuda'
UNIFORM_RANGE = (-1, 1)
SEED = 0


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_kv_caches(num_blocks, block_size, num_kv_heads, head_size):
    x = 16
    key_shape = (num_blocks, num_kv_heads, head_size // x, block_size, x)
    val_shape = (num_blocks, num_kv_heads, head_size, block_size)
    kc = torch.empty(key_shape, dtype=bf16, device=dev)
    vc = torch.empty(val_shape, dtype=bf16, device=dev)
    kc.uniform_(*UNIFORM_RANGE)
    vc.uniform_(*UNIFORM_RANGE)
    return kc, vc


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


def torch_ref_attention(query, key_cache, value_cache, block_tables,
                        context_lengths, key_scale, value_scale):
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
        qg = num_query_heads // num_heads
        q_grouped = q_b.view(num_heads, qg, head_dim)
        scores = torch.einsum('gqd,tgd->gqt', q_grouped, keys) * softmax_scale
        attn = torch.softmax(scores, dim=-1)
        out_b = torch.einsum('gqt,tgd->gqd', attn, vals)
        outputs.append(out_b.reshape(num_query_heads, head_dim))
    return torch.stack(outputs).to(query.dtype)


def run_single(num_query_heads, num_kv_heads, batch_size, context_length,
               block_size=16, trans_v=False, ps=True, quant_q=True,
               num_iters=100, test_graph=False):
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

    # ── FlyDSL (sliding-window kernel) ───────────────────────────
    if quant_q:
        fd_query = quantized_query
        fd_q_scale = q_scale
    else:
        fd_query, fd_q_scale = per_tensor_quant(query, quant_dtype=aiter_dtypes.fp8)
    if not isinstance(fd_q_scale, torch.Tensor):
        fd_q_scale = torch.tensor([float(fd_q_scale)], device=dev, dtype=torch.float32)

    fd_k_scale = key_scale if isinstance(key_scale, torch.Tensor) else torch.tensor([key_scale.item()], device=dev, dtype=torch.float32)
    fd_v_scale = val_scale if isinstance(val_scale, torch.Tensor) else torch.tensor([val_scale.item()], device=dev, dtype=torch.float32)

    fd_out = torch.empty(batch_size, num_query_heads, head_size, dtype=bf16, device=dev)

    mode_str = pa_decode_sw_launch(
        fd_out, fd_query, q_keys, q_vals,
        context_lengths, block_tables,
        softmax_scale, fd_q_scale, fd_k_scale, fd_v_scale,
        sliding_window=0, max_context_len=context_length,
        kv_block_size=block_size,
    )
    torch.cuda.synchronize()

    # ── Verify ──────────────────────────────────────────────────
    diff_tol = 5e-2
    fd_flat = fd_out.float()
    gl_flat = gl_out.float()
    ref_flat = ref_out.float()

    err_gl = checkAllclose(ref_flat, gl_flat, atol=diff_tol, rtol=diff_tol,
                           msg=f"[Ref vs Gluon]", printLog=False)
    err_fd = checkAllclose(ref_flat, fd_flat, atol=diff_tol, rtol=diff_tol,
                           msg=f"[Ref vs FlyDSL]", printLog=False)
    gl_ok = err_gl < 0.05
    fd_ok = err_fd < 0.05

    # ── Perf (torch.profiler trace → self_device_time_total) ──
    def launch_fd(out, q, kc, bt):
        pa_decode_sw_launch(out, q, kc, q_vals, context_lengths, bt,
                            softmax_scale, fd_q_scale, fd_k_scale, fd_v_scale,
                            sliding_window=0, max_context_len=context_length,
                            kv_block_size=block_size)

    def launch_gl(out, q, kc, vc, bt):
        pa_decode_gluon(out, q, kc, vc, context_lengths,
                        bt, softmax_scale, 1, gl_mp, CPSZ, fp8,
                        q_scale, key_scale, val_scale,
                        gl_es, gl_ml, gl_tmp, None, ps=ps)

    _, fd_us = run_perftest(launch_fd, fd_out, fd_query,
                            q_keys, block_tables,
                            num_iters=num_iters, num_warmup=5)
    _, gl_us = run_perftest(launch_gl, gl_out, quantized_query, q_keys,
                            q_vals, block_tables,
                            num_iters=num_iters, num_warmup=5)

    torch.cuda.empty_cache(); gc.collect()
    return fd_ok, gl_ok, fd_us, gl_us, mode_str, err_fd


# ── Test configs ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PA Decode FP8 benchmark")
    parser.add_argument("--test_graph", "-tg", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=100)
    args = parser.parse_args()

    NH, NKV, BS = 16, 1, 1024
    tv, qq = True, False

    print(f"{'mode':>14} |  NH  NKV |   BS | batch |   CTX | tv qq |   FlyDSL   |   Gluon    | ratio | status")
    print("-" * 100)

    configs = [
        (128, 8192, False),
        (128, 8192, True),
        (128, 4096, False),
    ]

    for batch, CTX, use_ps in configs:
        try:
            fd_ok, gl_ok, fd_us, gl_us, mode_str, err_fd = run_single(
                NH, NKV, batch, CTX, block_size=BS, trans_v=tv, quant_q=qq,
                ps=use_ps, num_iters=args.num_iters, test_graph=args.test_graph)
            sp = gl_us / fd_us
            fd_s = "PASS" if fd_ok else "FAIL"
            gl_s = "PASS" if gl_ok else "FAIL"
            print(f"{mode_str:>14} | {NH:>3} {NKV:>4} | {BS:>4} | {batch:>5} | {CTX:>5} |  T  F | {fd_us:>8.1f}us | {gl_us:>8.1f}us | {sp:>5.2f}x | {fd_s:>4} {gl_s:>4} (err={err_fd:.4f})")
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"{'ERROR':>14} | {NH:>3} {NKV:>4} | {BS:>4} | {batch:>5} | {CTX:>5} | ps={'T' if use_ps else 'F'} | EXCEPTION")
    print("-" * 100)
