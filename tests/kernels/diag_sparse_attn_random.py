"""Diagnostic: random KV with various Q to isolate residual bug."""
import sys
import os
import math
import torch

sys.path.insert(0, ".")
from kernels.sparse_attn_gfx1250 import compile_sparse_attn_gfx1250


def reference(Q, KV, topk_idxs, sm_scale):
    """Simple reference: gather KV by topk, compute attention."""
    B, H, D = Q.shape
    _, n_kv, _ = KV.shape
    _, k = topk_idxs.shape
    out = torch.zeros_like(Q, dtype=torch.float32)
    Qf = Q.float()
    KVf = KV.float()
    for b in range(B):
        valid = topk_idxs[b] != -1
        idx = topk_idxs[b].clone()
        idx[~valid] = 0
        kv_g = KVf[b, idx]   # (k, D)
        scores = (Qf[b] @ kv_g.t()) * sm_scale  # (H, k)
        scores = torch.where(valid.unsqueeze(0), scores, torch.full_like(scores, float("-inf")))
        weights = torch.softmax(scores, dim=-1)
        out[b] = weights @ kv_g
    return out


def main():
    import os as _os_local
    B = 1
    n_kv = int(_os_local.environ.get("DIAG_NKV", "64"))
    topk = int(_os_local.environ.get("DIAG_TOPK", "64"))
    H, D = 128, 512
    block_h, block_n = 16, 64
    num_waves, num_kv_bufs = 4, 2
    device = "cuda"
    mode = os.environ.get("DIAG_MODE", "qones_kvrand")  # default

    torch.manual_seed(0)
    # v1.1: KV is fp8_e4m3 + per-block fp32 scale, shape (n_kv, head_dim/128).
    # Each mode generates "real" KV values then quantizes per-block.
    KV_SCALE_BLOCK = 128
    assert D % KV_SCALE_BLOCK == 0
    n_blocks = D // KV_SCALE_BLOCK
    # Uniform per-block scale (single value reused) so the diagnostic stays
    # interpretable; random per-block scale is exercised in the pytest suite.
    KV_SCALE = 0.05  # |kv_real|/scale ≤ 20, well within fp8 e4m3 ±448
    kv_scale = torch.full((n_kv, n_blocks), KV_SCALE, dtype=torch.float32, device=device)
    scale_bcast = (
        kv_scale.view(1, n_kv, n_blocks, 1)
        .expand(B, n_kv, n_blocks, KV_SCALE_BLOCK)
        .reshape(B, n_kv, D)
    )
    def _to_fp8(kv_real_bf16):
        return (kv_real_bf16.float() / scale_bcast).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    if mode == "qones_kvrand":
        Q = torch.ones(B, H, D, dtype=torch.bfloat16, device=device)
        KV_real = (torch.rand(B, n_kv, D, device=device) * 2 - 1).to(torch.bfloat16)
    elif mode == "qrand_kvones":
        Q = (torch.rand(B, H, D, device=device) * 2 - 1).to(torch.bfloat16)
        KV_real = torch.ones(B, n_kv, D, dtype=torch.bfloat16, device=device)
    elif mode == "all_rand":
        Q = (torch.rand(B, H, D, device=device) * 2 - 1).to(torch.bfloat16)
        KV_real = (torch.rand(B, n_kv, D, device=device) * 2 - 1).to(torch.bfloat16)
    elif mode == "qones_kvkrows":
        Q = torch.ones(B, H, D, dtype=torch.bfloat16, device=device)
        kv_vals = torch.arange(1, n_kv + 1, dtype=torch.float32, device=device) / 100.0
        KV_real = kv_vals.view(1, n_kv, 1).expand(B, n_kv, D).contiguous().to(torch.bfloat16)
    elif mode == "qones_kvdcols":
        Q = torch.ones(B, H, D, dtype=torch.bfloat16, device=device)
        kv_vals = torch.arange(1, D + 1, dtype=torch.float32, device=device) / 100.0
        KV_real = kv_vals.view(1, 1, D).expand(B, n_kv, D).contiguous().to(torch.bfloat16)
    elif mode == "kid":
        Q = torch.ones(B, H, D, dtype=torch.bfloat16, device=device)
        k_vals = torch.arange(n_kv, dtype=torch.float32, device=device).view(1, n_kv, 1) * 0.01
        d_vals = torch.arange(D, dtype=torch.float32, device=device).view(1, 1, D) * 0.0001
        KV_real = (k_vals + d_vals).expand(B, n_kv, D).contiguous().to(torch.bfloat16)
    else:
        raise ValueError(mode)

    KV = _to_fp8(KV_real)
    # Reference uses dequant'd KV (matches kernel's gather-time per-block dequant)
    KV_ref = (KV.float() * scale_bcast).to(torch.bfloat16)

    topk_idxs = torch.arange(topk, dtype=torch.int32, device=device).view(1, topk).expand(B, topk).contiguous()
    O = torch.zeros(B, H, D, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / math.sqrt(D)
    launch = compile_sparse_attn_gfx1250(
        nheads_q=H, head_dim=D, topk=topk,
        block_h=block_h, block_n=block_n, sm_scale=sm_scale,
        NUM_WAVES=num_waves, NUM_KV_BUFS=num_kv_bufs,
    )
    stream = torch.cuda.current_stream()
    launch(Q, KV, kv_scale, topk_idxs, O, B, n_kv, topk, stream)
    torch.cuda.synchronize()

    # Self-consistency: run again, verify deterministic
    O2 = torch.zeros(B, H, D, dtype=torch.bfloat16, device=device)
    launch(Q, KV, kv_scale, topk_idxs, O2, B, n_kv, topk, stream)
    torch.cuda.synchronize()
    self_diff = (O.float() - O2.float()).abs().max().item()
    print(f"=== SELF-CONSISTENCY: max diff between two runs = {self_diff:.4e} ===")

    ref = reference(Q, KV_ref, topk_idxs, sm_scale).cpu()
    got = O.float().detach().cpu()
    diff = (got - ref).abs()
    print(f"=== mode={mode} ===")
    print(f"max_diff={diff.max().item():.4e}  mean_diff={diff.mean().item():.4e}")
    print(f"ref     range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")
    print(f"got     range: [{got.min().item():.4f}, {got.max().item():.4f}]")

    # Look at where biggest errors are
    flat = diff.flatten()
    top_idx = flat.argsort(descending=True)[:5]
    for idx in top_idx:
        b = idx // (H * D)
        h = (idx % (H * D)) // D
        d = idx % D
        print(f"  b={b.item():>2} h={h.item():>3} d={d.item():>3}: "
              f"got={got[b,h,d].item():>+10.4f} ref={ref[b,h,d].item():>+10.4f} "
              f"diff={diff[b,h,d].item():.4f}")

    # Per-d max error histogram (head 0)
    print("\nPer-d max error (head 0, d-axis sliced into 64-col bc tiles):")
    h0_diff = diff[0, 0]
    for bc in range(D // 64):
        d_start = bc * 64
        bc_diff = h0_diff[d_start:d_start+64]
        max_d_in_bc = bc_diff.argmax().item() + d_start
        print(f"  bc={bc} (cols {d_start}..{d_start+63}): max_diff={bc_diff.max().item():.4e} "
              f"at d={max_d_in_bc} (got={got[0,0,max_d_in_bc].item():+.4f} "
              f"ref={ref[0,0,max_d_in_bc].item():+.4f})")


if __name__ == "__main__":
    main()
