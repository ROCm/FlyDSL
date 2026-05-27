#!/usr/bin/env python3
"""Reproduce bench heap state: free tensors after each M, then check M=1024 VA + crash."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250

B, K, N = 16, 4096, 1024
GROUP_K, GROUP_N = 128, 128
TILE_K, TILE_N = 128, 128

def ptr_lo_hi(t):
    addr = t.data_ptr()
    return addr & 0xFFFFFFFF, addr >> 32

def check_b_overflow(b_lo, label=""):
    """Return True if any (bz, step) combo overflows addr_lo_b."""
    any_overflow = False
    for bz in range(B):
        addr_lo = (b_lo + bz * K * N) & 0xFFFFFFFF
        lo = addr_lo
        for step in range(32):
            old = lo
            lo = (lo + 0x20000) & 0xFFFFFFFF
            if lo < old:
                where = "PRO" if step < 2 else "MAIN"
                print(f"  {label}bz={bz:2d} lo=0x{addr_lo:08x} overflow at {where} step {step}")
                any_overflow = True
    return any_overflow

def run_one(M, tile_m, nb, label):
    """Allocate tensors, print B addr, check overflow, run kernel. Free tensors after."""
    torch.manual_seed(0)
    M_pad = ((M + tile_m - 1) // tile_m) * tile_m
    a = torch.randn((B, M_pad, K), dtype=torch.bfloat16).cuda().contiguous()
    b = torch.zeros((B, K, N), dtype=torch.float8_e4m3fn).cuda().contiguous()
    scale_fp32 = torch.rand((B, K // GROUP_K, N // GROUP_N)) * 0.1 + 0.01
    log2_s = torch.log2(scale_fp32.clamp(min=1e-38))
    e8m0 = (log2_s.round().to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
    scale = e8m0.cuda().contiguous()
    c = torch.zeros((B, M_pad, N), dtype=torch.bfloat16).cuda()

    a_flat = a.view(-1)
    b_flat = b.view(-1)
    scale_flat = scale.view(-1)
    c_flat = c.view(-1)

    b_lo, b_hi = ptr_lo_hi(b_flat)
    a_lo, a_hi = ptr_lo_hi(a_flat)
    print(f"\n{'='*60}")
    print(f"{label} M={M} tile_m={tile_m} nb={nb}")
    print(f"  A ptr: 0x{a_hi:08x}_{a_lo:08x}  size={a_flat.nbytes//1024//1024}MB")
    print(f"  B ptr: 0x{b_hi:08x}_{b_lo:08x}  size={b_flat.nbytes//1024//1024}MB")
    has_overflow = check_b_overflow(b_lo, label=f"{label} ")

    fn = compile_bmm_a16w8_gfx1250(
        B=B, M=M_pad, N=N, K=K,
        group_k=GROUP_K, group_n=GROUP_N,
        tile_m=tile_m, tile_n=TILE_N, tile_k=TILE_K,
        m_warp=2, n_warp=4, num_buffers=nb, cluster_n=1,
        use_e8m0_scale=True)
    stream = torch.cuda.current_stream()
    try:
        fn(c_flat, a_flat, b_flat, scale_flat, M_pad, stream)
        torch.cuda.synchronize()
        print(f"  kernel: OK{'  (has addr_lo overflow handled by carry)' if has_overflow else ''}")
    except Exception as e:
        print(f"  kernel: CRASH - {e}")
    # Tensors freed here (end of function) — same as bench_config

# Simulate bench M_SWEEP order — free tensors after each run
M_SWEEP = [
    (1,    64, 3, "M=1   dec"),
    (8,    64, 3, "M=8   dec"),
    (32,   64, 3, "M=32  dec"),
    (64,   64, 3, "M=64  dec"),
    (128,  64, 3, "M=128 dec"),
    (256, 128, 2, "M=256 pre"),
    (512,  64, 3, "M=512 pre"),
    (1024, 64, 3, "M=1024 pre"),
]

for M, tile_m, nb, label in M_SWEEP:
    run_one(M, tile_m, nb, label)
