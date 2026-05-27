#!/usr/bin/env python3
"""Minimal M=512 debug: print tensor addresses and check addr_lo_b overflow risk."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250

B, K, N = 16, 4096, 1024
TILE_K, TILE_N = 128, 128

def ptr_lo_hi(t):
    addr = t.data_ptr()
    return addr & 0xFFFFFFFF, addr >> 32

def check_overflow(base_lo, batch, adv=131072, steps=32, pre_loaded=2):
    """Simulate addr_lo_b advances; report which step overflows."""
    lo = (base_lo + batch * K * N) & 0xFFFFFFFF
    hi = 0  # relative, just tracking carry count
    for step in range(steps):
        old_lo = lo
        lo = (lo + adv) & 0xFFFFFFFF
        carry = 1 if lo < old_lo else 0
        hi += carry
        where = "PROLOGUE" if step < pre_loaded else "MAINLOOP"
        if carry:
            print(f"  step {step:2d} ({where}): CARRY! old_lo=0x{old_lo:08x} "
                  f"-> new_lo=0x{lo:08x}, carry -> addr_hi += 1")

def analyze_tensors(M):
    print(f"\n{'='*60}")
    print(f"M={M}")
    torch.manual_seed(0)
    a = torch.randn((B, M, K), dtype=torch.bfloat16).cuda().contiguous()
    b = torch.zeros((B, K, N), dtype=torch.float8_e4m3fn).cuda().contiguous()
    scale_fp32 = torch.rand((B, K // 128, N // 128)) * 0.1 + 0.01
    log2_s = torch.log2(scale_fp32.clamp(min=1e-38))
    e8m0 = (log2_s.round().to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
    scale = e8m0.cuda().contiguous()
    c = torch.zeros((B, M, N), dtype=torch.bfloat16).cuda()

    a_flat = a.view(-1); b_flat = b.view(-1)
    scale_flat = scale.view(-1); c_flat = c.view(-1)

    a_lo, a_hi = ptr_lo_hi(a_flat)
    b_lo, b_hi = ptr_lo_hi(b_flat)
    print(f"A ptr: 0x{a_hi:08x}_{a_lo:08x}  size={a_flat.nbytes//1024//1024}MB")
    print(f"B ptr: 0x{b_hi:08x}_{b_lo:08x}  size={b_flat.nbytes//1024//1024}MB")

    # B tensor: adv=0x20000 per step, 32 steps, pre_loaded=2
    print(f"\nB overflow (adv=0x20000, 32 steps, pre_loaded=2):")
    for bz in range(16):
        addr_lo = (b_lo + bz * K * N) & 0xFFFFFFFF
        lo = addr_lo; overflows = []
        for step in range(32):
            old = lo; lo = (lo + 0x20000) & 0xFFFFFFFF
            if lo < old: overflows.append(("PRO" if step < 2 else "MAIN", step))
        tag = f"OVERFLOW{overflows}" if overflows else "safe"
        if overflows:
            print(f"  bz={bz:2d} lo=0x{addr_lo:08x} -> {tag}")

    # A tensor: adv=256 per step, 32 steps
    a_off = M * K * 2  # bytes per batch
    print(f"\nA overflow (adv=256, 32 steps):")
    for bz in range(16):
        addr_lo = (a_lo + bz * a_off) & 0xFFFFFFFF
        lo = addr_lo; overflows = []
        for step in range(32):
            old = lo; lo = (lo + 256) & 0xFFFFFFFF
            if lo < old: overflows.append(step)
        if overflows:
            print(f"  bz={bz:2d} lo=0x{addr_lo:08x} -> OVERFLOW at steps {overflows}")

    print("\nRunning kernel...")
    try:
        fn = compile_bmm_a16w8_gfx1250(
            B=B, M=M, N=N, K=K,
            group_k=128, group_n=128,
            tile_m=64, tile_n=TILE_N, tile_k=TILE_K,
            m_warp=2, n_warp=4, num_buffers=3, cluster_n=1,
            use_e8m0_scale=True)
        stream = torch.cuda.current_stream()
        fn(c_flat, a_flat, b_flat, scale_flat, M, stream)
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"CRASH: {e}")
    return a_flat, b_flat, scale_flat, c_flat  # keep alive

# Run in sweep order to reproduce heap state
refs = []
for M in [1, 8, 32, 64, 128, 256, 512, 1024]:
    refs.append(analyze_tensors(M))
