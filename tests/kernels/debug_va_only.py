#!/usr/bin/env python3
"""Print VA for each M in bench-sweep order, no kernel run. Fast diagnosis."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

B, K, N = 16, 4096, 1024
GROUP_K, GROUP_N = 128, 128

def ptr_lo_hi(t):
    addr = t.data_ptr()
    return addr & 0xFFFFFFFF, addr >> 32

def check_b_overflow(b_lo, label=""):
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

def check_c_overflow(c_lo, M_pad, label=""):
    """Check C TDM store addr_lo overflow (adv = N*elem_bytes = 1024*2 = 2048 bytes)."""
    # C is [B, M_pad, N] bf16; each warp tile row advance = N*2 = 2048 bytes
    # Kernel uses warp_tile_m rows per warp, each row advances addr_lo_c by N*2
    adv_c = N * 2  # 2048 bytes per row
    rows_per_warp = 64  # warp_tile_m (tile_m=64, m_warp=2 -> each warp gets 32 rows... hmm)
    # Actually TDM store writes warp_tile_m x warp_tile_n tile
    # adv per step in TDM store = elem_bytes * pad_interval = 2 * N = 2048
    # steps = warp_tile_m = tile_m // m_warp = 64 // 2 = 32 rows
    steps = 32
    any_overflow = False
    for bz in range(B):
        # This is approximate - actual offset depends on blk_m, warp_m_off
        addr_lo = (c_lo + bz * M_pad * N * 2) & 0xFFFFFFFF
        lo = addr_lo
        for step in range(steps):
            old = lo
            lo = (lo + adv_c) & 0xFFFFFFFF
            if lo < old:
                where = "C-store"
                print(f"  {label}C bz={bz:2d} lo=0x{addr_lo:08x} overflow at step {step} ({where})")
                any_overflow = True
    return any_overflow

def alloc_and_print(M, tile_m, nb, label, free_after=True):
    torch.manual_seed(0)
    M_pad = ((M + tile_m - 1) // tile_m) * tile_m
    a = torch.randn((B, M_pad, K), dtype=torch.bfloat16).cuda().contiguous()
    b = torch.zeros((B, K, N), dtype=torch.float8_e4m3fn).cuda().contiguous()
    e8m0 = torch.zeros((B, K // GROUP_K, N // GROUP_N), dtype=torch.uint8).cuda()
    c = torch.zeros((B, M_pad, N), dtype=torch.bfloat16).cuda()

    a_flat = a.view(-1); b_flat = b.view(-1)
    scale_flat = e8m0.view(-1); c_flat = c.view(-1)

    b_lo, b_hi = ptr_lo_hi(b_flat)
    a_lo, a_hi = ptr_lo_hi(a_flat)
    c_lo, c_hi = ptr_lo_hi(c_flat)
    print(f"\n{'='*60}")
    print(f"{label} M={M} tile_m={tile_m} nb={nb}")
    print(f"  A ptr: 0x{a_hi:08x}_{a_lo:08x}  size={a_flat.nbytes//1024//1024}MB")
    print(f"  B ptr: 0x{b_hi:08x}_{b_lo:08x}  size={b_flat.nbytes//1024//1024}MB")
    print(f"  C ptr: 0x{c_hi:08x}_{c_lo:08x}  size={c_flat.nbytes//1024//1024}MB")

    b_overflow = check_b_overflow(b_lo, label=f"B ")
    c_overflow = check_c_overflow(c_lo, M_pad, label=f"C ")
    if not b_overflow and not c_overflow:
        print(f"  → no overflow detected")

    if not free_after:
        return a, b, e8m0, c  # keep alive

M_SWEEP = [
    (1,    64, 3, "M=1   dec", True),
    (8,    64, 3, "M=8   dec", True),
    (32,   64, 3, "M=32  dec", True),
    (64,   64, 3, "M=64  dec", True),
    (128,  64, 3, "M=128 dec", True),
    (256, 128, 2, "M=256 pre", True),
    (512,  64, 3, "M=512 pre", True),
    (1024, 64, 3, "M=1024 pre", False),  # keep last one to see final VA
]

refs = []
for M, tile_m, nb, label, free_after in M_SWEEP:
    result = alloc_and_print(M, tile_m, nb, label, free_after)
    if result is not None:
        refs.append(result)
