#!/usr/bin/env python3
"""B=1 M=64 K=4096 N=128 run for csim — 1 WG, exercises full 32-K-tile loop."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250

# Single WG: gx=1, gy=1, gz=1
# tile_n == group_n is a hard constraint (bench uses TILE_N=128=GROUP_N)
# K=4096 → 32 K-tiles → tests full prologue+main(29)+tail pipeline
B, K, N = 1, 4096, 128
GROUP_K, GROUP_N = 128, 128
TILE_K, TILE_N = 128, 128
M = 64
tile_m = 64   # gx=M//tile_m=1
nb = 3        # K/tile_k=32 >= 3 OK

print(f"Config: B={B} M={M} K={K} N={N} tile_m={tile_m} tile_n={TILE_N} nb={nb}")
print(f"Grid: gx={M//tile_m} gy={N//TILE_N} gz={B} -> {M//tile_m * N//TILE_N * B} WGs total")

torch.manual_seed(0)
a = torch.randn((B, M, K), dtype=torch.bfloat16).cuda().contiguous()
b = torch.zeros((B, K, N), dtype=torch.float8_e4m3fn).cuda().contiguous()
scale_fp32 = torch.rand((B, K // GROUP_K, N // GROUP_N)) * 0.1 + 0.01
log2_s = torch.log2(scale_fp32.clamp(min=1e-38))
e8m0 = (log2_s.round().to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
scale = e8m0.cuda().contiguous()
c = torch.zeros((B, M, N), dtype=torch.bfloat16).cuda()

a_flat = a.view(-1); b_flat = b.view(-1)
scale_flat = scale.view(-1); c_flat = c.view(-1)

def ptr_lo_hi(t):
    addr = t.data_ptr()
    return addr & 0xFFFFFFFF, addr >> 32

a_lo, a_hi = ptr_lo_hi(a_flat)
b_lo, b_hi = ptr_lo_hi(b_flat)
c_lo, c_hi = ptr_lo_hi(c_flat)

print(f"A ptr: 0x{a_hi:08x}_{a_lo:08x}  size={a_flat.nbytes}B")
print(f"B ptr: 0x{b_hi:08x}_{b_lo:08x}  size={b_flat.nbytes}B")
print(f"C ptr: 0x{c_hi:08x}_{c_lo:08x}  size={c_flat.nbytes}B")

# B addr_lo carry analysis: adv_b = TILE_K * N = 128*128 = 16384 per step
adv_b = TILE_K * N  # 0x4000
n_k_tiles = K // TILE_K  # 32
for bz in range(B):
    base_lo = (b_lo + bz * K * N) & 0xFFFFFFFF
    lo = base_lo
    for step in range(n_k_tiles):
        old = lo
        lo = (lo + adv_b) & 0xFFFFFFFF
        if lo < old:
            where = "PRO" if step < (nb-1) else "MAIN"
            print(f"  B carry: bz={bz} lo=0x{base_lo:08x} at step {step} ({where})")

fn = compile_bmm_a16w8_gfx1250(
    B=B, M=M, N=N, K=K,
    group_k=GROUP_K, group_n=GROUP_N,
    tile_m=tile_m, tile_n=TILE_N, tile_k=TILE_K,
    m_warp=2, n_warp=2, num_buffers=nb, cluster_n=1,
    use_e8m0_scale=True)

print("Launching kernel...")
stream = torch.cuda.current_stream()
fn(c_flat, a_flat, b_flat, scale_flat, M, stream)
torch.cuda.synchronize()
print("OK - csim K=4096 full-loop test passed")
