#!/usr/bin/env python3
"""Tiny BMM run for csim — small B/M/K/N to complete quickly on cycle-accurate sim."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250

# Tiny dims: 2 WGs in M, 1 WG in N, 2 batches -> 4 WGs total
B, K, N = 2, 512, 128
GROUP_K, GROUP_N = 128, 128
TILE_K, TILE_N = 128, 128
M = 128
tile_m = 64   # -> gx=2, gy=1, gz=2 -> 4 WGs total
nb = 3        # needs K/tile_k=4 >= nb=3 OK

print(f"Config: B={B} M={M} K={K} N={N} tile_m={tile_m} nb={nb}")
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

b_lo, b_hi = ptr_lo_hi(b_flat)
a_lo, a_hi = ptr_lo_hi(a_flat)
c_lo, c_hi = ptr_lo_hi(c_flat)

print(f"A ptr: 0x{a_hi:08x}_{a_lo:08x}  size={a_flat.nbytes}B")
print(f"B ptr: 0x{b_hi:08x}_{b_lo:08x}  size={b_flat.nbytes}B")
print(f"C ptr: 0x{c_hi:08x}_{c_lo:08x}  size={c_flat.nbytes}B")

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
print("OK - csim small test passed")
