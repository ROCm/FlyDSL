#!/usr/bin/env python3
"""Standalone runner for the generated FA FlyDSL kernel.

Config: B=1, H=1, SQ=128, SK=1024, D=128
"""

import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))

# Detect which location we're running from:
_fly_kernel = os.path.join(_this_dir, "..", "..", "..", "kernels", "fa_gfx1250.py")
if os.path.exists(_fly_kernel):
    _REPO_ROOT = os.path.abspath(os.path.join(_this_dir, "..", "..", ".."))
    _kernel_dir = os.path.join(_REPO_ROOT, "kernels")
else:
    _REPO_ROOT = os.path.abspath(os.path.join(_this_dir, "..", "..", "flydsl-prev"))
    _kernel_dir = _this_dir

_BUILD = os.path.join(_REPO_ROOT, "build-fly", "python_packages")
_SRC = os.path.join(_REPO_ROOT, "python")
sys.path = [p for p in sys.path if p != _SRC]
sys.path.insert(0, _BUILD)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.compiler as flyc  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl.compiler.jit_function import CompilationContext  # noqa: E402

# Import the generated kernel
sys.path.insert(0, _kernel_dir)
import fa_gfx1250 as fa_kernel  # noqa: E402

kernel = fa_kernel.attn_fwd_kernel

# ── Tensors ──────────────────────────────────────────────────────────────────
B, H, SQ, SK, D = 1, 1, 128, 1024, 128
torch.manual_seed(42)
q = torch.randn((B, H, SQ, D), dtype=torch.bfloat16).cuda()
k = torch.randn((B, H, SK, D), dtype=torch.bfloat16).cuda()
v = torch.randn((B, H, SK, D), dtype=torch.bfloat16).cuda()
o = torch.zeros((B, H, SQ, D), dtype=torch.float32).cuda()

sm_scale = 1.0 / (D**0.5)
ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=sm_scale)

# ── Launcher ─────────────────────────────────────────────────────────────────


@flyc.jit
def launch_fa(q, k, v, o, sq_z, sq_h, sq_m, sk_z, sk_h, sk_n, sv_z, sv_h, sv_n, so_z, so_h, so_m):
    fa_kernel._lds_allocator.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        fa_kernel._lds_allocator.finalize()
    launcher = kernel(q, k, v, o, sq_z, sq_h, sq_m, sk_z, sk_h, sk_n, sv_z, sv_h, sv_n, so_z, so_h, so_m)
    launcher.launch(grid=[B, H, SQ // 128], block=[128, 1, 1])


# ── Run ──────────────────────────────────────────────────────────────────────
print(f"FlyDSL FA kernel: B={B} H={H} SQ={SQ} SK={SK} D={D}")
print(f"Grid: [{B}, {H}, {SQ // 128}]  Block: [128, 1, 1]")
print("Compiling and running...")

launch_fa(
    q,
    k,
    v,
    o,
    q.stride(0),
    q.stride(1),
    q.stride(2),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    v.stride(0),
    v.stride(1),
    v.stride(2),
    o.stride(0),
    o.stride(1),
    o.stride(2),
)
torch.cuda.synchronize()

print("Kernel done. Checking numerics...")
o_cpu = o.cpu().float()
ref_cpu = ref.cpu().float()
print(f"  o   : mean={o_cpu.mean():.4f} max={o_cpu.max():.4f} min={o_cpu.min():.4f}")
print(f"  ref : mean={ref_cpu.mean():.4f} max={ref_cpu.max():.4f} min={ref_cpu.min():.4f}")

try:
    torch.testing.assert_close(o_cpu, ref_cpu, rtol=0.005, atol=0.005)
    print("PASS: numerics match PyTorch reference!")
except AssertionError as e:
    max_err = (o_cpu - ref_cpu).abs().max().item()
    print(f"FAIL: max abs error = {max_err:.6f}")
    print(str(e)[:500])
