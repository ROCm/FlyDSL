#!/usr/bin/env python3
"""MLA FA kernel host launcher. Run from flydsl-kernels/ dir after sourcing env_ffm.sh."""

import sys, os

_this_dir = os.path.dirname(os.path.abspath(__file__))

# Detect which location we're running from:
_fly_kernel = os.path.join(_this_dir, "..", "..", "..", "kernels", "mla_gfx1250.py")
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

import importlib.util, math, torch
import flydsl.expr as fx, flydsl.compiler as flyc
from flydsl._mlir import ir
from flydsl.compiler.jit_function import CompilationContext

spec = importlib.util.spec_from_file_location("mla_fa_kernel", os.path.join(_kernel_dir, "mla_gfx1250.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
kernel = mod.mla_attn_fwd_pipelined_kernel

# Config
B, H, SQ, SK = 1, 1, 128, 512
HEAD_SZ_NOPE = 128
HEAD_SZ_ROPE = 64
HEAD_SZ_QK = HEAD_SZ_NOPE + HEAD_SZ_ROPE  # 192
HEAD_SZ_V = 128
BLOCK_M = 128
sm_scale = 1.0 / (HEAD_SZ_QK**0.5)

torch.manual_seed(42)
q = torch.randn((B, H, SQ, HEAD_SZ_QK), dtype=torch.bfloat16).cuda()
k = torch.randn((B, H, SK, HEAD_SZ_QK), dtype=torch.bfloat16).cuda()
v = torch.randn((B, H, SK, HEAD_SZ_V), dtype=torch.bfloat16).cuda()
o = torch.zeros((B, H, SQ, HEAD_SZ_V), dtype=torch.float32).cuda()

# Reference
q_nope = q[:, :, :, :HEAD_SZ_NOPE].float()
q_rope = q[:, :, :, HEAD_SZ_NOPE:].float()
k_nope = k[:, :, :, :HEAD_SZ_NOPE].float()
k_rope = k[:, :, :, HEAD_SZ_NOPE:].float()
qk = (torch.matmul(q_nope, k_nope.transpose(-2, -1)) + torch.matmul(q_rope, k_rope.transpose(-2, -1))) * sm_scale
p = torch.softmax(qk, dim=-1)
ref = torch.matmul(p, v.float())


@flyc.jit
def launch(q, k, v, o, sq_z, sq_h, sq_m, sk_z, sk_h, sk_n, sv_z, sv_h, sv_n, so_z, so_h, so_m):
    mod._lds_allocator.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        mod._lds_allocator.finalize()
    launcher = kernel(q, k, v, o, sq_z, sq_h, sq_m, sk_z, sk_h, sk_n, sv_z, sv_h, sv_n, so_z, so_h, so_m)
    launcher.launch(grid=[B, H, SQ // BLOCK_M], block=[128, 1, 1])


print(f"FlyDSL MLA FA kernel: B={B} H={H} SQ={SQ} SK={SK}")
print(f"  HEAD_SZ_NOPE={HEAD_SZ_NOPE} HEAD_SZ_ROPE={HEAD_SZ_ROPE} HEAD_SZ_V={HEAD_SZ_V}")
print(f"Grid: [{B}, {H}, {SQ // BLOCK_M}]  Block: [128, 1, 1]")
print("Compiling and running...")

launch(
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

o_cpu = o.cpu().float()
ref_cpu = ref.cpu().float()
print(f"  o   : mean={o_cpu.mean():.4f} max={o_cpu.max():.4f} min={o_cpu.min():.4f}")
print(f"  ref : mean={ref_cpu.mean():.4f} max={ref_cpu.max():.4f} min={ref_cpu.min():.4f}")

max_err = (o_cpu - ref_cpu).abs().max().item()
print(f"  max_err={max_err:.6f}")
print("PASS" if max_err < 0.05 else f"FAIL: max_err={max_err}")
