#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
def _path_key(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _resolve_flydsl_paths():
    candidates = []
    for env_name in ("FLYDSL_ROOT", "FLYDSL_PREV_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(os.path.abspath(os.path.expanduser(value)))

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.abspath(os.path.join(here, "..")),
            os.path.abspath(os.path.join(here, "..", "..")),
            os.path.abspath(os.path.join(here, "..", "..", "flydsl-prev")),
        ]
    )

    seen = set()
    for root in candidates:
        key = _path_key(root)
        if key in seen:
            continue
        seen.add(key)
        build_path = os.path.join(root, "build-fly", "python_packages")
        source_path = os.path.join(root, "python")
        if os.path.isdir(build_path) and os.path.isdir(source_path):
            return build_path, source_path

    raise RuntimeError(
        "Could not locate FlyDSL. Set FLYDSL_ROOT to the flydsl-prev root directory."
    )


_build, _src = _resolve_flydsl_paths()
_remove_keys = {_path_key(_build), _path_key(_src)}
sys.path = [path for path in sys.path if _path_key(path) not in _remove_keys]
sys.path.insert(0, _build)

_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import flydsl.compiler as flyc
from flydsl._mlir import ir
from flydsl.compiler.jit_function import CompilationContext

from kernels.mha_pingpong_bf16_gfx1250 import attn_fwd_pingpong_pipelined_kernel as kernel
from kernels import mha_pingpong_bf16_gfx1250 as mha_mod

B = 1
H = 1
SQ = 256
SK = 512
HEAD_SZ_QK = 192
HEAD_SZ_V = 128
BLOCK_M = 256
BLOCK_THREADS = 256
sm_scale = 1.0 / (HEAD_SZ_QK ** 0.5)

torch.manual_seed(42)
q = torch.randn((B, H, SQ, HEAD_SZ_QK), dtype=torch.bfloat16).cuda()
k = torch.randn((B, H, SK, HEAD_SZ_QK), dtype=torch.bfloat16).cuda()
v = torch.randn((B, H, SK, HEAD_SZ_V), dtype=torch.bfloat16).cuda()
o = torch.zeros((B, H, SQ, HEAD_SZ_V), dtype=torch.float32).cuda()

qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
p = torch.softmax(qk, dim=-1)
ref = torch.matmul(p, v.float())

@flyc.jit
def launch_mha_fa(q, k, v, o, sq_z, sq_h, sq_m, sk_z, sk_h, sk_n, sv_z, sv_h, sv_n, so_z, so_h, so_m):
    if hasattr(mha_mod, "_lds_allocator"):
        mha_mod._lds_allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            mha_mod._lds_allocator.finalize()
    launcher = kernel(q, k, v, o, sq_z, sq_h, sq_m, sk_z, sk_h, sk_n, sv_z, sv_h, sv_n, so_z, so_h, so_m)
    launcher.launch(grid=[B, H, (SQ + BLOCK_M - 1) // BLOCK_M], block=[BLOCK_THREADS, 1, 1])

print("FlyDSL MHA kernel:")
print(f"  variant='pingpong' B={B} H={H} SQ={SQ} SK={SK}")
print(f"  HEAD_SZ_QK={HEAD_SZ_QK} HEAD_SZ_V={HEAD_SZ_V}")
print(f"  Grid: [{B}, {H}, {(SQ + BLOCK_M - 1) // BLOCK_M}]  Block: [{BLOCK_THREADS}, 1, 1]")
print("Compiling and running...")

launch_mha_fa(
    q, k, v, o,
    q.stride(0), q.stride(1), q.stride(2),
    k.stride(0), k.stride(1), k.stride(2),
    v.stride(0), v.stride(1), v.stride(2),
    o.stride(0), o.stride(1), o.stride(2),
)
torch.cuda.synchronize()

o_cpu = o.cpu().float()
ref_cpu = ref.cpu().float()
max_err = (o_cpu - ref_cpu).abs().max().item()
print(f"  o   : mean={o_cpu.mean():.4f} max={o_cpu.max():.4f} min={o_cpu.min():.4f}")
print(f"  ref : mean={ref_cpu.mean():.4f} max={ref_cpu.max():.4f} min={ref_cpu.min():.4f}")
print(f"  max_err={max_err:.6f}")
torch.testing.assert_close(o_cpu, ref_cpu, rtol=0.005, atol=0.005)
print("PASS")
