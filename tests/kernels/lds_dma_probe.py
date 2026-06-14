# SPDX-License-Identifier: Apache-2.0
"""Minimal standalone root-cause probe for fx.rocdl.buffer_load_to_lds (global->LDS DMA).

Copies a tile of int32 from a global buffer into LDS via buffer_load_to_lds, barriers,
then each thread reads back from LDS and stores to a global output. Compare on CPU.

MODE env selects the addressing variant under test:
  ok      : uniform LDS base ptr, per-lane voffset = tid*4   (hypothesis: correct)
  perlane : per-lane LDS ptr (base + tid*4), voffset = tid*4 (v12/ck style; suspected wrong)
  perm    : uniform LDS base, voffset = (tid^1)*4            (diagnostic: is LDS dest = lane order?)
  uniform_voff0 : uniform LDS base, voffset = 0              (diagnostic: all lanes read inp[0]?)

Run:  HIP_VISIBLE_DEVICES=6 MODE=ok python3 tests/kernels/lds_dma_probe.py
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "kernels"))

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

MODE = os.environ.get("MODE", "ok")
NTHREADS = int(os.environ.get("NT", "64"))  # 256 => 4 waves, exercises per-wave M0
N = NTHREADS           # number of i32 dwords copied

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="lds_dma_probe_smem")
_LDS_BYTES = N * 4
_alloc.ptr = _LDS_BYTES

_VMCNT0 = 0x3F70  # s_waitcnt vmcnt(0): wait for outstanding VMEM incl. buffer_load_to_lds DMA


@flyc.kernel(known_block_size=[NTHREADS, 1, 1])
def probe_kernel(IN: fx.Tensor, OUT: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)

    rin = fx.buffer_ops.create_buffer_resource(IN)
    rout = fx.buffer_ops.create_buffer_resource(OUT)

    lds = SmemPtr(_alloc.get_base(), 0, fx.typing.T.i32, shape=(N,)).get()
    lds_base = memref.extract_aligned_pointer_as_index(lds)
    lds_ptr_base = fx.buffer_ops.create_llvm_ptr(arith.index_cast(fx.typing.T.i64, lds_base), address_space=3)

    if fx.const_expr(MODE == "perlane"):
        # v12/ck style: per-lane LDS destination pointer.
        lds_ptr = fx.buffer_ops.get_element_ptr(lds_ptr_base, byte_offset=fx.Index(tid * fx.Int32(4)))
        voff = tid * fx.Int32(4)
    elif fx.const_expr(MODE == "perm"):
        lds_ptr = lds_ptr_base
        voff = (tid ^ fx.Int32(1)) * fx.Int32(4)
    elif fx.const_expr(MODE == "uniform_voff0"):
        lds_ptr = lds_ptr_base
        voff = fx.Int32(0)
    elif fx.const_expr(MODE == "mw"):
        # Multi-wave correct pattern: each wave's M0 must be its own uniform base
        # (base + wave*64*4). voffset stays the per-lane global fetch addr (tid*4).
        wave = tid // fx.Int32(64)
        lds_ptr = fx.buffer_ops.get_element_ptr(lds_ptr_base, byte_offset=fx.Index(wave * fx.Int32(64 * 4)))
        voff = tid * fx.Int32(4)
    elif fx.const_expr(MODE == "perlane_scatter"):
        # per-lane LDS dest = base + (tid^1)*4 (disagrees with natural lane order),
        # voffset = tid*4. If per-lane ptr is RESPECTED: out[k]=inp[k^1]. If IGNORED
        # (hw uses M0+lane*4): out[k]=inp[k] (identity). This is the v12/ck failure mode.
        lds_ptr = fx.buffer_ops.get_element_ptr(lds_ptr_base, byte_offset=fx.Index((tid ^ fx.Int32(1)) * fx.Int32(4)))
        voff = tid * fx.Int32(4)
    else:  # "ok"
        lds_ptr = lds_ptr_base
        voff = tid * fx.Int32(4)

    fx.rocdl.buffer_load_to_lds(rin, lds_ptr, voff, size_bytes=4)
    fx.rocdl.s_waitcnt(_VMCNT0)
    fx.gpu.barrier()

    val = memref.load(lds, [fx.Index(tid).ir_value()])
    fx.buffer_ops.buffer_store(val, rout, tid)


@flyc.jit
def run_probe(IN: fx.Tensor, OUT: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    probe_kernel(IN, OUT).launch(grid=(1,), block=(NTHREADS,), stream=stream)


def main():
    torch.manual_seed(0)
    inp = torch.arange(N, dtype=torch.int32, device="cuda")
    out = torch.full((N,), -1, dtype=torch.int32, device="cuda")
    run_probe(inp, out)
    torch.cuda.synchronize()
    inp_c = inp.cpu()
    out_c = out.cpu()
    if MODE == "perm":
        expect = inp_c.clone()
        expect = expect.view(-1).clone()
        idx = torch.arange(N)
        expect = inp_c[idx ^ 1]
    elif MODE == "perlane_scatter":
        idx = torch.arange(N)
        identity = inp_c
        respected = inp_c[idx ^ 1]
        print(f"MODE={MODE}")
        print("in :", inp_c[:16].tolist())
        print("out:", out_c[:16].tolist())
        print(f"if ptr RESPECTED -> {respected[:16].tolist()}")
        print(f"if ptr IGNORED   -> {identity[:16].tolist()}")
        if torch.equal(out_c, identity):
            print("VERDICT: per-lane LDS pointer IGNORED; hw writes LDS[M0_uniform + lane*4]")
        elif torch.equal(out_c, respected):
            print("VERDICT: per-lane LDS pointer RESPECTED")
        else:
            print("VERDICT: neither -> something else")
        return
    elif MODE == "uniform_voff0":
        expect = None
    else:
        expect = inp_c
    print(f"MODE={MODE}")
    print("in :", inp_c[:16].tolist())
    print("out:", out_c[:16].tolist())
    if expect is not None:
        exact = torch.equal(out_c, expect)
        nmis = int((out_c != expect).sum())
        print(f"expect:{expect[:16].tolist()}")
        print(f"RESULT: {'BIT-EXACT PASS' if exact else 'FAIL'}  mismatches={nmis}/{N}")
    else:
        print("(diagnostic mode, no fixed expectation)")


if __name__ == "__main__":
    main()
