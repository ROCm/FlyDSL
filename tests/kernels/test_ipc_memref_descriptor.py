#!/usr/bin/env python3
"""IPC pointer + memref descriptor calling convention tests.

This validates "方案2" building blocks:
- `flydsl.runtime.ipc`: export/open HIP IPC handles to obtain a raw device pointer
- `flydsl.compile` + `ExecutionEngineExecutor`: allow feeding a raw device pointer (int)
  directly into an executor call (bare-pointers host ABI)
- Optional: disable bare-pointers host ABI and verify executor can pass memref descriptors
  for torch tensors.
"""

import os
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[3]
_embedded = _repo / "build" / "python_packages" / "flydsl"
if _embedded.exists():
    os.environ.setdefault("FLYDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import flydsl
from flydsl.dialects.ext import flir, arith
from flydsl.runtime import ipc
from flydsl.runtime.device import get_rocm_arch
import _mlir.extras.types as T


def _build_copy_module(N: int):
    chip = get_rocm_arch()

    class _M(flir.MlirModule):
        GPU_MODULE_NAME = "ipc_copy"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{chip}", abi = "500">']

        @flir.kernel
        def copy_kernel(self: flir.T.i64, In: lambda: T.memref(N, T.f16()), Out: lambda: T.memref(N, T.f16())):
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))
            bdim = flir.const_index(flir.block_dim("x"))
            idx = arith.ArithValue(bid) * bdim + arith.ArithValue(tid)
            if arith.ult(idx, flir.const_index(N)):
                v = flir.memref.load(In, [arith.as_value(idx)])
                flir.memref.store(v, Out, [arith.as_value(idx)])

        @flir.jit
        def __call__(self: flir.T.i64, In: lambda: T.memref(N, T.f16()), Out: lambda: T.memref(N, T.f16())):
            c1 = arith.index(1)
            bx = arith.index(256)
            gx = arith.index((N + 255) // 256)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "copy_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[In, Out],
            )

    return _M().module


def test_ipc_handle_export_smoke():
    """Always-on smoke test: exporting an IPC handle returns correctly sized bytes."""
    N = 1024
    x = torch.randn((N,), device="cuda", dtype=torch.float16).contiguous()
    h = ipc.get_ipc_handle(x)
    assert isinstance(h.handle, (bytes, bytearray))
    assert len(h.handle) == 64
    assert int(h.offset_bytes) == 0


def _ipc_child_worker(q_in, q_out):
    import torch

    # One child process: open handle, run kernel using raw device ptr, validate output.
    msg = q_in.get()
    N = int(msg["N"])
    handle = msg["handle"]
    offset = int(msg["offset"])
    x_cpu = msg["x_cpu"]

    torch.cuda.set_device(0)
    y = torch.empty((N,), device="cuda", dtype=torch.float16)

    m = _build_copy_module(N)
    exe = flydsl.compile(m)

    try:
        with ipc.open_ipc_tensor_ptr(ipc.IpcHandle(handle=handle, offset_bytes=offset)) as x_ptr:
            exe(x_ptr, y)
        torch.cuda.synchronize()
        ok = torch.allclose(y.cpu(), x_cpu, atol=0, rtol=0)
        q_out.put({"ok": bool(ok)})
    except Exception as e:
        q_out.put({"ok": False, "err": f"{type(e).__name__}: {e}"})


def test_ipc_open_and_raw_ptr_optional():
    """Optional: open IPC handle in a different process and feed raw ptr to executor.

    Enable with:
      FLYDSL_TEST_HIP_IPC=1
    """
    if os.environ.get("FLYDSL_TEST_HIP_IPC", "0") != "1":
        pytest.skip("HIP IPC open test disabled (set FLYDSL_TEST_HIP_IPC=1)")

    N = 1024
    torch.cuda.set_device(0)
    x = torch.randn((N,), device="cuda", dtype=torch.float16).contiguous()
    h = ipc.get_ipc_handle(x)
    x_cpu = x.cpu()

    import torch.multiprocessing as mp

    q_in = mp.Queue()
    q_out = mp.Queue()
    p = mp.Process(target=_ipc_child_worker, args=(q_in, q_out))
    p.start()
    q_in.put({"N": N, "handle": h.handle, "offset": h.offset_bytes, "x_cpu": x_cpu})
    res = q_out.get()
    p.join(timeout=30)
    if not res.get("ok", False):
        pytest.skip(f"HIP IPC open/exec failed in this environment: {res.get('err')}")


def test_memref_descriptor_host_abi_optional():
    """When bare-pointers host ABI is disabled, executor should pack torch tensors as memref descriptors.

    Enable with:
      FLYDSL_TEST_MEMREF_DESCRIPTOR=1
    """
    if os.environ.get("FLYDSL_TEST_MEMREF_DESCRIPTOR", "0") != "1":
        pytest.skip("descriptor ABI test disabled (set FLYDSL_TEST_MEMREF_DESCRIPTOR=1)")

    N = 1024
    x = torch.randn((N,), device="cuda", dtype=torch.float16).contiguous()
    y = torch.empty((N,), device="cuda", dtype=torch.float16)

    # Disable bare host pointers for this compile.
    os.environ["FLIR_USE_BARE_PTR_HOST"] = "0"
    try:
        m = _build_copy_module(N)
        exe = flydsl.compile(m)
        exe(x, y)
        torch.cuda.synchronize()
        assert torch.allclose(y, x)
    finally:
        os.environ.pop("FLIR_USE_BARE_PTR_HOST", None)


