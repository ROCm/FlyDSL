"""HIP module executor for the experimental ASM backend.

This bypasses MLIR ExecutionEngine and launches GPU kernels directly via HIP
module APIs, using the `gpu.launch_func` site in the lowered MLIR to determine
grid/block sizes and kernel argument wiring.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Optional


def _ssa(value) -> str:
    try:
        if value is not None and hasattr(value, "get_name"):
            n = value.get_name()
            if n:
                return str(n)
    except Exception:
        pass
    return str(value)


@dataclass(frozen=True)
class _LaunchPlan:
    kernel_name: str
    # MLIR `gpu.launch_func` operands: first 6 are grid/block dims, remaining are kernargs.
    grid_vals: tuple[Any, Any, Any]
    block_vals: tuple[Any, Any, Any]
    kernarg_vals: tuple[Any, ...]


class AsmHipExecutor:
    """Callable executor that launches a single `gpu.launch_func` kernel."""

    def __init__(
        self,
        *,
        hsaco_bytes: bytes,
        launch_plan: _LaunchPlan,
    ):
        self._hsaco_bytes = hsaco_bytes
        self._launch_plan = launch_plan
        self._hip = None
        self._mod = ctypes.c_void_p()
        self._fn = ctypes.c_void_p()

    def _ensure_loaded(self):
        if self._hip is not None:
            return

        hip = ctypes.CDLL("libamdhip64.so")

        hip.hipInit.argtypes = [ctypes.c_uint]
        hip.hipInit.restype = ctypes.c_int
        hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        hip.hipModuleLoad.restype = ctypes.c_int
        hip.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        hip.hipModuleGetFunction.restype = ctypes.c_int
        hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        hip.hipModuleLaunchKernel.restype = ctypes.c_int

        rc = hip.hipInit(0)
        if rc != 0:
            raise RuntimeError(f"hipInit failed: {rc}")

        # HIP module APIs load from file path; persist HSACO bytes to a temp file.
        # Keep the file for the lifetime of this executor.
        td = tempfile.mkdtemp(prefix="flydsl_asm_hsaco_")
        hsaco_path = os.path.join(td, "kernel.hsaco")
        with open(hsaco_path, "wb") as f:
            f.write(self._hsaco_bytes)

        rc = hip.hipModuleLoad(ctypes.byref(self._mod), hsaco_path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"hipModuleLoad failed: {rc}")

        rc = hip.hipModuleGetFunction(
            ctypes.byref(self._fn), self._mod, self._launch_plan.kernel_name.encode("utf-8")
        )
        if rc != 0:
            raise RuntimeError(
                f"hipModuleGetFunction({self._launch_plan.kernel_name!r}) failed: {rc}"
            )

        self._hip = hip

    @staticmethod
    def _as_int(x) -> int:
        # Accept python ints and small scalar tensors.
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return int(x)
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                return int(x.item())
            except Exception:
                pass
        return int(x)

    @staticmethod
    def _as_device_ptr(x) -> int:
        if hasattr(x, "data_ptr") and callable(getattr(x, "data_ptr")):
            return int(x.data_ptr())
        raise TypeError(f"Expected a tensor-like with .data_ptr(); got {type(x)}")

    def __call__(self, *args):
        # Late import torch so CPU-only envs can still import flydsl.
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("ASM backend requires torch for stream + device pointers") from e

        self._ensure_loaded()

        # Evaluate grid/block dims. These are either constants or derived from user args.
        grid_x = self._as_int(self._launch_plan.grid_vals[0](*args) if callable(self._launch_plan.grid_vals[0]) else self._launch_plan.grid_vals[0])
        grid_y = self._as_int(self._launch_plan.grid_vals[1](*args) if callable(self._launch_plan.grid_vals[1]) else self._launch_plan.grid_vals[1])
        grid_z = self._as_int(self._launch_plan.grid_vals[2](*args) if callable(self._launch_plan.grid_vals[2]) else self._launch_plan.grid_vals[2])

        block_x = self._as_int(self._launch_plan.block_vals[0](*args) if callable(self._launch_plan.block_vals[0]) else self._launch_plan.block_vals[0])
        block_y = self._as_int(self._launch_plan.block_vals[1](*args) if callable(self._launch_plan.block_vals[1]) else self._launch_plan.block_vals[1])
        block_z = self._as_int(self._launch_plan.block_vals[2](*args) if callable(self._launch_plan.block_vals[2]) else self._launch_plan.block_vals[2])

        # Pack kernel parameters (void*[] of pointers to each argument storage).
        storages = []
        params = []

        def _push_ptr(p: int):
            v = ctypes.c_void_p(int(p))
            storages.append(v)
            params.append(ctypes.cast(ctypes.byref(v), ctypes.c_void_p))

        def _push_i64(v: int):
            vv = ctypes.c_int64(int(v))
            storages.append(vv)
            params.append(ctypes.cast(ctypes.byref(vv), ctypes.c_void_p))

        # Evaluate kernargs from launch plan: they reference user args by index.
        for ka in self._launch_plan.kernarg_vals:
            if callable(ka):
                v = ka(*args)
            else:
                v = ka
            # Heuristic: tensors -> device ptr, scalars -> i64.
            if hasattr(v, "data_ptr") and callable(getattr(v, "data_ptr")):
                _push_ptr(self._as_device_ptr(v))
            else:
                _push_i64(self._as_int(v))

        params_arr = (ctypes.c_void_p * len(params))(*params)

        # Launch on current torch stream.
        stream = torch.cuda.current_stream().cuda_stream
        stream_h = ctypes.c_void_p(int(stream))

        rc = self._hip.hipModuleLaunchKernel(
            self._fn,
            ctypes.c_uint(grid_x),
            ctypes.c_uint(grid_y),
            ctypes.c_uint(grid_z),
            ctypes.c_uint(block_x),
            ctypes.c_uint(block_y),
            ctypes.c_uint(block_z),
            ctypes.c_uint(0),
            stream_h,
            params_arr,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"hipModuleLaunchKernel failed: {rc}")


def build_launch_plan_from_module(module) -> _LaunchPlan:
    """Extract a single `gpu.launch_func` plan from a lowered MLIR module.

    This is intentionally minimal and only supports the patterns used by
    `tests/kernels/test_vec_add.py` and `tests/kernels/test_softmax.py`:
    - a single `gpu.launch_func` in `func.func @__call__`
    - grid/block sizes computed from arith.{constant,addi,subi,muli,divsi,divui,select,cmpi}
      over `__call__` args and constants
    """
    from _mlir import ir  # type: ignore
    from _mlir.dialects import arith as arith_d  # type: ignore
    from _mlir.dialects import func as func_d  # type: ignore
    from _mlir.dialects import gpu as gpu_d  # type: ignore

    # Find __call__ and its first launch.
    call_fn = None
    for op in module.operation.regions[0].blocks[0].operations:
        if isinstance(op, func_d.FuncOp) and op.sym_name.value == "__call__":
            call_fn = op
            break
    if call_fn is None:
        raise ValueError("ASM backend compile expects a host wrapper func.func @__call__")

    entry = call_fn.entry_block
    launch = None
    for op in entry.operations:
        if isinstance(op, gpu_d.LaunchFuncOp):
            launch = op
            break
    if launch is None:
        raise ValueError("ASM backend compile expects a gpu.launch_func in __call__")

    # Parse kernel name like: @mod::@fn
    launch_attrs = {a.name: a.attr for a in launch.attributes}
    kernel_attr = launch_attrs.get("kernel")
    kernel_ref = str(kernel_attr) if kernel_attr is not None else ""
    kernel_name = kernel_ref.split("::")[-1].lstrip("@").strip()
    if not kernel_name:
        raise ValueError(f"Could not parse kernel name from gpu.launch_func kernel={kernel_ref!r}")

    # Build a def map for simple SSA evaluation in __call__.
    defs = {}
    for op in entry.operations:
        try:
            results = list(op.operation.results)
        except Exception:
            results = []
        if len(results) == 1:
            defs[_ssa(results[0])] = op

    arg_ssas = [_ssa(a) for a in entry.arguments]
    arg_idx = {n: i for i, n in enumerate(arg_ssas)}

    def eval_ssa(v):
        # Return either a constant int, or a callable(args)->int.
        if isinstance(v, ir.Value):
            name = _ssa(v)
        else:
            name = _ssa(v)

        # Block argument -> runtime argument by position.
        if name in arg_idx:
            i = int(arg_idx[name])
            return lambda *user_args: user_args[i]

        # Constant.
        op = defs.get(name)
        if op is None:
            raise ValueError(f"Cannot evaluate SSA {name}: no def in __call__")

        if isinstance(op, arith_d.ConstantOp):
            # Parse leading integer.
            txt = str(op.value).strip()
            n = ""
            for ch in txt:
                if ch in "+-" and not n:
                    n += ch
                    continue
                if ch.isdigit():
                    n += ch
                    continue
                break
            if not n or n in {"+", "-"}:
                raise ValueError(f"Non-integer arith.constant for {name}: {txt!r}")
            return int(n)

        # Binary integer ops.
        def _bin(opv, fn):
            a = eval_ssa(opv.operands[0])
            b = eval_ssa(opv.operands[1])
            if callable(a) or callable(b):
                return lambda *user_args: fn(
                    AsmHipExecutor._as_int(a(*user_args) if callable(a) else a),
                    AsmHipExecutor._as_int(b(*user_args) if callable(b) else b),
                )
            return fn(int(a), int(b))

        if isinstance(op, arith_d.AddIOp):
            return _bin(op, lambda a, b: a + b)
        if isinstance(op, arith_d.SubIOp):
            return _bin(op, lambda a, b: a - b)
        if isinstance(op, arith_d.MulIOp):
            return _bin(op, lambda a, b: a * b)
        if isinstance(op, arith_d.DivSIOp):
            return _bin(op, lambda a, b: int(a // b))
        if isinstance(op, arith_d.DivUIOp):
            return _bin(op, lambda a, b: int(a // b))

        if isinstance(op, arith_d.CmpIOp):
            pred = str(op.predicate).strip()
            a = eval_ssa(op.lhs)
            b = eval_ssa(op.rhs)
            def _cmp(x, y):
                if "ult" in pred:
                    return int(x < y)
                if "ule" in pred:
                    return int(x <= y)
                if "ugt" in pred:
                    return int(x > y)
                if "uge" in pred:
                    return int(x >= y)
                if "eq" in pred:
                    return int(x == y)
                if "ne" in pred:
                    return int(x != y)
                raise ValueError(f"Unsupported cmpi predicate: {pred}")
            if callable(a) or callable(b):
                return lambda *user_args: _cmp(
                    AsmHipExecutor._as_int(a(*user_args) if callable(a) else a),
                    AsmHipExecutor._as_int(b(*user_args) if callable(b) else b),
                )
            return _cmp(int(a), int(b))

        if isinstance(op, arith_d.SelectOp):
            c = eval_ssa(op.condition)
            t = eval_ssa(op.true_value)
            f = eval_ssa(op.false_value)
            return lambda *user_args: (
                t(*user_args) if AsmHipExecutor._as_int(c(*user_args) if callable(c) else c) else f(*user_args)
            )

        raise ValueError(f"Unsupported __call__ op for grid eval: {op.operation.name}")

    # Launch operand layout:
    #   0..2: grid sizes (blocks in)
    #   3..5: block sizes (threads in)
    #   6.. : kernel args
    if len(launch.operands) < 6:
        raise ValueError("gpu.launch_func operand count < 6")

    grid_vals = (eval_ssa(launch.operands[0]), eval_ssa(launch.operands[1]), eval_ssa(launch.operands[2]))
    block_vals = (eval_ssa(launch.operands[3]), eval_ssa(launch.operands[4]), eval_ssa(launch.operands[5]))
    kernarg_vals = tuple(eval_ssa(v) if _ssa(v) in arg_idx or _ssa(v) in defs else (lambda *user_args, vv=v: vv) for v in launch.operands[6:])

    return _LaunchPlan(
        kernel_name=kernel_name,
        grid_vals=grid_vals,
        block_vals=block_vals,
        kernarg_vals=kernarg_vals,
    )

