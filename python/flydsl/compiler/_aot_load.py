# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Implementation of ``flyc.load_module``: the loading counterpart of ``export_to_c``."""

import ctypes
import json
import operator
import os
import sys
import weakref
from pathlib import Path
from typing import Union

from .._mlir._mlir_libs._mlirDialectsFly import DLTensorAdaptor, _host_target_triple, _HostObject
from ..expr import numeric
from ..expr.typing import Stream
from ..utils._elf import _defined_global_symbols, _symbol_contents
from .jit_argument import PointerJitArg, _LayoutPlan
from .jit_executor import _EMPTY, CallState, _resolve_runtime_libs
from .jit_function import CompiledFunction
from .protocol import c_abi_spec

_SCHEMA_VERSION = 3
_METADATA_SUFFIX = "__metadata"

# Scalar slot (ctype, encoding) -> the Numeric type whose fill packs it.
_SCALAR_TYPES = {
    ("bool", None): numeric.Boolean,
    ("int8", None): numeric.Int8,
    ("int16", None): numeric.Int16,
    ("int32", None): numeric.Int32,
    ("int64", None): numeric.Int64,
    ("uint8", None): numeric.Uint8,
    ("uint16", None): numeric.Uint16,
    ("uint32", None): numeric.Uint32,
    ("uint64", None): numeric.Uint64,
    ("float", None): numeric.Float32,
    ("double", None): numeric.Float64,
    ("uint16", "float16_bits"): numeric.Float16,
    ("uint16", "bfloat16_bits"): numeric.BFloat16,
}

_NOT_LOADED = -2
_CURRENT_DEVICE = -1  # module_load argument selecting the current device
_STATUS_MESSAGES = {-1: "module not initialized", _NOT_LOADED: "module not loaded on the current device"}


def _dlpack(t):
    """DLPack view of a tensor argument that is not a torch tensor, or of its FlyDSL wrapper."""
    return DLTensorAdaptor((t.dltensor if hasattr(t, "dltensor") else t).__dlpack__())


def _tensor_data_fill(a, s):
    t = a.torch_tensor if hasattr(a, "torch_tensor") else a
    s.value = t.data_ptr() if hasattr(t, "data_ptr") else _dlpack(t).data_ptr


def _tensor_layout_fill(slot):
    plan = _LayoutPlan(tuple(slot["shape_dims"]), tuple(slot["stride_dims"]), slot.get("stride_bits") == 32)
    # One function per slot with the dimensions unrolled, like the JIT's tensor fills.
    terms = [f"sh[{d}]" for d in plan.shape] + [f"st[{d}]" for d in plan.stride]
    src = (
        "def fill(a, s, _pack=_pack, _dlpack=_dlpack):\n"
        "    t = a.torch_tensor if hasattr(a, 'torch_tensor') else a\n"
        "    if hasattr(t, 'stride'):\n"
        "        sh, st = t.shape, t.stride()\n"
        "    else:\n"
        "        ad = _dlpack(t)\n"
        "        sh, st = ad.shape, ad.stride\n"
        f"    _pack(s, 0, {', '.join(terms)})\n"
    )
    ns = {"_pack": plan.codec.pack_into, "_dlpack": _dlpack}
    exec(compile(src, "<flydsl-load-module-fill>", "exec"), ns)
    return plan.buf_ctype, ns["fill"]


def _slot_spec(slot):
    """``(ctype, fill)`` packing one metadata ABI slot from a launch argument."""
    kind = slot["kind"]
    if kind == "tensor_data":
        ctype, fill = ctypes.c_void_p, _tensor_data_fill
    elif kind == "tensor_layout":
        ctype, fill = _tensor_layout_fill(slot)
    elif kind == "stream":
        ((ctype, fill),) = c_abi_spec(Stream())
    elif kind == "pointer":
        ((ctype, fill),) = c_abi_spec(PointerJitArg(numeric.Int8, None))
    elif kind == "scalar":
        numeric_type = _SCALAR_TYPES.get((slot["ctype"], slot.get("encoding")))
        if numeric_type is None:
            raise ValueError(f"flyc.load_module: unsupported scalar slot {slot['ctype']!r}")
        ((ctype, fill),) = c_abi_spec(numeric_type(0))
    else:
        raise ValueError(f"flyc.load_module: unsupported ABI slot kind {kind!r}")
    if ctypes.sizeof(ctype) != slot["size"]:
        raise ValueError(f"flyc.load_module: {kind} slot packs {ctypes.sizeof(ctype)} bytes, expected {slot['size']}")
    field = slot.get("field")
    if field is not None:
        get_field = operator.attrgetter(field)

        def fill(a, s, _fill=fill, _get=get_field):
            _fill(_get(a), s)

    return ctype, fill


def _check_status(symbol, status):
    if status:
        reason = _STATUS_MESSAGES.get(status, f"HIP error {status}")
        raise RuntimeError(f"{symbol} failed with status {status} ({reason})")


class _LoadedModule:
    """Keeps a loaded object linked and its GPU modules loaded while a function uses it."""

    def __init__(self, host_object, unload):
        # The finalizer keeps the object linked until its modules are unloaded.
        # It is skipped at interpreter exit, when the device runtime may already be gone.
        weakref.finalize(self, lambda keep=host_object: unload()).atexit = False

    def max_blocks_per_cu(self) -> int:
        raise NotImplementedError("max_blocks_per_cu() is not available for a function from flyc.load_module")


def load_module(path: Union[str, "os.PathLike[str]"]) -> CompiledFunction:
    """Load an object written by :meth:`CompiledFunction.export_to_c` into this process.

    The object is linked in memory, with no system linker or shared library on
    disk. Its GPU module is loaded on the current device now and on any other
    device the first time it is called there. The returned :class:`CompiledFunction`
    takes the launcher's arguments positionally, like the one ``flyc.compile``
    returns; constexpr arguments are ignored and omitted trailing arguments take
    their recorded defaults. Linux x86-64 only.
    """
    if not sys.platform.startswith("linux"):
        raise NotImplementedError("flyc.load_module: only Linux ELF objects are supported")
    data = Path(path).read_bytes()
    candidates = [name for name in _defined_global_symbols(data) if name.endswith(_METADATA_SUFFIX)]
    if len(candidates) != 1:
        raise ValueError(f"flyc.load_module: {path} is not an object written by export_to_c")

    metadata = json.loads(_symbol_contents(data, candidates[0]).split(b"\0", 1)[0])
    if metadata.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"flyc.load_module: {path} has metadata schema {metadata.get('schema_version')}, "
            f"expected {_SCHEMA_VERSION}; export it again with this FlyDSL version"
        )
    if metadata["host_triple"] != _host_target_triple():
        raise ValueError(f"flyc.load_module: {path} targets host {metadata['host_triple']}")
    slot_specs = []
    for slot in metadata["abi"]:
        if slot["kind"] == "stream" and slot.get("arg_index") is None:
            slot_specs.append((-1, ctypes.c_void_p, None))  # auto-stream: NULL selects the default stream
        else:
            slot_specs.append((slot["arg_index"], *_slot_spec(slot)))
    params = tuple((p["name"], p.get("default", _EMPTY)) for p in metadata["params"])

    # Everything above can reject the object; only now is it linked. Resolving the
    # runtime libraries also rejects a backend this build does not have.
    host_object = _HostObject(data, _resolve_runtime_libs(metadata["backend"]))

    def lifecycle(key, *argtypes):
        symbol = metadata[key]
        fn = ctypes.CFUNCTYPE(ctypes.c_int32, *argtypes)(host_object.lookup(symbol))
        return lambda *args: _check_status(symbol, fn(*args))

    symbol = metadata["symbol"]
    entry = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)(host_object.lookup(symbol))
    module_init = lifecycle("module_init_symbol")
    module_load = lifecycle("module_load_symbol", ctypes.c_int32)
    module_unload = lifecycle("module_unload_symbol")

    module_init()
    # From here on, dropping `loaded` unloads every device the module was loaded on.
    loaded = _LoadedModule(host_object, module_unload)
    module_load(_CURRENT_DEVICE)

    def func_exe(packed, _entry=entry, _symbol=symbol, _load=module_load):
        status = _entry(packed)
        if status == _NOT_LOADED:
            # First call on this device: nothing was launched, so load and launch again.
            _load(_CURRENT_DEVICE)
            status = _entry(packed)
        if status:
            _check_status(_symbol, status)

    compiled = CompiledFunction(CallState(tuple(slot_specs), func_exe, params=params), loaded)
    compiled._backend = metadata["backend"]
    compiled._aot_error = (RuntimeError, "a function from flyc.load_module cannot be exported again")
    return compiled
