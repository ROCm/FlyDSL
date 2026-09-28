# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Internal implementation of ``CompiledFunction.export_to_c``."""

import ctypes
import dataclasses
import fcntl
import filecmp
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .._mlir import ir
from .._mlir._mlir_libs._mlirDialectsFly import _emit_host_object, _host_target_triple
from .._mlir.dialects import llvm
from ..expr.numeric import BFloat16, Float16, Float32, Float64, Numeric
from ..expr.struct import _runtime_fields, is_struct_type
from ..expr.typing import Stream
from ..runtime._libraries import _find_runtime_libraries, _RuntimeLibrary
from ..utils._elf import _defined_global_symbols
from .backends import _get_backend_class
from .jit_argument import MemRefJitArg, PointerJitArg
from .jit_function import _create_mlir_context
from .protocol import c_abi_spec

_SCHEMA_VERSION = 1

_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INT_CTYPE = re.compile(r"u?int(8|16|32|64)")

_HOST_OPT_LEVEL = 2


@dataclass(frozen=True)
class _AbiSlot:
    """One host-storage entry in the packed ``void **args`` ABI."""

    arg_index: Optional[int]
    arg_name: Optional[str]
    kind: str
    ctype: str
    size: int
    alignment: int
    shape_dims: Optional[Tuple[int, ...]] = None
    stride_dims: Optional[Tuple[int, ...]] = None
    shape_bits: Optional[int] = None
    stride_bits: Optional[int] = None
    encoding: Optional[str] = None
    field: Optional[str] = None

    def _to_dict(self) -> Dict[str, Any]:
        """JSON-compatible dict; optional fields are omitted when unset."""
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is None and f.default is None:
                continue
            d[f.name] = list(v) if isinstance(v, tuple) else v
        return d


@dataclass(frozen=True)
class _ReturnAbi:
    """Return value of the exported entry and lifecycle functions."""

    kind: str = "status"
    ctype: str = "int32"
    success: int = 0

    def _to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class _ExportMetadata:
    schema_version: int
    symbol: str
    module_init_symbol: str
    module_load_symbol: str
    module_unload_symbol: str
    metadata_symbol: str
    runtime_libraries: Tuple[_RuntimeLibrary, ...]
    abi: Tuple[_AbiSlot, ...]
    return_abi: _ReturnAbi
    backend: str
    arch: str
    host_triple: str
    flydsl_version: str

    def _to_dict(self) -> Dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["runtime_libraries"] = [{"soname": lib.soname} for lib in self.runtime_libraries]
        d["abi"] = [slot._to_dict() for slot in self.abi]
        d["return_abi"] = self.return_abi._to_dict()
        return d


_NUMERIC_ENCODINGS = {Float16: "float16_bits", BFloat16: "bfloat16_bits"}
_NUMERIC_FLOAT_CTYPES = {Float32: "float", Float64: "double"}


def _semantic_slots(name: str, jit_arg, field: Optional[str] = None) -> List[Dict[str, Any]]:
    """Describe one launcher's ABI slots in ``c_abi_spec`` order."""
    if is_struct_type(type(jit_arg)):
        return [
            slot
            for sub_name, value in _runtime_fields(jit_arg)
            for slot in _semantic_slots(name, value, sub_name if field is None else f"{field}.{sub_name}")
        ]
    slots = _leaf_semantic_slots(name if field is None else f"{name}.{field}", jit_arg)
    if field is not None:
        for slot in slots:
            slot["field"] = field
    return slots


def _leaf_semantic_slots(name: str, jit_arg) -> List[Dict[str, Any]]:
    if isinstance(jit_arg, MemRefJitArg):
        slots = [{"kind": "tensor_data", "ctype": "pointer"}]
        if jit_arg.is_layout_dynamic:
            slots.append(
                {
                    "kind": "tensor_layout",
                    "ctype": "bytes",
                    "shape_dims": tuple(jit_arg.shape_dyn_indices),
                    "stride_dims": tuple(jit_arg.stride_dyn_indices),
                }
            )
        return slots
    if isinstance(jit_arg, Stream):
        return [{"kind": "stream", "ctype": "pointer"}]
    if isinstance(jit_arg, PointerJitArg):
        return [{"kind": "pointer", "ctype": "pointer"}]
    if isinstance(jit_arg, Numeric):
        cls = type(jit_arg)
        if jit_arg.signed is not None and jit_arg.width in (1, 8, 16, 32, 64):
            ctype = "bool" if jit_arg.width == 1 else f"{'int' if jit_arg.signed else 'uint'}{jit_arg.width}"
            return [{"kind": "scalar", "ctype": ctype}]
        if cls in _NUMERIC_FLOAT_CTYPES:
            return [{"kind": "scalar", "ctype": _NUMERIC_FLOAT_CTYPES[cls]}]
        if cls in _NUMERIC_ENCODINGS:
            return [{"kind": "scalar", "ctype": "uint16", "encoding": _NUMERIC_ENCODINGS[cls]}]
        raise NotImplementedError(f"flyc.compile: argument {name!r} has unsupported scalar type {cls.__name__}")
    raise NotImplementedError(
        f"flyc.compile: argument {name!r} has unsupported type {type(jit_arg).__name__}; "
        "AOT export supports tensors, pointers, scalars and streams"
    )


def _is_type(ty: ir.Type, cls) -> bool:
    return ty.typeid == cls.static_typeid


def _integer_width(ty: ir.Type) -> Optional[int]:
    return ir.IntegerType(ty).width if _is_type(ty, ir.IntegerType) else None


def _lowered_type_matches(semantic: Dict[str, Any], ty: ir.Type) -> bool:
    """Whether the lowered entry parameter ``ty`` carries the slot's C type."""
    ctype = semantic["ctype"]
    if ctype == "pointer":
        return _is_type(ty, llvm.PointerType)
    if ctype == "bytes":
        return _is_type(ty, llvm.StructType)
    if ctype == "bool":
        return _integer_width(ty) == 1
    if ctype == "float":
        return _is_type(ty, ir.F32Type)
    if ctype == "double":
        return _is_type(ty, ir.F64Type)
    encoding = semantic.get("encoding")
    if encoding == "float16_bits":
        return _is_type(ty, ir.F16Type)
    if encoding == "bfloat16_bits":
        return _is_type(ty, ir.BF16Type)
    match = _INT_CTYPE.fullmatch(ctype)
    return match is not None and _integer_width(ty) == int(match.group(1))


def _host_size_align(ty: ir.Type) -> Tuple[int, int]:
    """Host (LP64) size and alignment of a lowered LLVM parameter type."""
    if _is_type(ty, llvm.PointerType):
        return ctypes.sizeof(ctypes.c_void_p), ctypes.alignment(ctypes.c_void_p)
    width = _integer_width(ty)
    if width is not None:
        size = {1: 1, 8: 1, 16: 2, 32: 4, 64: 8}.get(width)
        if size is None:
            raise NotImplementedError(f"flyc.compile: unsupported host integer type {ty}")
        return size, size
    for float_type, size in ((ir.F16Type, 2), (ir.BF16Type, 2), (ir.F32Type, 4), (ir.F64Type, 8)):
        if _is_type(ty, float_type):
            return size, size
    if _is_type(ty, llvm.StructType):
        st = llvm.StructType(ty)
        size, align = 0, 1
        for elem in st.body:
            esize, ealign = _host_size_align(elem)
            if not st.packed:
                size = (size + ealign - 1) // ealign * ealign
                align = max(align, ealign)
            size += esize
        return (size + align - 1) // align * align, align
    raise NotImplementedError(f"flyc.compile: unsupported host argument type {ty}")


def _struct_leaf_widths(ty: ir.Type) -> List[int]:
    if _is_type(ty, llvm.StructType):
        return [w for elem in llvm.StructType(ty).body for w in _struct_leaf_widths(elem)]
    width = _integer_width(ty)
    if width is None:
        raise NotImplementedError(f"flyc.compile: unsupported tensor layout element type {ty}")
    return [width]


def _group_width(widths: List[int], what: str, name: str) -> Optional[int]:
    if not widths:
        return None
    if len(set(widths)) != 1:
        raise NotImplementedError(f"flyc.compile: tensor {name!r} has mixed-width dynamic {what} values {widths}")
    return widths[0]


def _build_abi(sig, param_names, jit_args, has_user_stream, entry_arg_types: List[ir.Type]) -> Tuple[_AbiSlot, ...]:
    """Build and validate the packed C ABI against the lowered entry."""
    param_index = {name: i for i, name in enumerate(sig.parameters)}
    user_args = jit_args[: len(param_names)]
    if sum(isinstance(a, Stream) for a in user_args) > 1:
        raise NotImplementedError("flyc.compile: a launcher may declare at most one stream argument")

    pending = []  # (arg_index, arg_name, semantic slot, ctypes type used by JIT dispatch)
    for name, jit_arg in zip(param_names, user_args):
        semantic = _semantic_slots(name, jit_arg)
        ctypes_seq = [ct for ct, _fill in c_abi_spec(jit_arg)]
        if len(ctypes_seq) != len(semantic):
            raise NotImplementedError(
                f"flyc.compile: argument {name!r} ({type(jit_arg).__name__}) packs {len(ctypes_seq)} C ABI slots, "
                f"expected {len(semantic)}"
            )
        pending.extend((param_index[name], name, s, ct) for s, ct in zip(semantic, ctypes_seq))
    if not has_user_stream:
        pending.append((None, None, {"kind": "stream", "ctype": "pointer"}, ctypes.c_void_p))

    if len(pending) != len(entry_arg_types):
        raise RuntimeError(
            f"flyc.compile: launcher packs {len(pending)} C ABI slots but the lowered entry "
            f"takes {len(entry_arg_types)} arguments"
        )

    slots = []
    for (arg_index, arg_name, semantic, ctype), ty in zip(pending, entry_arg_types):
        where = f"argument {arg_name!r} slot {semantic['kind']}"
        if not _lowered_type_matches(semantic, ty):
            raise RuntimeError(f"flyc.compile: {where} ({semantic['ctype']}) does not match lowered type {ty}")
        size, alignment = _host_size_align(ty)
        if ctypes.sizeof(ctype) != size:
            raise RuntimeError(
                f"flyc.compile: {where} packs {ctypes.sizeof(ctype)} bytes but the lowered type {ty} has {size}"
            )
        extra = {}
        if semantic["kind"] == "tensor_layout":
            widths = _struct_leaf_widths(ty)
            n_shape = len(semantic["shape_dims"])
            if len(widths) != n_shape + len(semantic["stride_dims"]):
                raise RuntimeError(f"flyc.compile: tensor {arg_name!r} layout {ty} does not match its dynamic dims")
            extra["shape_bits"] = _group_width(widths[:n_shape], "shape", arg_name)
            extra["stride_bits"] = _group_width(widths[n_shape:], "stride", arg_name)
        slots.append(_AbiSlot(arg_index, arg_name, size=size, alignment=alignment, **semantic, **extra))
    return tuple(slots)


def _make_aot_compiled(module, entry, sig, param_names, jit_args, has_user_stream, backend, arch, *, ir_text=None):
    """Build an exportable artifact from an already-lowered module."""
    entry_op = ir.SymbolTable(module.operation)[entry]
    entry_arg_types = [arg.type for arg in entry_op.regions[0].blocks[0].arguments]
    abi = _build_abi(sig, param_names, jit_args, has_user_stream, entry_arg_types)
    return _AOTCompiledFunction(
        ir_text=str(module) if ir_text is None else ir_text,
        entry=entry,
        abi=abi,
        backend=backend,
        arch=arch,
        host_triple=_host_target_triple(),
    )


def _aot_from_jit_artifact(
    artifact, *, sig, param_names, jit_args, has_user_stream, backend: str, arch: str
) -> "_AOTCompiledFunction":
    """Create an export view from an existing lowered JIT artifact."""
    if artifact._post_load_processors:
        raise NotImplementedError(
            "export_to_c: launchers with post-load processors (Python callbacks run after GPU "
            "module load) cannot be exported"
        )
    with _create_mlir_context():
        module = ir.Module.parse(artifact._ir_text)
        return _make_aot_compiled(
            module,
            artifact._entry,
            sig,
            param_names,
            jit_args,
            has_user_stream,
            backend,
            arch,
            ir_text=artifact._ir_text,
        )


def _linkage(name: str) -> ir.Attribute:
    return ir.Attribute.parse(f"#llvm.linkage<{name}>")


def _is_definition(op) -> bool:
    """Return whether a top-level symbol op is a definition."""
    name = op.name
    if name == "llvm.func":
        return len(op.regions[0].blocks) > 0
    if name == "llvm.mlir.global":
        return "value" in op.attributes or len(op.regions[0].blocks) > 0
    return name == "gpu.binary"


def _namespace_symbols(module: ir.Module, prefix: str) -> Dict[str, str]:
    """Prefix and internalize definitions while preserving external declarations."""
    renamed = {}
    for op in list(module.body.operations):
        op = op.operation
        if "sym_name" not in op.attributes or not _is_definition(op):
            continue
        old = ir.StringAttr(op.attributes["sym_name"]).value
        new = f"{prefix}__{old}"
        if new in ir.SymbolTable(module.operation):
            raise RuntimeError(f"export_to_c: symbol {new!r} already exists")
        ir.SymbolTable.replace_all_symbol_uses(old, new, module.operation)
        ir.SymbolTable.set_symbol_name(op, new)
        if op.name in ("llvm.func", "llvm.mlir.global"):
            op.attributes["linkage"] = _linkage("internal")
        renamed[old] = new
    return renamed


def _add_entry_wrapper(module: ir.Module, symbol: str, target: str, abi, take_error: str) -> None:
    """Define ``int32_t symbol(void **args)``: load each slot from its storage,
    call ``target``, and return the first runtime error recorded during the call."""
    table = ir.SymbolTable(module.operation)
    arg_types = [arg.type for arg in table[target].regions[0].blocks[0].arguments]
    if len(arg_types) != len(abi):
        raise RuntimeError(f"export_to_c: {target} takes {len(arg_types)} arguments, ABI has {len(abi)} slots")

    ptr = llvm.PointerType.get()
    i32 = ir.IntegerType.get_signless(32)
    with ir.InsertionPoint(module.body), ir.Location.unknown():
        if take_error not in table:
            llvm.LLVMFuncOp(take_error, ir.TypeAttr.get(llvm.FunctionType.get(i32, [])))
        wrapper = llvm.LLVMFuncOp(symbol, ir.TypeAttr.get(llvm.FunctionType.get(i32, [ptr])))
        block = wrapper.body.blocks.append(ptr)
        with ir.InsertionPoint(block):
            # Drop errors left by earlier calls on this thread so the status is ours.
            llvm.CallOp(i32, [], [], [], callee=take_error)
            values = []
            for i, (ty, slot) in enumerate(zip(arg_types, abi)):
                slot_addr = llvm.GEPOp(ptr, block.arguments[0], [], [i], ptr, None).result
                storage = llvm.LoadOp(ptr, slot_addr).result
                values.append(llvm.LoadOp(ty, storage, alignment=slot.alignment).result)
            llvm.CallOp(None, values, [], [], callee=target)
            status = llvm.CallOp(i32, [], [], [], callee=take_error).result
            llvm.ReturnOp(arg=status)


def _add_metadata_global(module: ir.Module, symbol: str, metadata: bytes) -> None:
    """Define ``const char symbol[]`` holding ``metadata`` plus a NUL terminator."""
    data = metadata + b"\0"
    array = llvm.ArrayType.get(ir.IntegerType.get_signless(8), len(data))
    with ir.InsertionPoint(module.body), ir.Location.unknown():
        llvm.GlobalOp(array, symbol, _linkage("external"), constant=True, value=ir.StringAttr.get(data))


def _emit_export_object(
    ir_text: str,
    entry: str,
    abi: Tuple[_AbiSlot, ...],
    function_name: str,
    metadata_symbol: str,
    metadata: bytes,
    backend_cls,
) -> bytes:
    """Rewrite a fresh copy of the lowered module for export and emit its host object."""
    with _create_mlir_context():
        module = ir.Module.parse(ir_text)
        renamed = _namespace_symbols(module, function_name)
        handler = ir.Attribute.parse(backend_cls._aot_offloading_handler(function_name))
        for op in module.body.operations:
            if op.operation.name == "gpu.binary":
                op.operation.attributes["offloadingHandler"] = handler
        _add_entry_wrapper(module, function_name, renamed[entry], abi, backend_cls._aot_take_error_symbol())
        _add_metadata_global(module, metadata_symbol, metadata)
        module.operation.verify()
        return _emit_host_object(module.operation, _HOST_OPT_LEVEL)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically; permissions follow the umask."""
    tmp = _temporary_sibling(path)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _publish_file(source: Path, destination: Path) -> None:
    """Publish atomically, accepting an existing byte-identical file."""
    if destination.is_symlink():
        raise RuntimeError(f"export_to_c: runtime library collision: {destination} is a symbolic link")
    if destination.exists() and destination.samefile(source):
        return
    tmp = _temporary_sibling(destination)
    try:
        shutil.copy2(source, tmp)
        try:
            os.link(tmp, destination)
        except FileExistsError:
            if destination.is_symlink() or not filecmp.cmp(tmp, destination, shallow=False):
                raise RuntimeError(
                    f"export_to_c: runtime library collision: {destination} exists with content "
                    f"different from {source}"
                ) from None
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _output_directory_lock(directory: Path):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


_HEADER_CTYPES = {
    "pointer": "void *",
    "bool": "bool",
    "float": "float",
    "double": "double",
    **{f"{s}int{w}": f"{s}int{w}_t" for s in ("", "u") for w in (8, 16, 32, 64)},
}

_C_RESERVED = frozenset(
    "alignas alignof and and_eq asm atomic_cancel atomic_commit atomic_noexcept auto bitand bitor bool break case "
    "catch char char8_t char16_t char32_t class co_await co_return co_yield compl concept const consteval constexpr "
    "constinit const_cast continue contract_assert decltype default delete do double dynamic_cast else enum explicit "
    "export extern false float for friend goto if import inline int long module mutable namespace new noexcept not "
    "not_eq nullptr operator or or_eq private protected public reflexpr register reinterpret_cast requires restrict return short "
    "signed sizeof static static_assert static_cast struct switch synchronized template this thread_local throw true "
    "try typedef typeid typename typeof typeof_unqual union unsigned using virtual void volatile wchar_t while xor "
    "xor_eq _Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local".split()
)

_HEADER_ARGS_ARRAY = "flydsl_args_"


def _is_implementation_reserved(name: str, *, file_scope: bool) -> bool:
    """Whether ``name`` is reserved to the C/C++ implementation."""
    return name.startswith("__") or (len(name) > 1 and name[0] == "_" and (file_scope or name[1].isupper()))


def _validate_export_symbol(name) -> None:
    """Require a source-level C/C++ function name, not only an ELF symbol."""
    if (
        not isinstance(name, str)
        or not _C_IDENTIFIER.fullmatch(name)
        or name in _C_RESERVED
        or _is_implementation_reserved(name, file_scope=True)
    ):
        raise ValueError(f"export_to_c: function_name must be a non-reserved C identifier, got {name!r}")


def _path_identity(path: Path) -> str:
    """Stable identity for detecting aliases among paths that may not exist yet."""
    return os.path.normcase(str(path.resolve(strict=False)))


def _check_output_path_collisions(object_path: Path, header_path: Optional[Path], installed, bundled) -> None:
    """Reject output aliases that could overwrite another artifact or runtime."""
    outputs = {"object": object_path}
    if header_path is not None:
        outputs["header"] = header_path
    identities = {}
    for kind, path in outputs.items():
        identity = _path_identity(path)
        if identity in identities:
            raise ValueError(f"export_to_c: {kind} path aliases {identities[identity]} path: {path}")
        identities[identity] = kind

    installed_paths = {_path_identity(Path(lib.path)) for lib in installed}
    for kind, path in outputs.items():
        if _path_identity(path) in installed_paths:
            raise ValueError(f"export_to_c: {kind} path would overwrite an installed runtime library: {path}")

    for lib in bundled:
        identity = _path_identity(Path(lib.path))
        if identity in identities:
            raise ValueError(
                f"export_to_c: bundled runtime library {lib.soname!r} would overwrite the "
                f"{identities[identity]} path: {lib.path}"
            )
        identities[identity] = f"bundled runtime library {lib.soname!r}"


def _resolve_export_paths(file_path, file_name, function_prefix):
    directory = Path(file_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"export_to_c: output directory does not exist: {directory}")
    if not isinstance(file_name, str) or not file_name or file_name in (".", "..") or Path(file_name).name != file_name:
        raise ValueError(f"export_to_c: file_name must be a non-empty base name, got {file_name!r}")
    symbol = file_name if function_prefix in (None, "") else function_prefix
    return directory / f"{file_name}.o", symbol, directory / f"{file_name}.h"


def _render_header(metadata: _ExportMetadata) -> str:
    sym = metadata.symbol
    macro = sym.upper()
    guard = f"FLYDSL_AOT_{macro}_H"
    types, params, arg_exprs = [], [], []
    used = {_HEADER_ARGS_ARRAY}
    used_types = set()

    def param_name(base: str) -> str:
        name = base + "_" if base in _C_RESERVED else base
        if _is_implementation_reserved(name, file_scope=False):
            name = f"flydsl{name}"
        while name in used:
            name += "_"
        used.add(name)
        return name

    for slot in metadata.abi:
        base = slot.arg_name if slot.arg_name is not None else "stream"
        if slot.field is not None:
            base = f"{base}_{slot.field.replace('.', '_')}"
        if slot.kind == "tensor_layout":
            tname = f"{sym}_{base}_layout_t"
            while tname in used_types:
                tname += "_"
            used_types.add(tname)
            fields = []
            if slot.shape_dims:
                fields.append(f"int{slot.shape_bits}_t shape[{len(slot.shape_dims)}];")
            if slot.stride_dims:
                fields.append(f"int{slot.stride_bits}_t stride[{len(slot.stride_dims)}];")
            types += [
                f"/* Dynamic shape dims {list(slot.shape_dims)} and stride dims {list(slot.stride_dims)} of "
                f"'{slot.arg_name}'. */",
                f"typedef struct __attribute__((packed)) {{ {' '.join(fields)} }} {tname};",
                f'FLYDSL_AOT_STATIC_ASSERT(sizeof({tname}) == {slot.size}, "ABI size of {tname}");',
            ]
            name = param_name(f"{base}_layout")
            params.append(f"const {tname} *{name}")
            arg_exprs.append(f"(void *){name}")
            continue
        name = param_name(f"{base}_data" if slot.kind == "tensor_data" else base)
        ctype = _HEADER_CTYPES[slot.ctype]
        params.append(f"{ctype}{'' if ctype.endswith('*') else ' '}{name}")
        arg_exprs.append(f"(void *)&{name}")

    return "\n".join(
        [
            "/* Generated by FlyDSL export_to_c. Do not edit. */",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <stdbool.h>",
            "#include <stdint.h>",
            "",
            "#ifdef __cplusplus",
            "#define FLYDSL_AOT_STATIC_ASSERT static_assert",
            'extern "C" {',
            "#else",
            "#define FLYDSL_AOT_STATIC_ASSERT _Static_assert",
            "#endif",
            "",
            "#ifndef FLYDSL_AOT_SUCCESS",
            "#define FLYDSL_AOT_SUCCESS 0",
            "#define FLYDSL_AOT_ERR_NOT_INITIALIZED (-1)",
            "#define FLYDSL_AOT_ERR_NOT_LOADED (-2)",
            "#endif",
            "",
            f"#define {macro}_SCHEMA_VERSION {metadata.schema_version}",
            f"#define {macro}_NUM_ARGS {len(metadata.abi)}",
            "",
            *types,
            "",
            "/* Packed entry: args[i] points to storage for ABI slot i. Returns 0 once all launches are submitted. */",
            f"int32_t {sym}(void **args);",
            f"int32_t {metadata.module_init_symbol}(void);",
            f"int32_t {metadata.module_load_symbol}(int32_t device);",
            f"int32_t {metadata.module_unload_symbol}(void);",
            f"extern const char {metadata.metadata_symbol}[];",
            "",
            f"static inline int32_t {sym}_call({', '.join(params) or 'void'}) {{",
            f"  void *{_HEADER_ARGS_ARRAY}[{max(len(arg_exprs), 1)}] = {{{', '.join(arg_exprs) or '0'}}};",
            f"  return {sym}({_HEADER_ARGS_ARRAY});",
            "}",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            f"#endif /* {guard} */",
            "",
        ]
    )


class _AOTCompiledFunction:
    """Lowered export state without references to example arguments."""

    def __init__(
        self, *, ir_text: str, entry: str, abi: Tuple[_AbiSlot, ...], backend: str, arch: str, host_triple: str
    ):
        self._ir_text = ir_text
        self._entry = entry
        self._abi = abi
        self._backend = backend
        self._arch = arch
        self._host_triple = host_triple

    def _export_to_c(
        self,
        file_path: Union[str, "os.PathLike[str]"],
        file_name: str,
        function_prefix: str = "",
    ) -> None:
        """Write ``<file_name>.o`` and ``<file_name>.h`` into ``file_path``."""
        import flydsl

        object_path, function_name, header_path = _resolve_export_paths(file_path, file_name, function_prefix)
        _validate_export_symbol(function_name)
        if self._host_triple != _host_target_triple():
            raise NotImplementedError(
                f"export_to_c: artifact targets host {self._host_triple}; cross-compiling from "
                f"{_host_target_triple()} is not supported"
            )

        backend_cls = _get_backend_class(self._backend)
        lifecycle = backend_cls._aot_module_symbols(function_name)
        metadata_symbol = f"{function_name}__metadata"
        exported_symbols = {function_name, metadata_symbol, *lifecycle.values()}
        installed = _find_runtime_libraries(self._backend)
        runtime_symbols = {sym for lib in installed for sym in _defined_global_symbols(lib.path, dynamic=True)}
        clashes = sorted(exported_symbols & runtime_symbols)
        if clashes:
            raise ValueError(f"export_to_c: function_name {function_name!r} collides with runtime symbols {clashes}")
        output_directory = object_path.parent.resolve()
        libraries = tuple(_RuntimeLibrary(str(output_directory / lib.soname), lib.soname) for lib in installed)
        _check_output_path_collisions(object_path, header_path, installed, libraries)

        export_metadata = _ExportMetadata(
            schema_version=_SCHEMA_VERSION,
            symbol=function_name,
            module_init_symbol=lifecycle["init"],
            module_load_symbol=lifecycle["load"],
            module_unload_symbol=lifecycle["unload"],
            metadata_symbol=metadata_symbol,
            runtime_libraries=libraries,
            abi=self._abi,
            return_abi=_ReturnAbi(),
            backend=self._backend,
            arch=self._arch,
            host_triple=self._host_triple,
            flydsl_version=flydsl.__version__,
        )
        embedded = export_metadata._to_dict()
        metadata = json.dumps(embedded, sort_keys=True, separators=(",", ":")).encode("utf-8")

        data = _emit_export_object(
            self._ir_text, self._entry, self._abi, function_name, metadata_symbol, metadata, backend_cls
        )
        exported = set(_defined_global_symbols(data))
        if exported != exported_symbols:
            raise RuntimeError(
                f"export_to_c: object exports {sorted(exported)}, expected exactly {sorted(exported_symbols)}"
            )
        header = _render_header(export_metadata).encode("utf-8")

        with _output_directory_lock(object_path.parent):
            for lib, copy in zip(installed, libraries):
                _publish_file(Path(lib.path), Path(copy.path))
            _write_atomically(object_path, data)
            _write_atomically(header_path, header)
