# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __version__
from ..expr.typing import Stream
from .jit_argument import _is_constexpr_annotation, _is_type_param_annotation
from .jit_function import JitFunction


@dataclass(frozen=True)
class ExportedHsaco:
    hsaco_path: Path
    metadata_path: Path
    metadata: Dict[str, Any]


def _decode_mlir_string_attr(text: str, start: int) -> tuple[bytes, int]:
    """Decode an MLIR string attribute body starting at the opening quote."""
    if start >= len(text) or text[start] != '"':
        raise ValueError("expected MLIR string attribute opening quote")

    out = bytearray()
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == '"':
            return bytes(out), i + 1
        if ch == "\\":
            if i + 1 >= len(text):
                raise ValueError("unterminated MLIR string escape")
            nxt = text[i + 1]
            if nxt == "\\":
                out.append(ord("\\"))
                i += 2
                continue
            if nxt == '"':
                out.append(ord('"'))
                i += 2
                continue
            if i + 3 <= len(text):
                hex_str = text[i + 1 : i + 3]
                try:
                    out.append(int(hex_str, 16))
                    i += 3
                    continue
                except ValueError:
                    pass
        out.extend(ch.encode("utf-8"))
        i += 1

    raise ValueError("unterminated MLIR string attribute")


def _find_matching_angle(text: str, start: int) -> int:
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
        elif ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unterminated angle attribute")


def _top_level_comma(text: str) -> int:
    depth = 0
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            return i
        i += 1
    return -1


def extract_gpu_objects(mlir_asm: str) -> List[Dict[str, Any]]:
    """Extract GPU binary objects from compiled MLIR assembly."""
    objects: List[Dict[str, Any]] = []
    marker = "#gpu.object<"
    pos = 0
    while True:
        start = mlir_asm.find(marker, pos)
        if start == -1:
            break
        angle_start = start + len("#gpu.object")
        end = _find_matching_angle(mlir_asm, angle_start)
        body = mlir_asm[angle_start + 1 : end]
        comma = _top_level_comma(body)
        if comma == -1:
            pos = end + 1
            continue
        target = body[:comma].strip()
        rest = body[comma + 1 :].lstrip()
        if not rest.startswith('"'):
            pos = end + 1
            continue
        data, _ = _decode_mlir_string_attr(rest, 0)
        objects.append(
            {
                "target": target,
                "data": data,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
        pos = end + 1

    if objects:
        return objects

    # Older GPU dialect spellings used named binary fields in the object attr.
    for m in re.finditer(r"(?:bin|object|offload)\s*=\s*\"", mlir_asm):
        data, _ = _decode_mlir_string_attr(mlir_asm, m.end() - 1)
        objects.append(
            {
                "target": "",
                "data": data,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    return objects


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "kernel"


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    return getattr(annotation, "__name__", str(annotation))


def _ctype_for_arg(value: Any, annotation: Any) -> str:
    if annotation is Stream or getattr(annotation, "_is_stream_param", False):
        return "hip_stream"
    if hasattr(value, "data_ptr"):
        return "void_p"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int32"
    if isinstance(value, float):
        return "float32"
    if hasattr(value, "__fly_ptrs__"):
        name = type(value).__name__.lower()
        if "stream" in name:
            return "hip_stream"
        if "tensor" in name:
            return "void_p"
        if "float" in name:
            return "float32"
        if "int" in name or "uint" in name:
            return "int32"
    return "opaque"


def _host_arg_metadata(jit_fn: JitFunction, args_tuple: tuple) -> List[Dict[str, Any]]:
    sig = jit_fn._sig
    result = []
    runtime_index = 0
    for source_index, (name, param) in enumerate(sig.parameters.items()):
        annotation = param.annotation
        if annotation is not inspect.Parameter.empty and (
            _is_constexpr_annotation(annotation) or _is_type_param_annotation(annotation)
        ):
            continue
        value = args_tuple[source_index]
        result.append(
            {
                "name": name,
                "index": runtime_index,
                "source_index": source_index,
                "annotation": _annotation_name(annotation),
                "ctype": _ctype_for_arg(value, annotation),
                "python_type": type(value).__name__,
            }
        )
        runtime_index += 1
    return result


def _specialization_metadata(jit_fn: JitFunction, args_tuple: tuple) -> Dict[str, Any]:
    sig = jit_fn._sig
    values = {}
    for source_index, (name, param) in enumerate(sig.parameters.items()):
        annotation = param.annotation
        if annotation is not inspect.Parameter.empty and (
            _is_constexpr_annotation(annotation) or _is_type_param_annotation(annotation)
        ):
            value = args_tuple[source_index]
            values[name] = repr(value)
    return values


def _build_metadata(
    *,
    name: str,
    hsaco_path: Path,
    object_info: Dict[str, Any],
    jit_fn: JitFunction,
    compilation,
) -> Dict[str, Any]:
    if not compilation.launches:
        raise RuntimeError("Cannot export HSACO metadata: no gpu.launch_func was emitted")
    if len(compilation.launches) != 1:
        raise NotImplementedError(
            "export_hsaco currently supports exactly one gpu.launch_func per @jit launcher; "
            f"found {len(compilation.launches)}"
        )

    launch = compilation.launches[0]
    host_args = _host_arg_metadata(jit_fn, compilation.args_tuple)
    host_arg_by_name = {arg["name"]: arg for arg in host_args}
    kernel_args = []
    for arg in launch["kernel_args"]:
        entry = dict(arg)
        if arg.get("kind") == "host_arg":
            host_arg = host_arg_by_name.get(arg["name"])
            if host_arg is not None:
                entry["ctype"] = host_arg["ctype"]
        kernel_args.append(entry)

    metadata = {
        "schema_version": 1,
        "name": name,
        "entry": jit_fn.func.__name__,
        "target": {
            "backend": compilation.target.backend,
            "arch": compilation.target.arch,
            "warp_size": compilation.target.warp_size,
            "object_target": object_info["target"],
        },
        "binary": {
            "path": hsaco_path.name,
            "sha256": object_info["sha256"],
            "size": object_info["size"],
            "format": "hip_module_code_object",
        },
        "host_args": host_args,
        "constexpr": compilation.constexpr_values,
        "specialization": _specialization_metadata(jit_fn, compilation.args_tuple),
        "compile_hints": dict(jit_fn.compile_hints),
        "launch": {
            "kernel_name": launch["kernel_name"],
            "kernel_args": kernel_args,
            "grid": launch["grid"],
            "block": launch["block"],
            "smem": launch["smem"],
            "stream": launch["stream"],
            "cluster": launch["cluster"],
        },
        "flydsl": {
            "version": __version__,
            "cache_key": str(compilation.cache_key),
        },
    }
    metadata["route_key"] = hashlib.sha256(
        json.dumps(
            {
                "target": metadata["target"],
                "entry": metadata["entry"],
                "specialization": metadata["specialization"],
                "compile_hints": metadata["compile_hints"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return metadata


def export_hsaco(
    launch_fn: JitFunction,
    *sample_args,
    out_dir: str | Path,
    name: Optional[str] = None,
    metadata_overrides: Optional[Dict[str, Any]] = None,
    **sample_kwargs,
) -> ExportedHsaco:
    """Compile a FlyDSL launcher to a HIP-loadable code object plus metadata."""
    if not isinstance(launch_fn, JitFunction):
        raise TypeError(f"export_hsaco() expects a @flyc.jit function, got {type(launch_fn).__name__}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compilation = launch_fn.compile_for_export(*sample_args, **sample_kwargs)
    objects = extract_gpu_objects(compilation.artifact.ir)
    if not objects:
        raise RuntimeError("Compiled module does not contain a gpu.binary object")
    if len(objects) != 1:
        raise NotImplementedError(f"export_hsaco currently supports one GPU object; found {len(objects)}")

    default_name = compilation.launches[0]["kernel_name"] if compilation.launches else launch_fn.func.__name__
    export_name = _sanitize_name(name or default_name)
    hsaco_path = out_dir / f"{export_name}.hsaco"
    hsaco_path.write_bytes(objects[0]["data"])

    metadata = _build_metadata(
        name=export_name,
        hsaco_path=hsaco_path,
        object_info=objects[0],
        jit_fn=launch_fn,
        compilation=compilation,
    )
    if metadata_overrides:
        metadata.update(metadata_overrides)

    metadata_path = out_dir / f"{export_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ExportedHsaco(hsaco_path=hsaco_path, metadata_path=metadata_path, metadata=metadata)
