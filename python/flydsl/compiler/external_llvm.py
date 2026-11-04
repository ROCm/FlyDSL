# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .._mlir import ir
from ..utils import env


class ExternalLLVMError(RuntimeError):
    """Raised when external LLVM final code generation fails."""


def _format_llvm_cli_options(opts: dict) -> list[str]:
    """Convert ``{"enable-post-misched": False}`` to ``["--enable-post-misched=false"]``."""
    args: list[str] = []
    for name, value in opts.items():
        if isinstance(value, bool):
            args.append(f"--{name}={'true' if value else 'false'}")
        else:
            args.append(f"--{name}={value}")
    return args


def _llvm_dir() -> Path:
    raw = env.compile.llvm_dir.strip()
    if not raw:
        raise ExternalLLVMError(
            "External LLVM codegen requires FLYDSL_COMPILE_LLVM_DIR to point at an LLVM/MLIR install prefix."
        )
    return Path(raw).expanduser().resolve()


def _tool_candidates(prefix: Path, name: str) -> list[Path]:
    return [prefix / "bin" / name]


def _tool(prefix: Path, name: str) -> Path:
    for path in _tool_candidates(prefix, name):
        if path.is_file():
            if not os.access(path, os.X_OK):
                raise ExternalLLVMError(f"External LLVM tool is not executable: {path}")
            return path
    candidates = ", ".join(str(p) for p in _tool_candidates(prefix, name))
    raise ExternalLLVMError(f"External LLVM tool '{name}' not found. Tried: {candidates}")


def _subprocess_env(prefix: Path) -> dict:
    run_env = dict(os.environ)
    lib_dirs = [prefix / "lib"]
    existing = run_env.get("LD_LIBRARY_PATH", "")
    found_lib_dirs = [str(p) for p in lib_dirs if p.is_dir()]
    if found_lib_dirs:
        run_env["LD_LIBRARY_PATH"] = ":".join(found_lib_dirs + ([existing] if existing else []))
    path_dirs = [prefix / "bin"]
    existing_path = run_env.get("PATH", "")
    found_path_dirs = [str(p) for p in path_dirs if p.is_dir()]
    if found_path_dirs:
        run_env["PATH"] = ":".join(found_path_dirs + ([existing_path] if existing_path else []))
    return run_env


def _file_hash(path: Path) -> str:
    """SHA-256 hash of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=8)
def external_llvm_fingerprint(llvm_dir: Optional[str] = None) -> str:
    prefix = Path(llvm_dir).expanduser().resolve() if llvm_dir else _llvm_dir()
    mlir_opt = _tool(prefix, "mlir-opt")
    digest = hashlib.sha256(_file_hash(mlir_opt).encode())
    assembler = prefix / "bin" / "llvm-mc"
    if assembler.is_file():
        digest.update(_file_hash(assembler).encode())
    return f"external-binary:{prefix}:{digest.hexdigest()}"


def _link_external_assembly(module: ir.Module, binary: ir.Operation, prefix: Path, work_dir: Path) -> bool:
    """Assemble with the selected LLVM, then use FlyDSL's explicit linker pass."""
    from .._mlir.dialects import gpu
    from .._mlir.passmanager import PassManager
    from .backends.rocm import BINARY_PASS, get_rocm_toolchain, pass_path

    objects = [gpu.ObjectAttr(attr) for attr in ir.ArrayAttr(binary.attributes["objects"])]
    # Retain support for callers that already received a linked binary.
    if not any(obj.format == gpu.CompilationTarget.Assembly for obj in objects):
        return False
    source = _single_top_level_op(module, "gpu.module")
    if _symbol_name(source) != _symbol_name(binary):
        raise ExternalLLVMError("External LLVM changed the GPU module name")
    targets = list(ir.ArrayAttr(source.attributes["targets"]))
    if len(objects) != len(targets) or any(obj.target != target for obj, target in zip(objects, targets)):
        raise ExternalLLVMError("External LLVM changed the GPU target list")
    assembler = _tool(prefix, "llvm-mc")
    object_paths = []
    for index, obj in enumerate(objects):
        if obj.format != gpu.CompilationTarget.Assembly or not str(obj.target).startswith("#rocdl.target"):
            raise ExternalLLVMError("External assembly must contain only ROCDL assembly objects")
        isa = obj.object.decode("utf-8")
        # The directive is emitted by the external code generator, including
        # when chip/triple are implicit in the printed target attribute.
        directive = re.search(r'(?m)^\s*\.amdgcn_target\s+"(.+?)-(gfx[0-9a-z-]+)(:[^"\n]*)?"', isa)
        if not directive:
            raise ExternalLLVMError("External ISA has no AMDGCN target directive")
        triple, chip = directive[1], directive[2]
        processor_features = (directive[3] or "").split(":")[1:]
        settings = dict(re.findall(r'\b(triple|chip|features)\s*=\s*"([^"]*)"', str(obj.target)))
        features = settings.get("features", "")
        if not features:
            features = ",".join(f"{f[-1]}{f[:-1]}" for f in processor_features if f[-1:] in ("+", "-"))
        assembly_path = work_dir / f"external_{index}.s"
        object_path = work_dir / f"external_{index}.o"
        assembly_path.write_text(isa, encoding="utf-8")
        command = [
            str(assembler),
            "--filetype=obj",
            f"--triple={settings.get('triple', triple)}",
            f"--mcpu={settings.get('chip', chip)}",
            str(assembly_path),
            "-o",
            str(object_path),
        ]
        if features:
            command.append(f"--mattr={features}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, env=_subprocess_env(prefix))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ExternalLLVMError(f"External GPU assembly failed: {command}\n{exc.stderr}") from exc
        object_paths.append(object_path)
    toolchain = get_rocm_toolchain()
    options = toolchain.pass_options() if toolchain else ""
    inputs = ",".join(pass_path(path) for path in object_paths)
    pipeline = f"builtin.module({BINARY_PASS}{{format=fatbin {options} object-files={{{inputs}}}}})"
    PassManager.parse(pipeline, context=module.context).run(module.operation)
    return True


def _single_top_level_op(module: ir.Module, op_name: str) -> ir.Operation:
    matches = [op.operation for op in module.body.operations if op.operation.name == op_name]
    if len(matches) != 1:
        raise ExternalLLVMError(f"Expected exactly one {op_name}, found {len(matches)}.")
    return matches[0]


def _symbol_name(op: ir.Operation) -> str:
    try:
        return ir.StringAttr(op.attributes["sym_name"]).value
    except Exception as exc:
        raise ExternalLLVMError(f"{op.name} is missing a string sym_name attribute.") from exc


def _replace_gpu_module_with_binary_op(module: ir.Module, external_binary_module: ir.Module) -> None:
    gpu_module = _single_top_level_op(module, "gpu.module")
    gpu_binary = _single_top_level_op(external_binary_module, "gpu.binary")

    module_name = _symbol_name(gpu_module)
    binary_name = _symbol_name(gpu_binary)
    if module_name != binary_name:
        raise ExternalLLVMError(
            f"External LLVM produced gpu.binary @{binary_name}, but bundled module contains gpu.module @{module_name}."
        )

    ir.InsertionPoint(gpu_module).insert(gpu_binary.clone())
    gpu_module.erase()


def run_external_binary_codegen(
    module: ir.Module,
    binary_fragment: str,
    *,
    llvm_options: Optional[dict] = None,
    work_dir: Optional[Path] = None,
    stage_prefix: str = "external_binary",
) -> None:
    """Use external LLVM only for device binary bytes.

    Mutates ``module`` in-place: the bundled ``gpu.module`` is replaced by the
    external toolchain's ``gpu.binary`` op.  Host-side MLIR stays owned by the
    bundled MLIR runtime.
    """

    prefix = _llvm_dir()
    mlir_opt = _tool(prefix, "mlir-opt")
    pipeline = f"builtin.module({binary_fragment})"

    tmp_dir_obj = None
    if work_dir is None:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="flydsl_external_llvm_")
        work_dir = Path(tmp_dir_obj.name)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir.resolve()

    llvm_cli_args = _format_llvm_cli_options(llvm_options) if llvm_options else []

    input_path = work_dir / f"{stage_prefix}_input.mlir"
    external_output_path = work_dir / f"{stage_prefix}_external_output.mlir"
    output_path = work_dir / f"{stage_prefix}_output.mlir"

    def run_mlir_opt(*, pass_pipeline: str, input_path: Path, output_path: Path) -> None:
        cmd = [
            str(mlir_opt),
            str(input_path),
            f"--pass-pipeline={pass_pipeline}",
            "-o",
            str(output_path),
            *llvm_cli_args,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
                env=_subprocess_env(prefix),
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalLLVMError(
                f"External LLVM codegen timed out after 600s.\n" f"command: {' '.join(cmd)}\n" f"work_dir: {work_dir}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ExternalLLVMError(
                "External LLVM codegen failed.\n"
                f"llvm_dir: {prefix}\n"
                f"command: {' '.join(cmd)}\n"
                f"work_dir: {work_dir}\n"
                f"pipeline: {pass_pipeline}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

    # Serialize only the gpu.module into a minimal wrapper so the external
    # tool never sees host-side IR that may fail to parse with a different
    # LLVM version.
    gpu_module_op = _single_top_level_op(module, "gpu.module")
    wrapper = ir.Module.create(loc=ir.Location.unknown(module.context))
    wrapper.operation.attributes["gpu.container_module"] = ir.UnitAttr.get(module.context)
    ir.InsertionPoint.at_block_begin(wrapper.body).insert(gpu_module_op.operation.clone())
    input_path.write_text(wrapper.operation.get_asm(enable_debug_info=env.debug.enable_debug_info), encoding="utf-8")

    try:
        run_mlir_opt(pass_pipeline=pipeline, input_path=input_path, output_path=external_output_path)
        if not external_output_path.is_file():
            raise ExternalLLVMError(f"External LLVM did not create output file: {external_output_path}")
        external_binary_module = ir.Module.parse(
            external_output_path.read_text(encoding="utf-8"), context=module.context
        )
        binary = _single_top_level_op(external_binary_module, "gpu.binary")
        if not _link_external_assembly(module, binary, prefix, work_dir):
            _replace_gpu_module_with_binary_op(module, external_binary_module)
        output_path.write_text(
            module.operation.get_asm(enable_debug_info=env.debug.enable_debug_info), encoding="utf-8"
        )
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()
