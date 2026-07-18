# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import hashlib
import os
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
    return f"external-binary:{prefix}:{_file_hash(mlir_opt)}"


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
        _replace_gpu_module_with_binary_op(module, external_binary_module)
        output_path.write_text(
            module.operation.get_asm(enable_debug_info=env.debug.enable_debug_info), encoding="utf-8"
        )
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


def llvm_opt_fingerprint(pipeline: str, plugins: Optional[list] = None) -> str:
    """Cache fingerprint for a custom LLVM-opt configuration: the pipeline
    string plus each plugin's path and content hash, so editing a plugin .so
    (or the pipeline) invalidates cached artifacts."""
    parts = [f"llvm-opt:{pipeline}"]
    for p in plugins or []:
        path = Path(p).expanduser()
        try:
            parts.append(f"{path}:{_file_hash(path.resolve())}")
        except OSError:
            parts.append(f"{path}:<missing>")
    return ";".join(parts)


def _run_tool(cmd: list, *, prefix: Path, what: str, work_dir: Path) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600, env=_subprocess_env(prefix))
    except subprocess.TimeoutExpired as exc:
        raise ExternalLLVMError(
            f"{what} timed out after 600s.\ncommand: {' '.join(cmd)}\nwork_dir: {work_dir}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ExternalLLVMError(
            f"{what} failed.\nllvm_dir: {prefix}\ncommand: {' '.join(cmd)}\n"
            f"work_dir: {work_dir}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc


def run_llvm_opt_then_binary(
    module: ir.Module,
    *,
    llvm_ir: str,
    attach_fragment: str,
    binary_fragment: str,
    pipeline: str,
    plugins: Optional[list] = None,
    llvm_options: Optional[dict] = None,
    work_dir: Optional[Path] = None,
    stage_prefix: str = "llvm_opt",
) -> None:
    """Run a custom LLVM new-PM pass pipeline on the device kernel's (pre-link)
    LLVM IR, then re-codegen the device binary and splice it back into *module*.

    Flow: ``opt --passes`` (with optional ``--load-pass-plugin``) on ``llvm_ir``
    -> ``mlir-translate --import-llvm`` -> wrap into a ``gpu.module`` -> external
    ``mlir-opt`` running ``attach_fragment`` (ROCDL target at O=0) then
    ``binary_fragment`` (``gpu-module-to-binary``) -> replace the in-process
    ``gpu.module`` with the produced ``gpu.binary``.
    """
    prefix = _llvm_dir()
    opt = _tool(prefix, "opt")
    mlir_translate = _tool(prefix, "mlir-translate")
    mlir_opt = _tool(prefix, "mlir-opt")

    gpu_module = _single_top_level_op(module, "gpu.module")
    name = _symbol_name(gpu_module)
    data_layout = None
    if "llvm.data_layout" in gpu_module.attributes:
        try:
            data_layout = ir.StringAttr(gpu_module.attributes["llvm.data_layout"]).value
        except Exception:
            data_layout = None

    llvm_cli_args = _format_llvm_cli_options(llvm_options) if llvm_options else []

    tmp_dir_obj = None
    if work_dir is None:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="flydsl_llvm_opt_")
        work_dir = Path(tmp_dir_obj.name)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    in_ll = work_dir / f"{stage_prefix}_pre_opt.ll"
    out_ll = work_dir / f"{stage_prefix}_post_opt.ll"
    imported_path = work_dir / f"{stage_prefix}_imported.mlir"
    wrapped_path = work_dir / f"{stage_prefix}_wrapped.mlir"
    bin_path = work_dir / f"{stage_prefix}_binary.mlir"

    try:
        in_ll.write_text(llvm_ir, encoding="utf-8")

        plugin_args = [f"--load-pass-plugin={Path(p).expanduser()}" for p in (plugins or [])]
        _run_tool(
            [str(opt), str(in_ll), "-S", f"--passes={pipeline}", *plugin_args, *llvm_cli_args, "-o", str(out_ll)],
            prefix=prefix,
            what="LLVM opt pass pipeline",
            work_dir=work_dir,
        )

        _run_tool(
            [str(mlir_translate), "--import-llvm", str(out_ll), "-o", str(imported_path)],
            prefix=prefix,
            what="mlir-translate --import-llvm",
            work_dir=work_dir,
        )

        # Wrap the re-imported LLVM-dialect IR back into a gpu.module (no target;
        # attach_fragment adds it).  The original gpu.module's data layout is
        # re-applied; gpu-module-to-binary will produce gpu.binary @<name>.
        imported = ir.Module.parse(imported_path.read_text(encoding="utf-8"), context=module.context)
        body = "\n".join(op.operation.get_asm() for op in imported.body.operations)
        dl_attr = f' attributes {{llvm.data_layout = "{data_layout}"}}' if data_layout else ""
        wrapped_path.write_text(
            f"module attributes {{gpu.container_module}} {{\n" f"  gpu.module @{name}{dl_attr} {{\n{body}\n  }}\n}}\n",
            encoding="utf-8",
        )

        _run_tool(
            [
                str(mlir_opt),
                str(wrapped_path),
                f"--pass-pipeline=builtin.module({attach_fragment},{binary_fragment})",
                *llvm_cli_args,
                "-o",
                str(bin_path),
            ],
            prefix=prefix,
            what="external gpu-module-to-binary codegen",
            work_dir=work_dir,
        )

        if not bin_path.is_file():
            raise ExternalLLVMError(f"external codegen did not create output file: {bin_path}")
        binary_module = ir.Module.parse(bin_path.read_text(encoding="utf-8"), context=module.context)
        _replace_gpu_module_with_binary_op(module, binary_module)
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


# ---------------------------------------------------------------------------
# Custom-codegen path: fly-llc (IR -> obj with injectable MIR passes) + ld.lld
# ---------------------------------------------------------------------------


def _fly_llc_path() -> Path:
    raw = env.compile.fly_llc.strip()
    if raw:
        return Path(raw).expanduser()
    cand = _llvm_dir() / "bin" / "fly-llc"
    if cand.is_file():
        return cand
    raise ExternalLLVMError(
        "fly-llc tool not found: set FLYDSL_COMPILE_FLY_LLC or build fly-llc into <FLYDSL_COMPILE_LLVM_DIR>/bin."
    )


def _lld_path() -> Path:
    raw = env.compile.lld.strip()
    if raw:
        return Path(raw).expanduser()
    cand = _llvm_dir() / "bin" / "ld.lld"
    if cand.is_file():
        return cand
    raise ExternalLLVMError(
        "fly-llc codegen path needs ld.lld: set FLYDSL_COMPILE_LLD or place ld.lld in <FLYDSL_COMPILE_LLVM_DIR>/bin."
    )


def fly_llc_codegen_fingerprint(
    passes: Optional[list] = None, plugins: Optional[list] = None, insert_after: Optional[list] = None
) -> str:
    """Cache fingerprint for a fly-llc codegen configuration: the pass names plus
    the fly-llc binary's and each plugin's content hash."""
    parts = ["fly-llc-codegen:" + ",".join(passes or []) + "|after:" + ",".join(insert_after or [])]
    try:
        parts.append(_file_hash(_fly_llc_path().resolve()))
    except OSError:
        parts.append("<no-fly-llc>")
    except ExternalLLVMError:
        parts.append("<no-fly-llc>")
    for p in plugins or []:
        path = Path(p).expanduser()
        try:
            parts.append(f"{path}:{_file_hash(path.resolve())}")
        except OSError:
            parts.append(f"{path}:<missing>")
    return ";".join(parts)


def _gpu_binary_module_text(name: str, target_cpu: str, hsaco: bytes) -> str:
    """Build a ``builtin.module`` text embedding *hsaco* as a ``gpu.binary @name``
    (every byte escaped as ``\\XX`` for the MLIR string attribute)."""
    esc = "".join("\\%02X" % b for b in hsaco)
    return (
        "module attributes {gpu.container_module} {\n"
        f'  gpu.binary @{name} [#gpu.object<#rocdl.target<chip = "{target_cpu}">, kernels = <>, bin = "{esc}">]\n'
        "}\n"
    )


def run_fly_llc_codegen(
    module: ir.Module,
    *,
    llvm_ir: str,
    codegen_passes: Optional[list] = None,
    codegen_plugins: Optional[list] = None,
    codegen_insert_after: Optional[list] = None,
    target_triple: str,
    target_cpu: str,
    work_dir: Optional[Path] = None,
    stage_prefix: str = "fly_llc",
) -> None:
    """Codegen the device kernel's LLVM IR with injectable MIR passes and splice
    the result back into *module*.

    Flow: ``fly-llc <in.ll> -o <obj> --load=<plugin> [--pre-emit-pass=<pass>]
    [--insert-after=<anchor>=<pass>]`` (custom MIR passes run inside the standard
    codegen — pre-emit and/or at named earlier stages) -> ``ld.lld -shared`` ->
    wrap the HSACO bytes into a ``gpu.binary`` -> replace the in-process
    ``gpu.module``.
    """
    fly_llc = _fly_llc_path()
    lld = _lld_path()
    prefix = _llvm_dir()

    gpu_module = _single_top_level_op(module, "gpu.module")
    name = _symbol_name(gpu_module)

    tmp_dir_obj = None
    if work_dir is None:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="flydsl_fly_llc_")
        work_dir = Path(tmp_dir_obj.name)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    in_ll = work_dir / f"{stage_prefix}_pre_codegen.ll"
    obj = work_dir / f"{stage_prefix}.o"
    hsaco = work_dir / f"{stage_prefix}.hsaco"
    bin_mlir = work_dir / f"{stage_prefix}_binary.mlir"

    try:
        in_ll.write_text(llvm_ir, encoding="utf-8")

        plugin_args = [f"--load={Path(p).expanduser()}" for p in (codegen_plugins or [])]
        pass_args = [f"--pre-emit-pass={n}" for n in (codegen_passes or [])]
        insert_after_args = [f"--insert-after={spec}" for spec in (codegen_insert_after or [])]
        _run_tool(
            [
                str(fly_llc),
                str(in_ll),
                "-o",
                str(obj),
                f"-mtriple={target_triple}",
                f"-mcpu={target_cpu}",
                *plugin_args,
                *pass_args,
                *insert_after_args,
            ],
            prefix=prefix,
            what="fly-llc codegen",
            work_dir=work_dir,
        )

        _run_tool(
            [str(lld), "-shared", str(obj), "-o", str(hsaco)],
            prefix=prefix,
            what="ld.lld HSACO link",
            work_dir=work_dir,
        )

        if not hsaco.is_file():
            raise ExternalLLVMError(f"ld.lld did not create HSACO: {hsaco}")
        text = _gpu_binary_module_text(name, target_cpu, hsaco.read_bytes())
        bin_mlir.write_text(text, encoding="utf-8")
        binary_module = ir.Module.parse(text, context=module.context)
        _replace_gpu_module_with_binary_op(module, binary_module)
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()
