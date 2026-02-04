from contextlib import ExitStack

from mlir import ir as upstream_ir
from mlir.passmanager import PassManager

from ..lang import MlirModule
from .executor import Executor
from .._mlir.interop import to_upstream


def _decode_mlir_escaped_bytes(s: str) -> str:
    """Decode MLIR string attr content that uses \\xx hex byte escapes (e.g. \\0A, \\09, \\22).

    This is what gpu-module-to-binary emits for `assembly = "..."` (and often `bin = "..."`).
    """
    out_chars = []
    i = 0
    n = len(s)

    def _is_hex(c: str) -> bool:
        return ("0" <= c <= "9") or ("a" <= c <= "f") or ("A" <= c <= "F")

    while i < n:
        ch = s[i]
        if ch != "\\":
            out_chars.append(ch)
            i += 1
            continue

        # Backslash escape.
        if i + 2 < n and _is_hex(s[i + 1]) and _is_hex(s[i + 2]):
            byte = int(s[i + 1 : i + 3], 16)
            out_chars.append(chr(byte))
            i += 3
            continue

        # Common C-style single-char escapes (rare here, but harmless).
        if i + 1 < n:
            nxt = s[i + 1]
            if nxt == "n":
                out_chars.append("\n")
                i += 2
                continue
            if nxt == "t":
                out_chars.append("\t")
                i += 2
                continue
            if nxt == "r":
                out_chars.append("\r")
                i += 2
                continue
            if nxt in ['"', "\\"]:
                out_chars.append(nxt)
                i += 2
                continue
            # Unknown escape: keep the escaped char as-is.
            out_chars.append(nxt)
            i += 2
            continue

        # Trailing backslash.
        i += 1

    return "".join(out_chars)


def _extract_mlir_string_attr(asm: str, attr_name: str) -> str | None:
    """Extract and decode a string attribute like `attr_name = "..."` from an MLIR asm dump."""
    marker = f'{attr_name} = "'
    start = asm.find(marker)
    if start == -1:
        return None

    i = start + len(marker)
    # Find the closing quote. Skip over \xx escapes as two hex bytes.
    while i < len(asm):
        if asm[i] == "\\" and i + 2 < len(asm):
            # Skip the escape introducer and two following chars (typically hex digits).
            i += 3
            continue
        if asm[i] == '"':
            end = i
            encoded = asm[start + len(marker) : end]
            return _decode_mlir_escaped_bytes(encoded)
        i += 1
    return None


def compile(
    fx_module: MlirModule, verify=True, print_after_all=False, output_format="fatbin"
):
    # gpu-module-to-binary formats are backend-dependent. For ROCm/ROCDL, "isa"
    # is the human-readable assembly/ISA dump and "fatbin" is an object container.
    fmt_map = {
        "fatbin": "fatbin",
        "assembly": "isa",
    }
    if output_format not in fmt_map:
        raise ValueError(
            f"Unsupported output_format: {output_format}. Use one of {list(fmt_map)}"
        )

    pipeline = (
        "builtin.module("
        "gpu-kernel-outlining{data-layout-str=},"
        "fly-canonicalize,"
        "fly-layout-lowering,"
        "convert-fly-to-rocdl,"
        "canonicalize,"
        "gpu.module("
        "convert-vector-to-llvm,"
        "canonicalize,"
        "convert-gpu-to-rocdl{ chipset=gfx000 index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true}"
        "),"
        "rocdl-attach-target{O=2 abi=600 chip=gfx942 correct-sqrt=true daz=false fast=false features= finite-only=false  module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true},"
        "gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true},"
        "reconcile-unrealized-casts,"
        f"gpu-module-to-binary{{format={fmt_map[output_format]}  opts= section= toolkit=}}"
        ")"
    )
    mlir_module = fx_module.module
    
    # CRITICAL: Get module ASM and re-parse in a fresh upstream context
    # This ensures proper resource management and avoids cross-domain issues
    asm = mlir_module.operation.get_asm(enable_debug_info=True)
    
    # Create a fresh upstream context for compilation
    upstream_ctx = upstream_ir.Context()
    upstream_ctx.allow_unregistered_dialects = True
    upstream_ctx.load_all_available_dialects()
    
    # Import _fly to ensure passes are registered
    from flydsl._mlir._mlir_libs import _fly
    
    # CRITICAL: Register fly dialect in this context
    # Create a flydsl context wrapper sharing the same underlying C pointer
    flydsl_ctx = _fly.Context._CAPICreate(upstream_ctx._CAPIPtr)
    _fly._register_dialect(flydsl_ctx, load=True)
    
    module = None
    try:
        with ExitStack() as stack:
            stack.enter_context(upstream_ctx)
            
            # Parse module in this context
            module = upstream_ir.Module.parse(asm)
            
            pm = PassManager.parse(pipeline)
            pm.enable_verifier(verify)
            pm.enable_ir_printing(print_after_all=print_after_all)

            pm.run(module.operation)
    except Exception as e:
        print(e)

    # Default: produce a runnable executor (requires gpu-module-to-binary to have produced
    # a launchable binary container).
    if output_format == "fatbin":
        return Executor(module)

    # Debug output: return textual assembly/ISA emitted into gpu.binary's `assembly` attribute
    # (or `bin` in some toolchains).
    # If the toolchain doesn't embed it (or it was elided), fall back to returning the MLIR.
    asm = module.operation.get_asm(enable_debug_info=True, large_elements_limit=1 << 30)
    text = _extract_mlir_string_attr(asm, "assembly")
    if text is not None:
        return text
    text = _extract_mlir_string_attr(asm, "bin")
    if text is not None:
        return text
    return asm
