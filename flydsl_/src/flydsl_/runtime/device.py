import os
from typing import Optional


def get_rocm_arch() -> str:
    """Best-effort ROCm GPU arch string (e.g. 'gfx942') without requiring HIP python bindings."""
    env = (
        os.environ.get("FLIR_CHIP")
        or os.environ.get("FLIR_GPU_ARCH")
        or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    )
    if env:
        # Accept 'gfx942' or '9.4.2' (HSA_OVERRIDE_GFX_VERSION).
        if env.startswith("gfx"):
            return env
        if env.count(".") == 2:
            parts = env.split(".")
            return f"gfx{parts[0]}{parts[1]}{parts[2]}"

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            arch = getattr(props, "gcnArchName", None) or getattr(props, "gcn_arch_name", None)
            if arch:
                # MLIR/LLVM expects the processor name without feature suffixes.
                # Example: "gfx942:sramecc+:xnack-" -> "gfx942".
                return str(arch).split(":", 1)[0]
    except Exception:
        pass

    # Conservative default for this repo's primary test environment.
    return "gfx942"


# Arch prefixes that support bf16 global atomics (single source for checks and error messages).
_BF16_GLOBAL_ATOMICS_ARCH_PREFIXES = ("gfx942", "gfx950", "gfx12")


def supports_bf16_global_atomics(arch: str) -> bool:
    """True if the given ROCm arch supports bf16 global atomics (e.g. for MoE reduction)."""
    arch = str(arch).strip()
    return any(arch.startswith(prefix) for prefix in _BF16_GLOBAL_ATOMICS_ARCH_PREFIXES)


def bf16_global_atomics_arch_description() -> str:
    """Human-readable list of archs that support bf16 global atomics (for error messages)."""
    return "/".join(_BF16_GLOBAL_ATOMICS_ARCH_PREFIXES)


