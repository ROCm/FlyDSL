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


