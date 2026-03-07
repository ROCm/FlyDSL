from _mlir.dialects import builtin, gpu as _gpu
from flydsl.dialects.ext import buffer_ops
from flydsl.runtime.device import get_rocm_arch


def get_warp_size(arch: str = None) -> int:
    """Return the wavefront/warp size for the given GPU architecture.

    CDNA (gfx9xx) uses wave64, RDNA (gfx10xx/gfx11xx/gfx12xx) uses wave32.
    """
    if arch is None:
        arch = get_rocm_arch()
    if arch.startswith("gfx10") or arch.startswith("gfx11") or arch.startswith("gfx12"):
        return 32
    return 64


def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
    stream_llvm_ptr = buffer_ops.create_llvm_ptr(stream_ptr_value)

    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp(
        [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
    )
    return cast_op.results[0]
