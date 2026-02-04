"""
FlyDSL GPU Dialect Wrapper

This module re-exports the upstream mlir.dialects.gpu operations.

CRITICAL: GlobalRAIIMLIRContext now sets BOTH flydsl and upstream contexts,
so upstream GPU operations should work directly.
"""

# Import the upstream GPU dialect operations and re-export
try:
    from mlir.dialects import _gpu_ops_gen as _upstream_gpu
    
    # Direct re-export - GlobalRAIIMLIRContext handles context bridging
    module = _upstream_gpu.module
    launch = _upstream_gpu.launch
    GPUModuleOp = _upstream_gpu.GPUModuleOp
    LaunchOp = _upstream_gpu.LaunchOp
    
    # Re-export all other GPU ops
    import sys
    _current_module = sys.modules[__name__]
    for name in dir(_upstream_gpu):
        if not name.startswith('_'):
            setattr(_current_module, name, getattr(_upstream_gpu, name))
    
except ImportError as e:
    raise ImportError(
        f"Failed to import upstream GPU dialect. "
        f"Make sure mlir.dialects is available. Error: {e}"
    )

__all__ = ['module', 'launch', 'GPUModuleOp', 'LaunchOp']
