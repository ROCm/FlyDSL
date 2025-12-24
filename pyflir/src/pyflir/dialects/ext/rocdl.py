"""ROCDL dialect extension for ROCm/AMD GPU programming.

This module provides access to ROCm-specific GPU operations including:
- Thread/block/grid identifiers and dimensions
- Synchronization primitives (barriers, wait operations)
- Matrix multiplication acceleration (MFMA, WMMA, SMFMAC)
- Data movement and shuffle operations
- Atomic operations
- Type conversion operations

Example:
    >>> from pyflir.dialects.ext import rocdl
    >>> tid_x = rocdl.workitem_id_x()
    >>> rocdl.barrier()
"""

from _mlir.dialects.rocdl import *

__all__ = [
    # Thread/Block/Grid IDs and dimensions
    'workitem_id_x', 'workitem_id_y', 'workitem_id_z',
    'workgroup_id_x', 'workgroup_id_y', 'workgroup_id_z', 
    'workgroup_dim_x', 'workgroup_dim_y', 'workgroup_dim_z',
    'grid_dim_x', 'grid_dim_y', 'grid_dim_z',
    'wavefrontsize',
    
    # Synchronization
    'barrier', 's_barrier', 's_barrier_signal', 's_barrier_wait',
    's_waitcnt', 's_wait_loadcnt', 's_wait_storecnt',
    's_wait_dscnt', 's_wait_expcnt',
    
    # Matrix operations - MFMA (Matrix Fused Multiply-Add)
    'mfma_f32_32x32x8f16', 'mfma_f32_16x16x16f16',
    'mfma_f32_32x32x4bf16', 'mfma_f32_16x16x8bf16',
    'mfma_i32_32x32x8i8', 'mfma_i32_16x16x16i8',
    
    # Matrix operations - WMMA (Wave Matrix Multiply-Accumulate)
    'wmma_f32_16x16x16_f16', 'wmma_f32_16x16x16_bf16',
    'wmma_i32_16x16x16_iu8',
    
    # Matrix operations - SMFMAC (Sparse Matrix FMA)
    'smfmac_f32_32x32x16_f16', 'smfmac_f32_32x32x16_bf16',
    'smfmac_i32_32x32x32_i8',
    
    # Shuffle and permutation
    'ds_swizzle', 'ds_bpermute',
    'permlanex16', 'permlane16_swap', 'permlane32_swap',
    'readlane', 'readfirstlane',
    'update_dpp',
    'ballot',
    
    # Data movement
    'raw_buffer_load', 'raw_buffer_store',
    'raw_ptr_buffer_load', 'raw_ptr_buffer_store',
    'load_to_lds', 'global_load_lds',
    'make_buffer_rsrc',
    
    # Atomic operations
    'raw_buffer_atomic_fadd', 'raw_buffer_atomic_fmax',
    'raw_buffer_atomic_smax', 'raw_buffer_atomic_umin',
    'raw_ptr_buffer_atomic_fadd', 'raw_ptr_buffer_atomic_fmax',
    
    # Bit manipulation
    'mbcnt_lo', 'mbcnt_hi',
    
    # Scheduling and optimization
    's_setprio', 's_sleep',
    'sched_barrier', 'sched_group_barrier',
    'iglp_opt',
    
    # Type conversions
    'cvt_f32_bf8', 'cvt_f32_fp8',
    'cvt_pk_f32_bf8', 'cvt_pk_f32_fp8',
]
