import sys
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Union, Type, Tuple, Any, Optional, Callable, Dict

from .._mlir import ir
from .._mlir.dialects import arith, memref, gpu
from .._mlir.extras import types as T
from flydsl.expr.gpu import lds_space

# ==============================================================================
# Type Utilities
# ==============================================================================

def get_mlir_type_size(mlir_type: ir.Type) -> int:
    """Returns the size in bytes of an MLIR type."""
    if mlir_type == T.f32() or mlir_type == T.i32(): return 4
    if mlir_type == T.f16() or mlir_type == T.bf16() or mlir_type == T.i16(): return 2
    if mlir_type == T.i8(): return 1
    # FP8 types
    if isinstance(
        mlir_type,
        (
            ir.Float8E4M3FNType,
            ir.Float8E5M2Type,
            # AMD MFMA uses FNUZ variants on ROCm (1 byte each).
            ir.Float8E4M3FNUZType,
        ),
    ):
        return 1
    if mlir_type == T.f64() or mlir_type == T.i64(): return 8
    if isinstance(mlir_type, ir.VectorType):
        total = 1
        for s in mlir_type.shape: total *= s
        return total * get_mlir_type_size(mlir_type.element_type)
    # Fallback / Default
    return 4

def get_mlir_type_align(mlir_type: ir.Type) -> int:
    """Returns the alignment requirement in bytes."""
    # For Vector types, usually align to total size, capped at 16 bytes (float4)
    size = get_mlir_type_size(mlir_type)
    return min(size, 16) 

def get_op_result_or_value(op_or_val):
    if hasattr(op_or_val, 'value'): # ArithValue or similar wrapper
        return op_or_val.value
    if isinstance(op_or_val, ir.Value):
        return op_or_val
    if hasattr(op_or_val, 'result'):
        return op_or_val.result
    if hasattr(op_or_val, 'results'):
        return op_or_val.results[0]
    return op_or_val

# ==============================================================================
# Pointer Abstraction
# ==============================================================================

class SmemPtr:
    """
    Represents a typed pointer into Shared Memory.
    Analogue to a typed pointer wrapper.
    """
    def __init__(self, base_memref: ir.Value, byte_offset: int, element_type: ir.Type, shape: Optional[Tuple[int, ...]] = None):
        self.base_memref = base_memref # The raw i8 buffer
        self.byte_offset = byte_offset # Static offset
        self.element_type = element_type
        self.shape = shape
        self._view_cache = None

    def _get_value(self, op_or_val):
        if hasattr(op_or_val, 'value'): # ArithValue or similar wrapper
            return op_or_val.value
        if isinstance(op_or_val, ir.Value):
            return op_or_val
        if hasattr(op_or_val, 'result'):
            return op_or_val.result
        if hasattr(op_or_val, 'results'):
            return op_or_val.results[0]
        return op_or_val

    def get(self) -> ir.Value:
        """Dereference: Returns a memref view."""
        if self._view_cache: return self._view_cache

        offset_op = arith.constant(T.index(), self.byte_offset)
        offset_val = get_op_result_or_value(offset_op)
        
        # Construct a structured memref view using the provided shape or default to scalar.
        
        if self.shape:
            target_shape = self.shape
        else:
            target_shape = (1,) # Scalar treated as 1-element array for view simplicity
            
        target_type = T.memref(*target_shape, self.element_type, memory_space=lds_space())
        
        # memref.view(source, byte_shift, sizes)
        # sizes are needed for dynamic dimensions. Since we use static shapes here, sizes=[]
        self._view_cache = memref.view(target_type, self.base_memref, offset_val, sizes=[])
        return self._view_cache

    def load(self, idxs=None):
        """Helper to load value. If scalar, idxs defaults to [0]."""
        view = self.get()
        if idxs is None: 
            # If scalar (shape is None or (1,)), access index 0
            idxs = [
                get_op_result_or_value(arith.constant(T.index(), 0))
                for _ in range(len(self.shape) if self.shape else 1)
            ]
        else:
            idxs = [get_op_result_or_value(i) for i in idxs]
        return memref.load(get_op_result_or_value(view), idxs)
    
    def store(self, val, idxs=None):
        """Helper to store value. If scalar, idxs defaults to [0]."""
        view = self.get()
        if idxs is None: 
            idxs = [
                get_op_result_or_value(arith.constant(T.index(), 0))
                for _ in range(len(self.shape) if self.shape else 1)
            ]
        else:
            idxs = [get_op_result_or_value(i) for i in idxs]
        memref.store(get_op_result_or_value(val), get_op_result_or_value(view), idxs)

# ==============================================================================
# Allocator
# ==============================================================================

class SmemAllocator:
    def __init__(self, ctx, arch: Optional[str] = None, global_sym_name: str = "smem_storage"):
        self.ctx = ctx
        self.ptr = 0
        self.max_size = 0
        self.alignment = 128 # Base alignment for the whole buffer
        self.finalized = False
        self.base_buffer_val = None 
        self.global_sym_name = global_sym_name
        self.arch = arch

    def _align(self, ptr, align):
        if ptr % align == 0: return ptr
        return (ptr + align - 1) // align * align

    def finalize(self):
        """
        Generates the global buffer allocation. 
        Must be called inside the gpu.module body.
        """
        if self.finalized: return
        
        # Final padding to block alignment
        total_size = self._align(self.ptr, 128) 
        if total_size == 0: total_size = 128
        
        # Create Global
        memref_type = T.memref(total_size, T.i8(), memory_space=lds_space())
        self.global_op = memref.global_(
            sym_name=self.global_sym_name,
            type_=memref_type,
            alignment=1024 # High alignment for base
        )
        self.finalized = True
        return self.global_op

    def get_base(self):
        """
        Call inside kernel to get the pointer.
        """
        # We need to recreate the memref type to access the global
        # The size must match what was allocated (or at least be large enough, but for global access exact match is good)
        total_size = self._align(self.ptr, 128)
        if total_size == 0: total_size = 128
        
        memref_type = T.memref(total_size, T.i8(), memory_space=lds_space())
        op = memref.get_global(memref_type, self.global_sym_name)
        return get_op_result_or_value(op)

# ==============================================================================
# Shared Memory Capacity Check
# ==============================================================================

SMEM_CAPACITY_MAP = {
    # ===================== AMD CDNA Architectures (Data Center Compute Cards) =====================
    # CDNA 3 (MI300 Series) - 64KB LDS per CU
    "gfx942": 65536,   # MI300A / MI300X: 64KB LDS per CU
    # CDNA 4 (MI350 Series) - 160KB LDS per CU (key upgrade for CDNA4)
    "gfx950": 163840,   # MI300C / MI300X Enhanced Models: 64KB LDS per CU
}

def check_smem_capacity(allocated_bytes: int, arch: str = None):
    """
    Checks if the allocated shared memory fits within the device capacity.
    """
    if arch is None:
        # Try to detect arch from environment or flir context if possible
        # For now, default to a safe limit or skip check if unknown
        return
        
    if arch in SMEM_CAPACITY_MAP:
        limit = SMEM_CAPACITY_MAP[arch]
        if allocated_bytes > limit:
            raise RuntimeError(
                f"Shared Memory Overflow: Requested {allocated_bytes} bytes, "
                f"but device {arch} limit is {limit} bytes."
            )
    else:
        # Unknown arch, maybe warn or skip
        pass


