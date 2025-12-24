"""Python bindings for Flir ROCm dialect operations.

This module provides Python wrappers for AMD GPU-specific operations
for GFX942 (MI300 series), including MFMA, copy operations, and LDS management.
"""

from typing import Optional
from _mlir.ir import Type, Value, Location, InsertionPoint


#===----------------------------------------------------------------------===//
# Copy Operations for AMD GPUs
#===----------------------------------------------------------------------===//

class CopyUniversalOp:
    """Universal copy operation for AMD GPUs.
    
    This operation represents a generic data movement instruction that can
    be instantiated for different memory spaces:
    - Global to LDS (buffer_load)
    - LDS to Register (ds_read)
    - Register to Global (buffer_store)
    - Register to LDS (ds_write)
    
    The actual instruction is selected based on the source and destination
    memory spaces during lowering.
    """
    
    def __init__(self):
        """Initialize universal copy operation."""
        pass
    
    def __repr__(self):
        return "CopyUniversalOp()"
    
    def __str__(self):
        return "Universal AMD GPU copy operation"


class CopyG2LOp:
    """Global to LDS copy operation (buffer_load)."""
    
    def __init__(self, vector_size: int = 8):
        """Initialize G2L copy operation.
        
        Args:
            vector_size: Number of elements per load instruction
        """
        self.vector_size = vector_size
    
    def __repr__(self):
        return f"CopyG2LOp(vec={self.vector_size})"
    
    def __str__(self):
        return f"Global -> LDS copy (buffer_load x{self.vector_size})"


class CopyL2ROp:
    """LDS to Register copy operation (ds_read)."""
    
    def __init__(self, vector_size: int = 4):
        """Initialize L2R copy operation.
        
        Args:
            vector_size: Number of elements per read instruction
        """
        self.vector_size = vector_size
    
    def __repr__(self):
        return f"CopyL2ROp(vec={self.vector_size})"
    
    def __str__(self):
        return f"LDS -> Register copy (ds_read x{self.vector_size})"


class CopyR2GOp:
    """Register to Global copy operation (buffer_store)."""
    
    def __init__(self, vector_size: int = 8):
        """Initialize R2G copy operation.
        
        Args:
            vector_size: Number of elements per store instruction
        """
        self.vector_size = vector_size
    
    def __repr__(self):
        return f"CopyR2GOp(vec={self.vector_size})"
    
    def __str__(self):
        return f"Register -> Global copy (buffer_store x{self.vector_size})"


class CopyR2LOp:
    """Register to LDS copy operation (ds_write)."""
    
    def __init__(self, vector_size: int = 4):
        """Initialize R2L copy operation.
        
        Args:
            vector_size: Number of elements per write instruction
        """
        self.vector_size = vector_size
    
    def __repr__(self):
        return f"CopyR2LOp(vec={self.vector_size})"
    
    def __str__(self):
        return f"Register -> LDS copy (ds_write x{self.vector_size})"


#===----------------------------------------------------------------------===//
# ROCm Operation Helpers
#===----------------------------------------------------------------------===//

def make_tensor(ptr: Value, layout: Value, element_type: Type, 
                loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a tensor from pointer and layout.
    
    Args:
        ptr: Pointer to memory
        layout: Layout describing the tensor structure
        element_type: Element type of the tensor
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Tensor value
    """
    # This would use flir_rocm.make_tensor operation
    # For now, return a placeholder
    return ptr


def make_fragment(layout: Value, element_type: Type,
                  loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a fragment (register memory tensor).
    
    Args:
        layout: Layout describing the fragment structure
        element_type: Element type of the fragment
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Fragment value
    """
    # This would use flir_rocm.make_fragment operation
    # For now, return a placeholder
    return layout


def partition_src(tiled_copy: Value, src_tensor: Value, thr_idx: Value,
                  loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Partition source tensor according to tiled copy layout.
    
    Args:
        tiled_copy: Tiled copy descriptor
        src_tensor: Source tensor to partition
        thr_idx: Thread index
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Partitioned tensor for this thread
    """
    # This would use flir_rocm.partition_src operation
    return src_tensor


def partition_dst(tiled_copy: Value, dst_tensor: Value, thr_idx: Value,
                  loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Partition destination tensor according to tiled copy layout.
    
    Args:
        tiled_copy: Tiled copy descriptor
        dst_tensor: Destination tensor to partition
        thr_idx: Thread index
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Partitioned tensor for this thread
    """
    # This would use flir_rocm.partition_dst operation
    return dst_tensor


#===----------------------------------------------------------------------===//
# MFMA Operations
#===----------------------------------------------------------------------===//

class MfmaOp:
    """MFMA (Matrix Fused Multiply-Add) operation descriptor.
    
    Represents an AMD matrix core instruction for GFX942.
    """
    
    def __init__(self, shape: tuple, a_type: Type, b_type: Type, c_type: Type, arch: str = "gfx942"):
        """Initialize MFMA operation.
        
        Args:
            shape: (M, N, K) dimensions of the MFMA instruction
            a_type: Element type for A operand
            b_type: Element type for B operand
            c_type: Element type for C/D operand
            arch: Target architecture (default: gfx942)
        """
        self.shape = shape
        self.a_type = a_type
        self.b_type = b_type
        self.c_type = c_type
        self.arch = arch
    
    def __repr__(self):
        return f"MfmaOp({self.shape}, {self.arch})"
    
    def __str__(self):
        m, n, k = self.shape
        return f"MFMA {m}x{n}x{k} ({self.arch})"


__all__ = [
    # Copy operations
    "CopyUniversalOp",
    "CopyG2LOp",
    "CopyL2ROp",
    "CopyR2GOp",
    "CopyR2LOp",
    # MFMA operations
    "MfmaOp",
    # Operation helpers
    "make_tensor",
    "make_fragment",
    "partition_src",
    "partition_dst",
]

