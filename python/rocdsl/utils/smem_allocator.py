import sys
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Union, Type, Tuple, Any, Optional, Callable, Dict

from mlir import ir
from mlir.dialects import arith, memref, gpu
import mlir.extras.types as T
from rocdsl.dialects.ext.gpu import lds_space

# ==============================================================================
# Type Utilities
# ==============================================================================

def get_mlir_type_size(mlir_type: ir.Type) -> int:
    """Returns the size in bytes of an MLIR type."""
    if mlir_type == T.f32() or mlir_type == T.i32(): return 4
    if mlir_type == T.f16() or mlir_type == T.bf16() or mlir_type == T.i16(): return 2
    if mlir_type == T.i8(): return 1
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
    Analogue to cute::Pointer<T>.
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
            idxs = [arith.constant(T.index(), 0) for _ in range(len(self.shape) if self.shape else 1)]
        return memref.load(view, idxs)
    
    def store(self, val, idxs=None):
        """Helper to store value. If scalar, idxs defaults to [0]."""
        view = self.get()
        if idxs is None: 
            idxs = [arith.constant(T.index(), 0) for _ in range(len(self.shape) if self.shape else 1)]
        memref.store(val, view, idxs)

# ==============================================================================
# Struct Support
# ==============================================================================

class SmemStructInstance:
    """
    A proxy object that intercepts attribute access and maps them to SmemPtrs.
    """
    def __init__(self, base_memref, start_offset, field_layout: Dict[str, Tuple[int, ir.Type]]):
        self._base_memref = base_memref
        self._start_offset = start_offset
        self._field_layout = field_layout # Dict[name, (offset, type)]

    def __getattr__(self, name):
        if name in self._field_layout:
            rel_offset, dtype = self._field_layout[name]
            # Return a SmemPtr to the field,
            # assuming scalars (recursion needed for nested structs).
            return SmemPtr(self._base_memref, self._start_offset + rel_offset, dtype)
        raise AttributeError(f"Struct has no field '{name}'")

# ==============================================================================
# Allocator
# ==============================================================================

class SmemAllocator:
    def __init__(self, ctx, arch: Optional[str] = None):
        self.ctx = ctx
        self.ptr = 0
        self.max_size = 0
        self.alignment = 128 # Base alignment for the whole buffer
        self.finalized = False
        self.base_buffer_val = None 
        self.global_sym_name = "smem_storage"
        self.arch = arch
        
    def init_tracker(self):
        """
        Call this at the start of compilation to reset tracking.
        """
        self.ptr = 0
        self.max_size = 0
        self.finalized = False
    
    def _align(self, ptr, align):
        if ptr % align == 0: return ptr
        return (ptr + align - 1) // align * align

    def get_dyn_smem(self, dtype=None, alignment=1024):
        """
        Analogue to get_dyn_smem.
        Returns the 'base' pointer generator for dynamic shared memory.
        
        Currently, MLIR GPU dialect usually handles dynamic smem via `gpu.dynamic_shared_memory`.
        This method is a placeholder to align with CuTe semantics.
        """
        if dtype is None:
            dtype = T.i8()
        # But usually you choose one.
        raise NotImplementedError("Dynamic SMEM support is not yet fully implemented.")

    def allocate(self, size_or_type_or_struct: Union[int, ir.Type, Any], alignment=None):
        """
        The master allocation function.
        Returns a generator function that accepts the base pointer.
        """
        
        allocated_bytes = 0
        generator = None

        # Mode 1: Raw Bytes
        if isinstance(size_or_type_or_struct, int):
            size_bytes = size_or_type_or_struct
            align = alignment if alignment else 1
            
            offset = self._align(self.ptr, align)
            self.ptr = offset + size_bytes
            allocated_bytes = size_bytes
            
            generator = lambda base: SmemPtr(base, offset, T.i8(), shape=(size_bytes,))

        # Mode 2: MLIR Type (Scalar)
        elif isinstance(size_or_type_or_struct, ir.Type):
            dtype = size_or_type_or_struct
            size_bytes = get_mlir_type_size(dtype)
            align = alignment if alignment else get_mlir_type_align(dtype)
            
            offset = self._align(self.ptr, align)
            self.ptr = offset + size_bytes
            allocated_bytes = size_bytes
            
            generator = lambda base: SmemPtr(base, offset, dtype, shape=None)

        # Mode 3: Struct (Python Class decorated with @dataclass or similar)
        elif is_dataclass(size_or_type_or_struct) or hasattr(size_or_type_or_struct, '__annotations__'):
            cls = size_or_type_or_struct
            
            # Calculate Layout
            current_struct_offset = 0
            struct_align = 1
            field_layout = {} # name -> (offset, type)
            
            # Get type hints
            hints = getattr(cls, '__annotations__', {})
            
            for name, type_hint in hints.items():
                field_dtype = type_hint 
                if not isinstance(field_dtype, ir.Type):
                    if callable(field_dtype):
                        try:
                            field_dtype = field_dtype()
                        except:
                            pass
                    
                if not isinstance(field_dtype, ir.Type):
                    raise ValueError(f"Field '{name}' in struct '{cls.__name__}' must be typed with an MLIR Type")
                
                f_size = get_mlir_type_size(field_dtype)
                f_align = get_mlir_type_align(field_dtype)
                
                # Align field
                current_struct_offset = self._align(current_struct_offset, f_align)
                field_layout[name] = (current_struct_offset, field_dtype)
                
                current_struct_offset += f_size
                struct_align = max(struct_align, f_align)
            
            # Struct total size usually aligned to max align
            total_struct_size = self._align(current_struct_offset, struct_align)
            
            # Allocate in global buffer
            align = alignment if alignment else struct_align
            base_offset = self._align(self.ptr, align)
            self.ptr = base_offset + total_struct_size
            allocated_bytes = total_struct_size
            
            def struct_generator(base):
                return SmemStructInstance(base, base_offset, field_layout)
            
            generator = struct_generator

        else:
            raise ValueError(f"Unsupported argument to allocate: {size_or_type_or_struct}")

        # Check Capacity
        check_smem_capacity(self.ptr, self.arch)
        
        return generator

    def allocate_array(self, dtype: ir.Type, num_elems: int, alignment=None):
        """
        Allocates a contiguous array of type dtype.
        """
        elem_size = get_mlir_type_size(dtype)
        total_size = elem_size * num_elems
        align = alignment if alignment else get_mlir_type_align(dtype)
        
        offset = self._align(self.ptr, align)
        self.ptr = offset + total_size
        
        check_smem_capacity(self.ptr, self.arch)
        
        def array_generator(base):
            return SmemPtr(base, offset, dtype, shape=(num_elems,))
            
        return array_generator

    def allocate_tensor(self, layout, element_type, swizzle=None):
        """
        allocate_tensor(Layout, Type, Swizzle) -> Tensor Generator
        
        layout: Must be a tuple (shape) or object with .cosize()
        """
        # 1. Calculate cosize (domain size)
        if hasattr(layout, 'cosize'):
            # Assuming layout is a Rocir layout or similar
            num_elements = layout.cosize()
            shape = getattr(layout, 'shape', None) # Try to preserve shape info
        elif isinstance(layout, tuple):
            # Simple shape tuple
            num_elements = 1
            for s in layout: num_elements *= s
            shape = layout
        else:
            raise ValueError("Layout must be a tuple or have .cosize()")
                
        element_size = get_mlir_type_size(element_type)
        total_bytes = num_elements * element_size
        
        # 2. Allocate
        # Tensor allocations usually want high alignment for Vectorized Access
        align = 16 
        offset = self._align(self.ptr, align)
        self.ptr = offset + total_bytes
        
        check_smem_capacity(self.ptr, self.arch)
        
        # 3. Return Tensor Generator
        def tensor_generator(base):
            # Returns a SmemPtr viewing memory as the specified tensor shape,
            # currently serving as a simplified placeholder for full layout logic.
            # It uses the provided shape tuple directly
            # or defaults to a flat 1D view for opaque layout objects.
            
            tensor_shape = shape if isinstance(shape, tuple) else (num_elements,)
            
            return SmemPtr(base, offset, element_type, shape=tensor_shape)
            
        return tensor_generator

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
        # Try to detect arch from environment or rocdsl context if possible
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


