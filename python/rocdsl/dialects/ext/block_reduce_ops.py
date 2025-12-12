"""
High-level collective operations for GPU programming.

This module provides block-level and warp-level collective operations that
combine multiple low-level GPU primitives (shuffle, shared memory, barriers)
into commonly used patterns for parallel reductions and communication.

Example:
    >>> from rocdsl.dialects.ext import collective_ops
    >>> max_val = collective_ops.block_reduce_max(val, smem, tid, num_warps=4)
"""

from typing import Optional, Callable
from mlir.ir import Value, InsertionPoint
from mlir.dialects import gpu as mlir_gpu, memref, scf

try:
    from rocdsl.dialects.ext import arith
except ImportError:
    from . import arith


def _block_reduce(
    val: Value,
    red_smem: Value,
    tid: Value,
    reduction_op: Callable,
    identity_val: float = 0.0,
    num_warps: int = 4,
    warp_size: int = 64,
) -> Value:
    """Generic block-level reduction operation.
    
    This is an internal helper function that implements the common pattern
    for block-level reductions. It should not be called directly; instead,
    use the specific reduction functions like block_reduce_max or block_reduce_sum.
    
    Args:
        val: Input value to reduce
        red_smem: Shared memory buffer for inter-warp reduction
        tid: Thread ID within the block
        reduction_op: Binary operation function (e.g., arith.maximum, operator.add)
        identity_val: Identity value for the reduction (e.g., 0.0 for sum, -inf for max)
        num_warps: Number of warps per block
        warp_size: Number of threads per warp
    
    Returns:
        The reduced value across all threads in the block (as ArithValue).
    """
    # Stage 1: Warp-level reduction using shuffle
    current_val = arith.ArithValue(val)
    
    # Perform log₂(warp_size) shuffle reductions
    # For warp_size=64: [32, 16, 8, 4, 2, 1]
    shuffle_offsets = [warp_size // (2 ** i) for i in range(1, 
                       int(warp_size).bit_length())]
    
    for offset in shuffle_offsets:
        offset_val = arith.i32(offset)
        width_val = arith.i32(warp_size)
        shuffled_op = mlir_gpu.ShuffleOp(
            current_val.value,
            offset_val.value,
            width_val.value,
            mode=mlir_gpu.ShuffleMode.XOR,
        )
        shuffled_val = shuffled_op.results[0]
        # Apply the reduction operation
        current_val = reduction_op(current_val, shuffled_val)
    
    # Wrap tid in ArithValue for arithmetic operations
    tid_arith = arith.ArithValue(tid)
    
    # Calculate warp ID and lane ID
    c_warp_size = arith.index(warp_size)
    c_0_idx = arith.index(0)
    warp_id = tid_arith / c_warp_size
    lane_id = tid_arith % c_warp_size
    
    # Stage 2: Lane 0 of each warp writes to shared memory
    is_lane_0 = lane_id == c_0_idx
    
    if_op = scf.IfOp(is_lane_0.value)
    with InsertionPoint(if_op.then_block):
        memref.store(current_val.value, red_smem, [warp_id.value])
        scf.YieldOp([])
    
    # Synchronize all threads before reading from shared memory
    mlir_gpu.BarrierOp()
    
    # Stage 3: Thread 0 computes final reduction across warps
    is_thread_0 = tid_arith == c_0_idx
    
    if_op2 = scf.IfOp(is_thread_0.value)
    with InsertionPoint(if_op2.then_block):
        # Initialize accumulator with identity value
        final_val = arith.ArithValue(arith.f32(identity_val))
        
        # Reduce across all warp results
        for w in range(num_warps):
            c_w_val = arith.index(w).value
            val_from_warp = memref.load(red_smem, [c_w_val])
            # Apply the reduction operation
            final_val = reduction_op(final_val, val_from_warp)
        
        # Write final result back to shared memory
        memref.store(final_val.value, red_smem, [c_0_idx.value])
        scf.YieldOp([])
    
    # Stage 4: All threads read final result
    mlir_gpu.BarrierOp()
    final_result = memref.load(red_smem, [c_0_idx.value])
    
    return arith.ArithValue(final_result)


def block_reduce_max(
    val: Value,
    red_smem: Value,
    tid: Value,
    num_warps: int = 4,
    warp_size: int = 64,
) -> Value:
    """Perform block-level max reduction across all threads.
    
    This function reduces a per-thread value to find the maximum across all
    threads in the block. It uses a two-stage reduction:
    1. Warp-level reduction using shuffle operations (fast)
    2. Block-level reduction via shared memory (slower but necessary for cross-warp)
    
    Args:
        val: Input value to reduce (typically f32)
        red_smem: Shared memory buffer for inter-warp reduction
                  Shape should be at least (num_warps,)
        tid: Thread ID within the block (0 to BLOCK_SIZE-1)
        num_warps: Number of warps per block (default: 4)
        warp_size: Number of threads per warp (default: 64 for AMD GPUs)
    
    Returns:
        The maximum value across all threads in the block.
        This value is identical for all threads (as ArithValue).
    
    Implementation Details:
        - Stage 1: Each warp performs shuffle-based reduction (log₂(warp_size) steps)
        - Stage 2: Lane 0 of each warp writes result to shared memory
        - Stage 3: Thread 0 reads all warp results and computes final max
        - Stage 4: All threads read the final result from shared memory
        
    Memory Requirements:
        - Shared memory: num_warps * sizeof(element_type) bytes
        - No registers spilled to memory
    
    Example:
        >>> # Find maximum value across 256 threads
        >>> red_smem = allocate_shared(4, f32)  # 4 warps
        >>> tid = gpu.thread_id("x")
        >>> max_val = block_reduce_max(my_val, red_smem, tid, num_warps=4)
    """
    return _block_reduce(
        val=val,
        red_smem=red_smem,
        tid=tid,
        reduction_op=arith.maximum,
        identity_val=0.0,  # For max, we could use -inf, but 0.0 works for positive values
        num_warps=num_warps,
        warp_size=warp_size,
    )


def block_reduce_sum(
    val: Value,
    red_smem: Value,
    tid: Value,
    num_warps: int = 4,
    warp_size: int = 64,
) -> Value:
    """Perform block-level sum reduction across all threads.
    
    Similar to block_reduce_max but computes the sum instead of maximum.
    
    Args:
        val: Input value to reduce
        red_smem: Shared memory buffer for inter-warp reduction
        tid: Thread ID within the block
        num_warps: Number of warps per block (default: 4)
        warp_size: Number of threads per warp (default: 64)
    
    Returns:
        The sum of values across all threads in the block (as ArithValue).
    
    Example:
        >>> # Sum values across 256 threads
        >>> red_smem = allocate_shared(4, f32)  # 4 warps
        >>> tid = gpu.thread_id("x")
        >>> total = block_reduce_sum(my_val, red_smem, tid, num_warps=4)
    """
    # Use lambda for addition since ArithValue supports __add__
    return _block_reduce(
        val=val,
        red_smem=red_smem,
        tid=tid,
        reduction_op=lambda a, b: a + b,
        identity_val=0.0,
        num_warps=num_warps,
        warp_size=warp_size,
    )


def block_reduce_min(
    val: Value,
    red_smem: Value,
    tid: Value,
    num_warps: int = 4,
    warp_size: int = 64,
) -> Value:
    """Perform block-level min reduction across all threads.
    
    Similar to block_reduce_max but computes the minimum instead.
    
    Args:
        val: Input value to reduce
        red_smem: Shared memory buffer for inter-warp reduction
        tid: Thread ID within the block
        num_warps: Number of warps per block (default: 4)
        warp_size: Number of threads per warp (default: 64)
    
    Returns:
        The minimum value across all threads in the block (as ArithValue).
    
    Example:
        >>> # Find minimum value across 256 threads
        >>> red_smem = allocate_shared(4, f32)  # 4 warps
        >>> tid = gpu.thread_id("x")
        >>> min_val = block_reduce_min(my_val, red_smem, tid, num_warps=4)
    """
    return _block_reduce(
        val=val,
        red_smem=red_smem,
        tid=tid,
        reduction_op=arith.minimum,
        identity_val=float('inf'),  # For min, use +inf as identity
        num_warps=num_warps,
        warp_size=warp_size,
    )


__all__ = [
    'block_reduce_max',
    'block_reduce_sum',
    'block_reduce_min',
]
