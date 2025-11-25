"""HIP runtime utilities for GPU kernel execution"""

import ctypes
from typing import Tuple, Optional
import time


def hip_check(result):
    """Check HIP API call result and raise exception on error"""
    from hip import hip, hiprtc
    
    if isinstance(result, tuple):
        err = result[0]
        if len(result) > 1:
            ret_val = result[1] if len(result) == 2 else result[1:]
        else:
            ret_val = None
    else:
        err = result
        ret_val = None
    
    if err != hip.hipError_t.hipSuccess:
        error_str = hip.hipGetErrorString(err)
        raise RuntimeError(f'HIP Error: {error_str}')
    
    return ret_val


def get_hip_arch() -> str:
    """Get the HIP GPU architecture string (e.g., 'gfx942')"""
    from hip import hip
    
    # Create props struct
    props = hip.hipDeviceProp_t()
    
    # Get properties for device 0
    hip_check(hip.hipGetDeviceProperties(props, 0))
    
    # Extract gfx architecture
    gcn_arch_name = props.gcnArchName.decode('utf-8')
    # Format: gfxXXX
    if ':' in gcn_arch_name:
        gcn_arch_name = gcn_arch_name.split(':')[0]
    
    return gcn_arch_name


def launch_kernel(
    function_ptr: int,
    blocks_x: int,
    blocks_y: int,
    blocks_z: int,
    threads_x: int,
    threads_y: int,
    threads_z: int,
    stream: int,
    shared_mem_bytes: int,
    *kernel_args
) -> float:
    """Launch a GPU kernel and return execution time in milliseconds
    
    Args:
        function_ptr: Pointer to the kernel function
        blocks_x, blocks_y, blocks_z: Grid dimensions
        threads_x, threads_y, threads_z: Block dimensions
        stream: CUDA/HIP stream
        shared_mem_bytes: Shared memory size in bytes
        *kernel_args: Kernel arguments (device pointers or scalar values)
        
    Returns:
        Execution time in milliseconds
    """
    from hip import hip
    
    # Create events for timing
    start_event = hip_check(hip.hipEventCreate())
    stop_event = hip_check(hip.hipEventCreate())
    
    # Convert arguments to ctypes pointers
    args_list = []
    for arg in kernel_args:
        if isinstance(arg, int):
            # Device pointer or scalar integer
            arg_ptr = ctypes.c_void_p(arg)
            args_list.append(arg_ptr)
        elif isinstance(arg, float):
            arg_val = ctypes.c_float(arg)
            args_list.append(arg_val)
        else:
            args_list.append(arg)
    
    # Create array of pointers to arguments
    kernel_params = (ctypes.c_void_p * len(args_list))()
    for i, arg in enumerate(args_list):
        kernel_params[i] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p).value
    
    # Record start event
    hip_check(hip.hipEventRecord(start_event, stream))
    
    # Launch kernel
    hip_check(hip.hipModuleLaunchKernel(
        ctypes.c_void_p(function_ptr),
        blocks_x, blocks_y, blocks_z,
        threads_x, threads_y, threads_z,
        shared_mem_bytes,
        stream,
        kernel_params,
        None
    ))
    
    # Record stop event
    hip_check(hip.hipEventRecord(stop_event, stream))
    
    # Wait for completion
    hip_check(hip.hipEventSynchronize(stop_event))
    
    # Calculate elapsed time
    elapsed_ms = hip_check(hip.hipEventElapsedTime(start_event, stop_event))
    
    # Cleanup events
    hip_check(hip.hipEventDestroy(start_event))
    hip_check(hip.hipEventDestroy(stop_event))
    
    return elapsed_ms
