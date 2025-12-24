"""Shared utilities for GPU testing, compilation, and benchmarking."""

from pyflir.compiler.pipeline import Pipeline, run_pipeline
from pyflir.runtime.device import get_rocm_arch
from _mlir import ir
import os
import torch
import functools
import time
from typing import Optional

def compile_to_hsaco(mlir_module, kernel_name="kernel", waves_per_eu: Optional[int] = None):
    """
    Compile MLIR module to HSACO binary for AMD GPUs.
    
    Pipeline:
    1. Apply flir coordinate lowering (flir ops -> arithmetic)
    2. Lower remaining Flir ops to standard dialects (flir-to-standard)
    2. Canonicalize and CSE
    3. Attach ROCDL target for current GPU architecture
    4. Convert GPU dialect to ROCDL
    5. Lower to LLVM
    6. Generate binary
    
    Environment Variables:
    - FLIR_DUMP_IR=1: Enable IR dumping at each compilation stage
    - FLIR_DUMP_DIR=/path/to/dir: Directory to save IR files (default: /tmp/flir_dump)
    - FLIR_ENABLE_IR_PRINTING=1: Print IR to console during compilation
    
    Args:
        mlir_module: MLIR module containing GPU kernels
        kernel_name: Name prefix for dumped files (default: "kernel")
        
    Returns:
        bytes: HSACO binary object
    """
    _ctx = getattr(mlir_module, "context", None)
    if _ctx is None and hasattr(mlir_module, "operation"):
        _ctx = getattr(mlir_module.operation, "context", None)

    if _ctx is None:
        # Best-effort fallback (older callsites may rely on an ambient default context).
        return _compile_to_hsaco_impl(mlir_module, kernel_name=kernel_name, waves_per_eu=waves_per_eu)

    with _ctx:
        return _compile_to_hsaco_impl(mlir_module, kernel_name=kernel_name, waves_per_eu=waves_per_eu)


def _compile_to_hsaco_impl(mlir_module, kernel_name="kernel", waves_per_eu: Optional[int] = None):
    """Implementation of compile_to_hsaco; assumes an MLIR context is already active."""
    # Check environment variables for IR dumping
    dump_ir = os.environ.get('FLIR_DUMP_IR', '0') == '1'
    dump_dir = os.environ.get('FLIR_DUMP_DIR', '/tmp/flir_dump')
    enable_ir_printing = os.environ.get('FLIR_ENABLE_IR_PRINTING', '0') == '1'
    time_stages = os.environ.get("FLIR_TIME_COMPILE", "0") == "1"
    t0_all = time.perf_counter()
    stage_times_ms = {}

    def _timeit(name, fn):
        if not time_stages:
            return fn()
        t0 = time.perf_counter()
        out = fn()
        stage_times_ms[name] = (time.perf_counter() - t0) * 1e3
        return out
    
    # Create dump directory if needed
    if dump_ir:
        os.makedirs(dump_dir, exist_ok=True)
        print(f"  📁 IR dump directory: {dump_dir}")
    
    def dump_stage(module, stage_name):
        """Helper function to dump IR at a specific stage."""
        if dump_ir:
            ir_str = str(module)
            filename = os.path.join(dump_dir, f"{kernel_name}_{stage_name}.mlir")
            with open(filename, 'w') as f:
                f.write(ir_str)
            print(f"  Dumped {stage_name}: {filename}")
            
            if enable_ir_printing:
                print(f"\n{'='*80}")
                print(f"IR Stage: {stage_name}")
                print(f"{'='*80}")
                print(ir_str)
                print(f"{'='*80}\n")

    def _apply_waves_per_eu_on_llvm_funcs(module):
        """Apply AMDGPU waves-per-eu hint to llvm.func ops via LLVM passthrough."""
        if waves_per_eu is None:
            return
        w = int(waves_per_eu)
        new_entry = ir.StringAttr.get(f"amdgpu-waves-per-eu={w},{w}")
        new_entry_str = str(new_entry).strip('"')

        def _append_passthrough(func_op):
            existing = func_op.attributes.get("passthrough", None)
            if existing is None:
                func_op.attributes["passthrough"] = ir.ArrayAttr.get([new_entry])
                return

            # Best-effort: if it's not an ArrayAttr-like object, just overwrite.
            try:
                existing_entries = list(existing)
            except TypeError:
                func_op.attributes["passthrough"] = ir.ArrayAttr.get([new_entry])
                return

            if any(str(a).strip('"') == new_entry_str for a in existing_entries):
                return
            func_op.attributes["passthrough"] = ir.ArrayAttr.get(existing_entries + [new_entry])

        try:
            for op in module.body.operations:
                if getattr(op, "OPERATION_NAME", None) != "gpu.module":
                    continue
                for inner_op in op.body.blocks[0].operations:
                    if getattr(inner_op, "OPERATION_NAME", None) != "llvm.func":
                        continue
                    if "gpu.kernel" not in inner_op.attributes:
                        continue
                    _append_passthrough(inner_op)
        except Exception:
            # Best-effort only.
            return
    
    # Dump initial IR (with flir ops)
    dump_stage(mlir_module, "01_initial")
    
    # Lower Flir ops (includes coordinate lowering) before any GPU/LLVM conversion.
    lowered_module = _timeit("02_flir_to_standard",
                             lambda: run_pipeline(mlir_module, Pipeline().flir_to_standard()))
    dump_stage(lowered_module, "02_flir_to_standard")

    # Dead-code elimination (module-level). In our embedded environment upstream
    # MLIR does not register a generic 'dce/adce', so we provide a standalone
    # pass 'trivial-dce'.
    lowered_module = _timeit("02c_trivial_dce",
                             lambda: run_pipeline(lowered_module, Pipeline().add_pass("trivial-dce")))
    dump_stage(lowered_module, "02c_trivial_dce")
    
    # Get the current GPU architecture
    gpu_arch = get_rocm_arch()
    
    # Build pipeline step by step for intermediate dumps
    # Stage 1: Canonicalize
    canonicalized = _timeit(
        "03_canonicalize",
        lambda: run_pipeline(lowered_module, Pipeline().canonicalize()),
    )
    dump_stage(canonicalized, "03_canonicalized")
    
    # Stage 2: CSE
    cse_result = _timeit(
        "04_cse",
        lambda: run_pipeline(canonicalized, Pipeline().cse()),
    )
    dump_stage(cse_result, "04_cse")
    
    # Stage 3: Attach ROCDL target
    with_target = _timeit(
        "05_attach_target",
        lambda: run_pipeline(cse_result, Pipeline().rocdl_attach_target(chip=gpu_arch)),
    )
    dump_stage(with_target, "05_with_target")
    
    # Stage 4: Convert GPU to ROCDL (with SCF to CF inside GPU module)
    rocdl_converted = _timeit(
        "06_convert_gpu_to_rocdl",
        lambda: run_pipeline(
            with_target,
            Pipeline().Gpu(
                Pipeline()
                .convert_scf_to_cf()  # Lower SCF loops first inside GPU module
                .convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP")
                .reconcile_unrealized_casts()
            ),  # Clean up inside GPU module
        ),
    )
    dump_stage(rocdl_converted, "06_rocdl")
    
    # Stage 5: GPU to LLVM  
    llvm_converted = _timeit(
        "07_gpu_to_llvm",
        lambda: run_pipeline(rocdl_converted, Pipeline().gpu_to_llvm().reconcile_unrealized_casts()),
    )
    dump_stage(llvm_converted, "07_llvm_gpu")
    
    # Stage 6: Lower to LLVM (includes CF to LLVM conversion)
    llvm_lowered = _timeit(
        "08_lower_to_llvm",
        lambda: run_pipeline(llvm_converted, Pipeline().lower_to_llvm().reconcile_unrealized_casts()),
    )
    _apply_waves_per_eu_on_llvm_funcs(llvm_lowered)
    dump_stage(llvm_lowered, "08_llvm_final")
    
    # Stage 7: Generate binary (ISA/assembly)
    # For assembly dump, we need to use a different format first
    if dump_ir:
        try:
            # Try to get assembly output
            asm_module = run_pipeline(
                ir.Module.parse(str(llvm_lowered), context=llvm_lowered.context),
                Pipeline().gpu_module_to_binary(format="isa")
            )
            from pyflir.dialects.ext.gpu import get_compile_object_bytes
            asm_bytes = get_compile_object_bytes(asm_module)
            asm_filename = os.path.join(dump_dir, f"{kernel_name}_09_assembly.s")
            with open(asm_filename, 'wb') as f:
                f.write(asm_bytes)
            print(f"  Dumped assembly: {asm_filename}")
        except Exception as e:
            print(f"  Could not dump assembly: {e}")
    
    # Final binary generation
    lowered = _timeit(
        "10_gpu_module_to_binary_bin",
        lambda: run_pipeline(llvm_lowered, Pipeline().gpu_module_to_binary(format="bin")),
    )
    dump_stage(lowered, "10_binary_module")
    
    from pyflir.dialects.ext.gpu import get_compile_object_bytes
    hsaco_bytes = _timeit("11_get_compile_object_bytes", lambda: get_compile_object_bytes(lowered))
    
    # Save HSACO binary if dumping
    if dump_ir:
        hsaco_filename = os.path.join(dump_dir, f"{kernel_name}_11_final.hsaco")
        with open(hsaco_filename, 'wb') as f:
            f.write(hsaco_bytes)
        print(f"  Dumped HSACO binary: {hsaco_filename}")
    
    if time_stages:
        total_ms = (time.perf_counter() - t0_all) * 1e3
        print(f"\n[FLIR TIME] compile_to_hsaco({kernel_name}) arch={gpu_arch}")
        for k, v in stage_times_ms.items():
            print(f"  - {k}: {v:.1f} ms")
        print(f"  - total: {total_ms:.1f} ms")
    return hsaco_bytes


# Simple dtypes namespace for compatibility
class dtypes:
    fp32 = torch.float32
    fp16 = torch.float16
    bf16 = torch.bfloat16
    i8 = torch.int8
    fp8 = torch.float8_e4m3fn

@functools.lru_cache()
def get_dtype_max(dtype):
    """Get max value for a given dtype."""
    try:
        dtypeMax = torch.finfo(dtype).max
    except:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax

def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=dtypes.fp32,
    quant_dtype=dtypes.i8,
    dtypeMax=None,
):
    x = x.to(dtypes.fp32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = scale
    if scale is None:
        # [m, 1]
        per_token_amax, _ = torch.max(
            input=torch.abs(hidden_states), dim=-1, keepdim=True
        )
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_