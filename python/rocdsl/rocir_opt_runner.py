"""Run rocir-opt as subprocess to execute passes."""

import subprocess
import tempfile
import re
from pathlib import Path
from mlir.ir import Module

def convert_generic_to_pretty(ir_str: str) -> str:
    """Convert MLIR generic format to a pretty layout-specific format."""
    def convert_op(match):
        prefix = match.group(1)
        op_name = match.group(2)
        operands = match.group(3).strip()
        type_sig = match.group(4)
        
        # Check operand count
        operand_list = [op.strip() for op in operands.split(',') if op.strip()]
        
        if len(operand_list) == 1:
            # Single operand: remove parens from input type
            type_sig = re.sub(r':\s*\(([^)]+)\)\s*->', r': \1 ->', type_sig)
        
        return f'{prefix}{op_name} {operands} {type_sig}'
    
    # Pattern: "rocir.op"(operands) : types
    pattern = r'(%\w+\s*=\s*)"(cute\.\w+)"\(([^)]*)\)(\s*:.+?)(?=\n|$)'
    return re.sub(pattern, convert_op, ir_str)

def run_rocir_opt(module: Module, passes: str) -> Module:
    """Run rocir-opt with given passes and return transformed module."""
    print(f"[DEBUG] run_cute_opt called with passes: {passes}")
    
    # Get rocir-opt path
    cute_opt = Path("/mnt/raid0/felix/rocDSL/build/tools/rocir-opt/rocir-opt")
    if not cute_opt.exists():
        raise RuntimeError(f"rocir-opt not found at {rocir_opt}")
    
    # Write module to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        # Convert generic format to pretty format
        module_str = str(module)
        converted_str = convert_generic_to_pretty(module_str)
        
        f.write(converted_str)
        input_file = f.name
        print(f"[DEBUG] Wrote converted IR to: {input_file}")
    
    try:
        # Run rocir-opt
        cmd = [str(rocir_opt), f"--{passes}", input_file]
        print(f"[DEBUG] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"[DEBUG] rocir-opt exit code: {result.returncode}")
        print(f"[DEBUG] Output length: {len(result.stdout)} chars")
        print(f"[DEBUG] First 200 chars: {result.stdout[:200]}")
        
        # Parse output back to Module
        return Module.parse(result.stdout, context=module.context)
        
    except subprocess.CalledProcessError as e:
        print(f"[DEBUG] rocir-opt failed with exit code: {e.returncode}")
        print(f"[DEBUG] stderr: {e.stderr[:500]}")
        raise RuntimeError(f"rocir-opt execution failed: {e.stderr}") from e
    finally:
        # Clean up temp file
        Path(input_file).unlink(missing_ok=True)

def rocir_to_standard(module: Module) -> Module:
    """Lower Rocir IR to standard dialects using rocir-opt."""
    return run_rocir_opt(module, "rocir-to-standard")
