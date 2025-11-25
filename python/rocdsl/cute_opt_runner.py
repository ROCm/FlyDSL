"""Run cute-opt as subprocess to execute passes."""

import subprocess
import tempfile
import re
from pathlib import Path
from mlir.ir import Module

def run_cute_opt(module: Module, passes: str) -> Module:
    """Run cute-opt with given passes and return transformed module.
    
    Args:
        module: Input MLIR module
        passes: Pass pipeline string (e.g., "cute-to-standard")
    
    Returns:
        Transformed MLIR module
    """
    # Get cute-opt path
    cute_opt = Path("/mnt/raid0/felix/rocDSL/build/tools/cute-opt/cute-opt")
    if not cute_opt.exists():
        raise RuntimeError(f"cute-opt not found at {cute_opt}")
    
    # Write module to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        # Get module string and convert generic format to pretty format
        module_str = str(module)
        # Replace generic "cute.op_name" with pretty cute.op_name
        module_str = re.sub(r'"(cute\.\w+)"', r'\1', module_str)
        
        f.write(module_str)
        input_file = f.name
    
    try:
        # Run cute-opt
        cmd = [str(cute_opt), f"--{passes}", input_file]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output back to Module
        return Module.parse(result.stdout, context=module.context)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"cute-opt execution failed: {e.stderr}") from e
    finally:
        # Clean up temp file
        Path(input_file).unlink(missing_ok=True)

def cute_to_standard(module: Module) -> Module:
    """Lower CuTe IR to standard dialects using cute-opt.
    
    Args:
        module: Input MLIR module with CuTe operations
    
    Returns:
        Lowered MLIR module
    """
    return run_cute_opt(module, "cute-to-standard")
