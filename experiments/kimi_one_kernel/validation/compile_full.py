"""Compile the full S4 MTP layer without initializing HIP or launching a kernel."""
import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys

os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_GPU_ARCH"] = "gfx950"
import flydsl.compiler as flyc
from flydsl.expr.typing import Stream
from kernels.kimi_k3_monokernel import kernel
from kernels.monokernel.config import MAX_LAYERS_PER_STEP, KIMI_K3_CONFIG

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
tree = ast.parse(Path(kernel.__file__).read_text())
launch = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "launch")
args = []
for arg in launch.args.args:
    annotation = ast.unparse(arg.annotation)
    if annotation == "Int64":
        args.append(1 << 48)
    elif annotation == "Int32":
        args.append(0)
    elif annotation == "Stream":
        args.append(Stream(None))
    else:
        raise ValueError((arg.arg, annotation))
layer_idx = 1
block = KIMI_K3_CONFIG.attn_res_block_size
config = dict(samples=4, npes=8, launches_per_step=MAX_LAYERS_PER_STEP,
              attn_res_blocks=(layer_idx + block - 1) // block,
              block_write_idx=layer_idx // block if layer_idx % block == 0 else -1,
              fuse_moe=True, mtp=True)
if "--attention" in sys.argv[2:]:
    config["fuse_moe"] = False
print(json.dumps(dict(stage="compile_only", config=config, source=kernel.__file__)), flush=True)
flyc.compile(kernel.build_kimi_k3_monokernel(**config), *args)

records = []
for path in Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"]).rglob("*.pkl"):
    artifact = pickle.loads(path.read_bytes())
    ir = artifact.ir
    match = re.search(r'gpu.kernel_metadata<"(kimi_k3_[^"]+)".*?metadata = \{([^}]+)\}', ir)
    if not match:
        continue
    name, meta = match.groups()
    values = {k: int(v) for k, v in re.findall(r"(\w+) = (\d+) : i64", meta)}
    i = ir.index('bin = "') + len('bin = "')
    binary = bytearray()
    while ir[i] != '"':
        if ir[i] == "\\":
            if ir[i + 1] in '\\"':
                binary.append(ord(ir[i + 1]))
                i += 2
            else:
                binary.append(int(ir[i + 1:i + 3], 16))
                i += 3
        else:
            binary.append(ord(ir[i]))
            i += 1
    hsaco = out / (path.parent.name + ".hsaco")
    hsaco.write_bytes(binary)
    records.append(dict(name=name, hsaco=str(hsaco),
                        sha256=hashlib.sha256(binary).hexdigest(), **values))
result = dict(config=config, source=kernel.__file__,
              source_sha256=hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
              kernels=records, gpu_execution=False, hip_occupancy_checked=False)
(out / "compile.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
assert len(records) == 1, records
assert records[0]["max_flat_workgroup_size"] == 512, records
