"""Compile a fixed B/S and compile-path specialization without executing GPU code."""
import argparse
import ast
from dataclasses import asdict
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
from kernels.kimi_k3_monokernel.compile_config import KimiK3CompileConfig
from kernels.monokernel.config import MAX_LAYERS_PER_STEP, KIMI_K3_CONFIG

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch', type=int, choices=range(1, 9), required=True)
parser.add_argument('--seq', type=int, choices=range(1, 5), required=True)
parser.add_argument('--compile-path', choices=['auto', 'small_batch', 'general'], default='auto')
parser.add_argument('--input-row-groups', type=int, choices=[1, 2, 4, 8])
parser.add_argument('--staged-samples', type=int, choices=[1, 2, 3, 4])
parser.add_argument('--input-schedule', choices=['auto', 'cta', 'flat'], default='auto')
parser.add_argument('--gate-schedule', choices=['auto', 'serial', 'overlap'], default='auto')
parser.add_argument('--router-reduction', choices=['auto', 'two_pass', 'pair'], default='auto')
parser.add_argument('--route-publication', choices=['auto', 'stream', 'wave'], default='auto')
parser.add_argument('--output-prefetch-units', choices=['auto', '0', '1', '6'], default='auto')
parser.add_argument('--conv-schedule', choices=['auto', 'serial', 'independent'], default='auto')
parser.add_argument('--norm-reduction', choices=['auto', 'mailbox', 'direct'], default='auto')
parser.add_argument('--down-task-mapping', choices=['auto','linear','ready64'], default='auto')
parser.add_argument('--input-mfma', choices=['auto','f32','bf16_k16'], default='auto')
parser.add_argument('--out', type=Path, required=True)
cli = parser.parse_args()
extra = {'input_schedule': cli.input_schedule} if cli.input_schedule != 'auto' else {}
for field in ['gate_schedule', 'router_reduction', 'route_publication', 'conv_schedule', 'norm_reduction', 'down_task_mapping', 'input_mfma']:
    if getattr(cli, field) != 'auto':
        extra[field] = getattr(cli, field)
if cli.output_prefetch_units != 'auto':
    extra['output_prefetch_units'] = int(cli.output_prefetch_units)
specialization = KimiK3CompileConfig(cli.compile_path, cli.staged_samples, cli.input_row_groups, **extra)
resolved = specialization.resolve(cli.batch * cli.seq, cli.seq > 1, cli.seq)
out = cli.out
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
config = dict(samples=cli.batch * cli.seq, npes=8, launches_per_step=MAX_LAYERS_PER_STEP,
              attn_res_blocks=(layer_idx + block - 1) // block,
              block_write_idx=layer_idx // block if layer_idx % block == 0 else -1,
              fuse_moe=True, mtp=cli.seq > 1, mtp_seq_len=cli.seq,
              compile_config=specialization)
print(json.dumps(dict(stage="compile_only", specialization=asdict(resolved), source=kernel.__file__)), flush=True)
flyc.compile(kernel.build_kimi_k3_monokernel(**config), *args)
config['compile_config'] = asdict(specialization)

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
source_root = Path(kernel.__file__).parents[2]
manifest = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for subdir in ['kernels/kimi_k3_monokernel', 'kernels/monokernel']
            for p in (source_root / subdir).rglob('*.py')}
result = dict(config=config, specialization=asdict(resolved), grid_blocks=getattr(resolved, 'grid_blocks', kernel._BLOCKS), source_manifest=manifest, source=kernel.__file__,
              source_sha256=hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
              kernels=records, gpu_execution=False, hip_occupancy_checked=False)
(out / "compile.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
assert len(records) == 1, records
assert records[0]["max_flat_workgroup_size"] == 512, records
