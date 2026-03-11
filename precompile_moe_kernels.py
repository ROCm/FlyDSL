#!/usr/bin/env python3
"""Precompile FlyDSL MOE fp8 kernels for use inside PyTorch processes.

Two-step pipeline:
  1. Compile kernel in subprocess (no PyTorch) → dump MLIR IR via FLYDSL_DUMP_IR
  2. Run external mlir-opt to lower to GPU binary → extract .hsaco ELF

Usage:
    DSL2_ROOT=/path/to/FlyDSL MLIR_PATH=/path/to/mlir_install \\
        python3 precompile_moe_kernels.py [--configs CONFIG_JSON]
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".flydsl" / "precompiled"
DUMP_DIR = Path.home() / ".flydsl" / "debug"

STAGE2_CONFIGS = [
    dict(model_dim=7168, inter_dim=2048, experts=33, topk=10,
         tile_m=32, tile_n=128, tile_k=128,
         doweight=True, a_dtype="fp8", out_dtype="bf16", accumulate=False),
    dict(model_dim=7168, inter_dim=2048, experts=33, topk=10,
         tile_m=64, tile_n=128, tile_k=128,
         doweight=True, a_dtype="fp8", out_dtype="bf16", accumulate=False),
    dict(model_dim=7168, inter_dim=2048, experts=33, topk=10,
         tile_m=32, tile_n=128, tile_k=128,
         doweight=False, a_dtype="fp8", out_dtype="bf16", accumulate=True),
    dict(model_dim=7168, inter_dim=2048, experts=33, topk=10,
         tile_m=64, tile_n=128, tile_k=128,
         doweight=False, a_dtype="fp8", out_dtype="bf16", accumulate=True),
]


def hsaco_name(cfg, chip):
    acc = "acc1" if cfg["accumulate"] else "acc0"
    dw = "dw1" if cfg["doweight"] else "dw0"
    return (
        f"moe_gemm2_{cfg['a_dtype']}_{cfg['a_dtype']}_{cfg['out_dtype']}_"
        f"{cfg['model_dim']}x{cfg['inter_dim']}_e{cfg['experts']}_t{cfg['topk']}_"
        f"tile{cfg['tile_m']}x{cfg['tile_n']}x{cfg['tile_k']}_{dw}_{acc}_{chip}"
    )


def extract_gpu_binary(ir_text):
    pattern = r'bin\s*=\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, ir_text, re.DOTALL)
    if not match:
        return None
    escaped = match.group(1)
    raw = bytearray()
    i = 0
    while i < len(escaped):
        if escaped[i] == '\\' and i + 1 < len(escaped):
            i += 1
            c = escaped[i]
            if c == '\\':   raw.append(0x5C)
            elif c == '"':  raw.append(0x22)
            elif c == 'n':  raw.append(0x0A)
            elif c == 't':  raw.append(0x09)
            else:
                if i + 1 < len(escaped):
                    try:
                        raw.append(int(escaped[i:i+2], 16))
                        i += 1
                    except ValueError:
                        raw.append(ord(c))
                else:
                    raw.append(ord(c))
        else:
            raw.append(ord(escaped[i]))
        i += 1
    return bytes(raw)


def compile_one(cfg, chip, cache_dir, mlir_opt_path, dsl2_root):
    name = hsaco_name(cfg, chip)
    hsaco_file = cache_dir / f"{name}.hsaco"
    meta_file = cache_dir / f"{name}.json"

    if hsaco_file.exists():
        print(f"  [cached] {name}")
        return True

    # Step 1: Compile kernel in subprocess to get MLIR dump
    compile_script = f'''
import os, sys, shutil
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_EXTERNAL_GPU_COMPILE"] = "0"
dsl2 = "{dsl2_root}"
sys.path.insert(0, os.path.join(dsl2, "python"))
sys.path.insert(0, os.path.join(dsl2, "kernels"))
sys.path.insert(0, dsl2)

from moe_gemm_2stage import compile_moe_gemm2
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.jit_function import MlirCompiler
from flydsl.compiler.jit_argument import Stream, Tensor, Int32
from flydsl.compiler.protocol import fly_types, fly_construct
from flydsl.compiler.kernel_function import CompilationContext, create_gpu_module, get_gpu_module_body
from flydsl._mlir.dialects import func
import inspect

jit_func = compile_moe_gemm2(
    model_dim={cfg["model_dim"]}, inter_dim={cfg["inter_dim"]},
    experts={cfg["experts"]}, topk={cfg["topk"]},
    tile_m={cfg["tile_m"]}, tile_n={cfg["tile_n"]}, tile_k={cfg["tile_k"]},
    doweight_stage2={cfg["doweight"]}, in_dtype="{cfg["a_dtype"]}",
    out_dtype="{cfg["out_dtype"]}", accumulate={cfg["accumulate"]},
)

sig = inspect.signature(jit_func.func)
dummy_jit_args = []
for name, param in sig.parameters.items():
    ann = param.annotation
    if ann is fx.Tensor:
        dummy_jit_args.append(Tensor(None))
    elif ann is fx.Int32:
        dummy_jit_args.append(Int32(1))
    elif ann is fx.Stream:
        dummy_jit_args.append(Stream(None))
    else:
        dummy_jit_args.append(Tensor(None))

ir_types = fly_types(dummy_jit_args)

with ir.Context() as ctx:
    ctx.load_all_available_dialects()
    loc = ir.Location.unknown(ctx)
    module = ir.Module.create(loc=loc)
    module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

    with ir.InsertionPoint(module.body), loc:
        gpu_module = create_gpu_module("kernels", targets=['#rocdl.target<chip = "{chip}">'])
        func_op = func.FuncOp(jit_func.func.__name__, (ir_types, []))
        func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        entry_block = func_op.add_entry_block()

        with CompilationContext.create() as comp_ctx:
            comp_ctx.gpu_module_op = gpu_module
            comp_ctx.gpu_module_body = get_gpu_module_body(gpu_module)

            with ir.InsertionPoint(entry_block):
                ir_args = list(func_op.regions[0].blocks[0].arguments)
                comp_ctx.stream_arg = ir_args[-1]
                dsl_types = [type(a) for a in dummy_jit_args]
                dsl_args = fly_construct(dsl_types, dummy_jit_args, ir_args)
                param_names = list(sig.parameters.keys())
                named_args = dict(zip(param_names, dsl_args))
                jit_func.func(**named_args)
                func.ReturnOp([])

    compiled = MlirCompiler.compile(module, chip="{chip}", func_name=jit_func.func.__name__)
print("COMPILE_OK")
'''
    print(f"  Compiling {name}...", end=" ", flush=True)
    env = os.environ.copy()
    env["DSL2_ROOT"] = dsl2_root
    result = subprocess.run(
        [sys.executable, "-c", compile_script],
        env=env, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0 or "COMPILE_OK" not in result.stdout:
        print(f"FAILED")
        if result.stderr:
            print(f"    stderr: {result.stderr[-300:]}")
        return False

    # Step 2: Find the reconcile_unrealized_casts MLIR dump
    dump_candidates = sorted(DUMP_DIR.rglob("*reconcile_unrealized_casts.mlir"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
    if not dump_candidates:
        print("FAILED (no MLIR dump found)")
        return False
    pre_binary_mlir = dump_candidates[0]

    # Step 3: Run mlir-opt to produce GPU binary
    result = subprocess.run(
        [mlir_opt_path, "--gpu-module-to-binary", str(pre_binary_mlir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"FAILED (mlir-opt)")
        return False

    # Step 4: Extract ELF binary
    binary = extract_gpu_binary(result.stdout)
    if binary is None or len(binary) < 100:
        print(f"FAILED (no ELF in output)")
        return False

    # Save .hsaco and metadata
    hsaco_file.write_bytes(binary)
    meta = {
        "kernel_name": "moe_gemm2_0",
        **cfg,
        "chip": chip,
        "block": [256, 1, 1],
        "shared_mem": 8192,
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"OK ({len(binary)} bytes)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chip", type=str, default="gfx942")
    parser.add_argument("--configs", type=str, default=None,
                        help="JSON file with kernel configs (optional)")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    dsl2_root = os.environ.get("DSL2_ROOT", "")
    if not dsl2_root:
        print("ERROR: DSL2_ROOT not set")
        sys.exit(1)

    mlir_path = os.environ.get("MLIR_PATH", "")
    mlir_opt = os.path.join(mlir_path, "bin", "mlir-opt") if mlir_path else "mlir-opt"
    if not os.path.isfile(mlir_opt):
        print(f"ERROR: mlir-opt not found at {mlir_opt}")
        sys.exit(1)

    configs = STAGE2_CONFIGS
    if args.configs:
        with open(args.configs) as f:
            configs = json.load(f)

    print(f"Precompiling {len(configs)} FlyDSL MOE stage2 kernels for {args.chip}")
    print(f"Cache: {args.cache_dir}")
    print()

    ok = 0
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}]", end=" ")
        if compile_one(cfg, args.chip, args.cache_dir, mlir_opt, dsl2_root):
            ok += 1

    print(f"\nDone: {ok}/{len(configs)} kernels precompiled")


if __name__ == "__main__":
    main()
