#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Compare OPUS GQA result MD5 from the host exe and the Python wrapper."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_EXE = THIS_DIR / "build" / "gqa_attn.exe"
UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123

# Tuples from FlyDSL/tests/kernels/test_flash_opus_attn.py lines 53-83:
# (batch, seq_len, num_heads, num_kv_heads, head_dim)
SELECTED_CONFIGS = [
    (8, 128, 64, 64, 128),
    (8, 256, 64, 64, 128),
    (8, 512, 64, 64, 128),
    (1, 128, 64, 64, 128),
    (1, 256, 64, 64, 128),
    (1, 512, 64, 64, 128),
    (1, 1024, 64, 64, 128),
    (1, 2048, 64, 64, 128),
    (1, 4096, 64, 64, 128),
    (1, 8192, 64, 64, 128),
    (4, 8192, 64, 64, 128),
    (1, 2048, 32, 32, 128),
    (1, 4096, 32, 32, 128),
    (1, 8192, 32, 32, 128),
    (8, 8192, 32, 32, 128),
    (1, 2048, 16, 16, 128),
    (1, 4096, 16, 16, 128),
    (1, 8192, 16, 16, 128),
    (16, 8192, 16, 16, 128),
    (1, 2048, 8, 8, 128),
    (1, 4096, 8, 8, 128),
    (1, 8192, 8, 8, 128),
    (32, 8192, 8, 8, 128),
    (16, 8192, 64, 64, 128),
    (16, 8192, 64, 8, 128),
]

ACTIVE_CONFIGS = [
    (16, 8192, 64, 64, 128),
    # (16, 8192, 64, 8, 128),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="Path to build/gqa_attn.exe")
    parser.add_argument(
        "--configs",
        choices=("selected", "active"),
        default="selected",
        help="'selected' runs every tuple shown in the referenced block; 'active' runs only active DEFAULT_CONFIGS.",
    )
    parser.add_argument("--device", type=str, default=None, help="Set HIP_VISIBLE_DEVICES for both calls")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed seed for Python input generation")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each exe call")
    parser.add_argument(
        "--include-unsupported", action="store_true", help="Run configs even if this kernel rejects them"
    )
    return parser.parse_args()


def tensor_md5(torch, tensor) -> str:
    return hashlib.md5(tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()


def file_md5(path: Path, expected_bytes: int) -> str:
    data = path.read_bytes()
    if len(data) != expected_bytes:
        raise RuntimeError(f"{path} has unexpected size: expected {expected_bytes} bytes, got {len(data)}")
    return hashlib.md5(data).hexdigest()


def setup_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def dump_tensor(torch, tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tofile(path)


def is_supported_config(config: tuple[int, int, int, int, int]) -> tuple[bool, str]:
    _batch, seq_len, _num_heads, _num_kv_heads, head_dim = config
    if head_dim == 128:
        q_tile, kv_tile, num_warps = 32, 64, 8
    elif head_dim == 512:
        q_tile, kv_tile, num_warps = 16, 32, 8
    else:
        return False, "D must be 128 or 512"

    if seq_len % kv_tile != 0 or seq_len // kv_tile < 6:
        return False, f"N must be a multiple of {kv_tile} and span at least 6 KV tiles"
    if seq_len % (q_tile * num_warps) != 0:
        return False, f"N must be a multiple of {q_tile * num_warps}"
    return True, ""


def shell_result_md5(
    exe: Path,
    config: tuple[int, int, int, int, int],
    causal: bool,
    input_dir: Path,
    env: dict[str, str],
    timeout: int,
) -> str:
    batch, seq_len, num_heads, num_kv_heads, head_dim = config
    output_file = input_dir / "cpp_o.bin"
    cmd = [
        str(exe),
        "-b",
        str(batch),
        "-n",
        str(seq_len),
        "-h_q",
        str(num_heads),
        "-h_kv",
        str(num_kv_heads),
        "-d",
        str(head_dim),
        "--causal" if causal else "--no-causal",
        "--input-dir",
        str(input_dir),
        "--output-file",
        str(output_file),
        "--skip-benchmark",
    ]
    proc = subprocess.run(
        cmd,
        cwd=THIS_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"exe failed with rc={proc.returncode}\n{proc.stdout}")

    if not output_file.is_file():
        raise RuntimeError(
            "exe did not write output file. Rebuild gqa_attn.exe after updating gqa_host.cc.\n" + proc.stdout
        )
    return file_md5(output_file, batch * seq_len * num_heads * head_dim * 2)


def python_result_md5(
    torch,
    opus_attn,
    config: tuple[int, int, int, int, int],
    causal: bool,
    input_dir: Path,
    seed: int,
) -> str:
    batch, seq_len, num_heads, num_kv_heads, head_dim = config
    device = "cuda"

    setup_seed(torch, seed)
    torch.cuda.empty_cache()

    q = torch.empty(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
    k = torch.empty(batch, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device).uniform_(
        *UNIFORM_RANGE
    )
    v = torch.empty(batch, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device).uniform_(
        *UNIFORM_RANGE
    )
    out = torch.zeros_like(q)

    dump_tensor(torch, q, input_dir / "q.bin")
    dump_tensor(torch, k, input_dir / "k.bin")
    dump_tensor(torch, v, input_dir / "v.bin")

    opus_attn.forward_out(q, k, v, out, causal=causal)
    torch.cuda.synchronize()
    md5 = tensor_md5(torch, out)

    del q, k, v, out
    torch.cuda.empty_cache()
    return md5


def main() -> int:
    args = parse_args()
    if args.device is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = args.device

    if not args.exe.is_file():
        print(f"ERROR: {args.exe} not found. Build it with: cd {THIS_DIR} && make -j", file=sys.stderr)
        return 1

    import torch
    import opus_attn

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm device is not available", file=sys.stderr)
        return 1

    configs = SELECTED_CONFIGS if args.configs == "selected" else ACTIVE_CONFIGS
    env = os.environ.copy()
    any_failed = False

    print("dtype=bf16")
    print("status mode      B      N   H  H_KV   D  shell_md5                         python_md5")
    print("-" * 104)

    for config in configs:
        supported, reason = is_supported_config(config)
        if not supported and not args.include_unsupported:
            batch, seq_len, num_heads, num_kv_heads, head_dim = config
            for mode_name in ("causal", "nocausal"):
                print(
                    f"SKIP   {mode_name:<8} {batch:>2} {seq_len:>6} {num_heads:>3} {num_kv_heads:>5} "
                    f"{head_dim:>3}  {reason}"
                )
            continue

        for mode_name, causal in (("causal", True), ("nocausal", False)):
            batch, seq_len, num_heads, num_kv_heads, head_dim = config
            try:
                with tempfile.TemporaryDirectory(prefix="opus_gqa_inputs_") as tmp:
                    input_dir = Path(tmp)
                    py_md5 = python_result_md5(torch, opus_attn, config, causal, input_dir, args.seed)
                    shell_md5 = shell_result_md5(args.exe, config, causal, input_dir, env, args.timeout)
                status = "PASS" if shell_md5 == py_md5 else "FAIL"
                any_failed |= status == "FAIL"
                print(
                    f"{status:<6} {mode_name:<8} {batch:>2} {seq_len:>6} {num_heads:>3} {num_kv_heads:>5} "
                    f"{head_dim:>3}  {shell_md5}  {py_md5}"
                )
            except Exception as exc:
                any_failed = True
                print(
                    f"ERROR  {mode_name:<8} {batch:>2} {seq_len:>6} {num_heads:>3} {num_kv_heads:>5} "
                    f"{head_dim:>3}  {exc}"
                )

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
