#!/usr/bin/env python3

"""Benchmark FlyDSL gfx1100 Flash Attention against Triton and/or a baseline tree.

Key properties required by ROCm/FlyDSL#1216:

* configurable head counts, head dims and sequence lengths
* dtype sweep over bf16 and f16
* causal vs. non-causal forward paths
* decode benchmarks via ``--decode-seq-q``
* optional correctness check (max ULP against an fp64 reference)
* JSON output for automated ingestion ``(--json)``
* ``--compare-src`` measures two FlyDSL trees **inside the same process**;
  every round times the current tree, the other tree and Triton back to back,
  and each tree runs with its own ``kernels.*`` modules installed
* the benchmark keeps cycling distinct Q/K/V replicas so that every round
  streams more than 2 × 96 MiB through Infinity Cache, matching the
  methodology used in the PR review

Triton baseline: ``flash_attn.flash_attn_func`` from flash-attn 2.8.3 on its
Triton AMD backend (export ``FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE``).
Numbers in the PR were taken with Triton 3.7.1 (ROCm 7.14 nightly) and
torch 2.13.0a0 on a W7900D.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - optional runtime dependency
    from flash_attn.flash_attn_interface import flash_attn_func as triton_flash_attn_func
except Exception:  # pragma: no cover - handled when --triton is requested
    triton_flash_attn_func = None


CACHE_CYCLE_BYTES = 192 * 1024 * 1024  # 2 × 96 MiB Infinity Cache target
DEFAULT_ITERATIONS = 64
DEFAULT_WARMUP = 4


# ──────────────────────────────────────────────────────────────────────────────
# Helper dataclasses


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    num_heads: int
    head_dim: int
    seq_q: int
    seq_kv: int
    causal: bool
    dtype: str

    def as_tuple(self) -> Tuple[int, int, int, int, int, bool, str]:
        return (
            self.batch,
            self.num_heads,
            self.head_dim,
            self.seq_q,
            self.seq_kv,
            self.causal,
            self.dtype,
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "bf16":
            return torch.bfloat16
        if self.dtype == "f16":
            return torch.float16
        raise ValueError(f"unsupported dtype {self.dtype}")

    def flop_count(self) -> float:
        return 2.0 * self.batch * self.num_heads * self.seq_q * self.seq_kv * self.head_dim

    def per_case_bytes(self) -> int:
        elem_size = torch.tensor([], dtype=self.torch_dtype).element_size()
        q_bytes = self.batch * self.seq_q * self.num_heads * self.head_dim * elem_size
        kv_bytes = self.batch * self.seq_kv * self.num_heads * self.head_dim * elem_size
        return q_bytes + 2 * kv_bytes


# ──────────────────────────────────────────────────────────────────────────────
# FlyDSL loader utilities


TREE_PREFIXES = ("kernels",)
# flash_attn_interface imports the gfx1100 kernel inside the builder, at call
# time, so each tree keeps its own copy of these installed while it runs.
TREE_MODULES = (
    "kernels.attention.flash_attn_interface",
    "kernels.attention.flash_attn_gfx1100",
    "kernels.attention.flash_attn_gfx1100_autotune",
)
_active_tree: List[Optional[str]] = [None]


class FlyDSLImplementation:
    """Wrapper around a FlyDSL flash attention implementation."""

    def __init__(self, label: str, modules: Dict[str, ModuleType]):
        module = modules["kernels.attention.flash_attn_interface"]
        if not hasattr(module, "flydsl_flash_attn_func"):
            raise AttributeError(f"module {module.__name__} lacks flydsl_flash_attn_func")
        self.label = label
        self.module = module
        self.modules = modules
        self.kernel_src = modules["kernels.attention.flash_attn_gfx1100"].__file__
        self.callable = module.flydsl_flash_attn_func

    def activate(self) -> None:
        if _active_tree[0] == self.label:
            return
        for name in list(sys.modules):
            if name.startswith(TREE_PREFIXES):
                sys.modules.pop(name, None)
        sys.modules.update(self.modules)
        _active_tree[0] = self.label

    def launch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, causal: bool) -> None:
        self.activate()
        self.callable(q, k, v, out=out, causal=causal)


def current_implementation(label: str = "current") -> FlyDSLImplementation:
    """Wrap the worktree's own implementation (the one imported at startup)."""
    for name in TREE_MODULES:
        importlib.import_module(name)
    _active_tree[0] = label
    return FlyDSLImplementation(label, _snapshot_modules(TREE_PREFIXES))


def _snapshot_modules(prefixes: Sequence[str]) -> Dict[str, ModuleType]:
    return {name: module for name, module in sys.modules.items() if any(name.startswith(p) for p in prefixes)}


def _restore_modules(snapshot: Dict[str, ModuleType], prefixes: Sequence[str]) -> None:
    for name in list(sys.modules):
        if any(name.startswith(p) for p in prefixes) and name not in snapshot:
            sys.modules.pop(name, None)
    for name, module in snapshot.items():
        sys.modules[name] = module


def load_flydsl_from_tree(root: Path, label: str) -> FlyDSLImplementation:
    """Load flydsl_flash_attn_func from ``root`` without leaving modules behind."""

    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    snapshot = _snapshot_modules(TREE_PREFIXES)

    for name in list(sys.modules):
        if name.startswith(TREE_PREFIXES):
            sys.modules.pop(name, None)

    sys.path.insert(0, str(root))
    try:
        importlib.invalidate_caches()
        for name in TREE_MODULES:
            importlib.import_module(name)
        impl = FlyDSLImplementation(label, _snapshot_modules(TREE_PREFIXES))
    finally:
        sys.path.pop(0)
        _restore_modules(snapshot, TREE_PREFIXES)

    if not Path(impl.kernel_src).resolve().is_relative_to(root):
        raise RuntimeError(f"{label}: gfx1100 kernel resolved to {impl.kernel_src}, outside {root}")
    return impl


# ──────────────────────────────────────────────────────────────────────────────
# Buffer management


class BufferPool:
    """Pre-generates Q/K/V replicas so each round streams > 192 MiB."""

    def __init__(self, cfg: BenchConfig, device: torch.device, rng: torch.Generator):
        self.cfg = cfg
        self.device = device
        self.dtype = cfg.torch_dtype
        self.elem_size = torch.tensor([], dtype=self.dtype).element_size()
        per_case = cfg.per_case_bytes()
        replicas = max(3, math.ceil(CACHE_CYCLE_BYTES / per_case))
        self.replicas = replicas
        self.q_list: List[torch.Tensor] = []
        self.k_list: List[torch.Tensor] = []
        self.v_list: List[torch.Tensor] = []
        for _ in range(replicas):
            self.q_list.append(
                torch.randn(
                    cfg.batch,
                    cfg.seq_q,
                    cfg.num_heads,
                    cfg.head_dim,
                    dtype=self.dtype,
                    device=device,
                    generator=rng,
                )
            )
            self.k_list.append(
                torch.randn(
                    cfg.batch,
                    cfg.seq_kv,
                    cfg.num_heads,
                    cfg.head_dim,
                    dtype=self.dtype,
                    device=device,
                    generator=rng,
                )
            )
            self.v_list.append(
                torch.randn(
                    cfg.batch,
                    cfg.seq_kv,
                    cfg.num_heads,
                    cfg.head_dim,
                    dtype=self.dtype,
                    device=device,
                    generator=rng,
                )
            )
        self._outs: Dict[str, List[torch.Tensor]] = {}

    def outputs_for(self, label: str) -> List[torch.Tensor]:
        outs = self._outs.get(label)
        if outs is None:
            template = self.q_list[0]
            outs = [torch.empty_like(template) for _ in range(self.replicas)]
            self._outs[label] = outs
        return outs

    def iterator(self, start: int = 0) -> Iterator[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        idx = start % self.replicas
        while True:
            yield idx, self.q_list[idx], self.k_list[idx], self.v_list[idx]
            idx = (idx + 1) % self.replicas


# ──────────────────────────────────────────────────────────────────────────────
# Numeric helpers


def _bottom_right_causal_mask(seq_q: int, seq_kv: int, device: torch.device) -> torch.Tensor:
    offset = seq_kv - seq_q
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_kv, device=device).unsqueeze(0)
    allowed = offset + q_idx
    return k_idx <= allowed


def _softmax(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    scores = scores - scores.max(dim=-1, keepdim=True).values
    return torch.softmax(scores, dim=-1)


def _reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    q_64 = q.to(torch.float64)
    k_64 = k.to(torch.float64)
    v_64 = v.to(torch.float64)
    batch, seq_q, num_heads, head_dim = q.shape
    seq_kv = k.shape[1]
    q_flat = q_64.permute(0, 2, 1, 3).reshape(batch * num_heads, seq_q, head_dim)
    k_flat = k_64.permute(0, 2, 1, 3).reshape(batch * num_heads, seq_kv, head_dim)
    v_flat = v_64.permute(0, 2, 1, 3).reshape(batch * num_heads, seq_kv, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    mask = _bottom_right_causal_mask(seq_q, seq_kv, q.device).unsqueeze(0) if causal else None
    scores = torch.matmul(q_flat, k_flat.transpose(1, 2)) * scale
    probs = _softmax(scores, mask)
    out = torch.matmul(probs, v_flat)
    return out.view(batch, num_heads, seq_q, head_dim).permute(0, 2, 1, 3).contiguous()


def _max_ulp(tensor: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> float:
    if tensor.shape != ref.shape:
        raise ValueError(f"shape mismatch: {tensor.shape} vs {ref.shape}")
    err = (tensor - ref).abs().to(torch.float64)
    ref_abs = ref.abs().to(torch.float64)
    rows = err.reshape(-1, err.shape[-1])
    ref_rows = ref_abs.reshape_as(rows)
    row_max = rows.max(dim=-1).values
    ref_row_max = ref_rows.max(dim=-1).values
    dtype_info = torch.finfo(dtype)
    safe_ref = torch.where(ref_row_max == 0, torch.full_like(ref_row_max, dtype_info.tiny), ref_row_max)
    ref_cast = safe_ref.to(dtype)
    next_up = torch.nextafter(ref_cast, torch.full_like(ref_cast, float("inf")))
    ulp = (next_up - ref_cast).to(torch.float64)
    ulp = torch.where(ref_row_max == 0, torch.full_like(ulp, dtype_info.tiny), ulp)
    ratios = row_max / ulp
    return float(ratios.max().item())


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark engine


def _time_callable(fn, iterations: int, warmup: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


def _flydsl_round(
    impl: FlyDSLImplementation,
    cfg: BenchConfig,
    buffers: BufferPool,
    round_idx: int,
    iterations: int,
    warmup: int,
) -> float:
    outputs = buffers.outputs_for(impl.label)
    iterator = buffers.iterator(start=round_idx)

    def _call() -> None:
        idx, q, k, v = next(iterator)
        impl.launch(q, k, v, outputs[idx], cfg.causal)

    return _time_callable(_call, iterations, warmup)


def _triton_round(cfg: BenchConfig, buffers: BufferPool, round_idx: int, iterations: int, warmup: int) -> float:
    if triton_flash_attn_func is None:
        raise RuntimeError("flash-attn package not available; install flash-attn>=2.5")
    outputs = buffers.outputs_for("triton")
    iterator = buffers.iterator(start=round_idx)
    scale = 1.0 / math.sqrt(cfg.head_dim)

    def _call() -> None:
        idx, q, k, v = next(iterator)
        outputs[idx].copy_(triton_flash_attn_func(q, k, v, 0.0, scale, cfg.causal))

    return _time_callable(_call, iterations, warmup)


def _summarize(cfg: BenchConfig, ms_rounds: List[float]) -> Tuple[float, List[float], List[float]]:
    tflops = [cfg.flop_count() / (ms * 1e-3) / 1e12 for ms in ms_rounds]
    return statistics.median(tflops), tflops, ms_rounds


def run_benchmark(
    cfg: BenchConfig,
    rounds: int,
    iterations: int,
    warmup: int,
    check: bool,
    current_impl: FlyDSLImplementation,
    compare_impl: Optional[FlyDSLImplementation],
    device: torch.device,
    rng: torch.Generator,
) -> Dict[str, object]:
    buffers = BufferPool(cfg, device, rng)
    result: Dict[str, object] = {
        "batch": cfg.batch,
        "heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "seq_q": cfg.seq_q,
        "seq_kv": cfg.seq_kv,
        "causal": cfg.causal,
        "dtype": cfg.dtype,
    }

    # One round of every implementation before the next round, so clock or
    # thermal drift hits all of them alike.
    cur_ms: List[float] = []
    cmp_ms: List[float] = []
    tri_ms: List[float] = []
    triton_error = None if triton_flash_attn_func is not None else "flash-attn not installed"
    for round_idx in range(rounds):
        cur_ms.append(_flydsl_round(current_impl, cfg, buffers, round_idx, iterations, warmup))
        if compare_impl is not None:
            cmp_ms.append(_flydsl_round(compare_impl, cfg, buffers, round_idx, iterations, warmup))
        if triton_error is None:
            try:
                tri_ms.append(_triton_round(cfg, buffers, round_idx, iterations, warmup))
            except RuntimeError as exc:  # pragma: no cover - optional dependency
                triton_error = str(exc)

    cur_med, result["flydsl_tflops_rounds"], result["flydsl_ms_rounds"] = _summarize(cfg, cur_ms)
    result["flydsl_tflops"] = cur_med
    result["flydsl_kernel_src"] = current_impl.kernel_src

    if compare_impl is not None:
        cmp_med, result["baseline_tflops_rounds"], result["baseline_ms_rounds"] = _summarize(cfg, cmp_ms)
        result["baseline_tflops"] = cmp_med
        result["baseline_kernel_src"] = compare_impl.kernel_src
        result["delta_percent_vs_baseline"] = ((cur_med - cmp_med) / cmp_med * 100.0) if cmp_med else float("nan")

    if triton_error is None:
        tri_med, result["triton_tflops_rounds"], result["triton_ms_rounds"] = _summarize(cfg, tri_ms)
        result["triton_tflops"] = tri_med
        result["ratio_vs_triton"] = (cur_med / tri_med) if tri_med else float("nan")
    else:
        result["triton_error"] = triton_error

    if check:
        q0 = buffers.q_list[0].to(torch.float64)
        k0 = buffers.k_list[0].to(torch.float64)
        v0 = buffers.v_list[0].to(torch.float64)
        ref = _reference_attention(q0, k0, v0, cfg.causal)
        out = buffers.outputs_for(current_impl.label)[0].to(torch.float64)
        result["max_ulp"] = _max_ulp(out, ref, cfg.torch_dtype)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing / CLI


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_dtype_list(value: str) -> List[str]:
    if not value:
        return []
    dtypes = []
    for item in value.split(","):
        name = item.strip().lower()
        if name not in {"bf16", "f16"}:
            raise argparse.ArgumentTypeError(f"unsupported dtype '{item}' (choose bf16,f16)")
        dtypes.append(name)
    return dtypes


def _unique_configs(configs: Iterable[BenchConfig]) -> List[BenchConfig]:
    dedup: Dict[Tuple[int, int, int, int, int, bool, str], BenchConfig] = {}
    for cfg in configs:
        dedup[cfg.as_tuple()] = cfg
    return list(dedup.values())


def build_configs(args) -> List[BenchConfig]:
    configs: List[BenchConfig] = []
    dtypes = args.dtype or ["bf16", "f16"]
    causal_flags = [bool(c) for c in (args.causal or [0, 1])]

    for dtype in dtypes:
        for head_dim in args.dims:
            for heads in args.heads:
                for causal in causal_flags:
                    for seq in args.seqs:
                        configs.append(
                            BenchConfig(
                                batch=args.batch,
                                num_heads=heads,
                                head_dim=head_dim,
                                seq_q=seq,
                                seq_kv=seq,
                                causal=causal,
                                dtype=dtype,
                            )
                        )
                        for decode_q in args.decode_seq_q:
                            configs.append(
                                BenchConfig(
                                    batch=args.batch,
                                    num_heads=heads,
                                    head_dim=head_dim,
                                    seq_q=decode_q,
                                    seq_kv=seq,
                                    causal=causal,
                                    dtype=dtype,
                                )
                            )
    return _unique_configs(configs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark FlyDSL gfx1100 Flash Attention")
    parser.add_argument("--heads", type=_parse_int_list, default="1,4,8,32")
    parser.add_argument("--dims", type=_parse_int_list, default="64,128,256")
    parser.add_argument("--seqs", type=_parse_int_list, default="1024")
    parser.add_argument("--dtype", type=_parse_dtype_list, default="bf16,f16")
    parser.add_argument("--causal", type=_parse_int_list, default="0,1")
    parser.add_argument("--decode-seq-q", type=_parse_int_list, default="1,16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--check", action="store_true", help="validate against an fp64 reference")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--compare-src", type=str, default=None, help="path to another FlyDSL checkout")

    args = parser.parse_args(argv)

    if not torch.cuda.is_available():  # pragma: no cover - GPU only
        raise SystemExit("GPU (ROCm) is required for benchmarking")

    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    current_impl = current_implementation()
    compare_impl: Optional[FlyDSLImplementation] = None
    if args.compare_src:
        compare_impl = load_flydsl_from_tree(Path(args.compare_src), label="baseline")

    configs = build_configs(args)
    device = torch.device("cuda")
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    results: List[Dict[str, object]] = []
    for cfg in configs:
        res = run_benchmark(
            cfg,
            rounds=args.rounds,
            iterations=args.iterations,
            warmup=args.warmup,
            check=args.check,
            current_impl=current_impl,
            compare_impl=compare_impl,
            device=device,
            rng=rng,
        )
        results.append(res)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for entry in results:
        base = (
            f"batch={entry['batch']} heads={entry['heads']} dim={entry['head_dim']} "
            f"seq_q={entry['seq_q']} seq_kv={entry['seq_kv']} causal={entry['causal']} dtype={entry['dtype']}"
        )
        msg = [base, f"FlyDSL={entry['flydsl_tflops']:.4f} TF/s"]
        if "baseline_tflops" in entry:
            msg.append(f"Baseline={entry['baseline_tflops']:.4f} TF/s")
            msg.append(f"Delta={entry['delta_percent_vs_baseline']:.2f}%")
        if "triton_tflops" in entry:
            msg.append(f"Triton={entry['triton_tflops']:.4f} TF/s")
            msg.append(f"FlyDSL/Triton={entry['ratio_vs_triton']:.4f}")
        if entry.get("triton_error"):
            msg.append(f"TritonError={entry['triton_error']}")
        if "max_ulp" in entry:
            msg.append(f"max_ulp={entry['max_ulp']:.3f}")
        print(" | ".join(msg))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
