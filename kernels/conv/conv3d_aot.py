#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""AOT for the FlyDSL implicit-GEMM convolution (lightweight native port).

Each tuned CSV row becomes two conv jobs (``out_ndhwc`` True/False) plus the
NCDHW->NDHWC pre-transpose. Same parse/dedupe/dyn_hw rules as aiter
``aiter/aot/flydsl/conv.py``.

    python -m kernels.conv.conv3d_aot
    python -m kernels.conv.conv3d_aot --csv kernels/conv/configs/wan21_vae_bf16_tuned_conv3d.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

from .conv3d_gfx950_utils import (
    LDG_VEC,
    make_conv_geometry,
    make_launch_grid,
    make_output_scatter_plan,
    make_tile_config,
    out_extent,
    unit_divisors,
)
from .conv3d_im2col import make_im2col_plan
from .conv3d_implicit import (
    FLYDSL_CONV3D_DYN_HW,
    LIBTYPE_FLYDSL,
    TUNED_KEY_COLUMNS,
    TUNED_LIBTYPE_COLUMN,
    TUNED_RESULT_COLUMNS,
    _dispatch,
    _dyn_hw_ok,
    _implicit_param_from_problem,
    _is_matmul_fast_path,
    _pad_channels,
    _parse_tuned_bool,
    _resolve_splitk,
    _resolve_tuned_csv_paths,
)
from .conv3d_implicit_gfx950 import (
    _dyn_hw_closure_key,
    compile_conv3d_implicit,
)
from .conv3d_transpose import (
    TR_MAX_BIG_S,
    TR_VEC,
    compile_transpose_ncdhw_ndhwc,
)

assert LDG_VEC % TR_VEC == 0, (
    f"channel padding rounds to a multiple of LDG_VEC={LDG_VEC}, which no longer "
    f"guarantees the transpose's c % {TR_VEC} == 0; parse_csv has to test it again"
)

CONV_AOT_ARCH_DEFAULT = "gfx950"
_PROBE_EXTENT = 8
_INT_COLS = tuple(c for c in TUNED_KEY_COLUMNS if c != "bias")
_CONFIG_COLS = TUNED_RESULT_COLUMNS
_RESOLUTION_COLS = ("N", "D", "H", "W")

_CU_NUM_TO_ARCH = {
    80: "gfx942",
    304: "gfx942",
    256: "gfx950",
}


def cu_num_to_arch(cu_num: int, default: str = CONV_AOT_ARCH_DEFAULT) -> str:
    return _CU_NUM_TO_ARCH.get(cu_num, default)


def job_identity(job: dict) -> tuple:
    return tuple(sorted(job.items()))


def job_arch(cu_num: int = 0, gfx: str = "") -> str:
    return gfx or cu_num_to_arch(cu_num, default=CONV_AOT_ARCH_DEFAULT)


def default_csv_paths():
    paths = _resolve_tuned_csv_paths()
    return [str(p) for p in paths if p.is_file()]


def _requested_archs():
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS")
    if not arch:
        return None
    return {a.strip() for a in re.split(r"[;,]", arch) if a.strip()} or None


def _row_npq_per_sample(shape) -> int:
    return (
        out_extent(shape["D"], shape["pad_d"], shape["dil_d"], shape["kT"], shape["stride_d"])
        * out_extent(shape["H"], shape["pad_h"], shape["dil_h"], shape["kH"], shape["stride_h"])
        * out_extent(shape["W"], shape["pad_w"], shape["dil_w"], shape["kW"], shape["stride_w"])
    )


def _conv_dedupe_key(job):
    if not job["dyn_hw"]:
        return job_identity(job)
    tile = (job["tile_m"], job["tile_n"], job["wave_m"], job["wave_n"])
    try:
        param = _implicit_param_from_problem(
            job["N"],
            job["C"],
            job["D"],
            job["H"],
            job["W"],
            job["K"],
            job["kT"],
            job["kH"],
            job["kW"],
            job["stride_d"],
            job["stride_h"],
            job["stride_w"],
            job["pad_d"],
            job["pad_h"],
            job["pad_w"],
            job["dil_d"],
            job["dil_h"],
            job["dil_w"],
            job["groups"],
            job["has_bias"],
            job["splitk"],
            tile,
            job["wgm"],
            job["out_ndhwc"],
            "zeros",
            True,
        )
        cfg = make_tile_config(param.tile)
        geom = make_conv_geometry(param)
        grid = make_launch_grid(param, geom, cfg)
        closure = _dyn_hw_closure_key(
            grid._replace(grid_x=0, grid_z=0, grid_m=0),
            make_im2col_plan(param, geom, cfg),
            make_output_scatter_plan(param, geom, cfg, grid),
            unit_divisors(param, geom),
        )
    except (AssertionError, ValueError):
        return job_identity(job)
    return (closure,) + tuple(sorted((k, v) for k, v in job.items() if k not in _RESOLUTION_COLS))


def _resolve_dyn_hw(*, shape, splitk, tile, out_ndhwc, has_bias):
    probe = _implicit_param_from_problem(
        shape["N"],
        shape["C"],
        shape["D"],
        shape["H"],
        shape["W"],
        shape["K"],
        shape["kT"],
        shape["kH"],
        shape["kW"],
        shape["stride_d"],
        shape["stride_h"],
        shape["stride_w"],
        shape["pad_d"],
        shape["pad_h"],
        shape["pad_w"],
        shape["dil_d"],
        shape["dil_h"],
        shape["dil_w"],
        shape["groups"],
        has_bias,
        splitk,
        tile,
        1,
        out_ndhwc,
        "zeros",
        False,
    )
    geom = make_conv_geometry(probe)
    return _dyn_hw_ok(probe.n, probe.c, probe.d, probe.h, probe.w, probe.k, geom.npq, tile)


def parse_csv(csv_path: str):
    """Parse the tuned conv CSV into unique conv and transpose compile jobs."""
    jobs = []
    seen = set()
    keep_archs = _requested_archs()

    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            libtype = row.get(TUNED_LIBTYPE_COLUMN, "")
            if libtype and libtype != LIBTYPE_FLYDSL:
                continue
            missing = [c for c in (*_INT_COLS, *_CONFIG_COLS) if c not in row]
            if missing:
                print(f"  [WARN] {csv_path}: missing columns {missing}, skipping row")
                continue
            try:
                shape = {c: int(row[c]) for c in _INT_COLS}
                config = {c: int(row[c]) for c in _CONFIG_COLS}
                has_bias = _parse_tuned_bool(row.get("bias"))
                raw_splitk = row.get("splitK", "")
                splitk = (int(raw_splitk) or 1) if raw_splitk else None
            except ValueError as exc:
                print(f"  [WARN] {csv_path}: unparsable row ({exc}), skipping")
                continue

            if _is_matmul_fast_path(shape):
                continue

            cu_num = int(row.get("cu_num") or 0)
            gfx = row.get("gfx", "")
            if keep_archs is not None and job_arch(cu_num, gfx) not in keep_archs:
                continue

            groups = shape["groups"]
            cgp = _pad_channels(shape["C"] // groups)
            c_padded = groups * cgp

            tile = (
                config["tile_m"],
                config["tile_n"],
                config["wave_m"],
                config["wave_n"],
            )

            crs = cgp * shape["kT"] * shape["kH"] * shape["kW"]
            npq = shape["N"] * _row_npq_per_sample(shape)
            splitk = _resolve_splitk(
                splitk,
                npq,
                crs,
                shape["K"],
                None,
                tile,
                groups,
                num_cu=cu_num or None,
            )
            for out_ndhwc in (False, True):
                conv_job = {
                    "kind": "conv3d",
                    "kernel_name": "conv3d_implicit_kernel",
                    "cu_num": cu_num,
                    "gfx": gfx,
                    "has_bias": has_bias,
                    "splitk": splitk,
                    "out_ndhwc": out_ndhwc,
                    "dyn_hw": _resolve_dyn_hw(
                        shape=shape,
                        splitk=splitk,
                        tile=tile,
                        out_ndhwc=out_ndhwc,
                        has_bias=has_bias,
                    ),
                    **shape,
                    **config,
                }
                key = _conv_dedupe_key(conv_job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(conv_job)

            s = shape["D"] * shape["H"] * shape["W"]
            big = shape["N"] * c_padded * s > 0x7FFFFFFF
            if not (big and s > TR_MAX_BIG_S):
                tr_job = {
                    "kind": "transpose",
                    "kernel_name": "transpose_ncdhw_ndhwc",
                    "cu_num": cu_num,
                    "gfx": gfx,
                    "N": shape["N"],
                    "c_padded": c_padded,
                    "s": s,
                }
                key = ("transpose", gfx, cu_num, shape["N"], c_padded, big)
                if key not in seen:
                    seen.add(key)
                    jobs.append(tr_job)

    return jobs


def collect_aot_jobs(csv_paths):
    jobs = []
    seen = set()
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"  [WARN] CSV not found: {csv_path}")
            continue
        for job in parse_csv(csv_path):
            key = job_identity(job)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)
    return jobs


def _probe(rank: int, dtype_is_fp32: bool = False):
    import torch

    return torch.empty(
        (1,) * (rank - 1) + (_PROBE_EXTENT,),
        device=torch.device("cpu"),
        dtype=torch.float32 if dtype_is_fp32 else torch.bfloat16,
    )


def _conv_probe_args(splitk: int):
    y = _probe(2, dtype_is_fp32=True) if splitk > 1 else _probe(5)
    return y, _probe(5), _probe(2), _probe(1, dtype_is_fp32=True)


def _compile_conv3d_to_cache(
    *,
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    K: int,
    kT: int,
    kH: int,
    kW: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dil_d: int,
    dil_h: int,
    dil_w: int,
    groups: int,
    has_bias: bool,
    splitk: int,
    tile_m: int,
    tile_n: int,
    wave_m: int,
    wave_n: int,
    wgm: int,
    out_ndhwc: bool = False,
    dyn_hw: bool = False,
):
    # No **kwargs: a job field this does not name is a compile-time parameter
    # being dropped, which would cache an artifact under the default and leave
    # the runtime JITing the one it asked for. Let it raise TypeError instead.
    exe = compile_conv3d_implicit(
        _implicit_param_from_problem(
            N,
            C,
            D,
            H,
            W,
            K,
            kT,
            kH,
            kW,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            dil_d,
            dil_h,
            dil_w,
            groups,
            has_bias,
            splitk,
            (tile_m, tile_n, wave_m, wave_n),
            wgm,
            out_ndhwc,
            "zeros",
            dyn_hw,
        )
    )
    prev = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        _dispatch(exe, *_conv_probe_args(splitk), stream=None)
    finally:
        if prev is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev


def _compile_transpose_to_cache(*, N: int, c_padded: int, s: int):
    exe = compile_transpose_ncdhw_ndhwc(N, c_padded, s)
    prev = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        _dispatch(exe, _probe(5), _probe(5), stream=None)
    finally:
        if prev is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev


def compile_one_config(
    kind: str,
    kernel_name: str,
    cu_num: int = 0,
    gfx: str = "",
    **kwargs,
) -> dict:
    aot_arch = job_arch(cu_num, gfx)
    if kind == "transpose":
        shape_str = f"{kernel_name}  N={kwargs['N']} C={kwargs['c_padded']} S={kwargs['s']}"
    else:
        shape_str = (
            f"{kernel_name}  {kwargs['N']}x{kwargs['C']}x{kwargs['D']}x"
            f"{kwargs['H']}x{kwargs['W']}->{kwargs['K']} "
            f"k{kwargs['kT']}{kwargs['kH']}{kwargs['kW']} "
            f"tile={kwargs['tile_m']}x{kwargs['tile_n']} "
            f"out={'NDHWC' if kwargs.get('out_ndhwc') else 'NCDHW'}"
        )
    result = {
        "kernel_name": kernel_name,
        "kind": kind,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    t0 = time.time()
    prev_arch = os.environ.get("FLYDSL_GPU_ARCH")
    os.environ["FLYDSL_GPU_ARCH"] = aot_arch
    try:
        if kind == "conv3d":
            _compile_conv3d_to_cache(**kwargs)
        elif kind == "transpose":
            _compile_transpose_to_cache(**kwargs)
        else:
            raise ValueError(f"Unknown conv AOT kind: {kind}")
        result["compile_time"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")
    finally:
        if prev_arch is None:
            os.environ.pop("FLYDSL_GPU_ARCH", None)
        else:
            os.environ["FLYDSL_GPU_ARCH"] = prev_arch
    return result


def _max_workers(n_jobs: int) -> int:
    env = os.environ.get("FLYDSL_CONV3D_AOT_WORKERS")
    if env:
        return min(max(int(env), 1), n_jobs)
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    workers = min(cpus, 64, n_jobs)
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / (1024**3)
        workers = min(workers, max(1, int(avail_gb / 2.0)))
    except Exception:  # noqa: BLE001
        pass
    return max(1, workers)


def _aot_worker(idx_job):
    """Top-level so ProcessPoolExecutor can pickle it under spawn/fork."""
    idx, job = idx_job
    return idx, compile_one_config(**job)


def run_jobs_parallel(jobs):
    if not jobs:
        return []
    max_workers = _max_workers(len(jobs))
    print(f"[flydsl] conv3d AOT: {len(jobs)} kernels, {max_workers} workers", flush=True)
    ctx = multiprocessing.get_context("fork")
    results = [None] * len(jobs)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_aot_worker, (i, job)) for i, job in enumerate(jobs)]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)} kernels done", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL conv3d kernels from tuned CSV config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to tuned CSV config file(s); defaults to shipped configs/",
    )
    args = parser.parse_args(argv)

    csv_paths = [os.path.abspath(p) for p in (args.csv or default_csv_paths())]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

    cache_dir = os.path.expanduser(os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache"))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS")

    all_jobs = collect_aot_jobs(csv_paths)
    conv_jobs = [j for j in all_jobs if j["kind"] == "conv3d"]
    tr_jobs = [j for j in all_jobs if j["kind"] == "transpose"]

    print("=" * 72)
    print("FlyDSL conv3d AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:              {csv_path}")
    n_dyn = sum(1 for j in conv_jobs if j.get("dyn_hw"))
    print(f"  conv3d jobs:      {len(conv_jobs)}  ({n_dyn} variable-resolution)")
    print(f"  transpose jobs:   {len(tr_jobs)}")
    print(f"  Total jobs:       {len(all_jobs)}")
    print(f"  Cache dir:        {cache_dir}")
    print(f"  Target arch:      {arch or '(all archs found in CSVs)'}")
    print(f"  FLYDSL_CONV3D_DYN_HW={FLYDSL_CONV3D_DYN_HW}")
    print("=" * 72)

    total_t0 = time.time()
    results = run_jobs_parallel(all_jobs)
    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r and r["compile_time"] is not None)
    fail = sum(1 for r in results if not r or r["compile_time"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    print(f"  Cache dir:    {cache_dir}")
    print()

    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        sys.exit(1)
    print("All compilations succeeded. Cache is ready.")


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    main()
