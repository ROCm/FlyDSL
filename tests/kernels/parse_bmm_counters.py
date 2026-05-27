#!/usr/bin/env python3
"""Parse rocprofv3 counter CSVs from run_bmm_a16w8_profile.sh and print a diagnosis table.

Usage:
    python tests/kernels/parse_bmm_counters.py /tmp/bmm_profile
    python tests/kernels/parse_bmm_counters.py /tmp/bmm_profile --csv summary.csv

rocprofv3 writes one CSV per config dir. Each row = one profiled dispatch.
This script averages across the N_PROFILE=5 dispatches per config and computes
derived metrics for the 5-bottleneck diagnosis.

Expected CSV columns (subset):
    Dispatch_ID, Kernel_Name, GPU_ID, Queue_ID, ...
    GRBM_GUI_ACTIVE, SQG_BUSY_CYCLES, SQ_INST_ISSUE_ALL_STALL,
    SQ_INST_ISSUE_TEX_STALL, SQ_INST_ISSUE_LDS_STALL,
    SQ_WAIT_CNT_LOAD, SQ_WAIT_CNT_DS, SQ_WAIT_BARRIER,
    SQ_VALU_WMMA_FLOP_BF16,
    TX_VCA_LDS_LOAD_BANDWIDTH_BYTES, TX_VCA_LDS_STORE_BANDWIDTH_BYTES,
    GL2C_EA_REQ_VC4_STALL, GL2C_LATENCY_FIFO_FULL,
    GC_EA_SE_SARB_DRAM_RD_SIZE_REQ, GC_EA_SE_SARB_DRAM_WR_SIZE_REQ,
    GL1XC_STALL_BUFFER_FULL, TX_VMW_GL1_VMW_BACK_PRESSURE,
    GLARBC_STALL_GL2_GL1, GLARBC_BUSY, GL2C_BUSY, GL2C_CYCLE
"""

import argparse
import csv
import os
import sys
from pathlib import Path


# gfx1250 hardware constants
GPU_CLK_HZ = 2.1e9    # ~2.1 GHz sclk (adjust if known from rocm-smi)
NUM_SE = 8             # 4 SE/XCD × 2 XCDs; verify with rocminfo
EA_REQ_BYTES = 32      # each EA DRAM request = 32B
# WMMA BF16 16×16×32: 2×16×16×32 = 16384 FLOPs per instruction per wave
# rocprofv3 SQ_VALU_WMMA_FLOP_BF16 counts *total* FLOPs across all waves.
# Use directly: XDL_util = FLOP / (peak_FLOPS × kernel_cycles / GPU_CLK)
PEAK_BF16_TFLOPS = 4200.0  # gfx1250 BF16 WMMA peak (TFLOPS)


def _get(row, key, default=0.0):
    v = row.get(key, "")
    try:
        return float(v) if v != "" else default
    except ValueError:
        return default


def parse_config_dir(config_name, config_dir):
    """Find and parse the rocprofv3 output CSV in config_dir."""
    # rocprofv3 may write to a subdirectory or directly into config_dir
    csvs = list(Path(config_dir).rglob("*.csv"))
    if not csvs:
        print(f"  [{config_name}] WARNING: no CSV found in {config_dir}")
        return None

    # Use the most recently modified CSV (rocprofv3 may create multiple)
    csv_path = max(csvs, key=lambda p: p.stat().st_mtime)
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"  [{config_name}] WARNING: CSV empty: {csv_path}")
        return None

    print(f"  [{config_name}] {len(rows)} dispatches from {csv_path.name}")
    return rows


def avg_counters(rows):
    """Average all numeric counter columns across rows."""
    if not rows:
        return {}
    keys = rows[0].keys()
    result = {}
    for k in keys:
        vals = [_get(r, k) for r in rows]
        try:
            result[k] = sum(vals) / len(vals)
        except Exception:
            result[k] = 0.0
    return result


def compute_metrics(c, kernel_us):
    """Derive diagnostic ratios from raw counter averages.

    kernel_us: measured kernel latency in µs (from bench sweep; optional).
               Used to compute absolute BW. If None, BW is not computed.
    """
    gui = c.get("GRBM_GUI_ACTIVE", 1)
    sqg_busy = c.get("SQG_BUSY_CYCLES", 0)
    all_stall = c.get("SQ_INST_ISSUE_ALL_STALL", 0)
    tex_stall = c.get("SQ_INST_ISSUE_TEX_STALL", 0)
    lds_stall = c.get("SQ_INST_ISSUE_LDS_STALL", 0)
    valu_stall = c.get("SQ_INST_ISSUE_VALU_STALL", 0)
    wait_load = c.get("SQ_WAIT_CNT_LOAD", 0)
    wait_ds = c.get("SQ_WAIT_CNT_DS", 0)
    wait_bar = c.get("SQ_WAIT_BARRIER", 0)
    wait_any = c.get("SQ_WAIT_ANY", 0)
    wmma_flop = c.get("SQ_VALU_WMMA_FLOP_BF16", 0)
    lds_load_bw = c.get("TX_VCA_LDS_LOAD_BANDWIDTH_BYTES", 0)
    lds_store_bw = c.get("TX_VCA_LDS_STORE_BANDWIDTH_BYTES", 0)
    vc4_stall = c.get("GL2C_EA_REQ_VC4_STALL", 0)
    gl2_cycle = c.get("GL2C_CYCLE", 1)
    gl2_busy = c.get("GL2C_BUSY", 0)
    gl2_fifo_full = c.get("GL2C_LATENCY_FIFO_FULL", 0)
    ea_rd = c.get("GC_EA_SE_SARB_DRAM_RD_SIZE_REQ", 0)
    ea_wr = c.get("GC_EA_SE_SARB_DRAM_WR_SIZE_REQ", 0)
    gl1_stall = c.get("GL1XC_STALL_BUFFER_FULL", 0)
    gl1_cycle = c.get("GL1C_CYCLE", 1)
    gl1_busy = c.get("GL1C_BUSY", 0)
    gl0_bp = c.get("TX_VMW_GL1_VMW_BACK_PRESSURE", 0)
    glarb_stall = c.get("GLARBC_STALL_GL2_GL1", 0)
    glarb_busy = c.get("GLARBC_BUSY", 1)
    wave_lvl = c.get("SQ_LEVEL_WAVE", 0)

    m = {}

    # A-bound
    m["WGP_util"]         = sqg_busy / gui if gui else 0
    m["issue_stall_rate"] = all_stall / gui if gui else 0
    m["VMEM_stall_frac"]  = tex_stall / all_stall if all_stall else 0
    m["LDS_stall_frac"]   = lds_stall / all_stall if all_stall else 0
    m["VALU_stall_frac"]  = valu_stall / all_stall if all_stall else 0
    m["wave_avg"]         = wave_lvl / gui if gui else 0

    # E-bound (wave wait breakdown)
    m["wait_VMEM_frac"]   = wait_load / wait_any if wait_any else 0
    m["wait_LDS_frac"]    = wait_ds / wait_any if wait_any else 0
    m["wait_bar_frac"]    = wait_bar / wait_any if wait_any else 0
    m["wait_total_frac"]  = wait_any / (gui * max(m["wave_avg"], 1)) if gui else 0

    # XDL utilization (workaround for broken SQ_INST_CYCLES_VALU_WMMA)
    # kernel_cycles = gui (GRBM_GUI_ACTIVE counts kernel-active cycles per CU)
    # peak FLOP/cycle = PEAK_BF16_TFLOPS × 1e12 / GPU_CLK_HZ
    peak_flop_per_cycle = PEAK_BF16_TFLOPS * 1e12 / GPU_CLK_HZ
    m["XDL_util"]         = wmma_flop / (peak_flop_per_cycle * gui) if gui else 0

    # LDS BW (absolute, divide by kernel time if known)
    if kernel_us and kernel_us > 0:
        m["LDS_load_TBps"]  = lds_load_bw / (kernel_us * 1e-6) / 1e12
        m["LDS_store_TBps"] = lds_store_bw / (kernel_us * 1e-6) / 1e12
        m["HBM_rd_TBps"]    = ea_rd * NUM_SE * EA_REQ_BYTES / (kernel_us * 1e-6) / 1e12
        m["HBM_wr_TBps"]    = ea_wr * NUM_SE * EA_REQ_BYTES / (kernel_us * 1e-6) / 1e12
        m["HBM_total_TBps"] = m["HBM_rd_TBps"] + m["HBM_wr_TBps"]

    # D-bound (HBM)
    m["HBM_credit_stall"] = vc4_stall / gl2_cycle if gl2_cycle else 0
    m["GL2_fifo_sat"]     = gl2_fifo_full / gl2_cycle if gl2_cycle else 0
    m["GL2_util"]         = gl2_busy / gl2_cycle if gl2_cycle else 0

    # C-bound (cache hierarchy)
    m["GL1_util"]         = gl1_busy / gl1_cycle if gl1_cycle else 0
    m["GL1_sat"]          = gl1_stall / gl1_cycle if gl1_cycle else 0
    m["GL2_stall_GLARB"]  = glarb_stall / glarb_busy if glarb_busy else 0

    return m


def diagnose(m):
    """Return primary bottleneck label based on metrics."""
    if m.get("HBM_credit_stall", 0) > 0.4:
        return "D-bound (HBM saturated)"
    if m.get("GL2_fifo_sat", 0) > 0.3:
        return "C/D-bound (GL2 saturated)"
    if m.get("GL1_sat", 0) > 0.3:
        return "C-bound (GL1 full)"
    if m.get("VMEM_stall_frac", 0) > 0.5:
        return "C-bound (VMEM → memory latency)"
    if m.get("LDS_stall_frac", 0) > 0.3:
        return "E/C-bound (LDS lgkmcnt stall)"
    if m.get("XDL_util", 0) > 0.7:
        return "A-bound (compute, XDL high)"
    if m.get("issue_stall_rate", 0) > 0.3:
        return "A-bound (issue stall)"
    if m.get("wait_bar_frac", 0) > 0.15:
        return "E-bound (barrier)"
    return "unclear — check raw counters"


def print_metrics(name, m):
    print(f"\n  {'─'*60}")
    print(f"  Config: {name}")
    print(f"  {'─'*60}")
    print(f"  [A] WGP_util={m['WGP_util']:.2%}  wave_avg={m['wave_avg']:.1f}  "
          f"issue_stall={m['issue_stall_rate']:.2%}")
    print(f"      VMEM_stall={m['VMEM_stall_frac']:.2%}  LDS_stall={m['LDS_stall_frac']:.2%}  "
          f"VALU_stall={m['VALU_stall_frac']:.2%}")
    print(f"  [A] XDL_util={m['XDL_util']:.2%} (BF16 WMMA)")
    print(f"  [E] wait_VMEM={m['wait_VMEM_frac']:.2%}  wait_LDS={m['wait_LDS_frac']:.2%}  "
          f"wait_bar={m['wait_bar_frac']:.2%}")
    print(f"  [C] GL1_util={m['GL1_util']:.2%}  GL1_sat={m['GL1_sat']:.2%}  "
          f"GL2_util={m['GL2_util']:.2%}  GL2_sat(FIFO)={m['GL2_fifo_sat']:.2%}")
    print(f"      GL2_stall_GLARB={m['GL2_stall_GLARB']:.2%}")
    print(f"  [D] HBM_credit_stall={m['HBM_credit_stall']:.2%}  "
          f"GL2_FIFO_full={m['GL2_fifo_sat']:.2%}")
    if "HBM_rd_TBps" in m:
        print(f"      HBM_rd={m['HBM_rd_TBps']:.2f} TB/s  "
              f"HBM_wr={m['HBM_wr_TBps']:.2f} TB/s  "
              f"total={m['HBM_total_TBps']:.2f} TB/s")
    if "LDS_load_TBps" in m:
        print(f"  [LDS] load={m['LDS_load_TBps']:.2f} TB/s  "
              f"store={m['LDS_store_TBps']:.2f} TB/s")
    print(f"  ⇒ {diagnose(m)}")


def main():
    parser = argparse.ArgumentParser(description="Parse bmm_a16w8 rocprofv3 counter CSVs")
    parser.add_argument("results_dir", help="Output dir from run_bmm_a16w8_profile.sh")
    parser.add_argument("--csv", help="Write aggregated metrics to CSV")
    parser.add_argument("--kernel-us", type=float, default=None,
                        help="Measured kernel latency in µs (from bench sweep) for BW calc")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: {results_dir} is not a directory")
        sys.exit(1)

    # Find config dirs (each subdirectory = one config)
    config_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if not config_dirs:
        print(f"No config subdirs found in {results_dir}")
        sys.exit(1)

    print(f"\n=== bmm_a16w8 Counter Analysis ===")
    print(f"Results dir: {results_dir}")
    print(f"Configs found: {[p.name for p in config_dirs]}")

    csv_rows = []

    for config_dir in config_dirs:
        name = config_dir.name
        rows = parse_config_dir(name, config_dir)
        if rows is None:
            continue

        avg = avg_counters(rows)
        m = compute_metrics(avg, args.kernel_us)
        print_metrics(name, m)

        if args.csv:
            row = {"config": name}
            row.update({k: f"{v:.4f}" if isinstance(v, float) else v
                        for k, v in m.items()})
            row["bottleneck"] = diagnose(m)
            csv_rows.append(row)

    if args.csv and csv_rows:
        fieldnames = ["config"] + [k for k in csv_rows[0] if k != "config"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nMetrics written to {args.csv}")


if __name__ == "__main__":
    main()
