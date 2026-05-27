#!/usr/bin/env python3
"""Counter collection script for bmm_a16w8_gfx1250. Run under rocprofv3.

Usage (single config):
    rocprofv3 -i tests/kernels/bmm_a16w8_counters.yaml \\
        -- python tests/kernels/profile_bmm_a16w8.py dec_e8m0

Usage (all configs, one rocprofv3 session per config):
    bash tests/kernels/run_bmm_a16w8_profile.sh

Dispatch layout per config:
    Dispatches 0 .. N_WARMUP-1   : warmup (L2 warm-up, NOT for analysis)
    Dispatches N_WARMUP .. N_WARMUP+N_PROFILE-1 : profiled dispatches

The YAML uses dispatch_index to skip warmup. Set N_WARMUP to match.

Configs profiled:
    dec_e8m0        M=64,  tile_m=64,  nb=3, cl=1  — decode baseline (memory-bound)
    dec_e8m0_cl8    M=64,  tile_m=64,  nb=3, cl=8  — decode + A multicast
    pre_e8m0_m256   M=256, tile_m=128, nb=2, cl=1  — prefill (near compute-bound)
    dec_noscale     M=64,  tile_m=64,  nb=3, cl=1  — no_scale mode (fp8→bf16 direct)
    dec_tm32        M=64,  tile_m=32,  nb=3, cl=1  — smaller tile (compare vs tm64)

Counter analysis (see bmm_a16w8_counters.yaml for full list):
    GRBM_GUI_ACTIVE           → total active cycles (denominator for all util)
    SQG_BUSY_CYCLES           → WGP utilization = SQG_BUSY / GRBM_GUI_ACTIVE
    SQ_INST_ISSUE_ALL_STALL   → issue stall rate (A-bound indicator)
    SQ_INST_ISSUE_TEX_STALL   → VMEM issue stall (→ memory latency exposure)
    SQ_INST_ISSUE_LDS_STALL   → LDS issue stall (→ LDS bank conflict / lgkmcnt)
    SQ_WAIT_CNT_LOAD          → wave cycles spent waiting on VMEM loads
    SQ_WAIT_CNT_DS            → wave cycles spent waiting on LDS
    SQ_VALU_WMMA_FLOP_BF16    → actual BF16 WMMA FLOPs (XDL util proxy)
    TX_VCA_LDS_LOAD_BW_BYTES  → LDS load bandwidth (bytes)
    TX_VCA_LDS_STORE_BW_BYTES → LDS store bandwidth (bytes)
    GL2C_LATENCY_FIFO_FULL    → GL2 saturation (C/D-bound)
    GL2C_EA_REQ_VC4_STALL     → HBM credit stall (D-bound: HBM MC saturated)
    GC_EA_SE_SARB_DRAM_RD_SIZE_REQ → actual HBM read BW (× #SE × 32B = bytes)
    GC_EA_SE_SARB_DRAM_WR_SIZE_REQ → actual HBM write BW
    TX_VMW_GL1_VMW_BACK_PRESSURE    → GL0→GL1 pressure (GL1 is the bottleneck)
    GLARBC_STALL_GL2_GL1      → GL2 stalling GLARB (GL2 is the bottleneck)
"""

import argparse
import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from flydsl.runtime.device import get_rocm_arch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250

# ---------------------------------------------------------------------------
# Dispatch counts — MUST MATCH bmm_a16w8_counters.yaml dispatch_index
# ---------------------------------------------------------------------------
N_WARMUP = 10    # dispatches 0..9  : L2 warm-up (skipped by YAML)
N_PROFILE = 5    # dispatches 10..14: profiled (captured by YAML)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
B, K, N_OUT = 16, 4096, 1024
GROUP_K, GROUP_N = 128, 128

CONFIGS = {
    # label: (M, tile_m, m_warp, n_warp, nb, cluster_n, use_e8m0, no_scale)
    "dec_e8m0": dict(
        M=64,  tile_m=64,  m_warp=2, n_warp=4, num_buffers=3,
        cluster_n=1, use_e8m0_scale=True, no_scale=False,
    ),
    "dec_e8m0_cl8": dict(
        M=64,  tile_m=64,  m_warp=2, n_warp=4, num_buffers=3,
        cluster_n=8, use_e8m0_scale=True, no_scale=False,
    ),
    "pre_e8m0_m256": dict(
        M=256, tile_m=128, m_warp=2, n_warp=4, num_buffers=2,
        cluster_n=1, use_e8m0_scale=True, no_scale=False,
    ),
    "dec_noscale": dict(
        M=64,  tile_m=64,  m_warp=2, n_warp=4, num_buffers=3,
        cluster_n=1, use_e8m0_scale=False, no_scale=True,
    ),
    "dec_tm32": dict(
        M=64,  tile_m=32,  m_warp=2, n_warp=4, num_buffers=3,
        cluster_n=1, use_e8m0_scale=True, no_scale=False,
    ),
}


def _align_up(v, a):
    return ((v + a - 1) // a) * a


def _make_inputs(M_pad, use_e8m0, no_scale):
    torch.manual_seed(42)
    a = torch.randn((B, M_pad, K), dtype=torch.bfloat16).cuda().contiguous()
    b = torch.randn((B, K, N_OUT), dtype=torch.float32).clamp(-1, 1)
    b = b.to(torch.float8_e4m3fn).cuda().contiguous()

    if no_scale:
        scale = torch.zeros(1, dtype=torch.uint8).cuda()
    elif use_e8m0:
        sf = torch.rand((B, K // GROUP_K, N_OUT // GROUP_N)) * 0.1 + 0.01
        e8m0 = (torch.log2(sf.clamp(1e-38)).round().int() + 127).clamp(0, 255).byte()
        scale = e8m0.cuda().contiguous()
    else:
        scale = (torch.rand((B, K // GROUP_K, N_OUT // GROUP_N)) * 0.1 + 0.01).cuda().contiguous()

    c = torch.zeros((B, M_pad, N_OUT), dtype=torch.bfloat16).cuda()
    return a.view(-1), b.view(-1), scale.view(-1), c.view(-1)


def run_config(name, cfg):
    M = cfg["M"]
    M_pad = _align_up(M, cfg["tile_m"])

    print(f"\n[profile] config={name}  M={M} tile_m={cfg['tile_m']} "
          f"mw={cfg['m_warp']} nw={cfg['n_warp']} nb={cfg['num_buffers']} "
          f"cl={cfg['cluster_n']} e8m0={cfg['use_e8m0_scale']} "
          f"noscale={cfg['no_scale']}")

    fn = compile_bmm_a16w8_gfx1250(
        B=B, M=M_pad, N=N_OUT, K=K,
        group_k=GROUP_K, group_n=GROUP_N,
        tile_m=cfg["tile_m"], tile_n=128, tile_k=128,
        m_warp=cfg["m_warp"], n_warp=cfg["n_warp"],
        num_buffers=cfg["num_buffers"],
        cluster_n=cfg["cluster_n"], cluster_m=1,
        use_e8m0_scale=cfg["use_e8m0_scale"],
        no_scale=cfg["no_scale"],
    )

    a, b, scale, c = _make_inputs(M_pad, cfg["use_e8m0_scale"], cfg["no_scale"])
    stream = torch.cuda.current_stream()

    # Warmup — dispatches 0..N_WARMUP-1, NOT captured by rocprofv3 (dispatch_index skips them)
    print(f"  warmup: {N_WARMUP} dispatches (indices 0..{N_WARMUP - 1}, skipped in YAML)")
    for i in range(N_WARMUP):
        fn(c, a, b, scale, M_pad, stream)
    torch.cuda.synchronize()

    # Profiled — dispatches N_WARMUP..N_WARMUP+N_PROFILE-1, captured by rocprofv3
    print(f"  profiled: {N_PROFILE} dispatches (indices {N_WARMUP}..{N_WARMUP + N_PROFILE - 1})")
    for i in range(N_PROFILE):
        fn(c, a, b, scale, M_pad, stream)
    torch.cuda.synchronize()

    print(f"  [profile] done: config={name}")


def main():
    parser = argparse.ArgumentParser(description="bmm_a16w8 counter collection")
    parser.add_argument("config", nargs="?", default=None,
                        choices=list(CONFIGS.keys()),
                        help="Config to profile (default: all)")
    parser.add_argument("--list", action="store_true", help="List available configs")
    args = parser.parse_args()

    arch = str(get_rocm_arch())
    if not arch.startswith("gfx1250"):
        print(f"ERROR: requires gfx1250, got {arch}")
        sys.exit(1)

    if args.list:
        for k, v in CONFIGS.items():
            print(f"  {k:<20}  M={v['M']:<4} tile_m={v['tile_m']:<4} "
                  f"nb={v['num_buffers']} cl={v['cluster_n']}")
        return

    if args.config:
        targets = {args.config: CONFIGS[args.config]}
    else:
        targets = CONFIGS

    for name, cfg in targets.items():
        run_config(name, cfg)

    print(f"\n[profile] All configs done. rocprofv3 captured dispatches "
          f"{N_WARMUP}..{N_WARMUP + N_PROFILE - 1} per config.")


if __name__ == "__main__":
    main()
