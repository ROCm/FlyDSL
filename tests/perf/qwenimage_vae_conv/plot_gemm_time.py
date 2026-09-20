#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Plot per-shape GEMM time and achieved throughput from gemm_time.json.

Companion to ``plot_gemm_vs_hipblaslt.py``: that one plots the speedup ratio,
this one plots absolute µs next to achieved TFLOP/s. Rows are ordered by
``Cout`` so the small-channel throughput deficit reads off the right panel.

Usage::

    python tests/perf/qwenimage_vae_conv/plot_gemm_time.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FLY, BLAS, GREY, LOSE = "#2a6f6f", "#b0803a", "#5c5c5c", "#b44a3c"


def lab(r):
    s = f"{r['cin']}→{r['cout']} @{r['hin']}²" + ("" if r["stride"] == 1 else " s2")
    return f"{s}   ×{r['freq']}"


def sub(r):
    return f"M={r['M']}  N={r['N']}  K={r['K']}  tile {r['tile']}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=HERE / "gemm_time.json")
    p.add_argument("--out", type=Path, default=HERE / "figures" / "gemm_time_by_case.png")
    args = p.parse_args()

    rows = json.loads(args.json.read_text())
    rows.sort(key=lambda r: (r["cout"], r["K"], r["M"]))

    plt.rcParams.update(
        {
            "font.sans-serif": ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )

    n = len(rows)
    y = np.arange(n)
    h = 0.38
    fig, (axL, axR) = plt.subplots(
        1,
        2,
        figsize=(15.2, 9.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.06},
    )

    fly_us = np.array([r["gemm"] for r in rows])
    mm_us = np.array([r["mm"] for r in rows])
    fly_tf = np.array([r["flops"] / r["gemm"] / 1e6 for r in rows])
    mm_tf = np.array([r["flops"] / r["mm"] / 1e6 for r in rows])

    axL.barh(y - h / 2, fly_us, height=h, color=FLY, zorder=3, label="FlyDSL conv3d_implicit GEMM kernel")
    axL.barh(y + h / 2, mm_us, height=h, color=BLAS, zorder=3, label="hipBLASLt torch.mm（同 M/N/K）")
    for i, r in enumerate(rows):
        ratio = r["mm"] / r["gemm"]
        axL.text(
            max(fly_us[i], mm_us[i]) + 8,
            i,
            f"{ratio:.2f}×",
            va="center",
            fontsize=9,
            color=FLY if ratio >= 1 else LOSE,
            fontweight="bold",
            zorder=5,
        )
    axL.set_xlim(0, max(fly_us.max(), mm_us.max()) * 1.18)
    axL.set_xlabel("on-device kernel 耗时（µs，越短越好）")
    axL.set_title("单次调用耗时", fontsize=12, loc="left")
    axL.grid(axis="x", color="#e6e6e6", zorder=0)
    axL.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="none", fontsize=9.5)

    axR.barh(y - h / 2, fly_tf, height=h, color=FLY, zorder=3)
    axR.barh(y + h / 2, mm_tf, height=h, color=BLAS, zorder=3)
    for i in range(n):
        axR.text(max(fly_tf[i], mm_tf[i]) + 16, i, f"{fly_tf[i]:.0f}", va="center", fontsize=9, color=GREY, zorder=5)
    axR.set_xlim(0, max(fly_tf.max(), mm_tf.max()) * 1.12)
    axR.set_xlabel("达成算力（TFLOP/s，有效 K = Cin·R·S）")
    axR.set_title("达成算力 —— 小 channel 的真实差距在这一栏", fontsize=12, loc="left")
    axR.grid(axis="x", color="#e6e6e6", zorder=0)

    groups: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["cout"], []).append(i)
    for ax in (axL, axR):
        for idxs in list(groups.values())[:-1]:
            ax.axhline(idxs[-1] + 0.5, color="#d0d0d0", lw=0.8, zorder=1)

    axL.set_yticks(y)
    axL.set_yticklabels([lab(r) for r in rows], fontsize=9.5)
    axL.set_ylim(n - 0.5, -0.5)
    for i, r in enumerate(rows):
        axL.text(-6, i + 0.30, sub(r), ha="right", va="center", fontsize=7.2, color="#8a8a8a", zorder=5)
    for cout, idxs in groups.items():
        axR.text(
            axR.get_xlim()[1] * 0.995,
            float(np.mean(idxs)),
            f"Cout={cout}",
            ha="right",
            va="center",
            fontsize=9,
            color="#a0a0a0",
            rotation=-90,
        )

    fig.suptitle(
        "Qwen-Image VAE conv（bf16, gfx950）：FlyDSL 出厂选核 vs hipBLASLt 同 M/N/K —— 按 Cout 从小到大排列",
        fontsize=13.5,
        y=0.945,
        x=0.012,
        ha="left",
    )

    main_path = [r for r in rows if r["path"] == "1024"]
    w_fly = sum(r["gemm"] * r["freq"] for r in main_path) / 1e3
    w_mm = sum(r["mm"] * r["freq"] for r in main_path) / 1e3
    for dy, text in (
        (
            0.055,
            f"×N = 一次 T2I encode+decode 里该 shape 的调用次数。1024 路径加权：FlyDSL {w_fly:.2f} ms vs hipBLASLt "
            f"{w_mm:.2f} ms（{w_mm / w_fly:.3f}×）。左栏 FlyDSL 走 tile=None 出厂路径，不含 NCHW→NHWC 转置。",
        ),
        (
            0.032,
            "非同类对比：hipBLASLt 读已物化的 M×K im2col 矩阵（96→96 @1024² 达 1.81 GB），"
            "FlyDSL 直接从约 9 倍小的源张量 gather，物化代价未计入 hipBLASLt 一侧。",
        ),
        (
            0.009,
            "小 channel 达成算力低的两个直接原因：① N 方向掩码空转 —— Cout=96 用 128 宽 N tile（75% 有效）、"
            "Cout=3 用 32 宽（9%）；② K=Cin·9 太小时 k-tile 不够，软流水藏不住 global 延迟"
            "（3→96 的 K=27 只有 1 个 k-tile，16→384 的 K=144 只有 4.5 个）。",
        ),
    ):
        fig.text(0.012, dy, text, fontsize=8.4, color=GREY)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
