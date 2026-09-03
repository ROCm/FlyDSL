#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Plot GEMM kernel quality vs hipBLASLt (口径 B) from fair_baseline.json.

Usage::

    python tests/perf/qwenimage_vae_conv/plot_gemm_vs_hipblaslt.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
WIN, LOSE, DEF_WIN, DEF_LOSE, NEUTRAL = (
    "#2a6f6f",
    "#b44a3c",
    "#9ec9c9",
    "#e8b4ad",
    "#5c5c5c",
)


def lab(r):
    conv = f"{r['cin']}→{r['cout']} @{r['hin']}²"
    if r["stride"] != 1:
        conv += " s2"
    return f"{conv}  {r['M']}×{r['N']}×{r['K']}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=HERE / "fair_baseline.json")
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "figures" / "gemm_vs_hipblaslt.png",
    )
    args = p.parse_args()
    rows = json.loads(args.json.read_text())

    plt.rcParams.update(
        {
            "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )

    max_f = max(r["freq"] for r in rows)

    def bar_h(freq):
        return 0.22 + 0.72 * (freq / max_f)

    fig, ax = plt.subplots(figsize=(13.4, 10.8))
    gap = 0.22
    centers = []
    cursor = 0.0
    for r in rows:
        h = bar_h(r["freq"])
        cursor += h / 2
        centers.append(cursor)
        cursor += h / 2 + gap
    centers = np.array(centers)

    for yc, r in zip(centers, rows):
        h = bar_h(r["freq"])
        sb = r["mm"] / r["gemm_best"]
        sd = r["mm"] / r["gemm_def"]
        win = sb >= 1
        ax.barh(yc, sb, height=h, color=WIN if win else LOSE, zorder=2)
        ax.barh(yc, sd, height=h, color=DEF_WIN if win else DEF_LOSE, zorder=3)
        ax.text(
            sb + 0.03,
            yc,
            f"{sb:.2f}  ×{r['freq']}",
            va="center",
            fontsize=10,
            color=WIN if win else LOSE,
            zorder=5,
        )

    ax.axvline(1.0, color=NEUTRAL, ls="--", lw=1.2, zorder=4)
    ax.set_yticks(centers)
    ax.set_yticklabels([lab(r) for r in rows], fontsize=10)
    ax.set_ylim(centers[-1] + bar_h(rows[-1]["freq"]) / 2 + gap, -gap)
    ax.set_xlabel(
        "GEMM 加速比  hipBLASLt on-device µs / FlyDSL GEMM kernel on-device µs"
    )
    ax.set_xlim(0, 3.35)
    ax.set_title(
        "口径 B：GEMM kernel 质量对标 hipBLASLt（条宽 = 调用次数；同 M/N/K，同 FLOP）"
    )
    fig.subplots_adjust(left=0.32)
    ax.legend(
        handles=[
            Patch(facecolor=WIN, label="最优 tile，≥1 跑赢 hipBLASLt"),
            Patch(facecolor=LOSE, label="最优 tile，<1 落后"),
            Patch(facecolor=DEF_WIN, label="默认 tile（赢，浅青绿）"),
            Patch(facecolor=DEF_LOSE, label="默认 tile（输，浅红）"),
        ],
        loc="center right",
        bbox_to_anchor=(1.0, 0.34),
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        fontsize=10,
    )
    fig.text(
        0.01,
        0.012,
        "条越厚调用越密。核均为 3×3；标签中 a×b×c 是等价 GEMM 的 M×N×K（M=输出空间，N=Cout，K=Cin·R·S）。"
        "1024 路径加权 1.083×，高频且 GEMM-K≥1728 普遍落后。",
        color=NEUTRAL,
        fontsize=8,
    )
    fig.text(
        0.01,
        -0.004,
        "非同类对比：hipBLASLt 读已物化的 M×K 矩阵，FlyDSL 从约 9 倍小的原张量 gather；"
        "物化代价（1024 路径 23.77 ms）未计入 hipBLASLt。",
        color=NEUTRAL,
        fontsize=8,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
