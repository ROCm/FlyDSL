"""Post-process the mixed-dtype sweep jsonl: prepend a best-per-shape
summary (lowest combined dispatch+combine latency among the 9
``(block_num, warp_per_block)`` configs) to the markdown report.
"""

from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()

    records: list[dict] = []
    with open(args.jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Group by (dispatch_dtype, p2p_read, max_tokens) and pick the config
    # with the lowest dispatch_avg + combine_avg.
    groups: dict[tuple[str, bool, int], list[dict]] = {}
    for rec in records:
        if rec.get("failed"):
            continue
        c = rec["case"]
        key = (c["dispatch_dtype"], not c["use_external_inp_buf"], c["max_tokens"])
        groups.setdefault(key, []).append(rec)

    def _fly_total(r):
        fly = r.get("flydsl", {})
        return fly.get("dispatch", {}).get("avg_us", float("inf")) + fly.get("combine", {}).get("avg_us", float("inf"))

    def _mori_total(r):
        mori = r.get("mori", {})
        if not mori or "dispatch" not in mori or "combine" not in mori:
            return float("inf")
        return mori["dispatch"].get("avg_us", float("inf")) + mori["combine"].get("avg_us", float("inf"))

    def _fmt(v: float) -> str:
        return "n/a" if v == float("inf") else f"{v:.1f}"

    summary_lines: list[str] = []
    summary_lines.append("## Best configs per shape (FlyDSL vs mori best, lowest dispatch + combine us)\n\n")
    summary_lines.append(
        "For each ``(dispatch_dtype, p2p_read, max_tokens)`` we independently "
        "pick the ``(block_num, warp_per_block)`` that minimizes the impl's "
        "own ``dispatch + combine us`` (so FlyDSL and mori may end up with "
        "different chosen configs). ``fly_x`` = ``mori_best_total / "
        "fly_best_total`` -- > 1.00x means FlyDSL's best config is faster "
        "than mori's best config.\n\n"
    )

    for d_dtype in ("fp4", "fp8_ocp"):
        for p2p in (False, True):
            tag = f"{d_dtype} -> bf16, P2P-read = {p2p}"
            summary_lines.append(f"### {tag}\n\n")
            summary_lines.append(
                "| max_tokens | fly cfg (block/warp) | fly_d_us | fly_c_us "
                "| fly_total | mori cfg (block/warp) | mori_d_us | mori_c_us "
                "| mori_total | fly_x |\n"
            )
            summary_lines.append(
                "|-----------:|:--------------------:|---------:|---------:"
                "|---------:|:---------------------:|----------:|----------:"
                "|----------:|------:|\n"
            )
            for bs in sorted({c["case"]["max_tokens"] for c in records}):
                recs = groups.get((d_dtype, p2p, bs), [])
                if not recs:
                    summary_lines.append(f"| {bs} | -- | -- | -- | -- | -- | -- | -- | -- | -- |\n")
                    continue
                fly_best = min(recs, key=_fly_total)
                mori_best = min(recs, key=_mori_total)
                fly_d = fly_best["flydsl"]["dispatch"]["avg_us"]
                fly_c = fly_best["flydsl"]["combine"]["avg_us"]
                fly_tot = fly_d + fly_c
                mori_sect = mori_best.get("mori") or {}
                mori_avail = "dispatch" in mori_sect and "combine" in mori_sect
                if mori_avail:
                    mori_d = mori_sect["dispatch"]["avg_us"]
                    mori_c = mori_sect["combine"]["avg_us"]
                    mori_tot = mori_d + mori_c
                    fly_x = f"{mori_tot / fly_tot:.2f}x" if fly_tot > 0 else "n/a"
                    mori_d_str = f"{mori_d:.1f}"
                    mori_c_str = f"{mori_c:.1f}"
                    mori_tot_str = f"{mori_tot:.1f}"
                    mori_cfg = f"{mori_best['case']['block_num']}/" f"{mori_best['case']['warp_per_block']}"
                else:
                    fly_x = "n/a"
                    mori_d_str = "n/a"
                    mori_c_str = "n/a"
                    mori_tot_str = "n/a"
                    mori_cfg = "n/a"
                summary_lines.append(
                    f"| {bs} "
                    f"| {fly_best['case']['block_num']}/"
                    f"{fly_best['case']['warp_per_block']} "
                    f"| {_fmt(fly_d)} | {_fmt(fly_c)} | {_fmt(fly_tot)} "
                    f"| {mori_cfg} | {mori_d_str} | {mori_c_str} "
                    f"| {mori_tot_str} | {fly_x} |\n"
                )
            summary_lines.append("\n")

    with open(args.md) as fh:
        body = fh.read()
    summary_text = "".join(summary_lines)
    # Inject after the front matter (after the "Total cases" line).
    marker = "Total cases:"
    pos = body.find(marker)
    if pos == -1:
        new_body = summary_text + body
    else:
        # Find end of the line containing ``Total cases:``.
        line_end = body.find("\n", pos)
        if line_end == -1:
            new_body = body + "\n\n" + summary_text
        else:
            new_body = body[: line_end + 1] + "\n" + summary_text + body[line_end + 1 :]
    with open(args.md, "w") as fh:
        fh.write(new_body)
    print(f"Wrote summary -> {args.md}")


if __name__ == "__main__":
    main()
