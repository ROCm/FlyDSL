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

    summary_lines: list[str] = []
    summary_lines.append("## Best configs per shape (lowest dispatch + combine us)\n\n")
    summary_lines.append(
        "Picked the ``(block_num, warp_per_block)`` minimizing "
        "``dispatch_avg_us + combine_avg_us`` for each "
        "``(dispatch_dtype, p2p_read, max_tokens)``.\n\n"
    )

    for d_dtype in ("fp4", "fp8_ocp"):
        for p2p in (False, True):
            tag = f"{d_dtype} -> bf16, P2P-read = {p2p}"
            summary_lines.append(f"### {tag}\n\n")
            summary_lines.append(
                "| max_tokens | best block_num | best warp_per_block | "
                "dispatch us | combine us | dispatch GB/s | combine GB/s "
                "| total_recv |\n"
            )
            summary_lines.append(
                "|-----------:|---------------:|--------------------:|"
                "-----------:|-----------:|--------------:|-------------:"
                "|-----------:|\n"
            )
            for bs in sorted({c["case"]["max_tokens"] for c in records}):
                recs = groups.get((d_dtype, p2p, bs), [])
                if not recs:
                    summary_lines.append(f"| {bs} | -- | -- | -- | -- | -- | -- | -- |\n")
                    continue

                def _total(r):
                    fly = r.get("flydsl", {})
                    return fly.get("dispatch", {}).get("avg_us", float("inf")) + fly.get("combine", {}).get(
                        "avg_us", float("inf")
                    )

                best = min(recs, key=_total)
                bc = best["case"]
                bd = best["flydsl"]["dispatch"]
                bcb = best["flydsl"]["combine"]
                tr = best.get("total_recv", 0)
                summary_lines.append(
                    f"| {bs} | {bc['block_num']} | {bc['warp_per_block']} "
                    f"| {bd['avg_us']:.1f} | {bcb['avg_us']:.1f} "
                    f"| {bd['bw_gbps']:.1f} | {bcb['bw_gbps']:.1f} | {tr} |\n"
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
