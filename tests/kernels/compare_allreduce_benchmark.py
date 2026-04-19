#!/usr/bin/env python3
"""Compare two allreduce benchmark CSVs (main vs PR) and flag regressions.

Usage:
    python3 compare_benchmark.py <main.csv> <pr.csv>

Exit code 1 if any case regresses more than BOTH thresholds:
    - relative increase > MAX_REGRESSION_PCT  (default 10%)
    - absolute increase > MIN_ABS_REGRESSION_US (default 5 us)
"""
import sys
import pandas as pd

MAX_REGRESSION_PCT = 10.0
MIN_ABS_REGRESSION_US = 5.0


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <main.csv> <pr.csv>")
        sys.exit(2)

    main_csv, pr_csv = sys.argv[1], sys.argv[2]

    main_df = pd.read_csv(main_csv)
    pr_df = pd.read_csv(pr_csv)

    main_agg = main_df[main_df["rank"] == "aggregate"].copy()
    pr_agg = pr_df[pr_df["rank"] == "aggregate"].copy()

    # Detect cases that failed/skipped in PR but succeeded on main
    pr_agg_indexed = pr_agg.set_index(["shape", "dtype"])
    main_agg_indexed = main_agg.set_index(["shape", "dtype"])

    pr_broken = pr_agg_indexed[
        (pr_agg_indexed["avg_time_us"] <= 0) | pr_agg_indexed["avg_time_us"].isna()
    ]
    main_ok = main_agg_indexed[
        (main_agg_indexed["avg_time_us"] > 0) & main_agg_indexed["avg_time_us"].notna()
    ]
    newly_broken = pr_broken.index.intersection(main_ok.index)

    # Performance comparison for cases that both sides ran successfully
    pr_valid = pr_agg_indexed[["avg_time_us"]].loc[
        (pr_agg_indexed["avg_time_us"] > 0) & pr_agg_indexed["avg_time_us"].notna()
    ]
    main_valid = main_agg_indexed[["avg_time_us"]].loc[
        (main_agg_indexed["avg_time_us"] > 0) & main_agg_indexed["avg_time_us"].notna()
    ]
    merged = pr_valid.join(main_valid, lsuffix="_pr", rsuffix="_main").dropna()

    fail_count = 0

    if not merged.empty:
        merged["delta_us"] = merged["avg_time_us_pr"] - merged["avg_time_us_main"]
        merged["delta_pct"] = (merged["delta_us"] / merged["avg_time_us_main"]) * 100.0

        print("=== Allreduce Benchmark: PR vs main ===")
        for (shape, dtype), row in merged.iterrows():
            regressed = (
                row["delta_pct"] > MAX_REGRESSION_PCT
                and row["delta_us"] > MIN_ABS_REGRESSION_US
            )
            tag = "REGRESSION" if regressed else "OK"
            if regressed:
                fail_count += 1
            print(
                f"  {shape:>20s} {dtype:>4s}  "
                f"main={row['avg_time_us_main']:8.2f} us  "
                f"PR={row['avg_time_us_pr']:8.2f} us  "
                f"delta={row['delta_us']:+8.2f} us ({row['delta_pct']:+5.1f}%)  "
                f"[{tag}]"
            )

    if len(newly_broken) > 0:
        print("\n=== Cases BROKEN in PR (work on main but fail on PR) ===")
        for shape, dtype in newly_broken:
            fail_count += 1
            err = pr_agg_indexed.loc[(shape, dtype)].get("error", "unknown")
            print(f"  {shape:>20s} {dtype:>4s}  [BROKEN]  error: {err}")

    if fail_count > 0:
        print(f"\nFAILED: {fail_count} issue(s) detected.")
        sys.exit(1)
    else:
        print("\nPASSED: No regression or breakage detected.")


if __name__ == "__main__":
    main()
