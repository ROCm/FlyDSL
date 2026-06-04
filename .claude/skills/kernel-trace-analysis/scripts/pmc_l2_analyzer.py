"""
PMC L2 / HBM efficiency analyzer.

Parses rocprofv3 PMC counter-collection output and reports L2 cache behaviour
and HBM read efficiency for a kernel.  Complements hotspot_analyzer.py, which
reads ATT instruction timing (no cache counters).

Supported input formats:
    CSV  : ``*_counter_collection.csv`` produced by older rocprofv3 or manually
           exported from a rocpd DB.  Expected columns:
               Dispatch_Id, Kernel_Name, Counter_Name, Counter_Value
    rocpd: SQLite DB (``*.db``) written by ``rocprofv3 --pmc`` on ROCm >= 7.x.
           Tables are discovered automatically by UUID suffix:
               rocpd_info_kernel_symbol_<uuid>
               rocpd_kernel_dispatch_<uuid>
               rocpd_info_pmc_<uuid>
               rocpd_pmc_event_<uuid>

Counters expected (collect via capture-kernel-trace "PMC mode"):
    L2 hit rate:        TCC_HIT_sum, TCC_MISS_sum, TCC_REQ_sum
    line utilization:   TCC_EA0_RDREQ_sum, TCC_EA0_RDREQ_32B_sum
    HBM traffic:        TCC_EA0_RDREQ_DRAM_sum
    L1->L2:             TCP_TCC_READ_REQ_sum

Usage:
    # CSV path
    python pmc_l2_analyzer.py pmc_l2_counter_collection.csv \\
        [pmc_ea_counter_collection.csv ...] \\
        [--kernel pa_decode_ps_kernel_0] [--ideal-gb 8.59] [--ea-channels 2]

    # rocpd DB path (ROCm >= 7.x)
    python pmc_l2_analyzer.py pmc_l2_results.db [pmc_ea_results.db ...] \\
        --kernel pa_decode_ps_kernel_0

Interpretation:
    L2 hit rate    : HIT/(HIT+MISS).  For decode with independent per-sequence
                     paged KV there is no inter-CTA reuse, so ~1-3% is EXPECTED
                     and correct (streaming).  A high value only appears when
                     the workload has real reuse (e.g. shared-prefix serving).
    32B fraction   : TCC_EA0_RDREQ_32B / TCC_EA0_RDREQ.  Fraction of HBM reads
                     that are partial 32B lines.  High % => scattered access /
                     poor spatial locality => wasted bandwidth.  ~0% => full
                     64B lines, no line-level waste.
    over-fetch     : measured HBM read bytes / ideal bytes.  ~1.0 => the kernel
                     reads exactly what it needs; >>1.0 => redundant fetches.
"""

import argparse
import csv
import re
import sqlite3
from collections import defaultdict


def _load_counters_from_csv(path, kernel):
    """Load PMC counters from a rocprofv3 CSV counter-collection file."""
    agg = defaultdict(float)
    dispatches = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            kn = r.get("Kernel_Name", "")
            if kernel and kernel not in kn:
                continue
            name = r.get("Counter_Name")
            val = r.get("Counter_Value")
            if name is None or val in (None, ""):
                continue
            agg[name] += float(val)
            dispatches.add(r.get("Dispatch_Id"))
    return agg, dispatches


def _load_counters_from_db(path, kernel):
    """Load PMC counters from a rocpd SQLite DB (rocprofv3 >= ROCm 7.x).

    rocprofv3 writes UUID-suffixed tables; the UUID is discovered by searching
    for ``rocpd_pmc_event_*``.  Required tables:

        rocpd_info_kernel_symbol_<uuid> : id | name
        rocpd_kernel_dispatch_<uuid>    : id | kernel_symbol_id | ...
        rocpd_info_pmc_<uuid>           : id | name
        rocpd_pmc_event_<uuid>          : id | pmc_id | dispatch_id | value

    Returns the same ``(agg, dispatches)`` pair as the CSV path.
    """
    agg = defaultdict(float)
    dispatches = set()

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        tables = {t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        # Discover UUID from rocpd_pmc_event_<uuid>
        uuid = None
        for t in sorted(tables):
            m = re.match(r"rocpd_pmc_event_(.+)", t)
            if m:
                uuid = m.group(1)
                break
        if uuid is None:
            return agg, dispatches

        sym_t = f"rocpd_info_kernel_symbol_{uuid}"
        disp_t = f"rocpd_kernel_dispatch_{uuid}"
        pmc_info_t = f"rocpd_info_pmc_{uuid}"
        pmc_event_t = f"rocpd_pmc_event_{uuid}"

        required = {sym_t, disp_t, pmc_info_t, pmc_event_t}
        if not required.issubset(tables):
            missing = required - tables
            print(f"  Warning: rocpd DB missing tables: {', '.join(sorted(missing))}")
            return agg, dispatches

        # Resolve dispatch ids matching the kernel filter.
        if kernel:
            rows = cur.execute(
                f"SELECT d.id FROM {disp_t} d" f" JOIN {sym_t} s ON d.kernel_symbol_id = s.id" f" WHERE s.name LIKE ?",
                (f"%{kernel}%",),
            ).fetchall()
        else:
            rows = cur.execute(f"SELECT id FROM {disp_t}").fetchall()
        dispatch_ids = [r[0] for r in rows]
        if not dispatch_ids:
            return agg, dispatches

        dispatches = set(dispatch_ids)

        # Aggregate counter values for the matched dispatches.
        placeholders = ",".join("?" * len(dispatch_ids))
        events = cur.execute(
            f"SELECT p.name, e.value"
            f" FROM {pmc_event_t} e"
            f" JOIN {pmc_info_t} p ON e.pmc_id = p.id"
            f" WHERE e.dispatch_id IN ({placeholders})",
            dispatch_ids,
        ).fetchall()

        for name, val in events:
            agg[name] += float(val)
    finally:
        conn.close()

    return agg, dispatches


def load_counters(paths, kernel):
    """Load and aggregate PMC counters from one or more CSV or rocpd DB files.

    Each path is handled independently based on its extension:
      ``.db``  → rocpd SQLite DB (rocprofv3 >= ROCm 7.x)
      anything else → CSV with columns Dispatch_Id/Kernel_Name/Counter_Name/Counter_Value

    Returns ``(agg, n_dispatches)`` where ``agg`` maps counter name to total
    value summed over all matched dispatches and files.
    """
    agg = defaultdict(float)
    all_dispatches = set()
    fmt_used = set()

    for p in paths:
        if p.endswith(".db"):
            sub_agg, sub_disp = _load_counters_from_db(p, kernel)
            fmt_used.add("rocpd")
        else:
            sub_agg, sub_disp = _load_counters_from_csv(p, kernel)
            fmt_used.add("csv")
        for k, v in sub_agg.items():
            agg[k] += v
        all_dispatches |= sub_disp

    if fmt_used:
        print(f"  Input format(s): {', '.join(sorted(fmt_used))}")
    return agg, len(all_dispatches)


def main():
    ap = argparse.ArgumentParser(description="PMC L2/HBM efficiency analyzer")
    ap.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE",
        help="PMC counter file(s): *_counter_collection.csv or *.db (rocpd SQLite)",
    )
    ap.add_argument("--kernel", default="", help="substring filter on Kernel_Name")
    ap.add_argument(
        "--ideal-gb",
        type=float,
        default=0.0,
        help="ideal HBM read bytes per dispatch in GB (for over-fetch ratio)",
    )
    ap.add_argument(
        "--ea-channels",
        type=int,
        default=2,
        help="EA interfaces to scale single-channel EA0 counters by (default 2)",
    )
    args = ap.parse_args()

    agg, ndisp = load_counters(args.inputs, args.kernel)
    if not agg:
        print("No matching counter rows found.")
        return 1

    print(f"  Dispatches matched: {ndisp}")
    hit = agg.get("TCC_HIT_sum", 0)
    miss = agg.get("TCC_MISS_sum", 0)
    ea = agg.get("TCC_EA0_RDREQ_sum", 0)
    ea32 = agg.get("TCC_EA0_RDREQ_32B_sum", 0)
    dram = agg.get("TCC_EA0_RDREQ_DRAM_sum", 0)
    tcp = agg.get("TCP_TCC_READ_REQ_sum", 0)

    print("\n  L2 cache")
    print("  --------")
    if hit + miss > 0:
        print(f"  TCC_HIT_sum  = {hit:,.0f}")
        print(f"  TCC_MISS_sum = {miss:,.0f}")
        print(f"  L2 hit rate  = {100*hit/(hit+miss):.1f}%   (streaming decode: ~1-3% expected)")
    if tcp:
        print(f"  TCP->TCC read req (L1->L2) = {tcp:,.0f}")

    if ea > 0:
        ea64 = ea - ea32
        bytes_ea = (ea64 * 64 + ea32 * 32) * args.ea_channels
        print("\n  HBM read efficiency")
        print("  -------------------")
        print(f"  TCC_EA0_RDREQ (L2->HBM) = {ea:,.0f}")
        print(f"  32B partial fraction    = {100*ea32/ea:.1f}%   (~0% = full 64B lines, no waste)")
        print(f"  DRAM reads              = {dram:,.0f}")
        print(f"  est HBM read bytes      = {bytes_ea/1e9:.1f} GB  (EA0 x{args.ea_channels} channels)")
        if args.ideal_gb > 0 and ndisp:
            ideal = args.ideal_gb * ndisp * 1e9
            print(f"  ideal bytes             = {ideal/1e9:.1f} GB  ({args.ideal_gb} GB x {ndisp} disp)")
            print(f"  over-fetch ratio        = {bytes_ea/ideal:.2f}x   (~1.0 = no redundant fetch)")

    print("\n  Verdict")
    print("  -------")
    if hit + miss > 0:
        hr = 100 * hit / (hit + miss)
        if hr < 5:
            print("  L2 hit rate is near-zero => pure streaming, no reuse to exploit.")
            print("  Improving 'L2 hit rate' is a non-goal here; only real KV reuse")
            print("  (shared-prefix serving) would change it. ")
    if ea > 0 and ea32 / ea < 0.05:
        print("  Line utilization is full (>=95% 64B) => no spatial-locality waste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
