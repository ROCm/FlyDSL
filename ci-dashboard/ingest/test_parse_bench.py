#!/usr/bin/env python3
"""Pure-Python tests for parse_bench (no GPU, no network).

Run:  python3 -m pytest ci-dashboard/ingest/test_parse_bench.py
  or:  python3 ci-dashboard/ingest/test_parse_bench.py
"""

import parse_bench as pb

# A miniature job log: timestamp prefixes + ANSI, a current table, a baseline-rebuild
# table (must be ignored), two comparison blocks, and a FlyDSL-vs-AIter sweep.
# Spacing is intentionally tight; the parser tokenizes on \\s+ so any padding works.
P = "2026-06-09T09:00:00.0Z "  # a representative Actions-log timestamp prefix
LOG = "".join(
    P + line + "\n"
    for line in [
        "op                     shape                          dtype       TB/s     TFLOPS",
        "softmax                32768x8192                     bf16       3.938          -",
        "gemm                   5120x5120x8320_tile64x256x128  fp8        0.430   1361.400",
        "gemm                   8192x8192x8192_tile64x256x256  fp4         skip       skip",
        "\x1b[36;1mfound_main=1\x1b[0m",
        "op                     shape                          dtype       TB/s     TFLOPS",
        "softmax                32768x8192                     bf16       4.105          -",
        "=== Benchmark: current vs main ===",
        "  softmax 32768x8192 bf16  main=  4.105 TB/s  current=  3.938 TB/s  ratio= 0.959x  delta= -0.167 ( -4.1%)",
        "  gemm 5120x5120x8320_tile64x256x128 fp8  main= 1336.46 TFLOPS  current= 1361.40 TFLOPS  ratio= 1.019x  delta= +24.9 ( +1.9%)",  # noqa: E501
        "=== Benchmark: current vs v0.2.0 ===",
        "  softmax 32768x8192 bf16  v0.2.0=4.040 TB/s  current=  3.938 TB/s  ratio= 0.975x  delta= -0.102 ( -2.5%)",
        "op         shape              dtype  FlyDSL(gpu us)  AIter(gpu us)    speedup",
        "wmma_gemm  4096x4096x4096     bf16          1,264.1        1,119.7      1.13x",
        "fp8_gemm   32x8192x8192       fp8              68.9          163.0      0.42x",
    ]
)


def _by_key(recs):
    return {(r.op, r.shape, r.dtype, r.metric): r for r in recs}


def test_metric_selection_and_skip():
    d = _by_key(pb.parse_log(LOG))
    assert d[("softmax", "32768x8192", "bf16", "TB/s")].value == 3.938  # memory op -> TB/s
    assert d[("gemm", "5120x5120x8320_tile64x256x128", "fp8", "TFLOPS")].value == 1361.4  # compute -> TFLOPS
    assert d[("gemm", "8192x8192x8192_tile64x256x256", "fp4", "TFLOPS")].status == "skip"


def test_first_table_wins_over_baseline_rebuild():
    # current softmax is 3.938; the later (baseline) table's 4.105 must not override.
    d = _by_key(pb.parse_log(LOG))
    assert d[("softmax", "32768x8192", "bf16", "TB/s")].value == 3.938


def test_comparison_main_and_tag_and_regression():
    d = _by_key(pb.parse_log(LOG))
    sm = d[("softmax", "32768x8192", "bf16", "TB/s")]
    assert sm.vs_main["delta_pct"] == -4.1 and sm.regression is True  # <= -3% gate
    assert sm.vs_tag["tag"] == "v0.2.0" and sm.vs_tag["delta_pct"] == -2.5
    gm = d[("gemm", "5120x5120x8320_tile64x256x128", "fp8", "TFLOPS")]
    assert gm.vs_main["delta_pct"] == 1.9 and gm.regression is False


def test_aiter_speedup_with_commas():
    sp = _by_key(pb.parse_aiter_compare(LOG))
    w = sp[("wmma_gemm", "4096x4096x4096", "bf16", "speedup")]
    assert w.value == 1.13 and w.extra["flydsl_us"] == 1264.1 and w.extra["aiter_us"] == 1119.7
    assert sp[("fp8_gemm", "32x8192x8192", "fp8", "speedup")].value == 0.42


def test_attach_meta_infers_arch():
    recs = pb.parse_log(LOG)
    out = pb.attach_meta(recs, runner="linux-flydsl-mi355-1", commit="abc", run_id=1, ts="t")
    assert all(r["arch"] == "gfx950" and r["source"] == "ci" for r in out)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all parser tests passed")
