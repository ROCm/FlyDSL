# MiniMax M3 MXFP8 prefill: dense eight-wave GEMM extended to MoE

The validated TP8, 32k-token path takes **2.231 ms including GPU sorting**, versus
**2.887 ms for AITER on identical inputs**: **1.29x throughput, 22.7% less GPU time**.
GEMM1 reaches **1,898,166 GFLOPS** and GEMM2 **987,738 GFLOPS**, counting only useful
routed GEMM operations. The geometry-matched balance control below reaches 2,148 TFLOPS; grouped GEMM1
with SwiGLU reaches 2,123 TFLOPS on the same M/N/K geometry.

Measured 2026-09-11 on gfx950, 256 CUs, ROCm 7.2.4, PyTorch 2.10.0. Exact versions,
commits, and the selected AITER CSV rows are in
[environment.json](minimax_m3_mxfp8/environment.json). The implementation shares
[kernels/gemm/mxfp8_gemm_8wave.py](../../kernels/gemm/mxfp8_gemm_8wave.py) with the
recent dense GEMM merge, `578cfc7a`, on FlyDSL base `acf7e67b`.

The target comes from
`../aiter/aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv`:

| Parameter | TP8 target |
| --- | ---: |
| Tokens | 32768 |
| Hidden dimension | 6144 |
| Intermediate dimension per rank | 384 |
| Experts / top-k | 129 / 5 |
| Activation / weight | FP8 E4M3, E8M0 scale per 32 values |
| Output | BF16 |
| Activation | MiniMax clamped SwiGLU, alpha=1.702, limit=7 |
| Gate layout / routing weights | G1U1; routing weights applied after GEMM2 |
| Physical K for GEMM2 | 512; compute only logical K=384 |
| Rows padded to 256 per expert | 177920 for the recorded input |

For tokens T, top-k R, hidden H, and intermediate I, useful FLOPs are
`4*T*R*H*I` for GEMM1 and `2*T*R*H*I` for GEMM2. GFLOPS are `FLOPs / us / 1000`.
Neither padded rows nor the padded K tail is counted. Activation, quantization,
and reduction operations are excluded from the numerator even when timed.

| Measured operation | Time (us) | Useful GFLOPS |
| --- | ---: | ---: |
| Input quantization + scatter scales | 192.5 | — |
| GEMM1 + SwiGLU | 814.6 | 1,898,166 |
| Intermediate quantization | 34.3 | — |
| GEMM2, before top-k reduction | 782.7 | 987,738 |
| Weighted top-k reduction | 462.5 | — |
| Pipeline, prepared routing | 2180.1 | 1,063,820 |
| Sorting + route maps, measured separately | 39.7 | — |
| Pipeline including sorting + maps | 2230.6 | 1,039,745 |
| AITER pipeline including sorting | 2886.8 | 803,411 |

The separately measured AITER stage1 wrapper takes 1252.8 us; its stage2 wrapper
takes 1436.6 us **including reduction**. Compare that stage2 wrapper with both
new GEMM2 and reduction (about 1245 us when summing their separate measurements),
not with the 782.7 us GEMM alone. Independently timed stages need not sum exactly
to the timed pipeline because their cache state and launch sequence differ.

Timings use `run_perftest`, 3 warmups and 20 iterations, and sum GPU kernel
durations. They exclude CPU dispatch gaps, router top-k selection, weight packing,
compilation, allocation, and the initial readback used to size the fixed-input
routing workspace. Both full pipelines include their own GPU sorting. This is a
kernel/pipeline benchmark with prepared storage, not serving wall-clock latency.
The random BF16 inputs, weights, and router-score distribution follow
`test_moe_2stage`; the direct `--aiter` comparison passes the same tensors and
routing decisions to both implementations.

[TP8 raw results](minimax_m3_mxfp8/tp8_32k.json),
[intermediate=768 results](minimax_m3_mxfp8/inter768_32k.json), and
[1k-token results](minimax_m3_mxfp8/tp8_1k.json) contain the individual measurements.
The intermediate=768 context shape takes 3.266 ms versus AITER's 4.529 ms (1.39x).
The 1k TP8 case takes 0.406 ms; it is a correctness/padding check, not the tuning target.

To reproduce AITER's saved configuration on this machine:

```bash
cd ../aiter
HIP_VISIBLE_DEVICES=0 \
AITER_CONFIG_FMOE=/root/aiter/aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv \
python op_tests/test_moe_2stage.py --no-flydsl-csv -q 9 -t 32768 \
  -dim 6144,384 6144,768 -e 129 -k 5 -a swiglu -hip 0,0 --kernel
```

The CSV's historical TP8 times are 884.5 us + 1347.9 us. They are not this
machine's reproduced times. The paired AITER integration corrects the full reference to quantize activations
at both stages; its four final 16k/32k CSV cases pass the unchanged normalized
accuracy check with differences of about 4e-6. The earlier BF16 pass-through
reference omitted activation quantization and is superseded.

From the FlyDSL root:

```bash
HIP_VISIBLE_DEVICES=0 python scripts/bench_mxfp8_moe_8wave.py \
  --aiter --check --output /tmp/minimax_tp8_32k.json
HIP_VISIBLE_DEVICES=0 python scripts/bench_mxfp8_moe_8wave.py \
  --aiter --check --inter-dim 768 --output /tmp/minimax_inter768_32k.json
HIP_VISIBLE_DEVICES=0 python scripts/bench_mxfp8_moe_8wave.py \
  --check --tokens 1024 --output /tmp/minimax_tp8_1k.json
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/kernels/test_mxfp8_moe_8wave.py \
  tests/kernels/test_mxfp8_gemm_8wave.py -q
```

The benchmark defaults to swizzle 1 for GEMM1 and 3 for GEMM2. Use
`--sweep-swizzle` to retune. The dense tests with `-s` also print the large-K and
K=512 dense controls; both quoted controls use M=N=8192, preshuffled B, swizzle 4.
Those controls isolate some shape effects, but are not identical to routed MoE.

The implementation adds the following to
[kernels/moe/mxfp8_moe_8wave.py](../../kernels/moe/mxfp8_moe_8wave.py):

- Expert IDs select packed weights/scales per 256-row tile; the dense launcher's
  public arguments remain unchanged.
- Quantize each source token once and scatter only scales into sorted order;
  GEMM1 gathers the FP8 payload. This avoids top-k replication of the large input.
- Fuse MiniMax SwiGLU into GEMM1. Reuse a separate existing LDS allocation per wave
  for the output exchange; the eight static LDS fields are not assumed contiguous.
- Use a transpose in LDS for coalesced BF16 GEMM2 stores, then a weighted FP32
  reduction. Preserve logical K=384 with a 512-element storage stride.
- Extend XCD remapping to grids not divisible by eight. Expert padding often
  produces those grids; the mapping remains a bijection.
- Adapt the existing FlyDSL sorter through a small packed-ID conversion kernel.

Correctness checks use an independent Torch dequantize/matmul/quantize reference,
with RoundUp E8M0 scales and both activation quantization steps. TP8's normalized
logits diff is **1.015e-6**, and intermediate=768's is about **1.4e-6**. The default
reference checks 128 tokens spread over the entire sequence, including both ends;
all output elements are checked for finiteness. `--reference-tokens` increases the
reference coverage. Outputs are also checked for exact repeatability. Full-sized
intermediate tensors were compared across five identical launches while debugging.
The full TP8 output differs from AITER by 1.179e-5 under the same logits metric.
This is numerical validation, not a model-quality evaluation.

The 25 MoE tests cover expert permutations, empty experts, ragged XCD grids,
logical K tails, quantization, routing maps, long-K gathered input with padding,
dynamic valid rows, physical B stride 384, both GEMM1 tiles, missing routes,
and repeated execution. The 16 existing dense tests also pass. AITER's initial
output is cloned before timing its stage2 wrapper: the atomic variant otherwise
mutates the correctness sample during repeated microbenchmark launches.

A correctness issue was found during the padding check: the dense pipeline's
`vmcnt(6)` allowance was too loose for grouped/gathered inputs. The final grouped
path uses `vmcnt(2)` before rotating LDS buffers, completing the A prefetches and
leaving only the last B-half loads outstanding. A separate barrier reunites the
staggered M wave groups before the epilogue reuses compute LDS. Initial 2.13–2.15 ms
results used the unsafe wait allowance and **are superseded**, even though a
prefix-only reference sample initially passed. The epilogue class source is also
included in the JIT cache key so edits cannot silently reuse stale binaries.

ATT and PMC were collected in separate passes using the installed `flyprof`
capture/analyze/report skills. Reproduce each stage in its own bundle:

```bash
flyprof capture mxfp8_moe_8wave --worktree /root/FlyDSL \
  --invocation 'python scripts/bench_mxfp8_moe_8wave.py --profile-stage 1' \
  --tag big --with-pmc --gpu 3 --bundle /tmp/m3_stage1 --timeout 600 -f json
flyprof counters --bundle /tmp/m3_stage1 --worktree /root/FlyDSL -f json
flyprof bubbles --bundle /tmp/m3_stage1 --tag big -f json
flyprof map --bundle /tmp/m3_stage1 -f json
flyprof report --bundle /tmp/m3_stage1 -f json
```

Repeat with `--profile-stage 2` and a new bundle. `--profile-stage` launches only
the selected prepared GEMM and avoids nested profiling. Full traces remain under
`/tmp/flydsl_mxfp8_moe/profile_wait2_stage{1,2}`; compact evidence, source locations,
and tool recommendations are in
[profile_summary.json](minimax_m3_mxfp8/profile_summary.json).
[profile_expectations.json](minimax_m3_mxfp8/profile_expectations.json) records the
known capture expectations: a real dispatch, at least 95% source mapping, and
barrier as the leading stall class. Investigate changes rather than loosening
these expectations to accommodate an empty or wrong capture.

| Final trace/PMC evidence | GEMM1 | GEMM2 |
| --- | ---: | ---: |
| Source mapping | 100.0% rounded | 99.9% |
| L2 hit rate | 67.6% | 60.4% |
| Barrier share of attributed stall cycles | 30.52% | 30.70% |
| VM wait share of attributed stall cycles | 17.63% | 17.79% |
| MFMA share of attributed stall cycles | 28.24% | 14.94% |
| Allocated VGPRs / thread | 256 | 251 |
| LDS / CTA | 128 KiB | 128 KiB |

These are trace attribution ratios, not percentages of wall-clock runtime saved
by deleting an instruction. The counter tool incorrectly infers BF16 compute
from the output dtype; its automatic compute roofline is therefore not used.
Its EA0 bandwidth estimate is an uncalibrated channel lower bound, not total HBM
bandwidth. The capture does not establish a calibrated LDS conflict rate.

There is further optimization space, with the following priorities:

1. **Overlap loads while preserving cross-wave dependencies.** Source-mapped
   barriers and VM waits are the strongest evidence. Compared with a fully drained
   `vmcnt(0)` grouped implementation, the final stage1 profile drops VM-wait share
   from 28.83% to 17.63%, barrier share from 35.70% to 30.52%, and profiled duration
   from 969 to 819 us. This is a measured improvement. The remaining barriers
   guard staggered LDS reads/reuse; the tool's suggestion to remove barriers is
   not proof that any particular barrier is redundant.
2. **Design a short-K tile with lower LDS/register cost.** The current 128 KiB LDS
   allocation permits only one CTA per CU on this device. GEMM2 has just three
   128-wide K iterations, so startup, synchronization, and output traffic are
   expensive relative to its useful FLOPs. More resident CTAs could hide latency,
   but smaller tiles may increase weight traffic. This is an unverified direction.
3. **Reduce the GEMM2/reduction traffic.** The sorted BF16 partial is about 2.19 GB.
   Writing it, reading useful routed rows, and writing final output account for
   about 4.60 GB of logical traffic before other operands. Reduction alone takes
   462 us. An output-layout or tile-ownership redesign is more promising than the
   tested contended BF16 atomic scheme, but requires new correctness/perf evidence.
4. **Fuse intermediate quantization if its synchronization cost stays low.** It
   currently costs 34 us for TP8 and also reads/writes the stage1 intermediate.
   Its standalone time is a modest fraction of the pipeline; it is a lower
   priority than stage2 traffic and load/barrier overlap.

The whole-K-in-LDS short-K variants were slower (roughly 1.04–1.07 ms for GEMM2
in their trials). Fused BF16 atomic reduction was much slower (about 2.91 ms for
GEMM2). An early-scale-prefetch trial failed correctness. These variants were
removed. No parity with the 8192-cubed dense GEMM is claimed.

The delivered launcher requires gfx950, N and expert-padded M aligned to 256,
32-value scale blocks, the documented weight packing, and the MiniMax activation.
The benchmark prepares storage for a fixed routing input; dynamic serving
workspace management and automatic AITER dispatch integration are separate work.


## Native AITER integration and geometry-matched follow-up

The paired AITER branch integrates this core with standard packed weights,
GPU-resident valid row counts, native CSV dispatch, and AOT precompilation.
The standalone measurements above use prepared workspaces; the integrated
adapter creates its workspaces through the normal AITER stage interface.
On identical seed-42 random input, the native results are:

| Tokens | I per rank | Original (us) | Eight-wave (us) | Speedup | GEMM1/GEMM2 swizzle |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16384 | 384 | 1567.5 | 1314.3 | 1.193x | 3 / 0 |
| 32768 | 384 | 2851.6 | 2347.5 | 1.215x | 1 / 3 |
| 16384 | 768 | 2182.1 | 1908.2 | 1.144x | 3 / 0 |
| 32768 | 768 | 4505.6 | 3437.1 | 1.311x | 1 / 3 |

The [full native sweep](minimax_m3_mxfp8/integrated_sweep.json) checks every
candidate against the original AITER path and verifies full-output repeatability.
The selected tile is 256x256 for all four rows. The grouped launcher now also
supports a GPU valid-row bound, independent B stride (`b_k=384` for TP8), and
128x512 as a tuning candidate. Static and dense launcher argument lists are
unchanged. AITER's two graph tests and eight independent-cache AOT/run-only
stage checks pass.

For balance mode, each expert receives mean M=1270.08 (padded to 1280),
N=768 and K=6144. Expert count is a batch dimension, not a multiplier of N/K.
GEMM1 + SwiGLU measures 728.2 us / 2123.2 TFLOPS; the grouped kernel without
activation measures 710.0 us / 2177.7 TFLOPS. This control writes twice the
BF16 columns, so the difference is not pure activation latency. A dense control
with the same total M/N/K but sharing one expert's weight measures 719.7 us /
2148.3 TFLOPS. A single-expert dense launch achieves only 228.1 useful TFLOPS
because its 15 CTAs cannot fill 256 CUs. See
[equivalent_tp8.json](minimax_m3_mxfp8/equivalent_tp8.json).

```bash
python scripts/bench_mxfp8_moe_8wave.py --balanced --check --equivalent-gemm \
  --output /tmp/equivalent_tp8.json
python scripts/bench_mxfp8_moe_8wave.py --balanced --check --stage1-tile 128x512
```

128x512 is slower here: TP8 GEMM1 1252.8 us / 1234 TFLOPS versus 727.0 us /
2127 TFLOPS for 256x256 in the direct tile comparison. N768 occupies two N512
tiles, with 25% of computed columns unused, and LDS rises to 160 KiB. At I768
(N1536, no N tail), it still loses: 1848.7 versus 1370.8 us.

Fresh [native ATT/PMC evidence](minimax_m3_mxfp8/native_profile.json) confirms
the same remaining bottlenecks: barrier attribution 30.82% / 29.39%, VM wait
17.43% / 16.82%, L2 hits 66.2% / 60.4%, and 128 KiB LDS for GEMM1/GEMM2.
The main further opportunities are short-K GEMM2 state reduction, less traffic
between GEMM2 and top-k reduction, and better overlap of scale/payload loads.
The native capture's debug locations coalesce some control flow onto the CTA
guard, so ISA waitcnt producer edges provide the precise dependency evidence.
The same roofline, bandwidth calibration, and stall-attribution limitations
above apply. The core has not been changed to remove required barriers.
