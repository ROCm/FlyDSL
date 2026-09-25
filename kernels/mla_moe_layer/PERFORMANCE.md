# GLM-5 MLA + MoE: TileRT assembly inside FlyDSL

[简体中文](PERFORMANCE_zh.md)

Worktree: `/root/FlyDSL-glm5-perf`, branch `codex/glm5-perf`, based on
`2fc0cefb`. Claude's nine local tuning commits were recovered from
`/root/FlyDSL-glm5mono`; its staged changes are preserved there. Local
measurements, candidate snapshots, disassembly and logs are in
`/root/glm5-perf-results`.

The imported assembly, the hybrid with the final MoE reduction written in
FlyDSL, and the tuned regular `Glm5MlaMoeLayer` now reach approximately TileRT
latency for the measured shards. Select the regular kernel with `--backend flydsl`
or the hybrid with `--backend tilert_inline --replace ffn`. The regular kernel
does not import the TileRT assembly body.

## Measured scope and results

Hardware: 8 × MI355X, gfx950. Torch: `2.12.0+rocm7.14.0`.
TileRT: `0.1.6.post2`. Position 3000, sparse top-2048, hidden size 6144.
Each rank has **8 heads and expert intermediate size 256** at every GPU count.
Consequently, 2/4-GPU results measure smaller collectives on the same shard;
they are **not full-model TP2/TP4 shapes**.

Latency is microseconds per layer: a 128-launch HIP graph, two warmup replays,
nine measured replays, median of the slowest rank per replay. Tags and peer
buffers are cleared outside the timed interval. Weights and inputs are shared
between implementations. Graph output must match eager output exactly.

| GPUs | Implementation | S=1 | S=2 | S=4 |
|---:|---|---:|---:|---:|
| 2 | Adapted TileRT ASM in FlyDSL | 34.59 | 41.10 | 52.95 |
| 2 | Hybrid: FlyDSL final reduction/output | 34.54 | 40.82 | 52.54 |
| 2 | Regular FlyDSL kernel | 33.69 | 40.01 | 53.48 |
| 4 | Adapted TileRT ASM in FlyDSL | 35.39 | 42.08 | 53.60 |
| 4 | Hybrid: FlyDSL final reduction/output | 35.33 | 42.06 | 54.52 |
| 4 | Regular FlyDSL kernel | 34.55 | 41.52 | 54.57 |
| 8 | Native TileRT | 35.85 | 42.90 | 55.93 |
| 8 | TileRT ASM in FlyDSL | 36.10 | 42.83 | 55.69 |
| 8 | Hybrid: FlyDSL final reduction/output | 36.54 | 43.11 | 55.63 |
| 8 | Regular FlyDSL kernel | 35.65 | 43.03 | 56.32 |

Sources: `corrected-inline{2,4,8}.jsonl`, `corrected-tilert8.jsonl`,
`final-ffn{2,4,8}.jsonl`, `verified-flydsl{2,4,8}.jsonl`, and
`verified-tilert8.jsonl` in the results directory. Both the hybrid and regular
FlyDSL kernel are within about 2% of their corresponding assembly/native
reference on the slowest measured case; several regular-kernel cases are faster.
The previous regular eight-GPU S=4 result was 61.91 µs (`final-flydsl8.jsonl`).

**Superseded measurements:** earlier `matched-*`, `inline*`, `dispatch*`, and
`ffn*` runs without the `corrected-`/`final-` prefix used an incorrect Q-B
conversion. TileRT requires all non-positional rows followed by all RoPE rows,
with 64-row scale blocks. The corrected adapter reorders rows and expands the
original 128-row scales. Independent golden checks caught this error; those
older runs must not be used as matched-workload performance evidence.

## Implemented segments

| File | Responsibility |
|---|---|
| `tools/asm_bridge/capture.cpp` | Capture the 560-byte native argument structure and exact specialization name at an armed launch. A skip mode discovers the 8-peer ABI without executing it on a smaller group. |
| `tools/asm_bridge/inline.py` | Disassemble one specialization, relocate branches to labels, and execute it in a real `llvm.InlineAsmOp`. Check the code-object hash and ABI before use. |
| `tools/asm_bridge/replacements.py` | Replace CTA-role dispatch and thread coordinates with FlyDSL expressions. Keep argument prefetches and the live block-ID sign register. |
| `tools/asm_bridge/peer_count.py` | Adapt the collective for 2/4 peers. Preserve the seven-remote-slot stride, zero absent slots, and mask their sends and receives. |
| `tools/asm_bridge/epilogue.py` | Return from the assembly after expert-down computation; perform peer reduction, residual addition and BF16 output in FlyDSL. |
| `tools/asm_bridge/trace.py` | Timestamp producer/consumer boundaries for S=1/2/4. |

The S=1 intact import was checked instruction by instruction: all **6,254**
native instruction encodings matched the generated FlyDSL code object after
its two-instruction argument-pointer prologue. Native LDS size and launch shape
remain 256 CTAs × 512 threads. The bridge does not use runtime ASM substitution.

The final-reduction replacement preserves the native down stage's out-of-line
basic blocks. Its ASM inputs are declared read/write so LLVM preserves live
argument and thread/block values when control returns to FlyDSL. Local down
outputs begin at LDS byte 7168 / 6144 / 4096 for S=1 / 2 / 4; the residual is
at byte 27152. The replacement batches all peer polls and sums in rank order.
A separately tested 64-byte packet translation was slightly slower and is
archived as `ffn-packets-epilogue.py` with `packets8.jsonl`.

Native argument offsets established by differential capture:

- Epoch fields: 120, 392, 496, 536.
- Attention rank/count: 176, 180.
- FFN rank/count: 488, 492.

The released native wrapper supports only 1/8 peers. Its 2/4-peer adaptation is
validated against the independent golden; there is no released native TP2/TP4
whole-layer benchmark to quote.

The regular FlyDSL kernel uses one wave per peer destination, one peer pointer
per wave, mask-based CTA mapping, and eight-intermediate up/gate tiles pipelined
over samples. Two changes guided by the assembly comparison closed the remaining
gap: packed BF16 peer payloads and one sample per router CTA. The S=4 compiler
dump uses **224 VGPRs, 94 SGPRs and zero private/scratch bytes**, versus 256 VGPRs
and spills in the recovered version. Obsolete up/gate schedules were removed;
the reserved scratch layout remains compatible.

The peer format changes numerical behavior: each rank rounds its attention and
FFN partial to BF16 before the ordered FP32 reduction, matching TileRT's peer
precision. The independent end-to-end golden still uses FP32 peer partials and
the original 5% relative-L2 limit. The down-segment golden models the BF16 payload
rounding and retains its original tight output tolerance. The extended check
first exposed this reference mismatch on two eight-GPU S=1 inputs; the payload
precision is now explicit, without relaxing either tolerance.

## Segment-guided tuning

The S=4 rank-0 traces exposed later attention publication, router completion,
and down computation in the earlier regular kernel. The uninstrumented
eight-GPU ablation was:

| Candidate | S=1 | S=2 | S=4 |
|---|---:|---:|---:|
| Before peer-format/router changes | 36.22 | 45.21 | 61.91 |
| Packed BF16 peer payload | 35.61 | 43.27 | 57.93 |
| Plus one sample per router CTA | — | 42.94 | 56.09 |

These come from `final-flydsl8.jsonl`, `packed-peer8.jsonl`, and
`router-split8.jsonl`. Earlier shared-expert execution improved S=2 slightly but
regressed S=4; a larger activation poll batch did not improve S=4. Both experiments
are archived rather than selected. The final matrix above remeasures the cleaned
candidate. Native and regular traces are in `final-trace-inline-s4` and
`verified-trace-flydsl-s4`; the prior regular trace is `final-trace-flydsl-s4`.
Last-CTA milestones in these diagnostic traces are:

| S=4 milestone | Earlier FlyDSL | TileRT ASM | Tuned FlyDSL |
|---|---:|---:|---:|
| Attention state published | 30.23 | 27.59 | 27.66 |
| Router scores published | 35.12 | 31.83 | 30.65 |
| Routed up/gate published | 50.12 | 44.96 | 45.73 |
| Down computation complete | 57.52 | 53.18 | 52.97 |

Units are microseconds from launch entry, with instrumentation enabled. The
plot and CSV are `s4-segment-milestones.png` and `s4-segment-milestones.csv` in
the results directory.

## Reproduce

Use the existing compiler build, this worktree, and the installed TileRT package:

```bash
cd /root/FlyDSL-glm5-perf
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-perf:/root/tilert_pkg
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
mkdir -p /root/glm5-perf-results/repro
g++ -shared -fPIC -O2 kernels/mla_moe_layer/tools/asm_bridge/capture.cpp \
  -ldl -o /root/glm5-perf-results/repro/capture.so
export LD_PRELOAD=/root/glm5-perf-results/repro/capture.so

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend tilert_inline --replace ffn --npes 8 --samples 1 2 4 \
  --verify-golden --changing-inputs 5 \
  --output /root/glm5-perf-results/repro/hybrid8.jsonl \
  --asm-artifacts /root/glm5-perf-results/repro/asm
```

Repeat with `--npes 2` and `--npes 4`. Use `--replace none` for the imported
assembly, `--replace dispatch` on 1/8 peers for dispatch replacement,
`--backend tilert` on eight GPUs for native TileRT, and `--backend flydsl`
without `--replace`, `--verify-golden`, or `--changing-inputs` for the regular
kernel. Run GPU benchmarks sequentially.

The imported binary is `/root/tilert_isa/b14.co`, SHA256
`6e0517e924f042d63a8b5a4e13f86696ae5bc4c1eee32765b28456d65bd2eb65`.
Generated assembly and code objects are local artifacts, not checked-in source.
Another TileRT build needs a new ABI/segment audit before changing this guard.

For diagnostics, add `--trace --layers 16` to the intact inline or regular
FlyDSL benchmark. Inspect a rank's trace with:

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  /root/glm5-perf-results/repro/asm/s4/rank0/trace.pt
```

Inline traces for S=1/2/4 were run on eight GPUs and preserved exact native
output. Regular FlyDSL tracing also runs on 2/4 peers; NP2/S4 was exercised in
`verified-trace-flydsl2-s4`. Inline tracing of the 2/4-peer adapter remains
unsupported because it shares instrumentation registers.

Trace timestamps use the 100 MHz realtime counter. Instrumentation drains
memory operations and changes scheduling; use uninstrumented graph measurements
for performance claims. Some publication marks cover a subset of CTAs or the
last iteration of a repeated stage.

The hybrid checks five changing inputs per S/GPU combination against native
TileRT on eight GPUs and the adapted intact assembly on two/four GPUs, requiring
bitwise equality and rank agreement. Independent golden checks also cover the
initial input. Corrected-input dispatch replacement also passes all three sample
sizes on eight GPUs (`corrected-dispatch8.jsonl`). The regular kernel checks five
changing inputs for every 2/4/8-GPU × S=1/2/4 combination, with exact agreement
across ranks and segment goldens. The independent end-to-end comparison passed
41 of 45 inputs at relative L2 1.49–2.85%; the existing near-tied-routing check
skipped four inputs (NP2/S2 once, NP8/S2 once, NP8/S4 twice). Those inputs still
passed the segment checks and rank equality. Its test
CLI returns nonzero on failure:

```bash
/opt/venv/bin/python tests/kernels/test_glm5_mla_moe_layer.py \
  --npes 8 -S 4 --pos 3000 --iters 5
```
