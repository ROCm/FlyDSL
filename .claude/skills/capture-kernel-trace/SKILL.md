---
name: capture-kernel-trace
description: >
  Capture GPU kernel ATT (Advanced Thread Trace) via rocprofv3 on a remote Docker
  container or locally. Discovers kernel names, configures input.yaml with the target
  kernel_include_regex, runs rocprofv3 -i input.yaml with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1,
  and downloads the latest ui_output_agent_* directory for analysis.
  Usage: /capture-kernel-trace <test_script.py> [kernel_name_pattern]
allowed-tools: Bash,Read,Write,Edit,Grep,Glob
---

# Capture Kernel Trace

Capture rocprofv3 ATT traces from a GPU environment (local or remote Docker container),
then download the trace output for analysis.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<test_script>` | Yes | Python test/bench script to profile, e.g. `bench_ps_pingpong.py` |
| `[kernel_pattern]` | No | Kernel name regex. If omitted, discover via `--stats` first |

If no test script is provided, ask the user.

## Connection Info

**Ask the user for their remote access configuration** (or reuse what they gave
earlier in the session): SSH host and user, Docker container name (if any), and
the FlyDSL path on the remote (default: project root).

Commands below are written for a local GPU. On a remote container, run each one
through this wrapper, without its `PYTHONPATH`, `ROCR_VISIBLE_DEVICES` and
`FLYDSL_DEBUG_ENABLE_DEBUG_INFO` prefixes (the wrapper sets them; drop
`docker exec` when the checkout is on the host):

```bash
ssh $USER@$HOST \
  "docker exec -e PYTHONPATH=<flydsl_root>/python:<flydsl_root>/tests \
   -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 -e ROCR_VISIBLE_DEVICES=<gpu> \
   $CONTAINER bash -c 'cd /tmp && <CMD>'"
```

---

## Step 1: Deploy Test Script (remote only)

Skip this if the script is already on the remote (e.g. in the FlyDSL tests dir):

```bash
scp $TEST_SCRIPT $USER@$HOST:/tmp/
ssh $USER@$HOST "docker cp /tmp/$TEST_SCRIPT $CONTAINER:/tmp/"
```

---

## Step 2: Kernel Discovery (if no pattern provided)

```bash
rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $TEST_SCRIPT 2>&1
cat /tmp/discover_kernel_stats.csv
```

Present the kernel list and let the user pick, or auto-select the FlyDSL/target kernel
(typically contains `pa_decode`, `kernel_0`, or the function name from the test script).

Use these durations to find names, not to time kernels. Profiler overhead per
dispatch dominates short kernels: paged-attention decode kernels of a few
microseconds all read about 4.1-4.3 us here on gfx950. Time them with HIP events
or graph replay.

---

## Step 3: Configure input.yaml

Write `/tmp/input_trace.yaml` with the target `kernel_include_regex`:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_PATTERN>
       kernel_iteration_range: "[1, [2-4]]"
       output_file: out
       output_directory: /tmp/kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Key configuration:
- `kernel_include_regex`: Exact name or regex from Step 2
- `kernel_iteration_range`: `"[1, [2-4]]"` skips warmup (iteration 0), traces iterations 2-4
- `att_target_cu: 1`: Single CU for manageable output. Only waves placed on that CU are traced (see 5.3)
- `att_buffer_size: "0x6000000"`: 96MB per SE (increase to `0xC000000` if truncated)

---

## Step 4: Run rocprofv3 with ATT

Before running, check two things:

- **Launch the kernel eagerly.** ATT captures under HIP graph replay have been
  observed to hang. If the script times its kernel by replaying a graph, add an
  eager mode (a plain launch loop) and trace that.
- **Pin one GPU with `ROCR_VISIBLE_DEVICES`.** `HIP_VISIBLE_DEVICES` only filters
  devices inside HIP. rocprofv3 works at the ROCr level, and with only the HIP
  filter set it has been observed to set up ATT on every GPU in the node,
  including GPUs that other jobs are timing on.

```bash
rm -rf /tmp/kernel_trace_output
ROCR_VISIBLE_DEVICES=<gpu> FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=./ \
  rocprofv3 -i /tmp/input_trace.yaml -- python $TEST_SCRIPT 2>&1
```

`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` compiles DWARF debug info into the HSACO, so
`code.json` maps each ISA instruction to its source file:line. Allow 3-5
minutes for JIT compilation + trace collection.

---

## Step 5: Download Trace Output

### 5.1 Find the latest ui_output_agent_* directory

```bash
ls -td /tmp/kernel_trace_output/ui_output_agent_* 2>/dev/null | head -5
```

The output directories are named `ui_output_agent_<PID>_dispatch_<N>`. Pick the latest.

### 5.2 Download to local (remote only)

```bash
LOCAL_TRACE_DIR=./trace_data/$(date +%Y%m%d_%H%M%S)_$KERNEL_SHORT_NAME
mkdir -p $LOCAL_TRACE_DIR
UI_OUTPUT_DIR=<latest ui_output_agent_* path>

ssh $USER@$HOST "docker cp $CONTAINER:$UI_OUTPUT_DIR /tmp/ui_trace_download"
scp -r $USER@$HOST:/tmp/ui_trace_download/* $LOCAL_TRACE_DIR/
# Kernel trace CSV (timing, VGPR info)
ssh $USER@$HOST "docker cp $CONTAINER:/tmp/kernel_trace_output/out_kernel_trace.csv /tmp/"
scp $USER@$HOST:/tmp/out_kernel_trace.csv $LOCAL_TRACE_DIR/
```

### 5.3 Verify download

```bash
ls -la $LOCAL_TRACE_DIR/
# Should contain: code.json, occupancy.json, filenames.json, wstates*.json, se*_*.json

# Quick validation
python3 -c "
import glob, json
d = '$LOCAL_TRACE_DIR'
waves = len(glob.glob(f'{d}/se*_sm*_sl*_wv*.json'))
code = json.load(open(f'{d}/code.json')).get('code') or []
has_src = sum(1 for i in code if i[3])
print(f'Waves: {waves}, instructions: {len(code)}, with source mapping: {has_src} ({100*has_src//max(len(code),1)}%)')
if not waves or not code:
    raise SystemExit('no waves on the target CU: rerun with another att_target_cu')
"
```

If no wave of the dispatch runs on `att_target_cu`, rocprofv3 still exits 0 and
writes the dispatch directory, but with `"code": null` and no wave files, and
the failure surfaces only later in an analyzer. Grids smaller than the CU count
(a split-K reduce of a few dozen workgroups, say) are the usual cause; this was
observed on gfx942 (80 CUs) with a reduce kernel traced on CU 0. Rerun with
another `att_target_cu`, keep `att_shader_engine_mask: "0xf"`, or trace a
larger launch of the same kernel.

---

## PMC Mode: Cache / HBM Counter Capture (separate from ATT)

ATT (above) gives per-instruction stall timing but **no cache counters**. To
answer "what is the L2 hit rate / HBM read efficiency", capture hardware
performance counters (PMC) in a **separate** run. PMC and ATT cannot be
combined in one job.

**Counter set** (L2 + HBM read efficiency):

```yaml
# /tmp/pmc_l2.yaml
jobs:
  - pmc: [TCC_HIT_sum, TCC_MISS_sum, TCC_REQ_sum]
    kernel_include_regex: pa_decode_ps_kernel_0
    output_file: pmc_l2
    output_directory: /tmp/pmc_out
    output_format: [csv]
```

```yaml
# /tmp/pmc_ea.yaml  (HBM-facing read requests + 32B-partial fraction)
jobs:
  - pmc: [TCC_EA0_RDREQ_sum, TCC_EA0_RDREQ_32B_sum, TCC_EA0_RDREQ_DRAM_sum, TCP_TCC_READ_REQ_sum]
    kernel_include_regex: pa_decode_ps_kernel_0
    output_file: pmc_ea
    output_directory: /tmp/pmc_ea_out
    output_format: [csv]
```

Run each job separately, pinned as in Step 4:

```bash
ROCR_VISIBLE_DEVICES=<gpu> PYTHONPATH=./ rocprofv3 -i /tmp/pmc_l2.yaml -- python <test_script> --perf
ROCR_VISIBLE_DEVICES=<gpu> PYTHONPATH=./ rocprofv3 -i /tmp/pmc_ea.yaml -- python <test_script> --perf
```

**CRITICAL — keep each job to a single hardware pass (≤ ~4 TCC counters).**
Multi-pass collection has been observed to trigger a **GPU Hang (HW
Exception)** on gfx942; split counters into single-pass jobs as above.

Discover available counters with:
```bash
rocprofv3 --list-avail 2>/dev/null | grep -iE "TCC_HIT|TCC_MISS|TCC_EA0_RDREQ|TCP_TCC_READ"
```

Output lands in `<output_directory>/pass_1/<output_file>_counter_collection.csv`.
Analyze with `kernel-trace-analysis/scripts/pmc_l2_analyzer.py` (see that skill).

Quick interpretation:
- **L2 hit rate** = `TCC_HIT/(TCC_HIT+TCC_MISS)`. For independent per-sequence
  paged-KV decode, ~1-3% is **expected** (streaming, no reuse) — not a bug.
- **32B fraction** = `TCC_EA0_RDREQ_32B/TCC_EA0_RDREQ`. ~0% = full 64B lines,
  no spatial-locality waste.

---

## Output

After capture, report:

1. **Trace location**: Local path to the downloaded trace directory
2. **Kernel info**: Name, VGPR/AGPR counts, grid size, duration (from out_kernel_trace.csv;
   the profiler inflates the duration of short kernels, see Step 2)
3. **Source mapping**: Whether debug info is present (% of instructions with source annotations)
4. **Instruction count**: Total instructions in code.json
5. **Next step**: Suggest running `/kernel-trace-analysis` on the downloaded trace for bottleneck analysis

Example output:
```
Trace captured: ./trace_data/20260325_153000_pa_decode/
  Kernel: pa_decode_sw_kernel_0
  Duration: 208.3 us
  arch_vgpr=96, accum_vgpr=128, SGPR=80
  Instructions: 2692, source-mapped: 2105 (78%)

Run /kernel-trace-analysis to analyze bottlenecks.
```

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | Install decoder: see kernel-trace-analysis skill Step 3 |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch, update both |
| Empty ui_output_agent_* | kernel_include_regex didn't match -- re-check kernel name from Step 2 |
| `code.json` has `"code": null`, no `se*_wv*.json` files | No wave ran on `att_target_cu` (grid smaller than the CU count) -- rerun with another CU (5.3) |
| No source mapping in code.json | Ensure `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` is set |
| Trace truncated (missing instructions) | Increase `att_buffer_size` to `0xC000000` (192MB) |
| SSH timeout | Increase timeout, check host connectivity |
| `kernel_iteration_range` mismatch | Test runs fewer iterations than expected -- use `"[0, [1-2]]"` |
| GPU Hang / HW Exception during PMC capture | Counter list forced multi-pass — split into single-pass jobs of ≤ ~4 TCC counters |
| ATT capture hangs and the process never exits | The script replays a HIP graph — trace an eager launch loop instead (Step 4) |
| Profiler activity or memory on GPUs you did not select | Only `HIP_VISIBLE_DEVICES` was set — pin with `ROCR_VISIBLE_DEVICES` (Step 4) |
| PMC CSV empty / wrong kernel row | Match `Kernel_Name` substring; the first row may be a torch/elementwise kernel |
