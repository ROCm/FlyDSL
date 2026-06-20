---
name: capture-kernel-trace
description: >
  Capture GPU kernel ATT (Advanced Thread Trace) via rocprofv3 on a remote Docker
  container or locally. Discovers kernel names, configures input.yaml with the target
  kernel_include_regex, runs rocprofv3 -i input.yaml with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1,
  and downloads the latest ui_output_agent_* directory for analysis.
  Usage: /capture-kernel-trace <test_script.py> [kernel_name_pattern]
tools: Bash,Read,Write,Edit,Grep,Glob
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

**Check MEMORY.md for the user's current remote access configuration.** If not found, ask the user for:
- SSH host and user
- Docker container name (if applicable)
- FlyDSL install path on remote (default: project root)

SSH command pattern (adjust per environment):
```bash
ssh $USER@$HOST \
  "docker exec -e PYTHONPATH=<flydsl_root>/python:<flydsl_root>/tests \
   -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
   $CONTAINER bash -c '<CMD>'"
```

For local execution (no SSH/Docker):
```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=./ <CMD>
```

---

## Workflow

```
Step 1: Deploy test script to remote container (if remote)
Step 2: Discover kernel names (if pattern not provided)
Step 3: Configure input.yaml with kernel_include_regex
Step 4: Run rocprofv3 -i input.yaml to collect ATT trace
Step 5: Find and download latest ui_output_agent_* to local
```

---

## Step 1: Deploy Test Script

If running on a remote container, copy the test script:

```bash
# Copy local file to container via SSH + docker cp
scp $TEST_SCRIPT $USER@$HOST:/tmp/
ssh $USER@$HOST "docker cp /tmp/$TEST_SCRIPT $CONTAINER:/tmp/"
```

If the test script is already on the remote (e.g., in the FlyDSL tests dir), skip this step.

---

## Step 2: Kernel Discovery (if no pattern provided)

Run rocprofv3 in stats mode to list kernel names:

```bash
# Remote
ssh $USER@$HOST \
  "docker exec -e PYTHONPATH=<flydsl_root>/python:<flydsl_root>/tests \
   $CONTAINER bash -c \
   'cd /tmp && rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $TEST_SCRIPT 2>&1'"

# Local
rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $TEST_SCRIPT 2>&1
```

Parse output to find kernel names:

```bash
cat /tmp/discover_kernel_stats.csv
```

Present the kernel list and let the user pick, or auto-select the FlyDSL/target kernel
(typically contains `pa_decode`, `kernel_0`, or the function name from the test script).

---

## Step 3: Configure input.yaml

Create the input.yaml with the target `kernel_include_regex`:

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
- `att_target_cu: 1`: Single CU for manageable output
- `att_buffer_size: "0x6000000"`: 96MB per SE (increase to `0xC000000` if truncated)

---

## Step 4: Run rocprofv3 with ATT

```bash
# Remote
ssh $USER@$HOST \
  "docker exec -e PYTHONPATH=<flydsl_root>/python:<flydsl_root>/tests \
   -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
   $CONTAINER bash -c \
   'cd /tmp && rm -rf /tmp/kernel_trace_output && rocprofv3 -i /tmp/input_trace.yaml -- python $TEST_SCRIPT 2>&1'"

# Local
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=./ \
  rocprofv3 -i /tmp/input_trace.yaml -- python $TEST_SCRIPT 2>&1
```

**IMPORTANT**: Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` to get source-to-assembly mapping
in the trace output. This enables DWARF debug info in the compiled HSACO, so `code.json`
will contain source file:line annotations for each ISA instruction.

Timeout: allow 3-5 minutes for JIT compilation + trace collection.

---

## Step 5: Download Trace Output

### 5.1 Find the latest ui_output_agent_* directory

```bash
# Remote
ssh $USER@$HOST \
  "docker exec $CONTAINER bash -c \
   'ls -td /tmp/kernel_trace_output/ui_output_agent_* 2>/dev/null | head -5'"

# Local
ls -td /tmp/kernel_trace_output/ui_output_agent_* 2>/dev/null | head -5
```

The output directories are named `ui_output_agent_<PID>_dispatch_<N>`. Pick the latest.

### 5.2 Download to local (remote only)

```bash
# Create local destination
LOCAL_TRACE_DIR=./trace_data/$(date +%Y%m%d_%H%M%S)_$KERNEL_SHORT_NAME
mkdir -p $LOCAL_TRACE_DIR

# Copy from container to host, then to local
UI_OUTPUT_DIR=<latest ui_output_agent_* path>

ssh $USER@$HOST "docker cp $CONTAINER:$UI_OUTPUT_DIR /tmp/ui_trace_download"
scp -r $USER@$HOST:/tmp/ui_trace_download/* $LOCAL_TRACE_DIR/
```

Also download supporting files:

```bash
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
import json, sys
with open('$LOCAL_TRACE_DIR/code.json') as f:
    data = json.load(f)
n = len(data.get('code', []))
has_src = sum(1 for i in data.get('code', []) if i[3])
print(f'Instructions: {n}, with source mapping: {has_src} ({100*has_src//max(n,1)}%)')
"
```

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

Run each (cache disabled is NOT needed — PMC doesn't use source mapping, so
leave `FLYDSL_RUNTIME_ENABLE_CACHE=1` for speed):

```bash
HIP_VISIBLE_DEVICES=<gpu> FLYDSL_RUNTIME_ENABLE_CACHE=1 PYTHONPATH=./ \
  rocprofv3 -i /tmp/pmc_l2.yaml -- python <test_script> --perf
HIP_VISIBLE_DEVICES=<gpu> FLYDSL_RUNTIME_ENABLE_CACHE=1 PYTHONPATH=./ \
  rocprofv3 -i /tmp/pmc_ea.yaml -- python <test_script> --perf
```

**CRITICAL — keep each job to a single hardware pass (≤ ~4 TCC counters).**
Packing many counters into one job forces multi-pass collection, which on
gfx942 has been observed to trigger a **GPU Hang (HW Exception)**. Split into
multiple single-pass jobs (as above) instead of one big counter list.

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
2. **Kernel info**: Name, VGPR/AGPR counts, grid size, duration (from out_kernel_trace.csv)
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
| No source mapping in code.json | Ensure `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` is set |
| Trace truncated (missing instructions) | Increase `att_buffer_size` to `0xC000000` (192MB) |
| SSH timeout | Increase timeout, check host connectivity |
| `kernel_iteration_range` mismatch | Test runs fewer iterations than expected -- use `"[0, [1-2]]"` |
| GPU Hang / HW Exception during PMC capture | Counter list forced multi-pass — split into single-pass jobs of ≤ ~4 TCC counters |
| PMC CSV empty / wrong kernel row | Match `Kernel_Name` substring; the first row may be a torch/elementwise kernel |
