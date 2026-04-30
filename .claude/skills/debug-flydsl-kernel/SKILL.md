---
name: debug-flydsl-kernel
description: >
  Debug FlyDSL GPU kernels that produce NaN, inf, wrong results, or crash.
  Covers cache invalidation, tracing pitfalls (runtime conditionals, range vs
  range_constexpr), loop-carried state packing, buffer_load addressing, MFMA operand
  layout verification, LDS bank conflict diagnosis, and systematic error
  isolation (all-1s test, single-partition test, host-side tensor inspection).
  Use when a FlyDSL kernel produces incorrect output or compilation errors.
  Usage: /debug-flydsl-kernel
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Debug FlyDSL Kernel

## Step 0: Clear All Caches (ALWAYS DO THIS FIRST)

FlyDSL aggressively caches compiled kernels. Stale cache is the #1 cause of "my fix didn't work":

```bash
rm -rf ~/.flydsl /tmp/flydsl*
```

Also clear Python-level caches if using `@functools.lru_cache`:
```python
compile_my_kernel.cache_clear()
```

## Step 1: Classify the Error

| Symptom | Likely Cause | Go to |
|---|---|---|
| All NaN output | Softmax -inf/-inf, division by zero, uninitialized buffer | Section 2 |
| All zeros output | Wrong output address, uninitialized temp buffer | Section 3 |
| Partially wrong (>50% mismatch) | Wrong partition count, missing partitions, layout mismatch | Section 4 |
| Small errors (1-5% mismatch) | FP8 quantization, scale factor, off-by-one masking | Section 5 |
| Compilation error / crash | Type mismatch, loop-carried state, range vs range_constexpr | Section 6 |
| GPU hang | Infinite loop, deadlock in barrier, OOB memory access | Section 7 |

## 2. Debugging NaN

### 2.1 Softmax NaN: -inf minus -inf

When ALL tokens in a partition are masked (out of context), `qk_max = -inf`. Then `exp(s - qk_max) = exp(-inf - (-inf)) = exp(NaN) = NaN`.

**Fix**: Guard the exp calculation:
```python
safe_diff = (qk_max > NEG_INF).select(diff, ZERO_F)
```

### 2.2 Division by zero in normalization

When `exp_sum = 0` (all probs zero), `1/exp_sum = inf`.

**Fix**:
```python
safe_sum = (running_sum > ZERO_F).select(running_sum, fx.Float32(1.0))
inv_sum = fx.Float32(1.0) / safe_sum
```

### 2.3 Host-side NaN check

Add prints in the Python launch function to check intermediate buffers:
```python
torch.cuda.synchronize()
print(f"exp_sums nan={exp_sums.isnan().sum()}, inf={exp_sums.isinf().sum()}")
print(f"max_logits nan={max_logits.isnan().sum()}, range=[{max_logits.min():.4f}, {max_logits.max():.4f}]")
print(f"temp_out nan={temporary_output.isnan().sum()}")
```

## 3. Debugging All-Zeros Output

### 3.1 Wrong output address

Check stride parameters: if `stride_out_seq` or `stride_out_part` is wrong, output writes go to incorrect locations. Print strides:
```python
print(f"out strides: {output.stride()}, temp strides: {temporary_output.stride()}")
```

### 3.2 Partition slot mismatch

For multi-partition kernels, verify the output is written to `part_z` slot (not absolute partition index). The reduce kernel reads from `part_z = 0..grid_z-1` slots.

### 3.3 exp_sums at zero / max_logits at -inf

If the main kernel doesn't write exp_sums/max_logits, the reduce kernel produces zeros. Initialize sentinel values before kernel launch:
```python
exp_sums.fill_(-999.0)  # sentinel
# ... launch kernel ...
torch.cuda.synchronize()
print(f"exp_sums[0,0,0,:4] = {exp_sums[0,0,0,:4]}")  # should NOT be -999
```

## 4. Debugging Large Mismatch (>50%)

### 4.1 Missing partitions

If `grid_z < total_partitions` and the kernel processes only ONE partition per CTA (no loop), most of the context is skipped. Verify:
```python
total_parts = math.ceil(context_len / KV_COMPUTE_BLOCK)
print(f"grid_z={grid_z}, total_parts={total_parts}")
assert grid_z == total_parts or kernel_has_multi_partition_loop
```

### 4.2 All-1s isolation test

Fill query, key_cache, value_cache with 1.0 to eliminate data-dependent bugs:
```python
query.fill_(1.0)
key_cache.fill_(1.0)
value_cache.fill_(1.0)
```
With uniform input: all softmax probs are equal, PV output = 1.0. Any deviation reveals layout/addressing bugs.

**Caveat**: All-1s test does NOT catch V/P operand misalignment (since uniform values produce correct results regardless of ordering).

### 4.3 Single-partition test

Force `max_context_partition_num=1` (one_shot mode) to bypass the reduce kernel and test the main kernel in isolation.

### 4.4 Compare against Gluon

Run both Gluon and FlyDSL on the same input and compare element-wise:
```python
torch.testing.assert_close(flydsl_output, gluon_output, atol=5e-3, rtol=5e-3)
```

## 5. Debugging Small Errors (1-5%)

### 5.1 FP8 probability requantization

FP8 PV MFMA introduces ~0.03 max error vs bf16 reference. This is inherent to the FP8 data path and NOT a bug. Expected tolerance: `atol=5e-3`.

### 5.2 Per-tensor vs per-row quantization

If the reference uses per-row Q quantization but FlyDSL uses per-tensor, expect ~1-3% mismatch. Verify quantization mode matches.

### 5.3 Scale factor mismatch

Verify `_scale = softmax_scale * q_scale * k_scale` matches the reference. Common bug: applying v_scale twice (once in prob scaling, once after PV).

## 6. Compilation Errors

### 6.1 `range()` vs `range_constexpr()` inside @flyc.kernel

FlyDSL's AST rewriter converts runtime `range()` loops into MLIR loops. Use `range_constexpr()` for compile-time unrolled loops:
```python
# WRONG: i becomes an ArithValue, can't index Python lists
for i in range(4): result[i] = ...

# CORRECT: i is a Python int
for i in range_constexpr(4): result[i] = ...
```

### 6.2 Runtime vs compile-time conditionals

Current FlyDSL supports runtime comparisons in Python `if`; the AST rewriter lowers dynamic conditions to `scf.IfOp`. Prefer readable DSL operators for runtime SSA values:
```python
tid = gpu.thread_id("x")
lane = tid % fx.Index(64)
c_zero = fx.Index(0)
c_limit = fx.Index(8)

# Runtime condition, lowered to scf.IfOp
if lane == c_zero:
    fx.printf("lane zero")

# Runtime predicate for select
val = (lane < c_limit).select(good_val, zero_val)

```

Avoid spelling simple integer comparisons as `arith.cmpi(arith.CmpIPredicate.slt, lane, c_limit)` unless you are manually constructing low-level MLIR. If you pass a condition directly to `scf.IfOp`, unwrap the DSL boolean:
```python
cond = arith.unwrap(partition_idx >= visible_tile_count)
if_op = scf.IfOp(cond, has_else=False)
```

Use `const_expr(...)` only for compile-time decisions:
```python
if const_expr(trans_v):
    ...
```

Do not use `const_expr(lane == 0)`: even with `known_block_size`, `gpu.thread_id("x")`, `lane`, and `warp_id` are runtime SSA values. The compiler knows their range, not the current executing lane.

### 6.3 Loop-carried state packing

Prefer FlyDSL internal types (`fx.Int32`, `fx.Float32`, `Vector`, `ArithValue`) for loop-carried state. Unwrap only when a low-level helper explicitly requires raw `ir.Value`:
```python
def _unwrap(v):
    return v.ir_value() if hasattr(v, 'ir_value') else v

init_state = [_unwrap(v) for v in [val1, val2, vec_val]]
```

Supported state types: `f32` (scalar), vector values, `i32`, `i64`, `index`.

### 6.4 buffer_load type mismatch

`buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=T.i32)` — the offset is in units of `dtype`. For FP8 data addressed in bytes, divide by element size:
```python
k_addr_bytes = ...  # address in FP8 elements (= bytes for FP8)
k_4xi32 = buffer_ops.buffer_load(k_rsrc, k_addr_bytes // 4, vec_width=4, dtype=T.i32)
```

### 6.5 Vector stores require vector values

`Vector.store` requires the value to be a vector, not scalar:
```python
# WRONG
Vec(scalar_i32).store(lds_ptr, [idx])

# CORRECT
vec = Vec.from_elements([scalar_i32], fx.Int32)
vec.store(lds_ptr, [idx])
```

## 7. GPU Hang

### 7.1 Infinite runtime loop

If loop bounds are wrong (`stop < start` with unsigned comparison issues, or `step=0`), the GPU hangs. Verify bounds on host:
```python
print(f"loop: start={part_start}, stop={part_end}, step={cpb}")
```

### 7.2 Barrier deadlock

`gpu.barrier()` requires ALL threads in the workgroup to reach it. If some threads take a different branch (runtime `if`), the barrier deadlocks. FlyDSL doesn't support divergent barriers.

### 7.3 Recovery from GPU hang

```bash
# Check GPU state
rocm-smi
# If GPU shows 100% usage with no progress, reset:
sudo amdgpu-reset  # or reboot
```

## 8. Diagnostic Workflow

```
1. Clear caches (rm -rf ~/.flydsl)
2. Run with all-1s input → passes? Layout is OK, data issue
3. Run with single partition (one_shot) → passes? Multi-partition/reduce bug
4. Add host-side prints (tensor shapes, strides, NaN checks)
5. Compare intermediate buffers (exp_sums, max_logits, temp_out)
6. If layout bug suspected: trace one thread's addresses manually
   (tid=0: lane16id=0, rowid=0, warp_id=0)
7. For MFMA bugs: verify operand order (K is LHS, Q is RHS for QK)
```

## 9. Common Pitfalls Checklist

- [ ] Cleared `~/.flydsl` cache after code change
- [ ] `range_constexpr()` for all compile-time loops (not `range()`)
- [ ] No Python `if` on runtime GPU values
- [ ] `buffer_load` offset units match dtype (bytes/4 for i32)
- [ ] Vector stores use `Vector` values (not scalars)
- [ ] `range(..., init=...)` state uses internal types, unwrapped only at hard boundaries
- [ ] Output written to correct partition slot (`part_z`, not absolute index)
- [ ] `exp_sums`/`max_logits` strides match actual tensor layout
- [ ] Softmax guards against `-inf - (-inf) = NaN`
- [ ] Division by zero guarded (`select(sum > 0, sum, 1.0)`)
- [ ] K/V address calculation matches tensor layout (4D vs 5D trans_v)
- [ ] MFMA operand order: `mfma(LHS, RHS, acc)` — LHS→M, RHS→N
