# MoE Blockscale Stage1/Stage2 性能优化记录

## 概述

对 `kernels/moe_blockscale_2stage.py` 的 MoE blockscale FP8 GEMM stage1/stage2 kernel 进行了系统性优化，在保持 `waves_per_eu=2` 的前提下，从 **0.18x CK** 优化到 **1.37x 超越 CK**。

**测试配置**: M=16384, dim=7168, inter_dim=2048, E=32, topk=8, tile=64×128×128 (gfx950/MI350)

| 阶段 | stage1 (us) | stage2 (us) | total (us) | vs CK |
|---|---|---|---|---|
| 优化前 | 38646 | 4495 | 43141 | 0.27x |
| 最终 | 5421 | 3568 | 8996 | **1.37x** |
| CK | 7042 | 5234 | 12274 | 1.0x |

---

## 优化1: 逐(m,n)复用 block_acc + vec4 FMA

**问题**: `compute_tile_bs_s1` 一次性分配所有 block accumulator（`block_gate_accs = [acc_init] * (m_repeat * num_acc_n)` = 8 vec4 gate + 8 vec4 up = **64 VGPRs**），然后通过 `_do_scale_fma` 逐元素 extract → scalar FMA → from_elements 更新全局 accumulator。

**ISA 表现**: 177 VGPR spill, 596 bytes scratch, 229 条 scratch 指令。

**CK 做法**: 只用 **1 个** `c_thread_buf_per_scale`（4 VGPRs），逐 (m, n) 处理：MFMA → 立即 scale-FMA → 复用 block_acc。

**修复**:
```python
# 之前: 分配全部 block_accs, 最后统一 _do_scale_fma
block_gate_accs = [acc_init] * (num_acc_n * m_repeat)  # 64 VGPRs!
for mi: for ni: block_gate_accs[idx] = mfma(...)
_do_scale_fma(block_gate_accs, ...)  # extract+scalar_fma+from_elements

# 之后: 逐(mi,ni)复用, 立即 vec4 FMA
for mi:
    for ni:
        blk_g = mfma(a, b_gate, acc_init)  # 4 VGPRs, 复用
        blk_u = mfma(a, b_up, acc_init)
        current_gate[idx] = math.fma(blk_g, combined_vec4, current_gate[idx])
        current_up[idx] = math.fma(blk_u, combined_vec4, current_up[idx])
```

同时 `load_scales_s1` 的 combined_scale 从 4 个标量列表改为 `vector.from_elements(vec4_f32, ...)` 打包的 vec4。

**效果**: 177 → 63 VGPR spill, stage1 38646 → 12199 us (3.17x 提速)

---

## 优化2: B tiles 和 a0_prefetch 从 loop-carried state 移出

**问题**: `scf.for` 的 loop-carried state 有 34 个值（16 acc + 16 B tile + 2 a0_prefetch = 100 VGPRs），远超 CK 的 `do-while`（0 个 phi node）。

**修复**: 将 `do_one_stage` 函数内部化 B tile 加载和 scale 加载，loop state 只保留 16 个 accumulator（64 VGPRs）。

```python
# 之前: 34 loop-carried values
for k_iv, inner in range(..., init=[acc_gate, acc_up, b_gate, b_up, a0_pf]):
    ...
    yield [acc_gate, acc_up, b_gate_next, b_up_next, a0_pf_new]

# 之后: 16 loop-carried values
for k_iv, inner in range(..., init=list(acc_gate) + list(acc_up)):
    acc_gate_s0, acc_up_s0 = do_one_stage(...)  # B/scale在内部加载
    yield list(acc_gate_s1) + list(acc_up_s1)
```

**效果**: 63 → 15 VGPR spill (全在 prologue/epilogue, 热循环零 scratch)

---

## 优化3: 消除 scale validity cndmask

**问题**: `load_scales_s1` 中 `arith.select(t_valid, s_a_val, 0.0)` 在热循环每个 K-tile 产生 16 条 `v_cndmask_b32`。但 buffer_load OOB 自动返回 0，这些 select 完全多余。

**修复**: 移除 `arith.select` 和对应的 `_pre_t_valid` 数组。prologue 保留 `select(valid, id, 0)` 确保 scale index 不越界。

```python
# 之前
s_a_val = buffer_ops.buffer_load(sx_rsrc, sa_idx, ...)
s_a_val = arith.select(t_valid, s_a_val, fx.Float32(0.0))  # 多余!

# 之后
s_a_val = buffer_ops.buffer_load(sx_rsrc, sa_idx, ...)  # OOB返回0
```

**效果**: 热循环 cndmask 32 → 0, SGPR 72 → 50, stage1 7331 → 6033 us (**超越 CK 17%**)

---

## 优化4: Split scale 计算 (blk × s_a + fma(tmp, s_w_bc))

**问题**: 预计算 `combined = s_a_vec4 × broadcast(s_w)` 再 `fma(blk, combined, acc)` 产生大量 v_mul + 不完全 packed 的 v_fma（82 条标量 fma vs 23 条 pk_fma），且 combined 存活时间长导致 s_nop 增多。

**修复**: 拆分为 `blk × s_a` + `fma(tmp, s_w_broadcast, acc)`，让 s_w 以 broadcast 标量形式进入 fma，编译器更容易使用 `v_pk_fma op_sel_hi:[1,0]`。

```python
# 之前: 预计算 combined_scale
combined = s_a_vec4 * broadcast(s_w)
acc = fma(blk, combined, acc)  # 128 scalar muls + 128 scalar fmas

# 之后: 分离 s_a 和 s_w
tmp = blk * s_a_vec4           # MFMA结果直接乘s_a
acc = fma(tmp, broadcast(s_w), acc)  # s_w做broadcast FMA
```

**效果**: s_nop 49 → 23, 热循环 423 → 397 行, stage1 6033 → 5682 us (**超越 CK 24%**)

---

## 优化5: Stage2 tile_n 加倍

**问题**: Stage2 只有单 side（无 gate+up），用 tile_n=128 时 CTA 粒度过细。

**修复**: Stage2 使用 tile_n=256（num_acc_n=4），增加每个 CTA 的计算密度。通过 `--tile_n2` 参数独立配置。

**效果**: stage2 4583 → 3517 us (**超越 CK 49%**)

---

## 优化6: scale_w 去重 (ScaleBlockN 内共享)

**问题**: 同一 wave 内 `num_acc_n` 个 ni 的 scale_w 地址在 `ScaleBlockN=128` 下指向同一 N-block，导致重复 buffer_load。Stage1 有 gate+up 翻倍，更浪费。

**修复**: 编译期判断 `n_per_wave <= 128`，若成立则 ni>0 复用 ni=0 的 scale_w，不再重复加载。

```python
_sw_shared_n = (n_per_wave <= 128)
for ni in range_constexpr(num_acc_n):
    if ni == 0 or not _sw_shared_n:
        s_w_gate = buffer_ops.buffer_load(sw_rsrc, sw_gate_idx, ...)
    s_w_gate_vals.append(s_w_gate)  # ni>0 复用 ni=0 的值
```

**效果**: stage1 热循环 scale_w buffer_load 8 → 4，stage1 5682 → 5421 us (**超越 CK 30%**)

---

## ISA 指标演进 (Stage1)

| 指标 | 优化前 | 优化1 | 优化2 | 优化3 | 优化4 | 优化6 |
|---|---|---|---|---|---|---|
| VGPR spill | 177 | 63 | 15 | 14 | 0 | 14 |
| scratch (bytes) | 596 | 256 | 64 | 60 | 0 | 60 |
| 热循环 scratch | 229 | ~50 | 0 | 0 | 0 | 0 |
| 热循环 cndmask | 32 | 32 | 32 | 0 | 0 | 0 |
| 热循环 s_nop | ~49 | ~49 | ~49 | ~49 | 23 | 23 |
| 热循环 buf_load_dword | 40 | 40 | 40 | 40 | 40 | 36 |
| 热循环行数 | ~480 | ~450 | ~423 | ~423 | 397 | 399 |

## 尝试过但未采纳的方案

1. **per-group scale_a** (每 MRepeat 组只加载1个 scale_a，broadcast 给4元素): err=0.0875，精度不可接受。CK 的 `Scale_Block_M=1` 确认需要 per-element。

2. **ds_bpermute 共享 scale_a**: 16个lane重复加载同一 scale_a，用 ds_bpermute 从 lane 0-3 广播。但 ds_bpermute 延迟 (~20cy) 在关键路径上，stage1 5421 → 6061 us 反而更慢。

3. **Async LDS DMA (raw_ptr_buffer_load_lds)**: 替代 buffer_load_dwordx4 → ds_write 的 X tile 路径。MoE 的 gather 访问模式（每thread不同 token row）下，DMA 比 texture-cached buffer_load 更慢，stage1 5421 → 6093 us。该优化仅适用于线性/stride 访问（如 preshuffle_gemm 的 A tile）。

## 关键设计原则

1. **逐(m,n)复用 block_acc**: 对齐 CK 的 `c_thread_buf_per_scale` 模式，每个 (mi, ni) 只用 1 对 vec4 block_acc（8 VGPRs），MFMA 后立即 scale-FMA，不累积全量。

2. **最小化 loop-carried state**: 只保留 accumulator 在 scf.for 循环中，B tile 和 scale 在循环体内部加载。

3. **利用 buffer OOB**: AMD GPU 的 buffer_load OOB 返回 0，无需显式 validity masking。

4. **分离 s_a 和 s_w 乘法**: `blk × s_a` + `fma(tmp, broadcast(s_w), acc)` 比预计算 `combined = s_a × s_w` 更利于编译器 pack 和 MFMA 延迟隐藏。

5. **Stage-specific tile 配置**: Stage1（gate+up 双 side）用 tile_n=128，Stage2（单 side）用 tile_n=256。

6. **scale_w 去重**: 同一 ScaleBlockN 内的 ni 共享 weight scale，编译期判断避免重复 buffer_load。
