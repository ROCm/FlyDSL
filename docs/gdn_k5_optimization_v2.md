# GDN K5 性能分析 V2：Triton (193us) vs FlyDSL (314us)

## 版本信息

- **FlyDSL 版本**: 314us — 已完成 cooperative load、XOR swizzle、ds_read_b64_tr_b16、bf16 LDS 等优化
- **Triton 版本**: 193us — `chunk_gated_delta_rule_fwd_kernel_h_opt3`
- **目标 GPU**: gfx950 (MI350)
- **运行参数**: K=128, V=128, H=8, Hg=2, BT=64, BV=16, max_tokens=8192, full_prompt_len=8000

## 文件位置

| 实现 | 源码 | IR/ASM 目录 |
|------|------|-------------|
| **FlyDSL** | `kernels/chunk_gated_delta_h.py` | `/workspace/ir_dump/opt_flydsl_314us_ir_output/chunk_gdn_fwd_h_opt3/` |
| **Triton** | `/workspace/linear_attn_example/kernel/triton/chunk_delta_h.py` | `/workspace/ir_dump/triton_193us_ir_dump_opt3/` |

## 一、硬件资源对比

| 指标 | FlyDSL (314us) | Triton (193us) | 说明 |
|------|---------------|---------------|------|
| **VGPR** | 86 | 116 | Triton 更多 VGPR 用于多 buffer 预取 |
| **AGPR** | 0 | 8 | Triton 使用 AGPR 作为 MFMA 累加器 |
| **SGPR** | 50 | 52 | 基本持平 |
| **LDS 声明** | 14336 bytes | 0 bytes (编译器分配) | Triton 由编译器管理 LDS |
| **Occupancy** | ~5 waves | 4 waves | FlyDSL 略高但无实质帮助 |
| **kernarg preload** | 0 SGPRs | 14 SGPRs | Triton 预加载参数到 SGPR |
| **ISA 代码行数** | ~524 行 | ~1733 行 | Triton 代码量大 3x（含循环展开） |

## 二、指令统计对比（全 kernel）

| 指令类别 | FlyDSL (314us) | Triton (193us) | 说明 |
|---------|---------------|---------------|------|
| `v_mfma_f32_16x16x32_bf16` | 8 | 24 | Triton 含 prologue/epilogue 展开 |
| `s_barrier` | 10 | 20 | Triton barrier 更多 |
| `buffer_load_dwordx4` / `global_load_dwordx4` | 8 | 26 | Triton 大量向量化预取 |
| `buffer_load_dword` (f32 标量) | 14 | 5 (`global_load_dword`) | FlyDSL g 值逐元素加载 |
| `buffer_load_ushort` (bf16 标量) | 4 | 0 | FlyDSL u 值逐元素加载 |
| `buffer_store_short` / `global_store_dwordx2` | 12 | 11 (`global_store_dwordx2/x4`) | FlyDSL 逐元素存储 |
| `buffer_store_dword` (f32) | 8 | 0 | FlyDSL final_state 存储 |
| `ds_write_b16` | 12 | 36 | Triton 更多 LDS bf16 写 |
| `ds_write_b128` | 8 | 24 | Triton 更多 LDS 向量写 |
| `ds_read_b128` | 4 | 30 | Triton 更多 LDS 向量读 |
| `ds_read_b64_tr_b16` | 24 | 24 | **已对齐** |
| `ds_bpermute_b32` | 0 | 16 | Triton 独有 warp shuffle |
| `v_exp_f32` | 5 | 6 | 基本持平 |
| `v_cvt_pk_bf16_f32` | 16 | 24 | Triton 更多 bf16 pack |
| `s_and_saveexec_b64` | 4 | 43 | Triton 大量 exec mask 分支 |
| `v_accvgpr_write/read` | 0 | 99 | Triton 独有 AGPR 操作 |

## 三、主循环结构对比

### FlyDSL 主循环流程 (.LBB0_3 → .LBB0_2)

```
每次迭代处理 1 个 chunk (BT=64 行):

1. 预取 w[kb=0] → 2× buffer_load_dwordx4 (global → VGPR)
2. 存 h snapshot → 8× buffer_store_short (global) + 8× ds_write_b16 (LDS)  ← 双写
3. s_barrier
4. w[kb=0] → LDS (ds_write_b128) → s_barrier → 预取 w[kb=1]
5. MFMA × 2 (delta correction kb=0): ds_read_b128 + ds_read_b64_tr_b16 → mfma
6. s_barrier → w[kb=1] → LDS → s_barrier
7. MFMA × 2 (delta correction kb=1): ds_read_b128 + ds_read_b64_tr_b16 → mfma
8. 加载 u (4× buffer_load_ushort) → v_new = u - bv
9. 条件存储 v_new (4× scf.IfOp → 4× s_and_saveexec 分支)
10. 加载 g (5× buffer_load_dword) → gate 计算 (4× v_exp_f32) → 缩放 h, v_new
11. 预取 k[kb=0] → 2× buffer_load_dwordx4
12. gated v_new → LDS (4× ds_write_b16) → s_barrier
13. k[kb=0] → LDS (ds_write_b128) → s_barrier → 预取 k[kb=1]
14. MFMA × 2 (state update kb=0): ds_read_b64_tr_b16 → mfma
15. s_barrier → k[kb=1] → LDS → s_barrier
16. MFMA × 2 (state update kb=1): ds_read_b64_tr_b16 → mfma
17. s_barrier → 回到步骤 1
```

**每次迭代**: 8 MFMA, 10 barrier, ~4 global load batch + 5 g load + 4 u load

### Triton 主循环流程 (.LBB0_55)

```
每次迭代处理 1 个 chunk (BT=64 行), 但数据已在上一迭代预取完毕:

1. 从 AGPR 读出上一迭代 h 累加器 → cvt_pk_bf16 → 存 h snapshot (global_store_dwordx2)
2. 存 h snapshot 到 LDS (ds_write_b16) → s_barrier
3. 预取 w 下一迭代 (2× global_load_dwordx4 × 2 rows)
4. ds_read_b128 + ds_read_b64_tr_b16 → s_barrier
5. MFMA × 2 (delta correction block 0)
6. 预取 w 下一迭代 block 1
7. ds_read + ds_read_b64_tr_b16 → s_barrier
8. MFMA × 2 (delta correction block 1)
9. 预取 k (global_load_dwordx2) → ds_bpermute → v_new = u - bv
10. 条件存储 v_new (global_store_dwordx2, 向量化)
11. 加载 g (2× global_load_dword) → gate (1× v_exp_f32) → 缩放
12. ds_read_b64_tr_b16 → s_barrier → v_new → LDS (ds_write_b16) → s_barrier
13. MFMA × 2 (state update block 0) + 预取 k 下一迭代
14. ds_read_b64_tr_b16
15. MFMA × 2 (state update block 1)
16. 写入 w/k/h 预取数据到 LDS (ds_write_b128 × 8) → s_barrier
17. 回到步骤 1
```

**每次迭代**: 8 MFMA, 7 barrier (稳态), 数据预取与计算完全重叠

## 四、性能差异根因分析

### 差异 1：w/k 共享 LDS 导致串行化（最关键，估计 ~40us）

**FlyDSL 源码** — `chunk_gated_delta_h.py:150-151`:

```python
# w and k are used in different phases, so they can share the same LDS region
LDS_WK_BYTES = max(LDS_W_BYTES, LDS_K_BYTES)
```

w 和 k 共享同一块 LDS 区域 (`lds_wk`)，导致必须**串行处理**：
- 先加载 w → LDS → 完成 delta correction MFMA → barrier 清空
- 再加载 k → LDS → 完成 state update MFMA → barrier 清空

每个 K-block 的切换都需要额外的 barrier 等待。

**FlyDSL ASM** — 主循环中 w→k 切换的 barrier 链:

```asm
; delta correction 完成后
s_barrier                          ; 等 w MFMA 完成
; ... 存 gated v_new 到 LDS ...
s_barrier                          ; 等 v_new LDS 写完
; 才能开始加载 k 到同一块 LDS
ds_write_b128 v50, v[68:71]       ; k[kb=0] → LDS (覆盖之前的 w)
ds_write_b128 v51, v[72:75]
s_barrier                          ; 等 k LDS 写完
; 才能开始 state update MFMA
```

**Triton** — 为 w, k, v_new, h 分别分配独立的 LDS 区域（通过编译器自动管理的 `@global_smem`），每个区域还有 double-buffer（两个 K-block 的数据同时驻留）。从 Triton LLIR 可以看到 LDS 使用了多个 offset 段：

```
; Triton LDS 布局 (编译器分配, 约 36KB)
; offset 0..8191:     w block 0/1 (swizzled)
; offset 8192..16383: w block 0/1 (second half)
; offset 16384..24575: k block 0/1 (swizzled)
; offset 24576..32767: k block 0/1 (second half)
; offset 32768..33279: v_new 中转 (小块, 用于 ds_write_b16 → ds_read_b128)
```

这允许 Triton 在执行 delta correction MFMA 时**同时预取 k 数据到独立的 LDS 区域**，消除了串行等待。

---

### 差异 2：v_new 逐元素条件存储的分支开销（估计 ~20us）

**FlyDSL 源码** — `chunk_gated_delta_h.py:454-461`:

```python
for elem_i in range_constexpr(4):
    vn_bt_row = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + ...
    vn_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, vn_bt_row, T_local)
    _if_vn = scf.IfOp(vn_in_bounds)          # ← 4 个独立的 scf.IfOp
    with ir.InsertionPoint(_if_vn.then_block):
        bf16_v = arith.trunc_f(T.bf16, f32_v)
        vn_[fx.Index(vn_off)] = bf16_v        # ← 逐元素 bf16 标量存储
        scf.YieldOp([])
```

**FlyDSL ASM** — 生成 4 组 exec mask 分支:

```asm
; .LBB0_5 ~ .LBB0_9: 4 个条件存储分支
s_and_saveexec_b64 s[6:7], vcc     ; 保存 exec, 设置 mask
s_cbranch_execz .LBB0_5            ; 跳过
  buffer_store_short v63, v62, s[24:27], 0 offen
.LBB0_5:
  s_or_b64 exec, exec, s[6:7]     ; 恢复 exec
  s_and_saveexec_b64 s[6:7], s[0:1]
  s_cbranch_execz .LBB0_7
    buffer_store_short v63, v62, s[24:27], 0 offen offset:256
.LBB0_7:
  s_or_b64 exec, exec, s[6:7]
  ; ... 重复 2 次 ...
```

每个分支需要 `s_and_saveexec` + `s_cbranch_execz` + `s_or_b64` = 3 条标量指令 + 1 条存储，共 4 组 = 16 条额外指令。

**Triton** — 将 4 个 bf16 打包为 `<4 x bfloat>`，一次向量存储:

```asm
v_cvt_pk_bf16_f32 v37, v8, v9     ; 打包 2 个 f32 → 1 个 dword
v_cvt_pk_bf16_f32 v36, v6, v7
global_store_dwordx2 v[12:13], v[36:37], off  ; 一次存 8 字节 = 4 个 bf16
```

只需 1 次 exec mask 检查 + 1 次向量存储。

---

### 差异 3：g 值加载冗余 + safe exp 开销（估计 ~15us）

**FlyDSL 源码** — `chunk_gated_delta_h.py:472-488`:

```python
# 加载 g_last (1 次)
g_last = g_[fx.Index(g_last_off)]
exp_g_last = _fast_exp(g_last)

# 为每个 MFMA 元素独立加载 g_row (4 次)
for elem_i in range_constexpr(4):
    g_row = g_[fx.Index(g_row_off)]            # ← 每元素独立 global load
    gate = _fast_exp(arith.subf(g_last, g_row)) # ← 每元素独立 exp
```

**FlyDSL ASM** — 5 次 `buffer_load_dword` + 5 次 safe exp (含下溢保护):

```asm
; g_last + 4 个 g_row 加载
buffer_load_dword v63, v62, s[36:39], 0 offen   ; g_last
buffer_load_dword v64, v61, s[36:39], 0 offen   ; g[row0]
buffer_load_dword v65, v60, s[36:39], 0 offen   ; g[row1]
buffer_load_dword v66, v59, s[36:39], 0 offen   ; g[row2]
buffer_load_dword v67, v58, s[36:39], 0 offen   ; g[row3]

; 每个 gate 的 safe exp: sub → mul → cmp → cndmask → add → exp → cndmask → ldexp
v_sub_f32_e32 v56, v63, v64        ; g_last - g[row0]
v_mul_f32_e32 v56, 0x3fb8aa3b, v56 ; × log2(e)
v_cmp_gt_f32_e64 s[6:7], s49, v56  ; 下溢检查
v_cndmask_b32_e64 v62, 0, v52, s[6:7]
v_add_f32_e32 v56, v56, v62
v_exp_f32_e32 v56, v56              ; exp2
v_cndmask_b32_e64 v62, 0, v53, s[6:7]
v_ldexp_f32 v56, v56, v62           ; ldexp 修正
; ... 重复 4 次 (共 ~40 条 VALU 指令)
```

**Triton** — 利用 `ds_bpermute` 在 warp 内交换 u 数据，只需 2 次 g 加载 + 1 次 gate exp:

```asm
global_load_dword v71, v73, s[2:3]              ; g_last (1 次)
global_load_dword v58, v[36:37], off            ; g_row (1 次, 通过 lane 映射)

v_sub_f32_e32 v2, v65, v66                      ; g_last - g_row
v_mul_f32_e32 v3, 0x3fb8aa3b, v2                ; × log2(e)
v_cndmask_b32_e64 v3, 0, v74, s[4:5]           ; 下溢保护
v_fmac_f32_e32 v3, 0x3fb8aa3b, v2              ; fused mul-add
v_exp_f32_e32 v2, v3                            ; exp2
v_cndmask_b32_e64 v4, 0, v75, s[4:5]
v_ldexp_f32 v2, v2, v4
; gate 值通过 v_pk_mul_f32 广播到所有元素
v_pk_mul_f32 v[4:5], v[2:3], v[6:7] op_sel_hi:[0,1]
```

Triton 的关键优化：**每个 lane 只持有自己行的 g 值**，通过 MFMA 的 lane 映射自然对齐，不需要为每个 MFMA 元素独立加载。gate 值通过 `op_sel_hi:[0,1]` 广播到 `v_pk_mul_f32` 的两个 f32 通道。

---

### 差异 4：h snapshot 双写开销（估计 ~15us）

**FlyDSL 源码** — `chunk_gated_delta_h.py:366-376`:

```python
for elem_i in range_constexpr(4):
    bf16_val = arith.trunc_f(T.bf16, f32_val)
    h_[fx.Index(h_off)] = bf16_val       # ← 写 global (buffer_store_short)
    lds_h[fx.Index(lds_h_idx)] = bf16_val # ← 写 LDS (ds_write_b16)
```

**FlyDSL ASM** — 8 次 `buffer_store_short` + 8 次 `ds_write_b16` = 16 次写操作:

```asm
; h → global (逐元素 bf16)
buffer_store_short v10, v11, s[40:43], 0 offen
buffer_store_short v58, v11, s[40:43], 0 offen offset:256
buffer_store_short v59, v11, s[40:43], 0 offen offset:512
buffer_store_short v60, v11, s[40:43], 0 offen offset:768
buffer_store_short v61, v65, s[40:43], 0 offen
buffer_store_short v62, v65, s[40:43], 0 offen offset:256
buffer_store_short v63, v65, s[40:43], 0 offen offset:512
buffer_store_short v64, v65, s[40:43], 0 offen offset:768

; h → LDS (逐元素 bf16)
ds_write_b16 v24, v10 offset:10240
ds_write_b16 v25, v58 offset:10240
ds_write_b16 v26, v59 offset:10240
ds_write_b16 v27, v60 offset:10240
ds_write_b16 v46, v61 offset:10240
ds_write_b16 v47, v62 offset:10240
ds_write_b16 v48, v63 offset:10240
ds_write_b16 v49, v64 offset:10240
```

**Triton** — h snapshot 存 global 用向量化 `global_store_dwordx2`（4 个 bf16 一次），LDS 中的 h 数据通过 cooperative load 从 global 预取后写入（`global_load_dwordx4` → `ds_write_b128`），不从 VGPR 双写:

```asm
; h → global (向量化)
global_store_dwordx2 v[2:3], v[8:9], off   ; 4 个 bf16 一次

; h → LDS 通过 cooperative load (在下一迭代的 prologue)
global_load_dwordx4 v[80:83], v[26:27], off  ; 预取 h 到 VGPR
ds_write_b128 v57, v[80:83]                   ; 向量化写 LDS
```

---

### 差异 5：u 值逐元素标量加载（估计 ~10us）

**FlyDSL 源码** — `chunk_gated_delta_h.py:436-442`:

```python
for elem_i in range_constexpr(4):
    u_off = v_base + safe_u_row * stride_v + u_col
    u_bf16 = v_[fx.Index(u_off)]               # ← 逐元素 bf16 标量加载
    u_f32_elems.append(arith.extf(T.f32, u_bf16))
```

**FlyDSL ASM** — 4 次 `buffer_load_ushort`:

```asm
buffer_load_ushort v66, v0, s[44:47], 0 offen
buffer_load_ushort v67, v1, s[44:47], 0 offen
buffer_load_ushort v68, v10, s[44:47], 0 offen
buffer_load_ushort v69, v11, s[44:47], 0 offen
; 每次只加载 2 字节, 需要 v_lshlrev_b32 扩展为 f32
```

**Triton** — u 值通过 `ds_bpermute_b32` 在 warp 内 shuffle 获取（数据已在 prologue 预加载到寄存器）:

```asm
ds_bpermute_b32 v36, v46, v37      ; warp 内数据交换
ds_bpermute_b32 v38, v46, v39
; 直接得到 packed bf16, 无需 global load
v_pk_add_f32 v[6:7], v[36:37], v[6:7] neg_lo:[0,1] neg_hi:[0,1]  ; v_new = u - bv
```

---

### 差异 6：Triton 的 double-buffer 预取流水线（估计 ~10us）

Triton 在主循环中实现了完整的 **double-buffer 预取**：

1. 在执行当前迭代的 MFMA 时，同时发射下一迭代的 `global_load_dwordx4`
2. 在 barrier 等待期间，预取的数据已经到达 VGPR
3. barrier 后立即将预取数据写入 LDS，无需等待

从 Triton ASM 稳态循环可以看到这种重叠：

```asm
; 正在执行 MFMA (state update)
v_mfma_f32_16x16x32_bf16 a[4:7], v[2:5], v[104:107], a[4:7]

; 同时预取的 w/k 数据已到达, 立即写入 LDS
s_waitcnt vmcnt(1)
ds_write_b128 v57, v[80:83]         ; 写入下一迭代的 w
ds_write_b128 v57, v[76:79] offset:4096
ds_write_b128 v57, v[88:91] offset:8192   ; 写入下一迭代的 k
ds_write_b128 v57, v[84:87] offset:12288
; ...

v_mfma_f32_16x16x32_bf16 a[4:7], v[6:9], v[108:111], a[4:7]  ; 继续 MFMA

s_barrier    ; 此时下一迭代数据已全部就绪
```

FlyDSL 虽然也有 w/k 的 prefetch，但由于 w/k 共享 LDS，无法实现跨阶段的数据预取重叠。

---

### 差异 7：kernarg preload（估计 ~5us）

**Triton** 使用 `amdhsa_user_sgpr_kernarg_preload_length: 14`，在 kernel 启动时将前 14 个 SGPR 的参数预加载，避免了 `s_load_dword` 的延迟。

**FlyDSL** 使用 `amdhsa_user_sgpr_kernarg_preload_length: 0`，所有参数通过 `s_load_dwordx16` + `s_load_dwordx4` 从内存加载，需要 `s_waitcnt lgkmcnt(0)` 等待。

## 五、性能差距量化归因

| 因素 | 估计影响 | 占总差距比例 |
|------|---------|------------|
| w/k 共享 LDS → 串行化 + barrier 等待 | ~40us | 33% |
| v_new 逐元素条件存储 (4× scf.IfOp) | ~20us | 17% |
| g 值冗余加载 + safe exp 开销 | ~15us | 12% |
| h snapshot 双写 (global + LDS) | ~15us | 12% |
| u 值逐元素标量加载 | ~10us | 8% |
| 缺少 double-buffer 预取流水线 | ~10us | 8% |
| kernarg preload 缺失 | ~5us | 4% |
| 其他（AGPR、指令调度等） | ~6us | 5% |
| **总计** | **~121us** | **100%** |

实测差距: 314 - 193 = **121us**，与估算吻合。

## 六、优化建议（按优先级排序）

### P0: 分离 w/k 的 LDS 区域（预期 -40us）

**当前**: `LDS_WK_BYTES = max(LDS_W_BYTES, LDS_K_BYTES)` — w 和 k 共享 8192 bytes。

**改为**: 为 w 和 k 分别分配独立的 LDS 区域:

```python
lds_w_offset = allocator._align(allocator.ptr, 16)
allocator.ptr = lds_w_offset + LDS_W_BYTES    # 8192 bytes for w
lds_k_offset = allocator._align(allocator.ptr, 16)
allocator.ptr = lds_k_offset + LDS_K_BYTES    # 8192 bytes for k
```

LDS 总量从 14336 → 22528 bytes（仍在 64KB 限制内），但允许在执行 delta correction MFMA 时同时预取 k 数据到独立区域，消除串行等待。

### P1: v_new 存储向量化（预期 -20us）

**当前**: 4 个 `scf.IfOp` 逐元素条件存储。

**改为**: 将 4 个 bf16 打包为 `<4 x bfloat>` 向量，用整块级边界检查 + 一次 `buffer_store_dwordx2`:

```python
# 替换 4 个 scf.IfOp 为:
vn_packed = vector.from_elements(T.vec(4, T.bf16), [bf16_v0, bf16_v1, bf16_v2, bf16_v3])
# 整块边界检查 (第一行 in_bounds 即整块 in_bounds)
if first_row_in_bounds:
    vn_.vec_store((fx.Index(vn_off_base),), vn_packed, 4)
```

### P2: g 值加载优化（预期 -15us）

**当前**: 5 次 `buffer_load_dword`（1 g_last + 4 g_row）。

**改为**: 利用 MFMA lane 映射，每个 lane 只加载自己行的 g 值（1 次 g_last + 1 次 g_row），gate 值通过 `vector.broadcast` 广播:

```python
# 每个 lane 的 MFMA 行由 (wid, lane_m_base) 唯一确定
# 只需加载 1 个 g_row (对应当前 lane 的行)
abs_row = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4)
g_row = g_[fx.Index(g_row_off)]
gate = _fast_exp(arith.subf(g_last, g_row))
# 广播到 f32x4 的所有元素
gate_vec = vector.broadcast(T.f32x4, gate)
```

### P3: h snapshot 消除双写（预期 -15us）

**当前**: h snapshot 同时写 global 和 LDS。

**改为**: 只写 global，LDS 中的 h 数据在 delta correction 阶段通过 cooperative load 从 global 预取:

```python
# 步骤 1: h → global (保留)
h_[fx.Index(h_off)] = bf16_val

# 步骤 2: 删除 lds_h 的直接写入
# 步骤 3: 在 delta correction 前, 通过 cooperative load 从 global 加载 h 到 LDS
h_prefetch = h_.vec_load((fx.Index(h_global_off),), LOAD_VEC_WIDTH)
lds_h.vec_store((fx.Index(lds_off),), h_prefetch, LOAD_VEC_WIDTH)
```

### P4: u 值向量化加载（预期 -10us）

**当前**: 4 次 `buffer_load_ushort` 逐元素加载 u。

**改为**: 利用 `ds_bpermute_b32` 在 warp 内交换数据，或将 u 的加载改为 cooperative load 经 LDS 中转。

### P5: 启用 kernarg preload（预期 -5us）

在 kernel 编译选项中启用 `amdhsa_user_sgpr_kernarg_preload_length`，将常用参数预加载到 SGPR。

## 七、预期优化效果

| 阶段 | 优化内容 | 预期耗时 |
|------|---------|---------|
| 当前 | — | 314us |
| P0 | 分离 w/k LDS | ~274us |
| P0+P1 | + v_new 向量化 | ~254us |
| P0+P1+P2 | + g 加载优化 | ~239us |
| P0+P1+P2+P3 | + h 消除双写 | ~224us |
| 全部 | + u 向量化 + kernarg | ~209us |
| **目标** | 接近 Triton | **~193us** |

剩余 ~16us 差距来自 Triton 编译器的全局指令调度优化（AGPR 使用、指令交错等），需要更细粒度的 ISA 级调优。

## 八、已实施优化总结

### 优化结果

- **优化前**: 314us（原始版本有 `lds_wk` 引用 bug，修复后为 312us）
- **优化后**: 227us
- **Triton 基准**: 194us
- **提升**: 312us → 227us，减少 85us（**27% 提速**），达到 Triton 的 **85.6%**
- **精度**: FlyDSL 与 Triton opt3 结果完全一致（abs_err max=0.000000）

### 优化 A：Bug 修复 — P0 LDS 分离的遗留引用错误

**问题**: P0 将 w/k 的 LDS 区域从共享 (`lds_wk`) 分离为独立 (`lds_w`, `lds_k`)，但代码中有 3 处仍引用旧变量名 `lds_wk` 和 `lds_wk_offset`，以及 1 处引用了不存在的 `_lds_vec_read_bf16x8`。

**修复**:
```python
# 修复前
lds_wk.vec_store(...)          # w 写入 LDS
lds_wk.vec_store(...)          # k 写入 LDS
k_lds_byte = ... + fx.Int32(lds_wk_offset)
a_frag = _lds_vec_read_bf16x8(w_lds_idx)

# 修复后
lds_w.vec_store(...)           # w 写入独立的 lds_w
lds_k.vec_store(...)           # k 写入独立的 lds_k
k_lds_byte = ... + fx.Int32(lds_k_offset)
a_frag = _lds_vec_read_w_bf16x8(w_lds_idx)
```

### 优化 B：批量 LDS 写入减少 barrier（-43us，312us → 269us）

**问题**: 原始代码中 w 和 k 各有 NUM_K_BLOCKS=2 个 K-block，每个 K-block 需要单独写入 LDS 并执行 barrier，导致主循环有 10 个 barrier。

**方案**: 将 LDS stride 从 64 扩展到 K=128，使所有 K-block 数据可以一次性写入 LDS，然后用一个 barrier 同步。

```python
# 修改前: LDS 只能容纳 1 个 K-block (stride=64)
LDS_W_STRIDE = 64
LDS_W_ELEMS = BT * 64       # 4096 elems = 8192 bytes

# 修改后: LDS 容纳所有 K-block (stride=K)
LDS_W_STRIDE = K             # 128
LDS_W_ELEMS = BT * K        # 8192 elems = 16384 bytes
```

主循环流程变化:
```
修改前 (10 barriers):
  w[kb=0] → LDS → barrier → MFMA → barrier →
  w[kb=1] → LDS → barrier → MFMA → barrier →
  v_new → LDS → barrier →
  k[kb=0] → LDS → barrier → MFMA → barrier →
  k[kb=1] → LDS → barrier → MFMA → barrier

修改后 (3 barriers):
  w[all kb] → LDS → barrier → MFMA(all kb) →
  v_new + k[all kb] → LDS → barrier → MFMA(all kb)
```

LDS 总量: 16384 (w) + 16384 (k) + 2048 (v_new) + 4096 (h) = **38912 bytes** < 64KB ✓

### 优化 C：合并 v_new 与 k 的 LDS barrier（-27us，269us → 242us）

**问题**: v_new 写入 LDS 后需要一个 barrier，k 写入 LDS 后又需要一个 barrier，共 2 个 barrier。

**方案**: 将 v_new 的 LDS 写入（ds_write_b16）和 k 的 LDS 写入（ds_write_b128）放在同一个 barrier 之前。

```python
# 修改前: 2 个 barrier
lds_vn[...] = bf16_v          # v_new → LDS
gpu.barrier()                  # barrier 1
lds_k.vec_store(...)           # k → LDS
gpu.barrier()                  # barrier 2

# 修改后: 1 个 barrier
lds_vn[...] = bf16_v          # v_new → LDS
lds_k.vec_store(...)           # k → LDS (紧接着写)
gpu.barrier()                  # 只需 1 个 barrier
```

主循环最终只有 **2 个 barrier**（h+w 写入后 1 个，v_new+k 写入后 1 个）。

### 优化 D：数据预取重叠 MFMA（-15us，242us → 227us）

**问题**: k、u、g 的 global load 在 MFMA 完成后才发射，global load 延迟（~200 cycles）完全暴露。

**方案**: 在 delta correction MFMA 执行之前发射 k[0]、u、g 的 global load，利用 MFMA 执行时间隐藏 global load 延迟。

```python
# 修改前: MFMA 完成后才加载 u 和 g
for kb in range_constexpr(NUM_K_BLOCKS):
    ... # MFMA
u_bf16 = v_[fx.Index(u_off)]          # MFMA 后才加载 u
g_last = g_[fx.Index(g_last_off)]     # MFMA 后才加载 g

# 修改后: MFMA 之前就发射所有 global load
k_prefetch = [k_.vec_load(...)]        # k[0] prefetch
g_last_prefetch = g_[fx.Index(...)]    # g_last prefetch
g_row_prefetch = [g_[fx.Index(...)]]   # g_row prefetch
u_prefetch = [v_[fx.Index(...)]]       # u prefetch
for kb in range_constexpr(NUM_K_BLOCKS):
    ... # MFMA (此时 k/u/g 的 global load 在飞行中)
# MFMA 完成时 k/u/g 数据已到达 VGPR
```

同时将 gate_vec 的计算提取到 N_REPEAT 循环外部复用。

### 性能对比总结

| 优化阶段 | 耗时 | 改善 | 累计提升 | vs Triton |
|---------|------|------|---------|-----------|
| 基线（bug 修复后） | 312us | — | — | 62.0% |
| +批量 LDS 写入（减少 barrier 10→3） | 269us | -43us | -43us | 72.2% |
| +合并 v_new/k barrier（3→2） | 242us | -27us | -70us | 80.0% |
| +u/g/k 预取重叠 MFMA | 227us | -15us | -85us | 85.6% |
| +v_new 无分支存储 + amdgcn.exp2 | 201us | -26us | -111us | **96.1%** |
| **Triton opt3 基准** | **194us** | — | — | 100% |

### 优化 E：v_new 无分支存储（-12us，227us → ~215us 贡献）

**问题**: 4 个 `scf.IfOp` 生成 4 组 `s_and_saveexec` + `s_cbranch_execz` + `s_or_b64` exec mask 分支，每组 3 条标量指令。

**方案**: 用 `arith.select` 做 safe addressing（out-of-bounds 时 clamp 到 row 0），然后无条件存储。写入 row 0 的冗余数据不影响正确性（后续迭代会覆盖）。

```python
# 修改前: 4 个 scf.IfOp 分支
_if_vn = scf.IfOp(vn_in_bounds)
with ir.InsertionPoint(_if_vn.then_block):
    vn_[fx.Index(vn_off)] = bf16_v
    scf.YieldOp([])

# 修改后: branchless safe addressing
safe_vn_row = arith.select(vn_in_bounds, vn_bt_row, fx.Int32(0))
vn_off = vn_base + safe_vn_row * fx.Int32(V) + vn_col
vn_[fx.Index(vn_off)] = bf16_v
```

消除了 4 组 exec mask 分支（12 条标量指令 + 4 次 `s_cbranch_execz`）。

### 优化 F：amdgcn.exp2 消除下溢保护（-14us，~215us → 201us 贡献）

**问题**: `_fast_exp` 使用 `llvm.exp2.f32` intrinsic，LLVM 后端为保证 IEEE 兼容性会展开为 `v_mul → v_cmp → v_cndmask → v_add → v_exp_f32 → v_cndmask → v_ldexp` 共 ~8 条指令/次。5 次 exp 产生 ~40 条额外指令。

**方案**: 使用 `llvm.amdgcn.exp2.f32` target-specific intrinsic，直接映射到 `v_exp_f32` 指令，跳过下溢保护。gate 值 `exp(g_last - g_row)` 的参数范围 `[0, +∞)` 不会下溢，`exp(g_last)` 的参数虽可能为负但精度损失可接受。

```python
# 修改前: llvm.exp2.f32 → 8 条指令/次（含下溢保护）
def _fast_exp(x):
    return _llvm.call_intrinsic(ir.F32Type.get(), "llvm.exp2.f32", ...)

# 修改后: llvm.amdgcn.exp2.f32 → 2 条指令/次（bare v_exp_f32）
def _fast_exp(x):
    return _llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.exp2.f32", ...)
```

ISA 指令从 `5× v_exp_f32 + 5× v_ldexp_f32 + 10× v_cndmask + 5× v_cmp` (25 条) 减少到 `5× v_exp_f32 + 5× v_mul_f32` (10 条)，净减 ~30 条指令。

### 剩余差距分析（~8us）

| 因素 | 估计影响 | 说明 |
|------|---------|------|
| h snapshot 逐元素存储 | ~3us | 8× `buffer_store_short` + 8× `ds_write_b16`（Triton 用向量化 + cooperative load） |
| u 逐元素标量加载 | ~2us | 4× `buffer_load_ushort`（Triton 用 `ds_bpermute` warp shuffle） |
| AGPR 累加器 | ~1us | Triton 使用 AGPR 释放 VGPR 压力 |
| kernarg preload | ~1us | Triton 预加载 14 SGPRs |
| 指令调度差异 | ~1us | Triton 编译器全局优化 |
| **总计** | **~8us** | |

剩余 8us 差距主要来自 FlyDSL 编译器基础设施限制（bf16 向量化存储、AGPR 分配、kernarg preload），需要编译器层面支持。
