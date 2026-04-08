# GDN K5 性能分析：Triton (193us) vs FlyDSL (279us)

## 原始 Kernel 代码位置

| 实现 | 文件路径 | 入口函数 |
|------|---------|----------|
| **Triton** | `/workspace/linear_attn_example/kernel/triton/chunk_delta_h.py:970` | `chunk_gated_delta_rule_fwd_kernel_h_opt3` |
| **FlyDSL** | `/workspace/FlyDSL/kernels/chunk_gated_delta_h.py:149` | `@flyc.kernel(name="chunk_gdn_fwd_h_opt3")` 内 `gdn_h_kernel` |
| **FlyDSL wrapper** | `/workspace/FlyDSL/kernels/chunk_gated_delta_h.py:520` | `chunk_gated_delta_rule_fwd_h_flydsl` |

## IR / ASM 文件位置

| 实现 | 目录 |
|------|------|
| Triton 193us | `/workspace/ir_dump/triton_193us_ir_dump_opt3/` |
| FlyDSL 279us | `/workspace/ir_dump/opt_flydsl_279us_ir_output/chunk_gdn_fwd_h_opt3/` |

## 关键指标对比

| 指标 | FlyDSL (279us) | Triton (193us) | 说明 |
|------|---------------|---------------|------|
| **VGPR** | 95 | 116+8 AGPR = 124 | Triton 用了 AGPR |
| **SGPR** | 78 | 52 | FlyDSL SGPR 压力更大 |
| **LDS 声明** | 8192 bytes | 0 bytes (编译器分配) | FlyDSL 显式 LDS |
| **Occupancy** | ~5 | 4 | 差异不大 |
| **MFMA 指令数** | 8 | 24 | **Triton 3x 多** |
| **Barrier 数** | 2 | 20 | Triton 10x 多 |
| **LDS 读写** | 53 | 130 | Triton LDS 操作更多 |
| **全局内存操作** | 75 (buffer_load/store) | 45 (global_load/store) | FlyDSL 更多 |
| **exec mask 分支** | 4 (s_and_saveexec) | 43 | Triton 大量分支 |
| **ds_read_b64_tr_b16** | 0 | 24 | **Triton 独有** |
| **v_accvgpr** | 0 | 99 | **Triton 独有** |
| **代码长度** | ~758 行 ISA | ~1733 行 ISA | Triton 代码大得多 |

## 性能差异根因分析

### 1. 数据加载向量化不足（最关键）

**FlyDSL** 使用 `buffer_load_ushort`（2B/次）逐元素加载 bf16 数据：

```asm
buffer_load_ushort v58, v71, s[36:39], 0 offen
buffer_load_ushort v59, v72, s[36:39], 0 offen
... (每次只加载 2 字节)
```

**Triton** 使用 `global_load_dwordx4`（16B/次）向量化加载：

```asm
global_load_dwordx4 v[2:5], v[2:3], off  ; 一次加载 16 字节 = 8 个 bf16
```

FlyDSL 需要约 32 次 ushort load 才能组装一个 MFMA 的 8xbf16 操作数，Triton 只需 1 次 dwordx4。

### 2. 缺少 `ds_read_b64_tr_b16` transpose read

Triton 利用了 gfx950 的 `ds_read_b64_tr_b16` 指令（24次），从 LDS 中一步完成读取+转置，直接生成 MFMA 操作数。

FlyDSL 需要 `ds_read2_b32` + `v_cvt_pk_bf16_f32` + `v_perm_b32` 多步组装。

### 3. LDS 中间数据用 f32 而非 bf16

FlyDSL 将 delta correction 结果以 f32 存入 LDS（占用 2x 空间和带宽），读出时还需额外的 f32→bf16 转换。Triton 直接以 bf16 存储。

### 4. MFMA 计算密度低

FlyDSL 每次循环迭代 4 个 MFMA（2 delta correction + 2 state update），Triton 8 个。Triton 的计算/访存比更高。

## 性能差异根因与源码/汇编对应关系

### 1. 数据加载向量化不足 → 源码/汇编定位

**FlyDSL 源码** — `kernels/chunk_gated_delta_h.py` 中 k/w 的逐元素标量加载：

```python
# chunk_gated_delta_h.py:431-437  (state update 阶段加载 k)
for ki in range_constexpr(8):
    k_t_row_raw = i_t_i32 * fx.Int32(BT) + fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(8) + fx.Int32(ki)
    k_row_valid = arith.cmpi(arith.CmpIPredicate.slt, k_t_row_raw, T_local)
    k_t_row = arith.select(k_row_valid, k_t_row_raw, fx.Int32(0))
    k_off = k_base + k_t_row * stride_k + k_col
    k_val = k_[fx.Index(k_off)]                    # ← 逐元素 bf16 标量加载
    k_a_elems.append(arith.select(k_row_valid, k_val, arith.constant(0.0, type=T.bf16)))
```

```python
# chunk_gated_delta_h.py:323-328  (delta correction 阶段加载 w)
w_bt_row_raw = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + lane_n
w_off = w_base + safe_w_row * stride_w + w_col
a_frag = w_.vec_load((fx.Index(w_off),), 8)         # ← vec_load(8) 但地址不连续
```

**FlyDSL 汇编** — `17_final_isa.s:345-435` 生成大量 `buffer_load_ushort`（2B/次）：

```asm
; 17_final_isa.s:345-352  (delta correction 加载 w)
buffer_load_ushort v58, v71, s[36:39], 0 offen
buffer_load_ushort v59, v72, s[36:39], 0 offen
buffer_load_ushort v61, v73, s[36:39], 0 offen
buffer_load_ushort v62, v74, s[36:39], 0 offen
buffer_load_ushort v63, v75, s[36:39], 0 offen
buffer_load_ushort v64, v1, s[36:39], 0 offen
buffer_load_ushort v71, v11, s[36:39], 0 offen
buffer_load_ushort v76, v57, s[36:39], 0 offen
; ... 共约 32 次 buffer_load_ushort 来组装两组 MFMA 的 8xbf16 操作数
```

**Triton 源码** — `chunk_delta_h.py:1075-1077` 使用 `tl.make_block_ptr` 块加载：

```python
# chunk_delta_h.py:1075-1077  (delta correction 加载 w)
p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
b_w = tl.load(p_w, boundary_check=(0, 1))           # ← 块加载整个 [BT, 64] tile
b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
```

```python
# chunk_delta_h.py:1131-1133  (state update 加载 k)
p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
b_k = tl.load(p_k, boundary_check=(0, 1))           # ← 块加载整个 [64, BT] tile
b_h1 += tl.dot(b_k, b_v)
```

**Triton 汇编** — `.amdgcn:79,186,208` 等处生成 `global_load_dwordx4`（16B/次）：

```asm
; .amdgcn:79   (initial state 加载)
global_load_dwordx4 v[2:5], v[2:3], off             ; 一次 16 字节 = 8 个 bf16

; .amdgcn:186  (w 块加载)
global_load_dwordx4 v[36:39], v[4:5], off

; .amdgcn:360  (k 块加载)
global_load_dwordx4 v[68:71], v[4:5], off
```

> **根因**：FlyDSL 的 `GTensor` 逐元素索引 `k_[fx.Index(k_off)]` 产生标量 `buffer_load_ushort`（2B），Triton 的 `tl.make_block_ptr` + `tl.load` 产生向量化 `global_load_dwordx4`（16B），带宽利用率差 **8x**。

---

### 2. 缺少 `ds_read_b64_tr_b16` → 源码/汇编定位

**FlyDSL 源码** — `chunk_gated_delta_h.py:330-339` 逐元素从 LDS 读取 bf16 组装 MFMA B 操作数：

```python
# chunk_gated_delta_h.py:330-339  (delta correction 从 LDS 读 h snapshot)
for nr in range_constexpr(N_REPEAT):
    b_elems = []
    for bi in range_constexpr(8):
        lds_r = fx.Int32(ks * WMMA_K) + lane_m_base * fx.Int32(8) + fx.Int32(bi)
        lds_c = fx.Int32(nr * 16) + lane_n
        lds_idx = lds_r * fx.Int32(BV) + lds_c
        b_elems.append(lds_h[fx.Index(lds_idx)])     # ← 逐元素 bf16 LDS 读取
    b_frag = vector.from_elements(T.vec(8, T.bf16), b_elems)
    bv_accs[nr] = _mfma_bf16_16x16x32(a_frag, b_frag, bv_accs[nr])
```

**FlyDSL 汇编** — `17_final_isa.s:464-479` 使用 `ds_read2_b32` + `v_cvt_pk_bf16_f32` + `v_perm_b32` 多步组装：

```asm
; 17_final_isa.s:464-479  (从 LDS 读 h snapshot → 组装 MFMA B 操作数)
ds_read2_b32 v[10:11], v37 offset0:96 offset1:112   ; 读 f32 对
ds_read2_b32 v[60:61], v37 offset0:64 offset1:80
ds_read2_b32 v[64:65], v37 offset0:32 offset1:48
ds_read2_b32 v[66:67], v37 offset1:16
; ... waitcnt ...
v_cvt_pk_bf16_f32 v63, v10, v11                     ; f32 → bf16 pack
v_cvt_pk_bf16_f32 v62, v60, v61
v_cvt_pk_bf16_f32 v61, v64, v65
v_cvt_pk_bf16_f32 v60, v66, v67
; 然后才能送入 MFMA:
v_mfma_f32_16x16x32_bf16 v[6:9], v[56:59], v[60:63], v[6:9]
```

**Triton 汇编** — `.amdgcn:815-818,846` 使用 gfx950 的 `ds_read_b64_tr_b16` 一步完成读取+转置：

```asm
; .amdgcn:815-818  (从 LDS 读 k^T @ v_new 的 B 操作数)
ds_read_b64_tr_b16 v[92:93], v28 offset:16384       ; 一步读取+转置
ds_read_b64_tr_b16 v[94:95], v36 offset:512
ds_read_b64_tr_b16 v[96:97], v38 offset:4096
ds_read_b64_tr_b16 v[98:99], v36 offset:4608
; 直接作为 MFMA 操作数:
; .amdgcn:846
v_mfma_f32_16x16x32_bf16 a[0:3], v[2:5], v[92:95], a[0:3]
```

> **根因**：Triton 利用 gfx950 专有 `ds_read_b64_tr_b16` 一步完成 LDS 读取+转置直接生成 MFMA 操作数，FlyDSL 需要 `ds_read2_b32` → `v_cvt_pk_bf16_f32` → `v_perm_b32` 三步，额外消耗大量指令槽和延迟。

---

### 3. LDS 中间数据用 f32 而非 bf16 → 源码/汇编定位

**FlyDSL 源码** — `chunk_gated_delta_h.py:190-196` 声明 LDS 为 f32 类型：

```python
# chunk_gated_delta_h.py:190-196  (LDS 分配)
lds_vn_ptr = SmemPtr(
    lds_base_ptr,
    lds_vn_offset,
    T.f32,                                           # ← f32 类型，占 2x 空间
    shape=(LDS_VN_ELEMS,),
)
lds_vn = STensor(lds_vn_ptr, dtype=T.f32, shape=(LDS_VN_ELEMS,))
```

```python
# chunk_gated_delta_h.py:413-420  (gated v_new 以 f32 写入 LDS)
for elem_i in range_constexpr(4):
    f32_v = vector.extract(vn_val, static_position=[elem_i], dynamic_position=[])
    lds_row = wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
    lds_idx = lds_row * fx.Int32(BV) + lds_col
    lds_vn[fx.Index(lds_idx)] = f32_v                # ← f32 写入 LDS
```

```python
# chunk_gated_delta_h.py:441-448  (从 LDS 读 f32 再转 bf16)
f32_val = lds_vn[fx.Index(lds_elem_idx)]
vn_b_elems.append(arith.trunc_f(T.bf16, f32_val))   # ← 读出后额外 f32→bf16 转换
```

**FlyDSL 汇编** — `17_final_isa.s:321-324` 使用 `ds_write_b32`（4B/elem）：

```asm
; 17_final_isa.s:321-324  (gated v_new 以 f32 写入 LDS)
ds_write_b32 v32, v10                               ; 4 字节/元素
ds_write_b32 v33, v11
ds_write_b32 v34, v0
ds_write_b32 v35, v1
```

**Triton 源码** — `chunk_delta_h.py:1129` 转为 bf16 后参与 dot（LDS 中以 bf16 存储）：

```python
# chunk_delta_h.py:1129
b_v = b_v.to(k.dtype.element_ty)                     # ← 转为 bf16
```

**Triton 汇编** — `.amdgcn:822-825` 使用 `ds_write_b16`（2B/elem）：

```asm
; .amdgcn:822-825  (v_new 以 bf16 写入 LDS)
ds_write_b16 v61, v2 offset:32768                   ; 2 字节/元素
ds_write_b16_d16_hi v61, v2 offset:32896
ds_write_b16 v62, v3 offset:33024
ds_write_b16_d16_hi v62, v3 offset:33152
```

> **根因**：FlyDSL 的 `lds_vn` 声明为 `T.f32`，每个元素占 4B（`ds_write_b32`），LDS 空间和带宽消耗 2x，且读出后需额外 `trunc_f` 转换。Triton 直接以 bf16 存储（`ds_write_b16`），节省空间和带宽。

---

### 4. MFMA 计算密度低 → 源码/汇编定位

**FlyDSL 源码** — 主循环中 delta correction 2 MFMA + state update 2 MFMA = 4 MFMA/iter：

```python
# chunk_gated_delta_h.py:322-339  (delta correction: K_STEPS=2, 每步 1 MFMA × N_REPEAT=1)
for ks in range_constexpr(K_STEPS):       # K_STEPS = K // WMMA_K = 2
    ...
    for nr in range_constexpr(N_REPEAT):  # N_REPEAT = 1
        bv_accs[nr] = _mfma_bf16_16x16x32(a_frag, b_frag, bv_accs[nr])  # 2 MFMA

# chunk_gated_delta_h.py:427-451  (state update: BT_STEPS=2, 每步 1 MFMA × N_REPEAT=1)
for bt_s in range_constexpr(BT_STEPS):    # BT_STEPS = BT // WMMA_K = 2
    ...
    for nr in range_constexpr(N_REPEAT):  # N_REPEAT = 1
        h_accs_in[acc_idx] = _mfma_bf16_16x16x32(k_a_frag, vn_b_frag, h_accs_in[acc_idx])  # 2 MFMA
```

**FlyDSL 汇编** — `17_final_isa.s` 主循环中 4 条 MFMA：

```asm
; 17_final_isa.s:479   (delta correction step 0)
v_mfma_f32_16x16x32_bf16 v[6:9], v[56:59], v[60:63], v[6:9]
; 17_final_isa.s:520   (delta correction step 1)
v_mfma_f32_16x16x32_bf16 v[6:9], v[56:59], v[64:67], v[6:9]
; 17_final_isa.s:543   (state update step 0)
v_mfma_f32_16x16x32_bf16 v[0:3], v[56:59], v[60:63], v[2:5]
; 17_final_isa.s:559   (state update step 1)
v_mfma_f32_16x16x32_bf16 v[2:5], v[56:59], v[64:67], v[0:3]
```

**Triton 源码** — 处理 K=128 的 2 个 64-block，每个 block 有 delta+update 各 2 MFMA = 8 MFMA/iter：

```python
# chunk_delta_h.py:1077   b_v = tl.dot(b_w, b_h1)    → 2 MFMA (delta corr block 0)
# chunk_delta_h.py:1081   b_v += tl.dot(b_w, b_h2)   → 2 MFMA (delta corr block 1)
# chunk_delta_h.py:1133   b_h1 += tl.dot(b_k, b_v)   → 2 MFMA (state update block 0)
# chunk_delta_h.py:1137   b_h2 += tl.dot(b_k, b_v)   → 2 MFMA (state update block 1)
```

**Triton 汇编** — `.amdgcn` 稳态循环 `.LBB0_55` 中 8 条 MFMA：

```asm
; .amdgcn:1057  (delta corr block 0, step 0)
v_mfma_f32_16x16x32_bf16 a[0:3], v[92:95], v[84:87], 0
; .amdgcn:1060  (delta corr block 0, step 1)
v_mfma_f32_16x16x32_bf16 a[0:3], v[96:99], v[88:91], a[0:3]
; .amdgcn:1100  (delta corr block 1, step 0)
v_mfma_f32_16x16x32_bf16 a[0:3], v[6:9], v[26:29], a[0:3]
; .amdgcn:1107  (delta corr block 1, step 1)
v_mfma_f32_16x16x32_bf16 a[0:3], v[96:99], v[92:95], a[0:3]
; .amdgcn:1281  (state update block 0, step 0)
v_mfma_f32_16x16x32_bf16 a[0:3], v[2:5], v[108:111], a[0:3]
; .amdgcn:1283  (state update block 0, step 1)
v_mfma_f32_16x16x32_bf16 a[0:3], v[6:9], v[112:115], a[0:3]
; .amdgcn:1326  (state update block 1, step 0)
v_mfma_f32_16x16x32_bf16 a[4:7], v[2:5], v[104:107], a[4:7]
; .amdgcn:1343  (state update block 1, step 1)
v_mfma_f32_16x16x32_bf16 a[4:7], v[6:9], v[108:111], a[4:7]
```

> **根因**：FlyDSL 仅处理 1 个 K=64 block（NUM_K_BLOCKS=1），每次迭代 4 MFMA；Triton 处理 K=128 的 2 个 block，每次迭代 8 MFMA，计算/访存比高 2x。

---

### 5. AGPR 使用差异（附加观察）

**FlyDSL 汇编** — MFMA 累加器使用普通 VGPR：

```asm
; 17_final_isa.s:804-805
.set chunk_gdn_fwd_h_opt3.num_vgpr, 95
.set chunk_gdn_fwd_h_opt3.num_agpr, 0               ; ← 未使用 AGPR
; MFMA 写入普通 VGPR v[6:9], v[0:3]
v_mfma_f32_16x16x32_bf16 v[6:9], v[56:59], v[60:63], v[6:9]
```

**Triton 汇编** — MFMA 累加器使用 AGPR，通过 `v_accvgpr_write/read` 交互：

```asm
; .amdgcn:1781-1782
.set chunk_gated_delta_rule_fwd_kernel_h_opt3.num_vgpr, 116
.set chunk_gated_delta_rule_fwd_kernel_h_opt3.num_agpr, 8   ; ← 使用 8 个 AGPR

; .amdgcn:840-843  (将 VGPR 值写入 AGPR 作为 MFMA 累加器初始值)
v_accvgpr_write_b32 a0, v30
v_accvgpr_write_b32 a1, v31
v_accvgpr_write_b32 a2, v32
v_accvgpr_write_b32 a3, v33

; .amdgcn:846  (MFMA 结果写入 AGPR a[0:3])
v_mfma_f32_16x16x32_bf16 a[0:3], v[2:5], v[92:95], a[0:3]

; .amdgcn:676-679  (从 AGPR 读出结果到 VGPR)
v_accvgpr_read_b32 v5, a3
v_accvgpr_read_b32 v4, a2
v_accvgpr_read_b32 v3, a1
v_accvgpr_read_b32 v2, a0
```

> AGPR 是 CDNA 架构专用的累加寄存器，MFMA 可直接写入 AGPR 而不占用 VGPR 寄存器压力。FlyDSL 未使用 AGPR，所有累加器占用普通 VGPR。

---

## 对应关系总结表

| 性能差异 | FlyDSL 源码位置 | FlyDSL 汇编特征 | Triton 源码位置 | Triton 汇编特征 |
|---------|----------------|-----------------|----------------|-----------------|
| **向量化加载** | `chunk_gated_delta_h.py:431-436` `k_[fx.Index(k_off)]` 逐元素 | `buffer_load_ushort` (2B) ×32+ | `chunk_delta_h.py:1075-1076` `tl.make_block_ptr` + `tl.load` | `global_load_dwordx4` (16B) |
| **LDS transpose read** | `chunk_gated_delta_h.py:330-337` 逐元素 `lds_h[fx.Index()]` | `ds_read2_b32` + `v_cvt_pk_bf16_f32` + `v_perm_b32` | `chunk_delta_h.py:1077` `tl.dot(b_w, b_h1)` 内部 | `ds_read_b64_tr_b16` (gfx950) |
| **LDS f32 vs bf16** | `chunk_gated_delta_h.py:190-196` `SmemPtr(..., T.f32, ...)` | `ds_write_b32` (4B/elem) | `chunk_delta_h.py:1129` `b_v.to(k.dtype.element_ty)` | `ds_write_b16` (2B/elem) |
| **MFMA 密度** | `chunk_gated_delta_h.py:339,451` 各 2 MFMA = 4 total | 4× `v_mfma_f32_16x16x32_bf16` | `chunk_delta_h.py:1077,1081,1133,1137` 各 2 MFMA = 8 total | 8× `v_mfma_f32_16x16x32_bf16` |
| **AGPR 使用** | 无（VGPR 累加） | `num_agpr=0` | MFMA 写入 `a[0:7]` | `num_agpr=8`, `v_accvgpr_write/read` |

## w/k 加载分析与优化方案

> 详见独立文档 **[gdn_k5_wk_load_optimization.md](gdn_k5_wk_load_optimization.md)**，包含：
>
> - 汇编级 w/k 加载处理对比（FlyDSL `buffer_load_ushort` vs Triton `global_load_dwordx4` → LDS → MFMA）
> - Triton TTGIR 中的 LDS 布局编码（`swizzled_shared`、`dot_op`、`#mma`）
> - 5 项具体改动方案（LDS 空间分配、XOR Swizzle、Cooperative 向量化加载、`ds_read_b64_tr_b16`、v_new bf16 化）
> - 改动后完整主循环数据流图
> - 代码改动清单（10 项）
> - 预期性能提升（279us → ~200us）

## 优化建议

1. 将 w/k 改为 cooperative 向量化加载经 LDS 中转（`buffer_load_dwordx4` → `ds_write_b128` → `ds_read_b128`/`ds_read_b64_tr_b16`）
2. 将 delta correction 结果以 bf16 格式写入 LDS，而非 f32
3. 引入 `ds_read_b64_tr_b16` intrinsic 来高效读取 MFMA 操作数
4. 增大循环体内的计算量（更多 MFMA per iteration）以提高计算密度
5. 统一边界检查为整块级别（`s_and_saveexec_b64`），避免逐元素 `v_cmp` + `v_cndmask` 分支开销
6. 添加 XOR swizzle 消除 LDS bank conflict
