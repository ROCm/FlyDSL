# GDN K5 w/k 加载分析与优化方案

> 从 [gdn_k5_perf_analysis.md](gdn_k5_perf_analysis.md) 拆分。聚焦 w/k 的全局加载 → LDS → MFMA 操作数的完整数据流对比与改造方案。

## 一、汇编级 w/k 加载处理对比

### 1. w 的加载（delta correction 阶段 A 操作数）

#### FlyDSL：`buffer_load_ushort` 逐元素标量加载（2B/次）

**源码** — `chunk_gated_delta_h.py:322-328`，`vec_load(..., 8)` 请求 8 个 bf16：

```python
for ks in range_constexpr(K_STEPS):
    w_bt_row_raw = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + lane_n
    w_col = fx.Int32(ks * WMMA_K) + lane_m_base * fx.Int32(8)
    w_off = w_base + safe_w_row * stride_w + w_col
    a_frag = w_.vec_load((fx.Index(w_off),), 8)
```

**汇编** — `17_final_isa.s:345-435`，生成大量 `buffer_load_ushort`（每条不同地址寄存器，编译器未合并为向量加载）：

```asm
; 17_final_isa.s:345-352  (w 加载，K_STEPS=0)
buffer_load_ushort v58, v71, s[36:39], 0 offen
buffer_load_ushort v59, v72, s[36:39], 0 offen
buffer_load_ushort v61, v73, s[36:39], 0 offen
buffer_load_ushort v62, v74, s[36:39], 0 offen
buffer_load_ushort v63, v75, s[36:39], 0 offen
buffer_load_ushort v64, v1, s[36:39], 0 offen
buffer_load_ushort v71, v11, s[36:39], 0 offen
buffer_load_ushort v76, v57, s[36:39], 0 offen
; ... K_STEPS=1 再重复 8 条，加上第二组 K-block 的 16 条
; 共约 32 条 buffer_load_ushort
```

每条 `buffer_load_ushort` 只加载 2B（1 个 bf16），且每个元素都有独立的 `v_cmp_gt_i32` + `v_cndmask_b32` 边界检查。

**w 不经过 LDS**，直接从 Global Memory → VGPR → MFMA A 操作数。

#### Triton：`global_load_dwordx4` → LDS → `ds_read_b128` → MFMA A

**源码** — `chunk_delta_h.py:1075-1077`，块加载整个 `[BT, 64]` tile：

```python
p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
b_w = tl.load(p_w, boundary_check=(0, 1))
b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
```

**汇编** — 四步流水线：

**Step 1**：从全局内存向量化加载（`.amdgcn:186`）：

```asm
; .amdgcn:186  (w 块加载，16B/次 = 8 个 bf16)
global_load_dwordx4 v[36:39], v[4:5], off
```

**Step 2**：写入 LDS（`.amdgcn:474-478`）：

```asm
; .amdgcn:474-478  (w 写入 LDS，ds_write_b128 = 16B/次)
ds_write_b128 v57, v[36:39]
ds_write_b128 v57, v[42:45] offset:4096
ds_write_b128 v57, v[64:67] offset:8192
ds_write_b128 v3, v[60:63] offset:4096
```

**Step 3**：从 LDS 读取作为 MFMA A 操作数（`.amdgcn:597-598`）：

```asm
; .amdgcn:597-598  (从 LDS 读 w 的 MFMA A 操作数)
ds_read_b128 v[76:79], v59
ds_read_b128 v[80:83], v60
```

**Step 4**：送入 MFMA（`.amdgcn:613,616`）：

```asm
; .amdgcn:613  (delta correction step 0)
v_mfma_f32_16x16x32_bf16 a[0:3], v[84:87], v[76:79], 0
; .amdgcn:616  (delta correction step 1)
v_mfma_f32_16x16x32_bf16 a[0:3], v[96:99], v[80:83], a[0:3]
```

> 注意：这里 `v[84:87]`/`v[96:99]` 是 w 的 A 操作数（从 `ds_read_b128` 读出），`v[76:79]`/`v[80:83]` 是 h snapshot 的 B 操作数（也从 LDS 读出）。

---

### 2. k 的加载（state update 阶段 A 操作数）

#### FlyDSL：逐元素 `buffer_load_ushort`（2B/次）

**源码** — `chunk_gated_delta_h.py:431-437`，循环 8 次逐元素加载：

```python
for ki in range_constexpr(8):
    k_t_row_raw = i_t_i32 * fx.Int32(BT) + fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(8) + fx.Int32(ki)
    k_off = k_base + k_t_row * stride_k + k_col
    k_val = k_[fx.Index(k_off)]                    # ← vec_size=1，逐元素
    k_a_elems.append(arith.select(k_row_valid, k_val, arith.constant(0.0, type=T.bf16)))
```

**汇编** — `17_final_isa.s:633-636`，使用不同 buffer resource `s[56:59]`（k 的 buffer）：

```asm
; 17_final_isa.s:633-636  (k 加载，state update)
buffer_load_ushort v59, v1, s[56:59], 0 offen
buffer_load_ushort v68, v10, s[56:59], 0 offen
buffer_load_ushort v69, v11, s[56:59], 0 offen
buffer_load_ushort v70, v0, s[56:59], 0 offen
; ... 共 8 条 × BT_STEPS × NUM_K_BLOCKS
```

**k 不经过 LDS**，直接从 Global Memory → VGPR → `vector.from_elements` 组装 → MFMA A 操作数。

#### Triton：`global_load_dwordx4` → LDS → `ds_read_b64_tr_b16` → MFMA A

**源码** — `chunk_delta_h.py:1131-1133`，块加载 `[64, BT]` tile：

```python
p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
b_k = tl.load(p_k, boundary_check=(0, 1))
b_h1 += tl.dot(b_k, b_v)
```

**汇编** — 四步流水线：

**Step 1**：从全局内存向量化加载（`.amdgcn:360`）：

```asm
; .amdgcn:360  (k 块加载)
global_load_dwordx4 v[68:71], v[4:5], off
```

主循环中（`.amdgcn:1195`）：

```asm
; .amdgcn:1195  (k 块加载，下一迭代 prefetch)
global_load_dwordx4 v[92:95], v[38:39], off
```

**Step 2**：写入 LDS（`.amdgcn:480,918-919,1338-1339`）：

```asm
; .amdgcn:480  (k 写入 LDS)
ds_write_b128 v57, v[68:71] offset:16384
; .amdgcn:918-919  (主循环中 k 写入 LDS)
ds_write_b128 v57, v[84:87] offset:16384
ds_write_b128 v57, v[88:91] offset:20480
; .amdgcn:1338-1339  (稳态循环中 k 写入 LDS)
ds_write_b128 v57, v[92:95] offset:16384
ds_write_b128 v57, v[100:103] offset:20480
```

**Step 3**：从 LDS 用 `ds_read_b64_tr_b16` 读取+转置（`.amdgcn:815-818,1219-1222`）：

```asm
; .amdgcn:815-818  (k 从 LDS 读取+转置，gfx950 专有)
ds_read_b64_tr_b16 v[92:93], v28 offset:16384
ds_read_b64_tr_b16 v[94:95], v36 offset:512
ds_read_b64_tr_b16 v[96:97], v38 offset:4096
ds_read_b64_tr_b16 v[98:99], v36 offset:4608
; .amdgcn:1219-1222  (稳态循环中 k 的 transpose read)
ds_read_b64_tr_b16 v[108:109], v42
ds_read_b64_tr_b16 v[110:111], v43
ds_read_b64_tr_b16 v[112:113], v44
ds_read_b64_tr_b16 v[114:115], v45
```

**Step 4**：送入 MFMA（`.amdgcn:846,851,1281,1283`）：

```asm
; .amdgcn:846  (state update step 0)
v_mfma_f32_16x16x32_bf16 a[0:3], v[2:5], v[92:95], a[0:3]
; .amdgcn:851  (state update step 1)
v_mfma_f32_16x16x32_bf16 a[0:3], v[6:9], v[96:99], a[0:3]
; .amdgcn:1281  (稳态循环 state update step 0)
v_mfma_f32_16x16x32_bf16 a[0:3], v[2:5], v[108:111], a[0:3]
; .amdgcn:1283  (稳态循环 state update step 1)
v_mfma_f32_16x16x32_bf16 a[0:3], v[6:9], v[112:115], a[0:3]
```

> 这里 MFMA 的 A 操作数 `v[2:5]`/`v[6:9]` 是 gated v_new（从 LDS 中 `ds_read_b128` 读出），B 操作数 `v[92:95]`/`v[96:99]` 是 k（从 LDS 中 `ds_read_b64_tr_b16` 读出+转置）。

---

### 3. 汇编级对比总结表

| 方面 | FlyDSL | Triton |
|------|--------|--------|
| **全局内存加载指令** | `buffer_load_ushort`（2B/次） | `global_load_dwordx4`（16B/次） |
| **加载带宽利用率** | 每次 2B，需 8 条指令加载 16B | 每次 16B，1 条指令加载 16B |
| **w/k 是否经过 LDS** | **否**，直接 Global → VGPR → MFMA A | **是**，Global → VGPR → LDS → VGPR → MFMA A |
| **LDS 写入 w/k** | 不涉及 | `ds_write_b128`（16B/次，高效） |
| **LDS 读取 w** | 不涉及 | `ds_read_b128`（16B/次） |
| **LDS 读取 k** | 不涉及 | `ds_read_b64_tr_b16`（gfx950 专有，读取+转置一步完成） |
| **MFMA 操作数组装** | 8 次 `buffer_load_ushort` + `v_perm_b32` 手动组装 8xbf16 | 编译器自动从 LDS 读取并组装 |
| **边界检查** | 每个元素 `v_cmp` + `v_cndmask`（~32 条分支指令） | `s_and_saveexec_b64` 整块跳过（~4 条） |
| **w 加载总指令数** | ~32 条 `buffer_load_ushort` + ~32 条 cmp/cndmask | ~4 条 `global_load_dwordx4` + ~4 条 `ds_write_b128` + ~2 条 `ds_read_b128` |
| **k 加载总指令数** | ~16 条 `buffer_load_ushort` + ~16 条 cmp/cndmask | ~4 条 `global_load_dwordx4` + ~4 条 `ds_write_b128` + ~4 条 `ds_read_b64_tr_b16` |

### 4. Triton 为什么选择 Global → LDS → MFMA 而非直接 Global → MFMA

Triton 将 w/k 先写入 LDS 再读出，看似多了一步，但有三个关键优势：

1. **`ds_read_b64_tr_b16` 硬件转置**：gfx950 的这条指令在 LDS 读取时同时完成数据转置，直接生成 MFMA 需要的操作数布局。FlyDSL 没有这条指令，需要 `v_perm_b32` + `v_cvt_pk_bf16_f32` 多步软件转置。

2. **跨 warp 数据共享**：`tl.dot` 中的矩阵乘需要多个 warp 协作，LDS 是 warp 间共享数据的唯一途径。FlyDSL 的 4 个 warp 各自独立从全局内存加载自己需要的数据片段，存在重复加载。

3. **向量化加载效率**：`global_load_dwordx4`（16B）比 `buffer_load_ushort`（2B）的带宽利用率高 8 倍。Triton 的块加载天然保证地址连续性，而 FlyDSL 的逐元素索引计算导致编译器无法证明地址连续，只能退化为标量加载。

---

## 二、优化方案：使 FlyDSL w/k 加载与 Triton 完全一致

### 目标数据流对比

**当前 FlyDSL（279us，慢）**：

```
w/k:  Global --[buffer_load_ushort × 8]--> VGPR --[v_perm_b32]--> MFMA A
h:    VGPR --[ds_write_b16]--> LDS --[ds_read2_b32 + v_cvt_pk_bf16]--> MFMA B
v_new: VGPR --[ds_write_b32(f32)]--> LDS --[ds_read2_b32 + trunc_f]--> MFMA B
```

**目标（与 Triton 193us 一致）**：

```
w:    Global --[buffer_load_dwordx4]--> VGPR --[ds_write_b128]--> LDS --[ds_read_b128]--> MFMA A
k:    Global --[buffer_load_dwordx4]--> VGPR --[ds_write_b128]--> LDS --[ds_read_b64_tr_b16]--> MFMA A
h:    VGPR --[ds_write_b128]--> LDS --[ds_read_b128]--> MFMA B
v_new: VGPR --[ds_write_b16(bf16)]--> LDS --[ds_read_b64_tr_b16]--> MFMA B
```

---

### Triton TTGIR 中的 LDS 布局编码

从 Triton 的 TTGIR 中提取的关键布局定义：

```
#blocked  = #ttg.blocked<{sizePerThread=[8,1], threadsPerWarp=[8,8], warpsPerCTA=[1,4], order=[0,1]}>
#blocked2 = #ttg.blocked<{sizePerThread=[1,8], threadsPerWarp=[8,8], warpsPerCTA=[4,1], order=[1,0]}>
#mma      = #ttg.amd_mfma<{version=4, warpsPerCTA=[4,1], instrShape=[16,16], isTransposed=true}>
#shared   = #ttg.swizzled_shared<{vec=8, perPhase=2, maxPhase=8, order=[1,0]}>   -- w 用
#shared1  = #ttg.swizzled_shared<{vec=8, perPhase=2, maxPhase=8, order=[0,1]}>   -- k / v_new / h 用
```

| 张量 | SMEM 编码 | 寄存器布局 | dot_op 角色 |
|------|----------|-----------|------------|
| w `[BT,64]` | `#shared` (order=[1,0]) | `#blocked2` → `dot_op opIdx=0` | MFMA A |
| k `[64,BT]` | `#shared1` (order=[0,1]) | `#blocked` → `dot_op opIdx=0` | MFMA A |
| h `[64,BV]` | `#shared1` (经 `local_alloc`) | `dot_op opIdx=1` | MFMA B |
| v_new `[BT,BV]` | `#shared1` (经 `local_alloc`) | `dot_op opIdx=1` | MFMA B |

Triton 的 TTGIR 数据流：

```
# Delta correction: b_v = dot(w, h)
%w_lds   = ttg.local_load %w_smem   → tensor<64x64xbf16, dot_op<opIdx=0>>   -- ds_read_b128
%h_lds   = ttg.local_load %h_smem   → tensor<64x16xbf16, dot_op<opIdx=1>>   -- ds_read_b64_tr_b16
%b_v     = tt.dot %w_lds, %h_lds    → tensor<64x16xf32, #mma>

# State update: h += dot(k, v_new)
%k_lds   = ttg.local_load %k_smem   → tensor<64x64xbf16, dot_op<opIdx=0>>   -- ds_read_b64_tr_b16
%vn_lds  = ttg.local_load %vn_smem  → tensor<64x16xbf16, dot_op<opIdx=1>>   -- ds_read_b64_tr_b16
%h_new   = tt.dot %k_lds, %vn_lds   → tensor<64x16xf32, #mma>
```

---

### 改动 1：新增 LDS 空间给 w 和 k

**当前 LDS 分配**（`chunk_gated_delta_h.py:133-145`）：

```python
# 当前
LDS_VN_BYTES = BT * BV * 4   # f32, 64×32×4 = 8192 bytes
LDS_H_BYTES  = K * BV * 2    # bf16, 128×32×2 = 8192 bytes
# 总计: 16384 bytes
```

**改造后**：

```python
# w tile: [BT, 64] bf16, 一个 K-block
LDS_W_BYTES  = BT * 64 * 2           # 64×64×2 = 8192 bytes

# k tile: [64, BT] bf16, 一个 K-block
LDS_K_BYTES  = 64 * BT * 2           # 64×64×2 = 8192 bytes

# v_new: [BT, BV] bf16 (从 f32 改为 bf16)
LDS_VN_BYTES = BT * BV * 2           # 64×16×2 = 2048 bytes (BV=16)

# h snapshot: [K, BV] bf16, 不变
LDS_H_BYTES  = K * BV * 2            # 128×16×2 = 4096 bytes (BV=16)
```

> 注：w 和 k 在不同阶段使用（delta correction vs state update），可以复用同一块 LDS 空间。Triton 为 w 和 k 各分配了 `NUM_K_BLOCKS × 64 × 64 × 2` bytes 的 LDS（含 double-buffer）。

---

### 改动 2：XOR Swizzle 消除 LDS bank conflict

Triton 使用 `swizzled_shared<{vec=8, perPhase=2, maxPhase=8}>`，等价于以下 XOR swizzle：

```python
def xor_swizzle(row, col, vec=8, perPhase=2, maxPhase=8):
    """Triton-style XOR swizzle.
    
    对于 bf16 元素 (2B)，vec=8 表示 8 个元素 = 16 bytes 为一组。
    phase = (row // perPhase) % maxPhase
    swizzled_col = col ^ (phase * vec)
    """
    phase = (row // perPhase) % maxPhase
    return col ^ (phase * vec)
```

写入和读取 LDS 时**必须使用相同的 swizzle 函数**。

FlyDSL 仓库中 `flash_attn_func.py` 已有类似实现可参考：

```python
# flash_attn_func.py:394 — K 的 XOR swizzle
def _k_swizzle(row_idx, col_idx):
    mask = (row_idx & arith.index(0x7)) << arith.index(4)
    return col_idx ^ mask

# flash_attn_func.py:548 — V 的 XOR swizzle
def _v_swizzle(row_idx, col_idx):
    mask = (row_idx & arith.index(0x3)) << arith.index(4)
    return col_idx ^ mask
```

---

### 改动 3：Cooperative 向量化加载 w/k 到 LDS

当前每个 warp 独立从全局内存逐元素加载自己需要的 w/k 片段。改为**全 block 256 线程协作加载**整个 tile 到 LDS。

**线程分解**：

```python
LOAD_VEC_WIDTH = 8                                # 8 bf16 = 16B = dwordx4
ELEMS_PER_ROW = 64                                # K-block 宽度
THREADS_PER_ROW = ELEMS_PER_ROW // LOAD_VEC_WIDTH # 64/8 = 8
ROWS_PER_BATCH = BLOCK_THREADS // THREADS_PER_ROW # 256/8 = 32
NUM_BATCHES = BT // ROWS_PER_BATCH                # 64/32 = 2

load_row_in_batch = tid // THREADS_PER_ROW        # 0..31
load_col_base = (tid % THREADS_PER_ROW) * LOAD_VEC_WIDTH  # 0,8,16,...,56
```

**w 的协作加载**（参考 `flash_attn_func.py:398-425` 的 `coop_load_k` 模式）：

```python
def coop_load_w_to_lds(i_t_i32, kb):
    """全 block 协作加载 w[i_t*BT:(i_t+1)*BT, kb*64:(kb+1)*64] 到 LDS_w。"""
    for batch in range_constexpr(NUM_BATCHES):
        row = fx.Int32(batch * ROWS_PER_BATCH) + load_row_in_batch
        abs_row = i_t_i32 * fx.Int32(BT) + row
        
        # 整块边界检查 (替代逐元素 v_cmp)
        in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
        safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
        
        # 向量化全局加载: buffer_load vec_width=8 → buffer_load_dwordx4
        g_off = w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
        vec = w_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH)
        
        # XOR swizzle 写入 LDS
        swz_col = load_col_base ^ ((row & fx.Int32(0x7)) << fx.Int32(3))
        lds_idx = row * fx.Int32(64) + swz_col
        lds_w.vec_store((fx.Index(lds_idx),), vec, LOAD_VEC_WIDTH)
    
    gpu.barrier()
```

**k 的协作加载**（k 的内存布局 `[T, Hg*K]`，K 维度 stride=1，连续）：

```python
def coop_load_k_to_lds(i_t_i32, kb):
    """全 block 协作加载 k 的转置 tile 到 LDS_k。
    
    k 在全局内存中是 [T, Hg*K]，每行 K 个元素连续。
    加载 k[i_t*BT:(i_t+1)*BT, kb*64:(kb+1)*64] 并以 [64, BT] 转置存入 LDS。
    """
    for batch in range_constexpr(NUM_BATCHES):
        row = fx.Int32(batch * ROWS_PER_BATCH) + load_row_in_batch
        abs_row = i_t_i32 * fx.Int32(BT) + row
        in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
        safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
        
        # 全局加载 k 的一行 (K 维度连续，天然向量化)
        g_off = k_base + safe_row * stride_k + fx.Int32(kb * 64) + load_col_base
        vec = k_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH)
        
        # 写入 LDS: 行主序 [BT, 64]，后续用 ds_read_b64_tr_b16 做硬件转置
        swz_col = load_col_base ^ ((row & fx.Int32(0x7)) << fx.Int32(3))
        lds_idx = row * fx.Int32(64) + swz_col
        lds_k.vec_store((fx.Index(lds_idx),), vec, LOAD_VEC_WIDTH)
    
    gpu.barrier()
```

> **关键**：`GTensor.vec_load(..., vec_size=8)` 底层调用 `buffer_ops.buffer_load(rsrc, offset, vec_width=8, dtype=bf16)`，生成 `rocdl.RawPtrBufferLoadOp` 结果类型为 `vector<8xbf16>`，LLVM 后端会选择 `buffer_load_dwordx4`（16B）指令。

---

### 改动 4：从 LDS 读取 w/k 作为 MFMA 操作数

#### w 的 LDS 读取（delta correction A 操作数）

w 在 LDS 中是 `[BT, 64]` 行主序（与 Triton `#shared` order=[1,0] 一致），MFMA A 操作数需要沿 K 维度连续的 8xbf16。使用 `ds_read_b128`：

```python
def read_w_a_frag(ks):
    """从 LDS 读取 w 的 MFMA A 操作数 (8xbf16)。"""
    # 每个 lane 需要 BT 维度上的一个位置，K 维度上连续 8 个 bf16
    row = wid * fx.Int32(16) + lane_n          # BT 维度
    col = fx.Int32(ks * WMMA_K) + lane_m_base * fx.Int32(8)  # K 维度，8 连续
    swz_col = col ^ ((row & fx.Int32(0x7)) << fx.Int32(3))
    lds_idx = row * fx.Int32(64) + swz_col
    return lds_w.vec_load((fx.Index(lds_idx),), 8)  # → ds_read_b128
```

#### k 的 LDS 读取（state update A 操作数）— 使用 `ds_read_b64_tr_b16`

k 需要做转置（从 `[BT, 64]` 读出 `[64, BT]` 的视角），使用 gfx950 专有的 `ds_read_b64_tr_b16` 硬件转置读取。

参考 `flash_attn_func.py:292-307` 的已有实现：

```python
v4bf16_type = T.vec(4, T.bf16)

def ds_read_tr_bf16x4(lds_elem_idx):
    """ds_read_b64_tr_b16: 从 LDS 读取 4xbf16 并做硬件转置。
    
    在每 16 个 lane 的块内，硬件对 4 组 × 4 lane 做 4×4 转置。
    转置后 result[lane, e] = Input[source_lane, lane%4]
    其中 source_lane = e*4 + (lane%16)//4。
    """
    byte_offset = lds_elem_idx * fx.Int32(2) + fx.Int32(lds_k_byte_offset)
    byte_i64 = arith.index_cast(T.i64, byte_offset)
    ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
    return rocdl.ds_read_tr16_b64(v4bf16_type, ptr).result

def read_k_a_frag(bt_s):
    """从 LDS 用 ds_read_b64_tr_b16 读取 k 的 MFMA A 操作数 (8xbf16)。"""
    # lane 映射 (参考 flash_attn 的 tr_col_sub / tr_k_group 分解)
    tr_col_sub = lane % fx.Int32(4)
    tr_col_half = (lane % fx.Int32(32)) // fx.Int32(16)
    tr_k_group = (lane % fx.Int32(16)) // fx.Int32(4)
    lane_div_32 = lane // fx.Int32(32)
    
    k_row = wid * fx.Int32(16) + tr_col_half * fx.Int32(16) + tr_col_sub * fx.Int32(4)
    bt_col = fx.Int32(bt_s * WMMA_K) + lane_div_32 * fx.Int32(4) + tr_k_group
    swz_col = bt_col ^ ((k_row & fx.Int32(0x7)) << fx.Int32(3))
    lds_base = k_row * fx.Int32(64) + swz_col  # 注意 k 在 LDS 中仍是 [BT,64]
    
    # ds_read_b64_tr_b16 返回 4xbf16，需要 2 次调用 + shuffle 得到 8xbf16
    lo = ds_read_tr_bf16x4(lds_base)
    hi = ds_read_tr_bf16x4(lds_base + fx.Int32(8 * 64))  # 偏移 8 行
    return vector.shuffle(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7])
```

---

### 改动 5：gated v_new 改为 bf16 写入 LDS

**当前**（`chunk_gated_delta_h.py:412-420`）：

```python
# f32 写入 LDS → ds_write_b32 (4B/elem)
f32_v = vector.extract(vn_val, static_position=[elem_i], dynamic_position=[])
lds_vn[fx.Index(lds_idx)] = f32_v
```

**改造后**：

```python
# 先截断为 bf16，再写入 LDS → ds_write_b16 (2B/elem)
f32_v = vector.extract(vn_val, static_position=[elem_i], dynamic_position=[])
bf16_v = arith.trunc_f(T.bf16, f32_v)
lds_vn_bf16[fx.Index(lds_idx)] = bf16_v
```

state update 阶段从 LDS 读取 v_new 时，也改用 `ds_read_b64_tr_b16`（v_new 是 MFMA B 操作数，需要转置读取）。

---

### 改动后的完整主循环数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│  for i_t in range(NT):   (chunk 循环)                               │
│                                                                     │
│  ┌─ STEP 1: Store h snapshot ──────────────────────────────────┐    │
│  │  h_accs → trunc bf16 → global store (h_out)                │    │
│  │  h_accs → trunc bf16 → ds_write_b128 → LDS_h              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─ STEP 2: Cooperative load w ────────────────────────────────┐    │
│  │  全 256 线程: buffer_load_dwordx4 → ds_write_b128 → LDS_w  │    │
│  │  (XOR swizzle, 每线程 16B, 2 批次覆盖 [BT,64])             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  barrier                                                            │
│                                                                     │
│  ┌─ STEP 3: Delta correction  b_v = w @ h ─────────────────────┐   │
│  │  for ks in K_STEPS:                                          │   │
│  │    w_a = ds_read_b128(LDS_w)          -- MFMA A operand     │   │
│  │    h_b = ds_read_b128(LDS_h)          -- MFMA B operand     │   │
│  │    bv_acc = mfma_bf16_16x16x32(w_a, h_b, bv_acc)           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ STEP 4: v_new = u - b_v, store v_new ──────────────────────┐   │
│  │  u_val = buffer_load(u)                                      │   │
│  │  vn = u_val - bv_acc                                         │   │
│  │  buffer_store(vn, v_new_out)                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ STEP 5: Gating + store gated v_new to LDS (bf16) ──────────┐   │
│  │  gate = exp(g_last - g_row)                                  │   │
│  │  vn_gated = vn * gate                                        │   │
│  │  trunc_f(bf16) → ds_write_b16 → LDS_vn                     │   │
│  │  h_accs *= exp(g_last)                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ STEP 6: Cooperative load k ────────────────────────────────┐    │
│  │  全 256 线程: buffer_load_dwordx4 → ds_write_b128 → LDS_k  │    │
│  │  (XOR swizzle, 每线程 16B, 2 批次覆盖 [BT,64])             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  barrier                                                            │
│                                                                     │
│  ┌─ STEP 7: State update  h += k^T @ v_new ────────────────────┐   │
│  │  for bt_s in BT_STEPS:                                       │   │
│  │    k_a = ds_read_b64_tr_b16(LDS_k)   -- MFMA A (HW transpose) │ │
│  │    vn_b = ds_read_b64_tr_b16(LDS_vn) -- MFMA B (HW transpose) │ │
│  │    h_acc = mfma_bf16_16x16x32(k_a, vn_b, h_acc)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  yield h_accs → 下一个 chunk                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、代码改动清单

| # | 文件位置 | 改动内容 |
|---|---------|---------|
| 1 | `chunk_gated_delta_h.py:133-145` | 新增 `LDS_W`、`LDS_K` 空间分配；`LDS_VN` 从 f32 改为 bf16 |
| 2 | `chunk_gated_delta_h.py:188-206` | 新增 `lds_w`、`lds_k` 的 `SmemPtr`/`STensor` 声明；`lds_vn` 改为 bf16 |
| 3 | `chunk_gated_delta_h.py:256-258` | 新增 cooperative load 线程分解（`load_row_in_batch`、`load_col_base`） |
| 4 | 新增 helper | `xor_swizzle(row, col)` — XOR swizzle 函数 |
| 5 | 新增 helper | `ds_read_tr_bf16x4(lds_elem_idx)` — 参考 `flash_attn_func.py:294-307` |
| 6 | 新增 helper | `coop_load_w_to_lds(i_t, kb)` — 全 block 协作加载 w |
| 7 | 新增 helper | `coop_load_k_to_lds(i_t, kb)` — 全 block 协作加载 k |
| 8 | `chunk_gated_delta_h.py:322-339` | **重写 delta correction**：`coop_load_w_to_lds` + `ds_read_b128` 读 w + `ds_read_b128` 读 h → MFMA |
| 9 | `chunk_gated_delta_h.py:412-420` | **v_new 写 LDS**：f32 → bf16 `trunc_f` 后 `ds_write_b16` |
| 10 | `chunk_gated_delta_h.py:427-451` | **重写 state update**：`coop_load_k_to_lds` + `ds_read_b64_tr_b16` 读 k + `ds_read_b64_tr_b16` 读 v_new → MFMA |

---

## 四、预期性能提升

| 指标 | 改动前 (279us) | 改动后 (预期) | 提升 |
|------|---------------|-------------|------|
| w 加载 | ~32× `buffer_load_ushort` (2B) | ~4× `buffer_load_dwordx4` (16B) + ~4× `ds_write_b128` + ~2× `ds_read_b128` | 全局带宽利用率 ×8 |
| k 加载 | ~16× `buffer_load_ushort` (2B) | ~4× `buffer_load_dwordx4` (16B) + ~4× `ds_write_b128` + ~4× `ds_read_b64_tr_b16` | 全局带宽利用率 ×8 |
| v_new LDS | `ds_write_b32` (4B) + `ds_read2_b32` + `trunc_f` | `ds_write_b16` (2B) + `ds_read_b64_tr_b16` | LDS 带宽减半，消除 trunc 开销 |
| 边界检查 | 逐元素 `v_cmp` + `v_cndmask` (~64 条) | 整块 `s_and_saveexec` (~4 条) | 分支指令 ×16 减少 |
| MFMA 操作数组装 | `v_perm_b32` + `v_cvt_pk_bf16_f32` 多步 | 硬件直接生成（`ds_read_b128` / `ds_read_b64_tr_b16`） | 消除软件转置 |

综合预期：kernel 时间从 **~279us 降到 ~200us** 左右（接近 Triton 的 193us）。

---

## 五、仓库内参考实现

FlyDSL 仓库中 `kernels/flash_attn_func.py` 已有完整的参考模式：

| 模式 | flash_attn 位置 | GDN K5 对应 |
|------|----------------|------------|
| Cooperative 向量化加载 | `coop_load_k()` (L398-425) | `coop_load_w_to_lds` / `coop_load_k_to_lds` |
| XOR swizzle | `_k_swizzle()` (L394) / `_v_swizzle()` (L548) | `xor_swizzle()` |
| `ds_read_b64_tr_b16` | `ds_read_tr_v4f16()` (L294-307) | `ds_read_tr_bf16x4()` |
| `vector.store` 写 LDS | L417, L424 | `lds_w.vec_store()` |
| `_gep_load` 向量化全局加载 | `load_global_f16xN()` (L352) | `w_.vec_load(..., 8)` / `k_.vec_load(..., 8)` |
