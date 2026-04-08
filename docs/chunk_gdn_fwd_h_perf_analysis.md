# chunk_gdn_fwd_h_opt3 性能分析：FlyDSL vs Triton

FlyDSL kernel (293 us) 与 Triton opt3 kernel (193 us) 的 IR / ISA 对比分析，定位 FlyDSL 编译产物的性能瓶颈并给出优化建议。

> 测试配置：Qwen3.5-397B-A17B TP=8, K=V=128, H=8, Hg=2, BT=64, full_prompt_len=8000, T=8192, gfx950

---

## 一、基础指标对比

| 指标 | Triton opt3 (193 us) | FlyDSL (293 us) | 差异 |
|------|---------------------|-----------------|------|
| Kernel 耗时 | 193 us | 293 us | FlyDSL 慢 **52%** |
| ISA 行数 | 2011 | 826 | Triton 代码更大但更高效 |
| LLVM IR 行数 | 1386 | 791 | — |
| Kernel 代码大小 | 7152 bytes | 更小 | — |
| VGPR | **124** (116 + 8 AGPR) | **62** (0 AGPR) | FlyDSL 未用 AGPR |
| SGPR | **52** | **78** | FlyDSL 标量寄存器压力更高 |
| LDS 大小 | **0** (静态) | **6656 bytes** | 策略不同 |
| Occupancy | **4** waves/SIMD | 取决于 VGPR/LDS | — |
| 线程配置 | 256 threads/WG | 256 threads/WG | 相同 |

---

## 二、核心性能差异

### 2.1 MFMA 计算密度不足

| 指标 | Triton | FlyDSL |
|------|--------|--------|
| 每次迭代 MFMA 数 | **8** | **4** |
| 全 kernel MFMA 总数 | **26** | **9** |
| MFMA 指令类型 | `mfma.f32.16x16x32.bf16` | 相同 |

Triton 在每个循环迭代中执行 **8 次 MFMA**，FlyDSL 只有 **4 次**。这意味着 FlyDSL 的计算密度只有 Triton 的一半。在同样的循环迭代次数下，FlyDSL 的有效计算吞吐显著更低。

**根因**: FlyDSL 在 K 维度上的 tiling 策略不如 Triton，没有充分展开计算。

### 2.2 全局内存访问：标量 vs 向量化（最关键瓶颈）

| 指标 | Triton | FlyDSL |
|------|--------|--------|
| 循环内 load 指令数 | **11** (`global_load_dwordx4` 等) | **37** (`buffer_load_dword/ushort`) |
| 全局 load 向量宽度 | `<8 x bfloat>` / `<4 x float>` | 大量**标量 bf16** load |
| 地址模式 | `addrspace(1)` flat global | `raw.ptr.buffer.load` offset |

**Triton** 的全局内存访问是**宽向量化**的：
- `global_load_dwordx4` 一次加载 128-bit (8 个 bf16 或 4 个 float)
- 直接加载 `<8 x bfloat>` 向量喂给 MFMA

**FlyDSL** 的全局内存访问大量退化为**标量**操作：
- 使用 `buffer_load_ushort`（单个 bf16，16-bit）逐元素加载
- 加载后用 `insertelement` 手工拼装成 `<8 x bfloat>` 向量
- k 矩阵每次需要 8 个标量 load + 8 个 select + 1 个 vector.from_elements

**量化对比**: FlyDSL 循环体 37 次 load（多数是标量），Triton 仅 11 次 load（全部向量化），但搬运的数据量更大。标量 load 不仅指令数膨胀 3 倍以上，还无法充分利用内存带宽（每条 load 只搬运 2-4 bytes vs Triton 的 16 bytes）。

### 2.3 LDS 使用策略差异

| 指标 | Triton | FlyDSL |
|------|--------|--------|
| LDS 分配 | 0 bytes (静态) | 6656 bytes |
| `ds_*` 指令数 | **~119** | **~10** |
| `s_barrier` 数 | **~21** | **2** |
| 关键 intrinsic | `ds.read.tr16.b64` + `ds.bpermute` | 无 |

Triton 虽然静态 LDS=0，但实际大量使用 LDS 进行**数据转置和 lane 间通信**：
- `ds.read.tr16.b64.v4bf16` — 转置读取，从 LDS 读数据时自动完成 layout 变换为 MFMA 友好格式
- `ds.bpermute` — 跨 lane 数据交换，用于高效重组数据

FlyDSL 分配了 LDS 但只做简单 store + load 中继（写入 bf16 tile，barrier，再读回），未利用 GFX950 的高级 LDS 指令。

### 2.4 exp 指令实现差异

| 指标 | Triton | FlyDSL |
|------|--------|--------|
| 实现方式 | `exp2(x * (1/ln2))` | `exp(x)` 直接调用 |
| 循环内 exp 次数 | **2** | **5** |
| gate 处理 | 向量化 `<2 x float>` 操作 | 逐元素标量 fsub + exp + select |

FlyDSL 对 gate 的计算是完全标量化的：先 load 4 个 g 值，分别做 `fsub` → `exp` → `select`，再 `insertelement` 拼装。Triton 则利用向量化批量处理。

### 2.5 v_new 存储的分支开销

FlyDSL 对 `v_new` 的存储使用了 **4 个独立的 `scf.if` 分支**（每个元素一个条件判断），在 ISA 层面变成 4 组 `s_and_saveexec_b64` + `s_cbranch_execz`。Triton 使用 **masked store** 一次性完成所有元素的条件写入。

```
// FlyDSL: 4个独立分支
scf.if %cond0 { store v_new[0] }
scf.if %cond1 { store v_new[1] }
scf.if %cond2 { store v_new[2] }
scf.if %cond3 { store v_new[3] }

// Triton: 单次 masked store
tt.store %ptr, %data, %mask   // 一条指令，mask 控制哪些写入
```

### 2.6 AGPR 累加器未使用

| 指标 | Triton | FlyDSL |
|------|--------|--------|
| AGPR 数量 | **8** | **0** |
| 累加器位置 | AGPR (专用) | VGPR (通用) |

GFX950 的 AGPR (Accumulator GPR) 是专门为 MFMA 累加器设计的寄存器文件。使用 AGPR 可以释放 VGPR 压力，允许更多寄存器用于数据暂存，进而提升 occupancy 或减少 spill。

### 2.7 Software Pipelining

| 指标 | Triton | FlyDSL |
|------|--------|--------|
| Pipelining | 有 prologue（循环外预加载） | 无 |
| 循环剥离 | 有 (`llvm.loop.peeled.count = 1`) | 无 |

Triton 的 TTGIR 中明确标注了 `amd.pipeliner_part = "prologue"` 的预加载操作，将下一迭代的数据加载提前到当前迭代的计算阶段，实现 **load-compute overlap**。FlyDSL 的循环没有这种优化。

---

## 三、LLVM IR 层面关键差异汇总

| 方面 | Triton (`.llir`) | FlyDSL (`16_llvm_ir.ll`) |
|------|------------------|--------------------------|
| 全局寻址 | `addrspace(1)` load/store | `raw.ptr.buffer.load/store` |
| 全局向量 | `<8 x bfloat>`, `<4 x float>` 常见 | 部分 `v8bf16`，大量标量 bf16/f32 |
| LDS 高级指令 | `ds.read.tr16`, `ds.bpermute` | 无 |
| Exp 实现 | `llvm.exp2.f32` + scale | `llvm.exp.f32`（多次） |
| Barrier 数量 | ~21 (精细流水线控制) | 2 (简单同步) |
| 循环结构 | 有 peel + software pipeline | 简单 counted loop |

---

## 四、ISA 层面关键差异汇总

| 方面 | Triton (`.amdgcn`) | FlyDSL (`17_final_isa.s`) |
|------|---------------------|---------------------------|
| MFMA / 迭代 | 8 | 4 |
| global/buffer load / 迭代 | 11 (向量化) | 37 (大量标量) |
| `s_waitcnt` / 迭代 | 18 | 41 |
| `ds_*` 操作 / 全kernel | ~119 | ~10 |
| `s_barrier` / 全kernel | ~21 | 2 |
| `v_exp_f32` / 迭代 | 2 | 5 |
| 基本块数（循环附近） | ~30 (精细调度) | ~8 |
| 代码总大小 | 7152 bytes | 更小 |

---

## 五、优化建议（按优先级排序）

### P0（高优先级，预估收益最大）

#### 1. 全局内存访问向量化

**问题**: 标量 `buffer_load_ushort` (bf16) 逐个加载 → `insertelement` 拼装向量

**目标**: 合并连续地址的标量 load 为 `buffer_load_dwordx4` 等宽向量指令

**预估收益**: 30-50%

**具体方向**:
- FlyDSL 编译器在 layout lowering 阶段，识别连续地址的标量 load 模式
- 将 8 个连续 bf16 load 合并为 1 个 `buffer_load_dwordx4`（128-bit）
- 减少 k 矩阵加载的指令数从 ~8 条降为 ~1 条

#### 2. 增加每次迭代 MFMA 数量

**问题**: 每次循环迭代仅 4 次 MFMA，计算密度不足

**目标**: K 维度更好的 tiling，每次迭代 8 次 MFMA

**预估收益**: 20-40%

**具体方向**:
- 调整 K 维度的 tile 大小和展开因子
- 参考 Triton 的 `b_h1` / `b_h2` 双 tile 策略，在 V 维度做 2-way tiling
- 确保 MFMA 链之间有足够的数据复用

### P1（中优先级）

#### 3. 利用 `ds_read_tr16_b64` 完成 LDS 转置读取

**问题**: FlyDSL 使用普通 LDS load (align 2)，没有利用硬件转置能力

**目标**: 使用 `ds.read.tr16.b64.v4bf16` 在 LDS 读取时完成 layout 变换

**预估收益**: 10-20%

**具体方向**:
- 这是 GFX950 新增的 LDS 指令，可在读取时自动转置数据
- 适配 MFMA 的输入 layout 要求，避免寄存器中额外的 permute 操作

#### 4. 合并 v_new 条件存储

**问题**: 4 个独立的 `scf.if` 分支，产生 4 组 exec mask 切换

**目标**: 合并为 masked vector store

**预估收益**: 5-10%

**具体方向**:
- 在 FlyDSL lowering 阶段识别连续条件写入模式
- 生成带 exec mask 的向量 store，避免分支开销

### P2（低优先级）

#### 5. 使用 AGPR 作为 MFMA 累加器

**问题**: MFMA 结果存在 VGPR 中，占用通用寄存器

**目标**: 使用 AGPR 释放 VGPR 压力

**预估收益**: 5-10%

#### 6. 减少 gate 计算的标量 exp 次数

**问题**: 5 次 `v_exp_f32` / 迭代，全部标量处理

**目标**: 向量化 gate 计算流程，减少 exp 调用

**预估收益**: 3-5%

#### 7. 实现 Software Pipelining

**问题**: 循环无 load-compute overlap

**目标**: 将下一迭代的 global load 提前到当前迭代的 MFMA 执行期间

**预估收益**: 5-15%

---

## 六、已实施优化及效果

> 日期: 2026-04-08

### 6.1 优化结果总览

| 版本 | Kernel 耗时 | 相对 Triton | 变化 |
|------|------------|------------|------|
| FlyDSL 原始 | **293 us** | 0.66x | — |
| FlyDSL 优化后 | **279 us** | 0.69x | **-14 us (-5%)** |
| Triton opt3 | **193 us** | 1.00x | — |

精度验证: 优化后 FlyDSL 与 Triton 输出**位精确匹配** (abs_err max=0.000000)。

### 6.2 成功应用的优化

#### exp → exp2 内联指令 (收益: 293us → 279us, -5%)

**问题**: `math_dialect.ExpOp` 经 MLIR 管线降级为 `@llvm.exp.f32` 内联函数，LLVM 后端将其展开为 ~10 条指令的完整范围缩减序列:

```
v_mul_f32   ; x * log2(e)
v_fma_f32   ; 高精度补偿
v_rndne_f32 ; 取整
v_fmac_f32  ; 残差修正
v_sub_f32   ; 分离整数/小数部分
v_add_f32   ; 合并
v_exp_f32   ; 2^frac
v_cvt_i32   ; 整数部分
v_ldexp_f32 ; 2^int * 2^frac
v_cndmask   ; 范围钳位
```

**方案**: 使用 `_llvm.call_intrinsic("llvm.exp2.f32", ...)` 直接发射 LLVM `exp2` 内联指令，手动实现 `exp(x) = exp2(x * log2(e))`:

```python
def _fast_exp(x):
    log2e = arith.constant(math.log2(math.e), type=T.f32)
    return _llvm.call_intrinsic(ir.F32Type.get(), "llvm.exp2.f32",
                                [_to_raw(arith.mulf(x, log2e))], [], [])
```

优化后每个 exp 仅需 2 条指令:

```
v_mul_f32  v56, 0x3fb8aa3b, v56   ; x * log2(e)
v_exp_f32  v56, v56                ; 2^(x*log2e) = e^x
```

**LLVM IR 变化**:
- 原始: `call float @llvm.exp.f32(float %x)` × 5 → 展开为 ~50 条 ISA
- 优化: `call float @llvm.exp2.f32(float %mul)` × 5 → 仅 ~10 条 ISA

**修改文件**: `kernels/chunk_gated_delta_h.py` — 添加 `_llvm_exp2_f32()` / `_fast_exp()` 辅助函数，替换 gating 中的 `math_dialect.ExpOp`。

### 6.3 新增基础设施

#### FlatGTensor (tensor_shim.py)

添加了基于 LLVM GEP + load/store 的 flat global 内存访问类 `FlatGTensor`，使用 `addrspace(0)` 指针和 `llvm.GEPOp` 进行元素寻址。

该类作为基础设施已就绪，但本次优化中**未最终采用**（见 6.4 节）。

### 6.4 尝试但回退的优化方案

| 方案 | 预期收益 | 实测结果 | 回退原因 |
|------|---------|---------|---------|
| **Flat global 替代 buffer load** | 5-15% | 293→425 us (+45%) | 64 位地址计算 (`s_add_u32/s_addc_u32` 对) 开销远大于 buffer load 的 32 位 VGPR offset |
| **LDS staging for k matrix** | 10-20% | 293→561 us (+91%) | 无 `ds_read_tr16_b64` 时，额外的 LDS 写入 + barrier + 逐元素 LDS 读取反而增加开销 |
| **去掉 scf.if 用 masked buffer store** | 5-10% | 279→616 us (+121%) | `buffer_store` 的 mask 实现将 OOB offset 设为 `0x7FFFFFFF`，触发极慢的 OOB 处理路径 |
| **math.Exp2Op (MLIR math dialect)** | 3-5% | 293→661 us (+126%) | MLIR `math.exp2` 降级为 `@__ocml_exp2_f32` 库函数调用（非内联），引入函数调用开销 |

**关键教训**:
1. AMD buffer load 在此 kernel 中比 flat global 更高效，因为 buffer 描述符的 SGPR base + 32-bit VGPR offset 模式避免了 64 位地址运算。
2. LDS staging 只有在配合 `ds_read_tr16_b64` 等硬件转置指令时才有收益；纯 LDS 中转反而增加延迟。
3. MLIR math dialect 的 `Exp2Op` 和直接使用 `llvm.exp2.f32` 内联指令走的是完全不同的降级路径，性能差异巨大。

### 6.5 优化后 ISA 指标对比

| 指标 | Triton opt3 (193 us) | FlyDSL 原始 (293 us) | FlyDSL 优化 (279 us) |
|------|---------------------|---------------------|---------------------|
| ISA 行数 | 2011 | 826 | **897** |
| LLVM IR 行数 | 1386 | 791 | **1203** |
| VGPR | 124 (116+8 AGPR) | 62 (0 AGPR) | **95** (0 AGPR) |
| SGPR | 52 | 78 | **78** |
| MFMA 总数 | 24 | 8 | **8** |
| `buffer_load` 总数 | 0 | 55 | **55** |
| `ds_*` 操作总数 | ~130 | ~8 | **~53** |
| `s_barrier` 总数 | ~21 | 2 | **2** |
| `v_exp_f32` 总数 | 6 | 5 | **5** |
| `s_cbranch` 总数 | — | 8 | **6** |
| Exp LLVM IR | `@llvm.exp2.f32` | `@llvm.exp.f32` | **`@llvm.exp2.f32`** |

### 6.6 剩余性能差距分析 (279 us vs 193 us, ~45%)

剩余差距主要来自 Triton 编译器的以下能力，在 FlyDSL 手动 MFMA 编程模型中难以直接复制:

1. **`ds_read_tr16_b64` 硬件转置 LDS 读取** (~130 ds 操作 vs 53): Triton 将 k/v_new 数据通过 LDS 中转并使用硬件转置指令，大幅减少全局内存访问次数和寄存器中的 permute 操作。
2. **`ds_bpermute` 跨 lane 数据交换**: Triton 用于 v_new 的 bf16 分发，避免 LDS roundtrip。
3. **XOR swizzle LDS 布局**: 消除 LDS bank conflict，需要复杂的地址计算 (`v_bitop3_b32`)。
4. **AGPR 累加器** (8 AGPRs): Triton 使用专用累加器寄存器，释放 VGPR 用于数据暂存。
5. **Software pipelining**: Triton 编译器自动交错下一迭代的 global load 与当前迭代的 MFMA 计算。
6. **Tile-level 向量化**: Triton 的 `tl.exp(b_g_last - b_g)` 对整个 BT 维度一次性向量化处理，而 FlyDSL 在 MFMA fragment 级别逐元素处理 (4 个 exp / warp)。
7. **MFMA 展开** (24 vs 8): Triton 在循环外展开了更多 MFMA 指令（3x unroll），提高指令级并行度。

### 6.7 后续优化方向建议

| 优先级 | 方向 | 预估收益 | 难度 |
|--------|------|---------|------|
| P0 | 实现 `ds_read_tr16_b64` + XOR swizzle LDS 布局用于 k 矩阵 | 15-25% | 高 — 需要精确匹配 Triton 的 swizzle pattern |
| P0 | 实现 `ds_bpermute` 用于 v_new bf16 分发 | 5-10% | 中 — 参考 `flash_attn_func.py` 已有实现 |
| P1 | AGPR 累加器 | 5-10% | 中 — 需要修改 MFMA intrinsic 调用方式 |
| P1 | Software pipelining (load-compute overlap) | 5-15% | 高 — 需要手动构建 prologue/epilogue |
| P2 | 循环展开 (3x unroll) | 3-5% | 低 — 增加代码大小换取 ILP |

---

## 七、数据来源

| 文件 | 路径 |
|------|------|
| Triton LLVM IR | `/workspace/ir_dump/triton_193us_ir_dump_opt3/chunk_gated_delta_rule_fwd_kernel_h_opt3.llir` |
| Triton ISA | `/workspace/ir_dump/triton_193us_ir_dump_opt3/chunk_gated_delta_rule_fwd_kernel_h_opt3.amdgcn` |
| FlyDSL 原始 LLVM IR | `/workspace/ir_dump/origin_flydsl_293us_ir_output/chunk_gdn_fwd_h_opt3/16_llvm_ir.ll` |
| FlyDSL 原始 ISA | `/workspace/ir_dump/origin_flydsl_293us_ir_output/chunk_gdn_fwd_h_opt3/17_final_isa.s` |
| FlyDSL 优化后 LLVM IR | `/workspace/ir_dump/opt_flydsl_ir_output/chunk_gdn_fwd_h_opt3/16_llvm_ir.ll` |
| FlyDSL 优化后 ISA | `/workspace/ir_dump/opt_flydsl_ir_output/chunk_gdn_fwd_h_opt3/17_final_isa.s` |
| FlyDSL 内核源码 | `/workspace/FlyDSL/kernels/chunk_gated_delta_h.py` |
| FlyDSL 内存抽象 | `/workspace/FlyDSL/kernels/tensor_shim.py` |
| Triton 参考实现 | `/workspace/linear_attn_example/kernel/triton/chunk_delta_h.py` |
