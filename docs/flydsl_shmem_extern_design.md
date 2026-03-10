# FlyDSL Shmem Extern 集成 — 设计要点与实现记录

## 1. 背景与目标

**目标**：在 FlyDSL 中实现等价于 Triton `core.extern_elementwise` 的机制，使 dispatch/combine kernel 能用 `@flyc.kernel` Python 语法编写，shmem 函数调用体验与 Triton 保持一致。

**参考实现**：`mori/examples/shmem/ir/test_triton_shmem.py`

```python
# Triton 中调用 shmem 的方式（目标效果）
@triton.jit
def shmem_put_kernel(symm_buf_ptr, value):
    dest_pe = mori_shmem_device.my_pe() + 1
    mori_shmem_device.int32_p(symm_buf_ptr, value, dest_pe, 0)
    mori_shmem_device.quiet_thread()

# v2 中调用 shmem 的方式（实现效果）
@flyc.kernel
def ep_dispatch(addr_buf: fx.Int64, ...):
    pe = mori_shmem.my_pe()
    mori_shmem.int32_p(local_sym_addr, val, dest_pe, 0)
```

---

## 2. 文件清单

### 2.1 新增文件

| 文件 | 说明 |
|------|------|
| `mori/python/mori/ir/flydsl/__init__.py` | mori shmem FlyDSL 集成包入口 |
| `mori/python/mori/ir/flydsl/ops.py` | 从 `MORI_DEVICE_FUNCTIONS` 自动生成所有 shmem 函数的 `ExternFunction` 包装 |
| `mori/python/mori/ir/flydsl/runtime.py` | `get_bitcode_path()` + `install_hook()` |
| `FlyDSL/python/flydsl/expr/extern.py` | `ExternFunction` 类：在 `@flyc.kernel` 编译期发射 `llvm.LLVMFuncOp` 声明 + `llvm.CallOp` |
| `FlyDSL/python/flydsl/expr/lowlevel.py` | 低层 GPU op 包装：`ballot_i64`, `readlane`, `store_v4i32_global`, `idx_to_i32`, `as_index`, `atomic_add_i32_at`, `load_i32_global` 等 |
| `FlyDSL/python/flydsl/compiler/shmem_compile.py` | `compile_shmem_kernel()` + `ShmemKernel` 可调用对象 |
| `FlyDSL/kernels/dispatch_combine_intranode_v2.py` | 用 `@flyc.kernel` 重写的 dispatch/combine kernel（工厂函数模式） |
| `FlyDSL/kernels/dispatch_combine_intranode_op_v2.py` | v2 算子包装器（Python API 与 v1 兼容） |
| `FlyDSL/tests/kernels/test_v2_accuracy_perf.py` | 精度 + 性能测试（支持 EP=2/8, spawn/torchrun 两种模式） |

### 2.2 修改的现有文件

| 文件 | 改动 |
|------|------|
| `FlyDSL/tests/kernels/test_bench_dispatch_combine_flydsl.py` | 加入 `--mode v2-compile` 编译测试模式；加入 mori 路径 |

---

## 3. 核心架构

```
@flyc.kernel (Python)
    │
    ├── mori_shmem.ptr_p2p(addr, rank, pe)   ← ExternFunction.__call__()
    │       │
    │       └── 发射 llvm.LLVMFuncOp 声明 + llvm.CallOp
    │
    ↓
FlyDSL MLIR passes（up to reconcile-unrealized-casts）
    │
    ↓
mlir-translate → kernel.ll
    │
llvm-link kernel.ll libmori_shmem_device.bc → linked.bc
    │
clang -target amdgcn-amd-amdhsa → kernel.hsaco
    │
hipModuleLoad + shmem_module_init
    │
ShmemKernel.launch() via ctypes
```

---

## 4. ExternFunction 机制

### 4.1 原理

`ExternFunction` 是 `mori_shmem.*` 函数在 `@flyc.kernel` 中的 Python 代理对象。在内核编译（tracing）阶段调用时，它向当前 GPU module 发射 MLIR 声明和调用指令。

```python
# 用户侧（dispatch_combine_intranode_v2.py）
import mori.ir.flydsl as mori_shmem

@flyc.kernel
def ep_dispatch(addr_buf: fx.Int64, ...):
    rem = mori_shmem.ptr_p2p(addr_buf, rank, dest_pe)  # 发射 llvm.call
    mori_shmem.int32_p(local_sym_addr, val, dest_pe, 0)
```

### 4.2 关键实现细节

```python
class ExternFunction:
    def __call__(self, *args):
        ctx = CompilationContext.get_current()  # 获取当前编译上下文
        self._ensure_declared(ctx.gpu_module_body)  # 确保 llvm.LLVMFuncOp 已声明
        call = llvm.CallOp(ret_type, raw_args, ..., callee=symbol_ref)
        return call.result   # 返回 ir.Value（自动提升为 ArithValue）
```

**类型处理**：
- `uint64`/`int32` 等 string 类型名 → 映射为 `IntegerType.get_signless(N)`
- Python int 常量 → 按参数期望类型自动 materialize（`const_i32`, `const_i64`）
- 参数类型不匹配 → 自动 trunc/extui 转换

---

## 5. compile_shmem_kernel() 编译流程

### 5.1 与标准 FlyDSL 编译的区别

| 步骤 | 标准 `@flyc.jit` 路径 | `compile_shmem_kernel` 路径 |
|------|-----------------------|---------------------------|
| 1 | FlyDSL passes + `gpu-module-to-binary{format=fatbin}` | FlyDSL passes（停在 `reconcile-unrealized-casts`） |
| 2 | ExecutionEngine 加载 fatbin | `mlir-translate --mlir-to-llvmir` → kernel.ll |
| 3 | — | `llvm-link kernel.ll libmori_shmem_device.bc` → linked.bc |
| 4 | — | `clang -target amdgcn-amd-amdhsa -mcpu={chip}` → .hsaco |
| 5 | — | `hipModuleLoad` + `shmem_module_init` |
| 调度 | `ExecutionEngine` | `ctypes.hipModuleLaunchKernel` |

### 5.2 GPU module LLVM IR 提取

在 `reconcile-unrealized-casts` 之后，MLIR 模块结构：
```mlir
module {
  gpu.module @kernels [...] {
    llvm.func @ep_dispatch_0(...) attributes {gpu.kernel, rocdl.kernel} { ... }
  }
  llvm.func @_shmem_stub(...) { gpu.launch_func ... }
}
```

`_extract_gpu_module_mlir()` 从中提取 `gpu.module` 的 `llvm.func` 体，包装成独立 MLIR module，再由 `mlir-translate` 转换为 LLVM IR。

---

## 6. @flyc.kernel 编码规则

调试过程中发现 FlyDSL ASTRewriter 有若干约束，必须遵守：

### 规则 1：编译期常量用闭包变量，不用 `fx.Constexpr[int]` 参数

```python
# ✗ 错误：Constexpr 参数不被 ASTRewriter 正确检测
@flyc.kernel
def bad(A: fx.Tensor, rank: fx.Constexpr[int]):  ...

# ✓ 正确：闭包捕获（对标 FlyDSL softmax_kernel.py 的做法）
def make_dispatch_kernel(rank, npes, hidden_dim, ...):
    @flyc.kernel
    def ep_dispatch(addr_buf: fx.Int64, ...):
        # rank, npes, hidden_dim 均为 Python 闭包变量
        rem = mori_shmem.ptr_p2p(addr_buf, rank, dest_pe)
    return ep_dispatch
```

### 规则 2：所有 Buffer 地址以 `fx.Int64` 传入

```python
# ✗ 不可行：fly.memref 不支持 llvm.ptrtoint（类型时序问题）
@flyc.kernel
def bad(inp: fx.Tensor):
    base = ptrtoint(inp)  # 编译时报错

# ✓ 正确：data_ptr() 在 host 侧计算，以 i64 传入
@flyc.kernel
def good(addr_inp: fx.Int64):
    val = load_i32_at(addr_inp, offset)

# 调用侧
args = [ctypes.c_int64(tensor.data_ptr()), ...]
```

### 规则 3：动态 `if` 条件必须包含函数调用

```python
# ✗ ASTRewriter 不会转换（Compare 节点，无函数调用）
if lane == 0:  ...

# ✓ ASTRewriter 检测到函数调用，生成 scf.IfOp
if icmp_eq_i32(lane, const_i32(0)):  ...
```

**原理**：`ReplaceIfWithDispatch._could_be_dynamic()` 只检查条件 AST 中是否有 `ast.Call` 节点。

### 规则 4：for 循环必须在 kernel 顶层

```python
# ✗ scf.if 闭包内的 for 循环不会被 InsertEmptyYieldForSCFFor 变换
if some_cond:
    for ec4 in range(lane4, n_i32, 256):  # 被变换为生成器但闭包语义错误
        ...

# ✓ for 循环在顶层，条件写在循环体内
for ec4 in range(as_index(lane4), as_index(n_i32), 256):
    if some_cond:
        ...
```

### 规则 5：scf.ForOp 归纳变量是 index 类型，需要显式转换

```python
for i in range(as_index(start), as_index(stop), as_index(step)):
    i = idx_to_i32(i)   # 必须！否则算术运算类型不匹配
    tok_id = i // wpt
```

**`idx_to_i32()` 实现**：`arith.IndexCastUIOp(i32, index_val)` + 包装为 `ArithValue`（确保 Python 算符 `//`, `+` 等正确工作）。

### 规则 6：跨 scf.for 迭代的累加必须用 range_constexpr 展开

```python
# ✗ SSA domination 违规：acc 在 scf.for 内定义，循环后无法使用
acc = zero
for j in range(0, k, 1):   # → scf.ForOp, acc 不能流出
    acc = acc + val_j

# ✓ range_constexpr 展开（Python compile-time loop，无 scf.for）
from flydsl.expr import range_constexpr
acc = zero
for j_py in range_constexpr(k):   # → Python 展开，直接生成 k 个 AddFOp
    acc = acc + val_j_py
```

---

## 7. mori shmem API 使用要点

### 7.1 函数签名与 MORI_DEVICE_FUNCTIONS 的差异

实际 bitcode 中的签名与 `MORI_DEVICE_FUNCTIONS` 记录的 string 类型名存在差异：

| 函数 | MORI_DEVICE_FUNCTIONS 声明 | 实际 bitcode 签名 |
|------|--------------------------|-----------------|
| `int32_p` | `(uint64, int32, int32, int32) → int32` | `(ptr, i32, i32, i32) → i32` |
| `uint32_atomic_add_thread` | `(uint64, uint32, int32, int32) → int32` | `(ptr, i32, i32, i32) → i32` |
| `int32_wait_until_equals` | `(uint64, int32) → int32` | `(ptr, i32) → i32` |
| `ptr_p2p` | `(uint64, int32, int32) → uint64` | `(i64, i32, i32) → i64` ✓ |

**关键发现**：`ptr` 和 `i64` 在 AMD GPU (64-bit) 中大小相同，`llvm-link` 可以合并，运行时行为等价。

### 7.2 地址语义（最重要！）

```
shmem 函数                 第一个地址参数含义
─────────────────────────────────────────────────────
int32_p(addr, val, pe, qp)     addr = LOCAL 对称堆地址
                               （NIC 自动将其翻译为 pe 侧的对应地址）

float_p(addr, val, pe, qp)     addr = LOCAL 对称堆地址

uint32_atomic_add_thread(addr, val, pe, qp)
                               addr = LOCAL 对称堆地址

int32_wait_until_equals(addr, val)
                               addr = 可以是 LOCAL 或 ptr_p2p 返回的 P2P 地址
                               （函数直接在该地址自旋读，两种地址都可以）

ptr_p2p(local_sym_addr, my_pe, target_pe)
                               → 返回 P2P 映射地址（XGMI BAR 地址）
                               可用于：store_v4i32_global / load_i32_global
```

**错误示例（导致 GPU fault）**：
```python
# ✗ 把 ptr_p2p 返回的远端地址传给 int32_p
rem = mori_shmem.ptr_p2p(base, rank, dest_pe)
mori_shmem.int32_p(rem, val, dest_pe, 0)  # GPU fault! rem 是远端地址

# ✓ 正确：传 LOCAL 对称堆地址
mori_shmem.int32_p(local_sym_addr + offset, val, dest_pe, 0)
```

### 7.3 XGMI 直写（ptr_p2p + store_v4i32_global）

```python
# ptr_p2p 返回值是 XGMI 映射地址，可以直接用 addrspace(1) store
rem = mori_shmem.ptr_p2p(addr_out_tok, rank, dest_pe)
store_v4i32_global(vec4, rem + byte_offset)  # ✓ 通过 XGMI 写远端 GPU 内存
```

**适用场景**：`inp_tok`（普通 device tensor，不在 shmem 堆）写入远端 shmem 缓冲区。

---

## 8. 动态列表索引：`_sel_pe()` 模式

Python list 不能用运行时 MLIR 值索引，需要用 select 链展开：

```python
def _sel_pe(rem_list, dest_pe):
    """用 select_i64 链实现运行时动态索引。"""
    result = rem_list[-1]
    for pe in reversed(range(len(rem_list) - 1)):
        result = select_i64(icmp_eq_i32(dest_pe, const_i32(pe)), rem_list[pe], result)
    return result

# 使用
rem_tok = [mori_shmem.ptr_p2p(base, rank, pe) for pe in range(npes)]
tok_remote = _sel_pe(rem_tok, dest_pe) + tok_boff
```

对于 npes=8，展开为 7 个 select 指令，编译期展开无运行时开销。

---

## 9. 测试结果

### 9.1 精度

| 测试 | 结果 |
|------|------|
| EP=2, dispatch token 内容 | ✓ max_diff=0.0000 |
| EP=2, total_recv 路由计数 | ✓ 与 mori 参考完全一致 |
| EP=8, dispatch token 内容 | ✓ max_diff=0.0000 |
| EP=8, total_recv 路由计数 | ✓ 与 mori 参考完全一致 |

### 9.2 性能

| 配置 | dispatch v2 | dispatch mori ref | combine v2 | combine mori ref |
|------|-------------|-------------------|------------|-----------------|
| EP=2, max_tok=64, h=128, k=2 | 0.54 ms | 0.09 ms | 0.15 ms | 0.05 ms |
| EP=8, max_tok=128, h=256, k=4 | 6.40 ms | 0.15 ms | 0.23 ms | 0.08 ms |

**dispatch 性能差距原因**：v2 对 weights/indices 使用逐元素 `int32_p`（4 bytes/次 NIC 操作），而 mori 参考使用 `putmem_nbi_warp` 批量传输。token embedding 已使用 `store_v4i32_global` XGMI 直写（16 bytes/次）。

**combine 性能差距原因**：Stage 3 使用 `range_constexpr` 展开 k 次 f32 累加，产生更多指令。

---

## 10. 已知限制与后续优化方向

### 10.1 dispatch 性能优化

- **根本原因**：`inp_tok` 是普通 device tensor，不在 shmem 对称堆中，无法使用 `putmem_nbi_warp` 批量传输 weights/indices。
- **解决方案**：在算子初始化时为输入 token 额外分配一个对称 shmem 缓冲区，dispatch 前先将 inp_tok 复制过去，再用 `putmem_nbi_warp` 批量发送。

### 10.2 combine 权重支持

当前 v2 combine 仅做无权重累加（sum），mori 参考支持 weighted sum。后续可在 Stage 3 中集成权重读取（`shmem_inp_wts` P2P 读 + FMA 累加）。

### 10.3 编译缓存

每次创建算子实例都会重新编译内核（~15 秒/kernel）。后续可实现基于 closure 参数（rank, npes, hidden_dim, ...）的哈希缓存，跳过重复编译。

---

## 11. 快速上手

```bash
# 1. 单进程编译测试（验证 ExternFunction 机制）
cd /home/yashao/FlyDSL
LD_LIBRARY_PATH=python/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH \
python tests/kernels/test_bench_dispatch_combine_flydsl.py \
    --mode v2-compile --chip gfx942

# 2. EP=2 精度 + 性能测试
python tests/kernels/test_v2_accuracy_perf.py \
    --mode both --world-size 2 \
    --max-tokens 64 --hidden-dim 128 --k 2 --num-experts 8 \
    --block-num 4 --warp-per-block 2 --n-rounds 3

# 3. EP=8 性能测试
python tests/kernels/test_v2_accuracy_perf.py \
    --mode bench --world-size 8 \
    --max-tokens 4096 --hidden-dim 7168 --k 8 --num-experts 256 \
    --block-num 80 --warp-per-block 16

# 4. 查看 FlyDSL 中间 IR（调试）
./scripts/dumpir.sh python kernels/dispatch_combine_intranode_v2.py
# IR 输出到 /tmp/flydsl_dump_ir/
```

---

## 12. 汇编分析与 Bug 修复（2026-03-10）

### 问题

V2 dispatch kernel 使用 **155 VGPRs**，V1 仅需 **61 VGPRs**（通过 `llvm-objdump` 对比汇编确认）。高 VGPR 占用导致 GPU 占用率降低、计时差异，进而引发 GPU Memory Access Fault（特别是 k≥4 或大 max_tok 的 benchmark 模式）。

### 根因

FlyDSL 的 `ArithValue.__floordiv__`（Python 的 `//` 运算符）生成 `arith.floordivsi`，最终编译为 AMD GPU 上代价极高的 **`sdiv`（有符号整数除法）**。而 V1 的 LLVM IR 手写代码使用 **`udiv`（无符号除法）**，代价约为 `sdiv` 的 1/3。

```bash
# 对比 VGPR 用量（通过 llvm-objdump 分析）
V1 dispatch: max VGPR = 60
V2 dispatch: max VGPR = 154  (修复前)  → 152 (修复后)

# 对比除法指令（通过 LLVM IR 分析）
V1: udiv i32 %i, %experts_per_token    (无符号，高效)
V2: sdiv i32 %i, %experts_per_token    (有符号，慢 ~3×)
```

### 修复

在 `FlyDSL/python/flydsl/expr/lowlevel.py` 新增 `divui()` 和 `remui()` 辅助函数：

```python
def divui(a, b) -> ir.Value:   # arith.DivUIOp → udiv（比 sdiv 快 ~3×）
def remui(a, b) -> ir.Value:   # arith.RemUIOp → urem
```

在 `dispatch_combine_intranode_v2.py` 中，将所有已知非负值的 `//` 和 `%` 替换：

```python
# Before（生成 sdiv）                  After（生成 udiv）
src_tok = i // experts_per_token   →  src_tok = divui(i, experts_per_token)
dest_pe = dest_exp // experts_per_rank → dest_pe = divui(dest_exp, experts_per_rank)
```

**修复效果**：k=4 benchmark 和 max_tok=256, h=512 等大配置不再 GPU Fault。

### 编码规则（补充）

> **规则 7：对已知非负整数使用 `divui()`/`remui()` 而非 Python 的 `//`/`%`**
>
> FlyDSL 的 `//` 生成有符号除法 (`sdiv`)，对非负值来说既无必要又低效。
> AMD GPU 无硬件整数除法指令，`sdiv` 会展开为多条指令，而 `udiv` 相对简单。

---

## 13. 关键文件索引

```
FlyDSL/
├── python/flydsl/
│   ├── expr/
│   │   ├── extern.py          # ExternFunction（核心机制）
│   │   └── lowlevel.py        # 低层 GPU ops（ballot, readlane, store_v4i32_global...）
│   └── compiler/
│       └── shmem_compile.py   # compile_shmem_kernel() + ShmemKernel
├── kernels/
│   ├── dispatch_combine_intranode_v2.py      # @flyc.kernel 实现（工厂函数）
│   └── dispatch_combine_intranode_op_v2.py   # 算子包装器（兼容 v1 API）
└── tests/kernels/
    └── test_v2_accuracy_perf.py              # 精度 + 性能测试

mori/python/mori/ir/flydsl/
├── __init__.py    # 包入口
├── ops.py         # shmem 函数 ExternFunction 包装
└── runtime.py     # bitcode 路径 + install_hook()
```
