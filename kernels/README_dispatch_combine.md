# FlyDSL Dispatch/Combine IntraNode — 使用说明

## 文件结构

```
kernels/
├── dispatch_combine_intranode_kernel_v2.py      # v2 Kernel 工厂（Python FlyDSL 语法）★ 主文件
├── dispatch_combine_intranode_op_v2.py   # v2 算子包装器（对外 API）              ★ 主文件
├── dispatch_combine_intranode_kernel.py  # v1 Kernel（手写 LLVM IR，Legacy）
└── dispatch_combine_intranode_op.py      # v1 算子包装器（Legacy）

tests/kernels/
└── test_v2_accuracy_perf.py             # v2 精度 + 性能测试脚本                  ★ 测试入口
```

---

## 精度测试

### 快速启动（spawn 模式，适合 2–8 GPU）

```bash
cd /home/yashao/FlyDSL
export MORI_SHMEM_HEAP_SIZE=6G
export LD_LIBRARY_PATH=/home/yashao/FlyDSL/python/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH

# EP=2，精度测试（与 mori 参考对比）
python tests/kernels/test_v2_accuracy_perf.py \
    --mode test \
    --world-size 2 \
    --max-tokens 64 \
    --hidden-dim 128 \
    --k 2 \
    --num-experts 8

# EP=8，精度测试
python tests/kernels/test_v2_accuracy_perf.py \
    --mode test \
    --world-size 8 \
    --max-tokens 64 \
    --hidden-dim 128 \
    --k 4 \
    --num-experts 64
```

### torchrun 模式（适合生产环境）

```bash
torchrun --nproc_per_node=8 tests/kernels/test_v2_accuracy_perf.py \
    --mode test \
    --max-tokens 256 \
    --hidden-dim 512 \
    --k 4
```

### 精度输出说明

```
[精度测试] EP=2, max_tok=64, hidden=128, k=2, experts=8
  Round 1: cur_tok=39  total_recv v2=67 ref=67 ✓
    dispatch:  tok max_diff (v2 vs ref, sorted)      = 0.0000 ✓
    combine:   v2/k vs ref (should match)           = 0.xxxx ✓
    combine:   v2/k vs original_input                = ~x.xxxx (expected: (1-n_valid/k)*max_orig)
```

- **dispatch max_diff = 0.0000** ✓ 表示 dispatch 完全正确（与 mori 参考比特级一致）
- **combine v2/k vs orig** 的差值是预期行为：v2 combine 做无权重累加，`out = n_valid * inp`；
  对于 npes=2, k=2 场景，期望差值约为 `0.5 * max_inp`（每 token 平均 1 个非重复专家）

---

## 性能测试

### 单次运行（精度 + 性能）

```bash
python tests/kernels/test_v2_accuracy_perf.py \
    --mode both \
    --world-size 2 \
    --max-tokens 256 \
    --hidden-dim 512 \
    --k 4 \
    --num-experts 16 \
    --block-num 8 \
    --warp-per-block 4 \
    --warmup 3 \
    --iters 10
```

### 只跑性能

```bash
python tests/kernels/test_v2_accuracy_perf.py \
    --mode bench \
    --world-size 8 \
    --max-tokens 4096 \
    --hidden-dim 7168 \
    --k 8 \
    --num-experts 256 \
    --block-num 80 \
    --warp-per-block 16
```

### 性能输出说明

```
[性能测试] EP=2, max_tok=256, hidden=512, k=4, block_num=8, wpb=4
                    FlyDSL v2     mori ref          对比
  ────────────── ──────────── ────────────  ──────────
  dispatch           2.132 ms     0.135 ms  (0.06x ref)
  combine            0.230 ms     0.113 ms  (0.55x ref)
  dispatch+comb      2.362 ms     0.248 ms  (0.11x ref)
```

---

## 全参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `both` | `test`/`bench`/`both` |
| `--world-size` | 2 | GPU 数量（EP 度） |
| `--max-tokens` | 512 | 每 rank 最大 token 数 |
| `--hidden-dim` | 512 | Hidden dimension |
| `--k` | 4 | Top-k 专家数 |
| `--num-experts` | ws×32 | 专家总数（若不指定则自动推导） |
| `--block-num` | 16 | CUDA block 数 |
| `--warp-per-block` | 4 | 每 block 的 warp 数 |
| `--chip` | `gfx942` | AMD GPU 型号 |
| `--n-rounds` | 3 | 精度测试轮数 |
| `--warmup` | 3 | 性能测试预热轮数 |
| `--iters` | 10 | 性能测试正式轮数 |
| `--no-compare` | — | 跳过与 mori ref 的对比 |

---

## 在 Python 代码中直接使用

```python
import torch
from kernels.dispatch_combine_intranode_op_v2 import (
    FlyDSLDispatchCombineConfigV2,
    FlyDSLDispatchCombineIntraNodeOpV2,
)

cfg = FlyDSLDispatchCombineConfigV2(
    rank=rank,
    world_size=world_size,
    hidden_dim=7168,
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32,   # 256 experts / 8 GPUs
    top_k=8,
    data_type=torch.bfloat16,
    warp_num_per_block=16,
    block_num=80,
    chip="gfx942",
)

op = FlyDSLDispatchCombineIntraNodeOpV2(cfg)

# 每轮推理
op.reset()
dispatch_out, wts, _, idx, total_recv = op.dispatch(inp, weights, None, expert_indices)

# expert 处理 ...

combine_out, _ = op.combine(expert_output, weights, idx)
```

---

## 性能分析：v2 vs mori 汇编对比

> 工具：`/opt/rocm/llvm/bin/llvm-objdump -d <hsaco>` 反汇编，对比 `/tmp/v2_disp_*.hsaco`（v2）与 `/tmp/ep_dispatch_gfx942_r0.hsaco`（mori 参考）。

### 1. Dispatch 内核代码规模对比（EP=2, k=2, hidden=128）

| 指标 | v2 FlyDSL（修复前）| v2 FlyDSL（修复后）| mori 参考 |
|------|:---------:|:---------:|:--------:|
| 内核代码字节数 | 29,832 | 24,636 (≈) | 21,972 |
| 总指令数 | 5,486 | 4,910 | 5,493 |
| `flat_atomic_cmpswap`（软件 CAS）| 13 | 9 | — |
| `global_atomic_add`（XGMI 硬件原子）| 0 | 1 | — |
| `s_waitcnt` 总数 | 410 | — | 356 |
| `flat_load` 总数 | 296 | — | 165 |

### 2. Dispatch 15.8× 性能差距根因（逐条分析）

#### **[主因] `uint32_atomic_fetch_add_thread` 从全部 64 个 lane 调用**

```
旧代码（dispatch_combine_intranode_kernel_v2.py, ~行 162）：

add_one  = select_i32(lane0, select_i32(is_dup, 0, 1), 0)   # lane1-63: add_one=0
dest_tok = mori_shmem.uint32_atomic_fetch_add_thread(         # 全 64 lane 调用！
               addr_tok_off, add_one, dest_pe, 0)
```

| 项目 | v2 旧版 | mori 参考 |
|------|---------|----------|
| 调用 lane 数 | **64**（含 63 个 add=0 的无效调用）| **1**（仅 lane0） |
| 底层实现 | 软件 CAS 循环（flat_atomic_cmpswap + branch）| `atomicAdd()` 直接 XGMI 硬件原子 |
| 每次 XGMI 往返 | ~10+ 条指令 × retry | 1 条 `global_atomic_add_ret` 指令 |
| 竞争时行为 | O(64) 轮 CAS 重试（所有 lane 竞争同一 tok_off 地址）| 无重试 |

**估算 XGMI 流量倍数：64 lanes × CAS 重试轮次 ≈ 640× 更多 XGMI 操作**

mori 对应代码（`intranode.hpp`）：
```cpp
if (laneId == 0) {
    destTokId = atomicAdd(args.dispTokOffsetMemObj->GetAs<index_t*>(destPe), 1);
}
destTokId = __shfl(destTokId, 0);   // 广播
```
只有 lane0 执行一次原生 GPU `atomicAdd`，对应 1 条 `flat_atomic_add_ret` 指令。

#### **[次因] 软件 CAS 循环 vs 硬件原子指令**

- `mori_shmem_uint32_atomic_fetch_add_thread`（P2P 路径）= `GetGlobalGpuStatesPtr()` + `peerPtrs[pe]` 地址计算 + 软件 CAS 循环
- CAS 循环：`flat_load_dword sc0 sc1` → `v_add_u32` → `flat_atomic_cmpswap sc0 sc1` → 检查 → 重试分支
- 对比：`global_atomic_add_ret`（addrspace 1，XGMI）= **1 条硬件指令**，不会重试

#### **[次次因] 重复 token 时仍加载所有数据**

```python
# v2：内层 token 数据循环在 dup 检测之外
for ec4 in range(as_index(lane4), as_index(n_i32), 256):
    vec4 = load_v4i32(inp_src_b + ec4_byt)   # 总是加载（即使 is_dup=True）
    if _icmp_eq_i64(dup_ballot, const_i64(0)):
        store_v4i32_global(vec4, tok_remote + ec4_byt)  # 只有非重复才存
```
mori 使用 `continue` 跳过重复 token 的所有工作（包括数据加载）。对于 npes=2、k=4，约 50% token 为重复，额外浪费 ~50% 带宽。

### 3. 已实施修复：Phase 1 tok_off 原子操作

**修复方案**：用 `scf.IfOp` 将 exec mask 收缩到 lane 0，使用 `atomic_fetch_add_i32_global`（生成 `global_atomic_add_ret`，单条 XGMI 硬件指令）替代软件 CAS 循环。

```python
# 新增：预计算各 PE 的 tok_off XGMI 地址（在循环外）
rem_tok_off = [mori_shmem.ptr_p2p(addr_tok_off, rank, pe) for pe in range(npes)]

# Phase 1 loop 内：
from flydsl._mlir.dialects import scf as _scf_d
from flydsl._mlir.ir import InsertionPoint as _IP, IntegerType as _IT_mlir
_i32_ty = _IT_mlir.get_signless(32)
_lane0_cond = _lv_unwrap(icmp_eq_i32(lane, const_i32(0)))
_if_lane0 = _scf_d.IfOp(_lane0_cond, [_i32_ty], has_else=True)
with _IP(_if_lane0.then_block):
    _tok_off_xgmi = _sel_pe(rem_tok_off, dest_pe)   # XGMI addrspace(1) 地址
    _add_delta = select_i32(is_dup, const_i32(0), const_i32(1))
    _old_tok = atomic_fetch_add_i32_global(_tok_off_xgmi, _add_delta)  # 单条指令!
    _scf_d.YieldOp([_lv_unwrap(_old_tok)])
with _IP(_if_lane0.else_block):
    _scf_d.YieldOp([_lv_unwrap(const_i32(0))])      # lanes 1-63：不执行原子
dest_tok     = _if_lane0.result
dest_tok_all = readlane(dest_tok, 0)                 # 广播 lane0 结果
```

**修复效果**（汇编对比）：
- `flat_atomic_cmpswap`：13 → 9（Phase 1 的 CAS 循环消除）
- `global_atomic_add`：0 → 1（新增 XGMI 硬件原子指令）
- 总指令数：5,486 → 4,910（减少 576 条，约 10.5%）
- 精度测试：dispatch max_diff = 0.0000 ✓

### 4. Combine 2× 差距分析

v2 combine 与 mori combine 的汇编指令数量几乎**完全相同**（误差 <1%），说明 combine 2× 差距不来自代码质量问题，可能来自：

- **运行时数据分布差异**：v2 combine 使用均匀权重 1/k，而 mori 有可能用了不同的权重策略
- **内存访问步长差异**：v2 combine 内层循环用 lane 步长 64（1 i32/lane），dispatch 用 lane×4 步长 256（4 i32/lane），需验证 cache 行为
- **`total_recv` 计算路径**：v2 combine 依赖 dispatch 阶段写入 shmem 的计数，若存在额外同步等待则增加延迟

### 5. 待优化项（后续）

| 优先级 | 优化点 | 预期收益 |
|--------|--------|---------|
| P0 ✅ | tok_off 原子：64 lane CAS → lane0 硬件原子 | ~10-64× dispatch 加速 |
| P1 | 重复 token：数据加载移到 dup 检测之后 | ~2× dispatch 数据带宽节省 |
| P2 | Phase 2 信号传递：`uint32_atomic_add_thread` 也走 CAS | 信号延迟降低 |
| P3 | `int32_p`/`float_p` 改用 XGMI 直写（同 mori `GetAs<T*>(pe)`）| 小幅提升 |
| P4 | combine：验证是否能用 4×i32 向量读替代逐元素 load | cache 利用率提升 |

---

## 常见问题

**Q: 编译时间很长？**  
A: 首次编译约 15-30 秒，HSACO 会缓存在 `/tmp/v2_disp_*.hsaco`，后续重启进程可直接加载。
缓存文件名包含配置参数，不同配置不会互相污染。

**Q: `hipModuleLaunchKernel` 返回 error 719（资源不足）？**  
A: 减小 `--warp-per-block`（如从 8 改为 4）或 `--block-num`，降低每 block 的线程数。

**Q: dispatch max_diff 不是 0？**  
A: 检查精度测试是否跨轮次（多轮精度测试时存在 shmem barrier 配对问题），建议单轮（`--n-rounds 1`）验证。
