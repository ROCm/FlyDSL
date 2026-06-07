# FlyDSL vs up-aiter MoE GEMM — 学习笔记（M=16384 / bm=128 / KIMI-K2.5）

> 目标：搞清楚 `bench.py` 在最大形状 M=16384 上为什么 FlyDSL 的 gemm 比 up-aiter mxfp4 慢
> （trace 数据：gemm1 fly 784us vs mx 678us；gemm2 fly 909us vs mx 849us）。
> 形状常量：NE=385, H=7168 (=K), INTER=512 (gate/up 各 512), TOPK=9。

---

## 1. 入口 & 实际跑的是哪个 kernel

`~/FlyDSL/bench.py` 两边都通过 `aiter.fused_moe.fused_moe()` 进入，CSV 决定走哪条路：

| 边 | dispatch 路径 | M=16384 实际 kernel |
|---|---|---|
| **mxfp4** | `shuffle_kind="mxfp4_moe"` 标签命中 `kimik2_5_mxfp4_tuned_fmoe.csv` 的 `BM128` 行 | `aiter::mxfp4_moe::gemm1::kernel<655360, 385, 7168, 1024, BM=128, kUseNT=false, kInlineQuant=false, kXcdSwizzle=0>` |
| **flydsl** | 无 `shuffle_kind` 标签，命中 untagged `kimik2_fp4_tuned_fmoe.csv`（CSV 里 M=16384 是 `t64x128x256`，但 padding 落到 M=8192 行）→ `_flydsl_stage1_wrapper` → `compile_mixed_moe_gemm1` | `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` |

源码位置：
- mxfp4 gemm1：`/root/up-aiter/csrc/kernels/mxfp4_moe/gemm_a4w4/gemm1_a4w4.cuh:22-593`
- mxfp4 BM128 实例化：`/root/up-aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128.cu`
- FlyDSL gemm1：`/root/FlyDSL/kernels/mixed_moe_gemm_2stage.py:97-2415`
- FlyDSL wrapper：`/root/up-aiter/aiter/fused_moe.py:845-906`

---

## 2. 形状 & tile 对照（M=16384）

| 项目 | mxfp4 (BM=128) | FlyDSL (t128x128x256) |
|---|---|---|
| 单 block 输出 | `BM×BN = 128×256`（gate/up 拼在 N 维） | `tile_m × tile_n × 2 = 128×128 × {gate,up}` ⇒ 实质 128×256 |
| K 维 tile | `BK=256`, `K_TILES = K/BK = 28` | `tile_k=256`，K_TILES=28 |
| 单 block MFMA 数 | 28 × 32 = **896** | 28 × 64 = **1792**（gate + up 各一组） |
| Grid（一次） | `n_blocks=N_OUT/256=4`, `m_blocks≈⌈(M·TOPK+NE·127)/128⌉≈1534` → **6136** | `n_blocks=inter_dim/tile_n=4`（只覆盖 gate side，up 在同 block 算），`m_blocks` 同 ≈ **6136** |
| 总 MFMA | ~5.5 M | ~11 M ❌ |

⚠️ 看起来 FlyDSL 总 MFMA 是 mxfp4 的 2 倍。但每个 MFMA 是 `mfma_scale_f32_16x16x128_f8f6f4`，输出 16×16 块。两边总输出 token 数一样，所以**实际 MFMA 计数应该相等**。差别在于**记账方式**：
- mxfp4 把 (gate || up) 拼成 BN=256 一维 → 每个 J 簇里 8 MFMA × 4 J × kSubBlocks=4
- FlyDSL 把 gate/up 当两条平行流 → `compute_bmajor_mfma_phase` 里每个 (k,ni) 同时下两条 MFMA（gate + up）

数学上相等。但在汇编层 FlyDSL 多了一次 LDS-A 读使用、多了一份 B-VMEM 读、accumulator VGPR 占用更高 → 实际 FLOPs 一样而 VGPR 压力大。

---

## 3. 关键差异（按"对性能的影响"排序）

### 3.1 A-scale 加载策略（最大差距之一）

**mxfp4** (`gemm1_a4w4.cuh:165-186, 432`)：
```cpp
// 在 run_one 开头，K-loop 之前 ONCE：一次性把整个 M-block 的所有 K-tile 的 A-scale
// 直接 buffer_load_lds 到 LDS，s_Ascale[kSubBlocks * K_TILES * 256]
issue_a_scale_load(m_row);    // 28 K-tiles × kSubBlocks 一次塞进 LDS
// K-loop 里只做 LDS read：
issue_a_scale_ds_read(kt);    // 纯 LDS 操作，零 VMEM
```

**FlyDSL** (`mixed_moe_gemm_2stage.py:1366-1379`)：
```python
# _interleaved_half 的 phase 0 每个 K-tile 都做一次 VMEM buffer_load 拿 A-scale
_raw_as = buffer_ops.buffer_load(sx_rsrc, _a_scale_bases[_mi_p] + _k_off, ...)
```

**影响**：mxfp4 把 A-scale "K_TILES × kSubBlocks" 次 VMEM 集中到 run_one 头部一次性发起，靠 buffer_load_lds 直通 LDS 不占 VGPR；FlyDSL 每个 K-tile 都重新从 VMEM 取一遍，并经 VGPR 再写 LDS。28 个 K-tile × m_repeat_packed × VMEM 发射 ≈ 80+ 次 VMEM ops 散在 main loop 里，会和 B-load 抢 vmcnt 槽位。

### 3.2 AGPR 使用（VGPR 压力）

**mxfp4** (`gemm1_a4w4.cuh:49,352-361`)：
```cpp
constexpr bool kUseAGPR = (BM == 128);
// BM=128 时初始化用 mfma_f4f4_agpr_init_zero<…>(accm, …)
// 把累加器放进 AGPR，释放 VGPR 给 B-load / A-load / scale 用
```
`__launch_bounds__(256, 1)` —— BM=128 时每 CU 1 个 block，让出几乎全部寄存器。

**FlyDSL** (`mixed_moe_gemm_2stage.py:1257-1285`)：
直接调 `rocdl.mfma_scale_f32_16x16x128_f8f6f4` 写回 VGPR `acc_gate[]`/`acc_up[]`，**没有 AGPR 切换**。CSV 行 `_w2` 表示 `waves_per_eu=2` → 8 wave per CU = 2 blocks per CU，每 block 寄存器更紧。两份累加器（gate + up）各占 `m_repeat × num_acc_n × f32x4 = 8×2×4 = 64` VGPR，合计 128 VGPR 只给累加器，留给 B-load 双缓冲 / A-LDS 读寄存器的余地很少 —— spill 风险大或 occupancy 被压低。

### 3.3 gate/up 是 "block 内双流" vs "block 间分摊"

**mxfp4**：N_OUT = 2·inter_dim = 1024，gate 和 up 在 N 维拼起来。`num_n_blocks = 4`，每个 block 只算 gate **或** up 中的一段 BN=256 输出 —— 单 accumulator，单 B 流。

**FlyDSL**：tile_n=128（只覆盖 gate side 的一段 128 列），但同 block 用 `acc_gate / acc_up` 两个独立累加器，B 加载也分 `_b_gate_all / _b_up_all` 两路。

后果：
1. 同 block 内 **B 的 VMEM 流量翻倍**（gate B + up B，每 K-tile 各 4 × 16 B = 128B per thread → 总 32 KB per K-tile per block，和 mxfp4 一致；但 mxfp4 是 1 个 256 宽 N，FlyDSL 是 2 个 128 宽 N 各发独立 VMEM 指令 → 调度更碎）。
2. MFMA 里 `compute_bmajor_mfma_phase` 一对 (gate_mfma, up_mfma) 共享同一个 A 操作数 —— 看起来 A 复用率更高。但因为同时维护 2 个 acc，pipeline 里 **prev_gate_w / prev_up_w / prev_gate_bs / prev_up_bs** 的 register live-range 都翻倍 → VGPR 高峰大幅上升。

mxfp4 把 gate/up 拆成不同 block 实质上把 register pressure 摊开到不同 CU。

### 3.4 XCD swizzle & launch_bounds

| | mxfp4 | FlyDSL |
|---|---|---|
| BM=128 launch_bounds | `__launch_bounds__(256, 1)` —— per-CU 1 block，最大寄存器 | `waves_per_eu=2`（CSV `_w2`），4 waves/block × 2 = 8 waves/CU = 2 blocks/CU |
| XCD swizzle | `kXcdSwizzle=0`（BM=128 instance 没用 swizzle） | `xcd_swizzle=4` → 走 `remap_xcd_grouped` 8-XCD remap（`mixed_moe_gemm_2stage.py:496-527`） |

更激进的 occupancy（FlyDSL 2 blocks/CU）正常能换 latency hiding，但前提是寄存器够；当 B 双流 + A scale 在 main loop 反复取 + VGPR 累加器同时存在时，更可能 spill。

mxfp4 BM=128 直接放弃 XCD swizzle —— K2.5 这个形状（NE=385, 多 expert）走自然 m_block_idx 顺序 cache locality 已经够；FlyDSL 还启用 xcd_swizzle=4，是否带来好处需要进一步 attr / nvprof 验证。

### 3.5 调度原语

**mxfp4** BM=128 路径（`gemm1_a4w4.cuh:529-543`）：
```cpp
if constexpr (BM != 128) {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
}
issue_mfma_cluster<J, ...>(slot_b);
if constexpr (BM != 128) { __builtin_amdgcn_s_setprio(0); }
__builtin_amdgcn_sched_barrier(0);
issue_b_load_j<K_C>(b[slot_b], J);
__builtin_amdgcn_sched_barrier(0);
```
**BM=128 关闭了 setprio**，让编译器自己排 MFMA / VMEM 重叠（依赖 hipcc 的 LLVM scheduler 对 1-block-per-CU 寄存器布局做整体优化）。

**FlyDSL**（`mixed_moe_gemm_2stage.py:1421-1462`）：
所有 BM 共用 `sched_barrier(0) + s_setprio(1)/(0)` 包住 MFMA，**手动钉死 phase 顺序**。在 BM=128 这种"寄存器富裕单 block 模型" 上反而限制了 LLVM 调度器自由度。

### 3.6 inline-quant 缺失（影响 stage 间 overhead）

mxfp4 在 BM=16/32 路径上有 `kInlineQuant=true`，把 hidden_states → fp4 量化 **熔进 gemm1 第一层**（`gemm1_a4w4.cuh:197-310`），完全消掉外部的 sort/quant kernel。FlyDSL 任何 BM 都依赖外部 `opus_moe_sorting_entry + fused_mx_quant_moe_sort_kernel` 两个 kernel —— 小 M 多 11–20us，大 M 因为 moe_sort 还会 launch 4 次/iter 多 47–76us。

虽然 BM=128 这条路径 mxfp4 自己也用 sort_quant_fused（trace 显示 0us，但是其实它把这部分塞进了 gemm1 内部），但**结果是 FlyDSL 多了一个 `fused_mx_quant_moe_sort_kernel` 跨 kernel 同步**，对 cudagraph 调度也不友好。

---

## 4. 量化的差距来源（M=16384，per iter）

```
              fly       mx     Δ      原因
gemm1        784.3    678.6  +105.8  3.1/3.2/3.3/3.5 综合 ~15% 慢
gemm2        908.7    849.3  + 59.3  同上结构问题
moe_sort      76.1      —    + 76.1  3.6（mx 内联到 gemm 里）
quant          —        —      —     bm=128 两边都不外置
sum overhead 1769     1528   +241    GPU 净 kernel 时间
launch ovhd   556      295   +261    fly 6 个 kernel/iter vs mx 2 个/iter
bench wall   2325     1823   +502    端到端
```

---

## 5. 可能的优化方向（待验证）

按预期 ROI 排序：

1. **A-scale hoist 到 LDS**（3.1）：把 28 个 K-tile 的 A-scale 在 run_one 头部一次性 `buffer_load_lds` 进 LDS，main loop 只做 LDS read。预估省 ~30-50us / iter（对 BM=128 主循环 vmcnt 压力极大）。
2. **AGPR 累加器**（3.2）：BM=128 时把 `acc_gate / acc_up` 放进 AGPR，释放 VGPR 给 B 双流和 prefetch。FlyDSL 已有 AGPR 支持的话直接打开即可；没有就是较大工程。
3. **去掉 `_w2` 改 `waves_per_eu=1` + 试试 launch_bounds(256, 1)**（3.2）：和 mxfp4 对齐，单 block per CU 试一把。需要寄存器分析配合。
4. **gate/up 改成 N-split**（3.3）：把 grid 扩成 `n_blocks = 2 × inter_dim / tile_n`，每个 block 只算 gate 或 up。register 压力立刻降一半，编译器调度也变简单。这是结构调整，工作量大但对齐 mxfp4 设计思路。
5. **关掉 `xcd_swizzle=4`**（3.4）：对照实验，确认 swizzle 是否真带来收益；如果无影响就关掉减少地址计算。
6. **去掉 `s_setprio` for BM=128**（3.5）：mfma 簇前后只留 `sched_barrier(0)`，把调度决定让给 LLVM。代价低。
7. **inline-quant for BM≤32**（3.6，和大 M 性能无关，但能改善小 M）：在 stage1 头部熔进 hidden_states→fp4 量化，消掉一个独立 kernel。

---

## 6. 复现 / 后续

- bench：`python ~/FlyDSL/bench.py 2>&1 | grep "| "`
- trace 落在工作目录 `1.json.gz` ~ `18.json.gz`（奇数 = fly，偶数 = mx，按 M 顺序 9 对）
- 分析脚本：`/root/analyze.py`、`/root/timing.py`、`/root/kernels_per_iter.py`
- 进一步定位单 kernel 瓶颈：可用 `/capture-kernel-trace` + `/kernel-trace-analysis` 跑 ATT trace 把 fly 的 `mfma_moe1_..._t128x128x256_pm1_async_xcd4_v32` 单独抓出来，对比 VMEM-wait/lgkmcnt/MFMA stall 分布。
