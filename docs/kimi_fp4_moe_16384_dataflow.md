# Kimi FP4 MoE 16384 FlyDSL Dataflow

这份文档只解释 `kimi_fp4_moe_16384.py` 里固定 M=16384 这条 FlyDSL 路径的 GEMM1 和 GEMM2 数据流。重点是 kernel tiling: 每个 CTA/wave 负责哪块 C，A/B 怎么搬，K loop 怎么走，以及按 tile 之后一次 forward 大概要发多少 A/B load。

## 固定形状

| 项 | 值 |
| --- | ---: |
| `TOKEN` | 16384 |
| `TOPK` | 9 |
| routed rows, no padding | `TOKEN * TOPK = 147456` |
| `EXPERTS` | 385 |
| model dim | 7168 |
| intermediate dim | 512 |
| sort block | 64 |
| CTA threads | 256 |
| waves per CTA | 4 |

所有 A/W 都是 fp4 packed 成 `fp4x2` byte。代码里很多地方的 `tile_k=256` 表示 256 个 fp4 K 元素；按 global memory 字节算时要除以 2。

本文的数据量默认是每个 active CTA 自己发出的 logical traffic，不扣 L2/cache 命中，也不计很小的 sorted id、scale、topk weight 读流量。真实 DRAM traffic 会被 cache、expert 分布、padding 和 guard 影响。

## FlyDSL 数据流术语

这份 kernel 的主数据路径可以按下面几层看：

| 动作 | FlyDSL/ROCDL 位置 | 含义 |
| --- | --- | --- |
| global pointer 变成 resource | `_ptr_buffer_resource` | 给 `buffer_ops` / raw buffer load 用 |
| A global -> LDS | `rocdl.raw_ptr_buffer_load_lds` in `dma_x_tile_to_lds` | 256 threads 合作把一个 A tile 搬进 ping/pong LDS |
| A LDS -> VGPR | `vector.load_op` in `lds_load_packs_k64` | 每个 wave 从 LDS 读它要喂给 MFMA 的 A subtile |
| B global -> VGPR | `_buffer_load_vec` in `load_b_packs_k64` | B 不进 LDS，按当前 wave 的 N slice 直接进寄存器 |
| scales global -> VGPR | `buffer_ops.buffer_load` in `prefetch_ab_scale_tile` | e8m0 scale，给 `mfma_scale` 用 |
| matrix compute | `rocdl.mfma_scale_f32_16x16x128_f8f6f4` | 真正发 MFMA 指令的地方 |
| C fragment -> LDS | `write_row_to_lds` | 把 lane-fragment 格式的 accumulator 整理进 LDS |
| LDS -> global C | `c_shuffle_epilog(... store_pair ...)` | C shuffle 后用 `llvm.StoreOp` 写回 global |

直观理解：A 放 LDS 是因为同一个 A tile 会被 4 个 waves 分别乘不同 N slice 的 B；B 是每个 wave 独有的 N slice，直接进寄存器更合适。

## GEMM1: Stage1 Tiling

Stage1 做的是:

```text
gate = hidden @ W_gate
up   = hidden @ W_up
out  = silu(gate) * up
```

输出是 bf16 intermediate:

```text
[TOKEN, TOPK, INTER_DIM] = [16384, 9, 512]
```

固定 tile:

| 项 | 值 |
| --- | ---: |
| `tile_m` | 64 routed rows |
| `tile_n` | 128 intermediate columns |
| `tile_k` | 256 model-dim fp4 elements |
| K tiles | `7168 / 256 = 28` |
| N tiles | `512 / 128 = 4` |
| waves | 4 |
| `n_per_wave` | `128 / 4 = 32` |
| `m_repeat` | `64 / 16 = 4` |
| `num_acc_n` | `32 / 16 = 2` |
| `k_unroll` | `256 / 128 = 2` MFMA K chunks |

### CTA 和 Wave 负责的 C 块

一个 Stage1 CTA 负责一个 logical C tile:

```text
M: 64 sorted routed rows
N: 128 intermediate columns
K: loop over 7168
```

4 个 waves 平分 N 维：

| wave | N columns |
| ---: | --- |
| 0 | `by_n + 0..31` |
| 1 | `by_n + 32..63` |
| 2 | `by_n + 64..95` |
| 3 | `by_n + 96..127` |

每个 wave 内部再拆成 16x16 MFMA tile：

```text
M side: 4 blocks of 16 rows  -> mi_idx = 0..3
N side: 2 blocks of 16 cols  -> ni_idx = 0..1
```

所以每个 wave 同时维护:

```text
4 * 2 = 8 accumulator fragments for gate
4 * 2 = 8 accumulator fragments for up
```

`acc_idx = mi_idx * num_acc_n + ni_idx`。每个 accumulator 是一个 `vec4_f32` lane fragment，整个 wave 合起来表示一个 16x16 C micro-tile。K loop 结束后，gate/up accumulator 先做 `silu(gate) * up`，再进入 C shuffle epilogue。

### GEMM1 Per K Tile 数据流

一个 Stage1 CTA 的一个 K tile 是:

```text
A: 64 x 256 fp4  -> 64 * 256 / 2 = 8192 B = 8 KiB
B_gate: 128 x 256 fp4 -> 16 KiB
B_up:   128 x 256 fp4 -> 16 KiB
```

也就是每个 K tile:

| 数据 | 路径 | 每 CTA footprint |
| --- | --- | ---: |
| A | global -> LDS | 8 KiB |
| A | LDS -> VGPR, all waves total | 32 KiB LDS reads |
| B gate | global -> VGPR | 16 KiB |
| B up | global -> VGPR | 16 KiB |
| B total | global -> VGPR | 32 KiB |

A 的 global load 是整个 CTA 合作完成的。`_num_dma_loads = 2`，所以每个 thread 发 2 次 16B raw LDS load:

```text
256 threads * 2 * 16 B = 8192 B
```

B 的 load 是按 wave 的 N slice 发的。每个 wave 每个 K tile 的 B footprint:

```text
gate: 32 N cols * 256 K / 2 = 4 KiB
up:   32 N cols * 256 K / 2 = 4 KiB
total per wave = 8 KiB
```

4 waves 加起来就是 32 KiB。

### GEMM1 MFMA 数量

每个 wave 每个 K tile:

```text
m_repeat(4) * num_acc_n(2) * k_unroll(2) * gate/up(2)
= 32 MFMA
```

每个 CTA 每个 K tile:

```text
4 waves * 32 = 128 MFMA
```

整个 K=7168:

```text
28 K tiles * 128 = 3584 MFMA per CTA
```

### GEMM1 K Loop

Stage1 的 K 维有 28 个 tile。代码里的关键变量是:

```text
num_k_tiles_py = 28
tail_tiles = 2
k_main2_py = (28 - 2) * 256 = 6656
```

主循环:

```text
for k_iv_py in range(0, 6656, 512):
    _interleaved_half(...)
    _interleaved_half(...)
```

一个 `_interleaved_half` 可以理解成:

```text
compute previous K tile
load current K tile's B and scales into VGPR
read current K tile's A from LDS into VGPR
DMA a future A tile into the other LDS buffer
return current tile as next half's previous tile
```

启动阶段先准备:

```text
K0 A -> lds_x_pong
K0 scales
K1 A -> lds_x_ping
K0 B gate/up -> regs
K0 A LDS -> regs
```

主循环的 26 个 half 会计算 K0..K25，并持续把未来的 A tile 填到 ping/pong LDS。最后 tail 处理 K26 和 K27:

```text
compute K26
compute K27
```

所以 Stage1 每个 CTA 总 global load footprint 约为:

| 数据 | 公式 | 每 CTA |
| --- | --- | ---: |
| A | `28 * 8 KiB` | 224 KiB |
| B gate+up | `28 * 32 KiB` | 896 KiB |
| C store | `64 * 128 * 2 B` | 16 KiB |
| 合计, 不含 scales/sorted ids |  | 1136 KiB |

### GEMM1 写回

每个 wave 得到的是 MFMA lane-fragment layout，不是 contiguous C layout。写回分两步：

1. `write_row_to_lds` 把 `acc[mi, ni]` 中的 f32 转 bf16，按 row/col 写到 `lds_out`。
2. `c_shuffle_epilog` 让线程从 LDS 读出连续片段，再由 `store_pair` 用 `llvm.StoreOp` 写到 global。

Stage1 的 global out row 是 `(token_id, topk_id)` 对应的 `ts_idx = token * TOPK + topk`，列是 `INTER_DIM` 里的 `by_n + local_n`。padding row 会被 `row_valid` guard 掉。

## GEMM2: Stage2 Tiling

Stage2 做的是:

```text
out_topk_slot = intermediate @ W_down
out_topk_slot *= topk_weight
```

当前实现先写一个很大的 bf16 staging buffer:

```text
target: [TOKEN, TOPK, MODEL_DIM]
```

然后 Python wrapper 里再做:

```python
torch.sum(target.view(TOKEN, TOPK, MODEL_DIM), dim=1, out=out)
```

固定 tile:

| 项 | 值 |
| --- | ---: |
| `tile_m` | 64 routed rows |
| `tile_n` | 256 model columns |
| `tile_k` | 256 intermediate fp4 elements |
| K tiles | `512 / 256 = 2` |
| N tiles | `7168 / 256 = 28` |
| waves | 4 |
| `n_per_wave` | `256 / 4 = 64` |
| `m_repeat` | `64 / 16 = 4` |
| `num_acc_n` | `64 / 16 = 4` |
| `k_unroll` | `256 / 128 = 2` MFMA K chunks |

Stage2 是 persistent-M grid:

```text
grid.x = 28 N tiles
grid.y = CU count
```

每个 launched block 不是只做一个 fixed M tile，而是在 `scf.ForOp` 里按 CU 分配到多个 sorted M tiles。逻辑上仍然可以把它理解成很多 `(M tile, N tile)` CTA work items:

```text
M tile = 64 sorted routed rows
N tile = 256 model columns
```

### CTA 和 Wave 负责的 C 块

一个 logical Stage2 CTA 负责:

```text
M: 64 sorted routed rows
N: 256 model columns
K: 512 intermediate dim
```

4 个 waves 平分 N 维：

| wave | N columns |
| ---: | --- |
| 0 | `by_n + 0..63` |
| 1 | `by_n + 64..127` |
| 2 | `by_n + 128..191` |
| 3 | `by_n + 192..255` |

每个 wave 内部:

```text
M side: 4 blocks of 16 rows -> mi_idx = 0..3
N side: 4 blocks of 16 cols -> ni_idx = 0..3
```

所以每个 wave 维护:

```text
4 * 4 = 16 accumulator fragments
```

GEMM2 没有 gate/up 双 accumulator，只有一套 `acc`。

### GEMM2 Per K Tile 数据流

一个 Stage2 CTA 的一个 K tile:

```text
A: 64 x 256 fp4 -> 8 KiB
B: 256 x 256 fp4 -> 32 KiB
```

| 数据 | 路径 | 每 CTA footprint |
| --- | --- | ---: |
| A | global -> LDS | 8 KiB |
| A | LDS -> VGPR, all waves total | 32 KiB LDS reads |
| B | global -> VGPR | 32 KiB |

和 Stage1 对比：

```text
Stage1: tile_n=128, 但要同时读 gate/up 两套 B -> 32 KiB B per K tile
Stage2: tile_n=256, 只读一套 down B       -> 32 KiB B per K tile
```

所以两者每个 K tile 的 B global footprint 恰好一样大。

### GEMM2 MFMA 数量

每个 wave 每个 K tile:

```text
m_repeat(4) * num_acc_n(4) * k_unroll(2)
= 32 MFMA
```

每个 logical CTA 每个 K tile:

```text
4 waves * 32 = 128 MFMA
```

整个 K=512:

```text
2 K tiles * 128 = 256 MFMA per logical CTA
```

### GEMM2 K Loop

GEMM2 的 K 只有两个 tile，所以没有长 main loop:

```text
num_k_tiles = 2
tail_tiles = 2
k_main2_py = 0
```

实际路径是 even-tail:

1. load K0 的 B low half、scales。
2. DMA K0 的 A 到 `lds_x_pong`。
3. 从 `lds_x_pong` 预取 K0 的 A subtile。
4. DMA K1 的 A 到 `lds_x_ping`。
5. load K1 的 B、scales。
6. compute K0。
7. wait/barrier。
8. 从 `lds_x_ping` 读 K1 的 A subtile。
9. compute K1，并预取 epilogue 需要的 `sorted_weights`。

每个 logical CTA 总 global load/store footprint 约为:

| 数据 | 公式 | 每 logical CTA |
| --- | --- | ---: |
| A | `2 * 8 KiB` | 16 KiB |
| B | `2 * 32 KiB` | 64 KiB |
| C staging store | `64 * 256 * 2 B` | 32 KiB |
| 合计, 不含 scales/sorted ids/topk weights |  | 112 KiB |

### GEMM2 写回和 TOPK Reduce

GEMM2 epilogue 在 `write_row_to_lds` 里多做了一步:

```text
v = accumulator * sorted_weight
```

然后和 Stage1 一样走 `c_shuffle_epilog` 写回 global。不过它写回的不是最终 `[TOKEN, MODEL_DIM]`，而是 topk slot staging:

```text
target[token, topk, model_col]
```

最后 `torch.sum(target.view(TOKEN, TOPK, MODEL_DIM), dim=1)` 再把 9 个 topk slot 合并。这是当前 FlyDSL 路径最显眼的数据流开销之一：

| buffer | 大小 |
| --- | ---: |
| Stage1 intermediate `[16384, 9, 512]` bf16 | 144 MiB |
| Stage2 staging `[16384, 9, 7168]` bf16 | 2016 MiB |
| `torch.sum` 读取 staging | 2016 MiB |
| `torch.sum` 写最终 out `[16384, 7168]` bf16 | 224 MiB |

这里的 MiB 按 1024 进制算。

## 一次 Forward 的 Tile 级 Traffic 公式

令:

```text
M_tiles = ceil(num_valid_ids[0] / 64)
```

`num_valid_ids[0]` 由 sort kernel 产生，可能包含为了 expert/block 对齐产生的 padding。没有 padding 的理论下界是:

```text
ceil(16384 * 9 / 64) = 2304
```

当前 wrapper 分配的 `sorted_token_ids` 上界接近:

```text
max_sorted = TOKEN * TOPK + EXPERTS * BLOCK_M - TOPK = 172087
floor(max_sorted / 64) = 2688 blocks
```

真实 active work 由 `num_valid_ids[0]`、`exp_valid`、`tile_has_tokens` 和 row guard 共同决定。

### Stage1

Stage1 logical CTA 数:

```text
CTA_s1 = M_tiles * 4
```

因为 N tile 数是 4。

Traffic 公式:

```text
A_s1 = CTA_s1 * 28 * 8 KiB
B_s1 = CTA_s1 * 28 * 32 KiB
C_s1 = CTA_s1 * 16 KiB
```

如果只按没有 padding 的 2304 个 M tiles 算:

| 项 | 约 GiB |
| --- | ---: |
| A global load | 1.97 |
| B global load, logical | 7.88 |
| C store | 0.14 |
| total | 9.98 |

如果按 2688 个 M tiles 上界算:

| 项 | 约 GiB |
| --- | ---: |
| A global load | 2.30 |
| B global load, logical | 9.19 |
| C store | 0.16 |
| total | 11.65 |

### Stage2

Stage2 logical CTA 数:

```text
CTA_s2 = M_tiles * 28
```

因为 N tile 数是 28。

Traffic 公式:

```text
A_s2 = CTA_s2 * 2 * 8 KiB
B_s2 = CTA_s2 * 2 * 32 KiB
C_s2 = CTA_s2 * 32 KiB
```

如果只按没有 padding 的 2304 个 M tiles 算:

| 项 | 约 GiB |
| --- | ---: |
| A global load | 0.98 |
| B global load, logical | 3.94 |
| C staging store | 1.97 |
| total | 6.89 |

如果按 2688 个 M tiles 上界算:

| 项 | 约 GiB |
| --- | ---: |
| A global load | 1.15 |
| B global load, logical | 4.59 |
| C staging store | 2.30 |
| total | 8.04 |

Stage2 后面的 `torch.sum` 还会额外读约 1.97 GiB staging，并写约 0.22 GiB final output。

## 读代码时的最短路径

GEMM1 建议按这个顺序看：

1. `Stage1Config`: 确认 `tile_m/tile_n/tile_k`。
2. `compile_kimi_fp4_stage1_16384`: 看 derived constants，比如 `n_per_wave`、`m_repeat`、`k_unroll`。
3. `moe_gemm1`: 看 grid/block id 怎么映射到 `by_n` 和 `bx_m`。
4. `dma_x_tile_to_lds`: 看 A global -> LDS。
5. `load_b_packs_k64`: 看 B global -> VGPR。
6. `compute_bmajor_mfma_phase`: 看 MFMA 真正在哪里发。
7. `_interleaved_half`: 看 K loop pipeline。
8. `write_row_to_lds` + `c_shuffle_epilog`: 看 accumulator 怎么变成 global C。

GEMM2 建议按这个顺序看：

1. `Stage2Config`: 确认 `64x256x256` 和 `persist_m=-1`。
2. `compile_kimi_fp4_stage2_16384`: 看 persistent grid 和 derived constants。
3. `_moe_gemm2_then_body`: 看一个 logical M tile 的实际工作。
4. `load_b_tile_lo/load_b_tile_hi`: 看 B split load。
5. `compute_tile`: 看 A LDS read、B pack、scale、MFMA。
6. `write_row_to_lds`: 看 topk weight 乘在哪里做。
7. wrapper `kimi_fp4_stage2_16384`: 看 `target` staging 和最终 `torch.sum`。

## 最重要的性能理解点

1. 每个 CTA 的 A global load 很小，但同一个 M tile 会被不同 N tiles 重复加载到 LDS。Stage1 重复 4 次，Stage2 重复 28 次。
2. B global logical traffic 比 A 大很多，尤其 Stage1 的 28 个 K tiles 每 CTA 要读 896 KiB 的 gate/up B。实际 DRAM 是否这么大取决于 expert 分布和 L2 复用。
3. Stage1 和 Stage2 每个 K tile 每个 CTA 都是 128 条 wave-level MFMA，但 Stage1 有 28 个 K tiles，Stage2 只有 2 个 K tiles。
4. 当前 Stage2 的 GEMM 本身不是唯一问题。它写 bf16 `[TOKEN, TOPK, MODEL_DIM]` staging，再用 PyTorch reduce，这会带来约 2 GiB 级别的额外读写。
5. 对齐 mxfp4 性能时，除了 GEMM tile 本身，还要对齐 Stage2 输出格式和 reduce 路径，否则 `torch.sum` 这段数据流会一直留在 critical path 上。
