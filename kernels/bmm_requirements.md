# DeepSeek-V4 Grouped Output Projection BMM — 需求文档

**日期**：2026-05-18  
**状态**：Draft，待模型侧对齐确认  

---

## 1. 背景

DeepSeek-V4 的 Attention 模块在标准 MHA 输出投影（单一 W_o）的基础上，引入了 **Grouped Output Projection**：将 128 个 attention head 分成 16 组，每组独立做低秩下投影，再拼接做上投影。这是 V4 相对 V3 新增的结构，需要独立的 BMM kernel 支持。

当前 aiter 团队的任务：**针对 MI450（gfx1250）优化该 BMM 算子**。

---

## 2. 算子定义

### 2.1 数学形式

```python
# 来源：model.py L537-541
o    = o.view(num_tokens, n_groups, -1)              # [T, G, D]
wo_a = self.wo_a.weight.view(n_groups, o_lora_rank, -1)  # [G, R, D]
out  = torch.einsum("sgd,grd->sgr", o, wo_a)         # [T, G, R]
```

等价于 **G 个独立 GEMM**，每个：

```
[T, D] @ [D, R]^T  →  [T, R]
```

### 2.2 形状参数（V4-Pro config.json）

| 参数 | 值 | 说明 |
|---|---|---|
| G | 16 | n_groups（TP=1） |
| D | 4096 | 每组输入维度 = (128 heads / 16 groups) × 512 head_dim |
| R | 1024 | o_lora_rank（输出秩） |
| T | **1 ～ 65536** | num_tokens，decode 时 T 极小，prefill 时最大 64k |

### 2.3 数据类型（已确认）

| 张量 | 存储 dtype | MMA dtype | 说明 |
|---|---|---|---|
| o（输入激活） | bf16 | bf16 | a16 |
| wo_a（权重） | **fp8** | bf16（dequant 后） | w8，checkpoint 以 fp8 存储 |
| out（输出） | bf16 | — | |
| 累加器 | fp32 | — | 标准 bf16 WMMA 累加 |

> **MMA 路径**：fp8 weight 从 HBM 读入后 dequant → bf16，再走 bf16 WMMA。不走 fp8 MMA（activation 保持 bf16 精度）。

### 2.4 后续算子

wo_a BMM 之后紧接：
```python
x = self.wo_b(out.flatten(2))  # [T, G*R=16384] @ [16384, 7168] → [T, 7168]
```
wo_b 是标准 GEMM，不在本需求范围内，但 **fuse wo_a + wo_b** 是值得评估的优化选项。

---

## 3. 范围界定

### 3.1 在范围内

- Grouped Output Projection：`einsum("sgd,grd->sgr")`，即本文描述的 BMM
- 目标硬件：MI450 / gfx1250

### 3.2 不在范围内

- Attention 内部 GEMM（Q@K^T、P@V）：属于 sparse_attn kernel，由 attention team 负责
- wo_b（上投影）：标准 GEMM，独立优化
- Lightning Indexer scoring（einsum "bshd,btd->bsht"）：另一个 BMM，本期不涉及
- MoE expert FFN：Grouped GEMM，独立需求

---

## 4. 硬件参数（MI450，ubench 实测）

| 指标 | 5/12 | 5/13 |
|---|---|---|
| HBM read BW | 15.6 TB/s | 19.6 TB/s |
| HBM r/w BW | 15.5 TB/s | 18.5 TB/s |
| bf16 WMMA peak | 4067 TFLOPS | 4200 TFLOPS |
| fp8 WMMA peak | 14541 TFLOPS | 15036 TFLOPS |

**wo_a 权重 footprint**：G×R×D × 1B(fp8) = 16×1024×4096 = **67.1 MB**

---

## 5. 计算模型分析

### 5.1 Roofline（bf16 MMA，5/13 数据）

| T | AI (FLOP/B) | bound | roofline 时间 | MMA 利用率 |
|---|---|---|---|---|
| 1 | 2.0 | memory | 3.43 µs | < 1% |
| 64 | 110.7 | memory | 3.96 µs | 52% |
| 128 | 195.0 | memory | 4.49 µs | 91% |
| **145** | **~214** | **crossover** | — | — |
| 256 | 315.1 | compute | 8.18 µs | 68% |
| 1024 | 585.1 | compute | 32.72 µs | 37% |
| 65536 | 814.1 | compute | 2094 µs | 26% |

**AI 渐近值（T→∞）= 819.2 FLOP/B**（= 2×R×D / (D×2B + R×2B)）

### 5.2 分区间结论

**Decode（T ≤ 64）：memory-bound**
- 瓶颈：67MB wo_a 每次 forward 从 HBM 搬入，MMA 几乎空转
- 优化目标：打满 HBM BW（19.6 TB/s），让实际耗时逼近 3.43µs roofline

**Prefill（T ≥ 256）：compute-bound**
- bf16 WMMA 在大 T 下利用率仅 26~68%，仍有提升空间
- 优化目标：提高 WMMA occupancy，选合适 tile size

---

## 6. 优化方向

### 6.1 Decode（T = 1 ～ 64）

**核心问题**：T=1 时退化为 16 个独立 GEMV（每个 [1,4096]×[4096,1024]），G 维度无法合并为单一 dense GEMM（各组权重矩阵独立）。

**方向 A：Cluster Async 覆盖宽 N**
- 不在 G、S 维度上安排 warp，cluster 在 N 方向展开（cluster_width=16，每 WG 处理 N/16=64 个输出）
- 16 个 WG 同时向 HBM 发请求 → 并发度提升 16x → HBM 利用率从 ~60% 提升至 ~90%
- Cluster multicast：WG-0 用 TENSOR_LOAD_TO_LDS 将 o[g,:] (8KB) broadcast 到所有 cluster 成员，节省 (C-1)×8KB 的重复输入读取（小但免费）
- 注：总字节数不变，收益来自 HBM 并发度，非字节减少

**方向 B：GL2 Weight Caching（intra-kernel）**
- 加载 wo_a 时设 NV=1 + TH=HT：barrier 不刷 NV=1 cacheline，weight tile 跨 K-loop iteration 留在 GL2
- **跨 kernel dispatch 无法 pin**：ROCm runtime dispatch 间 GCR 会清掉 NV=1 lines，gfx1250 没有类似 CUDA L2 residency 的硬件 pin 机制

**方向 C：Fuse wo_a + wo_b**
- 消掉 wo_a 和 wo_b 之间的独立 kernel dispatch，省去一次 HBM round-trip（out 张量的写入和读取）
- out [T, G, R] = [T, 16384]（bf16）= T×32KB，T=1 时 32KB，T=64 时 2MB，fuse 收益随 T 增大

**方向 D：Multi-request batching**
- 等 batch 攒够 T≥16，M 维度才能填满 WMMA tile（WMMA 最小 M=16）
- 需要推理框架侧（vLLM/SGLang）配合做 continuous batching

### 6.2 Prefill（T = 256 ～ 65536）

**方向 A：Tile size 优化**
- 当前 compute-bound，WMMA 利用率 26~68%，主要损失来自 tile 选取不当导致 pipeline stall
- 目标：M×N×K tile 选取使 V_NOP hazard 最小、LDS bank conflict 为零、TDM prefetch 深度合适

**方向 B：K-loop N=3 pipeline**
- 参考 GEMM checklist：K-loop 内 N=3 stage（prefetch + compute + epilogue 重叠）
- wo_a 按列（K 方向）分块，async TDM load 掩盖 HBM latency

**方向 C：fp8 dequant overhead 评估**
- fp8→bf16 dequant 是否成为瓶颈（per-group scale 需要额外读取）
- Scale footprint：67MB / 128（group_size）× 4B(fp32) ≈ 2MB，可忽略

---

## 7. 开放问题（需对齐）

| # | 问题 | 影响 |
|---|---|---|
| Q1 | 实际部署的 decode batch size 分布？T=1 是主路径还是 T=8/16/32？ | 决定 decode 优化的优先级和 cluster_width 选择 |
| Q2 | wo_a 和 wo_b 是否可以 fuse？框架侧是否支持 fused epilogue？ | 影响优化收益的上限 |
| Q3 | fp8 dequant 的 scale 粒度？per-128 还是 per-tensor？ | 影响 dequant overhead 和 weight load pattern |
| Q4 | TP > 1 时 G = n_groups / world_size，最小 G=1？需覆盖此场景 | 影响 cluster 设计（G=1 时退化为单 GEMV） |

---

## 8. 交付物（TBD）

- [ ] MI450 baseline kernel（a16w8，grouped GEMM，无优化）+ 精度测试
- [ ] Decode 优化 kernel（cluster async，NV=1）+ 性能对比
- [ ] Prefill 优化 kernel（tile/pipeline 调优）+ 性能对比
- [ ] Fuse wo_a+wo_b 评估（可行性 + 收益估算）
