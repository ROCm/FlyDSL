# bmm_a16w8_gfx1250 性能分析与 Silicon Benchmark 指南

## 模拟器 vs 真机的区别

| 工具 | 用途 | 不能做 |
|------|------|--------|
| **ffmlite** | 精度验证（atol/rtol） | 性能 benchmark（数字无意义） |
| **AM** | cycle-accurate timing | 快速迭代（>10min 启动，串行执行 WG） |
| **silicon** | 一切 | — |

AM 170k sclk 的来源：AM 串行执行 8192 个 WMMA op（无 WG 并行）≈ 170k cycles。
不代表真机延迟，不可用于对比 hardware peak。

---

## V4 Decode 形状的并发度分析

形状：`B=16, M=64, N=1024, K=4096`，`tile_m=64, tile_n=128, tile_k=128`

```
Grid:  gx=1 × gy=8 × gz=16 = 128 WGs
MI450: 304 CU = 152 WGP (SPX 模式)
WGP 利用率: 128 / 152 = 84%  → 不是瓶颈
```

### LDS 限制（为什么 split-K 没用）

```
lds_a_bytes = 64 × (128+8) × 2B = 17,408 B
lds_b_bytes = 128 × (128+8) × 1B = 17,408 B
每 stage: ~35 KB,  3 stages: 105 KB/WG

gfx1250 LDS = 128 KB / WGP
105 KB > 64 KB → 1 WG per WGP，放不下第 2 个
```

**Split-K 无效**：split-K 靠多 WG 并行 K 维，但 LDS 已锁定 1 WG/WGP，
新 WG 只能排队，不缩短延迟。且 waves_per_eu 调整也无效（LDS 才是瓶颈，不是 VGPR）。

### HBM Traffic（含 cluster_n=8 multicast）

```
A (cluster multicast ÷8): 16 × 64 × 4096 × 2B / 8 = 1 MB
B (fp8, 无 multicast):    16 × 4096 × 1024 × 1B    = 64 MB
C (bf16 write):           16 × 64 × 1024 × 2B       = 2 MB
Total:                                               ≈ 67 MB
```

### L2 Cache 效应（关键）

MI450 SPX 模式 GL2 = **192 MB**。B 矩阵 = 64 MB，**完全放得进 L2**。

```
Cold call (首次):  67 MB @ 8 TB/s ≈ 8 µs 理论下限
Warm call (L2命中): A(1MB) + C(2MB) ≈ 3 MB @ 8 TB/s ≈ 0.4 µs 理论下限
```

**Benchmark 必须测 warm 态**——cold 数字不代表推理服务的 steady-state 性能。

---

## Silicon Benchmark 步骤

### 1. 前置：correctness 验证（ffmlite）

```bash
cd ~/flydsl

# no_scale 精度
python -m pytest tests/kernels/test_bmm_a16w8_gfx1250.py::test_a16w8_no_scale -v

# cluster_n=4 精度
python -m pytest tests/kernels/test_bmm_a16w8_gfx1250.py::test_a16w8_cluster -v
```

两组都过了再上机。

### 2. Warm-up Latency（主 benchmark）

```bash
cd ~/flydsl
python tests/kernels/test_bmm_a16w8_gfx1250.py \
    --B 16 --M 64 --N 1024 --K 4096 \
    --tile-m 64 --tile-n 128 --tile-k 128 \
    --num-buffers 3 \
    --no-scale \
    --cluster-n 8 \
    --iters 100 --warmup 20
```

**对照组**（无 cluster，其余相同）：
```bash
python tests/kernels/test_bmm_a16w8_gfx1250.py \
    --B 16 --M 64 --N 1024 --K 4096 \
    --tile-m 64 --tile-n 128 --tile-k 128 \
    --num-buffers 3 \
    --no-scale \
    --iters 100 --warmup 20
```

记录 **p50 latency**（warm 态），以及 cluster vs 无 cluster 的比值。

理论上 cluster_n=8 把 A HBM 从 8MB→1MB，节省 7MB（总 67MB 减少约 10%）。
如果 B 已 L2 命中，A 减少的影响更大（从 3MB 减到更小）。

### 3. 对比矩阵

| 配置 | cluster_n | scale_mode | 预期收益 |
|------|-----------|-----------|---------|
| baseline | 1 | e8m0 | 基准 |
| cluster | 8 | e8m0 | A HBM −87.5% |
| no_scale + cluster | 8 | no_scale | A HBM −87.5% + scale load 省去 |
| prefill M=256 | 1 | no_scale | 更高 WGP 利用率 |

### 4. 如果结果不符预期：profiling

```bash
# B 是否 L2 命中（验证 warm 假设）
rocprofv3 --counter TCP_TCC_HIT_sum,TCP_TCC_MISS_sum \
    python tests/kernels/test_bmm_a16w8_gfx1250.py --warmup 5 --iters 1 ...

# VMEM stall（验证 pipeline 效果）
rocprofv3 --counter VMEM_STALL_ON_SALU_sum \
    python tests/kernels/test_bmm_a16w8_gfx1250.py ...

# CU 利用率
rocprofv3 --counter SQ_BUSY_CU_CYCLES_sum ...
```

---

## 预期结论

- **Warm 态目标**：latency ≤ 5 µs（T=1 decode single call）
- **Cold miss**：首次调用约 8 µs，不代表服务性能，不作为优化目标
- **Split-K**：不适合此形状，LDS 瓶颈决定了并发上限已经饱和
- **waves_per_eu**：LDS 限制 1 WG/WGP，改 waves_per_eu 无额外收益
- **cluster_n=8 收益**：主要在 cold miss 路径（A HBM 减 7MB）；warm 态收益取决于 A 是否也在 L2

---

## 优化点分析

以下各项按预估收益从大到小排列。参考实现：`wmma_gemm_gfx1250.py`（bf16 GEMM）、`gemm_fp8fp4_gfx1250.py`（WMMA_SCALE GEMM）。

### 1. [最大] 改用 WMMA_SCALE fp8 路径 — 理论 3.5× 算力提升

**现状**：当前计算路径是 fp8→bf16 转换 + bf16 WMMA (WMMA_K=32)：

```
ds_load_tr8_b64 (LDS load fp8)
→ V_CVT_SCALE_PK8_BF16_FP8 (fp8→bf16 转换)
→ [V_PK_MUL_BF16 (residual scale, fp32路径)]
→ V_WMMA_F32_16x16x32_BF16
```

**优化**：gfx1250 的 `V_WMMA_SCALE_F32_16x16x128_F8F6F4` 指令支持 fp8×fp8 fused MAC with E8M0 scale，WMMA_K=128（4× wider）。`gemm_fp8fp4_gfx1250.py` 已完整实现该路径。

- MI450 ubench: fp8 TFLOPS ≈ 15,036 vs bf16 TFLOPS ≈ 4,200 → **3.6× 理论算力**
- WMMA_K=128 → LDS load 次数减少 4×，同时省去 CVT + MUL 的 VALU 开销

**代价**：activation 需要先 cast 成 fp8，有精度损失（mantissa: 7bit→3bit for E4M3）。建议新增 `compile_bmm_a8w8_gfx1250()` 变体，端到端精度验证通过后替换当前路径。

### 2. [中] Wave-specialized TDM

**现状**：所有 wave 都发射 A + B 两个 TDM 指令（`wave_specialized=False`），loop-carry state 包含 3 个值（addr_lo_a, addr_lo_b, addr_hi_b）。

**优化**：`wmma_gemm_gfx1250.py` 已实现 `wave_specialized_tdm=True`——wave 0 专门发 A，wave 1 专门发 B，其余 wave 纯计算。每 wave 从 2 条 TDM 降为 1 条，loop-carry 从 3 个降为 1 个。

预估收益：热路径 10-20%（compute-bound 场景下更显著）。参考实现可直接移植，改动集中在 prologue / main loop / tail 的 TDM 发射逻辑。

### 3. [中] Scale 加载延迟暴露

**现状**：`_load_scale` 在每个 K-tile 的 `pipeline_fence_wait` 之后、`compute_tile` 之前**同步**加载。若 scale 不在 L2，400-600 cycle HBM 延迟完全暴露在关键路径上。

**优化**：把下一个 K-tile 的 scale buffer_load 移入当前 tile 的 `mid_compute_callback`，与 WMMA 并行执行，隐藏 scale 加载延迟。

影响范围：仅默认 fp32 scale 路径；`no_scale` / `use_e8m0_scale` 路径 scale 开销已足够小（常量或 1 byte load），无需修改。

### 4. [小-中] LDS bank conflict — A fragment load

**现状**：`LDS_PAD_A=8` bf16 → `lds_a_stride = 136 elem = 272 bytes`。A frag 用 `ds_load_b128`（128-bit）。wave 内 lane16=0 与 lane16=8 的行地址差 = 8 × 272 = 2176 bytes，bank offset 差 = 2176/4 = 544 ≡ 0 (mod 32)，产生 **2-way bank conflict**。

**修复**：改 `LDS_PAD_A=12` → stride=140 elem=280 bytes → bank offset=70 ≡ 6 (mod 32)，GCD(6,32)=2，消除 2-way conflict。

预估收益：LDS A 读吞吐提升 ~10-15%（仅在 LDS 是瓶颈时有效；decode memory-bound 场景可能不明显）。

### 5. [小] Tile 参数调优与 occupancy

**V4 decode M=64**：默认 `tile_m=128 > M=64`，浪费 50% tile。改 `tile_m=64` 直接消除浪费。

**occupancy**：tile=64×128×128，3-stage：每 stage A≈17KB + B≈17KB = 34KB，三 stage ≈ 102KB。gfx1250 LDS = 320KB/WGP，102KB < 160KB，理论上可支持 **2 WG/WGP**，需实测 arena alignment 后确认。

推荐流程：固定 `tile_m=64, tile_n=128, tile_k=128, num_buffers=3`，用 autotuner 搜索 `m_warp × n_warp` 和 `waves_per_eu`。

### 6. [附赠] bf16 BMM 潜在溢出 bug

`bmm_gfx1250.py`（bf16 BMM）的 `adv_b_i32 = tile_k × N × 2 = 262,144 bytes`。K=4096, tile_k=128 → 32 步累计进量 = 8MB。若初始 B ptr 的 low-32 位 ≥ 0xFC000000，addr_lo_b **会溢出 i32**，导致错误 TDM 地址。

`bmm_a16w8` 已正确实现 carry propagation。应在 `bmm_gfx1250.py` 补充同样逻辑。

### 汇总

| # | 优化点 | 预估收益 | 难度 | 适用场景 |
|---|--------|---------|------|---------|
| 1 | WMMA_SCALE fp8 路径（新变体） | **3.5× 算力** | 高 | 精度验证后 |
| 2 | Wave-specialized TDM | 10-20% | 中 | Compute-bound |
| 3 | Scale buffer_load prefetch | 5-10% | 中 | fp32 scale 路径 |
| 4 | LDS_PAD_A bank conflict 修复 | 5-10% | 低 | Compute-bound |
| 5 | Tile 调优 + autotune | 10-30% | 低 | 各场景 |
| 6 | bf16 BMM carry propagation | bug fix | 低 | bmm_gfx1250.py |

**优先级**：先验证精度后走 WMMA_SCALE 路径（单项收益超过其余所有项之和）；tile 调优和 wave_specialized_tdm 是近期低风险的改善项。
