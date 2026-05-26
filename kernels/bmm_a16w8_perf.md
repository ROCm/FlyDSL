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
