# MoE Stage-1 Megakernel — 实验记录 & 进度（records / notes）

> 设计与算子说明以 `docs/moe_stage1_mega.md` 为准。本文件只存**有价值的实验记录、性能基线、进度日志**。
> 口径统一：8×MI355X（gfx950，CDNA4），topk=6，profiler GPU device-time，8 卡均值，`--from-bf16`。

---

## 1. 进度日志（progress log）

- **decode（`scheme="fixedslot"`）= 已验证、已落地、生产可用。** 每行映射回源 token 后逐元素 vs ATOM 生产栈通过（§7 oracle，容差比对）；最终性能见 `moe_stage1_mega.md` §6（bs≥8 对生产 ATOM 净胜 1.2–1.9×，靠 xcd 的 L2 复用 + gather-free + 省 sort/scale-sort）。
- **prefill（`scheme="handshake"`）= P1+P2 已落地，待 GPU 释放后按 `moe_stage1_mega.md` §9.9 验证；P3/P4 待 P1/P2 验证后接力。**
  - **P1**：block0-自足 handshake（P0 256线程直方图 → 跨PE计数 all-gather → CMP 元数据 → SCT inv，全部逐 workgroup `fx.barrier`，零 grid barrier）+ 每次 forward 复位的 `meta_flag`（绝对阈值 `wait>=1`，任意 wave 序不死锁）。跨 PE epoch 用单调 `gb1`（block0 每 launch +1，永不 reset，CUDAGraph 安全）。
  - **P2**：解 co-resident，grid=`(gx+np_cols, gy)`、`gy=α·cu/(gx+np_cols)`、total=`α·cu`。env：`FUSED_MEGA_ALPHA`（默认 1=共相驻先验证；2=oversubscribe 性能模式）、`FUSED_MEGA_NP_COLS`（默认 8，扫 {8,16,32,64}）、`FUSED_MEGA_GY`（覆盖）。
  - **P3（待落地）**：swizzle 折进 producer + consumer 切 `gemm_sw`（`raw_a_scale=False`），去 ~16KB scale-LDS，occupancy 2→3（见 §3）。
  - **P4（可选）**：叠 `xcd_swizzle`（`inter%256==0`、`tile_n=256`，守卫 `gx|8`）。
- **代码隔离**：所有 prefill 改动封在 `fused_moe_gemm_2stage.py` 的 `const_expr(_fuse_hs)` 分支 + `fused_moe_megakernel.py` 的 handshake-only 复位里；decode（`_fuse_fs`）路径未改动。

### 验证顺序（GPU 释放后）
```bash
# oracle = mega vs ATOM 逐元素（bench --check-correctness，见 moe_stage1_mega.md §7）
TR="torchrun --standalone --nproc_per_node=8 tests/kernels/bench_moe_intranode_stage1_groupgemm.py"
# 0) decode 不回归
MORI_SHMEM_HEAP_SIZE=8G $TR --mega --mega-scheme fixedslot --from-bf16 --check-correctness
# 1) P1 handshake 正确（α=1 共相驻，先排除 oversubscribe 变量）
MORI_SHMEM_HEAP_SIZE=8G $TR --mega --mega-scheme handshake --from-bf16 --check-correctness
# 2) P2 oversubscribe 仍 PASS + 无死锁
FUSED_MEGA_ALPHA=2 MORI_SHMEM_HEAP_SIZE=8G $TR --mega --mega-scheme handshake --from-bf16 --check-correctness
# 3) 性能扫（找 fused/atom_fp8 >= 1.0 的 np_cols/α）
#   $TR --mega --mega-scheme handshake --from-bf16 \
#     --atom-fp8-dispatch --profiler-time --check-correctness
```

---

## 2. decode 性能杠杆 & 扫描结果

构造期静态二选一（`fused_moe_megakernel.py`），非 autotune：

| 路径 | 判据 | tile_m | tile_n | tile_k |
|---|---|---:|---:|---:|
| decode | `mtpr < 4096` | 64 | 256（`inter%256==0`，否则 128） | 256 |
| prefill | `mtpr ≥ 4096` / handshake | 128 | 128 | 256 |

`gx = inter/tile_n`、`gy = cu/gx`、`Tm = ceil(num_valid/tile_m)`。

**杠杆**：① `tile_n 64→256` 提 MFMA 强度（decode 快 4–6%）；② **`xcd_swizzle`（最大头）**：把共用同一 N-tile 权重的 CTA 聚到同一 XCD → L2 复用，仅持久受益、仅 `gx|8`（gx∈{1,2,4,8}）合法（详见 `moe_stage1_mega.md` §5）；③ split-K（见 §4，未落地）。

### Exp A — tile 扫（v4_flash a8w4，fused/atom_fp8，越高越好；xcd=0）
```
tm=64 tn=64   bs1/8/16/32/64 = 0.838/0.709/0.705/0.719/0.753
tm=64 tn=128  bs1/8/16/32/64 = 0.850/0.857/0.859/0.880/0.896
tm=64 tn=256  bs1/8/16/32/64 = 0.835/0.781/0.772/0.799/0.821
tm=32 tn=256  bs1/8/16/32/64 = 1.589(bs1 噪声)/0.847/0.857/0.845/0.856
tm=128 tn=128 bs1/8/16/32/64 = 0.772/0.775/0.832/0.803/0.816
```
### Exp B — xcd 扫（tile_n=256，fused/atom_fp8）
```
tm=64 tn=256 xcd=4  bs8/32/64 = 1.162/1.361/1.343
tm=64 tn=256 xcd=8  bs8/32/64 = 1.161/1.327/1.350
```
结论：**tile_n=256 + xcd 是 decode 决胜组合**（裸 tile 扫均 <1.0，xcd 一开就 1.16–1.36×）。已设为默认。

---

## 3. prefill handshake overlap —— 实验根因（R1/R2/R3）& occupancy

旧 handshake overlap 实测 **0.21–0.49×**（净输），三个**结构性**根因（`moe_stage1_mega.md` §9.2 据此重排）：

- **R1 CU 对半砍**：旧默认 `np_cols=gx/2`（~1/3–1/2 CU 划给 producer）+ co-resident 上限 `(gx+np_cols)·gy ≤ cu` ⇒ consumer 并行度腰斩。
- **R2 串行 + 原子风暴**：4 道 agent grid barrier 在 256 block 上 `atomic_add+spin`（实测每道 ~11–14µs），且 P0/CMP/scatter 串行排在 GEMM 前。
- **R3 共享 CU 池两相都 resource-bound**：dispatch 是带宽/发射-bound（实测 np1→np8 越多越快，producer 不是「少 CU 即饱和」），GEMM 是 compute-bound；静态切 CU 不腾空闲资源 ⇒ overlap 无增益。

**dispatch 的三方案（均 bit-exact 验证过）**：per-token 远端原子（fixedslot）/ counts-first all-gather + dense 直写（handshake）/ counting-sort scatter（handshake 最优，megakernel producer 主体即此）。fence 配对：producer `fence_system_release` 在 `atomic_add(payload_done)` 之前；consumer 过闸后 `fence_system_acquire` ⇒ 读到刚落地 payload。

**scale-LDS → occupancy（rocprofv3 实测，tile_m=128）**：raw scale-LDS ~16–28KB；去掉后 per-block LDS ~50KB，gfx950 的 163840 B/CU 下 occupancy **2→3 blocks/CU**（LDS 驱动，非 VGPR）。这是 a4w4 prefill 早先回落（v4_flash bs8192/16384≈0.87/0.89×）的根因，预-swizzle 修复后反超 1.22/1.26×。⇒ P3 的依据。

---

## 4. split-K（未落地，未来杠杆）

decode 是 M-tiny / K-huge（model_dim 4096–7168）→ GEMM grid 饿（少 M-tile，闲 CU）。split-K 加 grid-Z 维并行 K reduction → 填 CU、提 occupancy。fused 适配需 **dispatch-prologue z-gate**：dispatch+arrival 仅 `blockIdx.z==0` 跑；`z>0` 平面跳 dispatch、等 `meta_flag` 后算各自 K 切片、atomic-add 到预清零 `out`。改动共享 prologue、风险高 → 留作 supervised 落地（先 k_batch=1 strict 回归哨兵，再开 2/4，用 bench `--check-correctness` 验证）。

---

## 5. 性能基线：a8w4 vs **fp8-dispatch** baseline（prefill 仍落后点，§9 攻击目标）

对**生产 bf16-dispatch** baseline，a8w4 全面正（1.0–1.4×）；但对**更强的 fp8-dispatch** baseline（双方都 fp8 dispatch + scale-sort，差距收敛到 GEMM 本身），优势收窄 —— 这是 prefill 单核 overlap（§9）要吃掉的：

| net | bs | atom_fp8 µs | flydsl µs | fused µs | fused/flydsl | **fused/atom_fp8** | bucket |
|---|---|---|---|---|---|---|---|
| v4_flash | 4096 | 1004.9 | 830.8 | 721.1 | 1.39 | **1.15** | prefill |
| v4_pro | 8192 | 3652.9 | 2906.2 | 3168.5 | 1.15 | **0.92** | prefill |
| v4_pro | 16384 | 7245.6 | 5701.2 | 6210.7 | 1.17 | **0.92** | prefill |
| v4_pro | 32768 | 14496.5 | 11389.2 | 11770.6 | 1.23 | **0.97** | prefill |
| r1_v3 | 4096 | 1690.7 | 1309.0 | 1546.0 | 1.09 | **0.85** | prefill |
| r1_v3 | 8192 | 2964.0 | 2212.9 | 2784.3 | 1.07 | **0.80** | prefill |
| r1_v3 | 16384 | 5903.9 | 4388.5 | 5077.8 | 1.16 | **0.86** | prefill |
| r1_v3 | 32768 | 11921.5 | 8850.8 | 9965.5 | 1.20 | **0.89** | prefill |

**geomean（vs atom_fp8）**：v4_flash prefill(≥4096) **1.087**；v4_pro prefill **0.959**；r1_v3 prefill **0.848**。
⇒ §9 单核 overlap 的目标网络 = **v4_pro / r1_v3 prefill**（把 dispatch 串行暴露藏到 MFMA 下、推过 1.0），v4_flash prefill（已 1.087×）进一步拉开。
