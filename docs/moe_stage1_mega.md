# MoE Stage-1 单launch融合算子 `mega` —— 架构设计 & 算子说明（AMD MI355X / CDNA4）

面向**节点内专家并行（EP）推理**的 MoE **stage-1**（dispatch → gate/up GEMM）**单 kernel 融合算子**。把生产基线的多 kernel 流水（量化 → dispatch → moe_sorting → scale-sort → group-GEMM）压成 **一次 launch 的 2 个 kernel**：①（外部）激活量化；②一个「**dispatch ⊕ 持久 group-GEMM**」融合核（本文档主角）。

- **入口**：`kernels/fused_moe_megakernel.py::FusedMoEMegaStage1(scheme="fixedslot")`
- **kernel**：`kernels/fused_moe_gemm_2stage.py::compile_fused_moe_gemm1(fuse_dispatch="fixedslot", ...)`
- **复用缓冲**：`kernels/ep_dispatch_groupmajor_op.py::FlyDSLDispatchGroupMajorOp`（只用其对称/P2P 缓冲，不单独 launch dispatch kernel）
- **正确性**：`bench_moe_intranode_stage1_groupgemm.py --mega --check-correctness`（每行映射回源 token 后逐元素 vs ATOM 生产栈，见 §7）

> **最终性能（8×MI355X，topk=6，profiler GPU device-time，8 卡均值，`--from-bf16`，`fixedslot` 持久 + `tile_n=256` + xcd，全部 bit-exact）：decode 对生产 ATOM 净胜——v4_flash a8w4 bs≥8 1.19–1.40×、r1_v3 a4w4 bs≥8 1.44–1.88×、v4_pro a8w4 ≈持平（xcd 不适用，见 §5）。** 完整数据见 §6。

---

## 1. 架构与数据流

生产基线（5-kernel）：`量化 → dispatch → moe_sorting → scale-sort → group-GEMM`。本算子压成 **2 个 kernel**：

```
                       ┌──────────────────────── 融合 megakernel（1 次 launch）─────────────────────────┐
 bf16 token            │  Phase-1  dispatch 前奏        ┊ grid    Phase-2  持久 group-GEMM              │
 + routing  ──①量化──▶ │  (每块写 peer 的 expert-major  ┊ barrier (每块 round-robin 扫 M-tile:          │ ──▶ out
 (外部 kernel,         │   固定槽 + 跨 PE done-barrier)  ┊ (§4)    raw_a_scale 折入, MFMA, silu(g)*u)     │   expert-major
  per_1x32_mx_quant)   │  block0: post-pass 产 tile 元数据┊         gather-free, 直接写 expert-major out  │   连续 [nv,inter]
                       └──────────────────────────────────────────────────────────────────────────────┘
```

**数据流（融合核内部）**：

1. **输入**：A = 量化后的激活（a8w4=fp8 / a4w4=fp4 packed）+ e8m0 行 scale；W = 专家权重（fp4 packed）+ e8m0 列 scale；`topk_ids`、routing weights。
2. **Phase-1 dispatch 前奏**：每个 block 把本卡 token 按 `(token, expert)` 写到**目标 PE 的 expert-major 缓冲** `rx_em`（固定槽 `slot = le*cap + atomic(running[le])`，`cap = npes*mtpr` 证明无溢出），scale/idx/srcmap 同写。跨 PE done-barrier 保证 peer 写入可见。
3. **Phase-1.5 post-pass（仅 block0）**：扫描 per-expert 计数，**紧凑产出 GEMM 要消费的 tile 元数据**：`tile_row_base[ci]`（紧凑 tile ci → 起始行）、`sorted_expert_ids[ci]`、`num_valid`；并 fold 掉 running 计数复位。发布 `meta_flag`。
4. **grid barrier**（§4）：到达计数 + meta_flag 广播,把 Phase-1 的产物对所有块可见。
5. **Phase-2 持久 group-GEMM**：每块 round-robin 扫 M-tile，读 `rx_em`（A,已 expert-major **gather-free**）+ raw e8m0 scale（`raw_a_scale` 折进 GEMM,**省独立 swizzle**）+ W,沿 K MFMA 累加,`silu(gate)*up`,**直接写 expert-major 连续 `out`**（无 scatter）。
6. **输出**：`out[num_valid, inter]` f16,expert-major 固定槽。expert `le` 的真实行 = `[le*cap, le*cap+count[le])`;pad 行下游 combine 按 srcmap 取真实行、永不读 → 无害。

**关键设计点**：① 固定槽免 moe_sorting/scale-sort 两个 kernel；② `raw_a_scale` 把激活 scale 折进 GEMM 免 swizzle；③ dispatch 产 expert-major 连续布局 → GEMM gather-free + 输出免 scatter；④ 全程不落地中间 buffer、不跨 kernel。

---

## 2. 正确路径与输出布局

- **唯一正确路径 = `fixedslot`（持久 round-robin GEMM）**：结构上无条件覆盖全部 occupied tile（每个 CTA 循环 `ceil(num_valid/tile_m / grid_y)` 个 M-tile，按 device 端 `num_valid` 现算）。
- 输出布局：expert-major 固定槽，expert `le` 真实行 `[le*cap, le*cap+count[le])`，`cap = npes*mtpr`。

---

## 3. Kernel 流水排布（pipeline scheduling）

### 3.1 宏观（核内相位）

decode 是 **strict-phase**：`dispatch 前奏 → grid barrier → GEMM`，两相**不重叠**（decode 无足够 compute 可藏 dispatch 延迟）。一次 launch 内顺序执行：

```
[Phase-1 写 payload(peer atomics)]──[CTA内 barrier]──[到达 atomic + block0 等待 + 跨PE + post-pass + 发 meta_flag]──[所有块等 meta_flag]──[Phase-2 GEMM]
        无 compute,纯访存/同步                              ↑ 唯一 grid barrier(§4)                                      ↑ 计算密集
```

> prefill 用 `handshake` scheme 做**生产者/消费者 overlap**（producer 块写 payload ‖ consumer 块 GEMM,per-expert 闸 `payload_done`），把 xGMI 搬运藏到 MFMA 之下。与 decode 同核、host-only 策略选择,**优化设计见 §9**。

### 3.2 微观（GEMM K-loop 软件流水）

Phase-2 每个 tile `(bx, by)` 的 GEMM 沿 K 维做**双缓冲软件流水**（`use_async_copy=True`）：

```
持久外层:  for mi in [0, ceil(Tm/gy)):   bx = bx_persist + mi*gy      # round-robin M-tile
  K 内层(ping/pong LDS 双缓冲):
    prefetch A_tile,W_tile(K=0) → LDS[ping]        (async global→LDS)
    for kk in K-steps:
        async copy A,W(kk+1) → LDS[pong]            # 预取下一片,与下面 MFMA 重叠
        fx.barrier()                                # 等当前片就绪
        MFMA accumulate  使用 LDS[ping]             # 计算当前片
        swap(ping, pong)
    epilogue: silu(gate)*up,(CShuffle)写 expert-major out
```

要点：① **async copy 把下一 K 片的 global→LDS 搬运与当前片 MFMA 重叠**（隐藏访存延迟）；② 双缓冲 LDS（ping/pong）使取数与计算不互锁；③ 累加器底盘 `num_acc_n·m_repeat·2`（gate+up）vec4f32 常驻寄存器；④ 持久外层的 round-robin backedge 把 K-流水 live-range 跨迭代并集 → 持久变体 **VGPR≈156**（非持久≈104），换来无条件全覆盖。

---

## 4. Grid 配置与同步

### 4.1 网格映射

融合核 grid = `(gx, gy)`：

- `blockIdx.x = by ∈ [0, gx)` —— **N-tile**（`gx = inter_dim / tile_n`）
- `blockIdx.y = bx_persist ∈ [0, gy)` —— **M 方向持久索引**

每块 round-robin 扫 M-tile：`bx = bx_persist + mi*gy`，`mi ∈ [0, ceil(Tm/gy))`，`Tm = ceil(num_valid/tile_m)`（device 端现算）；`blk_valid = (bx < Tm)`。tile `(bx, by)` 读 `tile_row_base[bx]`→A 行基址、`sorted_expert_ids[bx]`→该 tile 的专家(决定 W 的列段 by)。**循环步长 = `grid_dim.y = gy`,覆盖与 gy 大小无关（循环兜住）→ 减小 gy 不影响正确性。**

### 4.2 co-residency（为何 `gy = cu//gx`，与 gemm2 的区别）

**融合核有「核内 grid barrier」**——dispatch 写完后,block0 自旋等**所有**块到达 + 跨 PE 同步 + 发布元数据,然后所有块才读 dispatch 产物做 GEMM。这要求**所有块同时驻留（co-resident）**,否则排队块永不到达 → block0 死等 → 死锁。故：

```
gx * gy ≤ cu_num   →   gy = cu_num // gx
```

**对比 `compile_mixed_moe_gemm2`（stage2 独立核）**：它**没有核内 barrier**（输入已 dispatch 完）,所以能启动 `grid_y = cu_num`（总块 = gx·cu_num ≫ cu）,硬件按 wave 自然调度。融合核为了「核内直接读 dispatch 产物」付出的代价就是 co-resident 约束。

### 4.3 grid barrier 实现（原子计数,非硬件 grid.sync）

decode 只有**一处** grid barrier（`fused_moe_gemm_2stage.py` 的 fixedslot prologue）,实现为 N→1 到达 + 1→N 广播：

```
每块: fx.barrier()(CTA内) → atomic_add_agent(gb1)         # 到达,不自旋
block0: int64_wait_until_equals(gb1, total)               # 等所有块到达(N→1)
        → 跨 PE done-barrier(与 peer rank 互等)
        → post-pass(产 tile 元数据)
        → fence + release-store meta_flag = epoch          # 发布
其余块: int32_wait_until_greater_than(meta_flag, e0)       # 等广播(1→N,1 writer/N readers,无原子风暴)
```

payload 写是独立 peer atomic,无需「所有块已启动」的初始 barrier → **已从 3 个砍到 1 个**(decode 剩余 gap 大头就是这个 barrier 的原子争用 + 跨 PE 延迟地板)。

### 4.4 tile 配置


| 路径              | 判据                        | tile_m | tile_n  | tile_k |
| --------------- | ------------------------- | ------ | ------- | ------ |
| decode（默认 auto） | `mtpr < 4096`             | 64     | **256** | 256    |
| prefill         | `mtpr ≥ 4096` / handshake | 128    | 128     | 256    |


facade `tile_n=-1`（AUTO）：decode→256（当 `inter%256==0`,否则 128）,prefill→128;显式传入优先。`tile_n=256` 提 MFMA 强度且使 `gx=inter/256`（v4_flash/r1_v3→gx=8,可开 xcd）。约束:tile_n 须满足 epilogue 的 `e_vec`(=`min(tile_n//32,8)`)为 2 的幂 ∈{2,4,8}（CShuffle 向量 store 对齐）→ 合法 tile_n ∈ {64,128,256,512,768,...}。

---

## 5. XCD swizzle 优化（仅 `gx | 8` 时开）

MI355X 有 **8 个 XCD（chiplet），每个独立 L2**；硬件默认 `block i → XCD (i%8)`。GEMM 中同一 N-tile 的权重 `W[:, nj]`（`[K, tile_n]`,几 MB）被该列所有 M-tile 复用；若这些 CTA 散到 8 个 XCD,`W[:,nj]` 被各 L2 各拉一份 → 8× 冗余带宽。`xcd_swizzle` 用 WGM 重映射（`fused_moe_gemm_2stage.py` 的 xcd_swizzle 分支）把**共用同一权重的 CTA 聚到同一 XCD** → L2 复用 → 提速。

- `**xcd_swizzle=4` = WGM 组大小（不是 4 个 XCD；XCD 恒 8）**：把 **4 行 M-tile × 全部 gx 个 N-tile** 打包成一个单元、整块给同一 XCD,该专家权重 slab 在该 XCD 的 L2 只加载一次、被 4 行复用。aiter 生产搜索空间 `{0, 4}`。
- **正确性约束：`gx | 8`**（gx ∈ {1,2,4,8}）。WGM 组 `4·gx` 必须装进单个 XCD 的块块 `cu/8=32` → `32 % (4·gx)==0` → `gx | 8`。
  - **gx=8（v4_flash/r1_v3）→ 合法 + 大收益**；gx=4 也合法。
  - **gx=12 / gx=16 → WGM 组跨 XCD 边界 → 数值损坏** → facade 自动关。
- **只有持久受益**：持久 CTA 长驻留 + 循环扫多 tile,XCD 的 L2 里权重跨循环迭代反复命中;非持久 CTA 算一个 tile 即退,多波时硬件按 block-id 重填、聚类被打散。

**v4_pro（inter=3072）拿不到 xcd —— 已穷举实测**：合法 tile_n（256 倍数下 e_vec=pow2:256/512/768）→ gx ∈ {12,6,4};要 `gx|8` 只有 gx=4（tile_n=768,实测**慢 4.6×**:271→1253µs,累加器爆 + N 并行仅 4 路）。gx=8 需 tile_n=384(强制 e_vec=4 可正确,bit-exact)但大 tile 抵消 xcd → 仍 ~0.97×。∴ v4_pro 保持 **tile_n=256 + xcd 自动关 = 持平**。

> facade 守卫:`cu % gx != 0` 自动关 xcd（并警告）。env `FUSED_MEGA_XCD`(开关)、`FUSED_MEGA_EVEC`(强制 e_vec,实验用)。

---

## 6. 性能数据（最终）

口径：8×MI355X，topk=6，profiler GPU device-time，8 卡均值，`--from-bf16`，`scheme=fixedslot`，`tile_n=256`，xcd（gx=8 网络生效、v4_pro 自动关）。baseline：a8w4 = `atom_fp8`（fp8-dispatch），a4w4 = `atom`。


| bs  | **v4_flash a8w4** (/atom_fp8) | **v4_pro a8w4** (/atom_fp8) | **r1_v3 a4w4** (/atom) |
| --- | ----------------------------- | --------------------------- | ---------------------- |
| 1   | 0.94×                         | 0.98×                       | 0.81×                  |
| 8   | 1.19×                         | 0.98×                       | 1.44×                  |
| 16  | 1.28×                         | 0.98×                       | 1.57×                  |
| 32  | 1.35×                         | 0.99×                       | 1.65×                  |
| 64  | 1.35×                         | 0.99×                       | 1.71×                  |
| 128 | 1.40×                         | 1.02×                       | 1.88×                  |


绝对 µs（baseline → mega，bs128）：v4_flash 101→72，r1_v3 165→88，v4_pro 291→284。网络维度（md/inter/experts）：v4_flash=4096/2048/256，v4_pro=7168/3072/384，r1_v3=7168/2048/256。

**规律**：

- **gx=8 网络（v4_flash/r1_v3）bs≥8 净胜 1.2–1.9×**（xcd 的 L2 复用 + gather-free + 省 sort/scale-sort）。
- **v4_pro（gx=12）≈持平**：xcd 不适用，融合本身打平（无回退无损坏）。
- **bs1 普遍略输**（0.81–0.98×）：单 token work 极小，dispatch 延迟主导，是物理天花板。

---

## 7. 正确性体系

- **oracle = mega vs ATOM（生产 aiter 栈），逐元素比对**：`bench_moe_intranode_stage1_groupgemm.py --mega --check-correctness`。两边把每个输出行**映射回源 token**用同一把 32-bit 键 `(k_slot<<24)|(src_pe*mtpr+src_tok)`：
  - mega：`srcmap_em[row]` 即该键，值 = `out[row]`（real 行：fixedslot `[le*cap, le*cap+count)`；handshake dense `ebase`）。
  - atom：`dispatch_combine` dispatch → `aiter.moe_sorting_fwd` → `aiter.mxfp4_moe_sort_hip`（scale-sort）→ `mixed_moe_gemm1`；recv 槽 `s` 的第 `k` 个 LOCAL expert → 键 `(k<<24)|tok_id_to_src[s]`，值 = `at_out[s,k]`。
- 判据：**键集一致**（无丢 tile / 多余行）**且**每个键逐元素误差 < tol（atom 与 mega 是不同 dispatch/sort/GEMM 栈，非 bit-exact，用容差 `1e-2`）。丢 tile ⇒ 键集不符；数值损坏 ⇒ 逐行误差 → 都判 FAIL。
- atom `tok_id_to_src` 与 mega `srcmap_em` 同一套 `rank*mtpr+tok` 编码，键直接可比；atom 行经 `tok_id_to_src`+`out_idx` 直接定位源 token，不依赖 aiter sort 的内部置换。

---

## 8. CUDAGraph / 约束 / 待办

- **CUDAGraph 安全**：整算子一次 launch；grid 同步用原子计数 + 单调 meta_flag（per-launch memset / 单调推进,无 reset 竞争）；per-step 输入指针走标量 launch 参数（capture 期不写 device 表,避免非法操作）。
- **co-residency**：`gx·gy ≤ cu_num`（构造期成立）。
- **split-K（slice-K）**：未落地的未来杠杆 —— 需 dispatch prologue z-gate（dispatch+arrival 仅 `blockIdx.z==0` 跑;z>0 平面等 meta_flag 后算各自 K 切片,atomic-add 到预清零 out）。因改动共享 prologue、风险高,留作 supervised 落地（设计见记录文档 `moe_stage1_mega_notes.md` §4）。
- **prefill**：`handshake` scheme（生产者/消费者 overlap）,与 decode 同核——**优化设计见 §9**（解 co-resident 约束 + per-expert flag overlap）。

---

## 9. Prefill 优化设计：per-expert overlap 单核（counts-first + 自由网格 producer/consumer）

> **状态 = P1+P2 已落地（`scheme="handshake"`），待 GPU 释放后跑 §9.9 验证；P3/P4 待 P1/P2 验证后接力。** decode 走 §1–§8 的 `fixedslot` strict-phase（**未改动**，所有 prefill 改动均封在 `const_expr(_fuse_hs)` 分支 + handshake-only facade 复位里）；prefill 走本节。两模式同一 megakernel、host-only 构造期二选一、CUDAGraph 安全，每个 bucket 都保留 2-launch 回退（零风险）。
>
> **已落地（`fused_moe_gemm_2stage.py` / `fused_moe_megakernel.py`）**：
> - **P1 block0-自足 handshake**：flat-id-0 块独自跑完 counts-first（P0 256线程直方图 → 跨PE计数 all-gather → CMP 元数据 → SCT inv），全部用**逐 workgroup 的 `fx.barrier`（非 grid barrier）**；完事 release-store `meta_flag`（**每次 forward in-graph 复位为 0 → 绝对阈值 `wait>=1`**，任意 wave 序都不死锁）。producer/consumer 相**零 grid barrier**，只靠 `payload_done` 逐专家闸耦合。跨 PE epoch 用单调 `gb1`（block0 每 launch +1，永不 reset，CUDAGraph 安全）。
> - **P2 解 co-resident**：grid = `(gx+np_cols, gy)`，`gy = α·cu/(gx+np_cols)` ⇒ total = `α·cu`。env：`FUSED_MEGA_ALPHA`（默认 **1=共相驻**先验证 P1；置 **2** 开 §9.5 oversubscribe 性能模式）、`FUSED_MEGA_NP_COLS`（默认 8，扫 {8,16,32,64}）、`FUSED_MEGA_GY`（直接覆盖）。
> - disp 表新增 idx24 = `meta_flag`；facade `forward()` 对 handshake 额外 `self._meta.zero_()`。`FUSED_MEGA_TILECOUNT` 仅限 fixedslot/np（handshake 的 `payload_done[0]` 是活闸，不可借用 → handshake 覆盖率由 srcmap-keyed bit-exact 反证：丢 tile ⇒ 该行未算 ⇒ 值不符）。

### 9.1 目标与 overlap 机制（= 用户思路）

prefill 有大块 GEMM compute 可藏 dispatch 搬运（decode 没有，§3.1 / 记录文档）。机制正是用户描述的：**dispatch 按 expert 发数据；接收端某 expert 的全部 token 落地即打 per-expert flag；GEMM 的该 expert tile 立刻过闸开做 MFMA。**

- **flag = 接收端（owner rank）的 `payload_done[le]`**（per local-expert 计数器，对称缓冲）。
- 每个 sender 写完自己路由到 le 的全部 token 后，对 owner 的 `payload_done[le]` `**atomic_add` 自己的计数**（前置 `fence_system_release`）。
- consumer 的某 tile 进 MFMA 前自旋 `payload_done[le] ≥ expected_real[le]`（`expected_real[le]` = 全 npes sender 之和，counts-first 提前已知）+ `fence_system_acquire` 读新鲜 payload。
- 因 `payload_done[le]` 达 `expected_real[le]` ⟺ 该 expert 全部 sender 数据齐 ⇒ **per-expert 早就绪、gather-free 开算**。该闸 (`overlap_gate`) 在 `fused_moe_gemm_2stage.py` 的 consumer GEMM body。

### 9.2 为什么旧 handshake overlap 没赢（记录文档 实测根因）

三个**结构性**根因，必须逐一拆掉，否则 overlap 必输（旧实测 0.21–0.49×）：

- **R1　CU 对半砍**：旧默认 `np_cols = gx/2`（近 1/3–1/2 的 CU 划给 producer）；又因 co-resident 上限 `(gx+np_cols)·gy ≤ cu_num` 必须成立，producer 多则 consumer 少（np8: 128 producer / 128 consumer）⇒ **GEMM 并行度腰斩**。
- **R2　counts-first 串行开销 + 256-block grid barrier 原子风暴**：4 道 agent grid barrier 在 256 block 上 `atomic_add + spin`（记录文档 实测每道 ~11–14µs），且 P0/CMP/scatter 串行排在 GEMM 之前。
- **R3　共享 CU 池上两相都 resource-bound**：dispatch 是带宽/发射-bound（散射写 + 索引解码，要**很多** block 才灌满 xGMI——实测 np1→np8 越多越快，说明 producer 不是「少 CU 即饱和」），GEMM 是 compute-bound；**静态切 CU 不腾出空闲资源给对方** ⇒ overlap 无增益（记录文档 物理结论）。

### 9.3 重新设计：解开 co-resident 约束，让硬件去 overlap

**核心认识**：consumer 是按 **per-tile 数据依赖（`payload_done[le]`）** 过闸，**不是** grid barrier；producer 各自独立发 `payload_done[le]`。所以**「产/消」相根本不需要任何 grid 级全核同步** → 网格可以 **oversubscribe（total > cu_num）**，硬件按 wave 调度，producer 的 xGMI 写与 consumer 的 MFMA 自然交叠（同一 wave 并存 + producer 退役后其 CU 被 consumer wave 回填）。唯一串行的只有 **block0 的 counts-first handshake**，用单调/复位 flag 发布；block0 自足（不依赖任何其他块）⇒ grid 可超 cu_num 不死锁。

这正好拆掉三根因：


| 根因      | 本设计的拆法                                                                                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R1 / R3 | 解 256 co-resident 上限 ⇒ producer 与 consumer **都开足**（各 ~cu_num 量级、跨 wave）；overlap 不靠「切一块固定 CU」而靠**时间复用 + wave 回填 + 同 CU 功能单元互填**（consumer 过闸自旋时 MFMA 空,同驻 producer 发 xGMI store） |
| R2      | **产/消相零 grid barrier**；handshake 仅 block0 串行 + **1-writer / N-reader 单调 flag**（无 256-way 原子 RMW 风暴,仿记录文档 decode 的 barrier→flag）                                          |


并叠加两个效率改动：

- `**np_cols` 最小化**：只取「灌满 xGMI 写所需」的量（env `FUSED_MEGA_NP_COLS` 扫 {8,16,32,64}），而非旧的 `gx/2`。这是相对旧设计最关键的差别。
- **swizzle 折进 producer + consumer 走 `gemm_sw`（`raw_a_scale=False`）**：去掉 ~16KB scale-LDS，occupancy **2→3**（记录文档 实测），更高 occupancy = 每 CU 更易让「consumer 过闸自旋」与「producer 发 xGMI」共驻互填气泡。producer 写完 le 的 scale 即就地 swizzle 成紧凑 A-scale 再发 `payload_done[le]`，**swizzle 纳入这条流水**而非单独一相。

### 9.3.1 协作式持久 megakernel 架构（相位 + 角色划分）

一个 kernel、一次 launch；把量化（可选）/ handshake / producer 写 / consumer GEMM 收进同一持久 grid 的**相位（phase）与角色（role）**。与 decode 的 strict-phase（§1 图、`fixedslot`）相比，prefill 的关键差异是 **producer 相与 consumer 相物理并发、且产/消相零 grid barrier**：

```
                 ┌──────────────  一个持久 grid（产/消相零 grid barrier ⇒ 可 oversubscribe，total>cu_num）  ──────────────┐
 Phase 0 (可选)  │ 量化：本地 x[bs] bf16 → fp4/fp8 + e8m0   (Inc 3 才并入；否则沿用外部 aiter 量化前置 kernel)         │
 Phase H (block0)│ counts-first handshake：P0 本地直方图 → 跨 PE 计数 all-gather → CMP 元数据/my_base → SCT 散射     │
 ─────────────── │  → release-store meta_flag（单调）。其余块 acquire-spin meta_flag（1-writer/N-reader,非 barrier）  │
 Phase P (by≥gx) │ producer 块：按 expert 写 DENSE payload + swizzle → 每 (rank,expert) 发一次 payload_done[le]      │
 Phase C (by<gx) │ consumer 块：持久 round-robin occupied tile，每 tile 过 per-expert 闸 (payload_done≥expected) →MFMA│
                 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
   Phase P ‖ Phase C 在不同 block/wave 上物理并发；唯一耦合 = payload_done[le] 数据闸（无全核同步）→ overlap 由硬件 wave 调度自然产生
```

- **复用持久 grid**：consumer 本就 round-robin occupied tile（§9.4），megakernel 只是把 handshake + producer 写**并到同一批已驻留 CTA**，GEMM 相零额外 launch。
- **相间同步**：Phase H → P/C 用单调 `meta_flag`（release/acquire，agent-scope，免 cooperative-launch API）；跨 PE 用单调 epoch done-barrier（CUDAGraph 安全、永不 reset）；P → C 用 per-expert `payload_done`（system-scope）。
- **资源包络** = 各相取 max（GEMM 主导 VGPR/LDS）；producer 相 comm/BW-bound，低占用率无害（warp 数远超饱和 xGMI 写所需）。

### 9.4 persistent 工作模式

**consumer（`block_idx.x < gx`，列 = N-tile）= 持久 round-robin，全覆盖：**

```
acquire-wait meta_flag ≥ epoch                 # 等 block0 发布 tile_row_base/sorted_expert_ids/num_valid
Tm = ceil(num_valid / tile_m)                   # device 端现算
for mi in [0, ceil(Tm/gy)):                      # 步长 = grid_dim.y → 全覆盖（与是否 co-resident 无关）
    bx = block_idx.y + mi*gy
    if bx ≥ Tm: continue                          # 早退（pad 行无害）
    le = sorted_expert_ids[bx] − rank*epr
    int32_wait_until(payload_done[le] ≥ expected_real[le]); fence_system_acquire   # ← per-expert 闸
    A = rx_em[ tile_row_base[bx] .. ]            # DENSE expert-major，gather-free
    K-loop 双缓冲 MFMA（记录文档）→ silu(gate)*up → 写 expert-major out
```

持久 round-robin 步长 `grid_dim.y` ⇒ `occupied_tiles > gy` 时也**全覆盖**（不像非持久 1-tile/CTA 只盖 gy 个会丢 tile）；oversubscribe 下每个 `block_idx.y ∈ [0,gy)` 行都会被某 wave 执行 → **覆盖与 co-resident 解耦**。

**producer（`block_idx.x ≥ gx`，np_cols 列）= expert-grouped scatter 写 + per-(sender,expert) 信号：**

```
acquire-wait meta_flag ≥ epoch                  # 等 my_base / inv / local_prefix
for ge in (pbid, total_experts, num_prod_blocks):     # 本 block 拥有的全局专家（取模分配）
    le=ge%epr; dest=ge//epr
    cnt=local_hist[ge]; base=local_prefix[ge]; slot0=my_base[ge]   # 致密前缀,无 cap 空洞
    for off in (warp, cnt, num_waves):           # counting-sort: 只碰真实 token,无扫描
        wk = inv[base+off]                         # 致密 send-order → work_idx
        写 payload(fp4)+idx+wts+srcmap+swizzled-scale 到 dest 远端 DENSE 槽 slot0+off
    block barrier; fence_system_release
    atomic_add_system(dest.payload_done[le], cnt) # ← 每 (本 rank, ge) 只发一次 → 争用 = npes(=8)
```

producer 主体即 `_fuse_hs` 的 **counting-sort scatter**（`fused_moe_gemm_2stage.py` 的 producer 循环，记录文档 §3 三方案里最好的一条，已 bit-exact）。

### 9.5 grid & tile（精确）

- `gx = inter_dim / tile_n`（consumer N-tile 列；launcher 现公式 `(2·inter − pad + 2·tile_n − 1)/tile_n/2`）。
- grid = `(gx + np_cols, gy)`；**total 不再 cap 到 cu_num**（产/消相无 grid barrier ⇒ 可 oversubscribe，目标 ~2×cu_num 让 wave 交叠）。
- `gy ≈ round(α·cu_num / (gx + np_cols))`，α≈2（env `FUSED_MEGA_GY` 覆盖）；`np_cols` 默认取「灌满 xGMI 的最小量」（env `FUSED_MEGA_NP_COLS`，扫 {8,16,32,64}）。
- **block0（flat id 0）= 协调者**：做完 handshake、release-store `meta_flag` 后并入 consumer。升序 block-id 调度保证 block0 在首 wave ⇒ flag 必被发出，无死锁（block0 自足范式）。
- **tile**：`tile_m=128, tile_n=128, tile_k=256`（prefill 高 MFMA 强度）；`inter%256==0` 时可 `tile_n=256`（gx 减半，叠 `xcd_swizzle` 拿 L2 复用，与 overlap 正交，守卫 `gx|8`，§5）。
- **scale 路径 = `gemm_sw`**（`raw_a_scale=False`，swizzle 折进 producer）以提 occupancy。


| net      | inter | tile_n | gx  | np_cols(初值) | gy(α=2) | producer/consumer 块 |
| -------- | ----- | ------ | --- | ----------- | ------- | ------------------- |
| v4_flash | 2048  | 128    | 16  | 8           | 21      | 168 / 336           |
| r1_v3    | 2048  | 128    | 16  | 8           | 21      | 168 / 336           |
| v4_pro   | 3072  | 128    | 24  | 8           | 16      | 128 / 384           |


> 表中 np_cols/gy 为**起扫初值**，最优值由 P2 实测扫定（见 §9.9）。

### 9.6 数据流 & 同步协议（单调 flag，CUDAGraph 安全）

```
block0 串行（其余块 acquire-spin meta_flag）：
  P0  自给直方图 local_hist[ge] + 每 token 致密 off（block0 用满 256 线程,LDS 直方图,O(T·topk)）
  PUB 跨 PE 计数 all-gather → bigcnt（一次跨卡轮; ~xGMI 延迟地板,与 2-launch 共享,非新增）
  CMP my_base[ge]（致密 tile-padded 前缀）+ tile_row_base/sorted_expert_ids/num_valid
      + expected_real[le](=ll_count) + local_prefix[ge]
  SCT counting-sort: inv[ local_prefix[ge] + off ] = wk
  release-store meta_flag = epoch              # 单调,永不 reset → CUDAGraph 安全
producer:  acquire flag → 写 payload(system release) → atomic_add 远端 payload_done[le]
consumer:  acquire flag → 读 num_valid/元数据 → 每 tile spin payload_done[le]≥expected_real[le] + system acquire → MFMA
```

**fence 配对（新鲜性，记录文档 同 pairing，已 bit-exact）**：producer `fence_system_release` 在 `atomic_add(payload_done)` 之前；consumer 过闸后 `fence_system_acquire` ⇒ 读到刚落地 payload 而非陈旧 L2。`expected_real`/元数据在 payload **之前**就已知（counts-first）⇒ per-expert 早就绪。

### 9.6.1 三层掩盖（overlap 在哪、怎么藏）

megakernel 的掩盖分三层；prefill 的净增益主要来自 **L1**（用户要的 per-expert overlap），L2/L3 是底层流水：


| 层                     | 粒度          | 谁 ‖ 谁                             | 同步件                              | 生效模式              |
| --------------------- | ----------- | --------------------------------- | -------------------------------- | ----------------- |
| **L1 inter-block**    | 粗（专家级）      | dispatch payload 搬运 ‖ GEMM 计算     | per-expert `payload_done` 闸      | **prefill**（本节主角） |
| **L2 intra-GEMM**     | 细（K-tile 级） | 下一拍访存 ‖ 本拍 MFMA                   | ping/pong LDS + `use_async_copy` | both              |
| **L3 latency-hiding** | 相内          | quant‖xGMI写、权重预取‖cross-PE barrier | 无（本地、无依赖）                        | decode            |


### 9.6.2 图2 — prefill per-expert 滚动流水（真正的掩盖）

producer 块写 payload + 发信号，consumer 块过闸 + MFMA，**物理上跑在不同 CU/wave 上并行**，per-expert 闸把两条流串起来；counts 先到 ⇒ tile 元数据先知 ⇒ 闸可早开：

```
producer blocks (by≥gx) ‖ consumer blocks (by<gx)；counts 先到 → tile 元数据先知 → 闸串两流
t ───────────────────────────────────────────────────────────────────────────►
producers │handshake│ wr e0 │ wr e1 │ wr e2 │ wr e3 │ wr e4 │ …      ← xGMI 带宽流(全节点 all-to-all)
(写+swz    │ counts  │ ↓pd0  │ ↓pd1  │ ↓pd2  │ ↓pd3  │ ↓pd4  │          pd[le] 达峰 = le 全 peer 写完
 +发信号)  │
consumers │handshake│metadata│fill │MFMA e0│MFMA e1│MFMA e2│MFMA e3│… ← 算力流
(gate+MFMA │ counts  │(counts→tiles 已知) ↑gate pd0 ↑gate pd1 ↑gate pd2
                                           └ e1/e2 的 xGMI 落在 e0/e1 的 MFMA 之下 ┘
稳态：总时 ≈ handshake + 一次 fill(e0) + max(Σ xGMI, Σ MFMA)，而非串行 Σ xGMI + Σ MFMA
```

让这条流水真正满起来的三个设计动作 → 见 §9.8（producer 按专家序写完即发信号 / swizzle 折进 producer / consumer 过闸 acquire-fence）。

### 9.6.3 图3 — intra-GEMM K-loop 双缓冲（L2，CDNA4 友好）

每个 tile 的 K 循环沿 K 维做 ping/pong 双缓冲，把下一拍 global→LDS 搬运压到本拍 MFMA 之下（MI355X 160KB LDS + L1→LDS 直灌 + 256B/clk 读 ⇒ 双缓冲可做更深、掩盖更稳）：

```
单个 tile 的 K 循环（async-copy / MI355X L1→LDS 直灌 + ping/pong）
K iter :   k         k+1        k+2        k+3
X→LDS  : [→ping]   [→pong]    [→ping]    [→pong]      ← 预取下一拍
B load : [B k ]    [B k+1]    [B k+2]    …
MFMA   :     └─[MFMA k]─┘└─[MFMA k+1]─┘└─[MFMA k+2]─┘  ← 访存延迟压在 MFMA 之下
```

### 9.7 正确性 / co-residency / CUDAGraph / fail-fast（**重点防错**）

- **全覆盖**：consumer 持久 round-robin 步长 `grid_dim.y`，盖全部 `Tm` tile；oversubscribe 下每个 `block_idx.y` 行必被某 wave 执行 ⇒ 无丢 tile（与非持久「1 tile/CTA 只盖 gy 个」本质不同）。哨兵：decode 用 `FUSED_MEGA_TILECOUNT=1` 核验 `gemm_body_execs == Tm × gx`；handshake 的覆盖由 srcmap-keyed bit-exact 反证（丢 tile ⇒ 该行未算 ⇒ 值不符）。
- **无死锁**：产/消相**零 grid barrier**；唯一等待是 `payload_done[le]`（数据依赖，对应 producer 必达）与 `meta_flag`（block0 必发、首 wave）。block0 自足、不等任何别的块 ⇒ **grid 可超 cu_num 不死锁**。⚠️ 前提：升序 block-id 调度让 block0 进首 wave。
- **gate 阈值**：`int32_wait_until_greater_than(payload_done+le, expected_real[le]−1)` ⟺ `≥ expected_real`（代码 `_thr = exp_cnt − 1`）。block-uniform（`blk_valid ∧ exp_valid`）⇒ 同 block 全线程同闸，无 warp 分歧死锁。
- **expert 序一致**：CMP 按 le 升序排 `tile_row_base/sorted_expert_ids`，producer 按 ge 升序取模拥有 ⇒ 低专家先达峰、consumer 低 tile 先过闸（§9.8）。
- **CUDAGraph**：一次 launch；`meta_flag`/cross-PE 用单调 epoch（永不 reset）；`local_hist`/`payload_done` 每 forward in-graph memset（graph 安全，已在 `forward()` 的 handshake 分支）。
- **显存 / fail-fast**：handshake 紧凑布局 `num_valid_max = npes·mtpr·topk + epr·tile_m`（按真实占用、无 cap 空洞，记录文档），比 `fixedslot` 的 `epr·cap` 省显存；溢出 host 断言 `cur_tok ≤ mtpr`。

### 9.8 让流水满起来（最小化 fill）

1. **producer 按 expert 升序拥有/遍历**（取模分配，低专家由低 producer block 先写）→ 低专家先达峰 `payload_done`，consumer 的低 tile（同按专家升序排布）先过闸 ⇒ fill 只等 `e0`。
2. **producer block 放低 flat-id**（block0 之后）→ 与 consumer 同处首 wave 并发，而非「先 producer wave、后 consumer wave」（否则退化成串行，无 overlap）。
3. **swizzle 纳入 producer 的 per-expert 工作**（写完 le 的 scale 即 swizzle 再发 `payload_done[le]`），不另起一相。

### 9.9 落地步骤 & 验证 & 回退（每步独立可测可退）


| 步          | 内容                                                                                                             | 验证 / 红线                                                                         | 状态 |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --- |
| **P1**     | 现 `handshake` 的 4 道 grid barrier → 去掉，改 **block0 自足 + 单调/复位 flag**；产/消相零 barrier | 小 config **bit-exact**；无死锁                                    | ✅ 已落地，待 GPU 验证 |
| **P2**     | grid **解 cap**（total=α·cu，α=2 oversubscribe）+ `np_cols`/`gy`/`α` env 扫                                               | prefill bs≥4096 扫出 `fused/atom_fp8 ≥ 1.0` 的 (np_cols, gy, α)；**坐实「最小 CU 灌满 xGMI」** | ✅ 已落地，待 GPU 扫 |
| **P3**     | swizzle 折进 producer + consumer 切 `gemm_sw`（occupancy 2→3）                                                      | bit-exact；rocprofv3 复核 occupancy↑ + perf                                        | ⏳ 待 P1/P2 验证后接力 |
| **P4**(可选) | 叠 `xcd_swizzle`（tile_n=256、`inter%256==0`）                                                                     | `gx|8` 守卫；bit-exact                                                             | ⏳ |


- **红线**：prefill 任一 bucket 目标 `fused/atom(_fp8) ≥ 1.0`；decode 走 §1–§8 `fixedslot` 不回归。
- **GPU 释放后的验证顺序**（每步绿了再下一步；oracle = §7 的 mega-vs-atom 逐元素比对，命令前缀 `TR="torchrun --standalone --nproc_per_node=8 tests/kernels/bench_moe_intranode_stage1_groupgemm.py"`）：
  1. **decode 不回归**：`MORI_SHMEM_HEAP_SIZE=8G $TR --mega --mega-scheme fixedslot --from-bf16 --check-correctness` → `[CORRECTNESS] ... PASS`。
  2. **P1 handshake 正确（共相驻 α=1）**：`MORI_SHMEM_HEAP_SIZE=8G $TR --mega --mega-scheme handshake --from-bf16 --check-correctness` → PASS（键集一致 + 逐元素 < tol）。
  3. **P2 oversubscribe**：`FUSED_MEGA_ALPHA=2 MORI_SHMEM_HEAP_SIZE=8G $TR --mega --mega-scheme handshake --from-bf16 --check-correctness` → 仍 PASS 且无死锁。
  4. **性能扫**：`$TR --mega --mega-scheme handshake --from-bf16 --atom-fp8-dispatch --profiler-time`，扫 `FUSED_MEGA_NP_COLS∈{8,16,32,64}` / `FUSED_MEGA_ALPHA` 找 `fused/atom_fp8 ≥ 1.0`。

### 9.10 收益预估与判据

上界 ≈ 把 dispatch 搬运从关键路径移到 MFMA 之下：`handshake_serial + fill(e0) + max(ΣxGMI, ΣMFMA)` vs 2-launch 的 `Σdispatch + ΣGEMM`。

- 目标网络 = 记录文档 中 `atom_fp8` 下 prefill 仍落后的 **v4_pro / r1_v3（0.80–0.96×）**——把 dispatch 串行暴露吃掉、推过 1.0；v4_flash prefill（已 1.087×）应进一步拉开。
- **关键前提 = P2 坐实**：`np_cols` 真能用「最小 CU」灌满 xGMI（这是相对旧 `np_cols=gx/2` 的核心区别）；若实测显示 dispatch 仍需 ~半数 CU 才饱和（R3 未消），则 overlap 上界受限，需回到「降 dispatch 每字节指令数 / 更宽 vec 写」单独提 producer 效率，再评估。
- **诚实边界**：本设计是把已验证正确的部件重排以攻击三根因；**净赢与否取决于 P2 实测**，定稿后把 prefill 数据补入 §6。

### 9.11 路径选择（host-only，构造期二选一，CUDAGraph 安全）

单 megakernel（`FusedMoEMegaStage1`），构造期按 bucket 选 scheme，**无 2-launch 回退**：


| bucket                             | scheme                                          |
| ---------------------------------- | ----------------------------------------------- |
| decode（小 bs）                       | `fixedslot`（strict-phase + 持久 GEMM，§1–§8，已落地）   |
| prefill（大 bs / handshake）          | `handshake`（block0-自足 + producer/consumer overlap，§9.1–9.10） |


> 入口：`FusedMoEMegaStage1(scheme="fixedslot"|"handshake")`（facade `kernels/fused_moe_megakernel.py`）。kernel 与 dispatch 缓冲：`fused_moe_gemm_2stage.py`（`compile_fused_moe_gemm1(fuse_dispatch=...)`）+ `ep_dispatch_groupmajor_op.py` / `ep_dispatch_groupmajor_kernel.py`。

