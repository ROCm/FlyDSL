# MoE Stage-1 单 launch 融合算子 `mega` —— 架构、流水排布与收益来源（AMD MI355X / CDNA4）

面向**节点内专家并行（EP）推理**的 MoE **stage-1**（dispatch → gate/up GEMM）融合算子。把生产基线的 5-kernel 流水（`量化 → dispatch → moe_sorting → scale-sort → group-GEMM`）压成 **一次 launch 的 2 个 kernel**：①（外部）激活量化；②一个「**dispatch ⊕ 持久 group-GEMM**」融合核（本文档主角）。

- **入口（stage1）**：`kernels/fused_moe_megakernel.py::FusedMoEMegaStage1(scheme="fixedslot")`
- **入口（端到端 stage1+stage2）**：`kernels/fused_moe_stage1_stage2.py::MegaMoE`（§9）
- **kernel**：`kernels/fused_moe_gemm_2stage.py::compile_fused_moe_gemm1(fuse_dispatch="fixedslot", ...)`
- **缓冲**：`kernels/ep_dispatch_groupmajor_op.py::FlyDSLDispatchGroupMajorOp`（只用其对称 / P2P 缓冲，不单独 launch dispatch kernel）
- **正确性**：`bench_moe_intranode_stage1_groupgemm.py --mega --check-correctness`（逐行映射回源 token 后 vs ATOM 生产栈逐元素比对，见 §7）；端到端 `--full-e2e`（§9.6/§10）

> **最终性能（8×MI355X，topk=6，profiler GPU device-time，8 卡均值，`--from-bf16`，全 bit-exact，30-iter × 3 网络 0 FAIL）**：
> - **v4_flash a8w4：bs8–512 全胜 1.01–1.26×**（vs `atom_fp8`）
> - **r1_v3 a4w4：bs8–512 全胜 1.02–1.27×**（vs `atom`）
> - **v4_pro a8w4：bs8 1.03×，bs16+ 0.86–0.99×**（结构性，inter=3072 拿不到 XCD，见 §5）
>
> 完整数据见 §6，收益拆解见 §4。

---

## 1. 架构与数据流

整算子 = **一次 launch 的单个 megakernel**：核内顺序跑「dispatch → 同步 → group-GEMM」三段（持久 grid `gx×gy` 同时驻留）。下面**从上到下**画三张图：①固定槽位（默认，小-中 bs）；②紧凑 compact（大 bs，绕 4GB）；③接 stage2 的适配（atom 契约）。

**名词速查**（图里出现的所有缩写，先看这个）：

| 名字 | 在哪 | 含义 |
| --- | --- | --- |
| `running[le]` | HBM, 每卡 epr 个 | dispatch 期 `atomicAdd` 累的「本地专家 le 当前收到第几个 token」(= 定槽用的游标) |
| `ll_count[le]` | HBM, 每卡 epr 个 | = **每个本地专家收到的 token 总数**（`running` 的稳定拷贝；`ll` 只是前缀，**不是 L1**）。GEMM 读它推 tile |
| `num_valid` | HBM | 本卡所有专家 tile-padded 后的总有效行数（GEMM/combine 的循环界） |
| `gb1` | HBM i64 | grid-barrier **到达计数器**（每块 +1；block0 等它凑满） |
| `done2[pe]` | HBM i32×npes (对称) | 跨卡 done 握手：peer pe「已写完」的 epoch 标记 |
| `meta_flag` | HBM i32 | **放行信号**：block0 置 = 本次 epoch ⇒「元数据就绪、GEMM 可跑」 |
| `epoch` | — | = 第几次 launch（cudagraph replay 不重置 → 单调累加，`epoch=gb1/nblk`） |
| `rx_em / scale_em / srcmap_em` | HBM (对称, P2P) | 收到的 激活 / e8m0行scale / 源键`(k_slot<<24)\|src_global` |
| `le` / `cap` / `src_global` | — | 本地专家号 / 每专家预留槽数`=npes*mtpr` / 源全局token`=src卡*mtpr+src_tok` |

**fence 速查**（图里 `[…]` 标的就是它）：`release`=把我之前的写**推出去**让别人能看见；`acquire`=把别人的写**拉进来**我能看见（成对用）。**agent**=只在本卡 GPU 内可见（便宜）；**system**=跨卡 xGMI 可见（贵，仅跨卡 payload 才用）。激活量化是上一步外部 kernel，不在本核、不计时。

### 1.1 固定槽位（fixed-slot，默认路径）

```
输入(每卡):  A=量化激活 fp8/fp4 + e8m0行scale  |  W=权重 fp4 + e8m0列scale  |  topk_ids + routing wts
   │
   ▼ 【阶段①  DISPATCH】所有块并行 · 单遍 · 纯访存 · 无预排序
   │    for 每个 (token, expert):
   │      slot = le*cap + atomicAdd(目标卡.running[le])              # 在「目标卡」上定槽; cap=npes*mtpr 无溢出
   │      ─P2P 写「目标卡」的 expert-major 缓冲─►  rx_em[slot]=激活 ; scale_em[slot]=e8m0 ;
   │                                              srcmap_em[slot]=(k_slot<<24)|src_global   ◀═ 给 stage2 的钥匙
   │      (atom 接 stage2 时: 同一趟内顺带去重 dest_PE → dest_ctr, 无 payload 后二次 token 扫描)
   │
   ▼ 【同步】strict-phase: 必须「全卡 token 落盘」后才进 GEMM。grid barrier 拆成 到达(N→1) + 放行(1→N) 两半:
   │
   │  每个块(payload 写完):
   │    CTA内 fx.barrier() ─►[fence agent-RELEASE]─► atomicAdd(gb1,+1) ─► 直接跳到 (f) 等放行  (不在 gb1 上自旋)
   │      └本块所有warp落盘    └把本块的写推到agent可见   └「我到了」,每块仅1次
   │
   │  block0 独自做 (b)→(e):
   │    (b) 自旋 gb1==nblk ─►[fence agent-ACQUIRE]   # 本卡所有块到齐 = 「我这卡 P2P 写给别人」的都写完了
   │                                                  #   ⚠ gb1 是「本卡私有」计数器: EP8 = 8 个独立 kernel/8 GPU,
   │                                                  #      各有各的 gb1, 本卡看不到别卡的 gb1; 故 (b) 只知本卡状态,
   │                                                  #      完全不知道别的 7 张卡写完没 (它们此刻还在往我的 rx_em 写)
   │    (c) 跨卡 done-barrier (= ~12µs xGMI「地板」, 删不掉):   # ← 补上「别人写给我」这一向
   │          本卡 GEMM 待会读的 rx_em 正是别卡写进来的, 所以必须等所有别卡都说"写完":
   │          [fence system-RELEASE]─► 给每个 peer 写 done2[本卡]=epoch (P2P, 告诉它「我全写完了」)
   │          ─► 自旋等 done2[每个 peer]==epoch (= 每个 peer 都已告诉我它写完) ─►[fence system-ACQUIRE]
   │          ⇒ 此刻"别人写进我 rx_em 的 token"才全部落盘+可见, GEMM 才能安全读
   │    (d) atom 接 stage2 时: recv-count 握手 → total_recv              # 用 payload loop 内 inline 维护的 dest_ctr
   │        算 num_valid ; running ──拷──► ll_count                     # HBM↔寄存器(readlane warp shuffle), 不碰 LDS
   │    (e) [fence agent-RELEASE]─► meta_flag = epoch                    # 「放行」, 1 个 writer
   │          (running 清零 + [fence system-RELEASE] 推迟到此后, 离开关键路径)
   │
   │  其余所有块:
   │    (f) 自旋 meta_flag > 上轮epoch ─►[fence agent-ACQUIRE]           # N 个 reader 读同一 cacheline, 无原子风暴
   │
   ▼ 【阶段②  持久 group-GEMM】所有块 round-robin 扫 M-tile
   │    每 tile: 由 ll_count 前缀(readlane)现推 (expert e, k)
   │      A 行 = e*cap + k*tile_m                # 直接命中 dispatch 写的槽 → 无 gather
   │      读 rx_em + e8m0 scale(折进 K-loop, 免 swizzle) + W → K-loop ping/pong 双缓冲 MFMA → silu(gate)*up
   │
   ▼ 输出 out[num_valid, inter]  expert-major 固定槽 · dtype=输入dtype(fp8/fp4)+e8m0 scale
       专家 le 真实行 = [le*cap, le*cap+ll_count[le]); pad 行下游按 srcmap 跳过, 无害
```

#### 1.1.1 同步段逐项说明（回答「这些到底是什么」）

decode 是 **strict-phase**：dispatch 全写完 → 同步 → 才开始 GEMM（两段不重叠）。同步要做两件事：① 本卡所有块都写完了吗？② 别的卡 P2P 写进我的 token 都落盘了吗？拆开讲：

- **⚠ 前提：EP8 是 8 个「各自独立」的 kernel（8 进程 / 8 GPU），互相没有共享 grid、没有共享计数器**。`gb1` 是**每张卡私有**的（8 张卡 = 8 个独立 `gb1`），只数**本卡**的块。所以「本卡 `gb1` 凑满」**只代表本卡写完**，本卡**根本看不到别卡的 `gb1`**、不知道别的 7 张卡写没写完——这正是为什么光靠 `gb1`（=步骤 b）**不够**、还必须有跨卡握手（c）。常见误解：「所有块 atomicAdd gb1 = 所有人都写完了」——错，那个 `gb1` 只是**我这一张卡**的。
- **「grid barrier」是什么、怎么保证**（**本卡内**的同步，不跨卡）：就是「本卡全网格同步点」——本卡所有块都到这儿了，谁也别先走。这里**不是**用对称 barrier（那样 N 个块互相自旋、256 路原子风暴），而是**拆成两半**：
  - **到达半（counter，N→1）**：`gb1` 是 HBM 上一个 `i64` 计数器。每个块把自己所有 warp 在 CTA 内 `fx.barrier()` 同步好（保证本块 payload 全落 HBM）后，对 `gb1` 做**一次** `atomicAdd(+1)`，就是「我到了」。**做完就走**（去等 `meta_flag`），**不在 `gb1` 上自旋**——这就是「不自旋」的意思（省掉 N 个块互等的开销）。
  - **放行半（flag，1→N）**：只有 `block0` 自旋等 `gb1` 凑满 `nblk`（= 本卡块数），凑满即「本卡全到齐」。然后 block0 干完跨卡 + 算元数据，把 `meta_flag` 一推，所有块放行。
- **gb1 为什么是「凑满 nblk」而不是「==nblk」**：CUDAGraph 会反复 replay 同一个核且**不重置** `gb1`/`meta_flag`（重置会和 replay 竞争；改成**单调累加**才 graph-safe）。所以 `gb1` 跨 launch 一直涨；block0 是「向上取整到下一个 `nblk` 的整数倍」，`epoch = gb1/nblk` = 第几次 launch。`meta_flag` 也存这个 `epoch`。
- **跨卡 done-barrier 是什么**：dispatch 是 **all-to-all**——别的 7 张卡都会 P2P 写 token 进**我这张卡**的 `rx_em`。本卡块到齐（gb1）只说明**我写给别人**的写完了，**不**代表**别人写给我**的已落盘。所以 block0 再做一道**跨卡**握手：给每个 peer 的 `done2[我]` 写上 `epoch`（P2P `store_i32_system`，告诉它「我全写完了」），再自旋等**我自己**的 `done2[每个 peer]` 都变成 `epoch`（= 每个 peer 都告诉我它写完了）。前面 `fence_system_release` 把我的写推过 xGMI，后面 `fence_system_acquire` 让我能看见 peer 落盘的 payload。
- **「地板」是什么意思**：= **不可压缩的延迟下限**。上面这道跨卡往返走 xGMI（~12µs），是物理硬限——GEMM 必须等所有卡的 token 真正落到本卡 HBM 才能读，strict-phase decode 下藏不掉、删不掉，所以叫「地板」。
- **`meta_flag` 是什么标记**：HBM 上一个 `i32`，存当前 `epoch`，含义 = **「元数据就绪、GEMM 可以开跑」的放行信号**。block0 用 `fence_agent_release` + 一次写把它推进到 `epoch`。各块进同步段时先 `snapshot` 上一轮的值 `_e0`，然后自旋 `meta_flag > _e0`（严格大于上一轮）+ `fence_agent_acquire`。
- **「广播 meta_flag」怎么广播**：不是发消息，就是 **1 个 writer / N 个 reader 共享同一个 HBM cacheline**——block0 写一次，其余 N−1 个块都去**读同一个地址**。硬件把这条 cacheline 缓存住，N 个读者基本命中缓存（**不是** N 个原子，所以没有原子风暴）。这就是「放行半」便宜的原因。
- **`running` 拷成 `ll_count` 是 HBM 还是 LDS**：**HBM↔寄存器，不碰 LDS**。`running[le]`（HBM）是 dispatch 期间 `atomicAdd` 累的「专家 le 收到几个 token」。block0 让 **lane le** 把 `running[le]` 读进**寄存器**，用 **readlane**（warp 内跨 lane 的寄存器 shuffle，不是 LDS）跨 lane 归约出 `num_valid`，再把值写进 `ll_count[le]`（HBM）。拷一份是为了**解耦**：GEMM 读稳定的 `ll_count`，而 `running` 要清零给下一轮——把清零挪到 `meta_flag` 之后（不在关键路径上）。
- **fence 用 agent 还是 system**：`meta_flag`/`num_valid`/`ll_count` 都是**卡内**数据 → 便宜的 **agent** fence 足够；**只有**跨卡 payload 可见性才用贵的 **system** fence（就在 done-barrier 那道）。

> 固定槽位每专家预留 `cap` 行 ⇒ 缓冲 = `epr*npes*mtpr` 行。**大 bs 会逼近 32-bit buffer-resource 的 4GB voffset 上限 → 自动切下面的 compact**。

### 1.2 紧凑 compact（大 bs，绕 4GB）

与固定槽位**同一个核、同一个 GEMM**，只把「定槽」换成「数一遍再紧凑摆放」（行数 ~`topk/cap` 倍小，永不撞 4GB）：

```
输入: 同上                                                       # fence 标注同 §1.1: [agent]=卡内 [system]=跨卡
   │
   ▼ 【阶段0  COUNT】所有块: 每块 LDS 专家直方图(全线程 ds_add) → 收尾每非空专家 1 次全局原子 → local_hist[ge]
   │     然后每块: [fence agent-REL]─► atomicAdd(gb1,+1)             # 到达(同固定槽的「到达半」)
   │
   ▼ 【block0: 跨卡#1(数量 all-gather) + 算紧凑布局】
   │     自旋 gb1==nblk ─►[fence agent-ACQ]
   │     跨卡#1: [fence system-REL]─► 把本卡 local_hist 写进每个 peer 的 bigcnt(P2P) ─► cnt_done 握手 ─►[fence system-ACQ]
   │     CMP:  my_base[ge] = 各专家 tile-padded 前缀 + 本卡发送前缀     # 紧凑 dense 基址(无 cap 浪费)
   │     算 ll_count / num_valid / sorted_expert_ids(_se) / tile_row_base(_trb)
   │     [fence agent-REL]─► meta = epoch        →  各块自旋 meta ─►[fence agent-ACQ]
   │
   ▼ 【阶段②  WRITE】所有块 · for (token, expert):
   │     slot = my_base[expert] + atomicAdd(local_cursor[expert])      # dense 槽(紧凑)
   │     P2P 写目标卡: rx / scale / idx / wts / srcmap[slot]
   │     (atom 接 stage2 时: 此处顺带核内去重 dest_ctr → total_recv, 见 §9.3)
   │
   ▼ 【同步】每块 [fence agent-REL]─►atomicAdd(gb_cnt); block0 自旋齐 ─►[fence agent-ACQ]
   │     跨卡#2 done-barrier: [fence system-REL]─► done2 互发/互等 ─►[fence system-ACQ]   # payload 落盘(~地板)
   │     [fence agent-REL]─► meta2 = epoch  →  各块自旋 meta2 ─►[fence agent-ACQ]
   │
   ▼ 【阶段② GEMM】持久 group-GEMM(同固定槽, 但行基址走 dense)
   │     每 tile: 行基址 = tile_row_base[tile] (dense), 专家 = sorted_expert_ids[tile]; 其余(scale折入/双缓冲/silu)一致
   │
   ▼ 输出 out[num_valid, inter]   紧凑 dense + e8m0 scale
```

> compact 比固定槽多一道**跨卡#1**(数量 all-gather,为了算紧凑基址)→ 故有两道 system fence 地板(数量 + payload);固定槽只有一道(payload)。`local_hist[ge]`=本卡发往全局专家 ge 的 token 数;`bigcnt`=all-gather 后的全卡计数;`my_base[ge]`=紧凑 dense 写入基址。

### 1.3 输出给 stage2 的适配（atom 契约，`atom_contract=True`）

stage2 (GEMM2+combine) **硬编码**：a2 按 **逻辑行 `t*topk+s`** 读、combine 用 `sorted_token_ids` 的 `(t,s)` 反查源。所以接 stage2 时，GEMM epilogue **不写 expert-major/dense 槽行，改按 dispatch 写的 `srcmap` 写到逻辑行**，并额外吐出 stage2 要的元数据。**全部映射**（一把 32-bit 钥匙 `srcmap` 贯穿）：

```
dispatch 写下:  srcmap_em[recv_slot] = (k_slot<<24) | src_global          (src_global = src卡*mtpr + src_tok)
                          │  GEMM1 epilogue 解码 (t=src_global, s=k_slot)
        ┌─────────────────┼──────────────────────────────────────────────────────────┐
        ▼                 ▼                  ▼                  ▼                       ▼
  a2 值                a2 scale          _sti[行]           _se_atom              _sw_atom[t*topk+s]
  写 out[(t*topk+s)     写 sorted(tile)   = (k_slot<<24)     = 本地专家id          = recv routing 权重
     *inter + col]      行                 | src_global       (32-row sub-tile)
   = LOGICAL 行          ▲                  ▲                  ▲                      ▲
   (stage2 GEMM2 这样读) │                  │ stage2: t,s      │ stage2 选 W2          │ stage2 combine 加权
                        └ stage2 按 sorted 行读 scale

  另外两项(combine 用):
    total_recv    = 核内去重的 distinct-recv 数 (Plan A, 直接写进 combine 的 buffer)   → combine 循环界
    tok_id_to_src = identity (arange)        ← 因 _sti 已含 src_global, 反查即源, 无需桥接

  两条路径接 stage2 的差别(其余完全相同):
    · 固定槽位:  A-gather 用 static(e*cap);  _sti/_se_atom 直接 in-place 写 sorted ARG
    · 紧凑 combo: A-gather 用 dense(_trb/_se 占着 ARG);  _sti/_se_atom 写「独立槽 disp[40/41]」;
                 padding 槽的 srcmap 由 block0 核内 sentinel 填(零额外 host 动作) —— 详见 §9.5
```

> 端到端封装 `MegaMoE`（§9）= stage1(atom 契约) ⊕ stage2(GEMM2+combine)，`forward` 内**零桥接、零冗余 dispatch**。**看不懂上面的映射？→ §9.2.1 有一个 2 卡/4 专家的小张量逐行演示**（srcmap / a2值 / a2 scale / _sti / _se_atom 怎么写怎么读）；逐项原理/适配方法见 §9。

---

## 2. 正确路径与输出布局

- **唯一正确路径 = `fixedslot` 持久 round-robin GEMM**：结构上无条件覆盖全部 occupied tile（每 CTA 循环 `ceil(num_valid/tile_m / gy)` 个 tile，按 device 端 `num_valid` 现算，循环兜住）。
- 输出布局：expert-major 固定槽，expert `le` 真实行 `[le*cap, le*cap+count[le])`，`cap = npes*mtpr`。
- 大 bs（`fixedslot` 缓冲将逼近 32-bit num_records 的 4GB 上限，≥3GB）自动切 **compact 紧凑布局**（行数 ~`topk/cap` 更小，OOM-free），同核同正确路径。
- **接 stage2 时**：stage1 用 `atom_contract=True`（a2@logical + srcmap，§9.2）。大 bs 的 compact 接 stage2 需 **compact+atom combo**（`FUSED_MEGA_COMPACT_ATOM=1`，§9.5）——compact dispatch 绕 4GB + GEMM1 写 atom-logical a2，stage2 接线不变、零额外动作、逐位一致。

---

## 3. Kernel 流水排布（pipeline scheduling）

### 3.1 宏观（核内相位）

decode 是 **strict-phase**：`dispatch → 同步 → GEMM`，两相不重叠（decode 无足够 compute 可藏 dispatch 延迟；overlap 在此 regime 不赢，见 §7 附录）。一次 launch 内顺序：

```
[Phase-1 写 payload(peer atomics)]─[CTA内 barrier]─[到达 + block0 等齐 + 跨PE done + lean post-pass + 发 meta]─[各块等 meta]─[Phase-2 GEMM]
       无 compute,纯访存                              ↑ 唯一 grid barrier + 跨 PE 地板(§3.3)                          ↑ 计算密集
```

### 3.2 微观（GEMM K-loop 软件流水）

Phase-2 每个 tile `(bx, by)` 沿 K 做 **ping/pong 双缓冲软件流水**（`use_async_copy=True`）：

```
持久外层:  for mi in [0, ceil(Tm/gy)):   bx = bx_persist + mi*gy        # round-robin M-tile
  K 内层(ping/pong LDS 双缓冲):
    prefetch A,W(K=0) → LDS[ping]                       (async global→LDS)
    for kk in K-steps:
        async copy A,W(kk+1) → LDS[pong]                # 预取下一片,与 MFMA 重叠
        fx.barrier(); MFMA accumulate(LDS[ping]); swap(ping,pong)
    epilogue: silu(gate)*up, (CShuffle) 写 expert-major out
```

要点：① async copy 把下一 K 片 global→LDS 与当前片 MFMA 重叠（藏访存延迟）；② ping/pong LDS 取数与计算不互锁；③ 累加器 `num_acc_n·m_repeat·2`(gate+up) vec4f32 常驻寄存器；④ 持久变体 VGPR≈156（非持久≈104），换无条件全覆盖。

### 3.3 同步协议（dispatch → GEMM 之间）

static-tiles 之后，两相之间只剩 4 件事，其中**跨 PE done-barrier 是唯一不可压的地板**：

```
每块: fx.barrier()(CTA内) → atomic_add_agent(gb1)            # 到达,不自旋
block0: int64_wait_until_equals(gb1, nblk*epoch)             # 等本卡所有块到齐 (N→1)
        → 跨 PE done-barrier(fence_system_release → 发 done2 给 peers → 等 peers → fence_system_acquire)   # ~12µs xGMI 地板,GEMM 必须等所有卡数据
        → atom 接 stage2 时: recv-count 握手累 total_recv     # dest_ctr 已在 payload loop inline 去重维护
        → lean post-pass: readlane 算 num_valid + copy running→ll_count   # ~1µs,无 se/trb 列表
        → fence_AGENT_release + release-store meta_flag=epoch              # meta 是卡内 flag → agent fence 足矣
        → running reset=0 + fence_SYSTEM_release                          # 推迟到 meta 之后,离开关键路径(GEMM 读 ll_count 不读 running)
各块: int32_wait_until_greater_than(meta_flag, e0) + fence_agent_acquire   # 等广播(1 writer/N readers,无原子风暴)
GEMM 块读 peer payload: 自身 agent-acquire 从本卡 HBM 读(payload 已由跨 PE barrier 落地)
```

- **唯一 grid barrier**：N→1 到达 + 1→N 广播（meta_flag）。payload 写是独立 peer atomic，无需「所有块已启动」的初始 barrier。
- **meta 用 agent fence（非 system）**：meta / num_valid / ll_count 都是**卡内**数据，GEMM 块靠自身 agent-acquire 读；peer payload 的跨 PE 可见性由前面的 done-barrier 保证。把 system fence（xGMI flush，贵）降为 agent 是关键收益（§4.6）。
- **`running` reset 推迟**：GEMM 读 `ll_count`（copy 出来的）而非 `running`，故 reset 可挪到 meta 之后、与各块 GEMM 重叠；其 cross-PE 可见性（下次 launch peers 的 remote atomic）由挪后的 system fence 保证。

---

## 4. 收益来源（gain breakdown）

按对最终性能的贡献，从「融合本身」到「逐项削减」：

| # | 收益项 | 机制 | 省掉什么 |
| --- | --- | --- | --- |
| 1 | **5→2 kernel 融合** | dispatch+GEMM 同核、不落地中间 buffer | 2 次 launch + 中间 HBM 往返 |
| 2 | **固定槽 dispatch** | `slot = le*cap + atomic(running[le])`，atom 路径同趟 inline 维护 `dest_ctr` | 整个 `moe_sorting` kernel + payload 后二次 token 去重扫描 |
| 3 | **`raw_a_scale` 折入** | 激活 e8m0 scale 直接进 GEMM K-loop | 整个 `scale-sort` swizzle kernel |
| 4 | **expert-major 连续布局** | dispatch 即写成 GEMM 要的布局 | GEMM 的 gather + 输出的 scatter |
| 5 | **static-tiles GEMM 驱动** | GEMM 用 readlane prefix-scan `ll_count` 现推 `(expert,k)`、行走静态固定槽位 | block0 串行建 se/trb 紧凑 tile 列表（~5–7µs，在关键路径上） |
| 6 | **reset-defer + agent-fence** | `running` reset 挪到 meta 之后；meta fence `system→agent` | 关键路径上一道 **xGMI-flush system fence**（最大的单项削减，tiny bs 因此翻盘） |
| 7 | **XCD swizzle**（`gx|8`） | WGM 重映射，同权重 CTA 聚到同 XCD | W 的 8× 冗余 L2 带宽（§5） |

> #1–#4 是融合架构的底盘（大 bs 净胜的主因）；**#5/#6 专削 decode 小 bs 的「GEMM 之前那段」**——把 dispatch→GEMM 之间从 ~27µs 压到 ~21µs（其中 ~12µs 是跨 PE 不可压地板），让 r1_v3 tiny bs 从 0.96/0.99/1.00 翻到 1.02/1.05/1.07、v4_flash tiny bs 涨到 1.09–1.17。详见 §4.5 / §4.6。

### 4.5 static-tiles（删 post-pass 紧凑列表）

**问题**：原 block0 post-pass 串行扫 per-expert 计数、紧凑产出 `tile_row_base[ci]` / `sorted_expert_ids[ci]`（GEMM 要消费的 tile 列表），实测 ~5–7µs，卡在 dispatch 与 GEMM 之间。

**做法**（`FUSED_MEGA_STATIC_TILES`，默认开）：GEMM 不读紧凑列表，改走**静态固定槽位**——
- loop 前用 **readlane** 把 `ll_count[le]` 的 ntiles 前缀算进寄存器（loop-invariant，64 lane 一次算完）；
- 每个紧凑 tile `bx` 用 readlane prefix-scan 现推 `(expert e, k)`，行 = `e*cap + k*tile_m`（正是 dispatch 写的槽位）；
- block0 只留 `num_valid` + `running→ll_count` copy（全 readlane 并行）；`ll_count` 经 `expected_real` 槽传给 GEMM。
- `epr>64` 自动回退原 se/trb 路径（安全）。

### 4.6 reset-defer + agent-fence（最大单项）

**问题**：static-tiles 后，meta 之前仍有一道 `fence_system_release`（xGMI flush，比预想贵得多）。它本是为 `running` reset 的 cross-PE 可见性而设，但 meta / num_valid / ll_count 全是卡内数据，只需 agent fence。

**做法**：`running` reset（+ 其 system fence）**挪到 meta 之后**（GEMM 读 ll_count、不读 running → reset 不在关键路径）；meta 那道 fence 从 `fence_system_release` 降为 `fence_agent_release`。correctness-preserving（peer payload 由 GEMM 块自身 agent-acquire 从 HBM 读；reset 的 cross-PE 由挪后的 system fence 保证），30-iter × 3 网络 0 FAIL。

---

## 5. XCD swizzle 优化（仅 `gx | 8` 时开）

MI355X 有 **8 个 XCD（chiplet），每个独立 L2**；硬件默认 `block i → XCD (i%8)`。GEMM 中同一 N-tile 的权重 `W[:, nj]`（`[K, tile_n]`，几 MB）被该列所有 M-tile 复用；若散到 8 个 XCD → 各 L2 各拉一份 → 8× 冗余带宽。`xcd_swizzle` 用 WGM 重映射把**共用同权重的 CTA 聚到同一 XCD** → L2 复用。

- **`xcd_swizzle=4` = WGM 组大小**：4 行 M-tile × 全部 gx 个 N-tile 打包给同一 XCD，专家权重 slab 在该 XCD L2 只加载一次。
- **正确性约束 `gx | 8`**（gx ∈ {1,2,4,8}）：WGM 组 `4·gx` 须装进单 XCD 的块块 `cu/8=32` → `32 % (4·gx)==0`。gx=8（v4_flash/r1_v3）合法 + 大收益；gx=12/16 跨 XCD 边界损坏 → facade 自动关。
- **只有持久受益**：持久 CTA 长驻 + 循环扫多 tile，L2 跨迭代反复命中。
- **v4_pro（inter=3072）拿不到 xcd（已穷举）**：合法 tile_n（e_vec=pow2）→ gx ∈ {12,6,4}；`gx|8` 只有 gx=4（tile_n=768，慢 4.6×，累加器爆 + N 并行仅 4 路）；gx=8 需 tile_n=384（大 tile 抵消 xcd，仍 ~0.97×）。∴ v4_pro 保持 tile_n=256 + xcd 自动关。

> facade 守卫：`cu % gx != 0` 自动关 xcd。env `FUSED_MEGA_XCD`（组大小，0=关）。

---

## 6. 性能数据（最终）

口径：8×MI355X，网络 native topk（v4=6,r1_v3=8），**stage1-only `_profiler_ms` device-time**，8 卡均值，`--stage1-out quant --from-bf16`，`scheme=fixedslot`（static-tiles + reset-defer 默认开 / 大 bs 自动 compact），mega decode tile 使用 `_MEGA_DECODE_TILE` 自动表（2026-06-18 刷新）。baseline = **FlyDSL fp8/fp4 dispatch → aiter moe_sorting → aiter mxscale_sort → flydsl GEMM1**（两边都吃预量化 fp8/fp4、都不含激活量化耗时）。

> **公平性（必读）**：baseline 计时圈内的 `a2_e.zero_()`（大 bs 是 GB 级 memset、mega 无等价物)**已移出计时圈**（harness buffer 复位、非 stage1 compute）；remap glue（`.to/where/copy`,FlyDSL-dispatch→aiter-sort 适配)保留(mega 用核内 counting-sort 融合掉它)。rocprofv3 交叉验证:mega 融合核 device-time = `_profiler_ms`(bs512 152.9 vs 153.0µs ✓);GEMM1 两边同一 kernel(纯算力相同),**mega 的赢 = 融合 dispatch+sort+remap(省独立 sort 核 + torch glue + launch)**,不是 GEMM/dispatch 更快。
> **精度**:dequant 后 `relL2(mega,atom)=0`(逐位一致,6 个随机种子 × 多 bs 全 0)、`relL2(mega,f16)=0.027(fp8)/0.143(fp4)`(纯量化噪声)。

**2026-06-18 全 bs `--stage1-sweep` speedup(mega / baseline,profiler time,输出 fp8/fp4 a2)**:

| bs | **v4_flash a8w4** | **v4_pro a8w4** | **r1_v3 a4w4** |
| --- | --- | --- | --- |
| 1 | 0.973 | **1.044** | 0.770 |
| 4 | **1.049** | 0.856 | **1.065** |
| 8 | **1.115** | **1.014** | **1.075** |
| 16 | **1.221** | 0.893 | **1.061** |
| 32 | **1.232** | 0.913 | **1.084** |
| 64 | **1.234** | 0.918 | **1.109** |
| 128 | **1.243** | 0.968 | **1.159** |
| 256 | **1.146** | **1.060** | **1.089** |
| 512 | **1.253** | **1.081** | **1.035** |
| 1024 | **1.077** | **4.972**† | **1.001** |
| 2048 | **1.017** | **4.523**† ◇ | 0.963 |
| 4096 | 0.944 ◇ | **4.156**† ◇ | **1.048** ◇ |
| 8192 | **1.009** ◇ | **4.086**† ◇ | **1.130** ◇ |
| 16384 | **1.031** ◇ | **3.935**† ◇ | **1.112** ◇ |
| 32768 | **1.059** ◇ | 0.947 ◇ | **1.140** ◇ |

◇=compact（其余 fixed-slot）。† v4_pro baseline 在 bs1024–16384 命中异常慢的 ATOM-tuned baseline tile，speedup 被 baseline 弱化放大；bs32768 fallback 到 operator-default 后恢复。网络维度:v4_flash=4096/2048/256(gx=8)、r1_v3=7168/2048/256(gx=8)、v4_pro=7168/3072/384(gx=12)。

**覆盖**:三网络 15 个 bs（1,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768）全部完成，无 skip / OOM / FAIL。完整 us 表见 canvas `moe-stage1-profiler-sweep.canvas.tsx`。

**v4_flash a8w4 stage1 profiler-time(us)**:

| bs | mode | baseline | megav1 | speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | fixedslot | 46.3 | 47.6 | 0.973 |
| 4 | fixedslot | 60.0 | 57.2 | 1.049 |
| 8 | fixedslot | 66.7 | 59.8 | 1.115 |
| 16 | fixedslot | 75.4 | 61.7 | 1.221 |
| 32 | fixedslot | 77.9 | 63.2 | 1.232 |
| 64 | fixedslot | 81.5 | 66.1 | 1.234 |
| 128 | fixedslot | 97.6 | 78.5 | 1.243 |
| 256 | fixedslot | 127.5 | 111.2 | 1.146 |
| 512 | fixedslot | 168.7 | 134.6 | 1.253 |
| 1024 | fixedslot | 249.0 | 231.2 | 1.077 |
| 2048 | fixedslot | 407.8 | 401.2 | 1.017 |
| 4096 | compact | 723.9 | 767.1 | 0.944 |
| 8192 | compact | 1377.2 | 1364.8 | 1.009 |
| 16384 | compact | 2828.9 | 2742.9 | 1.031 |
| 32768 | compact | 5684.2 | 5370.0 | 1.059 |

**v4_pro a8w4 stage1 profiler-time(us)**:

| bs | mode | baseline | megav1 | speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | fixedslot | 60.6 | 58.0 | 1.044 |
| 4 | fixedslot | 140.9 | 164.5 | 0.856 |
| 8 | fixedslot | 179.7 | 177.2 | 1.014 |
| 16 | fixedslot | 221.1 | 247.6 | 0.893 |
| 32 | fixedslot | 225.9 | 247.3 | 0.913 |
| 64 | fixedslot | 230.7 | 251.3 | 0.918 |
| 128 | fixedslot | 251.4 | 259.9 | 0.968 |
| 256 | fixedslot | 283.4 | 267.2 | 1.060 |
| 512 | fixedslot | 389.4 | 360.2 | 1.081 |
| 1024 | fixedslot | 2666.5 | 536.4 | 4.972 |
| 2048 | compact | 4497.9 | 994.4 | 4.523 |
| 4096 | compact | 7070.1 | 1701.3 | 4.156 |
| 8192 | compact | 12752.3 | 3121.3 | 4.086 |
| 16384 | compact | 23580.8 | 5992.3 | 3.935 |
| 32768 | compact | 11363.0 | 11994.0 | 0.947 |

**r1_v3 a4w4 stage1 profiler-time(us)**:

| bs | mode | baseline | megav1 | speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | fixedslot | 52.9 | 68.7 | 0.770 |
| 4 | fixedslot | 97.9 | 92.0 | 1.065 |
| 8 | fixedslot | 113.6 | 105.6 | 1.075 |
| 16 | fixedslot | 121.7 | 114.7 | 1.061 |
| 32 | fixedslot | 122.6 | 113.1 | 1.084 |
| 64 | fixedslot | 128.0 | 115.4 | 1.109 |
| 128 | fixedslot | 142.3 | 122.8 | 1.159 |
| 256 | fixedslot | 173.0 | 158.9 | 1.089 |
| 512 | fixedslot | 234.5 | 226.6 | 1.035 |
| 1024 | fixedslot | 351.4 | 351.1 | 1.001 |
| 2048 | fixedslot | 587.7 | 610.1 | 0.963 |
| 4096 | compact | 1331.7 | 1271.3 | 1.048 |
| 8192 | compact | 2627.6 | 2324.5 | 1.130 |
| 16384 | compact | 4990.2 | 4489.2 | 1.112 |
| 32768 | compact | 9742.0 | 8542.7 | 1.140 |

**规律**:

- **v4_flash**：bs16–512 最稳（1.22–1.25×），高 bs compact 约持平到小赢；bs4096 是 compact 切换低点。
- **r1_v3**：刷新 tile 后 bs256/512/1024 回到约持平或小赢；compact 高 bs 8192+ 明显赢（1.13–1.14×）。
- **v4_pro**：gx=12 自动关 XCD；fixedslot 小 bs 仍有波动。bs1024–16384 的高 speedup 主要来自 baseline ATOM tile 异常慢，不能解读为 mega 绝对性能突增。

---

## 7. 正确性体系

- **oracle = mega vs ATOM 生产栈，逐元素比对**：`bench_... --mega --check-correctness`。两边把每个输出行**映射回源 token**用同一把 32-bit 键 `(k_slot<<24)|(src_pe*mtpr+src_tok)`：
  - mega：`srcmap_em[row]` 即键，值 = `out[row]`（real 行 `[le*cap, le*cap+count)`）。
  - atom：`dispatch_combine` → `aiter.moe_sorting_fwd` → scale-sort → `mixed_moe_gemm1`；recv 槽 `s` 第 `k` 个 LOCAL expert → 键 `(k<<24)|tok_id_to_src[s]`。
- 判据：**键集一致**（无丢 tile / 多余行）**且**每键逐元素误差 < tol（不同 dispatch/sort/GEMM 栈，非 bit-exact，容差 `1e-2`）。丢 tile ⇒ 键集不符；数值损坏 ⇒ 逐行误差 → 都判 FAIL。

### 7.1 量化输出口径（`--stage1-out quant`，生产默认）

两边都输出 fp8/fp4 + e8m0 scale，bench 把每行按键 dequant 回 f32 再比 relL2。三组 relL2 同时报：
- `relL2(mega,atom)` —— **mega vs 生产 atom 的量化 a2**，实测 **0**（逐位一致），是最强独立交叉验证；
- `relL2(mega,f16)` / `relL2(atom,f16)` —— vs 同一份生产 `mixed_moe_gemm1`（f16 输出）的量化噪声地板，fp8≈0.027 / fp4≈0.143；
- `relL2(*,torch)` —— vs bf16 真值 MoE，定位 dispatch/GEMM 是否整体跑偏。

判据：`only_a==only_m==0 且 relL2(mega,f16) ≤ 量化地板（fp8 0.05 / fp4 0.20）`；atom 量化参考健康时（`relL2(atom,f16) ≤ 地板）再加严 `relL2(mega,atom) ≤ tol`。

> **scale 行序坑（接 stage2 必读）**：`mixed_moe_gemm1` 把 fp **值** scatter 到逻辑行 `token*topk+slot`，但 e8m0 **scale** 写在 **sorted 行**（`sorted_token_ids` 的位置，padded 到 `sort_block_m`，tiled-offset `d0*inter_dim + d3*256 + d5*64 + d2*4 + d4*2 + d1`，行序取 sorted_row）。dequant / 接 stage2 时 scale 必须按 sorted_row 读，不能用逻辑行。mega 因值与 scale 同在 slot（内部）行序，读取一致。

### 7.2 接 stage2 — 见 §9（端到端 EP-MoE 单算子 + 所有适配点）

---

## 8. CUDAGraph / 约束

- **CUDAGraph 安全**：整算子一次 launch；grid 同步用原子计数 + 单调 `meta_flag`（per-launch memset / 单调推进，无 reset 竞争）；per-step 输入指针走标量 launch 参数（capture 期不写 device 表）。
- **co-residency**：`gx·gy ≤ cu_num`（`gy = cu_num // gx`，构造期成立）——因核内有 grid barrier，所有块须同时驻留，否则排队块永不到达 → block0 死等。这也是融合核相对独立 GEMM 核（无核内 barrier、可超 cu 启动）为「核内直读 dispatch 产物」付出的代价。
### 8.1 开关 / 环境变量（默认即最优，无需任何设置）

**生产开关（默认值已是最优路径）**：

| env | 默认 | 含义 |
| --- | --- | --- |
| `FUSED_MEGA_STATIC_TILES` | `1`（开） | 静态固定槽位 GEMM 驱动 + reset-defer + agent-fence（§4.5/§4.6 的全部收益）。`epr>64` 自动回退 se/trb 路径(安全)。`=0` 关 |
| `FUSED_MEGA_XCD` | `4`（开） | XCD swizzle 组大小；`cu%gx≠0` 自动关（v4_pro gx=12）。`=0` 关 |
| `FUSED_MEGA_WAVES` | `4` | GEMM waves/block（实测 8 更慢，4 最优） |
| `FUSED_MEGA_MAXNREG` | `0` | 0 = 编译器默认 VGPR 分配 |
| `FLIR_CK_LDS128` / `FLIR_MOE_STAGE1_CSHUFFLE` | `1` | LDS128 / CShuffle epilogue（已调优，勿动） |

> 上述默认值就是本文档报告的最优配置——**不设任何 env 即得最优**。reset-defer + agent-fence 无独立开关，随 `FUSED_MEGA_STATIC_TILES` 默认生效。

**调试 / 诊断开关（默认关，仅排错用，不影响默认路径）**：

| env | 默认 | 用途 |
| --- | --- | --- |
| `FUSED_MEGA_PHASE_TS` | `0` | 核内各 phase 时间戳（block0 `s_memrealtime`→`payload_done[k]` i64），host 差分得每 phase µs |
| `FUSED_MEGA_SKIP_GEMM` | `0` | 跳过 GEMM → 计时 dispatch 前奏 |
| `FUSED_MEGA_COMPACT_ATOM` | `0` | 开启 **compact + atom combo**(大 bs compact dispatch 接 stage2,§9.5);关闭时大 bs e2e 走非紧凑 atom(撞 4GB) |
| `FUSED_MEGA_FORCE_COMPACT` | `0` | 强制 compact 布局(小 bs 也用,做 combo 正确性对拍) |
| `FUSED_MEGA_SWEEP_SIDE` | `both` | bench-only:`mega`/`base` 只计时一侧(rocprofv3 隔离同名 `moe_gemm1_0`) |
| `FUSED_MEGA_GY` | `0` | 手动覆盖 GEMM 网格 gy(0=自动 `cu//gx`,仅扫参用) |

---

## 9. stage1 → stage2 接入（端到端 EP-MoE 单算子 + 所有适配点）

> 本节是「stage1 怎么接 stage2」的权威说明：端到端封装、stage2 的输入契约、stage1 输出与之的**映射关系**、以及**每一个适配点的原理 + 适配方法**。

### 9.0 端到端封装 `MegaMoE`（零桥接）

- **入口**：`kernels/fused_moe_stage1_stage2.py::MegaMoE`，对外两个入口：`forward(x_q, scales, wts, topk_ids) -> bf16 [run_tokens, model_dim]`（前置量化）与 `forward_bf16(x_bf16, wts, topk_ids) -> bf16`（内部 per_1x32 量化，输出与 `forward` 逐位一致）。
- **组成**：`stage1 = FusedMoEMegaStage1(atom_contract=True)` ⊕ `stage2 = FlyDSLMoeGemm2CombineOp`（GEMM2 epilogue 内联 combine Stage-1 P2P scatter + Stage-2/3）。
- **零桥接（与 fixslot 同构）**:`forward` 内**只有**：① `stage1.forward`(1 个 megakernel launch);② `g2.run`(GEMM2+combine)。**无冗余 dispatch、无 host 数据搬运/适配**。两个关键之所以零桥接:
  - `total_recv`(combine 需要的 distinct-recv 计数)由 stage1 **megakernel 核内直接写进 combine op 的 buffer**(Plan A,§9.3),不再像历史那样为拿 fresh `total_recv` 额外跑一次 `comb_op.dispatch`。
  - `tok_id_to_src`(combine 的 scatter-back 映射)= **identity(`arange`)**,因为 stage1 产出的 `sorted_token_ids` 已编码 `src_global = dest_token`(§9.2);故构造期设一次、`forward` 内不碰。

### 9.1 stage2 的输入契约（GEMM2+combine 读什么）

stage2 `g2.run(a2, w2, a2_scale, w2_scale, sorted_token_ids, sorted_expert_ids, sorted_weights, num_valid_ids, wts_buf, ...)` 读取：

| 输入 | 含义 | stage2 怎么用 |
| --- | --- | --- |
| `a2`(=stage1 `_out`) | GEMM1 输出激活(量化 fp8/fp4) | GEMM2 的 A;**行号 = `t*topk+s`**(logical,`mixed_moe_gemm_2stage.py:3268` 硬编码) |
| `a2_scale`(=`_osd`) | a2 的 e8m0 行 scale | 按 **sorted 行** `bx_m` 读(swizzled) |
| `sorted_token_ids`(=`_sti`) | 每行的 `(t=src_global, s=k_slot)` 编码 | ① GEMM2 A 行 = `t*topk+s`;② combine scatter:`t`→`tok_id_to_src[t]`→`(dest_pe,dest_lid)`,`slot=dest_lid*topk+s` |
| `sorted_expert_ids`(=`_se_atom`) | 每(sub-)tile 的**本地**专家 id | GEMM2 选 W2 |
| `sorted_weights`/`wts_buf`(=`_sw_atom`) | 逻辑行 `t*topk+s` 的 routing 权重 | combine Stage-3 加权 |
| `num_valid_ids`(=`_nv`) | 有效行数(tile-padded) | GEMM2/ combine 循环界 |
| `total_recv`(combine op 内) | distinct recv token 数 | combine 循环界(scatter-back) |
| `tok_id_to_src` | recv 行 → 源 (dest_pe,dest_lid) | combine scatter dest;atom 下 = identity |

**核心约束(决定一切适配)**:GEMM2 的 **a2 行号硬编码为 logical `t*topk+s`**,combine 的 scatter 靠 `sorted_token_ids` 的 `(t,s)` + `tok_id_to_src` 解。所以 stage1 必须产出「**a2@logical 行 + `_sti` 编码 `t=src_global,s=k_slot` + identity tis**」这套 **atom 契约**。

### 9.2 atom 契约(`atom_contract=True`)= 映射关系的本体

- **a2 值 @ logical 行**:GEMM1 epilogue 不写「expert-major 槽行」,而是按 **dispatch 写下的 srcmap** 把每行结果写到 `out[(src_global*topk + k_slot) * inter_dim + col]`。srcmap = dispatch 时每个 recv 槽记录的 `(k_slot<<24)|(src_pe*mtpr+src_tok)`(`src_global=src_pe*mtpr+src_tok`)。
- **a2 scale @ sorted 行**:量化 epilogue 把 e8m0 scale 写在 **compact sorted 行**(`bx*sort_block_m+row`),与 stage2 的 scale 读法一一对应(§7.1 的 scale 行序坑)。
- **`_sti`(sorted_token_ids 输出)= srcmap 值**:`_sti[row] = (k_slot<<24)|src_global`。stage2 解 `t=src_global`、`s=k_slot`。
- **`_se_atom`= 本地专家 id**(32-row sub-tile 粒度);**`_sw_atom`= recv routing 权重写到逻辑行 `t*topk+s`**。
- **identity tis**:`t=src_global=src_pe*mtpr+src_lid` → `dest_pe=t>>log2(mtpr)`、`dest_lid=t&(mtpr-1)`,正是源,故 `tok_id_to_src` 恒等。
- **`total_recv` Plan A**(§9.3):核内算 distinct-recv,直接写 combine buffer。

#### 9.2.1 小张量演示（一眼看懂 `srcmap` 怎么贯穿 stage1→stage2）

设定(故意取小): `npes=2`(卡0/卡1), `mtpr=2`(每卡最多2 token), `topk=2`, `experts=4`(卡0 拥 e0/e1, 卡1 拥 e2/e3)。
`src_global = src卡*mtpr + src_tok` ⇒ 卡0 的 token=0,1;卡1 的 token=2,3。**只看「卡0」收到了什么**(它拥 e0/e1):

各源 token 的 topk 路由里、落到卡0(e0/e1)的部分:

```
src_global=0 (卡0 tok0): topk=[e0,e3] → e0@k0 落卡0
src_global=1 (卡0 tok1): topk=[e1,e0] → e1@k0、e0@k1 都落卡0
src_global=2 (卡1 tok0): topk=[e0,e2] → e0@k0 落卡0
src_global=3 (卡1 tok1): topk=[e1,e2] → e1@k0 落卡0
```

**第1步｜dispatch 落地后(GEMM 的输入,按 recv 槽 `i` 索引)**——以下 5 个数组都是「每 recv 槽一项」,`i` 就是数组下标(变量名 = 实际 buffer):

```
i  │ 来自(src,k)│ rx_em[i]=激活(GEMM输入A) │ scale_em[i]=输入scale │ srcmap_em[i]=(k<<24)|src │ ll_count归属专家
0  │ (0,k0)e0  │ X0                       │ sc0                   │ 0                        │ e0
1  │ (1,k1)e0  │ X1                       │ sc1                   │ 16777217 (=0x01000001)   │ e0
2  │ (2,k0)e0  │ X2                       │ sc2                   │ 2                        │ e0
3  │ (1,k0)e1  │ X3                       │ sc3                   │ 1                        │ e1
4  │ (3,k0)e1  │ X4                       │ sc4                   │ 3                        │ e1
```
> `srcmap_em` 解码: `src = v & 0xFFFFFF`,`k = v >> 24`（槽1 的 16777217 → src=1,k=1）。`ll_count=[3,2]`（e0 收3、e1 收2）。

**第2步｜GEMM1 主体**：对 sorted 行 `i` 读 `rx_em[i]`+`scale_em[i]`+`W` → 算出结果行 `Ai = silu(gate)*up`（`A0..A4`,各 `inter` 长）。

**第3步｜GEMM1 epilogue 遍历**——`for i in [0, num_valid):`（**被遍历的就是 `srcmap_em`**，下标变量在核里叫 `_tid_row = bx_m+tx`；先把 `srcmap_em[i]` 载进 LDS `lds_tid` 再解码）：

```
t,s   = decode(srcmap_em[i])                  # 上表
out[(t*topk+s)*inter + col] = Ai              # ① a2 值 → LOGICAL 行
out_scale_sorted[i]         = Ai 的输出e8m0   # ② a2 scale → sorted 行 i
sorted_token_ids[i]         = srcmap_em[i]    # ③ (代码里叫 _sti) 原样透出
sorted_expert_ids[i 的子块] = 本地专家(e0/e1) # ④ (代码里叫 _se_atom)
sw_atom[t*topk+s]           = wts[i]          # ⑤ combine 权重 → LOGICAL 行
```
> 注:`sorted_token_ids` 就是图里的 `_sti`;`sorted_expert_ids` 就是 `_se_atom`;`a2 scale` 就是 `out_scale_sorted`/`_osd`。**值@logical、scale 与 _sti/_se_atom @sorted（=recv 槽序）**。

**第4步｜stage1 输出的全部数组**（卡0,`topk=2`,logical 行数 `=npes*mtpr*topk=8`,sorted 有效行 `=5`）：

```
a2 值 out            (logical, 8 行): [A0,  _,  A3, A1, A2,  _,  A4,  _ ]   # _ = 空(那(token,k)没来卡0)
                       行号:           0   1   2   3   4   5   6   7
sw_atom              (logical, 8 行): [w0,  _,  w3, w1, w2,  _,  w4,  _ ]   # 与 a2 值同布局
a2 scale  (_osd)      (sorted,  5 行): [so0, so1, so2, so3, so4]            # so_i = Ai 的输出e8m0, 按 recv 槽序(=sorted 行)
sorted_token_ids(_sti)(sorted, 5 行): [0,   16777217, 2,  1,  3]
sorted_expert_ids(_se_atom)(sorted):  [e0,  e0,  e0,  e1,  e1]
num_valid = 5
```

**stage2 GEMM2+combine** 反着用同一把钥匙(以 sorted 行 r=1 为例):

```
r=1: t,s = decode(_sti[1]=16777217) = (src=1, k=1)
     A(a2) 读 logical 行 t*topk+s = 1*2+1 = 3      ← 正是 ① 写进去的那行 ✔ (值@logical 对上了)
     scale 读 sorted 行 1                           ← 正是 ② 写的那行 ✔
     W2 选 _se_atom[1]=e0
     算完 down-proj 后 combine scatter-back:
       tok_id_to_src[t]=t=1 (identity)  →  dest_pe = 1>>log2(2) = 0,  dest_lid = 1 & 1 = 1
       → 写回「卡0、本地 token1」的 combine 槽 dest_lid*topk+s = 1*2+1 = 3, 最后按 k 求和归约
```

**一句话**: `srcmap` 是唯一的桥——dispatch 写下它;GEMM1 用它把 a2 摆到 logical 行 + 透出 `_sti`;GEMM2 用 `_sti` 把 a2 读回来、再用 `t` 散射回源 token。**所有"适配"就是让这把钥匙在两段之间对齐**,没有任何额外 host 搬运。

> compact+atom combo 唯一不同(§9.5):上表的 `_sti`/`_se_atom` 因 A-gather 占用了 sorted ARG,改写到**独立槽 disp[40]/[41]**;pad 槽的 `srcmap` 由 block0 核内填 sentinel(`t=npes*mtpr` → stage2 的 `t<tokens` 判否、跳过)。映射规则与上表**完全一致**。

#### 9.2.2 整个输出张量：「按 srcmap 重写」vs「不适配」对照

沿用 §9.2.1 的卡0。记 GEMM1 对 5 个 recv 槽算出的结果行向量为 `A0..A4`(`Ai` = 槽 i 那个 token 的 `silu(gate)*up`,长度 `inter`)。**同一批结果**,两种摆法:

**(A) 不适配(`atom_contract=False`,stage1 standalone 的天然输出)= recv 槽序 / expert-major,紧凑 5 行**
```
a2_noadapt  (5 行, dense, 行号 = recv 槽号)         sorted 元数据(也都按这 5 行槽序):
  行0 = A0   (e0)                                   _se   = [e0,e0,e0, e1,e1]
  行1 = A1   (e0)                                   sorted_token_ids = 槽内 token 序(本地)
  行2 = A2   (e0)                                   srcmap 也记了, 但 a2 *不按它摆*
  行3 = A3   (e1)
  行4 = A4   (e1)
```
这是 `--stage1-sweep` / 只测 stage1 / 训练里 stage1 单独用时的输出:**最紧凑、最快**(连续写,无散射)。

**(B) 适配(`atom_contract=True`)= 按 srcmap 把每行"重写"到 logical 行 `src*topk+k`,稀疏 8 行**
```
重写循环(GEMM1 epilogue 干的就是这件事):
  for 每个 recv 槽 i:
      key      = srcmap[i]
      logical  = (key & 0xFFFFFF)*topk + (key >> 24)     # = src*topk + k
      a2[logical] = Ai                                    # ← "重写"= 一次 scatter

  槽0:key=0        → logical 0 → a2[0]=A0
  槽1:key=16777217 → logical 3 → a2[3]=A1
  槽2:key=2        → logical 4 → a2[4]=A2
  槽3:key=1        → logical 2 → a2[2]=A3
  槽4:key=3        → logical 6 → a2[6]=A4

a2_atom  (8 行 = npes*mtpr*topk, 稀疏):
  行0=A0 │ 行1=空 │ 行2=A3 │ 行3=A1 │ 行4=A2 │ 行5=空 │ 行6=A4 │ 行7=空
        (空行 = 那些 (token,k) 路由到了别的卡, 卡0 这边没有; stage2 靠 _sti 只读非空行, 不碰空行)
```

**两者关系**:`a2_atom[ srcmap 解出的 logical ] = a2_noadapt[槽]`——就是一个**由 srcmap 定义的置换/散射**。stage1 多花的代价 = 这次散射(写到相隔较远的 logical 行,coalescing 略差)+ 维护 8 行稀疏 buffer;换来 stage2 **零适配**直接读。

**stage2 怎么取(逐行演示)** — GEMM2 对每个 sorted 行 `r=0..num_valid-1`,**只用 `_sti[r]`**:解出 `t,s` → 读 `a2[t*topk+s]` 取激活、读 `scale[r]` 取 scale、用 `_se_atom[r]` 选 W2;算完用 `t` 散回源。`_sti=[0,16777217,2,1,3]`:

```
      _sti[r]    → (t, s)   读 a2 下标 = t*topk+s
(B) 适配后 a2_atom = [A0, _, A3, A1, A2, _, A4, _]   (8 行, 行号=logical)
  r=0 │ 0        │ (0,0) │ a2_atom[0] = A0  ✔   ← 这一行本就该是 A0
  r=1 │ 16777217 │ (1,1) │ a2_atom[3] = A1  ✔   ← 散射时 A1 正写到了行3, 这里读回 A1
  r=2 │ 2        │ (2,0) │ a2_atom[4] = A2  ✔
  r=3 │ 1        │ (1,0) │ a2_atom[2] = A3  ✔
  r=4 │ 3        │ (3,0) │ a2_atom[6] = A4  ✔
  ⇒ stage2 按 sorted 行依次拿到 A0 A1 A2 A3 A4 —— 每行的激活都对得上 ✓✓
```

**没适配 stage2 的行为(同一份 `_sti`,但 a2 留在槽序 (A))** — GEMM2 的下标公式不变(还是 `t*topk+s`),但 (A) 的 a2 是**槽序**、只有 5 行,下标对不上 → 读错行 / 越界:

```
(A) 不适配 a2_noadapt = [A0, A1, A2, A3, A4]   (5 行, 行号=recv 槽号)
  r=0 │ (0,0) │ 读 a2_noadapt[0] = A0   (碰巧对: 槽0 的 logical 也是 0)
  r=1 │ (1,1) │ 读 a2_noadapt[3] = A3   ✗ 应是 A1, 却拿到 A3
  r=2 │ (2,0) │ 读 a2_noadapt[4] = A4   ✗ 应是 A2
  r=3 │ (1,0) │ 读 a2_noadapt[2] = A2   ✗ 应是 A3
  r=4 │ (3,0) │ 读 a2_noadapt[6] = 越界! (只有 0..4 行) → 读到垃圾/0
  ⇒ 拿到 A0 A3 A4 A2 垃圾 —— 行全串了 + 越界 → 整个输出全错 (relL2≈1.0)
```
> 根因一句话:GEMM2 用的下标是 **logical** (`t*topk+s`),只有当 a2 也按 logical 摆(B)才对得上;a2 停在槽序(A)就是「拿 logical 下标去索引槽序数组」→ 必错。本项目早期把未重写的 a2 喂 stage2,看到的正是 `relL2≈1.0`。

所以**必须**二选一:

| 方案 | 做法 | 取舍 |
| --- | --- | --- |
| **本项目采用**:stage1 适配 | GEMM1 epilogue 按 srcmap 重写到 logical(上面 (B)) | stage2 **零改动**(已验证);stage1 多一次散射 |
| 备选(未采用):stage2 适配 | 改 GEMM2 读「槽序 a2 + 用 srcmap 取 scatter 目标」 | stage1 省散射;但要动**已验证的 stage2 kernel**,风险大;实测散射并非瓶颈(rocprofv3),故不值 |

> 一句话:「不适配」= 输出停在 (A) 槽序(stage1 自己用最快、但喂 stage2 必错);「适配」= 多做一次 **srcmap 定义的散射**变成 (B) logical 序,让 stage2 不用改一行就能正确读。

### 9.3 Plan A：核内产出 `total_recv`（distinct recv，零额外 dispatch）

- **为何要**:combine 的 scatter-back 循环以「本卡收到多少个 distinct 源 token」为界。一个源 token 可能路由到本卡多个 local expert(占多个 recv 槽),要**去重**才得 distinct 数。历史做法是额外跑一次 `comb_op.dispatch` 拿,是冗余。
- **怎么做(fixed-slot 默认路径)**:payload loop 正在处理 `(src_token,k_slot)` 时,lane0 同步判断这个 `dest_pe` 是否是该 token 的首次出现;首次出现才 `atomic_add_agent(dest_ctr[dest_pe])`。也就是说 `dest_ctr` 与 payload 写**同趟完成**;旧的 payload 后“再扫一遍 token×topk 做去重”的实现已删除。block0 在跨 PE drain 后做一轮 recv-count 握手(signal peer `recv_num[rank]=dest_ctr[peer]+1` → 累加 `total_recv += peer_signal-1`),`recv_num`/`dest_ctr` 核内自复位。结果直接落进 combine op 的 `total_recv`(facade 把 `total_recv_buf` 指给它)。
- **compact 大 bs 的坑 + 修复(关键)**:compact 初版去重用 lane0-only **全局原子**撞 `dest_ctr[npes]`,~1024 个 warp-lane0 抢 8 个计数器 → 串行,把 compact 的 xPE#2 grid-wait **翻倍**(rocprofv3/phase-ts 实证)。**修复 = LDS 直方图去重**:每 block 全线程网格跨步遍历 token、每线程 bitmask 去重 → ds_add 到 LDS dest 直方图 → 每 dest 每 block **1 次全局原子**。把劣化从 e2e 0.83 拉回 **0.92**(与非-atom compact 持平)。

### 9.4 两条接入路径

| | **fixed-slot atom**(bs ≤ ~2048) | **compact + atom combo**(大 bs，env `FUSED_MEGA_COMPACT_ATOM=1`) |
| --- | --- | --- |
| dispatch | 固定槽(cap=npes·mtpr/专家) | compact dense(count-first all-gather,行数 ~topk/cap 小) |
| 为何 | 单趟最快 | 固定槽 rx 在大 bs 撞 **32-bit buffer-resource 4GB voffset wrap** → 必须 compact |
| A-gather | static-tiles(`e*cap+k*tile_m`,readlane prefix) | sparse_tiles 走 `_trb`/`_se`(compact dense tile) |
| atom 输出 emit | in-place 到 `sorted_token_ids`/`sorted_expert_ids` ARG(static A-gather 不占它们) | **必须到独立 disp 槽 40/41**(`_trb`/`_se` 占着 ARG 作 A-gather) |
| a2 行 | logical(`precompute_row` 读 lds_tid=srcmap) | 同 logical(同一段 epilogue) |
| padding mask | static 的 `_kf*tile_m+row < ll_count` | block0 核内 **srcmap gap-fill sentinel**(§9.5) |

> compact+atom 默认由 env 门控(关闭时大 bs e2e 仍走非紧凑 atom、会撞 4GB);两条路径 stage2 接线**完全一致**,只是 stage1 内部布局不同。

### 9.5 compact + atom combo 的所有适配点（原理 + 方法）

GEMM2 硬编码 a2@logical,所以 combo 必须「compact dispatch(绕 4GB)+ GEMM1 写 atom-logical a2」。逐点:

1. **disp 表共存**(`fused_moe_megakernel.py::_build_disp_table`)。compact_ag 不用 idx 19/20(那是非-ag compact 的 compact_base);故 combo 覆写:
   - `disp[19] = srcmap_em`(GEMM atom epilogue 硬编码读它 → a2@logical);
   - `disp[20] = _sw_atom`(combine 权重输出);
   - `disp[36..39] = total_recv / dest_ctr / recv_num / p2p_recv_num`(Plan A,避开 compact_ag 已占的 21=gb_cnt、24=meta2);
   - `disp[40] = _sti`、`disp[41] = _se_atom`(atom 元数据输出,**独立于** A-gather 的 `_trb`/`_se` ARG)。
2. **A-gather 参数 vs 输出分离**(`forward`)。combo 下 `_sorted_arg=_trb`、`_se_arg=_se`(compact tile 结构,A-gather 用),`_wt_arg=wts_em`;`_sti`/`_se_atom` 由 GEMM emit 到 disp 40/41。非-compact atom 则 `_sorted_arg=_sti`、`_se_arg=_se_atom`(static A-gather 不占 ARG,可 in-place)。
3. **GEMM1 emit**(`fused_moe_gemm_2stage.py` lds_tid 装载处,`_ca` 门控)。每 tile 取 `srcmap[bx_m+tx]`→ 写 `_sti[disp40]`、`_sw_atom[disp20]`(real 行,logical index)、`_se_atom[disp41]`(本地专家)。a2-logical 写仍走通用 `precompute_row`(读 lds_tid → t_ok 掩码 → 写 `out[t*topk+s]`)。
4. **padding sentinel = block0 核内 gap-fill**(§9.5 关键,**对齐 fixslot 零额外动作**)。compact dense 的 tile-pad gap 槽(每专家 <tile_m)从不被 peer 写、srcmap 残留上一launch → 必须 sentinel。做法:**block0 在 metadata 阶段**(已算出每专家 dense base `_ebase` + 计数 `ll_count`)对 `srcmap[ebase+cnt : ebase+ntiles*tile_m]` 写 sentinel(`t=npes*mtpr`)。**核内、无 host memset、只填 gap(极少)**;GEMM 直接读 srcmap(pad=sentinel → `t_ok` 自动掩码 → 不写 a2、`_sti` 携带 sentinel → stage2 跳过)。
   - (曾用 host `srcmap_em.fill_` 整 buffer memset → 大 bs 多一道 GB 级动作;也试过核内 prefix-scan count-mask → 每 tile 32-iter readlane 拖慢 ~2%;**gap-fill 两全:零额外 host 动作 + 无 per-tile scan**。)
5. **Plan A 去重**(§9.3):fixed-slot 在 payload loop 内 inline 维护 `dest_ctr`(无第二遍 token pass);combo 的 `total_recv` 走核内 LDS 直方图去重(避免全局原子争用)。
6. **buffer 尺寸**:combo a2(`_out`)= npes·mtpr·topk 行(logical);srcmap_em = nvm(compact dense)。

### 9.6 验证（精度逐位、零额外动作）

- **`--full-e2e`(`MegaMoE` 真链路)**:`mega-vs-baseline relL2 = 0.000`(逐位复刻生产 atom 路径)——v4_flash a8w4 bs 8/128/512/2048/4096/8192/16384、r1_v3 a4w4 bs 8/512/2048/8192,**6 个随机种子(0/1/42/123/7/999)全 0**;默认关(无 env)非紧凑路径不变。
- **零额外动作**:combo `forward` 的额外动作 = 0(`total_recv` 核内、`tok_id_to_src` 一次性、padding gap-fill 核内);compact dispatch 自身的 `local_hist`/`local_cursor` 复位是其固有机制(非 combo 引入),与 fixslot 的 `total_recv.zero_` 同类。
- **gemm2_combine 本体 bit-exact**:`test_profiler_moe_gemm2_combine.py --mode verify` 三网络 `out_tok`/`out_wts` abs/rel max=0。
- **fp8 a2 跨 dtype 拷贝**(若上层手动搬 a2)必须走 `uint8` view(bitcast),直接赋值会数值 cast 损坏字节。

---

## 10. bench：精度 / 性能对比方法与执行

入口 `tests/kernels/bench_moe_intranode_stage1_groupgemm.py`(8-rank torchrun,EP8)。统一口径:`--from-bf16`(生产对齐,量化在计时圈外、不计时)、cudagraph 计时。

### 10.1 三种运行模式

| 模式 | flag | 测什么 | mega / baseline |
| --- | --- | --- | --- |
| **stage1 性能扫** | `--stage1-sweep` | stage1(dispatch+gemm1)纯性能,全 bs | mega=单核;baseline=FlyDSL dispatch→aiter sort→aiter mxscale_sort→flydsl gemm1 |
| **端到端** | `--full-e2e` | stage1+stage2 性能 + 精度 | mega=`MegaMoE`;baseline=atom_fp8 全栈 |
| **stage1 正确性** | `--check-correctness`(+`--mega`) | 逐元素映射回源 token 对拍 | — |

### 10.2 计时标准 = `_profiler_ms`

`_cg_time` 统一走 `_profiler_ms`:capture cudagraph 一次 → profiler 取每次 replay 的 chrome-trace **`gpu_user_annotation` dur**(纯 GPU device-time,**排除 launch 开销**) → 跳前 5 → 均值 → 跨 8 卡 `_all_mean`。**不要**用 CUDA-event wall(小 bs 会把 per-replay launch 开销算进去、稀释比值)。

### 10.3 公平性（必须遵守）

- **量化在计时圈外**:`x_q/x_sc = per_1x32_mx_quant_hip(...)` 在两个 body 之前算好,两边都吃预量化、都不含 quant 耗时。
- **两边都只到 gemm1**(stage1-sweep):无 gemm2/combine。
- **baseline 的 `a2_e.zero_()` 已移出计时圈**(harness buffer 复位、mega 无等价物;大 bs 是 GB 级 memset)。remap glue(`.to/where/copy`)保留(是 baseline dispatch→sort 的必要适配,mega 用核内 counting-sort 融合掉它)。
- **rocprofv3 交叉验证**:`rocprofv3 --kernel-trace -f csv` 逐 kernel(取 median 滤 cold-compile);`FUSED_MEGA_SWEEP_SIDE=mega|base` 隔离(mega 融合核与 baseline gemm1 同名 `moe_gemm1_0`,需分开)。实证 mega 融合核 device-time = `_profiler_ms`。

### 10.4 精度判据

`--full-e2e` 同报三组 relL2:① `relL2(mega,atom)` = **0**(逐位,最强交叉验证);② `relL2(mega/atom, f16)` = 量化噪声地板(fp8≈0.027/0.21、fp4≈0.143/0.29);③ `relL2(*, torch)` = vs bf16 真值(定位整体跑偏)。判据:`mega-vs-baseline==0` 且 `relL2(mega,f16) ≤ 地板`。**多种子**(`--seed`)验证防 trivial 假通过。

### 10.5 常用命令

```bash
# stage1 全 bs 性能扫(v4_flash a8w4)
torchrun --nnodes=1 --nproc_per_node=8 ... bench_moe_intranode_stage1_groupgemm.py \
  --network v4_flash --quant a8w4 --bs-list 1,8,64,512,4096,32768 \
  --stage1-out quant --stage1-sweep --iters 20 --warmup 5
# 端到端正确性 + 性能(大 bs compact 接 stage2 需开 combo env)
FUSED_MEGA_COMPACT_ATOM=1 torchrun ... --network v4_flash --quant a8w4 \
  --bs-list 128,4096,16384 --full-e2e --iters 15 --warmup 5
# 小 bs 强制 compact 做 combo 正确性对拍(多种子)
FUSED_MEGA_COMPACT_ATOM=1 FUSED_MEGA_FORCE_COMPACT=1 torchrun ... \
  --network r1_v3 --quant a4w4 --bs-list 128,2048 --full-e2e --seed 42
# 分阶段打点(dispatch 各相位 tick)
FUSED_MEGA_PHASE_TS=1 torchrun ... --stage1-sweep --bs-list 4096
```

> 网络 keys:`v4_flash`(4096/2048/256/top6)、`v4_pro`(7168/3072/384/top6)、`r1_v3`(7168/2048/256/top8)。tune csv:`(r1_v3,a4w4)`、`(v4_pro,a8w4)` 有,`v4_flash` 无(用默认 tile)。
