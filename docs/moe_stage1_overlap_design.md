# MoE Stage-1 `mega` — decode v4_pro overlap 设计（消 grid barrier / 藏 xGMI 地板）

> 目标：把 v4_pro decode 从 0.86–0.99× 拉回 ≥1.0×，**不动** v4_flash / r1_v3 的既有最优路径、**不降** occupancy。
> 只给核心设计；§2–§7 的性能数字原为分析目标，现已在 8×MI355X/gfx950 真机实测（见 §7.8 / §7.8.6 决定性归因 + §8 验证脚本）。
> §2–§6 是「藏权重」基线（A/B/C，单 floor）；§7 是「藏计算」精细化（渐进 frontier / 数据到即算）。

---

## 1. 问题定位（为什么 v4_pro decode 慢）

> **真机复盘（2026-07-07，见 §7.8.6）**：下面「权重-BW-bound 的 GEMM 是全算子主导项」被真机数据**证实**（GEMM 占 80~88%）；但「同步气泡 / 把 dispatch 藏进 GEMM 是收益本体」这一前提被**证伪**——真实 dispatch 只占 12~19%，且本设计的 frontier readiness 协调开销（+169~577µs）**大于它要隐藏的整个 dispatch**，overlap 未发生（`full=dispatch+GEMM` 精确串行）。故 §2–§7 的 overlap 路线对本 workload 净亏损；真正杠杆是 §1 结尾就已指出的**权重-BW/GEMM 本身**。以下保留原始设计推演，结论以 §7.8.6 为准。

现状 strict-phase 时间线（单卡，块视角；`fixedslot` + static-tiles，见 `fused_moe_gemm_2stage.py:781-1019`）：

```
所有块:  [写 payload(peer atomics)]─[CTA barrier]─┐
                                                  │ 所有非block0块: 空转自旋 meta_flag  ← 全程 IDLE
block0:                              [等本卡到齐]──[跨PE done floor ~12µs]──[算 num_valid/ll_count]──[发 meta]
                                                  └──────────── grid barrier ────────────┘
所有块:                                                                        [各块 acquire]─[GEMM K-loop]
                                                                                              ↑ 计算密集(v4_pro 权重BW-bound)
```

三条结构性开销叠加，恰好吃掉 v4_pro decode 的 ~11%：

1. **同步气泡**：grid barrier 期间**所有块空转**。构造期注释实测 mega **occupancy 仅 5–6%（sync-bound，`fused_moe_gemm_2stage.py:485`）**——绝大多数算力在自旋。
2. **co-residency 上限**：核内 barrier 要求全块常驻 ⇒ `gx*gy ≤ cu_num`（`:472-487`）。baseline 的独立 GEMM 无核内 barrier、可超 cu 启动 ⇒ M 并行更足。mega 为「核内直读 dispatch 产物」付出这份并行度。
3. **block0 串行元数据**在关键路径尾（`num_valid`/`ll_count`，虽已 readlane 化 ~1µs，但仍在 floor 之后串行）。

**为什么偏偏是 v4_pro**：inter=3072 → `gx=12` → `gx∤8` → **XCD swizzle 自动关**（§5 主文档）⇒ 权重 8× 冗余 L2 带宽 ⇒ **GEMM 是权重-BW-bound、且是全算子主导项**。这正是 overlap 的最佳标的：**要藏的东西（权重流）最多、能藏的窗口（floor+气泡）也最实**。反观 v4_flash/r1_v3 有 XCD，权重已便宜，可藏的少——这就是主文档「decode overlap 不赢」结论成立的场景，本设计**不碰它们**。

**关键可分解性**：GEMM 的 **权重加载（W gate/up + w-scale）完全不依赖 dispatch**；只有 **激活读（rx_em + a-scale）** 依赖跨 PE floor。current 设计把两者一起挡在 barrier 之后——**这是浪费的根源**。

---

## 2. 核心思想

> **跨 PE floor 是真数据依赖（激活来自 peer 写），删不掉；但它旁边那段「全块空转」可以填。**
> 把 barrier 的**空转**换成**权重预取**，把 barrier 的**串行元数据**换成**静态自门控 tile**。floor 依旧在，但被权重流盖住；v4_pro 的权重-BW 开销正好塞进这个洞。

三步（互相独立、可分级落地）：

- **A｜静态自门控 tile**：GEMM tile→(expert e, k) 变成**纯静态**映射（不等 block0 的紧凑列表），tile 自己按 live `running[le]` 判占用/掩码。→ 把「算 num_valid/ll_count」搬下关键路径。
- **B｜provisional 元数据（floor 前）**：block0 在**本卡到齐后、跨PE floor 前**先用 live `running` 广播一份 *provisional* 占用（`running` 单调增 ⇒ provisional ≤ final ⇒ **只会漏预取、绝不预取错**）。
- **C｜权重预取 ∥ floor**：各块拿 provisional 占用，在 floor 期间把自己 tile 的权重 K-片流进 L2 / 首个 LDS buffer；floor 放行后**激活一到就算**，权重已热。

---

## 3. Overlap 后的时间线与核内相位

```
所有块:  [写 payload]─[CTA barrier]─[到达 gb1]                                    ┌ floor 放行
block0:                            └[等本卡到齐]─[REL]─►provisional_flag=epoch    │
                                            └────► [跨PE done floor ~12µs] ───────┘
所有块(非block0):  ─►自旋 provisional_flag─[ACQ]─►┌ 权重预取 W(e,k) K-片 → L2/LDS ┐  ← 原本 IDLE 的窗口
                                                └──── 与 floor 并行 ────────────┘
所有块:                          自旋 payload_ready(floor 完)─[ACQ system]─►[GEMM K-loop: 激活热读 + 权重已热]
```

对应 §1.1 主文档风格的核内相位图（`[…]`=fence，agent=卡内 / system=跨卡）：

```
输入(每卡): A=量化激活 + e8m0行scale | W=fp4 + 列scale | topk_ids + wts
   │
   ▼ 【① DISPATCH】所有块并行 · 单遍 · peer atomic 定槽（同 fixedslot，§1.1 不变）
   │     slot = le*cap + atomicAdd(目标卡.running[le]) ; P2P 写 rx/scale/srcmap
   │     每块 payload 写完 ─►[fence agent-REL]─► atomicAdd(gb1,+1)   # 到达，不自旋
   │
   ▼ 【②a block0: provisional（floor 之前！）】
   │     自旋 gb1 凑满 nblk ─►[fence agent-ACQ]
   │     lane le 读 live running[le] → prov_nt[le]=⌈running/tile_m⌉      # 单调，≤ final
   │     写 ll_prov[le] ─►[fence agent-REL]─► prov_flag = epoch          # 早放行(不等 floor)
   │     （随后才做 ▼②b 的跨 PE floor，两者对其余块并行）
   │
   ▼ 【②b block0: 跨 PE done floor】（唯一不可压地板，~12µs xGMI）
   │     [fence system-REL]─► done2 互发 ─► 自旋等 peers ─►[fence system-ACQ]
   │     再读 final running → 覆写 ll_count[le] + num_valid ─►[fence agent-REL]─► payload_ready = epoch
   │     （running reset + system-REL 仍推迟到此后，离关键路径，同 §4.6）
   │
   ▼ 【③ 各块: 权重预取 ∥ floor】  自旋 prov_flag ─►[fence agent-ACQ]
   │     用 ll_prov 的 readlane 前缀现推本块 tile 的 (e,k)（静态 e*cap+k*tile_m）
   │     for 本块前若干 tile: 流 W_gate/W_up 的首 K-片(+w-scale) → L2（async global→LDS[ping]）
   │        # 纯权重、零 dispatch 依赖；把 floor 的空转变成有用带宽
   │
   ▼ 【④ GEMM】各块: 自旋 payload_ready ─►[fence system-ACQ]
   │     用 final ll_count 现推 (e,k) + 行掩码（k*tile_m<ll_count[e]）
   │     读 rx_em + a-scale（激活此刻才可见）→ K-loop 双缓冲 MFMA（权重已热）→ silu(gate)*up
   │
   ▼ 输出 out（expert-major 固定槽，与 strict-phase 逐位一致）
```

> 相对 strict-phase，**删掉的是**：非 block0 块在 floor 期间的纯空转（§1 开销 1）+ block0 元数据在关键路径尾（§1 开销 3）。**保留的是**：跨 PE floor 本体（真依赖）。**新增的是**：一个 provisional flag（1 writer/N reader，无原子风暴）。

---

## 4. 三个改动点（producer / block0 / consumer）

### 4.1 Consumer：静态自门控 tile（改动点 A，最小必需）
GEMM tile 派生已是 readlane 前缀现推 `(e,k)` + 行 = `e*cap+k*tile_m`（`:1848-1899`），本就静态。**唯一变化**：把「循环界 `num_valid` / 前缀源 `ll_count`」的读取时机拆成两次——
- **预取期**读 `ll_prov`（provisional，来自 prov_flag）；
- **计算期**读 `ll_count`（final，来自 payload_ready）做行掩码。

两次都用同一段 readlane 前缀代码，只换 buffer 句柄。tile 占用判据 `k*tile_m < ll_count[e]`（现有 `_cnt_ef` 掩码逻辑 `:1891` 直接复用）。→ **block0 不必在关键路径尾产元数据给 GEMM 起步**。

### 4.2 block0：provisional 早放行（改动点 B）
在现有 `_gwid==0` 分支里，把「读 running→算 ntiles/num_valid」这段（`:958-975`）**复制一份提前到 floor（`:922-929`）之前**，写入新 buffer `ll_prov` 并 `fence_agent_release` + `prov_flag=epoch`。floor 之后保留原逻辑，读 **final** running 覆写 `ll_count`/`num_valid` 并置 `payload_ready`。
- 正确性：`running` 由 peer atomic 单调增 ⇒ `ll_prov ≤ ll_count` ⇒ 预取**只会漏、不会错**；计算期一律用 final。
- 成本：block0 多一次 readlane 前缀（~1µs，与 floor 并行，净零）。

### 4.3 Consumer：权重预取（改动点 C，收益本体）
放行 prov_flag 后、payload_ready 之前，插入一段**只取权重**的预取：对本块（round-robin）负责的 tile，用 provisional (e,k) 计算 `W_gate/W_up` 地址，发 K=0..(预取深度) 的 async global→LDS[ping]（**复用 K-loop 现有的 prologue async-copy 与 ping buffer**，见 §3.2 主文档）。payload_ready 后 K-loop 从 ping 直接进入，激活片补入 pong。
- **深度自适应**：预取深度 = min(填满 floor 所需 K-片, tile 总 K-片)。floor≈12µs → 按 HBM/L2 BW 折算 K-片数，作 env `FUSED_MEGA_PREFETCH_KTILES`（0=关，默认按 shape 表）。
- v4_pro（无 XCD）预取直接热 L2，后续 K-loop 权重命中 L2 ⇒ 削 8× 冗余的**启动段**。

### 4.4 Producer：本设计**不改** dispatch 写序
保持现有 (token,topk) round-robin 写（`:812`）。**不引入** expert-major 重排 / per-peer frontier（那是 §6 的可选进阶，复杂且需实测，本期不做——遵循 Simplicity First）。因此 floor 仍是**全局一道**（不是 per-expert 渐进），但已被权重预取盖住，足以回收 v4_pro 缺口。

---

## 5. 资源与「不劣化」保障

| 维度 | 现状(persist) | 本设计 | 说明 |
|---|---|---|---|
| VGPR | ≈156 | ≈156（+少量地址寄存器，预取复用 K-loop 地址计算） | 不新增 accumulator，不加流水级 |
| LDS/block | 现有 ping/pong | **零新增**（预取写进现有 ping buffer） | occupancy 不受 LDS floor(`:596-603`) 影响 |
| 网格 | `gx*gy=cu_num` | **不变**（不 oversubscribe） | 保 co-residency ⇒ **无死锁风险**；prov/payload flag 均 block0 单写、N 读 cacheline |
| 新增 HBM | — | `ll_prov[epr]` + `prov_flag` i32（对齐 meta_flag 单调 epoch，per-launch 免 memset） | CUDAGraph-safe（同 meta_flag 机制） |

**不劣化的三重闸门**：
1. **shape/arch 门控**：仅 `xcd_swizzle==0`（即 v4_pro 类 `gx∤8`、无 XCD）且 `inter_dim` 大 时启用；v4_flash/r1_v3（有 XCD）**走原 strict-phase，一行不改**。
2. **env 开关** `FUSED_MEGA_OVERLAP`（默认按门控自动，可强制 0 回退）——回退即现路径，天然安全网。
3. **预取深度 0 = 纯 A+B 的静态-tile 化**（无预取带宽消耗），可单独验证「拆 barrier 本身」不亏，再逐级开预取。

---

## 6. 可选进阶（本期不做，仅登记）

- **per-peer expert-frontier 渐进就绪**：各 peer 向目标卡发布单调「expert 前沿」，目标卡 expert e 就绪 ⟺ 所有 peer 前沿越过 e ⇒ 低号 expert 先算、真正 per-expert overlap。**已在 §7 正式展开（G1）**；本条仅保留其 tile 粒度进阶（G2，§7.6）。
- **relax co-residency（oversubscribe GEMM）**：需保证 dispatch 写块与 block0 全常驻、消费块不因排队 producer 而自旋死锁——风险高，单列。

---

## 7. 精细化 overlap：从「藏权重」到「藏计算」（数据到即算 / 渐进 frontier）

> §2–§4 的 A/B/C 把 **floor 的空转**换成**权重预取**——floor 仍整体挡住**所有计算**。本节再进一步：把「等一道全局 floor 再算」换成「**expert 一到齐就算**」，让 slow peer 还在 dispatch 高号 expert 时，本卡已在算低号 expert。这是 §6.1 登记项（G1）的正式展开；**与 §5 门控一致地默认关闭**，是 A/B/C 之上的可选再进一步。

### 7.1 依赖粒度：为什么 expert 是自然粒度，tile 不是（除非改 slot 布局）

grounding：fixedslot 落槽 `slot = le*cap + atomicAdd(dest.running[le])`（`fused_moe_gemm_2stage.py:823-827`）——**目标卡按到达顺序原子分槽**，同一 expert 的行**来自所有 peer 交错**填入。因此：

- 一个 tile `[k*tile_m,(k+1)*tile_m)` 属 expert e，但其 tile_m 行可能来自任意 peer ⇒ **该 tile 就绪 ⟺ 贡献 e 的所有 peer 都写完 e** ⇒ 现布局下 **tile 粒度 == expert 粒度**，细分不出更早的就绪点。
- 要做真·tile 粒度（fast peer 的 e 先算），须让同 expert 内**按 peer 连续**：`slot = e*cap + peer_base[p] + local`。这要么静态 `cap/npes` 分区（padding 浪费 + 改输出行序），要么先 count 一遍（即 compact 方案的 Phase-0，`:1071-1082`）。→ 归 §7.6 进阶（G2）。

所以主线取 **expert-frontier（G1）**：**不动目标卡落槽布局、不动输出行序**，只加「每源卡 per-expert 就绪信号」+「消费端升序、到即算」。

### 7.2 G1 三个改动点（producer / consumer / block0）

**① 源端：per-expert 完成信号（countdown，无 block0 串行）**
每源卡对**每目标卡**维护 `remaining[dest_pe][le]`（初值 = 本卡路由到该 (目标, 局部expert) 的 token 数，由 topk_ids 预统计一遍）。写 payload 的 block 每写完一个 e 行就 `atomic_sub_agent(remaining[dest_pe][le], 1)`；把它驱到 0 的那个 block 先 `fence_system_release`（保证**本 block** 的 e 写已刷到目标 L2），再向该目标卡发布 `dst.expert_ready[src_pe][le] = epoch`（system release）。
- 正确性核心：countdown 归零 ⟺ 所有贡献 e 的 block 都已各自递减完；只要**每个 block 在对某 e 递减前先对其 e 写做一次 release fence**，归零即蕴含「全卡 e 写系统可见」（fence 是 per-thread，但「递减前先 fence」使归零蕴含全员已 fence）。
- 成本：每 block 对它触及的 expert 各一次 system-release fence（≤epr 次，与写重叠）；`npes*epr` 个 i32 ready flag（例 8×64，KB 级）；epoch 编码同 meta_flag（单调、免 per-launch memset、CUDAGraph-safe）。
- 若不愿每 block 多次 fence：退化为 **expert-major 源写序**（block 内 LDS 按 e 分桶后连续 emit，把 `:811` 写循环改 e-outer），则每 block 对每 e 天然连续、访存更规整，frontier 可退成**单调标量** `dst.frontier[src_pe] = 已完成的最高连续 e`（更省 flag）。二取一，按实测。

**② 消费端：升序 + per-expert 门（复用 A 静态 tile、C 预取）**
静态 tile→(e,k) 映射不变（`:1875-1899` readlane 前缀），把**单一 `payload_ready` 门**换成 **per-expert frontier 门**：
- 每 block 的 tile 列表**按 e 升序**排（贴合 frontier 推进）；处理 expert e 的 tile 前，acquire-spin `∀p dst.expert_ready[p][e] >= epoch`；
- 就绪即读 `ll_count[e]`（此刻 `running[e]` 已是**终值**——所有 peer 都过了 e）→ 直接 MFMA（权重已被 C 预热）；
- 等 e 的 frontier 时，C 的权重预取正好流 e（及 e+1）的 W-片 ⇒ 空转→有用带宽。

**③ block0：只留 num_valid 上界，删 done2 单体 floor**
G1 下 `ll_count[e]` 由 frontier 保证终值，**不再需要** block0 的跨 PE done2 单体握手（`:922-929`）来「统一放行」——它退化为 per-expert 增量到达。block0 只需给 GEMM 循环界 `num_valid` 一个**上界**（§4.2 provisional 的 `running` 上界即可，逐 tile 掩码 `k*tile_m<ll_count[e]` 兜底精确性，`:1891`/`:1916`）。→ 原 ~12µs 单体 floor 拆成随数据到达的 per-expert 信号；**残留地板 ≈ 最后一个 expert 的一次 xGMI 往返**，且被前面所有 expert 的计算盖住。

### 7.3 时间线（对比 §3）

```
源卡:  [写 e0 payload]─fence─►ready[e0]  [写 e1]─fence─►ready[e1] ... [写 e_last]─►ready[last]
目标卡各块: 自旋 ready[e0]─ACQ─►[预取W(e0)]►[MFMA e0]┐自旋 ready[e1]─►[MFMA e1]┐ ...
                                                    └── 与源卡写 e1.. 并行 ──┘
残留 floor: 仅 last expert 一次往返（被 e0..e_{last-1} 的计算盖住）
```

### 7.4 正确性
- **单调**：`running`/frontier 均单调增；provisional ≤ final；升序消费 + per-tile 掩码 ⇒ 只会晚算、绝不算错。
- **可见性**：源端 per-block「先 `fence_system_release` 再递减/发 ready」+ 消费端 per-expert acquire ⇒ 与现 done2 的 release/acquire 同级，只是**从 1 道变 epr 道**。
- **epoch/CUDAGraph**：ready flag 用 meta_flag 式单调 epoch（`:1003`/`:1016` 同款）；back-to-back launch 用 `wait_until_greater_than` 防 lap（与 combine 修复 `dispatch_combine_intranode_kernel.py:1151` 同理，避免快 peer 用更晚 epoch 覆写导致 `==` 永久自旋）。

### 7.5 与 A/B/C 关系 / 不劣化
- **复用**：A 静态 tile、C 权重预取原样保留；G1 只把「floor 门」细化成「frontier 门」并删 done2 单体。
- **门控**：§5 三闸门 + 新增 `FUSED_MEGA_FRONTIER`（默认 0=走 A/B/C 单 floor；1=开 G1）。frontier=0 即退回 §3 时间线，天然安全网。
- **occupancy/网格**：不变（仍 co-resident，flag 单写多读 cacheline，无原子风暴）。
- **何时赢**：peer 间 dispatch 速率相近、expert 数 ≥ 数个、且 compute 能填住 dispatch（v4_pro 权重-BW-bound、compute 够重）——正是 §1 标的。peer 严重 straggler 或 expr 极少时 G1≈C（frontier 塌回单 floor），不亏。

### 7.5b 实现状态（`megamoeexp/`，2026-07-07）

G1 已在隔离目录 `megamoeexp/` 实现（生产 `kernels/` 零改动），env 开关 `FUSED_MEGA_FRONTIER=1`：

- **producer**（`fused_moe_gemm_2stage.py` fixedslot 写循环）：expert-parallel 写——每 warp 独占 strided global expert、扫全部 (token,topk) 写其 payload（复用 `_fz_write_pair` helper），写完 `fence_system_release` 后向 dest peer 发 `expert_ready[rank*epr+le]=epoch`（epoch=kernel 入口 meta 快照+1，跨卡单调、免 reset）。
- **block0**：删单体 done2 floor，改 per-expert 聚合——lane-strided 遍历本地 expert，等全 peer `expert_ready[*][e] >= epoch`（lap-safe `>`）、`fence_system_acquire`、`ll_count[e]=running[e]`、reset `running[e]`、发 `cnt_ready[e]=1`；`num_valid=epr*cap`（编译期常量）；meta 更新推到 gb1 全到齐之后。
- **consumer**：静态 tile `(e,k)=(bx//NT, bx%NT)`（NT=cap/tile_m），每 tile 先 spin `cnt_ready[e]>0`（本地 cacheline）+`fence_agent_acquire`，再读 `ll_count[e]` 做行掩码。
- **epoch/reset 关键决策**：`expert_ready` 跨卡 ⇒ 单调 epoch（不 reset,防 lap 用 `>epoch-1`）；`cnt_ready` 本地 ⇒ 主机每 launch in-graph memset=0 + block0 置 1 + 消费端等 `>0`（无跨卡竞争,免 epoch）。二者分治正是绕开 reset-race 的要点。
- **门控/不劣化**：`FUSED_MEGA_FRONTIER=0`（默认）走原路径,`const_expr` 门控使 traced kernel 逐位不变;on 时 `module_name` 加 `_frt` tag 避免 JIT 撞名。新增 buffer:`expert_ready`(sym, npes*epr)、`cnt_ready`(local, epr) + disp slot 26/27/28。

**验证**：本环境为 MI308/gfx942(CDNA3),而 a8w4/a4w4 stage1 用 gfx950/CDNA4 专属 scaled-MFMA(`mfma.scale.f32.16x16x128.f8f6f4`)+128-bit `buffer_load_to_lds` ⇒ 无法实跑。已用 `ARCH=gfx950 COMPILE_ONLY=1` **交叉编译验证**:frontier on/off 均为 gfx950 codegen 通过(v4_flash/v4_pro)。**数值正确性(megaexp≈prod vs oracle)+ 性能(exp-vs-prod、v4_pro decode 缺口)待 gfx950/MI355 空闲机跑 `megamoeexp/bench_megamoeexp.py --frontier` / `probe_stage1.py`**。stage2(gemm2→combine)同为 gfx950-only,本期按指示只调通 stage1。

### 7.6 进阶（登记，本期不做）：G2 真·tile 粒度
per-peer 区（`slot=e*cap+peer_base[p]+local`，复用 compact Phase-0 count 定 base）+ per-(src,e) frontier ⇒ fast peer 的 e-tile 先算。收益上限最高但**改输出行序**（需 srcmap 保逐位一致）、加 count 遍、padding 浪费——仅当 per-(peer,expert) 极不均衡时值得。

### 7.7 开发细节与落地清单（2026-07-07 对齐 §7.3；`megamoeexp/`）

> §7.5b 记录了 G1 的三点改动；本节补齐**让 overlap 收益真正落地**所必需的实现细节。三条 gate 全部 `const_expr` 门控、默认关：`FUSED_MEGA_FRONTIER=0` 逐位等于生产 floor 路径。

**① 空 tile 跳过（make-or-break，`fused_moe_gemm_2stage.py` frontier consumer）。**
G1 的 per-expert frontier **无法**在 block0 编译一份紧凑 tile 列表（那需要「全 expert 计数就绪」= 一道 floor），所以 `num_valid` 只能取**编译期常量 `epr*cap`（= 每 expert 全部 padding 槽）**，consumer 用静态 `(e,k)=(bx//NT, bx%NT)` 遍历**每一个** padding tile。
- 关键：consumer 读到 `ll_count[e]`（cnt_ready 门后）后，必须把 `k*tile_m < ll_count[e]` **并进 `blk_valid`**，让空 tile 跳过整个 `_if_blk`（权重 K-loop）本体，而不是只在 epilogue 做行掩码。
- 不做会怎样：decode 每 expert 实际占用 ≈ 1/NT 个 tile（`cap=npes*mtpr` 远大于实收），consumer 会对 **NT× 多的 tile** 跑权重-BW K-loop —— v4_pro 是权重-BW-bound，frontier 直接比**非-frontier 紧凑静态路径**（block0 post-pass 只发占用 tile，`num_valid=Σ⌈running/tile_m⌉·tile_m`）慢 NT 倍，**收益永远看不到**。这是先前实现与 §7.3 的首要出入。
- 残留开销：persist 循环仍迭代 `epr*NT` 个 tile（空 tile 只做「本地 cacheline spin cnt_ready + acquire + 读 ll_count + 跳过」），无权重加载；相对权重流可忽略。

**② 权重预取 C（收益本体，env `FUSED_MEGA_PREFETCH_KTILES`，默认 0）。**
在 consumer **spin `cnt_ready[e]` 之前**发 depth 条 `w_rsrc` 的 16B/线程 `buffer_load`（fp4 权重 expert-major，expert e 基址 `_ef*(inter_dim*model_dim/4)` i32 字）；载入值**跨 spin 存活于 VGPR**，acquire 后写到 `w_rsrc` 的**首个越界 i32 字**（buffer resource 丢弃越界 store）作 **DCE-safe sink**——零 LDS、不动 occupancy、不碰 §5 的「零新增 LDS」。`rocdl.sched_barrier(0)` 把 load 钉在 spin 前，使**长的 per-expert floor 与权重流并行**，K-loop 命中 L2（削 v4_pro 8× 冗余权重 BW 的启动段）。
- 空 tile 也会预取（占用未知于 floor 前），但 depth 有界 ⇒ 少量浪费（§8 风险 ②），且随即被 ① 跳过。
- 分级 bring-up（与 §5 一致）：先 `PREFETCH_KTILES=0` 验证「① 拆 floor」本身不亏，再 `2→4→8` 逐级开、按 §8 打点找拐点。
- 为什么不用 `rocdl.global_prefetch`：它 lower 到 `llvm.amdgcn.global.prefetch`，**gfx1250 专属**，gfx950 不可选；故用普通 `buffer_load`+越界 sink 走 L2。
- 调优备注（上机）：K-loop 权重用 `b_nt`（non-temporal），若预取被过早逐出，可在上机时试关权重 NT 或调 depth；若 scheduler 把 load 下沉过 spin（预取未与 floor 重叠），加/挪 `sched_barrier`。这两点是 MI355X 上的调参项，不影响正确性。

**③ 生产者开销（Gap3，登记为后续，本期不改）。**
当前 expert-parallel 写（每 warp 独占 strided expert、扫全部 `(token,topk)` 只写本 expert 行）是 **O(total_experts × tokens×topk)** 的 index 扫描（v4_pro 384×）。§7.2① 的 countdown 或 expert-major LDS 分桶可把它降回 O(tokens×topk)；若上机看到**生产者成为瓶颈**（`FUSED_MEGA_PHASE_TS` 的 `_ts(1)-_ts(0)` 段偏大），再按 §7.2① 落地。本期遵循 Simplicity First 不动写序。

**④ 如何确认收益（上机）。** `FUSED_MEGA_PHASE_TS=1` 看 `_ts` 段是否呈 **per-expert 交错**而非单块 floor；stage1-only bench（§8）的 `exp-vs-prod` / `exp-vs-baseline` 加速比是最终判据；`PREFETCH_KTILES` 0 vs N 的差即 C 的净收益。

---

## 7.8 真机实测与实现更新（2026-07-07, 8×MI355X/gfx950）

> **环境更正**：本机实为 **8×MI355X / gfx950 (CDNA4)**，不是文档早先假设的 gfx942。stage1 可**真机实跑**（非交叉编译）。以下均为 8 卡真机结论。生产 `kernels/` 未改动；全部改动在 `megamoeexp/`，`const_expr` 门控、默认关。

### 7.8.1 精度：G1 frontier 从「不可用」修到 bit-exact

真机首跑暴露两个**从未在硬件上跑过**的 bug（§7.5b/§8 此前只做过交叉编译，codegen 通过 ≠ 能跑）：

1. **死锁（consumer 尾块越界）** —— 持久循环界 `ceil(total_m_tiles/gy)` 使部分块尾迭代 `bx ≥ total_m_tiles → _ef = bx//NT ≥ epr`。而 per-tile `cnt_ready` 自旋用**裸地址** `_cr_base + _ef*4`（无 buffer 边界检查）、且在 `blk_valid` 门**之前**执行 → 越界读 `cnt_ready[epr]`（恒 ≤0）→ `int32_wait_until_greater_than(>0)` 永不满足 → 全 8 卡 100% 挂死。**修复**：`_ef = select(blk_valid, _ef, 0)`（尾块被 blk_valid 屏蔽，不影响输出）。
2. **冷启动数值全错（`num_valid` 竞态）** —— frontier consumer 无 metadata gate（per-expert cnt_ready 取代单 floor），却仍从 buffer 读 `num_valid` → 与 block0 的写形成竞态。冷启动 `num_valid=0` 时抢读到 0 → `tiles_per_block=0` → 跳过所有 tile → a2≈0。**关键现象**：只有 launch #1 错，launch #2 起因缓冲残留上次正确数据而「变对」，所以 CUDAGraph perf（warmup 多轮）看不出、**单发 accuracy 才暴露**（`relL2≈0.85`）。**修复**：frontier 下 `num_valid` 直接用编译期常量 `epr*cap`（设计 §7.7① 本就声明是常量）。

修复后（`diag_accuracy.py`，FLOW A=冷启动无 barrier / FLOW B=有 barrier）：`relL2(megaexp_a2, megav1_a2) = 0.000e+00`（**bit-exact**），vs torch-oracle 0.205 < 量化地板 0.22。

### 7.8.2 Kernel 内部详细相位图（单次 megakernel launch，expert-major 路径）

```
输入(rank r): A_q[T,md] fp8/fp4 + a_scale(e8m0) | topk_ids[T,k] | W_fp4[epr,2·inter,md] + w_scale
对称缓冲(mori-shmem, 每卡; peer 可 P2P 写): rx_em/scale_em/idx_em/wts_em/srcmap_em[nvm]
本地/对称计数: running[epr] ll_count[epr] expert_ready[npes·epr] cnt_ready[epr](本地) meta(本地)
网格: gx=N-tiles, gy=cu//gx, co-resident; 块=(by=block_id.x=N-tile, bx_persist=block_id.y=M-round)

┌──────────────────────────── 所有 warp 并行 ────────────────────────────┐
│【① PRODUCER (expert-parallel)】gwid = flat·nwave + warp                  │
│   for ge in {gwid, gwid+gwn, …} < total_experts:   # 每 warp 独占 strided 全局 expert
│      for wk in 0..cur_tok·topk:                     # 扫本卡全部 (token,topk)
│         if topk_ids[wk]==ge:                        # 只写本 expert 的行
│            slot = le·cap + atomicAdd(dst.running[le])   # dst = ge//epr; le = ge%epr
│            P2P→ dst.rx_em/scale/idx/wts/srcmap[slot]
│      [fence system-REL]                             # 本 expert 写全局可见
│      dst.expert_ready[r·epr + le] = epoch           # 发布 per-(src,expert) 就绪 (system store)
│   ── fx.barrier ; tid0:[fence agent-REL]; atomicAdd(gb1,+1)   # 到达，不自旋
└─────────────────────────────────────────────────────────────────────────┘

【② BLOCK0 (gwid==0) 逐 expert 聚合 — 无单体 done2 floor】
   for e in lane..epr step 64:
      for p in 0..npes: int32_wait_until_greater_than(expert_ready[p·epr+e], epoch-1)  # 等全 peer 写完 e
      [fence system-ACQ]; ll_count[e]=running[e]; running[e]=0; [fence agent-REL]; cnt_ready[e]=1
   等 gb1==nblk·L ; [fence agent-REL]; meta=epoch ; [fence system-REL]     # num_valid = epr·cap(常量)

┌──────────────────────────── 所有块并行 (消费) ─────────────────────────┐
│【③ CONSUMER — EXPERT-MAJOR (FUSED_MEGA_EMAJOR=1)】                         │
│  # 本 M-block(bx_persist) 静态拥有 experts E={bx_persist+j·gy : j<NEB}, NEB=⌈epr/gy⌉
│  ── pre-loop 前缀 (unrolled NEB, 每 expert 仅一次): ──
│     for j in 0..NEB: e_j=bx_persist+j·gy(若<epr)
│        int32_wait_until_greater_than(cnt_ready[e_j], 0); [fence agent-ACQ]  # 每 expert 只 spin 1 次
│        nt_j=⌈ll_count[e_j]/tile_m⌉; pref[j+1]=pref[j]+nt_j                  # 动态 occupied tile 数
│     T_b = pref[NEB]                                          # 本块 occupied tile 总数(密集)
│  ── persist 循环 (只遍历 occupied tile, expert 连续): ──
│     for mi in 0..T_b:
│        (e,k) = 前缀反查(mi)                                   # expert-major, 全 occupied
│        A = rx_em[e·cap + k·tile_m …] (fixed-slot) + a_scale ; W = W_fp4[e] + w_scale
│        K-loop 双缓冲 MFMA → silu(gate)·up → a2 @ 逻辑行(srcmap)
│        gpu.barrier                                           # 仅 occupied tile 付 barrier
└─────────────────────────────────────────────────────────────────────────┘
输出 out = a2[npes·mtpr·topk, inter]（逻辑行序，与 strict-phase 逐位一致）
```

**对照 —— 默认 frontier（`FUSED_MEGA_EMAJOR=0`）的 consumer** 是 `epr·cap` **全 padding tile** 的 flat round-robin：`for mi: bx=bx_persist+mi·gy; (e,k)=(bx//NT, bx%NT); spin cnt_ready[e]（每 tile）; acquire; if k·tile_m<ll_count[e]: MFMA else 跳过`。expert-major 把「每 tile spin + 稀疏 occupied」换成「每 expert spin 一次 + 密集 occupied」。

### 7.8.3 env 开关一览（全部 `const_expr` 门控，默认关 → traced kernel 逐位等于生产）

| env | 作用 |
|---|---|
| `FUSED_MEGA_FRONTIER=1` | 开 G1 per-expert frontier（producer expert-parallel + block0 逐 expert + consumer per-expert 门） |
| `FUSED_MEGA_EMAJOR=1` | consumer 改 expert-major 密集内循环（需 FRONTIER；§7.2②/§7.8.2） |
| `FUSED_MEGA_PREFETCH_KTILES=N` | 权重预取 C 深度（§4.3；当前 emajor 路径未接，默认 0） |
| `FUSED_MEGA_SKIP_GEMM=1` | DIAG：consumer 循环清零 → 只计 dispatch |
| `FUSED_MEGA_SKIP_BODY=1` | DIAG：保留循环+per-tile 开销、跳 GEMM body → 隔离循环开销 |

### 7.8.4 实测性能（v4_pro a8w4 bs=64, 8 卡, CUDAGraph device time, us）

| 配置 | full | 说明 |
|---|---|---|
| 生产 megav1 (prod stage1) | **247–254** | 基准 |
| frontier 默认 consumer | 541 (0.47×) | epr·cap 全 padding tile |
| frontier + **expert-major** | **459 (0.54×)** | 密集 occupied tile |

分段归因（`SKIP_GEMM`/`SKIP_BODY`, expert-major）：`SKIP_GEMM ≈ 204µs`，`per-tile 循环开销 ≈ 0`（expert-major 消掉了默认路径的 ~153µs），`consumer(GEMM) ≈ 247µs`。

> **重要更正（2026-07-07，见 §7.8.6）**：上面那 `SKIP_GEMM ≈ 204µs` **不是** dispatch 本身，而是 **frontier readiness 协议的开销**。关掉 frontier 后同一 kernel 的真实 dispatch **只有 35µs**。即之前把「dispatch=213µs」当瓶颈是错的——真瓶颈是 GEMM，frontier 协议纯属额外开销。详见 §7.8.6 的完整 2×2 归因矩阵与结论。

straggler 扫描（`perf_probe.py --skew`，见 §7.8.5）：`skew 0.0→0.54×, 0.6→0.51×, 0.9→0.51×`（skew 无收益，与 §7.8.6 结论一致）。

### 7.8.5 straggler 测试台（`perf_probe.py --skew f`, f∈[0,1]）

对称 8 卡 bench 只能证明「不亏」，证不了「赢」（overlap 需要 per-expert 就绪时间差）。`--skew` 制造该时间差：每 token 的 topk 按权重采样，前 `experts//16` 个「热 expert」权重 = `1+f·200`，f 越大越集中 → 热 expert 收更多 token → 其 dispatch 更晚完成 → per-expert 就绪方差。`f=0` 即均匀（对称）。只改路由，不改权重/数值。低显存实现（随机 fp4 权重、只本卡 local experts、~1GB），不需等大显存窗口。

### 7.8.6 决定性归因：frontier 是净亏损 + 零重叠（2026-07-07 补测，供开发评审）

**方法**：CUDAGraph device time（`perf_probe.py`，无 ATT 逐指令偏差），用 `SKIP_GEMM`（清空 consumer loop = 只测 dispatch）与 `SKIP_BODY`（保留 loop+spin、跳过 GEMM 数学）做相位拆分。对 **frontier off（生产路径）** 与 **frontier on（emajor）** 两条路径各测一遍，得到完整 2×2 矩阵（v4_pro / a8w4 / 8×MI355X）：

| bs | 路径 | dispatch (`SKIP_GEMM`) | GEMM (`full−SKIP_GEMM`) | full | vs ProdS1 |
|---|---|---|---|---|---|
| 64  | 生产路径 frontier **off** | **35µs** | 264µs | 299µs | 1.21× |
| 64  | frontier **on** (emajor) | **204µs** | 248µs | 452µs | **1.83× 慢** |
| 256 | 生产路径 frontier **off** | **63µs** | 266µs | 328µs | 1.14× |
| 256 | frontier **on** (emajor) | **639µs** | 253µs | 893µs | **3.1× 慢** |

（ProdS1 真·生产 = 247µs@bs64 / 288µs@bs256。）

**三条硬结论，全部来自数据：**

1. **kernel 是 GEMM-bound，不是 dispatch-bound。** 生产路径 dispatch 仅占 **12%(bs64) ~ 19%(bs256)**，GEMM ~260µs 占 80~88%。之前「dispatch≈213µs 是瓶颈」是把 frontier 协议自身开销（204µs）误当成 dispatch；**真实 dispatch 只有 35~63µs**。

2. **frontier 零重叠，且是净亏损。** 每个配置都精确满足 `full = dispatch + GEMM`（452≈204+248、893≈639+253、299≈35+264、328≈63+266）——dispatch 与 GEMM **完全串行，零重叠**。frontier 的 readiness 协议（block0 跨 PE 逐 (src,expert) 聚合 spin）把 dispatch 从 35→204µs(bs64)、63→639µs(bs256) 撑大，**overhead(+169 / +577µs) 本身就远超它想隐藏的整个 dispatch**，且随 token 数爆炸增长。消费者的 `int32_wait_until_greater_than` spin（`fused_moe_gemm_2stage.py:2073`）在真实稳态里 device-time ≈0（`SKIP_BODY≈SKIP_GEMM`，差 0.5µs）；ATT 早期看到的「75% spin」是采集时 per-replay barrier 造成的假象，非真实开销。

3. **G2（紧凑消费者 / 权重复用）帮不上忙。** emajor 的 GEMM（248/253µs）与生产路径 GEMM（264/266µs）基本相等——emajor 的稠密内循环已追平生产 GEMM；GEMM 既非瓶颈、也未被 fixed-slot 拖慢。重构消费者最多省 ~10µs。

**对 Plan B（producer/consumer split）的天花板分析：**

- 完美重叠下限 = `max(raw_dispatch, GEMM) ≈ GEMM ≈ 250~266µs`。对 ProdS1：bs64 ≈ 持平（甚至更差，因实验路径 GEMM 本就比 ProdS1 慢 ~6%），bs256 ≈ **8~12% 上限**。
- 要摸到该上限须**同时**：(a) 把 169~577µs 的 readiness 协调开销**彻底消掉**（不是搬移，"便宜 countdown producer" 只解决一部分，且已遇死锁）；(b) 不同 CU 近乎完美并发。
- 当前实现是 **1.83~3.1× 更慢**。投入产出比差：高风险、≤12% 上限、GEMM-bound 下不值得为隐藏 12~19% 的 dispatch 加协调机制。

**建议**：overlap/frontier/split 方向的前提（dispatch/地板 主导）对本 workload **不成立**——已被数据证伪。真正的杠杆是那 ~260µs GEMM（80~88%，对生产与任何新设计都受益）；ATT 显示 occupancy 仅 1 wave/SIMD、VGPR-bound（512 combined）。

**复现**（8×MI355X，rank0 单独挂 rocprofv3 的 `--no-python` 包装 + eager/graph trace 驱动）：
```bash
# device-time 相位归因（无 ATT 偏差）：
for m in "" FUSED_MEGA_SKIP_GEMM=1; do
  env $m FUSED_MEGA_EMAJOR=1 MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=8 \
    megamoeexp/perf_probe.py --network v4_pro --quant a8w4 --tokens 64 --frontier; done   # frontier on
for m in "" FUSED_MEGA_SKIP_GEMM=1; do
  env $m MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=8 \
    megamoeexp/perf_probe.py --network v4_pro --quant a8w4 --tokens 64; done              # frontier off
# ATT 单 CU trace（rank0 only，需 rocprof-trace-decoder .so → /opt/rocm-7.0.2/lib/）：
TRACE_YAML=/tmp/att.yaml FUSED_MEGA_FRONTIER=1 FUSED_MEGA_EMAJOR=1 FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
  torchrun --no-python --standalone --nproc_per_node=8 megamoeexp/trace_wrap.sh
```
（trace 驱动：`megamoeexp/trace_drv.py`；rank0-only 包装：`megamoeexp/trace_wrap.sh`。）

---

## 8. 正确性与上机验证（已在 8×MI355X/gfx950 真机实跑；下为脚本与判据）

> **更正（2026-07-07）**：本机实为 **8×MI355X/gfx950(CDNA4)**，stage1 已**真机实跑**（非交叉编译）——见 §7.8 的真机精度/性能结论。下方 bench 判据仍适用；已验证：`FUSED_MEGA_FRONTIER=0` 逐位等于生产；`FRONTIER=1`（+`EMAJOR=1`）修复两个真机 bug 后 `relL2(megaexp,megav1)=0`（bit-exact）。

**stage1-only bench（`megamoeexp/bench_megamoeexp.py`，只跑 megastage1，不建/不计 stage2）：**
- **精度（8 卡，MAX-reduce，任一 rank 挂即 FAIL）**：accuracy pass 用 `out_dtype="f16"`（反量化 a2，免解包 MX scale）。
  1. `relL2(megaexp_a2, megav1_a2)` —— **硬门 ~0**：G1 frontier / 预取是纯性能改动，不得改数值；
  2. 二者 vs torch a2 oracle `silu(x_t@Wg_e)*(x_t@Wu_e)`（bf16 参考权重），落在 fp8/fp4 量化地板（`--oracle-max` 以上 bs 只跑第 1 项，oracle 需 all_gather peers 的 x/topk_ids）。
- **性能（8 卡，CUDAGraph device time，us；用生产原生 a2 dtype fp8/fp4）**：`megaexp(dev)` vs `megav1(prod stage1)` vs `atom baseline stage1`，报 `exp-vs-prod` / `exp-vs-baseline` 加速比。**只有 stage1 数字，无 e2e。**
- **G1 frontier 分级**：`--frontier` off vs on 精度逐位对拍（应 ~0）+ 性能对比；再 `--prefetch-ktiles 0→4→8` 找 C 的净收益拐点。
- **分相位打点**：`FUSED_MEGA_PHASE_TS=1` 看 `_ts` 段呈 per-expert 交错（而非单块 floor），确认 last-expert 之外的 floor 被前序 expert 计算/预取盖住。
- **占用**：`rocprofv3` 看 `VGPRs`/`Wavefronts` 与生产 persist 一致（frontier 默认关时逐位不变；开时仅多几枚地址/预取寄存器）。

上机命令（8×MI355X）见交付说明；回归护栏：`FUSED_MEGA_FRONTIER=0` 下 megaexp 逐位等于生产 megav1（sanity floor）。

**风险登记**：① 预取深度过大反而挤占计算期带宽（env 可调、默认 0）；② 空 tile / 尾 tile 的预取浪费（depth 有界，随即被 §7.7① 跳过，数值正确）；③ floor 短于预取所需时预取未完（K-loop 自然补齐，正确性无损）；④ scheduler 把预取 load 下沉过 spin（加 `sched_barrier`，见 §7.7②）。

---

## 9. Plan B 落地：block 级 producer/consumer split（`FUSED_MEGA_SPLIT`）

> §7.8.6 证伪了 frontier overlap（GEMM-bound、frontier readiness 协议净亏损、`full=dispatch+GEMM` 零重叠）。**Plan B 是据此提出的新方向**：既然 dispatch（xGMI P2P 写）与 GEMM（HBM 权重读）用**不同的资源**（fabric vs HBM），就在**不同的 CU** 上让二者**真正并发**——少数 CU 当 producer 饱和 xGMI，其余 CU 当 consumer 跑 HBM-bound GEMM，**两者之间没有 grid barrier**。完美重叠下限 = `max(dispatch, GEMM) ≈ GEMM`（§7.8.6 天花板分析）。
>
> 已在 `megamoeexp/fused_moe_gemm_2stage.py` 落地，env 门控 `FUSED_MEGA_SPLIT=1`（需 `FUSED_MEGA_EMAJOR=1`⇒`FUSED_MEGA_FRONTIER=1`）+ `FUSED_MEGA_GP=N`（producer M-round 数，默认 1）。`const_expr` 门控，默认关⇒traced kernel 逐位等于生产。**状态：gfx942 交叉编译通过（`ARCH=gfx950 COMPILE_ONLY=1`，GP=1/2）；死锁修复见 §9.3；真机数值/性能待 gfx950/MI355 验证。**

### 9.1 网格与角色划分（block_id.y 分界，block-uniform）

```
grid: gx = grid_dim.x = N-tiles;  gy = grid_dim.y = M-rounds;  co-resident (gx*gy <= cu_num)
flat = block_idx.y*gx + block_idx.x    # block_idx.y = M-round(分界维), block_idx.x = N-tile
gwid = flat*num_waves + warp           # 全局 warp id

by = block_idx.y ∈ [0, gp)  ─►  producer M-rounds  (gp*gx 个 block)
      ├─ flat == 0            ─►  ①AGGREGATOR      (唯一, by=0 & bx=0)
      └─ flat ∈ [1, gp*gx)    ─►  ②PRODUCER        (gp*gx-1 个)
by ∈ [gp, gy)                ─►  ③CONSUMER         ((gy-gp)*gx 个; 有效 M-round = by-gp, gy_cons = gy-gp)
```

producer/consumer 分属不同 block_id.y ⇒ 不同 block ⇒ 落到不同 CU（co-resident），**三方全程并发、无 grid barrier**。

### 9.2 完整相位图（单次 megakernel launch；三方并发）

```
输入(rank r): A_q[T,md] fp8 + a_scale(e8m0) | topk_ids[T,k]+wts | W_fp4[epr,2·inter,md]+w_scale
对称缓冲(mori-shmem): rx/scale/idx/wts/srcmap[epr·cap] | running[epr] ll_count[epr] |
                       expert_ready[npes·epr](sym, 单调 epoch) | remaining[total_experts](本地, host 偏置=count+1) |
                       cnt_ready[epr](本地, host 每 launch memset 0)
epoch = kernel 入口 meta 快照(L-1) + 1  ── 跨卡单调, 免 reset

━━━━━━━━━━━━━━━━━━━━━━━━━ 三方并发, 无 grid barrier ━━━━━━━━━━━━━━━━━━━━━━━━━
② PRODUCER warp (flat∈[1,gp·gx), warp-strided; prodwid=(flat-1)·nw+warp, prodwn=(gp·gx-1)·nw)
   │  cheap countdown, O(tokens·topk) —— 取代旧 O(experts·tokens) expert-parallel 扫描
   │ ┌ 写 pass:   for wk in [prodwid : T·k : prodwn]:  _fz_write_pair(wk)
   │ │              slot = le·cap + atomicAdd(dst.running[le]) ; P2P→ dst.rx/scale/idx/wts/srcmap[slot]
   │ ├ [fence system-REL]  ── 本 warp 全部写一次性刷到目标 L2 (在任何 decrement 之前)
   │ ├ 递减 pass: for wk: lane0: ge=idx[wk]; old=atomicAdd_system(remaining[ge], -1)
   │ │              if old==1:  _sp_publish(ge)                      ← 1→0 边沿(可能落这)
   │ └ finalize:  for ge in [prodwid : total_experts : prodwn]: lane0: old=atomicAdd_system(remaining[ge],-1)
   │                if old==1:  _sp_publish(ge)                      ← 或落这 (0-token expert 必在此)
   │   _sp_publish(ge): [fence system-REL] ; dst.expert_ready[r·epr + ge%epr] = epoch  (system store)
   ▼   （remaining 偏置 count+1 ⇒ 每 expert 恰好一次 1→0 边沿；见 §9.3 死锁修复）

① AGGREGATOR (flat==0, gwid==0; lane-strided epr)  —— 无单体 done2 floor, 无 gb1 到齐等待(split)
   │  num_valid = epr·cap (编译期常量)
   │  for e in lane..epr step 64:
   │     for p in 0..npes: int32_wait_until_greater_than(expert_ready[p·epr+e], epoch-1)  # 等全 peer 写完 e
   │     [fence system-ACQ] ; ll_count[e]=running[e] ; running[e]=0 ; [fence agent-REL]
   │     cnt_ready[e] = 1                                            # 本地发布, 消费端等 >0
   ▼  [fence agent-REL] ; meta = epoch ; [fence system-REL]         # 供下次 launch(stream 排序)

③ CONSUMER block (by∈[gp,gy); em_byp = by-gp; 拥有 experts E={em_byp + j·gy_cons : j<NEB}, NEB=⌈epr/gy_cons⌉)
   │ ┌ pre-loop 前缀 (unrolled NEB, 每 expert spin 一次):
   │ │   for j<NEB: e_j = em_byp + j·gy_cons (若<epr)
   │ │      int32_wait_until_greater_than(cnt_ready[e_j], 0) ; [fence agent-ACQ]   # 本地 cacheline
   │ │      nt_j=⌈ll_count[e_j]/tile_m⌉ ; pref[j+1]=pref[j]+nt_j                    # occupied tile 数
   │ │   T_b = pref[NEB]                                            # 本块 occupied tile 总数(密集)
   │ └ persist 循环 (只遍历 occupied tile, expert 连续 ⇒ 权重 L2/pipeline 复用):
   │     for mi in 0..T_b: (e,k)=前缀反查(mi)
   │        A = rx[e·cap + k·tile_m …] + a_scale ; W = W_fp4[e] + w_scale
   │        K-loop 双缓冲 MFMA → silu(gate)·up → a2 @ 逻辑行(srcmap) ; gpu.barrier
   ▼        # producer 还在写高号 expert 时, consumer 已在算已就绪的低号 expert ⇒ 真重叠
输出 out = a2[...]（逻辑行序，与 strict-phase 逐位一致）
```

**与 §7.8.2（非-split emajor）的区别**：非-split 是「所有块先并行 producer → block0 单体聚合 → 所有块转 consumer」（producer 与 consumer 在时间上串行，靠 frontier 门做 per-expert 交错，但 §7.8.6 实测零重叠）。split 把 producer 与 consumer **绑到不同 block_id.y**，二者从 kernel 启动就在**不同 CU 上同时跑**，consumer 的 `cnt_ready` spin 与 producer 的 countdown 天然重叠——这才是 Plan B 想要的 CU 级并发。

### 9.3 死锁修复：publish-on-1→0-edge（本次改动）

cheap countdown 的 readiness 依赖「把 `remaining[ge]` 驱到 0 的那次递减」来 publish `expert_ready[ge]`。host 把 `remaining[ge]` 偏置成 `count[ge]+1`（`fused_moe_megakernel.py:505-521`，使 0-token expert 也能被 publish）⇒ 对每个 expert **恰好一次**递减产生 1→0 边沿。

- **bug**：该边沿**可能落在递减 pass（某行），也可能落在 finalize pass**。旧代码只在 finalize pass 检测边沿并 publish；当边沿落在递减 pass（finalize 已先跑过）时，该 expert **永不 publish** ⇒ aggregator 的 `int32_wait_until_greater_than` 永久自旋 ⇒ **全卡死锁**。
- **修复**（`fused_moe_gemm_2stage.py:988-1021`）：抽出 lane0-only 的 `_sp_publish(ge)`（`[fence system-REL]` + 写 peer `expert_ready`），在**递减 pass 与 finalize pass 都检测 `old==1` 边沿**并调用它。每 expert 只有一次 1→0 边沿 ⇒ **恰好 publish 一次**，不重复。producer 在任何递减之前已 `fence_system_release` 过本 warp 的全部写 ⇒ 1→0 归零蕴含「全部贡献 warp 的 e-行系统可见」，release→acquire 链对 aggregator 成立。

### 9.4 状态与验证清单

- ✅ **交叉编译**（gfx942 host, `ARCH=gfx950 COMPILE_ONLY=1`）：`FUSED_MEGA_SPLIT=1 EMAJOR=1 FRONTIER=1`，`GP=1` 与 `GP=2` 均 `compilation succeeded (arch=gfx950)`。
- ⬜ **真机（gfx950/MI355）**：`bench_megamoeexp.py` 数值 `relL2(split, megav1)≈0`（split 是纯性能改动）+ 死锁不复现；`perf_probe.py` 报 `exp-vs-prod` 加速比，对照 §7.8.6 天花板（bs256 上限 ~8-12%）。producer CU 数 = `gp·gx`，需扫 `FUSED_MEGA_GP` 找「够饱和 xGMI 又不挤占 consumer」的拐点。
- **风险**：① producer CU 太少 ⇒ dispatch 拖尾成瓶颈；太多 ⇒ 挤占 consumer GEMM 并行度（`GP` 可调）。② consumer `cnt_ready` 的本地 cacheline spin 若 device-time 非 0（§7.8.6 稳态 ≈0），重叠收益打折。③ co-resident 约束下 `gp·gx` 个 block 让给 producer，consumer 的 M-parallelism 降低——GEMM-bound 下这是主要代价，须实测净收益。
