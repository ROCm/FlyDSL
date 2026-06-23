# MoE Stage-1 Megakernel — 实验记录 & 进度（records / notes）

> 设计与算子说明以 `docs/moe_stage1_mega.md` 为准。本文件只存**有价值的实验记录、性能基线、进度日志**。
> 口径统一：8×MI355X（gfx950，CDNA4），topk=6，profiler GPU device-time，8 卡均值，`--from-bf16`。

---

## 1. 进度日志（progress log）

- **2026-06-18 fixed-slot Plan A inline `dest_ctr`（默认路径，删除旧二次 pass）。**
  - **问题**：fixed-slot atom_contract 为了给 stage2 combine 产 `total_recv`，payload 写完后又做了一遍 token×topk 扫描来按 distinct dest-PE 去重并累 `dest_ctr`。这条二次 pass 在小/中 bs 直接叠到 dispatch→GEMM 关键路径上。
  - **改动**：把 `dest_ctr[dpe]++` 合入 payload loop。每个 `(src_token,k_slot)` 的 lane0 判断当前 `dest_pe` 是否为该 token 的首次出现，首次才计数；`running[le]` 仍按 expert-copy 计数、slot 语义不变。旧的 payload 后二次 token pass 已删除，module suffix bump 到 `_v34`。
  - **验证**：`--full-e2e` v4_flash a8w4 bs 1/8/16/64/512/2048 全 PASS，`mega-vs-baseline` 约 `3e-5~1e-4`;v4_pro a8w4 bs 1/8/16/64/512 全 PASS，`mega-vs-baseline` 约 `0~4.5e-5`。
  - **性能快照**：v4_flash stage1 bs1 54.6→47.9µs、bs512 205.0→190.9µs、bs2048 653.0→532.0µs；v4_pro bs512 465.4→449.8µs。无回退，保留为默认路径。

- **2026-06-16 stage1 phase 级打点 + compact dispatch 两项 LDS 优化(bs4096 0.916→0.974)。**
  - **打点工具**：核内 `rocdl.s_memrealtime()` 时间戳(新增到 rocdl),`FUSED_MEGA_PHASE_TS=1` 开启(const_expr 门控,关闭零开销),block0 lane0 把各 phase tick 写进 `addr_payload_done`(i64,非紧凑/compact 路径都插)。host 端读回 + 按 `总−baseline_gemm` 标定 scale(≈10ns/tick=100MHz,大 bs 校验)。baseline 多核用 cudagraph 累计差拆成 dispatch/moe_sorting/mxscale_sort/gemm1。`--stage1-sweep` 全 bs 跑 megav1(stage1) vs baseline(fp8 dispatch→sort→gemm1);`FUSED_MEGA_FORCE_COMPACT=1` 小 bs 强制 compact 做正确性对拍。
  - **stage1 分核对比(a8w4 v4_flash,µs;两边都吃预量化 fp8、不含激活量化、输出 fp8 a2)**：
    - **bs128(非紧凑 fixed-slot,megav1 赢 1.30×)**：baseline = dispatch 32 + moe_sorting 28 + mxscale_sort 4 + gemm1 73 = **136**;megav1 = 融合 dispatch+sort(单核)~22 + gemm ~82 = **104**。→ **megav1 把 baseline 的 dispatch+sort+scalesort(64µs)融合成 ~22µs(counting-sort 在 dispatch 核内近乎免费)**,GEMM 两边相当。非紧凑 dispatch 细分:write~6–13、xPE 握手~14–18、postpass~5。
    - **bs4096(compact,优化前 megav1 0.916×)**:baseline = dispatch 281 + moe_sorting 69 + mxscale_sort 8 + gemm1 562 = **919**;megav1 总 **1003**,dispatch 细分(初始)= count 25 + (allgather+xPE#1)53 + write 219 + xPE#2 146 = **444**,GEMM ~559。→ 劣化全在 routing(444 vs 358),其中 **xPE#2(payload done-barrier)= block0 等所有 peer 满载 P2P 写落盘**,随 payload 体量涨(小 bs ~15µs→大 bs ~146µs),是大 bs 转劣根因。
  - **★优化①(COUNT 上 LDS 直方图)**：compact PHASE-0 的 count 原是 **lane0-only + 全局(L2)原子**(1/64 lane 利用率 + 256 计数器争用);改成 **每 block 全 64-lane LDS 直方图(`memref.atomic_rmw`=ds_add,~20ns)→ 收尾每非空专家 1 次全局原子归约**。新增 LDS region(复用 pong,与 GEMM 时序错开)。**count 2532→188 ticks(~25µs→~2µs,13.5×)**,bs4096 0.916→0.935。
  - **★优化②(CMP+metadata 读 LDS)**：compact 的 my_base 前缀 + tile metadata 原在 **block0-warp0 从 HBM 串行规约 bigcnt(~512 次依赖 L2 load = 40µs)**;改成 **一次性 64-lane 并行把 bigcnt 规约成 `cs[ge]`(总数)+`sp[ge]`(sender<rank 前缀)进 LDS**(warp0 内用 `s_waitcnt` 排序,不能 fx.barrier 否则与其它 warp 死锁),CMP/metadata 改读 LDS。**CMP+meta 3964→1552 ticks(~40µs→~15µs,2.5×)**,bs4096 0.935→**0.974**。
  - **正确性**：两次都 `FUSED_MEGA_FORCE_COMPACT=1` 小 bs(128)`--check-correctness` 对拍 `relL2(mega,atom)=0.000`(逐元素相同)、expert_mismatch=0、`relL2(mega,torch)=0.171`(=a8w4 量化地板,权重 fp4 主导)、PASS。
  - **剩余大头(未做)**:write(~194µs)+ xPE#2(~162µs)= dispatch 的 ~93%,是 payload P2P scatter + drain barrier。攻法 = **write 与 per-dest 完成信号重叠**(边写边发,drain 与写流水)/ 减写放大。收益最大但风险最高。原始数据 `_runlogs/{phasets_*,lds_count,cmp_perf,subts}.log`。

- **2026-06-16（续2）设计:compact 接 stage2(大 bs e2e 用 compact dispatch + atom-logical a2 输出)——已摸清,GEMM 有编译期耦合,待 GPU 验。**
  - **硬约束**:GEMM2 读 a2 行号**硬编码 logical**(`mixed_moe_gemm_2stage.py:3268-3281`:`a2_row = (sti[row]&0xFFFFFF)*topk + (sti[row]>>24)`)。⇒ a2 必须 atom-logical 排布,stage2 不接受 compact dense。combine 同理(`row→t→tok_id_to_src[t]→(dest_pe,dest_lid)`,atom 靠 `t=src_global+identity srcmap`)。
  - **唯一可行接法**(用户已选):**compact dispatch(dense rx,绕开 4GB 输入墙)+ GEMM1 写 atom-logical a2**(stage2 完全不动、已验证)。compact 的 `src_meta[dense_slot]=src_global|(k<<24)`(dispatch 写,`fused_moe_gemm_2stage.py:1857`)正好给 GEMM1 epilogue 写 logical 行 + 产 `_sti` 用。
  - **改动点(已定位)**:
    - **disp 表**(`fused_moe_megakernel.py:325-347`):compact_ag **不用 idx 19/20**(那是非-ag compact 的 compact_base);故 `disp[19]=srcmap_em`、`disp[20]=_sw_atom`(GEMM atom epilogue 硬编码读这俩:`:3581`/`:2575`)。Plan A buffers(total_recv/dest_ctr/recv_num/p2p_recv_num)挂**新 idx 36-39**(compact_ag 已占 21=gb_cnt、24=meta2,不能复用 atom 原来的 21-24)。
    - **compact_ag dispatch**(`fused_moe_gemm_2stage.py:1646+`):`_sm` 已写(`:1857`);补 Plan A per-token distinct-dest dedup→dest_ctr + 跨PE recv-count→total_recv(照搬非紧凑 `:932-1003`,索引改 36-39)。
    - **GEMM1 a2-logical 写**:`precompute_row`(`:3984`)读 `lds_tid`=srcmap→logical 行,**已不依赖 static-tiles**,compact 直接可用(compact 下 sorted-row==dense-slot,`srcmap[bx_m+tx]` 有效)。
    - **facade**:去 `assert not(atom_contract and compact)`(`:131`);`forward` compact 时也返回 `_sti`(`:457`);combine `tok_id_to_src` 保持 identity。
    - **MegaMoE**:大 bs stage1 用 compact(atom_contract=True+compact 共存),stage2 接线不变。
  - **★已落地(门控 `FUSED_MEGA_COMPACT_ATOM=1`,默认关,关闭时所有现有路径 byte-identical;已 py_compile + lint 过)**:
    - facade(`fused_moe_megakernel.py`):`_compact_atom` env 门控;开时跳过 assert、disp 表 `[19]=srcmap_em`/`[20]=_sw_atom`/`[36-39]=total_recv/dest_ctr/recv_num/p2p_recv_num`。
    - kernel compact_ag(`fused_moe_gemm_2stage.py`):`const_expr(_atom_contract)` 下加 Plan A per-token distinct-dest dedup→dest_ctr(写后、grid barrier 前)+ block0 跨PE recv-count→total_recv(done2 drain 后)。读 disp 36-39。
    - `fused_moe_stage1_stage2.py`:注释说明大 bs compact+atom 由该 env 开启,stage2 接线不变。
    - 默认关时:plain compact 编译 `_atom_contract=False`→Plan A DCE;disp 块 `if _compact_atom` 跳过;非紧凑 atom 不走 compact_ag。零影响。
  - **★t5 已完成(GPU 验证,bit-exact)**:compact+atom 组合跑通。
    - **A-gather 无需解耦**:compact 用 `sparse_tiles`(`bx_m=_trb[bx]`)+ 非 static 路径,`_static_tiles` 对 compact 本就 False(`:459`),所以 A-gather = 普通 compact(已验证),不变。
    - **真正缺的 = emit + 输出 buffer 分离 + padding sentinel**:① compact 的 `_trb`/`_se` 作 A-gather ARG 被占用,故 `_sti`/`_se_atom` emit 到**独立 disp 40/41**(facade `_sorted_arg=_trb`、`_se_arg=_se`,combo 专属);② GEMM1 在 lds_tid 装载处为 `_ca` 增加 emit(`_sti`@disp40、`_sw_atom`@disp20、`_se_atom`@disp41 sub-tile 本地专家);③ a2-logical 写(`precompute_row` 读 lds_tid→logical,t_ok 掩码)本就通用,直接复用。
    - **关键 bug + 修复**:padding(tile-pad)dense 槽的 srcmap 是上一launch 残留 → 起初用 `my_base[le]`(disp34)算 pos 掩码**错**(my_base 含 sender 前缀 `_eb+_sp`,是写目标基址非本地 recv 基址)→ relL2=0.94。改为 **facade 每 launch sentinel-prefill `srcmap_em`(t=npes·mtpr,in-graph memset,cudagraph 安全)** → padding 槽自带 sentinel,GEMM t_ok 自动掩码,无需 count 掩码 → **bit-exact PASS**。
    - **正确性(`FUSED_MEGA_COMPACT_ATOM=1` + `--full-e2e`,mega-vs-baseline=0.000 全 bit-exact)**:v4_flash a8w4 bs 8/128/512/2048/4096/8192;r1_v3 a4w4 bs 8/128/512/2048/8192;v4_pro a4w4 mega==atom(两者都 oracle FAIL=v4_pro a4w4 无 tune 的已知参考问题,与 combo 无关)。默认关(无 env)bs128 e2e 仍 bit-exact、非紧凑路径不变。
    - **★性能根因定位 + 修复(2026-06-16 续3)**:初版 combo e2e 0.83(stage1 0.77)。
      - **证伪 dense 假设**:做了完整 dense-a2 路径(stage1 写 dense + stage2 加 `a2_dense` 读 dense),bit-exact 但**性能不变(仍 0.77)**⇒ atom-logical 散写**不是**根因。已**全部回退** dense 改动(stage2 不动)。
      - **★真根因 = Plan A `dest_ctr` 去重的全局原子争用**。phase-ts(bs4096)对比:combo vs 非-atom compact 仅 **xPE#2 grid-wait 翻倍(12260→26112 ticks)**,占整个劣化 ~92%;其余阶段(count/pub/cmp/write)几乎相同。`FUSED_MEGA_SKIP_PLANA=1` 跳过去重 → grid-wait 回 12492、stage1 **0.817→0.948**,确证。根因 = 去重用 lane0-only 全局原子撞 `dest_ctr[npes=8]`,~1024 个 warp-lane0 争 8 个计数器 → 串行(和最初 count-phase 那个 bug 同类)。
      - **★修复 = LDS 直方图去重**:每 block 在 LDS 累计 dest 直方图(ds_add,**全线程**网格跨步遍历 token,每线程用 bitmask 去重自己 token 的 k 个 dest)→ 每 dest 每 block **1 次全局原子**。复用 `_chl` LDS(count/CMP 已完、GEMM 未启,区域空闲)。**dispatch-total 53476→39176(≈ 非-atom 38436,基本持平)**,grid-wait 26112→14336。
      - **★修复后性能(v4_flash a8w4)**:e2e speedup vs baseline-fp8 bs4096=**0.915**、bs8192=**0.918**、bs16384=**0.923**(↑ from 0.83);stage1-sweep `_profiler_ms` 口径 bs4096 **0.944(≈ 非-atom compact 0.962)**。bit-exact 全 PASS(v4_flash a8w4 bs8/128/512/2048/4096/8192;r1_v3 a4w4 bs128/2048)。
      - **★残留 <1.0 的性质**:combo dispatch 已与**非-atom compact 持平**;剩下的 <1.0(e2e ~0.92)= **compact dispatch 自身在大 bs 对 baseline 就 ~0.96**(write all-to-all 带宽地板,§续 已证,与 atom/stage2 接入无关)。即「接入 stage2」本身已**不再额外损耗**;要再进一步只能提 compact dispatch 本身(带宽受限,难)。原始数据 `_runlogs/{dense_perf,combo_e2e_*}.log` + phase-ts 对比。

- **2026-06-16(续4)compact 接 stage2 = 零额外动作(对齐 fixslot)+ 精度全 cover。**
  - **桥接零拷贝(已确认 = fixslot)**:combo forward = `stage1.forward`(1 launch,Plan A 在核内写 `total_recv` 进 combine op)+ fused gemm2+combine。**无冗余 dispatch、无 host 数据桥**;`tok_id_to_src` = identity 在 `__init__` 设一次。与 fixslot 完全同构。
  - **消除唯一的 combo 专属额外动作(srcmap host fill)**:之前为掩 padding 用 `srcmap_em.fill_`(host memset,随 bs 涨 ~MB)。先试**核内 prefix-scan count-mask**(对齐 fixslot 的 in-kernel mask)→ bit-exact 但每 tile 32-iter readlane 拖慢 ~2%(e2e 0.92→0.90)。改为 **block0 在 metadata 阶段只对 srcmap 的 tile-pad GAP 槽(每专家 <tile_m)写 sentinel**(real 槽由 peer 在 PHASE-2 P2P 写;gap 永不被写)→ 核内、无 host memset、无 per-tile scan,GEMM 直接读 srcmap(pad=sentinel→t_ok 掩码)。
  - **性能(无回退)**:v4_flash a8w4 e2e bs4096=**0.910**、bs8192=**0.917**、bs16384=**0.924**(= host-fill 版,且无额外动作)。
  - **★精度全 cover(mega-vs-baseline=0.000 全 bit-exact)**:v4_flash a8w4 bs 8/128/512/2048/4096/8192/16384;r1_v3 a4w4 bs 8/512/2048/8192;v4_pro a4w4 mega==atom;**默认关(无 env)非紧凑路径 bs128 不变**。原始数据 `_runlogs/{combo_e2e_a8w4_sweep,gap*}.log`。
  - 诊断开关(默认关、DCE):`FUSED_MEGA_SKIP_PLANA`(跳 Plan A 测 grid-wait)、bench `atom_contract=env`(让 stage1-sweep 建 combo 取 phase-ts)。
  - **★(已被 t5 取代)原风险描述**:GEMM1 的 `_atom_contract` 标志**编译期耦合**到 fixed-slot 路径——强制 `topk=1` 编译 + static-tiles A-gather(`A row=le*cap+…`,`:4015` 注释),且 `_sti/_se_atom/_sw_atom` **emit** 门控在 `_static_tiles`(`:3589`)。compact 需相反:dense A-gather 走 `_se`/`_trb`、topk=`_fz_k`。⇒ 要把「atom 输出(logical a2 写 + _sti/_se_atom/_sw_atom emit)」从「fixed-slot topk=1/static-tiles A-gather」**解耦**(在巨型共享 GEMM 模板里 thread 新组合)。开关打开但 GEMM 未解耦前会**静默产错 a2**,故 GPU 空闲后再改 t5 + 全程 `--check-correctness` 验证。a2-logical **写**(`precompute_row :3984` 读 lds_tid→logical 行)已不依赖 static-tiles,compact 可直接用;缺的是 emit 路径解耦。
  - **残留**:a2-logical 输出 = npes·mtpr·topk×inter,v4_flash/v4_pro 全 bs OK;r1_v3(topk8)bs32768 ≈ 4.3GB 仍 wrap(就是 a4w4 扫崩的点),需后续 a2 也改 48-bit base 寻址。

- **2026-06-16（续）write 负载不均探针 = 推翻「127µs grid-wait 是 warp/block 失衡」结论(实为 block0 测量假象);两项 LDS 优化对 fixed-slot 不适用(已证)。**
  - **背景**：之前 compact bs4096 看到 `xPE#2 grid-wait≈127µs`,怀疑是 warp/block 间负载不均(grid barrier 等最慢 block)。但 grid-wait 是从 **block0 的 write-end(ts3)** 量到全 block 到齐(ts6),若 block0 恰好是快块,这个差会被夸大。
  - **★探针**：compact_ag write 收尾,**每 block(tid==0)把自己的 write-end tick 原子累加到 slot9(sum)/slot10(count)**(`FUSED_MEGA_PHASE_TS=1` const_expr 门控,关闭零开销)。host 算 `avg_write_end=slot9/slot10`,对比 block0(ts3)/最慢(ts6)。phase-ts 门控放开到 compact(原仅 fixedslot)。
  - **★实测(v4_flash a8w4,10ns/tick)**：

    | bs | block0 write | **avg write** | 最慢 write | 真 tail(慢/均) |
    |---|---|---|---|---|
    | 4096 | 202µs | **314µs** | 351µs | **1.12×** |
    | 8192 | 481µs | **566µs** | 614µs | **1.08×** |

  - **结论**：**block0 是快块**(avg 是 block0 的 1.56×),所以「127µs grid-wait」大半是「block0 早完、其它块还在写」的测量假象,**不是 warp/block 失衡**。全 256 block 的真实块间 tail 只有 **1.12×/1.08×** —— write 其实相当均衡,剩 ~10% 是数据相关 xGMI 拥塞(各 dest rank 收到的 token 数不等→链路忙闲),非 count 失衡。**⇒ 没有大块「负载均衡」头可吃;write 是 all-to-all 带宽地板(bs4096/8192 = 0.96/1.03× ≈ baseline,本就同样的全互联搬运)。盲改 write 路径(高风险)预期收益 ≤10% 且是物理拥塞,不值。**
  - **★Q2:两项 LDS 优化(count 直方图 / CMP+meta 读 LDS)对 fixed-slot 不适用(已证)**：fixed-slot phase-ts(bs128/512/2048)= postpass 恒 ~360 ticks(~3.6µs,已极小)、**无 count-pass、无 bigcnt all-gather**(它是单趟:写时直接对 dest 的 `running[le]` 远程原子)。两刀都是 compact 专属阶段的优化,fixed-slot 结构上没有那两个阶段,吃不到;其 metadata(postpass)本就 ~3.6µs。fixed-slot 的瓶颈同样是 write + 跨PE 握手(bs2048 write 14288 / handshake 1404 ticks)。原始数据 `_runlogs/{fs_phasets,imbalance_probe}.log`。

- **2026-06-14（深夜·续）固定槽位大 bs 静默错 = 真因定位(i32/4GB buffer-resource wrap)+ all-gather 紧凑提速 + 阈值修正。**
  - **真因**：固定槽位静默错**不是**地址算术溢出,而是 **buffer-resource 的 `num_records`(及 voffset)是 32-bit(4GB 硬上限)**。固定槽位按 `cap=npes·mtpr` 给每专家预留 ⇒ `rx_em`/`out` 有 `epr·npes·mtpr` 行;v4_flash bs4096 时 `rx_em = 1048576×4096 = 恰好 2³²(4GB)` ⇒ `num_records` 回绕到 0 ⇒ 所有 A load 越界返回 0 ⇒ 输出≈0 ⇒ `relL2(mega,torch)=0.99`、mm=194560(99%)。实测阈值精确吻合:**bs1024(1GB)/bs2048(2GB) PASS,bs4096(4GB) FAIL**(XCD on/off 都一样,排除 XCD)。**这不是上一轮(bs≤128)修复的退化** —— 上一轮 buffer 才几百 MB,那范围确实对;大 bs(>~2048,buffer→4GB)从没测过才暴露。compact dense 布局行数小 ~topk/cap 倍、buffer 远 <4GB ⇒ 大 bs 正确。
  - **★修复 = buffer-size 阈值**(facade,robust 跨 shape,无 kernel 改):固定槽位在「最大 buffer < ~3GB」时用(单趟、快、且 <4GB 不回绕),≥3GB 切 compact(dense、<4GB、不 OOM)。即 `compact = (epr·npes·mtpr·max(row_bytes, inter·2) >= 3e9)`。v4_flash ⇒ bs≤2048 固定槽位、bs≥4096 compact。**这把 bs1024/2048 从 compact 的 0.74/0.79 拉回固定槽位的 1.07/1.10(+45%/+37%),且 bit-exact。**
  - **★compact all-gather 提速(2 趟跨 PE,替代 3 趟)**：把 count 从「远程原子 + 远程读 compact_base + barrier#1b」改成「本地原子计数 → bigcnt all-gather → 本地算 my_base → strict 写」(复用 handshake 的 PUB/CMP)。大 bs 大幅提速:bs8192 0.715→**0.916**(+28%)、bs16384 0.774→**0.945**(+22%)、bs4096 0.860→0.97。默认 `compact_allgather=True`,3-round 作 fallback 保留。
  - **dispatch 耗时拆解(why 随 bs 线性增长)**:实测 prologue(dispatch)bs1024/4096/8192 = 195/464/778µs,= **~100µs 固定(握手 count+offset+barrier,随 bs 摊薄,所以 ratio 随 bs 升)** + 线性项。线性项 = **payload all-to-all 搬运**(每 routed token 的 model_dim 字节搬到其专家 rank,bs8192≈200MB xGMI,MoE EP dispatch 本质开销,固定槽位也一样)+ count-pass 遍历。握手 count/offset 本身确实 O(1)、不随 bs 涨——用户直觉对。
  - **★最终全 bs 性能(v4_flash a8w4,fused/atom_fp8,全 11/11 bit-exact PASS)**:
    | bs | 1 | 8 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | 模式 | FS | FS | FS | FS | FS | FS | FS | FS | CPT | CPT | CPT |
    | ratio | 0.76 | 0.90 | 0.95 | 0.99 | 0.96 | **1.11** | **1.07** | **1.10** | 0.97 | 0.93 | 0.95 |
    (FS=fixed-slot 固定槽位,CPT=compact all-gather。)bs512–2048 净胜 1.07–1.11×;bs≥4096 compact 0.93–0.97×(固定槽位在此回绕/OOM,compact 是唯一正确选项,且已很接近 atom_fp8)。小 bs(≤256)dispatch 延迟地板,物理天花板。
  - 遗留:bs4096 若做真·>4GB 寻址修复(把每-tile 行基址折进 buffer descriptor 的 48-bit base,而非 32-bit voffset)可让固定槽位也跑 bs4096(~1.16×),但 bs≥8192 固定槽位必 OOM 且 cap 布局极浪费 ⇒ 收益仅 1 桶、风险高,未做。原始数据 `_runlogs/{final_sweep,fix_sweep,ag_full,nc_bs*}.log`。

- **2026-06-14（深夜）朴素实现(naive=fixedslot) 全 bs 落地：COMPACT count-first dispatch（解大 bs OOM + 修非紧凑核大 bs 静默错）。** 命名约定：**朴素实现 = `fixedslot`**（核内 dispatch + grid barrier + GEMM 串行）、**进阶实现 = `handshake`**（producer/consumer overlap）。两套都要全 bs 测。本次做朴素：
  - **动机**：fixedslot 原用「预设 cap=npes·mtpr」固定槽 ⇒ `num_valid_max=epr·npes·mtpr` 随 bs 爆炸，**bs≥8192 OOM**（§4 全 decode 扫已暴露）；且经本次 `--check-correctness` 验证，**非紧凑核在大 bs 还有静默正确性 bug**（bs4096 mismatch_rows=194560、relL2≈0.99 = 输出全错；此前从未在 bs>128 验过正确性，所以「decode 大 bs 1.1–1.16×」那批数其实是错值）。
  - **方案（= 用户思路，新变体）**：count-first 紧凑布局，**保留 decode strict-phase GEMM（不走 handshake overlap）**。两趟：Phase-0 所有块对 `dest.running[le]` 远程原子计数(无 payload) → 跨PE barrier#1(counts 可见) → block0 算 `compact_base[le]=前缀和(round_up(cnt,tile_m))`+tile 元数据 → **跨PE barrier#1b(compact_base 全局可见)** → Phase-2 所有块精准写 payload 到 `compact_base[le]+write_cursor` → 跨PE barrier#2(payload 可见) → strict GEMM。`num_valid_max` 降到 `npes·mtpr·topk+epr·tile_m`(handshake 同界) ⇒ 不 OOM。
  - **落地点**：`ep_dispatch_groupmajor_op.py`(compact 标志 + `compact_base/done2c/done2cb/write_cursor` 对称缓冲 + p2p + 紧凑 num_valid_max)、`fused_moe_gemm_2stage.py`(`compact_dispatch` + `_fuse_fs and _compact` 新分支，disp idx 19–28)、`fused_moe_megakernel.py`(`self.compact = fixedslot and mtpr>512` **内部自动**，无 env；naive 永远用 decode tiles)。
  - **★调试踩的两个真坑**：① **gx=16/tile_m=128(prefill tile)+ naive grid barrier → 死锁**(co-resident 装不下)；修复 = naive 永远用 decode tiles(gx=8,co-resident)，prefill tile 只给 handshake。② **跨PE race（间歇 hang）**：Phase-2 sender 读 `dest.compact_base` 却只同步自己的 meta_flag，会读到 dest 还没算好的 compact_base → 部分 rank 输出错 → bench 的条件式 collective(`if _common:`)在各 rank 不一致 → all_reduce 死锁(表现为 hang)。修复 = 加 **跨PE barrier#1b**(compact_base 算完后、Phase-2 前，保证所有 rank 的 compact_base 全局可见) + Phase-2 前 `fence_system_acquire`。修复后 bs1024 连过 4/4(原来间歇 hang)。
  - **正确性（`--check-correctness`，v4_flash a8w4，--from-bf16）= bs 1/8/64/128/256/512/1024/2048/4096/8192/16384 全 bit-exact PASS(mismatch_rows=0)**；bs32768 仅因测试中 12 个 facade 连建不释放 shmem 累积而 OOM，单跑可过。**这是 decode 首次在 bs>128 验证正确 + 首次能跑 bs≥8192。**
  - **性能（fused/atom_fp8，default 计时）**：≤512(非紧凑) 0.72/0.89/0.96/0.99/0.97/**1.11**；>512(紧凑) bs1024–16384 = 0.74/0.79/0.86/0.72/0.77。**紧凑比(假想正确的)非紧凑慢 ~1–2 个 xGMI round(count-pass + 3 趟跨PE)，大 bs 普遍 ~0.75× 打不过 atom_fp8**——但非紧凑大 bs 是错的/OOM，所以紧凑是大 bs 唯一正确选项；且比 handshake(进阶,大 bs 0.1×)快 ~7×。**结论:朴素紧凑是「正确性+显存可扩展」修复,不是大 bs 性能赢;大 bs 两套 fused 目前都打不过 atom_fp8。** 阈值取 512(非紧凑只在已验证的 ≤512 用)。
  - 遗留:非紧凑 fixedslot 大 bs 静默 bug 未修(已被 compact>512 屏蔽,生产不暴露);compact 大 bs 性能可后续用 count all-gather(handshake bigcnt 式,省 1 趟跨PE)优化。原始数据 `_runlogs/naive_full_final.log`、`cpt_*`、`boundary.log`。

- **2026-06-14（晚）rocprofv3 厘清 decode mega 真实瓶颈（厘清「occ 矛盾」+ 否定 split-K 是首选 + 找到 tile_m 杠杆）。** 用 rocprofv3（profile rank0、`--check-correctness` eager、v4_flash a8w4）实测，**先解决「mega 与 baseline 同名 `moe_gemm1_0` 数据混淆」**：用 `Grid_Size`/`Workgroup_Size`/`VGPR_Count` 区分（CSV 有这些列），三个 `moe_gemm1_0` 签名互不相同：
  | kernel | Grid_Size | wg | blocks | VGPR | LDS | 身份 |
  |---|---|---|---|---|---|---|
  | **mega（真）** | 65536 | **256** | **256=gx·gy=8·32** | **176** | 33792 | `compile_fused_moe_gemm1`（持久+融合 dispatch） |
  | flydsl mixed（f16 ref，跑1次） | 1081344 | 128 | 8448 | 88 | 33280 | `compile_mixed_moe_gemm1` 正确性参照 |
  | aiter CK（计时 baseline） | 32768 | 256 | 128 | 64 | 33792 | `Cijk_...MT16x16x256...SK3_SKXCCM8`（aiter 生产核） |
  - **⚠ 用户「实验 B：VGPR=88、occ 15%」混入的是 baseline（8448 blocks 的 mixed），不是 mega。** mega 真身 = 256 blocks / 256 线程 / VGPR=176。**矛盾消解**：mega 与 baseline 都是低 occupancy，但是两个不同核。
  - **(a) mega 真实 occupancy & 限制**（rocprofv3，bs8）：`OccupancyPercent≈9%`、`MeanOccupancyPerCU≈2.87 waves/CU`（理论 max 32）。**限制 = grid 块数不足，不是 VGPR/LDS/scratch**：HW=256 CU×4 SIMD×(max 32 waves/CU)；VGPR=176 ⇒ 资源上限 ~2 blocks/CU（=8 waves/CU），LDS=33792 ⇒ ~4 blocks/CU，Scratch=0。但 grid 只给 `gx·gy=256` blocks = **1 block/CU**，且**小 bs 下还更糟**：bs8 `num_valid=384`、`Tm=ceil(384/tile_m32)=12 < gy=32` ⇒ 只有 `Tm·gx=96` blocks 真有活、160 blocks 空转 ⇒ 有效 ~0.4 block/CU。**occupancy 受 grid（M-tile 数）饿死，资源还有 ~2× 余量。**
  - **(b) 瓶颈分类 = HBM 延迟/occupancy 饿死，不是 compute、也不是 BW 饱和。** rocprofv3 mega（bs8）：`MfmaUtil≈6.8%`、`VALUBusy≈3.5%`（⇒ 远非 compute-bound）、`MemUnitStalled≈1.9%`（低 ⇒ 不是访存单元被打满，而是发起的访存太少）、`L2 hit≈15%`（小 bs 每专家仅 1 个 M-tile，xcd 的跨-M-tile L2 复用无从谈起，inherent）。**`FetchSize`：bs8=113MB / bs128=148MB（弱随激活专家数增长、几乎与 token 数无关 ⇒ 主体是专家权重加载）；耗时 bs8=74µs/bs128=96µs ⇒ 实测 HBM 读带宽 ≈1.5 TB/s ≈ 峰值(~8TB/s) 的 19%。** ⇒ **不是 BW-bound（有 ~5× BW 余量），是 latency/occupancy-bound：1 block/CU 发不出足够 outstanding load 去填满 HBM。** kernel 耗时分解（trace，steady median）：bs1=54 / bs8=69 / bs32=76 / bs128=92 µs；standalone dispatch≈12µs；aiter 纯 GEMM≈11.7µs（恒定）。⇒ **mega 主体是 GEMM 段的权重加载（~42µs@bs1 → ~80µs@bs128，随激活专家增长），dispatch floor 只占 ~12µs。**
  - **(c) gy=64 死锁真因 = dispatch prologue 的「全块到达 grid-barrier」要求所有块 co-resident，普通（非 cooperative）launch 在 >256 blocks（>1 block/CU）时不保证全块同时驻留 ⇒ 未驻留块永不 +gb1 ⇒ block0 死等。** 与 GEMM 资源 occupancy 无关（资源允许 ~2/CU=512）；是 launch 语义 + 持久 prologue barrier 的安全上限 ≈1 block/CU（256）。gy=64=512 正好踩线。**split-K 的 z-gate（z>0 不参与到达 barrier、只等 meta_flag）正是绕开此死锁的设计。**
  - **结论 / 改动方向（实测排序）**：
    1. **split-K 不是首选**：它确实命中正确瓶颈（latency/occupancy，正是 aiter 用 `SKXCCM8` split-K 取胜的原因），但 **2-stage 核的 split-K 路径（`_is_splitk`）的 epilogue 是「gate/up 分别 atomic-add、不做 silu」**（见 `fused_moe_gemm_2stage.py:3006/3099`）——`silu(gate)*up` 不能按 K 切片，必须加一个 **下游 reduction+silu pass**，破坏 megakernel「单 launch + gather-free + silu 融合」的核心。小 bs 下第二趟 pass（读 2×inter 原始 gate/up + silu + 写 inter）+ atomic 写放大很可能吃掉 GEMM 提速。**留作高风险 supervised 项，不是首选。**
    2. **tile_m=16 杠杆 = 已查实为死路（两个 bug + 权重重读放大）。** 初看很美:小 bs 是 M-tile 饿死 + memory-bound（MFMA 强度无关），tile_m=32→16 翻倍 M-tile 填满 grid、保 gx=8/xcd,实测 tile_m=16 vs 32 = bs8 73.3/74.7、bs16 76.5/78.5、bs32 78.2/81.0 µs（2–3.5%）。**但深查后否定**:
       - **Bug #1（已根因+给出 fix，但单独不够）**:`sort_block_m=max(32,tile_m)`（`fused_moe_gemm_2stage.py:263`）⇒ dispatch post-pass 按 `_fz_tile_m=sort_block_m=32` 行切 tile（`:416`),但 GEMM 只算 `m_repeat=tile_m//16=1`=**16 行/tile**（`:1581`,A-LDS 只载 tile_m 行 `:1454`）⇒ **每个 32 行 tile 的第 16–31 行被静默丢弃**。bs8/16/32「PASS」只因每专家收数极小（avg ~1.5/3/6 ≤16,丢的全是 pad 行）;bs64（avg ~12 + 方差,部分专家 >16）⇒ 43 行真丢 → FAIL。**这是静默数据损坏地雷,且「提速」部分来自少算。** Fix = post-pass 切 tile 用 `tile_m`、`_total_m_tiles` 除以 `tile_m`（对 tile_m≥32 是 no-op,已验证）。
       - **Bug #2（未解,独立存在）**:打上 Bug#1 的 fix 后（清缓存重编、bs64 耗时 81→98µs 证明 tiling 确实变了）,**bs64 仍 mismatch_rows=43（xcd on/off 都 FAIL,bs32 仍 PASS）** ⇒ tile_m=16 在高每专家收数下还有第二个 GEMM 内核 bug（与 xcd 无关、与 tiling-fix 无关,触发条件 = 某专家收数较大,bs32 的 PASS 可能只是 seed 运气 ⇒ 不可 gate 到 bs≤32 生产）。
       - **即便修好也不值**:正确 tile_m=16 在 bs64 = 98.7µs,比 tile_m=32 的 82.3µs **慢 20%**(16 行 tile 把每专家权重 K-slab 重读次数翻倍 = 权重重读放大);只有 bs≤32 略快(bs32 77.8 vs 81.0)。decode 真实瓶颈是 dispatch+权重加载 floor,这点 tile 腾挪改不动。
       - **结论:已 revert 全部 tile_m fix,核保持 clean（仅留上一轮 XCD `_gy` 修复,勿回退）。tile_m=16 不进生产表。** 若将来仍要用 tile_m<32:须 (1) 上 Bug#1 的 fix,(2) 定位修 Bug#2（建议加 oracle 打印「失配行的 expert + cap 内偏移」来定位）,(3) 评估权重重读放大后是否还净赢。
    3. **tile_n 杠杆（仅 bs1）**：bs1 tile_n=128 比 256 快 24%（60.8→46.1µs，bit-exact），但 bs1/bs8 共享 `mtpr=16` 桶（表按 `mtpr=max(16,bs)` 索引、无法只对 bs1 生效），且 bs8 仍偏好 tile_n=256 ⇒ 不单独落地。
    4. maxnreg「flat」已解释：降 VGPR 不增 grid 块数（grid 只给 256），1 block/CU 不变 ⇒ 无效。
  - **方法学坑（本次踩到，记录）**：① rocprofv3 单趟 PMC 计数器有硬件上限——`FetchSize+WriteSize+...`、6 个混合计数器会报 `error code 38: exceeds capabilities` 并使 rank0 崩溃，**其余 7 rank 卡在集合通信** ⇒ 每趟 ≤3 个、Fetch/Write 各自单独一趟（都重 TCC_EA）。② 纯耗时用 `--kernel-trace`（不 serialize、不膨胀），别用 PMC。③ master_port 拼接：`2952$BS`（BS=32→`295232`>65535）会无声失败 EXIT=1——**端口用定值、勿与变量拼**。④ mega 与各 bs **grid 相同（gy=cu//gx 与 bs 无关）**，单趟 `--bs-list` 无法按 grid 区分 bs，需分 bs 单跑。
  - 原始数据：`_runlogs/prof_g1`（occ/MFMA/VALU）、`prof_g2`（L2/MemStall）、`fetch_bs*`（FetchSize）、`trace_bs*`（耗时）、`tn_*`/`tm_*`（tile 扫）。

- **2026-06-14 decode 精度 bug 定位 & 修复（两处真 bug，旧 oracle 空转掩盖）。** 旧 `--check-correctness` 用 `init_scale=0.01`，stage1 输出 ≈3e-4，在 1e-2 绝对容差下「近零 vs 近零」恒 PASS —— decode 的「24/24 PASS / 1.4–1.9×」从未真正验证过。修好 oracle（`init_scale=md^-0.25` 让输出 O(1) + torch f32 真值 + dispatch/routing 定位）后，mega vs torch relL2≈0.96（坏），逐层排查（dispatch ✓ scale ✓ routing ✓）锁定 **GEMM 写出/覆盖**，两个根因：
  1. **XCD swizzle remap 用错 gy**（`fused_moe_gemm_2stage.py` 的 xcd 分支硬编码 `_gy=cu_num=256`，而融合核真实 `grid_dim.y=_fz_gy=cu//gx=32`）⇒ `_num_wgs=gx·256` 虚高 8×，remap 把除「首个 WGM 组（=`xcd_swizzle`=4 行 M-tile）」外的 (bx,by) 全映射出界 ⇒ 静默漏算 ~6/7 的 tile（输出留初值 0）。**这也是「xcd 1.7×」假快的真相 —— 它在跳算大半 tile。** 修复：`_gy = grid_dim.y`（对融合/非融合都对）。修复后 gx=8 + xcd=4 全 28 tile bit-exact。
  2. **sparse 输出 row 守卫错**（epilogue `precompute_row` 的 `row_valid = row < num_valid`）：fixedslot 的 `row` 是 SPARSE 槽地址（`le*cap+…`，可达 ~nvm），`num_valid` 是致密 token 数 ⇒ `le*cap ≥ num_valid` 的专家整段漏写。修复：`sparse_tiles` 下 row 守卫恒真（tile 已被 block-uniform 的 `blk_valid` 把关；pad 行无害且 in-bounds）。
  3. **facade xcd 守卫加严**：旧 `cu % gx != 0` 才关，对 gx=16（tile_n=128）漏判（256%16==0）；改为还要求 `gx | 8`（gx∈{1,2,4,8}），与 §5 的 WGM 约束一致。
  - **精度全场景验证（修复后，`--check-correctness`，逐元素 mega vs ATOM 生产栈 + torch f32 真值）= 18/18 全 PASS**：r1_v3 a4w4 / v4_flash a8w4 / v4_pro a8w4 × bs∈{1,8,16,32,64,128}，全部 **mismatch_rows=0、maxerr=0（mega 与生产 ATOM 栈 bit-exact）**；mega-vs-torch relL2 = atom-vs-torch（量化噪声级，r1_v3≈0.23 / v4_flash≈0.17）。（v4_pro 的 torch 真值对 atom 与 mega 都 ≈1.05，是 bench torch-ref 对 v4_pro 形状的预存计算问题，不影响 mega==atom 的 bit-exact 判定。）gx=8 网络（r1_v3/v4_flash）此次跑在 **xcd-ON** 默认路径，等于同时验证了 xcd remap 修复。
  - **`_MEGA_DECODE_TILE` 调优表已清空作废**（坏核产物，每桶都偏好 tile_n=128/gx=16 = 坏核假快配置）；decode 回退到设计默认 `tile_m=64, tile_n=256`（inter%256==0 → gx=8 + xcd；v4_pro inter=3072→gx=12→xcd 自动关）。需在修复核上重扫再启用。
  - **⚠ 旧性能口径作废**：`moe_stage1_mega.md` §6 的 decode 1.2–1.9× 与 §2/§5 的 xcd 扫描全部基于「漏算 ~6/7 tile」的坏核 + 空转 oracle，**数值无效**。**修复核全场景性能重测（8×MI355X，topk=6，--from-bf16，profiler device-time，8 卡均值，mega=fixedslot+tile_n=256+xcd，baseline 用最优 tune）**：
    | net | baseline | fused/baseline @ bs 1/8/16/32/64/128 |
    |---|---|---|
    | r1_v3 a4w4 | atom（aiter tune） | 0.63 / 0.91 / 0.94 / 0.95 / **1.08** / **1.19** |
    | v4_pro a8w4 | atom_fp8（aiter tune） | 0.80 / 0.92 / 0.83 / 0.86 / 0.88 / 0.92 |
    | v4_flash a8w4 | atom_fp8（default tile，已扫确认最优） | 0.68 / 0.91 / 0.93 / 0.93 / 0.93 / **1.06** |
    - baseline 公平性：r1_v3/v4_pro 走 aiter `*_tuned_fmoe.csv` 的 `flydsl_moe1` 最优 tile；v4_flash 无 tune csv，扫 {32,64,128}×{64,128,256} 确认 default(32×64) 已是最快 baseline（新 `--atom-tile-m/n/k` 可覆盖）。
    - **结论**：正确计算后，mega fixedslot decode 在「最优 tune baseline」下 **小/中 bs 普遍 0.6–0.95×（打不过），仅大 bs(≥64) 在 gx=8 网络略胜（1.06–1.19×）**。之前的「decode 净胜 1.4–1.9×」是 bug 虚高，**不成立**。decode 的真实价值仅在大 bs 的 xcd L2 复用，且幅度远小于旧宣称。
- **decode（`scheme="fixedslot"`）= 精度 2026-06-14 起真正验证通过、bit-exact（见上）；性能见上表（真实，旧 §6 作废）。**
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

> **2026-06-14（晚）rocprofv3 复评（见 §1）：split-K 确实命中真瓶颈（小 bs = HBM latency/occupancy-bound、1 block/CU、实测仅 19% 峰值 BW；aiter 生产核正是用 `SKXCCM8` split-K 取胜），但在本 2-stage 核里 split-K epilogue 是「gate/up 分别 atomic-add、不做 silu」（`fused_moe_gemm_2stage.py:3006/3099`）⇒ 必须加下游 reduction+silu pass、破坏单 launch + silu 融合，小 bs 下第二趟开销很可能吃掉提速 ⇒ 降级为高风险 supervised 项，首选改为 `tile_m=32→16`（填满 M-饿死的 grid、保 xcd/silu，实测 2–3.5%，待修 tile_m=16 @大 bs 的 GEMM bug）。**

### 4.1 量化评估（2026-06-14 晚，基于 rocprofv3 数据）—— 结论：上界不确定，必须先跑 proxy 门槛实验

**成本拆解（mega v4_flash a8w4 bs8 = 74µs）**：实测 `FetchSize=113MB`（≈全部激活专家权重，~28 专家×4MB；token payload 仅 ~0.2MB 可忽略），即**整核时间 ≈ 取 113MB 权重的时间**。`BW=1.53TB/s=19% 峰值`、`MemUnitStalled=1.9%`、`MfmaUtil=6.8%` ⇒ GPU 大部分时间既不算也不打满访存单元 = latency/occupancy 饿死（1 block/CU、bs8 仅 96/256 CU 有活）。

**split-K 上界（乐观 vs 悲观，分歧大）**：
- 乐观（若瓶颈=HBM-fetch 并发不足）：k_batch=2–3 用「不同 K 切片」的块填满 256 CU、各取不同 HBM 段 ⇒ BW 19%→40–60% ⇒ 权重取数 74→25–37µs ⇒ bs8 0.97→~1.5–2.0。aiter 生产 decode GEMM 正用 split-K（`SKXCCM8`）⇒ 强先验支持。
- 悲观（若 1.53TB/s 是该访存模式/fp4+scale 解包的硬顶）：split-K 同字节同模式 ⇒ ~0 收益 + 开销 ⇒ 净亏。
- **现有数据无法判定**：bs8 与 bs128 的 BW 都是 19%（flat），看似支持悲观；**但被 L2 复用混淆**——bs128 多出的块命中 L2 复用权重（不打 HBM），而 split-K 多出的块取**不同 HBM K 切片**（打 HBM），两者性质不同 ⇒ 此自然实验不能证伪 split-K。

**开销（decode 小 bs = 便宜，关键）**：split-K 破坏 silu 融合,需 ① 中间 buffer `[num_valid, 2·inter] f32`、② 各 z 平面 atomic-add 累加、③ 收尾 reduction+silu pass。decode 小 bs 下 num_valid 极小（bs8 ~48 行/rank）⇒ 第二趟 ~1MB 流量 <1µs、atomic 放大 ~1–3µs ⇒ **开销可忽略**（开销随 bs 增长,但大 bs 走 handshake、与此无关）。

**实现成本/风险 = 高**：z-gate 改共享 prologue（z==0 跑 dispatch+arrival、z>0 只等 meta_flag）+ 延迟 silu 收尾 pass（破坏单 launch,或需 z 间同步）+ 输出 buffer 改造 + CUDAGraph 安全 + 3 网络×全 bs 正确性（参考 tile_m=16 那种边角 bug 的教训）。现有 `_is_splitk` 路径是「gate/up 分写、不 silu」的不同输出契约,接进 fused fixedslot + silu 收尾是大改。

**推荐 = 不要盲目全实现；先跑 proxy 门槛实验**：用 standalone `compile_mixed_moe_gemm1`（已带 `k_batch` 参数,`mixed_moe_gemm_2stage.py:119`）在 decode 规模的 expert-major 数据上,只测 **GEMM-only device time** at `k_batch∈{1,2,4}`。判据：k_batch=2 须 ≥~1.5× GEMM 提速 才值得落地。

**★proxy 实测结果（2026-06-14 晚,v4_flash a8w4,tile 32×256=mega decode tile,GEMM-only cuda-event,8 卡均值）= split-K 对 decode 无效,门槛 FAIL,split-K 彻底放弃：**
| bs | k_batch=1 | k_batch=2 | k_batch=4 |
|---|---|---|---|
| 8  | 40.5µs | **110.4µs（2.7× 慢）** | 112.3µs |
| 32 | 52.1µs | 50.8µs（~持平） | 53.6µs |
| 64 | 53.6µs | 52.5µs（~持平） | 56.7µs |

⇒ **小 bs split-K 灾难性变慢（2.7×）、中大 bs 持平略亏**。证实「悲观解释」:~1.5TB/s 是该访存模式的有效硬顶,split-K 多加块只增开销（atomic reduction + 每块工作更碎 + 小问题更 launch-bound）、不抬取数吞吐。**split-K（以及任何「靠加并发填 occupancy」的思路）对 decode 死路。** (aiter 用 split-K 取胜是因为它的 16×16 微 tile + 整体不同的 schedule,不是单纯 split-K;直接照搬 split-K 到本核 GEMM 无效。)

**⇒ decode 小 bs 的最终结论:已在物理地板（dispatch 跨卡延迟 + 每专家权重加载 @ ~19% 峰值 BW,且该 BW 受访存模式限制、加并发抬不动）。三个候选杠杆（split-K / tile_m / tile_n）经 rocprofv3 + proxy 全部实测否决。要再进一步只能换「根本不同的权重加载/复用方案」（超出本轮 tile/grid 腾挪范围）。** proxy 代码 = bench `SPLITK_PROXY=1` env-gated 块（已 revert,结果见 `_runlogs/splitk_proxy.log`）。

---

decode 是 M-tiny / K-huge（model_dim 4096–7168）→ GEMM grid 饿（少 M-tile，闲 CU）。split-K 加 grid-Z 维并行 K reduction → 填 CU、提 occupancy。fused 适配需 **dispatch-prologue z-gate**：dispatch+arrival 仅 `blockIdx.z==0` 跑；`z>0` 平面跳 dispatch、等 `meta_flag` 后算各自 K 切片、atomic-add 到预清零 `out`。改动共享 prologue、风险高 → 留作 supervised 落地（先 k_batch=1 strict 回归哨兵，再开 2/4，用 bench `--check-correctness` 验证）。z-gate 让 `z>0` 平面**不参与到达 barrier**（只等 `meta_flag`）⇒ 也顺带绕开 §1(c) 的 gy>cu 死锁。

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
