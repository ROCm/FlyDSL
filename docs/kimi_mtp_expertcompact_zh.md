# Kimi MTP3：双阶段 expert compact 实现与验证

2026-09-28。接续 [计算数据流与 HBM 审计](kimi_mtp_dataflow_zh.md)，实际实现 UG 和 down 共同按 expert 分组，并验证资源、正确性、HBM/L2 请求、ATT 和无插桩完整层性能。

**权重复用生效，但本轮没有得到完整层加速。** 最快 compact 候选的三 seed HBM 读请求降低 **17.4–18.2%**；完整层中位数为 **145.531 μs**，原 pipeline 为 **143.528 μs**，staged 为 **134.488 μs**。九个候选均保留为独立实验源码，未替换工作区 kernel。

## 实现改变了哪些数据流

输入保持一个请求的 true-MTP3、S4、TP8、staged KDA、native MXFP4 routed experts。融合范围仍为 routed UG/SiTU/down/combine/TP，router 与 shared/latent tail 仍独立。

第一版 `compact` 复用 router 已有的 `sorted_expert_ids`、`sorted_token_ids`、`num_valid_ids`，构造 CTA 内的 expert、sample、route 和系数映射。UG 的同一权重片段供四个 sample 列使用；每个活跃 producer CTA 仅 stage 一次四份 BF16 activation，并跨 task 复用。down 也遍历独立 expert，同一权重片段服务其有效 sample 列，末轮不再预取未消费的专家。

SiTU 的 BF16 舍入边界、携带 epoch 的 intermediate payload、确定的输出 tile 所有权、routed TP/tail 和 true-MTP 状态语义均保留。消费端使用与正确 tag 一起读到的 payload；没有把独立 ready flag 当作读取旧数据的许可。

后续最快版本 `compact_vecmid_loopug` 增加了两项改变：UG 使用有界运行时循环控制预取活跃范围；down 由每 wave 串行处理四个 sample，改成每个 sample 分配 16 lanes，并行轮询后以四个 packed BF16 pair 为单位写入 LDS。仍使用原 tagged mid ABI，未改变中间精度。

完整源码与补丁在 [实验目录](../../results/kimi_mtp_expertcompact_20260928/)。最快版本见 [routed_pipeline.py](../../results/kimi_mtp_expertcompact_20260928/node50/compact_vecmid_loopug/kernels/kimi_k3_monokernel/routed_pipeline.py)，其 staged 调用适配也在同一源树。

## 资源与候选筛选

所有 polling geometry 都先 compile-only，再调用实际 HIP occupancy 接口；512 CTA 要求至少 2 CTA/CU，768 CTA 要求至少 3 CTA/CU。四 wave 版本以实际 256 threads 查询，其他版本为 512 threads。编译前启用 debug info；下表均无 VGPR/SGPR spill 或 scratch。

| 候选 | 主要变化 | TP8 VGPR | threads/CTA | resident CTA/CU | 实测 grid | 完整层初筛，μs |
|---|---|---:|---:|---:|---:|---:|
| compact | 两级 expert 分组 | 102 | 512 | 2 | 512 | 147.911 |
| compact_loopug | 有界 UG 循环 | 96 | 512 | 2 | 512 | 148.038 |
| compact_w4 | 四 wave；两轮覆盖八个 TP peer | 124 | 256 | 4 | 768 | 160.401 |
| compact_dn1_w4 | 四 wave；down 仅预取一个 K128 | 122 | 256 | 4 | 768 | 162.219 |
| compact_vecmid | 四 sample 并行轮询、向量写 LDS | 102 | 512 | 2 | 512 | 146.072 |
| compact_vecmid_loopug | 并行轮询 + 有界 UG 循环 | 96 | 512 | 2 | 512 | **145.021** |
| compact_lds | 再按 MFMA 片段交错 LDS sample 布局 | 104 | 512 | 2 | 512 | 146.016 |

上表为 node50、seed1234、50 次 graph replay 的完整层初筛。同轮 staged 为 134.236 μs，原 768/P544 pipeline 为 143.206 μs。八 wave compact 的 768 配置驻留不合格，只有编译资源记录，没有启动它们。

另外两个候选先在 node47 完成：`compact_dn1` 为 VGPR100、2 CTA/CU、147.565 μs；再将 UG 预取量降至 2 的 `compact_dn1_ug2` 为 VGPR102、2 CTA/CU、147.591 μs。同节点第一版 compact 为 147.193 μs。它们没有恢复驻留或性能，后续转向有界循环与四 wave，而不是继续扩大预取参数矩阵。

八 wave compact 的 LDS 为 39,168 B/CTA，四 wave 为 35,072 B/CTA；原 pipeline 为 15,360 B/CTA、VGPR78、3 CTA/CU。四 wave 恢复了驻留，但同时增加单 wave 的 K/专家工作量，实测整体更慢。单次改动同时影响调度、活跃值和等待，不能只凭 VGPR 或占用率解释全部耗时。

## 三 seed 的完整层结果

以下全在 node50 上重新测量。每配置 100 次、每次 16 层 graph replay，使用 GPU events，逐次取最慢 rank；配置顺序随 seed 轮换。soft、ATT、PMC 均关闭，包含完整 attention→MoE→tail，不通过减去 MoE-only 估算 attention 时间。

| seed | staged，μs | 原 pipeline 768/P544，μs | 最快 compact 512/P256，μs | compact − 原 pipeline |
|---|---:|---:|---:|---:|
| 1234 | 134.488 | 143.875 | 146.000 | +2.125 |
| 2025 | 134.511 | 143.528 | 145.531 | +2.003 |
| 42 | 134.001 | 142.942 | 144.857 | +1.915 |
| 三 seed 中位数 | **134.488** | **143.528** | **145.531** | **+2.003** |

最快 compact 相对原 pipeline 慢约 1.4%，相对 staged 慢约 8.2%。初筛选择没有在确认阶段转变为完整层收益。

图：[完整层与内存请求 PNG](../../results/kimi_mtp_expertcompact_20260928/node50/compact_comparison.png) / [SVG](../../results/kimi_mtp_expertcompact_20260928/node50/compact_comparison.svg)。左图与右侧 PMC 来自不同运行；内存误差线为八卡 min/max，不是置信区间。

## HBM 确实减少，L2 请求没有同比减少

复用前一轮已验证的四个计数器，同组采集每个目标 kernel 第 65 次匹配 dispatch、八个 GPU。`TCC_EA0_RDREQ_DRAM_32B_sum × 32` 表示 DRAM/HBM 读请求字节，`TCC_READ_SECTORS_sum × 32` 表示 L2 读请求字节。下表为每卡中位数，MiB=2²⁰ bytes。

| seed | 独立 expert | 原 pipeline HBM | 最快 compact HBM | HBM 减少 | 原 pipeline L2 | 最快 compact L2 |
|---|---:|---:|---:|---:|---:|---:|
| 1234 | 48 | 125.840 | 103.481 | 17.77% | 203.372 | 213.245 |
| 2025 | 47 | 122.760 | 101.352 | 17.44% | 208.377 | 204.154 |
| 42 | 47 | 123.902 | 101.333 | 18.21% | 209.059 | 204.563 |

seed1234 的第一版 compact 为 HBM 103.361、L2 229.321 MiB；最快版降低了它的 L2 请求，但相对原 pipeline 的 L2 变化仍是一个 seed 上升、两个小幅下降。同节点 staged GEMM1+GEMM2 为 HBM 102.152、L2 112.556 MiB。pipeline 包含 routed TP，staged 计数仅相加两个 routed GEMM，因此不把两者全部差距归因到单一实现变化。

48/47 个独立 expert 的逻辑权重加 scale 预算分别为 100.406/98.314 MiB。实测 compact 已接近这个预算；继续省相同权重的重复 HBM 读取，空间明显小于这轮之前。保留的 tagged mid 只有 96 KiB，但被 224 个输出 owner 反复读取和轮询；减少唯一 HBM 字节数并不等于消除这些请求与依赖。

PMC 会扰动调度和缓存，其 dispatch 时间及应用 event 时间均未混入性能表。原始计数、逐卡范围和 metadata 见 [memory_summary.json](../../results/kimi_mtp_expertcompact_20260928/node50/memory_summary.json)。

## ATT 对照与下一步定位

三份 ATT 共 **224 个采样 wave**，全部完整 stitched；动态指令次数、stall 与 interval 累加和 decoder code table 一致。每份 capture 的目标 `.text` 与对应 compile-only HSACO 一致，debug info 和源码快照逐字节核对通过。表格只比较具有 down 工作的 consumer wave；第一版另有 16 个不执行 MFMA 的采样 wave。

| consumer 指标 | 原 pipeline | 第一版 compact | 最快 compact |
|---|---:|---:|---:|
| 采样 consumer wave 数 | 32 | 16 | 32 |
| 每 wave down MFMA | 96 | 72 | 72 |
| 每 wave raw weight load | 27 | 18 | 18 |
| scoped LDS read 指令 | 96 | 156 | 102 |
| scoped LDS write 指令 | 24 | 73–75 | 19–21 |
| VMEM-wait 占 scoped stall | 83.59% | 72.84% | 75.47% |
| LDS/SMEM-wait 占 scoped stall | 6.13% | 17.66% | 12.42% |

scoped 范围为每 wave entry 到最后一次 MFMA，排除后续 pack/TP。LDS 数字包含该范围内的 metadata 操作，不能全部视作 intermediate 数据字节。权重与 MFMA 指令减少证明两级复用进入了实际机器执行；并行轮询/向量搬运的 LDS 指令减少也得到验证。

最快 compact 约 73.1% 的 scoped stall 映射到 intermediate 初始读取和重试调用点。权重更少、LDS 操作更少，并没有移除 producer→consumer 的跨 CTA 依赖；其完整层表现仍受等待及资源代价影响。没有对各项代价做独立因果拆分。

图：[三份 ATT wave timeline PNG](../../results/kimi_mtp_expertcompact_20260928/node50/att_analysis/att_wave_comparison.png) / [SVG](../../results/kimi_mtp_expertcompact_20260928/node50/att_analysis/att_wave_comparison.svg)。各 capture 单独以最早采样 entry 为原点，纵轴按角色分组，不是逻辑 CTA ID。ATT 只使用 cycles，不转换为 μs、不跨 GPU 对齐，也不用不同采样角色/数量的 stall 总量判定性能。

进一步改变计算流，应评估 **同一 CTA 完成 expert 的 intermediate 行块后立即做 down，再由输出 owner 归约部分和**，以改变全局 mid 的重复读取与等待依赖。其代价仍是确定性归约和更大的部分和 mailbox：现有预算中 U32/U64 的 tagged 部分和写+读分别为 42/21 MiB，且并行度不同。这是下一项算法层面的候选，尚未实现，不能把本轮结果解释为它已经有效。

## 正确性、环境与归档

本轮 **32 次 TP8** 完整检查通过：node47 三次、node50 二十九次，其中 profiler 运行也执行相同检查。检查包括独立层参考、有限值、八 rank 一致性、routing、routed TP/tail 精确关系、改变 payload 后 replay，以及 true-MTP incoming state 保留和状态快照。

单卡有六次原有独立 FP32 参考/三次 payload replay 检查，以及七个候选的改变路由检查。后者保持 graph 内指针不变，依次切换 16、17、47、48、63、64 个专家，同时更新 expert ID、route slot、系数和输入；**42 个变化路由 case** 均逐次对照 staged 与独立 FP32 参考通过。UG/下游 LDS 向量映射、UG K-tail、expert/wave task 覆盖另有静态检查。

node47 的初始三个性能结果在 14:51 UTC 前完成。另一项 46/47 EP16 作业于 14:51:13 启动后，本轮旧 512 基线发生 GPU 阻塞；已仅终止自己的进程，保留 `-15`、命令与环境冲突记录，没有记作正确性失败或有效性能值。此后 GPU 实验迁至空闲 node50，重新编译、查询 occupancy，并在每个 subprocess 启动前检查无 KFD 进程。没有终止他人的任务。

node50 使用 Podman `hzm_work`，PyTorch `2.9.1+rocm7.2.0.git7e1940d4`，复制了同一份 FlyDSL runtime；runtime 文件 SHA256 和环境信息单独记录。两节点结果分开保存，没有拿跨节点时间差声称加速。所有 GPU 作业串行，结束后 node50 无 KFD 进程。

证据包含九个候选的源码/补丁/生成脚本、命令/日志/rc、实际 occupancy 和 HSACO、TP1/TP8 结果、原始 PMC、三份 ATT 原始 trace/目标 code object/源码、分析脚本及图。省略无关库 code object 和可重建编译缓存，保留相应 inventory。

- [node47 审计](../../results/kimi_mtp_expertcompact_20260928/node47/experiment_audit.json)、[node50 审计](../../results/kimi_mtp_expertcompact_20260928/node50/experiment_audit.json)。
- [node50 验证汇总](../../results/kimi_mtp_expertcompact_20260928/node50/verification_summary.json)。
- [node47 归档](../../results/kimi_mtp_expertcompact_20260928/kimi-compact-node47-evidence.tgz)：5,406 个文件，SHA256 `f063ce59e2073550f774209ebf05318e17124176fb10bffbbe1b01161c288cb3`。
- [node50 归档](../../results/kimi_mtp_expertcompact_20260928/kimi-compact-node50-evidence.tgz)：7,031 个文件，SHA256 `c56f549046dac70055653bdfc66aba53d95972d7ff1e1bbf15ab3ae02af61072`。

本轮没有接入负收益候选，工作区原 kernel 和独立正确性修复 `cb147e89d02a38dcc35a7929cf6593aa84784d18` 保留，HEAD 仍为 `3563ef29b7c9f9030e5f9645940948299b03ca46`。没有提交或推送。
