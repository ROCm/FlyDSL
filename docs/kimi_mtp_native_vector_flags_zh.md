# Kimi K3 MTP：native UG、直接 GEMV 与独立 ready flag

这些实验均未达到相对最快公开 pipeline 降低完整层延迟 10% 的目标。所有实现仍在独立 overlay 中，未合入默认 kernel。最新同节点 public 对照为 143.577 μs；此前三 seed public 为 143.528 μs，目标约 129.18 μs。

## 完整层结果

下表均为 node50、TP8、S4 true-MTP、16 层 graph、50 repeats、seed1234，无插桩 GPU event 计时，取最慢 rank 的中位数。表中数据用于筛选，不能替代多 seed 交替复测。

| 方案 | grid | 完整层 μs | 结论 |
|---|---:|---:|---|
| public 对照 | 768 | 143.577 | 比较基准 |
| native compact、独立 pack、F1 | 768 | 157.217 | 回退 |
| native public、独立 pack、F1 | 768 | 158.225 | 回退 |
| native public、独立 pack、F2 | 768 | 158.066 | 回退 |
| native inline pack、F1 | 768 | 149.127 | 回退 |
| native inline pack、F2 | 768 | 144.245 | 无收益 |
| native public、metadata 入 LDS | 768 | 145.097 | 无收益 |
| native compact、metadata 入 LDS | 768 | 148.926 | 回退 |
| 直接 BF16/FP4 GEMV、F1 | 512 | 155.736 | 回退 |
| bounded GEMV、F1 | 768 | 149.279 | 回退 |
| bounded GEMV、F2 | 512 | 151.479 | 回退 |
| raw mid + 独立 route tile flag | 512 | 394.773 | 严重回退 |

native 独立 pack 的追加对照为 157.106、158.256、158.137 μs，与筛选结果一致。不能把较慢 native 方案作为优化目标的分母。

## exact BF16 → 两项 FP8

对每组 K32 BF16，以截断低四位尾数的值作为高项，将高项和残差分别编码为带 scale 的 E4M3。只有 FP32 重建逐元素等于原 BF16，且整数范围检查通过，才允许 native 计算；否则整组 K128 回到原 BF16 MFMA。SiTU 和 down 保持原 BF16 语义。

GPU 检查覆盖全部 65,280 个有限 BF16 模式：2,040 个 block 中 1,824 个走 native，216 个保守 fallback。还覆盖三 seed、normal、变更 payload、零、宽指数、tiny、最大有限值及 graph replay。BF16 极小值的整数范围检查防止下溢后错误接受。原生产精度阈值未放宽。

FP8 B operand 的 lane 布局由硬件 probe 确认：两个 16-byte 半块相距 64 bytes；K32 scale 对应 lane//16。早期错误布局及过严的 probe 相对误差检查保留在 audit 中，最终 probe 同时验证 relative L2 和逐元素 FP32 gamma_256 误差界。

最初逐 K32 fallback 使用 164 VGPR，仅 1 CTA/CU，未启动 polling kernel。后续 F1/F2 的真实 HIP occupancy 满足相应 grid 要求。inline F2 使用 78 VGPR、3 CTA/CU，但完整层仍略慢于 public。

## ATT 发现

`att_native_public_f2_g768` 的 96 个 wave 全部完成拼接，64 producer、32 consumer，静态代码表、动态 totals、debug source 与 code object 校验一致。该次捕获 producer 执行 28 或 56 条 native MFMA，没有 BF16 fallback。

统计区间为每个 wave 从入口至最后一条 MFMA，排除后续 pack/TP；单位为 ATT cycles，不能换成 μs，也不跨 GPU 对齐。

- producer 的 scoped stalls 中，VMEM wait 占 76.56%，VMEM load 占 13.12%。
- 37.10% 位于 fallback flag 依赖位置，34.72% 位于依赖全局 FP8 scale 的 native MFMA 位置；两处合计约 71.83%。
- consumer 的 VMEM wait 占 89.77%，主要等待中间结果。

把 metadata 放入 LDS、或合入 activation staging 后，确实消除了大部分回退，但尚未赢过 public。减少 MFMA 条数并未自动转化成完整层收益。

图表与逐 wave 数据：`../../results/kimi_mtp_nativebf16_20260928/node50/att_analysis/`。

## GEMV 和 ready flag 的含义

GEMV 让八个 lane 直接计算一行，避免 MFMA 的重复 sample 列，但增加标量 unpack/归约和寄存器压力。F2 原版使用 166 VGPR、1 CTA/CU，未启动；bounded 版本减小 live range 后可运行，仍慢于 public。

独立 flag 版本由 producer 写 raw BF16 mid，经 device release fence 和 CTA barrier 发布每个 UG tile 的 epoch。consumer 用 atomic monotonic load + ballot 检查某条 route 的 12 个 tile，再 acquire 并把 K384 放入 LDS。正确性通过，但静态 ISA 有两处 `buffer_wbl2` 和 145 处 `s_waitcnt`，完整层 394.773 μs。此版本没有 ATT，不能把所有回退定量归因于某一条 fence；也不能直接删掉 fence 而放弃 payload 可见性的证明。

## 验证与证据

所有已启动候选通过 TP1 独立 FP32 reference、routing/payload graph replay，以及 TP8 exact routing、rank equality、TP/tail 和 true-MTP state 检查。varied-scale 检查覆盖逐 block UG/down scale 变化和 normal、wide、mixed、tiny、zero 输入。阈值仍为 mid relative L2 < 2e-3，output < 1e-2。

本地证据位于 `../../results/`，归档 SHA256：

| 归档 | manifest 已校验条目 | SHA256 |
|---|---:|---|
| kimi-native-node50-evidence.tgz | 8,360 | 46805c1b0699d442de1365137305507bf7958f3dd0902dca4fd9a3908c4e4ad2 |
| kimi-native-node46-compile-evidence.tgz | 3,166 | 04773c3152835f25b54f5a1b8a0c715b93cac055aff39a56c5882aaff1312026 |
| kimi-vector-node50-evidence.tgz | 3,256 | 6c2d801ea03aa7aa38cd1bd547550244b61c6e9c149359fe317891cf32387a09 |
| kimi-routeflags-node50-evidence.tgz | 834 | 7d0485559105d71a3d3eaba3f03fe67b335db9f739501c34fd88c719f52b9e82 |

下一实验是 expert-local U192，把 UG→SiTU→down 保留在同一 CTA，用一个 64-bit word 原子更新 FP32 数值与贡献计数，尝试同时避开大块 partial 回写与独立完成 flag 的 fence。它的 CAS 争用、初始化开销、精度和实际收益需要单独验证。
