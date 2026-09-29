# Kimi MTP：U192 局部 down 与计数原子累加

这轮实现了 UG→BF16 SiTU→down 全部留在同一 CTA 的 U192 方案，并比较原子累加、连续写出、分片与 tagged partial。四个版本正确性通过，但完整层均慢于约 143.58 μs 的 public pipeline，未合入默认 kernel，10% 目标仍未达到。

## 数据流与正确性

每个独立 expert 分成两个 U192 task，最多 128 个 producer CTA。每个 producer 用三轮 U64 计算 UG，将 SiTU 的 BF16 结果保存在 LDS，再计算这一 U192 对全部 H3584 输出的贡献。grid512、512 threads/CTA；224 个输出 owner 与 producer 集合不相交。

原子版用一个对齐的 64-bit word 保存每个输出的 FP32 sum 和贡献 count。单次 monotonic CAS 同时更新两者；失败后从返回的旧值重新计算，原贡献保持不变。消费者从 count 达标的同一次 atomic load 读取数值，避免独立 ready flag 的发布/可见性问题。所有有效路由都计数，包括权重为零的路由。累加器在同一 stream 每次调用前清零，清零包含在 graph 与完整层 event 时间内。

第一版 16 个 lane 各写四个输出，形成四轮串行 CAS。`atomicdown_wave` 利用 MFMA 中重复的 sample 列，让 64 个 lane 各选择一个输出，合成一轮连续地址的 CAS。四分片版再按 top-k slot/4 分配累加器，消费者对四片求和。单片 112 KiB、目标 count32；四片共 448 KiB、每片 count8。

`localdown192_coalesced` 保留相同的 U192 计算，以 `ds_bpermute` 将 16×4 的结果转为 sample-major，然后连续写出每 route 的 tagged FP32 partial。每 16 lanes 写一个连续 128-byte tile；消费者沿用已验证的 tagged partial 轮询与 FP32 归约。该版本没有原子累加器清零，partial 数据为 3.5 MiB，每次写出携带 epoch。

所有版本保留 diagnostic tagged BF16 mid。TP1 独立 FP32 reference 与 staged 对照、16/17/47/48/63/64 个 expert 的改变路由、逐 block scale 变化、normal/wide/mixed/tiny/zero 输入以及零系数检查均通过。每个路由 case 重放 8 次、每 graph 16 层；原子版本检查最终 count 精确为 32 或 8。TP8 检查精确 routing、rank equality、独立 mid/output、TP/tail 和 true-MTP state，且在 graph replay 后重查。仍使用 mid < 2e-3、output < 1e-2 的原阈值。

## 资源与完整层时间

node50、TP8、S4、seed1234、16-layer graph、50 repeats，无插桩 GPU events，取最慢 rank 的中位数。所有候选先 compile-only，再以实际 512 threads 查询 HIP occupancy，LDS 均为 40,704 B，无 SGPR/VGPR spill 或 private scratch。

| 候选 | TP1 VGPR / CTA每CU | TP8 VGPR / CTA每CU | 完整层 μs |
|---|---:|---:|---:|
| U192，四轮 CAS | 104 / 2 | 72 / 3 | 265.438 |
| U192，单轮 full-wave CAS | 104 / 2 | 70 / 3 | 200.590 |
| U192，full-wave CAS + 四片 | 106 / 2 | 104 / 2 | 188.301 |
| U192，连续 tagged partial | 74 / 3 | 72 / 3 | 181.638 |

四版均实际运行 grid512。占用率达到 3 的版本也没有据此声称验证了 grid768。最新既有 public 对照为 143.577 μs；上一轮 U64 local-down 为约 157.22 μs。这里是负收益单 seed 筛选，没有开展多 seed 获益确认。

## ATT：原子延迟存在，但不是大量 CAS 重试

捕获四分片与 tagged U192 两版，分别 64 个 wave，全部完整 stitched；动态指令总数/stall/interval 与 code table 一致，`.text`、debug info 和源码快照与 compile-only 产物匹配。两版各采到 24 个 producer wave。四分片另外有 24 consumer/16 idle wave，tagged 有 16 consumer/24 idle wave；idle wave 只执行初始化和任务判断，未混入 producer/consumer 分母。

producer 的统计区间从入口到最后的 CTA barrier，consumer 从入口到输出 pack 后第一处 CTA barrier；均排除后续 TP。单位是 ATT shader cycles，不换算 μs、不跨 GPU 对齐。两个捕获的样本组成不同，不能直接用 wave 个数推断整卡各类占比。

| producer 指标 | 四分片 CAS | tagged partial |
|---|---:|---:|
| 每 wave UG MFMA | 336 | 336 |
| 每 wave down MFMA | 168 | 168 |
| VMEM wait，占 scoped stalls | 72.21% | 56.29% |
| LDS/SMEM wait | 14.46% | 23.46% |
| scoped cycles 中位数 | 144,562 | 124,180 |

四分片每个 producer wave 预期执行 28 次 CAS；动态记录为 28–32 次，24 waves 合计 694 条，基础次数为 672 条。这里是 wave 级指令次数，不能换算每 lane 的成功更新/重试总数。此捕获不支持“大量重试主导”的解释；约 53.78% producer stalls 位于原子旧值依赖与 CAS 相关源码位置，说明原子往返和串行依赖仍昂贵。

tagged 版约 38.79% producer stalls 位于 UG 计算中的依赖位置，31.42% 位于最后 producer barrier 的源码位置；该位置也包含编译器安排的等待，不能全部归因于 `s_barrier` 指令。消费者 VMEM wait 占 87.68%，主要等待 tagged partial。

四分片 consumer 总体 barrier stalls 为 87.00%，原因需结合分工解释：只有 wave0 的一部分 lane 轮询输出，其余 wave 等待 pack 后的 CTA barrier 再执行 TP。不能将这个占比当成独立 barrier 优化能够兑现的加速比例。

## 实测内存请求

同一目标 kernel 的第 65 次匹配 dispatch，四个计数器同组、8 GPUs，表中为八卡均值，MiB=2²⁰ bytes。PMC 与 ATT 的应用时间未混入性能表。

| 计数器换算后的请求量 | 四分片 CAS | tagged U192 |
|---|---:|---:|
| HBM read | 118.997 | 108.585 |
| HBM write | 0.203 | 3.703 |
| L2 read sectors×32 | 246.634 | 480.703 |
| L2 write sectors×32 | 0.641 | 4.141 |

原子版本的普通 write-sector 与 dispatch 内 DRAM write 计数不能直接当作全部 atomic RMW 物理流量；缓存驻留、atomic 计数分类和后续写回影响这个边界。清零 kernel 也不在这次目标 dispatch 的 PMC 范围内，虽然已计入完整层时间。因此不能宣称原子写流量只有 0.203 MiB，或用这些数字计算整个 layer 的总内存流量。

tagged U192 的写请求接近 3.5 MiB partial 数据加原有写出，连续写出布局生效；但 L2 read 仍达到约 481 MiB，远高于既有 public 的约 206 MiB。减少唯一 HBM 字节数仍未消除反复读取/等待及计算资源分配成本。

## 证据与下一项实验

源码、patch、验证、资源、原始 ATT/PMC、分析器和图表在 [实验目录](../../results/kimi_mtp_atomicdown_20260929/)。每版使用独立 source overlay；没有修改公开默认 kernel。编译阶段显式整数右移参数的一次修正，以及分析器的空白/idle-wave 分类修正，均保留日志，未改变 kernel 精度标准。

下一项受控实验回到 U32 compact：先完成 UG，再由独立 kernel 执行 compact down/combine/TP。stream 上的 kernel 边界保证 48 KiB raw BF16 intermediate 已就绪，移除跨 CTA 中间结果轮询、大块 down partial 和 atomic combine；完整层仍包含两次 launch。它牺牲 UG/down overlap 换取资源分离，必须由新的正确性与完整层计时判断。
