# Kimi K3 MTP3：768 CTA 与 route metadata 预取

本轮接续 [GLM ATT 对照](glm_kimi_pipeline_att_zh.md)，在 node47 的 MI355X 上验证 producer 供给、consumer 交接粒度和编译器调度。输入为一个请求的 MTP3，即当前 token 加三个候选 token（`--mtp --samples 4`），TP8、staged KDA、native MXFP4 routed experts。

**768 CTA / 544 producer 的整层延迟比 512 CTA / 256 producer 降低约 2 μs，但仍未超过 staged。** 已将验证过的 route-prefetch 及 768 配置接入显式实验开关，默认路径仍为 staged。两种没有获益的变体只保留在结果目录。

后续已补充 [512/768 的无 soft 插桩 ATT 对照及八个候选实验](kimi_mtp_att_followup_zh.md)。该轮确认了 consumer 的末轮冗余预取和持续 intermediate 等待，但没有得到稳定的新整层加速，未改变本报告中的已接入配置。

## 非插桩 A/B

所有数字均来自关闭 soft/ATT 的 GPU events，graph 内 16 层、50 次 replay，每次取最慢 rank。使用 seed 1234、2025、42，轮换三个方案的执行顺序，并反转其中一组的 scope 顺序。

512 和 768 两个候选都采用相同的 route-prefetch：同一路由的三个 K128 权重片段只加载一次 expert ID/coefficient，仍然逐 chunk 等待 intermediate 和计算。只改变 grid 与 producer 数，down 固定 224 个 D16 task，UG 为 U32/F4/interleaved。

| scope，μs | staged | route-prefetch 512/P256 | route-prefetch 768/P544 |
|---|---:|---:|---:|
| MoE 段，三个 seed 中位数 | 67.102 | 68.556 | 67.433 |
| 整层，三个 seed 中位数 | 133.815 | 144.121 | 142.086 |

| seed | staged 整层 | 512 整层 | 768 整层 | 768 − 512 |
|---|---:|---:|---:|---:|
| 1234 | 134.049 | 144.410 | 142.564 | −1.846 |
| 2025 | 133.815 | 144.121 | 142.086 | −2.036 |
| 42 | 133.689 | 143.435 | 141.691 | −1.744 |

整层三个 seed 都改善，中位数下降约 1.41%；相对 staged 仍慢约 **6.18%**。MoE 的逐 seed 变化为 −1.123、+0.601、−2.660 μs，方向并非全部一致。两版在 seed 2025 的 MoE 段都偏快，不能据这个 seed 的局部结果宣称 fusion 已超过 baseline。MoE-only 与整层是不同的缓存/通信上下文，不通过两者相减拆分 attention 时间。

## 驻留条件与正确性

所有 GPU launch 前均先 compile-only，再通过 HIP occupancy API 检查轮询 grid 的驻留条件。编译前统一启用 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`，不同实验使用独立缓存并保留 HSACO、源码和命令。

| TP8 版本 | VGPR | SGPR | LDS / CTA | spill / scratch | HIP CTA/CU |
|---|---:|---:|---:|---:|---:|
| route-prefetch 512/P256 | 78 | 73 | 15,360 B | 0 | 3 |
| route-prefetch 768/P544 | 78 | 73 | 15,360 B | 0 | 3 |
| 三个 chunk 一次 staging | 76 | 73 | 15,360 B | 0 | 3 |
| GMM2 八个 route 全展开 | 150 | 72 | 15,360 B | 0 | **1** |
| 轻量时间点 | 78 | 80 | 15,360 B | 0 | 3 |

最终接口另做了 512/P256 route-prefetch 的 TP1/TP8 × soft1/soft2 四组 compile-only 检查，均为 VGPR124、无 spill/scratch、2 CTA/CU，满足 512 CTA 驻留条件。这四组只验证编译资源；768 配置仍禁止使用公开 soft 模式。

512 原配置有 256 producer、224 consumer、32 个无任务 CTA；768 配置为 544 producer + 224 consumer。UG 共 768 个任务，前者每个 producer 做 3 个，后者 224 个 producer 做 2 个、320 个做 1 个。CTA 数不是物理 CU 分布的直接证明。

共完成 30 次 TP8 运行：初始检查 1 次、三 seed 矩阵 18 次、route staging 对照 4 次、轻量打点 4 次、最终接口验证 3 次。全部通过 8-rank 检查和 graph replay 重查，包括：

- attention、conv/recurrent state 与独立参考的既定误差门槛；MTP incoming snapshot 保持不变。
- 独立 expert mid/routed partial 参考；routing 精确一致。
- routed TP、tail reduce/residual、AttnRes delta 精确检查。

route staging 和最终接口还各通过单卡独立 FP32 参考及改变 payload 后的 replay 检查。未启用 route-prefetch 的原路径通过最终 TP8 检查；原有 S1/S2/S8 配置另做了 compile-only 检查，这不构成这些 shape 的新性能结论。

最终 API 中 512/768 route-prefetch 的 `.text` 与用于三 seed A/B 的实验产物**逐字节一致**，分别为 9,920 / 9,984 bytes。最终接口再运行的 seed 1234 得到 MoE 66.906 μs、整层 142.907 μs，作为接入验证单独保存，不替换三 seed 矩阵结果。

## 轻量时间点解释

原来的完整 soft instrumentation 需要更多 VGPR，只能达到 2 CTA/CU。为诊断 768 配置，使用独立 `candidate_lite`：仅 wave 0 记录绝对时间点，不跨循环保留 polling 累加器。LDS 和 VGPR 均保持原值，3 CTA/CU 通过资源 gate。

诊断输出包含 CTA entry、UG end、down start、wave-0 第一 K128 chunk 完成、down end、pack/TP end、CTA exit 和 epoch。使用额外 output buffer，未增加 barrier，未改模型输出。所有非 wave-0 槽位为零，四组 × 八 rank 的时间点单调性和 epoch 均已验证。

下表是每个 rank 的 warm graph 最后一层快照，再在八个 rank 之间取中位数，单位 μs；不能作为多轮时延分布或替代上面的 event A/B。

| 整层上下文中的指标 | 512/P256 | 768/P544 |
|---|---:|---:|
| 最后 UG end，相对该 rank 最早 CTA entry | 24.28 | 19.99 |
| producer 持续时间 p50 | 21.40 | 16.66 |
| down 持续时间 p50，包含 polling | 27.16 | 25.06 |
| 最早第一 K128 chunk 完成 | 7.43 | 8.52 |
| pack + TP 持续时间 p50 | 3.20 | 2.55 |

两种配置中，224/224 consumer 的第一个 chunk 都在最后 UG end 前完成。更多 producer 明显缩短 UG 尾部，却让第一批 down 稍晚完成；收益来自供给与竞争的重新平衡，不能把 UG 提前约 4.3 μs 全部折算成整层收益。

轻量版整层 event 为 145.077 / 143.532 μs，较相应 seed 的未插桩版增加约 0.46% / 0.68%。MoE-only 对照存在更明显运行差异，不据其正负变化估算打点开销或优化收益。

图表：[PNG](../../results/kimi_mtp_grid768_20260928/node47/grid768_lite_timeline.png) / [SVG](../../results/kimi_mtp_grid768_20260928/node47/grid768_lite_timeline.svg)。横轴按各 rank 内部的 100 MHz `s_memrealtime` 归一化，纵轴为逻辑 CTA ID，不是物理 CU ID。灰蓝色区间包含等待，不能解释为纯 MFMA 时间。该独立脚本的输出 schema 与公开 `soft_profile=1` 不同，仅用于本次诊断，没有覆盖生产工具的原有 soft 语义。

## 两个未采用的方向

**同一路由三个 chunk 一次 staging。** 将 wave-local LDS arena 从 64 words 扩为 192 words，仍在原有 X 区内，减少了波内屏障和 polling 次数；每个 chunk 的数学顺序不变。资源和正确性均通过，但 seed 1234 的结果退化：

| scope | 原 512 | stage-three 512 | 原 768 | stage-three 768 |
|---|---:|---:|---:|---:|
| MoE | 68.556 | 71.282 | 67.433 | 68.744 |
| 整层 | 144.410 | 150.327 | 142.564 | 146.480 |

这一单 seed 对照不用于概括所有 shape，但足以停止当前变体的推广。它说明直接扩大 readiness 粒度并不自动获得 GLM 式 staging 的收益；更粗等待和数据调度的代价仍需要单独分析。

**完整展开八个 route。** 按 GLM 的方式显式展开，并删除最后一批未使用的预取，但 VGPR 增至 150、仅 1 CTA/CU。资源 gate 返回失败，**没有启动该变体的 512/768 轮询 grid**，也没有把未测数据写作性能结果。其源码、资源和 gate 失败日志保留。

## 接口与重现

主路径新增 `routed_route_prefetch` 参数；kernel builder 和 `RoutedTilePipeline` 对应参数为 `route_prefetch`。仅在明确开启时选择新分支，其他 shape 继续走原实现。

最终接入的 768 配置限制为 S4、TP8、P544、U32/F4、默认 sample grouping/D16、interleaved、wave handoff、complete TP、关闭 trace/soft。512 配置为 P256，可用于单卡验证和原有 soft 分析。非法组合在 launch 前拒绝。

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --staged --mtp --samples 4 --check --bench --layers 16 --repeats 50 \
  --routed-pipeline --routed-route-prefetch \
  --routed-grid 768 --routed-producers 544 --routed-up-order interleaved \
  --seed 1234 --output /tmp/kimi-mtp3-grid768.json
```

MoE-only 增加 `--moe-only-bench`。512 对照仅将 grid/producers 改为 512/256。staged 对照去掉全部 `--routed-*` 参数。编译器或调度参数变化后，应重新检查实际 occupancy，不能仅沿用本轮的 3 CTA/CU。

单卡独立参考：

```bash
python -m kernels.kimi_k3_monokernel.tools.routed_pipeline \
  --samples 4 --grid 512 --producers 256 --up-order interleaved --route-prefetch
```

当前收益不足以替换 staged。后续更值得检查的是 producer 的 activation 重复 staging、跨 UG 任务的有界预取，以及更少活跃寄存器的 consumer 软件流水；本次不将这些尚未测量的方向写入性能结论。

## 证据与工作区状态

本地结果：`/home/zihuang/work/mega_transformer/results/kimi_mtp_grid768_20260928`。远端：`/tmp/flydsl-kimi-mtp-grid768-20260928`。

结果保留原始 event 数组、所有 rank 正确性、轻量 raw timestamps、每次命令/env/exit code、全部实验源码、HSACO 资源记录、最终 ISA 比较、SHA256 manifest，以及未满足资源 gate 的展开变体。`analyze_results.py` 和 `analyze_lite.py` 可在本地重查结果；绘图额外需要 matplotlib。

归档 `kimi-mtp-grid768-node47.tgz` 为 12,123,990 bytes，包含 3,770 个文件的 SHA256 manifest。归档 SHA256：

```text
04c35fdc39029f67ba9b20745333031de6771a1c6938f6ba7beb965cfc89a393
```

原 correctness commit `cb147e89d02a38dcc35a7929cf6593aa84784d18` 未改动。本轮没有新增 commit，也没有推送 remote。
