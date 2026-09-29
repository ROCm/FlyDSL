# Kimi K3 true-MTP routed tile pipeline 实验

本实验基于 PR #1204 的 `3563ef29b7c9f9030e5f9645940948299b03ca46`，目标为一个请求的 MTP3：`--mtp --samples 4` 表示当前 token 加三个候选 token，使用 `[S+1]` 的状态快照链。

本轮保持 staged KDA、MTP 状态链及 shared/latent tail 的计算布局不变。仅把 routed GEMM1 + SiTU + GEMM2 + combine + routed TP 接入一个 tile kernel。Router/latent-down/shared-up 和最终 fused tail 仍为独立 kernel；这不是完整 decoder-layer MonoKernel。

## 结果与结论

**tile overlap 已实现并通过正确性检查，但 TP8 MoE 和整层性能目标尚未达到，因此默认关闭。**

| 测量范围 | staged 中位数 | tile 中位数 | 延迟变化 |
|---|---:|---:|---:|
| 单卡 routed UG + SiTU + down + combine | 33.16 μs | 29.49 μs | 降低 11.1% |
| TP8 MoE 段，真实 MTP 输入 | 67.15 μs | 70.30 μs | 增加 4.7% |
| TP8 完整 staged KDA/MTP 层 | 134.24 μs | 146.59 μs | 增加 9.2% |

单卡和整层各为三个 seed 的中位数。MoE 候选包含三组 A/B 加三组独立确认，共六个中位数：65.90、71.28、71.42、70.30、70.29、69.60 μs；基线为三组。65.90 μs 的局部偏快结果未在追加复测中复现，全部原始记录均保留，不据此宣称收益。整层候选三组为 147.36、146.59、145.94 μs，回退稳定。

预热后的 graph 最后一层时间线在 **8/8 rank** 上都确认 `first_down_math < last_up`、`first_tp < last_down_math`，证明 UG/down 和 down/TP 确实重叠。插桩计时不进入上表。首次 cold JIT 时间线中的跨 rank 启动等待也不用于性能分析。

所有最终及确认运行均通过：routing、incoming snapshot、routed TP、tail reduction/residual 的精确检查；graph replay 后再次通过。跨全部 rank/seed 的最大 relative L2 为 attention 0.001543 以下、conv state 0.000087 以下、recurrent state 0.000349 以下、output 0.0040 以下；新 expert mid 0.000072 以下、routed partial 0.001679 以下。

## 调优观察

首轮 S4 单卡 down tile 16/32/64 分别约 34.1/45.3/63.9 μs。扩大 tile 缩减了消费者并行度，未带来收益。原先 256-CTA 混合生产/消费调度的 TP8 MoE 段约 77.1 μs；512 个共驻留 CTA 配合交错发布降到约 70–71 μs。

owner reduce/broadcast、同专家多 token 的 UG 权重复用、LDS padding、权重 cache modifier 2 均完成相应编译/正确性/性能对照，没有超过当前候选。UG 复用和 cache modifier 实验保存在结果目录 overlay 中，不留在当前默认实验路径。完整 MTP 输入的 64 条 route 对应 47–48 个专家；单卡随机路由有不同的专家重叠率，不能将单卡 11.1% 的收益外推到 TP8。

下一步应针对完整输入下的 down 微内核、专家权重复用和 TP 等待做进一步分解；现有结果不足以把回退完全归因于某一项。继续扩大 down tile、增加 CTA 或仅减少通信字节量，都没有获得目标收益。

## 实现

- `kernels/kimi_k3_monokernel/routed_pipeline.py`：UG producer 发布带 epoch 的 BF16 tile，down consumer 按 K128 片段等待与计算。每个输出 tile 自行 combine 并启动 TP。
- `staged.py`：通过 `routed_pipeline=True` 接入新路径。默认关闭；完整 `KimiK3MonoKernel` 的 attention 实现没有变化。
- `tail.py`：已完成 routed TP 时直接读取 reduced 值，其余 norm、shared/latent projection 和最终 TP/residual 沿用原布局。
- `tools/monokernel.py`：同一 staged KDA/MTP 路径上的整层和 MoE 段计时、所有 rank 的校验和结果记录。
- `tools/tail_check.py`：从真实 BF16 latent 和实际选择的专家独立计算参考值，并按固定 rank 顺序验证 TP 和尾部残差。

当前最快候选使用 `grid=512, producers=256, up_tile=32, down_tile=16, prefetch=4`。224 个 down consumer 与 256 个 UG producer 分开驻留；其余 32 个 CTA 无任务。UG 任务交错安排各 sample 的 slot 0/8、1/9 等，使每个 down wave 的首批输入更早就绪。所有 rank 使用相同的静态输出任务顺序。

512-CTA 配置必须满足至少两个 block/CU 的实际 HIP occupancy。编译资源检查保存在结果目录中；没有根据 launch 数量直接假定能够共驻留。`reduce_scatter` 为未获益的对照选项，当前候选使用 `complete` TP。

## 测量方法

硬件为 node46 的 MI355X，TP8、BF16 activation、MXFP4 routed expert weights。A/B 使用相同 source snapshot、相同输入和 seed、相同 staged KDA attention。基线与候选均包含此前正确性 commit `cb147e89` 的两处计算修复，迁移到新包路径。原 commit 未修改。

整层计时涵盖完整 staged KDA MoE decoder layer。`--moe-only-bench` 在真实 MTP forward 完成预热和输入准备后，仅捕获 router/latent-down/shared-up/sort 到最终 tail/residual 的 `_moe` 调用；它不包含 post-AttnRes 的输入量化。两者是独立测量，不能相减推导 attention 时间，也不能用局部收益代替整层收益。

使用非插桩 GPU events，graph 每次含 16 层、重复 100 次，按每次的最慢 rank 统计中位数。最终复测 seed 为 1234、2025、42，部分顺序反转。单卡 routed 对照使用独立随机路由，其专家分布与完整 MTP 层不同。

正确性门槛：所有 rank routing 相等、MTP 初始快照 bitwise 不变，attention relative L2 < 2e-3，conv/recurrent state < 5e-4，输出 < 1e-2。新 routed 中间值独立参考误差 < 2e-3、routed partial < 1e-2；routed TP、最终 TP 和 residual 必须精确相等。graph replay 后重查 routing、快照、专家和 tail。

GPU 内 timestamp 运行只用于确认 overlap，不能作为性能值。首次预编译脚本误用了默认会执行的 `flyc.compile`，占位地址触发非法访问；已改为显式 `COMPILE_ONLY=1`，错误日志保留，未计入任何性能结论。

## 重现

基线：

```bash
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --staged --mtp --samples 4 --layer-idx 1 \
  --check --bench --layers 16 --repeats 100 --seed 1234
```

候选在相同命令追加：

```bash
--routed-pipeline --routed-grid 512 --routed-producers 256 \
--routed-up-order interleaved
```

MoE 段对两侧均追加 `--moe-only-bench`。单卡诊断：

```bash
python -m kernels.kimi_k3_monokernel.tools.routed_pipeline \
  --samples 4 --grid 512 --producers 256 --up-order interleaved \
  --repeats 100 --seed 1234
```

完整日志、每次命令、exit code、所有 rank 的数值检查、资源记录、不同实验 source overlay 和 SHA256 清单保存在 `/home/zihuang/work/mega_transformer/results/kimi_mtp_tile_20260928`。本轮验证范围是最新 PR 的 S4 true-MTP；其他 shape 不能沿用本轮性能结论。
