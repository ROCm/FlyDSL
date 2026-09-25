# GLM-5 MLA + MoE：在 FlyDSL 中运行 TileRT 汇编

[English](PERFORMANCE.md)

工作目录为 `/root/FlyDSL-glm5-perf`，分支为 `codex/glm5-perf`，基于 `2fc0cefb`。
已从 `/root/FlyDSL-glm5mono` 找回 Claude 的九个本地调优提交，该目录的暂存修改仍保留。
测量数据、候选快照、反汇编和日志位于 `/root/glm5-perf-results`。

导入汇编版、用 FlyDSL 实现最终 MoE 归约的混合版，以及调优后的常规
`Glm5MlaMoeLayer`，在已测分片上均已接近 TileRT 延迟。使用 `--backend flydsl`
选择常规 kernel，使用 `--backend tilert_inline --replace ffn` 选择混合版。
常规 kernel 不导入 TileRT 汇编主体，也不含 `InlineAsmOp`；剩余的 wave 归约和轮询循环
调度操作均使用 FlyDSL API。

## 测量范围与结果

硬件：8 × MI355X，gfx950。Torch：`2.12.0+rocm7.14.0`。TileRT：`0.1.6.post2`。
位置 3000，稀疏 top-2048，hidden size 6144。所有 GPU 数量下，每个 rank 固定为
**8 个 heads、专家 intermediate size 256**。因此二卡、四卡结果测量的是相同分片
配合较小通信组，**不是完整模型 TP2/TP4 的形状**。

延迟单位为每层微秒：HIP graph 包含 128 次启动，预热重放两次，正式重放九次，取每轮
最慢 rank 延迟的中位数。tag 和通信缓冲区在计时区间外清零。不同实现使用相同权重和
输入，graph 输出必须与 eager 输出逐位一致。

| GPU 数量 | 实现 | S=1 | S=2 | S=4 |
|---:|---|---:|---:|---:|
| 2 | FlyDSL 中的适配版 TileRT 汇编 | 34.59 | 41.10 | 52.95 |
| 2 | 混合版：FlyDSL 最终归约与输出 | 34.54 | 40.82 | 52.54 |
| 2 | 常规 FlyDSL kernel | 33.98 | 39.80 | 53.55 |
| 4 | FlyDSL 中的适配版 TileRT 汇编 | 35.39 | 42.08 | 53.60 |
| 4 | 混合版：FlyDSL 最终归约与输出 | 35.33 | 42.06 | 54.52 |
| 4 | 常规 FlyDSL kernel | 34.30 | 41.51 | 54.42 |
| 8 | 原生 TileRT | 35.85 | 42.90 | 55.93 |
| 8 | FlyDSL 中的 TileRT 汇编 | 36.10 | 42.83 | 55.69 |
| 8 | 混合版：FlyDSL 最终归约与输出 | 36.54 | 43.11 | 55.63 |
| 8 | 常规 FlyDSL kernel | 35.55 | 42.72 | 56.32 |

数据来源为结果目录中的 `corrected-inline{2,4,8}.jsonl`、`corrected-tilert8.jsonl`、
`final-ffn{2,4,8}.jsonl`、`fx-api-candidate{2,4,8}.jsonl` 和
`verified-tilert8.jsonl`。混合版和常规 FlyDSL kernel 仍接近对应汇编或原生基线；
常规 kernel 没有任何一项慢超过 1.6%，部分项目更快。此前常规八卡 S=4 为
61.91 µs（`final-flydsl8.jsonl`）。

**已作废的测量：** 未带 `corrected-`/`final-` 前缀的早期 `matched-*`、`inline*`、
`dispatch*`、`ffn*` 运行使用了错误的 Q-B 转换。TileRT 要求先排列全部非位置编码行，
再排列全部 RoPE 行，并使用 64 行的 scale block。修正后的适配器会重排行并展开原有
的 128 行 scales。独立 golden 检查发现了这个问题；旧结果不能作为相同工作负载下的
性能比较依据。

## 已实现的分段替换

| 文件 | 职责 |
|---|---|
| `tools/asm_bridge/capture.cpp` | 显式启用捕获时保存原生启动的 560 字节参数结构及准确特化名称。skip 模式可发现八卡 ABI，而不在较小通信组上执行八卡 kernel。 |
| `tools/asm_bridge/inline.py` | 反汇编一个特化，将分支重定位到标签，在真正的 `llvm.InlineAsmOp` 中执行；使用前检查 code object 哈希和 ABI。 |
| `tools/asm_bridge/replacements.py` | 用 FlyDSL 表达式替换 CTA 任务分配和线程坐标，保留参数预取及后续仍使用的 block-ID 符号寄存器。 |
| `tools/asm_bridge/peer_count.py` | 适配 2/4 peers，保留七个远端槽位的步长，清零缺席槽位，屏蔽其发送与接收。 |
| `tools/asm_bridge/epilogue.py` | 专家 down 计算结束后从汇编返回 FlyDSL，执行跨卡归约、残差相加和 BF16 输出。 |
| `tools/asm_bridge/trace.py` | 为 S=1/2/4 的生产者、消费者边界记录时间戳。 |

S=1 完整导入版做过逐条检查：FlyDSL code object 的两条参数指针前导指令之后，全部
**6,254 条**原生指令编码逐字节一致。保留原生 LDS 大小与 256 CTAs × 512 threads
启动形状，没有使用运行时 ASM 替换机制。

最终归约替换保留原生 down 阶段位于后方的基本块。ASM 输入声明为可读写，使 LLVM
在返回 FlyDSL 时保留仍存活的参数及线程、block 值。S=1/2/4 的本地 down 输出分别
位于 LDS 字节偏移 7168 / 6144 / 4096，残差位于 27152。替换版本批量轮询 peers，
按 rank 顺序求和。另行测试的 64 字节 packet 翻译略慢，已归档为
`ffn-packets-epilogue.py`，数据见 `packets8.jsonl`。

差分捕获确认的参数偏移：epoch 为 120、392、496、536；attention rank/count 为
176、180；FFN rank/count 为 488、492。发布版原生 wrapper 只支持 1/8 peers；
2/4 GPU 适配版已与独立 golden 比较，不能称为发布版原生 TileRT 的 TP2/TP4 整层基准。

常规 FlyDSL kernel 使用每个 wave 对应一个 peer 的发送方式、每个 wave 一个 peer
指针、按位与计算 CTA 映射，以及按 sample 流水执行的八个 intermediate 的 up/gate
tile。汇编对比指导了最后两项优化：BF16 打包通信，以及每个 router CTA 只处理一个
sample。API 替换后的 S=4 编译结果使用 **216 个 VGPR、94 个 SGPR，
private/scratch 为零**，此前为 224 个 VGPR，恢复版本则使用 256 个 VGPR 且有
spill。已删除不再使用的 up/gate 调度，保留预留的 scratch 布局以维持兼容。

常规 kernel 中最后两处内联汇编已改用 FlyDSL API：无符号 top-k 归约调用
`fx.coop.warp_reduce(..., fx.ReductionOp.MAX, width=64)`，mailbox 重试循环使用
`rocdl.s_nop(0)`。生成的 gfx950 ISA 在 S=4 下仍包含 48 条融合
`v_max_u32_dpp`，源代码和生成的 LLVM IR 中均无内联汇编。八卡前后紧邻测量的
S=1/2/4 分别为 35.44/42.99/56.02 µs 与 35.55/42.72/56.32 µs，变化为
+0.31%、-0.65%、+0.55%。产物位于 `fx-api-baseline8.jsonl`、
`fx-api-candidate8.jsonl` 和 `fx-api-isa8-s4/`。

通信格式改变了数值行为：每个 rank 的 attention 和 FFN partial 先舍入为 BF16，
再按 rank 顺序以 FP32 累加，与 TileRT 的通信精度一致。独立端到端 golden 仍使用
FP32 partial 和原有的 5% 相对 L2 阈值。down 分段 golden 明确建模 BF16 payload
舍入，保留原有严格输出容差。扩展检查首先在两个八卡 S=1 输入上发现了参考精度
不一致；修正后精度约定明确，两类检查均未放宽容差。

## 按分段证据调优

S=4 rank-0 trace 显示此前常规 kernel 的 attention 发布、router 完成和 down 计算
均较晚。八卡未插桩的逐项测量如下：

| 候选 | S=1 | S=2 | S=4 |
|---|---:|---:|---:|
| 通信格式和 router 调整前 | 36.22 | 45.21 | 61.91 |
| BF16 打包通信 | 35.61 | 43.27 | 57.93 |
| 再改为每个 router CTA 一个 sample | — | 42.94 | 56.09 |

数据分别来自 `final-flydsl8.jsonl`、`packed-peer8.jsonl`、`router-split8.jsonl`。
提前计算 shared expert 略微改善 S=2，但使 S=4 变慢；扩大 activation 轮询批次也未
改善 S=4，因此两项实验均已归档，未启用。上方最终矩阵重新测量了清理后的候选。
原生和常规 trace 分别位于 `final-trace-inline-s4`、`verified-trace-flydsl-s4`，
此前常规 trace 位于 `final-trace-flydsl-s4`。诊断 trace 中最后一个 CTA 的里程碑：

| S=4 里程碑 | 此前 FlyDSL | TileRT ASM | 调优后 FlyDSL |
|---|---:|---:|---:|
| Attention 状态发布 | 30.23 | 27.59 | 27.66 |
| Router scores 发布 | 35.12 | 31.83 | 30.65 |
| Routed up/gate 发布 | 50.12 | 44.96 | 45.73 |
| Down 计算完成 | 57.52 | 53.18 | 52.97 |

单位为插桩运行中从启动开始计时的微秒。图表和 CSV 位于结果目录的
`s4-segment-milestones.png`、`s4-segment-milestones.csv`。

## 复现

使用现有编译器 build、当前 worktree 和已安装的 TileRT 包：

```bash
cd /root/FlyDSL-glm5-perf
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-perf:/root/tilert_pkg
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
mkdir -p /root/glm5-perf-results/repro
g++ -shared -fPIC -O2 kernels/mla_moe_layer/tools/asm_bridge/capture.cpp \
  -ldl -o /root/glm5-perf-results/repro/capture.so
export LD_PRELOAD=/root/glm5-perf-results/repro/capture.so

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend tilert_inline --replace ffn --npes 8 --samples 1 2 4 \
  --verify-golden --changing-inputs 5 \
  --output /root/glm5-perf-results/repro/hybrid8.jsonl \
  --asm-artifacts /root/glm5-perf-results/repro/asm
```

再分别使用 `--npes 2`、`--npes 4`。`--replace none` 选择导入汇编；1/8 GPU 下的
`--replace dispatch` 选择 dispatch 替换。八卡下的 `--backend tilert` 选择原生
TileRT。常规 kernel 使用 `--backend flydsl`，不带 `--replace`、`--verify-golden`、
`--changing-inputs`。GPU 基准应串行运行。

导入二进制为 `/root/tilert_isa/b14.co`，SHA256：
`6e0517e924f042d63a8b5a4e13f86696ae5bc4c1eee32765b28456d65bd2eb65`。
生成的汇编与 code objects 保存在本地，未作为源代码加入仓库。更换 TileRT build 后
应重新核对 ABI 和分段边界，再修改哈希检查。

诊断时，可给完整汇编或常规 FlyDSL 基准添加 `--trace --layers 16`。查看某个 rank：

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  /root/glm5-perf-results/repro/asm/s4/rank0/trace.pt
```

S=1/2/4 的导入汇编 trace 均已在八卡运行，并保持与原生输出逐位一致。常规 FlyDSL
trace 也支持 2/4 peers；NP2/S4 已在 `verified-trace-flydsl2-s4` 实测。由于适配器
与插桩共用寄存器，导入汇编的 2/4-peer trace 暂不支持。

时间戳来自 100 MHz realtime counter。插桩会等待内存操作完成并改变调度，性能结论
应使用未插桩的 graph 测量。部分发布标记只覆盖部分 CTAs，或记录重复阶段的最后一次迭代。

混合版在每种 S/GPU 组合下检查五组变化输入：八卡与原生 TileRT 对比，二卡、四卡与
适配后的完整汇编对比，要求逐位一致且各 rank 一致。初始输入还检查独立 golden。
修正输入后的 dispatch 替换也通过了八卡三种 sample 数量的检查，数据见
`corrected-dispatch8.jsonl`。常规 kernel 在 2/4/8 GPU × S=1/2/4 的每种组合下
检查五组变化输入、跨 rank 逐位一致和分段 golden。独立端到端比较在 45 个输入中
通过了 41 个，相对 L2 为 1.49–2.85%；原有的近似并列 routing 检查跳过四个输入
（NP2/S2 一次、NP8/S2 一次、NP8/S4 两次），这些输入仍通过分段和跨 rank 检查。
API-only 后续修改重新运行了相同矩阵，结果不变；日志位于
`fx-api-check-np{2,4,8}-s{1,2,4}.log`。测试 CLI 会在失败时返回非零退出码：

```bash
/opt/venv/bin/python tests/kernels/test_glm5_mla_moe_layer.py \
  --npes 8 -S 4 --pos 3000 --iters 5
```
