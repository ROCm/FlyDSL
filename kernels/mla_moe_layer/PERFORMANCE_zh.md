# Indexed sparse MLA + MoE block

[English](PERFORMANCE.md)

本目录包含一个固定 GLM-5 MLA + MoE 分片的 FlyDSL 实现。生产路径通过
`Glm5IndexedMlaMoeBlock` 提供，并通过 FlyDSL 与 ROCDL API 生成。

该 block 接收调用方准备好的 sparse-attention indices。GLM-5 的 78 层生产调度中，
57 个 MoE 层使用本实现覆盖的对称 8-head/rank reuse 拓扑。另外 18 个 MoE 层使用非对称
refresh 拓扑：rank 1-7 运行 padded 10-head whole-layer 路径，rank 0 参加 attention
reduction 后再调用 standalone MoE。该 refresh 拓扑和 standalone 入口尚不在此 block
内。生成新索引的 selector/indexer/top-2048/broadcast 独立链路也不在本 block 内，且
不属于 MoE 计算。

在已支持的拓扑中，本 block 完整包含 MoE router、expert top-8、shared/routed experts、
up/gate activation、down projection、expert weighting 和 TP peer reduction。refresh
路径使用相同的 MoE 数学阶段，但其生产装载方式和 rank 布局仍需单独集成。

`native_baseline.py` 仍可选择性依赖 TileRT，用于把同一组生成权重转换给已发布的
TileRT wrapper，从而直接比较两个实现。FlyDSL 执行路径不会导入该适配器。

外部集成与测量基线参考
[InferenceX commit 8ac98344](https://github.com/SemiAnalysisAI/InferenceX/commit/8ac98344b038a3f2da20a565fe9b974772a67ef9)，
该提交固定了对比所用的 GLM-5.3 MI355X TileRT 环境。

## 固定分片与计算模式

每个 rank 固定使用 8 个 attention heads、专家 intermediate size 256、hidden size
6144、256 个 routed experts、1 个 shared expert 和 top-8 routing。因此二卡、四卡
结果测量的是相同固定分片配合较小通信组，不是完整模型 TP2 或 TP4 的形状。

公开的 `MoeMode` 包含：

| 模式 | 专家 activation | 专家 weight | Up/gate 到 down 的交接格式 |
|---|---|---|---|
| `w8a8` | 每 128 个元素动态量化 FP8 E4M3 | block-scaled FP8 E4M3 | FP8 |
| `w8a16` | BF16 | block-scaled FP8 E4M3 | BF16 |
| `a16w4` | BF16 | 每 1x32 使用 E8M0 scale 的 MXFP4 | BF16 |
| `a8w4` | 每 1x32 使用 E8M0 scale 的 MXFP8 E4M3 | 每 1x32 使用 E8M0 scale 的 MXFP4 | MXFP8 |

所有模式的 attention weights 都保持 block-scaled FP8。当前支持 sample count 1、2、4、8
以及 peer count 1、2、4、8。host wrapper 会在分配 GPU buffer 前验证完整固定分片约束。

## 代码结构

| 文件 | 职责 |
|---|---|
| `config.py` | 固定维度、公开计算模式和 host 参数验证。 |
| `../common/hip_ipc.py` | 可复用的 HIP IPC handle 与 allocation host wrapper。 |
| `../common/mx_formats.py` | 可供其他 MoE wrapper 复用的 Torch MXFP4/MXFP8 量化与反量化。 |
| `packing.py` | FP8、BF16 和 MXFP4 matrix 的 MFMA weight packing。 |
| `runtime.py` | 自有 symmetric HIP IPC buffer，以及远端 handle 的确定性清理。 |
| `indexed_mla_moe_kernel.py` | FlyDSL kernel 调度、通信、MLA、routing 和专家计算。 |
| `layer.py` | 公开 host wrapper、scratch 分配、启动参数、trace 和生命周期。 |
| `reference.py` | 独立 Torch 分段与端到端计算。 |
| `native_baseline.py` | 可选的同权重 TileRT 对比适配器。 |

kernel 使用 FlyDSL 操作实现 wave reduction、硬件数学指令、mailbox polling、buffer
访问和 MFMA。每个 rank 的 peer payload 先舍入为 BF16，再按 rank 顺序累加，因此所有
rank 得到逐位一致的 hidden state 和 routing 结果。

attention 与 FFN reduction 各自使用两个按 epoch 奇偶选择的 symmetric-buffer slot。
这样在一个 graph 连续捕获多层时，较快 rank 不会覆盖仍由较慢 peer 读取的 epoch `k` 数据。

`Glm5IndexedMlaMoeBlock` 持有远端 HIP IPC mappings。应在最后一次 rank barrier 后调用
`close()`，也可以把该对象作为 context manager 使用。

## 正确性状态

最终 peer slot 修复没有改动算术路径；固定 8-head reuse 算术路径的 `w8a8` 和
`w8a16` 均已通过 2/4/8 GPU x S=1/2/4 的完整矩阵。九种配置各运行五组变化输入，
并检查：

- 分段输出与独立 Torch 计算一致；
- 所有 rank 的最终输出逐位一致；
- 最终 down projection 和 BF16 peer reduction；
- 输出有限，且 HIP graph replay 稳定。

最终源码随后又通过 TP8 下每 step 1、75 和 128 次 launch 的 eager-versus-replay 精确
检查，验证了常规单层实例和 75-launch 压力序列中的 slot 轮换；该检查不能代替尚未实现
的非对称 refresh 拓扑。

独立 FP32 端到端检查通过 41 个输入，相对 L2 为 1.49-2.85%。另有四个输入沿用已有的
近似并列 routing 跳过规则，因为 attention 的一个 BF16 ulp 差异改变了专家选择；这些
输入仍通过分段检查和跨 rank 逐位一致检查。现有容差没有放宽。一个 NP2/S4 `w8a16`
intermediate 使用已有的一个 BF16 ulp 上限，而其最终 down/output 仍精确匹配。

S=8 也在 1、2、4、8 GPU 上分别以一组新输入通过 `w8a8` 和 `w8a16` 的完整分段检查。
更大的 peer payload 使用两个 64-lane send batches，仍保持所有 rank 输出逐位一致。
NP4 `w8a16` 的 normalized expert input 有一个元素与独立 reduction 相差一个 BF16 ulp，
处于已有 BF16 handoff 上限内。

新增的 `a16w4` 和 `a8w4` 在单卡 S=1、S=8，以及双卡和八卡 S=8 上通过完整分段
检查。检查覆盖 packed MXFP4 weight、每 1x32 E8M0 scale、A8W4 activation 量化、
最终 BF16 peer reduction，以及所有 rank 最终输出逐位一致。TP8/S8 的独立端到端
相对 L2 分别为 `a16w4` 0.430%、`a8w4` 2.87%。没有放宽现有容差。

TP1/S1 下使用相同权重直接对比 TileRT wrapper，结果为：

| 模式 | 最大绝对误差 | 相对 L2 |
|---|---:|---:|
| `w8a8` | 0.1171875 | 2.263% |
| `w8a16` | 0.03125 | 0.357% |

## 性能状态

下列配对结果比较的是对称 reuse-selection 拓扑。每个 HIP graph 包含 128 次 layer
launch，并使用 5 个 eager warmup step、2 次舍弃的 graph 计时和 9 次正式 graph
replay，结果取每轮最慢 rank 的中位数。硬件为 8 x MI355X（gfx950），位置 3000，
seed 1234，sparse top-2048。Delta 越低表示 FlyDSL 越快。

| 模式 | S | FlyDSL | TileRT | Delta |
|---|---:|---:|---:|---:|
| `w8a8` | 1 | 35.179 us | 35.995 us | -2.27% |
| `w8a8` | 2 | 42.461 us | 42.712 us | -0.59% |
| `w8a8` | 4 | 56.217 us | 55.361 us | +1.55% |
| `w8a16` | 1 | 35.747 us | 36.305 us | -1.54% |
| `w8a16` | 2 | 43.414 us | 43.448 us | -0.08% |
| `w8a16` | 4 | 57.246 us | 55.563 us | +3.03% |

FlyDSL 在六项已测配置中的四项更快。两个 S=4 配置与 TileRT 的差距均不超过 3.03%，
因此 W8A8 和 W8A16 在已测范围内接近性能持平，但不能表述为所有配置都更快。同一源码
还通过了每 step 1 次和 75 次 launch 的 TP8 重复 graph，直接覆盖单层使用方式和
75-launch 压力序列，未再发生 peer slot 覆盖。

当前已发布的 TileRT 对比适配器只接受 `w8a8`、`w8a16`、1 或 8 peers，以及
S=1/2/4。它没有 `a16w4`、`a8w4` 或 S=8 的有效同权重整层基线，因此这些 FlyDSL
扩展已有正确性覆盖，但这里不作 TileRT 性能结论。TP2 和 TP4 结果使用固定的每-rank
分片，只表示 scaling 检查，不是完整模型 TP2/TP4 对比。

分段 trace 指导了两项保留的调度修改：BF16 packed peer exchange，以及每个 router
CTA 只处理一个 sample。此前 S=4 调度中，最后一个插桩 CTA 到达 attention 发布、
router 发布、routed up/gate 发布和 down 完成的时间分别为 30.23、35.12、50.12、
57.52 us；修改后为 27.66、30.65、45.73、52.97 us。对比图位于
`/root/glm5-perf-results/s4-segment-milestones.png`。

## 复现

使用现有 FlyDSL compiler build 和本 worktree：

```bash
cd /root/FlyDSL-glm5-perf
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-perf:/root/tilert_pkg
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

/opt/venv/bin/python tests/kernels/test_glm5_indexed_mla_moe.py \
  --npes 8 -S 8 --pos 3000 --iters 1 --moe-mode a8w4

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode w8a8 --npes 8 --samples 1 2 4 \
  --layers 128 --repeats 9 --seed 1234 --pos 3000
```

正确性命令应替换其他模式和 peer count 重复运行。性能命令还应使用
`--moe-mode w8a16` 重复，再用 `--backend tilert` 得到配对基线。GPU 任务应串行执行。

如需与已发布实现直接比较，保留 `/root/tilert_pkg` 在 `PYTHONPATH` 中，并把
`--backend flydsl` 改为 `--backend tilert`。原生 wrapper 只支持 `w8a8`/`w8a16`、
1 或 8 peers，以及 sample count 1/2/4。S=8 和 MXFP4 模式是本 harness 中的
FlyDSL 独有扩展。

FlyDSL benchmark 可添加 `--trace --layers 16 --trace-dir <directory>` 记录分段时间戳，
然后查看某个 rank：

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  <directory>/w8a8-s4/rank0/trace.pt
```

trace 插桩会等待内存操作完成并改变调度。延迟比较应使用未插桩的 graph 测量。
