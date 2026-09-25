# Shared/reuse MLA + MoE kernel

[English](PERFORMANCE.md)

本目录包含一个固定 GLM-5 MLA + MoE 分片的 FlyDSL 实现。生产路径通过
`SharedReuseMlaMoeLayer` 提供，并由 `build_shared_reuse_kernel` 生成。该路径不导入
TileRT kernel 主体，也不嵌入汇编。原有的汇编捕获、改写和启动桥接代码已全部删除。

`native_baseline.py` 仍可选择性依赖 TileRT，用于把同一组生成权重转换给已发布的
TileRT wrapper，从而直接比较两个实现。FlyDSL 执行路径不会导入该适配器。

## 固定分片与计算模式

每个 rank 固定使用 8 个 attention heads、专家 intermediate size 256、hidden size
6144、256 个 routed experts、1 个 shared expert 和 top-8 routing。因此二卡、四卡
结果测量的是相同固定分片配合较小通信组，不是完整模型 TP2 或 TP4 的形状。

公开的 `MoeMode` 包含：

| 模式 | 专家 activation | 专家 weight | Up/gate 到 down 的交接格式 |
|---|---|---|---|
| `w8a8` | 动态 FP8 E4M3 | block-scaled FP8 E4M3 | FP8 |
| `w8a16` | BF16 | block-scaled FP8 E4M3 | BF16 |

两种模式的 attention weights 都保持 block-scaled FP8。当前支持 sample count 1、2、4
以及 peer count 1、2、4、8。host wrapper 会在分配 GPU buffer 前验证完整固定分片约束。

## 代码结构

| 文件 | 职责 |
|---|---|
| `config.py` | 固定维度、公开计算模式和 host 参数验证。 |
| `packing.py` | FP8 和 BF16 matrix 共用的 MFMA weight packing。 |
| `runtime.py` | 自有 symmetric HIP IPC buffer，以及远端 handle 的确定性清理。 |
| `shared_reuse_moe_kernel.py` | FlyDSL kernel 调度、通信、MLA、routing 和专家计算。 |
| `layer.py` | 公开 host wrapper、scratch 分配、启动参数、trace 和生命周期。 |
| `reference.py` | 独立 Torch 分段与端到端计算。 |
| `native_baseline.py` | 可选的同权重 TileRT 对比适配器。 |

kernel 使用 FlyDSL 操作实现 wave reduction、硬件数学指令、mailbox polling、buffer
访问和 MFMA。每个 rank 的 peer payload 先舍入为 BF16，再按 rank 顺序累加，因此所有
rank 得到逐位一致的 hidden state 和 routing 结果。

`SharedReuseMlaMoeLayer` 持有远端 HIP IPC mappings。应在最后一次 rank barrier 后调用
`close()`，也可以把该对象作为 context manager 使用。

## 正确性状态

`w8a8` 和 `w8a16` 均通过 2/4/8 GPU x S=1/2/4 的完整矩阵。九种配置各运行五组变化
输入，并检查：

- 分段输出与独立 Torch 计算一致；
- 所有 rank 的最终输出逐位一致；
- 最终 down projection 和 BF16 peer reduction；
- 输出有限，且 HIP graph replay 稳定。

独立 FP32 端到端检查通过 41 个输入，相对 L2 为 1.49-2.85%。另有四个输入沿用已有的
近似并列 routing 跳过规则，因为 attention 的一个 BF16 ulp 差异改变了专家选择；这些
输入仍通过分段检查和跨 rank 逐位一致检查。现有容差没有放宽。一个 NP2/S4 `w8a16`
intermediate 使用已有的一个 BF16 ulp 上限，而其最终 down/output 仍精确匹配。

TP1/S1 下使用相同权重直接对比 TileRT wrapper，结果为：

| 模式 | 最大绝对误差 | 相对 L2 |
|---|---:|---:|
| `w8a8` | 0.1171875 | 2.263% |
| `w8a16` | 0.03125 | 0.357% |

## 性能状态

下表为既有 W8A8 HIP graph 测量：每个 graph 含 128 次 layer launch，预热两次，正式
测量九次，并取各轮最慢 rank 的中位数。硬件为 8 x MI355X（gfx950），位置 3000，
sparse top-2048。

| GPU 数量 | 后端 | S=1 | S=2 | S=4 |
|---:|---|---:|---:|---:|
| 2 | FlyDSL | 33.98 us | 39.80 us | 53.55 us |
| 4 | FlyDSL | 34.30 us | 41.51 us | 54.42 us |
| 8 | FlyDSL | 35.55 us | 42.72 us | 56.32 us |
| 8 | TileRT | 35.85 us | 42.90 us | 55.93 us |

FlyDSL 在已测八卡 S=1、S=2 中更快，在 S=4 中慢 0.7%，所以尚未在所有配置上超过
TileRT。

一次使用 8 层、1 次正式 replay 的 TP1/S1 短 smoke 测量中，`w8a8` 为 33.840 us，
TileRT 为 33.520 us；`w8a16` 为 34.735 us，TileRT 为 32.895 us。这些短运行用于验证
benchmark 路径，不属于可发布的性能数据。

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

/opt/venv/bin/python tests/kernels/test_shared_reuse_mla_moe_layer.py \
  --npes 8 -S 4 --pos 3000 --iters 5 --moe-mode w8a8

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode w8a8 --npes 8 --samples 1 2 4 \
  --layers 128 --repeats 9 \
  --output /root/glm5-perf-results/flydsl-w8a8-tp8.jsonl
```

正确性命令应分别使用 `--moe-mode w8a16`、peer count 2/4/8 和 sample count 1/2/4
重复运行。GPU 任务应串行执行。

如需与已发布实现直接比较，保留 `/root/tilert_pkg` 在 `PYTHONPATH` 中，并把
`--backend flydsl` 改为 `--backend tilert`。原生 wrapper 只支持 1 或 8 peers，以及
sample count 1/2/4。

FlyDSL benchmark 可添加 `--trace --layers 16 --trace-dir <directory>` 记录分段时间戳，
然后查看某个 rank：

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  <directory>/w8a8-s4/rank0/trace.pt
```

trace 插桩会等待内存操作完成并改变调度。延迟比较应使用未插桩的 graph 测量。
