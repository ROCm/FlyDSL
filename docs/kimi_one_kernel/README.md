# Kimi-K3 one-kernel 当前版本

本分支 `codex/kimi-k3_one_kernel` 基于 session
`01a10f4c-f6ac-7563-ad99-faf460c01281` 保留的
`opt254_swizzled_input_lds_s4`，该版本已在提交
`9cb1d40e0c3c344007852be1d9aa92b5b69a4ce3` 中原样保存。
当前版本为 Kimi 的 dspark 7 场景扩展到 seq 8，即当前 token 加七个投机 token。

入口为 [KimiK3MonoKernel](../../kernels/kimi_k3_monokernel/op.py)，
支持 TP8、batch 1–8、seq 1–8 的完整 KDA MoE layer 单次 launch，最多 64 tokens。
`samples = batch * seq`；`seq_len` 指定每个独立状态链的连续 token 数，
`seq > 1` 使用 MTP 状态快照，每条链有 `seq + 1` 个索引。
`KimiK3CompileConfig(path="auto")` 在编译期选择配置；seq 5–8 使用 general 路径，
seq 1–4 保留 Opt254 配置。非默认配置需要单独验证精度及完整网格驻留。
完整精度覆盖和性能计时针对 layer 1；layer 0 dense FFN 不在范围内。

## seq 8 实现

MXFP8 激活 scale 的读写支持第二个 32-row tile，完整 kernel 的 mailbox 和
host buffer 支持 64 tokens。MTP 状态索引覆盖完整九个快照，batch 之间独立。
新形状的输入 staging/row groups 根据实际编译资源选择，避免 polling kernel
分配私有内存；原有形状的编译配置保留。输入 GEMM 的 K 分区同时按新增总 token 数
匹配原生 FP32 累加顺序，避免 BF16 舍入误差沿 KDA 状态链放大；
[逐位复现记录](../../experiments/kimi_one_kernel/seq8/native_input_orders.json)覆盖新增形状的所有总 token 数。

完整 kernel 不再构造未使用的 staged-tail launcher，避免 64-token 初始化时除零。
检查工具按完整 kernel 的实际权重格式构造 golden，仍使用原有量化语义及精度阈值。
`--staged` 比较路径的 fused tail 仍限于 32 tokens。

严格回放 oracle 已迁入
[kernel tools](../../kernels/kimi_k3_monokernel/tools/full_replay.py)，数值计算及阈值沿用 Opt254；
仅将 slot 模式扩展到 seq 8。回放覆盖线性、排列、单个负 slot、全负 slot、
单 slot 别名和交替别名，每类三轮变化数据，并检查新 epoch 重放、未写 slot 和跨 rank 一致性。

## 验证与性能

当前 seq 8 的编译、驻留、严格回放和 GPU-event 结果保存在
[验证记录](../../experiments/kimi_one_kernel/seq8/validation.json)及
[64 个形状的编译记录](../../experiments/kimi_one_kernel/seq8/compiled_shapes.json)。

64 个默认形状全部通过编译及八卡完整网格驻留检查，private/VGPR 显存 spill 均为零。
其中 seq 1–4 的 32 个二进制与 Opt254 完全一致。
新增 32 个形状 × 3 seeds 全部通过严格回放：96 cases、768 rank checks、13824 records；
另有 B1/S1、B1/S4、B2/S4、B8/S4 的 seed1234 回归，共计 800 rank checks、14376 records。
原有数值阈值保持不变；端到端诊断与阶段回放误差在摘要中分别保留。

B1/S8 的三 seed 完整层耗时为 261.160240、259.577692、257.075161 μs，
中位数 **259.577692 μs**。

Opt254 的历史数据为 B1/S1 71.826909 μs、B1/S4 114.204876 μs。
计时协议为 MI355 节点46、TP8、完整 layer 1、无插桩 GPU events，
16-layer graph、50 repeats；每个 repeat 取最慢 rank，随后取中位数。
原 seq 1–4 的历史六 seed 严格回放覆盖 192 cases、1536 rank checks、26496 records。
这些原始结果与此次新增验证分开保留。

## 运行与复核

在配置好 gfx950 FlyDSL/PyTorch 的八卡环境，确认设备空闲后先编译并检查驻留：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
FLYDSL_RUNTIME_CACHE_DIR=/tmp/kimi_seq8_b1_s8_cache \
python experiments/kimi_one_kernel/seq8/compile_shape.py \
  --batch 1 --seq 8 --out /tmp/kimi_seq8_b1_s8_resources
python experiments/kimi_one_kernel/seq8/check_shape_occupancy.py \
  /tmp/kimi_seq8_b1_s8_resources
```

每个形状使用独立的空缓存目录。重新编译后的资源和二进制可能受工具链影响，
历史驻留证据不能替代新二进制的检查。通过后，使用同一编译缓存运行：

```bash
FLYDSL_RUNTIME_CACHE_DIR=/tmp/kimi_seq8_b1_s8_cache \
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --batch 1 --seq 8 --layer-idx 1 --seed 1234 --check --full-replay-check \
  --bench --layers 16 --repeats 50 --output /tmp/kimi_b1_s8.json
```

`--mtp --samples 8` 等价于 `--batch 1 --seq 8`。严格回放无需额外的 oracle PYTHONPATH。
新增形状验证 seeds 为 1234、2025、3141。

仓库内源码、配置和历史归档的 CPU 复核命令：

```bash
PYTHONPATH=. python3 tests/kernels/test_kimi_k3_shapes.py
python3 experiments/kimi_one_kernel/verify_archive.py
```

完整原始回放、GPU-event 数据、编译二进制和冻结源码在工作区的
`../results/kimi_seq8_20261008/`（相对于此仓库根目录），46 容器对应目录为
`/tmp/flydsl-kimi-seq8-20261008/`。压缩包 SHA-256 记录在验证摘要中。
解压后可独立复核原始证据：

```bash
python3 experiments/kimi_one_kernel/seq8/audit_evidence.py /path/to/extracted/evidence \
  --opt254-shapes experiments/kimi_one_kernel/opt254/compiled_shapes.json \
  --out /tmp/kimi_seq8_audit
```

## 历史归档

[Opt254 源码 manifest](../../experiments/kimi_one_kernel/opt254/source_manifest.json)、
[选择记录](../../experiments/kimi_one_kernel/opt254/CURRENT_SELECTION.json)、
[完整功能覆盖](../../experiments/kimi_one_kernel/opt254/opt254_inheritance_validation.json)及
[性能验证](../../experiments/kimi_one_kernel/opt254/opt254_validation.json)保留原始记录。
其中 `repository_checkout_changed: false` 等字段描述实验采用时的状态。
Opt254 原始证据位于 `../results/kimi_shapes_20261006/`；46 容器对应目录为
`/tmp/flydsl-kimi-shapes-20261006/`。

[2026-09-29 历史归档](archive_20260929.md)中的 92 个补丁仍以
`8137256d25ccea61eeca4f43d26fc3aad280cb1d` 为基底。
校验器分别通过 Git 中的原始提交校验历史基底与 Opt254，再检查当前 seq 8 源码。
