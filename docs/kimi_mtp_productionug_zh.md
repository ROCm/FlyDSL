# Kimi MTP：生产 UG 与 compact down 的受控组合

生产版 UG 替换通过正确性检查，完整层单 seed 筛选为 **148.461 μs**。比 compact UG 两阶段版本的 150.139 μs 快约 1.68 μs，仍慢于 public pipeline 的约 143.577 μs，未达到约 129.18 μs 的目标。没有接入默认实现。

该实现使用已调优的 `kimi_k3_mxfp4_gemm1`，S4 配置为 A16W4、TILE_N64、TILE_K256、k_wave2、XCD4、256 threads。UG 输出保留 sorted-position BF16 布局。224 个 512-thread down CTA 通过 expert/sample 到 sorted-position 的 LDS 映射直接读取，无额外 reorder kernel；保持 down/combine/TP 算法和 BF16 边界。两次 launch 都计入完整层 GPU events。

诊断用 `diagnostic_mid()` 在计时与 graph 之外 gather 实际 UG 结果，仅转换布局，不重算参考数学。补齐了未在本轮运行的独立 helper 对此接口的调用；源码清单保留 helper 修改前后的 hashes。

三份产物经 compile-only、HIP module load 及实际 block size 的 occupancy 检查：生产 UG 为 152 VGPR、32 KiB LDS、3 CTA/CU；TP1/TP8 down 都为 72 VGPR、40,192 B LDS、3 CTA/CU。均无 register spill 或 private scratch。

TP1 常规与 varied-scale 检查覆盖 expert 数 16/17/47/48/63/64、零系数、变更 routing/payload、每例 8 次 16-layer graph 重放。TP8 全层检查覆盖独立 FP32 reference、routing、rank equality、mid/output、TP/tail 与 true-MTP states。阈值未放宽。

性能测量为 node50、TP8、S4、seed1234、16-layer graph、50 repeats、无插桩 GPU events，按每次最慢 rank 取中位数：148.460999 μs。本候选未做多 seed 确认或 ATT/PMC，因筛选已慢于基线；不能将 1.68 μs 的差异解释为某个阶段的精确耗时变化。

证据位于 [productionug 结果目录](../../results/kimi_mtp_productionug_20260929/)。归档已下载，838 项 manifest 全部验证，SHA256 为 `caa1bc2bd9295246d819ca2b94eea641672e8f93dda5d8618e4557955a4e9ec8`。后续转向 attention 的 MTP 递归状态驻留与 CTA 分工，检验能否减少连续 token 间的 global handoff。
