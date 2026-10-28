# Turn MI355X hardware limits into a kernel budget

Use this before choosing workgroup geometry, tiles, fusion boundaries, or buffering. Produce a short budget for the current kernel: **visible CUs → resident workgroups and task rounds → instruction and traffic bounds → binding limit → predicted change**. Recompute the affected terms after compilation. Keep it with the experiment; the hardware budget should prune candidates before timing.

## Public specifications and their scope

The following describe a full MI355X (`gfx950`, CDNA 4). Check the runtime-visible device, compute/memory partition, CU mask, compiler version, and measured clocks before applying card-wide limits. AMD's [architecture overview][arch] and [GPU specification table][spec] distinguish per-CU, per-XCD, and whole-device quantities.

| Resource | Published limit or organization | Consequence for kernel design |
|---|---|---|
| Execution | 8 XCDs × 32 active CUs = 256 CUs; wave64, 4 SIMDs/CU, at most 8 waves/SIMD. Check the selected target with the [occupancy calculator][occupancy] | Derive role-local tasks and waves from the actual visible CU count; small grids may leave execution resources idle |
| Registers | 512 KiB vector register file and 12.5 KiB scalar register file per CU. [ISA §3.6][isa]: VGPR allocation in groups of 8, SGPR allocation in groups of 16; regular and accumulator VGPR pools share the allocation budget | Use allocated VGPR/AGPR/SGPR resources, including alignment and reserved registers; a source-level variable count or requested compiler hint is insufficient |
| LDS | 160 KiB/CU; [ISA §3.6.5][isa] specifies 1,280-byte allocation granularity. 64 banks × 4 bytes; [white paper][whitepaper] gives peak reads of 256 bytes/CU/clock | Budget all simultaneously live buffers and metadata; evaluate actual DS access phases and lane addresses |
| Cache | Vector L1: 32 KiB/CU with 128-byte lines. L2: 4 MiB/XCD, 32 MiB total. Infinity Cache: 256 MiB/device | Coalescing, XCD placement, and weight-address reuse affect which level supplies traffic |
| HBM | 288 GiB in the [GPU specification table][spec]; up to 8 TB/s aggregate theoretical bandwidth | Budget capacity from runtime-reported bytes. Match the device/partition scope of HBM traffic; hot-cache logical reads cannot be divided by this bandwidth to predict elapsed time |
| Matrix execution | Approximate full-card dense peaks: BF16/FP16 2.5 PFLOP/s, FP8 5 PFLOP/s, MXFP6/MXFP4 10 PFLOP/s; the [architecture table][arch] separates dense and sparse modes | Select the ceiling for the emitted instruction and both operand formats, then account for padding, inactive columns, and available parallelism |

These are capacity and peak-throughput facts. Instruction latency, sustainable bandwidth, occupancy during a stage, and the number of outstanding requests also depend on the generated program and runtime conditions.

## Calculate residency before tuning registers or worker counts

For the actual compiled entry point and launch, obtain threads/workgroup, static plus dynamic LDS, allocated regular/accumulator VGPRs, SGPRs, private segment, and spill metadata. Reconcile compiler resource remarks and code-object metadata with `hipFuncGetAttributes` and `hipOccupancyMaxActiveBlocksPerMultiprocessor` where the runtime exposes the kernel handle. The occupancy query must use the actual block size and dynamic LDS.

```text
waves_per_workgroup = ceil(threads_per_workgroup / 64)
allocated_LDS = 1280 * ceil((static_LDS + dynamic_LDS) / 1280)
LDS_workgroups_per_CU = floor(163840 / allocated_LDS)  # omit if no LDS

R = compiler-charged regular + accumulator vector allocation per wave
register_waves_per_SIMD <= min(8, floor(512 / max(8, round_up(R, 8))))
workgroups_per_CU <= min(LDS limit, register-placement limit,
                         wave-slot limit, workgroup/barrier-slot limit)
resident_workgroups <= visible_CUs * workgroups_per_CU
```

Here R is in wave64 register rows, each storing 64 32-bit lane values. Respect the compiler's pool layout and [metadata definitions][llvm]; do not add AGPRs again if its reported count already includes them. Include the SGPR limit and reserved state. SIMD placement and workgroup granularity can reduce the bound. Where available, use [LLVM's occupancy calculator][occupancy] from the matching toolchain to inspect each constraint; omitted resources are treated as unconstrained. A lower VGPR count helps residency only if it changes the limiting allocation tier. It can still reduce scratch traffic or alter scheduling without adding a resident workgroup.

**Existing GLM example.** The [rows10 kernel](../../tune/references/cases.md#shape-cliff) used 113,196/113,224 LDS bytes per workgroup, both rounding to 113,920 bytes. `floor(163840 / 113920) = 1`: LDS alone caps it at one workgroup/CU. Reducing VGPRs from 256 to 215/216 cannot admit a second workgroup while that LDS footprint remains. Two workgroups would require at most 81,920 allocated LDS bytes each, plus feasible register and wave budgets. This rules out an unsupported occupancy explanation for the measured speedup; it does not rule out other benefits of the code change.

For persistent kernels, compare each role's tasks with its resident workers and derive its last task round. Then establish [forward progress](persistent.md), including whether consumers can prevent needed producers from running. Increasing the grid beyond the residency bound needs a scheduling argument.

## Match instruction and memory bounds to the critical path

Read the emitted MFMA opcode, operand formats, tile shape, and accumulation dependencies. [ISA §7.1][isa] lists `16x16x128_F8F6F4`, including its scaled form, at 16 or 32 cycles; either operand being F8 selects the larger value. The `32x32x64` form uses 32 or 64 cycles on the same rule. An FP8×FP4 path therefore needs the mixed-format ceiling, rather than the pure-MXFP4 peak. Preserve the instruction's operand mapping and the dependency/wait rules in ISA §§4.4–4.5 when changing the schedule.

Separate useful arithmetic from issued padded/masked work. Model memory traffic at the level that serves it: registers/LDS, L1/L2, Infinity Cache, HBM, or a peer link. For ordinary dword LDS addresses, start with `bank = (byte_address / 4) % 64`, then apply the actual instruction's lane groups, access phases, and broadcast behavior. For global loads, derive per-wave address spans and touched cache lines before changing vector width or packing. Use the [layout cases](../../tune/references/att-cases.md#att-layout) to connect these mappings to measurements.

An optimistic lower bound for work that can overlap is:

```text
time >= max(issued_work / matching_compute_ceiling,
            HBM_bytes / HBM_bandwidth,
            LDS_bytes / LDS_bandwidth,
            serial_dependency_path)
```

Add costs that the dependency graph forces to execute serially. Use matching clock, datatype, CU/XCD scope, and traffic units; cache and peer-link bounds can be added where relevant. Low concurrency may lack enough independent waves, accumulators, or ready memory requests to approach a bandwidth/compute ceiling. Estimate what the proposed change removes from the critical path, including its added indexing, waits, and live ranges. Avoid selecting a generic target percentage of card peak.

Turn the binding limit into one prediction: an allocation tier changes, a task round disappears, fewer instructions/transactions execute, or a dependency becomes ready earlier. Choose [PMC/ATT evidence](../../tune/references/profiling.md) to distinguish that prediction, then retain the existing numerical, progress, and uninstrumented timing checks. Reuse the budget until the workload, code, or hardware configuration changes it.

## Sources

Public sources checked 2026-10-13. The ISA reference is dated 5 August 2025. Check the target project's compiler and runtime when applying allocation rules.

- [AMD MI350 Series microarchitecture][arch]: topology, hierarchy, dense/sparse peaks, partitioning.
- [AMD GPU specifications][spec]: register files and per-CU/cache capacities.
- [CDNA 4 ISA reference][isa]: §§3.6, 4.4–4.5, 7.1, and 11 for allocation, dependencies, MFMA, and LDS.
- [CDNA 4 architecture white paper][whitepaper]: CU memory pipeline and LDS bandwidth.
- [LLVM AMDGPU backend documentation][llvm]: target features, code-object metadata, kernel descriptors, and resource attributes.
- [LLVM occupancy calculator][occupancy]: target-specific allocation limits and per-constraint occupancy.

[arch]: https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html
[spec]: https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
[isa]: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
[whitepaper]: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
[llvm]: https://llvm.org/docs/AMDGPUUsage.html
[occupancy]: https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
