# MonoKernel

`kernels.monokernel` is the shared home for resident, fused decoder-layer
kernels. Model ownership is explicit:

- `glm/` contains GLM-5 scheduling, indexed attention, wrappers, and golden
  references.
- `k3/` contains Kimi-K3 MLA/KDA state handling, AttnRes, latent-MoE logic,
  staged baselines, and benchmark tools.
- `dsv4/` contains the DeepSeek-V4 decode layer: CSA / HCA sparse attention
  with its compressors and indexer, mHC, MoE, and its checkpoint loader.
  `kernel.py` builds the one launch from per-stage modules (`hc`, `qkv`,
  `indexer`, `attention`, `ffn`); `common.py` binds the shared helpers to it
  and adds the DSV4-only ones, and `plan.py` holds the host-side task counts
  and mailbox layout.

Reusable contracts and primitives stay at this package root. `config.py`,
`layout.py`, `ops.py`, `packing.py`, `reference.py`, `runtime.py`, and
`weights.py` define shared geometry, layouts, device operations, packing, host
runtime, and weight containers. `helpers.py` holds the device helpers that
GLM, Kimi-K3 MLA and DeepSeek-V4 share:

- tagged-pair mailboxes, `poll` and timeline `stamp`;
- block reductions;
- the MFMA GEMV units with `mma_units` / `run_units` / `reduce_rows`;
- RMSNorm and activation staging;
- the TP `peer_reduce` and task placement.

Each is a plain function whose first argument is the kernel's `LaunchState`,
the values of one launch. `LaunchState` also exposes every helper bound to
it (`st.put`, `st.poll`, ...), and the helpers call one another through it,
so a kernel that installs its own variant (for example
`st.poll = partial(helpers.poll, st, retry_spin_pause=True)`) changes it for
every helper. Variants are keyword arguments or separate functions
(`unit_mxfp4_atom`, `unit_mxfp4_rows`), each a build-time choice, so each
kernel compiles exactly its own. FlyDSL's JIT cache keys a
kernel only on its own directory's sources, so the kernels also capture
`SHARED_SOURCE_KEY`, a digest of `helpers.py` and `ops.py`, so that edits to
either still recompile them. `gemm_a16w16.py`, `mxfp8_linear.py`, and
`symmetric_allreduce.py` provide model-independent kernels used by the staged
paths.

Keep model-specific schedules next to their model. In particular, Kimi-K3's
ordered KDA snapshot chain and MTP recurrence must remain in `k3/`; extracting
compiler-sensitive hot-loop code solely for API symmetry can change generated
GPU code and must be justified by performance measurements.

The resident-grid schedule, tagged-pair mailbox protocol, and phase-overlaid
LDS arena follow the TileRT/GLM MonoKernel design lineage. The comparison
reference used during this work is:

https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9

Model semantics and the Kimi-K3 ordered state pipeline are implemented in
FlyDSL and remain model-specific.
