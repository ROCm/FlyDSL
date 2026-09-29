# MonoKernel

`kernels.monokernel` is the shared home for resident, fused decoder-layer
kernels. Model ownership is explicit:

- `glm/` contains GLM-5 scheduling, indexed attention, wrappers, and golden
  references.
- `k3/` contains Kimi-K3 MLA/KDA state handling, AttnRes, latent-MoE logic,
  staged baselines, and benchmark tools.

Reusable contracts and primitives stay at this package root. `config.py`,
`layout.py`, `ops.py`, `packing.py`, `reference.py`, `runtime.py`, and
`weights.py` define shared geometry, layouts, device operations, packing, host
runtime, and weight containers. `gemm_a16w16.py`, `mxfp8_linear.py`, and
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
