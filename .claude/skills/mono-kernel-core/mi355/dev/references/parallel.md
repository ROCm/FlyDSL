# Choose parallelization for the fused subgraph

Fusion changes launch overhead, intermediate-memory traffic, communication readiness, and resource residency. Reconsider tensor parallelism (TP), expert parallelism (EP), and context parallelism (CP) for the fused implementation. Preserve model arithmetic, global requests, required state updates, and user constraints. The EP guidance below describes design choices; it does not report measured EP speedups.

## Define data and output ownership

For each stage record actual group members, split axes/replicas, weight/KV/state owners, global→local indices, partial/final output owners and required transfers. Attention may use TP/CP while MoE uses EP with intra-expert TP. Include activation redistribution between groups; multiplying degree labels does not establish GPU count.

Global requests, query tokens and dispatched token-expert slots are different counts. Distinguish indexer CP, attention CP, physical KV sharding and intra-GPU context tiling. Bind weights, metadata, workspace, generations, and fused partition to the current plan.

| Strategy | Potential saving | Likely new cost and first control |
|---|---|---|
| TP | Less local tensor/expert math | Small local matrices and partial reduction; start with external collective, then test tile readiness |
| EP | Fuller expert-local matrices; fewer intra-expert reductions in some plans | Dispatch/return, small messages and expert hotspots; fix routes and measure owner work plus complete token round trip |
| CP | Less per-rank context scan/score/attention | Query/partial/candidate exchange, merge and reduced local tasks; compare the same algorithm with CP off/on |

## TP: connect local production to the actual reduction

Identify every required partial and its order; TP need not be all-reduce. Test local output and reconstructed final output before fusing communication. Tile-ready reduction is useful only if earlier partials advance completion. Peer waits must leave a path to produce unclaimed local outputs and allow transport to run. An inefficient small-matrix implementation can cost more than communication overlap saves; use fixed-plan N/P/F.

## EP: implement a complete token round trip

Start with `route → dispatch → owner-local UG/activation/down → return → original-token combine`, keeping communication external until each boundary works. Carry request/token, original route slot, global expert, destination/local expert and route weight. Apply weights, shared and residual contributions once at declared endpoints; retain nested TP where required.

Estimate grouping from received slots and distinct experts **on each owner after dispatch**. A global union copied from TP can mispredict reuse. Track per-rank/expert/peer live counts, capacity and tail. Many idle ranks cannot compensate for an overloaded expert owner. Empty peers must publish required completion; never silently drop overflow tokens.

Test dispatched records, local math, returns and final combine, including skew, repeated experts, zero messages and delayed peers. Fuse dispatch→compute or down→return only with explicit visibility, buffer reuse and transport progress. A resident kernel cannot occupy every resource needed by the communication kernel it awaits.

## CP: preserve complete score or attention-partial semantics

Measure live-context distribution, queries per request, capacity, page layout and refresh/reuse layers. Sparse selection may reduce later attention work enough that additional attention CP increases latency.

For indexer CP, establish selection unit, complete score formula, head aggregation, top-k, masks and total score/ID order. Historical M3 selected 16 blocks of 128 tokens **per head**; GLM selected 2048 tokens **after 32-head aggregation**. Similar final key counts do not make these equivalent. Selecting per-head candidates before a required weighted-sum score is generally wrong.

With disjoint context shards and one total order, global top-k lies in the union of shard-local top-k of complete scores. Budget roughly P*k candidates and merge cost; define NaNs, ties and fewer-than-k behavior. Use bounded tile/radix/histogram/compaction or candidate selectors: one million 32-bit keys require 4 MB, too large for a single workgroup's LDS.

For standard softmax attention retain local max m_p, exponential sum l_p and unnormalized weighted sum n_p. Merge with `m=max(m_p)`, `a_p=exp(m_p-m)`, then `sum(a_p*n_p)/sum(a_p*l_p)`. Do not average normalized outputs. Handle empty/all-masked shards without `-inf-(-inf)`; derive other attention formulas separately.

Reuse index-K across queries of the same request with each causal bound; different requests do not share KV pages. Splitting computation over replicated KV differs from physically sharding it, which changes writes/pages/ownership. Historical GLM 128K/S1 CP still regressed: long context is an opportunity, not a monotonic threshold rule.

## Compare the strategy independently from fusion

For old/new plans Q0/Q1 and staged/fused organizations U/F, `U0↔F0` isolates fusion at Q0; `U0↔U1` studies strategy; `U0↔F1` supports only a combined result without other controls. Build controls needed by the hypothesis, not four mandatory implementations.

Compare completion latency for the full target subgraph with the best known unfused plan under equal GPUs/topology, global work, precision, memory constraints and output/state scope. Include required communication and per-call conversions. Report one-time redistribution separately, including it if the measured operation actually performs it.

After a plan change revisit invalidated local contracts while retaining validated device routines. A rank-local PMC decrease may be work relocation: retain per-rank work and aggregate only compatible intervals. Validate global reconstruction and rank progress before ranking plans; retain the best configuration for each workload bucket when the tradeoff changes with workload.
