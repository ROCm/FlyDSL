# Preserve numerical behavior, routing, and state updates

## Specify the complete numerical path

For every stage retain `logical/packed shape → A/W/scales and groups → MFMA → accumulator → activation → rounding/quantization → route weighting → reduction`. Resolve scale formulas, format variants, saturation, padding and zero behavior from the current implementation and an independent oracle.

Specify the FP8 encoding and amax-to-scale calculation, and distinguish logical K from packed-byte offsets for FP4. BF16 round-to-nearest-even (RNE) differs from truncation. Record the exact scale-rounding formula rather than a label such as “upward.” Preserve required rounding when eliminating a tensor in global memory. Keep accumulation, precision, and error tolerances fixed when evaluating a layout or scheduling change.

Historical examples show why:

- The historical DSV4 routed-expert path used MXFP4 weights with one E8M0 scale per 32 elements, E4M3 inputs with scales rounded upward for each group of 32 elements, and a SwiGLU result kept in FP32 until FP8 quantization using the native exponent rule. The shared-expert branch used a different scale layout and BF16 RNE boundary. Adding BF16 rounding to routed intermediate activations violated that branch's contract.
- GLM's legacy BF16-MFMA oracle did not model a later native scaled MFMA path with 32-element MXFP8 blocks. Old/new candidate bitwise agreement did not supply the missing independent reference for the native path.
- K3 A16W4 ownership changes must initially retain input and intermediate formats and activation. Replacing atomic scatter with local combine changes summation/rounding organization and needs independent validation, even when all dtype labels match.

These are examples, not formats to impose on a new model. Check accumulation→activation→rounding→quantized values/scales as separate endpoints with zeros, signs, extrema, saturation and scale boundaries.

## Routing: distinguish selection from probability computation

Write the actual selection score, group selection if any, bias placement, top-k/tie order, probability normalization, route scale, and shared contribution. In one GLM sigmoid+bias path, selection used `s_i+bias_i`, but weights normalized unbiased `s_i` over selected experts. Other models need their own formula.

Diagnose mismatches on fixed logits/bias in three classes:

| Class | Meaning and action |
|---|---|
| Permutation | Same expert set, different slots: align by expert ID for diagnosis, retain raw slots, and honor any required slot ABI |
| Exact tie | Check the actual tie rule; a pivot/compact fast path may differ from lowest-ID-first. Resolve required behavior rather than changing the oracle silently |
| Near score | Distinct scores must not become ties through encoding or rounding; inspect full-score bits and upstream arithmetic |

GLM packed routing keys cleared eight score bits to store expert IDs, misordering distinct scores. K3 needed winner ID, score, and bias to move together; replacing `best_score-best_bias` with raw score could change FP32 rounding despite algebraic equivalence.

Fixed routing isolates expert math. It does not resolve a routing change caused by a one-unit-in-the-last-place (ULP) difference in an upstream result. For expert union/EP, keep `(token, original slot, global/local expert, weight)` aligned and apply route/shared/residual contributions once.

## Characterize reference nondeterminism and validate state updates

Repeat the original reference from identical snapshots. A historical DSV4 BF16 atomic split-K router could change near-tie routes; shared truncation/atomics also affected outputs. Preserve the nondeterministic reference results and identify any deterministic diagnostic control separately. A control using the candidate device routine in both implementations tests a narrower comparison, not independent correctness.

Output includes declared caches and recurrent state. Track written logical values/regions, byte-exact untouched regions, page/slot boundaries, aliases, padding and uncommitted rows. Use nonzero initial state and separate restored copies for R/C.

For recurrence or multi-token prediction, each request owns its predecessor chain. Historical K3 used `state_indices[S+1]`, token s reading slot[s] and writing slot[s+1]; flattening B requests must not link their chains. Apply snapshot/accept/commit tests only where this kernel owns those effects. Validate consecutive steps, interleaved requests and legal reuse for the affected ABI.

Report equivalence to the previous baseline, staged/fused equivalence with shared device routines, and correctness against an independent reference separately. Missing oracle arithmetic or unexecuted state checks remain gaps; see [unit validation](../../test/references/validation.md).
