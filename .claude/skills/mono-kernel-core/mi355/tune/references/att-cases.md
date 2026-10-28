# ATT cases: from stalls to measured improvements

Historical GLM cases cover weight working sets, shared-expert prefetch, and A4W4 layout. Keep their precision, binary, and sampling scopes separate. Hit means traced instruction execution, not cache hit; cycles/hit below means `sum(stalls)/sum(hits)`. See [profiling](profiling.md) for collection and decoding methods.

<a id="att-workset"></a>

## 1. Larger working set: locate the added waiting

**Context.** TP8, H8/S4, one request; one/eight means independent addresses containing identical weights. With inputs and algorithm fixed, uninstrumented FlyDSL latency grew **57.166→62.102 µs**, while TileRT stayed **54.470→54.612 µs**. These were weight copies, not eight requests or a real layer stack.

ATT used GPU0, CU1/CU3, four SE streams each, selected SIMD, and three seeds. Each region retained 256 wavefronts after discarding the first eight of 16 launches. Instruction reconstruction passed; fixed native→Fly capture order remained a limitation. Selected stall sums, averaged over CUs/seeds, were:

| Category, Mcycles/region | One address | Eight addresses |
|---|---:|---:|
| VMEM-load | 5.150 | 7.373 |
| VMEM-wait | 7.547 | 7.933 |
| Barrier | 8.995 | 9.082 |
| All categories | 25.237 | 28.414 |

**Interpretation.** VMEM loads contributed about 70% of added traced stalls, not 70% of layer-latency growth. UG weight loads held at 15,360 hits/region while cycles/hit rose **77.52→101.49 (+30.9%)** in all six CU×seed pairs. Shared-scale cycles/hit rose **420.94→546.99** at 1,536 hits/region. The issue was more waiting per execution, not simply more instructions.

Matching binary/source mapping placed shared loads after routing even though their addresses were available earlier; routed loads depended on selection. This motivated prefetching the complete shared weight/scale set. No matching L2/HBM PMC established cache misses or queue saturation, and summed barrier stalls were not directly recoverable latency.

<a id="att-prefetch"></a>

## 2. Shared prefetch: stalls can move while latency improves

**Change.** On the A8W8 S>1 path, move shared weight/scale prefetch before routing. ISA confirmed 18 moved loads; VGPR 220, LDS 95,024 bytes, private segment zero, and 8,545 static instructions were unchanged.

Follow-up ATT used eight addresses, GPU0/CU1, four SEs, three seeds, and 256 retained wavefronts/region:

| Group | Hits/region | Before cycles/hit | After cycles/hit |
|---|---:|---:|---:|
| Shared scale | 1,536 | 532.57 | 284.78 |
| Shared weight | 3,072 | 55.83 | 58.25 |
| Routed weight | 12,288 | 101.26 | 155.56 |

Total traced stalls fell **28.314→27.575 Mcycles**; barriers fell **10.576→9.773 Mcycles**. Shared-scale waiting decreased while routed-weight waiting increased, consistent with redistributed stalls rather than universally faster memory access.

**Ablations and validation.** Scale-only prefetch gave 62.239→62.260 µs, no gain. Batched down consumption raised VGPR to 252 and regressed; deleting the preparation barrier produced no stable gain. Fresh uninstrumented confirmation used three rounds × three seeds, reversed order, 256 launches/HIP graph, and maximum-rank event median:

| Weight addresses | Before → after, µs | Latency reduction |
|---|---:|---:|
| One | 57.245 → 55.887 | 2.37% |
| Eight | 62.158 → 61.520 | 1.03% |

Original bitwise outputs and routing matched. Retain the scoped gain and investigate remaining routed loads; the working-set gap persists, and this A8W8 result does not validate an A8W4 transfer.

<a id="att-layout"></a>

## 3. A4W4 layout: normalize changed execution counts

**Context and change.** Scale rows 192 bytes apart scattered the tile's 16×4 scale bytes. Pack by 16 rows ×128 K and pair eight gate + eight up rows, changing initializer and loader together while preserving quantization/MFMA math. Screening also compared down batches 9/18 and prefetch distances 1/2.

ATT used GPU0/CU1, four SEs, eight addresses, S4, seed 42, and 256 retained wavefronts; debug and executed `.text` matched.

| Group | Original hits / stall cycles | Final hits / stall cycles | Original → final cycles/hit |
|---|---:|---:|---:|
| UG scale | 7,680 / 815,856 | 7,680 / 461,540 | 106.23 → 60.10 |
| UG weight | 7,680 / 1,288,324 | 7,680 / 801,168 | 167.75 → 104.32 |
| Down scale | 4,352 / 278,280 | 4,608 / 161,480 | 63.94 → 35.04 |
| Down weight | 4,608 / 485,976 | 4,608 / 474,100 | 105.46 → 102.89 |

Down-scale hits changed: `1-(161480/4608)/(278280/4352)≈45.2%` measures per-execution stall reduction. Normalization does not remove changes in executed paths.

**Selection evidence.** Eight-address screening controls were 57.244/57.195 µs. Scale tiling gave 54.119, scale+paired 53.678, down18 alone 57.308, and paired+down18 53.459 µs. Deeper prefetch did not win. Layout was the main direction, with down18 an increment; order/drift prevents treating differences as additive contributions.

Opposing evidence remained: paired layout with batch 9 increased barrier stalls **8,510,208→9,296,636 cycles**, while final batch 18 used **7,048,456**. Down-weight cycles/hit improved only 2.4%, and native-control total stalls drifted about 2.3%.

**Result and limits.** Fresh confirmation used two rounds × three seeds, 256 launches/HIP graph, slowest rank then mean across cases: **57.225→53.566 µs (6.39%)** for eight-address A4W4. Old/new output SHA matched. VGPR rose **162→194**, with unchanged LDS and zero private-segment allocation and reported spills. The PMC pilot had permission/perturbation limits and did not establish HBM/L2 improvements.

For another precision path, re-derive co-consumed weight/scale units and test its [packer and loader](directions.md#layout); these ATT numbers do not establish native scaled-MFMA behavior.

<a id="att-coverage"></a>

## 4. Trace coverage: keep conclusions within sampled stages

A later TP4 capture passed instruction reconstruction but mainly sampled indexer/attention on GPU0/CU3 with limited SE/SIMD. It retained **16 wavefronts/region**, versus 256 above, after dropping eight initial launches. Its stall totals could neither rank the two designs nor establish down as the largest layer bottleneck.

Map current roles/tasks to sampled wavefronts and matching debug/binary information before drawing UG/down conclusions. Until that coverage exists, use PMC/completion markers for a limited hypothesis; an old top-PC list does not diagnose the current kernel.
