# Baselines

## Full E2E c=256 background

These are the local non-DPA DeepSeek-V4-Pro `1k/1k c=256` results that motivated
the FlyDSL-only repro.

| stack | ATOM / deps | total tok/s | output tok/s | TPOT | TTFT |
| --- | --- | ---: | ---: | ---: | ---: |
| old good | `68a2d29`, old aiter, FlyDSL `0.1.9.dev599` | 8496.85 | 4250.48 | 58.04 ms | 719.77 ms |
| bad original | `2984891`, aiter `d988eaa`, FlyDSL `0.2.0` | 7810.44 | 3907.11 | 63.32 ms | 798.97 ms |
| full hotpath fix | FlyDSL + AITER qk/rawptr hotpath fixes | 8450.04 | 4227.06 | 58.30 ms | 768.43 ms |
| reviewer-safe nocachesig fix | FlyDSL direct-state only, no cache-sig change | 8225.67 | 4114.83 | 59.95 ms | 734.25 ms |

## Short hotpath background

Prior short c=256 hotpath checks:

| stack | output tok/s | TPOT |
| --- | ---: | ---: |
| bad pristine | 4042.2 | 56.95 ms |
| direct raw only | 4115.7 | 55.96 ms |
| direct raw + kwargs | 4478.6 | 51.50 ms |
| direct raw + kwargs + qkcache | 4550.2 | 50.58 ms |
| direct raw + kwargs + qkcache + rawptr | 4616.4 | 49.75 ms |

## FlyDSL-only repro baseline policy

Record fresh results from this script in `results/` for the machine under test.
Absolute numbers depend on CPU, Python, ROCm, GPU, and FlyDSL build options.
The pass/fail signal is the relative gap:

- original FlyDSL `jit_keyword_stream` vs fixed FlyDSL `jit_keyword_stream`
- `jit_keyword_stream` vs `compiled_positional`
- mixed replay original vs fixed

For the regression to be considered reproduced, `gpu_event_us_per_call` should
stay in the same range while host wall time changes materially.

## Local smoke baseline from this repro

Command shape:

```bash
python bench_flydsl_hotpath.py --windows 1 --warmup-windows 0 --gpu-event-calls 2
```

Environment:

| field | value |
| --- | --- |
| container | `yhl_dev` |
| GPU | `AMD Radeon Graphics` |
| arch | `gfx950` |
| original path | `local-c256-verify/flydsl-0.2.0-pristine` |
| fixed path | `local-c256-verify/flydsl-0.2.0-directraw-kw-patch` |

| stack | qk jit | qk compiled | fused jit | fused compiled | hca compress jit | hca compress compiled | hca scatter jit | hca scatter compiled | mixed jit | mixed compiled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 68.68 | 8.63 | 152.45 | 13.27 | 106.40 | 9.25 | 91.67 | 10.14 | 104.23 | 10.09 |
| fixed directraw-kw | 64.89 | 8.66 | 129.27 | 12.93 | 90.80 | 8.70 | 78.31 | 9.10 | 91.36 | 10.01 |

All values are host wall `us/call`.
