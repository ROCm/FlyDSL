## Decode 场景 Non-Temporal Cache 策略 (gfx1250)

gfx1250 (RDNA4/GFX12) 支持 non-temporal 特性，通过 TH (Temporal Hint, 3bit) + SCOPE (2bit) 控制，取代旧的 GLC/SLC/DLC。

### Decode 场景结论：全设 NT

Decode 阶段所有大块数据访问都是 memory-bandwidth bound 的一次性流式读取：

- **Weight load：设 NT**。tile M 很小（batch_size），GEMM 沿 N 遍历 weight，每个 tile 只读一次，无复用。
- **KV cache load：设 NT**。沿 seq_len 遍历一遍，下次读要等下一个 decode step，不复用。
- **Activation / Q：无所谓**。体积太小，缓存影响可忽略。

### 原因

L2 cache 对谁都没用，不设 NT 会导致 L2 无意义的 allocate + LRU update + evict thrashing，浪费带宽。设 NT 后减少 L2 管理开销，带宽更多留给实际数据传输。

### 用法

在 decode attention kernel 的 K/V load 和 GEMM kernel 的 weight load 上加：

```asm
global_load_b128 v[0:3], v[4:5], off th:TH_LOAD_NT scope:SCOPE_DEV
```

HIP/CK 中可用 `__builtin_nontemporal_load` 或内联汇编。

### 注意

RDNA4 的 NT 是 LRU 优先级降低，不是真正的 L2 bypass。命中时不更新 LRU 位，使 cache line 更容易被驱逐，但不会完全跳过 L2。
