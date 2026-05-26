"""
Grouped Output Projection BMM — Compute Model
==============================================

Operation (from model.py):
    o   : [T, G, D]  = [num_tokens, 16, 4096]
    wo_a: [G, R, D]  = [16, 1024, 4096]
    out  = einsum("sgd,grd->sgr", o, wo_a)   ->  [T, G, R]

Equivalent to G=16 independent GEMMs, each:
    [T, D] @ [D, R]^T  ->  [T, R]
    i.e., M=T, K=D=4096, N=R=1024, batch=G=16

What this model computes (hardware-agnostic):
  - FLOPs and memory traffic as a function of T and dtype
  - Arithmetic Intensity (AI = FLOPs / Bytes)
  - The AI formula is exact; bound conclusions need actual hw numbers.

To complete the roofline analysis, fill in:
  - peak_tflops_per_dtype: from MI450 SPG / characterization results
  - hbm_bw_GBs: from MI450 memory bandwidth benchmark (not spec sheet)
"""

# ---------------------------------------------------------------------------
# Problem parameters (fixed by V4 config.json)
# ---------------------------------------------------------------------------

G = 16      # n_local_groups  (o_groups with TP=1)
R = 1024    # o_lora_rank
D = 4096    # n_heads/G * head_dim = 8 * 512

DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8":  1,
    "fp4":  0.5,
}

# ---------------------------------------------------------------------------
# MI450 ubench results (full-card, single device)
# Two measurement dates — use as a range
# ---------------------------------------------------------------------------
#
#              5/12      5/13
#  read TB/s   15.6      19.6
#  r/w  TB/s   15.5      18.5
#  bf16 TFLOPS  4067      4200
#  fp8  TFLOPS 14541     15036
#  fp4  TFLOPS 30542     29322
#
# For a16w8 (act=bf16, wt=fp8):
#   Compute path: bf16 WMMA (dequant fp8 weight → bf16 before MMA)
#                 OR fp8 WMMA (cast bf16 act → fp8 on the fly)
#   Both cases tabulated below.
#
# Bandwidth: read-dominated (W >> A+C for small T), use read BW.

HW = {
    "5/12": {"read_TBs": 15.6,  "rw_TBs": 15.5,  "bf16_TFLOPS": 4067,  "fp8_TFLOPS": 14541},
    "5/13": {"read_TBs": 19.6,  "rw_TBs": 18.5,  "bf16_TFLOPS": 4200,  "fp8_TFLOPS": 15036},
}

# ---------------------------------------------------------------------------
# Arithmetic Intensity — hardware-independent
# ---------------------------------------------------------------------------

def flops(T: int, g: int = G, n: int = R, k: int = D) -> int:
    """Total FLOPs (2 per multiply-add)."""
    return 2 * g * T * n * k


def mem_bytes(T: int, act_dtype: str, wt_dtype: str, out_dtype: str,
              g: int = G, n: int = R, k: int = D) -> dict:
    """
    Ideal HBM traffic (no reuse assumed).
    A: activations read   [T, G, K]
    W: weight read        [G, N, K]  — constant, shared across all T tokens
    C: output write       [T, G, N]
    """
    A = T * g * k * DTYPE_BYTES[act_dtype]
    W =     g * n * k * DTYPE_BYTES[wt_dtype]
    C = T * g * n * DTYPE_BYTES[out_dtype]
    return {"A": A, "W": W, "C": C, "total": A + W + C,
            "no_weight": A + C}   # traffic if W is already in L2


def arithmetic_intensity(T: int, act_dtype: str, wt_dtype: str, out_dtype: str,
                         weight_in_l2: bool = False) -> float:
    """
    AI = FLOPs / Bytes.
    weight_in_l2=True: assume wo_a is pinned in GL2 across requests;
                       only A and C counted as HBM traffic.
    """
    f = flops(T)
    m = mem_bytes(T, act_dtype, wt_dtype, out_dtype)
    traffic = m["no_weight"] if weight_in_l2 else m["total"]
    return f / traffic


def crossover_T_formula(act_dtype: str, wt_dtype: str, out_dtype: str,
                        peak_tflops: float, hbm_bw_GBs: float) -> float:
    """
    Solve AI(T) = ridge_point for T.
    ridge_point = peak_tflops*1e12 / (hbm_bw_GBs*1e9)  [FLOP/Byte]

    Result is exact; only valid when hw numbers are filled in.
    """
    ridge = (peak_tflops * 1e12) / (hbm_bw_GBs * 1e9)
    a = DTYPE_BYTES[act_dtype]
    w = DTYPE_BYTES[wt_dtype]
    c = DTYPE_BYTES[out_dtype]
    # Derivation:
    #   AI(T) = 2*G*T*N*K / (T*G*K*a + G*N*K*w + T*G*N*c)
    #   Set = ridge, solve for T:
    #   T*(2*N*K - ridge*(K*a + N*c)) = ridge*N*K*w
    num   = ridge * R * D * w
    denom = 2 * R * D - ridge * (D * a + R * c)
    if denom <= 0:
        return float("inf")   # always memory-bound regardless of T
    return num / denom

# ---------------------------------------------------------------------------
# Report — AI only, no bound conclusions
# ---------------------------------------------------------------------------

def print_ai_table(act_dtype: str, wt_dtype: str, out_dtype: str,
                   weight_in_l2: bool = False,
                   t_list: list = None):
    if t_list is None:
        t_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    W_MB = G * R * D * DTYPE_BYTES[wt_dtype] / 1e6
    A_per_token_KB = G * D * DTYPE_BYTES[act_dtype] / 1024
    tag = " [W pinned in GL2]" if weight_in_l2 else " [W from HBM]"
    print(f"\n{'='*76}")
    print(f"Grouped Output Projection BMM  a16w8  [G={G}, K={D}, N={R}]{tag}")
    print(f"  act={act_dtype}, wt={wt_dtype}, out={out_dtype}")
    print(f"  W(wo_a) footprint : {W_MB:.1f} MB  (fp8, per-device)")
    print(f"  A per token       : {A_per_token_KB:.1f} KB  (bf16)")
    W_dom_T = W_MB * 1e6 / (G * D * DTYPE_BYTES[act_dtype] + G * R * DTYPE_BYTES[out_dtype])
    print(f"  W dominates A+C when T < {W_dom_T:.0f}  (i.e. W > A+C)")
    print(f"{'='*76}")
    print(f"  {'T':>7}  {'FLOPs':>14}  {'A+C':>9}  {'W':>8}  {'AI':>10}  note")
    print(f"  {'-'*7}  {'-'*14}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*20}")
    for T in t_list:
        f  = flops(T)
        m  = mem_bytes(T, act_dtype, wt_dtype, out_dtype)
        ai = arithmetic_intensity(T, act_dtype, wt_dtype, out_dtype, weight_in_l2)
        ac = (m["A"] + m["C"]) / 1e6
        w  = m["W"] / 1e6

        # annotations
        note = ""
        if T == 1:
            note = "decode bs=1"
        elif T == 64:
            note = "decode bs=64"
        elif T == 1024:
            note = "prefill ~1k"
        elif T == 8192:
            note = "prefill ~8k"
        elif T == 65536:
            note = "prefill ~64k"

        ac_str = f"{ac:.2f}MB" if ac < 1000 else f"{ac/1024:.2f}GB"
        print(f"  {T:>7,}  {f:>14,}  {ac_str:>9}  {w:>6.1f}MB  {ai:>10.1f}  {note}")

    print()
    ai_max = arithmetic_intensity(t_list[-1], act_dtype, wt_dtype, out_dtype, weight_in_l2)
    print(f"  AI range: {arithmetic_intensity(t_list[0], act_dtype, wt_dtype, out_dtype, weight_in_l2):.1f} → {ai_max:.1f} FLOP/B")
    print(f"  AI asymptote (T→∞, W negligible): {2*R*D / (D*DTYPE_BYTES[act_dtype] + R*DTYPE_BYTES[out_dtype]):.1f} FLOP/B")
    print()
    print("  ridge_point = peak_TFLOPS*1e12 / hbm_bw_GBs*1e9  [fill from SPG/bench]")
    print("  crossover_T = crossover_T_formula(act, wt, out, peak_tflops=?, hbm_bw_GBs=?)")


T_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
           16384, 32768, 65536]

def roofline_table(t_list, hw_date="5/13"):
    """
    Confirmed: bf16 MMA (fp8 weight dequant→bf16 on chip), output bf16.
    W HBM traffic = fp8 size; compute peak = bf16 TFLOPS.
    Bandwidth = read_TBs (read-dominated: W >> A+C for small T).
    """
    hw       = HW[hw_date]
    peak     = hw["bf16_TFLOPS"] * 1e12   # FLOP/s
    bw       = hw["read_TBs"]    * 1e12   # B/s
    ridge    = peak / bw                   # FLOP/B

    print(f"\n{'='*82}")
    print(f"Roofline — a16w8  bf16 MMA  [G={G}, K={D}, N={R}]  ({hw_date})")
    print(f"  bf16 peak = {hw['bf16_TFLOPS']:,} TFLOPS   read BW = {hw['read_TBs']} TB/s"
          f"   ridge = {ridge:.1f} FLOP/B")
    print(f"  crossover T ≈ {crossover_T_formula('bf16','fp8','bf16', hw['bf16_TFLOPS'], hw['read_TBs']*1e3):.0f} tokens")
    print(f"{'='*82}")
    print(f"  {'T':>7}  {'AI':>8}  {'bound':>7}  {'compute':>10}  {'memory':>10}  {'roofline':>10}  {'util%':>6}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
    for T in t_list:
        f   = flops(T)
        m   = mem_bytes(T, "bf16", "fp8", "bf16")
        ai  = arithmetic_intensity(T, "bf16", "fp8", "bf16")
        t_c = f   / peak * 1e6   # µs, compute-bound time
        t_m = m["total"] / bw * 1e6  # µs, memory-bound time
        t_r = max(t_c, t_m)
        bound = "compute" if ai >= ridge else "memory"
        # utilization of the binding resource
        util = (t_c / t_m * 100) if bound == "memory" else (t_m / t_c * 100)
        print(f"  {T:>7,}  {ai:>8.1f}  {bound:>7}  {t_c:>9.2f}µs  {t_m:>9.2f}µs"
              f"  {t_r:>9.2f}µs  {util:>5.1f}%")
    print(f"\n  util% = how busy the NON-bottleneck resource is (100% = perfectly balanced)")


if __name__ == "__main__":
    roofline_table(T_SWEEP, hw_date="5/13")
    roofline_table(T_SWEEP, hw_date="5/12")
