# gfx120x call graph

gfx120x means the device arch starts with `gfx120` (RDNA4, wave32, WMMA 16×16×16). It is not gfx1250. gfx1250 is CDNA5, also wave32, and its kernels are not these files.

Soft check: `kernels.common.gfx120x_arch.is_gfx120x`. It never raises. The shared FlashAttention router uses it and returns before the gfx950 and gfx942 paths.

Hard check: `require_gfx120x(device)`. Every gfx120x-only host that takes a tensor calls it with that tensor's device. A non-gfx120x caller gets `ValueError`. Compile-only builders (RoPE, RMS+RoPE, AdaLN, the fp8 quant modules, the stochastic fp8 module, and the SiLU×mul builders) have no device argument, so they do not call it. The examples that launch those builders check the process arch first. Nothing in this set patches a gfx950, gfx942, gfx11, or gfx1250 kernel.

Buffer helpers are `kernels.common.gfx120x_buf_helpers` only.

Elementwise SiLU×mul is `kernels.common.gfx120x_swiglu`. `kernels.quant.rdna4_swiglu` re-exports it. The fused MLP is `kernels.gemm.rdna4_fused_mlp_nmajor`. `kernels.common.act` is the CDNA helper.

`is_gfx120x` and `require_gfx120x` are implemented in `flydsl.runtime.device`. `kernels.common.gfx120x_arch` re-exports them.

## What calls what

```
flydsl_flash_attn_func                         kernels/attention/flash_attn_interface.py
  is_gfx120x?
    yes -> flash_attn_gfx120x_host.flydsl_flash_attn_func
           flydsl_flash_attn_varlen_func
           flydsl_flash_attn_paged_func
           flydsl_flash_attn_varlen_paged_func
    no  -> unchanged gfx950 / gfx942 path
           attn_mask is rejected on those arches. None leaves them unchanged.

flash_attn_gfx120x_host
  bf16/fp16 dense, varlen, paged, split-K
      -> flash_attn_gfx120x.build_flash_attn_func_* 
      -> flash_attn_gfx120x_splitk.build_splitk_combine_module   (only when num_kv_splits > 1)
  fp8  -> flash_attn_fp8_gfx120x
  int8 -> flash_attn_int8_gfx120x
  iu4  -> flash_attn_iu4_gfx120x
          The shared flydsl_flash_attn_func does not call this.
          Packed iu4 is torch.int8, so that entry cannot tell it from dense int8.
          Call flydsl_flash_attn_iu4_func on this host.
          if bias, ALiBi, sink, or return_lse is set:
              unpack nibbles to int8 and call flydsl_flash_attn_int8_func
          else: native body _flash_attn_iu4_native_body_gfx120x
  guards, bias tiles, fold_sink_lse, commit_caller_out
      -> flash_attn_gfx120x_ext
         this file does not run attention

int8_linear_auto                               rdna4_int8_linear_dispatch.py
  select_int8_kernel
    K < 256 -> w8a16_gemm                       rdna4_w8a16_linear.py
    else     -> quantize_int8_rowwise if a_int8 is omitted, then iu8
                rdna4_int8_linear
  int8_linear_auto does not call int8_linear_fused.

int8_linear_fused                              rdna4_int8_linear_fused.py
  K > 128 -> quantize_int8_rowwise, then int8_linear_dispatched("iu8")
             that path does not fall back to W8A16
  else    -> the fused module

scaled_mm_fp8_auto                             rdna4_scaled_mm_fp8_auto.py
  select_tile -> build_scaled_mm_fp8_module    rdna4_scaled_mm_fp8.py
  There is no function named scaled_mm_fp8.

iu4_gemm                                       rdna4_iu4_gemm.py
  build_iu4_gemm_module
  prefer_native False, or a shape the atom rejects -> unpack to iu8

fused_swiglu_mlp_nmajor                        rdna4_fused_mlp_nmajor.py
  K == FFN == 16, M and N_out multiples of 16 -> fused_swiglu_mlp_inreg
  else M, K, FFN, N_out all multiples of 16   -> fused_swiglu_mlp_lds
       one wave; K loop then FFN loop; mid spilled to LDS (32 lanes x 8)
  else                                        -> two GEMMs, mid in GMEM

convrot_w4a4_linear                            rdna4_convrot_w4a4.py
  quantize_convrot_w4a4_weight
  default linear_dtype="int4" -> iu4_gemm
  linear_dtype="int8"          -> unpack, then the int8 linear

quantize_int8_rowwise / quantize_int8_tensorwise
  their build_*_module, then the compiled kernel

rope / rms_rope / adaln
  build_*_module returns a callable.
  RoPE takes raw pointers (flyc.from_c_void_p) plus the pair counts.
  RMS+RoPE and AdaLN take fx.Tensor plus the row / group / eps arguments.
  See examples/12 (RoPE only) and tests/kernels/test_gfx120x_norm_rope.py.
```

## FlashAttention arguments that still raise

On bf16 and fp16 these run in the kernel: dense, `return_lse`, sink, ALiBi, packed varlen, paged `linear` / `linear3d` / `vectorized`, varlen together with paged, GQA when `Hq % Hkv == 0`, and split-K.

fp8 and int8 dense run bias, ALiBi, sink, LSE, and that same head index. They do not take varlen, paged KV, or split-K.

Still raise:

- fp8, int8, or iu4 with varlen, paged KV, or split-K
- varlen per-head bias, and varlen plus paged with per-head bias
- ragged paged KV with bias
- any other paged layout
- `Hq` not divisible by `Hkv`
- an explicit KV head count that does not match the KV tensor
- batch-varying ALiBi slopes

## Wave32, and what a wave64 port should copy

CDNA kernels in this repo use wave64 and MFMA (`examples/03-tiledMma.py`). gfx120x uses a block of 32 and WMMA. A wave64 port should not reuse the launch bounds. Copy the host order instead:

| Family | Example | Public call |
|---|---|---|
| One WMMA tile (bf16, iu8, fp8) | `examples/07-tiledMma_gfx120x.py` | the atom, not a library host |
| FlashAttention | `examples/08-flash_attn_gfx120x.py` | `flash_attn_gfx120x_host.flydsl_flash_attn_func` |
| Int8 linear | `examples/09-int8_linear_gfx120x.py` | `int8_linear_auto` |
| FP8 scaled matmul | `examples/10-scaled_mm_fp8_gfx120x.py` | `scaled_mm_fp8_auto`. The fixed tile is `build_scaled_mm_fp8_module`, launched by `tests/kernels/test_rdna4_scaled_mm_fp8.py`, not by the example. |
| ConvRot W4A4 | `examples/11-convrot_w4a4_gfx120x.py` | `quantize_convrot_w4a4_weight`, `convrot_w4a4_linear` |
| RoPE | `examples/12-norm_rope_gfx120x.py` | `build_rope_module` |
| Int8 rowwise quant | `examples/13-quant_int8_gfx120x.py` | `quantize_int8_rowwise` |
| iu4 GEMM | `examples/14-iu4_gemm_gfx120x.py` | `iu4_gemm` |
| Fused SwiGLU MLP | `examples/15-fused_mlp_gfx120x.py` | `fused_swiglu_mlp_inreg` |

`tests/arch_compat.py` allowlists examples 07 through 15 to gfx120x only. Other architectures skip them. Examples 01 through 06 are unchanged.

## Not in this tree

FP4, MXFP, na3d, and a wave64 build of these kernels. `fused_swiglu_mlp_inreg` keeps the mid in registers when K and the feed-forward width are both 16. `fused_swiglu_mlp_lds` is the in-kernel K/FFN loop for other multiples of 16: the swapped GEMM0 fragment is stored to LDS (32 lanes × 8 elements) and reloaded as GEMM1's A. `fused_swiglu_mlp_nmajor` calls those two. Shapes that are not multiples of 16 still use separate GEMMs and a global mid.

`CLAUDE.md` prefers `fx.copy` / `fx.gemm` over `copy_atom_call` / `mma_atom_call`. gfx120x kernels use that form, including a single atom: rank-1 `make_rmem_tensor` fragments plus `fx.gemm` for W8A16, iu8 linear, iu4 GEMM, int8 and iu4 FlashAttention, and the fused int8 linear. RMS+RoPE, AdaLN, and int8 rowwise quant use `fx.copy`. The authoring guide still shows `fx.copy_atom_call` for the gfx1250 TDM atom. That atom's base pointer comes from the copy operand, so the TDM example is unchanged. CDNA and gfx1250 kernels were not edited.
