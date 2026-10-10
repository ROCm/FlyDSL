# FlyDSL conv3d tuned configs

Per-model bf16 tables aligned with aiter `aiter/configs/model_configs/*_conv3d.csv`
(aiter main #5370). The host merges every `*_bf16_tuned_conv3d.csv` in this
directory at runtime; override with `FLYDSL_CONV3D_BF16_CONFIG` (colon-separated
paths). Duplicate `(gfx, cu_num, shape)` keys raise.

Retune with:

```bash
python -m kernels.conv.conv3d_tune \
  -i kernels/conv/configs/qwenimage_vae_bf16_untuned_conv3d.csv \
  -o kernels/conv/configs/qwenimage_vae_bf16_tuned_conv3d.csv
```
