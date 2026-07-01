# Committed offline autotune configs

Tuned kernel configs, committed so downstream consumers (and CI) get a fast,
deterministic config with **no search**. This is the aiter/SGLang "offline"
model. See `docs/autotune_guide.md`.

- **Filename is the key:** `op,shape...,dtype=...,device_name=...json`.
- **Generate:** run the kernel's autotuned entry point with
  `FLYDSL_AUTOTUNE=1 FLYDSL_AUTOTUNE_CONFIG_DIR=configs/autotune`, then commit
  the emitted JSON.
- **Guarded by** `tests/unit/test_autotune_configs.py` (GPU-free): every file
  here must be well-formed, loadable, and filename-consistent with its content.

Do not hand-edit configs; re-tune and re-commit instead.
