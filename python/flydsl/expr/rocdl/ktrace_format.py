# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCm device-record layout shared by the lowering pass and host decoder."""

# Record layout, 32 bytes, two dwordx4 stores.
#
#   u64 ts          offset 0
#   u64 payload     offset 8
#   u32 hw_id       offset 16
#   u32 xcc_id      offset 20
#   u32 event_id:24 | kind:7 | has_payload:1   offset 24
#   u32 start_slot  offset 28
RECORD_BYTES = 32

# Each wave reserves one counter slot plus up to 255 event slots. The matching
# constant in FlyKtraceToROCDL.cpp is checked by the lowering tests.
EVENTS_PER_WAVE = 256

# A suppressed START must never pair with a real record slot.
UNPAIRED_SLOT = 0xFFFFFFFF
