"""Query HIP residency before permitting any polling-kernel execution."""
import ctypes
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
result = json.loads((root / "compile.json").read_text())
lib = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
assert lib.hipInit(0) == 0
for record in result["kernels"]:
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    assert lib.hipModuleLoad(ctypes.byref(module), record["hsaco"].encode()) == 0
    assert lib.hipModuleGetFunction(ctypes.byref(function), module, record["name"].encode()) == 0
    blocks = ctypes.c_int()
    assert lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks), function, 512, ctypes.c_size_t(0)) == 0
    assert lib.hipModuleUnload(module) == 0
    record["resident_blocks_per_cu"] = blocks.value
    record["residency_pass"] = blocks.value >= 1
    record["no_scratch_pass"] = record["vgpr_spill_count"] == record["private_segment_fixed_size"] == 0
result["hip_occupancy_checked"] = True
result["pass"] = all(r["residency_pass"] and r["no_scratch_pass"] for r in result["kernels"])
(root / "occupancy.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
assert result["pass"], "Full-grid residency/resource gate failed; do not launch"
