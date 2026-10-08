"""Load compiled code and query actual residency on all eight local GPUs.

Run after idle-node admission. Does not launch a kernel. A passing result is
necessary but not sufficient for executing the subsequent correctness suite.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import torch

root = Path(sys.argv[1])
data = json.loads((root / 'compile.json').read_text())
source_root = Path(data['source']).parents[2]
for relative, digest in data['source_manifest'].items():
    assert hashlib.sha256((source_root / relative).read_bytes()).hexdigest() == digest, relative
assert len(data['kernels']) == 1
record = data['kernels'][0]
assert hashlib.sha256(Path(record['hsaco']).read_bytes()).hexdigest() == record['sha256']
assert record['max_flat_workgroup_size'] == 512
assert record['vgpr_spill_count'] == record['private_segment_fixed_size'] == 0, record

lib = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
assert lib.hipInit(0) == 0
devices = ctypes.c_int()
assert lib.hipGetDeviceCount(ctypes.byref(devices)) == 0
assert devices.value == 8, 'This experiment requires all eight GPUs on node46'
results = []
grid_blocks = data.get('grid_blocks', 256)
for device in range(devices.value):
    assert lib.hipSetDevice(device) == 0
    cu = torch.cuda.get_device_properties(device).multi_processor_count
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    assert lib.hipModuleLoad(ctypes.byref(module), record['hsaco'].encode()) == 0
    try:
        assert lib.hipModuleGetFunction(ctypes.byref(function), module, record['name'].encode()) == 0
        resident = ctypes.c_int()
        assert lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(resident), function, 512, ctypes.c_size_t(0)) == 0
        results.append(dict(device=device, compute_units=cu,
                            resident_blocks_per_cu=resident.value,
                            grid_blocks=grid_blocks, pass_residency=cu * resident.value >= grid_blocks))
    finally:
        assert lib.hipModuleUnload(module) == 0
data.update(hip_occupancy_checked=True, devices=results,
            passed=all(r['pass_residency'] for r in results), gpu_execution=False)
(root / 'occupancy.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(dict(passed=data['passed'], devices=results)), flush=True)
assert data['passed'], 'Do not execute: complete grid cannot be resident'
