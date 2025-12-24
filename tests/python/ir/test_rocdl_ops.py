#!/usr/bin/env python3
"""Test ROCDL dialect operations."""

from pyflir.dialects.ext import rocdl, flir
from pyflir.dialects.ext import arith as arith_ext
class _ROCDLOps(flir.MlirModule):
    @flir.jit
    def thread_test(self: flir.T.i64):
        i32 = flir.T.i32()
        rocdl.workitem_id_x(i32)
        rocdl.workitem_id_y(i32)
        rocdl.workitem_id_z(i32)
        rocdl.workgroup_id_x(i32)
        rocdl.workgroup_id_y(i32)
        rocdl.workgroup_id_z(i32)
        rocdl.workgroup_dim_x(i32)
        rocdl.grid_dim_x(i32)
        rocdl.wavefrontsize(i32)
        return []

def test_sync_operations():
    """Test synchronization operations."""
    class _M(flir.MlirModule):
        @flir.jit
        def sync_test(self: flir.T.i64):
            rocdl.barrier()
            rocdl.s_barrier()
            rocdl.s_waitcnt(0)
            return []
        
    mlir_str = str(_M().module)
    assert "rocdl.barrier" in mlir_str or "rocdl.workgroup.barrier" in mlir_str
    assert "rocdl.s.barrier" in mlir_str
    assert "rocdl.s.waitcnt" in mlir_str

def test_lane_operations():
    """Test lane/shuffle operations."""
    class _M(flir.MlirModule):
        @flir.jit
        def lane_test(self: flir.T.i64):
            i32 = flir.T.i32()
            src = arith_ext.constant(42, type=i32)
            lane = arith_ext.constant(0, type=i32)
            rocdl.readlane(i32, src.value, lane.value)
            rocdl.readfirstlane(i32, src.value)
            offset = arith_ext.constant(0x1F, type=i32)
            rocdl.ds_swizzle(i32, src.value, offset.value)
            return []
        
    mlir_str = str(_M().module)
    assert "rocdl.readlane" in mlir_str or "rocdl.ds.swizzle" in mlir_str

def test_summary():
    """Print summary."""
    all_ops = [x for x in dir(rocdl) if not x.startswith('_')]
    thread_ops = [x for x in all_ops if 'workitem' in x or 'workgroup' in x or 'grid' in x]
    sync_ops = [x for x in all_ops if 'barrier' in x or 'wait' in x]
    mfma_ops = [x for x in all_ops if 'mfma' in x]
    wmma_ops = [x for x in all_ops if 'wmma' in x]
    buffer_ops = [x for x in all_ops if 'buffer' in x]
    lane_ops = [x for x in all_ops if 'lane' in x or 'swizzle' in x or 'bpermute' in x]
    assert len(all_ops) > 0
    assert len(thread_ops) > 0
    assert len(sync_ops) > 0
    assert len(lane_ops) > 0

def test_thread_operations():
    """Test thread/workgroup ID operations."""
    mlir_str = str(_ROCDLOps().module)
    assert "rocdl.workitem.id.x" in mlir_str
    assert "rocdl.workitem.id.y" in mlir_str
    assert "rocdl.workgroup.id.x" in mlir_str
