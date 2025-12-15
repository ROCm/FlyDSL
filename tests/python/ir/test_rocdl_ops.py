#!/usr/bin/env python3
"""Test ROCDL dialect operations."""

import sys
import os


from mlir import ir
from mlir.dialects import func, arith
import mlir.extras.types as T
from rocdsl.dialects.ext import rocdl

print('=' * 70)
print('ROCDL Dialect Operations Test')
print('=' * 70)

def test_thread_operations():
    """Test thread/workgroup ID operations."""
    print('\n[Test 1] Thread/Workgroup ID Operations')
    print('-' * 70)
    
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        i32 = T.i32()
        
        with ir.InsertionPoint(module.body):
            @func.FuncOp.from_py_func()
            def thread_test():
                tid_x = rocdl.workitem_id_x(i32)
                tid_y = rocdl.workitem_id_y(i32)
                tid_z = rocdl.workitem_id_z(i32)
                wg_x = rocdl.workgroup_id_x(i32)
                wg_y = rocdl.workgroup_id_y(i32)
                wg_z = rocdl.workgroup_id_z(i32)
                wg_dim_x = rocdl.workgroup_dim_x(i32)
                grid_x = rocdl.grid_dim_x(i32)
                wavesize = rocdl.wavefrontsize(i32)
        
        mlir_str = str(module)
        assert 'rocdl.workitem.id.x' in mlir_str
        assert 'rocdl.workitem.id.y' in mlir_str
        assert 'rocdl.workgroup.id.x' in mlir_str
        
        print('Thread IDs: workitem_id_x/y/z')
        print('Workgroup IDs: workgroup_id_x/y/z')
        print('Dimensions: workgroup_dim, grid_dim, wavefrontsize')
        print('PASSED')
    return True

def test_sync_operations():
    """Test synchronization operations."""
    print('\n[Test 2] Synchronization Operations')
    print('-' * 70)
    
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        
        with ir.InsertionPoint(module.body):
            @func.FuncOp.from_py_func()
            def sync_test():
                rocdl.barrier()
                rocdl.s_barrier()
                rocdl.s_waitcnt(0)
        
        mlir_str = str(module)
        assert 'rocdl.barrier' in mlir_str or 'rocdl.workgroup.barrier' in mlir_str
        assert 'rocdl.s.barrier' in mlir_str
        assert 'rocdl.s.waitcnt' in mlir_str
        
        print('Barriers: barrier, s_barrier')
        print('Wait: s_waitcnt')
        print('PASSED')
    return True

def test_lane_operations():
    """Test lane/shuffle operations."""
    print('\n[Test 3] Lane/Shuffle Operations')
    print('-' * 70)
    
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        i32 = T.i32()
        
        with ir.InsertionPoint(module.body):
            @func.FuncOp.from_py_func()
            def lane_test():
                src = arith.constant(i32, 42)
                lane = arith.constant(i32, 0)
                val1 = rocdl.readlane(i32, src._value, lane._value)
                val2 = rocdl.readfirstlane(i32, src._value)
                offset = arith.constant(i32, 0x1F)
                val3 = rocdl.ds_swizzle(i32, src._value, offset._value)
        
        mlir_str = str(module)
        assert 'rocdl.readlane' in mlir_str or 'rocdl.ds.swizzle' in mlir_str
        
        print('Lane ops: readlane, readfirstlane, ds_swizzle')
        print('PASSED')
    return True

def test_summary():
    """Print summary."""
    print('\n' + '=' * 70)
    print('Test Summary')
    print('=' * 70)
    
    all_ops = [x for x in dir(rocdl) if not x.startswith('_')]
    thread_ops = [x for x in all_ops if 'workitem' in x or 'workgroup' in x or 'grid' in x]
    sync_ops = [x for x in all_ops if 'barrier' in x or 'wait' in x]
    mfma_ops = [x for x in all_ops if 'mfma' in x]
    wmma_ops = [x for x in all_ops if 'wmma' in x]
    buffer_ops = [x for x in all_ops if 'buffer' in x]
    lane_ops = [x for x in all_ops if 'lane' in x or 'swizzle' in x or 'bpermute' in x]
    
    print(f'Total ROCDL operations: {len(all_ops)}')
    print(f'  Thread/Block ops: {len(thread_ops)}')
    print(f'  Synchronization: {len(sync_ops)}')
    print(f'  Buffer ops: {len(buffer_ops)}')
    print(f'  Lane/Shuffle ops: {len(lane_ops)}')
    print(f'  MFMA ops: {len(mfma_ops)}')
    print(f'  WMMA ops: {len(wmma_ops)}')

if __name__ == '__main__':
    try:
        test_thread_operations()
        test_sync_operations()
        test_lane_operations()
        test_summary()
        
        print('\n' + '=' * 70)
        print('ALL TESTS PASSED')
        print('=' * 70)
        
    except Exception as e:
        print(f'\nTEST FAILED: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
