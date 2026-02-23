# SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Flir divide operations (partitioning)."""

import flydsl.dialects.ext.flir as flir
from flydsl.dialects.ext.arith import Index
from flydsl.dialects.ext import arith as arith_ext


def test_logical_divide(ctx, insert_point):
    """Test logical divide for basic partitioning."""
    
    @flir.jit
    def partition_layout():
        # Global layout: 128x256
        c128 = Index(128)
        c256 = Index(256)
        c16 = Index(16)
        c32 = Index(32)
        c1 = Index(1)
        
        global_shape = flir.make_shape(c128, c256)
        global_stride = flir.make_stride(c1, c128)
        global_layout = flir.make_layout(global_shape, global_stride)
        
        # Tile: 16x32
        tile_shape = flir.make_shape(c16, c32)
        tile_stride = flir.make_stride(c1, c16)
        tile = flir.make_layout(tile_shape, tile_stride)
        
        # Divide creates partitioned layout
        partitioned = flir.logical_divide(global_layout, tile)
        
        size = flir.size(partitioned)
        return [size]
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_zipped_divide(ctx, insert_point):
    """Test zipped divide."""
    
    @flir.jit
    def zipped_partition():
        c64 = Index(64)
        c128 = Index(128)
        c8 = Index(8)
        c16 = Index(16)
        c1 = Index(1)
        
        global_shape = flir.make_shape(c64, c128)
        global_stride = flir.make_stride(c1, c64)
        global_layout = flir.make_layout(global_shape, global_stride)
        
        tile_shape = flir.make_shape(c8, c16)
        tile_stride = flir.make_stride(c1, c8)
        tile = flir.make_layout(tile_shape, tile_stride)
        
        zipped = flir.zipped_divide(global_layout, tile)
        
        size = flir.size(zipped)
        return [size]
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_tiled_divide(ctx, insert_point):
    """Test tiled divide."""
    
    @flir.jit
    def tiled_partition():
        c32 = Index(32)
        c64 = Index(64)
        c4 = Index(4)
        c8 = Index(8)
        c1 = Index(1)
        
        global_shape = flir.make_shape(c32, c64)
        global_stride = flir.make_stride(c1, c32)
        global_layout = flir.make_layout(global_shape, global_stride)
        
        tile_shape = flir.make_shape(c4, c8)
        tile_stride = flir.make_stride(c1, c4)
        tile = flir.make_layout(tile_shape, tile_stride)
        
        tiled = flir.tiled_divide(global_layout, tile)
        
        size = flir.size(tiled)
        return [size]
    
    ctx.module.operation.verify()
    # Apply lowering

    


def test_flat_divide(ctx, insert_point):
    """Test flat divide."""
    
    @flir.jit
    def flat_partition():
        c16 = Index(16)
        c32 = Index(32)
        c4 = Index(4)
        c8 = Index(8)
        c1 = Index(1)
        
        global_shape = flir.make_shape(c16, c32)
        global_stride = flir.make_stride(c1, c16)
        global_layout = flir.make_layout(global_shape, global_stride)
        
        tile_shape = flir.make_shape(c4, c8)
        tile_stride = flir.make_stride(c1, c4)
        tile = flir.make_layout(tile_shape, tile_stride)
        
        flat = flir.flat_divide(global_layout, tile)
        
        size = flir.size(flat)
        return [size]
    
    ctx.module.operation.verify()
    # Apply lowering

    
