// RUN: rocir-opt %s --rocir-to-standard | FileCheck %s

// Comprehensive test covering all layout operations and lowering

module {
  // ============================================================================
  // Basic Operations Tests
  // ============================================================================
  
  // CHECK-LABEL: @test_make_shape
  func.func @test_make_shape() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    
    // CHECK: rocir.make_shape
    %shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
    
    // CHECK: arith.muli
    // CHECK: arith.muli
    %size = rocir.size %shape : !rocir.shape<(?,?)> -> index
    
    // CHECK: return
    return %size : index
  }
  
  // CHECK-LABEL: @test_make_layout
  func.func @test_make_layout() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    // CHECK: rocir.make_shape
    %shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
    // CHECK: rocir.make_stride
    %stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    
    // CHECK: rocir.make_layout
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: arith.muli
    %size = rocir.size %layout : !rocir.layout<(?,?)> -> index
    
    return %size : index
  }
  
  // CHECK-LABEL: @test_get_shape_stride
  func.func @test_get_shape_stride() -> index {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %shape = rocir.make_shape %c4, %c8 : (index, index) -> !rocir.shape<(?,?)>
    %stride = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<(?,?)>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.get_shape
    %extracted_shape = rocir.get_shape %layout : !rocir.layout<(?,?)> -> !rocir.shape<(?,?)>
    // CHECK: rocir.get_stride
    %extracted_stride = rocir.get_stride %layout : !rocir.layout<(?,?)> -> !rocir.stride<(?,?)>
    
    %size1 = rocir.size %extracted_shape : !rocir.shape<(?,?)> -> index
    %size2 = rocir.cosize %layout : !rocir.layout<(?,?)> -> index
    
    %result = arith.addi %size1, %size2 : index
    return %result : index
  }
  
  // ============================================================================
  // Product Operations Tests (Tiling)
  // ============================================================================
  
  // CHECK-LABEL: @test_logical_product
  func.func @test_logical_product() -> index {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    // Base layout: 16x32 column-major
    %base_shape = rocir.make_shape %c16, %c32 : (index, index) -> !rocir.shape<(?,?)>
    %base_stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<(?,?)>
    %base = rocir.make_layout %base_shape, %base_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // Tiler: 4x8
    %tile_shape = rocir.make_shape %c4, %c8 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<(?,?)>
    %tiler = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.logical_product
    // CHECK: rocir.composition
    // CHECK: arith.muli
    %tiled = rocir.logical_product %base, %tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %tiled : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_zipped_product
  func.func @test_zipped_product() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
    %base_stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    %base = rocir.make_layout %base_shape, %base_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c2, %c4 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c2 : (index, index) -> !rocir.stride<(?,?)>
    %tiler = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.zipped_product
    // CHECK: rocir.logical_product
    %zipped = rocir.zipped_product %base, %tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %zipped : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_tiled_product
  func.func @test_tiled_product() -> index {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<(?,?)>
    %base_stride = rocir.make_stride %c1, %c32 : (index, index) -> !rocir.stride<(?,?)>
    %base = rocir.make_layout %base_shape, %base_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    %tiler = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.tiled_product
    // CHECK: rocir.logical_product
    %tiled = rocir.tiled_product %base, %tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %tiled : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_flat_product
  func.func @test_flat_product() -> index {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = rocir.make_shape %c16, %c8 : (index, index) -> !rocir.shape<(?,?)>
    %base_stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<(?,?)>
    %base = rocir.make_layout %base_shape, %base_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c4, %c2 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<(?,?)>
    %tiler = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.flat_product
    // CHECK: rocir.logical_product
    %flat = rocir.flat_product %base, %tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
    
    %size = rocir.size %flat : !rocir.layout<(?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_raked_product
  func.func @test_raked_product() -> index {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = rocir.make_shape %c32, %c8 : (index, index) -> !rocir.shape<(?,?)>
    %base_stride = rocir.make_stride %c1, %c32 : (index, index) -> !rocir.stride<(?,?)>
    %base = rocir.make_layout %base_shape, %base_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c8, %c4 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    %tiler = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.raked_product
    // CHECK: rocir.logical_product
    %raked = rocir.raked_product %base, %tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %raked : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_blocked_product
  func.func @test_blocked_product() -> index {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = rocir.make_shape %c64, %c16 : (index, index) -> !rocir.shape<(?,?)>
    %base_stride = rocir.make_stride %c1, %c64 : (index, index) -> !rocir.stride<(?,?)>
    %base = rocir.make_layout %base_shape, %base_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c8, %c4 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    %tiler = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.blocked_product
    // CHECK: rocir.logical_product
    %blocked = rocir.blocked_product %base, %tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %blocked : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // ============================================================================
  // Divide Operations Tests (Partitioning)
  // ============================================================================
  
  // CHECK-LABEL: @test_logical_divide
  func.func @test_logical_divide() -> index {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    
    // Global layout: 128x256
    %global_shape = rocir.make_shape %c128, %c256 : (index, index) -> !rocir.shape<(?,?)>
    %global_stride = rocir.make_stride %c1, %c128 : (index, index) -> !rocir.stride<(?,?)>
    %global = rocir.make_layout %global_shape, %global_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // Tile: 16x32
    %tile_shape = rocir.make_shape %c16, %c32 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<(?,?)>
    %tile = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.logical_divide
    // CHECK: rocir.composition
    %partitioned = rocir.logical_divide %global, %tile : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %partitioned : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_zipped_divide
  func.func @test_zipped_divide() -> index {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = rocir.make_shape %c64, %c128 : (index, index) -> !rocir.shape<(?,?)>
    %global_stride = rocir.make_stride %c1, %c64 : (index, index) -> !rocir.stride<(?,?)>
    %global = rocir.make_layout %global_shape, %global_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    %tile = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.zipped_divide
    // CHECK: rocir.logical_divide
    %zipped = rocir.zipped_divide %global, %tile : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %zipped : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_tiled_divide
  func.func @test_tiled_divide() -> index {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<(?,?)>
    %global_stride = rocir.make_stride %c1, %c32 : (index, index) -> !rocir.stride<(?,?)>
    %global = rocir.make_layout %global_shape, %global_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c4, %c8 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<(?,?)>
    %tile = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.tiled_divide
    // CHECK: rocir.logical_divide
    %tiled = rocir.tiled_divide %global, %tile : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?,?,?)>
    
    %size = rocir.size %tiled : !rocir.layout<(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_flat_divide
  func.func @test_flat_divide() -> index {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = rocir.make_shape %c16, %c32 : (index, index) -> !rocir.shape<(?,?)>
    %global_stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<(?,?)>
    %global = rocir.make_layout %global_shape, %global_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %tile_shape = rocir.make_shape %c4, %c8 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<(?,?)>
    %tile = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.flat_divide
    // CHECK: rocir.logical_divide
    %flat = rocir.flat_divide %global, %tile : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
    
    %size = rocir.size %flat : !rocir.layout<(?,?)> -> index
    return %size : index
  }
  
  // ============================================================================
  // Local Operations Tests (Thread/Block Partitioning)
  // ============================================================================
  
  // CHECK-LABEL: @test_local_partition
  func.func @test_local_partition() -> index {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // Global tensor: 128x256
    %global_shape = rocir.make_shape %c128, %c256 : (index, index) -> !rocir.shape<(?,?)>
    %global_stride = rocir.make_stride %c1, %c128 : (index, index) -> !rocir.stride<(?,?)>
    %global = rocir.make_layout %global_shape, %global_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // Thread tile: 8x16
    %tile_shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
    %tile_stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    %tile = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.local_partition
    // CHECK: rocir.logical_divide
    %thread_data = rocir.local_partition %global, %tile, %c0 : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>, index) -> !rocir.layout<(?,?)>
    
    %size = rocir.size %thread_data : !rocir.layout<(?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_local_tile
  func.func @test_local_tile() -> index {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // Global tensor: 128x256
    %global_shape = rocir.make_shape %c128, %c256 : (index, index) -> !rocir.shape<(?,?)>
    %global_stride = rocir.make_stride %c1, %c128 : (index, index) -> !rocir.stride<(?,?)>
    %global = rocir.make_layout %global_shape, %global_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CTA tile shape: 32x64
    %cta_shape = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<(?,?)>
    
    // CTA coordinates: (0, 0)
    %cta_coord = rocir.make_shape %c0, %c0 : (index, index) -> !rocir.shape<(?,?)>
    
    // CHECK: rocir.local_tile
    // CHECK: rocir.logical_divide
    %cta_tile = rocir.local_tile %global, %cta_shape, %cta_coord : (!rocir.layout<(?,?)>, !rocir.shape<(?,?)>, !rocir.shape<(?,?)>) -> !rocir.layout<(?,?)>
    
    %size = rocir.size %cta_tile : !rocir.layout<(?,?)> -> index
    return %size : index
  }
  
  // ============================================================================
  // Composition Test
  // ============================================================================
  
  // CHECK-LABEL: @test_composition
  func.func @test_composition() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    
    %shape_a = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
    %stride_a = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
    %layout_a = rocir.make_layout %shape_a, %stride_a : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    %shape_b = rocir.make_shape %c4, %c2 : (index, index) -> !rocir.shape<(?,?)>
    %stride_b = rocir.make_stride %c2, %c1 : (index, index) -> !rocir.stride<(?,?)>
    %layout_b = rocir.make_layout %shape_b, %stride_b : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // CHECK: rocir.composition
    // CHECK: arith.muli
    // CHECK: arith.addi
    %composed = rocir.composition %layout_a, %layout_b : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
    
    %size = rocir.size %composed : !rocir.layout<(?,?)> -> index
    return %size : index
  }
}
