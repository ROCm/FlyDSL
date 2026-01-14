// RUN: flir-opt %s --flir-to-standard | FileCheck %s

// Comprehensive test covering all layout operations and lowering

module {
  // ============================================================================
  // Basic Operations Tests
  // ============================================================================
  
  // CHECK-LABEL: @test_make_shape
  func.func @test_make_shape() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    
    // CHECK: flir.make_shape
    %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    
    // CHECK: arith.muli
    // CHECK: arith.muli
    %size = flir.size %shape : !flir.shape<(?,?)> -> index
    
    // CHECK: return
    return %size : index
  }
  
  // CHECK-LABEL: @test_make_layout
  func.func @test_make_layout() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    // CHECK: flir.make_shape
    %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    // CHECK: flir.make_stride
    %stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    
    // CHECK: flir.make_layout
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: arith.muli
    %size = flir.size %layout : !flir.layout<(?,?):(?,?)> -> index
    
    return %size : index
  }
  
  // CHECK-LABEL: @test_get_shape_stride
  func.func @test_get_shape_stride() -> index {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %shape = flir.make_shape %c4, %c8 : (index, index) -> !flir.shape<(?,?)>
    %stride = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.get_shape
    %extracted_shape = flir.get_shape %layout : !flir.layout<(?,?):(?,?)> -> !flir.shape<(?,?)>
    // CHECK: flir.get_stride
    %extracted_stride = flir.get_stride %layout : !flir.layout<(?,?):(?,?)> -> !flir.stride<(?,?)>
    
    %size1 = flir.size %extracted_shape : !flir.shape<(?,?)> -> index
    %size2 = flir.cosize %layout : !flir.layout<(?,?):(?,?)> -> index
    
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
    %base_shape = flir.make_shape %c16, %c32 : (index, index) -> !flir.shape<(?,?)>
    %base_stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
    %base = flir.make_layout %base_shape, %base_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // Tiler: 4x8
    %tile_shape = flir.make_shape %c4, %c8 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
    %tiler = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.logical_product
    // CHECK: flir.composition
    // CHECK: arith.muli
    %tiled = flir.logical_product %base, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %tiled : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_zipped_product
  func.func @test_zipped_product() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    %base_stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    %base = flir.make_layout %base_shape, %base_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c2, %c4 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c2 : (index, index) -> !flir.stride<(?,?)>
    %tiler = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.zipped_product
    // CHECK: flir.logical_product
    %zipped = flir.zipped_product %base, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %zipped : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_tiled_product
  func.func @test_tiled_product() -> index {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = flir.make_shape %c32, %c64 : (index, index) -> !flir.shape<(?,?)>
    %base_stride = flir.make_stride %c1, %c32 : (index, index) -> !flir.stride<(?,?)>
    %base = flir.make_layout %base_shape, %base_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    %tiler = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.tiled_product
    // CHECK: flir.logical_product
    %tiled = flir.tiled_product %base, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %tiled : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_flat_product
  func.func @test_flat_product() -> index {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = flir.make_shape %c16, %c8 : (index, index) -> !flir.shape<(?,?)>
    %base_stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
    %base = flir.make_layout %base_shape, %base_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c4, %c2 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
    %tiler = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.flat_product
    // CHECK: flir.logical_product
    %flat = flir.flat_product %base, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %size = flir.size %flat : !flir.layout<(?,?):(?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_raked_product
  func.func @test_raked_product() -> index {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = flir.make_shape %c32, %c8 : (index, index) -> !flir.shape<(?,?)>
    %base_stride = flir.make_stride %c1, %c32 : (index, index) -> !flir.stride<(?,?)>
    %base = flir.make_layout %base_shape, %base_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c8, %c4 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    %tiler = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.raked_product
    // CHECK: flir.logical_product
    %raked = flir.raked_product %base, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %raked : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_blocked_product
  func.func @test_blocked_product() -> index {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = flir.make_shape %c64, %c16 : (index, index) -> !flir.shape<(?,?)>
    %base_stride = flir.make_stride %c1, %c64 : (index, index) -> !flir.stride<(?,?)>
    %base = flir.make_layout %base_shape, %base_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c8, %c4 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    %tiler = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.blocked_product
    // CHECK: flir.logical_product
    %blocked = flir.blocked_product %base, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %blocked : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
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
    %global_shape = flir.make_shape %c128, %c256 : (index, index) -> !flir.shape<(?,?)>
    %global_stride = flir.make_stride %c1, %c128 : (index, index) -> !flir.stride<(?,?)>
    %global = flir.make_layout %global_shape, %global_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // Tile: 16x32
    %tile_shape = flir.make_shape %c16, %c32 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
    %tile = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.logical_divide
    // CHECK: flir.composition
    %partitioned = flir.logical_divide %global, %tile : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %partitioned : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_zipped_divide
  func.func @test_zipped_divide() -> index {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = flir.make_shape %c64, %c128 : (index, index) -> !flir.shape<(?,?)>
    %global_stride = flir.make_stride %c1, %c64 : (index, index) -> !flir.stride<(?,?)>
    %global = flir.make_layout %global_shape, %global_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    %tile = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.zipped_divide
    // CHECK: flir.logical_divide
    %zipped = flir.zipped_divide %global, %tile : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %zipped : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_tiled_divide
  func.func @test_tiled_divide() -> index {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = flir.make_shape %c32, %c64 : (index, index) -> !flir.shape<(?,?)>
    %global_stride = flir.make_stride %c1, %c32 : (index, index) -> !flir.stride<(?,?)>
    %global = flir.make_layout %global_shape, %global_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c4, %c8 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
    %tile = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.tiled_divide
    // CHECK: flir.logical_divide
    %tiled = flir.tiled_divide %global, %tile : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?,?,?):(?,?,?,?)>
    
    %size = flir.size %tiled : !flir.layout<(?,?,?,?):(?,?,?,?)> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_flat_divide
  func.func @test_flat_divide() -> index {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = flir.make_shape %c16, %c32 : (index, index) -> !flir.shape<(?,?)>
    %global_stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
    %global = flir.make_layout %global_shape, %global_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %tile_shape = flir.make_shape %c4, %c8 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
    %tile = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.flat_divide
    // CHECK: flir.logical_divide
    %flat = flir.flat_divide %global, %tile : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %size = flir.size %flat : !flir.layout<(?,?):(?,?)> -> index
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
    %global_shape = flir.make_shape %c128, %c256 : (index, index) -> !flir.shape<(?,?)>
    %global_stride = flir.make_stride %c1, %c128 : (index, index) -> !flir.stride<(?,?)>
    %global = flir.make_layout %global_shape, %global_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // Thread tile: 8x16
    %tile_shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    %tile_stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    %tile = flir.make_layout %tile_shape, %tile_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.local_partition
    // CHECK: flir.logical_divide
    %thread_data = flir.local_partition %global, %tile, %c0 : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>, index) -> !flir.layout<(?,?):(?,?)>
    
    %size = flir.size %thread_data : !flir.layout<(?,?):(?,?)> -> index
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
    %global_shape = flir.make_shape %c128, %c256 : (index, index) -> !flir.shape<(?,?)>
    %global_stride = flir.make_stride %c1, %c128 : (index, index) -> !flir.stride<(?,?)>
    %global = flir.make_layout %global_shape, %global_stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CTA tile shape: 32x64
    %cta_shape = flir.make_shape %c32, %c64 : (index, index) -> !flir.shape<(?,?)>
    
    // CTA coordinates: (0, 0)
    %cta_coord = flir.make_shape %c0, %c0 : (index, index) -> !flir.shape<(?,?)>
    
    // CHECK: flir.local_tile
    // CHECK: flir.logical_divide
    %cta_tile = flir.local_tile %global, %cta_shape, %cta_coord : (!flir.layout<(?,?):(?,?)>, !flir.shape<(?,?)>, !flir.shape<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %size = flir.size %cta_tile : !flir.layout<(?,?):(?,?)> -> index
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
    
    %shape_a = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    %stride_a = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
    %layout_a = flir.make_layout %shape_a, %stride_a : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %shape_b = flir.make_shape %c4, %c2 : (index, index) -> !flir.shape<(?,?)>
    %stride_b = flir.make_stride %c2, %c1 : (index, index) -> !flir.stride<(?,?)>
    %layout_b = flir.make_layout %shape_b, %stride_b : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // CHECK: flir.composition
    // CHECK: arith.muli
    // CHECK: arith.addi
    %composed = flir.composition %layout_a, %layout_b : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    %size = flir.size %composed : !flir.layout<(?,?):(?,?)> -> index
    return %size : index
  }
}
