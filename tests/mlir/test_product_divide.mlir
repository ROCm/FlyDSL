// RUN: flir-opt %s --flir-to-standard | FileCheck %s

// Test logical_product operation
func.func @test_logical_product() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // Create base layout: (8, 16) with stride (1, 8)
  %shape_a = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride_a = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %layout_a = flir.make_layout %shape_a, %stride_a : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Create tiler layout: (2, 2) with stride (1, 2) 
  %shape_b = flir.make_shape %c2, %c2 : (index, index) -> !flir.shape<(?,?)>
  %stride_b = flir.make_stride %c1, %c2 : (index, index) -> !flir.stride<(?,?)>
  %layout_b = flir.make_layout %shape_b, %stride_b : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Logical product
  %product = flir.logical_product %layout_a, %layout_b : (!flir.layout<(?,?)>, !flir.layout<(?,?)>) -> !flir.layout<(?,?)>
  
  // Get size of result
  %size = flir.size %product : !flir.layout<(?,?)> -> index
  
  return %size : index
}

// Test zipped_product operation
func.func @test_zipped_product() -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c4_s = arith.constant 4 : index
  
  // Create layouts
  %shape_a = flir.make_shape %c8, %c4 : (index, index) -> !flir.shape<(?,?)>
  %stride_a = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %layout_a = flir.make_layout %shape_a, %stride_a : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %shape_b = flir.make_shape %c4, %c4 : (index, index) -> !flir.shape<(?,?)>
  %stride_b = flir.make_stride %c1, %c4_s : (index, index) -> !flir.stride<(?,?)>
  %layout_b = flir.make_layout %shape_b, %stride_b : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Zipped product
  %product = flir.zipped_product %layout_a, %layout_b : (!flir.layout<(?,?)>, !flir.layout<(?,?)>) -> !flir.layout<(?,?)>
  
  %size = flir.size %product : !flir.layout<(?,?)> -> index
  return %size : index
}

// Test logical_divide operation
func.func @test_logical_divide() -> index {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c32_s = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c8_s = arith.constant 8 : index
  
  // Create target layout: (32, 64) with stride (1, 32)
  %shape_target = flir.make_shape %c32, %c64 : (index, index) -> !flir.shape<(?,?)>
  %stride_target = flir.make_stride %c1, %c32_s : (index, index) -> !flir.stride<(?,?)>
  %layout_target = flir.make_layout %shape_target, %stride_target : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Create tiler layout: (8, 8) with stride (1, 8)
  %shape_tiler = flir.make_shape %c8, %c8 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c8_s : (index, index) -> !flir.stride<(?,?)>
  %layout_tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Logical divide
  %divided = flir.logical_divide %layout_target, %layout_tiler : (!flir.layout<(?,?)>, !flir.layout<(?,?)>) -> !flir.layout<(?,?)>
  
  %size = flir.size %divided : !flir.layout<(?,?)> -> index
  return %size : index
}

// Test tiled_divide operation
func.func @test_tiled_divide() -> index {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c16_s = arith.constant 16 : index
  %c4 = arith.constant 4 : index
  %c4_s = arith.constant 4 : index
  
  %shape_target = flir.make_shape %c16, %c32 : (index, index) -> !flir.shape<(?,?)>
  %stride_target = flir.make_stride %c1, %c16_s : (index, index) -> !flir.stride<(?,?)>
  %layout_target = flir.make_layout %shape_target, %stride_target : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %shape_tiler = flir.make_shape %c4, %c4 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c4_s : (index, index) -> !flir.stride<(?,?)>
  %layout_tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %divided = flir.tiled_divide %layout_target, %layout_tiler : (!flir.layout<(?,?)>, !flir.layout<(?,?)>) -> !flir.layout<(?,?)>
  
  %size = flir.size %divided : !flir.layout<(?,?)> -> index
  return %size : index
}
