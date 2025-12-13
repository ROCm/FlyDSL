// RUN: rocir-opt %s --rocir-to-standard | FileCheck %s

// Test logical_product operation
func.func @test_logical_product() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // Create base layout: (8, 16) with stride (1, 8)
  %shape_a = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride_a = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %layout_a = rocir.make_layout %shape_a, %stride_a : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Create tiler layout: (2, 2) with stride (1, 2) 
  %shape_b = rocir.make_shape %c2, %c2 : (index, index) -> !rocir.shape<(?,?)>
  %stride_b = rocir.make_stride %c1, %c2 : (index, index) -> !rocir.stride<(?,?)>
  %layout_b = rocir.make_layout %shape_b, %stride_b : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Logical product
  %product = rocir.logical_product %layout_a, %layout_b : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Get size of result
  %size = rocir.size %product : !rocir.layout<(?,?)> -> index
  
  return %size : index
}

// Test zipped_product operation
func.func @test_zipped_product() -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c4_s = arith.constant 4 : index
  
  // Create layouts
  %shape_a = rocir.make_shape %c8, %c4 : (index, index) -> !rocir.shape<(?,?)>
  %stride_a = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %layout_a = rocir.make_layout %shape_a, %stride_a : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %shape_b = rocir.make_shape %c4, %c4 : (index, index) -> !rocir.shape<(?,?)>
  %stride_b = rocir.make_stride %c1, %c4_s : (index, index) -> !rocir.stride<(?,?)>
  %layout_b = rocir.make_layout %shape_b, %stride_b : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Zipped product
  %product = rocir.zipped_product %layout_a, %layout_b : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
  
  %size = rocir.size %product : !rocir.layout<(?,?)> -> index
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
  %shape_target = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<(?,?)>
  %stride_target = rocir.make_stride %c1, %c32_s : (index, index) -> !rocir.stride<(?,?)>
  %layout_target = rocir.make_layout %shape_target, %stride_target : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Create tiler layout: (8, 8) with stride (1, 8)
  %shape_tiler = rocir.make_shape %c8, %c8 : (index, index) -> !rocir.shape<(?,?)>
  %stride_tiler = rocir.make_stride %c1, %c8_s : (index, index) -> !rocir.stride<(?,?)>
  %layout_tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Logical divide
  %divided = rocir.logical_divide %layout_target, %layout_tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
  
  %size = rocir.size %divided : !rocir.layout<(?,?)> -> index
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
  
  %shape_target = rocir.make_shape %c16, %c32 : (index, index) -> !rocir.shape<(?,?)>
  %stride_target = rocir.make_stride %c1, %c16_s : (index, index) -> !rocir.stride<(?,?)>
  %layout_target = rocir.make_layout %shape_target, %stride_target : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %shape_tiler = rocir.make_shape %c4, %c4 : (index, index) -> !rocir.shape<(?,?)>
  %stride_tiler = rocir.make_stride %c1, %c4_s : (index, index) -> !rocir.stride<(?,?)>
  %layout_tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %divided = rocir.tiled_divide %layout_target, %layout_tiler : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
  
  %size = rocir.size %divided : !rocir.layout<(?,?)> -> index
  return %size : index
}
