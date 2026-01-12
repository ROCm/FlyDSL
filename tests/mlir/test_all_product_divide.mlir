// Test all product and divide operations

func.func @test_tiled_product() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_block = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride_block = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %block = flir.make_layout %shape_block, %stride_block : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %shape_tiler = flir.make_shape %c2, %c2 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c2 : (index, index) -> !flir.stride<(?,?)>
  %tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %result = flir.tiled_product %block, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
  %size = flir.size %result : !flir.layout<(?,?):(?,?)> -> index
  return %size : index
}

func.func @test_flat_product() -> index {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  %shape_block = flir.make_shape %c8, %c4 : (index, index) -> !flir.shape<(?,?)>
  %stride_block = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %block = flir.make_layout %shape_block, %stride_block : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %shape_tiler = flir.make_shape %c4, %c4 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
  %tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %result = flir.flat_product %block, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
  %size = flir.size %result : !flir.layout<(?,?):(?,?)> -> index
  return %size : index
}

func.func @test_raked_product() -> index {
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_block = flir.make_shape %c16, %c8 : (index, index) -> !flir.shape<(?,?)>
  %stride_block = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
  %block = flir.make_layout %shape_block, %stride_block : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %shape_tiler = flir.make_shape %c2, %c2 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c2 : (index, index) -> !flir.stride<(?,?)>
  %tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %result = flir.raked_product %block, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
  %size = flir.size %result : !flir.layout<(?,?):(?,?)> -> index
  return %size : index
}

func.func @test_blocked_product() -> index {
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  
  %shape_block = flir.make_shape %c32, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride_block = flir.make_stride %c1, %c32 : (index, index) -> !flir.stride<(?,?)>
  %block = flir.make_layout %shape_block, %stride_block : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %shape_tiler = flir.make_shape %c4, %c4 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
  %tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %result = flir.blocked_product %block, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
  %size = flir.size %result : !flir.layout<(?,?):(?,?)> -> index
  return %size : index
}

func.func @test_zipped_divide() -> index {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  
  %shape_layout = flir.make_shape %c32, %c64 : (index, index) -> !flir.shape<(?,?)>
  %stride_layout = flir.make_stride %c1, %c32 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape_layout, %stride_layout : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %shape_tiler = flir.make_shape %c8, %c8 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %result = flir.zipped_divide %layout, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
  %size = flir.size %result : !flir.layout<(?,?):(?,?)> -> index
  return %size : index
}

func.func @test_flat_divide() -> index {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  
  %shape_layout = flir.make_shape %c16, %c32 : (index, index) -> !flir.shape<(?,?)>
  %stride_layout = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape_layout, %stride_layout : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %shape_tiler = flir.make_shape %c4, %c4 : (index, index) -> !flir.shape<(?,?)>
  %stride_tiler = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
  %tiler = flir.make_layout %shape_tiler, %stride_tiler : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %result = flir.flat_divide %layout, %tiler : (!flir.layout<(?,?):(?,?)>, !flir.layout<(?,?):(?,?)>) -> !flir.layout<(?,?):(?,?)>
  %size = flir.size %result : !flir.layout<(?,?):(?,?)> -> index
  return %size : index
}
