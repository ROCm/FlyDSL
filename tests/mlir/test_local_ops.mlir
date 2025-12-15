// Test local_partition and local_tile operations

func.func @test_local_partition() -> index {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  
  // Create a 32x64 layout
  %shape = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c32 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Create an 8x8 tile layout
  %tile_shape = rocir.make_shape %c8, %c8 : (index, index) -> !rocir.shape<(?,?)>
  %tile_stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %tile = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Partition for thread 0
  %result = rocir.local_partition %layout, %tile, %c0 : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>, index) -> !rocir.layout<(?,?)>
  %size = rocir.size %result : !rocir.layout<(?,?)> -> index
  return %size : index
}

func.func @test_local_tile() -> index {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  
  // Create a 128x256 layout (global tensor)
  %shape = rocir.make_shape %c128, %c256 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c128 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Tile shape: 32x64 (CTA tile)
  %tile_shape = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<(?,?)>
  
  // Coordinate: (0, 0)
  %coord = rocir.make_shape %c0, %c0 : (index, index) -> !rocir.shape<(?,?)>
  
  // Extract tile at coordinate (0,0)
  %result = rocir.local_tile %layout, %tile_shape, %coord : (!rocir.layout<(?,?)>, !rocir.shape<(?,?)>, !rocir.shape<(?,?)>) -> !rocir.layout<(?,?)>
  %size = rocir.size %result : !rocir.layout<(?,?)> -> index
  return %size : index
}

func.func @test_local_partition_thread() -> index {
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  
  // Create a 16x16 layout
  %shape = rocir.make_shape %c16, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Create a 2x2 thread tile
  %tile_shape = rocir.make_shape %c2, %c2 : (index, index) -> !rocir.shape<(?,?)>
  %tile_stride = rocir.make_stride %c1, %c2 : (index, index) -> !rocir.stride<(?,?)>
  %tile = rocir.make_layout %tile_shape, %tile_stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Partition for thread 5
  %result = rocir.local_partition %layout, %tile, %c5 : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>, index) -> !rocir.layout<(?,?)>
  %size = rocir.size %result : !rocir.layout<(?,?)> -> index
  return %size : index
}

func.func @test_local_tile_block() -> index {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // Create a 64x64 layout
  %shape = rocir.make_shape %c64, %c64 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c64 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Tile shape: 16x16
  %tile_shape = rocir.make_shape %c16, %c16 : (index, index) -> !rocir.shape<(?,?)>
  
  // Coordinate: (2, 2) - extract tile at block (2,2)
  %coord = rocir.make_shape %c2, %c2 : (index, index) -> !rocir.shape<(?,?)>
  
  %result = rocir.local_tile %layout, %tile_shape, %coord : (!rocir.layout<(?,?)>, !rocir.shape<(?,?)>, !rocir.shape<(?,?)>) -> !rocir.layout<(?,?)>
  %size = rocir.size %result : !rocir.layout<(?,?)> -> index
  return %size : index
}
