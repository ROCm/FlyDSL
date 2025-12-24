// RUN: flir-opt %s --flir-to-standard | FileCheck %s

module {
  func.func @test_crd2idx_2d() -> index {
    // Create 2D row-major layout: shape=(32, 64), stride=(64, 1)
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    
    %shape = flir.make_shape %c32, %c64 : (index, index) -> !flir.shape<(?,?)>
    %stride = flir.make_stride %c64, %c1 : (index, index) -> !flir.stride<(?,?)>
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
    
    // Create coordinate (2, 3)
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %coord = flir.make_coord %c2, %c3 : (index, index) -> !flir.coord<(?,?)>
    
    // Convert to linear index: 2*64 + 3*1 = 131
    %idx = flir.crd2idx %coord, %layout : (!flir.coord<(?,?)>, !flir.layout<(?,?)>) -> index
    
    // CHECK: %[[MUL0:.*]] = arith.muli %c2, %c64 : index
    // CHECK: %[[MUL1:.*]] = arith.muli %c3, %c1 : index
    // CHECK: %[[ADD:.*]] = arith.addi %[[MUL0]], %[[MUL1]] : index
    
    return %idx : index
  }
  
  func.func @test_idx2crd_2d() {
    // Create 2D layout: shape=(8, 16)
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
    %stride = flir.make_stride %c16, %c1 : (index, index) -> !flir.stride<(?,?)>
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
    
    // Convert index 35 to coordinate: row=35/16=2, col=35%16=3
    %c35 = arith.constant 35 : index
    %coord = flir.idx2crd %c35, %layout : (index, !flir.layout<(?,?)>) -> !flir.coord<(?,?)>
    
    // CHECK: %[[DIV:.*]] = arith.divui %c35, %c16 : index
    // CHECK: %[[REM:.*]] = arith.remui %c35, %c16 : index
    // CHECK: flir.make_coord %[[DIV]], %[[REM]]
    
    return
  }
  
  func.func @test_rank() -> index {
    %c0 = arith.constant 0 : index
    %shape = flir.make_shape %c0, %c0, %c0 : (index, index, index) -> !flir.shape<(?,?,?)>
    
    // Get rank of 3D shape
    %rank = flir.rank %shape : !flir.shape<(?,?,?)> -> index
    
    // CHECK: %[[RANK:.*]] = arith.constant 3 : index
    
    return %rank : index
  }
}
