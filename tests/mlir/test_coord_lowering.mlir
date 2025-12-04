// RUN: rocir-opt %s --rocir-coord-lowering | FileCheck %s

module {
  func.func @test_crd2idx_2d() -> index {
    // Create 2D row-major layout: shape=(32, 64), stride=(64, 1)
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    
    %shape = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<-1>
    %stride = rocir.make_stride %c64, %c1 : (index, index) -> !rocir.stride<-1>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<-1>, !rocir.stride<-1>) -> !rocir.layout<-1>
    
    // Create coordinate (2, 3)
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %coord = rocir.make_coord %c2, %c3 : (index, index) -> !rocir.coord<-1>
    
    // Convert to linear index: 2*64 + 3*1 = 131
    %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<-1>, !rocir.layout<-1>) -> index
    
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
    
    %shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<-1>
    %stride = rocir.make_stride %c16, %c1 : (index, index) -> !rocir.stride<-1>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<-1>, !rocir.stride<-1>) -> !rocir.layout<-1>
    
    // Convert index 35 to coordinate: row=35/16=2, col=35%16=3
    %c35 = arith.constant 35 : index
    %coord = rocir.idx2crd %c35, %layout : (index, !rocir.layout<-1>) -> !rocir.coord<-1>
    
    // CHECK: %[[DIV:.*]] = arith.divui %c35, %c16 : index
    // CHECK: %[[REM:.*]] = arith.remui %c35, %c16 : index
    // CHECK: rocir.make_coord %[[DIV]], %[[REM]]
    
    return
  }
  
  func.func @test_rank() -> index {
    %c0 = arith.constant 0 : index
    %shape = rocir.make_shape %c0, %c0, %c0 : (index, index, index) -> !rocir.shape<-1>
    
    // Get rank of 3D shape
    %rank = rocir.rank %shape : !rocir.shape<-1> -> index
    
    // CHECK: %[[RANK:.*]] = arith.constant 3 : index
    
    return %rank : index
  }
}
