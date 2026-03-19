gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
  gpu.func @vectorAddKernel_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) kernel {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %block_id_x = gpu.block_id  x
    %0 = arith.index_cast %block_id_x : index to i32
    %thread_id_x = gpu.thread_id  x
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %2 : index to i64
    %4 = llvm.getelementptr %arg0[%3] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %5 = arith.muli %0, %c64_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.index_cast %6 : index to i64
    %8 = llvm.getelementptr %arg2[%7] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %9 = arith.muli %0, %c64_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %11 = arith.index_cast %10 : index to i64
    %12 = llvm.getelementptr %arg3[%11] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %13 = llvm.alloca %c1_i64 x f32 : (i64) -> !llvm.ptr<5>
    %14 = llvm.alloca %c1_i64 x f32 : (i64) -> !llvm.ptr<5>
    %15 = llvm.alloca %c1_i64 x f32 : (i64) -> !llvm.ptr<5>
    %16 = arith.index_cast %thread_id_x : index to i64
    %17 = llvm.getelementptr %4[%16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    "llvm.intr.memcpy"(%13, %17, %c4_i64) <{isVolatile = false}> : (!llvm.ptr<5>, !llvm.ptr<1>, i64) -> ()
    %18 = arith.index_cast %thread_id_x : index to i64
    %19 = llvm.getelementptr %8[%18] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    "llvm.intr.memcpy"(%14, %19, %c4_i64) <{isVolatile = false}> : (!llvm.ptr<5>, !llvm.ptr<1>, i64) -> ()
    %20 = llvm.load %13 : !llvm.ptr<5> -> vector<1xf32>
    %21 = llvm.load %14 : !llvm.ptr<5> -> vector<1xf32>
    %22 = arith.addf %20, %21 : vector<1xf32>
    llvm.store %22, %15 : vector<1xf32>, !llvm.ptr<5>
    %23 = arith.index_cast %thread_id_x : index to i64
    %24 = llvm.getelementptr %12[%23] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    "llvm.intr.memcpy"(%24, %15, %c4_i64) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<5>, i64) -> ()
    gpu.return
  }
}
