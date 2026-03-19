module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @vectorAddKernel_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) kernel {
      %c64_i32 = arith.constant 64 : i32
      %block_id_x = gpu.block_id  x
      %0 = arith.index_cast %block_id_x : index to i32
      %thread_id_x = gpu.thread_id  x
      %1 = arith.index_cast %thread_id_x : index to i32
      %2 = arith.muli %0, %c64_i32 : i32
      %3 = fly.make_int_tuple(%2) : (i32) -> !fly.int_tuple<?{div=64}>
      %4 = arith.index_cast %2 : i32 to index
      %5 = arith.index_cast %4 : index to i64
      %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %7 = arith.muli %0, %c64_i32 : i32
      %8 = fly.make_int_tuple(%7) : (i32) -> !fly.int_tuple<?{div=64}>
      %9 = arith.index_cast %7 : i32 to index
      %10 = arith.index_cast %9 : index to i64
      %11 = llvm.getelementptr %arg2[%10] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %12 = arith.muli %0, %c64_i32 : i32
      %13 = fly.make_int_tuple(%12) : (i32) -> !fly.int_tuple<?{div=64}>
      %14 = arith.index_cast %12 : i32 to index
      %15 = arith.index_cast %14 : index to i64
      %16 = llvm.getelementptr %arg3[%15] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %17 = fly.make_atom : () -> !fly.atom.universal_copy<32>
      %18 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %19 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %20 = fly.make_layout(%18, %19) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %c1_i64 = arith.constant 1 : i64
      %21 = llvm.alloca %c1_i64 x f32 : (i64) -> !llvm.ptr<5>
      %22 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %23 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %24 = fly.make_layout(%22, %23) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %c1_i64_0 = arith.constant 1 : i64
      %25 = llvm.alloca %c1_i64_0 x f32 : (i64) -> !llvm.ptr<5>
      %26 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %27 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %28 = fly.make_layout(%26, %27) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %c1_i64_1 = arith.constant 1 : i64
      %29 = llvm.alloca %c1_i64_1 x f32 : (i64) -> !llvm.ptr<5>
      %30 = fly.make_int_tuple(%1) : (i32) -> !fly.int_tuple<?>
      %31 = arith.index_cast %1 : i32 to index
      %32 = arith.index_cast %31 : index to i64
      %33 = llvm.getelementptr %6[%32] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %34 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %35 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
      %36 = fly.make_layout(%34, %35) : (!fly.int_tuple<1>, !fly.int_tuple<0>) -> !fly.layout<1:0>
      %c4_i64 = arith.constant 4 : i64
      "llvm.intr.memcpy"(%21, %33, %c4_i64) <{isVolatile = false}> : (!llvm.ptr<5>, !llvm.ptr<1>, i64) -> ()
      %37 = fly.make_int_tuple(%1) : (i32) -> !fly.int_tuple<?>
      %38 = arith.index_cast %1 : i32 to index
      %39 = arith.index_cast %38 : index to i64
      %40 = llvm.getelementptr %11[%39] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %41 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %42 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
      %43 = fly.make_layout(%41, %42) : (!fly.int_tuple<1>, !fly.int_tuple<0>) -> !fly.layout<1:0>
      %c4_i64_2 = arith.constant 4 : i64
      "llvm.intr.memcpy"(%25, %40, %c4_i64_2) <{isVolatile = false}> : (!llvm.ptr<5>, !llvm.ptr<1>, i64) -> ()
      %44 = llvm.load %21 : !llvm.ptr<5> -> vector<1xf32>
      %45 = llvm.load %25 : !llvm.ptr<5> -> vector<1xf32>
      %46 = arith.addf %44, %45 : vector<1xf32>
      llvm.store %46, %29 : vector<1xf32>, !llvm.ptr<5>
      %47 = fly.make_int_tuple(%1) : (i32) -> !fly.int_tuple<?>
      %48 = arith.index_cast %1 : i32 to index
      %49 = arith.index_cast %48 : index to i64
      %50 = llvm.getelementptr %16[%49] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %51 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %52 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
      %53 = fly.make_layout(%51, %52) : (!fly.int_tuple<1>, !fly.int_tuple<0>) -> !fly.layout<1:0>
      %c4_i64_3 = arith.constant 4 : i64
      "llvm.intr.memcpy"(%50, %29, %c4_i64_3) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<5>, i64) -> ()
      gpu.return
    }
  }
  func.func @vectorAdd(%arg0: !llvm.ptr<1>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: i32, %arg5: !gpu.async.token) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>
    %1 = llvm.mlir.undef : !llvm.struct<packed ()>
    %2 = llvm.mlir.undef : !llvm.struct<packed (i32)>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %3 = llvm.extractvalue %arg1[0, 0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %4 = arith.addi %arg4, %c64_i32 : i32
    %5 = arith.subi %4, %c1_i32 : i32
    %6 = arith.floordivsi %5, %c64_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = llvm.insertvalue %3, %2[0] : !llvm.struct<packed (i32)> 
    %9 = llvm.insertvalue %8, %0[0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %10 = llvm.insertvalue %1, %9[1] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    gpu.launch_func <%arg5 : !gpu.async.token> @kernels::@vectorAddKernel_0 blocks in (%7, %c1, %c1) threads in (%c64, %c1, %c1)  args(%arg0 : !llvm.ptr<1>, %10 : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2 : !llvm.ptr<1>, %arg3 : !llvm.ptr<1>)
    return
  }
}
