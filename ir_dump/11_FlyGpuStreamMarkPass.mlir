module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
    llvm.func @vectorAddKernel_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {gpu.kernel, rocdl.kernel} {
      %0 = llvm.mlir.constant(4 : i64) : i64
      %1 = llvm.mlir.constant(1 : i64) : i64
      %2 = llvm.mlir.constant(64 : i32) : i32
      %3 = rocdl.workgroup.id.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.trunc %4 : i64 to i32
      %6 = rocdl.workitem.id.x : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.mul %5, %2 : i32
      %9 = llvm.sext %8 : i32 to i64
      %10 = llvm.getelementptr %arg0[%9] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %11 = llvm.mul %5, %2 : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.getelementptr %arg2[%12] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %14 = llvm.mul %5, %2 : i32
      %15 = llvm.sext %14 : i32 to i64
      %16 = llvm.getelementptr %arg3[%15] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %17 = llvm.alloca %1 x f32 : (i64) -> !llvm.ptr<5>
      %18 = llvm.alloca %1 x f32 : (i64) -> !llvm.ptr<5>
      %19 = llvm.alloca %1 x f32 : (i64) -> !llvm.ptr<5>
      %20 = llvm.getelementptr %10[%7] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      "llvm.intr.memcpy"(%17, %20, %0) <{isVolatile = false}> : (!llvm.ptr<5>, !llvm.ptr<1>, i64) -> ()
      %21 = llvm.getelementptr %13[%7] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      "llvm.intr.memcpy"(%18, %21, %0) <{isVolatile = false}> : (!llvm.ptr<5>, !llvm.ptr<1>, i64) -> ()
      %22 = llvm.load %17 : !llvm.ptr<5> -> vector<1xf32>
      %23 = llvm.load %18 : !llvm.ptr<5> -> vector<1xf32>
      %24 = llvm.fadd %22, %23 : vector<1xf32>
      llvm.store %24, %19 : vector<1xf32>, !llvm.ptr<5>
      %25 = llvm.getelementptr %16[%7] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      "llvm.intr.memcpy"(%25, %19, %0) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<5>, i64) -> ()
      llvm.return
    }
  }
  func.func @vectorAdd(%arg0: !llvm.ptr<1>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: i32, %arg5: !gpu.async.token) attributes {fly.stream_arg_index = 5 : index, llvm.emit_c_interface} {
    %c63_i32 = arith.constant 63 : i32
    %0 = llvm.mlir.undef : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>
    %1 = llvm.mlir.undef : !llvm.struct<packed ()>
    %2 = llvm.mlir.undef : !llvm.struct<packed (i32)>
    %c64_i32 = arith.constant 64 : i32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %3 = llvm.extractvalue %arg1[0, 0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %4 = arith.addi %arg4, %c63_i32 : i32
    %5 = arith.floordivsi %4, %c64_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = llvm.insertvalue %3, %2[0] : !llvm.struct<packed (i32)> 
    %8 = llvm.insertvalue %7, %0[0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %9 = llvm.insertvalue %1, %8[1] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    gpu.launch_func <%arg5 : !gpu.async.token> @kernels::@vectorAddKernel_0 blocks in (%6, %c1, %c1) threads in (%c64, %c1, %c1)  args(%arg0 : !llvm.ptr<1>, %9 : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2 : !llvm.ptr<1>, %arg3 : !llvm.ptr<1>)
    return
  }
}
