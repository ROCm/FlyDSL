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
  llvm.func @vectorAdd(%arg0: !llvm.ptr<1>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: i32, %arg5: !llvm.ptr) attributes {fly.stream_arg_index = 5 : index, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = llvm.mlir.undef : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>
    %2 = llvm.mlir.undef : !llvm.struct<packed ()>
    %3 = llvm.mlir.undef : !llvm.struct<packed (i32)>
    %4 = llvm.mlir.constant(64 : i32) : i32
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.extractvalue %arg1[0, 0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %8 = llvm.add %arg4, %0 : i32
    %9 = llvm.sdiv %8, %4 : i32
    %10 = llvm.mul %9, %4 : i32
    %11 = llvm.icmp "ne" %8, %10 : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.icmp "slt" %8, %12 : i32
    %14 = llvm.mlir.constant(false) : i1
    %15 = llvm.icmp "ne" %13, %14 : i1
    %16 = llvm.and %11, %15 : i1
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.add %9, %17 : i32
    %19 = llvm.select %16, %18, %9 : i1, i32
    %20 = llvm.sext %19 : i32 to i64
    %21 = llvm.insertvalue %7, %3[0] : !llvm.struct<packed (i32)> 
    %22 = llvm.insertvalue %21, %1[0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %23 = llvm.insertvalue %2, %22[1] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    gpu.launch_func  @kernels::@vectorAddKernel_0 blocks in (%20, %5, %5) threads in (%6, %5, %5) : i64 args(%arg0 : !llvm.ptr<1>, %23 : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2 : !llvm.ptr<1>, %arg3 : !llvm.ptr<1>)
    llvm.return
  }
}
