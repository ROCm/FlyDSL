; ModuleID = '/tmp/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM32_ATOMIC.cu'
source_filename = "/tmp/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM32_ATOMIC.cu"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%"struct.aiter::mxfp4_moe::gemm2::LDSLayout" = type { %union.anon }
%union.anon = type { [8192 x float] }
%struct.__hip_bfloat16 = type { %union.anon.17 }
%union.anon.17 = type { i16 }

$_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph = comdat any

$_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds = comdat any

@_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds = linkonce_odr hidden addrspace(3) global %"struct.aiter::mxfp4_moe::gemm2::LDSLayout" undef, comdat, align 16
@__hip_cuid_63e8a0938c35de0f = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_63e8a0938c35de0f to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef readonly captures(none) %1, ptr addrspace(1) noalias noundef readonly captures(none) %2, ptr addrspace(1) noalias noundef readonly captures(none) %3, ptr addrspace(1) noalias noundef readonly captures(none) %4, ptr addrspace(1) noalias noundef readonly captures(none) %5, ptr addrspace(1) noalias noundef readonly captures(none) %6, ptr addrspace(1) noalias noundef readonly captures(none) %7, i32 noundef %8, ptr addrspace(1) noalias noundef captures(none) %9, ptr addrspace(1) noalias noundef readnone captures(none) %10) local_unnamed_addr #0 comdat {
  %12 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %13 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %14 = icmp samesign ult i32 %13, 256
  tail call void @llvm.assume(i1 %14)
  %15 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %13)
  %16 = load i32, ptr addrspace(1) %5, align 4, !tbaa !7
  %17 = sdiv i32 %16, 32
  %18 = mul nsw i32 %17, 28
  %19 = icmp slt i32 %12, %18
  br i1 %19, label %20, label %410

20:                                               ; preds = %11
  %21 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone %3, i16 0, i64 44154880, i32 131072)
  %22 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone %1, i16 0, i64 10485760, i32 131072)
  %23 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone %2, i16 0, i64 706478080, i32 131072)
  %24 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone %0, i16 0, i64 167772160, i32 131072)
  %25 = and i32 %13, 63
  %26 = lshr i32 %15, 6
  %27 = sdiv i32 %12, 28
  %28 = mul i32 %27, 28
  %29 = sub i32 %12, %28
  %30 = sext i32 %27 to i64
  %31 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %30
  %32 = load i32, ptr addrspace(1) %31, align 4, !tbaa !7
  %33 = icmp samesign ult i32 %32, 385
  tail call void @llvm.assume(i1 %33)
  %34 = shl nsw i32 %29, 16
  %35 = mul nuw nsw i32 %32, 1835008
  %36 = add nsw i32 %35, %34
  %37 = shl i32 %26, 14
  %38 = add i32 %36, %37
  %39 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38)
  %40 = or disjoint i32 %37, 4096
  %41 = add i32 %40, %36
  %42 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %41)
  %43 = or disjoint i32 %37, 8192
  %44 = add i32 %43, %36
  %45 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44)
  %46 = or disjoint i32 %37, 12288
  %47 = add i32 %46, %36
  %48 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47)
  %49 = lshr i32 %25, 3
  %50 = shl nsw i32 %29, 12
  %51 = mul nuw nsw i32 %32, 114688
  %52 = shl i32 %26, 10
  %53 = add i32 %52, %50
  %54 = add i32 %51, %53
  %55 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %54)
  %56 = or disjoint i32 %53, 512
  %57 = add i32 %56, %51
  %58 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %57)
  %59 = shl nsw i32 %27, 5
  %60 = or disjoint i32 %49, %59
  %61 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 range(i32 -76695844, 67108863) %27)
  %62 = shl nsw i32 %61, 9
  %63 = shl i32 %26, 11
  %64 = shl i32 %60, 8
  %65 = add i32 %63, %64
  %66 = shl nuw nsw i32 %26, 3
  %67 = or disjoint i32 %66, %49
  %68 = shl nuw i32 %67, 3
  %69 = shl nuw nsw i32 %13, 4
  %70 = xor i32 %68, %69
  %71 = and i32 %70, 112
  %72 = or disjoint i32 %71, %65
  %73 = getelementptr inbounds nuw [128 x i8], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 %66
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %24, ptr addrspace(3) noundef %73, i32 noundef 16, i32 noundef %72, i32 noundef range(i32 0, 129) 0, i32 noundef 0, i32 noundef 0) #10
  %74 = getelementptr inbounds nuw [128 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 4096), i32 %66
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %24, ptr addrspace(3) noundef nonnull %74, i32 noundef 16, i32 noundef %72, i32 noundef range(i32 0, 129) 128, i32 noundef 0, i32 noundef 0) #10
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %75 = lshr i32 %25, 4
  %76 = and i32 %13, 15
  %77 = shl nuw nsw i32 %75, 6
  %78 = shl nuw nsw i32 %76, 2
  %79 = or disjoint i32 %77, %78
  %80 = or disjoint i32 %79, 256
  %81 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %22, i32 %79, i32 %62, i32 0)
  %82 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %22, i32 %80, i32 %62, i32 0)
  %83 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %21, i32 %79, i32 %55, i32 0)
  %84 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %21, i32 %79, i32 %58, i32 0)
  %85 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %21, i32 %80, i32 %55, i32 0)
  %86 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %21, i32 %80, i32 %58, i32 0)
  %87 = shl nuw nsw i32 %75, 8
  %88 = shl nuw nsw i32 %76, 4
  %89 = or disjoint i32 %87, %88
  %90 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %89, i32 %39, i32 0)
  %91 = or disjoint i32 %89, 1024
  %92 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %91, i32 %39, i32 0)
  %93 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %89, i32 %42, i32 0)
  %94 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %91, i32 %42, i32 0)
  %95 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %89, i32 %45, i32 0)
  %96 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %91, i32 %45, i32 0)
  %97 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %89, i32 %48, i32 0)
  %98 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %91, i32 %48, i32 0)
  %99 = or disjoint i32 %89, 2048
  %100 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %99, i32 %39, i32 0)
  %101 = or disjoint i32 %89, 3072
  %102 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %101, i32 %39, i32 0)
  %103 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %99, i32 %42, i32 0)
  %104 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %101, i32 %42, i32 0)
  %105 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %99, i32 %45, i32 0)
  %106 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %101, i32 %45, i32 0)
  %107 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %99, i32 %48, i32 0)
  %108 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %23, i32 %101, i32 %48, i32 0)
  tail call void asm sideeffect "s_waitcnt vmcnt(23)", "~{memory}"() #10, !srcloc !11
  tail call void @llvm.amdgcn.s.barrier()
  %109 = and i32 %13, 48
  %110 = shl nuw nsw i32 %13, 3
  %111 = and i32 %110, 112
  %112 = getelementptr [128 x i8], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 %76
  %113 = xor i32 %111, %109
  %114 = getelementptr i8, ptr addrspace(3) %112, i32 %113
  %115 = load <4 x i32>, ptr addrspace(3) %114, align 16, !tbaa !12
  %116 = getelementptr i8, ptr addrspace(3) %114, i32 2048
  %117 = load <4 x i32>, ptr addrspace(3) %116, align 16, !tbaa !12
  %118 = or disjoint i32 %109, 64
  %119 = xor i32 %118, %111
  %120 = getelementptr i8, ptr addrspace(3) %112, i32 %119
  %121 = load <4 x i32>, ptr addrspace(3) %120, align 16, !tbaa !12
  %122 = getelementptr i8, ptr addrspace(3) %120, i32 2048
  %123 = load <4 x i32>, ptr addrspace(3) %122, align 16, !tbaa !12
  %124 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %115, <4 x i32> noundef %90, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %81, i32 noundef 0, i32 noundef %83) #10
  %125 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %117, <4 x i32> noundef %90, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %81, i32 noundef 0, i32 noundef %83) #10
  %126 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %121, <4 x i32> noundef %92, <4 x float> noundef %124, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %81, i32 noundef 2, i32 noundef %83) #10
  %127 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %123, <4 x i32> noundef %92, <4 x float> noundef %125, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %81, i32 noundef 2, i32 noundef %83) #10
  %128 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %115, <4 x i32> noundef %93, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %81, i32 noundef 1, i32 noundef %83) #10
  %129 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %117, <4 x i32> noundef %93, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %81, i32 noundef 1, i32 noundef %83) #10
  %130 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %121, <4 x i32> noundef %94, <4 x float> noundef %128, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %81, i32 noundef 3, i32 noundef %83) #10
  %131 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %123, <4 x i32> noundef %94, <4 x float> noundef %129, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %81, i32 noundef 3, i32 noundef %83) #10
  %132 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %115, <4 x i32> noundef %95, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %81, i32 noundef 0, i32 noundef %84) #10
  %133 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %117, <4 x i32> noundef %95, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %81, i32 noundef 0, i32 noundef %84) #10
  %134 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %121, <4 x i32> noundef %96, <4 x float> noundef %132, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %81, i32 noundef 2, i32 noundef %84) #10
  %135 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %123, <4 x i32> noundef %96, <4 x float> noundef %133, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %81, i32 noundef 2, i32 noundef %84) #10
  %136 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %115, <4 x i32> noundef %97, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %81, i32 noundef 1, i32 noundef %84) #10
  %137 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %117, <4 x i32> noundef %97, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %81, i32 noundef 1, i32 noundef %84) #10
  %138 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %121, <4 x i32> noundef %98, <4 x float> noundef %136, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %81, i32 noundef 3, i32 noundef %84) #10
  %139 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %123, <4 x i32> noundef %98, <4 x float> noundef %137, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %81, i32 noundef 3, i32 noundef %84) #10
  tail call void asm sideeffect "s_waitcnt vmcnt(22)", "~{memory}"() #10, !srcloc !13
  tail call void @llvm.amdgcn.s.barrier()
  %140 = getelementptr [128 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 4096), i32 %76
  %141 = getelementptr i8, ptr addrspace(3) %140, i32 %113
  %142 = load <4 x i32>, ptr addrspace(3) %141, align 16, !tbaa !12
  %143 = getelementptr i8, ptr addrspace(3) %141, i32 2048
  %144 = load <4 x i32>, ptr addrspace(3) %143, align 16, !tbaa !12
  %145 = getelementptr i8, ptr addrspace(3) %140, i32 %119
  %146 = load <4 x i32>, ptr addrspace(3) %145, align 16, !tbaa !12
  %147 = getelementptr i8, ptr addrspace(3) %145, i32 2048
  %148 = load <4 x i32>, ptr addrspace(3) %147, align 16, !tbaa !12
  %149 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %142, <4 x i32> noundef %100, <4 x float> noundef %126, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %82, i32 noundef 0, i32 noundef %85) #10
  %150 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %144, <4 x i32> noundef %100, <4 x float> noundef %127, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %82, i32 noundef 0, i32 noundef %85) #10
  %151 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %146, <4 x i32> noundef %102, <4 x float> noundef %149, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %82, i32 noundef 2, i32 noundef %85) #10
  %152 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %148, <4 x i32> noundef %102, <4 x float> noundef %150, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %82, i32 noundef 2, i32 noundef %85) #10
  %153 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %142, <4 x i32> noundef %103, <4 x float> noundef %130, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %82, i32 noundef 1, i32 noundef %85) #10
  %154 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %144, <4 x i32> noundef %103, <4 x float> noundef %131, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %82, i32 noundef 1, i32 noundef %85) #10
  %155 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %146, <4 x i32> noundef %104, <4 x float> noundef %153, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %82, i32 noundef 3, i32 noundef %85) #10
  %156 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %148, <4 x i32> noundef %104, <4 x float> noundef %154, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %82, i32 noundef 3, i32 noundef %85) #10
  %157 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %142, <4 x i32> noundef %105, <4 x float> noundef %134, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %82, i32 noundef 0, i32 noundef %86) #10
  %158 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %144, <4 x i32> noundef %105, <4 x float> noundef %135, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %82, i32 noundef 0, i32 noundef %86) #10
  %159 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %146, <4 x i32> noundef %106, <4 x float> noundef %157, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %82, i32 noundef 2, i32 noundef %86) #10
  %160 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %148, <4 x i32> noundef %106, <4 x float> noundef %158, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %82, i32 noundef 2, i32 noundef %86) #10
  %161 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %142, <4 x i32> noundef %107, <4 x float> noundef %138, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %82, i32 noundef 1, i32 noundef %86) #10
  %162 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %144, <4 x i32> noundef %107, <4 x float> noundef %139, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %82, i32 noundef 1, i32 noundef %86) #10
  %163 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %146, <4 x i32> noundef %108, <4 x float> noundef %161, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %82, i32 noundef 3, i32 noundef %86) #10
  %164 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %148, <4 x i32> noundef %108, <4 x float> noundef %162, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %82, i32 noundef 3, i32 noundef %86) #10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !19)
  %165 = shl nuw nsw i32 %75, 12
  %166 = shl i32 %26, 8
  %167 = getelementptr i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 %166
  %168 = getelementptr float, ptr addrspace(3) %167, i32 %76
  %169 = extractelement <4 x float> %151, i64 0
  %170 = getelementptr i8, ptr addrspace(3) %168, i32 %165
  store float %169, ptr addrspace(3) %170, align 4, !tbaa !21, !noalias !23
  %171 = extractelement <4 x float> %151, i64 1
  %172 = or disjoint i32 %165, 1024
  %173 = getelementptr i8, ptr addrspace(3) %168, i32 %172
  store float %171, ptr addrspace(3) %173, align 4, !tbaa !21, !noalias !23
  %174 = extractelement <4 x float> %151, i64 2
  %175 = or disjoint i32 %165, 2048
  %176 = getelementptr i8, ptr addrspace(3) %168, i32 %175
  store float %174, ptr addrspace(3) %176, align 4, !tbaa !21, !noalias !23
  %177 = extractelement <4 x float> %151, i64 3
  %178 = or disjoint i32 %165, 3072
  %179 = getelementptr i8, ptr addrspace(3) %168, i32 %178
  store float %177, ptr addrspace(3) %179, align 4, !tbaa !21, !noalias !23
  %180 = getelementptr i8, ptr addrspace(3) %168, i32 64
  %181 = extractelement <4 x float> %155, i64 0
  %182 = getelementptr i8, ptr addrspace(3) %180, i32 %165
  store float %181, ptr addrspace(3) %182, align 4, !tbaa !21, !noalias !23
  %183 = extractelement <4 x float> %155, i64 1
  %184 = getelementptr i8, ptr addrspace(3) %180, i32 %172
  store float %183, ptr addrspace(3) %184, align 4, !tbaa !21, !noalias !23
  %185 = extractelement <4 x float> %155, i64 2
  %186 = getelementptr i8, ptr addrspace(3) %180, i32 %175
  store float %185, ptr addrspace(3) %186, align 4, !tbaa !21, !noalias !23
  %187 = extractelement <4 x float> %155, i64 3
  %188 = getelementptr i8, ptr addrspace(3) %180, i32 %178
  store float %187, ptr addrspace(3) %188, align 4, !tbaa !21, !noalias !23
  %189 = getelementptr i8, ptr addrspace(3) %168, i32 128
  %190 = extractelement <4 x float> %159, i64 0
  %191 = getelementptr i8, ptr addrspace(3) %189, i32 %165
  store float %190, ptr addrspace(3) %191, align 4, !tbaa !21, !noalias !23
  %192 = extractelement <4 x float> %159, i64 1
  %193 = getelementptr i8, ptr addrspace(3) %189, i32 %172
  store float %192, ptr addrspace(3) %193, align 4, !tbaa !21, !noalias !23
  %194 = extractelement <4 x float> %159, i64 2
  %195 = getelementptr i8, ptr addrspace(3) %189, i32 %175
  store float %194, ptr addrspace(3) %195, align 4, !tbaa !21, !noalias !23
  %196 = extractelement <4 x float> %159, i64 3
  %197 = getelementptr i8, ptr addrspace(3) %189, i32 %178
  store float %196, ptr addrspace(3) %197, align 4, !tbaa !21, !noalias !23
  %198 = getelementptr i8, ptr addrspace(3) %168, i32 192
  %199 = extractelement <4 x float> %163, i64 0
  %200 = getelementptr i8, ptr addrspace(3) %198, i32 %165
  store float %199, ptr addrspace(3) %200, align 4, !tbaa !21, !noalias !23
  %201 = extractelement <4 x float> %163, i64 1
  %202 = getelementptr i8, ptr addrspace(3) %198, i32 %172
  store float %201, ptr addrspace(3) %202, align 4, !tbaa !21, !noalias !23
  %203 = extractelement <4 x float> %163, i64 2
  %204 = getelementptr i8, ptr addrspace(3) %198, i32 %175
  store float %203, ptr addrspace(3) %204, align 4, !tbaa !21, !noalias !23
  %205 = extractelement <4 x float> %163, i64 3
  %206 = getelementptr i8, ptr addrspace(3) %198, i32 %178
  store float %205, ptr addrspace(3) %206, align 4, !tbaa !21, !noalias !23
  %207 = extractelement <4 x float> %152, i64 0
  %208 = or disjoint i32 %165, 16384
  %209 = getelementptr i8, ptr addrspace(3) %168, i32 %208
  store float %207, ptr addrspace(3) %209, align 4, !tbaa !21, !noalias !23
  %210 = extractelement <4 x float> %152, i64 1
  %211 = or disjoint i32 %165, 17408
  %212 = getelementptr i8, ptr addrspace(3) %168, i32 %211
  store float %210, ptr addrspace(3) %212, align 4, !tbaa !21, !noalias !23
  %213 = extractelement <4 x float> %152, i64 2
  %214 = or disjoint i32 %165, 18432
  %215 = getelementptr i8, ptr addrspace(3) %168, i32 %214
  store float %213, ptr addrspace(3) %215, align 4, !tbaa !21, !noalias !23
  %216 = extractelement <4 x float> %152, i64 3
  %217 = or disjoint i32 %165, 19456
  %218 = getelementptr i8, ptr addrspace(3) %168, i32 %217
  store float %216, ptr addrspace(3) %218, align 4, !tbaa !21, !noalias !23
  %219 = extractelement <4 x float> %156, i64 0
  %220 = getelementptr i8, ptr addrspace(3) %180, i32 %208
  store float %219, ptr addrspace(3) %220, align 4, !tbaa !21, !noalias !23
  %221 = extractelement <4 x float> %156, i64 1
  %222 = getelementptr i8, ptr addrspace(3) %180, i32 %211
  store float %221, ptr addrspace(3) %222, align 4, !tbaa !21, !noalias !23
  %223 = extractelement <4 x float> %156, i64 2
  %224 = getelementptr i8, ptr addrspace(3) %180, i32 %214
  store float %223, ptr addrspace(3) %224, align 4, !tbaa !21, !noalias !23
  %225 = extractelement <4 x float> %156, i64 3
  %226 = getelementptr i8, ptr addrspace(3) %180, i32 %217
  store float %225, ptr addrspace(3) %226, align 4, !tbaa !21, !noalias !23
  %227 = extractelement <4 x float> %160, i64 0
  %228 = getelementptr i8, ptr addrspace(3) %189, i32 %208
  store float %227, ptr addrspace(3) %228, align 4, !tbaa !21, !noalias !23
  %229 = extractelement <4 x float> %160, i64 1
  %230 = getelementptr i8, ptr addrspace(3) %189, i32 %211
  store float %229, ptr addrspace(3) %230, align 4, !tbaa !21, !noalias !23
  %231 = extractelement <4 x float> %160, i64 2
  %232 = getelementptr i8, ptr addrspace(3) %189, i32 %214
  store float %231, ptr addrspace(3) %232, align 4, !tbaa !21, !noalias !23
  %233 = extractelement <4 x float> %160, i64 3
  %234 = getelementptr i8, ptr addrspace(3) %189, i32 %217
  store float %233, ptr addrspace(3) %234, align 4, !tbaa !21, !noalias !23
  %235 = extractelement <4 x float> %164, i64 0
  %236 = getelementptr i8, ptr addrspace(3) %198, i32 %208
  store float %235, ptr addrspace(3) %236, align 4, !tbaa !21, !noalias !23
  %237 = extractelement <4 x float> %164, i64 1
  %238 = getelementptr i8, ptr addrspace(3) %198, i32 %211
  store float %237, ptr addrspace(3) %238, align 4, !tbaa !21, !noalias !23
  %239 = extractelement <4 x float> %164, i64 2
  %240 = getelementptr i8, ptr addrspace(3) %198, i32 %214
  store float %239, ptr addrspace(3) %240, align 4, !tbaa !21, !noalias !23
  %241 = extractelement <4 x float> %164, i64 3
  %242 = getelementptr i8, ptr addrspace(3) %198, i32 %217
  store float %241, ptr addrspace(3) %242, align 4, !tbaa !21, !noalias !23
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier(), !noalias !23
  fence syncscope("workgroup") acquire
  %243 = lshr i32 %13, 5
  %244 = shl nuw nsw i32 %13, 1
  %245 = and i32 %244, 62
  %246 = getelementptr float, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi32ELNS1_12EpilogPolicyE0ELb0ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 %245
  %247 = shl nsw i32 %29, 8
  %248 = sext i32 %247 to i64
  %249 = zext nneg i32 %245 to i64
  %250 = getelementptr %struct.__hip_bfloat16, ptr addrspace(1) %9, i64 %248
  %251 = getelementptr %struct.__hip_bfloat16, ptr addrspace(1) %250, i64 %249
  %252 = or disjoint i32 %59, %243
  %253 = sext i32 %252 to i64
  %254 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %253
  %255 = load i32, ptr addrspace(1) %254, align 4, !tbaa !7, !alias.scope !17, !noalias !24
  %256 = and i32 %255, 16777215
  %257 = icmp slt i32 %256, %8
  br i1 %257, label %258, label %290

258:                                              ; preds = %20
  %259 = getelementptr inbounds float, ptr addrspace(1) %7, i64 %253
  %260 = load float, ptr addrspace(1) %259, align 4, !tbaa !21, !alias.scope !19, !noalias !25
  %261 = shl nuw nsw i32 %243, 10
  %262 = getelementptr i8, ptr addrspace(3) %246, i32 %261
  %263 = getelementptr i8, ptr addrspace(3) %262, i32 256
  %264 = getelementptr i8, ptr addrspace(3) %262, i32 512
  %265 = getelementptr i8, ptr addrspace(3) %262, i32 768
  %266 = zext nneg i32 %256 to i64
  %267 = mul nuw nsw i64 %266, 14336
  %268 = getelementptr i8, ptr addrspace(1) %251, i64 %267
  %269 = load <2 x float>, ptr addrspace(3) %262, align 8, !tbaa !21, !noalias !23
  %270 = insertelement <2 x float> poison, float %260, i64 0
  %271 = shufflevector <2 x float> %270, <2 x float> poison, <2 x i32> zeroinitializer
  %272 = fmul contract <2 x float> %271, %269
  %273 = fptrunc <2 x float> %272 to <2 x bfloat>
  %274 = atomicrmw fadd ptr addrspace(1) %268, <2 x bfloat> %273 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %275 = load <2 x float>, ptr addrspace(3) %263, align 8, !tbaa !21, !noalias !23
  %276 = fmul contract <2 x float> %271, %275
  %277 = fptrunc <2 x float> %276 to <2 x bfloat>
  %278 = getelementptr inbounds nuw i8, ptr addrspace(1) %268, i64 128
  %279 = atomicrmw fadd ptr addrspace(1) %278, <2 x bfloat> %277 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %280 = load <2 x float>, ptr addrspace(3) %264, align 8, !tbaa !21, !noalias !23
  %281 = fmul contract <2 x float> %271, %280
  %282 = fptrunc <2 x float> %281 to <2 x bfloat>
  %283 = getelementptr inbounds nuw i8, ptr addrspace(1) %268, i64 256
  %284 = atomicrmw fadd ptr addrspace(1) %283, <2 x bfloat> %282 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %285 = load <2 x float>, ptr addrspace(3) %265, align 8, !tbaa !21, !noalias !23
  %286 = fmul contract <2 x float> %271, %285
  %287 = fptrunc <2 x float> %286 to <2 x bfloat>
  %288 = getelementptr inbounds nuw i8, ptr addrspace(1) %268, i64 384
  %289 = atomicrmw fadd ptr addrspace(1) %288, <2 x bfloat> %287 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  br label %290

290:                                              ; preds = %258, %20
  %291 = or disjoint i32 %243, 8
  %292 = or disjoint i32 %291, %59
  %293 = sext i32 %292 to i64
  %294 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %293
  %295 = load i32, ptr addrspace(1) %294, align 4, !tbaa !7, !alias.scope !17, !noalias !24
  %296 = and i32 %295, 16777215
  %297 = icmp slt i32 %296, %8
  br i1 %297, label %298, label %330

298:                                              ; preds = %290
  %299 = getelementptr inbounds float, ptr addrspace(1) %7, i64 %293
  %300 = load float, ptr addrspace(1) %299, align 4, !tbaa !21, !alias.scope !19, !noalias !25
  %301 = shl nuw nsw i32 %291, 10
  %302 = getelementptr i8, ptr addrspace(3) %246, i32 %301
  %303 = getelementptr i8, ptr addrspace(3) %302, i32 256
  %304 = getelementptr i8, ptr addrspace(3) %302, i32 512
  %305 = getelementptr i8, ptr addrspace(3) %302, i32 768
  %306 = zext nneg i32 %296 to i64
  %307 = mul nuw nsw i64 %306, 14336
  %308 = getelementptr i8, ptr addrspace(1) %251, i64 %307
  %309 = load <2 x float>, ptr addrspace(3) %302, align 8, !tbaa !21, !noalias !23
  %310 = insertelement <2 x float> poison, float %300, i64 0
  %311 = shufflevector <2 x float> %310, <2 x float> poison, <2 x i32> zeroinitializer
  %312 = fmul contract <2 x float> %311, %309
  %313 = fptrunc <2 x float> %312 to <2 x bfloat>
  %314 = atomicrmw fadd ptr addrspace(1) %308, <2 x bfloat> %313 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %315 = load <2 x float>, ptr addrspace(3) %303, align 8, !tbaa !21, !noalias !23
  %316 = fmul contract <2 x float> %311, %315
  %317 = fptrunc <2 x float> %316 to <2 x bfloat>
  %318 = getelementptr inbounds nuw i8, ptr addrspace(1) %308, i64 128
  %319 = atomicrmw fadd ptr addrspace(1) %318, <2 x bfloat> %317 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %320 = load <2 x float>, ptr addrspace(3) %304, align 8, !tbaa !21, !noalias !23
  %321 = fmul contract <2 x float> %311, %320
  %322 = fptrunc <2 x float> %321 to <2 x bfloat>
  %323 = getelementptr inbounds nuw i8, ptr addrspace(1) %308, i64 256
  %324 = atomicrmw fadd ptr addrspace(1) %323, <2 x bfloat> %322 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %325 = load <2 x float>, ptr addrspace(3) %305, align 8, !tbaa !21, !noalias !23
  %326 = fmul contract <2 x float> %311, %325
  %327 = fptrunc <2 x float> %326 to <2 x bfloat>
  %328 = getelementptr inbounds nuw i8, ptr addrspace(1) %308, i64 384
  %329 = atomicrmw fadd ptr addrspace(1) %328, <2 x bfloat> %327 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  br label %330

330:                                              ; preds = %298, %290
  %331 = or disjoint i32 %243, 16
  %332 = or disjoint i32 %331, %59
  %333 = sext i32 %332 to i64
  %334 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %333
  %335 = load i32, ptr addrspace(1) %334, align 4, !tbaa !7, !alias.scope !17, !noalias !24
  %336 = and i32 %335, 16777215
  %337 = icmp slt i32 %336, %8
  br i1 %337, label %338, label %370

338:                                              ; preds = %330
  %339 = getelementptr inbounds float, ptr addrspace(1) %7, i64 %333
  %340 = load float, ptr addrspace(1) %339, align 4, !tbaa !21, !alias.scope !19, !noalias !25
  %341 = shl nuw nsw i32 %331, 10
  %342 = getelementptr i8, ptr addrspace(3) %246, i32 %341
  %343 = getelementptr i8, ptr addrspace(3) %342, i32 256
  %344 = getelementptr i8, ptr addrspace(3) %342, i32 512
  %345 = getelementptr i8, ptr addrspace(3) %342, i32 768
  %346 = zext nneg i32 %336 to i64
  %347 = mul nuw nsw i64 %346, 14336
  %348 = getelementptr i8, ptr addrspace(1) %251, i64 %347
  %349 = load <2 x float>, ptr addrspace(3) %342, align 8, !tbaa !21, !noalias !23
  %350 = insertelement <2 x float> poison, float %340, i64 0
  %351 = shufflevector <2 x float> %350, <2 x float> poison, <2 x i32> zeroinitializer
  %352 = fmul contract <2 x float> %351, %349
  %353 = fptrunc <2 x float> %352 to <2 x bfloat>
  %354 = atomicrmw fadd ptr addrspace(1) %348, <2 x bfloat> %353 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %355 = load <2 x float>, ptr addrspace(3) %343, align 8, !tbaa !21, !noalias !23
  %356 = fmul contract <2 x float> %351, %355
  %357 = fptrunc <2 x float> %356 to <2 x bfloat>
  %358 = getelementptr inbounds nuw i8, ptr addrspace(1) %348, i64 128
  %359 = atomicrmw fadd ptr addrspace(1) %358, <2 x bfloat> %357 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %360 = load <2 x float>, ptr addrspace(3) %344, align 8, !tbaa !21, !noalias !23
  %361 = fmul contract <2 x float> %351, %360
  %362 = fptrunc <2 x float> %361 to <2 x bfloat>
  %363 = getelementptr inbounds nuw i8, ptr addrspace(1) %348, i64 256
  %364 = atomicrmw fadd ptr addrspace(1) %363, <2 x bfloat> %362 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %365 = load <2 x float>, ptr addrspace(3) %345, align 8, !tbaa !21, !noalias !23
  %366 = fmul contract <2 x float> %351, %365
  %367 = fptrunc <2 x float> %366 to <2 x bfloat>
  %368 = getelementptr inbounds nuw i8, ptr addrspace(1) %348, i64 384
  %369 = atomicrmw fadd ptr addrspace(1) %368, <2 x bfloat> %367 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  br label %370

370:                                              ; preds = %338, %330
  %371 = or disjoint i32 %243, 24
  %372 = or disjoint i32 %371, %59
  %373 = sext i32 %372 to i64
  %374 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %373
  %375 = load i32, ptr addrspace(1) %374, align 4, !tbaa !7, !alias.scope !17, !noalias !24
  %376 = and i32 %375, 16777215
  %377 = icmp slt i32 %376, %8
  br i1 %377, label %378, label %410

378:                                              ; preds = %370
  %379 = getelementptr inbounds float, ptr addrspace(1) %7, i64 %373
  %380 = load float, ptr addrspace(1) %379, align 4, !tbaa !21, !alias.scope !19, !noalias !25
  %381 = shl nuw nsw i32 %371, 10
  %382 = getelementptr i8, ptr addrspace(3) %246, i32 %381
  %383 = getelementptr i8, ptr addrspace(3) %382, i32 256
  %384 = getelementptr i8, ptr addrspace(3) %382, i32 512
  %385 = getelementptr i8, ptr addrspace(3) %382, i32 768
  %386 = zext nneg i32 %376 to i64
  %387 = mul nuw nsw i64 %386, 14336
  %388 = getelementptr i8, ptr addrspace(1) %251, i64 %387
  %389 = load <2 x float>, ptr addrspace(3) %382, align 8, !tbaa !21, !noalias !23
  %390 = insertelement <2 x float> poison, float %380, i64 0
  %391 = shufflevector <2 x float> %390, <2 x float> poison, <2 x i32> zeroinitializer
  %392 = fmul contract <2 x float> %391, %389
  %393 = fptrunc <2 x float> %392 to <2 x bfloat>
  %394 = atomicrmw fadd ptr addrspace(1) %388, <2 x bfloat> %393 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %395 = load <2 x float>, ptr addrspace(3) %383, align 8, !tbaa !21, !noalias !23
  %396 = fmul contract <2 x float> %391, %395
  %397 = fptrunc <2 x float> %396 to <2 x bfloat>
  %398 = getelementptr inbounds nuw i8, ptr addrspace(1) %388, i64 128
  %399 = atomicrmw fadd ptr addrspace(1) %398, <2 x bfloat> %397 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %400 = load <2 x float>, ptr addrspace(3) %384, align 8, !tbaa !21, !noalias !23
  %401 = fmul contract <2 x float> %391, %400
  %402 = fptrunc <2 x float> %401 to <2 x bfloat>
  %403 = getelementptr inbounds nuw i8, ptr addrspace(1) %388, i64 256
  %404 = atomicrmw fadd ptr addrspace(1) %403, <2 x bfloat> %402 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %405 = load <2 x float>, ptr addrspace(3) %385, align 8, !tbaa !21, !noalias !23
  %406 = fmul contract <2 x float> %391, %405
  %407 = fptrunc <2 x float> %406 to <2 x bfloat>
  %408 = getelementptr inbounds nuw i8, ptr addrspace(1) %388, i64 384
  %409 = atomicrmw fadd ptr addrspace(1) %408, <2 x bfloat> %407 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  br label %410

410:                                              ; preds = %378, %370, %11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #3

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #7

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #9

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="2" "denormal-fp-math-f32"="preserve-sign,preserve-sign" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #6 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}
!opencl.ocl.version = !{!6}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 1, !"Code Model", i32 4}
!5 = !{!"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.3 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"}
!6 = !{i32 2, i32 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{i64 7674843}
!12 = !{!9, !9, i64 0}
!13 = !{i64 7674938}
!14 = !{!15}
!15 = distinct !{!15, !16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi32EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf: argument 0"}
!16 = distinct !{!16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi32EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf"}
!17 = !{!18}
!18 = distinct !{!18, !16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi32EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf: argument 1"}
!19 = !{!20}
!20 = distinct !{!20, !16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi32EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf: argument 2"}
!21 = !{!22, !22, i64 0}
!22 = !{!"float", !9, i64 0}
!23 = !{!15, !18, !20}
!24 = !{!15, !20}
!25 = !{!15, !18}
!26 = !{!18, !20}
!27 = !{}
