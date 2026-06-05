; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@gemm2port_smem = external addrspace(3) global [32768 x i8], align 1024

define amdgpu_kernel void @gemm2_a4w4_port_bm32_atomic(ptr addrspace(1) %0, <{ <{ i32, i32 }>, <{ i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32 }>, <{ i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32 }>, <{ i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32 }> }> %9, ptr addrspace(1) %10, <{ <{ i32 }> }> %11, ptr addrspace(1) %12, <{ <{ i32 }> }> %13, ptr addrspace(1) %14, <{ <{ i32 }> }> %15, i32 %16, ptr addrspace(1) %17, <{ <{ i32, i32 }>, <{ i64 }> }> %18) #0 !reqd_work_group_size !1 {
  %20 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x()
  %21 = sext i32 %20 to i64
  %22 = call i32 @llvm.amdgcn.workgroup.id.x()
  %23 = sext i32 %22 to i64
  %24 = trunc i64 %21 to i32
  %25 = trunc i64 %23 to i32
  %26 = srem i32 %24, 64
  %27 = sdiv i32 %24, 64
  %28 = mul i32 %27, 64
  %29 = icmp ne i32 %24, %28
  %30 = icmp slt i32 %24, 0
  %31 = icmp ne i1 %30, false
  %32 = and i1 %29, %31
  %33 = add i32 %27, -1
  %34 = select i1 %32, i32 %33, i32 %27
  %35 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %34)
  %36 = ptrtoint ptr addrspace(1) %10 to i64
  %37 = inttoptr i64 %36 to ptr addrspace(1)
  %38 = load i32, ptr addrspace(1) %37, align 4
  %39 = sdiv i32 %38, 32
  %40 = mul i32 %39, 32
  %41 = icmp ne i32 %38, %40
  %42 = icmp slt i32 %38, 0
  %43 = icmp ne i1 %42, false
  %44 = and i1 %41, %43
  %45 = add i32 %39, -1
  %46 = select i1 %44, i32 %45, i32 %39
  %47 = mul i32 %46, 28
  %48 = icmp slt i32 %25, %47
  br i1 %48, label %49, label %856

49:                                               ; preds = %19
  %50 = srem i32 %25, 28
  %51 = sdiv i32 %25, 28
  %52 = mul i32 %51, 28
  %53 = icmp ne i32 %25, %52
  %54 = icmp slt i32 %25, 0
  %55 = icmp ne i1 %54, false
  %56 = and i1 %53, %55
  %57 = add i32 %51, -1
  %58 = select i1 %56, i32 %57, i32 %51
  %59 = mul i32 %58, 4
  %60 = ptrtoint ptr addrspace(1) %8 to i64
  %61 = sext i32 %59 to i64
  %62 = add i64 %60, %61
  %63 = inttoptr i64 %62 to ptr addrspace(1)
  %64 = load i32, ptr addrspace(1) %63, align 4
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64)
  %66 = mul i32 %58, 32
  %67 = addrspacecast ptr addrspace(1) %0 to ptr
  %68 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %67, i16 0, i64 167772160, i32 159744)
  %69 = addrspacecast ptr addrspace(1) %2 to ptr
  %70 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %69, i16 0, i64 10485760, i32 159744)
  %71 = addrspacecast ptr addrspace(1) %4 to ptr
  %72 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %71, i16 0, i64 706478080, i32 159744)
  %73 = addrspacecast ptr addrspace(1) %6 to ptr
  %74 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %73, i16 0, i64 44154880, i32 159744)
  %75 = sdiv i32 %26, 8
  %76 = mul i32 %75, 8
  %77 = icmp ne i32 %26, %76
  %78 = icmp slt i32 %26, 0
  %79 = icmp ne i1 %78, false
  %80 = and i1 %77, %79
  %81 = add i32 %75, -1
  %82 = select i1 %80, i32 %81, i32 %75
  %83 = srem i32 %26, 8
  %84 = sdiv i32 %26, 16
  %85 = mul i32 %84, 16
  %86 = icmp ne i32 %26, %85
  %87 = icmp slt i32 %26, 0
  %88 = icmp ne i1 %87, false
  %89 = and i1 %86, %88
  %90 = add i32 %84, -1
  %91 = select i1 %89, i32 %90, i32 %84
  %92 = srem i32 %26, 16
  %93 = mul i32 %65, 7168
  %94 = mul i32 %50, 256
  %95 = add i32 %93, %94
  %96 = mul i32 %35, 64
  %97 = add i32 %95, %96
  %98 = mul i32 %97, 256
  %99 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98)
  %100 = add i32 %97, 16
  %101 = mul i32 %100, 256
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101)
  %103 = add i32 %97, 32
  %104 = mul i32 %103, 256
  %105 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %104)
  %106 = add i32 %97, 48
  %107 = mul i32 %106, 256
  %108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %107)
  %109 = mul i32 %50, 8
  %110 = mul i32 %35, 2
  %111 = add i32 %109, %110
  %112 = mul i32 %65, 28672
  %113 = mul i32 %111, 128
  %114 = add i32 %112, %113
  %115 = mul i32 %114, 4
  %116 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115)
  %117 = add i32 %111, 1
  %118 = mul i32 %117, 128
  %119 = add i32 %112, %118
  %120 = mul i32 %119, 4
  %121 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %120)
  %122 = sdiv i32 %66, 32
  %123 = mul i32 %122, 32
  %124 = icmp ne i32 %66, %123
  %125 = icmp slt i32 %66, 0
  %126 = icmp ne i1 %125, false
  %127 = and i1 %124, %126
  %128 = add i32 %122, -1
  %129 = select i1 %127, i32 %128, i32 %122
  %130 = mul i32 %129, 512
  %131 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %130)
  %132 = mul i32 %35, 8
  %133 = add i32 %66, %132
  %134 = add i32 %133, %82
  %135 = add i32 %132, %82
  %136 = and i32 %135, 14
  %137 = shl i32 %136, 3
  %138 = mul i32 %83, 16
  %139 = xor i32 %138, %137
  %140 = mul i32 %134, 256
  %141 = add i32 %139, %140
  %142 = mul i32 %35, 1024
  %143 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %142
  %144 = sext i32 %143 to i64
  %145 = inttoptr i64 %144 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %68, ptr addrspace(3) %145, i32 16, i32 %141, i32 0, i32 0, i32 0)
  %146 = add i32 %142, 4096
  %147 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %146
  %148 = sext i32 %147 to i64
  %149 = inttoptr i64 %148 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %68, ptr addrspace(3) %149, i32 16, i32 %141, i32 128, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %150 = mul i32 %91, 16
  %151 = add i32 %150, %92
  %152 = mul i32 %151, 4
  %153 = sdiv i32 %152, 4
  %154 = mul i32 %153, 4
  %155 = icmp ne i32 %152, %154
  %156 = icmp slt i32 %152, 0
  %157 = icmp ne i1 %156, false
  %158 = and i1 %155, %157
  %159 = add i32 %153, -1
  %160 = select i1 %158, i32 %159, i32 %153
  %161 = mul i32 %160, 4
  %162 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %70, i32 %161, i32 %131, i32 0)
  %163 = add i32 %152, 256
  %164 = sdiv i32 %163, 4
  %165 = mul i32 %164, 4
  %166 = icmp ne i32 %163, %165
  %167 = icmp slt i32 %163, 0
  %168 = icmp ne i1 %167, false
  %169 = and i1 %166, %168
  %170 = add i32 %164, -1
  %171 = select i1 %169, i32 %170, i32 %164
  %172 = mul i32 %171, 4
  %173 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %70, i32 %172, i32 %131, i32 0)
  %174 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %161, i32 %116, i32 0)
  %175 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %161, i32 %121, i32 0)
  %176 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %172, i32 %116, i32 0)
  %177 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %172, i32 %121, i32 0)
  %178 = mul i32 %91, 256
  %179 = mul i32 %92, 16
  %180 = add i32 %178, %179
  %181 = sdiv i32 %180, 4
  %182 = mul i32 %181, 4
  %183 = icmp ne i32 %180, %182
  %184 = icmp slt i32 %180, 0
  %185 = icmp ne i1 %184, false
  %186 = and i1 %183, %185
  %187 = add i32 %181, -1
  %188 = select i1 %186, i32 %187, i32 %181
  %189 = mul i32 %188, 4
  %190 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %99, i32 0)
  %191 = add i32 %180, 1024
  %192 = sdiv i32 %191, 4
  %193 = mul i32 %192, 4
  %194 = icmp ne i32 %191, %193
  %195 = icmp slt i32 %191, 0
  %196 = icmp ne i1 %195, false
  %197 = and i1 %194, %196
  %198 = add i32 %192, -1
  %199 = select i1 %197, i32 %198, i32 %192
  %200 = mul i32 %199, 4
  %201 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %99, i32 0)
  %202 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %102, i32 0)
  %203 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %102, i32 0)
  %204 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %105, i32 0)
  %205 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %105, i32 0)
  %206 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %108, i32 0)
  %207 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %108, i32 0)
  %208 = add i32 %180, 2048
  %209 = sdiv i32 %208, 4
  %210 = mul i32 %209, 4
  %211 = icmp ne i32 %208, %210
  %212 = icmp slt i32 %208, 0
  %213 = icmp ne i1 %212, false
  %214 = and i1 %211, %213
  %215 = add i32 %209, -1
  %216 = select i1 %214, i32 %215, i32 %209
  %217 = mul i32 %216, 4
  %218 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %99, i32 0)
  %219 = add i32 %180, 3072
  %220 = sdiv i32 %219, 4
  %221 = mul i32 %220, 4
  %222 = icmp ne i32 %219, %221
  %223 = icmp slt i32 %219, 0
  %224 = icmp ne i1 %223, false
  %225 = and i1 %222, %224
  %226 = add i32 %220, -1
  %227 = select i1 %225, i32 %226, i32 %220
  %228 = mul i32 %227, 4
  %229 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %99, i32 0)
  %230 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %102, i32 0)
  %231 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %102, i32 0)
  %232 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %105, i32 0)
  %233 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %105, i32 0)
  %234 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %108, i32 0)
  %235 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %108, i32 0)
  call void asm sideeffect "s_waitcnt vmcnt(23)", ""()
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %236 = and i32 %92, 14
  %237 = shl i32 %236, 3
  %238 = xor i32 %150, %237
  %239 = mul i32 %92, 128
  %240 = add i32 %239, %238
  %241 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %240
  %242 = sext i32 %241 to i64
  %243 = inttoptr i64 %242 to ptr addrspace(3)
  %244 = load <4 x i32>, ptr addrspace(3) %243, align 16
  %245 = add i32 %92, 16
  %246 = mul i32 %245, 128
  %247 = add i32 %246, %238
  %248 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %247
  %249 = sext i32 %248 to i64
  %250 = inttoptr i64 %249 to ptr addrspace(3)
  %251 = load <4 x i32>, ptr addrspace(3) %250, align 16
  %252 = add i32 %150, 64
  %253 = xor i32 %252, %237
  %254 = add i32 %239, %253
  %255 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %254
  %256 = sext i32 %255 to i64
  %257 = inttoptr i64 %256 to ptr addrspace(3)
  %258 = load <4 x i32>, ptr addrspace(3) %257, align 16
  %259 = add i32 %246, %253
  %260 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %259
  %261 = sext i32 %260 to i64
  %262 = inttoptr i64 %261 to ptr addrspace(3)
  %263 = load <4 x i32>, ptr addrspace(3) %262, align 16
  %264 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %190, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 0, i32 %174)
  %265 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %251, <4 x i32> %190, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 0, i32 %174)
  %266 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %258, <4 x i32> %201, <4 x float> %264, i32 4, i32 4, i32 2, i32 %162, i32 2, i32 %174)
  %267 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %263, <4 x i32> %201, <4 x float> %265, i32 4, i32 4, i32 3, i32 %162, i32 2, i32 %174)
  %268 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %202, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 1, i32 %174)
  %269 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %251, <4 x i32> %202, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 1, i32 %174)
  %270 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %258, <4 x i32> %203, <4 x float> %268, i32 4, i32 4, i32 2, i32 %162, i32 3, i32 %174)
  %271 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %263, <4 x i32> %203, <4 x float> %269, i32 4, i32 4, i32 3, i32 %162, i32 3, i32 %174)
  %272 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %204, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 0, i32 %175)
  %273 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %251, <4 x i32> %204, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 0, i32 %175)
  %274 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %258, <4 x i32> %205, <4 x float> %272, i32 4, i32 4, i32 2, i32 %162, i32 2, i32 %175)
  %275 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %263, <4 x i32> %205, <4 x float> %273, i32 4, i32 4, i32 3, i32 %162, i32 2, i32 %175)
  %276 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %206, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 1, i32 %175)
  %277 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %251, <4 x i32> %206, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 1, i32 %175)
  %278 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %258, <4 x i32> %207, <4 x float> %276, i32 4, i32 4, i32 2, i32 %162, i32 3, i32 %175)
  %279 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %263, <4 x i32> %207, <4 x float> %277, i32 4, i32 4, i32 3, i32 %162, i32 3, i32 %175)
  call void asm sideeffect "s_waitcnt vmcnt(22)", ""()
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %280 = add i32 %239, 4096
  %281 = add i32 %280, %238
  %282 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %281
  %283 = sext i32 %282 to i64
  %284 = inttoptr i64 %283 to ptr addrspace(3)
  %285 = load <4 x i32>, ptr addrspace(3) %284, align 16
  %286 = add i32 %246, 4096
  %287 = add i32 %286, %238
  %288 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %287
  %289 = sext i32 %288 to i64
  %290 = inttoptr i64 %289 to ptr addrspace(3)
  %291 = load <4 x i32>, ptr addrspace(3) %290, align 16
  %292 = add i32 %280, %253
  %293 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %292
  %294 = sext i32 %293 to i64
  %295 = inttoptr i64 %294 to ptr addrspace(3)
  %296 = load <4 x i32>, ptr addrspace(3) %295, align 16
  %297 = add i32 %286, %253
  %298 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %297
  %299 = sext i32 %298 to i64
  %300 = inttoptr i64 %299 to ptr addrspace(3)
  %301 = load <4 x i32>, ptr addrspace(3) %300, align 16
  %302 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %285, <4 x i32> %218, <4 x float> %266, i32 4, i32 4, i32 0, i32 %173, i32 0, i32 %176)
  %303 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %291, <4 x i32> %218, <4 x float> %267, i32 4, i32 4, i32 1, i32 %173, i32 0, i32 %176)
  %304 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %296, <4 x i32> %229, <4 x float> %302, i32 4, i32 4, i32 2, i32 %173, i32 2, i32 %176)
  %305 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %301, <4 x i32> %229, <4 x float> %303, i32 4, i32 4, i32 3, i32 %173, i32 2, i32 %176)
  %306 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %285, <4 x i32> %230, <4 x float> %270, i32 4, i32 4, i32 0, i32 %173, i32 1, i32 %176)
  %307 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %291, <4 x i32> %230, <4 x float> %271, i32 4, i32 4, i32 1, i32 %173, i32 1, i32 %176)
  %308 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %296, <4 x i32> %231, <4 x float> %306, i32 4, i32 4, i32 2, i32 %173, i32 3, i32 %176)
  %309 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %301, <4 x i32> %231, <4 x float> %307, i32 4, i32 4, i32 3, i32 %173, i32 3, i32 %176)
  %310 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %285, <4 x i32> %232, <4 x float> %274, i32 4, i32 4, i32 0, i32 %173, i32 0, i32 %177)
  %311 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %291, <4 x i32> %232, <4 x float> %275, i32 4, i32 4, i32 1, i32 %173, i32 0, i32 %177)
  %312 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %296, <4 x i32> %233, <4 x float> %310, i32 4, i32 4, i32 2, i32 %173, i32 2, i32 %177)
  %313 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %301, <4 x i32> %233, <4 x float> %311, i32 4, i32 4, i32 3, i32 %173, i32 2, i32 %177)
  %314 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %285, <4 x i32> %234, <4 x float> %278, i32 4, i32 4, i32 0, i32 %173, i32 1, i32 %177)
  %315 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %291, <4 x i32> %234, <4 x float> %279, i32 4, i32 4, i32 1, i32 %173, i32 1, i32 %177)
  %316 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %296, <4 x i32> %235, <4 x float> %314, i32 4, i32 4, i32 2, i32 %173, i32 3, i32 %177)
  %317 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %301, <4 x i32> %235, <4 x float> %315, i32 4, i32 4, i32 3, i32 %173, i32 3, i32 %177)
  %318 = mul i32 %91, 4
  %319 = add i32 %96, %92
  %320 = mul i32 %91, 1024
  %321 = add i32 %320, %319
  %322 = extractelement <4 x float> %304, i64 0
  %323 = sext i32 %321 to i64
  %324 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %323
  store float %322, ptr addrspace(3) %324, align 4
  %325 = add i32 %318, 1
  %326 = mul i32 %325, 256
  %327 = add i32 %326, %319
  %328 = extractelement <4 x float> %304, i64 1
  %329 = sext i32 %327 to i64
  %330 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %329
  store float %328, ptr addrspace(3) %330, align 4
  %331 = add i32 %318, 2
  %332 = mul i32 %331, 256
  %333 = add i32 %332, %319
  %334 = extractelement <4 x float> %304, i64 2
  %335 = sext i32 %333 to i64
  %336 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %335
  store float %334, ptr addrspace(3) %336, align 4
  %337 = add i32 %318, 3
  %338 = mul i32 %337, 256
  %339 = add i32 %338, %319
  %340 = extractelement <4 x float> %304, i64 3
  %341 = sext i32 %339 to i64
  %342 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %341
  store float %340, ptr addrspace(3) %342, align 4
  %343 = add i32 %96, 16
  %344 = add i32 %343, %92
  %345 = add i32 %320, %344
  %346 = extractelement <4 x float> %308, i64 0
  %347 = sext i32 %345 to i64
  %348 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %347
  store float %346, ptr addrspace(3) %348, align 4
  %349 = add i32 %326, %344
  %350 = extractelement <4 x float> %308, i64 1
  %351 = sext i32 %349 to i64
  %352 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %351
  store float %350, ptr addrspace(3) %352, align 4
  %353 = add i32 %332, %344
  %354 = extractelement <4 x float> %308, i64 2
  %355 = sext i32 %353 to i64
  %356 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %355
  store float %354, ptr addrspace(3) %356, align 4
  %357 = add i32 %338, %344
  %358 = extractelement <4 x float> %308, i64 3
  %359 = sext i32 %357 to i64
  %360 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %359
  store float %358, ptr addrspace(3) %360, align 4
  %361 = add i32 %96, 32
  %362 = add i32 %361, %92
  %363 = add i32 %320, %362
  %364 = extractelement <4 x float> %312, i64 0
  %365 = sext i32 %363 to i64
  %366 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %365
  store float %364, ptr addrspace(3) %366, align 4
  %367 = add i32 %326, %362
  %368 = extractelement <4 x float> %312, i64 1
  %369 = sext i32 %367 to i64
  %370 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %369
  store float %368, ptr addrspace(3) %370, align 4
  %371 = add i32 %332, %362
  %372 = extractelement <4 x float> %312, i64 2
  %373 = sext i32 %371 to i64
  %374 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %373
  store float %372, ptr addrspace(3) %374, align 4
  %375 = add i32 %338, %362
  %376 = extractelement <4 x float> %312, i64 3
  %377 = sext i32 %375 to i64
  %378 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %377
  store float %376, ptr addrspace(3) %378, align 4
  %379 = add i32 %96, 48
  %380 = add i32 %379, %92
  %381 = add i32 %320, %380
  %382 = extractelement <4 x float> %316, i64 0
  %383 = sext i32 %381 to i64
  %384 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %383
  store float %382, ptr addrspace(3) %384, align 4
  %385 = add i32 %326, %380
  %386 = extractelement <4 x float> %316, i64 1
  %387 = sext i32 %385 to i64
  %388 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %387
  store float %386, ptr addrspace(3) %388, align 4
  %389 = add i32 %332, %380
  %390 = extractelement <4 x float> %316, i64 2
  %391 = sext i32 %389 to i64
  %392 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %391
  store float %390, ptr addrspace(3) %392, align 4
  %393 = add i32 %338, %380
  %394 = extractelement <4 x float> %316, i64 3
  %395 = sext i32 %393 to i64
  %396 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %395
  store float %394, ptr addrspace(3) %396, align 4
  %397 = add i32 %318, 16
  %398 = mul i32 %397, 256
  %399 = add i32 %398, %319
  %400 = extractelement <4 x float> %305, i64 0
  %401 = sext i32 %399 to i64
  %402 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %401
  store float %400, ptr addrspace(3) %402, align 4
  %403 = add i32 %318, 17
  %404 = mul i32 %403, 256
  %405 = add i32 %404, %319
  %406 = extractelement <4 x float> %305, i64 1
  %407 = sext i32 %405 to i64
  %408 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %407
  store float %406, ptr addrspace(3) %408, align 4
  %409 = add i32 %318, 18
  %410 = mul i32 %409, 256
  %411 = add i32 %410, %319
  %412 = extractelement <4 x float> %305, i64 2
  %413 = sext i32 %411 to i64
  %414 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %413
  store float %412, ptr addrspace(3) %414, align 4
  %415 = add i32 %318, 19
  %416 = mul i32 %415, 256
  %417 = add i32 %416, %319
  %418 = extractelement <4 x float> %305, i64 3
  %419 = sext i32 %417 to i64
  %420 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %419
  store float %418, ptr addrspace(3) %420, align 4
  %421 = add i32 %398, %344
  %422 = extractelement <4 x float> %309, i64 0
  %423 = sext i32 %421 to i64
  %424 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %423
  store float %422, ptr addrspace(3) %424, align 4
  %425 = add i32 %404, %344
  %426 = extractelement <4 x float> %309, i64 1
  %427 = sext i32 %425 to i64
  %428 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %427
  store float %426, ptr addrspace(3) %428, align 4
  %429 = add i32 %410, %344
  %430 = extractelement <4 x float> %309, i64 2
  %431 = sext i32 %429 to i64
  %432 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %431
  store float %430, ptr addrspace(3) %432, align 4
  %433 = add i32 %416, %344
  %434 = extractelement <4 x float> %309, i64 3
  %435 = sext i32 %433 to i64
  %436 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %435
  store float %434, ptr addrspace(3) %436, align 4
  %437 = add i32 %398, %362
  %438 = extractelement <4 x float> %313, i64 0
  %439 = sext i32 %437 to i64
  %440 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %439
  store float %438, ptr addrspace(3) %440, align 4
  %441 = add i32 %404, %362
  %442 = extractelement <4 x float> %313, i64 1
  %443 = sext i32 %441 to i64
  %444 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %443
  store float %442, ptr addrspace(3) %444, align 4
  %445 = add i32 %410, %362
  %446 = extractelement <4 x float> %313, i64 2
  %447 = sext i32 %445 to i64
  %448 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %447
  store float %446, ptr addrspace(3) %448, align 4
  %449 = add i32 %416, %362
  %450 = extractelement <4 x float> %313, i64 3
  %451 = sext i32 %449 to i64
  %452 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %451
  store float %450, ptr addrspace(3) %452, align 4
  %453 = add i32 %398, %380
  %454 = extractelement <4 x float> %317, i64 0
  %455 = sext i32 %453 to i64
  %456 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %455
  store float %454, ptr addrspace(3) %456, align 4
  %457 = add i32 %404, %380
  %458 = extractelement <4 x float> %317, i64 1
  %459 = sext i32 %457 to i64
  %460 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %459
  store float %458, ptr addrspace(3) %460, align 4
  %461 = add i32 %410, %380
  %462 = extractelement <4 x float> %317, i64 2
  %463 = sext i32 %461 to i64
  %464 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %463
  store float %462, ptr addrspace(3) %464, align 4
  %465 = add i32 %416, %380
  %466 = extractelement <4 x float> %317, i64 3
  %467 = sext i32 %465 to i64
  %468 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %467
  store float %466, ptr addrspace(3) %468, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %469 = sdiv i32 %24, 32
  %470 = mul i32 %469, 32
  %471 = icmp ne i32 %24, %470
  %472 = icmp slt i32 %24, 0
  %473 = icmp ne i1 %472, false
  %474 = and i1 %471, %473
  %475 = add i32 %469, -1
  %476 = select i1 %474, i32 %475, i32 %469
  %477 = srem i32 %24, 32
  %478 = mul i32 %477, 2
  %479 = add i32 %66, %476
  %480 = mul i32 %479, 4
  %481 = ptrtoint ptr addrspace(1) %12 to i64
  %482 = sext i32 %480 to i64
  %483 = add i64 %481, %482
  %484 = inttoptr i64 %483 to ptr addrspace(1)
  %485 = load i32, ptr addrspace(1) %484, align 4
  %486 = and i32 %485, 16777215
  %487 = icmp slt i32 %486, %16
  %488 = ptrtoint ptr addrspace(1) %14 to i64
  %489 = add i64 %488, %482
  %490 = inttoptr i64 %489 to ptr addrspace(1)
  %491 = load float, ptr addrspace(1) %490, align 4
  br i1 %487, label %492, label %573

492:                                              ; preds = %49
  %493 = mul i32 %486, 7168
  %494 = add i32 %493, %94
  %495 = add i32 %494, %478
  %496 = mul i32 %476, 256
  %497 = add i32 %496, %478
  %498 = sext i32 %497 to i64
  %499 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %498
  %500 = load float, ptr addrspace(3) %499, align 4
  %501 = fmul float %500, %491
  %502 = add i32 %497, 1
  %503 = sext i32 %502 to i64
  %504 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %503
  %505 = load float, ptr addrspace(3) %504, align 4
  %506 = fmul float %505, %491
  %507 = insertelement <2 x float> poison, float %501, i64 0
  %508 = insertelement <2 x float> %507, float %506, i64 1
  %509 = fptrunc <2 x float> %508 to <2 x bfloat>
  %510 = mul i32 %495, 2
  %511 = ptrtoint ptr addrspace(1) %17 to i64
  %512 = sext i32 %510 to i64
  %513 = add i64 %511, %512
  %514 = inttoptr i64 %513 to ptr addrspace(1)
  %515 = atomicrmw fadd ptr addrspace(1) %514, <2 x bfloat> %509 syncscope("agent") monotonic, align 4
  %516 = add i32 %497, 64
  %517 = sext i32 %516 to i64
  %518 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %517
  %519 = load float, ptr addrspace(3) %518, align 4
  %520 = fmul float %519, %491
  %521 = add i32 %497, 65
  %522 = sext i32 %521 to i64
  %523 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %522
  %524 = load float, ptr addrspace(3) %523, align 4
  %525 = fmul float %524, %491
  %526 = insertelement <2 x float> poison, float %520, i64 0
  %527 = insertelement <2 x float> %526, float %525, i64 1
  %528 = fptrunc <2 x float> %527 to <2 x bfloat>
  %529 = add i32 %495, 64
  %530 = mul i32 %529, 2
  %531 = sext i32 %530 to i64
  %532 = add i64 %511, %531
  %533 = inttoptr i64 %532 to ptr addrspace(1)
  %534 = atomicrmw fadd ptr addrspace(1) %533, <2 x bfloat> %528 syncscope("agent") monotonic, align 4
  %535 = add i32 %497, 128
  %536 = sext i32 %535 to i64
  %537 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %536
  %538 = load float, ptr addrspace(3) %537, align 4
  %539 = fmul float %538, %491
  %540 = add i32 %497, 129
  %541 = sext i32 %540 to i64
  %542 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %541
  %543 = load float, ptr addrspace(3) %542, align 4
  %544 = fmul float %543, %491
  %545 = insertelement <2 x float> poison, float %539, i64 0
  %546 = insertelement <2 x float> %545, float %544, i64 1
  %547 = fptrunc <2 x float> %546 to <2 x bfloat>
  %548 = add i32 %495, 128
  %549 = mul i32 %548, 2
  %550 = sext i32 %549 to i64
  %551 = add i64 %511, %550
  %552 = inttoptr i64 %551 to ptr addrspace(1)
  %553 = atomicrmw fadd ptr addrspace(1) %552, <2 x bfloat> %547 syncscope("agent") monotonic, align 4
  %554 = add i32 %497, 192
  %555 = sext i32 %554 to i64
  %556 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %555
  %557 = load float, ptr addrspace(3) %556, align 4
  %558 = fmul float %557, %491
  %559 = add i32 %497, 193
  %560 = sext i32 %559 to i64
  %561 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %560
  %562 = load float, ptr addrspace(3) %561, align 4
  %563 = fmul float %562, %491
  %564 = insertelement <2 x float> poison, float %558, i64 0
  %565 = insertelement <2 x float> %564, float %563, i64 1
  %566 = fptrunc <2 x float> %565 to <2 x bfloat>
  %567 = add i32 %495, 192
  %568 = mul i32 %567, 2
  %569 = sext i32 %568 to i64
  %570 = add i64 %511, %569
  %571 = inttoptr i64 %570 to ptr addrspace(1)
  %572 = atomicrmw fadd ptr addrspace(1) %571, <2 x bfloat> %566 syncscope("agent") monotonic, align 4
  br label %573

573:                                              ; preds = %492, %49
  %574 = add i32 %476, 8
  %575 = add i32 %66, %574
  %576 = mul i32 %575, 4
  %577 = sext i32 %576 to i64
  %578 = add i64 %481, %577
  %579 = inttoptr i64 %578 to ptr addrspace(1)
  %580 = load i32, ptr addrspace(1) %579, align 4
  %581 = and i32 %580, 16777215
  %582 = icmp slt i32 %581, %16
  %583 = add i64 %488, %577
  %584 = inttoptr i64 %583 to ptr addrspace(1)
  %585 = load float, ptr addrspace(1) %584, align 4
  br i1 %582, label %586, label %667

586:                                              ; preds = %573
  %587 = mul i32 %581, 7168
  %588 = add i32 %587, %94
  %589 = add i32 %588, %478
  %590 = mul i32 %574, 256
  %591 = add i32 %590, %478
  %592 = sext i32 %591 to i64
  %593 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %592
  %594 = load float, ptr addrspace(3) %593, align 4
  %595 = fmul float %594, %585
  %596 = add i32 %591, 1
  %597 = sext i32 %596 to i64
  %598 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %597
  %599 = load float, ptr addrspace(3) %598, align 4
  %600 = fmul float %599, %585
  %601 = insertelement <2 x float> poison, float %595, i64 0
  %602 = insertelement <2 x float> %601, float %600, i64 1
  %603 = fptrunc <2 x float> %602 to <2 x bfloat>
  %604 = mul i32 %589, 2
  %605 = ptrtoint ptr addrspace(1) %17 to i64
  %606 = sext i32 %604 to i64
  %607 = add i64 %605, %606
  %608 = inttoptr i64 %607 to ptr addrspace(1)
  %609 = atomicrmw fadd ptr addrspace(1) %608, <2 x bfloat> %603 syncscope("agent") monotonic, align 4
  %610 = add i32 %591, 64
  %611 = sext i32 %610 to i64
  %612 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %611
  %613 = load float, ptr addrspace(3) %612, align 4
  %614 = fmul float %613, %585
  %615 = add i32 %591, 65
  %616 = sext i32 %615 to i64
  %617 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %616
  %618 = load float, ptr addrspace(3) %617, align 4
  %619 = fmul float %618, %585
  %620 = insertelement <2 x float> poison, float %614, i64 0
  %621 = insertelement <2 x float> %620, float %619, i64 1
  %622 = fptrunc <2 x float> %621 to <2 x bfloat>
  %623 = add i32 %589, 64
  %624 = mul i32 %623, 2
  %625 = sext i32 %624 to i64
  %626 = add i64 %605, %625
  %627 = inttoptr i64 %626 to ptr addrspace(1)
  %628 = atomicrmw fadd ptr addrspace(1) %627, <2 x bfloat> %622 syncscope("agent") monotonic, align 4
  %629 = add i32 %591, 128
  %630 = sext i32 %629 to i64
  %631 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %630
  %632 = load float, ptr addrspace(3) %631, align 4
  %633 = fmul float %632, %585
  %634 = add i32 %591, 129
  %635 = sext i32 %634 to i64
  %636 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %635
  %637 = load float, ptr addrspace(3) %636, align 4
  %638 = fmul float %637, %585
  %639 = insertelement <2 x float> poison, float %633, i64 0
  %640 = insertelement <2 x float> %639, float %638, i64 1
  %641 = fptrunc <2 x float> %640 to <2 x bfloat>
  %642 = add i32 %589, 128
  %643 = mul i32 %642, 2
  %644 = sext i32 %643 to i64
  %645 = add i64 %605, %644
  %646 = inttoptr i64 %645 to ptr addrspace(1)
  %647 = atomicrmw fadd ptr addrspace(1) %646, <2 x bfloat> %641 syncscope("agent") monotonic, align 4
  %648 = add i32 %591, 192
  %649 = sext i32 %648 to i64
  %650 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %649
  %651 = load float, ptr addrspace(3) %650, align 4
  %652 = fmul float %651, %585
  %653 = add i32 %591, 193
  %654 = sext i32 %653 to i64
  %655 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %654
  %656 = load float, ptr addrspace(3) %655, align 4
  %657 = fmul float %656, %585
  %658 = insertelement <2 x float> poison, float %652, i64 0
  %659 = insertelement <2 x float> %658, float %657, i64 1
  %660 = fptrunc <2 x float> %659 to <2 x bfloat>
  %661 = add i32 %589, 192
  %662 = mul i32 %661, 2
  %663 = sext i32 %662 to i64
  %664 = add i64 %605, %663
  %665 = inttoptr i64 %664 to ptr addrspace(1)
  %666 = atomicrmw fadd ptr addrspace(1) %665, <2 x bfloat> %660 syncscope("agent") monotonic, align 4
  br label %667

667:                                              ; preds = %586, %573
  %668 = add i32 %476, 16
  %669 = add i32 %66, %668
  %670 = mul i32 %669, 4
  %671 = sext i32 %670 to i64
  %672 = add i64 %481, %671
  %673 = inttoptr i64 %672 to ptr addrspace(1)
  %674 = load i32, ptr addrspace(1) %673, align 4
  %675 = and i32 %674, 16777215
  %676 = icmp slt i32 %675, %16
  %677 = add i64 %488, %671
  %678 = inttoptr i64 %677 to ptr addrspace(1)
  %679 = load float, ptr addrspace(1) %678, align 4
  br i1 %676, label %680, label %761

680:                                              ; preds = %667
  %681 = mul i32 %675, 7168
  %682 = add i32 %681, %94
  %683 = add i32 %682, %478
  %684 = mul i32 %668, 256
  %685 = add i32 %684, %478
  %686 = sext i32 %685 to i64
  %687 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %686
  %688 = load float, ptr addrspace(3) %687, align 4
  %689 = fmul float %688, %679
  %690 = add i32 %685, 1
  %691 = sext i32 %690 to i64
  %692 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %691
  %693 = load float, ptr addrspace(3) %692, align 4
  %694 = fmul float %693, %679
  %695 = insertelement <2 x float> poison, float %689, i64 0
  %696 = insertelement <2 x float> %695, float %694, i64 1
  %697 = fptrunc <2 x float> %696 to <2 x bfloat>
  %698 = mul i32 %683, 2
  %699 = ptrtoint ptr addrspace(1) %17 to i64
  %700 = sext i32 %698 to i64
  %701 = add i64 %699, %700
  %702 = inttoptr i64 %701 to ptr addrspace(1)
  %703 = atomicrmw fadd ptr addrspace(1) %702, <2 x bfloat> %697 syncscope("agent") monotonic, align 4
  %704 = add i32 %685, 64
  %705 = sext i32 %704 to i64
  %706 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %705
  %707 = load float, ptr addrspace(3) %706, align 4
  %708 = fmul float %707, %679
  %709 = add i32 %685, 65
  %710 = sext i32 %709 to i64
  %711 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %710
  %712 = load float, ptr addrspace(3) %711, align 4
  %713 = fmul float %712, %679
  %714 = insertelement <2 x float> poison, float %708, i64 0
  %715 = insertelement <2 x float> %714, float %713, i64 1
  %716 = fptrunc <2 x float> %715 to <2 x bfloat>
  %717 = add i32 %683, 64
  %718 = mul i32 %717, 2
  %719 = sext i32 %718 to i64
  %720 = add i64 %699, %719
  %721 = inttoptr i64 %720 to ptr addrspace(1)
  %722 = atomicrmw fadd ptr addrspace(1) %721, <2 x bfloat> %716 syncscope("agent") monotonic, align 4
  %723 = add i32 %685, 128
  %724 = sext i32 %723 to i64
  %725 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %724
  %726 = load float, ptr addrspace(3) %725, align 4
  %727 = fmul float %726, %679
  %728 = add i32 %685, 129
  %729 = sext i32 %728 to i64
  %730 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %729
  %731 = load float, ptr addrspace(3) %730, align 4
  %732 = fmul float %731, %679
  %733 = insertelement <2 x float> poison, float %727, i64 0
  %734 = insertelement <2 x float> %733, float %732, i64 1
  %735 = fptrunc <2 x float> %734 to <2 x bfloat>
  %736 = add i32 %683, 128
  %737 = mul i32 %736, 2
  %738 = sext i32 %737 to i64
  %739 = add i64 %699, %738
  %740 = inttoptr i64 %739 to ptr addrspace(1)
  %741 = atomicrmw fadd ptr addrspace(1) %740, <2 x bfloat> %735 syncscope("agent") monotonic, align 4
  %742 = add i32 %685, 192
  %743 = sext i32 %742 to i64
  %744 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %743
  %745 = load float, ptr addrspace(3) %744, align 4
  %746 = fmul float %745, %679
  %747 = add i32 %685, 193
  %748 = sext i32 %747 to i64
  %749 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %748
  %750 = load float, ptr addrspace(3) %749, align 4
  %751 = fmul float %750, %679
  %752 = insertelement <2 x float> poison, float %746, i64 0
  %753 = insertelement <2 x float> %752, float %751, i64 1
  %754 = fptrunc <2 x float> %753 to <2 x bfloat>
  %755 = add i32 %683, 192
  %756 = mul i32 %755, 2
  %757 = sext i32 %756 to i64
  %758 = add i64 %699, %757
  %759 = inttoptr i64 %758 to ptr addrspace(1)
  %760 = atomicrmw fadd ptr addrspace(1) %759, <2 x bfloat> %754 syncscope("agent") monotonic, align 4
  br label %761

761:                                              ; preds = %680, %667
  %762 = add i32 %476, 24
  %763 = add i32 %66, %762
  %764 = mul i32 %763, 4
  %765 = sext i32 %764 to i64
  %766 = add i64 %481, %765
  %767 = inttoptr i64 %766 to ptr addrspace(1)
  %768 = load i32, ptr addrspace(1) %767, align 4
  %769 = and i32 %768, 16777215
  %770 = icmp slt i32 %769, %16
  %771 = add i64 %488, %765
  %772 = inttoptr i64 %771 to ptr addrspace(1)
  %773 = load float, ptr addrspace(1) %772, align 4
  br i1 %770, label %774, label %855

774:                                              ; preds = %761
  %775 = mul i32 %769, 7168
  %776 = add i32 %775, %94
  %777 = add i32 %776, %478
  %778 = mul i32 %762, 256
  %779 = add i32 %778, %478
  %780 = sext i32 %779 to i64
  %781 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %780
  %782 = load float, ptr addrspace(3) %781, align 4
  %783 = fmul float %782, %773
  %784 = add i32 %779, 1
  %785 = sext i32 %784 to i64
  %786 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %785
  %787 = load float, ptr addrspace(3) %786, align 4
  %788 = fmul float %787, %773
  %789 = insertelement <2 x float> poison, float %783, i64 0
  %790 = insertelement <2 x float> %789, float %788, i64 1
  %791 = fptrunc <2 x float> %790 to <2 x bfloat>
  %792 = mul i32 %777, 2
  %793 = ptrtoint ptr addrspace(1) %17 to i64
  %794 = sext i32 %792 to i64
  %795 = add i64 %793, %794
  %796 = inttoptr i64 %795 to ptr addrspace(1)
  %797 = atomicrmw fadd ptr addrspace(1) %796, <2 x bfloat> %791 syncscope("agent") monotonic, align 4
  %798 = add i32 %779, 64
  %799 = sext i32 %798 to i64
  %800 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %799
  %801 = load float, ptr addrspace(3) %800, align 4
  %802 = fmul float %801, %773
  %803 = add i32 %779, 65
  %804 = sext i32 %803 to i64
  %805 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %804
  %806 = load float, ptr addrspace(3) %805, align 4
  %807 = fmul float %806, %773
  %808 = insertelement <2 x float> poison, float %802, i64 0
  %809 = insertelement <2 x float> %808, float %807, i64 1
  %810 = fptrunc <2 x float> %809 to <2 x bfloat>
  %811 = add i32 %777, 64
  %812 = mul i32 %811, 2
  %813 = sext i32 %812 to i64
  %814 = add i64 %793, %813
  %815 = inttoptr i64 %814 to ptr addrspace(1)
  %816 = atomicrmw fadd ptr addrspace(1) %815, <2 x bfloat> %810 syncscope("agent") monotonic, align 4
  %817 = add i32 %779, 128
  %818 = sext i32 %817 to i64
  %819 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %818
  %820 = load float, ptr addrspace(3) %819, align 4
  %821 = fmul float %820, %773
  %822 = add i32 %779, 129
  %823 = sext i32 %822 to i64
  %824 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %823
  %825 = load float, ptr addrspace(3) %824, align 4
  %826 = fmul float %825, %773
  %827 = insertelement <2 x float> poison, float %821, i64 0
  %828 = insertelement <2 x float> %827, float %826, i64 1
  %829 = fptrunc <2 x float> %828 to <2 x bfloat>
  %830 = add i32 %777, 128
  %831 = mul i32 %830, 2
  %832 = sext i32 %831 to i64
  %833 = add i64 %793, %832
  %834 = inttoptr i64 %833 to ptr addrspace(1)
  %835 = atomicrmw fadd ptr addrspace(1) %834, <2 x bfloat> %829 syncscope("agent") monotonic, align 4
  %836 = add i32 %779, 192
  %837 = sext i32 %836 to i64
  %838 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %837
  %839 = load float, ptr addrspace(3) %838, align 4
  %840 = fmul float %839, %773
  %841 = add i32 %779, 193
  %842 = sext i32 %841 to i64
  %843 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %842
  %844 = load float, ptr addrspace(3) %843, align 4
  %845 = fmul float %844, %773
  %846 = insertelement <2 x float> poison, float %840, i64 0
  %847 = insertelement <2 x float> %846, float %845, i64 1
  %848 = fptrunc <2 x float> %847 to <2 x bfloat>
  %849 = add i32 %777, 192
  %850 = mul i32 %849, 2
  %851 = sext i32 %850 to i64
  %852 = add i64 %793, %851
  %853 = inttoptr i64 %852 to ptr addrspace(1)
  %854 = atomicrmw fadd ptr addrspace(1) %853, <2 x bfloat> %848 syncscope("agent") monotonic, align 4
  br label %855

855:                                              ; preds = %774, %761
  br label %856

856:                                              ; preds = %855, %19
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #7

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #7 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 256, i32 1, i32 1}
