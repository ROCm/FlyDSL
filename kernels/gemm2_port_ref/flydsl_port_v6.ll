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
  br i1 %48, label %49, label %767

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
  %61 = inttoptr i64 %60 to ptr addrspace(1)
  %62 = getelementptr i8, ptr addrspace(1) %61, i32 %59
  %63 = load i32, ptr addrspace(1) %62, align 4
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %63)
  %65 = mul i32 %58, 32
  %66 = addrspacecast ptr addrspace(1) %0 to ptr
  %67 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %66, i16 0, i64 167772160, i32 159744)
  %68 = addrspacecast ptr addrspace(1) %2 to ptr
  %69 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %68, i16 0, i64 10485760, i32 159744)
  %70 = addrspacecast ptr addrspace(1) %4 to ptr
  %71 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %70, i16 0, i64 706478080, i32 159744)
  %72 = addrspacecast ptr addrspace(1) %6 to ptr
  %73 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %72, i16 0, i64 44154880, i32 159744)
  %74 = sdiv i32 %26, 8
  %75 = mul i32 %74, 8
  %76 = icmp ne i32 %26, %75
  %77 = icmp slt i32 %26, 0
  %78 = icmp ne i1 %77, false
  %79 = and i1 %76, %78
  %80 = add i32 %74, -1
  %81 = select i1 %79, i32 %80, i32 %74
  %82 = srem i32 %26, 8
  %83 = sdiv i32 %26, 16
  %84 = mul i32 %83, 16
  %85 = icmp ne i32 %26, %84
  %86 = icmp slt i32 %26, 0
  %87 = icmp ne i1 %86, false
  %88 = and i1 %85, %87
  %89 = add i32 %83, -1
  %90 = select i1 %88, i32 %89, i32 %83
  %91 = srem i32 %26, 16
  %92 = mul i32 %64, 7168
  %93 = mul i32 %50, 256
  %94 = add i32 %92, %93
  %95 = mul i32 %35, 64
  %96 = add i32 %94, %95
  %97 = mul i32 %96, 256
  %98 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97)
  %99 = add i32 %96, 16
  %100 = mul i32 %99, 256
  %101 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %100)
  %102 = add i32 %96, 32
  %103 = mul i32 %102, 256
  %104 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %103)
  %105 = add i32 %96, 48
  %106 = mul i32 %105, 256
  %107 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106)
  %108 = mul i32 %50, 8
  %109 = mul i32 %35, 2
  %110 = add i32 %108, %109
  %111 = mul i32 %64, 28672
  %112 = mul i32 %110, 128
  %113 = add i32 %111, %112
  %114 = mul i32 %113, 4
  %115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %114)
  %116 = add i32 %110, 1
  %117 = mul i32 %116, 128
  %118 = add i32 %111, %117
  %119 = mul i32 %118, 4
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %119)
  %121 = sdiv i32 %65, 32
  %122 = mul i32 %121, 32
  %123 = icmp ne i32 %65, %122
  %124 = icmp slt i32 %65, 0
  %125 = icmp ne i1 %124, false
  %126 = and i1 %123, %125
  %127 = add i32 %121, -1
  %128 = select i1 %126, i32 %127, i32 %121
  %129 = mul i32 %128, 512
  %130 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129)
  %131 = mul i32 %35, 8
  %132 = add i32 %65, %131
  %133 = add i32 %132, %81
  %134 = add i32 %131, %81
  %135 = and i32 %134, 14
  %136 = shl i32 %135, 3
  %137 = mul i32 %82, 16
  %138 = xor i32 %137, %136
  %139 = mul i32 %133, 256
  %140 = add i32 %138, %139
  %141 = mul i32 %35, 1024
  %142 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %141
  %143 = sext i32 %142 to i64
  %144 = inttoptr i64 %143 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %67, ptr addrspace(3) %144, i32 16, i32 %140, i32 0, i32 0, i32 0)
  %145 = add i32 %141, 4096
  %146 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %145
  %147 = sext i32 %146 to i64
  %148 = inttoptr i64 %147 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %67, ptr addrspace(3) %148, i32 16, i32 %140, i32 128, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %149 = mul i32 %90, 16
  %150 = add i32 %149, %91
  %151 = mul i32 %150, 4
  %152 = sdiv i32 %151, 4
  %153 = mul i32 %152, 4
  %154 = icmp ne i32 %151, %153
  %155 = icmp slt i32 %151, 0
  %156 = icmp ne i1 %155, false
  %157 = and i1 %154, %156
  %158 = add i32 %152, -1
  %159 = select i1 %157, i32 %158, i32 %152
  %160 = mul i32 %159, 4
  %161 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %69, i32 %160, i32 %130, i32 0)
  %162 = add i32 %151, 256
  %163 = sdiv i32 %162, 4
  %164 = mul i32 %163, 4
  %165 = icmp ne i32 %162, %164
  %166 = icmp slt i32 %162, 0
  %167 = icmp ne i1 %166, false
  %168 = and i1 %165, %167
  %169 = add i32 %163, -1
  %170 = select i1 %168, i32 %169, i32 %163
  %171 = mul i32 %170, 4
  %172 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %69, i32 %171, i32 %130, i32 0)
  %173 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %160, i32 %115, i32 0)
  %174 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %160, i32 %120, i32 0)
  %175 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %171, i32 %115, i32 0)
  %176 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %171, i32 %120, i32 0)
  %177 = mul i32 %90, 256
  %178 = mul i32 %91, 16
  %179 = add i32 %177, %178
  %180 = sdiv i32 %179, 4
  %181 = mul i32 %180, 4
  %182 = icmp ne i32 %179, %181
  %183 = icmp slt i32 %179, 0
  %184 = icmp ne i1 %183, false
  %185 = and i1 %182, %184
  %186 = add i32 %180, -1
  %187 = select i1 %185, i32 %186, i32 %180
  %188 = mul i32 %187, 4
  %189 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %98, i32 0)
  %190 = add i32 %179, 1024
  %191 = sdiv i32 %190, 4
  %192 = mul i32 %191, 4
  %193 = icmp ne i32 %190, %192
  %194 = icmp slt i32 %190, 0
  %195 = icmp ne i1 %194, false
  %196 = and i1 %193, %195
  %197 = add i32 %191, -1
  %198 = select i1 %196, i32 %197, i32 %191
  %199 = mul i32 %198, 4
  %200 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %98, i32 0)
  %201 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %101, i32 0)
  %202 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %101, i32 0)
  %203 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %104, i32 0)
  %204 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %104, i32 0)
  %205 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %107, i32 0)
  %206 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %107, i32 0)
  %207 = add i32 %179, 2048
  %208 = sdiv i32 %207, 4
  %209 = mul i32 %208, 4
  %210 = icmp ne i32 %207, %209
  %211 = icmp slt i32 %207, 0
  %212 = icmp ne i1 %211, false
  %213 = and i1 %210, %212
  %214 = add i32 %208, -1
  %215 = select i1 %213, i32 %214, i32 %208
  %216 = mul i32 %215, 4
  %217 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %98, i32 0)
  %218 = add i32 %179, 3072
  %219 = sdiv i32 %218, 4
  %220 = mul i32 %219, 4
  %221 = icmp ne i32 %218, %220
  %222 = icmp slt i32 %218, 0
  %223 = icmp ne i1 %222, false
  %224 = and i1 %221, %223
  %225 = add i32 %219, -1
  %226 = select i1 %224, i32 %225, i32 %219
  %227 = mul i32 %226, 4
  %228 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %98, i32 0)
  %229 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %101, i32 0)
  %230 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %101, i32 0)
  %231 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %104, i32 0)
  %232 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %104, i32 0)
  %233 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %107, i32 0)
  %234 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %107, i32 0)
  call void asm sideeffect "s_waitcnt vmcnt(23)", ""()
  call void asm sideeffect "s_barrier", ""()
  %235 = and i32 %91, 14
  %236 = shl i32 %235, 3
  %237 = sext i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32) to i64
  %238 = inttoptr i64 %237 to ptr addrspace(3)
  %239 = xor i32 %149, %236
  %240 = mul i32 %91, 128
  %241 = add i32 %240, %239
  %242 = getelementptr i8, ptr addrspace(3) %238, i32 %241
  %243 = load <4 x i32>, ptr addrspace(3) %242, align 16
  %244 = add i32 %91, 16
  %245 = mul i32 %244, 128
  %246 = add i32 %245, %239
  %247 = getelementptr i8, ptr addrspace(3) %238, i32 %246
  %248 = load <4 x i32>, ptr addrspace(3) %247, align 16
  %249 = add i32 %149, 64
  %250 = xor i32 %249, %236
  %251 = add i32 %240, %250
  %252 = getelementptr i8, ptr addrspace(3) %238, i32 %251
  %253 = load <4 x i32>, ptr addrspace(3) %252, align 16
  %254 = add i32 %245, %250
  %255 = getelementptr i8, ptr addrspace(3) %238, i32 %254
  %256 = load <4 x i32>, ptr addrspace(3) %255, align 16
  %257 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %189, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 0, i32 %173)
  %258 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %189, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 0, i32 %173)
  %259 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %200, <4 x float> %257, i32 4, i32 4, i32 2, i32 %161, i32 2, i32 %173)
  %260 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %200, <4 x float> %258, i32 4, i32 4, i32 3, i32 %161, i32 2, i32 %173)
  %261 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %201, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 1, i32 %173)
  %262 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %201, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 1, i32 %173)
  %263 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %202, <4 x float> %261, i32 4, i32 4, i32 2, i32 %161, i32 3, i32 %173)
  %264 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %202, <4 x float> %262, i32 4, i32 4, i32 3, i32 %161, i32 3, i32 %173)
  %265 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %203, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 0, i32 %174)
  %266 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %203, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 0, i32 %174)
  %267 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %204, <4 x float> %265, i32 4, i32 4, i32 2, i32 %161, i32 2, i32 %174)
  %268 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %204, <4 x float> %266, i32 4, i32 4, i32 3, i32 %161, i32 2, i32 %174)
  %269 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %205, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 1, i32 %174)
  %270 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %205, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 1, i32 %174)
  %271 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %206, <4 x float> %269, i32 4, i32 4, i32 2, i32 %161, i32 3, i32 %174)
  %272 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %206, <4 x float> %270, i32 4, i32 4, i32 3, i32 %161, i32 3, i32 %174)
  call void asm sideeffect "s_waitcnt vmcnt(22)", ""()
  call void asm sideeffect "s_barrier", ""()
  %273 = add i32 %240, 4096
  %274 = add i32 %273, %239
  %275 = getelementptr i8, ptr addrspace(3) %238, i32 %274
  %276 = load <4 x i32>, ptr addrspace(3) %275, align 16
  %277 = add i32 %245, 4096
  %278 = add i32 %277, %239
  %279 = getelementptr i8, ptr addrspace(3) %238, i32 %278
  %280 = load <4 x i32>, ptr addrspace(3) %279, align 16
  %281 = add i32 %273, %250
  %282 = getelementptr i8, ptr addrspace(3) %238, i32 %281
  %283 = load <4 x i32>, ptr addrspace(3) %282, align 16
  %284 = add i32 %277, %250
  %285 = getelementptr i8, ptr addrspace(3) %238, i32 %284
  %286 = load <4 x i32>, ptr addrspace(3) %285, align 16
  %287 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %217, <4 x float> %259, i32 4, i32 4, i32 0, i32 %172, i32 0, i32 %175)
  %288 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %217, <4 x float> %260, i32 4, i32 4, i32 1, i32 %172, i32 0, i32 %175)
  %289 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %228, <4 x float> %287, i32 4, i32 4, i32 2, i32 %172, i32 2, i32 %175)
  %290 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %228, <4 x float> %288, i32 4, i32 4, i32 3, i32 %172, i32 2, i32 %175)
  %291 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %229, <4 x float> %263, i32 4, i32 4, i32 0, i32 %172, i32 1, i32 %175)
  %292 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %229, <4 x float> %264, i32 4, i32 4, i32 1, i32 %172, i32 1, i32 %175)
  %293 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %230, <4 x float> %291, i32 4, i32 4, i32 2, i32 %172, i32 3, i32 %175)
  %294 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %230, <4 x float> %292, i32 4, i32 4, i32 3, i32 %172, i32 3, i32 %175)
  %295 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %231, <4 x float> %267, i32 4, i32 4, i32 0, i32 %172, i32 0, i32 %176)
  %296 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %231, <4 x float> %268, i32 4, i32 4, i32 1, i32 %172, i32 0, i32 %176)
  %297 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %232, <4 x float> %295, i32 4, i32 4, i32 2, i32 %172, i32 2, i32 %176)
  %298 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %232, <4 x float> %296, i32 4, i32 4, i32 3, i32 %172, i32 2, i32 %176)
  %299 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %233, <4 x float> %271, i32 4, i32 4, i32 0, i32 %172, i32 1, i32 %176)
  %300 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %233, <4 x float> %272, i32 4, i32 4, i32 1, i32 %172, i32 1, i32 %176)
  %301 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %234, <4 x float> %299, i32 4, i32 4, i32 2, i32 %172, i32 3, i32 %176)
  %302 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %234, <4 x float> %300, i32 4, i32 4, i32 3, i32 %172, i32 3, i32 %176)
  %303 = sext i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32) to i64
  %304 = inttoptr i64 %303 to ptr addrspace(3)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %305 = mul i32 %90, 4
  %306 = add i32 %95, %91
  %307 = mul i32 %90, 1024
  %308 = add i32 %307, %306
  %309 = extractelement <4 x float> %289, i64 0
  %310 = mul i32 %308, 4
  %311 = getelementptr i8, ptr addrspace(3) %304, i32 %310
  store float %309, ptr addrspace(3) %311, align 4
  %312 = add i32 %305, 1
  %313 = mul i32 %312, 256
  %314 = add i32 %313, %306
  %315 = extractelement <4 x float> %289, i64 1
  %316 = mul i32 %314, 4
  %317 = getelementptr i8, ptr addrspace(3) %304, i32 %316
  store float %315, ptr addrspace(3) %317, align 4
  %318 = add i32 %305, 2
  %319 = mul i32 %318, 256
  %320 = add i32 %319, %306
  %321 = extractelement <4 x float> %289, i64 2
  %322 = mul i32 %320, 4
  %323 = getelementptr i8, ptr addrspace(3) %304, i32 %322
  store float %321, ptr addrspace(3) %323, align 4
  %324 = add i32 %305, 3
  %325 = mul i32 %324, 256
  %326 = add i32 %325, %306
  %327 = extractelement <4 x float> %289, i64 3
  %328 = mul i32 %326, 4
  %329 = getelementptr i8, ptr addrspace(3) %304, i32 %328
  store float %327, ptr addrspace(3) %329, align 4
  %330 = add i32 %95, 16
  %331 = add i32 %330, %91
  %332 = add i32 %307, %331
  %333 = extractelement <4 x float> %293, i64 0
  %334 = mul i32 %332, 4
  %335 = getelementptr i8, ptr addrspace(3) %304, i32 %334
  store float %333, ptr addrspace(3) %335, align 4
  %336 = add i32 %313, %331
  %337 = extractelement <4 x float> %293, i64 1
  %338 = mul i32 %336, 4
  %339 = getelementptr i8, ptr addrspace(3) %304, i32 %338
  store float %337, ptr addrspace(3) %339, align 4
  %340 = add i32 %319, %331
  %341 = extractelement <4 x float> %293, i64 2
  %342 = mul i32 %340, 4
  %343 = getelementptr i8, ptr addrspace(3) %304, i32 %342
  store float %341, ptr addrspace(3) %343, align 4
  %344 = add i32 %325, %331
  %345 = extractelement <4 x float> %293, i64 3
  %346 = mul i32 %344, 4
  %347 = getelementptr i8, ptr addrspace(3) %304, i32 %346
  store float %345, ptr addrspace(3) %347, align 4
  %348 = add i32 %95, 32
  %349 = add i32 %348, %91
  %350 = add i32 %307, %349
  %351 = extractelement <4 x float> %297, i64 0
  %352 = mul i32 %350, 4
  %353 = getelementptr i8, ptr addrspace(3) %304, i32 %352
  store float %351, ptr addrspace(3) %353, align 4
  %354 = add i32 %313, %349
  %355 = extractelement <4 x float> %297, i64 1
  %356 = mul i32 %354, 4
  %357 = getelementptr i8, ptr addrspace(3) %304, i32 %356
  store float %355, ptr addrspace(3) %357, align 4
  %358 = add i32 %319, %349
  %359 = extractelement <4 x float> %297, i64 2
  %360 = mul i32 %358, 4
  %361 = getelementptr i8, ptr addrspace(3) %304, i32 %360
  store float %359, ptr addrspace(3) %361, align 4
  %362 = add i32 %325, %349
  %363 = extractelement <4 x float> %297, i64 3
  %364 = mul i32 %362, 4
  %365 = getelementptr i8, ptr addrspace(3) %304, i32 %364
  store float %363, ptr addrspace(3) %365, align 4
  %366 = add i32 %95, 48
  %367 = add i32 %366, %91
  %368 = add i32 %307, %367
  %369 = extractelement <4 x float> %301, i64 0
  %370 = mul i32 %368, 4
  %371 = getelementptr i8, ptr addrspace(3) %304, i32 %370
  store float %369, ptr addrspace(3) %371, align 4
  %372 = add i32 %313, %367
  %373 = extractelement <4 x float> %301, i64 1
  %374 = mul i32 %372, 4
  %375 = getelementptr i8, ptr addrspace(3) %304, i32 %374
  store float %373, ptr addrspace(3) %375, align 4
  %376 = add i32 %319, %367
  %377 = extractelement <4 x float> %301, i64 2
  %378 = mul i32 %376, 4
  %379 = getelementptr i8, ptr addrspace(3) %304, i32 %378
  store float %377, ptr addrspace(3) %379, align 4
  %380 = add i32 %325, %367
  %381 = extractelement <4 x float> %301, i64 3
  %382 = mul i32 %380, 4
  %383 = getelementptr i8, ptr addrspace(3) %304, i32 %382
  store float %381, ptr addrspace(3) %383, align 4
  %384 = add i32 %305, 16
  %385 = mul i32 %384, 256
  %386 = add i32 %385, %306
  %387 = extractelement <4 x float> %290, i64 0
  %388 = mul i32 %386, 4
  %389 = getelementptr i8, ptr addrspace(3) %304, i32 %388
  store float %387, ptr addrspace(3) %389, align 4
  %390 = add i32 %305, 17
  %391 = mul i32 %390, 256
  %392 = add i32 %391, %306
  %393 = extractelement <4 x float> %290, i64 1
  %394 = mul i32 %392, 4
  %395 = getelementptr i8, ptr addrspace(3) %304, i32 %394
  store float %393, ptr addrspace(3) %395, align 4
  %396 = add i32 %305, 18
  %397 = mul i32 %396, 256
  %398 = add i32 %397, %306
  %399 = extractelement <4 x float> %290, i64 2
  %400 = mul i32 %398, 4
  %401 = getelementptr i8, ptr addrspace(3) %304, i32 %400
  store float %399, ptr addrspace(3) %401, align 4
  %402 = add i32 %305, 19
  %403 = mul i32 %402, 256
  %404 = add i32 %403, %306
  %405 = extractelement <4 x float> %290, i64 3
  %406 = mul i32 %404, 4
  %407 = getelementptr i8, ptr addrspace(3) %304, i32 %406
  store float %405, ptr addrspace(3) %407, align 4
  %408 = add i32 %385, %331
  %409 = extractelement <4 x float> %294, i64 0
  %410 = mul i32 %408, 4
  %411 = getelementptr i8, ptr addrspace(3) %304, i32 %410
  store float %409, ptr addrspace(3) %411, align 4
  %412 = add i32 %391, %331
  %413 = extractelement <4 x float> %294, i64 1
  %414 = mul i32 %412, 4
  %415 = getelementptr i8, ptr addrspace(3) %304, i32 %414
  store float %413, ptr addrspace(3) %415, align 4
  %416 = add i32 %397, %331
  %417 = extractelement <4 x float> %294, i64 2
  %418 = mul i32 %416, 4
  %419 = getelementptr i8, ptr addrspace(3) %304, i32 %418
  store float %417, ptr addrspace(3) %419, align 4
  %420 = add i32 %403, %331
  %421 = extractelement <4 x float> %294, i64 3
  %422 = mul i32 %420, 4
  %423 = getelementptr i8, ptr addrspace(3) %304, i32 %422
  store float %421, ptr addrspace(3) %423, align 4
  %424 = add i32 %385, %349
  %425 = extractelement <4 x float> %298, i64 0
  %426 = mul i32 %424, 4
  %427 = getelementptr i8, ptr addrspace(3) %304, i32 %426
  store float %425, ptr addrspace(3) %427, align 4
  %428 = add i32 %391, %349
  %429 = extractelement <4 x float> %298, i64 1
  %430 = mul i32 %428, 4
  %431 = getelementptr i8, ptr addrspace(3) %304, i32 %430
  store float %429, ptr addrspace(3) %431, align 4
  %432 = add i32 %397, %349
  %433 = extractelement <4 x float> %298, i64 2
  %434 = mul i32 %432, 4
  %435 = getelementptr i8, ptr addrspace(3) %304, i32 %434
  store float %433, ptr addrspace(3) %435, align 4
  %436 = add i32 %403, %349
  %437 = extractelement <4 x float> %298, i64 3
  %438 = mul i32 %436, 4
  %439 = getelementptr i8, ptr addrspace(3) %304, i32 %438
  store float %437, ptr addrspace(3) %439, align 4
  %440 = add i32 %385, %367
  %441 = extractelement <4 x float> %302, i64 0
  %442 = mul i32 %440, 4
  %443 = getelementptr i8, ptr addrspace(3) %304, i32 %442
  store float %441, ptr addrspace(3) %443, align 4
  %444 = add i32 %391, %367
  %445 = extractelement <4 x float> %302, i64 1
  %446 = mul i32 %444, 4
  %447 = getelementptr i8, ptr addrspace(3) %304, i32 %446
  store float %445, ptr addrspace(3) %447, align 4
  %448 = add i32 %397, %367
  %449 = extractelement <4 x float> %302, i64 2
  %450 = mul i32 %448, 4
  %451 = getelementptr i8, ptr addrspace(3) %304, i32 %450
  store float %449, ptr addrspace(3) %451, align 4
  %452 = add i32 %403, %367
  %453 = extractelement <4 x float> %302, i64 3
  %454 = mul i32 %452, 4
  %455 = getelementptr i8, ptr addrspace(3) %304, i32 %454
  store float %453, ptr addrspace(3) %455, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %456 = sdiv i32 %24, 32
  %457 = mul i32 %456, 32
  %458 = icmp ne i32 %24, %457
  %459 = icmp slt i32 %24, 0
  %460 = icmp ne i1 %459, false
  %461 = and i1 %458, %460
  %462 = add i32 %456, -1
  %463 = select i1 %461, i32 %462, i32 %456
  %464 = srem i32 %24, 32
  %465 = mul i32 %464, 2
  %466 = ptrtoint ptr addrspace(1) %12 to i64
  %467 = inttoptr i64 %466 to ptr addrspace(1)
  %468 = ptrtoint ptr addrspace(1) %14 to i64
  %469 = inttoptr i64 %468 to ptr addrspace(1)
  %470 = ptrtoint ptr addrspace(1) %17 to i64
  %471 = inttoptr i64 %470 to ptr addrspace(1)
  %472 = add i32 %65, %463
  %473 = mul i32 %472, 4
  %474 = getelementptr i8, ptr addrspace(1) %467, i32 %473
  %475 = load i32, ptr addrspace(1) %474, align 4
  %476 = and i32 %475, 16777215
  %477 = icmp slt i32 %476, %16
  %478 = getelementptr i8, ptr addrspace(1) %469, i32 %473
  %479 = load float, ptr addrspace(1) %478, align 4
  br i1 %477, label %480, label %544

480:                                              ; preds = %49
  %481 = mul i32 %476, 7168
  %482 = add i32 %481, %93
  %483 = add i32 %482, %465
  %484 = mul i32 %463, 256
  %485 = add i32 %484, %465
  %486 = mul i32 %485, 4
  %487 = getelementptr i8, ptr addrspace(3) %304, i32 %486
  %488 = load <2 x float>, ptr addrspace(3) %487, align 8
  %489 = extractelement <2 x float> %488, i64 0
  %490 = fmul float %489, %479
  %491 = extractelement <2 x float> %488, i64 1
  %492 = fmul float %491, %479
  %493 = insertelement <2 x float> poison, float %490, i64 0
  %494 = insertelement <2 x float> %493, float %492, i64 1
  %495 = fptrunc <2 x float> %494 to <2 x bfloat>
  %496 = mul i32 %483, 2
  %497 = getelementptr i8, ptr addrspace(1) %471, i32 %496
  %498 = atomicrmw fadd ptr addrspace(1) %497, <2 x bfloat> %495 syncscope("agent") monotonic, align 4
  %499 = add i32 %485, 64
  %500 = mul i32 %499, 4
  %501 = getelementptr i8, ptr addrspace(3) %304, i32 %500
  %502 = load <2 x float>, ptr addrspace(3) %501, align 8
  %503 = extractelement <2 x float> %502, i64 0
  %504 = fmul float %503, %479
  %505 = extractelement <2 x float> %502, i64 1
  %506 = fmul float %505, %479
  %507 = insertelement <2 x float> poison, float %504, i64 0
  %508 = insertelement <2 x float> %507, float %506, i64 1
  %509 = fptrunc <2 x float> %508 to <2 x bfloat>
  %510 = add i32 %483, 64
  %511 = mul i32 %510, 2
  %512 = getelementptr i8, ptr addrspace(1) %471, i32 %511
  %513 = atomicrmw fadd ptr addrspace(1) %512, <2 x bfloat> %509 syncscope("agent") monotonic, align 4
  %514 = add i32 %485, 128
  %515 = mul i32 %514, 4
  %516 = getelementptr i8, ptr addrspace(3) %304, i32 %515
  %517 = load <2 x float>, ptr addrspace(3) %516, align 8
  %518 = extractelement <2 x float> %517, i64 0
  %519 = fmul float %518, %479
  %520 = extractelement <2 x float> %517, i64 1
  %521 = fmul float %520, %479
  %522 = insertelement <2 x float> poison, float %519, i64 0
  %523 = insertelement <2 x float> %522, float %521, i64 1
  %524 = fptrunc <2 x float> %523 to <2 x bfloat>
  %525 = add i32 %483, 128
  %526 = mul i32 %525, 2
  %527 = getelementptr i8, ptr addrspace(1) %471, i32 %526
  %528 = atomicrmw fadd ptr addrspace(1) %527, <2 x bfloat> %524 syncscope("agent") monotonic, align 4
  %529 = add i32 %485, 192
  %530 = mul i32 %529, 4
  %531 = getelementptr i8, ptr addrspace(3) %304, i32 %530
  %532 = load <2 x float>, ptr addrspace(3) %531, align 8
  %533 = extractelement <2 x float> %532, i64 0
  %534 = fmul float %533, %479
  %535 = extractelement <2 x float> %532, i64 1
  %536 = fmul float %535, %479
  %537 = insertelement <2 x float> poison, float %534, i64 0
  %538 = insertelement <2 x float> %537, float %536, i64 1
  %539 = fptrunc <2 x float> %538 to <2 x bfloat>
  %540 = add i32 %483, 192
  %541 = mul i32 %540, 2
  %542 = getelementptr i8, ptr addrspace(1) %471, i32 %541
  %543 = atomicrmw fadd ptr addrspace(1) %542, <2 x bfloat> %539 syncscope("agent") monotonic, align 4
  br label %544

544:                                              ; preds = %480, %49
  %545 = add i32 %463, 8
  %546 = add i32 %65, %545
  %547 = mul i32 %546, 4
  %548 = getelementptr i8, ptr addrspace(1) %467, i32 %547
  %549 = load i32, ptr addrspace(1) %548, align 4
  %550 = and i32 %549, 16777215
  %551 = icmp slt i32 %550, %16
  %552 = getelementptr i8, ptr addrspace(1) %469, i32 %547
  %553 = load float, ptr addrspace(1) %552, align 4
  br i1 %551, label %554, label %618

554:                                              ; preds = %544
  %555 = mul i32 %550, 7168
  %556 = add i32 %555, %93
  %557 = add i32 %556, %465
  %558 = mul i32 %545, 256
  %559 = add i32 %558, %465
  %560 = mul i32 %559, 4
  %561 = getelementptr i8, ptr addrspace(3) %304, i32 %560
  %562 = load <2 x float>, ptr addrspace(3) %561, align 8
  %563 = extractelement <2 x float> %562, i64 0
  %564 = fmul float %563, %553
  %565 = extractelement <2 x float> %562, i64 1
  %566 = fmul float %565, %553
  %567 = insertelement <2 x float> poison, float %564, i64 0
  %568 = insertelement <2 x float> %567, float %566, i64 1
  %569 = fptrunc <2 x float> %568 to <2 x bfloat>
  %570 = mul i32 %557, 2
  %571 = getelementptr i8, ptr addrspace(1) %471, i32 %570
  %572 = atomicrmw fadd ptr addrspace(1) %571, <2 x bfloat> %569 syncscope("agent") monotonic, align 4
  %573 = add i32 %559, 64
  %574 = mul i32 %573, 4
  %575 = getelementptr i8, ptr addrspace(3) %304, i32 %574
  %576 = load <2 x float>, ptr addrspace(3) %575, align 8
  %577 = extractelement <2 x float> %576, i64 0
  %578 = fmul float %577, %553
  %579 = extractelement <2 x float> %576, i64 1
  %580 = fmul float %579, %553
  %581 = insertelement <2 x float> poison, float %578, i64 0
  %582 = insertelement <2 x float> %581, float %580, i64 1
  %583 = fptrunc <2 x float> %582 to <2 x bfloat>
  %584 = add i32 %557, 64
  %585 = mul i32 %584, 2
  %586 = getelementptr i8, ptr addrspace(1) %471, i32 %585
  %587 = atomicrmw fadd ptr addrspace(1) %586, <2 x bfloat> %583 syncscope("agent") monotonic, align 4
  %588 = add i32 %559, 128
  %589 = mul i32 %588, 4
  %590 = getelementptr i8, ptr addrspace(3) %304, i32 %589
  %591 = load <2 x float>, ptr addrspace(3) %590, align 8
  %592 = extractelement <2 x float> %591, i64 0
  %593 = fmul float %592, %553
  %594 = extractelement <2 x float> %591, i64 1
  %595 = fmul float %594, %553
  %596 = insertelement <2 x float> poison, float %593, i64 0
  %597 = insertelement <2 x float> %596, float %595, i64 1
  %598 = fptrunc <2 x float> %597 to <2 x bfloat>
  %599 = add i32 %557, 128
  %600 = mul i32 %599, 2
  %601 = getelementptr i8, ptr addrspace(1) %471, i32 %600
  %602 = atomicrmw fadd ptr addrspace(1) %601, <2 x bfloat> %598 syncscope("agent") monotonic, align 4
  %603 = add i32 %559, 192
  %604 = mul i32 %603, 4
  %605 = getelementptr i8, ptr addrspace(3) %304, i32 %604
  %606 = load <2 x float>, ptr addrspace(3) %605, align 8
  %607 = extractelement <2 x float> %606, i64 0
  %608 = fmul float %607, %553
  %609 = extractelement <2 x float> %606, i64 1
  %610 = fmul float %609, %553
  %611 = insertelement <2 x float> poison, float %608, i64 0
  %612 = insertelement <2 x float> %611, float %610, i64 1
  %613 = fptrunc <2 x float> %612 to <2 x bfloat>
  %614 = add i32 %557, 192
  %615 = mul i32 %614, 2
  %616 = getelementptr i8, ptr addrspace(1) %471, i32 %615
  %617 = atomicrmw fadd ptr addrspace(1) %616, <2 x bfloat> %613 syncscope("agent") monotonic, align 4
  br label %618

618:                                              ; preds = %554, %544
  %619 = add i32 %463, 16
  %620 = add i32 %65, %619
  %621 = mul i32 %620, 4
  %622 = getelementptr i8, ptr addrspace(1) %467, i32 %621
  %623 = load i32, ptr addrspace(1) %622, align 4
  %624 = and i32 %623, 16777215
  %625 = icmp slt i32 %624, %16
  %626 = getelementptr i8, ptr addrspace(1) %469, i32 %621
  %627 = load float, ptr addrspace(1) %626, align 4
  br i1 %625, label %628, label %692

628:                                              ; preds = %618
  %629 = mul i32 %624, 7168
  %630 = add i32 %629, %93
  %631 = add i32 %630, %465
  %632 = mul i32 %619, 256
  %633 = add i32 %632, %465
  %634 = mul i32 %633, 4
  %635 = getelementptr i8, ptr addrspace(3) %304, i32 %634
  %636 = load <2 x float>, ptr addrspace(3) %635, align 8
  %637 = extractelement <2 x float> %636, i64 0
  %638 = fmul float %637, %627
  %639 = extractelement <2 x float> %636, i64 1
  %640 = fmul float %639, %627
  %641 = insertelement <2 x float> poison, float %638, i64 0
  %642 = insertelement <2 x float> %641, float %640, i64 1
  %643 = fptrunc <2 x float> %642 to <2 x bfloat>
  %644 = mul i32 %631, 2
  %645 = getelementptr i8, ptr addrspace(1) %471, i32 %644
  %646 = atomicrmw fadd ptr addrspace(1) %645, <2 x bfloat> %643 syncscope("agent") monotonic, align 4
  %647 = add i32 %633, 64
  %648 = mul i32 %647, 4
  %649 = getelementptr i8, ptr addrspace(3) %304, i32 %648
  %650 = load <2 x float>, ptr addrspace(3) %649, align 8
  %651 = extractelement <2 x float> %650, i64 0
  %652 = fmul float %651, %627
  %653 = extractelement <2 x float> %650, i64 1
  %654 = fmul float %653, %627
  %655 = insertelement <2 x float> poison, float %652, i64 0
  %656 = insertelement <2 x float> %655, float %654, i64 1
  %657 = fptrunc <2 x float> %656 to <2 x bfloat>
  %658 = add i32 %631, 64
  %659 = mul i32 %658, 2
  %660 = getelementptr i8, ptr addrspace(1) %471, i32 %659
  %661 = atomicrmw fadd ptr addrspace(1) %660, <2 x bfloat> %657 syncscope("agent") monotonic, align 4
  %662 = add i32 %633, 128
  %663 = mul i32 %662, 4
  %664 = getelementptr i8, ptr addrspace(3) %304, i32 %663
  %665 = load <2 x float>, ptr addrspace(3) %664, align 8
  %666 = extractelement <2 x float> %665, i64 0
  %667 = fmul float %666, %627
  %668 = extractelement <2 x float> %665, i64 1
  %669 = fmul float %668, %627
  %670 = insertelement <2 x float> poison, float %667, i64 0
  %671 = insertelement <2 x float> %670, float %669, i64 1
  %672 = fptrunc <2 x float> %671 to <2 x bfloat>
  %673 = add i32 %631, 128
  %674 = mul i32 %673, 2
  %675 = getelementptr i8, ptr addrspace(1) %471, i32 %674
  %676 = atomicrmw fadd ptr addrspace(1) %675, <2 x bfloat> %672 syncscope("agent") monotonic, align 4
  %677 = add i32 %633, 192
  %678 = mul i32 %677, 4
  %679 = getelementptr i8, ptr addrspace(3) %304, i32 %678
  %680 = load <2 x float>, ptr addrspace(3) %679, align 8
  %681 = extractelement <2 x float> %680, i64 0
  %682 = fmul float %681, %627
  %683 = extractelement <2 x float> %680, i64 1
  %684 = fmul float %683, %627
  %685 = insertelement <2 x float> poison, float %682, i64 0
  %686 = insertelement <2 x float> %685, float %684, i64 1
  %687 = fptrunc <2 x float> %686 to <2 x bfloat>
  %688 = add i32 %631, 192
  %689 = mul i32 %688, 2
  %690 = getelementptr i8, ptr addrspace(1) %471, i32 %689
  %691 = atomicrmw fadd ptr addrspace(1) %690, <2 x bfloat> %687 syncscope("agent") monotonic, align 4
  br label %692

692:                                              ; preds = %628, %618
  %693 = add i32 %463, 24
  %694 = add i32 %65, %693
  %695 = mul i32 %694, 4
  %696 = getelementptr i8, ptr addrspace(1) %467, i32 %695
  %697 = load i32, ptr addrspace(1) %696, align 4
  %698 = and i32 %697, 16777215
  %699 = icmp slt i32 %698, %16
  %700 = getelementptr i8, ptr addrspace(1) %469, i32 %695
  %701 = load float, ptr addrspace(1) %700, align 4
  br i1 %699, label %702, label %766

702:                                              ; preds = %692
  %703 = mul i32 %698, 7168
  %704 = add i32 %703, %93
  %705 = add i32 %704, %465
  %706 = mul i32 %693, 256
  %707 = add i32 %706, %465
  %708 = mul i32 %707, 4
  %709 = getelementptr i8, ptr addrspace(3) %304, i32 %708
  %710 = load <2 x float>, ptr addrspace(3) %709, align 8
  %711 = extractelement <2 x float> %710, i64 0
  %712 = fmul float %711, %701
  %713 = extractelement <2 x float> %710, i64 1
  %714 = fmul float %713, %701
  %715 = insertelement <2 x float> poison, float %712, i64 0
  %716 = insertelement <2 x float> %715, float %714, i64 1
  %717 = fptrunc <2 x float> %716 to <2 x bfloat>
  %718 = mul i32 %705, 2
  %719 = getelementptr i8, ptr addrspace(1) %471, i32 %718
  %720 = atomicrmw fadd ptr addrspace(1) %719, <2 x bfloat> %717 syncscope("agent") monotonic, align 4
  %721 = add i32 %707, 64
  %722 = mul i32 %721, 4
  %723 = getelementptr i8, ptr addrspace(3) %304, i32 %722
  %724 = load <2 x float>, ptr addrspace(3) %723, align 8
  %725 = extractelement <2 x float> %724, i64 0
  %726 = fmul float %725, %701
  %727 = extractelement <2 x float> %724, i64 1
  %728 = fmul float %727, %701
  %729 = insertelement <2 x float> poison, float %726, i64 0
  %730 = insertelement <2 x float> %729, float %728, i64 1
  %731 = fptrunc <2 x float> %730 to <2 x bfloat>
  %732 = add i32 %705, 64
  %733 = mul i32 %732, 2
  %734 = getelementptr i8, ptr addrspace(1) %471, i32 %733
  %735 = atomicrmw fadd ptr addrspace(1) %734, <2 x bfloat> %731 syncscope("agent") monotonic, align 4
  %736 = add i32 %707, 128
  %737 = mul i32 %736, 4
  %738 = getelementptr i8, ptr addrspace(3) %304, i32 %737
  %739 = load <2 x float>, ptr addrspace(3) %738, align 8
  %740 = extractelement <2 x float> %739, i64 0
  %741 = fmul float %740, %701
  %742 = extractelement <2 x float> %739, i64 1
  %743 = fmul float %742, %701
  %744 = insertelement <2 x float> poison, float %741, i64 0
  %745 = insertelement <2 x float> %744, float %743, i64 1
  %746 = fptrunc <2 x float> %745 to <2 x bfloat>
  %747 = add i32 %705, 128
  %748 = mul i32 %747, 2
  %749 = getelementptr i8, ptr addrspace(1) %471, i32 %748
  %750 = atomicrmw fadd ptr addrspace(1) %749, <2 x bfloat> %746 syncscope("agent") monotonic, align 4
  %751 = add i32 %707, 192
  %752 = mul i32 %751, 4
  %753 = getelementptr i8, ptr addrspace(3) %304, i32 %752
  %754 = load <2 x float>, ptr addrspace(3) %753, align 8
  %755 = extractelement <2 x float> %754, i64 0
  %756 = fmul float %755, %701
  %757 = extractelement <2 x float> %754, i64 1
  %758 = fmul float %757, %701
  %759 = insertelement <2 x float> poison, float %756, i64 0
  %760 = insertelement <2 x float> %759, float %758, i64 1
  %761 = fptrunc <2 x float> %760 to <2 x bfloat>
  %762 = add i32 %705, 192
  %763 = mul i32 %762, 2
  %764 = getelementptr i8, ptr addrspace(1) %471, i32 %763
  %765 = atomicrmw fadd ptr addrspace(1) %764, <2 x bfloat> %761 syncscope("agent") monotonic, align 4
  br label %766

766:                                              ; preds = %702, %692
  br label %767

767:                                              ; preds = %766, %19
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

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
