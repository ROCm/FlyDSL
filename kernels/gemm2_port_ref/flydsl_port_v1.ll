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
  %36 = addrspacecast ptr addrspace(1) %10 to ptr
  %37 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %36, i16 0, i64 4294967295, i32 159744)
  %38 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %37, i32 0, i32 0, i32 0)
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
  br i1 %48, label %49, label %786

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
  %59 = addrspacecast ptr addrspace(1) %8 to ptr
  %60 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %59, i16 0, i64 4294967295, i32 159744)
  %61 = mul i32 %58, 4
  %62 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %60, i32 %61, i32 0, i32 0)
  %63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %62)
  %64 = mul i32 %58, 32
  %65 = addrspacecast ptr addrspace(1) %0 to ptr
  %66 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %65, i16 0, i64 167772160, i32 159744)
  %67 = addrspacecast ptr addrspace(1) %2 to ptr
  %68 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %67, i16 0, i64 10485760, i32 159744)
  %69 = addrspacecast ptr addrspace(1) %4 to ptr
  %70 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %69, i16 0, i64 706478080, i32 159744)
  %71 = addrspacecast ptr addrspace(1) %6 to ptr
  %72 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %71, i16 0, i64 44154880, i32 159744)
  %73 = sdiv i32 %26, 8
  %74 = mul i32 %73, 8
  %75 = icmp ne i32 %26, %74
  %76 = icmp slt i32 %26, 0
  %77 = icmp ne i1 %76, false
  %78 = and i1 %75, %77
  %79 = add i32 %73, -1
  %80 = select i1 %78, i32 %79, i32 %73
  %81 = srem i32 %26, 8
  %82 = sdiv i32 %26, 16
  %83 = mul i32 %82, 16
  %84 = icmp ne i32 %26, %83
  %85 = icmp slt i32 %26, 0
  %86 = icmp ne i1 %85, false
  %87 = and i1 %84, %86
  %88 = add i32 %82, -1
  %89 = select i1 %87, i32 %88, i32 %82
  %90 = srem i32 %26, 16
  %91 = mul i32 %63, 7168
  %92 = mul i32 %50, 256
  %93 = add i32 %91, %92
  %94 = mul i32 %35, 64
  %95 = add i32 %93, %94
  %96 = mul i32 %95, 256
  %97 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %96)
  %98 = add i32 %95, 16
  %99 = mul i32 %98, 256
  %100 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %99)
  %101 = add i32 %95, 32
  %102 = mul i32 %101, 256
  %103 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102)
  %104 = add i32 %95, 48
  %105 = mul i32 %104, 256
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105)
  %107 = mul i32 %50, 8
  %108 = mul i32 %35, 2
  %109 = add i32 %107, %108
  %110 = mul i32 %63, 28672
  %111 = mul i32 %109, 128
  %112 = add i32 %110, %111
  %113 = mul i32 %112, 4
  %114 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %113)
  %115 = add i32 %109, 1
  %116 = mul i32 %115, 128
  %117 = add i32 %110, %116
  %118 = mul i32 %117, 4
  %119 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118)
  %120 = sdiv i32 %64, 32
  %121 = mul i32 %120, 32
  %122 = icmp ne i32 %64, %121
  %123 = icmp slt i32 %64, 0
  %124 = icmp ne i1 %123, false
  %125 = and i1 %122, %124
  %126 = add i32 %120, -1
  %127 = select i1 %125, i32 %126, i32 %120
  %128 = mul i32 %127, 512
  %129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %128)
  %130 = mul i32 %35, 8
  %131 = add i32 %64, %130
  %132 = add i32 %131, %80
  %133 = add i32 %130, %80
  %134 = and i32 %133, 14
  %135 = shl i32 %134, 3
  %136 = mul i32 %81, 16
  %137 = xor i32 %136, %135
  %138 = mul i32 %132, 256
  %139 = add i32 %137, %138
  %140 = mul i32 %35, 1024
  %141 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %140
  %142 = sext i32 %141 to i64
  %143 = inttoptr i64 %142 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %66, ptr addrspace(3) %143, i32 16, i32 %139, i32 0, i32 0, i32 0)
  %144 = add i32 %140, 4096
  %145 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %144
  %146 = sext i32 %145 to i64
  %147 = inttoptr i64 %146 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %66, ptr addrspace(3) %147, i32 16, i32 %139, i32 128, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %148 = mul i32 %89, 16
  %149 = add i32 %148, %90
  %150 = mul i32 %149, 4
  %151 = sdiv i32 %150, 4
  %152 = mul i32 %151, 4
  %153 = icmp ne i32 %150, %152
  %154 = icmp slt i32 %150, 0
  %155 = icmp ne i1 %154, false
  %156 = and i1 %153, %155
  %157 = add i32 %151, -1
  %158 = select i1 %156, i32 %157, i32 %151
  %159 = mul i32 %158, 4
  %160 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %68, i32 %159, i32 %129, i32 0)
  %161 = add i32 %150, 256
  %162 = sdiv i32 %161, 4
  %163 = mul i32 %162, 4
  %164 = icmp ne i32 %161, %163
  %165 = icmp slt i32 %161, 0
  %166 = icmp ne i1 %165, false
  %167 = and i1 %164, %166
  %168 = add i32 %162, -1
  %169 = select i1 %167, i32 %168, i32 %162
  %170 = mul i32 %169, 4
  %171 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %68, i32 %170, i32 %129, i32 0)
  %172 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %72, i32 %159, i32 %114, i32 0)
  %173 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %72, i32 %159, i32 %119, i32 0)
  %174 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %72, i32 %170, i32 %114, i32 0)
  %175 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %72, i32 %170, i32 %119, i32 0)
  %176 = mul i32 %89, 256
  %177 = mul i32 %90, 16
  %178 = add i32 %176, %177
  %179 = sdiv i32 %178, 4
  %180 = mul i32 %179, 4
  %181 = icmp ne i32 %178, %180
  %182 = icmp slt i32 %178, 0
  %183 = icmp ne i1 %182, false
  %184 = and i1 %181, %183
  %185 = add i32 %179, -1
  %186 = select i1 %184, i32 %185, i32 %179
  %187 = mul i32 %186, 4
  %188 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %187, i32 %97, i32 0)
  %189 = add i32 %178, 1024
  %190 = sdiv i32 %189, 4
  %191 = mul i32 %190, 4
  %192 = icmp ne i32 %189, %191
  %193 = icmp slt i32 %189, 0
  %194 = icmp ne i1 %193, false
  %195 = and i1 %192, %194
  %196 = add i32 %190, -1
  %197 = select i1 %195, i32 %196, i32 %190
  %198 = mul i32 %197, 4
  %199 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %198, i32 %97, i32 0)
  %200 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %187, i32 %100, i32 0)
  %201 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %198, i32 %100, i32 0)
  %202 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %187, i32 %103, i32 0)
  %203 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %198, i32 %103, i32 0)
  %204 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %187, i32 %106, i32 0)
  %205 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %198, i32 %106, i32 0)
  %206 = add i32 %178, 2048
  %207 = sdiv i32 %206, 4
  %208 = mul i32 %207, 4
  %209 = icmp ne i32 %206, %208
  %210 = icmp slt i32 %206, 0
  %211 = icmp ne i1 %210, false
  %212 = and i1 %209, %211
  %213 = add i32 %207, -1
  %214 = select i1 %212, i32 %213, i32 %207
  %215 = mul i32 %214, 4
  %216 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %215, i32 %97, i32 0)
  %217 = add i32 %178, 3072
  %218 = sdiv i32 %217, 4
  %219 = mul i32 %218, 4
  %220 = icmp ne i32 %217, %219
  %221 = icmp slt i32 %217, 0
  %222 = icmp ne i1 %221, false
  %223 = and i1 %220, %222
  %224 = add i32 %218, -1
  %225 = select i1 %223, i32 %224, i32 %218
  %226 = mul i32 %225, 4
  %227 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %226, i32 %97, i32 0)
  %228 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %215, i32 %100, i32 0)
  %229 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %226, i32 %100, i32 0)
  %230 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %215, i32 %103, i32 0)
  %231 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %226, i32 %103, i32 0)
  %232 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %215, i32 %106, i32 0)
  %233 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %70, i32 %226, i32 %106, i32 0)
  call void asm sideeffect "s_waitcnt vmcnt(23)", ""()
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %234 = and i32 %90, 14
  %235 = shl i32 %234, 3
  %236 = xor i32 %148, %235
  %237 = mul i32 %90, 128
  %238 = add i32 %237, %236
  %239 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %238
  %240 = sext i32 %239 to i64
  %241 = inttoptr i64 %240 to ptr addrspace(3)
  %242 = load <4 x i32>, ptr addrspace(3) %241, align 16
  %243 = add i32 %90, 16
  %244 = mul i32 %243, 128
  %245 = add i32 %244, %236
  %246 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %245
  %247 = sext i32 %246 to i64
  %248 = inttoptr i64 %247 to ptr addrspace(3)
  %249 = load <4 x i32>, ptr addrspace(3) %248, align 16
  %250 = add i32 %148, 64
  %251 = xor i32 %250, %235
  %252 = add i32 %237, %251
  %253 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %252
  %254 = sext i32 %253 to i64
  %255 = inttoptr i64 %254 to ptr addrspace(3)
  %256 = load <4 x i32>, ptr addrspace(3) %255, align 16
  %257 = add i32 %244, %251
  %258 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %257
  %259 = sext i32 %258 to i64
  %260 = inttoptr i64 %259 to ptr addrspace(3)
  %261 = load <4 x i32>, ptr addrspace(3) %260, align 16
  %262 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %242, <4 x i32> %188, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %160, i32 0, i32 %172)
  %263 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %188, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %160, i32 0, i32 %172)
  %264 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %199, <4 x float> %262, i32 4, i32 4, i32 2, i32 %160, i32 2, i32 %172)
  %265 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %261, <4 x i32> %199, <4 x float> %263, i32 4, i32 4, i32 3, i32 %160, i32 2, i32 %172)
  %266 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %242, <4 x i32> %200, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %160, i32 1, i32 %172)
  %267 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %200, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %160, i32 1, i32 %172)
  %268 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %201, <4 x float> %266, i32 4, i32 4, i32 2, i32 %160, i32 3, i32 %172)
  %269 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %261, <4 x i32> %201, <4 x float> %267, i32 4, i32 4, i32 3, i32 %160, i32 3, i32 %172)
  %270 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %242, <4 x i32> %202, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %160, i32 0, i32 %173)
  %271 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %202, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %160, i32 0, i32 %173)
  %272 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %203, <4 x float> %270, i32 4, i32 4, i32 2, i32 %160, i32 2, i32 %173)
  %273 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %261, <4 x i32> %203, <4 x float> %271, i32 4, i32 4, i32 3, i32 %160, i32 2, i32 %173)
  %274 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %242, <4 x i32> %204, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %160, i32 1, i32 %173)
  %275 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %204, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %160, i32 1, i32 %173)
  %276 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %205, <4 x float> %274, i32 4, i32 4, i32 2, i32 %160, i32 3, i32 %173)
  %277 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %261, <4 x i32> %205, <4 x float> %275, i32 4, i32 4, i32 3, i32 %160, i32 3, i32 %173)
  call void asm sideeffect "s_waitcnt vmcnt(22)", ""()
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %278 = add i32 %237, 4096
  %279 = add i32 %278, %236
  %280 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %279
  %281 = sext i32 %280 to i64
  %282 = inttoptr i64 %281 to ptr addrspace(3)
  %283 = load <4 x i32>, ptr addrspace(3) %282, align 16
  %284 = add i32 %244, 4096
  %285 = add i32 %284, %236
  %286 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %285
  %287 = sext i32 %286 to i64
  %288 = inttoptr i64 %287 to ptr addrspace(3)
  %289 = load <4 x i32>, ptr addrspace(3) %288, align 16
  %290 = add i32 %278, %251
  %291 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %290
  %292 = sext i32 %291 to i64
  %293 = inttoptr i64 %292 to ptr addrspace(3)
  %294 = load <4 x i32>, ptr addrspace(3) %293, align 16
  %295 = add i32 %284, %251
  %296 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %295
  %297 = sext i32 %296 to i64
  %298 = inttoptr i64 %297 to ptr addrspace(3)
  %299 = load <4 x i32>, ptr addrspace(3) %298, align 16
  %300 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %216, <4 x float> %264, i32 4, i32 4, i32 0, i32 %171, i32 0, i32 %174)
  %301 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %289, <4 x i32> %216, <4 x float> %265, i32 4, i32 4, i32 1, i32 %171, i32 0, i32 %174)
  %302 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %294, <4 x i32> %227, <4 x float> %300, i32 4, i32 4, i32 2, i32 %171, i32 2, i32 %174)
  %303 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %299, <4 x i32> %227, <4 x float> %301, i32 4, i32 4, i32 3, i32 %171, i32 2, i32 %174)
  %304 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %228, <4 x float> %268, i32 4, i32 4, i32 0, i32 %171, i32 1, i32 %174)
  %305 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %289, <4 x i32> %228, <4 x float> %269, i32 4, i32 4, i32 1, i32 %171, i32 1, i32 %174)
  %306 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %294, <4 x i32> %229, <4 x float> %304, i32 4, i32 4, i32 2, i32 %171, i32 3, i32 %174)
  %307 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %299, <4 x i32> %229, <4 x float> %305, i32 4, i32 4, i32 3, i32 %171, i32 3, i32 %174)
  %308 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %230, <4 x float> %272, i32 4, i32 4, i32 0, i32 %171, i32 0, i32 %175)
  %309 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %289, <4 x i32> %230, <4 x float> %273, i32 4, i32 4, i32 1, i32 %171, i32 0, i32 %175)
  %310 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %294, <4 x i32> %231, <4 x float> %308, i32 4, i32 4, i32 2, i32 %171, i32 2, i32 %175)
  %311 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %299, <4 x i32> %231, <4 x float> %309, i32 4, i32 4, i32 3, i32 %171, i32 2, i32 %175)
  %312 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %232, <4 x float> %276, i32 4, i32 4, i32 0, i32 %171, i32 1, i32 %175)
  %313 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %289, <4 x i32> %232, <4 x float> %277, i32 4, i32 4, i32 1, i32 %171, i32 1, i32 %175)
  %314 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %294, <4 x i32> %233, <4 x float> %312, i32 4, i32 4, i32 2, i32 %171, i32 3, i32 %175)
  %315 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %299, <4 x i32> %233, <4 x float> %313, i32 4, i32 4, i32 3, i32 %171, i32 3, i32 %175)
  %316 = mul i32 %89, 4
  %317 = add i32 %94, %90
  %318 = mul i32 %89, 1024
  %319 = add i32 %318, %317
  %320 = extractelement <4 x float> %302, i64 0
  %321 = sext i32 %319 to i64
  %322 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %321
  store float %320, ptr addrspace(3) %322, align 4
  %323 = add i32 %316, 1
  %324 = mul i32 %323, 256
  %325 = add i32 %324, %317
  %326 = extractelement <4 x float> %302, i64 1
  %327 = sext i32 %325 to i64
  %328 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %327
  store float %326, ptr addrspace(3) %328, align 4
  %329 = add i32 %316, 2
  %330 = mul i32 %329, 256
  %331 = add i32 %330, %317
  %332 = extractelement <4 x float> %302, i64 2
  %333 = sext i32 %331 to i64
  %334 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %333
  store float %332, ptr addrspace(3) %334, align 4
  %335 = add i32 %316, 3
  %336 = mul i32 %335, 256
  %337 = add i32 %336, %317
  %338 = extractelement <4 x float> %302, i64 3
  %339 = sext i32 %337 to i64
  %340 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %339
  store float %338, ptr addrspace(3) %340, align 4
  %341 = add i32 %94, 16
  %342 = add i32 %341, %90
  %343 = add i32 %318, %342
  %344 = extractelement <4 x float> %306, i64 0
  %345 = sext i32 %343 to i64
  %346 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %345
  store float %344, ptr addrspace(3) %346, align 4
  %347 = add i32 %324, %342
  %348 = extractelement <4 x float> %306, i64 1
  %349 = sext i32 %347 to i64
  %350 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %349
  store float %348, ptr addrspace(3) %350, align 4
  %351 = add i32 %330, %342
  %352 = extractelement <4 x float> %306, i64 2
  %353 = sext i32 %351 to i64
  %354 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %353
  store float %352, ptr addrspace(3) %354, align 4
  %355 = add i32 %336, %342
  %356 = extractelement <4 x float> %306, i64 3
  %357 = sext i32 %355 to i64
  %358 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %357
  store float %356, ptr addrspace(3) %358, align 4
  %359 = add i32 %94, 32
  %360 = add i32 %359, %90
  %361 = add i32 %318, %360
  %362 = extractelement <4 x float> %310, i64 0
  %363 = sext i32 %361 to i64
  %364 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %363
  store float %362, ptr addrspace(3) %364, align 4
  %365 = add i32 %324, %360
  %366 = extractelement <4 x float> %310, i64 1
  %367 = sext i32 %365 to i64
  %368 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %367
  store float %366, ptr addrspace(3) %368, align 4
  %369 = add i32 %330, %360
  %370 = extractelement <4 x float> %310, i64 2
  %371 = sext i32 %369 to i64
  %372 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %371
  store float %370, ptr addrspace(3) %372, align 4
  %373 = add i32 %336, %360
  %374 = extractelement <4 x float> %310, i64 3
  %375 = sext i32 %373 to i64
  %376 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %375
  store float %374, ptr addrspace(3) %376, align 4
  %377 = add i32 %94, 48
  %378 = add i32 %377, %90
  %379 = add i32 %318, %378
  %380 = extractelement <4 x float> %314, i64 0
  %381 = sext i32 %379 to i64
  %382 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %381
  store float %380, ptr addrspace(3) %382, align 4
  %383 = add i32 %324, %378
  %384 = extractelement <4 x float> %314, i64 1
  %385 = sext i32 %383 to i64
  %386 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %385
  store float %384, ptr addrspace(3) %386, align 4
  %387 = add i32 %330, %378
  %388 = extractelement <4 x float> %314, i64 2
  %389 = sext i32 %387 to i64
  %390 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %389
  store float %388, ptr addrspace(3) %390, align 4
  %391 = add i32 %336, %378
  %392 = extractelement <4 x float> %314, i64 3
  %393 = sext i32 %391 to i64
  %394 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %393
  store float %392, ptr addrspace(3) %394, align 4
  %395 = add i32 %316, 16
  %396 = mul i32 %395, 256
  %397 = add i32 %396, %317
  %398 = extractelement <4 x float> %303, i64 0
  %399 = sext i32 %397 to i64
  %400 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %399
  store float %398, ptr addrspace(3) %400, align 4
  %401 = add i32 %316, 17
  %402 = mul i32 %401, 256
  %403 = add i32 %402, %317
  %404 = extractelement <4 x float> %303, i64 1
  %405 = sext i32 %403 to i64
  %406 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %405
  store float %404, ptr addrspace(3) %406, align 4
  %407 = add i32 %316, 18
  %408 = mul i32 %407, 256
  %409 = add i32 %408, %317
  %410 = extractelement <4 x float> %303, i64 2
  %411 = sext i32 %409 to i64
  %412 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %411
  store float %410, ptr addrspace(3) %412, align 4
  %413 = add i32 %316, 19
  %414 = mul i32 %413, 256
  %415 = add i32 %414, %317
  %416 = extractelement <4 x float> %303, i64 3
  %417 = sext i32 %415 to i64
  %418 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %417
  store float %416, ptr addrspace(3) %418, align 4
  %419 = add i32 %396, %342
  %420 = extractelement <4 x float> %307, i64 0
  %421 = sext i32 %419 to i64
  %422 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %421
  store float %420, ptr addrspace(3) %422, align 4
  %423 = add i32 %402, %342
  %424 = extractelement <4 x float> %307, i64 1
  %425 = sext i32 %423 to i64
  %426 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %425
  store float %424, ptr addrspace(3) %426, align 4
  %427 = add i32 %408, %342
  %428 = extractelement <4 x float> %307, i64 2
  %429 = sext i32 %427 to i64
  %430 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %429
  store float %428, ptr addrspace(3) %430, align 4
  %431 = add i32 %414, %342
  %432 = extractelement <4 x float> %307, i64 3
  %433 = sext i32 %431 to i64
  %434 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %433
  store float %432, ptr addrspace(3) %434, align 4
  %435 = add i32 %396, %360
  %436 = extractelement <4 x float> %311, i64 0
  %437 = sext i32 %435 to i64
  %438 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %437
  store float %436, ptr addrspace(3) %438, align 4
  %439 = add i32 %402, %360
  %440 = extractelement <4 x float> %311, i64 1
  %441 = sext i32 %439 to i64
  %442 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %441
  store float %440, ptr addrspace(3) %442, align 4
  %443 = add i32 %408, %360
  %444 = extractelement <4 x float> %311, i64 2
  %445 = sext i32 %443 to i64
  %446 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %445
  store float %444, ptr addrspace(3) %446, align 4
  %447 = add i32 %414, %360
  %448 = extractelement <4 x float> %311, i64 3
  %449 = sext i32 %447 to i64
  %450 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %449
  store float %448, ptr addrspace(3) %450, align 4
  %451 = add i32 %396, %378
  %452 = extractelement <4 x float> %315, i64 0
  %453 = sext i32 %451 to i64
  %454 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %453
  store float %452, ptr addrspace(3) %454, align 4
  %455 = add i32 %402, %378
  %456 = extractelement <4 x float> %315, i64 1
  %457 = sext i32 %455 to i64
  %458 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %457
  store float %456, ptr addrspace(3) %458, align 4
  %459 = add i32 %408, %378
  %460 = extractelement <4 x float> %315, i64 2
  %461 = sext i32 %459 to i64
  %462 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %461
  store float %460, ptr addrspace(3) %462, align 4
  %463 = add i32 %414, %378
  %464 = extractelement <4 x float> %315, i64 3
  %465 = sext i32 %463 to i64
  %466 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %465
  store float %464, ptr addrspace(3) %466, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %467 = sdiv i32 %24, 32
  %468 = mul i32 %467, 32
  %469 = icmp ne i32 %24, %468
  %470 = icmp slt i32 %24, 0
  %471 = icmp ne i1 %470, false
  %472 = and i1 %469, %471
  %473 = add i32 %467, -1
  %474 = select i1 %472, i32 %473, i32 %467
  %475 = srem i32 %24, 32
  %476 = mul i32 %475, 2
  %477 = addrspacecast ptr addrspace(1) %12 to ptr
  %478 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %477, i16 0, i64 4294967295, i32 159744)
  %479 = addrspacecast ptr addrspace(1) %14 to ptr
  %480 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %479, i16 0, i64 4294967295, i32 159744)
  %481 = addrspacecast ptr addrspace(1) %17 to ptr
  %482 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %481, i16 0, i64 4294967295, i32 159744)
  %483 = add i32 %64, %474
  %484 = mul i32 %483, 4
  %485 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %478, i32 %484, i32 0, i32 0)
  %486 = and i32 %485, 16777215
  %487 = icmp slt i32 %486, %16
  %488 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %480, i32 %484, i32 0, i32 0)
  br i1 %487, label %489, label %557

489:                                              ; preds = %49
  %490 = mul i32 %486, 7168
  %491 = add i32 %490, %92
  %492 = add i32 %491, %476
  %493 = mul i32 %474, 256
  %494 = add i32 %493, %476
  %495 = sext i32 %494 to i64
  %496 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %495
  %497 = load float, ptr addrspace(3) %496, align 4
  %498 = fmul float %497, %488
  %499 = add i32 %494, 1
  %500 = sext i32 %499 to i64
  %501 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %500
  %502 = load float, ptr addrspace(3) %501, align 4
  %503 = fmul float %502, %488
  %504 = insertelement <2 x float> poison, float %498, i64 0
  %505 = insertelement <2 x float> %504, float %503, i64 1
  %506 = fptrunc <2 x float> %505 to <2 x bfloat>
  %507 = mul i32 %492, 2
  %508 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %506, ptr addrspace(8) %482, i32 %507, i32 0, i32 0)
  %509 = add i32 %494, 64
  %510 = sext i32 %509 to i64
  %511 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %510
  %512 = load float, ptr addrspace(3) %511, align 4
  %513 = fmul float %512, %488
  %514 = add i32 %494, 65
  %515 = sext i32 %514 to i64
  %516 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %515
  %517 = load float, ptr addrspace(3) %516, align 4
  %518 = fmul float %517, %488
  %519 = insertelement <2 x float> poison, float %513, i64 0
  %520 = insertelement <2 x float> %519, float %518, i64 1
  %521 = fptrunc <2 x float> %520 to <2 x bfloat>
  %522 = add i32 %492, 64
  %523 = mul i32 %522, 2
  %524 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %521, ptr addrspace(8) %482, i32 %523, i32 0, i32 0)
  %525 = add i32 %494, 128
  %526 = sext i32 %525 to i64
  %527 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %526
  %528 = load float, ptr addrspace(3) %527, align 4
  %529 = fmul float %528, %488
  %530 = add i32 %494, 129
  %531 = sext i32 %530 to i64
  %532 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %531
  %533 = load float, ptr addrspace(3) %532, align 4
  %534 = fmul float %533, %488
  %535 = insertelement <2 x float> poison, float %529, i64 0
  %536 = insertelement <2 x float> %535, float %534, i64 1
  %537 = fptrunc <2 x float> %536 to <2 x bfloat>
  %538 = add i32 %492, 128
  %539 = mul i32 %538, 2
  %540 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %537, ptr addrspace(8) %482, i32 %539, i32 0, i32 0)
  %541 = add i32 %494, 192
  %542 = sext i32 %541 to i64
  %543 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %542
  %544 = load float, ptr addrspace(3) %543, align 4
  %545 = fmul float %544, %488
  %546 = add i32 %494, 193
  %547 = sext i32 %546 to i64
  %548 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %547
  %549 = load float, ptr addrspace(3) %548, align 4
  %550 = fmul float %549, %488
  %551 = insertelement <2 x float> poison, float %545, i64 0
  %552 = insertelement <2 x float> %551, float %550, i64 1
  %553 = fptrunc <2 x float> %552 to <2 x bfloat>
  %554 = add i32 %492, 192
  %555 = mul i32 %554, 2
  %556 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %553, ptr addrspace(8) %482, i32 %555, i32 0, i32 0)
  br label %557

557:                                              ; preds = %489, %49
  %558 = add i32 %474, 8
  %559 = add i32 %64, %558
  %560 = mul i32 %559, 4
  %561 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %478, i32 %560, i32 0, i32 0)
  %562 = and i32 %561, 16777215
  %563 = icmp slt i32 %562, %16
  %564 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %480, i32 %560, i32 0, i32 0)
  br i1 %563, label %565, label %633

565:                                              ; preds = %557
  %566 = mul i32 %562, 7168
  %567 = add i32 %566, %92
  %568 = add i32 %567, %476
  %569 = mul i32 %558, 256
  %570 = add i32 %569, %476
  %571 = sext i32 %570 to i64
  %572 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %571
  %573 = load float, ptr addrspace(3) %572, align 4
  %574 = fmul float %573, %564
  %575 = add i32 %570, 1
  %576 = sext i32 %575 to i64
  %577 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %576
  %578 = load float, ptr addrspace(3) %577, align 4
  %579 = fmul float %578, %564
  %580 = insertelement <2 x float> poison, float %574, i64 0
  %581 = insertelement <2 x float> %580, float %579, i64 1
  %582 = fptrunc <2 x float> %581 to <2 x bfloat>
  %583 = mul i32 %568, 2
  %584 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %582, ptr addrspace(8) %482, i32 %583, i32 0, i32 0)
  %585 = add i32 %570, 64
  %586 = sext i32 %585 to i64
  %587 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %586
  %588 = load float, ptr addrspace(3) %587, align 4
  %589 = fmul float %588, %564
  %590 = add i32 %570, 65
  %591 = sext i32 %590 to i64
  %592 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %591
  %593 = load float, ptr addrspace(3) %592, align 4
  %594 = fmul float %593, %564
  %595 = insertelement <2 x float> poison, float %589, i64 0
  %596 = insertelement <2 x float> %595, float %594, i64 1
  %597 = fptrunc <2 x float> %596 to <2 x bfloat>
  %598 = add i32 %568, 64
  %599 = mul i32 %598, 2
  %600 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %597, ptr addrspace(8) %482, i32 %599, i32 0, i32 0)
  %601 = add i32 %570, 128
  %602 = sext i32 %601 to i64
  %603 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %602
  %604 = load float, ptr addrspace(3) %603, align 4
  %605 = fmul float %604, %564
  %606 = add i32 %570, 129
  %607 = sext i32 %606 to i64
  %608 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %607
  %609 = load float, ptr addrspace(3) %608, align 4
  %610 = fmul float %609, %564
  %611 = insertelement <2 x float> poison, float %605, i64 0
  %612 = insertelement <2 x float> %611, float %610, i64 1
  %613 = fptrunc <2 x float> %612 to <2 x bfloat>
  %614 = add i32 %568, 128
  %615 = mul i32 %614, 2
  %616 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %613, ptr addrspace(8) %482, i32 %615, i32 0, i32 0)
  %617 = add i32 %570, 192
  %618 = sext i32 %617 to i64
  %619 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %618
  %620 = load float, ptr addrspace(3) %619, align 4
  %621 = fmul float %620, %564
  %622 = add i32 %570, 193
  %623 = sext i32 %622 to i64
  %624 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %623
  %625 = load float, ptr addrspace(3) %624, align 4
  %626 = fmul float %625, %564
  %627 = insertelement <2 x float> poison, float %621, i64 0
  %628 = insertelement <2 x float> %627, float %626, i64 1
  %629 = fptrunc <2 x float> %628 to <2 x bfloat>
  %630 = add i32 %568, 192
  %631 = mul i32 %630, 2
  %632 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %629, ptr addrspace(8) %482, i32 %631, i32 0, i32 0)
  br label %633

633:                                              ; preds = %565, %557
  %634 = add i32 %474, 16
  %635 = add i32 %64, %634
  %636 = mul i32 %635, 4
  %637 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %478, i32 %636, i32 0, i32 0)
  %638 = and i32 %637, 16777215
  %639 = icmp slt i32 %638, %16
  %640 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %480, i32 %636, i32 0, i32 0)
  br i1 %639, label %641, label %709

641:                                              ; preds = %633
  %642 = mul i32 %638, 7168
  %643 = add i32 %642, %92
  %644 = add i32 %643, %476
  %645 = mul i32 %634, 256
  %646 = add i32 %645, %476
  %647 = sext i32 %646 to i64
  %648 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %647
  %649 = load float, ptr addrspace(3) %648, align 4
  %650 = fmul float %649, %640
  %651 = add i32 %646, 1
  %652 = sext i32 %651 to i64
  %653 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %652
  %654 = load float, ptr addrspace(3) %653, align 4
  %655 = fmul float %654, %640
  %656 = insertelement <2 x float> poison, float %650, i64 0
  %657 = insertelement <2 x float> %656, float %655, i64 1
  %658 = fptrunc <2 x float> %657 to <2 x bfloat>
  %659 = mul i32 %644, 2
  %660 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %658, ptr addrspace(8) %482, i32 %659, i32 0, i32 0)
  %661 = add i32 %646, 64
  %662 = sext i32 %661 to i64
  %663 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %662
  %664 = load float, ptr addrspace(3) %663, align 4
  %665 = fmul float %664, %640
  %666 = add i32 %646, 65
  %667 = sext i32 %666 to i64
  %668 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %667
  %669 = load float, ptr addrspace(3) %668, align 4
  %670 = fmul float %669, %640
  %671 = insertelement <2 x float> poison, float %665, i64 0
  %672 = insertelement <2 x float> %671, float %670, i64 1
  %673 = fptrunc <2 x float> %672 to <2 x bfloat>
  %674 = add i32 %644, 64
  %675 = mul i32 %674, 2
  %676 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %673, ptr addrspace(8) %482, i32 %675, i32 0, i32 0)
  %677 = add i32 %646, 128
  %678 = sext i32 %677 to i64
  %679 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %678
  %680 = load float, ptr addrspace(3) %679, align 4
  %681 = fmul float %680, %640
  %682 = add i32 %646, 129
  %683 = sext i32 %682 to i64
  %684 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %683
  %685 = load float, ptr addrspace(3) %684, align 4
  %686 = fmul float %685, %640
  %687 = insertelement <2 x float> poison, float %681, i64 0
  %688 = insertelement <2 x float> %687, float %686, i64 1
  %689 = fptrunc <2 x float> %688 to <2 x bfloat>
  %690 = add i32 %644, 128
  %691 = mul i32 %690, 2
  %692 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %689, ptr addrspace(8) %482, i32 %691, i32 0, i32 0)
  %693 = add i32 %646, 192
  %694 = sext i32 %693 to i64
  %695 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %694
  %696 = load float, ptr addrspace(3) %695, align 4
  %697 = fmul float %696, %640
  %698 = add i32 %646, 193
  %699 = sext i32 %698 to i64
  %700 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %699
  %701 = load float, ptr addrspace(3) %700, align 4
  %702 = fmul float %701, %640
  %703 = insertelement <2 x float> poison, float %697, i64 0
  %704 = insertelement <2 x float> %703, float %702, i64 1
  %705 = fptrunc <2 x float> %704 to <2 x bfloat>
  %706 = add i32 %644, 192
  %707 = mul i32 %706, 2
  %708 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %705, ptr addrspace(8) %482, i32 %707, i32 0, i32 0)
  br label %709

709:                                              ; preds = %641, %633
  %710 = add i32 %474, 24
  %711 = add i32 %64, %710
  %712 = mul i32 %711, 4
  %713 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %478, i32 %712, i32 0, i32 0)
  %714 = and i32 %713, 16777215
  %715 = icmp slt i32 %714, %16
  %716 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %480, i32 %712, i32 0, i32 0)
  br i1 %715, label %717, label %785

717:                                              ; preds = %709
  %718 = mul i32 %714, 7168
  %719 = add i32 %718, %92
  %720 = add i32 %719, %476
  %721 = mul i32 %710, 256
  %722 = add i32 %721, %476
  %723 = sext i32 %722 to i64
  %724 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %723
  %725 = load float, ptr addrspace(3) %724, align 4
  %726 = fmul float %725, %716
  %727 = add i32 %722, 1
  %728 = sext i32 %727 to i64
  %729 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %728
  %730 = load float, ptr addrspace(3) %729, align 4
  %731 = fmul float %730, %716
  %732 = insertelement <2 x float> poison, float %726, i64 0
  %733 = insertelement <2 x float> %732, float %731, i64 1
  %734 = fptrunc <2 x float> %733 to <2 x bfloat>
  %735 = mul i32 %720, 2
  %736 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %734, ptr addrspace(8) %482, i32 %735, i32 0, i32 0)
  %737 = add i32 %722, 64
  %738 = sext i32 %737 to i64
  %739 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %738
  %740 = load float, ptr addrspace(3) %739, align 4
  %741 = fmul float %740, %716
  %742 = add i32 %722, 65
  %743 = sext i32 %742 to i64
  %744 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %743
  %745 = load float, ptr addrspace(3) %744, align 4
  %746 = fmul float %745, %716
  %747 = insertelement <2 x float> poison, float %741, i64 0
  %748 = insertelement <2 x float> %747, float %746, i64 1
  %749 = fptrunc <2 x float> %748 to <2 x bfloat>
  %750 = add i32 %720, 64
  %751 = mul i32 %750, 2
  %752 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %749, ptr addrspace(8) %482, i32 %751, i32 0, i32 0)
  %753 = add i32 %722, 128
  %754 = sext i32 %753 to i64
  %755 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %754
  %756 = load float, ptr addrspace(3) %755, align 4
  %757 = fmul float %756, %716
  %758 = add i32 %722, 129
  %759 = sext i32 %758 to i64
  %760 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %759
  %761 = load float, ptr addrspace(3) %760, align 4
  %762 = fmul float %761, %716
  %763 = insertelement <2 x float> poison, float %757, i64 0
  %764 = insertelement <2 x float> %763, float %762, i64 1
  %765 = fptrunc <2 x float> %764 to <2 x bfloat>
  %766 = add i32 %720, 128
  %767 = mul i32 %766, 2
  %768 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %765, ptr addrspace(8) %482, i32 %767, i32 0, i32 0)
  %769 = add i32 %722, 192
  %770 = sext i32 %769 to i64
  %771 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %770
  %772 = load float, ptr addrspace(3) %771, align 4
  %773 = fmul float %772, %716
  %774 = add i32 %722, 193
  %775 = sext i32 %774 to i64
  %776 = getelementptr inbounds nuw float, ptr addrspace(3) @gemm2port_smem, i64 %775
  %777 = load float, ptr addrspace(3) %776, align 4
  %778 = fmul float %777, %716
  %779 = insertelement <2 x float> poison, float %773, i64 0
  %780 = insertelement <2 x float> %779, float %778, i64 1
  %781 = fptrunc <2 x float> %780 to <2 x bfloat>
  %782 = add i32 %720, 192
  %783 = mul i32 %782, 2
  %784 = call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat> %781, ptr addrspace(8) %482, i32 %783, i32 0, i32 0)
  br label %785

785:                                              ; preds = %717, %709
  br label %786

786:                                              ; preds = %785, %19
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

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16(<2 x bfloat>, ptr addrspace(8) captures(none), i32, i32, i32 immarg) #5

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 256, i32 1, i32 1}
