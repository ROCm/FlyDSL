; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@mxscale_a8w4_32x32x256_2x2_2buf_arena = external addrspace(3) global [28672 x i8], align 1024

define amdgpu_kernel void @kernel_mxscale_gemm_0(ptr addrspace(1) %0, <{ <{ i32, i32 }>, <{ i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32 }>, <{ i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32 }>, <{ i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32 }>, <{ i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32, i32 }>, <{ i64 }> }> %9, i32 %10, i32 %11, i32 %12, i32 %13) #0 !reqd_work_group_size !1 {
  call void @llvm.amdgcn.s.setreg(i32 282, i32 1)
  %15 = call range(i32 0, 128) i32 @llvm.amdgcn.workitem.id.x()
  %16 = sext i32 %15 to i64
  %17 = call i32 @llvm.amdgcn.workgroup.id.x()
  %18 = sext i32 %17 to i64
  %19 = call i32 @llvm.amdgcn.workgroup.id.y()
  %20 = sext i32 %19 to i64
  %21 = mul i64 %18, 32
  %22 = mul i64 %20, 32
  %23 = trunc i64 %16 to i32
  %24 = sdiv i32 %23, 64
  %25 = srem i32 %24, 2
  %26 = sdiv i32 %23, 32
  %27 = srem i32 %26, 2
  %28 = sdiv i32 %23, 16
  %29 = srem i32 %28, 2
  %30 = srem i32 %23, 16
  %31 = sext i32 %25 to i64
  %32 = sext i32 %27 to i64
  %33 = sext i32 %29 to i64
  %34 = sext i32 %30 to i64
  %35 = mul i64 %31, 16
  %36 = mul i64 %32, 16
  %37 = sext i32 %10 to i64
  %38 = sext i32 %12 to i64
  %39 = sext i32 %13 to i64
  %40 = mul i64 %37, %39
  %41 = mul i64 %40, 2
  %42 = addrspacecast ptr addrspace(1) %0 to ptr
  %43 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %42, i16 0, i64 %41, i32 159744)
  %44 = call i32 @llvm.amdgcn.wave.id()
  %45 = icmp eq i32 %44, 0
  %46 = icmp eq i32 %44, 1
  %47 = icmp eq i32 %44, 2
  %48 = mul i64 %31, 2
  %49 = add i64 %48, %32
  %50 = mul i64 %49, 384
  %51 = mul i64 %34, 24
  %52 = add i64 %50, %51
  %53 = mul i64 %33, 8
  %54 = add i64 %52, %53
  %55 = sext i32 %44 to i64
  %56 = udiv i64 %55, 2
  %57 = urem i64 %55, 2
  %58 = mul i64 %56, 2
  %59 = add i64 %58, %57
  %60 = mul i64 %59, 768
  %61 = mul i64 %56, 16
  %62 = mul i64 %57, 16
  %63 = add i64 %21, %61
  %64 = add i64 %22, %62
  %65 = ptrtoint ptr addrspace(1) %0 to i64
  %66 = mul i64 %63, %39
  %67 = add i64 %66, %64
  %68 = mul i64 %67, 2
  %69 = add i64 %65, %68
  %70 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena to i64), %60
  %71 = trunc i64 %70 to i32
  %72 = trunc i64 %69 to i32
  %73 = lshr i64 %69, 32
  %74 = trunc i64 %73 to i32
  %75 = or i32 %74, -2147483648
  %76 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %71, i64 1
  %77 = insertelement <4 x i32> %76, i32 %72, i64 2
  %78 = insertelement <4 x i32> %77, i32 %75, i64 3
  %79 = trunc i64 %63 to i32
  %80 = sub i32 %10, %79
  %81 = call i32 @llvm.smax.i32(i32 %80, i32 0)
  %82 = and i32 %81, 65535
  %83 = shl i32 %82, 16
  %84 = lshr i32 %81, 16
  %85 = and i32 %84, 65535
  %86 = or i32 %85, 1572864
  %87 = insertelement <8 x i32> <i32 65536, i32 1048576, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %83, i64 2
  %88 = insertelement <8 x i32> %87, i32 %86, i64 3
  %89 = insertelement <8 x i32> %88, i32 16, i64 4
  %90 = insertelement <8 x i32> %89, i32 %13, i64 5
  %91 = insertelement <8 x i32> %90, i32 0, i64 6
  %92 = insertelement <8 x i32> %91, i32 0, i64 7
  %93 = ptrtoint ptr addrspace(1) %2 to i64
  %94 = mul i64 %21, %38
  %95 = add i64 %93, %94
  %96 = trunc i64 %95 to i32
  %97 = lshr i64 %95, 32
  %98 = trunc i64 %97 to i32
  %99 = or i32 %98, -2147483648
  %100 = trunc i64 %21 to i32
  %101 = sub i32 %10, %100
  %102 = call i32 @llvm.smax.i32(i32 %101, i32 0)
  %103 = and i32 %102, 65535
  %104 = shl i32 %103, 16
  %105 = lshr i32 %102, 16
  %106 = and i32 %105, 65535
  %107 = or i32 %106, 16777216
  %108 = insertelement <8 x i32> <i32 124780544, i32 16777216, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %104, i64 2
  %109 = insertelement <8 x i32> %108, i32 %107, i64 3
  %110 = insertelement <8 x i32> %109, i32 32, i64 4
  %111 = insertelement <8 x i32> %110, i32 %12, i64 5
  %112 = insertelement <8 x i32> %111, i32 0, i64 6
  %113 = insertelement <8 x i32> %112, i32 0, i64 7
  %114 = udiv i64 %22, 32
  %115 = mul i64 %114, 32
  %116 = ptrtoint ptr addrspace(1) %4 to i64
  %117 = mul i64 %114, 8192
  %118 = add i64 %116, %117
  %119 = trunc i64 %118 to i32
  %120 = lshr i64 %118, 32
  %121 = trunc i64 %120 to i32
  %122 = or i32 %121, -2147483648
  %123 = ptrtoint ptr addrspace(1) %6 to i64
  %124 = mul i64 %18, 512
  %125 = add i64 %123, %124
  %126 = trunc i64 %125 to i32
  %127 = lshr i64 %125, 32
  %128 = trunc i64 %127 to i32
  %129 = or i32 %128, -2147483648
  %130 = ptrtoint ptr addrspace(1) %8 to i64
  %131 = mul i64 %20, 512
  %132 = add i64 %130, %131
  %133 = trunc i64 %132 to i32
  %134 = lshr i64 %132, 32
  %135 = trunc i64 %134 to i32
  %136 = or i32 %135, -2147483648
  %137 = icmp slt i32 %44, 4
  %138 = zext i1 %137 to i32
  %139 = select i1 %47, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13056) to i32), i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13328) to i32)
  %140 = select i1 %46, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 8704) to i32), i32 %139
  %141 = select i1 %45, i32 ptrtoint (ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena to i32), i32 %140
  %142 = select i1 %47, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27392) to i32), i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27664) to i32)
  %143 = select i1 %46, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 23040) to i32), i32 %142
  %144 = select i1 %45, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 14336) to i32), i32 %143
  %145 = select i1 %47, i32 %126, i32 %133
  %146 = select i1 %46, i32 %119, i32 %145
  %147 = select i1 %45, i32 %96, i32 %146
  %148 = select i1 %47, i32 %129, i32 %136
  %149 = select i1 %46, i32 %122, i32 %148
  %150 = select i1 %45, i32 %99, i32 %149
  %151 = select i1 %46, <8 x i32> <i32 124780544, i32 16777216, i32 1048576, i32 16777216, i32 16, i32 256, i32 0, i32 0>, <8 x i32> <i32 2097152, i32 524288, i32 2097152, i32 524288, i32 32, i32 16, i32 0, i32 0>
  %152 = select i1 %45, <8 x i32> %113, <8 x i32> %151
  %153 = select i1 %46, i32 4096, i32 8
  %154 = select i1 %45, i32 256, i32 %153
  %155 = or i32 %106, 8388608
  %156 = insertelement <8 x i32> <i32 1194328064, i32 8388608, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %104, i64 2
  %157 = insertelement <8 x i32> %156, i32 %155, i64 3
  %158 = insertelement <8 x i32> %157, i32 32, i64 4
  %159 = insertelement <8 x i32> %158, i32 %12, i64 5
  %160 = insertelement <8 x i32> %159, i32 0, i64 6
  %161 = insertelement <8 x i32> %160, i32 0, i64 7
  %162 = select i1 %46, <8 x i32> <i32 124780544, i32 16777216, i32 524288, i32 16777216, i32 8, i32 256, i32 0, i32 0>, <8 x i32> <i32 -1048576, i32 262144, i32 2097152, i32 262144, i32 32, i32 16, i32 0, i32 0>
  %163 = select i1 %45, <8 x i32> %161, <8 x i32> %162
  %164 = add i64 %124, 4
  %165 = add i64 %123, %164
  %166 = trunc i64 %165 to i32
  %167 = lshr i64 %165, 32
  %168 = trunc i64 %167 to i32
  %169 = or i32 %168, -2147483648
  %170 = add i64 %131, 4
  %171 = add i64 %130, %170
  %172 = trunc i64 %171 to i32
  %173 = lshr i64 %171, 32
  %174 = trunc i64 %173 to i32
  %175 = or i32 %174, -2147483648
  %176 = add i64 %94, 128
  %177 = add i64 %93, %176
  %178 = trunc i64 %177 to i32
  %179 = lshr i64 %177, 32
  %180 = trunc i64 %179 to i32
  %181 = or i32 %180, -2147483648
  %182 = add i64 %115, 8
  %183 = mul i64 %182, 256
  %184 = add i64 %116, %183
  %185 = trunc i64 %184 to i32
  %186 = lshr i64 %184, 32
  %187 = trunc i64 %186 to i32
  %188 = or i32 %187, -2147483648
  %189 = select i1 %47, i32 %166, i32 %172
  %190 = select i1 %46, i32 %185, i32 %189
  %191 = select i1 %45, i32 %178, i32 %190
  %192 = select i1 %47, i32 %169, i32 %175
  %193 = select i1 %46, i32 %188, i32 %192
  %194 = select i1 %45, i32 %181, i32 %193
  %195 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138)
  %196 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %150)
  %197 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %141)
  %198 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %144)
  %199 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %150)
  %200 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %194)
  %201 = select i1 %46, i32 2176, i32 4
  %202 = select i1 %45, i32 128, i32 %201
  %203 = add i32 %141, %202
  %204 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %203)
  %205 = add i32 %144, %202
  %206 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %205)
  %207 = insertelement <2 x i32> poison, i32 %195, i64 0
  %208 = insertelement <2 x i32> %207, i32 %197, i64 1
  %209 = insertelement <2 x i32> poison, i32 %147, i64 0
  %210 = insertelement <2 x i32> %209, i32 %199, i64 1
  %211 = shufflevector <2 x i32> %208, <2 x i32> %210, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %211, <8 x i32> %163, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %212 = insertelement <2 x i32> poison, i32 %195, i64 0
  %213 = insertelement <2 x i32> %212, i32 %204, i64 1
  %214 = insertelement <2 x i32> poison, i32 %191, i64 0
  %215 = insertelement <2 x i32> %214, i32 %200, i64 1
  %216 = shufflevector <2 x i32> %213, <2 x i32> %215, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %216, <8 x i32> %163, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %217 = add i32 %147, %154
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  %218 = add i64 %35, %34
  %219 = mul i64 %218, 272
  %220 = mul i64 %33, 16
  %221 = add i64 %219, %220
  %222 = mul i64 %34, 16
  %223 = mul i64 %33, 544
  %224 = mul i64 %32, 272
  %225 = add i64 %224, %222
  %226 = add i64 %225, %223
  %227 = mul i64 %218, 8
  %228 = mul i64 %33, 4
  %229 = add i64 %227, %228
  %230 = add i64 %36, %34
  %231 = mul i64 %230, 8
  %232 = add i64 %231, %228
  %233 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13056) to i64), %229
  %234 = trunc i64 %233 to i32
  %235 = inttoptr i32 %234 to ptr addrspace(3)
  %236 = load <4 x i32>, ptr addrspace(3) %235, align 16
  %237 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13328) to i64), %232
  %238 = trunc i64 %237 to i32
  %239 = inttoptr i32 %238 to ptr addrspace(3)
  %240 = load <4 x i32>, ptr addrspace(3) %239, align 16
  %241 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena to i64), %221
  %242 = trunc i64 %241 to i32
  %243 = inttoptr i32 %242 to ptr addrspace(3)
  %244 = load <4 x i32>, ptr addrspace(3) %243, align 16
  %245 = getelementptr i8, ptr addrspace(3) %243, i32 32
  %246 = load <4 x i32>, ptr addrspace(3) %245, align 16
  %247 = getelementptr i8, ptr addrspace(3) %243, i32 64
  %248 = load <4 x i32>, ptr addrspace(3) %247, align 16
  %249 = getelementptr i8, ptr addrspace(3) %243, i32 96
  %250 = load <4 x i32>, ptr addrspace(3) %249, align 16
  %251 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 8704) to i64), %226
  %252 = trunc i64 %251 to i32
  %253 = inttoptr i32 %252 to ptr addrspace(3)
  %254 = load <4 x i32>, ptr addrspace(3) %253, align 16
  %255 = getelementptr i8, ptr addrspace(3) %253, i32 1088
  %256 = load <4 x i32>, ptr addrspace(3) %255, align 16
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %257 = insertelement <2 x i32> poison, i32 %195, i64 0
  %258 = insertelement <2 x i32> %257, i32 %198, i64 1
  %259 = insertelement <2 x i32> poison, i32 %217, i64 0
  %260 = insertelement <2 x i32> %259, i32 %196, i64 1
  %261 = shufflevector <2 x i32> %258, <2 x i32> %260, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %261, <8 x i32> %152, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 1)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %262 = extractelement <4 x i32> %236, i64 0
  %263 = shufflevector <4 x i32> %244, <4 x i32> %246, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %264 = shufflevector <4 x i32> %248, <4 x i32> %250, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %265 = shufflevector <8 x i32> %263, <8 x i32> %264, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %266 = shufflevector <4 x i32> %254, <4 x i32> %256, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %267 = extractelement <4 x i32> %240, i64 0
  %268 = getelementptr i8, ptr addrspace(3) %235, i32 4
  %269 = load <4 x i32>, ptr addrspace(3) %268, align 16
  %270 = getelementptr i8, ptr addrspace(3) %239, i32 4
  %271 = load <4 x i32>, ptr addrspace(3) %270, align 16
  %272 = getelementptr i8, ptr addrspace(3) %243, i32 128
  %273 = load <4 x i32>, ptr addrspace(3) %272, align 16
  %274 = getelementptr i8, ptr addrspace(3) %243, i32 160
  %275 = load <4 x i32>, ptr addrspace(3) %274, align 16
  %276 = getelementptr i8, ptr addrspace(3) %243, i32 192
  %277 = load <4 x i32>, ptr addrspace(3) %276, align 16
  %278 = getelementptr i8, ptr addrspace(3) %243, i32 224
  %279 = load <4 x i32>, ptr addrspace(3) %278, align 16
  %280 = getelementptr i8, ptr addrspace(3) %253, i32 2176
  %281 = load <4 x i32>, ptr addrspace(3) %280, align 16
  %282 = getelementptr i8, ptr addrspace(3) %253, i32 3264
  %283 = load <4 x i32>, ptr addrspace(3) %282, align 16
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %284 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %266, i32 0, <16 x i32> %265, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 %267, i32 0, i32 0, i32 %262, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %285 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27392) to i64), %229
  %286 = trunc i64 %285 to i32
  %287 = inttoptr i32 %286 to ptr addrspace(3)
  %288 = load <4 x i32>, ptr addrspace(3) %287, align 16
  %289 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27664) to i64), %232
  %290 = trunc i64 %289 to i32
  %291 = inttoptr i32 %290 to ptr addrspace(3)
  %292 = load <4 x i32>, ptr addrspace(3) %291, align 16
  %293 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 14336) to i64), %221
  %294 = trunc i64 %293 to i32
  %295 = inttoptr i32 %294 to ptr addrspace(3)
  %296 = load <4 x i32>, ptr addrspace(3) %295, align 16
  %297 = getelementptr i8, ptr addrspace(3) %295, i32 32
  %298 = load <4 x i32>, ptr addrspace(3) %297, align 16
  %299 = getelementptr i8, ptr addrspace(3) %295, i32 64
  %300 = load <4 x i32>, ptr addrspace(3) %299, align 16
  %301 = getelementptr i8, ptr addrspace(3) %295, i32 96
  %302 = load <4 x i32>, ptr addrspace(3) %301, align 16
  %303 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 23040) to i64), %226
  %304 = trunc i64 %303 to i32
  %305 = inttoptr i32 %304 to ptr addrspace(3)
  %306 = load <4 x i32>, ptr addrspace(3) %305, align 16
  %307 = getelementptr i8, ptr addrspace(3) %305, i32 1088
  %308 = load <4 x i32>, ptr addrspace(3) %307, align 16
  %309 = extractelement <4 x i32> %269, i64 0
  %310 = shufflevector <4 x i32> %273, <4 x i32> %275, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %311 = shufflevector <4 x i32> %277, <4 x i32> %279, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %312 = shufflevector <8 x i32> %310, <8 x i32> %311, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %313 = shufflevector <4 x i32> %281, <4 x i32> %283, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %314 = extractelement <4 x i32> %271, i64 0
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %315 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %313, i32 0, <16 x i32> %312, i16 0, <8 x float> %284, i32 0, i32 0, i32 %314, i32 0, i32 0, i32 %309, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %316 = extractelement <4 x i32> %288, i64 0
  %317 = shufflevector <4 x i32> %296, <4 x i32> %298, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %318 = shufflevector <4 x i32> %300, <4 x i32> %302, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %319 = shufflevector <8 x i32> %317, <8 x i32> %318, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %320 = shufflevector <4 x i32> %306, <4 x i32> %308, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %321 = extractelement <4 x i32> %292, i64 0
  %322 = getelementptr i8, ptr addrspace(3) %287, i32 4
  %323 = load <4 x i32>, ptr addrspace(3) %322, align 16
  %324 = getelementptr i8, ptr addrspace(3) %291, i32 4
  %325 = load <4 x i32>, ptr addrspace(3) %324, align 16
  %326 = getelementptr i8, ptr addrspace(3) %295, i32 128
  %327 = load <4 x i32>, ptr addrspace(3) %326, align 16
  %328 = getelementptr i8, ptr addrspace(3) %295, i32 160
  %329 = load <4 x i32>, ptr addrspace(3) %328, align 16
  %330 = getelementptr i8, ptr addrspace(3) %295, i32 192
  %331 = load <4 x i32>, ptr addrspace(3) %330, align 16
  %332 = getelementptr i8, ptr addrspace(3) %295, i32 224
  %333 = load <4 x i32>, ptr addrspace(3) %332, align 16
  %334 = getelementptr i8, ptr addrspace(3) %305, i32 2176
  %335 = load <4 x i32>, ptr addrspace(3) %334, align 16
  %336 = getelementptr i8, ptr addrspace(3) %305, i32 3264
  %337 = load <4 x i32>, ptr addrspace(3) %336, align 16
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %338 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %320, i32 0, <16 x i32> %319, i16 0, <8 x float> %315, i32 0, i32 0, i32 %321, i32 0, i32 0, i32 %316, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %339 = extractelement <4 x i32> %323, i64 0
  %340 = shufflevector <4 x i32> %327, <4 x i32> %329, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %341 = shufflevector <4 x i32> %331, <4 x i32> %333, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %342 = shufflevector <8 x i32> %340, <8 x i32> %341, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %343 = shufflevector <4 x i32> %335, <4 x i32> %337, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %344 = extractelement <4 x i32> %325, i64 0
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %345 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %343, i32 0, <16 x i32> %342, i16 0, <8 x float> %338, i32 0, i32 0, i32 %344, i32 0, i32 0, i32 %339, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %346 = add i64 %21, 32
  %347 = icmp sle i64 %346, %37
  br i1 %347, label %348, label %352

348:                                              ; preds = %14
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %349 = fptrunc <8 x float> %345 to <8 x bfloat>
  %350 = bitcast <8 x bfloat> %349 to <8 x half>
  %351 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i64 %54
  store <8 x half> %350, ptr addrspace(3) %351, align 2
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %78, <8 x i32> %92, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  br label %363

352:                                              ; preds = %14
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %353 = add i64 %21, %35
  %354 = add i64 %353, %34
  %355 = add i64 %22, %36
  %356 = add i64 %355, %53
  %357 = mul i64 %354, %39
  %358 = add i64 %357, %356
  %359 = mul i64 %358, 2
  %360 = fptrunc <8 x float> %345 to <8 x bfloat>
  %361 = bitcast <8 x bfloat> %360 to <4 x i32>
  %362 = trunc i64 %359 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %361, ptr addrspace(8) %43, i32 %362, i32 0, i32 0)
  br label %363

363:                                              ; preds = %348, %352
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setreg(i32 immarg, i32) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.wave.id() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.tensorcnt(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.wait.dscnt(i16 immarg) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 immarg, <8 x i32>, i32 immarg, <16 x i32>, i16 immarg, <8 x float>, i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg, i32, i1 immarg, i1 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #9

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.store.from.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #5

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #6 = { nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nofree nounwind willreturn }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 128, i32 1, i32 1}
