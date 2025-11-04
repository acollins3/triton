#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 8, 0], [0, 0, 32], [0, 0, 64], [0, 0, 128]], lane = [[0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0]], warp = [[0, 0, 8], [0, 0, 16]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_p_matmul_ogs_NNN_fp8e5xfp8e5xfp8e5_16x256x128x1(%arg0: !tt.tensordesc<tensor<1x16x256xf8E5M2, #shared>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg13: !tt.tensordesc<tensor<1x16x128xf8E5M2, #shared>>, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg24: !tt.tensordesc<tensor<1x128x256xf8E5M2, #shared>>, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg35: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32, %arg42: i32, %arg43: i32, %arg44: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c31_i32 = arith.constant 31 : i32
    %c932333861_i32 = arith.constant 932333861 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c2139095040_i32 = arith.constant 2139095040 : i32
    %cst_1 = arith.constant 1.000000e-30 : f32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #mma>
    %0 = tt.make_tensor_descriptor %arg7, [%arg1, %arg2, %arg3], [%arg4, %arg5, %c1_i64] : <f8E5M2>, <tensor<1x16x256xf8E5M2, #shared>>
    %1 = arith.muli %arg41, %arg42 : i32
    %2 = arith.muli %1, %arg43 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.subi %3, %c148_i32 : i32
    %5 = arith.muli %arg42, %arg43 : i32
    %6 = arith.muli %arg43, %c8_i32 : i32
    %7 = arith.addi %arg39, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.maxsi %8, %c1_i32 : i32
    %10 = arith.cmpi sgt, %9, %c0_i32 : i32
    %11 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %12 = tt.splat %arg37 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %13 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %14 = tt.splat %arg38 : i32 -> tensor<256xi32, #blocked>
    %15 = tt.bitcast %c932333861_i32 : i32 -> f32
    %16 = scf.for %arg45 = %3 to %2 step %c148_i32 iter_args(%arg46 = %4) -> (i32)  : i32 {
      %17 = arith.divsi %arg45, %5 {ttg.partition = array<i32: 1>} : i32
      %18 = arith.remsi %arg45, %5 {ttg.partition = array<i32: 1>} : i32
      %19 = arith.divsi %18, %6 {ttg.partition = array<i32: 1>} : i32
      %20 = arith.muli %19, %c8_i32 {ttg.partition = array<i32: 1>} : i32
      %21 = arith.subi %arg42, %20 {ttg.partition = array<i32: 1>} : i32
      %22 = arith.minsi %21, %c8_i32 {ttg.partition = array<i32: 1>} : i32
      %23 = arith.cmpi sge, %22, %c0_i32 {ttg.partition = array<i32: 1>} : i32
      llvm.intr.assume %23 : i1 {ttg.partition = array<i32: 1>}
      %24 = arith.remsi %18, %22 {ttg.partition = array<i32: 1>} : i32
      %25 = arith.addi %20, %24 {ttg.partition = array<i32: 1>} : i32
      %26 = arith.remsi %18, %6 {ttg.partition = array<i32: 1>} : i32
      %27 = arith.divsi %26, %22 {ttg.partition = array<i32: 1>} : i32
      %28 = arith.muli %25, %c16_i32 {ttg.partition = array<i32: 1>} : i32
      %29 = arith.muli %27, %c256_i32 {ttg.partition = array<i32: 1>} : i32
      llvm.intr.assume %10 : i1 {ttg.partition = array<i32: 0>}
      %30 = scf.for %arg47 = %c0_i32 to %9 step %c1_i32 iter_args(%arg48 = %cst_2) -> (tensor<16x256xf32, #mma>)  : i32 {
        %92 = arith.muli %arg47, %c128_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
        %93 = tt.descriptor_load %arg13[%17, %28, %92] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<1x16x128xf8E5M2, #shared>> -> tensor<16x128xf8E5M2, #blocked1>
        %94 = ttg.convert_layout %93 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<16x128xf8E5M2, #blocked1> -> tensor<16x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %95 = tt.descriptor_load %arg24[%17, %92, %29] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<1x128x256xf8E5M2, #shared>> -> tensor<128x256xf8E5M2, #blocked2>
        %96 = ttg.convert_layout %95 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x256xf8E5M2, #blocked2> -> tensor<128x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        %97 = tt.fp_to_fp %94 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<16x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %98 = tt.fp_to_fp %96 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        %99 = tt.dot %97, %98, %arg48, inputPrecision = tf32 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x256xf32, #mma>
        scf.yield {ttg.partition = array<i32: 0, 1>} %99 : tensor<16x256xf32, #mma>
      } {tt.scheduled_max_stage = 3 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      %31 = arith.addi %arg46, %c148_i32 {ttg.partition = array<i32: 0>} : i32
      %32 = arith.divsi %31, %5 {ttg.partition = array<i32: 0>} : i32
      %33 = arith.remsi %31, %5 {ttg.partition = array<i32: 0>} : i32
      %34 = arith.divsi %33, %6 {ttg.partition = array<i32: 0>} : i32
      %35 = arith.muli %34, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %36 = arith.subi %arg42, %35 {ttg.partition = array<i32: 0>} : i32
      %37 = arith.minsi %36, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %38 = arith.cmpi sge, %37, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      llvm.intr.assume %38 : i1 {ttg.partition = array<i32: 0>}
      %39 = arith.remsi %33, %37 {ttg.partition = array<i32: 0>} : i32
      %40 = arith.addi %35, %39 {ttg.partition = array<i32: 0>} : i32
      %41 = arith.remsi %33, %6 {ttg.partition = array<i32: 0>} : i32
      %42 = arith.divsi %41, %37 {ttg.partition = array<i32: 0>} : i32
      %43 = arith.muli %40, %c16_i32 {ttg.partition = array<i32: 0>} : i32
      %44 = arith.muli %42, %c256_i32 {ttg.partition = array<i32: 0>} : i32
      %45 = tt.splat %43 {ttg.partition = array<i32: 0>} : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #mma}>>
      %46 = arith.addi %45, %11 {ttg.partition = array<i32: 0>} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #mma}>>
      %47 = arith.cmpi slt, %46, %12 {ttg.partition = array<i32: 0>} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #mma}>>
      %48 = tt.splat %44 {ttg.partition = array<i32: 0>} : i32 -> tensor<256xi32, #blocked>
      %49 = arith.addi %48, %13 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked>
      %50 = arith.cmpi slt, %49, %14 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked>
      %51 = arith.muli %32, %arg36 {ttg.partition = array<i32: 0>} : i32
      %52 = tt.addptr %arg35, %51 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %53 = tt.splat %52 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
      %54 = tt.addptr %53, %49 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
      %55 = tt.load %54, %50, %cst {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked>
      %56 = tt.load %arg23 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>
      %57 = tt.addptr %arg34, %32 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %58 = tt.load %57 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>
      %59 = arith.mulf %56, %58 {ttg.partition = array<i32: 0>} : f32
      %60 = tt.splat %59 {ttg.partition = array<i32: 0>} : f32 -> tensor<16x256xf32, #mma>
      %61 = arith.mulf %30, %60 {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #mma>
      %62 = ttg.convert_layout %55 {ttg.partition = array<i32: 0>} : tensor<256xf32, #blocked> -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
      %63 = tt.expand_dims %62 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xf32, #mma>
      %64 = tt.broadcast %63 {ttg.partition = array<i32: 0>} : tensor<1x256xf32, #mma> -> tensor<16x256xf32, #mma>
      %65 = arith.addf %61, %64 {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #mma>
      %66 = tt.expand_dims %47 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xi1, #mma>
      %67 = tt.broadcast %66 {ttg.partition = array<i32: 0>} : tensor<16x1xi1, #mma> -> tensor<16x256xi1, #mma>
      %68 = arith.select %67, %65, %cst_2 {ttg.partition = array<i32: 0>} : tensor<16x256xi1, #mma>, tensor<16x256xf32, #mma>
      %69 = tt.addptr %arg11, %32 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %70 = tt.addptr %arg12, %32 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %71 = tt.load %69 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>
      %72 = arith.divf %cst_0, %71 {ttg.partition = array<i32: 0>} : f32
      %73 = tt.reshape %68 allow_reorder {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #mma> -> tensor<4096xf32, #blocked3>
      %74 = "tt.reduce"(%73) <{axis = 0 : i32}> ({
      ^bb0(%arg47: f32, %arg48: f32):
        %92 = tt.elementwise_inline_asm "{\0A    max.NaN.xorsign.abs.f32 $0, $1, $2;\0A    }" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true, ttg.partition = array<i32: 0>} %arg47, %arg48 : f32, f32 -> f32
        tt.reduce.return %92 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>} : (tensor<4096xf32, #blocked3>) -> f32
      %75 = tt.bitcast %74 {ttg.partition = array<i32: 0>} : f32 -> i32
      %76 = arith.andi %75, %c2147483647_i32 {ttg.partition = array<i32: 0>} : i32
      %77 = arith.minui %76, %c2139095040_i32 {ttg.partition = array<i32: 0>} : i32
      %78 = tt.bitcast %77 {ttg.partition = array<i32: 0>} : i32 -> f32
      %79 = math.fma %78, %15, %cst_1 {ttg.partition = array<i32: 0>} : f32
      %80 = tt.bitcast %79 {ttg.partition = array<i32: 0>} : f32 -> i32
      %81 = tt.bitcast %70 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> !tt.ptr<i32>
      %82 = arith.shrui %80, %c31_i32 {ttg.partition = array<i32: 0>} : i32
      %83 = arith.cmpi ne, %82, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      %84 = arith.cmpi eq, %82, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      %85 = tt.atomic_rmw max, relaxed, gpu, %81, %80, %84 {ttg.partition = array<i32: 0>} : (!tt.ptr<i32>, i32, i1) -> i32
      %86 = tt.atomic_rmw umin, relaxed, gpu, %81, %80, %83 {ttg.partition = array<i32: 0>} : (!tt.ptr<i32>, i32, i1) -> i32
      %87 = tt.splat %72 {ttg.partition = array<i32: 0>} : f32 -> tensor<16x256xf32, #mma>
      %88 = arith.mulf %68, %87 {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #mma>
      %89 = tt.fp_to_fp %88 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<16x256xf32, #mma> -> tensor<16x256xf8E5M2, #mma>
      %90 = tt.reshape %89 {ttg.partition = array<i32: 0>} : tensor<16x256xf8E5M2, #mma> -> tensor<1x16x256xf8E5M2, #linear>
      %91 = ttg.convert_layout %90 {ttg.partition = array<i32: 0>} : tensor<1x16x256xf8E5M2, #linear> -> tensor<1x16x256xf8E5M2, #blocked4>
      tt.descriptor_store %0[%32, %43, %44], %91 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x16x256xf8E5M2, #shared>>, tensor<1x16x256xf8E5M2, #blocked4>
      scf.yield {ttg.partition = array<i32: 0, 1>} %31 : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
