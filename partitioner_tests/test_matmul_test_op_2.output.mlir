#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 8, 0]], lane = [[0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0]], warp = [[0, 0, 8], [0, 0, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_p_matmul_ogs_NNN_fp32xfp8e5xfp8e5_16x16x128x5(%arg0: !tt.tensordesc<tensor<1x16x16xf32, #shared>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: !tt.tensordesc<tensor<1x16x128xf8E5M2, #shared1>>, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg22: !tt.tensordesc<tensor<1x128x16xf8E5M2, #shared2>>, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg33: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32, %arg40: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c30_i32 = arith.constant 30 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %c639_i32 = arith.constant 639 : i32
    %c640_i32 = arith.constant 640 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 0, parent = #mma}>>
    %0 = tt.make_tensor_descriptor %arg7, [%arg1, %arg2, %arg3], [%arg4, %arg5, %c1_i64] : <f32>, <tensor<1x16x16xf32, #shared>>
    %1 = arith.muli %arg39, %c5_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = arith.subi %2, %c30_i32 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %5 = tt.splat %arg36 : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %6 = scf.for %arg41 = %2 to %1 step %c30_i32 iter_args(%arg42 = %3) -> (i32)  : i32 {
      %7 = arith.divsi %arg41, %c5_i32 {ttg.partition = array<i32: 1>} : i32
      %8 = arith.remsi %arg41, %c5_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %9 = arith.remsi %8, %c5_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %10 = arith.divsi %8, %c5_i32 {ttg.partition = array<i32: 1>} : i32
      %11 = arith.divsi %10, %c8_i32 {ttg.partition = array<i32: 1>} : i32
      %12 = arith.muli %11, %c8_i32 {ttg.partition = array<i32: 1>} : i32
      %13 = arith.subi %c1_i32, %12 {ttg.partition = array<i32: 1>} : i32
      %14 = arith.minsi %13, %c8_i32 {ttg.partition = array<i32: 1>} : i32
      %15 = arith.cmpi sge, %14, %c0_i32 {ttg.partition = array<i32: 1>} : i32
      llvm.intr.assume %15 : i1 {ttg.partition = array<i32: 1>}
      %16 = arith.remsi %10, %14 {ttg.partition = array<i32: 1>} : i32
      %17 = arith.addi %12, %16 {ttg.partition = array<i32: 1>} : i32
      %18 = arith.remsi %10, %c8_i32 {ttg.partition = array<i32: 1>} : i32
      %19 = arith.divsi %18, %14 {ttg.partition = array<i32: 1>} : i32
      %20 = arith.muli %9, %c128_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %21 = arith.subi %arg37, %20 {ttg.partition = array<i32: 0, 1>} : i32
      %22 = arith.addi %21, %c639_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %23 = arith.divsi %22, %c640_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %24 = arith.muli %17, %c16_i32 {ttg.partition = array<i32: 1>} : i32
      %25 = arith.muli %19, %c16_i32 {ttg.partition = array<i32: 1>} : i32
      %26 = arith.maxsi %23, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %27 = arith.cmpi sgt, %26, %c0_i32 {ttg.partition = array<i32: 1>} : i32
      llvm.intr.assume %27 : i1 {ttg.partition = array<i32: 1>}
      %28 = scf.for %arg43 = %c0_i32 to %26 step %c1_i32 iter_args(%arg44 = %cst) -> (tensor<16x16xf32, #mma>)  : i32 {
        %67 = arith.muli %arg43, %c640_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
        %68 = arith.addi %20, %67 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
        %69 = tt.descriptor_load %arg11[%7, %24, %68] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<1x16x128xf8E5M2, #shared1>> -> tensor<16x128xf8E5M2, #blocked>
        %70 = ttg.convert_layout %69 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<16x128xf8E5M2, #blocked> -> tensor<16x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %71 = tt.descriptor_load %arg22[%7, %68, %25] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<1x128x16xf8E5M2, #shared2>> -> tensor<128x16xf8E5M2, #blocked1>
        %72 = ttg.convert_layout %71 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x16xf8E5M2, #blocked1> -> tensor<128x16xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        %73 = tt.fp_to_fp %70 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<16x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %74 = tt.fp_to_fp %72 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<128x16xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        %75 = tt.dot %73, %74, %arg44, inputPrecision = tf32 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x16xf32, #mma>
        scf.yield {ttg.partition = array<i32: 0, 1>} %75 : tensor<16x16xf32, #mma>
      } {tt.scheduled_max_stage = 3 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      %29 = arith.addi %arg42, %c30_i32 {ttg.partition = array<i32: 0>} : i32
      %30 = arith.divsi %29, %c5_i32 {ttg.partition = array<i32: 0>} : i32
      %31 = arith.remsi %29, %c5_i32 {ttg.partition = array<i32: 0>} : i32
      %32 = arith.remsi %31, %c5_i32 {ttg.partition = array<i32: 0>} : i32
      %33 = arith.divsi %31, %c5_i32 {ttg.partition = array<i32: 0>} : i32
      %34 = arith.divsi %33, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %35 = arith.muli %34, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %36 = arith.subi %c1_i32, %35 {ttg.partition = array<i32: 0>} : i32
      %37 = arith.minsi %36, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %38 = arith.cmpi sge, %37, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      llvm.intr.assume %38 : i1 {ttg.partition = array<i32: 0>}
      %39 = arith.remsi %33, %37 {ttg.partition = array<i32: 0>} : i32
      %40 = arith.addi %35, %39 {ttg.partition = array<i32: 0>} : i32
      %41 = arith.remsi %33, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %42 = arith.divsi %41, %37 {ttg.partition = array<i32: 0>} : i32
      %43 = arith.muli %40, %c16_i32 {ttg.partition = array<i32: 0>} : i32
      %44 = arith.muli %42, %c16_i32 {ttg.partition = array<i32: 0>} : i32
      %45 = tt.splat %44 {ttg.partition = array<i32: 0>} : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %46 = arith.addi %45, %4 {ttg.partition = array<i32: 0>} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %47 = arith.cmpi slt, %46, %5 {ttg.partition = array<i32: 0>} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %48 = arith.muli %30, %arg34 {ttg.partition = array<i32: 0>} : i32
      %49 = tt.addptr %arg33, %48 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %50 = tt.splat %49 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
      %51 = tt.addptr %50, %46 {ttg.partition = array<i32: 0>} : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %52 = arith.cmpi eq, %32, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      %53 = scf.if %52 -> (tensor<16xf32, #ttg.slice<{dim = 0, parent = #mma}>>) {
        %67 = tt.load %51, %47, %cst_0 {ttg.partition = array<i32: 0>} : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
        scf.yield {ttg.partition = array<i32: 0>} %67 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #mma}>>
      } else {
        scf.yield {ttg.partition = array<i32: 0>} %cst_0 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #mma}>>
      } {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]}
      %54 = tt.load %arg21 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>
      %55 = tt.addptr %arg32, %30 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %56 = tt.load %55 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>
      %57 = arith.mulf %54, %56 {ttg.partition = array<i32: 0>} : f32
      %58 = tt.splat %57 {ttg.partition = array<i32: 0>} : f32 -> tensor<16x16xf32, #mma>
      %59 = arith.mulf %28, %58 {ttg.partition = array<i32: 0>} : tensor<16x16xf32, #mma>
      %60 = tt.expand_dims %53 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<16xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x16xf32, #mma>
      %61 = tt.broadcast %60 {ttg.partition = array<i32: 0>} : tensor<1x16xf32, #mma> -> tensor<16x16xf32, #mma>
      %62 = arith.addf %59, %61 {ttg.partition = array<i32: 0>} : tensor<16x16xf32, #mma>
      %63 = tt.reshape %62 {ttg.partition = array<i32: 0>} : tensor<16x16xf32, #mma> -> tensor<1x16x16xf32, #linear>
      %64 = arith.muli %9, %arg39 {ttg.partition = array<i32: 0>} : i32
      %65 = arith.addi %64, %30 {ttg.partition = array<i32: 0>} : i32
      %66 = ttg.convert_layout %63 {ttg.partition = array<i32: 0>} : tensor<1x16x16xf32, #linear> -> tensor<1x16x16xf32, #blocked2>
      tt.descriptor_store %0[%65, %43, %44], %66 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x16x16xf32, #shared>>, tensor<1x16x16xf32, #blocked2>
      scf.yield {ttg.partition = array<i32: 0, 1>} %29 : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
