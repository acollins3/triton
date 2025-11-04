#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 2, 16], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 8], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 2, 1, 2, 1], order = [4, 3, 2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2, 4], threadsPerWarp = [1, 1, 16, 2, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 2, 1, 1, 4], threadsPerWarp = [1, 2, 16, 1, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 1, 2, 3, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 4], order = [4, 3, 2, 1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 2, 2], order = [4, 3, 2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[64, 0], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [128, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CTAsPerCGA = [1, 1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1, 1], CTAOrder = [4, 3, 2, 1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1, 1], CTAOrder = [4, 3, 2, 1, 0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1, 1], CTAOrder = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_p_matmul_ogs_NNT_fp32xbf16xmxfp4_16x256x128x9(%arg0: !tt.tensordesc<tensor<1x1x1x16x256xf32, #shared>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: !tt.tensordesc<tensor<1x1x1x16x128xbf16, #shared1>>, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: !tt.tensordesc<tensor<1x256x64xui8, #shared2>>, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: !tt.tensordesc<tensor<1x2x1x2x256xui8, #shared3>>, %arg40: i32, %arg41: i32, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i64, %arg46: i64, %arg47: i64, %arg48: i64, %arg49: i64, %arg50: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg51: i32 {tt.divisibility = 16 : i32}, %arg52: i32 {tt.divisibility = 16 : i32}, %arg53: i32 {tt.divisibility = 16 : i32}, %arg54: i32 {tt.divisibility = 16 : i32}, %arg55: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg56: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg57: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg58: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg59: i32, %arg60: i32, %arg61: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<256x16xf32, #blocked>
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c9_i32 = arith.constant 9 : i32
    %c148_i32 = arith.constant 148 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c127_i32 = arith.constant 127 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %c64_i32 = arith.constant 64 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c1151_i32 = arith.constant 1151 : i32
    %c576_i32 = arith.constant 576 : i32
    %c1152_i32 = arith.constant 1152 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %cst_2 = arith.constant dense<7> : tensor<256x4xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
    %0 = tt.make_tensor_descriptor %arg11, [%arg1, %arg2, %arg3, %arg4, %arg5], [%arg6, %arg7, %arg8, %arg9, %c1_i64] : <f32>, <tensor<1x1x1x16x256xf32, #shared>>
    %1 = tt.addptr %arg57, %c8_i32 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = arith.subi %arg59, %2 : i32
    %4 = arith.subi %arg59, %3 : i32
    %5 = arith.muli %4, %arg60 : i32
    %6 = arith.muli %5, %c9_i32 : i32
    %7 = tt.get_program_id x : i32
    %8 = arith.subi %7, %c148_i32 : i32
    %9 = arith.muli %arg60, %c8_i32 : i32
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %11 = tt.splat %arg52 : i32 -> tensor<256xi32, #blocked1>
    %12 = scf.for %arg62 = %7 to %6 step %c148_i32 iter_args(%arg63 = %c0_i32) -> (i32)  : i32 {
      %18 = arith.remsi %arg62, %6 : i32
      %19 = arith.remsi %18, %c9_i32 : i32
      %20 = arith.muli %19, %c128_i32 : i32
      %21 = arith.subi %arg53, %20 : i32
      %22 = arith.addi %21, %c1151_i32 : i32
      %23 = arith.divsi %22, %c1152_i32 : i32
      %24 = arith.maxsi %23, %c1_i32 : i32
      %25 = arith.maxsi %24, %c1_i32 : i32
      %26 = arith.addi %arg63, %25 : i32
      scf.yield %26 : i32
    }
    %13 = arith.subi %7, %c148_i32 : i32
    %14 = arith.addi %arg52, %c127_i32 : i32
    %15 = arith.divsi %14, %c128_i32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %16 = ttng.tmem_store %cst, %result[%token], %true : tensor<256x16xf32, #blocked> -> !ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %17:17 = scf.for %arg62 = %c0_i32 to %12 step %c1_i32 iter_args(%arg63 = %c0_i32, %arg64 = %13, %arg65 = %8, %arg66 = %c0_i32, %arg67 = %c0_i32, %arg68 = %c0_i32, %arg69 = %c0_i32, %arg70 = %c0_i32, %arg71 = %c0_i32, %arg72 = %c0_i32, %arg73 = %c0_i32, %arg74 = %c0_i32, %arg75 = %c0_i32, %arg76 = %c0_i32, %arg77 = %c0_i32, %arg78 = %false, %arg79 = %16) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      %18 = arith.cmpi eq, %arg63, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %19 = arith.select %18, %c0_i32, %arg68 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %20:12 = scf.if %18 -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
        %62 = arith.addi %arg64, %c148_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %63 = arith.remsi %62, %6 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %64 = arith.remsi %63, %c9_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %65 = arith.muli %64, %c128_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %66 = arith.subi %arg53, %65 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %67 = arith.addi %66, %c1151_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %68 = arith.divsi %67, %c1152_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %69 = arith.maxsi %68, %c1_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %70 = arith.maxsi %69, %c1_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %71 = arith.remsi %62, %6 {ttg.partition = array<i32: 0, 2>} : i32
        %72 = arith.remsi %71, %c9_i32 {ttg.partition = array<i32: 0, 2>} : i32
        %73 = arith.divsi %71, %c9_i32 {ttg.partition = array<i32: 2>} : i32
        %74 = arith.divsi %73, %9 {ttg.partition = array<i32: 2>} : i32
        %75 = arith.muli %74, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %76 = arith.subi %4, %75 {ttg.partition = array<i32: 2>} : i32
        %77 = arith.minsi %76, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %78 = arith.cmpi sge, %77, %c0_i32 {ttg.partition = array<i32: 2>} : i32
        llvm.intr.assume %78 : i1 {ttg.partition = array<i32: 2>}
        %79 = arith.remsi %73, %77 {ttg.partition = array<i32: 2>} : i32
        %80 = arith.addi %75, %79 {ttg.partition = array<i32: 2>} : i32
        %81 = arith.remsi %73, %9 {ttg.partition = array<i32: 2>} : i32
        %82 = arith.divsi %81, %77 {ttg.partition = array<i32: 2>} : i32
        %83 = arith.muli %72, %c128_i32 {ttg.partition = array<i32: 2>} : i32
        %84 = arith.muli %72, %c64_i32 {ttg.partition = array<i32: 2>} : i32
        %85 = arith.subi %arg53, %83 {ttg.partition = array<i32: 2>} : i32
        %86 = arith.addi %85, %c1151_i32 {ttg.partition = array<i32: 2>} : i32
        %87 = arith.divsi %86, %c1152_i32 {ttg.partition = array<i32: 2>} : i32
        %88 = tt.addptr %arg58, %80 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %89 = tt.load %88 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %90 = arith.andi %89, %c65535_i32 {ttg.partition = array<i32: 2>} : i32
        %91 = arith.shrsi %89, %c16_i32 {ttg.partition = array<i32: 2>} : i32
        %92 = tt.addptr %arg55, %90 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %93 = tt.load %92 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %94 = tt.addptr %arg56, %90 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %95 = tt.load %94 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %96 = arith.muli %91, %c16_i32 {ttg.partition = array<i32: 2>} : i32
        %97 = arith.muli %82, %c256_i32 {ttg.partition = array<i32: 2>} : i32
        %98 = arith.maxsi %87, %c1_i32 {ttg.partition = array<i32: 2>} : i32
        %99 = arith.cmpi sgt, %98, %c0_i32 {ttg.partition = array<i32: 2>} : i32
        llvm.intr.assume %99 : i1 {ttg.partition = array<i32: 2>}
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %72, %83, %84, %90, %93, %95, %96, %97, %98, %62, %69, %70 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg64, %arg66, %arg67 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 0, 1, 2>, array<i32: 1>, array<i32: 0, 1, 2>]}
      %21 = arith.muli %19, %c1152_i32 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %22 = arith.addi %20#1, %21 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %23 = arith.muli %19, %c576_i32 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %24 = arith.addi %20#2, %23 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %25 = arith.subi %c1073741824_i32, %20#4 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %26 = arith.addi %25, %20#6 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %27 = arith.addi %20#5, %20#4 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %28 = tt.descriptor_load %arg15[%c1073741824_i32, %27, %c0_i32, %26, %22] {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x1x1x16x128xbf16, #shared1>> -> tensor<16x128xbf16, #blocked4>
      %29 = tt.descriptor_load %arg29[%20#3, %20#7, %24] {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x256x64xui8, #shared2>> -> tensor<256x64xi8, #blocked5>
      %30 = ttg.convert_layout %29 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xi8, #blocked5> -> tensor<256x64xi8, #blocked6>
      %31 = arith.divsi %24, %c16_i32 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %32 = arith.muli %20#3, %15 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %33 = arith.divsi %20#7, %c128_i32 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %34 = arith.addi %32, %33 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %35 = arith.divsi %31, %c4_i32 {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %36 = tt.descriptor_load %arg39[%c0_i32, %34, %35, %c0_i32, %c0_i32] {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x2x1x2x256xui8, #shared3>> -> tensor<1x2x1x2x256xi8, #blocked7>
      %37 = tt.reshape %36 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<1x2x1x2x256xi8, #blocked7> -> tensor<2x1x32x4x4xi8, #blocked8>
      %38 = tt.trans %37 {loop.cluster = 2 : i32, loop.stage = 3 : i32, order = array<i32: 0, 3, 2, 1, 4>, ttg.partition = array<i32: 0>} : tensor<2x1x32x4x4xi8, #blocked8> -> tensor<2x4x32x1x4xi8, #blocked9>
      %39 = tt.reshape %38 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<2x4x32x1x4xi8, #blocked9> -> tensor<256x4xi8, #linear>
      %40 = ttg.fp4_to_fp %30 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xi8, #blocked6> -> tensor<256x128xbf16, #blocked10>
      %41 = ttg.convert_layout %39 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x4xi8, #linear> -> tensor<256x4xi8, #ttg.slice<{dim = 2, parent = #blocked3}>>
      %42 = arith.extui %41 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x4xi8, #ttg.slice<{dim = 2, parent = #blocked3}>> to tensor<256x4xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
      %43 = arith.shli %42, %cst_2 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x4xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
      %44 = tt.bitcast %43 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x4xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<256x4xbf16, #ttg.slice<{dim = 2, parent = #blocked3}>>
      %45 = tt.expand_dims %44 {axis = 2 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x4xbf16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<256x4x1xbf16, #blocked3>
      %46 = tt.broadcast %45 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x4x1xbf16, #blocked3> -> tensor<256x4x32xbf16, #blocked3>
      %47 = tt.reshape %46 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x4x32xbf16, #blocked3> -> tensor<256x128xbf16, #blocked10>
      %48 = arith.mulf %40, %47 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x128xbf16, #blocked10>
      %49 = ttg.convert_layout %48 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<256x128xbf16, #blocked10> -> tensor<256x128xbf16, #blocked11>
      %result_3 = ttng.tmem_alloc %49 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<256x128xbf16, #blocked11>) -> !ttg.memdesc<256x128xbf16, #tmem1, #ttng.tensor_memory>
      %50 = ttg.local_alloc %28 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 2>} : (tensor<16x128xbf16, #blocked4>) -> !ttg.memdesc<16x128xbf16, #shared4, #smem>
      %51 = ttg.memdesc_trans %50 {loop.cluster = 2 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<16x128xbf16, #shared4, #smem> -> !ttg.memdesc<128x16xbf16, #shared5, #smem>
      %52 = ttng.tc_gen5_mma %result_3, %51, %result[%arg79], %arg78, %true {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x16xbf16, #shared5, #smem>, !ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable>
      %53 = arith.addi %19, %c1_i32 {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : i32
      %54 = arith.subi %20#11, %c1_i32 {loop.cluster = 7 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %55 = arith.cmpi eq, %arg63, %54 {loop.cluster = 7 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %56 = arith.select %55, %false, %true {loop.cluster = 7 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : i1
      %57:2 = scf.if %55 -> (i32, !ttg.async.token) {
        %62 = arith.addi %arg65, %c148_i32 {ttg.partition = array<i32: 0>} : i32
        %63 = arith.remsi %62, %6 {ttg.partition = array<i32: 0>} : i32
        %64 = arith.remsi %63, %c9_i32 {ttg.partition = array<i32: 0>} : i32
        %65 = arith.divsi %63, %c9_i32 {ttg.partition = array<i32: 0>} : i32
        %66 = arith.divsi %65, %9 {ttg.partition = array<i32: 0>} : i32
        %67 = arith.muli %66, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %68 = arith.subi %4, %67 {ttg.partition = array<i32: 0>} : i32
        %69 = arith.minsi %68, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %70 = arith.cmpi sge, %69, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        llvm.intr.assume %70 : i1 {ttg.partition = array<i32: 0>}
        %71 = arith.remsi %65, %69 {ttg.partition = array<i32: 0>} : i32
        %72 = arith.addi %67, %71 {ttg.partition = array<i32: 0>} : i32
        %73 = arith.remsi %65, %9 {ttg.partition = array<i32: 0>} : i32
        %74 = arith.divsi %73, %69 {ttg.partition = array<i32: 0>} : i32
        %75 = tt.addptr %arg58, %72 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %76 = tt.load %75 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %77 = arith.andi %76, %c65535_i32 {ttg.partition = array<i32: 0>} : i32
        %78 = arith.shrsi %76, %c16_i32 {ttg.partition = array<i32: 0>} : i32
        %79 = tt.addptr %arg55, %77 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %80 = tt.load %79 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %81 = tt.addptr %arg56, %77 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %82 = tt.load %81 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %83 = arith.muli %78, %c16_i32 {ttg.partition = array<i32: 0>} : i32
        %84 = arith.muli %74, %c256_i32 {ttg.partition = array<i32: 0>} : i32
        %85 = tt.splat %84 {ttg.partition = array<i32: 0>} : i32 -> tensor<256xi32, #blocked1>
        %86 = arith.addi %85, %10 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked1>
        %87 = arith.cmpi slt, %86, %11 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked1>
        %88 = arith.muli %77, %arg51 {ttg.partition = array<i32: 0>} : i32
        %89 = tt.addptr %arg50, %88 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
        %90 = tt.splat %89 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
        %91 = tt.addptr %90, %86 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
        %92 = arith.cmpi eq, %64, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        %93 = scf.if %92 -> (tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>) {
          %103 = tt.load %91, %87, %cst_0 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked1>
          %104 = ttg.convert_layout %103 {ttg.partition = array<i32: 0>} : tensor<256xf32, #blocked1> -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
          scf.yield {ttg.partition = array<i32: 0>} %104 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        } else {
          scf.yield {ttg.partition = array<i32: 0>} %cst_1 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        } {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]}
        %result_4, %token_5 = ttng.tmem_load %result[%52] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x16xf32, #blocked>
        %94 = tt.trans %result_4 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<256x16xf32, #blocked> -> tensor<16x256xf32, #blocked2>
        %95 = tt.expand_dims %93 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xf32, #blocked2>
        %96 = tt.broadcast %95 {ttg.partition = array<i32: 0>} : tensor<1x256xf32, #blocked2> -> tensor<16x256xf32, #blocked2>
        %97 = arith.addf %94, %96 {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #blocked2>
        %98 = arith.subi %c1073741824_i32, %80 {ttg.partition = array<i32: 0>} : i32
        %99 = arith.addi %98, %83 {ttg.partition = array<i32: 0>} : i32
        %100 = arith.addi %82, %80 {ttg.partition = array<i32: 0>} : i32
        %101 = tt.reshape %97 {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #blocked2> -> tensor<1x1x1x16x256xf32, #blocked12>
        %102 = ttg.convert_layout %101 {ttg.partition = array<i32: 0>} : tensor<1x1x1x16x256xf32, #blocked12> -> tensor<1x1x1x16x256xf32, #blocked13>
        tt.descriptor_store %0[%c1073741824_i32, %100, %20#0, %99, %84], %102 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x16x256xf32, #shared>>, tensor<1x1x1x16x256xf32, #blocked13>
        scf.yield {ttg.partition = array<i32: 0, 1>} %62, %token_5 : i32, !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %arg65, %52 : i32, !ttg.async.token
      } {loop.cluster = 7 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
      %58 = arith.addi %arg63, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %59 = arith.subi %20#11, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %60 = arith.cmpi eq, %arg63, %59 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %61 = arith.select %60, %c0_i32, %58 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %61, %20#9, %57#0, %20#10, %20#11, %53, %20#0, %20#1, %20#2, %20#3, %20#4, %20#5, %20#6, %20#7, %20#8, %56, %57#1 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0>, array<i32: 1>, array<i32: 0, 1, 2>, array<i32: 2>, array<i32: 0>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
