#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 2, 1], order = [0, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 128, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 2], order = [0, 1, 2]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 8], threadsPerWarp = [1, 1, 1, 2, 16], warpsPerCTA = [1, 1, 1, 8, 1], order = [4, 3, 2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [1, 0], [2, 0], [4, 0]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 0]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1, 1], CTAOrder = [4, 3, 2, 1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_p_matmul_ogs_NNN_fp16xfp16xfp16_128x256x64x1(%arg0: !tt.tensordesc<tensor<1x1x1x128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: !tt.tensordesc<tensor<1x1x1x128x64xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: !tt.tensordesc<tensor<1x64x256xf16, #shared1>>, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: i32 {tt.divisibility = 16 : i32}, %arg43: i32 {tt.divisibility = 16 : i32}, %arg44: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg45: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg46: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg47: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg48: i32, %arg49: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked1>
    %0 = tt.make_tensor_descriptor %arg11, [%arg1, %arg2, %arg3, %arg4, %arg5], [%arg6, %arg7, %arg8, %arg9, %c1_i64] : <f16>, <tensor<1x1x1x128x128xf16, #shared>>
    %1 = tt.addptr %arg46, %c4_i32 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = arith.subi %arg48, %2 : i32
    %4 = arith.subi %arg48, %3 : i32
    %5 = tt.get_program_id x : i32
    %6 = arith.subi %5, %c5_i32 : i32
    %7 = arith.addi %arg42, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = arith.maxsi %8, %c1_i32 : i32
    %10 = arith.cmpi sgt, %9, %c0_i32 : i32
    %11 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %12 = tt.splat %arg41 : i32 -> tensor<256xi32, #blocked>
    %13 = arith.subi %4, %5 : i32
    %14 = arith.ceildivsi %13, %c5_i32 : i32
    %15 = arith.maxsi %9, %c1_i32 : i32
    %16 = arith.muli %14, %15 : i32
    %17 = arith.subi %5, %c5_i32 : i32
    %18 = arith.subi %15, %c1_i32 : i32
    %19 = arith.subi %15, %c1_i32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %20 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x256xf32, #blocked1> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %21:11 = scf.for %arg50 = %c0_i32 to %16 step %c1_i32 iter_args(%arg51 = %c0_i32, %arg52 = %17, %arg53 = %6, %arg54 = %c0_i32, %arg55 = %c0_i32, %arg56 = %c0_i32, %arg57 = %c0_i32, %arg58 = %c0_i32, %arg59 = %c0_i32, %arg60 = %false, %arg61 = %20) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      %22 = arith.cmpi eq, %arg51, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %23 = arith.select %22, %c0_i32, %arg54 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %24:6 = scf.if %22 -> (i32, i32, i32, i32, i32, i32) {
        %41 = arith.addi %arg52, %c5_i32 {ttg.partition = array<i32: 2>} : i32
        %42 = arith.remsi %41, %4 {ttg.partition = array<i32: 2>} : i32
        %43 = arith.divsi %42, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %44 = arith.muli %43, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %45 = arith.subi %4, %44 {ttg.partition = array<i32: 2>} : i32
        %46 = arith.minsi %45, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %47 = arith.cmpi sge, %46, %c0_i32 {ttg.partition = array<i32: 2>} : i32
        llvm.intr.assume %47 : i1 {ttg.partition = array<i32: 2>}
        %48 = arith.remsi %42, %46 {ttg.partition = array<i32: 2>} : i32
        %49 = arith.addi %44, %48 {ttg.partition = array<i32: 2>} : i32
        %50 = arith.remsi %42, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %51 = arith.divsi %50, %46 {ttg.partition = array<i32: 2>} : i32
        %52 = tt.addptr %arg47, %49 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %53 = tt.load %52 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %54 = arith.andi %53, %c65535_i32 {ttg.partition = array<i32: 2>} : i32
        %55 = arith.shrsi %53, %c16_i32 {ttg.partition = array<i32: 2>} : i32
        %56 = tt.addptr %arg44, %54 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %57 = tt.load %56 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %58 = tt.addptr %arg45, %54 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %59 = tt.load %58 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %60 = arith.muli %55, %c128_i32 {ttg.partition = array<i32: 2>} : i32
        %61 = arith.muli %51, %c256_i32 {ttg.partition = array<i32: 2>} : i32
        llvm.intr.assume %10 : i1 {ttg.partition = array<i32: 2>}
        scf.yield {ttg.partition = array<i32: 2>} %54, %57, %59, %60, %61, %41 : i32, i32, i32, i32, i32, i32
      } else {
        scf.yield {ttg.partition = array<i32: 2>} %arg55, %arg56, %arg57, %arg58, %arg59, %arg52 : i32, i32, i32, i32, i32, i32
      } {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>]}
      %25 = arith.muli %23, %c64_i32 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %26 = arith.subi %c1073741824_i32, %24#1 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %27 = arith.addi %26, %24#3 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %28 = arith.addi %24#2, %24#1 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %29 = tt.descriptor_load %arg15[%c1073741824_i32, %28, %c0_i32, %27, %25] {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x1x1x128x64xf16, #shared>> -> tensor<128x64xf16, #blocked2>
      %30 = ttg.local_alloc %29 {loop.cluster = 5 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared2, #smem>
      %31 = tt.descriptor_load %arg29[%24#0, %25, %24#4] {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x64x256xf16, #shared1>> -> tensor<64x256xf16, #blocked3>
      %32 = ttg.local_alloc %31 {loop.cluster = 5 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 2>} : (tensor<64x256xf16, #blocked3>) -> !ttg.memdesc<64x256xf16, #shared2, #smem>
      %33 = ttng.tc_gen5_mma %30, %32, %result[%arg61], %arg60, %true {loop.cluster = 5 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared2, #smem>, !ttg.memdesc<64x256xf16, #shared2, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %34 = arith.addi %23, %c1_i32 {loop.cluster = 8 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : i32
      %35 = arith.cmpi eq, %arg51, %18 {loop.cluster = 4 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %36 = arith.select %35, %false, %true {loop.cluster = 4 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : i1
      %37:2 = scf.if %35 -> (i32, !ttg.async.token) {
        %41 = arith.addi %arg53, %c5_i32 {ttg.partition = array<i32: 0>} : i32
        %42 = arith.remsi %41, %4 {ttg.partition = array<i32: 0>} : i32
        %43 = arith.divsi %42, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %44 = arith.muli %43, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %45 = arith.subi %4, %44 {ttg.partition = array<i32: 0>} : i32
        %46 = arith.minsi %45, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %47 = arith.cmpi sge, %46, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        llvm.intr.assume %47 : i1 {ttg.partition = array<i32: 0>}
        %48 = arith.remsi %42, %46 {ttg.partition = array<i32: 0>} : i32
        %49 = arith.addi %44, %48 {ttg.partition = array<i32: 0>} : i32
        %50 = arith.remsi %42, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %51 = arith.divsi %50, %46 {ttg.partition = array<i32: 0>} : i32
        %52 = tt.addptr %arg47, %49 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %53 = tt.load %52 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %54 = arith.andi %53, %c65535_i32 {ttg.partition = array<i32: 0>} : i32
        %55 = arith.shrsi %53, %c16_i32 {ttg.partition = array<i32: 0>} : i32
        %56 = tt.addptr %arg44, %54 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %57 = tt.load %56 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %58 = tt.addptr %arg45, %54 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %59 = tt.load %58 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %60 = arith.muli %55, %c128_i32 {ttg.partition = array<i32: 0>} : i32
        %61 = arith.muli %51, %c256_i32 {ttg.partition = array<i32: 0>} : i32
        %62 = tt.splat %61 {ttg.partition = array<i32: 0>} : i32 -> tensor<256xi32, #blocked>
        %63 = arith.addi %62, %11 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked>
        %64 = arith.cmpi slt, %63, %12 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked>
        %65 = arith.muli %54, %arg40 {ttg.partition = array<i32: 0>} : i32
        %66 = tt.addptr %arg39, %65 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
        %67 = tt.splat %66 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
        %68 = tt.addptr %67, %63 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
        %69 = tt.load %68, %64, %cst {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked>
        %result_1, %token_2 = ttng.tmem_load %result[%33] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked1>
        %70 = tt.reshape %result_1 {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #blocked1> -> tensor<128x2x128xf32, #blocked4>
        %71 = tt.trans %70 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x128xf32, #blocked4> -> tensor<128x128x2xf32, #blocked5>
        %72 = ttg.convert_layout %71 {ttg.partition = array<i32: 0>} : tensor<128x128x2xf32, #blocked5> -> tensor<128x128x2xf32, #blocked6>
        %outLHS, %outRHS = tt.split %72 {ttg.partition = array<i32: 0>} : tensor<128x128x2xf32, #blocked6> -> tensor<128x128xf32, #blocked7>
        %73 = tt.reshape %69 {ttg.partition = array<i32: 0>} : tensor<256xf32, #blocked> -> tensor<2x128xf32, #blocked8>
        %74 = tt.trans %73 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<2x128xf32, #blocked8> -> tensor<128x2xf32, #blocked9>
        %75 = ttg.convert_layout %74 {ttg.partition = array<i32: 0>} : tensor<128x2xf32, #blocked9> -> tensor<128x2xf32, #linear>
        %outLHS_3, %outRHS_4 = tt.split %75 {ttg.partition = array<i32: 0>} : tensor<128x2xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked7}>>
        %76 = tt.expand_dims %outLHS_3 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x128xf32, #blocked7>
        %77 = tt.broadcast %76 {ttg.partition = array<i32: 0>} : tensor<1x128xf32, #blocked7> -> tensor<128x128xf32, #blocked7>
        %78 = arith.addf %outLHS, %77 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked7>
        %79 = arith.truncf %78 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked7> to tensor<128x128xf16, #blocked7>
        %80 = arith.subi %c1073741824_i32, %57 {ttg.partition = array<i32: 0>} : i32
        %81 = arith.addi %80, %60 {ttg.partition = array<i32: 0>} : i32
        %82 = arith.addi %59, %57 {ttg.partition = array<i32: 0>} : i32
        %83 = tt.reshape %79 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked7> -> tensor<1x1x1x128x128xf16, #blocked10>
        tt.descriptor_store %0[%c1073741824_i32, %82, %c0_i32, %81, %61], %83 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x128x128xf16, #shared>>, tensor<1x1x1x128x128xf16, #blocked10>
        %84 = tt.expand_dims %outRHS_4 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x128xf32, #blocked7>
        %85 = tt.broadcast %84 {ttg.partition = array<i32: 0>} : tensor<1x128xf32, #blocked7> -> tensor<128x128xf32, #blocked7>
        %86 = arith.addf %outRHS, %85 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked7>
        %87 = arith.addi %61, %c128_i32 {ttg.partition = array<i32: 0>} : i32
        %88 = arith.truncf %86 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked7> to tensor<128x128xf16, #blocked7>
        %89 = tt.reshape %88 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked7> -> tensor<1x1x1x128x128xf16, #blocked10>
        tt.descriptor_store %0[%c1073741824_i32, %82, %c0_i32, %81, %87], %89 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x128x128xf16, #shared>>, tensor<1x1x1x128x128xf16, #blocked10>
        scf.yield {ttg.partition = array<i32: 0, 1>} %41, %token_2 : i32, !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %arg53, %33 : i32, !ttg.async.token
      } {loop.cluster = 10 : i32, loop.stage = 5 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
      %38 = arith.addi %arg51, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %39 = arith.cmpi eq, %arg51, %19 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %40 = arith.select %39, %c0_i32, %38 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %40, %24#5, %37#0, %34, %24#0, %24#1, %24#2, %24#3, %24#4, %36, %37#1 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, !ttg.async.token
    } {tt.scheduled_max_stage = 5 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 2>, array<i32: 0>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
