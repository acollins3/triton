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
  tt.func public @_p_matmul_ogs_NNN_fp16xfp16xfp16_128x256x64x1(%arg0: !tt.tensordesc<tensor<1x1x1x128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: !tt.tensordesc<tensor<1x1x1x128x64xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: !tt.tensordesc<tensor<1x64x256xf16, #shared1>>, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: i32 {tt.divisibility = 16 : i32}, %arg43: i32 {tt.divisibility = 16 : i32}, %arg44: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg45: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg46: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg47: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg48: i32 {tt.divisibility = 16 : i32}, %arg49: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked>
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked1>
    %0 = tt.make_tensor_descriptor %arg11, [%arg1, %arg2, %arg3, %arg4, %arg5], [%arg6, %arg7, %arg8, %arg9, %c1_i64] : <f16>, <tensor<1x1x1x128x128xf16, #shared>>
    %1 = tt.addptr %arg46, %c128_i32 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = arith.subi %arg48, %2 : i32
    %4 = arith.subi %arg48, %3 : i32
    %5 = tt.get_program_id x : i32
    %6 = arith.subi %5, %c64_i32 : i32
    %7 = arith.addi %arg42, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = arith.maxsi %8, %c1_i32 : i32
    %10 = arith.cmpi sgt, %9, %c0_i32 : i32
    %11 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %12 = tt.splat %arg41 : i32 -> tensor<256xi32, #blocked>
    %13 = scf.for %arg50 = %5 to %4 step %c64_i32 iter_args(%arg51 = %6) -> (i32)  : i32 {
      %14 = arith.remsi %arg50, %4 {ttg.partition = array<i32: 0, 7>} : i32
      %15 = arith.divsi %14, %c8_i32 {ttg.partition = array<i32: 0, 7>} : i32
      %16 = arith.muli %15, %c8_i32 {ttg.partition = array<i32: 0, 7>} : i32
      %17 = arith.subi %4, %16 {ttg.partition = array<i32: 0, 7>} : i32
      %18 = arith.minsi %17, %c8_i32 {ttg.partition = array<i32: 0, 7>} : i32
      %19 = arith.cmpi sge, %18, %c0_i32 {ttg.partition = array<i32: 7>} : i32
      llvm.intr.assume %19 : i1 {ttg.partition = array<i32: 7>}
      %20 = arith.remsi %14, %18 {ttg.partition = array<i32: 0>} : i32
      %21 = arith.addi %16, %20 {ttg.partition = array<i32: 0>} : i32
      %22 = arith.remsi %14, %c8_i32 {ttg.partition = array<i32: 7>} : i32
      %23 = arith.divsi %22, %18 {ttg.partition = array<i32: 7>} : i32
      %24 = tt.addptr %arg47, %21 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
      %25 = tt.load %24 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
      %26 = arith.andi %25, %c65535_i32 {ttg.partition = array<i32: 1>} : i32
      %27 = arith.shrsi %25, %c16_i32 {ttg.partition = array<i32: 4>} : i32
      %28 = tt.addptr %arg44, %26 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
      %29 = tt.load %28 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
      %30 = tt.addptr %arg45, %26 {ttg.partition = array<i32: 3>} : !tt.ptr<i32>, i32
      %31 = tt.load %30 {ttg.partition = array<i32: 3>} : !tt.ptr<i32>
      %32 = arith.muli %27, %c128_i32 {ttg.partition = array<i32: 4>} : i32
      %33 = arith.muli %23, %c256_i32 {ttg.partition = array<i32: 7>} : i32
      llvm.intr.assume %10 : i1 {ttg.partition = array<i32: 0>}
      %34 = arith.subi %c1073741824_i32, %29 {ttg.partition = array<i32: 4>} : i32
      %35 = arith.addi %34, %32 {ttg.partition = array<i32: 4>} : i32
      %36 = arith.addi %31, %29 {ttg.partition = array<i32: 3>} : i32
      %result, %token = ttng.tmem_alloc {ttg.partition = array<i32: 5, 6>} : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %37 = ttng.tmem_store %cst_0, %result[%token], %true {ttg.partition = array<i32: 5>} : tensor<128x256xf32, #blocked1> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %38:2 = scf.for %arg52 = %c0_i32 to %9 step %c1_i32 iter_args(%arg53 = %false, %arg54 = %37) -> (i1, !ttg.async.token)  : i32 {
        %88 = arith.muli %arg52, %c64_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 7>} : i32
        %89 = tt.descriptor_load %arg15[%c1073741824_i32, %36, %c0_i32, %35, %88] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 7>} : !tt.tensordesc<tensor<1x1x1x128x64xf16, #shared>> -> tensor<128x64xf16, #blocked2>
        %90 = ttg.local_alloc %89 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 7>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared2, #smem>
        %91 = tt.descriptor_load %arg29[%26, %88, %33] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 7>} : !tt.tensordesc<tensor<1x64x256xf16, #shared1>> -> tensor<64x256xf16, #blocked3>
        %92 = ttg.local_alloc %91 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 7>} : (tensor<64x256xf16, #blocked3>) -> !ttg.memdesc<64x256xf16, #shared2, #smem>
        %93 = ttng.tc_gen5_mma %90, %92, %result[%arg54], %arg53, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 6>} : !ttg.memdesc<128x64xf16, #shared2, #smem>, !ttg.memdesc<64x256xf16, #shared2, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 6, 7>} %true, %93 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 3 : i32, ttg.partition = array<i32: 6, 7>, ttg.partition.outputs = [array<i32: 6>, array<i32: 6>]}
      %39 = arith.addi %arg51, %c64_i32 {ttg.partition = array<i32: 5>} : i32
      %40 = arith.remsi %39, %4 {ttg.partition = array<i32: 5>} : i32
      %41 = arith.divsi %40, %c8_i32 {ttg.partition = array<i32: 5>} : i32
      %42 = arith.muli %41, %c8_i32 {ttg.partition = array<i32: 5>} : i32
      %43 = arith.subi %4, %42 {ttg.partition = array<i32: 5>} : i32
      %44 = arith.minsi %43, %c8_i32 {ttg.partition = array<i32: 5>} : i32
      %45 = arith.cmpi sge, %44, %c0_i32 {ttg.partition = array<i32: 5>} : i32
      llvm.intr.assume %45 : i1 {ttg.partition = array<i32: 5>}
      %46 = arith.remsi %40, %44 {ttg.partition = array<i32: 5>} : i32
      %47 = arith.addi %42, %46 {ttg.partition = array<i32: 5>} : i32
      %48 = arith.remsi %40, %c8_i32 {ttg.partition = array<i32: 5>} : i32
      %49 = arith.divsi %48, %44 {ttg.partition = array<i32: 5>} : i32
      %50 = tt.addptr %arg47, %47 {ttg.partition = array<i32: 5>} : !tt.ptr<i32>, i32
      %51 = tt.load %50 {ttg.partition = array<i32: 5>} : !tt.ptr<i32>
      %52 = arith.andi %51, %c65535_i32 {ttg.partition = array<i32: 5>} : i32
      %53 = arith.shrsi %51, %c16_i32 {ttg.partition = array<i32: 5>} : i32
      %54 = tt.addptr %arg44, %52 {ttg.partition = array<i32: 5>} : !tt.ptr<i32>, i32
      %55 = tt.load %54 {ttg.partition = array<i32: 5>} : !tt.ptr<i32>
      %56 = tt.addptr %arg45, %52 {ttg.partition = array<i32: 5>} : !tt.ptr<i32>, i32
      %57 = tt.load %56 {ttg.partition = array<i32: 5>} : !tt.ptr<i32>
      %58 = arith.muli %53, %c128_i32 {ttg.partition = array<i32: 5>} : i32
      %59 = arith.muli %49, %c256_i32 {ttg.partition = array<i32: 5>} : i32
      %60 = tt.splat %59 {ttg.partition = array<i32: 5>} : i32 -> tensor<256xi32, #blocked>
      %61 = arith.addi %60, %11 {ttg.partition = array<i32: 5>} : tensor<256xi32, #blocked>
      %62 = arith.cmpi slt, %61, %12 {ttg.partition = array<i32: 5>} : tensor<256xi32, #blocked>
      %63 = arith.muli %52, %arg40 {ttg.partition = array<i32: 5>} : i32
      %64 = tt.addptr %arg39, %63 {ttg.partition = array<i32: 5>} : !tt.ptr<f32>, i32
      %65 = tt.splat %64 {ttg.partition = array<i32: 5>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
      %66 = tt.addptr %65, %61 {ttg.partition = array<i32: 5>} : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
      %67 = tt.load %66, %62, %cst {ttg.partition = array<i32: 5>} : tensor<256x!tt.ptr<f32>, #blocked>
      %result_1, %token_2 = ttng.tmem_load %result[%38#1] {ttg.partition = array<i32: 5>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked1>
      %68 = tt.reshape %result_1 {ttg.partition = array<i32: 5>} : tensor<128x256xf32, #blocked1> -> tensor<128x2x128xf32, #blocked4>
      %69 = tt.trans %68 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 5>} : tensor<128x2x128xf32, #blocked4> -> tensor<128x128x2xf32, #blocked5>
      %70 = ttg.convert_layout %69 {ttg.partition = array<i32: 5>} : tensor<128x128x2xf32, #blocked5> -> tensor<128x128x2xf32, #blocked6>
      %outLHS, %outRHS = tt.split %70 {ttg.partition = array<i32: 5>} : tensor<128x128x2xf32, #blocked6> -> tensor<128x128xf32, #blocked7>
      %71 = tt.reshape %67 {ttg.partition = array<i32: 5>} : tensor<256xf32, #blocked> -> tensor<2x128xf32, #blocked8>
      %72 = tt.trans %71 {order = array<i32: 1, 0>, ttg.partition = array<i32: 5>} : tensor<2x128xf32, #blocked8> -> tensor<128x2xf32, #blocked9>
      %73 = ttg.convert_layout %72 {ttg.partition = array<i32: 5>} : tensor<128x2xf32, #blocked9> -> tensor<128x2xf32, #linear>
      %outLHS_3, %outRHS_4 = tt.split %73 {ttg.partition = array<i32: 5>} : tensor<128x2xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked7}>>
      %74 = tt.expand_dims %outLHS_3 {axis = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x128xf32, #blocked7>
      %75 = tt.broadcast %74 {ttg.partition = array<i32: 5>} : tensor<1x128xf32, #blocked7> -> tensor<128x128xf32, #blocked7>
      %76 = arith.addf %outLHS, %75 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked7>
      %77 = arith.truncf %76 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked7> to tensor<128x128xf16, #blocked7>
      %78 = arith.subi %c1073741824_i32, %55 {ttg.partition = array<i32: 5>} : i32
      %79 = arith.addi %78, %58 {ttg.partition = array<i32: 5>} : i32
      %80 = arith.addi %57, %55 {ttg.partition = array<i32: 5>} : i32
      %81 = tt.reshape %77 {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked7> -> tensor<1x1x1x128x128xf16, #blocked10>
      tt.descriptor_store %0[%c1073741824_i32, %80, %c0_i32, %79, %59], %81 {ttg.partition = array<i32: 5>} : !tt.tensordesc<tensor<1x1x1x128x128xf16, #shared>>, tensor<1x1x1x128x128xf16, #blocked10>
      %82 = tt.expand_dims %outRHS_4 {axis = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x128xf32, #blocked7>
      %83 = tt.broadcast %82 {ttg.partition = array<i32: 5>} : tensor<1x128xf32, #blocked7> -> tensor<128x128xf32, #blocked7>
      %84 = arith.addf %outRHS, %83 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked7>
      %85 = arith.addi %59, %c128_i32 {ttg.partition = array<i32: 5>} : i32
      %86 = arith.truncf %84 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked7> to tensor<128x128xf16, #blocked7>
      %87 = tt.reshape %86 {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked7> -> tensor<1x1x1x128x128xf16, #blocked10>
      tt.descriptor_store %0[%c1073741824_i32, %80, %c0_i32, %79, %85], %87 {ttg.partition = array<i32: 5>} : !tt.tensordesc<tensor<1x1x1x128x128xf16, #shared>>, tensor<1x1x1x128x128xf16, #blocked10>
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5, 6, 7>} %39 : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5, 6, 7>, ttg.partition.outputs = [array<i32: 5>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
