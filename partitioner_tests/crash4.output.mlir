#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 8], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 2, 1, 2, 1], order = [4, 3, 2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1, 16, 1], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 4], order = [4, 3, 2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 16], threadsPerWarp = [1, 1, 1, 2, 16], warpsPerCTA = [1, 1, 1, 4, 1], order = [4, 3, 2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1, 1], CTAOrder = [4, 3, 2, 1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1, 1], CTAOrder = [4, 3, 2, 1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared6 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared7 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared8 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [128, 0]]}, alignment = 128>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_p_matmul_ogs_NNT_fp8e5xfp8e5xmxfp4_16x256x128x1(%arg0: !tt.tensordesc<tensor<1x1x1x16x256xf8E5M2, #shared>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg16: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg17: !tt.tensordesc<tensor<1x1x1x16x128xf8E5M2, #shared>>, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg32: !tt.tensordesc<tensor<1x256x64xui8, #shared1>>, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: !tt.tensordesc<tensor<1x2x1x2x256xui8, #shared2>>, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: i32, %arg47: i32, %arg48: i64, %arg49: i64, %arg50: i64, %arg51: i64, %arg52: i64, %arg53: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg54: i32 {tt.divisibility = 16 : i32}, %arg55: i32 {tt.divisibility = 16 : i32}, %arg56: i32 {tt.divisibility = 16 : i32}, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg59: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg60: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg61: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg62: i32 {tt.divisibility = 16 : i32}, %arg63: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<127> : tensor<16x4xi8, #linear>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_1 = arith.constant dense<2147483647> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c932333861_i32 = arith.constant 932333861 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c2139095040_i32 = arith.constant 2139095040 : i32
    %cst_2 = arith.constant 1.000000e-30 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %true = arith.constant true
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked1>
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c127_i32 = arith.constant 127 : i32
    %c4_i32 = arith.constant 4 : i32
    %c31_i32 = arith.constant 31 : i32
    %c8_i32 = arith.constant 8 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<256x16xf32, #blocked2>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #blocked3>
    %0 = tt.make_tensor_descriptor %arg11, [%arg1, %arg2, %arg3, %arg4, %arg5], [%arg6, %arg7, %arg8, %arg9, %c1_i64] : <f8E5M2>, <tensor<1x1x1x16x256xf8E5M2, #shared>>
    %1 = tt.addptr %arg60, %c128_i32 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = arith.subi %arg62, %2 : i32
    %4 = arith.subi %arg62, %3 : i32
    %5 = tt.get_program_id x : i32
    %6 = arith.subi %5, %c64_i32 : i32
    %7 = arith.addi %arg56, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.maxsi %8, %c1_i32 : i32
    %10 = arith.cmpi sgt, %9, %c0_i32 : i32
    %11 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %13 = tt.splat %arg55 : i32 -> tensor<256xi32, #blocked1>
    %14 = arith.subi %4, %5 : i32
    %15 = arith.ceildivsi %14, %c64_i32 : i32
    %16 = arith.maxsi %9, %c1_i32 : i32
    %17 = arith.muli %15, %16 : i32
    %18 = arith.subi %5, %c64_i32 : i32
    %19 = arith.addi %arg55, %c127_i32 : i32
    %20 = arith.divsi %19, %c128_i32 : i32
    %21 = arith.subi %16, %c1_i32 : i32
    %22 = arith.subi %16, %c1_i32 : i32
    %result = ttng.tmem_alloc %cst : (tensor<16x4xi8, #linear>) -> !ttg.memdesc<16x4xi8, #tmem_scales, #ttng.tensor_memory>
    %result_7, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %23 = ttng.tmem_store %cst_5, %result_7[%token], %true : tensor<256x16xf32, #blocked2> -> !ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %24:12 = scf.for %arg64 = %c0_i32 to %17 step %c1_i32 iter_args(%arg65 = %c0_i32, %arg66 = %18, %arg67 = %6, %arg68 = %cst_0, %arg69 = %c0_i32, %arg70 = %c0_i32, %arg71 = %c0_i32, %arg72 = %c0_i32, %arg73 = %c0_i32, %arg74 = %c0_i32, %arg75 = %false, %arg76 = %23) -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      %41 = arith.cmpi eq, %arg65, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %42 = arith.select %41, %c0_i32, %arg69 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %43:6 = scf.if %41 -> (i32, i32, i32, i32, i32, i32) {
        %72 = arith.addi %arg66, %c64_i32 {ttg.partition = array<i32: 2>} : i32
        %73 = arith.remsi %72, %4 {ttg.partition = array<i32: 2>} : i32
        %74 = arith.divsi %73, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %75 = arith.muli %74, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %76 = arith.subi %4, %75 {ttg.partition = array<i32: 2>} : i32
        %77 = arith.minsi %76, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %78 = arith.cmpi sge, %77, %c0_i32 {ttg.partition = array<i32: 2>} : i32
        llvm.intr.assume %78 : i1 {ttg.partition = array<i32: 2>}
        %79 = arith.remsi %73, %77 {ttg.partition = array<i32: 2>} : i32
        %80 = arith.addi %75, %79 {ttg.partition = array<i32: 2>} : i32
        %81 = arith.remsi %73, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %82 = arith.divsi %81, %77 {ttg.partition = array<i32: 2>} : i32
        %83 = tt.addptr %arg61, %80 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %84 = tt.load %83 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %85 = arith.andi %84, %c65535_i32 {ttg.partition = array<i32: 2>} : i32
        %86 = arith.shrsi %84, %c16_i32 {ttg.partition = array<i32: 2>} : i32
        %87 = tt.addptr %arg58, %85 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %88 = tt.load %87 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %89 = tt.addptr %arg59, %85 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>, i32
        %90 = tt.load %89 {ttg.partition = array<i32: 2>} : !tt.ptr<i32>
        %91 = arith.muli %86, %c16_i32 {ttg.partition = array<i32: 2>} : i32
        %92 = arith.muli %82, %c256_i32 {ttg.partition = array<i32: 2>} : i32
        llvm.intr.assume %10 : i1 {ttg.partition = array<i32: 2>}
        scf.yield {ttg.partition = array<i32: 2>} %85, %88, %90, %91, %92, %72 : i32, i32, i32, i32, i32, i32
      } else {
        scf.yield {ttg.partition = array<i32: 2>} %arg70, %arg71, %arg72, %arg73, %arg74, %arg66 : i32, i32, i32, i32, i32, i32
      } {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>]}
      %44 = arith.muli %42, %c128_i32 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %45 = arith.muli %42, %c64_i32 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %46 = arith.subi %c1073741824_i32, %43#1 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %47 = arith.addi %46, %43#3 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %48 = arith.addi %43#2, %43#1 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %49 = tt.descriptor_load %arg17[%c1073741824_i32, %48, %c0_i32, %47, %44] {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x1x1x16x128xf8E5M2, #shared>> -> tensor<16x128xf8E5M2, #blocked4>
      %50 = tt.descriptor_load %arg32[%43#0, %43#4, %45] {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x256x64xui8, #shared1>> -> tensor<256x64xi8, #blocked5>
      %51 = ttg.local_alloc %50 {loop.cluster = 5 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 2>} : (tensor<256x64xi8, #blocked5>) -> !ttg.memdesc<256x64xi8, #shared3, #smem>
      %52 = arith.divsi %45, %c16_i32 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %53 = arith.muli %43#0, %20 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %54 = arith.divsi %43#4, %c128_i32 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %55 = arith.addi %53, %54 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %56 = arith.divsi %52, %c4_i32 {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %57 = tt.descriptor_load %arg42[%c0_i32, %55, %56, %c0_i32, %c0_i32] {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x2x1x2x256xui8, #shared2>> -> tensor<1x2x1x2x256xi8, #blocked6>
      %58 = ttg.local_alloc %57 {loop.cluster = 5 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 2>} : (tensor<1x2x1x2x256xi8, #blocked6>) -> !ttg.memdesc<1x2x1x2x256xi8, #shared2, #smem>
      %59 = ttg.local_alloc %49 {loop.cluster = 5 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 2>} : (tensor<16x128xf8E5M2, #blocked4>) -> !ttg.memdesc<16x128xf8E5M2, #shared4, #smem>
      %60 = ttg.memdesc_trans %59 {loop.cluster = 5 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<16x128xf8E5M2, #shared4, #smem> -> !ttg.memdesc<128x16xf8E5M2, #shared5, #smem>
      %61 = ttg.memdesc_reshape %58 {loop.cluster = 5 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x2x1x2x256xi8, #shared2, #smem> -> !ttg.memdesc<2x1x32x4x4xi8, #shared6, #smem>
      %62 = ttg.memdesc_trans %61 {loop.cluster = 5 : i32, loop.stage = 3 : i32, order = array<i32: 0, 3, 2, 1, 4>, ttg.partition = array<i32: 1>} : !ttg.memdesc<2x1x32x4x4xi8, #shared6, #smem> -> !ttg.memdesc<2x4x32x1x4xi8, #shared7, #smem>
      %63 = ttg.memdesc_reshape %62 {loop.cluster = 5 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<2x4x32x1x4xi8, #shared7, #smem> -> !ttg.memdesc<256x4xi8, #shared8, #smem>
      %64 = ttng.tc_gen5_mma_scaled %51, %60, %result_7[%arg76], %63, %result, %arg75, %true lhs = e2m1 rhs = e5m2 {loop.cluster = 5 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xi8, #shared3, #smem>, !ttg.memdesc<128x16xf8E5M2, #shared5, #smem>, !ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x4xi8, #shared8, #smem>, !ttg.memdesc<16x4xi8, #tmem_scales, #ttng.tensor_memory>
      %65 = arith.addi %42, %c1_i32 {loop.cluster = 8 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : i32
      %66 = arith.cmpi eq, %arg65, %21 {loop.cluster = 4 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %67 = arith.select %66, %false, %true {loop.cluster = 4 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : i1
      %68:3 = scf.if %66 -> (i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, !ttg.async.token) {
        %72 = arith.addi %arg67, %c64_i32 {ttg.partition = array<i32: 0>} : i32
        %73 = arith.remsi %72, %4 {ttg.partition = array<i32: 0>} : i32
        %74 = arith.divsi %73, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %75 = arith.muli %74, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %76 = arith.subi %4, %75 {ttg.partition = array<i32: 0>} : i32
        %77 = arith.minsi %76, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %78 = arith.cmpi sge, %77, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        llvm.intr.assume %78 : i1 {ttg.partition = array<i32: 0>}
        %79 = arith.remsi %73, %77 {ttg.partition = array<i32: 0>} : i32
        %80 = arith.addi %75, %79 {ttg.partition = array<i32: 0>} : i32
        %81 = arith.remsi %73, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %82 = arith.divsi %81, %77 {ttg.partition = array<i32: 0>} : i32
        %83 = tt.addptr %arg61, %80 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %84 = tt.load %83 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %85 = arith.andi %84, %c65535_i32 {ttg.partition = array<i32: 0>} : i32
        %86 = arith.shrsi %84, %c16_i32 {ttg.partition = array<i32: 0>} : i32
        %87 = tt.addptr %arg58, %85 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %88 = tt.load %87 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %89 = tt.addptr %arg59, %85 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %90 = tt.load %89 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %91 = arith.muli %86, %c16_i32 {ttg.partition = array<i32: 0>} : i32
        %92 = arith.muli %82, %c256_i32 {ttg.partition = array<i32: 0>} : i32
        %93 = tt.splat %91 {ttg.partition = array<i32: 0>} : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %94 = arith.addi %93, %11 {ttg.partition = array<i32: 0>} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %95 = tt.splat %88 {ttg.partition = array<i32: 0>} : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %96 = arith.cmpi slt, %94, %95 {ttg.partition = array<i32: 0>} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %97 = tt.splat %92 {ttg.partition = array<i32: 0>} : i32 -> tensor<256xi32, #blocked1>
        %98 = arith.addi %97, %12 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked1>
        %99 = arith.cmpi slt, %98, %13 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked1>
        %100 = arith.muli %85, %arg54 {ttg.partition = array<i32: 0>} : i32
        %101 = tt.addptr %arg53, %100 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
        %102 = tt.splat %101 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
        %103 = tt.addptr %102, %98 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
        %104 = tt.load %103, %99, %cst_4 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked1>
        %105 = tt.load %arg31 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>
        %106 = tt.splat %105 {ttg.partition = array<i32: 0>} : f32 -> tensor<256x16xf32, #blocked2>
        %result_8, %token_9 = ttng.tmem_load %result_7[%64] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x16xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x16xf32, #blocked2>
        %107 = arith.mulf %result_8, %106 {ttg.partition = array<i32: 0>} : tensor<256x16xf32, #blocked2>
        %108 = tt.trans %107 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<256x16xf32, #blocked2> -> tensor<16x256xf32, #blocked3>
        %109 = ttg.convert_layout %104 {ttg.partition = array<i32: 0>} : tensor<256xf32, #blocked1> -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %110 = tt.expand_dims %109 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xf32, #blocked3>
        %111 = tt.broadcast %110 {ttg.partition = array<i32: 0>} : tensor<1x256xf32, #blocked3> -> tensor<16x256xf32, #blocked3>
        %112 = arith.addf %108, %111 {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #blocked3>
        %113 = tt.expand_dims %96 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi1, #blocked3>
        %114 = tt.broadcast %113 {ttg.partition = array<i32: 0>} : tensor<16x1xi1, #blocked3> -> tensor<16x256xi1, #blocked3>
        %115 = arith.select %114, %112, %cst_6 {ttg.partition = array<i32: 0>} : tensor<16x256xi1, #blocked3>, tensor<16x256xf32, #blocked3>
        %116 = tt.reshape %115 allow_reorder {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #blocked3> -> tensor<32x128xf32, #blocked>
        %117 = "tt.reduce"(%116) <{axis = 0 : i32}> ({
        ^bb0(%arg77: f32, %arg78: f32):
          %131 = tt.elementwise_inline_asm "{\0A    max.NaN.xorsign.abs.f32 $0, $1, $2;\0A    }" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true, ttg.partition = array<i32: 0>} %arg77, %arg78 : f32, f32 -> f32
          tt.reduce.return %131 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>} : (tensor<32x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %118 = tt.bitcast %117 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %119 = arith.andi %118, %cst_1 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %120 = arith.maxui %arg68, %119 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %121 = tt.load %arg15 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>
        %122 = arith.divf %cst_3, %121 {ttg.partition = array<i32: 0>} : f32
        %123 = tt.splat %122 {ttg.partition = array<i32: 0>} : f32 -> tensor<16x256xf32, #blocked3>
        %124 = arith.mulf %115, %123 {ttg.partition = array<i32: 0>} : tensor<16x256xf32, #blocked3>
        %125 = tt.fp_to_fp %124 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<16x256xf32, #blocked3> -> tensor<16x256xf8E5M2, #blocked3>
        %126 = arith.subi %c1073741824_i32, %88 {ttg.partition = array<i32: 0>} : i32
        %127 = arith.addi %126, %91 {ttg.partition = array<i32: 0>} : i32
        %128 = arith.addi %90, %88 {ttg.partition = array<i32: 0>} : i32
        %129 = tt.reshape %125 {ttg.partition = array<i32: 0>} : tensor<16x256xf8E5M2, #blocked3> -> tensor<1x1x1x16x256xf8E5M2, #blocked7>
        %130 = ttg.convert_layout %129 {ttg.partition = array<i32: 0>} : tensor<1x1x1x16x256xf8E5M2, #blocked7> -> tensor<1x1x1x16x256xf8E5M2, #blocked8>
        tt.descriptor_store %0[%c1073741824_i32, %128, %c0_i32, %127, %92], %130 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x16x256xf8E5M2, #shared>>, tensor<1x1x1x16x256xf8E5M2, #blocked8>
        scf.yield {ttg.partition = array<i32: 0, 1>} %72, %120, %token_9 : i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %arg67, %arg68, %64 : i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, !ttg.async.token
      } {loop.cluster = 10 : i32, loop.stage = 5 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>]}
      %69 = arith.addi %arg65, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %70 = arith.cmpi eq, %arg65, %22 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %71 = arith.select %70, %c0_i32, %69 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %71, %43#5, %68#0, %68#1, %65, %43#0, %43#1, %43#2, %43#3, %43#4, %67, %68#2 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, i32, i32, i1, !ttg.async.token
    } {tt.scheduled_max_stage = 5 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 2>, array<i32: 0>, array<i32: 0>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %25 = tt.bitcast %24#3 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %26 = tt.reshape %25 allow_reorder : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<128xf32, #blocked9>
    %27 = "tt.reduce"(%26) <{axis = 0 : i32}> ({
    ^bb0(%arg64: f32, %arg65: f32):
      %41 = tt.elementwise_inline_asm "{\0A    max.NaN.xorsign.abs.f32 $0, $1, $2;\0A    }" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %arg64, %arg65 : f32, f32 -> f32
      tt.reduce.return %41 : f32
    }) : (tensor<128xf32, #blocked9>) -> f32
    %28 = tt.bitcast %27 : f32 -> i32
    %29 = arith.andi %28, %c2147483647_i32 : i32
    %30 = arith.minui %29, %c2139095040_i32 : i32
    %31 = tt.bitcast %30 : i32 -> f32
    %32 = tt.bitcast %c932333861_i32 : i32 -> f32
    %33 = math.fma %31, %32, %cst_2 : f32
    %34 = tt.bitcast %33 : f32 -> i32
    %35 = tt.bitcast %arg16 : !tt.ptr<f32> -> !tt.ptr<i32>
    %36 = arith.shrui %34, %c31_i32 : i32
    %37 = arith.cmpi ne, %36, %c0_i32 : i32
    %38 = arith.cmpi eq, %36, %c0_i32 : i32
    %39 = tt.atomic_rmw max, relaxed, gpu, %35, %34, %38 : (!tt.ptr<i32>, i32, i1) -> i32
    %40 = tt.atomic_rmw umin, relaxed, gpu, %35, %34, %37 : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}
