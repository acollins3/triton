#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_persistent_inner_loop_kernel(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.addi %arg22, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.divsi %3, %1 : i32
    %5 = arith.remsi %3, %1 : i32
    %6 = arith.cmpi slt, %0, %5 : i32
    %7 = scf.if %6 -> (i32) {
      %12 = arith.addi %4, %c1_i32 : i32
      scf.yield %12 : i32
    } else {
      scf.yield %4 : i32
    }
    %8 = tt.splat %arg24 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.splat %arg24 : f32 -> tensor<64x64xf32, #blocked>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %11 = scf.for %arg25 = %c0_i32 to %7 step %c1_i32 iter_args(%arg26 = %0) -> (i32)  : i32 {
      %12 = arith.muli %arg26, %c64_i32 {ttg.partition = array<i32: 0, 1, 3>} : i32
      %13 = tt.descriptor_load %arg0[%12, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
      %14 = ttg.local_alloc %13 {ttg.partition = array<i32: 3>} : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %result, %token = ttng.tmem_alloc {ttg.partition = array<i32: 0, 2>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_2, %token_3 = ttng.tmem_alloc {ttg.partition = array<i32: 1, 2>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %15 = ttng.tmem_store %cst_1, %result_2[%token_3], %true {ttg.partition = array<i32: 1>} : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %16:4 = scf.for %arg27 = %c0_i32 to %arg23 step %c64_i32 iter_args(%arg28 = %cst_0, %arg29 = %cst, %arg30 = %token, %arg31 = %15) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %30 = tt.descriptor_load %arg5[%arg27, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
        %31 = ttg.local_alloc %30 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
        %32 = ttg.memdesc_trans %31 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
        %33 = ttng.tc_gen5_mma %14, %32, %result[%arg30], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_6, %token_7 = ttng.tmem_load %result[%33] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
        %34 = "tt.reduce"(%result_6) <{axis = 1 : i32}> ({
        ^bb0(%arg32: f32, %arg33: f32):
          %55 = arith.maxnumf %arg32, %arg33 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %55 {ttg.partition = array<i32: 0>} : f32
        }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %35 = arith.mulf %34, %8 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %36 = arith.maxnumf %arg28, %35 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %37 = arith.mulf %result_6, %9 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %38 = tt.expand_dims %36 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
        %39 = tt.broadcast %38 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %40 = arith.subf %37, %39 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %41 = math.exp2 %40 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %42 = arith.subf %arg28, %36 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %43 = math.exp2 %42 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %44 = "tt.reduce"(%41) <{axis = 1 : i32}> ({
        ^bb0(%arg32: f32, %arg33: f32):
          %55 = arith.addf %arg32, %arg33 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %55 {ttg.partition = array<i32: 0>} : f32
        }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %45 = tt.expand_dims %43 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
        %46 = tt.broadcast %45 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %result_8, %token_9 = ttng.tmem_load %result_2[%arg31] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
        %47 = arith.mulf %result_8, %46 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : tensor<64x64xf32, #blocked>
        %48 = tt.descriptor_load %arg10[%arg27, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
        %49 = ttg.local_alloc %48 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 3>} : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
        %50 = arith.truncf %41 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
        %result_10 = ttng.tmem_alloc %50 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory>
        %51 = ttng.tmem_store %47, %result_2[%token_9], %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %52 = ttng.tc_gen5_mma %result_10, %49, %result_2[%51], %true, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %53 = arith.mulf %arg29, %43 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %54 = arith.addf %53, %44 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %36, %54, %token_7, %52 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 3 : i32, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 2>, array<i32: 1>]}
      %result_4, %token_5 = ttng.tmem_load %result_2[%16#3] {ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
      %17 = arith.truncf %result_4 {ttg.partition = array<i32: 1>} : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
      %18 = ttg.convert_layout %17 {ttg.partition = array<i32: 1>} : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #blocked2>
      tt.descriptor_store %arg15[%12, %c0_i32], %18 {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<64x64xf16, #shared>>, tensor<64x64xf16, #blocked2>
      %19 = tt.addptr %arg20, %12 {ttg.partition = array<i32: 0>} : !tt.ptr<f16>, i32
      %20 = tt.splat %19 {ttg.partition = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1>
      %21 = tt.addptr %20, %10 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1>
      %22 = arith.truncf %16#1 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = ttg.convert_layout %22 {ttg.partition = array<i32: 0>} : tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf16, #blocked1>
      tt.store %21, %23 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f16>, #blocked1>
      %24 = tt.addptr %arg21, %12 {ttg.partition = array<i32: 0>} : !tt.ptr<f16>, i32
      %25 = tt.splat %24 {ttg.partition = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1>
      %26 = tt.addptr %25, %10 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1>
      %27 = arith.truncf %16#0 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
      %28 = ttg.convert_layout %27 {ttg.partition = array<i32: 0>} : tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf16, #blocked1>
      tt.store %26, %28 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f16>, #blocked1>
      %29 = arith.addi %arg26, %1 {ttg.partition = array<i32: 0, 1, 3>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %29 : i32
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 3>], ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
