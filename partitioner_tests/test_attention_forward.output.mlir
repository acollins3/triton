#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_inner_loop_kernel(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.descriptor_load %arg0[%1, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
    %3 = ttg.local_alloc %2 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %4 = tt.splat %arg24 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.splat %arg24 : f32 -> tensor<64x64xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %6 = ttng.tmem_store %cst, %result_2[%token_3], %true : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %7:4 = scf.for %arg25 = %c0_i32 to %arg23 step %c64_i32 iter_args(%arg26 = %cst_0, %arg27 = %cst_1, %arg28 = %token, %arg29 = %6) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      %21 = tt.descriptor_load %arg5[%arg25, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %22 = ttg.local_alloc %21 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %23 = ttg.memdesc_trans %22 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %24 = ttng.tc_gen5_mma %3, %23, %result[%arg28], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_6, %token_7 = ttng.tmem_load %result[%24] {loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
      %25 = "tt.reduce"(%result_6) <{axis = 1 : i32}> ({
      ^bb0(%arg30: f32, %arg31: f32):
        %46 = arith.maxnumf %arg30, %arg31 : f32
        tt.reduce.return %46 : f32
      }) {loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %26 = arith.mulf %25, %4 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %27 = arith.maxnumf %arg26, %26 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %28 = arith.mulf %result_6, %5 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x64xf32, #blocked>
      %29 = tt.expand_dims %27 {axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %30 = tt.broadcast %29 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %31 = arith.subf %28, %30 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x64xf32, #blocked>
      %32 = math.exp2 %31 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x64xf32, #blocked>
      %33 = arith.subf %arg26, %27 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %34 = math.exp2 %33 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = "tt.reduce"(%32) <{axis = 1 : i32}> ({
      ^bb0(%arg30: f32, %arg31: f32):
        %46 = arith.addf %arg30, %arg31 : f32
        tt.reduce.return %46 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %36 = tt.expand_dims %34 {axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %37 = tt.broadcast %36 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg29] {loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
      %38 = arith.mulf %result_8, %37 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x64xf32, #blocked>
      %39 = tt.descriptor_load %arg10[%arg25, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %40 = ttg.local_alloc %39 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %41 = arith.truncf %32 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
      %result_10 = ttng.tmem_alloc %41 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory>
      %42 = ttng.tmem_store %38, %result_2[%token_9], %true {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %43 = ttng.tc_gen5_mma %result_10, %40, %result_2[%42], %true, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %44 = arith.mulf %arg27, %34 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %45 = arith.addf %44, %35 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %27, %45, %token_7, %43 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    %result_4, %token_5 = ttng.tmem_load %result_2[%7#3] : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
    %8 = arith.truncf %result_4 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %9 = ttg.convert_layout %8 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #blocked1>
    tt.descriptor_store %arg15[%1, %c0_i32], %9 : !tt.tensordesc<tensor<64x64xf16, #shared>>, tensor<64x64xf16, #blocked1>
    %10 = tt.addptr %arg20, %1 : !tt.ptr<f16>, i32
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked2>
    %12 = tt.splat %10 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked2>
    %13 = tt.addptr %12, %11 : tensor<64x!tt.ptr<f16>, #blocked2>, tensor<64xi32, #blocked2>
    %14 = arith.truncf %7#1 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %15 = ttg.convert_layout %14 : tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf16, #blocked2>
    tt.store %13, %15 : tensor<64x!tt.ptr<f16>, #blocked2>
    %16 = tt.addptr %arg21, %1 : !tt.ptr<f16>, i32
    %17 = tt.splat %16 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked2>
    %18 = tt.addptr %17, %11 : tensor<64x!tt.ptr<f16>, #blocked2>, tensor<64xi32, #blocked2>
    %19 = arith.truncf %7#0 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = ttg.convert_layout %19 : tensor<64xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf16, #blocked2>
    tt.store %18, %20 : tensor<64x!tt.ptr<f16>, #blocked2>
    tt.return
  }
}
