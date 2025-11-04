#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_tma_persistent_ws_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg6, %arg8], [%0, %c1_i64] : <f16>, <tensor<128x64xf16, #shared>>
    %2 = arith.extsi %arg4 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg7, %arg8], [%2, %c1_i64] : <f16>, <tensor<128x64xf16, #shared>>
    %4 = arith.extsi %arg5 : i32 to i64
    %5 = tt.make_tensor_descriptor %arg2, [%arg6, %arg7], [%4, %c1_i64] : <f16>, <tensor<128x128xf16, #shared>>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg6, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg7, %c127_i32 : i32
    %10 = arith.divsi %9, %c128_i32 : i32
    %11 = arith.addi %arg8, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    scf.for %arg9 = %6 to %13 step %c148_i32  : i32 {
      %15 = arith.divsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %16 = arith.muli %15, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %17 = arith.subi %8, %16 {ttg.partition = array<i32: 0, 2>} : i32
      %18 = arith.minsi %17, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %19 = arith.remsi %arg9, %18 {ttg.partition = array<i32: 0, 2>} : i32
      %20 = arith.addi %16, %19 {ttg.partition = array<i32: 0, 2>} : i32
      %21 = arith.remsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %22 = arith.divsi %21, %18 {ttg.partition = array<i32: 0, 2>} : i32
      %23 = arith.muli %20, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %24 = arith.muli %22, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %result, %token = ttng.tmem_alloc {ttg.partition = array<i32: 0, 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %25 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %26:2 = scf.for %arg10 = %c0_i32 to %12 step %c1_i32 iter_args(%arg11 = %false, %arg12 = %25) -> (i1, !ttg.async.token)  : i32 {
        %29 = arith.muli %arg10, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %30 = tt.descriptor_load %1[%23, %29] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %31 = ttg.local_alloc %30 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %32 = tt.descriptor_load %3[%24, %29] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %33 = ttg.local_alloc %32 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %34 = ttg.memdesc_trans %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %35 = ttng.tc_gen5_mma %31, %34, %result[%arg12], %arg11, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %35 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %result_0, %token_1 = ttng.tmem_load %result[%26#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %27 = arith.truncf %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %28 = ttg.convert_layout %27 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %5[%23, %24], %28 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
