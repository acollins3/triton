#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2], threadsPerWarp = [1, 1, 2, 2, 2, 2, 2, 1, 1, 1], warpsPerCTA = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1], order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0]], lane = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]], warp = [[0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0], [1], [0]], lane = [[0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear3 = #ttg.linear<{register = [[0], [0], [1]], lane = [[0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear4 = #ttg.linear<{register = [], lane = [[1], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear5 = #ttg.linear<{register = [[1]], lane = [[0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[0], [1], [0], [0], [0]], warp = [[0], [0]], block = []}>
#linear7 = #ttg.linear<{register = [[1], [0], [0]], lane = [[0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_topk_forward(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x4xi32, #linear>
    %cst_0 = arith.constant dense<1> : tensor<32x4xi32, #linear>
    %cst_1 = arith.constant dense<-1> : tensor<32x4xi16, #linear>
    %cst_2 = arith.constant dense<-32768> : tensor<32x4xi16, #linear>
    %cst_3 = arith.constant dense<0> : tensor<32x4xi32, #linear>
    %cst_4 = arith.constant dense<128> : tensor<32x4xi32, #linear>
    %cst_5 = arith.constant dense<-1> : tensor<32x32xi16, #blocked>
    %cst_6 = arith.constant dense<-32768> : tensor<32x32xi16, #blocked>
    %cst_7 = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    %cst_8 = arith.constant dense<16> : tensor<32x4xi32, #linear>
    %cst_9 = arith.constant dense<16> : tensor<32x32xi32, #blocked>
    %cst_10 = arith.constant dense<0xFC00> : tensor<32x32xf16, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %cst_11 = arith.constant dense<0> : tensor<32x4x1xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_12 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %cst_13 = arith.constant dense<1> : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %cst_14 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %cst_15 = arith.constant dense<1> : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %cst_16 = arith.constant dense<128> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_17 = arith.constant dense<-32> : tensor<32x32xi32, #blocked>
    %cst_18 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_19 = arith.constant dense<-96> : tensor<32x32xi32, #blocked>
    %cst_20 = arith.constant dense<-64> : tensor<32x32xi32, #blocked>
    %cst_21 = arith.constant dense<96> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = arith.cmpi sge, %1, %arg6 : i32
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %8 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %9 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %10 = arith.addi %7, %3 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = arith.addi %8, %4 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %12 = arith.addi %9, %5 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %13 = tt.expand_dims %10 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %14 = tt.expand_dims %11 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1xi32, #blocked3>
    %15 = tt.expand_dims %12 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %16 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked>
    %17 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked3>
    %18 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked4>
    %19 = arith.cmpi slt, %13, %16 : tensor<32x1xi32, #blocked>
    %20 = arith.cmpi slt, %14, %17 : tensor<32x1xi32, #blocked3>
    %21 = arith.cmpi slt, %15, %18 : tensor<32x1xi32, #blocked4>
    %22 = arith.addi %6, %cst_21 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %24 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked>
    %25 = arith.cmpi slt, %23, %24 : tensor<1x32xi32, #blocked>
    %26 = tt.splat %arg1 : i32 -> tensor<32x1xi32, #blocked>
    %27 = arith.muli %13, %26 : tensor<32x1xi32, #blocked>
    %28 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %29 = tt.addptr %28, %27 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %30 = tt.broadcast %29 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %31 = tt.broadcast %23 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %32 = tt.addptr %30, %31 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %33 = tt.broadcast %19 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %34 = tt.broadcast %25 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %35 = arith.andi %33, %34 : tensor<32x32xi1, #blocked>
    %36 = tt.load %32, %35, %cst_10 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %37 = tt.bitcast %36 : tensor<32x32xf16, #blocked> -> tensor<32x32xi16, #blocked>
    %38 = arith.andi %37, %cst_6 : tensor<32x32xi16, #blocked>
    %39 = arith.extui %38 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %40 = arith.cmpi ne, %39, %cst_7 : tensor<32x32xi32, #blocked>
    %41 = arith.select %40, %cst_5, %cst_6 : tensor<32x32xi1, #blocked>, tensor<32x32xi16, #blocked>
    %42 = arith.xori %37, %41 : tensor<32x32xi16, #blocked>
    %43 = arith.extui %42 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %44 = arith.shli %43, %cst_9 : tensor<32x32xi32, #blocked>
    %45 = arith.subi %cst_16, %22 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %46 = tt.expand_dims %45 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %47 = tt.broadcast %46 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %48 = arith.ori %44, %47 : tensor<32x32xi32, #blocked>
    %49 = tt.reshape %48 : tensor<32x32xi32, #blocked> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %50 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear2>
    %51 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear3>
    %52 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear4>
    %53 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear5>
    %54 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear6>
    %55 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear7>
    %56 = tt.reshape %50 : tensor<2xi32, #linear2> -> tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked2>
    %57 = tt.bitcast %49 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %58 = "tt.reduce"(%57) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %59 = tt.expand_dims %58 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %60 = tt.broadcast %59 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %61 = arith.xori %57, %60 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %62 = tt.bitcast %61 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %63 = tt.reshape %55 : tensor<2xi32, #linear7> -> tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked2>
    %64 = arith.cmpi ugt, %49, %62 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %65 = tt.broadcast %56 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked2> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked2>
    %66 = tt.broadcast %63 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked2> -> tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked2>
    %67 = arith.xori %65, %66 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked2>
    %68 = arith.extui %64 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %69 = tt.broadcast %67 : tensor<1x1x1x1x1x1x1x1x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %70 = arith.cmpi ne, %68, %69 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %71 = arith.select %70, %62, %49 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %72 = tt.reshape %51 : tensor<2xi32, #linear3> -> tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked2>
    %73 = tt.bitcast %71 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %74 = "tt.reduce"(%73) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>>
    %75 = tt.expand_dims %74 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2>
    %76 = tt.broadcast %75 : tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %77 = arith.xori %73, %76 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %78 = tt.bitcast %77 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %79 = arith.cmpi ugt, %71, %78 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %80 = tt.broadcast %72 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked2> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked2>
    %81 = tt.broadcast %56 : tensor<1x1x1x1x1x1x1x1x2x1xi32, #blocked2> -> tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked2>
    %82 = arith.xori %80, %81 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked2>
    %83 = arith.extui %79 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %84 = tt.broadcast %82 : tensor<1x1x1x1x1x1x1x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %85 = arith.cmpi ne, %83, %84 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %86 = arith.select %85, %78, %71 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %87 = tt.bitcast %86 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %88 = "tt.reduce"(%87) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %89 = tt.expand_dims %88 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %90 = tt.broadcast %89 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %91 = arith.xori %87, %90 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %92 = tt.bitcast %91 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %93 = arith.cmpi ugt, %86, %92 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %94 = tt.broadcast %72 : tensor<1x1x1x1x1x1x1x2x1x1xi32, #blocked2> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked2>
    %95 = tt.broadcast %63 : tensor<1x1x1x1x1x1x1x1x1x2xi32, #blocked2> -> tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked2>
    %96 = arith.xori %94, %95 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked2>
    %97 = arith.extui %93 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %98 = tt.broadcast %96 : tensor<1x1x1x1x1x1x1x2x1x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %99 = arith.cmpi ne, %97, %98 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %100 = arith.select %99, %92, %86 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %101 = "tt.reduce"(%100) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %102 = tt.reshape %52 : tensor<2xi32, #linear4> -> tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %103 = tt.bitcast %101 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %104 = "tt.reduce"(%103) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %105 = tt.expand_dims %104 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %106 = tt.broadcast %105 : tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %107 = arith.xori %103, %106 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %108 = tt.bitcast %107 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %109 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %110 = arith.cmpi ugt, %101, %108 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %111 = tt.broadcast %102 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %112 = tt.broadcast %109 : tensor<1x1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %113 = arith.xori %111, %112 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %114 = arith.extui %110 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %115 = tt.broadcast %113 : tensor<1x1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %116 = arith.cmpi ne, %114, %115 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %117 = arith.select %116, %108, %101 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %118 = tt.bitcast %117 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %119 = "tt.reduce"(%118) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %120 = tt.expand_dims %119 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %121 = tt.broadcast %120 : tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %122 = arith.xori %118, %121 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %123 = tt.bitcast %122 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %124 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %125 = arith.cmpi ugt, %117, %123 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %126 = tt.broadcast %102 : tensor<1x1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %127 = tt.broadcast %124 : tensor<1x1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %128 = arith.xori %126, %127 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %129 = arith.extui %125 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %130 = tt.broadcast %128 : tensor<1x1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %131 = arith.cmpi ne, %129, %130 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %132 = arith.select %131, %123, %117 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %133 = "tt.reduce"(%132) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %134 = tt.reshape %54 : tensor<2xi32, #linear6> -> tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %135 = tt.bitcast %133 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %136 = "tt.reduce"(%135) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %137 = tt.expand_dims %136 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %138 = tt.broadcast %137 : tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %139 = arith.xori %135, %138 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %140 = tt.bitcast %139 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %141 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %142 = arith.cmpi ugt, %133, %140 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %143 = tt.broadcast %134 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %144 = tt.broadcast %141 : tensor<1x1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %145 = arith.xori %143, %144 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %146 = arith.extui %142 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %147 = tt.broadcast %145 : tensor<1x1x1x1x1x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %148 = arith.cmpi ne, %146, %147 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %149 = arith.select %148, %140, %133 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %150 = tt.bitcast %149 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %151 = "tt.reduce"(%150) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %152 = tt.expand_dims %151 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %153 = tt.broadcast %152 : tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %154 = arith.xori %150, %153 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %155 = tt.bitcast %154 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %156 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %157 = arith.cmpi ugt, %149, %155 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %158 = tt.broadcast %134 : tensor<1x1x1x1x1x2x1x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %159 = tt.broadcast %156 : tensor<1x1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %160 = arith.xori %158, %159 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %161 = arith.extui %157 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %162 = tt.broadcast %160 : tensor<1x1x1x1x1x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %163 = arith.cmpi ne, %161, %162 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %164 = arith.select %163, %155, %149 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %165 = "tt.reduce"(%164) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %166 = tt.bitcast %165 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %167 = "tt.reduce"(%166) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %168 = tt.expand_dims %167 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %169 = tt.broadcast %168 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %170 = arith.xori %166, %169 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %171 = tt.bitcast %170 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %172 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #linear1>
    %173 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %174 = arith.cmpi ugt, %165, %171 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %175 = arith.xori %172, %cst_14 : tensor<1x1x1x1x1x2x1xi32, #linear1>
    %176 = arith.xori %173, %cst_15 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %177 = arith.extui %174 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %178 = tt.broadcast %175 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %179 = tt.broadcast %176 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %180 = arith.cmpi ne, %177, %179 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %181 = arith.select %180, %171, %165 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %182 = tt.bitcast %181 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %183 = "tt.reduce"(%182) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %184 = tt.expand_dims %183 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %185 = tt.broadcast %184 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %186 = arith.xori %182, %185 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %187 = tt.bitcast %186 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %188 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x2xi32, #linear1>
    %189 = tt.reshape %53 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %190 = arith.cmpi ugt, %181, %187 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %191 = arith.xori %188, %cst_12 : tensor<1x1x1x1x1x1x2xi32, #linear1>
    %192 = arith.xori %189, %cst_13 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %193 = arith.extui %190 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %194 = tt.broadcast %191 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %195 = tt.broadcast %192 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %196 = arith.cmpi ne, %193, %195 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %197 = arith.select %196, %187, %181 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %198 = tt.bitcast %197 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %199 = "tt.reduce"(%198) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %200 = tt.expand_dims %199 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %201 = tt.broadcast %200 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %202 = arith.xori %198, %201 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %203 = tt.bitcast %202 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %204 = arith.cmpi ugt, %197, %203 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %205 = arith.extui %204 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %206 = tt.broadcast %172 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %207 = tt.broadcast %173 : tensor<1x1x1x1x1x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %208 = arith.cmpi ne, %205, %207 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %209 = arith.select %208, %203, %197 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %210 = tt.bitcast %209 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %211 = "tt.reduce"(%210) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %212 = tt.expand_dims %211 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %213 = tt.broadcast %212 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %214 = arith.xori %210, %213 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %215 = tt.bitcast %214 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %216 = arith.cmpi ugt, %209, %215 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %217 = arith.extui %216 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %218 = tt.broadcast %188 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %219 = tt.broadcast %189 : tensor<1x1x1x1x1x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %220 = arith.cmpi ne, %217, %219 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %221 = arith.select %220, %215, %209 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %222 = tt.reshape %221 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<32x4xi32, #linear>
    %223 = tt.addptr %32, %cst_17 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %224 = arith.subi %22, %cst_18 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %225 = tt.load %223, %33, %cst_10 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %226 = tt.bitcast %225 : tensor<32x32xf16, #blocked> -> tensor<32x32xi16, #blocked>
    %227 = arith.andi %226, %cst_6 : tensor<32x32xi16, #blocked>
    %228 = arith.extui %227 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %229 = arith.cmpi ne, %228, %cst_7 : tensor<32x32xi32, #blocked>
    %230 = arith.select %229, %cst_5, %cst_6 : tensor<32x32xi1, #blocked>, tensor<32x32xi16, #blocked>
    %231 = arith.xori %226, %230 : tensor<32x32xi16, #blocked>
    %232 = arith.extui %231 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %233 = arith.shli %232, %cst_9 : tensor<32x32xi32, #blocked>
    %234 = arith.subi %cst_16, %224 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %235 = tt.expand_dims %234 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %236 = tt.broadcast %235 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %237 = arith.ori %233, %236 : tensor<32x32xi32, #blocked>
    %238 = tt.reshape %237 : tensor<32x32xi32, #blocked> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %239 = tt.bitcast %238 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %240 = "tt.reduce"(%239) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %241 = tt.expand_dims %240 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %242 = tt.broadcast %241 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %243 = arith.xori %239, %242 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %244 = tt.bitcast %243 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %245 = arith.cmpi ugt, %238, %244 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %246 = arith.extui %245 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %247 = arith.cmpi ne, %246, %69 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %248 = arith.select %247, %244, %238 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %249 = tt.bitcast %248 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %250 = "tt.reduce"(%249) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>>
    %251 = tt.expand_dims %250 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2>
    %252 = tt.broadcast %251 : tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %253 = arith.xori %249, %252 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %254 = tt.bitcast %253 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %255 = arith.cmpi ugt, %248, %254 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %256 = arith.extui %255 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %257 = arith.cmpi ne, %256, %84 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %258 = arith.select %257, %254, %248 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %259 = tt.bitcast %258 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %260 = "tt.reduce"(%259) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %261 = tt.expand_dims %260 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %262 = tt.broadcast %261 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %263 = arith.xori %259, %262 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %264 = tt.bitcast %263 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %265 = arith.cmpi ugt, %258, %264 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %266 = arith.extui %265 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %267 = arith.cmpi ne, %266, %98 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %268 = arith.select %267, %264, %258 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %269 = "tt.reduce"(%268) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %270 = tt.bitcast %269 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %271 = "tt.reduce"(%270) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %272 = tt.expand_dims %271 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %273 = tt.broadcast %272 : tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %274 = arith.xori %270, %273 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %275 = tt.bitcast %274 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %276 = arith.cmpi ugt, %269, %275 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %277 = arith.extui %276 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %278 = arith.cmpi ne, %277, %115 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %279 = arith.select %278, %275, %269 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %280 = tt.bitcast %279 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %281 = "tt.reduce"(%280) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %282 = tt.expand_dims %281 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %283 = tt.broadcast %282 : tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %284 = arith.xori %280, %283 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %285 = tt.bitcast %284 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %286 = arith.cmpi ugt, %279, %285 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %287 = arith.extui %286 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %288 = arith.cmpi ne, %287, %130 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %289 = arith.select %288, %285, %279 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %290 = "tt.reduce"(%289) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %291 = tt.bitcast %290 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %292 = "tt.reduce"(%291) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %293 = tt.expand_dims %292 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %294 = tt.broadcast %293 : tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %295 = arith.xori %291, %294 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %296 = tt.bitcast %295 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %297 = arith.cmpi ugt, %290, %296 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %298 = arith.extui %297 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %299 = arith.cmpi ne, %298, %147 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %300 = arith.select %299, %296, %290 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %301 = tt.bitcast %300 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %302 = "tt.reduce"(%301) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %303 = tt.expand_dims %302 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %304 = tt.broadcast %303 : tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %305 = arith.xori %301, %304 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %306 = tt.bitcast %305 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %307 = arith.cmpi ugt, %300, %306 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %308 = arith.extui %307 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %309 = arith.cmpi ne, %308, %162 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %310 = arith.select %309, %306, %300 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %311 = "tt.reduce"(%310) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %312 = tt.bitcast %311 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %313 = "tt.reduce"(%312) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %314 = tt.expand_dims %313 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %315 = tt.broadcast %314 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %316 = arith.xori %312, %315 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %317 = tt.bitcast %316 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %318 = arith.cmpi ugt, %311, %317 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %319 = arith.extui %318 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %320 = arith.cmpi ne, %319, %179 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %321 = arith.select %320, %317, %311 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %322 = tt.bitcast %321 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %323 = "tt.reduce"(%322) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %324 = tt.expand_dims %323 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %325 = tt.broadcast %324 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %326 = arith.xori %322, %325 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %327 = tt.bitcast %326 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %328 = arith.cmpi ugt, %321, %327 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %329 = arith.extui %328 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %330 = arith.cmpi ne, %329, %195 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %331 = arith.select %330, %327, %321 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %332 = tt.reshape %331 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<32x4xi32, #linear>
    %333 = arith.maxui %222, %332 : tensor<32x4xi32, #linear>
    %334 = tt.reshape %333 : tensor<32x4xi32, #linear> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %335 = tt.bitcast %334 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %336 = "tt.reduce"(%335) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>>
    %337 = tt.expand_dims %336 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>> -> tensor<2x2x2x2x2x1x2xi32, #linear1>
    %338 = tt.broadcast %337 : tensor<2x2x2x2x2x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %339 = arith.xori %335, %338 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %340 = tt.bitcast %339 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %341 = arith.cmpi ugt, %334, %340 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %342 = arith.extui %341 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %343 = arith.cmpi ne, %342, %206 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %344 = arith.select %343, %340, %334 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %345 = tt.bitcast %344 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %346 = "tt.reduce"(%345) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %347 = tt.expand_dims %346 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %348 = tt.broadcast %347 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %349 = arith.xori %345, %348 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %350 = tt.bitcast %349 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %351 = arith.cmpi ugt, %344, %350 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %352 = arith.extui %351 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %353 = arith.cmpi ne, %352, %218 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %354 = arith.select %353, %350, %344 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %355 = tt.reshape %354 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<32x4xi32, #linear>
    %356 = tt.addptr %32, %cst_20 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %357 = arith.subi %224, %cst_18 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %358 = tt.load %356, %33, %cst_10 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %359 = tt.bitcast %358 : tensor<32x32xf16, #blocked> -> tensor<32x32xi16, #blocked>
    %360 = arith.andi %359, %cst_6 : tensor<32x32xi16, #blocked>
    %361 = arith.extui %360 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %362 = arith.cmpi ne, %361, %cst_7 : tensor<32x32xi32, #blocked>
    %363 = arith.select %362, %cst_5, %cst_6 : tensor<32x32xi1, #blocked>, tensor<32x32xi16, #blocked>
    %364 = arith.xori %359, %363 : tensor<32x32xi16, #blocked>
    %365 = arith.extui %364 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %366 = arith.shli %365, %cst_9 : tensor<32x32xi32, #blocked>
    %367 = arith.subi %cst_16, %357 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %368 = tt.expand_dims %367 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %369 = tt.broadcast %368 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %370 = arith.ori %366, %369 : tensor<32x32xi32, #blocked>
    %371 = tt.reshape %370 : tensor<32x32xi32, #blocked> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %372 = tt.bitcast %371 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %373 = "tt.reduce"(%372) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %374 = tt.expand_dims %373 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %375 = tt.broadcast %374 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %376 = arith.xori %372, %375 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %377 = tt.bitcast %376 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %378 = arith.cmpi ugt, %371, %377 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %379 = arith.extui %378 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %380 = arith.cmpi ne, %379, %69 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %381 = arith.select %380, %377, %371 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %382 = tt.bitcast %381 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %383 = "tt.reduce"(%382) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>>
    %384 = tt.expand_dims %383 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2>
    %385 = tt.broadcast %384 : tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %386 = arith.xori %382, %385 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %387 = tt.bitcast %386 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %388 = arith.cmpi ugt, %381, %387 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %389 = arith.extui %388 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %390 = arith.cmpi ne, %389, %84 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %391 = arith.select %390, %387, %381 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %392 = tt.bitcast %391 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %393 = "tt.reduce"(%392) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %394 = tt.expand_dims %393 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %395 = tt.broadcast %394 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %396 = arith.xori %392, %395 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %397 = tt.bitcast %396 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %398 = arith.cmpi ugt, %391, %397 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %399 = arith.extui %398 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %400 = arith.cmpi ne, %399, %98 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %401 = arith.select %400, %397, %391 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %402 = "tt.reduce"(%401) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %403 = tt.bitcast %402 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %404 = "tt.reduce"(%403) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %405 = tt.expand_dims %404 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %406 = tt.broadcast %405 : tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %407 = arith.xori %403, %406 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %408 = tt.bitcast %407 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %409 = arith.cmpi ugt, %402, %408 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %410 = arith.extui %409 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %411 = arith.cmpi ne, %410, %115 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %412 = arith.select %411, %408, %402 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %413 = tt.bitcast %412 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %414 = "tt.reduce"(%413) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %415 = tt.expand_dims %414 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %416 = tt.broadcast %415 : tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %417 = arith.xori %413, %416 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %418 = tt.bitcast %417 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %419 = arith.cmpi ugt, %412, %418 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %420 = arith.extui %419 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %421 = arith.cmpi ne, %420, %130 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %422 = arith.select %421, %418, %412 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %423 = "tt.reduce"(%422) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %424 = tt.bitcast %423 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %425 = "tt.reduce"(%424) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %426 = tt.expand_dims %425 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %427 = tt.broadcast %426 : tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %428 = arith.xori %424, %427 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %429 = tt.bitcast %428 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %430 = arith.cmpi ugt, %423, %429 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %431 = arith.extui %430 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %432 = arith.cmpi ne, %431, %147 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %433 = arith.select %432, %429, %423 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %434 = tt.bitcast %433 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %435 = "tt.reduce"(%434) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %436 = tt.expand_dims %435 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %437 = tt.broadcast %436 : tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %438 = arith.xori %434, %437 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %439 = tt.bitcast %438 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %440 = arith.cmpi ugt, %433, %439 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %441 = arith.extui %440 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %442 = arith.cmpi ne, %441, %162 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %443 = arith.select %442, %439, %433 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %444 = "tt.reduce"(%443) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %445 = tt.bitcast %444 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %446 = "tt.reduce"(%445) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %447 = tt.expand_dims %446 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %448 = tt.broadcast %447 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %449 = arith.xori %445, %448 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %450 = tt.bitcast %449 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %451 = arith.cmpi ugt, %444, %450 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %452 = arith.extui %451 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %453 = arith.cmpi ne, %452, %179 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %454 = arith.select %453, %450, %444 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %455 = tt.bitcast %454 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %456 = "tt.reduce"(%455) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %457 = tt.expand_dims %456 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %458 = tt.broadcast %457 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %459 = arith.xori %455, %458 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %460 = tt.bitcast %459 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %461 = arith.cmpi ugt, %454, %460 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %462 = arith.extui %461 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %463 = arith.cmpi ne, %462, %195 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %464 = arith.select %463, %460, %454 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %465 = tt.reshape %464 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<32x4xi32, #linear>
    %466 = arith.maxui %355, %465 : tensor<32x4xi32, #linear>
    %467 = tt.reshape %466 : tensor<32x4xi32, #linear> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %468 = tt.bitcast %467 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %469 = "tt.reduce"(%468) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>>
    %470 = tt.expand_dims %469 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>> -> tensor<2x2x2x2x2x1x2xi32, #linear1>
    %471 = tt.broadcast %470 : tensor<2x2x2x2x2x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %472 = arith.xori %468, %471 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %473 = tt.bitcast %472 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %474 = arith.cmpi ugt, %467, %473 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %475 = arith.extui %474 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %476 = arith.cmpi ne, %475, %206 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %477 = arith.select %476, %473, %467 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %478 = tt.bitcast %477 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %479 = "tt.reduce"(%478) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %480 = tt.expand_dims %479 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %481 = tt.broadcast %480 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %482 = arith.xori %478, %481 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %483 = tt.bitcast %482 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %484 = arith.cmpi ugt, %477, %483 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %485 = arith.extui %484 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %486 = arith.cmpi ne, %485, %218 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %487 = arith.select %486, %483, %477 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %488 = tt.reshape %487 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<32x4xi32, #linear>
    %489 = tt.addptr %32, %cst_19 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %490 = arith.subi %357, %cst_18 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %491 = tt.load %489, %33, %cst_10 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %492 = tt.bitcast %491 : tensor<32x32xf16, #blocked> -> tensor<32x32xi16, #blocked>
    %493 = arith.andi %492, %cst_6 : tensor<32x32xi16, #blocked>
    %494 = arith.extui %493 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %495 = arith.cmpi ne, %494, %cst_7 : tensor<32x32xi32, #blocked>
    %496 = arith.select %495, %cst_5, %cst_6 : tensor<32x32xi1, #blocked>, tensor<32x32xi16, #blocked>
    %497 = arith.xori %492, %496 : tensor<32x32xi16, #blocked>
    %498 = arith.extui %497 : tensor<32x32xi16, #blocked> to tensor<32x32xi32, #blocked>
    %499 = arith.shli %498, %cst_9 : tensor<32x32xi32, #blocked>
    %500 = arith.subi %cst_16, %490 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %501 = tt.expand_dims %500 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %502 = tt.broadcast %501 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %503 = arith.ori %499, %502 : tensor<32x32xi32, #blocked>
    %504 = tt.reshape %503 : tensor<32x32xi32, #blocked> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %505 = tt.bitcast %504 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %506 = "tt.reduce"(%505) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %507 = tt.expand_dims %506 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %508 = tt.broadcast %507 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %509 = arith.xori %505, %508 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %510 = tt.bitcast %509 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %511 = arith.cmpi ugt, %504, %510 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %512 = arith.extui %511 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %513 = arith.cmpi ne, %512, %69 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %514 = arith.select %513, %510, %504 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %515 = tt.bitcast %514 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %516 = "tt.reduce"(%515) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>>
    %517 = tt.expand_dims %516 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2>
    %518 = tt.broadcast %517 : tensor<2x2x2x2x2x2x2x2x1x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %519 = arith.xori %515, %518 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %520 = tt.bitcast %519 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %521 = arith.cmpi ugt, %514, %520 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %522 = arith.extui %521 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %523 = arith.cmpi ne, %522, %84 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %524 = arith.select %523, %520, %514 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %525 = tt.bitcast %524 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %526 = "tt.reduce"(%525) <{axis = 9 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>>
    %527 = tt.expand_dims %526 {axis = 9 : i32} : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 9, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2>
    %528 = tt.broadcast %527 : tensor<2x2x2x2x2x2x2x2x2x1xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %529 = arith.xori %525, %528 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %530 = tt.bitcast %529 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2> -> tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %531 = arith.cmpi ugt, %524, %530 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %532 = arith.extui %531 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2> to tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %533 = arith.cmpi ne, %532, %98 : tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %534 = arith.select %533, %530, %524 : tensor<2x2x2x2x2x2x2x2x2x2xi1, #blocked2>, tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>
    %535 = "tt.reduce"(%534) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2x2xi32, #blocked2>) -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %536 = tt.bitcast %535 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %537 = "tt.reduce"(%536) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %538 = tt.expand_dims %537 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %539 = tt.broadcast %538 : tensor<2x2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %540 = arith.xori %536, %539 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %541 = tt.bitcast %540 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %542 = arith.cmpi ugt, %535, %541 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %543 = arith.extui %542 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %544 = arith.cmpi ne, %543, %115 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %545 = arith.select %544, %541, %535 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %546 = tt.bitcast %545 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %547 = "tt.reduce"(%546) <{axis = 8 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %548 = tt.expand_dims %547 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 8, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %549 = tt.broadcast %548 : tensor<2x2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %550 = arith.xori %546, %549 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %551 = tt.bitcast %550 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>> -> tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %552 = arith.cmpi ugt, %545, %551 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %553 = arith.extui %552 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>> to tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %554 = arith.cmpi ne, %553, %130 : tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %555 = arith.select %554, %551, %545 : tensor<2x2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 7, parent = #blocked2}>>, tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>
    %556 = "tt.reduce"(%555) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #blocked2}>>) -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %557 = tt.bitcast %556 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %558 = "tt.reduce"(%557) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %559 = tt.expand_dims %558 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %560 = tt.broadcast %559 : tensor<2x2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %561 = arith.xori %557, %560 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %562 = tt.bitcast %561 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %563 = arith.cmpi ugt, %556, %562 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %564 = arith.extui %563 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %565 = arith.cmpi ne, %564, %147 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %566 = arith.select %565, %562, %556 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %567 = tt.bitcast %566 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %568 = "tt.reduce"(%567) <{axis = 7 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %569 = tt.expand_dims %568 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 7, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %570 = tt.broadcast %569 : tensor<2x2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %571 = arith.xori %567, %570 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %572 = tt.bitcast %571 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> -> tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %573 = arith.cmpi ugt, %566, %572 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %574 = arith.extui %573 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>> to tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %575 = arith.cmpi ne, %574, %162 : tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %576 = arith.select %575, %572, %566 : tensor<2x2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>, tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>
    %577 = "tt.reduce"(%576) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.maxui %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>>) -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %578 = tt.bitcast %577 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %579 = "tt.reduce"(%578) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %580 = tt.expand_dims %579 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %581 = tt.broadcast %580 : tensor<2x2x2x2x2x1x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %582 = arith.xori %578, %581 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %583 = tt.bitcast %582 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %584 = arith.cmpi ugt, %577, %583 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %585 = arith.extui %584 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %586 = arith.cmpi ne, %585, %179 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %587 = arith.select %586, %583, %577 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %588 = tt.bitcast %587 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %589 = "tt.reduce"(%588) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>>
    %590 = tt.expand_dims %589 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>}>> -> tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %591 = tt.broadcast %590 : tensor<2x2x2x2x2x2x1xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %592 = arith.xori %588, %591 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %593 = tt.bitcast %592 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %594 = arith.cmpi ugt, %587, %593 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %595 = arith.extui %594 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> to tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %596 = arith.cmpi ne, %595, %195 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %597 = arith.select %596, %593, %587 : tensor<2x2x2x2x2x2x2xi1, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>, tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>>
    %598 = tt.reshape %597 : tensor<2x2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 6, parent = #ttg.slice<{dim = 7, parent = #blocked2}>}>}>> -> tensor<32x4xi32, #linear>
    %599 = arith.maxui %488, %598 : tensor<32x4xi32, #linear>
    %600 = arith.shli %599, %cst_8 : tensor<32x4xi32, #linear>
    %601 = arith.shrui %599, %cst_8 : tensor<32x4xi32, #linear>
    %602 = arith.ori %600, %601 : tensor<32x4xi32, #linear>
    %603 = tt.reshape %602 : tensor<32x4xi32, #linear> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %604 = tt.bitcast %603 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %605 = "tt.reduce"(%604) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %606 = tt.expand_dims %605 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %607 = tt.broadcast %606 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %608 = arith.xori %604, %607 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %609 = tt.bitcast %608 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %610 = arith.cmpi ugt, %603, %609 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %611 = tt.broadcast %172 : tensor<1x1x1x1x1x2x1xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %612 = tt.broadcast %188 : tensor<1x1x1x1x1x1x2xi32, #linear1> -> tensor<1x1x1x1x1x2x2xi32, #linear1>
    %613 = arith.xori %611, %612 : tensor<1x1x1x1x1x2x2xi32, #linear1>
    %614 = arith.extui %610 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %615 = tt.broadcast %613 : tensor<1x1x1x1x1x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %616 = arith.cmpi ne, %614, %615 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %617 = arith.select %616, %609, %603 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %618 = tt.bitcast %617 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %619 = "tt.reduce"(%618) <{axis = 5 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>>
    %620 = tt.expand_dims %619 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #linear1}>> -> tensor<2x2x2x2x2x1x2xi32, #linear1>
    %621 = tt.broadcast %620 : tensor<2x2x2x2x2x1x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %622 = arith.xori %618, %621 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %623 = tt.bitcast %622 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %624 = arith.cmpi ugt, %617, %623 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %625 = arith.extui %624 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %626 = arith.cmpi ne, %625, %178 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %627 = arith.select %626, %623, %617 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %628 = tt.bitcast %627 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %629 = "tt.reduce"(%628) <{axis = 6 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %685 = arith.xori %arg8, %arg9 : i32
      tt.reduce.return %685 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #linear1>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>>
    %630 = tt.expand_dims %629 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #linear1}>> -> tensor<2x2x2x2x2x2x1xi32, #linear1>
    %631 = tt.broadcast %630 : tensor<2x2x2x2x2x2x1xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %632 = arith.xori %628, %631 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %633 = tt.bitcast %632 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<2x2x2x2x2x2x2xi32, #linear1>
    %634 = arith.cmpi ugt, %627, %633 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %635 = arith.extui %634 : tensor<2x2x2x2x2x2x2xi1, #linear1> to tensor<2x2x2x2x2x2x2xi32, #linear1>
    %636 = arith.cmpi ne, %635, %194 : tensor<2x2x2x2x2x2x2xi32, #linear1>
    %637 = arith.select %636, %633, %627 : tensor<2x2x2x2x2x2x2xi1, #linear1>, tensor<2x2x2x2x2x2x2xi32, #linear1>
    %638 = tt.reshape %637 : tensor<2x2x2x2x2x2x2xi32, #linear1> -> tensor<32x4xi32, #linear>
    %639 = arith.shrui %638, %cst_8 : tensor<32x4xi32, #linear>
    %640 = arith.subi %cst_4, %639 : tensor<32x4xi32, #linear>
    %641 = arith.trunci %638 : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %642 = arith.andi %641, %cst_2 : tensor<32x4xi16, #linear>
    %643 = arith.extui %642 : tensor<32x4xi16, #linear> to tensor<32x4xi32, #linear>
    %644 = arith.cmpi eq, %643, %cst_3 : tensor<32x4xi32, #linear>
    %645 = arith.select %644, %cst_1, %cst_2 : tensor<32x4xi1, #linear>, tensor<32x4xi16, #linear>
    %646 = arith.xori %641, %645 : tensor<32x4xi16, #linear>
    %647 = tt.bitcast %646 : tensor<32x4xi16, #linear> -> tensor<32x4xf16, #linear>
    %648 = arith.extf %647 : tensor<32x4xf16, #linear> to tensor<32x4xf32, #linear>
    %649 = "tt.reduce"(%648) <{axis = 1 : i32}> ({
    ^bb0(%arg8: f32, %arg9: f32):
      %685 = arith.maxnumf %arg8, %arg9 : f32
      tt.reduce.return %685 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %650 = tt.expand_dims %649 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %651 = tt.broadcast %650 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %652 = arith.subf %648, %651 : tensor<32x4xf32, #linear>
    %653 = math.exp %652 : tensor<32x4xf32, #linear>
    %654 = "tt.reduce"(%653) <{axis = 1 : i32}> ({
    ^bb0(%arg8: f32, %arg9: f32):
      %685 = arith.addf %arg8, %arg9 : f32
      tt.reduce.return %685 : f32
    }) : (tensor<32x4xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %655 = tt.expand_dims %654 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xf32, #linear>
    %656 = tt.broadcast %655 : tensor<32x1xf32, #linear> -> tensor<32x4xf32, #linear>
    %657 = arith.divf %653, %656 : tensor<32x4xf32, #linear>
    %658 = arith.truncf %657 : tensor<32x4xf32, #linear> to tensor<32x4xf16, #linear>
    %659 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked3>
    %660 = arith.muli %14, %659 : tensor<32x1xi32, #blocked3>
    %661 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked3>
    %662 = tt.addptr %661, %660 : tensor<32x1x!tt.ptr<f16>, #blocked3>, tensor<32x1xi32, #blocked3>
    %663 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %664 = tt.expand_dims %663 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x4xi32, #blocked3>
    %665 = tt.broadcast %662 : tensor<32x1x!tt.ptr<f16>, #blocked3> -> tensor<32x4x!tt.ptr<f16>, #blocked3>
    %666 = tt.broadcast %664 : tensor<1x4xi32, #blocked3> -> tensor<32x4xi32, #blocked3>
    %667 = tt.addptr %665, %666 : tensor<32x4x!tt.ptr<f16>, #blocked3>, tensor<32x4xi32, #blocked3>
    %668 = tt.broadcast %20 : tensor<32x1xi1, #blocked3> -> tensor<32x4xi1, #blocked3>
    %669 = ttg.convert_layout %658 : tensor<32x4xf16, #linear> -> tensor<32x4xf16, #blocked3>
    tt.store %667, %669, %668 : tensor<32x4x!tt.ptr<f16>, #blocked3>
    %670 = tt.splat %arg3 : !tt.ptr<i16> -> tensor<32x1x!tt.ptr<i16>, #blocked3>
    %671 = tt.addptr %670, %660 : tensor<32x1x!tt.ptr<i16>, #blocked3>, tensor<32x1xi32, #blocked3>
    %672 = tt.broadcast %671 : tensor<32x1x!tt.ptr<i16>, #blocked3> -> tensor<32x4x!tt.ptr<i16>, #blocked3>
    %673 = tt.addptr %672, %666 : tensor<32x4x!tt.ptr<i16>, #blocked3>, tensor<32x4xi32, #blocked3>
    %674 = arith.trunci %640 : tensor<32x4xi32, #linear> to tensor<32x4xi16, #linear>
    %675 = ttg.convert_layout %674 : tensor<32x4xi16, #linear> -> tensor<32x4xi16, #blocked3>
    tt.store %673, %675, %668 : tensor<32x4x!tt.ptr<i16>, #blocked3>
    %676 = arith.divui %640, %cst : tensor<32x4xi32, #linear>
    %677 = arith.remui %640, %cst : tensor<32x4xi32, #linear>
    %678 = ttg.convert_layout %676 : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %679 = tt.expand_dims %678 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %680 = arith.shli %cst_0, %677 : tensor<32x4xi32, #linear>
    %681 = ttg.convert_layout %680 : tensor<32x4xi32, #linear> -> tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %682 = tt.expand_dims %681 {axis = 2 : i32} : tensor<32x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<32x4x1xi32, #blocked1>
    %683 = tt.splat %arg5 : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>, #blocked4>
    %684 = tt.addptr %683, %15 : tensor<32x1x!tt.ptr<i32>, #blocked4>, tensor<32x1xi32, #blocked4>
    scf.for %arg8 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %685 = tt.splat %arg8 : i32 -> tensor<32x4x1xi32, #blocked1>
      %686 = arith.cmpi eq, %679, %685 : tensor<32x4x1xi32, #blocked1>
      %687 = arith.select %686, %682, %cst_11 : tensor<32x4x1xi1, #blocked1>, tensor<32x4x1xi32, #blocked1>
      %688 = "tt.reduce"(%687) <{axis = 1 : i32}> ({
      ^bb0(%arg9: i32, %arg10: i32):
        %693 = arith.ori %arg9, %arg10 : i32
        tt.reduce.return %693 : i32
      }) : (tensor<32x4x1xi32, #blocked1>) -> tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %689 = arith.muli %arg8, %c32_i32 : i32
      %690 = tt.splat %689 : i32 -> tensor<32x1xi32, #blocked4>
      %691 = tt.addptr %684, %690 : tensor<32x1x!tt.ptr<i32>, #blocked4>, tensor<32x1xi32, #blocked4>
      %692 = ttg.convert_layout %688 : tensor<32x1xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked4>
      tt.store %691, %692, %21 : tensor<32x1x!tt.ptr<i32>, #blocked4>
    }
    tt.return
  }
}
