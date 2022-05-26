#include "bench_ff.hpp"

// register functions for benchmark
BENCHMARK(bench_ff_add);
BENCHMARK(bench_ff_sub);
BENCHMARK(bench_ff_mult);
BENCHMARK(bench_ff_pow);
BENCHMARK(bench_ff_inv);
BENCHMARK(bench_ff_div);

// main function to drive execution of benchmark
BENCHMARK_MAIN();
