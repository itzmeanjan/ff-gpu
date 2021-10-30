#pragma once
#include <CL/sycl.hpp>

// Benchmark addition on prime field F(2**64 - 2**32 + 1) elements
void benchmark_ff_p_addition(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count);
