#pragma once
#include <CL/sycl.hpp>

// I use it for benchmarking (binary extension) finite field addition
// performance
void benchmark_ff_addition(sycl::queue &q, const uint32_t dim,
                           const uint32_t wg_size, const uint32_t itr_count);
