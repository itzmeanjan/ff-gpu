#pragma once
#include <CL/sycl.hpp>

// Benchmark addition on prime field F(2**64 - 2**32 + 1) elements
void benchmark_ff_p_addition(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count);

// Benchmark subtraction on prime field F(2**64 - 2**32 + 1) elements
void benchmark_ff_p_subtraction(sycl::queue &q, const uint32_t dim,
                                const uint32_t wg_size,
                                const uint32_t itr_count);

// Benchmark multiplication on prime field F(2**64 - 2**32 + 1) elements
void benchmark_ff_p_multiplication(sycl::queue &q, const uint32_t dim,
                                   const uint32_t wg_size,
                                   const uint32_t itr_count);

// Benchmark division on prime field F(2**64 - 2**32 + 1) elements
void benchmark_ff_p_division(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count);

// Benchmark inversion on prime field F(2**64 - 2**32 + 1) elements
void benchmark_ff_p_inversion(sycl::queue &q, const uint32_t dim,
                                   const uint32_t wg_size,
                                   const uint32_t itr_count);
