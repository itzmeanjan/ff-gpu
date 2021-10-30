#pragma once
#include <CL/sycl.hpp>

// Benchmark binary extension field addition performance
void benchmark_ff_addition(sycl::queue &q, const uint32_t dim,
                           const uint32_t wg_size, const uint32_t itr_count);

// Benchmark binary extension field subtraction performance
void benchmark_ff_subtraction(sycl::queue &q, const uint32_t dim,
                              const uint32_t wg_size, const uint32_t itr_count);

// Benchmark binary extension field multiplication performance
void benchmark_ff_multiplication(sycl::queue &q, const uint32_t dim,
                                 const uint32_t wg_size,
                                 const uint32_t itr_count);

// Benchmark binary extension field division performance
void benchmark_ff_division(sycl::queue &q, const uint32_t dim,
                           const uint32_t wg_size, const uint32_t itr_count);

// Benchmark binary extension field inversion ( multiplicative inverse op )
// performance
void benchmark_ff_inversion(sycl::queue &q, const uint32_t dim,
                            const uint32_t wg_size, const uint32_t itr_count);
