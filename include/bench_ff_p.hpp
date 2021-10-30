#pragma once
#include <CL/sycl.hpp>

// computes a hilbert matrix of dimension (dim x dim)
// where each element is a prime field element,
// which are computed using (1 / (i + j + 1)) while following field
// rules, given i = row index, j = column index
//
// used for benchmarking prime field arithmetics
void gen_hilbert_matrix_ff_p(sycl::queue &q, uint32_t *const mat,
                             const uint dim, const uint wg_size);

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
                              const uint32_t wg_size, const uint32_t itr_count);

// Benchmark exponentiation on prime field F(2**64 - 2**32 + 1) elements
void benchmark_ff_p_exponentiation(sycl::queue &q, const uint32_t dim,
                                   const uint32_t wg_size,
                                   const uint32_t itr_count);
