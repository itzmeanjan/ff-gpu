#pragma once
#include <CL/sycl.hpp>

// computes a hilbert matrix of dimension (dim x dim)
// where each element is a finite field (binary extension) element,
// which are computed using (1 / (i + j + 1)) while following field
// rules, given i = row index, j = column index
//
// used for benchmarking binary field arithmetics
void gen_hilbert_matrix_ff(sycl::queue &q, uint32_t *const mat, const uint dim,
                           const uint wg_size);

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

// Benchmark binary extension field exponentiation ( raising element to some
// power )
void benchmark_ff_exponentiation(sycl::queue &q, const uint32_t dim,
                                 const uint32_t wg_size,
                                 const uint32_t itr_count);
