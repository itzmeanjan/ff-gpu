#pragma once
#include "ntt.hpp"
#include <random>

sycl::event compute_matrix_matrix_multiplication(
    sycl::queue &q, buf_2d_u64_t &mat_a, buf_2d_u64_t &mat_b,
    buf_2d_u64_t &mat_c, const uint64_t dim, const uint64_t wg_size);

void check_ntt_correctness(sycl::queue &q, const uint64_t dim,
                           const uint64_t wg_size);

void prepare_random_vector(uint64_t *const vec, const uint64_t size);

void check_ntt_forward_inverse_transform(sycl::queue &q, const uint64_t dim,
                                         const uint64_t wg_size);

void check_cooley_tukey_ntt(sycl::queue &q, const uint64_t dim,
                            const uint64_t wg_size);

// Asserts parallel in-place square matrix transposition
// by performing double transpose of same matrix & finally
// parallelly asserting cells, using atomics
void check_matrix_transposition(sycl::queue &q, const uint64_t dim,
                                const uint64_t wg_size);

void test_twiddle_factor_multiplication(sycl::queue &q, const uint64_t n1,
                                        const uint64_t n2,
                                        const uint64_t wg_size);

void test_six_step_fft(sycl::queue &q, const uint64_t dim,
                       const uint64_t wg_size);

void test_six_step_ifft(sycl::queue &q, const uint64_t dim,
                        const uint64_t wg_size);
