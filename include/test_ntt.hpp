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
