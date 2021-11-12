#pragma once
#include "ntt.hpp"

sycl::event compute_matrix_matrix_multiplication(
    sycl::queue &q, buf_2d_u64_t &mat_a, buf_2d_u64_t &mat_b,
    buf_2d_u64_t &mat_c, const uint64_t dim, const uint64_t wg_size);
