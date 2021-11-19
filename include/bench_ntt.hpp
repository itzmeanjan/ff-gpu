#pragma once
#include "ntt.hpp"
#include "test_ntt.hpp"
#include <chrono>

typedef std::chrono::_V2::steady_clock::time_point tp;

int64_t benchmark_forward_transform(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size);

int64_t benchmark_inverse_transform(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size);

int64_t benchmark_cooley_tukey_fft(sycl::queue &q, const uint64_t dim,
                                   const uint64_t wg_size);

int64_t benchmark_cooley_tukey_ifft(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size);

int64_t benchmark_matrix_transposition(sycl::queue &q, const uint64_t dim,
                                       const uint64_t wg_size);

int64_t benchmark_twiddle_factor_multiplication(sycl::queue &q,
                                                const uint64_t n1,
                                                const uint64_t n2,
                                                const uint64_t wg_size);
