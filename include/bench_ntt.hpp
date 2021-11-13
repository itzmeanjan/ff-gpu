#pragma once
#include "ntt.hpp"
#include "test_ntt.hpp"
#include <chrono>

typedef std::chrono::_V2::steady_clock::time_point tp;

int64_t benchmark_forward_transform(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size);

int64_t benchmark_inverse_transform(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size);
