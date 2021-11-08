#pragma once
#include <CL/sycl.hpp>

void benchmark_hash_elements(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count);
