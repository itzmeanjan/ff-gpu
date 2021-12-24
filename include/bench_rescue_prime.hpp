#pragma once
#include <CL/sycl.hpp>

int64_t
benchmark_hash_elements(sycl::queue& q,
                        const uint32_t dim,
                        const uint32_t wg_size,
                        const uint32_t itr_count);

int64_t
benchmark_merge(sycl::queue& q,
                const uint32_t dim,
                const uint32_t wg_size,
                const uint32_t itr_count);
