#pragma once
#include <CL/sycl.hpp>

// Kernel execution time returned in nanoseconds
uint64_t
benchmark_hash_elements(sycl::queue& q,
                        const uint32_t dim,
                        const uint32_t wg_size,
                        const uint32_t itr_count);

// Kernel execution time returned in nanosecond granularity
uint64_t
benchmark_merge(sycl::queue& q,
                const uint32_t dim,
                const uint32_t wg_size,
                const uint32_t itr_count);
