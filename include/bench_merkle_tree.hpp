#pragma once
#include <CL/sycl.hpp>

// Returns exact time of kernel execution in nanosecond
//
// Ensure SYCL queue has profiling enabled !
uint64_t
benchmark_merklize_approach_1(sycl::queue& q,
                              const size_t leaf_count,
                              const size_t wg_size);

// Returns sum of all kernel execution times in nanosecond
//
// Ensure SYCL queue has profiling enabled !
uint64_t
benchmark_merklize_approach_2(sycl::queue& q,
                              const size_t leaf_count,
                              const size_t wg_size);
