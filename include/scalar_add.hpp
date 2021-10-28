#pragma once
#include <CL/sycl.hpp>

// Adds a subset of F_2_q ( q = 32 ) elements
// and stores result in provided vector
//
// I use it for benchmarking finite field addition performance
void add_elements(sycl::queue &q, uint32_t *const vec, const uint32_t dim,
                  const uint32_t wg_size, const uint32_t itr_count);
