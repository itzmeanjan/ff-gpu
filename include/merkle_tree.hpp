#pragma once
#include <cassert>
#include <rescue_prime.hpp>

// Given N -many leaves of Binary Merkle Tree computes all (N - 1) -many
// intermediate nodes by using Rescue Prime `merge` function, which
// merges two Rescue Prime digests into single of width 256 -bit
//
// N needs to be power of two
//
// Returns sum of all kernel execution times with nanosecond
// level granularity
uint64_t
merklize(sycl::queue& q,
         const sycl::ulong* leaves,
         sycl::ulong* const intermediates,
         const size_t leaf_count,
         const size_t wg_size,
         const sycl::ulong16* mds,
         const sycl::ulong16* ark1,
         const sycl::ulong16* ark2);
