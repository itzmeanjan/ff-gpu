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
merklize_approach_1(sycl::queue& q,
                    const sycl::ulong* leaves,
                    sycl::ulong* const intermediates,
                    const size_t leaf_count,
                    const size_t wg_size,
                    const sycl::ulong16* mds,
                    const sycl::ulong16* ark1,
                    const sycl::ulong16* ark2);

// Same as above routine, serves similar purpose, when
// N -many leaves of binary merkle tree are provided, computes
// all N-1 -many intermediates, while using Rescue Prime merge function
// for merging two digests, which is equivalent to computing parent node
// from two children
//
// N -needs to be power of 2
//
// Check those assertions, I've written in implementation of this routine
// which are my assumptions, feel free to tinker with them !
//
// Returns total time spent on computing merkle tree intermediate nodes
// with nanosecond level granularity
//
// Have taken major motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/c48b8555e07eb9557a20383cc9f3a4aeec834317/rescue_prime.c#L153-L164
// where I wrote similar routine using OpenCL
uint64_t
merklize_approach_2(sycl::queue& q,
                    const sycl::ulong* leaves,
                    sycl::ulong* const intermediates,
                    const size_t leaf_count,
                    const size_t wg_size,
                    const sycl::ulong16* mds,
                    const sycl::ulong16* ark1,
                    const sycl::ulong16* ark2);
