#pragma once
#include <CL/sycl.hpp>
#include <limits>

inline constexpr uint64_t FIELD_MOD = 18446744069414584321ull;
inline constexpr uint64_t STATE_WIDTH = 12;
inline constexpr uint64_t RATE_WIDTH = 8;
inline constexpr uint64_t DIGEST_SIZE = 4;
inline constexpr uint64_t NUM_ROUNDS = 7;
inline constexpr uint64_t MAX_UINT = 0xffffffffull;

/*
  Note : Actually I wanted to use `marray` instead of `vec`, but seems that
  `sycl::mul_hi` is not yet able to take `marray` as input in SYCL/ DPCPP

  I will come and take a look later !
*/

// Performs element wise modular multiplication on two operand vectors
//
// Returned vector may not have all elements in canonical representation
// so consider running `res % FIELD_MOD` before consumption !
//
// Takes quite some motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L9-L36
SYCL_EXTERNAL sycl::ulong16 ff_p_vec_mul(sycl::ulong16 a, sycl::ulong16 b);

// Performs element wise modular addition ( on aforementioned 64-bit prime field
// ) on two supplied operands
//
// Before consumption consider performing `res % FIELD_MOD` so that all
// lanes are in canonical form
//
// Collects quite some motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L38-L66
SYCL_EXTERNAL sycl::ulong16 ff_p_vec_add(sycl::ulong16 a, sycl::ulong16 b);
