#pragma once
#include <CL/sycl.hpp>

// custom data type for dealing with two
// 64-bit field element multiplication
// resulting into 128-bit integer
typedef sycl::cl_ulong2 uint128_t;

// Prime modulas of selected field, F_p,
// where p = 2 ** 64 - 2 ** 32 + 1
inline constexpr uint64_t MOD =
    ((((uint64_t)1 << 63) - ((uint64_t)1 << 31)) << 1) + 1;

// modular addition of two prime field elements
//
// note: if a, b ∉ F_p, it will be converted by performing
// {a, b} % MOD
uint64_t ff_p_add(uint64_t a, uint64_t b);

// modular subtraction of two prime field elements
//
// note: if a, b ∉ F_p, it will be converted by performing
// {a, b} % MOD
uint64_t ff_p_sub(uint64_t a, uint64_t b);

// modular mulitiplication of two prime field elements
//
// note: if a, b ∉ F_p, it will be converted by performing
// {a, b} % MOD
//
// multiplication results into 128-bit integer, which
// is reduced to field element such that c ∈ [0, p)
// where p = field prime modulas
extern SYCL_EXTERNAL uint64_t ff_p_mul(uint64_t a, uint64_t b);
