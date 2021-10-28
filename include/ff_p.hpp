#pragma once
#include <CL/sycl.hpp>

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
