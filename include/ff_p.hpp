#pragma once
#include <CL/sycl.hpp>

// Prime modulas of selected field: 2 ** 64 - 2 ** 32 + 1
inline constexpr uint64_t MOD =
    ((((uint64_t)1 << 63) - ((uint64_t)1 << 31)) << 1) + 1;

// modular addition of two prime field elements
uint64_t ff_p_add(const uint64_t a, const uint64_t b);
