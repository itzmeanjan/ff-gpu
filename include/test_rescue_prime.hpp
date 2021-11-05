#pragma once
#include "rescue_prime.hpp"
#include "test.hpp"

inline constexpr uint64_t ALPHA = 7ull;
inline constexpr uint64_t INV_ALPHA = 10540996611094048183ull;

void test_alphas(sycl::queue &q);
