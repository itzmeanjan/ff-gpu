#pragma once
#include "test.hpp"
#include <iostream>
#include <rescue_prime_vectorized.hpp>

inline constexpr uint64_t ALPHA = 7ull;
inline constexpr uint64_t INV_ALPHA = 10540996611094048183ull;

void
test_alphas(sycl::queue& q);

void
random_hash_state(sycl::ulong16* state, const sycl::ulong n);

void
test_sbox(sycl::queue& q);

void
test_inv_sbox(sycl::queue& q);

void
test_permutation(sycl::queue& q);
