#pragma once
#include "rescue_prime.hpp"
#include "test.hpp"
#include <iostream>

inline constexpr uint64_t ALPHA = 7ull;
inline constexpr uint64_t INV_ALPHA = 10540996611094048183ull;

void test_alphas(sycl::queue &q);

void test_sbox(sycl::queue &q);

void random_rescue_prime_state(rescue_prime_state_t *arr);

void test_inv_sbox(sycl::queue &q);

void test_permutation(sycl::queue &q);
