#pragma once
#include <CL/sycl.hpp>

// custom data type for dealing with two
// 64-bit field element multiplication
// resulting into 128-bit integer
typedef sycl::ulonglong2 uint128_t;

// Prime modulas of selected field, F_p,
// where p = 2 ** 64 - 2 ** 32 + 1
inline constexpr uint64_t MOD =
    ((((uint64_t)1 << 63) - ((uint64_t)1 << 31)) << 1) + 1;

// modular addition of two prime field elements
//
// note: if a, b ∉ F_p, it will be converted by performing
// {a, b} % MOD
extern SYCL_EXTERNAL uint64_t ff_p_add(uint64_t a, uint64_t b);

// modular subtraction of two prime field elements
//
// note: if a, b ∉ F_p, it will be converted by performing
// {a, b} % MOD
extern SYCL_EXTERNAL uint64_t ff_p_sub(uint64_t a, uint64_t b);

// modular mulitiplication of two prime field elements
//
// note: if a, b ∉ F_p, it will be converted by performing
// {a, b} % MOD
//
// multiplication results into 128-bit integer, which
// is reduced to field element such that c ∈ [0, p)
// where p = field prime modulas
extern SYCL_EXTERNAL uint64_t ff_p_mult(uint64_t a, uint64_t b);

// modular exponentiation of prime field element by unsigned integer
//
// note: if first operand is not field element, it'll be converted into one
// by performing modulo operation
extern SYCL_EXTERNAL uint64_t ff_p_pow(uint64_t a, const uint64_t b);

// finds multiplicative inverse of field element, given that it's
// not additive identity
//
// note: if operand is not part of prime field, it's made so by performing
// modulo operation
//
// this function uses the fact a ** -1 = 1 / a = a ** (p - 2) ( mod p )
// where p = prime field modulas
//
// it raises operand to (p - 2)-th power, which is multiplicative
// inverse of operand
extern SYCL_EXTERNAL uint64_t ff_p_inv(uint64_t a);

// modular division of one prime field element by another one
//
// note: both operands to be converted into field elements
// by explicitly performing modulo at very beginning
//
// it computes a * (b ** -1), uses already defined multiplicative
// inverse finder function
extern SYCL_EXTERNAL uint64_t ff_p_div(uint64_t a, uint64_t b);
