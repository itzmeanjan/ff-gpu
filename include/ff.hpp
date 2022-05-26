#pragma once
#include <bit>
#include <cstdint>

// Finite field arithmetics for prime = 2^64 - 2^32 + 1
namespace ff {

// Prime modulus of field, F_p, where p = 2 ** 64 - 2 ** 32 + 1
constexpr uint64_t MOD = (((1ull << 63) - (1ull << 31)) << 1) + 1ull;

// Given a 64 -bit unsigned integer, this function converts it to canonical
// representation by computing a % MOD
static inline uint64_t
to_canonical(const uint64_t a)
{
  const uint64_t res[2] = { a, a - ff::MOD };
  return res[a >= ff::MOD];
}

// Modular addition of two prime field elements
//
// Note: operands doesn't necessarily need to ∈ F_p but second operand will be
// converted to canonical representation
//
// Return value may ∉ F_p, it's function invoker's responsibility to convert it
// to canonical representation
static inline uint64_t
add(const uint64_t a, uint64_t b)
{
  b = to_canonical(b);

  const uint64_t res0 = a + b;
  const bool over0 = a > UINT64_MAX - b;

  const uint64_t t0 = static_cast<uint64_t>(0u - static_cast<uint32_t>(over0));

  const uint64_t res1 = res0 + t0;
  const bool over1 = res0 > UINT64_MAX - t0;

  const uint64_t t1 = static_cast<uint64_t>(0u - static_cast<uint32_t>(over1));

  return res1 + t1;
}

// Modular subtraction of two prime field elements
//
// Note: operands doesn't necessarily need to ∈ F_p but second operand will be
// converted to canonical representation
//
// Return value may ∉ F_p, it's function invoker's responsibility to convert it
// to canonical representation
static inline uint64_t
sub(const uint64_t a, uint64_t b)
{
  b = to_canonical(b);

  const uint64_t res0 = a - b;
  const bool under0 = a < b;

  const uint64_t t0 = static_cast<uint64_t>(0u - static_cast<uint32_t>(under0));

  const uint64_t res1 = res0 - t0;
  const bool under1 = res0 < t0;

  const uint64_t t1 = static_cast<uint64_t>(0u - static_cast<uint32_t>(under1));

  return res1 + t1;
}

// Given two 64 -bit unsigned integers ( say a, b ), this function computes
// higher 64 -bits of  a * b
//
// See
// https://github.com/itzmeanjan/simd-rescue-prime/blob/c2b4de0/src/ff.rs#L10-L25
static inline uint64_t
mul_hi(const uint64_t a, const uint64_t b)
{
  const uint64_t a_lo = a & static_cast<uint64_t>(UINT32_MAX);
  const uint64_t a_hi = a >> 32;
  const uint64_t b_lo = b & static_cast<uint64_t>(UINT32_MAX);
  const uint64_t b_hi = b >> 32;

  const uint64_t a_x_b_hi = a_hi * b_hi;
  const uint64_t a_x_b_mid = a_hi * b_lo;
  const uint64_t b_x_a_mid = b_hi * a_lo;
  const uint64_t a_x_b_lo = a_lo * b_lo;

  const uint64_t t0 = static_cast<uint64_t>(static_cast<uint32_t>(a_x_b_mid));
  const uint64_t t1 = static_cast<uint64_t>(static_cast<uint32_t>(b_x_a_mid));
  const uint64_t t2 = a_x_b_lo >> 32;

  const uint64_t carry = (t0 + t1 + t2) >> 32;

  return a_x_b_hi + (a_x_b_mid >> 32) + (b_x_a_mid >> 32) + carry;
}

// Modular mulitiplication of two prime field elements
//
// Note: operands doesn't necessarily need to ∈ F_p
// but second operand will be made `b % MOD`
//
// Return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
static inline uint64_t
mult(uint64_t a, uint64_t b)
{
  if (b >= MOD) {
    b -= MOD;
  }

  uint64_t ab = a * b;
  uint64_t cd = mul_hi(a, b);
  uint64_t c = cd & 0x00000000ffffffff;
  uint64_t d = cd >> 32;

  uint64_t res_0 = ab - d;
  bool under_0 = ab < d;

  uint32_t zero = 0;
  uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(under_0 ? 1 : 0));
  res_0 -= tmp_0;

  uint64_t tmp_1 = (c << 32) - c;

  uint64_t res_1 = res_0 + tmp_1;
  bool over_0 = res_0 > UINT64_MAX - tmp_1;

  uint64_t tmp_2 = (uint64_t)(zero - (uint32_t)(over_0 ? 1 : 0));
  uint64_t res = res_1 + tmp_2;

  return res;
}

// Modular exponentiation of prime field element by unsigned integer
//
// Note: operands doesn't necessarily need to ∈ F_p
//
// Return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
static inline uint64_t
pow(uint64_t a, const uint64_t b)
{
  if (b == 0) {
    return 1;
  }

  if (b == 1) {
    return a;
  }

  if (a == 0) {
    return 0;
  }

  uint64_t r = b & 0b1 ? a : 1;
  for (uint8_t i = 1; i < 64 - std::countl_zero(b); i++) {
    a = mult(a, a);
    if ((b >> i) & 0b1) {
      r = mult(r, a);
    }
  }
  return r;
}

// Finds multiplicative inverse of field element, given that it's
// not additive identity
//
// Note: if operand is not ∈ F_p, it's made so by performing
// modulo operation
//
// This function uses the fact a ** -1 = 1 / a = a ** (p - 2) ( mod p )
// where p = prime field modulas
//
// It raises operand to (p - 2)-th power, which is multiplicative
// inverse of operand
//
// Return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
static inline uint64_t
inv(uint64_t a)
{
  if (a >= MOD) {
    a -= MOD;
  }

  if (a == 0) {
    // ** no multiplicative inverse of additive identity **
    //
    // I'm not throwing an exception from here, because
    // this function is supposed to be invoked from
    // kernel body, where exception throwing is not (yet) allowed !
    return 0;
  }

  const uint64_t exp = MOD - 2;
  return pow(a, exp);
}

// Modular division of one prime field element by another one
//
// Note: operands doesn't necessarily need to ∈ F_p
//
// It computes a * (b ** -1), uses already defined multiplicative
// inverse finder function
//
// Return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
static inline uint64_t
div(uint64_t a, uint64_t b)
{
  if (b == 0) {
    // ** no multiplicative inverse of additive identity **
    //
    // I'm not throwing an exception from here, because
    // this function is supposed to be invoked from
    // kernel body, where exception throwing is not (yet) allowed !
    return 0;
  }

  if (a == 0) {
    return 0;
  }

  uint64_t b_inv = inv(b);
  return mult(a, b_inv);
}

}
