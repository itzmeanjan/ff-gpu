#pragma once
#include <ff_p.hpp>
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

// Updates each element of rescue prime hash state ( 16 lane wide ) by
// exponentiating to their 7-th power
//
// Note this implementation doesn't use modular exponentiation routine, instead
// it uses multiple multiplications ( actually squaring )
//
// Collects huge motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L68-L88
SYCL_EXTERNAL sycl::ulong16 apply_sbox(sycl::ulong16 state);

// Applies rescue round key constants on hash state
//
// actually simple vectorized modular addition --- that's all this routine does
//
// inline it ?
//
// Taken from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L97-L106
SYCL_EXTERNAL sycl::ulong16 apply_constants(sycl::ulong16 state,
                                            sycl::ulong16 cnst);

// Reduces four prime field element vector into single accumulated prime
// field element, by performing modular addition
//
// Adapted from here
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L143-L166
SYCL_EXTERNAL sycl::ulong accumulate_vec4(sycl::ulong4 a);

// Accumulates state of rescue prime hash into single prime field element
//
// Takes some inspiration from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L168-L199
SYCL_EXTERNAL sycl::ulong accumulate_state(sycl::ulong16 state);

// Performs matrix vector multiplication; updates state of rescue prime
// hash by applying MDS matrix
//
// Adopted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L201-L231
SYCL_EXTERNAL sycl::ulong16 apply_mds(sycl::ulong16 state,
                                      sycl::ulong16 mds[12]);

// Instead of exponentiating hash state by some large number, this function
// helps in computing exponentiation by performing multiple modular
// multiplications
//
// This is invoked from following `apply_inv_sbox` function ( multiple times )
SYCL_EXTERNAL sycl::ulong16 exp_acc(const sycl::ulong m, sycl::ulong16 base,
                                    sycl::ulong16 tail);
