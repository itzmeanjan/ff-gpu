#include "ff_p.hpp"
#include <climits>

uint64_t ff_p_add(uint64_t a, uint64_t b) {
  a %= MOD;
  b %= MOD;

  uint64_t res_0 = a + b;
  bool over_0 = a > UINT64_MAX - b;

  uint32_t zero = 0;
  uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(over_0 ? 1 : 0));

  uint64_t res_1 = res_0 + tmp_0;
  bool over_1 = res_0 > UINT64_MAX - tmp_0;

  uint64_t tmp_1 = (uint64_t)(zero - (uint32_t)(over_1 ? 1 : 0));
  uint64_t res = res_1 + tmp_1;

  return res;
}

uint64_t ff_p_sub(uint64_t a, uint64_t b) {
  a %= MOD;
  b %= MOD;

  uint64_t res_0 = a - b;
  bool under_0 = a < b;

  uint32_t zero = 0;
  uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(under_0 ? 1 : 0));

  uint64_t res_1 = res_0 - tmp_0;
  bool under_1 = res_0 < tmp_0;

  uint64_t tmp_1 = (uint64_t)(zero - (uint32_t)(under_1 ? 1 : 0));
  uint64_t res = res_1 + tmp_1;

  return res;
}

uint64_t ff_p_mul(uint64_t a, uint64_t b) {
  a %= MOD;
  b %= MOD;

  uint128_t c_u128 = sycl::ulonglong2(a * b, sycl::mul_hi(a, b));

  uint64_t ab = c_u128.x();
  uint64_t cd = c_u128.y();
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

uint64_t ff_p_pow(uint64_t a, const uint64_t b) {
  a %= MOD;

  if (b == 0) {
    return 1;
  }

  if (a == 0) {
    return 0;
  }

  uint64_t r = b & 0b1 ? a : 1;
  for (uint8_t i = 1; i < 64 - sycl::clz(b); i++) {
    a = ff_p_mul(a, a);
    if ((b >> i) & 0b1) {
      r = ff_p_mul(r, a);
    }
  }
  return r;
}