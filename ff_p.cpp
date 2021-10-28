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
