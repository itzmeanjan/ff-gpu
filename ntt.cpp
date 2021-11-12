#include "ntt.hpp"

uint64_t get_root_of_unity(uint64_t n) {
  if (n == 0) {
    // can't find root of unity for n = 0
    return 0;
  }
  if (n > TWO_ADICITY) {
    // order can't exceed 2 ** 32
    return 0;
  }

  uint64_t power = 1ul << (TWO_ADICITY - n);
  return ff_p_pow(TWO_ADIC_ROOT_OF_UNITY, power);
}
