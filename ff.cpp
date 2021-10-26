#include "ff.hpp"
#include <stdexcept>

uint32_t ff_add(const uint32_t a, const uint32_t b) { return a ^ b; }

uint32_t ff_sub(const uint32_t a, const uint32_t b) { return a ^ b; }

uint32_t ff_neg(const uint32_t a) { return a; }

uint32_t ff_mult(const uint32_t a, const uint32_t b) {
  uint64_t a_ = a;
  uint32_t b_ = b;
  uint32_t c = 0;

  if (b > a) {
    a_ = b;
    b_ = a;
  }

  while (b_ > 0) {
    if (b_ & 0b1) {
      c = c ^ (uint32_t)a_;
    }

    b_ >>= 1;
    a_ <<= 1;

    if (a_ >= ORDER) {
      a_ ^= IRREDUCIBLE_POLY;
    }
  }

  return c;
}

uint32_t ff_inv(const uint32_t a) {
  if (a == 0) {
    throw std::invalid_argument(
        "no multiplicative inverse of additive identity");
  }

  uint32_t exp = ORDER - 0b10;
  uint32_t res_s = a;
  uint32_t res_m = 1;

  while (exp > 1) {
    if (exp % 2 == 0) {
      res_s = ff_mult(res_s, res_s);
      exp /= 2;
    } else {
      res_m = ff_mult(res_m, res_s);
      exp -= 1;
    }
  }

  uint32_t res = ff_mult(res_m, res_s);
  return res;
}

uint32_t ff_div(const uint32_t a, const uint32_t b) {
  if (b == 0) {
    throw std::invalid_argument(
        "no multiplicative inverse of additive identity");
  }

  if (a == 0) {
    return 0;
  }

  uint32_t b_inv = ff_inv(b);
  uint32_t res = ff_mult(a, b_inv);
  return res;
}

uint32_t ff_pow(const uint32_t a, const int32_t b) {
  if (a == 0 && b < 0) {
    throw std::invalid_argument(
        "no multiplicative inverse of additive identity");
  }

  if (b == 0) {
    return 1;
  }

  uint32_t a_ = a;
  uint32_t b_ = b < 0 ? std::abs(b) : (uint32_t)b;
  if (b < 0) {
    a_ = ff_inv(a);
  }

  uint32_t res_s = a_;
  uint32_t res_m = 1;

  while (b_ > 1) {
    if (b_ % 2 == 0) {
      res_s = ff_mult(res_s, res_s);
      b_ /= 2;
    } else {
      res_m = ff_mult(res_m, res_s);
      b_ -= 1;
    }
  }

  uint32_t res = ff_mult(res_m, res_s);
  return res;
}
