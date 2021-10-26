#include "ff.hpp"

using namespace sycl;

uint32_t ff_add(const uint32_t a, const uint32_t b) { return a ^ b; }

uint32_t ff_sub(const uint32_t a, const uint32_t b) { return a ^ b; }

uint32_t ff_neg(const uint32_t a) { return a; }

uint32_t ff_mult(const uint32_t a, const uint32_t b) {
  uint32_t a_ = a;
  uint32_t b_ = b;
  uint32_t c = 0;

  if (b > a) {
    a_ = b;
    b_ = a;
  }

  while (b_ > 0) {
    if (b_ & 0b1) {
      c ^= a_;
    }

    b_ >>= 1;
    a_ <<= 1;

    if(a_ >= ORDER) {
      a_ ^= IRREDUCIBLE_POLY;
    }
  }

  return c;
}
