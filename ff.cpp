#include "ff.hpp"

using namespace sycl;

uint32_t ff_add(const uint32_t a, const uint32_t b) { return a ^ b; }

uint32_t ff_sub(const uint32_t a, const uint32_t b) { return a ^ b; }
