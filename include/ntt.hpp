#pragma once
#include "ff_p.hpp"

inline constexpr uint64_t TWO_ADICITY = 32ul;
inline constexpr uint64_t TWO_ADIC_ROOT_OF_UNITY = 1753635133440165772ul;

extern SYCL_EXTERNAL uint64_t get_root_of_unity(uint64_t n);
