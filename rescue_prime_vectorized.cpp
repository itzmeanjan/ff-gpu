#include <rescue_prime_vectorized.hpp>

sycl::ulong16 ff_p_vec_mul(sycl::ulong16 a, sycl::ulong16 b) {
  sycl::ulong16 ab = a * b;
  sycl::ulong16 cd = sycl::mul_hi(a, b);
  sycl::ulong16 c = cd & 0xffffffff;
  sycl::ulong16 d = cd >> 32;

  sycl::ulong16 tmp_0 = ab - d;
  sycl::long16 und_0 = ab < d; // check if underflowed
  sycl::ulong16 tmp_1 = und_0.convert<ulong>();
  sycl::ulong16 tmp_2 = tmp_1 & 0xffffffff;
  sycl::ulong16 tmp_3 = tmp_0 - tmp_2;

  sycl::ulong16 tmp_4 = (c << 32) - c;

  sycl::ulong16 tmp_5 = tmp_3 + tmp_4;
  sycl::long16 ovr_0 = tmp_3 > std::numeric_limits<uint64_t>::max() - tmp_4;
  sycl::ulong16 tmp_6 = ovr_0.convert<ulong>();
  sycl::ulong16 tmp_7 = tmp_6 & 0xffffffff;

  return tmp_5 + tmp_7;
}
