#include <rescue_prime_vectorized.hpp>

sycl::ulong16 ff_p_vec_mul(sycl::ulong16 a, sycl::ulong16 b) {
  sycl::ulong16 ab = a * b;
  sycl::ulong16 cd = sycl::mul_hi(a, b);
  sycl::ulong16 c = cd & MAX_UINT;
  sycl::ulong16 d = cd >> 32;

  sycl::ulong16 tmp_0 = ab - d;
  sycl::long16 und_0 = ab < d; // check if underflowed
  sycl::ulong16 tmp_1 = und_0.convert<ulong>();
  sycl::ulong16 tmp_2 = tmp_1 & MAX_UINT;
  sycl::ulong16 tmp_3 = tmp_0 - tmp_2;

  sycl::ulong16 tmp_4 = (c << 32) - c;

  sycl::ulong16 tmp_5 = tmp_3 + tmp_4;
  sycl::long16 ovr_0 = tmp_3 > std::numeric_limits<uint64_t>::max() - tmp_4;
  sycl::ulong16 tmp_6 = ovr_0.convert<ulong>();
  sycl::ulong16 tmp_7 = tmp_6 & MAX_UINT;

  return tmp_5 + tmp_7;
}

sycl::ulong16 ff_p_vec_add(sycl::ulong16 a, sycl::ulong16 b) {
  // Following four lines are equivalent of writing
  // b % FIELD_MOD, which converts all lanes of `b` vector
  // into canonical representation
  sycl::ulong16 mod_vec = sycl::ulong16(FIELD_MOD);
  sycl::long16 over_0 = b >= FIELD_MOD;
  sycl::ulong16 tmp_0 = (over_0.convert<ulong>() >> 63) * mod_vec;
  sycl::ulong16 b_ok = b - tmp_0;

  sycl::ulong16 tmp_1 = a + b_ok;
  sycl::long16 over_1 = a > (std::numeric_limits<uint64_t>::max() - b_ok);
  sycl::ulong16 tmp_2 = over_1.convert<ulong>() & MAX_UINT;

  sycl::ulong16 tmp_3 = tmp_1 + tmp_2;
  sycl::long16 over_2 = tmp_1 > (std::numeric_limits<uint64_t>::max() - tmp_2);
  sycl::ulong16 tmp_4 = over_2.convert<ulong>() & MAX_UINT;

  return tmp_3 + tmp_4;
}

sycl::ulong16 apply_sbox(sycl::ulong16 state) {
  sycl::ulong16 state2 = ff_p_vec_mul(state, state);
  sycl::ulong16 state4 = ff_p_vec_mul(state2, state2);
  sycl::ulong16 state6 = ff_p_vec_mul(state2, state4);

  return ff_p_vec_mul(state, state6);
}
