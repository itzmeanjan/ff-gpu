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

sycl::ulong16 apply_constants(sycl::ulong16 state, sycl::ulong16 cnst) {
  return ff_p_vec_add(state, cnst);
}

sycl::ulong accumulate_vec4(sycl::ulong4 a) {
  uint64_t v0 = ff_p_add(a.x(), a.y());
  uint64_t v1 = ff_p_add(a.z(), a.w());

  return static_cast<sycl::ulong>(ff_p_add(v0, v1));
}

sycl::ulong accumulate_state(sycl::ulong16 state) {
  sycl::ulong8 state_lo = state.lo();
  sycl::ulong8 state_hi = state.hi();

  sycl::ulong v0 = accumulate_vec4(state_lo.lo());
  sycl::ulong v1 = accumulate_vec4(state_lo.hi());
  sycl::ulong v2 = accumulate_vec4(state_hi.lo());
  sycl::ulong v3 = accumulate_vec4(state_hi.hi());

  return accumulate_vec4(sycl::ulong4(v0, v1, v2, v3));
}

sycl::ulong16 apply_mds(sycl::ulong16 state, sycl::ulong16 mds[12]) {
  sycl::ulong v0 = accumulate_state(ff_p_vec_mul(state, mds[0]));
  sycl::ulong v1 = accumulate_state(ff_p_vec_mul(state, mds[1]));
  sycl::ulong v2 = accumulate_state(ff_p_vec_mul(state, mds[2]));
  sycl::ulong v3 = accumulate_state(ff_p_vec_mul(state, mds[3]));
  sycl::ulong v4 = accumulate_state(ff_p_vec_mul(state, mds[4]));
  sycl::ulong v5 = accumulate_state(ff_p_vec_mul(state, mds[5]));
  sycl::ulong v6 = accumulate_state(ff_p_vec_mul(state, mds[6]));
  sycl::ulong v7 = accumulate_state(ff_p_vec_mul(state, mds[7]));
  sycl::ulong v8 = accumulate_state(ff_p_vec_mul(state, mds[8]));
  sycl::ulong v9 = accumulate_state(ff_p_vec_mul(state, mds[9]));
  sycl::ulong v10 = accumulate_state(ff_p_vec_mul(state, mds[10]));
  sycl::ulong v11 = accumulate_state(ff_p_vec_mul(state, mds[11]));

  // note: last 4 vector lanes don't contribute anyway so, I'm
  // just filling them with 0
  return sycl::ulong16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, 0, 0,
                       0, 0);
}
