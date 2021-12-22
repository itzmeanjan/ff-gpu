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

sycl::ulong16 apply_mds(sycl::ulong16 state) {
  sycl::ulong v0 = accumulate_state(ff_p_vec_mul(state, MDS[0]));
  sycl::ulong v1 = accumulate_state(ff_p_vec_mul(state, MDS[1]));
  sycl::ulong v2 = accumulate_state(ff_p_vec_mul(state, MDS[2]));
  sycl::ulong v3 = accumulate_state(ff_p_vec_mul(state, MDS[3]));
  sycl::ulong v4 = accumulate_state(ff_p_vec_mul(state, MDS[4]));
  sycl::ulong v5 = accumulate_state(ff_p_vec_mul(state, MDS[5]));
  sycl::ulong v6 = accumulate_state(ff_p_vec_mul(state, MDS[6]));
  sycl::ulong v7 = accumulate_state(ff_p_vec_mul(state, MDS[7]));
  sycl::ulong v8 = accumulate_state(ff_p_vec_mul(state, MDS[8]));
  sycl::ulong v9 = accumulate_state(ff_p_vec_mul(state, MDS[9]));
  sycl::ulong v10 = accumulate_state(ff_p_vec_mul(state, MDS[10]));
  sycl::ulong v11 = accumulate_state(ff_p_vec_mul(state, MDS[11]));

  // note: last 4 vector lanes don't contribute anyway so, I'm
  // just filling them with 0
  return sycl::ulong16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, 0, 0,
                       0, 0);
}

sycl::ulong16 exp_acc(const sycl::ulong m, sycl::ulong16 base,
                      sycl::ulong16 tail) {
  sycl::ulong16 res = base; // just copies all vector lanes

  for (sycl::ulong i = 0; i < m; i++) {
    res = ff_p_vec_mul(res, res);
  }

  return ff_p_vec_mul(res, tail);
}

sycl::ulong16 apply_inv_sbox(sycl::ulong16 state) {
  sycl::ulong16 t1 = ff_p_vec_mul(state, state);
  sycl::ulong16 t2 = ff_p_vec_mul(t1, t1);

  sycl::ulong16 t3 = exp_acc(3, t2, t2);
  sycl::ulong16 t4 = exp_acc(6, t3, t3);
  t4 = exp_acc(12, t4, t4);

  sycl::ulong16 t5 = exp_acc(6, t4, t3);
  sycl::ulong16 t6 = exp_acc(31, t5, t5);

  sycl::ulong16 a = ff_p_vec_mul(ff_p_vec_mul(t6, t6), t5);
  a = ff_p_vec_mul(a, a);
  a = ff_p_vec_mul(a, a);
  sycl::ulong16 b = ff_p_vec_mul(ff_p_vec_mul(t1, t2), state);

  return ff_p_vec_mul(a, b);
}

sycl::ulong16 apply_permutation_round(sycl::ulong16 state, sycl::ulong16 ark1,
                                      sycl::ulong16 ark2) {
  state = apply_sbox(state);
  state = apply_mds(state);
  state = apply_constants(state, ark1);

  state = apply_inv_sbox(state);
  state = apply_mds(state);
  state = apply_constants(state, ark2);

  return state;
}

sycl::ulong16 apply_rescue_permutation(sycl::ulong16 state) {
  for (sycl::ulong i = 0; i < NUM_ROUNDS; i++) {
    state = apply_permutation_round(state, ARK1[i], ARK2[i]);
  }
  return state;
}

void hash_elements(const sycl::ulong *input_elements, const sycl::ulong count,
                   sycl::ulong *const hash) {
  sycl::ulong16 state = sycl::ulong16(0);
  state.sB() = count % FIELD_MOD;

  sycl::ulong i = 0;
  for (sycl::ulong j = 0; j < count; j++) {
    switch (i) {
    case 0:
      state.s0() = ff_p_add(state.s0(), *(input_elements + j));
      break;
    case 1:
      state.s1() = ff_p_add(state.s1(), *(input_elements + j));
      break;
    case 2:
      state.s2() = ff_p_add(state.s2(), *(input_elements + j));
      break;
    case 3:
      state.s3() = ff_p_add(state.s3(), *(input_elements + j));
      break;
    case 4:
      state.s4() = ff_p_add(state.s4(), *(input_elements + j));
      break;
    case 5:
      state.s5() = ff_p_add(state.s5(), *(input_elements + j));
      break;
    case 7:
      state.s7() = ff_p_add(state.s7(), *(input_elements + j));
      break;
    }

    if ((++i) % RATE_WIDTH == 0) {
      state = apply_rescue_permutation(state);
      i = 0;
    }
  }

  if (i > 0) {
    state = apply_rescue_permutation(state);
  }

  sycl::ulong4 digest = static_cast<sycl::ulong4>(state.swizzle<0, 1, 2, 3>());

  *(hash + 0) = digest.x();
  *(hash + 1) = digest.y();
  *(hash + 2) = digest.z();
  *(hash + 3) = digest.w();
}
