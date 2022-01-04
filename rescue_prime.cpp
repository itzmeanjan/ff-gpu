#include <rescue_prime.hpp>

sycl::ulong4
ff_p_vec_mul_(sycl::ulong4 a, sycl::ulong4 b)
{
  sycl::ulong4 ab = a * b;
  sycl::ulong4 cd = sycl::mul_hi(a, b);
  sycl::ulong4 c = cd & MAX_UINT;
  sycl::ulong4 d = cd >> 32;

  sycl::ulong4 tmp_0 = ab - d;
  sycl::long4 und_0 = ab < d; // check if underflowed
  sycl::ulong4 tmp_1 = und_0.convert<ulong>();
  sycl::ulong4 tmp_2 = tmp_1 & MAX_UINT;
  sycl::ulong4 tmp_3 = tmp_0 - tmp_2;

  sycl::ulong4 tmp_4 = (c << 32) - c;

  sycl::ulong4 tmp_5 = tmp_3 + tmp_4;
  sycl::long4 ovr_0 = tmp_3 > std::numeric_limits<uint64_t>::max() - tmp_4;
  sycl::ulong4 tmp_6 = ovr_0.convert<ulong>();
  sycl::ulong4 tmp_7 = tmp_6 & MAX_UINT;

  return tmp_5 + tmp_7;
}

void
ff_p_vec_mul(const sycl::ulong4* a,
             const sycl::ulong4* b,
             sycl::ulong4* const c)
{
  *(c + 0) = ff_p_vec_mul_(*(a + 0), *(b + 0));
  *(c + 1) = ff_p_vec_mul_(*(a + 1), *(b + 1));
  *(c + 2) = ff_p_vec_mul_(*(a + 2), *(b + 2));
}

sycl::ulong4
ff_p_vec_add_(sycl::ulong4 a, sycl::ulong4 b)
{
  // Following four lines are equivalent of writing
  // b % FIELD_MOD, which converts all lanes of `b` vector
  // into canonical representation
  sycl::ulong4 mod_vec = sycl::ulong4(MOD);
  sycl::long4 over_0 = b >= MOD;
  sycl::ulong4 tmp_0 = (over_0.convert<ulong>() >> 63) * mod_vec;
  sycl::ulong4 b_ok = b - tmp_0;

  sycl::ulong4 tmp_1 = a + b_ok;
  sycl::long4 over_1 = a > (std::numeric_limits<uint64_t>::max() - b_ok);
  sycl::ulong4 tmp_2 = over_1.convert<ulong>() & MAX_UINT;

  sycl::ulong4 tmp_3 = tmp_1 + tmp_2;
  sycl::long4 over_2 = tmp_1 > (std::numeric_limits<uint64_t>::max() - tmp_2);
  sycl::ulong4 tmp_4 = over_2.convert<ulong>() & MAX_UINT;

  return tmp_3 + tmp_4;
}

void
ff_p_vec_add(const sycl::ulong4* a,
             const sycl::ulong4* b,
             sycl::ulong4* const c)
{
  *(c + 0) = ff_p_vec_add_(*(a + 0), *(b + 0));
  *(c + 1) = ff_p_vec_add_(*(a + 1), *(b + 1));
  *(c + 2) = ff_p_vec_add_(*(a + 2), *(b + 2));
}

void
apply_sbox(const sycl::ulong4* state_in, sycl::ulong4* const state_out)
{
  sycl::ulong4 state_in_2[3] = {};
  ff_p_vec_mul(state_in, state_in, state_in_2);

  sycl::ulong4 state_in_4[3] = {};
  ff_p_vec_mul(state_in_2, state_in_2, state_in_4);

  sycl::ulong4 state_in_6[3] = {};
  ff_p_vec_mul(state_in_2, state_in_4, state_in_6);

  ff_p_vec_mul(state_in, state_in_6, state_out);
}

void
apply_constants(const sycl::ulong4* state_in,
                scratch_mem_1d_t cnst,
                const size_t cnst_offset,
                sycl::ulong4* const state_out)
{
  *(state_out + 0) = ff_p_vec_add_(*(state_in + 0), cnst[cnst_offset + 0]);
  *(state_out + 1) = ff_p_vec_add_(*(state_in + 1), cnst[cnst_offset + 1]);
  *(state_out + 2) = ff_p_vec_add_(*(state_in + 2), cnst[cnst_offset + 2]);
}

void
apply_constants(const sycl::ulong4* state_in,
                const sycl::ulong4* cnst,
                sycl::ulong4* const state_out)
{
  ff_p_vec_add(state_in, cnst, state_out);
}

sycl::ulong
accumulate_vec4(sycl::ulong4 a)
{
  uint64_t v0 = ff_p_add(a.x(), a.y());
  uint64_t v1 = ff_p_add(a.z(), a.w());

  return static_cast<sycl::ulong>(ff_p_add(v0, v1));
}

sycl::ulong
accumulate_state(const sycl::ulong4* state)
{
  sycl::ulong v0 = accumulate_vec4(*(state + 0));
  sycl::ulong v1 = accumulate_vec4(*(state + 1));
  sycl::ulong v2 = accumulate_vec4(*(state + 2));

  return static_cast<sycl::ulong>(ff_p_add(v2, ff_p_add(v0, v1)));
}

void
apply_mds(const sycl::ulong4* state_in,
          scratch_mem_1d_t mds,
          sycl::ulong4* const state_out)
{
  sycl::ulong4 scratch[3] = {};

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[0]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[1]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[2]);

  sycl::ulong v0 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[3]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[4]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[5]);

  sycl::ulong v1 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[6]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[7]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[8]);

  sycl::ulong v2 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[9]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[10]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[11]);

  sycl::ulong v3 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[12]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[13]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[14]);

  sycl::ulong v4 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[15]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[16]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[17]);

  sycl::ulong v5 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[18]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[19]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[20]);

  sycl::ulong v6 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[21]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[22]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[23]);

  sycl::ulong v7 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[24]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[25]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[26]);

  sycl::ulong v8 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[27]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[28]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[29]);

  sycl::ulong v9 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[30]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[31]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[32]);

  sycl::ulong v10 = accumulate_state(scratch);

  scratch[0] = ff_p_vec_mul_(*(state_in + 0), mds[33]);
  scratch[1] = ff_p_vec_mul_(*(state_in + 1), mds[34]);
  scratch[2] = ff_p_vec_mul_(*(state_in + 2), mds[35]);

  sycl::ulong v11 = accumulate_state(scratch);

  *(state_out + 0) = sycl::ulong4(v0, v1, v2, v3);
  *(state_out + 1) = sycl::ulong4(v4, v5, v6, v7);
  *(state_out + 2) = sycl::ulong4(v8, v9, v10, v11);
}

void
apply_mds(const sycl::ulong4* state_in,
          const sycl::ulong4* mds,
          sycl::ulong4* const state_out)
{
  sycl::ulong4 scratch[3] = {};

  ff_p_vec_mul(state_in, mds + 0, scratch);
  sycl::ulong v0 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 3, scratch);
  sycl::ulong v1 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 6, scratch);
  sycl::ulong v2 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 9, scratch);
  sycl::ulong v3 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 12, scratch);
  sycl::ulong v4 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 15, scratch);
  sycl::ulong v5 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 18, scratch);
  sycl::ulong v6 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 21, scratch);
  sycl::ulong v7 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 24, scratch);
  sycl::ulong v8 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 27, scratch);
  sycl::ulong v9 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 30, scratch);
  sycl::ulong v10 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 33, scratch);
  sycl::ulong v11 = accumulate_state(scratch);

  *(state_out + 0) = sycl::ulong4(v0, v1, v2, v3);
  *(state_out + 1) = sycl::ulong4(v4, v5, v6, v7);
  *(state_out + 2) = sycl::ulong4(v8, v9, v10, v11);
}

void
exp_acc(const sycl::ulong m,
        const sycl::ulong4* base,
        const sycl::ulong4* tail,
        sycl::ulong4* const out)
{
  sycl::ulong4 scratch[3] = {};

  *(out + 0) = *(base + 0);
  *(out + 1) = *(base + 1);
  *(out + 2) = *(base + 2);

  for (sycl::ulong i = 0; i < m; i++) {
    ff_p_vec_mul(out, out, scratch);

    *(out + 0) = *(scratch + 0);
    *(out + 1) = *(scratch + 1);
    *(out + 2) = *(scratch + 2);
  }

  ff_p_vec_mul(out, tail, scratch);

  *(out + 0) = *(scratch + 0);
  *(out + 1) = *(scratch + 1);
  *(out + 2) = *(scratch + 2);
}

void
apply_inv_sbox(const sycl::ulong4* state_in, sycl::ulong4* const state_out)
{
  sycl::ulong4 t1[3] = {};
  ff_p_vec_mul(state_in, state_in, t1);

  sycl::ulong4 t2[3] = {};
  ff_p_vec_mul(t1, t1, t2);

  sycl::ulong4 t3[3] = {};
  exp_acc(3, t2, t2, t3);

  sycl::ulong4 t4[3] = {};
  exp_acc(6, t3, t3, t4);

  sycl::ulong4 t5[3] = {};
  exp_acc(12, t4, t4, t5);

  sycl::ulong4 t6[3] = {};
  exp_acc(6, t5, t3, t6);

  sycl::ulong4 t7[3] = {};
  exp_acc(31, t6, t6, t7);

  sycl::ulong4 a[3] = {};
  sycl::ulong4 b[3] = {};
  sycl::ulong4 scratch[3] = {};

  ff_p_vec_mul(t7, t7, scratch);
  ff_p_vec_mul(t6, scratch, a);
  ff_p_vec_mul(a, a, scratch);
  ff_p_vec_mul(scratch, scratch, a);

  ff_p_vec_mul(t1, t2, scratch);
  ff_p_vec_mul(scratch, state_in, b);

  ff_p_vec_mul(a, b, state_out);
}

void
apply_permutation_round(const sycl::ulong4* state_in,
                        scratch_mem_1d_t mds,
                        scratch_mem_1d_t ark1,
                        scratch_mem_1d_t ark2,
                        const size_t ark_offset,
                        sycl::ulong4* const state_out)
{
  sycl::ulong4 scratch_0[3] = {};
  sycl::ulong4 scratch_1[3] = {};
  sycl::ulong4 scratch_2[3] = {};

  apply_sbox(state_in, scratch_0);
  apply_mds(scratch_0, mds, scratch_1);
  apply_constants(scratch_1, ark1, ark_offset, scratch_2);

  apply_inv_sbox(scratch_2, scratch_0);
  apply_mds(scratch_0, mds, scratch_1);
  apply_constants(scratch_1, ark2, ark_offset, state_out);
}

void
apply_permutation_round(const sycl::ulong4* state_in,
                        const sycl::ulong4* mds,
                        const sycl::ulong4* ark1,
                        const sycl::ulong4* ark2,
                        sycl::ulong4* const state_out)
{
  sycl::ulong4 scratch_0[3] = {};
  sycl::ulong4 scratch_1[3] = {};
  sycl::ulong4 scratch_2[3] = {};

  apply_sbox(state_in, scratch_0);
  apply_mds(scratch_0, mds, scratch_1);
  apply_constants(scratch_1, ark1, scratch_2);

  apply_inv_sbox(scratch_2, scratch_0);
  apply_mds(scratch_0, mds, scratch_1);
  apply_constants(scratch_1, ark2, state_out);
}

void
apply_rescue_permutation(const sycl::ulong4* state_in,
                         scratch_mem_1d_t mds,
                         scratch_mem_1d_t ark1,
                         scratch_mem_1d_t ark2,
                         sycl::ulong4* const state_out)
{
  sycl::ulong4 scratch_0[3] = {};
  sycl::ulong4 scratch_1[3] = {};
  sycl::ulong4 scratch_2[3] = {};

  apply_permutation_round(state_in, mds, ark1, ark2, 0, scratch_0);
  apply_permutation_round(scratch_0, mds, ark1, ark2, 3, scratch_1);
  apply_permutation_round(scratch_1, mds, ark1, ark2, 6, scratch_2);
  apply_permutation_round(scratch_2, mds, ark1, ark2, 9, scratch_0);
  apply_permutation_round(scratch_0, mds, ark1, ark2, 12, scratch_1);
  apply_permutation_round(scratch_1, mds, ark1, ark2, 15, scratch_2);
  apply_permutation_round(scratch_2, mds, ark1, ark2, 18, state_out);
}

void
apply_rescue_permutation(const sycl::ulong4* state_in,
                         const sycl::ulong4* mds,
                         const sycl::ulong4* ark1,
                         const sycl::ulong4* ark2,
                         sycl::ulong4* const state_out)
{
  sycl::ulong4 scratch_0[3] = {};
  sycl::ulong4 scratch_1[3] = {};
  sycl::ulong4 scratch_2[3] = {};

  apply_permutation_round(state_in, mds, ark1 + 0, ark2 + 0, scratch_0);
  apply_permutation_round(scratch_0, mds, ark1 + 3, ark2 + 3, scratch_1);
  apply_permutation_round(scratch_1, mds, ark1 + 6, ark2 + 6, scratch_2);
  apply_permutation_round(scratch_2, mds, ark1 + 9, ark2 + 9, scratch_0);
  apply_permutation_round(scratch_0, mds, ark1 + 12, ark2 + 12, scratch_1);
  apply_permutation_round(scratch_1, mds, ark1 + 15, ark2 + 15, scratch_2);
  apply_permutation_round(scratch_2, mds, ark1 + 18, ark2 + 18, state_out);
}

void
hash_elements(const sycl::ulong* input_elements,
              const sycl::ulong count,
              sycl::ulong* const hash,
              scratch_mem_1d_t mds,
              scratch_mem_1d_t ark1,
              scratch_mem_1d_t ark2)
{
  sycl::ulong4 state[3] = { sycl::ulong4(0),
                            sycl::ulong4(0),
                            sycl::ulong4(0, 0, 0, count % MOD) };
  sycl::ulong4 scratch[3] = {};

  sycl::ulong i = 0;
  for (sycl::ulong j = 0; j < count; j++) {
    switch (i) {
      case 0:
        state[0].x() = ff_p_add(state[0].x(), *(input_elements + j));
        break;
      case 1:
        state[0].y() = ff_p_add(state[0].y(), *(input_elements + j));
        break;
      case 2:
        state[0].z() = ff_p_add(state[0].z(), *(input_elements + j));
        break;
      case 3:
        state[0].w() = ff_p_add(state[0].w(), *(input_elements + j));
        break;
      case 4:
        state[1].x() = ff_p_add(state[1].x(), *(input_elements + j));
        break;
      case 5:
        state[1].y() = ff_p_add(state[1].y(), *(input_elements + j));
        break;
      case 6:
        state[1].z() = ff_p_add(state[1].z(), *(input_elements + j));
        break;
      case 7:
        state[1].w() = ff_p_add(state[1].w(), *(input_elements + j));
        break;
    }

    if ((++i) % RATE_WIDTH == 0) {
      apply_rescue_permutation(state, mds, ark1, ark2, scratch);
      i = 0;

      *(state + 0) = *(scratch + 0);
      *(state + 1) = *(scratch + 1);
      *(state + 2) = *(scratch + 2);
    }
  }

  if (i > 0) {
    apply_rescue_permutation(state, mds, ark1, ark2, scratch);

    *(state + 0) = *(scratch + 0);
    *(state + 1) = *(scratch + 1);
    *(state + 2) = *(scratch + 2);
  }

  *(hash + 0) = state[0].x();
  *(hash + 1) = state[0].y();
  *(hash + 2) = state[0].z();
  *(hash + 3) = state[0].w();
}

void
hash_elements(const sycl::ulong* input_elements,
              const sycl::ulong count,
              sycl::ulong* const hash,
              const sycl::ulong4* mds,
              const sycl::ulong4* ark1,
              const sycl::ulong4* ark2)
{
  sycl::ulong4 state[3] = { sycl::ulong4(0),
                            sycl::ulong4(0),
                            sycl::ulong4(0, 0, 0, count % MOD) };
  sycl::ulong4 scratch[3] = {};

  sycl::ulong i = 0;
  for (sycl::ulong j = 0; j < count; j++) {
    switch (i) {
      case 0:
        state[0].x() = ff_p_add(state[0].x(), *(input_elements + j));
        break;
      case 1:
        state[0].y() = ff_p_add(state[0].y(), *(input_elements + j));
        break;
      case 2:
        state[0].z() = ff_p_add(state[0].z(), *(input_elements + j));
        break;
      case 3:
        state[0].w() = ff_p_add(state[0].w(), *(input_elements + j));
        break;
      case 4:
        state[1].x() = ff_p_add(state[1].x(), *(input_elements + j));
        break;
      case 5:
        state[1].y() = ff_p_add(state[1].y(), *(input_elements + j));
        break;
      case 6:
        state[1].z() = ff_p_add(state[1].z(), *(input_elements + j));
        break;
      case 7:
        state[1].w() = ff_p_add(state[1].w(), *(input_elements + j));
        break;
    }

    if ((++i) % RATE_WIDTH == 0) {
      apply_rescue_permutation(state, mds, ark1, ark2, scratch);
      i = 0;

      *(state + 0) = *(scratch + 0);
      *(state + 1) = *(scratch + 1);
      *(state + 2) = *(scratch + 2);
    }
  }

  if (i > 0) {
    apply_rescue_permutation(state, mds, ark1, ark2, scratch);

    *(state + 0) = *(scratch + 0);
    *(state + 1) = *(scratch + 1);
    *(state + 2) = *(scratch + 2);
  }

  *(hash + 0) = state[0].x();
  *(hash + 1) = state[0].y();
  *(hash + 2) = state[0].z();
  *(hash + 3) = state[0].w();
}

void
merge(const sycl::ulong* input_hashes,
      sycl::ulong* const merged_hash,
      scratch_mem_1d_t mds,
      scratch_mem_1d_t ark1,
      scratch_mem_1d_t ark2)
{
  sycl::ulong4 state[3] = { sycl::ulong4(*(input_hashes + 0),
                                         *(input_hashes + 1),
                                         *(input_hashes + 2),
                                         *(input_hashes + 3)),
                            sycl::ulong4(*(input_hashes + 4),
                                         *(input_hashes + 5),
                                         *(input_hashes + 6),
                                         *(input_hashes + 7)),
                            sycl::ulong4(0, 0, 0, RATE_WIDTH) };
  sycl::ulong4 scratch[3] = {};

  apply_rescue_permutation(state, mds, ark1, ark2, scratch);

  *(merged_hash + 0) = scratch[0].x();
  *(merged_hash + 1) = scratch[0].y();
  *(merged_hash + 2) = scratch[0].z();
  *(merged_hash + 3) = scratch[0].w();
}

void
merge(const sycl::ulong* input_hashes,
      sycl::ulong* const merged_hash,
      const sycl::ulong4* mds,
      const sycl::ulong4* ark1,
      const sycl::ulong4* ark2)
{
  sycl::ulong4 state[3] = { sycl::ulong4(*(input_hashes + 0),
                                         *(input_hashes + 1),
                                         *(input_hashes + 2),
                                         *(input_hashes + 3)),
                            sycl::ulong4(*(input_hashes + 4),
                                         *(input_hashes + 5),
                                         *(input_hashes + 6),
                                         *(input_hashes + 7)),
                            sycl::ulong4(0, 0, 0, RATE_WIDTH) };
  sycl::ulong4 scratch[3] = {};

  apply_rescue_permutation(state, mds, ark1, ark2, scratch);

  *(merged_hash + 0) = scratch[0].x();
  *(merged_hash + 1) = scratch[0].y();
  *(merged_hash + 2) = scratch[0].z();
  *(merged_hash + 3) = scratch[0].w();
}

void
prepare_mds(sycl::ulong4* const mds)
{
  for (size_t i = 0; i < STATE_WIDTH * 3; i++) {
    *(mds + i) = sycl::ulong4(
      MDS[i * 4 + 0], MDS[i * 4 + 1], MDS[i * 4 + 2], MDS[i * 4 + 3]);
  }
}

void
prepare_ark1(sycl::ulong4* const ark1)
{
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    *(ark1 + i) = sycl::ulong4(
      ARK1[i * 4 + 0], ARK1[i * 4 + 1], ARK1[i * 4 + 2], ARK1[i * 4 + 3]);
  }
}

void
prepare_ark2(sycl::ulong4* const ark2)
{
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    *(ark2 + i) = sycl::ulong4(
      ARK2[i * 4 + 0], ARK2[i * 4 + 1], ARK2[i * 4 + 2], ARK2[i * 4 + 3]);
  }
}
