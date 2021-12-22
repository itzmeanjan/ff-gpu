#include "rescue_prime.hpp"

void
hash_elements(const uint64_t* elements,
              const uint64_t count,
              uint64_t* const hash)
{
  rescue_prime_state_t state{ .f_b = count >= MOD ? count - MOD : count };

  uint64_t i = 0;
  for (uint64_t j = 0; j < count; j++) {
    switch (i) {
      case 0:
        state.f_0 = ff_p_add(state.f_0, *(elements + j));
        break;
      case 1:
        state.f_1 = ff_p_add(state.f_1, *(elements + j));
        break;
      case 2:
        state.f_2 = ff_p_add(state.f_2, *(elements + j));
        break;
      case 3:
        state.f_3 = ff_p_add(state.f_3, *(elements + j));
        break;
      case 4:
        state.f_4 = ff_p_add(state.f_4, *(elements + j));
        break;
      case 5:
        state.f_5 = ff_p_add(state.f_5, *(elements + j));
        break;
      case 6:
        state.f_6 = ff_p_add(state.f_6, *(elements + j));
        break;
      case 7:
        state.f_7 = ff_p_add(state.f_7, *(elements + j));
        break;
      default:
        // we should not reach to this condition ever !
        break;
    }
    i++;
    if (i % RATE_WIDTH == 0) {
      apply_permutation(&state);
      i = 0;
    }
  }

  if (i > 0) {
    apply_permutation(&state);
  }

  *(hash + 0) = state.f_0;
  *(hash + 1) = state.f_1;
  *(hash + 2) = state.f_2;
  *(hash + 3) = state.f_3;
}

void
apply_permutation(rescue_prime_state_t* state)
{
  for (uint64_t i = 0; i < NUM_ROUNDS; i++) {
    apply_round(state, i);
  }
}

void
apply_round(rescue_prime_state_t* state, const uint64_t round)
{
  apply_sbox(state);
  apply_mds(state);
  apply_constants(state, ARK1[round]);

  apply_inv_sbox(state);
  apply_mds(state);
  apply_constants(state, ARK2[round]);
}

void
apply_sbox(rescue_prime_state_t* state)
{
  uint64_t t2 = 0ul, t4 = 0ul;

  t2 = ff_p_mult(state->f_0, state->f_0);
  t4 = ff_p_mult(t2, t2);
  state->f_0 = ff_p_mult(state->f_0, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_1, state->f_1);
  t4 = ff_p_mult(t2, t2);
  state->f_1 = ff_p_mult(state->f_1, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_2, state->f_2);
  t4 = ff_p_mult(t2, t2);
  state->f_2 = ff_p_mult(state->f_2, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_3, state->f_3);
  t4 = ff_p_mult(t2, t2);
  state->f_3 = ff_p_mult(state->f_3, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_4, state->f_4);
  t4 = ff_p_mult(t2, t2);
  state->f_4 = ff_p_mult(state->f_4, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_5, state->f_5);
  t4 = ff_p_mult(t2, t2);
  state->f_5 = ff_p_mult(state->f_5, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_6, state->f_6);
  t4 = ff_p_mult(t2, t2);
  state->f_6 = ff_p_mult(state->f_6, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_7, state->f_7);
  t4 = ff_p_mult(t2, t2);
  state->f_7 = ff_p_mult(state->f_7, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_8, state->f_8);
  t4 = ff_p_mult(t2, t2);
  state->f_8 = ff_p_mult(state->f_8, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_9, state->f_9);
  t4 = ff_p_mult(t2, t2);
  state->f_9 = ff_p_mult(state->f_9, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_a, state->f_a);
  t4 = ff_p_mult(t2, t2);
  state->f_a = ff_p_mult(state->f_a, ff_p_mult(t2, t4));

  t2 = ff_p_mult(state->f_b, state->f_b);
  t4 = ff_p_mult(t2, t2);
  state->f_b = ff_p_mult(state->f_b, ff_p_mult(t2, t4));
}

uint64_t
element_wise_accumulation(rescue_prime_state_t* state_a,
                          rescue_prime_state_t state_b)
{
  uint64_t res = 0ul;

  res = ff_p_add(res, ff_p_mult(state_a->f_0, state_b.f_0));
  res = ff_p_add(res, ff_p_mult(state_a->f_1, state_b.f_1));
  res = ff_p_add(res, ff_p_mult(state_a->f_2, state_b.f_2));
  res = ff_p_add(res, ff_p_mult(state_a->f_3, state_b.f_3));
  res = ff_p_add(res, ff_p_mult(state_a->f_4, state_b.f_4));
  res = ff_p_add(res, ff_p_mult(state_a->f_5, state_b.f_5));
  res = ff_p_add(res, ff_p_mult(state_a->f_6, state_b.f_6));
  res = ff_p_add(res, ff_p_mult(state_a->f_7, state_b.f_7));
  res = ff_p_add(res, ff_p_mult(state_a->f_8, state_b.f_8));
  res = ff_p_add(res, ff_p_mult(state_a->f_9, state_b.f_9));
  res = ff_p_add(res, ff_p_mult(state_a->f_a, state_b.f_a));
  res = ff_p_add(res, ff_p_mult(state_a->f_b, state_b.f_b));

  return res;
}

void
apply_mds(rescue_prime_state_t* state)
{
  rescue_prime_state_t res;

  res.f_0 = element_wise_accumulation(state, MDS[0]);
  res.f_1 = element_wise_accumulation(state, MDS[1]);
  res.f_2 = element_wise_accumulation(state, MDS[2]);
  res.f_3 = element_wise_accumulation(state, MDS[3]);
  res.f_4 = element_wise_accumulation(state, MDS[4]);
  res.f_5 = element_wise_accumulation(state, MDS[5]);
  res.f_6 = element_wise_accumulation(state, MDS[6]);
  res.f_7 = element_wise_accumulation(state, MDS[7]);
  res.f_8 = element_wise_accumulation(state, MDS[8]);
  res.f_9 = element_wise_accumulation(state, MDS[9]);
  res.f_a = element_wise_accumulation(state, MDS[10]);
  res.f_b = element_wise_accumulation(state, MDS[11]);

  state->f_0 = res.f_0;
  state->f_1 = res.f_1;
  state->f_2 = res.f_2;
  state->f_3 = res.f_3;
  state->f_4 = res.f_4;
  state->f_5 = res.f_5;
  state->f_6 = res.f_6;
  state->f_7 = res.f_7;
  state->f_8 = res.f_8;
  state->f_9 = res.f_9;
  state->f_a = res.f_a;
  state->f_b = res.f_b;
}

void
apply_constants(rescue_prime_state_t* state, rescue_prime_state_t ark)
{
  state->f_0 = ff_p_add(state->f_0, ark.f_0);
  state->f_1 = ff_p_add(state->f_1, ark.f_1);
  state->f_2 = ff_p_add(state->f_2, ark.f_2);
  state->f_3 = ff_p_add(state->f_3, ark.f_3);
  state->f_4 = ff_p_add(state->f_4, ark.f_4);
  state->f_5 = ff_p_add(state->f_5, ark.f_5);
  state->f_6 = ff_p_add(state->f_6, ark.f_6);
  state->f_7 = ff_p_add(state->f_7, ark.f_7);
  state->f_8 = ff_p_add(state->f_8, ark.f_8);
  state->f_9 = ff_p_add(state->f_9, ark.f_9);
  state->f_a = ff_p_add(state->f_a, ark.f_a);
  state->f_b = ff_p_add(state->f_b, ark.f_b);
}

void
element_wise_multiplication(rescue_prime_state_t* state_src_a,
                            rescue_prime_state_t* state_src_b,
                            rescue_prime_state_t* state_dst)
{
  state_dst->f_0 = ff_p_mult(state_src_a->f_0, state_src_b->f_0);
  state_dst->f_1 = ff_p_mult(state_src_a->f_1, state_src_b->f_1);
  state_dst->f_2 = ff_p_mult(state_src_a->f_2, state_src_b->f_2);
  state_dst->f_3 = ff_p_mult(state_src_a->f_3, state_src_b->f_3);

  state_dst->f_4 = ff_p_mult(state_src_a->f_4, state_src_b->f_4);
  state_dst->f_5 = ff_p_mult(state_src_a->f_5, state_src_b->f_5);
  state_dst->f_6 = ff_p_mult(state_src_a->f_6, state_src_b->f_6);
  state_dst->f_7 = ff_p_mult(state_src_a->f_7, state_src_b->f_7);

  state_dst->f_8 = ff_p_mult(state_src_a->f_8, state_src_b->f_8);
  state_dst->f_9 = ff_p_mult(state_src_a->f_9, state_src_b->f_9);
  state_dst->f_a = ff_p_mult(state_src_a->f_a, state_src_b->f_a);
  state_dst->f_b = ff_p_mult(state_src_a->f_b, state_src_b->f_b);
}

void
apply_inv_sbox(rescue_prime_state_t* state)
{
  rescue_prime_state_t t1;
  element_wise_multiplication(state, state, &t1);

  rescue_prime_state_t t2;
  element_wise_multiplication(&t1, &t1, &t2);

  rescue_prime_state_t t3;
  exp_acc(3, &t2, &t2, &t3);

  rescue_prime_state_t t4;
  exp_acc(6, &t3, &t3, &t4);

  rescue_prime_state_t tmp;
  exp_acc(12, &t4, &t4, &tmp);

  rescue_prime_state_t t5;
  exp_acc(6, &tmp, &t3, &t5);

  rescue_prime_state_t t6;
  exp_acc(31, &t5, &t5, &t6);

  // declare one time, use same registers multiple times
  uint64_t a = 0ul, b = 0ul;

  a = ff_p_mult(t5.f_0, ff_p_mult(t6.f_0, t6.f_0));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_0, ff_p_mult(t1.f_0, t2.f_0));
  state->f_0 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_1, ff_p_mult(t6.f_1, t6.f_1));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_1, ff_p_mult(t1.f_1, t2.f_1));
  state->f_1 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_2, ff_p_mult(t6.f_2, t6.f_2));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_2, ff_p_mult(t1.f_2, t2.f_2));
  state->f_2 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_3, ff_p_mult(t6.f_3, t6.f_3));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_3, ff_p_mult(t1.f_3, t2.f_3));
  state->f_3 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_4, ff_p_mult(t6.f_4, t6.f_4));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_4, ff_p_mult(t1.f_4, t2.f_4));
  state->f_4 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_5, ff_p_mult(t6.f_5, t6.f_5));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_5, ff_p_mult(t1.f_5, t2.f_5));
  state->f_5 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_6, ff_p_mult(t6.f_6, t6.f_6));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_6, ff_p_mult(t1.f_6, t2.f_6));
  state->f_6 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_7, ff_p_mult(t6.f_7, t6.f_7));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_7, ff_p_mult(t1.f_7, t2.f_7));
  state->f_7 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_8, ff_p_mult(t6.f_8, t6.f_8));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_8, ff_p_mult(t1.f_8, t2.f_8));
  state->f_8 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_9, ff_p_mult(t6.f_9, t6.f_9));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_9, ff_p_mult(t1.f_9, t2.f_9));
  state->f_9 = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_a, ff_p_mult(t6.f_a, t6.f_a));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_a, ff_p_mult(t1.f_a, t2.f_a));
  state->f_a = ff_p_mult(a, b);

  a = ff_p_mult(t5.f_b, ff_p_mult(t6.f_b, t6.f_b));
  a = ff_p_mult(a, a);
  a = ff_p_mult(a, a);
  b = ff_p_mult(state->f_b, ff_p_mult(t1.f_b, t2.f_b));
  state->f_b = ff_p_mult(a, b);
}

void
exp_acc(const uint64_t m,
        rescue_prime_state_t* base,
        rescue_prime_state_t* tail,
        rescue_prime_state_t* res)
{
  for (uint64_t i = 0; i < m; i++) {
    if (i == 0) {
      element_wise_multiplication(base, base, res);
    } else {
      element_wise_multiplication(res, res, res);
    }
  }

  element_wise_multiplication(res, tail, res);
}
