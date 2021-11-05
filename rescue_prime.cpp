#include "rescue_prime.hpp"

void hash_elements(const uint64_t *elements, const uint64_t count,
                   uint64_t *const hash) {
  uint64_t state[STATE_WIDTH] = {0};
  state[STATE_WIDTH - 1] = count >= MOD ? count - MOD : count;

  uint64_t i = 0;
  for (uint64_t j = 0; j < count; j++) {
    state[i] = ff_p_add(state[i], *(elements + j));
    i++;
    if (i % RATE_WIDTH == 0) {
      apply_permutation(state);
      i = 0;
    }
  }

  if (i > 0) {
    apply_permutation(state);
  }

  for (uint64_t i = 0; i < DIGEST_SIZE; i++) {
    *(hash + i) = state[i];
  }
}

void apply_permutation(uint64_t *const state) {
  for (uint64_t i = 0; i < NUM_ROUNDS; i++) {
    apply_round(state, i);
  }
}

void apply_round(uint64_t *const state, const uint64_t round) {
  apply_sbox(state);
  apply_mds(state);
  apply_constants(state, ARK1[round]);

  apply_inv_sbox(state);
  apply_mds(state);
  apply_constants(state, ARK2[round]);
}

void apply_sbox(uint64_t *const state) {
  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    uint64_t t2 = ff_p_mult(*(state + i), *(state + i));
    uint64_t t4 = ff_p_mult(t2, t2);

    *(state + i) = ff_p_mult(*(state + i), ff_p_mult(t2, t4));
  }
}

void apply_mds(uint64_t *state) {
  uint64_t res[STATE_WIDTH] = {0};
  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    for (uint64_t j = 0; j < STATE_WIDTH; j++) {
      *(res + i) = ff_p_add(*(res + i), ff_p_mult(MDS[i][j], *(state + j)));
    }

    *(state + i) = *(res + i);
  }
}

void apply_constants(uint64_t *const state, const uint64_t *ark) {
  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    *(state + i) = ff_p_add(*(state + i), *(ark + i));
  }
}

void apply_inv_sbox(uint64_t *const state) {
  uint64_t t1[STATE_WIDTH] = {0};
  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    t1[i] = ff_p_mult(*(state + i), *(state + i));
  }

  uint64_t t2[STATE_WIDTH] = {0};
  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    t2[i] = ff_p_mult(t1[i], t1[i]);
  }

  uint64_t t3[STATE_WIDTH] = {0};
  exp_acc(3, t2, t2, t3);

  uint64_t t4[STATE_WIDTH] = {0};
  exp_acc(6, t3, t3, t4);

  uint64_t tmp[STATE_WIDTH] = {0};
  exp_acc(12, t4, t4, tmp);

  uint64_t t5[STATE_WIDTH] = {0};
  exp_acc(6, tmp, t3, t5);

  uint64_t t6[STATE_WIDTH] = {0};
  exp_acc(31, t5, t5, t6);

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    uint64_t a = ff_p_mult(ff_p_mult(t6[i], t6[i]), t5[i]);
    a = ff_p_mult(a, a);
    a = ff_p_mult(a, a);
    uint64_t b = ff_p_mult(ff_p_mult(t1[i], t2[i]), *(state + i));

    *(state + i) = ff_p_mult(a, b);
  }
}

void exp_acc(const uint64_t m, const uint64_t *base, const uint64_t *tail,
             uint64_t *const res) {
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < STATE_WIDTH; j++) {
      if (i == 0) {
        *(res + j) = ff_p_mult(*(base + j), *(base + j));
      } else {
        *(res + j) = ff_p_mult(*(res + j), *(res + j));
      }
    }
  }

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    *(res + i) = ff_p_mult(*(res + i), *(tail + i));
  }
}
