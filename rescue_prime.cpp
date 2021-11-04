#include "rescue_prime.hpp"

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
  }
  state = res;
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
}

uint64_t *exp_acc(const uint64_t m, const uint64_t *base,
                  const uint64_t *tail) {
  uint64_t res[STATE_WIDTH] = {0};
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < STATE_WIDTH; j++) {
      if (i == 0) {
        res[j] = ff_p_mult(*(base + j), *(base + j));
      } else {
        res[j] = ff_p_mult(res[j], res[j]);
      }
    }
  }

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    res[i] = ff_p_mult(res[i], *(tail + i));
  }
  return res;
}
