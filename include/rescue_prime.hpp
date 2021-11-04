#include "ff_p.hpp"

inline constexpr uint64_t STATE_WIDTH = 12;
inline constexpr uint64_t RATE_WIDTH = 8;
inline constexpr uint64_t DIGEST_SIZE = 4;
inline constexpr uint64_t NUM_ROUNDS = 7;

extern SYCL_EXTERNAL void apply_permutation(uint64_t *const state);

extern SYCL_EXTERNAL void apply_round(uint64_t *const state,
                                      const uint64_t round);

extern SYCL_EXTERNAL void apply_sbox(uint64_t *const state);

extern SYCL_EXTERNAL void apply_mds(uint64_t *const state);

extern SYCL_EXTERNAL void apply_constants(uint64_t *const state,
                                          const uint64_t *ark);

extern SYCL_EXTERNAL void apply_inv_sbox(uint64_t *const state);
