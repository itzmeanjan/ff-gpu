#include "test_rescue_prime.hpp"

void test_alphas(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());

  uint64_t e = next_random(gen);
  uint64_t e_exp = operate(q, e, ALPHA, Op::power);

  assert(e == operate(q, e_exp, INV_ALPHA, Op::power));
}

void test_sbox(sycl::queue &q) {
  uint64_t *arr_0 =
      (uint64_t *)sycl::malloc_shared(sizeof(uint64_t) * STATE_WIDTH, q);
  uint64_t *arr_1 =
      (uint64_t *)sycl::malloc_shared(sizeof(uint64_t) * STATE_WIDTH, q);

  random_array(arr_0, STATE_WIDTH);
  q.memcpy(arr_1, arr_0, sizeof(uint64_t) * STATE_WIDTH).wait();
  q.single_task([=]() { apply_sbox(arr_0); });

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    *(arr_1 + i) = operate(q, *(arr_1 + i), ALPHA, Op::power);
  }

  q.wait();

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    assert(*(arr_0 + i) == *(arr_1 + i));
  }
}

void random_array(uint64_t *const arr, const uint64_t count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

  for (uint64_t i = 0; i < count; i++) {
    *(arr + i) = dis(gen) % MOD;
  }
}

void test_inv_sbox(sycl::queue &q) {
  uint64_t *arr_0 =
      (uint64_t *)sycl::malloc_shared(sizeof(uint64_t) * STATE_WIDTH, q);
  uint64_t *arr_1 =
      (uint64_t *)sycl::malloc_shared(sizeof(uint64_t) * STATE_WIDTH, q);

  random_array(arr_0, STATE_WIDTH);
  q.memcpy(arr_1, arr_0, sizeof(uint64_t) * STATE_WIDTH).wait();
  q.single_task([=]() { apply_inv_sbox(arr_0); });

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    *(arr_1 + i) = operate(q, *(arr_1 + i), INV_ALPHA, Op::power);
  }

  q.wait();

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    assert(*(arr_0 + i) == *(arr_1 + i));
  }
}

void test_permutation(sycl::queue &q) {
  uint64_t *state =
      (uint64_t *)sycl::malloc_shared(sizeof(uint64_t) * STATE_WIDTH, q);
  uint64_t expected[STATE_WIDTH] = {
      10809974140050983728ull % MOD, 6938491977181280539ull % MOD,
      8834525837561071698ull % MOD,  6854417192438540779ull % MOD,
      4476630872663101667ull % MOD,  6292749486700362097ull % MOD,
      18386622366690620454ull % MOD, 10614098972800193173ull % MOD,
      7543273285584849722ull % MOD,  9490898458612615694ull % MOD,
      9030271581669113292ull % MOD,  10101107035874348250ull % MOD,
  };

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    *(state + i) = i;
  }

  q.single_task([=]() { apply_permutation(state); }).wait();

  for (uint64_t i = 0; i < STATE_WIDTH; i++) {
    assert(*(state + i) == expected[i]);
  }
}
