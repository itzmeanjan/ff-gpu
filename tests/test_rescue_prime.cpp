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
