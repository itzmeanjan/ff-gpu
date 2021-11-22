#include "test_rescue_prime.hpp"

void test_alphas(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

  uint64_t e = dis(gen);
  uint64_t e_exp = operate(q, e, ALPHA, Op::power);

  assert(e == operate(q, e_exp, INV_ALPHA, Op::power));
}

void test_sbox(sycl::queue &q) {
  rescue_prime_state_t *arr_0 = (rescue_prime_state_t *)sycl::malloc_shared(
      sizeof(rescue_prime_state_t), q);
  rescue_prime_state_t *arr_1 = (rescue_prime_state_t *)sycl::malloc_shared(
      sizeof(rescue_prime_state_t), q);

  random_rescue_prime_state(arr_0);
  q.memcpy(arr_1, arr_0, sizeof(rescue_prime_state_t)).wait();
  q.single_task([=]() { apply_sbox(arr_0); });

  arr_1->f_0 = operate(q, arr_1->f_0, ALPHA, Op::power);
  arr_1->f_1 = operate(q, arr_1->f_1, ALPHA, Op::power);
  arr_1->f_2 = operate(q, arr_1->f_2, ALPHA, Op::power);
  arr_1->f_3 = operate(q, arr_1->f_3, ALPHA, Op::power);
  arr_1->f_4 = operate(q, arr_1->f_4, ALPHA, Op::power);
  arr_1->f_5 = operate(q, arr_1->f_5, ALPHA, Op::power);
  arr_1->f_6 = operate(q, arr_1->f_6, ALPHA, Op::power);
  arr_1->f_7 = operate(q, arr_1->f_7, ALPHA, Op::power);
  arr_1->f_8 = operate(q, arr_1->f_8, ALPHA, Op::power);
  arr_1->f_9 = operate(q, arr_1->f_9, ALPHA, Op::power);
  arr_1->f_a = operate(q, arr_1->f_a, ALPHA, Op::power);
  arr_1->f_b = operate(q, arr_1->f_b, ALPHA, Op::power);

  q.wait();

  assert(arr_0->f_0 == arr_1->f_0);
  assert(arr_0->f_1 == arr_1->f_1);
  assert(arr_0->f_2 == arr_1->f_2);
  assert(arr_0->f_3 == arr_1->f_3);
  assert(arr_0->f_4 == arr_1->f_4);
  assert(arr_0->f_5 == arr_1->f_5);
  assert(arr_0->f_6 == arr_1->f_6);
  assert(arr_0->f_7 == arr_1->f_7);
  assert(arr_0->f_8 == arr_1->f_8);
  assert(arr_0->f_9 == arr_1->f_9);
  assert(arr_0->f_a == arr_1->f_a);
  assert(arr_0->f_b == arr_1->f_b);
}

void random_rescue_prime_state(rescue_prime_state_t *arr) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

  arr->f_0 = dis(gen) % MOD;
  arr->f_1 = dis(gen) % MOD;
  arr->f_2 = dis(gen) % MOD;
  arr->f_3 = dis(gen) % MOD;

  arr->f_4 = dis(gen) % MOD;
  arr->f_5 = dis(gen) % MOD;
  arr->f_6 = dis(gen) % MOD;
  arr->f_7 = dis(gen) % MOD;

  arr->f_8 = dis(gen) % MOD;
  arr->f_9 = dis(gen) % MOD;
  arr->f_a = dis(gen) % MOD;
  arr->f_b = dis(gen) % MOD;
}

void test_inv_sbox(sycl::queue &q) {
  rescue_prime_state_t *arr_0 = (rescue_prime_state_t *)sycl::malloc_shared(
      sizeof(rescue_prime_state_t), q);
  rescue_prime_state_t *arr_1 = (rescue_prime_state_t *)sycl::malloc_shared(
      sizeof(rescue_prime_state_t), q);

  random_rescue_prime_state(arr_0);
  q.memcpy(arr_1, arr_0, sizeof(rescue_prime_state_t)).wait();
  q.single_task([=]() { apply_inv_sbox(arr_0); });

  arr_1->f_0 = operate(q, arr_1->f_0, INV_ALPHA, Op::power);
  arr_1->f_1 = operate(q, arr_1->f_1, INV_ALPHA, Op::power);
  arr_1->f_2 = operate(q, arr_1->f_2, INV_ALPHA, Op::power);
  arr_1->f_3 = operate(q, arr_1->f_3, INV_ALPHA, Op::power);
  arr_1->f_4 = operate(q, arr_1->f_4, INV_ALPHA, Op::power);
  arr_1->f_5 = operate(q, arr_1->f_5, INV_ALPHA, Op::power);
  arr_1->f_6 = operate(q, arr_1->f_6, INV_ALPHA, Op::power);
  arr_1->f_7 = operate(q, arr_1->f_7, INV_ALPHA, Op::power);
  arr_1->f_8 = operate(q, arr_1->f_8, INV_ALPHA, Op::power);
  arr_1->f_9 = operate(q, arr_1->f_9, INV_ALPHA, Op::power);
  arr_1->f_a = operate(q, arr_1->f_a, INV_ALPHA, Op::power);
  arr_1->f_b = operate(q, arr_1->f_b, INV_ALPHA, Op::power);

  q.wait();

  assert(arr_0->f_0 == arr_1->f_0);
  assert(arr_0->f_1 == arr_1->f_1);
  assert(arr_0->f_2 == arr_1->f_2);
  assert(arr_0->f_3 == arr_1->f_3);
  assert(arr_0->f_4 == arr_1->f_4);
  assert(arr_0->f_5 == arr_1->f_5);
  assert(arr_0->f_6 == arr_1->f_6);
  assert(arr_0->f_7 == arr_1->f_7);
  assert(arr_0->f_8 == arr_1->f_8);
  assert(arr_0->f_9 == arr_1->f_9);
  assert(arr_0->f_a == arr_1->f_a);
  assert(arr_0->f_b == arr_1->f_b);
}

void test_permutation(sycl::queue &q) {
  rescue_prime_state_t *state = (rescue_prime_state_t *)sycl::malloc_shared(
      sizeof(rescue_prime_state_t), q);
  rescue_prime_state_t expected = {
      10809974140050983728ull % MOD, 6938491977181280539ull % MOD,
      8834525837561071698ull % MOD,  6854417192438540779ull % MOD,
      4476630872663101667ull % MOD,  6292749486700362097ull % MOD,
      18386622366690620454ull % MOD, 10614098972800193173ull % MOD,
      7543273285584849722ull % MOD,  9490898458612615694ull % MOD,
      9030271581669113292ull % MOD,  10101107035874348250ull % MOD,
  };

  state->f_0 = 0;
  state->f_1 = 1;
  state->f_2 = 2;
  state->f_3 = 3;
  state->f_4 = 4;
  state->f_5 = 5;
  state->f_6 = 6;
  state->f_7 = 7;
  state->f_8 = 8;
  state->f_9 = 9;
  state->f_a = 10;
  state->f_b = 11;

  q.single_task([=]() { apply_permutation(state); }).wait();

  assert(state->f_0 == expected.f_0);
  assert(state->f_1 == expected.f_1);
  assert(state->f_2 == expected.f_2);
  assert(state->f_3 == expected.f_3);
  assert(state->f_4 == expected.f_4);
  assert(state->f_5 == expected.f_5);
  assert(state->f_6 == expected.f_6);
  assert(state->f_7 == expected.f_7);
  assert(state->f_8 == expected.f_8);
  assert(state->f_9 == expected.f_9);
  assert(state->f_a == expected.f_a);
  assert(state->f_b == expected.f_b);
}
