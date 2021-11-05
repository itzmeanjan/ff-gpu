#include "test_rescue_prime.hpp"

void test_alphas(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());

  uint64_t e = next_random(gen);
  uint64_t e_exp = operate(q, e, ALPHA, Op::power);

  assert(e == operate(q, e_exp, INV_ALPHA, Op::power));
}
