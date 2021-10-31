#include "test.hpp"

uint64_t operate(sycl::queue &q, uint64_t operand_1, uint64_t operand_2,
                 Op op) {
  uint64_t res = 0;

  {
    sycl::buffer<uint64_t, 1> buf{&res, sycl::range<1>{1}};

    q.submit([&](sycl::handler &h) {
      sycl::accessor<uint64_t, 1, sycl::access::mode::write,
                     sycl::access::target::global_buffer>
          acc{buf, h};

      h.single_task([=]() {
        switch (op) {
        case add:
          acc[0] = ff_p_add(operand_1, operand_2);
          break;
        case sub:
          acc[0] = ff_p_sub(operand_1, operand_2);
          break;
        case mult:
          acc[0] = ff_p_mult(operand_1, operand_2);
          break;
        case power:
          acc[0] = ff_p_pow(operand_1, operand_2);
          break;
        case inverse:
          acc[0] = ff_p_inv(operand_1);
          break;
        case division:
          acc[0] = ff_p_div(operand_1, operand_2);
          break;
        }
      });
    });
    q.wait();
  }

  return res % MOD;
}

uint64_t next_random(std::mt19937 gen) {
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
  return dis(gen);
}

void test_addition(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());

  uint64_t zero = 0;
  uint64_t one = 1;
  uint64_t two = 2;
  uint64_t r = next_random(gen);
  uint64_t t_1 = MOD - one;
  uint64_t t_2 = 4294967294;

  assert(r == operate(q, zero, r, Op::add));
  assert(5 == operate(q, 3, two, Op::add));
  assert(zero == operate(q, t_1, one, Op::add));
  assert(one == operate(q, t_1, two, Op::add));
  assert(t_2 == operate(q, t_1, 0xffffffff, Op::add));
}
