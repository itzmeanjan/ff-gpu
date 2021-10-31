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

void test_subtraction(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());

  uint64_t zero = 0;
  uint64_t two = 2;
  uint64_t r = next_random(gen);
  uint64_t t = MOD - two;

  assert(r == operate(q, r, zero, Op::sub));
  assert(two == operate(q, 5, 3, Op::sub));
  assert(t == operate(q, 3, 5, Op::sub));
}

uint64_t multiply_with_addition(sycl::queue &q, const uint64_t a,
                                const uint64_t b) {
  uint64_t a_ = 0;
  for (uint64_t i = 0; i < b; i++) {
    a_ = operate(q, a, a_, Op::add);
  }
  return a_;
}

void test_multiplication(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());

  uint64_t r = next_random(gen);
  uint64_t t = MOD - 1;
  uint64_t v = (MOD + 1) / 2;

  assert(0 == operate(q, r, 0, Op::mult));
  assert(r == operate(q, r, 1, Op::mult));
  assert(15 == operate(q, 3, 5, Op::mult));
  assert(1 == operate(q, t, t, Op::mult));
  assert(MOD - 2 == operate(q, t, 2, Op::mult));
  assert(MOD - 4 == operate(q, t, 4, Op::mult));
  assert(1 == operate(q, v, 2, Op::mult));
  assert(operate(q, r, 2, Op::mult) == operate(q, r, r, Op::add));

  // test whether multiplication & addition produces
  // same results by performing n * a == (0 + n-times ... + a)
  uint64_t rounds = 1 << 10;
  for (uint64_t i = 0; i < rounds; i++) {
    r = next_random(gen);
    assert(multiply_with_addition(q, r, i) == operate(q, r, i, Op::mult));
  }
}

void test_power(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());

  uint64_t r = next_random(gen);

  assert(operate(q, 0, 0, Op::power) == 1);
  assert(operate(q, 0, 1, Op::power) == 0);
  assert(operate(q, 1, 0, Op::power) == 1);
  assert(operate(q, 1, 1, Op::power) == 1);
  assert(operate(q, 1, 2, Op::power) == 1);
  assert(operate(q, r, 3, Op::power) ==
         operate(q, r, operate(q, r, r, Op::mult), Op::mult));
}

void test_inversion(sycl::queue &q) {
  std::random_device rd;
  std::mt19937 gen(rd());

  uint64_t r = next_random(gen);

  assert(1 == operate(q, 1, 0, Op::inverse));
  assert(0 == operate(q, 0, 0, Op::inverse));
  assert(operate(q, r, MOD - 2, Op::power) == operate(q, r, 0, Op::inverse));
}
