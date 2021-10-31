#include "ff_p.hpp"
#include <random>

enum Op { add, sub, mult, pow, inv, div };

// generic function to operate on field element operands,
// specified using operator enumeration; computation offloaded to device
//
// returned result will ∈ F_p
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
        case pow:
          acc[0] = ff_p_pow(operand_1, operand_2);
          break;
        case inv:
          acc[0] = ff_p_inv(operand_1);
          break;
        case div:
          acc[0] = ff_p_div(operand_1, operand_2);
          break;
        }
      });
    });
    q.wait();
  }

  return res % MOD;
}

// generate next random uint64 value using
// provided engine & randomization source
uint64_t next_random(std::mt19937 gen) {
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
  return dis(gen);
}
