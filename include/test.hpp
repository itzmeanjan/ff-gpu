#include "ff_p.hpp"
#include <random>

enum Op { add, sub, mult, power, inverse, division };

// generic function to operate on field element operands,
// specified using operator enumeration; computation offloaded to device
//
// returned result will ∈ F_p
uint64_t operate(sycl::queue &q, uint64_t operand_1, uint64_t operand_2, Op op);

// generate next random uint64 value using
// provided engine & randomization source
uint64_t next_random(std::mt19937 gen);

void test_addition(sycl::queue &q);
