#pragma once
#include "ff_p.hpp"
#include <random>

enum Op
{
  add,
  sub,
  mult,
  power,
  inverse,
  division
};

// generic function to operate on field element operands,
// specified using operator enumeration; computation offloaded to device
//
// returned result will ∈ F_p
uint64_t
operate(sycl::queue& q, uint64_t operand_1, uint64_t operand_2, Op op);

void
test_addition(sycl::queue& q);

void
test_subtraction(sycl::queue& q);

void
test_multiplication(sycl::queue& q);

void
test_power(sycl::queue& q);

void
test_inversion(sycl::queue& q);
