#include "test_ff.hpp"
#include <iostream>

int
main()
{
  test_ff_add();
  test_ff_sub();
  test_ff_mul();
  test_ff_exp();
  test_ff_inv();

  std::cout << "[test] Finite Field Arithmetics" << std::endl;

  return EXIT_SUCCESS;
}
