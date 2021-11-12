#include "test.hpp"
#include "test_ntt.hpp"
#include "test_rescue_prime.hpp"

using namespace sycl;

int main(int argc, char **argv) {
  device d{default_selector{}};
  queue q{d};

  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  test_addition(q);
  std::cout << "✅ passed addition tests" << std::endl;
  test_subtraction(q);
  std::cout << "✅ passed subtraction tests" << std::endl;
  test_multiplication(q);
  std::cout << "✅ passed multiplication tests" << std::endl;
  test_power(q);
  std::cout << "✅ passed exponentiation tests" << std::endl;
  test_inversion(q);
  std::cout << "✅ passed inversion tests" << std::endl;
  test_alphas(q);
  test_sbox(q);
  test_inv_sbox(q);
  test_permutation(q);
  std::cout << "✅ passed rescue prime tests" << std::endl;
  check_ntt_correctness(q, 1 << 10, 1 << 6);
  std::cout << "✅ passed NTT correctness test" << std::endl;

  return 0;
}
