#include "hilbert.hpp"
#include <chrono>

using namespace sycl;

const uint32_t N = 1 << 10;
const uint32_t B = 1 << 5;

int main(int argc, char **argv) {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  uint32_t *mat = (uint32_t *)malloc(sizeof(uint32_t) * N * N);

  std::chrono::_V2::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  gen_hilbert_matrix(q, mat, N, B);
  std::chrono::_V2::steady_clock::time_point end =
      std::chrono::steady_clock::now();

  int64_t tm =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << N << "x" << N << " hilbert matrix with F(2 ** 32) elements, in "
            << tm << " ms" << std::endl;

  std::free(mat);

  return 0;
}
