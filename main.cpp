#include "hilbert.hpp"
#include "scalar_add.hpp"
#include <chrono>
#include <iomanip>

using namespace sycl;

const uint32_t N = 1 << 10;
const uint32_t B = 1 << 5;

typedef std::chrono::_V2::steady_clock::time_point tp;

int main(int argc, char **argv) {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << std::endl;
  std::cout << "hilbert matrix generation with F(2 ** 32) elements\n"
            << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    uint32_t *mat = (uint32_t *)malloc(sizeof(uint32_t) * dim * dim);

    tp start = std::chrono::steady_clock::now();
    gen_hilbert_matrix(q, mat, dim, B);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << tm << " us" << std::endl;

    std::free(mat);
  }

  std::cout << "\nadd subsequence of F(2 ** 32) elements\n" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    uint32_t *vec = (uint32_t *)malloc(sizeof(uint32_t) * dim);

    tp start = std::chrono::steady_clock::now();
    add_elements(q, vec, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "\t\t" << std::setw(8)
              << std::right << N << "\t\t" << std::setw(10) << std::right << tm
              << " us" << std::endl;

    std::free(vec);
  }

  return 0;
}
