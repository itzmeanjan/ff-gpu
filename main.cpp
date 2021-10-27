#include "hilbert.hpp"
#include "utils.hpp"

using namespace sycl;

const uint32_t N = 4;
const uint32_t B = 4;

int main(int argc, char **argv) {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  uint32_t *mat = (uint32_t *)malloc(sizeof(uint32_t) * N * N);
  gen_hilbert_matrix(q, mat, N, B);
  show_matrix(mat, N);

  std::free(mat);

  return 0;
}
