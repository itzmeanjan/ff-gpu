#include "utils.hpp"

void show_matrix(const uint32_t *mat, const uint32_t dim) {
  for (uint i = 0; i < dim; i++) {
    for (uint j = 0; j < dim; j++) {
      std::cout << mat[i * dim + j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
