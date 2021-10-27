#include <CL/sycl.hpp>

// computes a hilbert matrix of dimension (dim x dim)
// where each element is a finite field element, which are
// computed using (1 / (i + j + 1)) while following field
// rules, given i = row index, j = column index
void gen_hilbert_matrix(sycl::queue &q, uint32_t *const mat, const uint dim,
                        const uint wg_size);
