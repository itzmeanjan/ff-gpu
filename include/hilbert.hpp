#pragma once
#include <CL/sycl.hpp>

// computes a hilbert matrix of dimension (dim x dim)
// where each element is a finite field (binary extension) element,
// which are computed using (1 / (i + j + 1)) while following field
// rules, given i = row index, j = column index
//
// used for benchmarking binary field arithmetics
void gen_hilbert_matrix_ff(sycl::queue &q, uint32_t *const mat, const uint dim,
                           const uint wg_size);

// computes a hilbert matrix of dimension (dim x dim)
// where each element is a prime field element, which are
// computed using (1 / (i + j + 1)) while following field
// rules, given i = row index, j = column index
//
// used for benchmarking prime field arithmetics
void gen_hilbert_matrix_ff_p(sycl::queue &q, uint32_t *const mat,
                             const uint dim, const uint wg_size);
