#include "test_ntt.hpp"

sycl::event compute_matrix_matrix_multiplication(
    sycl::queue &q, buf_2d_u64_t &mat_a, buf_2d_u64_t &mat_b,
    buf_2d_u64_t &mat_c, const uint64_t dim, const uint64_t wg_size) {
  sycl::event evt = q.submit([&](sycl::handler &h) {
    buf_2d_u64_rd_t acc_mat_a{mat_a, h};
    buf_2d_u64_rd_t acc_mat_b{mat_b, h};
    buf_2d_u64_rw_t acc_mat_c{mat_c, h, sycl::no_init};

    h.parallel_for<class kernelComputeDFTMatrixMatrixMultipication>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          sycl::sub_group sg = it.get_sub_group();
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t sum = 0ul;
          for (uint64_t i = 0; i < dim; i++) {
            sum = ff_p_add(sum,
                           ff_p_mult(sycl::group_broadcast(sg, acc_mat_a[r][i]),
                                     acc_mat_b[i][c]));
          }
          acc_mat_c[r][c] = sum;
        });
  });
  return evt;
}

void check_correctness(sycl::queue &q, const uint64_t dim,
                       const uint64_t wg_size) {
  assert(dim & (dim - 1ul) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t omega = 0;
  uint64_t omega_inv = 0ul;
  buf_1d_u64_t buf_omega{&omega, sycl::range<1>{1}};
  buf_1d_u64_t buf_omega_inv{&omega_inv, sycl::range<1>{1}};

  compute_omega(q, buf_omega, log_2_dim);
  compute_omega_inv(q, buf_omega_inv, log_2_dim);

  uint64_t *mat_a =
      static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));
  uint64_t *mat_b =
      static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));
  buf_2d_u64_t buf_mat_a{mat_a, sycl::range<2>{dim, dim}};
  buf_2d_u64_t buf_mat_b{mat_b, sycl::range<2>{dim, dim}};

  compute_dft_matrix(q, buf_mat_a, buf_omega, dim, wg_size);
  compute_dft_matrix(q, buf_mat_b, buf_omega_inv, dim, wg_size);

  uint64_t *mat_c =
      static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));
  buf_2d_u64_t buf_mat_c{mat_c, sycl::range<2>{dim, dim}};

  sycl::event evt = compute_matrix_matrix_multiplication(
      q, buf_mat_a, buf_mat_b, buf_mat_c, dim, wg_size);
  evt.wait();
}
