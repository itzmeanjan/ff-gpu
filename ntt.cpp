#include "ntt.hpp"

uint64_t get_root_of_unity(uint64_t n) {
  uint64_t power = 1ul << (TWO_ADICITY - n);
  return ff_p_pow(TWO_ADIC_ROOT_OF_UNITY, power);
}

sycl::event compute_omega(sycl::queue &q, buf_1d_u64_t &omega,
                          const uint64_t n) {
  sycl::event evt = q.submit([&](sycl::handler &h) {
    buf_1d_u64_wr_t acc_omega{omega, h, sycl::no_init};

    q.single_task([=]() { acc_omega[0] = get_root_of_unity(n); });
  });
  return evt;
}

sycl::event compute_omega_inv(sycl::queue &q, buf_1d_u64_t &omega_inv,
                              const uint64_t n) {
  sycl::event evt = q.submit([&](sycl::handler &h) {
    buf_1d_u64_wr_t acc_omega_inv{omega_inv, h, sycl::no_init};

    q.single_task([=]() { acc_omega_inv[0] = ff_p_inv(get_root_of_unity(n)); });
  });
  return evt;
}

void compute_dft_matrix(sycl::queue &q, buf_2d_u64_t &mat, buf_1d_u64_t &omega,
                        const uint64_t dim, const uint64_t wg_size) {
  q.submit([&](sycl::handler &h) {
    buf_2d_u64_wr_t acc_mat{mat, h, sycl::no_init};
    buf_1d_u64_rd_t acc_omega{omega, h};
    sycl::accessor<uint64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        loc_acc_omega{sycl::range<1>{1}, h};

    h.parallel_for<class kernelPowerSeriesOfOmega>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          sycl::sub_group sg = it.get_sub_group();
          const uint64_t c = it.get_global_id(0);

          if (sycl::ext::oneapi::leader(sg)) {
            loc_acc_omega[0] = acc_omega[0];
          }

          it.barrier(sycl::access::fence_space::local_space);

          acc_mat[0][c] = 1ul;
          acc_mat[1][c] = ff_p_pow(loc_acc_omega[0], c);
        });
  });

  q.submit([&](sycl::handler &h) {
    buf_2d_u64_rw_t acc_mat{mat, h};

    h.parallel_for<class kernelComputeDFTMatrix>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          if (r < 2) {
            return;
          }

          acc_mat[r][c] = c == 0 ? 1ul : ff_p_pow(acc_mat[1][c], r);
        });
  });
}

void compute_matrix_vector_multiplication(sycl::queue &q, buf_2d_u64_t &mat,
                                          buf_1d_u64_t &vec, buf_1d_u64_t &res,
                                          const uint64_t dim,
                                          const uint64_t wg_size) {
  q.submit([&](sycl::handler &h) {
    buf_2d_u64_rd_t acc_mat{mat, h};
    buf_1d_u64_rd_t acc_vec{vec, h};
    buf_1d_u64_rw_t acc_res{res, h, sycl::no_init};

    h.parallel_for<class kernelComputeDFTMatrixVectorMultipication>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          sycl::sub_group sg = it.get_sub_group();
          const uint64_t r = it.get_global_id(0);

          uint64_t sum = 0ul;
          for (uint64_t c = 0; c < dim; c++) {
            sum =
                ff_p_add(sum, ff_p_mult(acc_mat[r][c],
                                        sycl::group_broadcast(sg, acc_vec[c])));
          }
          acc_res[r] = sum;
        });
  });
}

void forward_transform(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size) {
  // size of input vector must be power of two !
  assert(dim & (dim - 1ul) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  // order can't exceed 2 ** 32 and can't also
  // find root of unity for n = 0
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t omega = 0ul;
  buf_1d_u64_t buf_omega{&omega, sycl::range<1>{1}};

  compute_omega(q, buf_omega, log_2_dim);

  uint64_t *mat = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));
  buf_2d_u64_t buf_mat{mat, sycl::range<2>{dim, dim}};

  compute_dft_matrix(q, buf_mat, buf_omega, dim, wg_size);
  compute_matrix_vector_multiplication(q, buf_mat, vec, res, dim, wg_size);

  q.wait();
}

void compute_matrix_matrix_multiplication(sycl::queue &q, buf_2d_u64_t &mat_a,
                                          buf_2d_u64_t &mat_b,
                                          buf_2d_u64_t &mat_c,
                                          const uint64_t dim,
                                          const uint64_t wg_size) {
  q.submit([&](sycl::handler &h) {
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
}
