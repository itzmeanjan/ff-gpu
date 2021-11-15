#include "ntt.hpp"

uint64_t get_root_of_unity(uint64_t n) {
  uint64_t power = 1ul << (TWO_ADICITY - n);
  return ff_p_pow(TWO_ADIC_ROOT_OF_UNITY, power);
}

sycl::event compute_omega(sycl::queue &q, buf_1d_u64_t &omega,
                          const uint64_t n) {
  sycl::event evt = q.submit([&](sycl::handler &h) {
    buf_1d_u64_wr_t acc_omega{omega, h, sycl::no_init};

    h.single_task([=]() { acc_omega[0] = get_root_of_unity(n); });
  });
  return evt;
}

sycl::event compute_omega_inv(sycl::queue &q, buf_1d_u64_t &omega_inv,
                              const uint64_t n) {
  sycl::event evt = q.submit([&](sycl::handler &h) {
    buf_1d_u64_wr_t acc_omega_inv{omega_inv, h, sycl::no_init};

    h.single_task([=]() { acc_omega_inv[0] = ff_p_inv(get_root_of_unity(n)); });
  });
  return evt;
}

sycl::event compute_dft_matrix(sycl::queue &q, buf_2d_u64_t &mat,
                               buf_1d_u64_t &omega, const uint64_t dim,
                               const uint64_t wg_size) {
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

  sycl::event evt = q.submit([&](sycl::handler &h) {
    buf_2d_u64_rw_t acc_mat{mat, h};

    h.parallel_for<class kernelComputeDFTMatrix>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          if (r < 2) {
            return;
          }

          acc_mat[r][c] = ff_p_pow(acc_mat[1][c], r);
        });
  });

  return evt;
}

sycl::event compute_matrix_vector_multiplication(
    sycl::queue &q, buf_2d_u64_t &mat, buf_1d_u64_t &vec, buf_1d_u64_t &res,
    const uint64_t dim, const uint64_t wg_size) {
  sycl::event evt = q.submit([&](sycl::handler &h) {
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
  return evt;
}

sycl::event compute_vector_scalar_multilication(sycl::queue &q,
                                                buf_1d_u64_t &vec,
                                                const uint64_t factor,
                                                const uint64_t dim,
                                                const uint64_t wg_size) {
  sycl::event evt = q.submit([&](sycl::handler &h) {
    buf_1d_u64_rw_t acc_vec{vec, h};

    h.parallel_for<class kernelInvDFTScalarMultiplication>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          acc_vec[r] = ff_p_mult(acc_vec[r], factor);
        });
  });
  return evt;
}

void forward_transform(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size) {
  // size of input vector must be power of two !
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  // order can't exceed 2 ** 32 and can't also
  // find root of unity for n = 0
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *omega = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *mat = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));

  // putting actual computation in block so that
  // allocated memory can be safely freed before
  // returning control back from function
  {
    buf_1d_u64_t buf_omega{omega, sycl::range<1>{1}};
    buf_2d_u64_t buf_mat{mat, sycl::range<2>{dim, dim}};

    compute_omega(q, buf_omega, log_2_dim);
    compute_dft_matrix(q, buf_mat, buf_omega, dim, wg_size);
    compute_matrix_vector_multiplication(q, buf_mat, vec, res, dim, wg_size)
        .wait();
  }

  std::free(omega);
  std::free(mat);
}

void inverse_transform(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size) {
  // size of input vector must be power of two !
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  // order can't exceed 2 ** 32 and can't also
  // find root of unity for n = 0
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *omega_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *mat = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));

  // putting actual computation in block so that
  // allocated memory can be safely freed before
  // returning control back from function
  {
    buf_1d_u64_t buf_omega_inv{omega_inv, sycl::range<1>{1}};
    buf_2d_u64_t buf_mat{mat, sycl::range<2>{dim, dim}};

    compute_omega_inv(q, buf_omega_inv, log_2_dim);
    compute_dft_matrix(q, buf_mat, buf_omega_inv, dim, wg_size);
    compute_matrix_vector_multiplication(q, buf_mat, vec, res, dim, wg_size);

    uint64_t inv_dim = 0ul;
    {
      buf_1d_u64_t buf_inv_dim{&inv_dim, sycl::range<1>{1}};

      q.submit([&](sycl::handler &h) {
         buf_1d_u64_wr_t acc_inv_dim{buf_inv_dim, h};

         h.single_task([=]() { acc_inv_dim[0] = ff_p_inv(dim); });
       }).wait();
    }

    compute_vector_scalar_multilication(q, res, inv_dim, dim, wg_size).wait();
  }

  std::free(omega_inv);
  std::free(mat);
}

uint64_t bit_rev(uint64_t v, uint64_t max_bit_width) {
  uint64_t v_rev = 0ul;
  for (uint64_t i = 0; i < max_bit_width; i++) {
    v_rev += ((v >> i) & 0b1) * (1ul << (max_bit_width - 1ul - i));
  }
  return v_rev;
}

void cooley_tukey_fft(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                      const uint64_t dim, const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *omega = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *staging = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  {
    buf_1d_u64_t buf_omega{omega, sycl::range<1>{1}};
    buf_1d_u64_t buf_staging{staging, sycl::range<1>{dim}};

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_rd_t acc_vec{vec, h};
      buf_1d_u64_wr_t acc_staging{buf_staging, h, sycl::no_init};

      h.copy(acc_vec, acc_staging);
    });

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_rd_t acc_vec{vec, h};
      buf_1d_u64_wr_t acc_res{res, h, sycl::no_init};

      h.copy(acc_vec, acc_res);
    });

    for (int64_t i = log_2_dim - 1ul; i >= 0; i--) {
      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_omega{buf_omega, h};
        buf_1d_u64_rd_t acc_res{res, h};
        buf_1d_u64_wr_t acc_staging{buf_staging, h};

        h.parallel_for<class kernelCooleyTukeyFFTMain>(
            sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
            [=](sycl::nd_item<1> it) {
              const uint64_t k = it.get_global_id(0);
              const uint64_t p = 1ul << i;
              const uint64_t q = dim / p;

              uint64_t z = ff_p_pow(acc_omega[0], p);

              if (k % p == k % (2 * p)) {
                uint64_t k_rev = bit_rev(k, log_2_dim) % q;
                uint64_t z_pow = ff_p_pow(z, k_rev);

                acc_staging[k] =
                    ff_p_add(acc_res[k], ff_p_mult(acc_res[k + p], z_pow));
                acc_staging[k + p] =
                    ff_p_sub(acc_res[k], ff_p_mult(acc_res[k + p], z_pow));
              }
            });
      });

      q.submit([&](sycl::handler &h) {
        buf_1d_u64_wr_t acc_res{res, h};
        buf_1d_u64_rd_t acc_staging{buf_staging, h};

        h.copy(acc_staging, acc_res);
      });
    }

    q.submit([&](sycl::handler &h) {
       buf_1d_u64_wr_t acc_res{res, h};
       buf_1d_u64_rd_t acc_staging{buf_staging, h};

       h.parallel_for<class kernelCooleyTukeyFFTFinalReorder>(
           sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
           [=](sycl::nd_item<1> it) {
             const uint64_t k = it.get_global_id(0);
             const uint64_t k_rev = bit_rev(k, log_2_dim);

             acc_res[k] = acc_staging[k_rev];
           });
     }).wait();
  }

  std::free(omega);
  std::free(staging);
}

void cooley_tukey_ifft(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *omega_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *dim_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *staging = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  {
    buf_1d_u64_t buf_omega_inv{omega_inv, sycl::range<1>{1}};
    buf_1d_u64_t buf_dim_inv{dim_inv, sycl::range<1>{1}};
    buf_1d_u64_t buf_staging{staging, sycl::range<1>{dim}};

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_rd_t acc_vec{vec, h};
      buf_1d_u64_wr_t acc_staging{buf_staging, h, sycl::no_init};

      h.copy(acc_vec, acc_staging);
    });

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_rd_t acc_vec{vec, h};
      buf_1d_u64_wr_t acc_res{res, h, sycl::no_init};

      h.copy(acc_vec, acc_res);
    });

    for (int64_t i = log_2_dim - 1ul; i >= 0; i--) {
      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_omega_inv{buf_omega_inv, h};
        buf_1d_u64_rd_t acc_res{res, h};
        buf_1d_u64_wr_t acc_staging{buf_staging, h};

        h.parallel_for<class kernelCooleyTukeyIFFTMain>(
            sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
            [=](sycl::nd_item<1> it) {
              const uint64_t k = it.get_global_id(0);
              const uint64_t p = 1ul << i;
              const uint64_t q = dim / p;

              uint64_t z = ff_p_pow(acc_omega_inv[0], p);

              if (k % p == k % (2 * p)) {
                uint64_t k_rev = bit_rev(k, log_2_dim) % q;
                uint64_t z_pow = ff_p_pow(z, k_rev);

                acc_staging[k] =
                    ff_p_add(acc_res[k], ff_p_mult(acc_res[k + p], z_pow));
                acc_staging[k + p] =
                    ff_p_sub(acc_res[k], ff_p_mult(acc_res[k + p], z_pow));
              }
            });
      });

      q.submit([&](sycl::handler &h) {
        buf_1d_u64_wr_t acc_res{res, h};
        buf_1d_u64_rd_t acc_staging{buf_staging, h};

        h.copy(acc_staging, acc_res);
      });
    }

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_wr_t acc_inv_dim{buf_dim_inv, h};

      h.single_task([=]() { acc_inv_dim[0] = ff_p_inv(dim); });
    });

    q.submit([&](sycl::handler &h) {
       buf_1d_u64_wr_t acc_res{res, h};
       buf_1d_u64_rd_t acc_staging{buf_staging, h};
       buf_1d_u64_rd_t acc_inv_dim{buf_dim_inv, h};

       h.parallel_for<class kernelCooleyTukeyIFFTFinalReorder>(
           sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
           [=](sycl::nd_item<1> it) {
             const uint64_t k = it.get_global_id(0);
             const uint64_t k_rev = bit_rev(k, log_2_dim);

             acc_res[k] = ff_p_mult(acc_staging[k_rev], acc_inv_dim[0]);
           });
     }).wait();
  }

  std::free(omega_inv);
  std::free(dim_inv);
  std::free(staging);
}
