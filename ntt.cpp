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

uint64_t rev_all_bits(uint64_t n) {
  uint64_t rev = 0;

  for (uint8_t i = 0; i < 64; i++) {
    if ((1ul << i) & n) {
      rev |= (1ul << (63 - i));
    }
  }

  return rev;
}

uint64_t permute_index(uint64_t idx, uint64_t size) {
  if (size == 1ul) {
    return 0ul;
  }

  uint64_t bits = sycl::ext::intel::ctz(size);
  return rev_all_bits(idx) >> (64ul - bits);
}

void cooley_tukey_fft(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                      const uint64_t dim, const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *omega = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));

  {
    // if you change this number, make sure
    // you also change `[[intel::reqd_sub_group_size(Z)]]`
    // below, such that SUBGROUP_SIZE == Z
    constexpr uint64_t SUBGROUP_SIZE = 1ul << 5;

    // don't change following assertions !
    assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1ul)) == 0ul &&
           (SUBGROUP_SIZE <= (1ul << 6)));
    assert((wg_size % SUBGROUP_SIZE) == 0);

    buf_1d_u64_t buf_omega{omega, sycl::range<1>{1}};
    buf_omega.set_write_back(false);

    compute_omega(q, buf_omega, log_2_dim);

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_rd_t acc_vec{vec, h};
      buf_1d_u64_wr_t acc_res{res, h, sycl::no_init};

      h.copy(acc_vec, acc_res);
    });

    for (int64_t i = log_2_dim - 1ul; i >= 0; i--) {
      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_omega{buf_omega, h};
        buf_1d_u64_rw_t acc_res{res, h};

        if ((1ul << i) >= SUBGROUP_SIZE) {
          h.parallel_for<class kernelCooleyTukeyGMFFT>(
              sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
              [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
                sycl::sub_group sg = it.get_sub_group();

                const uint64_t k = it.get_global_id(0);
                const uint64_t p = 1ul << i;
                const uint64_t q = dim / p;

                uint64_t k_rev = bit_rev(k, log_2_dim) % q;
                uint64_t ω = ff_p_pow(sycl::group_broadcast(sg, acc_omega[0]),
                                      p * k_rev);

                if (k < (k ^ p)) {
                  uint64_t tmp_k = acc_res[k];
                  uint64_t tmp_k_p = acc_res[k ^ p];
                  uint64_t tmp_k_p_ω = ff_p_mult(tmp_k_p, ω);

                  acc_res[k] = ff_p_add(tmp_k, tmp_k_p_ω);
                  acc_res[k ^ p] = ff_p_sub(tmp_k, tmp_k_p_ω);
                }
              });
        } else {
          h.parallel_for<class kernelCooleyTukeySGFFT>(
              sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
              [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
                sycl::sub_group sg = it.get_sub_group();

                const uint64_t k = it.get_global_id(0);
                const uint64_t p = 1ul << i; // mask for subgroup shuffle_xor
                const uint64_t q = dim / p;

                uint64_t k_rev = bit_rev(k, log_2_dim) % q;
                uint64_t ω = ff_p_pow(sycl::group_broadcast(sg, acc_omega[0]),
                                      p * k_rev);

                uint64_t tmp_k = acc_res[k];
                // obtain value with work-item index (k ^ p)
                uint64_t tmp_k_p = sg.shuffle_xor(tmp_k, p);
                // obtain ω of work-item with index (k ^ p)
                uint64_t ω_ = sg.shuffle_xor(ω, p);

                acc_res[k] = k < (k ^ p)
                                 ? ff_p_add(tmp_k, ff_p_mult(tmp_k_p, ω))
                                 : ff_p_sub(tmp_k_p, ff_p_mult(tmp_k, ω_));
              });
        }
      });
    }

    q.submit([&](sycl::handler &h) {
       buf_1d_u64_rw_t acc_res{res, h};

       h.parallel_for<class kernelCooleyTukeyFFTFinalReorder>(
           sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
           [=](sycl::nd_item<1> it) {
             const uint64_t k = it.get_global_id(0);
             const uint64_t k_perm = permute_index(k, dim);

             if (k_perm > k) {
               uint64_t a = acc_res[k];
               uint64_t b = acc_res[k_perm];

               a ^= b;
               b ^= a;
               a ^= b;

               acc_res[k] = a;
               acc_res[k_perm] = b;
             }
           });
     }).wait();
  }

  std::free(omega);
}

void cooley_tukey_ifft(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *omega_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *dim_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));

  {
    // if you change this number, make sure
    // you also change `[[intel::reqd_sub_group_size(Z)]]`
    // below, such that SUBGROUP_SIZE == Z
    constexpr uint64_t SUBGROUP_SIZE = 1ul << 5;

    // don't change following assertions !
    assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1ul)) == 0ul &&
           (SUBGROUP_SIZE <= (1ul << 6)));
    assert((wg_size % SUBGROUP_SIZE) == 0);

    buf_1d_u64_t buf_omega_inv{omega_inv, sycl::range<1>{1}};
    buf_1d_u64_t buf_dim_inv{dim_inv, sycl::range<1>{1}};

    buf_omega_inv.set_write_back(false);
    buf_dim_inv.set_write_back(false);

    compute_omega_inv(q, buf_omega_inv, log_2_dim);

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_wr_t acc_inv_dim{buf_dim_inv, h};

      h.single_task([=]() { acc_inv_dim[0] = ff_p_inv(dim); });
    });

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_rd_t acc_vec{vec, h};
      buf_1d_u64_wr_t acc_res{res, h, sycl::no_init};

      h.copy(acc_vec, acc_res);
    });

    for (int64_t i = log_2_dim - 1ul; i >= 0; i--) {
      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_omega_inv{buf_omega_inv, h};
        buf_1d_u64_rw_t acc_res{res, h};

        if ((1ul << i) >= SUBGROUP_SIZE) {
          h.parallel_for<class kernelCooleyTukeyGMIFFT>(
              sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
              [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
                sycl::sub_group sg = it.get_sub_group();

                const uint64_t k = it.get_global_id(0);
                const uint64_t p = 1ul << i;
                const uint64_t q = dim / p;

                uint64_t k_rev = bit_rev(k, log_2_dim) % q;
                uint64_t ω = ff_p_pow(
                    sycl::group_broadcast(sg, acc_omega_inv[0]), p * k_rev);

                if (k < (k ^ p)) {
                  uint64_t tmp_k = acc_res[k];
                  uint64_t tmp_k_p = acc_res[k ^ p];
                  uint64_t tmp_k_p_ω = ff_p_mult(tmp_k_p, ω);

                  acc_res[k] = ff_p_add(tmp_k, tmp_k_p_ω);
                  acc_res[k ^ p] = ff_p_sub(tmp_k, tmp_k_p_ω);
                }
              });
        } else {
          h.parallel_for<class kernelCooleyTukeySGIFFT>(
              sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
              [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
                sycl::sub_group sg = it.get_sub_group();

                const uint64_t k = it.get_global_id(0);
                const uint64_t p = 1ul << i; // mask for subgroup shuffle_xor
                const uint64_t q = dim / p;

                uint64_t k_rev = bit_rev(k, log_2_dim) % q;
                uint64_t ω = ff_p_pow(
                    sycl::group_broadcast(sg, acc_omega_inv[0]), p * k_rev);

                uint64_t tmp_k = acc_res[k];
                // obtain value with work-item index (k ^ p)
                uint64_t tmp_k_p = sg.shuffle_xor(tmp_k, p);
                // obtain ω of work-item with index (k ^ p)
                uint64_t ω_ = sg.shuffle_xor(ω, p);

                acc_res[k] = k < (k ^ p)
                                 ? ff_p_add(tmp_k, ff_p_mult(tmp_k_p, ω))
                                 : ff_p_sub(tmp_k_p, ff_p_mult(tmp_k, ω_));
              });
        }
      });
    }

    q.submit([&](sycl::handler &h) {
       buf_1d_u64_rw_t acc_res{res, h};
       buf_1d_u64_rd_t acc_inv_dim{buf_dim_inv, h};

       h.parallel_for<class kernelCooleyTukeyIFFTFinalReorder>(
           sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
           [=](sycl::nd_item<1> it) {
             sycl::sub_group sg = it.get_sub_group();
             uint64_t inv_dim = sycl::group_broadcast(sg, acc_inv_dim[0]);

             const uint64_t k = it.get_global_id(0);
             const uint64_t k_perm = permute_index(k, dim);

             if (k_perm == k) {
               acc_res[k] = ff_p_mult(acc_res[k], inv_dim);
             } else if (k_perm > k) {
               uint64_t a = acc_res[k];
               uint64_t b = acc_res[k_perm];

               a ^= b;
               b ^= a;
               a ^= b;

               acc_res[k] = ff_p_mult(a, inv_dim);
               acc_res[k_perm] = ff_p_mult(b, inv_dim);
             }
           });
     }).wait();
  }

  std::free(omega_inv);
  std::free(dim_inv);
}

sycl::event matrix_transpose(sycl::queue &q, uint64_t *data, const uint64_t dim,
                             std::vector<sycl::event> evts) {
  constexpr size_t TILE_DIM = 1 << 5;
  constexpr size_t BLOCK_ROWS = 1 << 3;

  assert(TILE_DIM >= BLOCK_ROWS);

  return q.submit([&](sycl::handler &h) {
    sycl::accessor<uint64_t, 2, sycl::access_mode::read_write,
                   sycl::target::local>
        tile_s{sycl::range<2>{TILE_DIM, TILE_DIM + 1}, h};
    sycl::accessor<uint64_t, 2, sycl::access_mode::read_write,
                   sycl::target::local>
        tile_d{sycl::range<2>{TILE_DIM, TILE_DIM + 1}, h};

    h.depends_on(evts);
    h.parallel_for<class kernelMatrixTransposition>(
        sycl::nd_range<2>{sycl::range<2>{dim / (TILE_DIM / BLOCK_ROWS), dim},
                          sycl::range<2>{BLOCK_ROWS, TILE_DIM}},
        [=](sycl::nd_item<2> it) {
          const size_t grp_id_x = it.get_group().get_id(1);
          const size_t grp_id_y = it.get_group().get_id(0);
          const size_t loc_id_x = it.get_local_id(1);
          const size_t loc_id_y = it.get_local_id(0);
          const size_t grp_width_x = it.get_group().get_group_range(1);

          // @note x denotes index along x-axis
          // while y denotes index along y-axis
          //
          // so in usual (row, col) indexing of 2D array
          // row = y, col = x
          const size_t x = grp_id_x * TILE_DIM + loc_id_x;
          const size_t y = grp_id_y * TILE_DIM + loc_id_y;

          const size_t width = grp_width_x * TILE_DIM;

          if (grp_id_y > grp_id_x) {
            size_t dx = grp_id_y * TILE_DIM + loc_id_x;
            size_t dy = grp_id_x * TILE_DIM + loc_id_y;

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_s[loc_id_y + j][loc_id_x] = *(data + (y + j) * width + x);
            }

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_d[loc_id_y + j][loc_id_x] = *(data + (dy + j) * width + dx);
            }

            it.barrier(sycl::access::fence_space::local_space);

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (dy + j) * width + dx) = tile_s[loc_id_x][loc_id_y + j];
            }

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (y + j) * width + x) = tile_d[loc_id_x][loc_id_y + j];
            }

            return;
          }

          if (grp_id_y == grp_id_x) {
            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_s[loc_id_y + j][loc_id_x] = *(data + (y + j) * width + x);
            }

            it.barrier(sycl::access::fence_space::local_space);

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (y + j) * width + x) = tile_s[loc_id_x][loc_id_y + j];
            }
          }
        });
  });
}

sycl::event matrix_transposed_initialise(sycl::queue &q, uint64_t *vec_src,
                                         uint64_t *vec_dst, const uint64_t rows,
                                         const uint64_t cols,
                                         const uint64_t width,
                                         const uint64_t wg_size,
                                         std::vector<sycl::event> evts) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(evts);

    h.parallel_for<class kernelMatrixTransposedInitialise>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          sycl::sub_group sg = it.get_sub_group();

          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          // read into work-item's private memory, using
          // subgroup collective function
          const uint64_t width_ = sycl::group_broadcast(sg, width);

          *(vec_dst + r * width_ + c) = *(vec_src + c * width_ + r);
        });
  });
}

sycl::event row_wise_transform(sycl::queue &q, uint64_t *vec, uint64_t *omega,
                               const uint64_t rows, const uint64_t cols,
                               const uint64_t width, const uint64_t wg_size,
                               std::vector<sycl::event> evts) {
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)cols);

  std::vector<sycl::event> _evts;
  _evts.reserve(log_2_dim);

  // if you change this number, make sure
  // you also change `[[intel::reqd_sub_group_size(Z)]]`
  // below, such that SUBGROUP_SIZE == Z
  constexpr uint64_t SUBGROUP_SIZE = 1ul << 5;

  // don't change following assertions !
  assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1ul)) == 0ul &&
         (SUBGROUP_SIZE <= (1ul << 6)));
  assert((wg_size % SUBGROUP_SIZE) == 0);

  for (int64_t i = log_2_dim - 1ul; i >= 0; i--) {
    sycl::event evt = q.submit([&](sycl::handler &h) {
      if (i == log_2_dim - 1ul) {
        // only first submission depends on
        // previous kernel executions, whose events
        // are passed as argument to this function
        h.depends_on(evts);
      } else {
        // all next kernel submissions
        // depend on just previous kernel submission
        h.depends_on(_evts.at(log_2_dim - (i + 2)));
      }

      if ((1ul << i) >= SUBGROUP_SIZE) {
        // cooley tukey based NTT implementation, with global memory
        // based communication ( relatively slow ! )
        h.parallel_for<class kernelCooleyTukeyRowWiseGMFFT>(
            sycl::nd_range<2>{sycl::range<2>{rows, cols},
                              sycl::range<2>{1, wg_size}},
            [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
              sycl::sub_group sg = it.get_sub_group();

              const uint64_t r = it.get_global_id(0);
              const uint64_t k = it.get_global_id(1);
              const uint64_t p = 1ul << i;
              const uint64_t q = cols / p;

              uint64_t k_rev = bit_rev(k, log_2_dim) % q;
              uint64_t ω =
                  ff_p_pow(sycl::group_broadcast(sg, *omega), p * k_rev);

              if (k < (k ^ p)) {
                uint64_t tmp_k = *(vec + r * width + k);
                uint64_t tmp_k_p = *(vec + r * width + (k ^ p));
                uint64_t tmp_k_p_ω = ff_p_mult(tmp_k_p, ω);

                *(vec + r * width + k) = ff_p_add(tmp_k, tmp_k_p_ω);
                *(vec + r * width + (k ^ p)) = ff_p_sub(tmp_k, tmp_k_p_ω);
              }
            });
      } else {
        // cooley tukey based NTT implementation, with subgroup
        // based shuffling ( much faster than global memory based ! )
        //
        // but I can't use this variant for all iterations, as
        // subgroup size is not always larger than (1ul << i)
        //
        // note, (1ul << i) is NTT butterfly gap for iteration `i`
        //
        // only when control flow reaches this conditional block
        // NTT butterfly shuffling will be in a subgroup itself
        //
        // so in that case, I get to enjoy faster subgroup collective
        // functions, which make use of SIMD in better way
        h.parallel_for<class kernelCooleyTukeyRowWiseSGFFT>(
            sycl::nd_range<2>{sycl::range<2>{rows, cols},
                              sycl::range<2>{1, wg_size}},
            [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
              sycl::sub_group sg = it.get_sub_group();

              const uint64_t r = it.get_global_id(0);
              const uint64_t k = it.get_global_id(1);
              const uint64_t p = 1ul << i; // mask for subgroup shuffling
              const uint64_t q = cols / p;

              uint64_t k_rev = bit_rev(k, log_2_dim) % q;
              uint64_t ω =
                  ff_p_pow(sycl::group_broadcast(sg, *omega), p * k_rev);

              uint64_t tmp_k = *(vec + r * width + k);
              uint64_t tmp_k_p = sg.shuffle_xor(tmp_k, p);
              uint64_t ω_ = sg.shuffle_xor(ω, p);

              *(vec + r * width + k) =
                  k < (k ^ p) ? ff_p_add(tmp_k, ff_p_mult(tmp_k_p, ω))
                              : ff_p_sub(tmp_k_p, ff_p_mult(tmp_k, ω_));

              // if (k % p == k % (2 * p)) {
              //   uint64_t tmp_k = *(vec + r * width + k);
              //   uint64_t tmp_k_p = *(vec + r * width + k + p);
              //   uint64_t tmp_k_p_ω = ff_p_mult(tmp_k_p, ω);

              //   *(vec + r * width + k) = ff_p_add(tmp_k, tmp_k_p_ω);
              //   *(vec + r * width + k + p) = ff_p_sub(tmp_k, tmp_k_p_ω);
              // }
            });
      }
    });
    _evts.push_back(evt);
  }

  return q.submit([&](sycl::handler &h) {
    // final reordering kernel depends on very
    // last kernel submission performed in above loop
    h.depends_on(_evts.at(log_2_dim - 1));
    h.parallel_for<class kernelCooleyTukeyRowWiseFFTFinalReorder>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t k = it.get_global_id(1);
          const uint64_t k_perm = permute_index(k, cols);

          if (k_perm > k) {
            uint64_t a = *(vec + r * width + k);
            uint64_t b = *(vec + r * width + k_perm);

            a ^= b;
            b ^= a;
            a ^= b;

            *(vec + r * width + k) = a;
            *(vec + r * width + k_perm) = b;
          }
        });
  });
}

sycl::event compute_twiddles(sycl::queue &q, uint64_t *twiddles,
                             uint64_t *omega, const uint64_t dim,
                             const uint64_t wg_size,
                             std::vector<sycl::event> evts) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          sycl::sub_group sg = it.get_sub_group();
          const uint64_t c = it.get_global_id(0);

          *(twiddles + c) = ff_p_pow(sycl::group_broadcast(sg, *omega), c);
        });
  });
}

sycl::event twiddle_multiplication(sycl::queue &q, uint64_t *vec,
                                   uint64_t *twiddles, const uint64_t rows,
                                   const uint64_t cols, const uint64_t width,
                                   const uint64_t wg_size,
                                   std::vector<sycl::event> evts) {
  assert(cols == width || 2 * cols == width);

  return q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelTwiddleFactorMultiplication>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          sycl::sub_group sg = it.get_sub_group();

          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          *(vec + r * width + c) = ff_p_mult(
              *(vec + r * width + c),
              ff_p_pow(sycl::group_broadcast(sg, *(twiddles + r)), c));
        });
  });
}

void six_step_fft(sycl::queue &q, uint64_t *vec, const uint64_t dim,
                  const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);

  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);
  uint64_t log_2_n1 = (uint64_t)sycl::log2((float)n1);
  uint64_t log_2_n2 = (uint64_t)sycl::log2((float)n2);

  assert(n1 == n2 || n2 == 2 * n1);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *vec_ =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * n * n, q));
  uint64_t *twiddles =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * n2, q));
  uint64_t *omega_dim =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));
  uint64_t *omega_n1 =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));
  uint64_t *omega_n2 =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));

  // compute i-th root of unity, where n = {dim, n1, n2}
  sycl::event evt_0 =
      q.single_task([=]() { *omega_dim = get_root_of_unity(log_2_dim); });
  sycl::event evt_1 =
      q.single_task([=]() { *omega_n1 = get_root_of_unity(log_2_n1); });
  sycl::event evt_2 =
      q.single_task([=]() { *omega_n2 = get_root_of_unity(log_2_n2); });

  // Step 1: Transpose Matrix
  sycl::event evt_3 =
      matrix_transposed_initialise(q, vec, vec_, n2, n1, n, wg_size, {});

  // Step 2: n2-many parallel n1-point Cooley-Tukey style FFT
  sycl::event evt_4 =
      row_wise_transform(q, vec_, omega_n1, n2, n1, n, wg_size, {evt_1, evt_3});

  // Step 3: Multiply by twiddle factors
  sycl::event evt_5 =
      compute_twiddles(q, twiddles, omega_dim, n2, wg_size, {evt_0});
  sycl::event evt_6 = twiddle_multiplication(q, vec_, twiddles, n2, n1, n,
                                             wg_size, {evt_4, evt_5});

  // Step 4: Transpose Matrix
  sycl::event evt_7 = matrix_transpose(q, vec_, n, {evt_6});

  // Step 5: n1-many parallel n2-point Cooley-Tukey FFT
  sycl::event evt_8 =
      row_wise_transform(q, vec_, omega_n2, n1, n2, n, wg_size, {evt_2, evt_7});

  // Step 6: Transpose Matrix
  sycl::event evt_9 = matrix_transpose(q, vec_, n, {evt_8});

  // copy result back to source matrix
  sycl::event evt_10 = q.submit([&](sycl::handler &h) {
    h.depends_on(evt_9);

    h.parallel_for<class kernelFFTCopyBack>(
        sycl::nd_range<2>{sycl::range<2>{n2, n1}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          *(vec + it.get_global_linear_id()) = *(vec_ + r * n + c);
        });
  });

  evt_10.wait();

  sycl::free(vec_, q);
  sycl::free(twiddles, q);
  sycl::free(omega_dim, q);
  sycl::free(omega_n1, q);
  sycl::free(omega_n2, q);
}

void six_step_ifft(sycl::queue &q, uint64_t *vec, const uint64_t dim,
                   const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);

  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);
  uint64_t log_2_n1 = (uint64_t)sycl::log2((float)n1);
  uint64_t log_2_n2 = (uint64_t)sycl::log2((float)n2);

  assert(n1 == n2 || n2 == 2 * n1);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *vec_ =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * n * n, q));
  uint64_t *twiddles =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * n2, q));
  uint64_t *omega_dim_inv =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));
  uint64_t *omega_n1_inv =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));
  uint64_t *omega_n2_inv =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));
  uint64_t *omega_domain_size_inv =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));

  // compute inverse of i-th root of unity, where n = {dim, n1, n2}
  sycl::event evt_0 = q.single_task(
      [=]() { *omega_dim_inv = ff_p_inv(get_root_of_unity(log_2_dim)); });
  sycl::event evt_1 = q.single_task(
      [=]() { *omega_n1_inv = ff_p_inv(get_root_of_unity(log_2_n1)); });
  sycl::event evt_2 = q.single_task(
      [=]() { *omega_n2_inv = ff_p_inv(get_root_of_unity(log_2_n2)); });
  sycl::event evt_3 =
      q.single_task([=]() { *omega_domain_size_inv = ff_p_inv(dim); });

  // Step 1: Transpose Matrix
  sycl::event evt_4 =
      matrix_transposed_initialise(q, vec, vec_, n2, n1, n, wg_size, {});

  // Step 2: n2-many parallel n1-point Cooley-Tukey style IFFT
  sycl::event evt_5 = row_wise_transform(q, vec_, omega_n1_inv, n2, n1, n,
                                         wg_size, {evt_1, evt_4});

  // Step 3: Multiply by twiddle factors
  sycl::event evt_6 =
      compute_twiddles(q, twiddles, omega_dim_inv, n2, wg_size, {evt_0});
  sycl::event evt_7 = twiddle_multiplication(q, vec_, twiddles, n2, n1, n,
                                             wg_size, {evt_5, evt_6});

  // Step 4: Transpose Matrix
  sycl::event evt_8 = matrix_transpose(q, vec_, n, {evt_7});

  // Step 5: n1-many parallel n2-point Cooley-Tukey IFFT
  sycl::event evt_9 = row_wise_transform(q, vec_, omega_n2_inv, n1, n2, n,
                                         wg_size, {evt_2, evt_8});

  // Step 6: Transpose Matrix
  sycl::event evt_10 = matrix_transpose(q, vec_, n, {evt_9});

  // copy result back to source matrix, while
  // also multiplying by inverse of domain size
  sycl::event evt_11 = q.submit([&](sycl::handler &h) {
    h.depends_on({evt_3, evt_10});

    h.parallel_for<class kernelIFFTCopyBack>(
        sycl::nd_range<2>{sycl::range<2>{n2, n1}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          sycl::sub_group sg = it.get_sub_group();

          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          *(vec + it.get_global_linear_id()) =
              ff_p_mult(sycl::group_broadcast(sg, *omega_domain_size_inv),
                        *(vec_ + r * n + c));
        });
  });

  evt_11.wait();

  sycl::free(vec_, q);
  sycl::free(twiddles, q);
  sycl::free(omega_dim_inv, q);
  sycl::free(omega_n1_inv, q);
  sycl::free(omega_n2_inv, q);
  sycl::free(omega_domain_size_inv, q);
}
