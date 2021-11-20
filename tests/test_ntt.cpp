#include "test_ntt.hpp"
#include <iostream>

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

void check_ntt_correctness(sycl::queue &q, const uint64_t dim,
                           const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *omega = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *omega_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  uint64_t *mat_a =
      static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));
  uint64_t *mat_b =
      static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));
  uint64_t *mat_c =
      static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim * dim));

  {
    buf_1d_u64_t buf_omega{omega, sycl::range<1>{1}};
    buf_1d_u64_t buf_omega_inv{omega_inv, sycl::range<1>{1}};
    buf_2d_u64_t buf_mat_a{mat_a, sycl::range<2>{dim, dim}};
    buf_2d_u64_t buf_mat_b{mat_b, sycl::range<2>{dim, dim}};
    buf_2d_u64_t buf_mat_c{mat_c, sycl::range<2>{dim, dim}};

    compute_omega(q, buf_omega, log_2_dim);
    compute_omega_inv(q, buf_omega_inv, log_2_dim);
    compute_dft_matrix(q, buf_mat_a, buf_omega, dim, wg_size);
    compute_dft_matrix(q, buf_mat_b, buf_omega_inv, dim, wg_size);
    compute_matrix_matrix_multiplication(q, buf_mat_a, buf_mat_b, buf_mat_c,
                                         dim, wg_size);

    uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
    memset(mismatch, 0, sizeof(uint64_t));
    {
      buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

      q.submit([&](sycl::handler &h) {
        buf_2d_u64_rd_t acc_mat_c{buf_mat_c, h};
        buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

        h.parallel_for<class kernelCheckNTTCorrectness>(
            sycl::nd_range<2>{sycl::range<2>{dim, dim},
                              sycl::range<2>{1, wg_size}},
            [=](sycl::nd_item<2> it) {
              const size_t r = it.get_global_id(0);
              const size_t c = it.get_global_id(1);

              uint64_t v = acc_mat_c[r][c] % MOD;

              sycl::ext::oneapi::atomic_ref<
                  uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_device_space>
                  corr_ref{acc_mismatch[0]};
              corr_ref.fetch_add((r == c ? v == dim : v == 0) ? 0 : 1);
            });
      });
      q.wait();
    }

    assert(*mismatch == 0);
    std::free(mismatch);
  }

  std::free(omega);
  std::free(omega_inv);
  std::free(mat_a);
  std::free(mat_b);
  std::free(mat_c);
}

void prepare_random_vector(uint64_t *const vec, const uint64_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

  for (uint64_t i = 0; i < size; i++) {
    *(vec + i) = dis(gen);
  }
}

void check_ntt_forward_inverse_transform(sycl::queue &q, const uint64_t dim,
                                         const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_fwd = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  prepare_random_vector(vec_src, dim);

  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_fwd{vec_fwd, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_inv{vec_inv, sycl::range<1>{dim}};

    forward_transform(q, buf_vec_src, buf_vec_fwd, dim, wg_size);
    inverse_transform(q, buf_vec_fwd, buf_vec_inv, dim, wg_size);

    uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
    memset(mismatch, 0, sizeof(uint64_t));
    {
      buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_vec_src{buf_vec_src, h};
        buf_1d_u64_rd_t acc_vec_inv{buf_vec_inv, h};
        buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

        h.parallel_for<class kernelCheckNTTForwardInverseTransform>(
            sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
            [=](sycl::nd_item<1> it) {
              const size_t r = it.get_global_id(0);

              sycl::ext::oneapi::atomic_ref<
                  uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_device_space>
                  corr_ref{acc_mismatch[0]};
              corr_ref.fetch_add(
                  acc_vec_src[r] % MOD == acc_vec_inv[r] % MOD ? 0 : 1);
            });
      });
      q.wait();
    }

    assert(*mismatch == 0);
    std::free(mismatch);
  }

  std::free(vec_src);
  std::free(vec_fwd);
  std::free(vec_inv);
}

void check_cooley_tukey_ntt(sycl::queue &q, const uint64_t dim,
                            const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_fwd = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  prepare_random_vector(vec_src, dim);

  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_fwd{vec_fwd, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_inv{vec_inv, sycl::range<1>{dim}};

    cooley_tukey_fft(q, buf_vec_src, buf_vec_fwd, dim, wg_size);
    cooley_tukey_ifft(q, buf_vec_fwd, buf_vec_inv, dim, wg_size);

    uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
    memset(mismatch, 0, sizeof(uint64_t));
    {
      buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_vec_src{buf_vec_src, h};
        buf_1d_u64_rd_t acc_vec_inv{buf_vec_inv, h};
        buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

        h.parallel_for<class kernelCheckCooleyTukeyNTT>(
            sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
            [=](sycl::nd_item<1> it) {
              const size_t r = it.get_global_id(0);

              sycl::ext::oneapi::atomic_ref<
                  uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_device_space>
                  corr_ref{acc_mismatch[0]};
              corr_ref.fetch_add(
                  acc_vec_src[r] % MOD == acc_vec_inv[r] % MOD ? 0 : 1);
            });
      });
      q.wait();
    }

    assert(*mismatch == 0);
    std::free(mismatch);
  }

  std::free(vec_src);
  std::free(vec_fwd);
  std::free(vec_inv);
}

void check_matrix_transposition(sycl::queue &q, const uint64_t dim,
                                const uint64_t wg_size) {
  uint64_t *vec_d = static_cast<uint64_t *>(
      sycl::malloc_device(sizeof(uint64_t) * dim * dim, q));
  uint64_t *vec_s = static_cast<uint64_t *>(
      sycl::malloc_shared(sizeof(uint64_t) * dim * dim, q));

  prepare_random_vector(vec_s, dim * dim);
  sycl::event evt_0 = q.memcpy(vec_d, vec_s, sizeof(uint64_t) * dim * dim);

  sycl::event evt_1 = matrix_transpose(q, vec_d, dim, wg_size, {evt_0});
  sycl::event evt_2 = matrix_transpose(q, vec_d, dim, wg_size, {evt_1});

  uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  memset(mismatch, 0, sizeof(uint64_t));

  {
    buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

    sycl::event evt_3 = q.submit([&](sycl::handler &h) {
      buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

      h.depends_on({evt_2});
      h.parallel_for(sycl::nd_range<2>{sycl::range<2>{dim, dim},
                                       sycl::range<2>{1, wg_size}},
                     [=](sycl::nd_item<2> it) {
                       const size_t l_idx = it.get_global_linear_id();

                       sycl::ext::oneapi::atomic_ref<
                           uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_device_space>
                           corr_ref{acc_mismatch[0]};
                       corr_ref.fetch_add(
                           *(vec_d + l_idx) == *(vec_s + l_idx) ? 0 : 1);
                     });
    });
    evt_3.wait();
  }

  assert(*mismatch == 0);
  std::free(mismatch);

  sycl::free(vec_d, q);
  sycl::free(vec_s, q);
}

void test_twiddle_factor_multiplication(sycl::queue &q, const uint64_t n1,
                                        const uint64_t n2,
                                        const uint64_t wg_size) {
  assert(n1 == n2 || n2 == 2 * n1);
  uint64_t n = std::max(n1, n2);

  uint64_t *vec_s =
      static_cast<uint64_t *>(sycl::malloc_shared(sizeof(uint64_t) * n * n, q));
  uint64_t *vec_d =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * n * n, q));
  uint64_t *omega =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));

  // first initialise all cells with zero
  q.memset(vec_s, 0, sizeof(uint64_t) * n * n).wait();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

  // then randomly assign field elements to n2 x n1 many cells
  for (uint64_t i = 0; i < n2; i++) {
    for (uint64_t j = 0; j < n1; j++) {
      *(vec_s + i * n + j) = dis(gen);
    }
  }

  // finally copy all n x n elements to device allocated vector,
  // some may be zeros ( not important cells, but still kept, just to ensure
  // that matrix is of square shape ! )
  sycl::event evt_0 = q.memcpy(vec_d, vec_s, sizeof(uint64_t) * n * n);

  sycl::event evt_1 = q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      *omega = get_root_of_unity((uint64_t)sycl::log2((float)n1 * n2));
    });
  });

  sycl::event evt_2 = twiddle_multiplication(q, vec_d, omega, n2, n1, n,
                                             wg_size, {evt_0, evt_1});

  uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  memset(mismatch, 0, sizeof(uint64_t));

  {
    buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

    sycl::event evt_3 = q.submit([&](sycl::handler &h) {
      buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

      h.depends_on({evt_2});
      h.parallel_for(
          sycl::nd_range<2>{sycl::range<2>{n, n}, sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) {
            const uint64_t r = it.get_global_id(0);
            const uint64_t c = it.get_global_id(1);

            sycl::ext::oneapi::atomic_ref<
                uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_device_space>
                corr_ref{acc_mismatch[0]};
            corr_ref.fetch_add(*(vec_d + r * n + c) % MOD ==
                                       ff_p_mult(*(vec_s + r * n + c),
                                                 ff_p_pow(*omega, r * c)) %
                                           MOD
                                   ? 0
                                   : 1);
          });
    });
    evt_3.wait();
  }

  assert(*mismatch == 0);
  std::free(mismatch);

  sycl::free(vec_s, q);
  sycl::free(vec_d, q);
  sycl::free(omega, q);
}

void test_six_step_fft(sycl::queue &q, const uint64_t dim,
                       const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_fwd = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_dev =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * dim, q));

  prepare_random_vector(vec_src, dim);
  q.memcpy(vec_dev, vec_src, sizeof(uint64_t) * dim).wait();

  // waits till completion
  six_step_fft(q, vec_dev, dim, wg_size);

  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_fwd{vec_fwd, sycl::range<1>{dim}};

    cooley_tukey_fft(q, buf_vec_src, buf_vec_fwd, dim, wg_size);

    uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
    memset(mismatch, 0, sizeof(uint64_t));
    {
      buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_vec_fwd{buf_vec_fwd, h};
        buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

        h.parallel_for<class kernelCompareCooleyTukeyFFTWithSSA>(
            sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
            [=](sycl::nd_item<1> it) {
              const size_t r = it.get_global_id(0);

              sycl::ext::oneapi::atomic_ref<
                  uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_device_space>
                  corr_ref{acc_mismatch[0]};
              corr_ref.fetch_add(
                  acc_vec_fwd[r] % MOD == *(vec_dev + r) % MOD ? 0 : 1);
            });
      });
      q.wait();
    }

    assert(*mismatch == 0);
    std::free(mismatch);
  }

  std::free(vec_src);
  std::free(vec_fwd);
  sycl::free(vec_dev, q);
}

void test_six_step_ifft(sycl::queue &q, const uint64_t dim,
                        const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_dev =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * dim, q));

  prepare_random_vector(vec_src, dim);
  q.memcpy(vec_dev, vec_src, sizeof(uint64_t) * dim).wait();

  // waits till completion
  six_step_ifft(q, vec_dev, dim, wg_size);

  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_inv{vec_inv, sycl::range<1>{dim}};

    cooley_tukey_ifft(q, buf_vec_src, buf_vec_inv, dim, wg_size);

    uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
    memset(mismatch, 0, sizeof(uint64_t));
    {
      buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

      q.submit([&](sycl::handler &h) {
        buf_1d_u64_rd_t acc_vec_inv{buf_vec_inv, h};
        buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

        h.parallel_for<class kernelCompareCooleyTukeyIFFTWithSSA>(
            sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
            [=](sycl::nd_item<1> it) {
              const size_t r = it.get_global_id(0);

              sycl::ext::oneapi::atomic_ref<
                  uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_device_space>
                  corr_ref{acc_mismatch[0]};
              corr_ref.fetch_add(
                  acc_vec_inv[r] % MOD == *(vec_dev + r) % MOD ? 0 : 1);
            });
      });
      q.wait();
    }

    assert(*mismatch == 0);
    std::free(mismatch);
  }

  std::free(vec_src);
  std::free(vec_inv);
  sycl::free(vec_dev, q);
}

void test_six_step_fft_and_ifft(sycl::queue &q, const uint64_t dim,
                                const uint64_t wg_size) {
  assert((dim & (dim - 1ul)) == 0);
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  assert(log_2_dim > 0 && log_2_dim <= TWO_ADICITY);

  uint64_t *vec_src =
      static_cast<uint64_t *>(sycl::malloc_shared(sizeof(uint64_t) * dim, q));
  uint64_t *vec_fwd =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * dim, q));
  uint64_t *vec_inv =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * dim, q));

  prepare_random_vector(vec_src, dim);

  q.memcpy(vec_fwd, vec_src, sizeof(uint64_t) * dim).wait();
  six_step_fft(q, vec_fwd, dim, wg_size);

  q.memcpy(vec_inv, vec_fwd, sizeof(uint64_t) * dim).wait();
  six_step_ifft(q, vec_inv, dim, wg_size);

  uint64_t *mismatch = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  memset(mismatch, 0, sizeof(uint64_t));
  {
    buf_1d_u64_t buf_mismatch{mismatch, sycl::range<1>{1}};

    q.submit([&](sycl::handler &h) {
      buf_1d_u64_rw_t acc_mismatch{buf_mismatch, h};

      h.parallel_for<class kernelCompareSSAFFTAndIFFT>(
          sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
          [=](sycl::nd_item<1> it) {
            const size_t r = it.get_global_id(0);

            sycl::ext::oneapi::atomic_ref<
                uint64_t, sycl::ext::oneapi::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_device_space>
                corr_ref{acc_mismatch[0]};
            corr_ref.fetch_add(
                *(vec_inv + r) % MOD == *(vec_src + r) % MOD ? 0 : 1);
          });
    });
    q.wait();
  }

  assert(*mismatch == 0);
  std::free(mismatch);

  sycl::free(vec_src, q);
  sycl::free(vec_fwd, q);
  sycl::free(vec_inv, q);
}
