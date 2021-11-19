#include "bench_ntt.hpp"

int64_t benchmark_forward_transform(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size) {
  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_fwd = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  prepare_random_vector(vec_src, dim);

  tp start = std::chrono::steady_clock::now();
  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_fwd{vec_fwd, sycl::range<1>{dim}};

    forward_transform(q, buf_vec_src, buf_vec_fwd, dim, wg_size);
  }
  tp end = std::chrono::steady_clock::now();

  std::free(vec_src);
  std::free(vec_fwd);

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

int64_t benchmark_inverse_transform(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size) {
  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  prepare_random_vector(vec_src, dim);

  tp start = std::chrono::steady_clock::now();
  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_inv{vec_inv, sycl::range<1>{dim}};

    inverse_transform(q, buf_vec_src, buf_vec_inv, dim, wg_size);
  }
  tp end = std::chrono::steady_clock::now();

  std::free(vec_src);
  std::free(vec_inv);

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

int64_t benchmark_cooley_tukey_fft(sycl::queue &q, const uint64_t dim,
                                   const uint64_t wg_size) {
  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_fwd = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  prepare_random_vector(vec_src, dim);

  tp start = std::chrono::steady_clock::now();
  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_fwd{vec_fwd, sycl::range<1>{dim}};

    cooley_tukey_fft(q, buf_vec_src, buf_vec_fwd, dim, wg_size);
  }
  tp end = std::chrono::steady_clock::now();

  std::free(vec_src);
  std::free(vec_fwd);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int64_t benchmark_cooley_tukey_ifft(sycl::queue &q, const uint64_t dim,
                                    const uint64_t wg_size) {
  uint64_t *vec_src = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));
  uint64_t *vec_inv = static_cast<uint64_t *>(malloc(sizeof(uint64_t) * dim));

  prepare_random_vector(vec_src, dim);

  tp start = std::chrono::steady_clock::now();
  {
    buf_1d_u64_t buf_vec_src{vec_src, sycl::range<1>{dim}};
    buf_1d_u64_t buf_vec_inv{vec_inv, sycl::range<1>{dim}};

    cooley_tukey_ifft(q, buf_vec_src, buf_vec_inv, dim, wg_size);
  }
  tp end = std::chrono::steady_clock::now();

  std::free(vec_src);
  std::free(vec_inv);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int64_t benchmark_matrix_transposition(sycl::queue &q, const uint64_t dim,
                                       const uint64_t wg_size) {
  uint64_t *vec_h = static_cast<uint64_t *>(
      sycl::malloc_host(sizeof(uint64_t) * dim * dim, q));
  uint64_t *vec_d = static_cast<uint64_t *>(
      sycl::malloc_device(sizeof(uint64_t) * dim * dim, q));

  prepare_random_vector(vec_h, dim * dim);

  sycl::event evt_0 = q.memcpy(vec_d, vec_h, sizeof(uint64_t) * dim * dim);
  evt_0.wait();

  tp start = std::chrono::steady_clock::now();
  matrix_transpose(q, vec_d, dim, wg_size, {}).wait();
  tp end = std::chrono::steady_clock::now();

  sycl::free(vec_h, q);
  sycl::free(vec_d, q);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int64_t benchmark_twiddle_factor_multiplication(sycl::queue &q,
                                                const uint64_t n1,
                                                const uint64_t n2,
                                                const uint64_t wg_size) {
  assert(n1 == n2 || n2 == 2 * n1);
  uint64_t n = std::max(n1, n2);

  uint64_t *vec_h =
      static_cast<uint64_t *>(sycl::malloc_host(sizeof(uint64_t) * n * n, q));
  uint64_t *vec_d =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t) * n * n, q));
  uint64_t *omega =
      static_cast<uint64_t *>(sycl::malloc_device(sizeof(uint64_t), q));

  q.memset(vec_h, 0, sizeof(uint64_t) * n * n).wait();
  q.single_task([=]() {
     *omega = get_root_of_unity((uint64_t)sycl::log2((float)n1 * n2));
   }).wait();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

  for (uint64_t i = 0; i < n2; i++) {
    for (uint64_t j = 0; j < n1; j++) {
      *(vec_h + i * n + j) = dis(gen);
    }
  }

  sycl::event evt_0 = q.memcpy(vec_d, vec_h, sizeof(uint64_t) * n * n);
  evt_0.wait();

  tp start = std::chrono::steady_clock::now();
  twiddle_multiplication(q, vec_d, omega, n2, n1, n, wg_size, {}).wait();
  tp end = std::chrono::steady_clock::now();

  sycl::free(vec_h, q);
  sycl::free(vec_d, q);
  sycl::free(omega, q);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}
