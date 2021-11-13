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