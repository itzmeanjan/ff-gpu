#include "bench_rescue_prime.hpp"
#include "rescue_prime.hpp"

void benchmark_hash_elements(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          uint64_t elements[8] = {0, 1, 2, 3, 4, 5, 6, 7};
          uint64_t hash[DIGEST_SIZE] = {0uLL};
          hash_elements(elements, 8, hash);
        });
  });
  evt.wait();
}
