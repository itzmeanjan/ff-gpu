#include "bench_rescue_prime.hpp"
#include "rescue_prime.hpp"

void benchmark_hash_elements(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count) {
  uint64_t *elements = (uint64_t *)sycl::malloc_device(sizeof(uint64_t) * 8, q);
  auto evt_0 = q.single_task([=]() {
    for (uint8_t i = 0; i < 8; i++) {
      *(elements + i) = i;
    }
  });

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on({evt_0});
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          uint64_t hash[DIGEST_SIZE] = {0uLL};
          hash_elements(elements, 8, hash);
        });
  });
  evt_1.wait();
}
