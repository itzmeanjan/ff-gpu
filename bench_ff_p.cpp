#include "bench_ff_p.hpp"
#include "ff_p.hpp"

void benchmark_ff_p_addition(sycl::queue &q, const uint32_t dim,
                           const uint32_t wg_size, const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t elem = r + c + 1;
          for (uint64_t i = 0; i < itr_count; i++) {
            ff_p_add(elem, elem+i+1);
          }
        });
  });
  evt.wait();
}
