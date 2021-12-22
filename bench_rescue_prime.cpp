#include "bench_rescue_prime.hpp"
#include "rescue_prime.hpp"

int64_t
benchmark_hash_elements(sycl::queue& q,
                        const uint32_t dim,
                        const uint32_t wg_size,
                        const uint32_t itr_count)
{
  uint64_t* elements =
    static_cast<uint64_t*>(sycl::malloc_device(sizeof(uint64_t) * 8, q));
  uint64_t* hashes = static_cast<uint64_t*>(
    sycl::malloc_device(sizeof(uint64_t) * dim * dim * DIGEST_SIZE, q));

  auto evt_0 = q.single_task([=]() {
    for (uint8_t i = 0; i < 8; i++) {
      *(elements + i) = i;
    }
  });

  using tp = std::chrono::_V2::steady_clock::time_point;
  tp start = std::chrono::steady_clock::now();

  auto evt_1 = q.submit([&](sycl::handler& h) {
    h.depends_on({ evt_0 });
    h.parallel_for<class kernelBenchmarkRescuePrimeHash>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

        for (uint32_t i = 0; i < itr_count; i++) {
          hash_elements(elements, 8, hashes + idx * 4);
        }
      });
  });
  evt_1.wait();

  tp end = std::chrono::steady_clock::now();

  sycl::free(elements, q);
  sycl::free(hashes, q);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
    .count();
}
