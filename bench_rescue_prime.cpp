#include "bench_rescue_prime.hpp"
#include "rescue_prime.hpp"

uint64_t
benchmark_hash_elements(sycl::queue& q,
                        const uint32_t dim,
                        const uint32_t wg_size,
                        const uint32_t itr_count)
{
  sycl::ulong* elements =
    static_cast<sycl::ulong*>(sycl::malloc_device(sizeof(sycl::ulong) * 8, q));
  sycl::ulong* hashes = static_cast<sycl::ulong*>(
    sycl::malloc_device(sizeof(sycl::ulong) * dim * dim * DIGEST_SIZE, q));
  sycl::ulong16* mds_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * STATE_WIDTH, q));
  sycl::ulong16* ark1_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark2_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* mds_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * STATE_WIDTH, q));
  sycl::ulong16* ark1_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark2_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * NUM_ROUNDS, q));

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  sycl::event evt_0 = q.single_task([=]() {
    for (uint8_t i = 0; i < 8; i++) {
      *(elements + i) = i;
    }
  });

  sycl::event evt_1 =
    q.memcpy(mds_d, mds_h, sizeof(sycl::ulong16) * STATE_WIDTH);
  sycl::event evt_2 =
    q.memcpy(ark1_d, ark1_h, sizeof(sycl::ulong16) * NUM_ROUNDS);
  sycl::event evt_3 =
    q.memcpy(ark2_d, ark2_h, sizeof(sycl::ulong16) * NUM_ROUNDS);

  sycl::event evt_4 = q.submit([&](sycl::handler& h) {
    h.depends_on({ evt_0, evt_1, evt_2, evt_3 });
    h.parallel_for<class kernelBenchmarkRescuePrimeHash>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

        for (uint32_t i = 0; i < itr_count; i++) {
          hash_elements(elements, 8, hashes + idx * 4, mds_d, ark1_d, ark2_d);
        }
      });
  });
  evt_4.wait();

  uint64_t start =
    evt_4.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end =
    evt_4.get_profiling_info<sycl::info::event_profiling::command_end>();

  sycl::free(elements, q);
  sycl::free(hashes, q);
  sycl::free(mds_h, q);
  sycl::free(ark1_h, q);
  sycl::free(ark2_h, q);
  sycl::free(mds_d, q);
  sycl::free(ark1_d, q);
  sycl::free(ark2_d, q);

  return (end - start);
}

uint64_t
benchmark_merge(sycl::queue& q,
                const uint32_t dim,
                const uint32_t wg_size,
                const uint32_t itr_count)
{
  sycl::ulong* input_hashes =
    static_cast<sycl::ulong*>(sycl::malloc_device(sizeof(sycl::ulong) * 8, q));
  sycl::ulong* merged_hashes = static_cast<sycl::ulong*>(
    sycl::malloc_device(sizeof(sycl::ulong) * dim * dim * DIGEST_SIZE, q));
  sycl::ulong16* mds_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * STATE_WIDTH, q));
  sycl::ulong16* ark1_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark2_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* mds_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * STATE_WIDTH, q));
  sycl::ulong16* ark1_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark2_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * NUM_ROUNDS, q));

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  sycl::event evt_0 = q.single_task([=]() {
    for (uint8_t i = 0; i < 8; i++) {
      *(input_hashes + i) = i;
    }
  });

  sycl::event evt_1 =
    q.memcpy(mds_d, mds_h, sizeof(sycl::ulong16) * STATE_WIDTH);
  sycl::event evt_2 =
    q.memcpy(ark1_d, ark1_h, sizeof(sycl::ulong16) * NUM_ROUNDS);
  sycl::event evt_3 =
    q.memcpy(ark2_d, ark2_h, sizeof(sycl::ulong16) * NUM_ROUNDS);

  sycl::event evt_4 = q.submit([&](sycl::handler& h) {
    h.depends_on({ evt_0, evt_1, evt_2, evt_3 });
    h.parallel_for<class kernelBenchmarkRescuePrimeMergeHashes>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();

        for (uint32_t i = 0; i < itr_count; i++) {
          merge(input_hashes, merged_hashes + idx * 4, mds_d, ark1_d, ark2_d);
        }
      });
  });
  evt_4.wait();

  uint64_t start =
    evt_4.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end =
    evt_4.get_profiling_info<sycl::info::event_profiling::command_end>();

  sycl::free(input_hashes, q);
  sycl::free(merged_hashes, q);
  sycl::free(mds_h, q);
  sycl::free(ark1_h, q);
  sycl::free(ark2_h, q);
  sycl::free(mds_d, q);
  sycl::free(ark1_d, q);
  sycl::free(ark2_d, q);

  return (end - start); // in nanoseconds
}
