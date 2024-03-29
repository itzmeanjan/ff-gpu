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
  sycl::ulong4* mds_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* mds_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  sycl::event evt_0 = q.single_task([=]() {
    for (uint8_t i = 0; i < 8; i++) {
      *(elements + i) = i;
    }
  });

  sycl::event evt_1 =
    q.memcpy(mds_d, mds_h, sizeof(sycl::ulong4) * STATE_WIDTH * 3);
  sycl::event evt_2 =
    q.memcpy(ark1_d, ark1_h, sizeof(sycl::ulong4) * NUM_ROUNDS * 3);
  sycl::event evt_3 =
    q.memcpy(ark2_d, ark2_h, sizeof(sycl::ulong4) * NUM_ROUNDS * 3);

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
  sycl::ulong4* mds_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* mds_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  sycl::event evt_0 = q.single_task([=]() {
    for (uint8_t i = 0; i < 8; i++) {
      *(input_hashes + i) = i;
    }
  });

  sycl::event evt_1 =
    q.memcpy(mds_d, mds_h, sizeof(sycl::ulong4) * STATE_WIDTH * 3);
  sycl::event evt_2 =
    q.memcpy(ark1_d, ark1_h, sizeof(sycl::ulong4) * NUM_ROUNDS * 3);
  sycl::event evt_3 =
    q.memcpy(ark2_d, ark2_h, sizeof(sycl::ulong4) * NUM_ROUNDS * 3);

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

uint64_t
benchmark_merge_using_scratch_pad(sycl::queue& q,
                                  const uint32_t dim,
                                  const uint32_t wg_size,
                                  const uint32_t itr_count)
{
  assert(wg_size >= 12);

  sycl::ulong* input_hashes =
    static_cast<sycl::ulong*>(sycl::malloc_device(sizeof(sycl::ulong) * 8, q));
  sycl::ulong* merged_hashes = static_cast<sycl::ulong*>(
    sycl::malloc_device(sizeof(sycl::ulong) * dim * dim * DIGEST_SIZE, q));
  sycl::ulong4* mds_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2_h = static_cast<sycl::ulong4*>(
    sycl::malloc_host(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* mds_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2_d = static_cast<sycl::ulong4*>(
    sycl::malloc_device(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  sycl::event evt_0 = q.single_task([=]() {
    for (uint8_t i = 0; i < 8; i++) {
      *(input_hashes + i) = i;
    }
  });

  sycl::event evt_1 =
    q.memcpy(mds_d, mds_h, sizeof(sycl::ulong4) * STATE_WIDTH * 3);
  sycl::event evt_2 =
    q.memcpy(ark1_d, ark1_h, sizeof(sycl::ulong4) * NUM_ROUNDS * 3);
  sycl::event evt_3 =
    q.memcpy(ark2_d, ark2_h, sizeof(sycl::ulong4) * NUM_ROUNDS * 3);

  sycl::event evt_4 = q.submit([&](sycl::handler& h) {
    scratch_mem_1d_t mds_loc =
      scratch_mem_1d_t{ sycl::range<1>{ STATE_WIDTH * 3 }, h };
    scratch_mem_1d_t ark1_loc =
      scratch_mem_1d_t{ sycl::range<1>{ NUM_ROUNDS * 3 }, h };
    scratch_mem_1d_t ark2_loc =
      scratch_mem_1d_t{ sycl::range<1>{ NUM_ROUNDS * 3 }, h };

    h.depends_on({ evt_0, evt_1, evt_2, evt_3 });
    h.parallel_for<class kernelBenchmarkRescuePrimeMergeHashesUsingScratchPad>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) {
        const size_t idx = it.get_global_linear_id();
        const size_t loc_idx = it.get_local_linear_id();
        sycl::group grp = it.get_group();

        if (loc_idx % 12 == loc_idx) {
          for (size_t j = 0; j < 3; j++) {
            const size_t k = j * 12 + loc_idx;
            mds_loc[k] = mds_d[k];
          }
        }

        if (loc_idx % 7 == loc_idx) {
          for (size_t j = 0; j < 3; j++) {
            const size_t k = j * 7 + loc_idx;
            ark1_loc[k] = ark1_d[k];
          }

          for (size_t j = 0; j < 3; j++) {
            const size_t k = j * 7 + loc_idx;
            ark2_loc[k] = ark2_d[k];
          }
        }

        // ensure all rescue prime constants written into local memory
        // and visible to all work-items in work-group
        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        for (uint32_t i = 0; i < itr_count; i++) {
          merge(
            input_hashes, merged_hashes + idx * 4, mds_loc, ark1_loc, ark2_loc);
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
