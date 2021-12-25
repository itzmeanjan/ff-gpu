#include <merkle_tree.hpp>

uint64_t
merklize(sycl::queue& q,
         const sycl::ulong* leaves,
         sycl::ulong* const intermediates,
         const size_t leaf_count,
         const size_t wg_size,
         const sycl::ulong16* mds,
         const sycl::ulong16* ark1,
         const sycl::ulong16* ark2)
{
  // ensure only working with powers of 2 -many leaves
  assert((leaf_count & (leaf_count - 1)) == 0);
  assert(wg_size <= (leaf_count >> 2));

  // so that only last half of tree is touched, where
  // intermediate nodes just above leaves are stored
  const size_t output_offset = leaf_count >> 1;

  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelMerklizeRescuePrimePhase0>(
      sycl::nd_range<1>{ sycl::range<1>{ leaf_count >> 1 },
                         sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        merge(leaves + idx * (DIGEST_SIZE >> 1),
              intermediates + (output_offset + idx) * DIGEST_SIZE,
              mds,
              ark1,
              ark2);
      });
  });

  const size_t log_wg_size =
    static_cast<size_t>(sycl::log2<float>(static_cast<float>(wg_size)));
  const size_t rounds =
    (leaf_count >> 2) == wg_size
      ? 1
      : (leaf_count >> 3) == wg_size
          ? (static_cast<size_t>(sycl::log2<float>(
               static_cast<float>((leaf_count >> 2) >> log_wg_size))) +
             1)
          : static_cast<size_t>(sycl::log2<float>(
              static_cast<float>((leaf_count >> 2) >> log_wg_size)));

  std::vector<sycl::event> evts;
  evts.reserve(rounds);

  size_t offset = (leaf_count >> 2);
  for (size_t r = 0; r < rounds; r++) {
    sycl::event evt = q.submit([&](sycl::handler& h) {
      if (r == 0) {
        h.depends_on(evt_0);
      } else {
        h.depends_on(evts.at(r - 1));
      }

      h.parallel_for<class kernelMerklizeRescuePrimePhase1>(
        sycl::nd_range<1>{
          sycl::range<1>{ offset },
          sycl::range<1>{ offset < wg_size ? offset : wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();

          merge(intermediates + (offset << 1) * DIGEST_SIZE +
                  idx * (DIGEST_SIZE >> 1),
                intermediates + (offset + idx) * DIGEST_SIZE,
                mds,
                ark1,
                ark2);

          size_t round = 1;
          const size_t loc_size = it.get_local_range(0);
          sycl::group<1> grp = it.get_group();

          sycl::group_barrier(grp, sycl::memory_scope_work_group);

          while ((1 << (round - 1)) < loc_size) {
            if (idx % (1 << round) == 0) {
              merge(intermediates + (offset >> (round - 1)) * DIGEST_SIZE +
                      (idx >> round) * (DIGEST_SIZE >> 1),
                    intermediates +
                      ((offset >> round) + (idx >> round)) * DIGEST_SIZE,
                    mds,
                    ark1,
                    ark2);
            }

            sycl::group_barrier(grp, sycl::memory_scope_work_group);
            round++;
          }
        });
    });

    evts.push_back(evt);
    offset >>= (log_wg_size + 1);
    offset = offset == 0 ? 1 : offset;
  }

  evts.at(rounds - 1).wait();

  // calculate sum of dispatched kernel execution times
  uint64_t ts = 0;

  uint64_t start =
    evt_0.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end =
    evt_0.get_profiling_info<sycl::info::event_profiling::command_end>();

  ts += (end - start);

  for (sycl::event evt : evts) {
    uint64_t start =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
      evt.get_profiling_info<sycl::info::event_profiling::command_end>();

    ts += (end - start);
  }

  return ts;
}
