#include <merkle_tree.hpp>

uint64_t
merklize_approach_1(sycl::queue& q,
                    const sycl::ulong* leaves,
                    sycl::ulong* const intermediates,
                    const size_t leaf_count,
                    const size_t wg_size,
                    const sycl::ulong4* mds,
                    const sycl::ulong4* ark1,
                    const sycl::ulong4* ark2)
{
  // ensure only working with powers of 2 -many leaves
  assert((leaf_count & (leaf_count - 1)) == 0);
  // checking that requested work group size for first
  // phase of kernel dispatch is valid
  //
  // for next rounds of kernel dispatches, work group
  // size will be adapted when required !
  assert(wg_size <= (leaf_count >> 1));

  const size_t output_offset = leaf_count >> 1;

  // this is first phase of kernel dispatch, where I compute
  // ( in parallel ) all intermediate nodes just above leaves of tree
  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelMerklizeRescuePrimeApproach1Phase0>(
      sycl::nd_range<1>{ sycl::range<1>{ output_offset },
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

  // for computing all remaining intermediate nodes, we'll need to
  // dispatch `rounds` -many kernels, where each round is data dependent
  // on just previous one
  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(leaf_count >> 1)));

  std::vector<sycl::event> evts;
  evts.reserve(rounds);

  for (size_t r = 0; r < rounds; r++) {
    sycl::event evt = q.submit([&](sycl::handler& h) {
      if (r == 0) {
        h.depends_on(evt_0);
      } else {
        h.depends_on(evts.at(r - 1));
      }

      // these many intermediate nodes to be computed during this
      // kernel dispatch round
      const size_t offset = leaf_count >> (r + 2);

      h.parallel_for<class kernelMerklizeRescuePrimeApproach1Phase1>(
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
        });
    });
    evts.push_back(evt);
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

uint64_t
merklize_approach_2(sycl::queue& q,
                    const sycl::ulong* leaves,
                    sycl::ulong* const intermediates,
                    const size_t leaf_count,
                    const size_t wg_size,
                    const sycl::ulong4* mds,
                    const sycl::ulong4* ark1,
                    const sycl::ulong4* ark2)
{
  // ensure only working with powers of 2 -many leaves
  assert((leaf_count & (leaf_count - 1)) == 0);
  assert(wg_size <= (leaf_count >> 2));

  // so that only last half of tree is touched, where
  // intermediate nodes just above leaves are stored
  const size_t output_offset = leaf_count >> 1;

  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelMerklizeRescuePrimeApproach2Phase0>(
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

      h.parallel_for<class kernelMerklizeRescuePrimeApproach2Phase1>(
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

          sycl::group_barrier(grp, sycl::memory_scope_device);

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

            sycl::group_barrier(grp, sycl::memory_scope_device);
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

uint64_t
merklize_approach_3(sycl::queue& q,
                    const sycl::ulong* leaves,
                    sycl::ulong* const intermediates,
                    const size_t leaf_count,
                    const size_t wg_size,
                    const sycl::ulong4* mds,
                    const sycl::ulong4* ark1,
                    const sycl::ulong4* ark2)
{
  // ensure only working with powers of 2 -many leaves
  assert((leaf_count & (leaf_count - 1)) == 0);
  // checking that requested work group size for first
  // phase of kernel dispatch is valid
  //
  // for next rounds of kernel dispatches, work group
  // size will be adapted when required !
  assert(wg_size <= (leaf_count >> 1));

  const size_t output_offset = leaf_count >> 1;

  // this is first phase of kernel dispatch, where I compute
  // ( in parallel ) all intermediate nodes just above leaves of tree
  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    if (wg_size >= 12) {
      scratch_mem_1d_t mds_loc =
        scratch_mem_1d_t{ sycl::range<1>{ STATE_WIDTH * 3 }, h };
      scratch_mem_1d_t ark1_loc =
        scratch_mem_1d_t{ sycl::range<1>{ NUM_ROUNDS * 3 }, h };
      scratch_mem_1d_t ark2_loc =
        scratch_mem_1d_t{ sycl::range<1>{ NUM_ROUNDS * 3 }, h };

      h.parallel_for<
        class kernelMerklizeRescuePrimeApproach3Phase0UsingScratchPad>(
        sycl::nd_range<1>{ sycl::range<1>{ output_offset },
                           sycl::range<1>{ wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();
          const size_t loc_idx = it.get_local_linear_id();
          sycl::group grp = it.get_group();

          if (loc_idx % 12 == loc_idx) {
            for (size_t j = 0; j < 3; j++) {
              const size_t k = j * 12 + loc_idx;
              mds_loc[k] = mds[k];
            }
          }

          if (loc_idx % 7 == loc_idx) {
            for (size_t j = 0; j < 3; j++) {
              const size_t k = j * 7 + loc_idx;
              ark1_loc[k] = ark1[k];
            }

            for (size_t j = 0; j < 3; j++) {
              const size_t k = j * 7 + loc_idx;
              ark2_loc[k] = ark2[k];
            }
          }

          // ensure all rescue prime constants written into local memory
          // and visible to all work-items in work-group
          sycl::group_barrier(grp, sycl::memory_scope::work_group);

          merge(leaves + idx * (DIGEST_SIZE >> 1),
                intermediates + (output_offset + idx) * DIGEST_SIZE,
                mds_loc,
                ark1_loc,
                ark2_loc);
        });
    } else {
      h.parallel_for<
        class kernelMerklizeRescuePrimeApproach3Phase0UsingGlobalMem>(
        sycl::nd_range<1>{ sycl::range<1>{ output_offset },
                           sycl::range<1>{ wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();

          merge(leaves + idx * (DIGEST_SIZE >> 1),
                intermediates + (output_offset + idx) * DIGEST_SIZE,
                mds,
                ark1,
                ark2);
        });
    }
  });

  // for computing all remaining intermediate nodes, we'll need to
  // dispatch `rounds` -many kernels, where each round is data dependent
  // on just previous one
  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(leaf_count >> 1)));

  std::vector<sycl::event> evts;
  evts.reserve(rounds);

  for (size_t r = 0; r < rounds; r++) {
    sycl::event evt = q.submit([&](sycl::handler& h) {
      if (r == 0) {
        h.depends_on(evt_0);
      } else {
        h.depends_on(evts.at(r - 1));
      }

      // these many intermediate nodes to be computed during this
      // kernel dispatch round
      const size_t offset = leaf_count >> (r + 2);
      const size_t new_wg_size = offset < wg_size ? offset : wg_size;

      if (new_wg_size >= 12) {
        scratch_mem_1d_t mds_loc =
          scratch_mem_1d_t{ sycl::range<1>{ STATE_WIDTH * 3 }, h };
        scratch_mem_1d_t ark1_loc =
          scratch_mem_1d_t{ sycl::range<1>{ NUM_ROUNDS * 3 }, h };
        scratch_mem_1d_t ark2_loc =
          scratch_mem_1d_t{ sycl::range<1>{ NUM_ROUNDS * 3 }, h };

        h.parallel_for<
          class kernelMerklizeRescuePrimeApproach3Phase1UsingScratchPad>(
          sycl::nd_range<1>{ sycl::range<1>{ offset },
                             sycl::range<1>{ new_wg_size } },
          [=](sycl::nd_item<1> it) {
            const size_t idx = it.get_global_linear_id();
            const size_t loc_idx = it.get_local_linear_id();
            sycl::group grp = it.get_group();

            if (loc_idx % 12 == loc_idx) {
              for (size_t j = 0; j < 3; j++) {
                const size_t k = j * 12 + loc_idx;
                mds_loc[k] = mds[k];
              }
            }

            if (loc_idx % 7 == loc_idx) {
              for (size_t j = 0; j < 3; j++) {
                const size_t k = j * 7 + loc_idx;
                ark1_loc[k] = ark1[k];
              }

              for (size_t j = 0; j < 3; j++) {
                const size_t k = j * 7 + loc_idx;
                ark2_loc[k] = ark2[k];
              }
            }

            // ensure all rescue prime constants written into local memory
            // and visible to all work-items in work-group
            sycl::group_barrier(grp, sycl::memory_scope::work_group);

            merge(intermediates + (offset << 1) * DIGEST_SIZE +
                    idx * (DIGEST_SIZE >> 1),
                  intermediates + (offset + idx) * DIGEST_SIZE,
                  mds_loc,
                  ark1_loc,
                  ark2_loc);
          });
      } else {
        h.parallel_for<
          class kernelMerklizeRescuePrimeApproach3Phase1UsingGlobalMem>(
          sycl::nd_range<1>{ sycl::range<1>{ offset },
                             sycl::range<1>{ new_wg_size } },
          [=](sycl::nd_item<1> it) {
            const size_t idx = it.get_global_linear_id();

            merge(intermediates + (offset << 1) * DIGEST_SIZE +
                    idx * (DIGEST_SIZE >> 1),
                  intermediates + (offset + idx) * DIGEST_SIZE,
                  mds,
                  ark1,
                  ark2);
          });
      }
    });
    evts.push_back(evt);
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
