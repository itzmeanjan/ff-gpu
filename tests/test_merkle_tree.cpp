#include <test_merkle_tree.hpp>

void
test_merklize(sycl::queue& q)
{
  const size_t leaf_count = 16;

  sycl::ulong* leaves = static_cast<sycl::ulong*>(
    sycl::malloc_shared(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));
  sycl::ulong* intermediates_a = static_cast<sycl::ulong*>(
    sycl::malloc_shared(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));
  sycl::ulong* intermediates_b = static_cast<sycl::ulong*>(
    sycl::malloc_shared(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));
  sycl::ulong16* mds = static_cast<sycl::ulong16*>(
    sycl::malloc_shared(sizeof(sycl::ulong16) * STATE_WIDTH, q));
  sycl::ulong16* ark1 = static_cast<sycl::ulong16*>(
    sycl::malloc_shared(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark2 = static_cast<sycl::ulong16*>(
    sycl::malloc_shared(sizeof(sycl::ulong16) * NUM_ROUNDS, q));

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

    for (uint64_t i = 0; i < leaf_count * DIGEST_SIZE; i++) {
      *(leaves + i) = static_cast<sycl::ulong>(dis(gen));
    }
  }

  // just to ensure that very first digest cell is never touched,
  // it should be zeroed, as being set here !
  q.memset(intermediates_a, 0, sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE);
  q.memset(intermediates_b, 0, sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE);
  q.wait();

  prepare_mds(mds);
  prepare_ark1(ark1);
  prepare_ark2(ark2);

  // host sychronization in function itself !
  merklize(q, leaves, intermediates_a, leaf_count, 2, mds, ark1, ark2);

  {
    const size_t output_offset = leaf_count >> 1;

    sycl::event evt_0 = q.submit([&](sycl::handler& h) {
      h.parallel_for<class kernelMerklizeRescuePrimePhase0Test>(
        sycl::nd_range<1>{ sycl::range<1>{ output_offset },
                           sycl::range<1>{ output_offset } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();

          merge(leaves + idx * (DIGEST_SIZE >> 1),
                intermediates_b + (output_offset + idx) * DIGEST_SIZE,
                mds,
                ark1,
                ark2);
        });
    });

    // manually compute merkle root !
    sycl::event evt_1 = q.submit([&](sycl::handler& h) {
      h.depends_on(evt_0);
      h.single_task([=]() {
        for (size_t idx = 0; idx < (leaf_count >> 2); idx++) {
          merge(intermediates_b + (leaf_count >> 1) * DIGEST_SIZE +
                  idx * (DIGEST_SIZE >> 1),
                intermediates_b + ((leaf_count >> 2) + idx) * DIGEST_SIZE,
                mds,
                ark1,
                ark2);
        }

        for (size_t idx = 0; idx < (leaf_count >> 3); idx++) {
          merge(intermediates_b + (leaf_count >> 2) * DIGEST_SIZE +
                  idx * (DIGEST_SIZE >> 1),
                intermediates_b + ((leaf_count >> 3) + idx) * DIGEST_SIZE,
                mds,
                ark1,
                ark2);
        }

        for (size_t idx = 0; idx < (leaf_count >> 4); idx++) {
          merge(intermediates_b + (leaf_count >> 3) * DIGEST_SIZE +
                  idx * (DIGEST_SIZE >> 1),
                intermediates_b + ((leaf_count >> 4) + idx) * DIGEST_SIZE,
                mds,
                ark1,
                ark2);
        }
      });
    });

    evt_1.wait();
  }

  // asserting that first digest in interemediate node holding
  // allocation is never touched !
  //
  // four consequtive memory locations are checked because
  // each rescue prime digest has width of 256 -bit which is
  // 4 field elements wide
  assert(*(intermediates_a + 0) == 0);
  assert(*(intermediates_a + 1) == 0);
  assert(*(intermediates_a + 2) == 0);
  assert(*(intermediates_a + 3) == 0);

  // check that root of merkle tree matches when computed
  // using actual kernel ( which is being tested ) with
  // manually computed intermediates ( hierarchically )
  assert(*(intermediates_a + 4) == *(intermediates_b + 4));
  assert(*(intermediates_a + 5) == *(intermediates_b + 5));
  assert(*(intermediates_a + 6) == *(intermediates_b + 6));
  assert(*(intermediates_a + 7) == *(intermediates_b + 7));

  // deallocate all memory resources
  sycl::free(leaves, q);
  sycl::free(intermediates_a, q);
  sycl::free(intermediates_b, q);
  sycl::free(mds, q);
  sycl::free(ark1, q);
  sycl::free(ark2, q);
}
