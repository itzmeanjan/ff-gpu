#include <test_merkle_tree.hpp>

void
test_merklize(sycl::queue& q)
{
  const size_t leaf_count = 1024;

  sycl::ulong* leaves = static_cast<sycl::ulong*>(
    sycl::malloc_shared(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));
  sycl::ulong* intermediates_a = static_cast<sycl::ulong*>(
    sycl::malloc_shared(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));
  sycl::ulong* intermediates_b = static_cast<sycl::ulong*>(
    sycl::malloc_shared(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));
  sycl::ulong4* mds = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1 = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2 = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));

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

  // following two kernel dispatches will compute same merkle tree ( i.e. all
  // intermediate nodes ) and finally root of two computed trees will be checked
  // against each other

  // host sychronization in function itself !
  merklize_approach_1(
    q, leaves, intermediates_a, leaf_count, 4, mds, ark1, ark2);

  // host sychronization in function itself !
  merklize_approach_2(
    q, leaves, intermediates_b, leaf_count, 8, mds, ark1, ark2);

  // asserting that first digest ( which spans 256 -bit ) in intermediate
  // node holder memory allocation is never touched !
  //
  // four consequtive memory locations are checked because
  // each rescue prime digest has width of 256 -bit i.e.
  // 4 field elements wide
  assert(*(intermediates_a + 0) == 0);
  assert(*(intermediates_a + 1) == 0);
  assert(*(intermediates_a + 2) == 0);
  assert(*(intermediates_a + 3) == 0);

  // same as above, this is done just to ensure that assertion kernel
  // never touches first four field elements, because it's never
  // desired/ required to be touched
  assert(*(intermediates_b + 0) == 0);
  assert(*(intermediates_b + 1) == 0);
  assert(*(intermediates_b + 2) == 0);
  assert(*(intermediates_b + 3) == 0);

  // asserting root of merkle tree, where each computed by two different
  // kernels
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
