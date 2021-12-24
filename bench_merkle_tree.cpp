#include <bench_merkle_tree.hpp>
#include <merkle_tree.hpp>
#include <random>

uint64_t
benchmark_merklize(sycl::queue& q,
                   const size_t leaf_count,
                   const size_t wg_size)
{
  sycl::ulong* leaves_h = static_cast<sycl::ulong*>(
    sycl::malloc_host(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));
  sycl::ulong* leaves_d = static_cast<sycl::ulong*>(
    sycl::malloc_device(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));

  sycl::ulong* intermediates = static_cast<sycl::ulong*>(
    sycl::malloc_device(sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE, q));

  sycl::ulong16* mds_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * STATE_WIDTH, q));
  sycl::ulong16* mds_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * STATE_WIDTH, q));

  sycl::ulong16* ark1_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark1_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * NUM_ROUNDS, q));

  sycl::ulong16* ark2_h = static_cast<sycl::ulong16*>(
    sycl::malloc_host(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark2_d = static_cast<sycl::ulong16*>(
    sycl::malloc_device(sizeof(sycl::ulong16) * NUM_ROUNDS, q));

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1ul, MOD);

    for (uint64_t i = 0; i < leaf_count * DIGEST_SIZE; i++) {
      *(leaves_h + i) = static_cast<sycl::ulong>(dis(gen));
    }
  }

  prepare_mds(mds_h);
  prepare_ark1(ark1_h);
  prepare_ark2(ark2_h);

  sycl::event evt_0 = q.memcpy(
    leaves_d, leaves_h, sizeof(sycl::ulong) * leaf_count * DIGEST_SIZE);
  sycl::event evt_1 =
    q.memcpy(mds_d, mds_h, sizeof(sycl::ulong16) * STATE_WIDTH);
  sycl::event evt_2 =
    q.memcpy(ark1_d, ark1_h, sizeof(sycl::ulong16) * NUM_ROUNDS);
  sycl::event evt_3 =
    q.memcpy(ark2_d, ark2_h, sizeof(sycl::ulong16) * NUM_ROUNDS);

  // wait for host to device copies to complete !
  q.wait();

  // this itself does host synchronization
  uint64_t ts = merklize(
    q, leaves_d, intermediates, leaf_count, wg_size, mds_d, ark1_d, ark2_d);

  sycl::free(leaves_h, q);
  sycl::free(leaves_d, q);
  sycl::free(intermediates, q);
  sycl::free(mds_h, q);
  sycl::free(ark1_h, q);
  sycl::free(ark2_h, q);
  sycl::free(mds_d, q);
  sycl::free(ark1_d, q);
  sycl::free(ark2_d, q);

  return ts;
}