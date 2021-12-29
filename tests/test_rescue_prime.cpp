#include "test_rescue_prime.hpp"

void
test_alphas(sycl::queue& q)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

  uint64_t e = dis(gen);
  uint64_t e_exp = operate(q, e, ALPHA, Op::power);

  assert(e == operate(q, e_exp, INV_ALPHA, Op::power));
}

void
random_hash_state(sycl::ulong4* const state, const sycl::ulong n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

  for (sycl::ulong i = 0; i < n; i++) {
    sycl::ulong4 v0 = sycl::ulong4(
      dis(gen) % MOD, dis(gen) % MOD, dis(gen) % MOD, dis(gen) % MOD);
    sycl::ulong4 v1 = sycl::ulong4(
      dis(gen) % MOD, dis(gen) % MOD, dis(gen) % MOD, dis(gen) % MOD);
    sycl::ulong4 v2 = sycl::ulong4(
      dis(gen) % MOD, dis(gen) % MOD, dis(gen) % MOD, dis(gen) % MOD);

    *(state + i * 3 + 0) = v0;
    *(state + i * 3 + 1) = v1;
    *(state + i * 3 + 2) = v2;
  }
}

void
test_sbox(sycl::queue& q)
{
  sycl::ulong4* in = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * 3, q));
  sycl::ulong4* out = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * 3, q));

  random_hash_state(in, 1);

  // ensuring host sychronization by using `.wait()`
  q.single_task([=]() { apply_sbox(in, out); }).wait();

  assert(operate(q, in[0].x(), ALPHA, Op::power) == out[0].x());
  assert(operate(q, in[0].y(), ALPHA, Op::power) == out[0].y());
  assert(operate(q, in[0].z(), ALPHA, Op::power) == out[0].z());
  assert(operate(q, in[0].w(), ALPHA, Op::power) == out[0].w());
  assert(operate(q, in[1].x(), ALPHA, Op::power) == out[1].x());
  assert(operate(q, in[1].y(), ALPHA, Op::power) == out[1].y());
  assert(operate(q, in[1].z(), ALPHA, Op::power) == out[1].z());
  assert(operate(q, in[1].w(), ALPHA, Op::power) == out[1].w());
  assert(operate(q, in[2].x(), ALPHA, Op::power) == out[2].x());
  assert(operate(q, in[2].y(), ALPHA, Op::power) == out[2].y());
  assert(operate(q, in[2].z(), ALPHA, Op::power) == out[2].z());
  assert(operate(q, in[2].w(), ALPHA, Op::power) == out[2].w());

  sycl::free(in, q);
  sycl::free(out, q);
}

void
test_inv_sbox(sycl::queue& q)
{
  sycl::ulong4* in = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * 3, q));
  sycl::ulong4* out = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * 3, q));

  random_hash_state(in, 1);

  // ensuring host sychronization by using `.wait()`
  q.single_task([=]() { apply_inv_sbox(in, out); }).wait();

  assert(operate(q, in[0].x(), INV_ALPHA, Op::power) == out[0].x());
  assert(operate(q, in[0].y(), INV_ALPHA, Op::power) == out[0].y());
  assert(operate(q, in[0].z(), INV_ALPHA, Op::power) == out[0].z());
  assert(operate(q, in[0].w(), INV_ALPHA, Op::power) == out[0].w());
  assert(operate(q, in[1].x(), INV_ALPHA, Op::power) == out[1].x());
  assert(operate(q, in[1].y(), INV_ALPHA, Op::power) == out[1].y());
  assert(operate(q, in[1].z(), INV_ALPHA, Op::power) == out[1].z());
  assert(operate(q, in[1].w(), INV_ALPHA, Op::power) == out[1].w());
  assert(operate(q, in[2].x(), INV_ALPHA, Op::power) == out[2].x());
  assert(operate(q, in[2].y(), INV_ALPHA, Op::power) == out[2].y());
  assert(operate(q, in[2].z(), INV_ALPHA, Op::power) == out[2].z());
  assert(operate(q, in[2].w(), INV_ALPHA, Op::power) == out[2].w());

  sycl::free(in, q);
  sycl::free(out, q);
}

void
test_permutation(sycl::queue& q)
{
  sycl::ulong4* out = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * 3, q));
  sycl::ulong4* mds = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * STATE_WIDTH * 3, q));
  sycl::ulong4* ark1 = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));
  sycl::ulong4* ark2 = static_cast<sycl::ulong4*>(
    sycl::malloc_shared(sizeof(sycl::ulong4) * NUM_ROUNDS * 3, q));

  sycl::ulong4 expected[3] = { sycl::ulong4(10809974140050983728ull,
                                            6938491977181280539ull,
                                            8834525837561071698ull,
                                            6854417192438540779ull),
                               sycl::ulong4(4476630872663101667ull,
                                            6292749486700362097ull,
                                            18386622366690620454ull,
                                            10614098972800193173ull),
                               sycl::ulong4(7543273285584849722ull,
                                            9490898458612615694ull,
                                            9030271581669113292ull,
                                            10101107035874348250ull) };

  prepare_mds(mds);
  prepare_ark1(ark1);
  prepare_ark2(ark2);

  // ensuring host sychronization by using `.wait()`
  q.single_task([=]() {
     sycl::ulong4 state[3] = { sycl::ulong4(0, 1, 2, 3),
                               sycl::ulong4(4, 5, 6, 7),
                               sycl::ulong4(8, 9, 10, 11) };

     apply_rescue_permutation(state, mds, ark1, ark2, out);
   })
    .wait();

  assert(out[0].x() == expected[0].x());
  assert(out[0].y() == expected[0].y());
  assert(out[0].z() == expected[0].z());
  assert(out[0].w() == expected[0].w());
  assert(out[1].x() == expected[1].x());
  assert(out[1].y() == expected[1].y());
  assert(out[1].z() == expected[1].z());
  assert(out[1].w() == expected[1].w());
  assert(out[2].x() == expected[2].x());
  assert(out[2].y() == expected[2].y());
  assert(out[2].z() == expected[2].z());
  assert(out[2].w() == expected[2].w());

  sycl::free(out, q);
  sycl::free(mds, q);
  sycl::free(ark1, q);
  sycl::free(ark2, q);
}
