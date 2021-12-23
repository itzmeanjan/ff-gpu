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
random_hash_state(sycl::ulong16* state, const sycl::ulong n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

  for (sycl::ulong i = 0; i < n; i++) {
    sycl::ulong16 s = sycl::ulong16(dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    dis(gen) % MOD,
                                    0,
                                    0,
                                    0,
                                    0);
    *(state + i) = s;
  }
}

void
test_sbox(sycl::queue& q)
{
  sycl::ulong16* in =
    static_cast<sycl::ulong16*>(sycl::malloc_shared(sizeof(sycl::ulong16), q));
  sycl::ulong16* out =
    static_cast<sycl::ulong16*>(sycl::malloc_shared(sizeof(sycl::ulong16), q));

  random_hash_state(in, 1);

  // ensuring host sychronization by using `.wait()`
  q.single_task([=]() { *out = apply_sbox(*in); }).wait();

  sycl::ulong16 in_state = *in;
  sycl::ulong16 out_state = *out;

  assert(operate(q, in_state.s0(), ALPHA, Op::power) == out_state.s0());
  assert(operate(q, in_state.s1(), ALPHA, Op::power) == out_state.s1());
  assert(operate(q, in_state.s2(), ALPHA, Op::power) == out_state.s2());
  assert(operate(q, in_state.s3(), ALPHA, Op::power) == out_state.s3());
  assert(operate(q, in_state.s4(), ALPHA, Op::power) == out_state.s4());
  assert(operate(q, in_state.s5(), ALPHA, Op::power) == out_state.s5());
  assert(operate(q, in_state.s6(), ALPHA, Op::power) == out_state.s6());
  assert(operate(q, in_state.s7(), ALPHA, Op::power) == out_state.s7());
  assert(operate(q, in_state.s8(), ALPHA, Op::power) == out_state.s8());
  assert(operate(q, in_state.s9(), ALPHA, Op::power) == out_state.s9());
  assert(operate(q, in_state.sA(), ALPHA, Op::power) == out_state.sA());
  assert(operate(q, in_state.sB(), ALPHA, Op::power) == out_state.sB());
  assert(0 == out_state.sC());
  assert(0 == out_state.sD());
  assert(0 == out_state.sE());
  assert(0 == out_state.sF());

  sycl::free(in, q);
  sycl::free(out, q);
}

void
test_inv_sbox(sycl::queue& q)
{
  sycl::ulong16* in =
    static_cast<sycl::ulong16*>(sycl::malloc_shared(sizeof(sycl::ulong16), q));
  sycl::ulong16* out =
    static_cast<sycl::ulong16*>(sycl::malloc_shared(sizeof(sycl::ulong16), q));

  random_hash_state(in, 1);

  // ensuring host sychronization by using `.wait()`
  q.single_task([=]() { *out = apply_inv_sbox(*in); }).wait();

  sycl::ulong16 in_state = *in;
  sycl::ulong16 out_state = *out;

  assert(operate(q, in_state.s0(), INV_ALPHA, Op::power) == out_state.s0());
  assert(operate(q, in_state.s1(), INV_ALPHA, Op::power) == out_state.s1());
  assert(operate(q, in_state.s2(), INV_ALPHA, Op::power) == out_state.s2());
  assert(operate(q, in_state.s3(), INV_ALPHA, Op::power) == out_state.s3());
  assert(operate(q, in_state.s4(), INV_ALPHA, Op::power) == out_state.s4());
  assert(operate(q, in_state.s5(), INV_ALPHA, Op::power) == out_state.s5());
  assert(operate(q, in_state.s6(), INV_ALPHA, Op::power) == out_state.s6());
  assert(operate(q, in_state.s7(), INV_ALPHA, Op::power) == out_state.s7());
  assert(operate(q, in_state.s8(), INV_ALPHA, Op::power) == out_state.s8());
  assert(operate(q, in_state.s9(), INV_ALPHA, Op::power) == out_state.s9());
  assert(operate(q, in_state.sA(), INV_ALPHA, Op::power) == out_state.sA());
  assert(operate(q, in_state.sB(), INV_ALPHA, Op::power) == out_state.sB());
  assert(0 == out_state.sC());
  assert(0 == out_state.sD());
  assert(0 == out_state.sE());
  assert(0 == out_state.sF());

  sycl::free(in, q);
  sycl::free(out, q);
}

void
test_permutation(sycl::queue& q)
{
  sycl::ulong16* out =
    static_cast<sycl::ulong16*>(sycl::malloc_shared(sizeof(sycl::ulong16), q));
  sycl::ulong16* mds = static_cast<sycl::ulong16*>(
    sycl::malloc_shared(sizeof(sycl::ulong16) * STATE_WIDTH, q));
  sycl::ulong16* ark1 = static_cast<sycl::ulong16*>(
    sycl::malloc_shared(sizeof(sycl::ulong16) * NUM_ROUNDS, q));
  sycl::ulong16* ark2 = static_cast<sycl::ulong16*>(
    sycl::malloc_shared(sizeof(sycl::ulong16) * NUM_ROUNDS, q));

  sycl::ulong16 expected = sycl::ulong16(10809974140050983728ull % MOD,
                                         6938491977181280539ull % MOD,
                                         8834525837561071698ull % MOD,
                                         6854417192438540779ull % MOD,
                                         4476630872663101667ull % MOD,
                                         6292749486700362097ull % MOD,
                                         18386622366690620454ull % MOD,
                                         10614098972800193173ull % MOD,
                                         7543273285584849722ull % MOD,
                                         9490898458612615694ull % MOD,
                                         9030271581669113292ull % MOD,
                                         10101107035874348250ull % MOD,
                                         0,
                                         0,
                                         0,
                                         0);

  prepare_mds(mds);
  prepare_ark1(ark1);
  prepare_ark2(ark2);

  // ensuring host sychronization by using `.wait()`
  q.single_task([=]() {
     *out = apply_rescue_permutation(
       sycl::ulong16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0),
       mds,
       ark1,
       ark2);
   })
    .wait();

  sycl::ulong16 out_state = *out;

  assert(out_state.s0() == expected.s0());
  assert(out_state.s1() == expected.s1());
  assert(out_state.s2() == expected.s2());
  assert(out_state.s3() == expected.s3());
  assert(out_state.s4() == expected.s4());
  assert(out_state.s5() == expected.s5());
  assert(out_state.s6() == expected.s6());
  assert(out_state.s7() == expected.s7());
  assert(out_state.s8() == expected.s8());
  assert(out_state.s9() == expected.s9());
  assert(out_state.sA() == expected.sA());
  assert(out_state.sB() == expected.sB());
  assert(out_state.sC() == expected.sC());
  assert(out_state.sD() == expected.sD());
  assert(out_state.sE() == expected.sE());
  assert(out_state.sF() == expected.sF());

  sycl::free(out, q);
  sycl::free(mds, q);
  sycl::free(ark1, q);
  sycl::free(ark2, q);
}
