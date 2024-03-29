#pragma once
#include "ff.hpp"
#include <cassert>
#include <random>

static void
test_ff_add()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  const uint64_t r = dis(gen);
  const uint64_t t0 = ff::MOD - 1ull;
  const uint64_t t1 = 4294967294ull;

  assert(r == ff::to_canonical(ff::add(r, 0ull)));
  assert(5ull == ff::to_canonical(ff::add(3ull, 2ull)));
  assert(0ull == ff::to_canonical(ff::add(t0, 1ull)));
  assert(1ull == ff::to_canonical(ff::add(t0, 2ull)));
  assert(t1 == ff::to_canonical(ff::add(t0, 0xffffffffull)));
}

static void
test_ff_sub()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  const uint64_t r = dis(gen);
  const uint64_t t = ff::MOD - 2ull;

  assert(r == ff::to_canonical(ff::sub(r, 0ull)));
  assert(2ull == ff::to_canonical(ff::sub(5ull, 3ull)));
  assert(t == ff::to_canonical(ff::sub(3ull, 5ull)));

  const size_t rounds = 1ull << 13;
  for (size_t i = 0; i < rounds; i++) {
    const uint64_t r0 = dis(gen);
    const uint64_t r1 = dis(gen);

    assert(ff::to_canonical(ff::sub(r0, r1)) ==
           ff::to_canonical(ff::add(r0, ff::sub(0, r1))));
  }
}

static inline uint64_t
multiply_with_addition(const uint64_t a, const uint64_t b)
{
  uint64_t a_ = 0;
  for (uint64_t i = 0; i < b; i++) {
    a_ = ff::to_canonical(ff::add(a, a_));
  }
  return a_;
}

static void
test_ff_mul()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  const uint64_t r = dis(gen);
  const uint64_t t = ff::MOD - 1ull;
  const uint64_t v = (ff::MOD + 1ull) >> 1;

  assert(0ull == ff::to_canonical(ff::mult(r, 0ull)));
  assert(r == ff::to_canonical(ff::mult(r, 1ull)));
  assert(15ull == ff::to_canonical(ff::mult(3ull, 5ull)));
  assert(1ull == ff::to_canonical(ff::mult(t, t)));
  assert(ff::MOD - 2ull == ff::to_canonical(ff::mult(t, 2ull)));
  assert(ff::MOD - 4ull == ff::to_canonical(ff::mult(t, 4ull)));
  assert(1ull == ff::to_canonical(ff::mult(v, 2ull)));
  assert(
    ff::to_canonical(ff::mult(r, 2ull) == ff::to_canonical(ff::add(r, r))));

  // test whether multiplication & addition produces
  // same results by performing n * a == (a + a + ... n-times ... + a + a)
  const size_t rounds = 1 << 10;
  for (uint64_t i = 0; i < rounds; i++) {
    const uint64_t r = dis(gen);
    assert(multiply_with_addition(r, i) == ff::to_canonical(ff::mult(r, i)));
  }
}

static inline uint64_t
exp_with_mult(const uint64_t a, const uint64_t b)
{
  uint64_t t0 = 1;
  for (uint64_t i = 0; i < b; i++) {
    t0 = ff::to_canonical(ff::mult(a, t0));
  }
  return t0;
}

static void
test_ff_exp()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  const uint64_t r = dis(gen);

  assert(ff::to_canonical(ff::pow(0ull, 0ull) == 1ull));
  assert(ff::to_canonical(ff::pow(0ull, 1ull) == 0ull));
  assert(ff::to_canonical(ff::pow(1ull, 0ull) == 1ull));
  assert(ff::to_canonical(ff::pow(1ull, 1ull) == 1ull));
  assert(ff::to_canonical(ff::pow(1ull, 2ull) == 1ull));
  assert(ff::to_canonical(ff::pow(r, 3ull) ==
                          ff::to_canonical(ff::mult(r, ff::mult(r, r)))));

  const size_t rounds = 1ull << 10;
  for (uint64_t i = 0; i < rounds; i++) {
    const uint64_t r = dis(gen);
    assert(exp_with_mult(r, i) == ff::to_canonical(ff::pow(r, i)));
  }
}

static void
test_ff_inv()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  const uint64_t r = dis(gen);

  assert(1ull == ff::to_canonical(ff::inv(1ull)));
  assert(0ull == ff::to_canonical(ff::inv(0ull)));
  assert(ff::to_canonical(ff::pow(r, ff::MOD - 2ull) ==
                          ff::to_canonical(ff::inv(r))));

  const size_t rounds = 1ull << 10;
  for (uint64_t i = 0; i < rounds; i++) {
    const uint64_t r = dis(gen);
    assert(ff::to_canonical(ff::mult(r, ff::to_canonical(ff::inv(r)))) == 1ull);
  }
}
