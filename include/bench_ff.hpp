#pragma once
#include "ff.hpp"
#include <benchmark/benchmark.h>
#include <random>

// Benchmark finite field addition on CPU system ( single core )
static void
bench_ff_add(benchmark::State& state)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (auto _ : state) {
    const uint64_t a = dis(gen);
    const uint64_t b = dis(gen);

    benchmark::DoNotOptimize(ff::add(a, b));
  }

  state.SetItemsProcessed(state.iterations());
}

// Benchmark finite field subtraction on CPU system ( single core )
static void
bench_ff_sub(benchmark::State& state)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (auto _ : state) {
    const uint64_t a = dis(gen);
    const uint64_t b = dis(gen);

    benchmark::DoNotOptimize(ff::sub(a, b));
  }

  state.SetItemsProcessed(state.iterations());
}

// Benchmark finite field multiplication on CPU system ( single core )
static void
bench_ff_mult(benchmark::State& state)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (auto _ : state) {
    const uint64_t a = dis(gen);
    const uint64_t b = dis(gen);

    benchmark::DoNotOptimize(ff::mult(a, b));
  }

  state.SetItemsProcessed(state.iterations());
}

// Benchmark finite field exponentiation on CPU system ( single core )
static void
bench_ff_pow(benchmark::State& state)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (auto _ : state) {
    const uint64_t a = dis(gen);
    const uint64_t b = dis(gen);

    benchmark::DoNotOptimize(ff::pow(a, b));
  }

  state.SetItemsProcessed(state.iterations());
}

// Benchmark finite field inversion on CPU system ( single core )
static void
bench_ff_inv(benchmark::State& state)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (auto _ : state) {
    const uint64_t a = dis(gen);

    benchmark::DoNotOptimize(ff::inv(a));
  }

  state.SetItemsProcessed(state.iterations());
}

// Benchmark finite field division on CPU system ( single core )
static void
bench_ff_div(benchmark::State& state)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (auto _ : state) {
    const uint64_t a = dis(gen);
    const uint64_t b = dis(gen);

    benchmark::DoNotOptimize(ff::div(a, b));
  }

  state.SetItemsProcessed(state.iterations());
}
