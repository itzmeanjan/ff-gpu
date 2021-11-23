#include "bench_ff.hpp"
#include "bench_ff_p.hpp"
#include "bench_ntt.hpp"
#include "bench_rescue_prime.hpp"
#include <iomanip>

using namespace sycl;

const uint32_t N = 1 << 10;
const uint32_t B = 1 << 7;

int main(int argc, char **argv) {
// device selection is based on flag provided to compiler
// such as `clang++ main.cpp {...}.cpp -D{CPU,GPU,HOST,DEFAULT} -fsycl
// -std=c++20 -Wall`
//
// but someone using make utility, should be invoking it as
// `DEVICE=cpu|gpu|host make <target>`
#if defined CPU
  device d{cpu_selector{}};
#elif defined GPU
  device d{gpu_selector{}};
#elif defined HOST
  device d{host_selector{}};
#else
  device d{default_selector{}};
#endif

  queue q{d};

  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;
  std::cout << "hilbert matrix generation with F(2 ** 32) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t\t" << std::setw(10) << "total" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    uint32_t *mat = (uint32_t *)malloc(sizeof(uint32_t) * dim * dim);

    tp start = std::chrono::steady_clock::now();
    gen_hilbert_matrix_ff(q, mat, dim, B);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << tm << " us" << std::endl;

    std::free(mat);
  }

  std::cout << "\naddition on F(2 ** 32) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_addition(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\nsubtraction on F(2 ** 32) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_subtraction(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\nmultiplication on F(2 ** 32) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_multiplication(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\ndivision on F(2 ** 32) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_division(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\ninversion on F(2 ** 32) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_inversion(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\nexponentiation on F(2 ** 32) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_exponentiation(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout
      << "\nhilbert matrix generation with F(2**64 - 2**32 + 1) elements 👇\n"
      << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t\t" << std::setw(10) << "total" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    uint32_t *mat = (uint32_t *)malloc(sizeof(uint32_t) * dim * dim);

    tp start = std::chrono::steady_clock::now();
    gen_hilbert_matrix_ff_p(q, mat, dim, B);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << tm << " us" << std::endl;

    std::free(mat);
  }

  std::cout << "\naddition on F(2**64 - 2**32 + 1) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_p_addition(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\nsubtraction on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_p_subtraction(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\nmultiplication on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_p_multiplication(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\ndivision on F(2**64 - 2**32 + 1) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_p_division(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\ninversion on F(2**64 - 2**32 + 1) elements 👇\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_p_inversion(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\nexponentiation on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_ff_p_exponentiation(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << N
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * N) << " ns" << std::endl;
  }

  std::cout << "\nrescue prime hash on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg"
            << "\t\t" << std::setw(20) << "op/s" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    tp start = std::chrono::steady_clock::now();
    benchmark_hash_elements(q, dim, B, N);
    tp end = std::chrono::steady_clock::now();

    int64_t tm =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << 1
              << "\t\t" << std::setw(15) << std::right << tm << " us"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * 1) << " us"
              << "\t\t" << std::setw(15) << std::right
              << 1e6 / ((double)tm / (double)(dim * dim * 1)) << std::endl;
  }

  std::cout << "\nForward NTT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(5) << "dimension"
            << "\t\t" << std::setw(10) << "total" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    int64_t tm = benchmark_forward_transform(q, dim, B);

    std::cout << std::setw(5) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << tm << " ms" << std::endl;
  }

  std::cout << "\nInverse NTT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(5) << "dimension"
            << "\t\t" << std::setw(10) << "total" << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    int64_t tm = benchmark_inverse_transform(q, dim, B);

    std::cout << std::setw(5) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << tm << " ms" << std::endl;
  }

  std::cout << "\nCooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(10) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = B; dim <= (1ul << 23); dim <<= 1) {
    int64_t tm = benchmark_cooley_tukey_fft(q, dim, B);

    std::cout << std::setw(8) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << (float)tm / 1000.f << " ms" << std::endl;
  }

  std::cout << "\nCooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(10) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = B; dim <= (1ul << 23); dim <<= 1) {
    int64_t tm = benchmark_cooley_tukey_ifft(q, dim, B);

    std::cout << std::setw(8) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << (float)tm / 1000.f << " ms" << std::endl;
  }

  std::cout
      << "\nSquare Matrix Transposition on F(2**64 - 2**32 + 1) elements 👇\n"
      << std::endl;
  std::cout << std::setw(10) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = B; dim <= (1ul << 12); dim <<= 1) {
    int64_t tm = benchmark_matrix_transposition(q, dim, B);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(15) << std::right
              << (float)tm / 1000.f << " ms" << std::endl;
  }

  std::cout
      << "\nTwiddle Factor Multiplication on F(2**64 - 2**32 + 1) elements 👇\n"
      << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint pow = 6; pow <= 10; pow++) {
    int64_t tm =
        benchmark_twiddle_factor_multiplication(q, 1 << pow, 1 << pow, 1 << 6);

    std::cout << std::setw(5) << std::left << (1 << pow) << "x" << std::setw(5)
              << std::right << (1 << pow) << "\t\t" << std::setw(15)
              << std::right << (float)tm / 1000.f << " ms" << std::endl;

    tm = benchmark_twiddle_factor_multiplication(q, 1 << pow, 1 << (pow + 1),
                                                 1 << 6);

    std::cout << std::setw(5) << std::left << (1 << pow) << "x" << std::setw(5)
              << std::right << (1 << (pow + 1)) << "\t\t" << std::setw(15)
              << std::right << (float)tm / 1000.f << " ms" << std::endl;
  }

  std::cout << "\nSix-Step FFT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 12; dim <= 24; dim++) {
    int64_t tm = benchmark_six_step_fft(q, 1ul << dim, 1 << 6);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step IFFT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 12; dim <= 24; dim++) {
    int64_t tm = benchmark_six_step_ifft(q, 1ul << dim, 1 << 6);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  return 0;
}
