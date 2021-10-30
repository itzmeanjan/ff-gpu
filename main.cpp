#include "bench_ff.hpp"
#include "bench_ff_p.hpp"
#include <chrono>
#include <iomanip>

using namespace sycl;

const uint32_t N = 1 << 10;
const uint32_t B = 1 << 5;

typedef std::chrono::_V2::steady_clock::time_point tp;

int main(int argc, char **argv) {
  device d{default_selector{}};
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

  return 0;
}
