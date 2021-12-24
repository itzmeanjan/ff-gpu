#include "bench_merkle_tree.hpp"
#include "bench_ntt.hpp"
#include "bench_rescue_prime.hpp"
#include <iomanip>

using namespace sycl;

const uint32_t N = 1 << 10;
const uint32_t B = 1 << 7;

int
main(int argc, char** argv)
{
// device selection is based on flag provided to compiler
// such as `clang++ main.cpp {...}.cpp -D{CPU,GPU,HOST,DEFAULT} -fsycl
// -std=c++20 -Wall`
//
// but someone using make utility, should be invoking it as
// `DEVICE=cpu|gpu|host make <target>`
#if defined CPU
  device d{ cpu_selector{} };
#elif defined GPU
  device d{ gpu_selector{} };
#elif defined HOST
  device d{ host_selector{} };
#else
  device d{ default_selector{} };
#endif

  queue q{ d, { sycl::property::queue::enable_profiling() } };

  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  std::cout << "\nRescue prime hash on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg"
            << "\t\t" << std::setw(20) << "op/s" << std::endl;

  for (uint dim = B; dim <= (1ul << 12); dim <<= 1) {
    // note iteration count is set to 1, so each work-item
    // only hashes input one time
    //
    // time in nanoseconds --- be careful !
    uint64_t tm = benchmark_hash_elements(q, dim, 1ul << 6, 1);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << 1
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * 1) << " ns"
              << "\t\t" << std::setw(15) << std::right
              << 1e9 / ((double)tm / (double)(dim * dim * 1)) << std::endl;
  }

  std::cout
    << "\nRescue prime merge hashes on F(2**64 - 2**32 + 1) elements 👇\n"
    << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(10) << "iterations"
            << "\t\t" << std::setw(15) << "total"
            << "\t\t" << std::setw(20) << "avg"
            << "\t\t" << std::setw(20) << "op/s" << std::endl;

  for (uint dim = B; dim <= (1ul << 12); dim <<= 1) {
    // time in nanoseconds --- beware !
    uint64_t tm = benchmark_merge(q, dim, 1ul << 6, 1);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t" << std::setw(8) << std::right << 1
              << "\t\t" << std::setw(15) << std::right << tm << " ns"
              << "\t\t" << std::setw(15) << std::right
              << (double)tm / (double)(dim * dim * 1) << " ns"
              << "\t\t" << std::setw(15) << std::right
              << 1e9 / ((double)tm / (double)(dim * dim * 1)) << std::endl;
  }

  std::cout
    << "\nMerklize using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇\n"
    << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = (1ul << 20); dim <= (1ul << 23); dim <<= 1) {
    // time in nanoseconds --- beware !
    uint64_t tm = benchmark_merklize(q, dim, 1ul << 5);

    std::cout << std::setw(11) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << tm * 1e-6 << " ms" << std::endl;
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

  for (uint dim = 1ul << 16; dim <= (1ul << 23); dim <<= 1) {
    int64_t tm = benchmark_cooley_tukey_fft(q, dim, B);

    std::cout << std::setw(8) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << (float)tm / 1000.f << " ms" << std::endl;
  }

  std::cout << "\nCooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(10) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 1ul << 16; dim <= (1ul << 23); dim <<= 1) {
    int64_t tm = benchmark_cooley_tukey_ifft(q, dim, B);

    std::cout << std::setw(8) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << (float)tm / 1000.f << " ms" << std::endl;
  }

  std::cout << "\nSix-Step FFT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 24; dim++) {
    int64_t tm = benchmark_six_step_fft(q, 1ul << dim, 1 << 6);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step IFFT on F(2**64 - 2**32 + 1) elements 👇\n"
            << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 24; dim++) {
    int64_t tm = benchmark_six_step_ifft(q, 1ul << dim, 1 << 6);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (float)tm / 1000.f << " ms"
              << std::endl;
  }

  return 0;
}
