#include "ff_p.hpp"

extern "C" uint64_t add(uint64_t a, uint64_t b) {
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};

  uint64_t *res = (uint64_t *)sycl::malloc_shared(sizeof(uint64_t), q);

  q.memset(res, 0, sizeof(uint64_t));
  q.single_task([=]() { *res = ff_p_add(a, b) % MOD; });
  q.wait();

  return *res;
}
