#include "ff_p.hpp"

extern "C" void
make_queue(void** wq)
{
  sycl::default_selector d_sel{};
  sycl::device d{ d_sel };
  sycl::context c{ d };
  sycl::queue* q = new sycl::queue{ c, d };

  *wq = q;
}

extern "C" uint64_t
add(void* wq, uint64_t a, uint64_t b)
{
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  uint64_t* res = (uint64_t*)sycl::malloc_shared(sizeof(uint64_t), *q);
  q->single_task([=]() { *res = ff_p_add(a, b) % MOD; }).wait();

  return *res;
}

extern "C" uint64_t
sub(void* wq, uint64_t a, uint64_t b)
{
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  uint64_t* res = (uint64_t*)sycl::malloc_shared(sizeof(uint64_t), *q);
  q->single_task([=]() { *res = ff_p_sub(a, b) % MOD; }).wait();

  return *res;
}

extern "C" uint64_t
multiply(void* wq, uint64_t a, uint64_t b)
{
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  uint64_t* res = (uint64_t*)sycl::malloc_shared(sizeof(uint64_t), *q);
  q->single_task([=]() { *res = ff_p_mult(a, b) % MOD; }).wait();

  return *res;
}

extern "C" uint64_t
exponentiate(void* wq, uint64_t a, uint64_t b)
{
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  uint64_t* res = (uint64_t*)sycl::malloc_shared(sizeof(uint64_t), *q);
  q->single_task([=]() { *res = ff_p_pow(a, b) % MOD; }).wait();

  return *res;
}

extern "C" uint64_t
inverse(void* wq, uint64_t a)
{
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  uint64_t* res = (uint64_t*)sycl::malloc_shared(sizeof(uint64_t), *q);
  q->single_task([=]() { *res = ff_p_inv(a) % MOD; }).wait();

  return *res;
}

extern "C" uint64_t
divide(void* wq, uint64_t a, uint64_t b)
{
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  uint64_t* res = (uint64_t*)sycl::malloc_shared(sizeof(uint64_t), *q);
  q->single_task([=]() { *res = ff_p_div(a, b) % MOD; }).wait();

  return *res;
}
