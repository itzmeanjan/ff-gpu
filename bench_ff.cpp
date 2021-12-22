#include "bench_ff.hpp"
#include "ff.hpp"

void
gen_hilbert_matrix_ff(sycl::queue& q,
                      uint32_t* const mat,
                      const uint dim,
                      const uint wg_size)
{
  sycl::buffer<uint32_t, 2> b_mat{ mat, sycl::range<2>{ dim, dim } };

  auto evt = q.submit([&](sycl::handler& h) {
    sycl::accessor<uint32_t,
                   2,
                   sycl::access::mode::write,
                   sycl::access::target::global_buffer>
      a_mat{ b_mat, h, sycl::no_init };

    h.parallel_for(sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                                      sycl::range<2>{ 1, wg_size } },
                   [=](sycl::nd_item<2> it) {
                     const uint32_t r = it.get_global_id(0);
                     const uint32_t c = it.get_global_id(1);

                     a_mat[r][c] = ff_div(1, ff_add(ff_add(r, c), 1));
                   });
  });
  evt.wait();
}

void
benchmark_ff_addition(sycl::queue& q,
                      const uint32_t dim,
                      const uint32_t wg_size,
                      const uint32_t itr_count)
{
  auto evt = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                                      sycl::range<2>{ 1, wg_size } },
                   [=](sycl::nd_item<2> it) {
                     const uint32_t r = it.get_global_id(0);
                     const uint32_t c = it.get_global_id(1);

                     uint32_t elem = r + c + 1;
                     for (uint32_t i = 0; i < itr_count; i++) {
                       ff_add(elem, elem + i + 1);
                     }
                   });
  });
  evt.wait();
}

void
benchmark_ff_subtraction(sycl::queue& q,
                         const uint32_t dim,
                         const uint32_t wg_size,
                         const uint32_t itr_count)
{
  auto evt = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                                      sycl::range<2>{ 1, wg_size } },
                   [=](sycl::nd_item<2> it) {
                     const uint32_t r = it.get_global_id(0);
                     const uint32_t c = it.get_global_id(1);

                     uint32_t elem = r + c + 1;
                     for (uint32_t i = 0; i < itr_count; i++) {
                       ff_sub(elem, elem + i + 1);
                     }
                   });
  });
  evt.wait();
}

void
benchmark_ff_multiplication(sycl::queue& q,
                            const uint32_t dim,
                            const uint32_t wg_size,
                            const uint32_t itr_count)
{
  auto evt = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                                      sycl::range<2>{ 1, wg_size } },
                   [=](sycl::nd_item<2> it) {
                     const uint32_t r = it.get_global_id(0);
                     const uint32_t c = it.get_global_id(1);

                     uint32_t elem = r + c + 1;
                     for (uint32_t i = 0; i < itr_count; i++) {
                       ff_mult(elem, elem + i + 1);
                     }
                   });
  });
  evt.wait();
}

void
benchmark_ff_division(sycl::queue& q,
                      const uint32_t dim,
                      const uint32_t wg_size,
                      const uint32_t itr_count)
{
  auto evt = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                                      sycl::range<2>{ 1, wg_size } },
                   [=](sycl::nd_item<2> it) {
                     const uint32_t r = it.get_global_id(0);
                     const uint32_t c = it.get_global_id(1);

                     uint32_t elem = r + c + 1;
                     for (uint32_t i = 0; i < itr_count; i++) {
                       ff_div(elem, elem + i + 1);
                     }
                   });
  });
  evt.wait();
}

void
benchmark_ff_inversion(sycl::queue& q,
                       const uint32_t dim,
                       const uint32_t wg_size,
                       const uint32_t itr_count)
{
  auto evt = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                                      sycl::range<2>{ 1, wg_size } },
                   [=](sycl::nd_item<2> it) {
                     const uint32_t r = it.get_global_id(0);
                     const uint32_t c = it.get_global_id(1);

                     uint32_t elem = r + c + 1;
                     for (uint32_t i = 0; i < itr_count; i++) {
                       ff_inv(elem + i + 1);
                     }
                   });
  });
  evt.wait();
}

void
benchmark_ff_exponentiation(sycl::queue& q,
                            const uint32_t dim,
                            const uint32_t wg_size,
                            const uint32_t itr_count)
{
  auto evt = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                                      sycl::range<2>{ 1, wg_size } },
                   [=](sycl::nd_item<2> it) {
                     const uint32_t r = it.get_global_id(0);
                     const uint32_t c = it.get_global_id(1);

                     uint32_t elem = r + c + 1;
                     for (uint32_t i = 0; i < itr_count; i++) {
                       ff_pow(elem + i + 1, i + 1);
                     }
                   });
  });
  evt.wait();
}
