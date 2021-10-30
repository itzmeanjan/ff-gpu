#include "bench_ff_p.hpp"
#include "ff_p.hpp"

void gen_hilbert_matrix_ff_p(sycl::queue &q, uint32_t *const mat,
                             const uint dim, const uint wg_size) {
  sycl::buffer<uint32_t, 2> b_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<uint32_t, 2, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        a_mat{b_mat, h, sycl::noinit};

    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          a_mat[r][c] = ff_p_div(1, ff_p_add(ff_p_add(r, c), 1));
        });
  });
  evt.wait();
}

void benchmark_ff_p_addition(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t one = 1;
          uint64_t op = one << (r + c);
          uint64_t elem = r + c + 1;
          for (uint64_t i = 0; i < itr_count; i++) {
            ff_p_add(elem, elem + i + op);
          }
        });
  });
  evt.wait();
}

void benchmark_ff_p_subtraction(sycl::queue &q, const uint32_t dim,
                                const uint32_t wg_size,
                                const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t one = 1;
          uint64_t op = one << (r + c);
          uint64_t elem = r + c + 1;
          for (uint64_t i = 0; i < itr_count; i++) {
            ff_p_sub(elem, elem + i + op);
          }
        });
  });
  evt.wait();
}

void benchmark_ff_p_multiplication(sycl::queue &q, const uint32_t dim,
                                   const uint32_t wg_size,
                                   const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t one = 1;
          uint64_t op = one << (r + c);
          uint64_t elem = r + c + 1;
          for (uint64_t i = 0; i < itr_count; i++) {
            ff_p_mult(elem, elem + i + op);
          }
        });
  });
  evt.wait();
}

void benchmark_ff_p_division(sycl::queue &q, const uint32_t dim,
                             const uint32_t wg_size, const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t one = 1;
          uint64_t op = one << (r + c);
          uint64_t elem = r + c + 1;
          for (uint64_t i = 0; i < itr_count; i++) {
            ff_p_div(elem, elem + i + op);
          }
        });
  });
  evt.wait();
}

void benchmark_ff_p_inversion(sycl::queue &q, const uint32_t dim,
                              const uint32_t wg_size,
                              const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t one = 1;
          uint64_t op = one << (r + c);
          uint64_t elem = r + c + 1;
          for (uint64_t i = 0; i < itr_count; i++) {
            ff_p_inv(elem + i + op);
          }
        });
  });
  evt.wait();
}

void benchmark_ff_p_exponentiation(sycl::queue &q, const uint32_t dim,
                                   const uint32_t wg_size,
                                   const uint32_t itr_count) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          uint64_t one = 1;
          uint64_t op = one << (r + c);
          uint64_t elem = r + c + 1;
          for (uint64_t i = 0; i < itr_count; i++) {
            ff_p_pow(elem + i + op, i + 1);
          }
        });
  });
  evt.wait();
}
