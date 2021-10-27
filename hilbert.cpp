#include "hilbert.hpp"
#include "ff.hpp"

void gen_hilbert_matrix(sycl::queue &q, uint32_t *const mat, const uint dim,
                        const uint wg_size) {
  sycl::buffer<uint32_t, 2> b_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<uint32_t, 2, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        a_mat{b_mat, h, sycl::noinit};

    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint32_t r = it.get_global_id(0);
          const uint32_t c = it.get_global_id(1);

          a_mat[r][c] = ff_div(1, ff_add(ff_add(r, c), 1));
        });
  });
  evt.wait();
}
