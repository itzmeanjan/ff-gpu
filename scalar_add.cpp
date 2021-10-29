#include "scalar_add.hpp"
#include "ff.hpp"

void add_elements(sycl::queue &q, uint32_t *const vec, const uint32_t dim,
                  const uint32_t wg_size, const uint32_t itr_count) {
  sycl::buffer<uint32_t, 1> b_vec{vec, sycl::range<1>{dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<uint32_t, 1, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        a_vec{b_vec, h};

    h.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_id(0);

          uint32_t elem = (uint32_t)idx;
          for (uint32_t i = 0; i < itr_count; i++) {
            elem = ff_add(elem, elem + i + 1);
          }

          a_vec[idx] = elem;
        });
  });
  evt.wait();
}
