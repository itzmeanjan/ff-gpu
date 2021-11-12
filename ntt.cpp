#include "ntt.hpp"

uint64_t get_root_of_unity(uint64_t n) {
  if (n == 0) {
    // can't find root of unity for n = 0
    return 0;
  }
  if (n > TWO_ADICITY) {
    // order can't exceed 2 ** 32
    return 0;
  }

  uint64_t power = 1ul << (TWO_ADICITY - n);
  return ff_p_pow(TWO_ADIC_ROOT_OF_UNITY, power);
}

void compute_dft_matrix(sycl::queue &q, buf_2d_u64_t &mat, buf_1d_u64_t &omega,
                        const uint64_t dim, const uint64_t wg_size) {
  q.submit([&](sycl::handler &h) {
    buf_2d_u64_wr_t acc_mat{mat, h, sycl::no_init};
    buf_1d_u64_rd_t acc_omega{omega, h};
    sycl::accessor<uint64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        loc_acc_omega{sycl::range<1>{1}, h};

    h.parallel_for<class kernelPowerSeriesOfOmega>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          sycl::sub_group sg = it.get_sub_group();
          const uint64_t c = it.get_global_id(0);

          if (sycl::ext::oneapi::leader(sg)) {
            loc_acc_omega[0] = acc_omega[0];
          }

          it.barrier(sycl::access::fence_space::local_space);

          acc_mat[1][c] = ff_p_pow(loc_acc_omega[0], c);
        });
  });
}
