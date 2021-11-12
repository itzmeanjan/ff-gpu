#pragma once
#include "ff_p.hpp"

typedef sycl::buffer<uint64_t, 2> buf_2d_u64_t;
typedef sycl::buffer<uint64_t, 1> buf_1d_u64_t;

typedef sycl::accessor<uint64_t, 2, sycl::access::mode::read,
                       sycl::access::target::global_buffer>
    buf_2d_u64_rd_t;
typedef sycl::accessor<uint64_t, 2, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
    buf_2d_u64_wr_t;
typedef sycl::accessor<uint64_t, 2, sycl::access::mode::read_write,
                       sycl::access::target::global_buffer>
    buf_2d_u64_rw_t;
typedef sycl::accessor<uint64_t, 1, sycl::access::mode::read,
                       sycl::access::target::global_buffer>
    buf_1d_u64_rd_t;
typedef sycl::accessor<uint64_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::global_buffer>
    buf_1d_u64_rw_t;

inline constexpr uint64_t TWO_ADICITY = 32ul;
inline constexpr uint64_t TWO_ADIC_ROOT_OF_UNITY = 1753635133440165772ul;

uint64_t get_root_of_unity(uint64_t n);

void compute_dft_matrix(sycl::queue &q, buf_2d_u64_t &mat, buf_1d_u64_t &omega,
                        const uint64_t dim, const uint64_t wg_size);

void compute_matrix_vector_multiplication(sycl::queue &q, buf_2d_u64_t &mat,
                                          buf_1d_u64_t &vec, buf_1d_u64_t &res,
                                          const uint64_t dim,
                                          const uint64_t wg_size);
