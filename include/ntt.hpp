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
typedef sycl::accessor<uint64_t, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
    buf_1d_u64_wr_t;
typedef sycl::accessor<uint64_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::global_buffer>
    buf_1d_u64_rw_t;

inline constexpr uint64_t TWO_ADICITY = 32ul;
inline constexpr uint64_t TWO_ADIC_ROOT_OF_UNITY = 1753635133440165772ul;

extern SYCL_EXTERNAL uint64_t get_root_of_unity(uint64_t n);

sycl::event compute_dft_matrix(sycl::queue &q, buf_2d_u64_t &mat,
                               buf_1d_u64_t &omega, const uint64_t dim,
                               const uint64_t wg_size);

sycl::event compute_matrix_vector_multiplication(
    sycl::queue &q, buf_2d_u64_t &mat, buf_1d_u64_t &vec, buf_1d_u64_t &res,
    const uint64_t dim, const uint64_t wg_size);

sycl::event compute_vector_scalar_multilication(sycl::queue &q,
                                                buf_1d_u64_t &vec,
                                                const uint64_t factor,
                                                const uint64_t dim,
                                                const uint64_t wg_size);

void forward_transform(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size);

void inverse_transform(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size);

sycl::event compute_omega(sycl::queue &q, buf_1d_u64_t &omega,
                          const uint64_t domain_size);

sycl::event compute_omega_inv(sycl::queue &q, buf_1d_u64_t &omega_inv,
                              const uint64_t domain_size);

extern SYCL_EXTERNAL uint64_t bit_rev(uint64_t v, uint64_t max_bit_width);

extern SYCL_EXTERNAL uint64_t rev_all_bits(uint64_t n);

extern SYCL_EXTERNAL uint64_t permute_index(uint64_t idx, uint64_t size);

void cooley_tukey_fft(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                      const uint64_t dim, const uint64_t wg_size);

void cooley_tukey_ifft(sycl::queue &q, buf_1d_u64_t &vec, buf_1d_u64_t &res,
                       const uint64_t dim, const uint64_t wg_size);

// Computes in-place parallel *square* matrix transposition
//
// @note if matrix is not square, consider padding empty rows,
// as it's easy to do in row-major indexing
//
// @note matrix is represented as 1d array
sycl::event matrix_transpose(sycl::queue &q, uint64_t *vec, const uint64_t dim,
                             const uint64_t wg_size,
                             std::vector<sycl::event> evts);

// Performs parallel in-place FFT based on Cooley-Tukey style while taking USM
// based memory pointer as input data location
//
// @note For kernel execution ordering consider using events vector parameter
// and return event type properly, otherwise it'll result into data race, as
// dependency needs to be managed manually due to no use of SYCL buffers
sycl::event row_fft(sycl::queue &q, uint64_t *vec, uint64_t *omega,
                    const uint64_t dim, const uint64_t wg_size,
                    std::vector<sycl::event> evts);
