#!/usr/bin/python3

import galois as gl
import math
import numpy as np

MODULUS = 2 ** 64 - 2 ** 32 + 1
gf = gl.GF(MODULUS)

GENERATOR = gf(7)  # primitive element
TWO_ADICITY = 32  # ((mod - 1) // 2 ** k) & 0b1 == 1, where k = 32
TWO_ADIC_ROOT_OF_UNITY = gf(1753635133440165772)  # 7 ** ((mod - 1) // 2 ** 32)


def get_root_of_unity(n: int):
    '''
      Returns root of unity of order 2 ** n

      Inspired from https://github.com/novifinancial/winterfell/blob/1685fa591aae9a5b426a590b7cf46c27607297de/math/src/field/traits.rs#L218-L227
    '''
    assert n != 0, "can't find root of unity for n = 0"
    assert n <= TWO_ADICITY, f"order can't exceed 2 ** {TWO_ADICITY}"

    power = 1 << (TWO_ADICITY - n)
    return TWO_ADIC_ROOT_OF_UNITY ** power


def forward_transform(vec):
    '''
        I adapted this complex DFT implementation
        https://gist.github.com/itzmeanjan/13b5efdff14f9c4877496947f1d9e449
    '''
    n = vec.shape[0]
    assert n & (n-1) == 0, "domain must be of power of two size"

    _omega = get_root_of_unity(int(math.log2(n)))

    index = np.arange(n)
    horz, vert = np.meshgrid(index, index)
    _omega_mat = _omega ** (horz * vert)

    return np.matmul(_omega_mat, vec)


def inverse_transform(vec):
    '''
        Adapted from same source, as `forward_transform` is
    '''
    n = vec.shape[0]
    assert n & (n-1) == 0, "domain must be of power of two size"

    _omega = get_root_of_unity(int(math.log2(n)))
    _omega_inv = gf(1) / _omega

    index = np.arange(n)
    horz, vert = np.meshgrid(index, index)
    _omega_mat = _omega_inv ** (horz * vert)

    n_inv = gf(1) / gf(n)
    return n_inv * np.matmul(_omega_mat, vec)


def check_correctness(n: int):
    '''
        Check `Proof of DFT/NTT correctness` section of
        https://www.nayuki.io/page/number-theoretic-transform-integer-dft
    '''
    assert n & (n-1) == 0, "domain must be of power of two size"

    _omega = get_root_of_unity(int(math.log2(n)))
    _omega_inv = gf(1) / _omega

    index = np.arange(n)
    horz, vert = np.meshgrid(index, index)
    powers = horz * vert
    _omega_mat = _omega ** powers
    _omega_inv_mat = _omega_inv ** powers

    assert np.all(gf.Identity(n) * gf(n) == np.matmul(_omega_mat,
                                                      _omega_inv_mat)), "AB = nI check failed, where A = forward DFT matrix, B = inverse DFT matrix"
    print(f"passed correctness check of AB = nI, with n = {n}")


def six_step_fft(vec):
    '''
        NTT based on six step algorithm, inspired from
        complex implementation https://doi.org/10.1109/FPT.2013.6718406
    '''
    n = vec.shape[0]
    assert n & (n-1) == 0, "domain size must be power of two"

    log_n = int(math.log2(n))
    n1 = 1 << (log_n // 2)
    n2 = n // n1

    # domain should be splitted into either two equal halves
    # or one double in size of another
    assert n1 == n2 or n2 == 2 * n1

    vec_ = vec.copy()
    # reshaping vector into n1 x n2 matrix
    vec_ = vec_.reshape((n1, n2))

    # step 1: Transpose
    vec_ = np.transpose(vec_)

    _omega_n1 = get_root_of_unity(int(math.log2(n1)))

    # step 2: n2-many (parallel) n1-point FFT
    for j1 in range(n2):

        # --- perform DFT for elements along selected row ---
        tmp = gf.Zeros(n1)
        for k2 in range(n1):
            for j2 in range(n1):
                tmp[k2] += (vec_[j1][j2] * (_omega_n1 ** (j2 * k2)))
        vec_[j1] = tmp

        # @note It's lot faster to replace above block
        # with vec_[j1] = forward_transform(vec_[j1])

    _omega = get_root_of_unity(log_n)

    # step 3: Multiplication by Twiddles
    for k2 in range(n2):
        for j1 in range(n1):
            vec_[k2][j1] *= (_omega ** (j1 * k2))

    # step 4: Transpose
    vec_ = np.transpose(vec_)

    _omega_n2 = get_root_of_unity(int(math.log2(n2)))

    # step 5: n1-many (parallel) n2-point FFT
    for j1 in range(n1):

        # --- perform DFT for elements along selected row ---
        tmp = gf.Zeros(n2)
        for k2 in range(n2):
            for j2 in range(n2):
                tmp[k2] += (vec_[j1][j2] * (_omega_n2 ** (j2 * k2)))
        vec_[j1] = tmp

        # @note It's lot faster to replace above block
        # with vec_[j1] = forward_transform(vec_[j1])

    # step 6: Transpose
    vec_ = np.transpose(vec_)

    # reshape back to vector
    return vec_.reshape(n)


def six_step_ifft(vec):
    '''
        Inverse NTT based on six step algorithm,
        adapted from same source as specified
        in `six_step_fft` function
    '''
    n = vec.shape[0]
    assert n & (n-1) == 0, "domain size must be power of two"

    log_n = int(math.log2(n))
    n1 = 1 << (log_n // 2)
    n2 = n // n1

    # domain should be splitted into either two equal halves
    # or one double in size of another
    assert n1 == n2 or n2 == 2 * n1

    vec_ = vec.copy()
    # reshaping vector into n1 x n2 matrix
    vec_ = vec_.reshape((n1, n2))

    # step 1: Transpose
    vec_ = np.transpose(vec_)

    _inv_omega_n1 = gf(1) / get_root_of_unity(int(math.log2(n1)))

    # step 2: n2-many (parallel) n1-point FFT
    for j1 in range(n2):

        # --- perform IDFT for elements along selected row ---
        tmp = gf.Zeros(n1)
        for k2 in range(n1):
            for j2 in range(n1):
                tmp[k2] += (vec_[j1][j2] * (_inv_omega_n1 ** (j2 * k2)))
        vec_[j1] = tmp

        # @note It's lot faster to replace above block
        # with vec_[j1] = inverse_transform(vec_[j1])

    _omega = gf(1) / get_root_of_unity(log_n)

    # step 3: Multiplication by Twiddles
    for k2 in range(n2):
        for j1 in range(n1):
            vec_[k2][j1] *= (_omega ** (j1 * k2))

    # step 4: Transpose
    vec_ = np.transpose(vec_)

    _inv_omega_n2 = gf(1) / get_root_of_unity(int(math.log2(n2)))

    # step 5: n1-many (parallel) n2-point FFT
    for j1 in range(n1):

        # --- perform DFT for elements along selected row ---
        tmp = gf.Zeros(n2)
        for k2 in range(n2):
            for j2 in range(n2):
                tmp[k2] += (vec_[j1][j2] * (_inv_omega_n2 ** (j2 * k2)))
        vec_[j1] = tmp

        # @note It's lot faster to replace above block
        # with vec_[j1] = inverse_transform(vec_[j1])

    # step 6: Transpose
    vec_ = np.transpose(vec_)
    _inv_n = gf(1) / gf(n)

    # reshape back to vector
    return vec_.reshape(n) * _inv_n


def main():
    for i in range(3, 9):
        n = 1 << i

        v = gf.Random(n)

        # Plain & simple (I)DFT
        v_dft = forward_transform(v)
        v_idft = inverse_transform(v_dft)

        # Six step algorithm based (I)FFT
        v_fft = six_step_fft(v)
        v_ifft = six_step_ifft(v_fft)

        assert np.all(v == v_idft)
        assert np.all(v == v_ifft)
        assert np.all(v_dft == v_fft)

        print(f'passed (I){{D,F}}FT for {n:>4}-sized random domain')


if __name__ == '__main__':
    main()
    check_correctness(1 << 8)
