#!/usr/bin/python3

'''
    Read https://www.nayuki.io/page/number-theoretic-transform-integer-dft for
    more information
'''

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
    n = vec.shape[0]
    assert n & (n-1) == 0, "domain must be of power of two size"

    _omega = get_root_of_unity(int(math.log2(n)))

    index = np.arange(n)
    horz, vert = np.meshgrid(index, index)
    _omega_mat = _omega ** (horz * vert)

    return np.matmul(_omega_mat, vec)


def inverse_transform(vec):
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


def main():
    for i in range(3, 9):
        n = 1 << i

        v = gf.Random(n)
        v_fwd = forward_transform(v)
        v_inv = inverse_transform(v_fwd)

        for i in range(n):
            assert v[i] == v_inv[i], f"expected {v[i]}, found {v_inv[i]}"

        print(
            f'passed forward/ inverse transform for {n:>4}-sized random domain')


if __name__ == '__main__':
    main()
    check_correctness(1 << 8)
