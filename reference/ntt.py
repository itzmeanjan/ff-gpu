#!/usr/bin/python3

from functools import reduce
import galois as gl
import math

MODULUS = 2 ** 64 - 2 ** 32 + 1
gf = gl.GF(MODULUS)

GENERATOR = gf(7)  # primitive element
TWO_ADICITY = 32  # ((mod - 1) // 2 ** k) & 0b1 == 1, where k = 32
TWO_ADIC_ROOT_OF_UNITY = gf(1753635133440165772)  # 7 ** ((mod - 1) // 2 ** 32)


def get_root_of_unity(n: int):
    '''
      Returns root of unity of order 2 ** n
    '''
    assert n != 0, "can't find root of unity for n = 0"
    assert n <= TWO_ADICITY, f"order can't exceed 2 ** {TWO_ADICITY}"

    power = 1 << (TWO_ADICITY - n)
    return TWO_ADIC_ROOT_OF_UNITY ** power


def forward_transform(vec):
    n = len(vec)
    assert n & (n-1) == 0, "domain must be of power of two size"

    _omega = get_root_of_unity(int(math.log2(n)))

    return [reduce(lambda acc, cur: acc + cur, [vec[j] * (_omega ** (i * j)) for j in range(n)], gf(0)) for i in range(n)]


def inverse_transform(vec):
    n = len(vec)
    assert n & (n-1) == 0, "domain must be of power of two size"

    _omega = get_root_of_unity(int(math.log2(n)))
    _omega_inv = gf(1) / _omega
    n_inv = gf(1) / gf(n)

    return [n_inv * reduce(lambda acc, cur: acc + cur, [vec[j] * (_omega_inv ** (i * j)) for j in range(n)], gf(0)) for i in range(n)]


def random_elements(n: int):
    return [gf.Random() for i in range(n)]


def main():
    for i in range(3, 8):
        n = 1 << i

        v = random_elements(n)
        v_fwd = forward_transform(v)
        v_inv = inverse_transform(v_fwd)

        for i in range(n):
            assert v[i] == v_inv[i], f"expected {v[i]}, found {v_inv[i]}"

        print(f'passed forward/ inverse transform for {n:>4}-sized random domain')


if __name__ == '__main__':
    main()
