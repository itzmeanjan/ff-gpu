#!/usr/bin/python3

import galois as gl
from ff_p import *
from time import time

MOD = 2 ** 64 - 2**32 + 1
ROUNDS = 1 << 7


def main():
    gf = gl.GF(MOD)

    start = time()
    for _ in range(ROUNDS):
        a = gf.Random()
        b = gf.Random()
        assert (a + b) == ff_p_add(a.item(), b.item())

    print(
        f'passed {ROUNDS} randomized {"addition":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = gf.Random()
        b = gf.Random()
        assert (a - b) == ff_p_sub(a.item(), b.item())

    print(
        f'passed {ROUNDS} randomized {"subtraction":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = gf.Random()
        b = gf.Random()
        assert (a * b) == ff_p_multiply(a.item(), b.item())

    print(
        f'passed {ROUNDS} randomized {"multiplication":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = gf.Random()
        b = gf.Random()
        assert (a ** b.item()) == ff_p_exponentiate(a.item(), b.item())

    print(
        f'passed {ROUNDS} randomized {"exponentiation":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    one = gf(1)
    for _ in range(ROUNDS):
        a = gf.Random()
        assert (one / a) == ff_p_inverse(a.item())

    print(
        f'passed {ROUNDS} randomized {"inversion":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = gf.Random()
        b = gf.Random()
        assert (a / b) == ff_p_divide(a.item(), b.item())

    print(
        f'passed {ROUNDS} randomized {"division":>16} tests !\t{time() - start:>16.2f} s')


if __name__ == '__main__':
    main()
