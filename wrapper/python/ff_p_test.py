#!/usr/bin/python3

import galois as gl
from ff_p import *
from time import time
from random import randint, seed

MIN = 0
MAX = 2 ** 64 - 1
MOD = 2 ** 64 - 2**32 + 1
ROUNDS = 1 << 7


def main():
    seed(time())
    gf = gl.GF(MOD)

    start = time()
    for _ in range(ROUNDS):
        a = randint(MIN, MAX)
        b = randint(MIN, MAX)
        assert (gf(a % MOD) + gf(b % MOD)) == ff_p_add(a, b)

    print(
        f'passed {ROUNDS} randomized {"addition":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = randint(MIN, MAX)
        b = randint(MIN, MAX)
        assert (gf(a % MOD) - gf(b % MOD)) == ff_p_sub(a, b)

    print(
        f'passed {ROUNDS} randomized {"subtraction":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = randint(MIN, MAX)
        b = randint(MIN, MAX)
        assert (gf(a % MOD) * gf(b % MOD)) == ff_p_multiply(a, b)

    print(
        f'passed {ROUNDS} randomized {"multiplication":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = randint(MIN, MAX)
        b = randint(MIN, MAX)
        assert (gf(a % MOD) ** b) == ff_p_exponentiate(a, b)

    print(
        f'passed {ROUNDS} randomized {"exponentiation":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    one = gf(1)
    for _ in range(ROUNDS):
        a = randint(MIN, MAX)
        assert (one / gf(a % MOD)) == ff_p_inverse(a)

    print(
        f'passed {ROUNDS} randomized {"inversion":>16} tests !\t{time() - start:>16.2f} s')

    start = time()
    for _ in range(ROUNDS):
        a = randint(MIN, MAX)
        b = randint(MIN, MAX)
        assert (gf(a % MOD) / gf(b % MOD)) == ff_p_divide(a, b)

    print(
        f'passed {ROUNDS} randomized {"division":>16} tests !\t{time() - start:>16.2f} s')


if __name__ == '__main__':
    main()
