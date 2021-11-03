#!/usr/bin/python3

import galois as gl
from ff_p import *

MOD = 2 ** 64 - 2**32 + 1


def main():
    gf = gl.GF(MOD)

    for _ in range(0, 1 << 7):
        a = gf.Random()
        b = gf.Random()
        assert (a + b) == ff_p_add(a.item(), b.item())

    print('passed addition tests !')

    for _ in range(0, 1 << 7):
        a = gf.Random()
        b = gf.Random()
        assert (a - b) == ff_p_sub(a.item(), b.item())

    print('passed subtraction tests !')


if __name__ == '__main__':
    main()
