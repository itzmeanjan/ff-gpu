#!/usr/bin/python3

import ctypes

ff_p_lib = ctypes.CDLL("../libff_p.so")


def ff_p_add(a: int, b: int):
    ff_p_lib.add.restype = ctypes.c_uint64
    ff_p_lib.add.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    return ff_p_lib.add(a, b)
