#!/usr/bin/python3

'''
  Make sure `make genlib` is run to generate shared object first,
  which is used for invoking prime field arithmetic functions
  on available default accelerator
'''

import ctypes

ff_p_lib = ctypes.CDLL("../libff_p.so")


def ff_p_add(a: int, b: int):
    '''
    Modular addition of two prime field elements
    '''
    ff_p_lib.add.restype = ctypes.c_uint64
    ff_p_lib.add.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    return ff_p_lib.add(a, b)


def ff_p_sub(a: int, b: int):
    '''
    Modular subtraction of two prime field elements
    '''
    ff_p_lib.sub.restype = ctypes.c_uint64
    ff_p_lib.sub.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    return ff_p_lib.sub(a, b)


def ff_p_multiply(a: int, b: int):
    '''
    Modular multiplication of two prime field elements
    '''

    ff_p_lib.multiply.restype = ctypes.c_uint64
    ff_p_lib.multiply.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    return ff_p_lib.multiply(a, b)


def ff_p_exponentiate(a: int, b: int):
    '''
      Modular exponentiation of one prime field element
      by exponent (second operand)
    '''
    ff_p_lib.exponentiate.restype = ctypes.c_uint64
    ff_p_lib.exponentiate.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    return ff_p_lib.exponentiate(a, b)


def ff_p_inverse(a: int):
    '''
    Muliplicative identity of non-zero prime field element
    '''

    ff_p_lib.inverse.restype = ctypes.c_uint64
    ff_p_lib.inverse.argtypes = [ctypes.c_uint64]

    return ff_p_lib.inverse(a)


def ff_p_divide(a: int, b: int):
    '''
    Modular division of one prime field element
    by another one
    '''

    ff_p_lib.divide.restype = ctypes.c_uint64
    ff_p_lib.divide.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    return ff_p_lib.divide(a, b)
