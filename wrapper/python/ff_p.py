#!/usr/bin/python3

'''
  Make sure `make genlib` is run to generate shared object first,
  which is used for invoking prime field arithmetic functions
  on available default accelerator
'''

import ctypes
from genericpath import exists
from posixpath import abspath


class ff_p:
    so_path: str = '../libff_p.so'
    sycl_q: ctypes.c_void_p = None
    so_lib: ctypes.CDLL = None

    def __init__(self) -> None:
        '''
        Creates an instance of `ff_p` class, along with backend resource(s)
        like setting up SYCL queue where compute jobs will be submitted,
        getting shared library ready for forwarding future function invocations
        '''
        if not exists(self.so_path):
            raise Exception(
                f'failed to find shared library `{abspath(self.so_path)}`')

        self.so_lib = ctypes.CDLL(self.so_path)

        self.so_lib.make_queue.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.sycl_q = ctypes.c_void_p()
        self.so_lib.make_queue(ctypes.byref(self.sycl_q))

        if self.sycl_q.value == None:
            raise Exception(f'failed to get default SYCL queue')

    def add(self, a: int, b: int) -> int:
        '''
        Modular addition of two prime field elements
        '''
        self.so_lib.add.restype = ctypes.c_uint64
        self.so_lib.add.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

        return self.so_lib.add(self.sycl_q, a, b)

    def sub(self, a: int, b: int) -> int:
        '''
        Modular subtraction of two prime field elements
        '''
        self.so_lib.sub.restype = ctypes.c_uint64
        self.so_lib.sub.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

        return self.so_lib.sub(self.sycl_q, a, b)

    def multiply(self, a: int, b: int) -> int:
        '''
        Modular multiplication of two prime field elements
        '''

        self.so_lib.multiply.restype = ctypes.c_uint64
        self.so_lib.multiply.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

        return self.so_lib.multiply(self.sycl_q, a, b)

    def exponentiate(self, a: int, b: int) -> int:
        '''
        Modular exponentiation of one prime field element
        by exponent (second operand)
        '''
        self.so_lib.exponentiate.restype = ctypes.c_uint64
        self.so_lib.exponentiate.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

        return self.so_lib.exponentiate(self.sycl_q, a, b)

    def inverse(self, a: int) -> int:
        '''
        Muliplicative identity of non-zero prime field element
        '''

        self.so_lib.inverse.restype = ctypes.c_uint64
        self.so_lib.inverse.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

        return self.so_lib.inverse(self.sycl_q, a)

    def divide(self, a: int, b: int) -> int:
        '''
        Modular division of one prime field element
        by another one
        '''

        self.so_lib.divide.restype = ctypes.c_uint64
        self.so_lib.divide.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

        return self.so_lib.divide(self.sycl_q, a, b)
