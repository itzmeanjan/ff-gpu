# ff-gpu
Finite Field Operations on GPGPU

## Background

In recent times, I've been interested in Finite Field operations, so I decided to implement few fields in SYCL DPC++, targeting accelerators ( specifically GPGPUs ).

In this repository, currently I keep implementation of two finite field's arithmetic operations, accompanied with relevant benchmarks on both CPU, GPGPU.

- Binary Extension Field `F(2 ** 32)`
- Prime Field `F(2 ** 64 - 2 ** 32 + 1)`

I've also written following implementations, along with benchmark results on CPU, GPU.

- Rescue Prime Hash function
- Six step algorithm based (I)*N*umber *T*heoretic *T*ransform
- Cooley-Tukey algorithm based (I)NTT
- DFT-style (I)NTT

## Prerequisites

- Make sure you've `make`, `clang-format` and `dpcpp`/ `clang++` installed
- You can build DPC++ compiler from source, check [here](https://intel.github.io/llvm-docs/GetStartedGuide.html#prerequisites)
- Or you may want to download pre-compiled Intel oneAPI toolkit, includes both compilers, check [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) [**recommended**]

## Benchmarks

- I'm running

```bash
$ lsb_release -d

Description:    Ubuntu 20.04.3 LTS
```

- As compiler, I'm using

```bash
$ dpcpp --version

Intel(R) oneAPI DPC++/C++ Compiler 2021.3.0 (2021.3.0.20210619)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/intel/oneapi/compiler/2021.3.0/linux/bin
```

- Compile, link & run

```bash
make # JIT kernel compilation, for AOT read below
./run
```

- Clean using

```bash
make clean
```

- Format source, if required

```bash
make format
```

- Aforementioned steps compile kernels JIT, but if target device is already known it's better to compile them AOT, which saves some time in runtime, though compiled binary is device specific now. 

- I provide AOT kernel compilation recipe for CPUs using `avx2` instructions. You can check whether your CPU supports that.

```bash
lscpu | grep -i avx
```

- AOT Compilation targeting CPU can be invoked using

```bash
make aot_cpu
```

- I also provide recipe for AOT compiling kernels targeting Intel Iris Xe Max Graphics

```bash
make aot_gpu
```

> You may have some other hardware, consider taking a look at AOT compilation [guidelines](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html) & make necessary changes in `Makefile`.

---

I run benchmarking code on both **CPU** and **GPGPU**, keeping results 👇

- [Arithmetics on `F(2 ** 32)`](./benchmarks/ff.md)
- [Arithmetics on `F(2 ** 64 - 2 ** 32 + 1)`](./benchmarks/ff_p.md)
- [Rescue Prime Hash on `F(2 ** 64 - 2 ** 32 + 1)`](./benchmarks/rescue_prime.md)
- [Six step algorithm-based (I)NTT on `F(2 ** 64 - 2 ** 32 + 1)`](./benchmarks/ntt.md#six-step-algorithm-based-intt)
- [Cooley-Tukey (I)NTT on `F(2 ** 64 - 2 ** 32 + 1)`](./benchmarks/ntt.md#cooley-tukey-inv-fft)
- [DFT-style (I)NTT on `F(2 ** 64 - 2 ** 32 + 1)`](./benchmarks/ntt.md#dft-style-ntt)

## Tests

You can run basic test cases using

```bash
make test
```

There's another set of randomised test cases, which asserts results *( obtained from my prime field implementation )* with another finite field implementation module, written in `Python`, named `galois`.

For running those, I suggest you first compile shared object using

```bash
make genlib
```

After that you can follow next steps [here](wrapper/python).
