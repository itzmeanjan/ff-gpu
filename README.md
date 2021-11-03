# ff-gpu
Finite Field Operations on GPGPU

## Background

In recent times, I've been interested in Finite Field operations, so I decided to implement few fields in SYCL DPC++, targeting accelerators ( specifically GPGPUs ).

In this repository, currently I keep implementation of two finite field's arithmetic operations, accompanied with relevant benchmarks on both CPU, GPGPU.

- Binary Extension Field `F(2 ** 32)`
- Prime Field `F(2 ** 64 - 2 ** 32 + 1)`

## Benchmarks

### Prerequisites

- Make sure you've `make`, `clang-format` and `dpcpp`/ `clang++` installed
- You can build DPC++ compiler from source, check [here](https://intel.github.io/llvm-docs/GetStartedGuide.html#prerequisites)
- Or you may want to download pre-compiled Intel oneAPI toolkit, includes both compilers, check [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) [**recommended**]

### Run

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
make
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

---

I run benchmarking code on both **CPU** and **GPGPU**.

- [Arithmetics on `F(2 ** 32)`](./benchmarks/ff.md)
- [Arithmetics on `F(2 ** 64 - 2 ** 32 + 1)`](./benchmarks/ff_p.md)

> More to come !
