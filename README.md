# ff-gpu
Finite Field Operations on GPGPU

## Background

In recent times, I've been interested in Finite Field operations, so I decided to implement those in SYCL DPC++, targeting accelerators ( specifically GPGPUs ).

In this repository, currently I keep addition(/ subtraction), multiplication, division, multiplicative inverse, exponentiation operation's implementations for `F(2 ** 32)`

> I plan to write better benchmarks, while improving current arithmetic operation's implementations.

## Usage

- Make sure you've Intel oneAPI toolkit installed. I found [this](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt) helpful.
- Compile & run

```bash
make
./run
```

- Clean using

```bash
make clean
```

## Benchmarks

I've implemented Hilbert Matrix computation of dimension `N x N`, where each cell is computed using field operations, so each cell is indeed a field element ∈ F(2 ** 32).

1. Running it on CPU

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
hilbert matrix generation with F(2 ** 32) elements

32   x   32			       139 ms
64   x   64			         2 ms
128  x  128			        10 ms
256  x  256			        40 ms
512  x  512			       163 ms
1024 x 1024			       655 ms
```

2. Running it on Intel Skylake-series CPU

```bash
running on Intel Xeon Processor (Skylake, IBRS)
hilbert matrix generation with F(2 ** 32) elements

32   x   32			        77 ms
64   x   64			         3 ms
128  x  128			         3 ms
256  x  256			         7 ms
512  x  512			        18 ms
1024 x 1024			        45 ms
```

3. On Intel Xe Max GPU, it's better

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]
hilbert matrix generation with F(2 ** 32) elements

32   x   32			        93 ms
64   x   64			         0 ms
128  x  128			         0 ms
256  x  256			         1 ms
512  x  512			         6 ms
1024 x 1024			        23 ms
```

4. Finally on another CPU target

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
hilbert matrix generation with F(2 ** 32) elements

32   x   32			        80 ms
64   x   64			         0 ms
128  x  128			         1 ms
256  x  256			         3 ms
512  x  512			        12 ms
1024 x 1024			        27 ms
```

---

I take an empty vector of length N and keep parallelly adding field elements starting from (i + 1), i ∈ [0, N) upto i + 1 + ITR_COUNT, ITR_COUNT = 1 << 10, also N < 2 ** 32. Finally computed sequence is stored in designated position in vector.

1. Running it on CPU

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
add subsequence of F(2 ** 32) elements

32   		    1024		     41099 us
64   		    1024		       150 us
128  		    1024		       130 us
256  		    1024		       118 us
512  		    1024		       137 us
1024 		    1024		       162 us
```

2. On Intel Skylake CPU field addition

```bash
running on Intel Xeon Processor (Skylake, IBRS)
add subsequence of F(2 ** 32) elements

32   		    1024		     28358 us
64   		    1024		        98 us
128  		    1024		       168 us
256  		    1024		       197 us
512  		    1024		       148 us
1024 		    1024		       368 us
```

3. On Intel Xe Max GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]
add subsequence of F(2 ** 32) elements

32   		    1024		     76484 us
64   		    1024		       363 us
128  		    1024		       329 us
256  		    1024		       346 us
512  		    1024		       337 us
1024 		    1024		       336 us
```

4. Finally on another CPU

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
add subsequence of F(2 ** 32) elements

32   		    1024		     22485 us
64   		    1024		       164 us
128  		    1024		       197 us
256  		    1024		       191 us
512  		    1024		       122 us
1024 		    1024		       163 us
```

---
> More benchmarks to come !
