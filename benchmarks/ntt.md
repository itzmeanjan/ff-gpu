## Benchmarking NTT on F(2 ** 64 - 2 ** 32 + 1) Elements

I benchmark both *DFT*-style and Cooley-Tukey Forward/ Inverse **N**umber **T**heoretic **T**ransform with random elements sampled from aforementioned prime field, with vector(s) of size N = 2 ** i, i > 0.

> DFT-style NTT computes DFT matrix & uses matrix-vector multiplication !

> Note: These benchmarks include time required to transfer input/ ouput vector(s) to/ from device.

## Cooley-Tukey (Inv-)FFT

I keep benchmark results of Cooley-Tukey (Inv-)FFT implementation of NTT 👇.

### On CPU

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

dimension		     total
  128		           3560 us
  256		            614 us
  512		            756 us
 1024		           1021 us
 2048		           1668 us
 4096		           2619 us
 8192		           4927 us
16384		          10018 us
32768		          20948 us
65536		          45364 us
131072		          99838 us
262144		         219063 us
524288		         486019 us
1048576		        1068197 us
2097152		        2338715 us
4194304		        5140759 us
8388608		       11139532 us

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

dimension		     total
  128		           5473 us
  256		            685 us
  512		            841 us
 1024		           1097 us
 2048		           1570 us
 4096		           2673 us
 8192		           4958 us
16384		           9946 us
32768		          21603 us
65536		          45598 us
131072		         101294 us
262144		         221742 us
524288		         485833 us
1048576		        1070807 us
2097152		        2334953 us
4194304		        5095512 us
8388608		       11063475 us
```

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

dimension		     total
  128		           2264 us
  256		            697 us
  512		            706 us
 1024		            832 us
 2048		            813 us
 4096		           1061 us
 8192		           1440 us
16384		           2149 us
32768		           3682 us
65536		           6231 us
131072		          10816 us
262144		          25082 us
524288		          51039 us
1048576		         101666 us
2097152		         210049 us
4194304		         443639 us
8388608		         974219 us

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

dimension		     total
  128		           4408 us
  256		            783 us
  512		            633 us
 1024		            866 us
 2048		           1083 us
 4096		           1037 us
 8192		           1745 us
16384		           2432 us
32768		           4263 us
65536		           7586 us
131072		          12181 us
262144		          25187 us
524288		          50939 us
1048576		          99174 us
2097152		         209442 us
4194304		         455056 us
8388608		         954022 us
```

### On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

dimension		     total
  128		            834 us
  256		            822 us
  512		            866 us
 1024		            865 us
 2048		            916 us
 4096		            985 us
 8192		           1100 us
16384		           1393 us
32768		           1905 us
65536		           2973 us
131072		           5198 us
262144		          10156 us
524288		          19938 us
1048576		          41287 us
2097152		          86872 us
4194304		         181725 us
8388608		         382721 us

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

dimension		     total
  128		           1312 us
  256		           1041 us
  512		           1047 us
 1024		           1126 us
 2048		           1156 us
 4096		           1174 us
 8192		           1321 us
16384		           1580 us
32768		           2038 us
65536		           3170 us
131072		           5523 us
262144		          10204 us
524288		          20541 us
1048576		          41602 us
2097152		          87092 us
4194304		         184488 us
8388608		         385528 us
```

## DFT-style NTT

Benchmark results of DFT-style forward/ inverse NTT implementation on multiple hardwares.

### On CPU

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Forward NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                         7 ms
  256                         2 ms
  512                         8 ms
 1024                        40 ms

Inverse NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                         3 ms
  256                         2 ms
  512                         8 ms
 1024                        39 ms
```

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Forward NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                         7 ms
  256                         1 ms
  512                         3 ms
 1024                         8 ms

Inverse NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                         2 ms
  256                         1 ms
  512                         2 ms
 1024                         7 ms
```

### On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Forward NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                        15 ms
  256                         1 ms
  512                         2 ms
 1024                         9 ms

Inverse NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                         1 ms
  256                         2 ms
  512                         2 ms
 1024                         7 ms
```

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

Forward NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                         9 ms
  256                         4 ms
  512                         7 ms
 1024                        26 ms

Inverse NTT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                         2 ms
  256                         2 ms
  512                         6 ms
 1024                        60 ms
```
