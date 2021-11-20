## Benchmarking NTT on F(2 ** 64 - 2 ** 32 + 1) Elements

I benchmark *DFT*-style, *Cooley-Tukey* Forward/ Inverse NTT and *Six-step algorithm* based (I)NTT with random elements sampled from aforementioned prime field, with vector(s) of size N = 2 ** i, i > 0.

> DFT-style NTT computes DFT matrix & uses matrix-vector multiplication - *so it's not efficient* !

## Six Step Algorithm based (I)NTT

I ran my (I)NTT implementation, based on six-step (I)FFT, on multiple CPU(s)/ GPU(s) & benchmark results are presented below.

### On CPU

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                        12.341 ms
     8192                         5.134 ms
    16384                         4.486 ms
    32768                         4.773 ms
    65536                         5.252 ms
   131072                         6.898 ms
   262144                        10.651 ms
   524288                        42.982 ms
  1048576                        23.412 ms
  2097152                        87.797 ms
  4194304                        83.066 ms
  8388608                       216.817 ms
 16777216                        331.26 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                          8.53 ms
     8192                         1.355 ms
    16384                         1.763 ms
    32768                           2.4 ms
    65536                         3.051 ms
   131072                         4.338 ms
   262144                         7.251 ms
   524288                        30.862 ms
  1048576                        22.251 ms
  2097152                        81.422 ms
  4194304                         85.92 ms
  8388608                       210.183 ms
 16777216                       313.521 ms
```

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         9.849 ms
     8192                         1.324 ms
    16384                         1.669 ms
    32768                         3.312 ms
    65536                         5.509 ms
   131072                         9.569 ms
   262144                        16.477 ms
   524288                        34.057 ms
  1048576                         65.39 ms
  2097152                           136 ms
  4194304                       261.959 ms
  8388608                         585.3 ms
 16777216                       1239.32 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         7.985 ms
     8192                         1.569 ms
    16384                         2.276 ms
    32768                         3.804 ms
    65536                         6.225 ms
   131072                        10.296 ms
   262144                        16.475 ms
   524288                         36.86 ms
  1048576                        66.395 ms
  2097152                       139.875 ms
  4194304                       268.897 ms
  8388608                       585.569 ms
 16777216                       1198.84 ms
```

```bash
running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         8.809 ms
     8192                         1.022 ms
    16384                         1.456 ms
    32768                         2.562 ms
    65536                         4.785 ms
   131072                         9.404 ms
   262144                        15.212 ms
   524288                        28.652 ms
  1048576                        55.302 ms
  2097152                       126.669 ms
  4194304                       262.377 ms
  8388608                       592.043 ms
 16777216                       1231.08 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                          6.43 ms
     8192                         0.874 ms
    16384                          1.36 ms
    32768                         2.366 ms
    65536                         4.275 ms
   131072                         7.561 ms
   262144                         12.46 ms
   524288                        28.938 ms
  1048576                        55.794 ms
  2097152                       126.037 ms
  4194304                       263.788 ms
  8388608                       592.917 ms
 16777216                        1234.2 ms
```

### On GPU


```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         0.689 ms
     8192                         0.783 ms
    16384                         0.933 ms
    32768                         1.448 ms
    65536                         2.159 ms
   131072                         3.703 ms
   262144                         7.186 ms
   524288                        15.541 ms
  1048576                          30.9 ms
  2097152                        67.108 ms
  4194304                       134.443 ms
  8388608                       290.295 ms
 16777216                       596.584 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         0.915 ms
     8192                         1.026 ms
    16384                         1.178 ms
    32768                         1.714 ms
    65536                         2.464 ms
   131072                          4.05 ms
   262144                         7.519 ms
   524288                        15.979 ms
  1048576                        31.234 ms
  2097152                        67.524 ms
  4194304                       134.651 ms
  8388608                       290.647 ms
 16777216                       596.639 ms
```

## Cooley-Tukey (Inv-)FFT

I keep benchmark results of Cooley-Tukey (Inv-)FFT implementation of NTT 👇.

### On CPU

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

 dimension                        total
     128                           2.92 ms
     256                          2.019 ms
     512                          1.821 ms
    1024                          2.086 ms
    2048                          2.371 ms
    4096                          3.102 ms
    8192                          3.276 ms
   16384                          3.683 ms
   32768                          4.077 ms
   65536                          4.399 ms
  131072                          6.279 ms
  262144                           9.99 ms
  524288                         18.095 ms
 1048576                         33.374 ms
 2097152                         64.707 ms
 4194304                        128.981 ms
 8388608                        263.787 ms

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

 dimension                        total
     128                          3.932 ms
     256                          1.031 ms
     512                          0.882 ms
    1024                          0.992 ms
    2048                          1.158 ms
    4096                          1.392 ms
    8192                           1.67 ms
   16384                          1.822 ms
   32768                          2.606 ms
   65536                          2.784 ms
  131072                          4.932 ms
  262144                          7.801 ms
  524288                         16.653 ms
 1048576                         35.476 ms
 2097152                         61.381 ms
 4194304                        131.921 ms
 8388608                        267.719 ms
```

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
