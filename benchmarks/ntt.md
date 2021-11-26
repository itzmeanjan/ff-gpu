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
     4096                        13.397 ms
     8192                         3.424 ms
    16384                         3.821 ms
    32768                         3.808 ms
    65536                         4.917 ms
   131072                         6.383 ms
   262144                        10.063 ms
   524288                        29.554 ms
  1048576                        21.893 ms
  2097152                        60.622 ms
  4194304                        86.085 ms
  8388608                       209.005 ms
 16777216                       327.247 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         7.356 ms
     8192                         1.399 ms
    16384                         1.392 ms
    32768                         2.252 ms
    65536                         2.931 ms
   131072                          4.65 ms
   262144                         8.079 ms
   524288                        27.875 ms
  1048576                        20.418 ms
  2097152                        60.818 ms
  4194304                         87.45 ms
  8388608                        205.47 ms
 16777216                        327.57 ms
```

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                        13.462 ms
     8192                         2.344 ms
    16384                         3.291 ms
    32768                         5.236 ms
    65536                         8.176 ms
   131072                         11.93 ms
   262144                        23.085 ms
   524288                        44.494 ms
  1048576                        72.567 ms
  2097152                       165.297 ms
  4194304                       315.133 ms
  8388608                       689.362 ms
 16777216                       1400.38 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         8.183 ms
     8192                         1.825 ms
    16384                         2.911 ms
    32768                         4.873 ms
    65536                         7.424 ms
   131072                        12.179 ms
   262144                        18.073 ms
   524288                        45.872 ms
  1048576                        78.676 ms
  2097152                       163.279 ms
  4194304                       315.296 ms
  8388608                        693.73 ms
 16777216                       1404.69 ms
```

```bash
running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                        12.066 ms
     8192                          1.75 ms
    16384                         2.621 ms
    32768                         4.888 ms
    65536                         8.084 ms
   131072                         12.55 ms
   262144                        17.075 ms
   524288                        35.661 ms
  1048576                        68.663 ms
  2097152                       153.032 ms
  4194304                       314.974 ms
  8388608                       693.434 ms
 16777216                       1435.38 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         6.811 ms
     8192                         1.076 ms
    16384                         1.507 ms
    32768                         2.605 ms
    65536                         4.598 ms
   131072                         8.582 ms
   262144                        15.765 ms
   524288                        37.308 ms
  1048576                        70.485 ms
  2097152                        154.82 ms
  4194304                       314.636 ms
  8388608                       691.992 ms
 16777216                          1434 ms
```

### On GPU


```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         1.081 ms
     8192                         1.069 ms
    16384                         1.208 ms
    32768                         1.758 ms
    65536                         2.326 ms
   131072                           3.9 ms
   262144                         6.926 ms
   524288                        15.181 ms
  1048576                        29.555 ms
  2097152                        63.418 ms
  4194304                       124.907 ms
  8388608                       267.661 ms
 16777216                       546.757 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension                       total
     4096                         1.251 ms
     8192                         1.354 ms
    16384                          1.53 ms
    32768                         1.978 ms
    65536                         2.616 ms
   131072                          4.23 ms
   262144                         7.366 ms
   524288                        15.397 ms
  1048576                         30.04 ms
  2097152                        63.358 ms
  4194304                       125.259 ms
  8388608                       267.972 ms
 16777216                       546.018 ms
```

## Cooley-Tukey (Inv-)FFT

I keep benchmark results of Cooley-Tukey (Inv-)FFT implementation of NTT 👇.

### On CPU

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                
 dimension                        total                                                                                                                                                                                                                                         
     128                          4.388 ms                                                                                                                                                                                                                                      
     256                          2.047 ms                                                                                                                                                                                                                                      
     512                          2.944 ms                                                                                                                                                                                                                                      
    1024                          3.035 ms                                                                                                                                                                                                                                      
    2048                           3.15 ms                                                                                                                                                                                                                                      
    4096                          4.674 ms                                                                                                                                                                                                                                      
    8192                          3.447 ms                                                                                                                                                                                                                                      
   16384                          3.007 ms                                                                                                                                                                                                                                      
   32768                          3.414 ms                                                                                                                                                                                                                                      
   65536                          4.127 ms                                                                                                                                                                                                                                      
  131072                          6.062 ms                                                                                                                                                                                                                                      
  262144                          8.413 ms                                                                                                                                                                                                                                      
  524288                         15.211 ms                                                                                                                                                                                                                                      
 1048576                         29.147 ms                                                                                                                                                                                                                                      
 2097152                         55.438 ms                                                                                                                                                                                                                                      
 4194304                        117.908 ms                                                                                                                                                                                                                                      
 8388608                        234.131 ms                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                
Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                
 dimension                        total                                                                                                                                                                                                                                         
     128                          5.674 ms                                                                                                                                                                                                                                      
     256                          1.146 ms                                                                                                                                                                                                                                      
     512                          1.496 ms                                                                                                                                                                                                                                      
    1024                          1.285 ms                                                                                                                                                                                                                                      
    2048                          1.158 ms                                                                                                                                                                                                                                      
    4096                          1.429 ms                                                                                                                                                                                                                                      
    8192                            1.6 ms                                                                                                                                                                                                                                      
   16384                          2.057 ms
   32768                          2.502 ms
   65536                          3.078 ms
  131072                          4.262 ms
  262144                          8.941 ms
  524288                          14.68 ms
 1048576                         33.476 ms
 2097152                         51.205 ms
 4194304                        115.247 ms
 8388608                        235.106 ms
```

```bash
running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

 dimension                        total
     128                          3.852 ms
     256                          0.618 ms
     512                          0.717 ms
    1024                          0.709 ms
    2048                          0.841 ms
    4096                          1.011 ms
    8192                          1.422 ms
   16384                          2.576 ms
   32768                          4.196 ms
   65536                          7.209 ms
  131072                         11.169 ms
  262144                         20.356 ms
  524288                         42.984 ms
 1048576                          92.55 ms
 2097152                        202.957 ms
 4194304                        431.258 ms
 8388608                        946.809 ms

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

 dimension                        total
     128                          5.268 ms
     256                          0.485 ms
     512                          0.496 ms
    1024                          0.572 ms
    2048                          0.518 ms
    4096                          0.645 ms
    8192                          0.931 ms
   16384                          1.529 ms
   32768                          2.745 ms
   65536                          4.819 ms
  131072                           9.93 ms
  262144                         20.306 ms
  524288                         44.245 ms
 1048576                         94.055 ms
 2097152                        199.423 ms
 4194304                        431.265 ms
 8388608                        947.322 ms
```

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                
 dimension                        total                                                                                                                                                                                                                                         
     128                            4.5 ms                                                                                                                                                                                                                                      
     256                          0.951 ms                                                                                                                                                                                                                                      
     512                          0.867 ms                                                                                                                                                                                                                                      
    1024                          1.032 ms                                                                                                                                                                                                                                      
    2048                          1.113 ms                                                                                                                                                                                                                                      
    4096                          1.345 ms                                                                                                                                                                                                                                      
    8192                          1.734 ms                                                                                                                                                                                                                                      
   16384                              3 ms                                                                                                                                                                                                                                      
   32768                          4.323 ms                                                                                                                                                                                                                                      
   65536                          7.287 ms                                                                                                                                                                                                                                      
  131072                         13.864 ms                                                                                                                                                                                                                                      
  262144                         22.543 ms                                                                                                                                                                                                                                      
  524288                         45.667 ms                                                                                                                                                                                                                                      
 1048576                          103.5 ms                                                                                                                                                                                                                                      
 2097152                         207.97 ms                                                                                                                                                                                                                                      
 4194304                        441.452 ms                                                                                                                                                                                                                                      
 8388608                        942.119 ms                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                
Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                
 dimension                        total                                                                                                                                                                                                                                         
     128                          6.035 ms                                                                                                                                                                                                                                      
     256                          0.642 ms                                                                                                                                                                                                                                      
     512                          0.879 ms                                                                                                                                                                                                                                      
    1024                          0.886 ms                                                                                                                                                                                                                                      
    2048                          1.083 ms                                                                                                                                                                                                                                      
    4096                          1.247 ms                                                                                                                                                                                                                                      
    8192                          1.668 ms                                                                                                                                                                                                                                      
   16384                          2.801 ms
   32768                          4.442 ms
   65536                          7.052 ms
  131072                         12.341 ms
  262144                         26.105 ms
  524288                         51.373 ms
 1048576                        102.383 ms
 2097152                        209.711 ms
 4194304                        440.504 ms
 8388608                        945.042 ms
```

### On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                
 dimension                        total                                                                                                                                                                                                                                         
     128                          0.816 ms                                                                                                                                                                                                                                      
     256                          0.933 ms                                                                                                                                                                                                                                      
     512                          0.832 ms                                                                                                                                                                                                                                      
    1024                          0.862 ms                                                                                                                                                                                                                                      
    2048                          0.886 ms                                                                                                                                                                                                                                      
    4096                          0.993 ms                                                                                                                                                                                                                                      
    8192                          1.138 ms                                                                                                                                                                                                                                      
   16384                          1.396 ms                                                                                                                                                                                                                                      
   32768                          2.388 ms                                                                                                                                                                                                                                      
   65536                          3.373 ms                                                                                                                                                                                                                                      
  131072                          6.171 ms                                                                                                                                                                                                                                      
  262144                         11.676 ms                                                                                                                                                                                                                                      
  524288                          24.18 ms                                                                                                                                                                                                                                      
 1048576                          53.15 ms                                                                                                                                                                                                                                      
 2097152                        114.183 ms                                                                                                                                                                                                                                      
 4194304                        253.065 ms                                                                                                                                                                                                                                      
 8388608                        554.653 ms                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                
Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                
 dimension                        total                                                                                                                                                                                                                                         
     128                          1.274 ms                                                                                                                                                                                                                                      
     256                            1.1 ms                                                                                                                                                                                                                                      
     512                          1.073 ms                                                                                                                                                                                                                                      
    1024                          1.102 ms                                                                                                                                                                                                                                      
    2048                          1.112 ms                                                                                                                                                                                                                                      
    4096                          1.207 ms                                                                                                                                                                                                                                      
    8192                          1.362 ms                                                                                                                                                                                                                                      
   16384                          1.588 ms
   32768                           2.27 ms
   65536                          3.677 ms
  131072                          6.733 ms
  262144                         12.789 ms
  524288                         25.923 ms
 1048576                         53.531 ms
 2097152                        116.104 ms
 4194304                        248.198 ms
 8388608                        555.098 ms
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
