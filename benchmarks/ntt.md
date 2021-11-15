## Benchmarking NTT on F(2 ** 64 - 2 ** 32 + 1) Elements

I benchmark both *DFT*-style and Cooley-Tukey Forward/ Inverse **N**umber **T**heoretic **T**ransform with random elements sampled from aforementioned prime field, with vector(s) of size N = 2 ** i, i > 0.

> DFT-style NTT computes DFT matrix & uses matrix-vector multiplication !

> Note: These benchmarks include time required to transfer input/ ouput vector(s) to/ from device.

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

## Cooley-Tukey (Inv-)FFT

I keep benchmark results of Cooley-Tukey (Inv-)FFT implementation of NTT 👇.

### On CPU

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      3739 us
  256                       972 us
  512                      1235 us
 1024                      1482 us

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      5102 us
  256                      1041 us
  512                      1086 us
 1024                      1518 us
```

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      2932 us
  256                       992 us
  512                      1174 us
 1024                      1218 us

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      3417 us
  256                      1065 us
  512                      1059 us
 1024                      1210 us
```

### On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      1883 us
  256                      1743 us
  512                      1835 us
 1024                      1911 us

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      1945 us
  256                      1941 us
  512                      2132 us
 1024                      2186 us
```

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      1528 us
  256                      1270 us
  512                      1971 us
 1024                      2381 us

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

dimension                    total
  128                      2134 us
  256                      1610 us
  512                      2073 us
 1024                      2303 us
```
