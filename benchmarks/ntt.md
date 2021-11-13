## Benchmarking NTT on F(2 ** 64 - 2 ** 32 + 1) Elements

I benchmark *DFT*-style Forward/ Inverse **N**umber **T**heoretic **T**ransform with random elements sampled from aforementioned prime field, with vector(s) of size N = 2 ** i, i > 0.

> DFT-style NTT computes DFT matrix & uses matrix-vector multiplication !

> Note: These benchmarks include time required to transfer input/ ouput vector(s) to/ from device.

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
