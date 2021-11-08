## Benchmark Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) Elements

I setup benchmarking with a 2D grid of work-items of dimension N x N, where N = {128, 256, 512, 1024}. Each work-item takes an input array of prime field elements of length 8 ( so 512-bit input ) and produces 256-bit output hash, consisting of four prime field elements.

`hash_elements` function is the only one benchmarked, with AOT compilation enabled for target platform, where workgroup size is set to **128**.

> Current implementation uses USM ( read pointer arithmetics used heavily ) for inteacting with memory systems.

### On CPU

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg                            op/s
128  x  128                    1                          54202 us                      3.30823 us                       302277
256  x  256                    1                         157476 us                      2.40289 us                       416165
512  x  512                    1                         603811 us                      2.30336 us                       434149
1024 x 1024                    1                        2379147 us                      2.26893 us                       440736
```

### On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg                            op/s
128  x  128                    1                        2428612 us                      148.231 us                      6746.24
256  x  256                    1                          31218 us                     0.476349 us                   2.0993e+06
512  x  512                    1                         107906 us                     0.411629 us                  2.42937e+06
1024 x 1024                    1                         433583 us                     0.413497 us                   2.4184e+06
```
