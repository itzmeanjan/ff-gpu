## Benchmark Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) Elements

I setup benchmarking with a 2D grid of work-items of dimension N x N, where N = {128, 256, 512, 1024, 2048, 4096}. Each work-item takes an input array of prime field elements of length 8 ( so 512-bit input ) and produces 256-bit output hash, consisting of four prime field elements.

`hash_elements` function is the only one benchmarked, with AOT compilation enabled for target platform, where workgroup size is set to **64**.

> Hash state is represented using register files, which is why I notice, on GPU performance is far superior than previous benchmark on GPU, while on CPU reverse trend is present. It makes sense because, on GPU, register files are larger than what it's in CPU. Note, at [commit](https://github.com/itzmeanjan/ff-gpu/blob/27f670fa955b8e33a76741cd364a8dbae7fa1959/benchmarks/rescue_prime.md) benchmark results using indexable arrays, and compare it with current benchmark results. On CPU, too much use of registers, probably resulting into register spilling which puts state data on RAM and access becomes slower, which seems to be reason behind decreased performance on CPU, when indexable arrays are not used. *I intend to keep register based implementation, instead of indexable array based [one](https://github.com/itzmeanjan/ff-gpu/blob/27f670fa955b8e33a76741cd364a8dbae7fa1959/rescue_prime.cpp), as my major interest is to get better performance when running on GPU.*

### On CPU

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇                                      

  dimension             iterations                        total                          avg                            op/s                                                                                                                                                    
128  x  128                    1                          54666 us                      3.33655 us                       299711                                                                                                                                                 
256  x  256                    1                         160172 us                      2.44403 us                       409160                                                                                                                                                 
512  x  512                    1                         635313 us                      2.42353 us                       412622                                                                                                                                                 
1024 x 1024                    1                        2530078 us                      2.41287 us                       414444                                                                                                                                                 
2048 x 2048                    1                       10154839 us                       2.4211 us                       413035                                                                                                                                                 
4096 x 4096                    1                       40395352 us                      2.40775 us                       415325
```

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg                            op/s
128  x  128                    1                          30587 us                      1.86688 us                       535652
256  x  256                    1                          51024 us                     0.778564 us                  1.28442e+06
512  x  512                    1                         163562 us                      0.62394 us                  1.60272e+06
1024 x 1024                    1                         635064 us                     0.605644 us                  1.65113e+06
2048 x 2048                    1                        2495802 us                     0.595046 us                  1.68054e+06
4096 x 4096                    1                        8674517 us                     0.517042 us                  1.93408e+06
```

### On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg                            op/s
128  x  128                    1                           1379 us                    0.0841675 us                  1.18811e+07
256  x  256                    1                            174 us                   0.00265503 us                  3.76644e+08
512  x  512                    1                            264 us                   0.00100708 us                   9.9297e+08
1024 x 1024                    1                            828 us                  0.000789642 us                   1.2664e+09
2048 x 2048                    1                           3100 us                  0.000739098 us                    1.353e+09
4096 x 4096                    1                          12174 us                  0.000725627 us                  1.37812e+09
```
