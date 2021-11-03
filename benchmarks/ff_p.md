## Benchmark F(2 ** 64 - 2 ** 32 + 1) Arithmetics

### On CPU

#### Hilbert Matrix Generation

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

hilbert matrix generation with F(2**64 - 2**32 + 1) elements 👇

  dimension                          total
32   x   32                         138218 us
64   x   64                           1265 us
128  x  128                           4351 us
256  x  256                          16835 us
512  x  512                          66869 us
1024 x 1024                         275600 us
```

#### Addition

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

addition on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                        1077112 ns                      1.02721 ns
64   x   64                 1024                          58380 ns                    0.0139189 ns
128  x  128                 1024                          65696 ns                   0.00391579 ns
256  x  256                 1024                          75742 ns                   0.00112864 ns
512  x  512                 1024                         115532 ns                   0.00043039 ns
1024 x 1024                 1024                         277929 ns                  0.000258842 ns
```

#### Subtraction

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

subtraction on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                         853570 ns                     0.814028 ns
64   x   64                 1024                          54212 ns                    0.0129251 ns
128  x  128                 1024                          55606 ns                   0.00331438 ns
256  x  256                 1024                          72608 ns                   0.00108194 ns
512  x  512                 1024                         119612 ns                  0.000445589 ns
1024 x 1024                 1024                         289037 ns                  0.000269187 ns
```

#### Multiplication

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

multiplication on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                         809511 ns                      0.77201 ns
64   x   64                 1024                          89723 ns                    0.0213916 ns
128  x  128                 1024                          63624 ns                   0.00379229 ns
256  x  256                 1024                          71413 ns                   0.00106414 ns
512  x  512                 1024                         110790 ns                  0.000412725 ns
1024 x 1024                 1024                         273030 ns                  0.000254279 ns
```

#### Division

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

division on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                         868418 ns                     0.828188 ns
64   x   64                 1024                          52363 ns                    0.0124843 ns
128  x  128                 1024                          83377 ns                   0.00496966 ns
256  x  256                 1024                          70559 ns                   0.00105141 ns
512  x  512                 1024                         114826 ns                   0.00042776 ns
1024 x 1024                 1024                         279563 ns                  0.000260363 ns
```

#### Inversion

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

inversion on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                         853072 ns                     0.813553 ns
64   x   64                 1024                          48846 ns                    0.0116458 ns
128  x  128                 1024                          60232 ns                   0.00359011 ns
256  x  256                 1024                          66781 ns                  0.000995114 ns
512  x  512                 1024                         116743 ns                  0.000434902 ns
1024 x 1024                 1024                         292384 ns                  0.000272304 ns
```

#### Exponentiation

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

exponentiation on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                         800526 ns                     0.763441 ns
64   x   64                 1024                          55009 ns                    0.0131152 ns
128  x  128                 1024                          57609 ns                   0.00343376 ns
256  x  256                 1024                          64887 ns                  0.000966892 ns
512  x  512                 1024                         100210 ns                  0.000373311 ns
1024 x 1024                 1024                         266936 ns                  0.000248604 ns
```

---

### On GPU

#### Hilbert Matrix Generation

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

hilbert matrix generation with F(2**64 - 2**32 + 1) elements 👇

  dimension                          total
32   x   32                         229889 us
64   x   64                            552 us
128  x  128                            533 us
256  x  256                           1267 us
512  x  512                           3762 us
1024 x 1024                          12161 us
```

#### Addition

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

addition on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                          51889 ns                    0.0494852 ns
64   x   64                 1024                          75536 ns                    0.0180092 ns
128  x  128                 1024                          35399 ns                   0.00210994 ns
256  x  256                 1024                          35855 ns                  0.000534281 ns
512  x  512                 1024                          43152 ns                  0.000160754 ns
1024 x 1024                 1024                          71440 ns                  6.65337e-05 ns
```

#### Subtraction

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

subtraction on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                          40848 ns                    0.0389557 ns
64   x   64                 1024                          35352 ns                   0.00842857 ns
128  x  128                 1024                          37347 ns                   0.00222605 ns
256  x  256                 1024                          36963 ns                  0.000550792 ns
512  x  512                 1024                          50454 ns                  0.000187956 ns
1024 x 1024                 1024                          76209 ns                  7.09752e-05 ns
```

#### Multiplication

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

multiplication on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                          40584 ns                    0.0387039 ns
64   x   64                 1024                          35113 ns                   0.00837159 ns
128  x  128                 1024                          36518 ns                   0.00217664 ns
256  x  256                 1024                          35513 ns                  0.000529185 ns
512  x  512                 1024                          48646 ns                   0.00018122 ns
1024 x 1024                 1024                          75797 ns                  7.05915e-05 ns
```

#### Division

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

division on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                          39211 ns                    0.0373945 ns
64   x   64                 1024                          34924 ns                   0.00832653 ns
128  x  128                 1024                          35585 ns                   0.00212103 ns
256  x  256                 1024                          43284 ns                  0.000644982 ns
512  x  512                 1024                          44524 ns                  0.000165865 ns
1024 x 1024                 1024                          75529 ns                  7.03419e-05 ns
```

#### Inversion

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

inversion on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                          38870 ns                    0.0370693 ns
64   x   64                 1024                          35101 ns                   0.00836873 ns
128  x  128                 1024                          32145 ns                   0.00191599 ns
256  x  256                 1024                          50119 ns                  0.000746831 ns
512  x  512                 1024                          50160 ns                  0.000186861 ns
1024 x 1024                 1024                          75807 ns                  7.06008e-05 ns
```

#### Exponentiation

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

exponentiation on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
32   x   32                 1024                          40555 ns                    0.0386763 ns
64   x   64                 1024                          34762 ns                   0.00828791 ns
128  x  128                 1024                          38597 ns                   0.00230056 ns
256  x  256                 1024                          40579 ns                  0.000604674 ns
512  x  512                 1024                          44622 ns                   0.00016623 ns
1024 x 1024                 1024                          74410 ns                  6.92997e-05 ns
```
