## Benchmark F(2 ** 32) Arithmetics

### On CPU

#### Hilbert Matrix Generation

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

hilbert matrix generation with F(2 ** 32) elements

  dimension                          total
32   x   32                         126920 us
64   x   64                           2811 us
128  x  128                          10358 us
256  x  256                          40784 us
512  x  512                         163589 us
1024 x 1024                         655595 us
```

#### Addition

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

addition on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                       77438550 ns                      73.8512 ns
64   x   64                 1024                         249679 ns                    0.0595281 ns
128  x  128                 1024                          90180 ns                   0.00537515 ns
256  x  256                 1024                          89057 ns                   0.00132705 ns
512  x  512                 1024                         120195 ns                  0.000447761 ns
1024 x 1024                 1024                         270457 ns                  0.000251883 ns
```

#### Subtraction

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

subtraction on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                         685632 ns                      0.65387 ns
64   x   64                 1024                          78116 ns                    0.0186243 ns
128  x  128                 1024                          59932 ns                   0.00357223 ns
256  x  256                 1024                          68233 ns                   0.00101675 ns
512  x  512                 1024                         121872 ns                  0.000454009 ns
1024 x 1024                 1024                         274567 ns                   0.00025571 ns
```

#### Multiplication

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

multiplication on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                         680773 ns                     0.649236 ns
64   x   64                 1024                          53883 ns                    0.0128467 ns
128  x  128                 1024                          53262 ns                   0.00317466 ns
256  x  256                 1024                          65145 ns                  0.000970736 ns
512  x  512                 1024                         102063 ns                  0.000380214 ns
1024 x 1024                 1024                         261642 ns                  0.000243673 ns
```

#### Division

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

division on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                         684562 ns                     0.652849 ns
64   x   64                 1024                          52228 ns                    0.0124521 ns
128  x  128                 1024                          52367 ns                   0.00312132 ns
256  x  256                 1024                          63755 ns                  0.000950024 ns
512  x  512                 1024                         111542 ns                  0.000415526 ns
1024 x 1024                 1024                         243703 ns                  0.000226966 ns
```

#### Inversion

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

inversion on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                         680572 ns                     0.649044 ns
64   x   64                 1024                          51444 ns                    0.0122652 ns
128  x  128                 1024                          54728 ns                   0.00326204 ns
256  x  256                 1024                          65594 ns                  0.000977427 ns
512  x  512                 1024                         130491 ns                  0.000486117 ns
1024 x 1024                 1024                         267599 ns                  0.000249221 ns
```

#### Exponentiation

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

exponentiation on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                         677030 ns                     0.645666 ns
64   x   64                 1024                          49722 ns                    0.0118546 ns
128  x  128                 1024                          50491 ns                    0.0030095 ns
256  x  256                 1024                          65832 ns                  0.000980973 ns
512  x  512                 1024                         109839 ns                  0.000409182 ns
1024 x 1024                 1024                         269690 ns                  0.000251168 ns
```

---

### On GPU

#### Hilbert Matrix Generation

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

hilbert matrix generation with F(2 ** 32) elements

  dimension                          total
32   x   32                         120629 us
64   x   64                            356 us
128  x  128                            583 us
256  x  256                           1681 us
512  x  512                           6070 us
1024 x 1024                          23687 us
```

#### Addition

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

addition on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                       81232118 ns                       77.469 ns
64   x   64                 1024                         118336 ns                    0.0282135 ns
128  x  128                 1024                          36772 ns                   0.00219178 ns
256  x  256                 1024                          37611 ns                  0.000560448 ns
512  x  512                 1024                          43900 ns                   0.00016354 ns
1024 x 1024                 1024                          85015 ns                  7.91764e-05 ns
```

#### Subtraction

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

subtraction on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                          42661 ns                    0.0406847 ns
64   x   64                 1024                          35729 ns                   0.00851846 ns
128  x  128                 1024                          34841 ns                   0.00207669 ns
256  x  256                 1024                          37372 ns                  0.000556886 ns
512  x  512                 1024                          50507 ns                  0.000188153 ns
1024 x 1024                 1024                          75626 ns                  7.04322e-05 ns

```

#### Multiplication

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

multiplication on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                          39925 ns                    0.0380754 ns
64   x   64                 1024                          36140 ns                   0.00861645 ns
128  x  128                 1024                          34156 ns                   0.00203586 ns
256  x  256                 1024                          35354 ns                  0.000526816 ns
512  x  512                 1024                          50392 ns                  0.000187725 ns
1024 x 1024                 1024                          75707 ns                  7.05076e-05 ns
```

#### Division

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

division on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                          39546 ns                     0.037714 ns
64   x   64                 1024                          34979 ns                   0.00833964 ns
128  x  128                 1024                          41367 ns                   0.00246567 ns
256  x  256                 1024                          37131 ns                  0.000553295 ns
512  x  512                 1024                          42717 ns                  0.000159133 ns
1024 x 1024                 1024                          75054 ns                  6.98995e-05 ns
```

#### Inversion

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

inversion on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                          39948 ns                    0.0380974 ns
64   x   64                 1024                          35504 ns                   0.00846481 ns
128  x  128                 1024                          41189 ns                   0.00245506 ns
256  x  256                 1024                          37129 ns                  0.000553265 ns
512  x  512                 1024                          42797 ns                  0.000159431 ns
1024 x 1024                 1024                          74894 ns                  6.97505e-05 ns
```

#### Exponentiation

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

exponentiation on F(2 ** 32) elements

  dimension             iterations                        total                          avg
32   x   32                 1024                          40034 ns                    0.0381794 ns
64   x   64                 1024                          35241 ns                   0.00840211 ns
128  x  128                 1024                          37286 ns                   0.00222242 ns
256  x  256                 1024                          35302 ns                  0.000526041 ns
512  x  512                 1024                          50148 ns                  0.000186816 ns
1024 x 1024                 1024                          75590 ns                  7.03987e-05 ns
```
