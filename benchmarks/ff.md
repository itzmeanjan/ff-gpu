## Benchmark F(2 ** 32) Arithmetics

### Hilbert Matrix Generation

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

### Addition

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

### Subtraction

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

### Multiplication

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

### Division

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

### Inversion

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

### Exponentiation

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
