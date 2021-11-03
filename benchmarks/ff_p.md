## Benchmark F(2 ** 64 - 2 ** 32 + 1) Arithmetics

### On CPU

#### Hilbert Matrix Generation

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

hilbert matrix generation with F(2**64 - 2**32 + 1) elements 👇

  dimension                          total
128  x  128                         114347 us
256  x  256                          15986 us
512  x  512                          62734 us
1024 x 1024                         249691 us
```

#### Addition

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

addition on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                        1038580 ns                    0.0619042 ns
256  x  256                 1024                         107417 ns                   0.00160064 ns
512  x  512                 1024                         263105 ns                  0.000980143 ns
1024 x 1024                 1024                         145885 ns                  0.000135866 ns
```

#### Subtraction

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

subtraction on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                         968130 ns                     0.057705 ns
256  x  256                 1024                          87515 ns                   0.00130408 ns
512  x  512                 1024                         116128 ns                  0.000432611 ns
1024 x 1024                 1024                         147496 ns                  0.000137366 ns
```

#### Multiplication

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

multiplication on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                         793775 ns                    0.0473127 ns
256  x  256                 1024                          70640 ns                   0.00105262 ns
512  x  512                 1024                          98206 ns                  0.000365846 ns
1024 x 1024                 1024                         116970 ns                  0.000108937 ns
```

#### Division

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

division on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                         786710 ns                    0.0468916 ns
256  x  256                 1024                          68288 ns                   0.00101757 ns
512  x  512                 1024                          73908 ns                  0.000275329 ns
1024 x 1024                 1024                         128005 ns                  0.000119214 ns
```

#### Inversion

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

inversion on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                         802661 ns                    0.0478423 ns
256  x  256                 1024                          63817 ns                  0.000950947 ns
512  x  512                 1024                          69328 ns                  0.000258267 ns
1024 x 1024                 1024                         134580 ns                  0.000125337 ns
```

#### Exponentiation

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

exponentiation on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                         784132 ns                    0.0467379 ns
256  x  256                 1024                          67957 ns                   0.00101264 ns
512  x  512                 1024                          74861 ns                  0.000278879 ns
1024 x 1024                 1024                         124116 ns                  0.000115592 ns
```

---

### On GPU

#### Hilbert Matrix Generation

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

hilbert matrix generation with F(2**64 - 2**32 + 1) elements 👇

  dimension                          total
128  x  128                         496676 us
256  x  256                           1230 us
512  x  512                           3317 us
1024 x 1024                          11240 us
```

#### Addition

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

addition on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                         107668 ns                   0.00641751 ns
256  x  256                 1024                          52328 ns                  0.000779748 ns
512  x  512                 1024                          60404 ns                  0.000225022 ns
1024 x 1024                 1024                          82924 ns                   7.7229e-05 ns
```

#### Subtraction

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

subtraction on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                          57050 ns                   0.00340044 ns
256  x  256                 1024                          65461 ns                  0.000975445 ns
512  x  512                 1024                          62747 ns                  0.000233751 ns
1024 x 1024                 1024                          81214 ns                  7.56364e-05 ns
```

#### Multiplication

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

multiplication on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                          55895 ns                    0.0033316 ns
256  x  256                 1024                          61597 ns                  0.000917867 ns
512  x  512                 1024                          66228 ns                  0.000246719 ns
1024 x 1024                 1024                          77487 ns                  7.21654e-05 ns
```

#### Division

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

division on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                      292349596 ns                      17.4254 ns
256  x  256                 1024                      895443633 ns                      13.3431 ns
512  x  512                 1024                     2677343714 ns                      9.97388 ns
1024 x 1024                 1024                    10743864974 ns                       10.006 ns
```

#### Inversion

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

inversion on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                      292032993 ns                      17.4065 ns
256  x  256                 1024                      828211506 ns                      12.3413 ns
512  x  512                 1024                     2481310870 ns                       9.2436 ns
1024 x 1024                 1024                    10067563654 ns                      9.37615 ns
```

#### Exponentiation

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

exponentiation on F(2**64 - 2**32 + 1) elements 👇

  dimension             iterations                        total                          avg
128  x  128                 1024                      176247119 ns                      10.5051 ns
256  x  256                 1024                      545426987 ns                       8.1275 ns
512  x  512                 1024                     2039514295 ns                      7.59778 ns
1024 x 1024                 1024                     9636617280 ns                       8.9748 ns
```
