## Benchmarking F(2 ** 32) Arithmetics with CUDA Backend

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

hilbert matrix generation with F(2 ** 32) elements 👇

  dimension			     total
128  x  128			      1232 us
256  x  256			       696 us
512  x  512			      1647 us
1024 x 1024			      5555 us

addition on F(2 ** 32) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          40712 ns		     0.00242662 ns
256  x  256		    1024		          19506 ns		    0.000290662 ns
512  x  512		    1024		          18193 ns		    6.77742e-05 ns
1024 x 1024		    1024		          27604 ns		    2.57082e-05 ns

subtraction on F(2 ** 32) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          24565 ns		     0.00146419 ns
256  x  256		    1024		          18235 ns		    0.000271723 ns
512  x  512		    1024		          17838 ns		    6.64517e-05 ns
1024 x 1024		    1024		          27123 ns		    2.52603e-05 ns

multiplication on F(2 ** 32) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          24633 ns		     0.00146824 ns
256  x  256		    1024		          18355 ns		    0.000273511 ns
512  x  512		    1024		          19659 ns		    7.32355e-05 ns
1024 x 1024		    1024		          28587 ns		    2.66237e-05 ns

division on F(2 ** 32) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          23520 ns		      0.0014019 ns
256  x  256		    1024		          17529 ns		    0.000261202 ns
512  x  512		    1024		          20052 ns		    7.46995e-05 ns
1024 x 1024		    1024		          29490 ns		    2.74647e-05 ns

inversion on F(2 ** 32) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          26186 ns		     0.00156081 ns
256  x  256		    1024		          17737 ns		    0.000264302 ns
512  x  512		    1024		          19427 ns		    7.23712e-05 ns
1024 x 1024		    1024		          30040 ns		    2.79769e-05 ns

exponentiation on F(2 ** 32) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          25294 ns		     0.00150764 ns
256  x  256		    1024		          18120 ns		    0.000270009 ns
512  x  512		    1024		          19715 ns		    7.34441e-05 ns
1024 x 1024		    1024		          29630 ns		    2.75951e-05 ns
```
