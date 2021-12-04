## Benchmarking F(2 ** 64 - 2 ** 32 + 1) Arithmetics with CUDA Backend

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

hilbert matrix generation with F(2**64 - 2**32 + 1) elements 👇

  dimension			     total
128  x  128			       793 us
256  x  256			       401 us
512  x  512			       626 us
1024 x 1024			      1497 us

addition on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          35123 ns		     0.00209349 ns
256  x  256		    1024		          20058 ns		    0.000298887 ns
512  x  512		    1024		          18216 ns		    6.78599e-05 ns
1024 x 1024		    1024		          28129 ns		    2.61972e-05 ns

subtraction on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          25239 ns		     0.00150436 ns
256  x  256		    1024		          17967 ns		    0.000267729 ns
512  x  512		    1024		          18895 ns		    7.03894e-05 ns
1024 x 1024		    1024		          27435 ns		    2.55508e-05 ns

multiplication on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          25098 ns		     0.00149596 ns
256  x  256		    1024		          17733 ns		    0.000264242 ns
512  x  512		    1024		          19380 ns		    7.21961e-05 ns
1024 x 1024		    1024		          33241 ns		    3.09581e-05 ns

division on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          25727 ns		     0.00153345 ns
256  x  256		    1024		          17568 ns		    0.000261784 ns
512  x  512		    1024		          19541 ns		    7.27959e-05 ns
1024 x 1024		    1024		          29171 ns		    2.71676e-05 ns

inversion on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          25469 ns		     0.00151807 ns
256  x  256		    1024		          17558 ns		    0.000261635 ns
512  x  512		    1024		          19377 ns		     7.2185e-05 ns
1024 x 1024		    1024		          29082 ns		    2.70847e-05 ns

exponentiation on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg
128  x  128		    1024		          24932 ns		     0.00148606 ns
256  x  256		    1024		          17979 ns		    0.000267908 ns
512  x  512		    1024		          19517 ns		    7.27065e-05 ns
1024 x 1024		    1024		          29278 ns		    2.72673e-05 ns
```
