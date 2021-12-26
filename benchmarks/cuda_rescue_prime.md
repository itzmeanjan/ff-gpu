## Benchmarking Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) with CUDA Backend

A 2D compute execution space is used for running these benchmarks, where I dispatch N x N -many work-items ( or let's call them independent invocations ), each computing Rescue Prime Hash ( either `hash_elements`/ `merge` function ) on provided input of 512 -bit length ( i.e. 8 contiguous 64 -bit prime field elements ), finally producing N x N -many Rescue Prime digests, each of width 256 -bit ( or in other terms 4 contiguous 64 -bit prime field elements ), which are written to global memory.

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

Rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		           1599 us		      0.0975952 us		    1.02464e+07
256  x  256		       1		           6232 us		      0.0950928 us		     1.0516e+07
512  x  512		       1		          21097 us		      0.0804787 us		    1.24257e+07
1024 x 1024		       1		          84189 us		      0.0802889 us		     1.2455e+07
2048 x 2048		       1		         311482 us		      0.0742631 us		    1.34656e+07
4096 x 4096		       1		        1161206 us		      0.0692133 us		    1.44481e+07

Rescue prime merge hashes on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		        2460449 ns		        150.174 ns		    6.65895e+06
256  x  256		       1		        5013672 ns		        76.5026 ns		    1.30715e+07
512  x  512		       1		       16635254 ns		        63.4585 ns		    1.57583e+07
1024 x 1024		       1		       66357421 ns		        63.2834 ns		    1.58019e+07
2048 x 2048		       1		      261927246 ns		        62.4483 ns		    1.60132e+07
4096 x 4096		       1		     1047354492 ns		        62.4272 ns		    1.60187e+07
```
