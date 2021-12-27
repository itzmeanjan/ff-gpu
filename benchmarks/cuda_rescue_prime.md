## Benchmarking Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) with CUDA Backend

A 2D compute execution space is used for running these benchmarks, where I dispatch N x N -many work-items ( or let's call them independent invocations ), each computing Rescue Prime Hash ( either `hash_elements`/ `merge` function ) on provided input of 512 -bit length ( i.e. 8 contiguous 64 -bit prime field elements ), finally producing N x N -many Rescue Prime digests, each of width 256 -bit ( or in other terms 4 contiguous 64 -bit prime field elements ), which are written to global memory.

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

Rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		        1021973 ns		        62.3763 ns		    1.60317e+07
256  x  256		       1		        4236328 ns		        64.6412 ns		      1.547e+07
512  x  512		       1		       14381104 ns		        54.8596 ns		    1.82284e+07
1024 x 1024		       1		       57822754 ns		        55.1441 ns		    1.81343e+07
2048 x 2048		       1		      221727540 ns		         52.864 ns		    1.89165e+07
4096 x 4096		       1		      818960449 ns		        48.8138 ns		     2.0486e+07

Rescue prime merge hashes on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		        1420411 ns		         86.695 ns		    1.15347e+07
256  x  256		       1		        3790039 ns		        57.8314 ns		    1.72916e+07
512  x  512		       1		       12962403 ns		        49.4476 ns		    2.02234e+07
1024 x 1024		       1		       52203125 ns		        49.7848 ns		    2.00865e+07
2048 x 2048		       1		      206751953 ns		        49.2935 ns		    2.02866e+07
4096 x 4096		       1		      827613769 ns		        49.3296 ns		    2.02718e+07
```
