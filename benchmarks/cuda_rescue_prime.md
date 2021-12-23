## Benchmarking Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) with CUDA Backend

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		           1599 us		      0.0975952 us		    1.02464e+07
256  x  256		       1		           6232 us		      0.0950928 us		     1.0516e+07
512  x  512		       1		          21097 us		      0.0804787 us		    1.24257e+07
1024 x 1024		       1		          84189 us		      0.0802889 us		     1.2455e+07
2048 x 2048		       1		         311482 us		      0.0742631 us		    1.34656e+07
4096 x 4096		       1		        1161206 us		      0.0692133 us		    1.44481e+07
```
