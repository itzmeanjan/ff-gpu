## Benchmarking Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) with CUDA Backend

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		           1917 us		       0.117004 us		    8.54669e+06
256  x  256		       1		           7179 us		       0.109543 us		    9.12885e+06
512  x  512		       1		          25410 us		      0.0969315 us		    1.03166e+07
1024 x 1024		       1		         101363 us		      0.0966673 us		    1.03448e+07
2048 x 2048		       1		         373258 us		      0.0889916 us		     1.1237e+07
4096 x 4096		       1		        1409368 us		      0.0840049 us		    1.19041e+07
```
