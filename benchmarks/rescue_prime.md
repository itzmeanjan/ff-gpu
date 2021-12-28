## Benchmark Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) Elements

I setup benchmarking with a 2D grid of work-items of dimension N x N, where N = {128, 256, 512, 1024, 2048, 4096}. Each work-item takes an input array of prime field elements of length 8 ( so 512-bit input ) and produces 256-bit output hash, consisting of four prime field elements.

`hash_elements`/ `merge` are two benchmarked functions, with AOT compilation enabled for target platform, where workgroup size is set to **64**.

### On CPU

```bash
DEVICE=cpu make aot_cpu && ./a.out
```

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		       33228041 ns		        2028.08 ns		         493078
256  x  256		       1		       68007486 ns		        1037.71 ns		         963659
512  x  512		       1		      194804485 ns		         743.12 ns		    1.34568e+06
1024 x 1024		       1		      723165970 ns		        689.665 ns		    1.44998e+06
2048 x 2048		       1		     3003509188 ns		        716.092 ns		    1.39647e+06
4096 x 4096		       1		    11916667514 ns		        710.289 ns		    1.40788e+06

Rescue prime merge hashes on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		       17881942 ns		        1091.43 ns		         916232
256  x  256		       1		       49560738 ns		        756.237 ns		    1.32234e+06
512  x  512		       1		      179926534 ns		        686.365 ns		    1.45695e+06
1024 x 1024		       1		      740464524 ns		        706.162 ns		    1.41611e+06
2048 x 2048		       1		     2973793361 ns		        709.008 ns		    1.41042e+06
4096 x 4096		       1		    11871040794 ns		        707.569 ns		    1.41329e+06
```
