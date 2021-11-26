## Benchmark Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) Elements

I setup benchmarking with a 2D grid of work-items of dimension N x N, where N = {128, 256, 512, 1024, 2048, 4096}. Each work-item takes an input array of prime field elements of length 8 ( so 512-bit input ) and produces 256-bit output hash, consisting of four prime field elements.

`hash_elements` function is the only one benchmarked, with AOT compilation enabled for target platform, where workgroup size is set to **64**.

> Hash state is represented using register files, which is why I notice, on GPU performance is far superior than previous benchmark on GPU, while on CPU reverse trend is present. It makes sense because, on GPU, register files are larger than what it's in CPU. Note, at [commit](https://github.com/itzmeanjan/ff-gpu/blob/27f670fa955b8e33a76741cd364a8dbae7fa1959/benchmarks/rescue_prime.md) benchmark results using indexable arrays, and compare it with current benchmark results. On CPU, too much use of registers, probably resulting into register spilling which puts state data on RAM and access becomes slower, which seems to be reason behind decreased performance on CPU, when indexable arrays are not used. *I intend to keep register based implementation, instead of indexable array based [one](https://github.com/itzmeanjan/ff-gpu/blob/27f670fa955b8e33a76741cd364a8dbae7fa1959/rescue_prime.cpp), as my major interest is to get better performance when running on GPU.*

### On CPU

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		          49722 us		        3.03479 us		         329512
256  x  256		       1		         162410 us		        2.47818 us		         403522
512  x  512		       1		         639436 us		        2.43925 us		         409961
1024 x 1024		       1		        2538022 us		        2.42045 us		         413147
2048 x 2048		       1		       10113576 us		        2.41126 us		         414720
4096 x 4096		       1		       40439263 us		        2.41037 us		         414874
```

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		          25644 us		        1.56519 us		         638902
256  x  256		       1		          49490 us		       0.755157 us		    1.32423e+06
512  x  512		       1		         163912 us		       0.625275 us		     1.5993e+06
1024 x 1024		       1		         636838 us		       0.607336 us		    1.64653e+06
2048 x 2048		       1		        2491102 us		       0.593925 us		    1.68371e+06
4096 x 4096		       1		        8382231 us		        0.49962 us		    2.00152e+06
```

### On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		          11589 us		       0.707336 us		    1.41375e+06
256  x  256		       1		          36109 us		        0.55098 us		    1.81495e+06
512  x  512		       1		         131971 us		       0.503429 us		    1.98638e+06
1024 x 1024		       1		         521590 us		       0.497427 us		    2.01035e+06
2048 x 2048		       1		        2076505 us		       0.495077 us		    2.01989e+06
4096 x 4096		       1		        8334105 us		       0.496751 us		    2.01308e+06
```
