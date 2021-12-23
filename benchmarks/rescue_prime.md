## Benchmark Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) Elements

I setup benchmarking with a 2D grid of work-items of dimension N x N, where N = {128, 256, 512, 1024, 2048, 4096}. Each work-item takes an input array of prime field elements of length 8 ( so 512-bit input ) and produces 256-bit output hash, consisting of four prime field elements.

`hash_elements` function is the only one benchmarked, with AOT compilation enabled for target platform, where workgroup size is set to **64**.

> Hash state is represented using register files, which is why I notice, on GPU performance is far superior than previous benchmark on GPU, while on CPU reverse trend is present. It makes sense because, on GPU, register files are larger than what it's in CPU. Note, at [commit](https://github.com/itzmeanjan/ff-gpu/blob/27f670fa955b8e33a76741cd364a8dbae7fa1959/benchmarks/rescue_prime.md) benchmark results using indexable arrays, and compare it with current benchmark results. On CPU, too much use of registers, probably resulting into register spilling which puts state data on RAM and access becomes slower, which seems to be reason behind decreased performance on CPU, when indexable arrays are not used. *I intend to keep register based implementation, instead of indexable array based [one](https://github.com/itzmeanjan/ff-gpu/blob/27f670fa955b8e33a76741cd364a8dbae7fa1959/rescue_prime.cpp), as my major interest is to get better performance when running on GPU.*

### On CPU

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		         215097 us		        13.1285 us		        76170.3
256  x  256		       1		         740699 us		        11.3022 us		        88478.6
512  x  512		       1		        2929226 us		        11.1741 us		        89492.6
1024 x 1024		       1		       11716377 us		        11.1736 us		        89496.6
2048 x 2048		       1		       46593422 us		        11.1087 us		        90019.2
4096 x 4096		       1		      186992626 us		        11.1456 us		        89721.3
```

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		          52454 us		        3.20154 us		         312350
256  x  256		       1		         100680 us		        1.53625 us		         650934
512  x  512		       1		         327716 us		        1.25014 us		         799912
1024 x 1024		       1		        1286976 us		        1.22736 us		         814760
2048 x 2048		       1		        5337248 us		         1.2725 us		         785855
4096 x 4096		       1		       21684922 us		        1.29252 us		         773681
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
