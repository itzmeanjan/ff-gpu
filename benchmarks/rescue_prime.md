## Benchmark Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) Elements

I setup benchmarking with a 2D grid of work-items of dimension N x N, where N = {128, 256, 512, 1024, 2048, 4096}. Each work-item takes an input array of prime field elements of length 8 ( so 512-bit input ) and produces 256-bit output hash, consisting of four prime field elements.

`hash_elements`/ `merge` are two benchmarked functions, with AOT compilation enabled for target platform, where workgroup size is set to **64**.

### On CPU

```bash
DEVICE=cpu make aot_cpu && ./a.out
```

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

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

Rescue prime hash on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		          52454 us		        3.20154 us		         312350
256  x  256		       1		         100680 us		        1.53625 us		         650934
512  x  512		       1		         327716 us		        1.25014 us		         799912
1024 x 1024		       1		        1286976 us		        1.22736 us		         814760
2048 x 2048		       1		        5337248 us		         1.2725 us		         785855
4096 x 4096		       1		       21684922 us		        1.29252 us		         773681

Rescue prime merge hashes on F(2**64 - 2**32 + 1) elements 👇

  dimension		iterations		          total		                 avg		                op/s
128  x  128		       1		       39734695 ns		        2425.21 ns		         412335
256  x  256		       1		       89310402 ns		        1362.77 ns		         733800
512  x  512		       1		      336179956 ns		        1282.42 ns		         779773
1024 x 1024		       1		     1361781516 ns		         1298.7 ns		         770003
2048 x 2048		       1		     5561879379 ns		        1326.06 ns		         754116
4096 x 4096		       1		    22200547412 ns		        1323.26 ns		         755712
```

### On GPU

```bash
DEVICE=gpu make aot_gpu && ./a.out
```

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
