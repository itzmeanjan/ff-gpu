## Benchmarking NTT on F(2 ** 64 - 2 ** 32 + 1) with CUDA Backend

```bash
DEVICE=gpu make cuda && ./run
```

### Six Step (I)FFT

```bash
running on Tesla V100-SXM2-16GB

Six-Step FFT on F(2**64 - 2**32 + 1) elements 👇

  dimension		          total
     4096		          0.495 ms
     8192		          0.386 ms
    16384		           0.39 ms
    32768		          0.399 ms
    65536		           0.41 ms
   131072		          0.681 ms
   262144		           1.12 ms
   524288		          1.948 ms
  1048576		          3.122 ms
  2097152		          6.663 ms
  4194304		         12.037 ms
  8388608		          24.11 ms
 16777216		         48.359 ms

Six-Step IFFT on F(2**64 - 2**32 + 1) elements 👇

  dimension		          total
     4096		          0.497 ms
     8192		          0.397 ms
    16384		          0.403 ms
    32768		          0.415 ms
    65536		           0.43 ms
   131072		          0.704 ms
   262144		          1.136 ms
   524288		          2.127 ms
  1048576		          3.224 ms
  2097152		          6.717 ms
  4194304		         12.017 ms
  8388608		         24.154 ms
 16777216		         48.301 ms
```

### Cooley-Tukey (I)FFT

```bash
running on Tesla V100-SXM2-16GB

Cooley-Tukey FFT on F(2**64 - 2**32 + 1) elements 👇

 dimension		          total
     128		          0.632 ms
     256		          0.617 ms
     512		          0.607 ms
    1024		          0.624 ms
    2048		          0.651 ms
    4096		          0.684 ms
    8192		          0.714 ms
   16384		          0.755 ms
   32768		           0.83 ms
   65536		          0.946 ms
  131072		          1.442 ms
  262144		          2.076 ms
  524288		          3.482 ms
 1048576		         11.779 ms
 2097152		         23.149 ms
 4194304		         46.459 ms
 8388608		         93.464 ms

Cooley-Tukey IFFT on F(2**64 - 2**32 + 1) elements 👇

 dimension		          total
     128		          0.868 ms
     256		          0.649 ms
     512		          0.667 ms
    1024		           0.69 ms
    2048		          0.717 ms
    4096		          0.742 ms
    8192		          0.778 ms
   16384		          0.853 ms
   32768		          0.906 ms
   65536		          0.973 ms
  131072		          1.414 ms
  262144		          2.096 ms
  524288		          3.312 ms
 1048576		          6.069 ms
 2097152		         11.805 ms
 4194304		         46.062 ms
 8388608		         98.826 ms
```
