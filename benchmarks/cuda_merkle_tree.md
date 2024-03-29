## Benchmarking Merklization using Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) with CUDA Backend

In benchmarking code, I construct all intermediate nodes of Binary Merkle Tree, where N -many leaves are provided. Each leaf is a Rescue Prime digest and Rescue Prime function `merge` is used for merging two children nodes & computing immediate parent node ( which is intermediate node, indeed ). Note, in all these case N must be power of 2. Each digest of Rescue Prime hash is 4 field elements ( i.e. 256 -bit ) wide. After Merklization (N - 1) -many intermediate nodes to be computed. 

I've applied three approaches in writing Merklization routine. Their underlying assumptions are similar, it's just that how work is distributed or more specifically speaking which work-item is orchestrated to do what is different along with whether Rescue Prime hash constants are read from global memory or cheaper scratch pad memory. Latest routine ( i.e. `approach 3` ) uses local scratch pad memory, otherwise it's very much same as what `approach 1` is.

> You may want to read [this post](https://itzmeanjan.in/pages/evaluate-merklizaion-design-performance-in-sycl.html) for understanding difference in between `approach {1, 2}`.

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

Merklize ( approach 1 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇

     leaves		          total
    1048576		        63.1163 ms
    2097152		        116.267 ms
    4194304		        219.799 ms
    8388608		        431.501 ms
   16777216		        849.147 ms

Merklize ( approach 2 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇

     leaves		          total
    1048576		        122.429 ms
    2097152		        231.646 ms
    4194304		        449.042 ms
    8388608		        886.979 ms
   16777216		        1746.06 ms

Merklize ( approach 3 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇

     leaves		          total
    1048576		        63.1895 ms
    2097152		        114.592 ms
    4194304		        212.439 ms
    8388608		        415.004 ms
   16777216		        792.812 ms
```
