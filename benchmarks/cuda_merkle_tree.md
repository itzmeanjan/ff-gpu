## Benchmarking Merklization using Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1) with CUDA Backend

In benchmarking code, I construct all intermediate nodes of Binary Merkle Tree, where N -many leaves are provided. Each leaf is a Rescue Prime digest and Rescue Prime function `merge` is used for merging two children nodes & computing immediate parent node ( which is intermediate node, indeed ). Note, in all these case N must be power of 2. Each digest of Rescue Prime hash is 4 field elements ( i.e. 256 -bit ) wide. After Merklization (N - 1) -many intermediate nodes to be computed. 

I've applied two approaches in writing Merklization routine. Their underlying assumptions are similar, it's just that how work is distributed or more specifically speaking which work-item is orchestrated to do what  is different. 

> I plan to write a detailed description/ comparison of both of these approaches. [ **WIP** ]

```bash
DEVICE=gpu make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

Merklize ( approach 1 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇

  leaves		          total
    1048576		        155.599 ms
    2097152		        288.474 ms
    4194304		        545.014 ms
    8388608		        1064.77 ms

Merklize ( approach 2 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇

  leaves		          total
    1048576		        89.3281 ms
    2097152		        162.115 ms
    4194304		        306.274 ms
    8388608		        593.398 ms
```
