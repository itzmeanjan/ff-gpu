## Benchmarking Merklization using Rescue Prime Hash on F(2 ** 64 - 2 ** 32 + 1)

Given N -many leaves of fully balanced Binary Merkle Tree, all (N-1) -many intermediate nodes are constructed by following two routines. Note, both of these approaches use Rescue Prime as underlying hash function for merging two child nodes into single immediate parent node. So an efficient Rescue Prime Hash function implementation boosts Merklization construction. Improvement suggestions/ ideas are welcome.

```bash
DEVICE=cpu make aot_cpu && ./a.out
```

```bash
running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Merklize ( approach 1 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇

     leaves		          total
    1048576		        1801.52 ms
    2097152		         3485.8 ms
    4194304		        6974.96 ms
    8388608		        13965.3 ms

Merklize ( approach 2 ) using Rescue Prime on F(2**64 - 2**32 + 1) elements 👇

     leaves		          total
    1048576		        770.228 ms
    2097152		        1500.47 ms
    4194304		        2987.23 ms
    8388608		        5974.54 ms
```
