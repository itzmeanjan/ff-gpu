## Background

I provide with `C` wrapper function definitions which can be compiled/ linked to relocatable shared object, having prime field arithmetics defined inside it. This shared object can now be interacted from higher level programming languages like `Python`, for offloading computation to default accelerator available on platform.

You can compile and produce dynamically linked shared object using

```bash
# assuming you're in this directory

cd ..
make genlib
cd wrapper; file libff_p.so
```

> For example on how to interact with shared object from `Python`, check [here](python).
