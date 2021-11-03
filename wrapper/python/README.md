## Background

I write respective `Python` wrapper functions for abstracting interaction with shared object, which is compiled using `make genlib`. Check wrapper definition in module [`ff_p`](ff_p.py).

> Before you use Python module defined in this directory, make sure you've run `make genlib` and generated shared object.

Now you can import `ff_p` and perform basic prime field arithmetics.

> Note, prime field modulus is `2**64 - 2**32 + 1`

```python3
from ff_p import *

assert 3 == ff_p_add(1, 2)
assert 18446744069414584320 == ff_p_sub(1, 2)
assert 19456 == ff_p_multiply(19, 1 << 10)
assert (1 << 10) == ff_p_exponentiate(2, 10)
assert 18302628881372282881 == ff_p_inverse(128)
assert 8198552919739815254 == ff_p_divide(2, 9)
```

## Tests

There're some **randomised** testcases I've written, for strongly asserting results produced by `ff_p` module are indeed correct, using another Finite Field implementation `galois` module.

For running those test cases, I suggest you start with enabling `virtualenv`


```bash
# assuming you're in this directory

python3 -m pip install --user virtualenv # if `venv` not installed
python3 -m venv .
source bin/activate

# you're inside `venv` now
```

Now you can install required dependencies inside `venv`

```bash
python3 -m pip install -r requirements.txt
```

Finally run randomized test cases

```bash
python3 ff_p_test.py
```

When done, consider deactivating virtual environment

```bash
deactivate
```
