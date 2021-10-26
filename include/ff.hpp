#include <CL/sycl.hpp>

// adds two finite field elements
uint32_t ff_add(const uint32_t a, const uint32_t b);

// subtracts one field element from another one, implementation
// is same as `ff_add`
uint32_t ff_sub(const uint32_t a, const uint32_t b);
