#include <CL/sycl.hpp>

// extension field of which prime number
inline constexpr uint32_t CHARACTERISTIC = 0b10;
// maximum degree of field polynomial
inline constexpr uint32_t DEGREE = 0b100000;
// number of elements present in field
inline constexpr uint32_t ORDER = CHARACTERISTIC << (DEGREE - 0b1);
// irreducible polynomial for 2**32 field:
// x^32 + x^15 + x^9 + x^7 + x^4 + x^3 + 1
inline constexpr uint32_t IRREDUCIBLE_POLY =
    0b100000000000000001000001010011001;

// adds two finite field elements
uint32_t ff_add(const uint32_t a, const uint32_t b);

// subtracts one field element from another one, implementation
// is same as `ff_add`
uint32_t ff_sub(const uint32_t a, const uint32_t b);

// negation of one finite field element is same
// as the given number, because each field element
// is additive inverse to self
uint32_t ff_neg(const uint32_t a);

// multiplies two finite field elements
// while respecting field rules
uint32_t ff_mult(const uint32_t a, const uint32_t b);

// inverts a field element, given that it's not
// additive identity of field
uint32_t ff_inv(const uint32_t a);
