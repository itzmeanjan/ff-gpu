#pragma once
#include <cassert>
#include <merkle_tree.hpp>
#include <random>

// Test that merkle tree construction kernel is working as expected !
void
test_merklize(sycl::queue& q);
