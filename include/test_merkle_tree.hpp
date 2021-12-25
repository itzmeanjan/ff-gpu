#pragma once
#include <merkle_tree.hpp>
#include <random>
#include <cassert>

// Test that merkle tree construction kernel is working as expected !
void
test_merklize(sycl::queue& q);
