#pragma once
#define tolerance 1e-3

#define assert_close(a, b) assert(std::abs(a - b) < tolerance)