#include "resolve_constraints/resolve_constraints.h"
#include <gtest/gtest.h>

using namespace gern;
TEST(ResolveConstraints, Simple) { resolve::solve({}); }