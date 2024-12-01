#include "resolve_constraints/resolve_constraints.h"
#include "test-utils.h"

#include <gtest/gtest.h>

using namespace gern;

TEST(ResolveConstraints, Simple) {
  Variable x("x");
  Variable y("y");
  auto sol = resolve::solve(x == y, x);

  ASSERT_TRUE(sol.ptr == y.ptr);

  sol = resolve::solve((x - 1) == y, x);
  auto string_sol = getStrippedString(sol);
  ASSERT_TRUE((string_sol == "(y+1)") || string_sol == "(1+y)");

  sol = resolve::solve(x == y - 1, x);
  string_sol = getStrippedString(sol);
  ASSERT_TRUE((string_sol == "(y+-1)") || string_sol == "(-1+y)");
}

TEST(ResolveConstraints, Multiply) {
  Variable x("x");
  Variable y("y");
  Variable z("z");

  auto sol = resolve::solve((x * 4) == y, x);
  auto string_sol = getStrippedString(sol);
  ASSERT_TRUE(string_sol == "(y*0.25)" || string_sol == "x");

  sol = resolve::solve(x * z == y, x);
  string_sol = getStrippedString(sol);
  ASSERT_TRUE((string_sol == "(y*(1/z))") || string_sol == "((1/z)*y)");
}
