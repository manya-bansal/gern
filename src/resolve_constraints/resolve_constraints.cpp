#include <ginac/ginac.h>

#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "resolve_constraints/resolve_constraints.h"
#include "utils/debug.h"

namespace gern {
namespace resolve {

// #define VISIT_AND_DECLARE(op, operation)                                       \
//   void visit(const op *node) {                                                 \
//     node->a.accept(this);                                                      \
//     auto a = expr;                                                             \
//     node->b.accept(this);                                                      \
//     expr = operation(a, expr);                                                 \
//   }

// #define COMPLAIN(op)                                                           \
//   void visit(const op *node) { assert("bad"); }


std::map<Variable, Expr> solve(std::vector<Expr> system_of_equations) {
  // Generate a SymEngine symbol for each variable node that we want to lower.
  GiNaC::symbol t("x");
  return {};
}
} // namespace resolve
} // namespace gern