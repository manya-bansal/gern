#include <ginac/ginac.h>

#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "resolve_constraints/resolve_constraints.h"
#include "utils/debug.h"

namespace gern {
namespace resolve {

#define VISIT_AND_DECLARE(op, operation)                                       \
  void visit(const op *node) {                                                 \
    node->a.accept(this);                                                      \
    auto a = expr;                                                             \
    node->b.accept(this);                                                      \
    expr = operation(a, expr);                                                 \
  }

#define COMPLAIN(op)                                                           \
  void visit(const op *node) { assert("bad"); }

typedef std::map<const VariableNode *, GiNaC::symbol> SymbolVariableMap;

GiNaC::ex convertToGinac() {}

std::map<Variable, Expr> solve(std::vector<Expr> system_of_equations) {
  // Generate a GiNaC symbol for each variable node that we want to lower.
  SymbolVariableMap symbols;

  for (const auto &eq : system_of_equations) {
    match(eq, std::function<void(const VariableNode *, Matcher *)>(
                  [&](const VariableNode *op, Matcher *ctx) {
                    symbols[op] = GiNaC::symbol(op->name);
                  }));
  }

  GiNaC::symbol y("x");
  return {};
}
} // namespace resolve
} // namespace gern