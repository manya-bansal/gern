#include <ginac/ginac.h>

#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "resolve_constraints/resolve_constraints.h"
#include "utils/debug.h"
#include "utils/error.h"

namespace gern {
namespace resolve {

#define VISIT_AND_DECLARE(op, operation)                                       \
  void visit(const op *node) {                                                 \
    this->visit(node->a);                                                      \
    auto a = ginacExpr;                                                        \
    this->visit(node->b);                                                      \
    ginacExpr = a operation ginacExpr;                                         \
  }

#define COMPLAIN(op)                                                           \
  void visit(const op *node) { assert("bad"); }

typedef std::map<const VariableNode *, GiNaC::symbol> SymbolVariableMap;

GiNaC::ex convertToGinac(Eq q, SymbolVariableMap names) {
  std::cout << q << std::endl;
  struct ExprToGinac : public ExprVisitorStrict {
    ExprToGinac(SymbolVariableMap names) : names(names) {}
    using ExprVisitorStrict::visit;

    void visit(const VariableNode *op) {
      if (names.count(op) == 0) {
        throw error::InternalError(
            "Map does not contain a symbol for variable");
      }
      std::cout << names[op] << std::endl;
      ginacExpr = names[op];
    }

    void visit(const LiteralNode *op) {
      switch (op->getDatatype().getKind()) {
      case Datatype::Int64:
        ginacExpr = op->getVal<int64_t>();
        break;
      case Datatype::Int32:
        ginacExpr = op->getVal<int32_t>();
        break;
      default:
        throw error::InternalError("Unimplemented");
        break;
      }
    }

    VISIT_AND_DECLARE(AddNode, +);
    VISIT_AND_DECLARE(SubNode, -);
    VISIT_AND_DECLARE(MulNode, *);
    VISIT_AND_DECLARE(DivNode, /);
    COMPLAIN(ModNode);

    GiNaC::ex ginacExpr;
    SymbolVariableMap names;
  };

  ExprToGinac convert_a{names};
  convert_a.visit(q.getA());
  GiNaC::ex a = convert_a.ginacExpr;
  ExprToGinac convert_b{names};
  convert_b.visit(q.getB());
  GiNaC::ex b = convert_b.ginacExpr;

  return a == b;
}

std::map<Variable, Expr> solve(std::vector<Eq> system_of_equations) {
  // Generate a GiNaC symbol for each variable node that we want to lower.
  SymbolVariableMap symbols;

  for (const auto &eq : system_of_equations) {
    match(eq, std::function<void(const VariableNode *, Matcher *)>(
                  [&](const VariableNode *op, Matcher *ctx) {
                    symbols[op] = GiNaC::symbol(op->name);
                    std::cout << symbols[op] << std::endl;
                  }));
  }

  GiNaC::lst ginacSystemOfEq;
  for (const auto &eq : system_of_equations) {
    std::cout << convertToGinac(eq, symbols) << std::endl;
    ginacSystemOfEq.append(convertToGinac(eq, symbols));
  }

  for (const auto &eq : system_of_equations) {
    match(eq, std::function<void(const VariableNode *, Matcher *)>(
                  [&](const VariableNode *op, Matcher *ctx) {
                    std::cout
                        << lsolve(ginacSystemOfEq, GiNaC::lst{symbols[op]});
                  }));
  }

  GiNaC::symbol y("x");
  return {};
}
} // namespace resolve
} // namespace gern