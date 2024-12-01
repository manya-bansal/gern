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

typedef std::map<const VariableNode *, GiNaC::symbol> VariableToSymbolMap;
typedef std::map<GiNaC::symbol, const VariableNode *> SymbolToVariableMap;

// Helper function to convert an equality constraint to a GiNaC
// expression. Currently, only equality constraints are accepted.
static GiNaC::ex convertToGinac(Eq q, VariableToSymbolMap names) {
  struct ExprToGinac : public ExprVisitorStrict {
    ExprToGinac(VariableToSymbolMap names) : names(names) {}
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
    VariableToSymbolMap names;
  };

  ExprToGinac convert_a{names};
  convert_a.visit(q.getA());
  GiNaC::ex a = convert_a.ginacExpr;
  ExprToGinac convert_b{names};
  convert_b.visit(q.getB());
  GiNaC::ex b = convert_b.ginacExpr;

  return a == b;
}

static Expr convertToGern(GiNaC::ex ginacExpr, SymbolToVariableMap names) {

  struct GinacToExpr : public GiNaC::visitor, public GiNaC::symbol::visitor {
    GinacToExpr(SymbolToVariableMap variables) : variables(variables) {}
    void visit(const GiNaC::symbol &s) {}

    SymbolToVariableMap variables;
    Expr e;
  };

  GinacToExpr convertor{names};
  ginacExpr.accept(convertor);

  return convertor.e;
}

std::map<Variable, Expr> solve(std::vector<Eq> system_of_equations) {
  // Generate a GiNaC symbol for each variable node that we want to lower.
  VariableToSymbolMap symbols;

  for (const auto &eq : system_of_equations) {
    match(eq, std::function<void(const VariableNode *, Matcher *)>(
                  [&](const VariableNode *op, Matcher *ctx) {
                    symbols[op] = GiNaC::symbol(op->name);
                  }));
  }

  GiNaC::lst ginacSystemOfEq;
  for (const auto &eq : system_of_equations) {
    DEBUG(convertToGinac(eq, symbols));
    ginacSystemOfEq.append(convertToGinac(eq, symbols));
  }

  SymbolToVariableMap variables;
  GiNaC::lst to_solve;
  for (const auto &var : symbols) {
    to_solve.append(var.second);
    variables[var.second] = var.first;
  }

  GiNaC::ex solutions = GiNaC::lsolve(ginacSystemOfEq, to_solve);
  std::cout << "Solved" << solutions;

  // class VariableCollector : public AnnotVisitor {
  // public:
  //   using AnnotVisitor::visit;
  //   void visit(Expr e) {
  //     if (dynamic_cast<const VariableNode *>(e.getNode().get()) != nullptr) {
  //       visit(Variable(e.getNode()));
  //     }
  //     AnnotVisitor::visit(e);
  //   };
  // };

  // Variable V{"V"};
  // Expr e = V;
  // VisitorTest v;
  // v.visit(e);

  GiNaC::symbol y("x");
  return {};
}
} // namespace resolve
} // namespace gern