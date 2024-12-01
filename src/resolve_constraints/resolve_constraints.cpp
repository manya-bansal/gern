#include <ginac/ginac.h>

#include "annotations/data_dependency_language.h"
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

struct GinacLess {
  bool operator()(const GiNaC::symbol &a, const GiNaC::symbol &b) const {
    return GiNaC::ex_is_less()(a, b);
  }
};

typedef std::map<Variable, GiNaC::symbol, ExprLess> VariableToSymbolMap;
typedef std::map<GiNaC::symbol, Variable, GinacLess> SymbolToVariableMap;

// Helper function to convert an equality constraint to a GiNaC
// expression. Currently, only equality constraints are accepted.
static GiNaC::ex convertToGinac(Eq q, VariableToSymbolMap names) {
  struct ExprToGinac : public ExprVisitorStrict {
    ExprToGinac(VariableToSymbolMap names) : names(names) {}
    using ExprVisitorStrict::visit;

    void visit(const VariableNode *op) {
      if (names.count(Variable(op)) == 0) {
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

static Expr convertToGern(GiNaC::ex ginacExpr, SymbolToVariableMap variables) {
  std::cout << "Total Expr = " << ginacExpr << std::endl;
  std::cout << "before" << std::endl;
  for (auto entry : variables) {
    std::cout << entry.first << " : " << entry.second << std::endl;
  }
  std::cout << "before" << std::endl;
  struct GinacToExpr : public GiNaC::visitor,
                       public GiNaC::symbol::visitor,
                       public GiNaC::add::visitor,
                       public GiNaC::mul::visitor,
                       public GiNaC::numeric::visitor {

    GinacToExpr(SymbolToVariableMap names) : names(names) {}

    void visit(const GiNaC::numeric &n) {
      if (n.is_integer()) {
        e = Expr(n.to_int());
      }
      if (n.is_real()) {
        e = Expr(n.to_double());
      } else {
        throw error::InternalError("Unimplemented");
      }
    }

    void visit(const GiNaC::symbol &s) {
      if (names.count(s) == 0) {
        throw error::InternalError(
            "Map does not contain a symbol for GiNac::symbol");
      }
      std::cout << "Symbol = " << s << std::endl;
      std::cout << "Return =" << names[s] << std::endl;

      for (auto entry : names) {
        std::cout << entry.first << " : " << entry.second << std::endl;
      }
      e = names[s];
    }

    void visit(const GiNaC::add &a) {
      a.op(0).accept(*this);
      Expr test = e;
      std::cout << "First = " << test << std::endl;
      for (size_t i = 1; i < a.nops(); i++) {
        a.op(i).accept(*this);
        std::cout << "Add Try=" << a.op(i) << "Mine=" << e << std::endl;
        test = test + e;
      }
      e = test;
      std::cout << "Addition = " << test << std::endl;
    }

    void visit(const GiNaC::mul &a) {
      a.op(0).accept(*this);
      Expr test = e;
      std::cout << "1: Try=" << a.op(0) << "Mine=" << e << std::endl;
      for (size_t i = 1; i < a.nops(); i++) {
        a.op(i).accept(*this);
        std::cout << "Try=" << a.op(i) << "Mine=" << e << std::endl;
        test = test * e;
      }
      e = test;
      std::cout << test << std::endl;
    }

    SymbolToVariableMap names;
    Expr e;
  };

  GinacToExpr convertor{variables};
  ginacExpr.accept(convertor);

  return convertor.e;
}

std::map<Variable, Expr, ExprLess> solve(std::vector<Eq> system_of_equations) {
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
    std::cout << var.second << " : " << var.first << std::endl;
    variables[var.second] = var.first;
  }

  std::cout << "before ----  " << std::endl;
  for (auto entry : variables) {
    std::cout << entry.first << " : " << entry.second << std::endl;
  }
  std::cout << "before" << std::endl;

  GiNaC::ex solutions = GiNaC::lsolve(ginacSystemOfEq, to_solve);

  std::cout << "before ----  " << std::endl;
  for (auto entry : variables) {
    std::cout << entry.first << " : " << entry.second << std::endl;
  }
  std::cout << "before" << std::endl;

  std::map<Variable, Expr, ExprLess> solved;
  for (const auto &sol : solutions) {
    assert(GiNaC::is_a<GiNaC::symbol>(sol.lhs()));
    GiNaC::symbol sym = GiNaC::ex_to<GiNaC::symbol>(sol.lhs());
    solved[variables[sym]] = convertToGern(sol.rhs(), variables);
    std::cout << sol.rhs() << std::endl;
  }
  std::cout << "before ----  " << std::endl;
  for (auto entry : variables) {
    std::cout << entry.first << " : " << entry.second << std::endl;
  }
  std::cout << "before" << std::endl;

  auto e = convertToGern(to_solve[0] - to_solve[1], variables);
  std::cout << e << std::endl;
  return solved;
}

} // namespace resolve
} // namespace gern