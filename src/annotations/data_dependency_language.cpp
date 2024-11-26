#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"

namespace gern {

Expr::Expr(uint8_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(uint16_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(uint32_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(uint64_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int8_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int16_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int32_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int64_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(float val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(double val) : Expr(std::make_shared<const LiteralNode>(val)) {}

Variable::Variable(const std::string &name)
    : Expr(std::make_shared<const VariableNode>(name)) {}

std::ostream &operator<<(std::ostream &os, const Expr &e) {
  Printer p{os};
  p.visit(e);
  return os;
}

#define DEFINE_BINARY_OPERATOR(CLASS_NAME, OPERATOR)                           \
  Expr operator OPERATOR(const Expr &a, const Expr &b) {                       \
    return CLASS_NAME(a, b);                                                   \
  }

DEFINE_BINARY_OPERATOR(Add, +)
DEFINE_BINARY_OPERATOR(Sub, -)
DEFINE_BINARY_OPERATOR(Mul, *)
DEFINE_BINARY_OPERATOR(Div, /)
DEFINE_BINARY_OPERATOR(Mod, %)
DEFINE_BINARY_OPERATOR(Eq, ==)
DEFINE_BINARY_OPERATOR(Neq, !=)
DEFINE_BINARY_OPERATOR(Leq, <=)
DEFINE_BINARY_OPERATOR(Geq, >=)
DEFINE_BINARY_OPERATOR(Less, <)
DEFINE_BINARY_OPERATOR(Greater, >)
DEFINE_BINARY_OPERATOR(And, &&)
DEFINE_BINARY_OPERATOR(Or, ||)

std::ostream &operator<<(std::ostream &os, const Stmt &s) {
  Printer p{os};
  p.visit(s);
  return os;
}

#define DEFINE_BINARY_EXPR_CONSTRUCTOR(CLASS_NAME)                             \
  CLASS_NAME::CLASS_NAME(Expr a, Expr b)                                       \
      : Expr(std::make_shared<const CLASS_NAME##Node>(a, b)) {}

DEFINE_BINARY_EXPR_CONSTRUCTOR(Add)
DEFINE_BINARY_EXPR_CONSTRUCTOR(Sub)
DEFINE_BINARY_EXPR_CONSTRUCTOR(Div)
DEFINE_BINARY_EXPR_CONSTRUCTOR(Mod)
DEFINE_BINARY_EXPR_CONSTRUCTOR(Mul)

DEFINE_BINARY_EXPR_CONSTRUCTOR(And);
DEFINE_BINARY_EXPR_CONSTRUCTOR(Or);

DEFINE_BINARY_EXPR_CONSTRUCTOR(Eq);
DEFINE_BINARY_EXPR_CONSTRUCTOR(Neq);
DEFINE_BINARY_EXPR_CONSTRUCTOR(Leq);
DEFINE_BINARY_EXPR_CONSTRUCTOR(Geq);
DEFINE_BINARY_EXPR_CONSTRUCTOR(Less);
DEFINE_BINARY_EXPR_CONSTRUCTOR(Greater);

Constraint::Constraint(Variable v, Expr where)
    : Stmt(std::make_shared<const ConstraintNode>(v, where)) {}

} // namespace gern