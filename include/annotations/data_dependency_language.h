#ifndef GERN_DATA_DEP_LANG_H
#define GERN_DATA_DEP_LANG_H

#include "annotations/abstract_nodes.h"
#include <memory>

namespace gern {

class Expr {
public:
  Expr() : node(std::shared_ptr<const ExprNode>(nullptr)) {}
  Expr(std::shared_ptr<const ExprNode> e) : node(e) {}

  Expr(uint8_t);
  Expr(uint16_t);
  Expr(uint32_t);
  Expr(uint64_t);
  Expr(int8_t);
  Expr(int16_t);
  Expr(int32_t);
  Expr(int64_t);
  Expr(float);
  Expr(double);

  bool isDefined() { return (node == nullptr); }
  std::shared_ptr<const ExprNode> getNode() { return node; }

private:
  std::shared_ptr<const ExprNode> node;
};

std::ostream &operator<<(std::ostream &os, const Expr &);
Expr operator+(const Expr &, const Expr &);
Expr operator-(const Expr &, const Expr &);
Expr operator*(const Expr &, const Expr &);
Expr operator/(const Expr &, const Expr &);
Expr operator%(const Expr &, const Expr &);

class Stmt {
public:
  Stmt() : node(std::shared_ptr<const StmtNode>(nullptr)) {}
  Stmt(std::shared_ptr<const StmtNode> e) : node(e) {}

  bool isDefined() { return (node == nullptr); }
  std::shared_ptr<const StmtNode> getNode() { return node; }

private:
  std::shared_ptr<const StmtNode> node;
};

std::ostream &operator<<(std::ostream &os, const Stmt &);
Expr operator==(const Expr &, const Expr &);
Expr operator!=(const Expr &, const Expr &);
Expr operator<=(const Expr &, const Expr &);
Expr operator>=(const Expr &, const Expr &);
Expr operator<(const Expr &, const Expr &);
Expr operator>(const Expr &, const Expr &);
Expr operator&&(const Expr &, const Expr &);
Expr operator||(const Expr &, const Expr &);

class Variable : public Expr {
public:
  Variable(const std::string &name);
};

#define DEFINE_BINARY_EXPR_CLASS(NAME)                                         \
  class NAME : public Expr {                                                   \
  public:                                                                      \
    NAME(Expr a, Expr b);                                                      \
    Expr a;                                                                    \
    Expr b;                                                                    \
  };

DEFINE_BINARY_EXPR_CLASS(Add);
DEFINE_BINARY_EXPR_CLASS(Sub);
DEFINE_BINARY_EXPR_CLASS(Div);
DEFINE_BINARY_EXPR_CLASS(Mul);
DEFINE_BINARY_EXPR_CLASS(Mod);

DEFINE_BINARY_EXPR_CLASS(And);
DEFINE_BINARY_EXPR_CLASS(Or);

DEFINE_BINARY_EXPR_CLASS(Eq);
DEFINE_BINARY_EXPR_CLASS(Neq);
DEFINE_BINARY_EXPR_CLASS(Leq);
DEFINE_BINARY_EXPR_CLASS(Geq);
DEFINE_BINARY_EXPR_CLASS(Less);
DEFINE_BINARY_EXPR_CLASS(Greater);

class VarDecl : public Stmt {
public:
  VarDecl(Variable b, Stmt where = Stmt());
};

} // namespace gern
#endif