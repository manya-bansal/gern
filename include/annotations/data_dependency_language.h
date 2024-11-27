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

  void accept(ExprVisitorStrict *v) const { node->accept(v); }
  bool isDefined() const { return (node == nullptr); }
  std::shared_ptr<const ExprNode> getNode() const { return node; }

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

  void accept(StmtVisitorStrict *v) const { node->accept(v); }
  bool isDefined() const { return (node == nullptr); }
  std::shared_ptr<const StmtNode> getNode() const { return node; }

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

class Constraint : public Expr {
public:
  Constraint(Expr e, Expr where = Expr());
};

class Subset : public Stmt {
public:
  Subset(std::shared_ptr<const AbstractDataType> data,
         std::vector<Expr> mdFields);
};

class Subsets : public Stmt {
public:
  Subsets(const std::vector<Subset> &subsets);
};

class For : public Stmt {
public:
  For(Variable v, Expr start, Expr end, Expr step, Stmt body,
      bool parallel = false);
};

class Produces : public Stmt {
public:
  Produces(Subset s);
};

class Consumes : public Stmt {
public:
  Consumes(Stmt stmt);
};

class Allocates : public Stmt {
public:
  Allocates(Expr reg = Expr(), Expr smem = Expr());
};

class Computes : public Stmt {
public:
  Computes(Produces p, Consumes c, Allocates a = Allocates());
};

/**
 * \brief Checks whether a data dependence pattern.
 *
 * A data dependence pattern is valid if:
 * - It contains 1 produces and 1 consumes node at the same nesting level
 *    inside a for node.
 * - The consumer can only contain a vector of subsets, optionally nested inside
 *    a for node.
 * - Different bound variables are introduced for every interval (no shadowing).
 *
 * \param s The data dependence pattern to check.
 * \return Whether the data dependence pattern is valid.
 */
bool isValidDataDependencyPattern(Stmt s);

} // namespace gern
#endif