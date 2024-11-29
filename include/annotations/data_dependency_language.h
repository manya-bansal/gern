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
  bool isDefined() const { return (node != nullptr); }
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

class Constraint {
public:
  Constraint() : node(std::shared_ptr<const ConstraintNode>(nullptr)) {}
  Constraint(std::shared_ptr<const ConstraintNode> node) : node(node) {}

  void accept(ConstraintVisitorStrict *v) const { node->accept(v); }
  bool isDefined() const { return (node != nullptr); }
  std::shared_ptr<const ConstraintNode> getNode() const { return node; }

private:
  std::shared_ptr<const ConstraintNode> node;
};
std::ostream &operator<<(std::ostream &os, const Constraint &);
Constraint operator==(const Expr &, const Expr &);
Constraint operator!=(const Expr &, const Expr &);
Constraint operator<=(const Expr &, const Expr &);
Constraint operator>=(const Expr &, const Expr &);
Constraint operator<(const Expr &, const Expr &);
Constraint operator>(const Expr &, const Expr &);
Constraint operator&&(const Expr &, const Expr &);
Constraint operator||(const Expr &, const Expr &);

class Stmt {
public:
  Stmt() : node(std::shared_ptr<const StmtNode>(nullptr)) {}
  Stmt(std::shared_ptr<const StmtNode> e) : node(e) {}

  /**
   * @brief Add a constraint to a statement
   *
   *  The function checks that only variables that are in
   *  scope are used within the constraint.
   *
   * @param constraint Constraint to add.
   * @return Stmt New statement with the constraint attached.
   */
  Stmt where(Constraint constraint);

  void accept(StmtVisitorStrict *v) const { node->accept(v); }
  bool isDefined() const { return (node != nullptr); }
  std::shared_ptr<const StmtNode> getNode() const { return node; }

private:
  Stmt(std::shared_ptr<const StmtNode> e, Constraint c) : node(e), c(c) {}
  std::shared_ptr<const StmtNode> node;
  Constraint c;
};

std::ostream &operator<<(std::ostream &os, const Stmt &);

class Variable : public Expr {
public:
  Variable(const std::string &name);
};

#define DEFINE_BINARY_CLASS(NAME, NODE)                                        \
  class NAME : public NODE {                                                   \
  public:                                                                      \
    NAME(Expr a, Expr b);                                                      \
    Expr a;                                                                    \
    Expr b;                                                                    \
  };

DEFINE_BINARY_CLASS(Add, Expr);
DEFINE_BINARY_CLASS(Sub, Expr);
DEFINE_BINARY_CLASS(Div, Expr);
DEFINE_BINARY_CLASS(Mul, Expr);
DEFINE_BINARY_CLASS(Mod, Expr);

DEFINE_BINARY_CLASS(And, Constraint);
DEFINE_BINARY_CLASS(Or, Constraint);

DEFINE_BINARY_CLASS(Eq, Constraint);
DEFINE_BINARY_CLASS(Neq, Constraint);
DEFINE_BINARY_CLASS(Leq, Constraint);
DEFINE_BINARY_CLASS(Geq, Constraint);
DEFINE_BINARY_CLASS(Less, Constraint);
DEFINE_BINARY_CLASS(Greater, Constraint);

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
  Allocates() : Stmt() {}
  Allocates(Expr reg, Expr smem = Expr());
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
 * - The consumer can only contain a vector of subsets, optionally nested
 * inside a for node.
 * - Different bound variables are introduced for every interval (no
 * shadowing).
 *
 * \param s The data dependence pattern to check.
 * \return Whether the data dependence pattern is valid.
 */
bool isValidDataDependencyPattern(Stmt s);

} // namespace gern
#endif