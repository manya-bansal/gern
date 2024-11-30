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
    Expr getA() const;                                                         \
    Expr getB() const;                                                         \
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

Add operator+(const Expr &, const Expr &);
Sub operator-(const Expr &, const Expr &);
Mul operator*(const Expr &, const Expr &);
Div operator/(const Expr &, const Expr &);
Mod operator%(const Expr &, const Expr &);
Eq operator==(const Expr &, const Expr &);
Neq operator!=(const Expr &, const Expr &);
Leq operator<=(const Expr &, const Expr &);
Geq operator>=(const Expr &, const Expr &);
Less operator<(const Expr &, const Expr &);
Greater operator>(const Expr &, const Expr &);
And operator&&(const Expr &, const Expr &);
Or operator||(const Expr &, const Expr &);

class Subset : public Stmt {
public:
  Subset(std::shared_ptr<const AbstractDataType> data,
         std::vector<Expr> mdFields);
};

class Produces : public Stmt {
public:
  Produces(Subset s);
};

struct ConsumesNode;

class Consumes : public Stmt {
public:
  Consumes(std::shared_ptr<const ConsumesNode>);
  Consumes(Subset s);
};

class ConsumeMany : public Consumes {
public:
  ConsumeMany(std::shared_ptr<const ConsumesNode> s) : Consumes(s) {};
};

class Subsets : public ConsumeMany {
public:
  Subsets(const std::vector<Subset> &subsets);
  Subsets(Subset s) : Subsets(std::vector<Subset>{s}) {}
};

// This ensures that a consumes node will only ever contain a for loop
// or a list of subsets. In this way, we can leverage the cpp type checker to
// ensures that only legal patterns are written down.
ConsumeMany For(Variable v, Expr start, Expr end, Expr step, ConsumeMany body,
                bool parallel = false);

class Allocates : public Stmt {
public:
  Allocates() : Stmt() {}
  Allocates(Expr reg, Expr smem = Expr());
};

struct PatternNode;
class Pattern : public Stmt {
public:
  Pattern(std::shared_ptr<const PatternNode>);
};

class Computes : public Pattern {
public:
  Computes(Produces p, Consumes c, Allocates a = Allocates());
};

// This ensures that a computes node will only ever contain a for loop
// or a (Produces, Consumes) node. In this way, we can leverage the cpp type
// checker to ensures that only legal patterns are written down.
Pattern For(Variable v, Expr start, Expr end, Expr step, Pattern body,
            bool parallel = false);

} // namespace gern
#endif