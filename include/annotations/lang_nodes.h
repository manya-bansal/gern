#ifndef GERN_LANG_NODES_H
#define GERN_LANG_NODES_H

#include "annotations/abstract_nodes.h"
#include "annotations/data_dependency_language.h"
#include "annotations/visitor.h"

#include <any>

namespace gern {

struct LiteralNode : public ExprNode {
  template <typename T>
  explicit LiteralNode(T val) : ExprNode(type<T>()), val(val) {}

  void accept(ExprVisitorStrict *v) const override { v->visit(this); }

  template <typename T> T getVal() const { return std::any_cast<T>(val); }

  std::any val;
};
std::ostream &operator<<(std::ostream &os, const LiteralNode &);

struct VariableNode : public ExprNode {
  VariableNode(const std::string &name)
      : ExprNode(Datatype::Kind::Int64), name(name) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  std::string name;
};

#define DEFINE_BINARY_EXPR_NODE(NAME)                                          \
  struct NAME : public ExprNode {                                              \
  public:                                                                      \
    NAME(Expr a, Expr b) : a(a), b(b) {};                                      \
    void accept(ExprVisitorStrict *v) const override { v->visit(this); }       \
    Expr a;                                                                    \
    Expr b;                                                                    \
  };

DEFINE_BINARY_EXPR_NODE(AddNode);
DEFINE_BINARY_EXPR_NODE(SubNode);
DEFINE_BINARY_EXPR_NODE(DivNode);
DEFINE_BINARY_EXPR_NODE(MulNode);
DEFINE_BINARY_EXPR_NODE(ModNode);

DEFINE_BINARY_EXPR_NODE(AndNode);
DEFINE_BINARY_EXPR_NODE(OrNode);

DEFINE_BINARY_EXPR_NODE(EqNode);
DEFINE_BINARY_EXPR_NODE(NeqNode);
DEFINE_BINARY_EXPR_NODE(LeqNode);
DEFINE_BINARY_EXPR_NODE(GeqNode);
DEFINE_BINARY_EXPR_NODE(LessNode);
DEFINE_BINARY_EXPR_NODE(GreaterNode);

struct ConstraintNode : public StmtNode {
  ConstraintNode(Variable v, Expr where = Expr()) : v(v), where(where) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  Variable v;
  Expr where;
};

} // namespace gern

#endif