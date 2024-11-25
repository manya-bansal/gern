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

struct AddNode : public ExprNode {
  AddNode(Expr a, Expr b) : a(a), b(b) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  Expr a;
  Expr b;
};

struct SubNode : public ExprNode {
  SubNode(Expr a, Expr b) : a(a), b(b) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  Expr a;
  Expr b;
};

struct DivNode : public ExprNode {
  DivNode(Expr a, Expr b) : a(a), b(b) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  Expr a;
  Expr b;
};

struct MulNode : public ExprNode {
  MulNode(Expr a, Expr b) : a(a), b(b) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  Expr a;
  Expr b;
};

struct ModNode : public ExprNode {
  ModNode(Expr a, Expr b) : a(a), b(b) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  Expr a;
  Expr b;
};

} // namespace gern

#endif