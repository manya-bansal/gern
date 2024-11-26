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
  VariableNode(const std::string &name, bool argument = false)
      : ExprNode(Datatype::Kind::Int64), name(name), argument(argument) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  std::string name;
  bool argument;
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

struct ConstraintNode : public ExprNode {
  ConstraintNode(Expr e, Expr where = Expr()) : e(e), where(where) {}
  void accept(ExprVisitorStrict *v) const override { v->visit(this); }
  Expr e;
  Expr where;
};

struct SubsetNode : public StmtNode {
  SubsetNode(std::shared_ptr<const AbstractDataType> data,
             std::vector<Expr> mdFields)
      : data(data), mdFields(mdFields) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  std::shared_ptr<const AbstractDataType> data;
  std::vector<Expr> mdFields;
};

struct SubsetsNode : public StmtNode {
  SubsetsNode(const std::vector<Subset> &subsets) : subsets(subsets) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  std::vector<Subset> subsets;
};

struct ProducesNode : public StmtNode {
  ProducesNode(Subset output) : output(output) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  Subset output;
};

struct ConsumesNode : public StmtNode {
  ConsumesNode(Stmt stmt) : stmt(stmt) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  Stmt stmt;
};

struct ForNode : public StmtNode {
  ForNode(Variable v, Expr start, Expr end, Expr step, Stmt body,
          bool parallel = false)
      : v(v), start(start), end(end), step(step), body(body) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  Variable v;
  Expr start;
  Expr end;
  Expr step;
  Stmt body;
  bool parallel;
};

struct ComputesNode : public StmtNode {
  ComputesNode(Produces p, Consumes c) : p(p), c(c) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  Produces p;
  Consumes c;
};

} // namespace gern

#endif