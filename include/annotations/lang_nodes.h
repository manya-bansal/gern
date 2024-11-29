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

#define DEFINE_BINARY_NODE(NAME, NODE)                                         \
  struct NAME : public NODE##Node {                                            \
  public:                                                                      \
    NAME(Expr a, Expr b) : a(a), b(b) {};                                      \
    void accept(NODE##VisitorStrict *v) const override { v->visit(this); }     \
    Expr a;                                                                    \
    Expr b;                                                                    \
  };

DEFINE_BINARY_NODE(AddNode, Expr);
DEFINE_BINARY_NODE(SubNode, Expr);
DEFINE_BINARY_NODE(DivNode, Expr);
DEFINE_BINARY_NODE(MulNode, Expr);
DEFINE_BINARY_NODE(ModNode, Expr);

DEFINE_BINARY_NODE(AndNode, Constraint);
DEFINE_BINARY_NODE(OrNode, Constraint);

DEFINE_BINARY_NODE(EqNode, Constraint);
DEFINE_BINARY_NODE(NeqNode, Constraint);
DEFINE_BINARY_NODE(LeqNode, Constraint);
DEFINE_BINARY_NODE(GeqNode, Constraint);
DEFINE_BINARY_NODE(LessNode, Constraint);
DEFINE_BINARY_NODE(GreaterNode, Constraint);

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

struct AllocatesNode : public StmtNode {
  AllocatesNode(Expr reg, Expr smem) : reg(reg), smem(smem) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  Expr reg;
  Expr smem;
};

struct ComputesNode : public StmtNode {
  ComputesNode(Produces p, Consumes c, Allocates a) : p(p), c(c), a(a) {}
  void accept(StmtVisitorStrict *v) const override { v->visit(this); }
  Produces p;
  Consumes c;
  Allocates a;
};

} // namespace gern

#endif