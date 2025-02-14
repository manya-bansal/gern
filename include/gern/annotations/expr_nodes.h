#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/expr.h"
#include "annotations/visitor.h"
#include <any>

namespace gern {

struct LiteralNode : public ExprNode {
    template<typename T>
    explicit LiteralNode(T val)
        : ExprNode(type<T>()), val(val) {
    }

    void accept(ExprVisitorStrict *v) const override {
        v->visit(this);
    }

    template<typename T>
    T getVal() const {
        return std::any_cast<T>(val);
    }

    std::any val;
};
std::ostream &operator<<(std::ostream &os, const LiteralNode &);

struct VariableNode : public ExprNode {
    VariableNode(const std::string &name,
                 Grid::Unit p = Grid::Unit::UNDEFINED,
                 Datatype type = Datatype::Int64, bool const_expr = false,
                 bool bound = false, int64_t val = 0)
        : ExprNode(Datatype::Kind::Int64), name(name), p(p),
          type(type), const_expr(const_expr), bound(bound), val(val) {
    }
    void accept(ExprVisitorStrict *v) const override {
        v->visit(this);
    }

    std::string name;
    Grid::Unit p;
    Datatype type;
    bool const_expr;
    bool bound;
    int64_t val;
};

struct ADTMemberNode : public ExprNode {
    ADTMemberNode(AbstractDataTypePtr ds,
                  const std::string &member,
                  bool const_expr)
        : ds(ds), member(member), const_expr(const_expr) {
    }
    AbstractDataTypePtr ds;
    std::string member;
    bool const_expr;
    void accept(ExprVisitorStrict *v) const override {
        v->visit(this);
    }
};

#define DEFINE_BINARY_NODE(NAME, NODE)                       \
    struct NAME : public NODE##Node {                        \
    public:                                                  \
        NAME(Expr a, Expr b) : a(a), b(b){};                 \
        void accept(NODE##VisitorStrict *v) const override { \
            v->visit(this);                                  \
        }                                                    \
        Expr a;                                              \
        Expr b;                                              \
    };

DEFINE_BINARY_NODE(AddNode, Expr)
DEFINE_BINARY_NODE(SubNode, Expr)
DEFINE_BINARY_NODE(DivNode, Expr)
DEFINE_BINARY_NODE(MulNode, Expr)
DEFINE_BINARY_NODE(ModNode, Expr)

DEFINE_BINARY_NODE(AndNode, Constraint)
DEFINE_BINARY_NODE(OrNode, Constraint)

DEFINE_BINARY_NODE(EqNode, Constraint)
DEFINE_BINARY_NODE(NeqNode, Constraint)
DEFINE_BINARY_NODE(LeqNode, Constraint)
DEFINE_BINARY_NODE(GeqNode, Constraint)
DEFINE_BINARY_NODE(LessNode, Constraint)
DEFINE_BINARY_NODE(GreaterNode, Constraint)

struct GridDimNode : public ExprNode {
    GridDimNode(const Grid::Dim &dim)
        : dim(dim) {
    }
    void accept(ExprVisitorStrict *v) const override {
        v->visit(this);
    }
    Grid::Dim dim;
};

}  // namespace gern
