#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/data_dependency_language.h"
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
    ADTMemberNode(AbstractDataTypePtr ds, const std::string &member)
        : ds(ds), member(member) {
    }
    AbstractDataTypePtr ds;
    std::string member;
    void accept(ExprVisitorStrict *v) const override {
        v->visit(this);
    }
};

#define DEFINE_BINARY_NODE(NAME, NODE)                       \
    struct NAME : public NODE##Node {                        \
    public:                                                  \
        NAME(Expr a, Expr b) : a(a), b(b) {};                \
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

struct GridDimNode : public ConstraintNode {
    GridDimNode(const Grid::Dim &dim)
        : dim(dim) {
    }
    void accept(ConstraintVisitorStrict *v) const override {
        v->visit(this);
    }
    Grid::Dim dim;
};

// Assignment Node
DEFINE_BINARY_NODE(AssignNode, Stmt)

struct SubsetNode : public StmtNode {
    SubsetNode(AbstractDataTypePtr data,
               std::vector<Expr> mdFields)
        : data(data), mdFields(mdFields) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    AbstractDataTypePtr data;
    std::vector<Expr> mdFields;
};

struct ProducesNode : public StmtNode {
    ProducesNode(AbstractDataTypePtr ds, std::vector<Variable> v)
        : output(SubsetObj(ds, std::vector<Expr>(v.begin(), v.end()))) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    std::vector<Variable> getFieldsAsVars() const;
    SubsetObj output;
};

struct ConsumesNode : public StmtNode {
    ConsumesNode() = default;
    virtual void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
};

struct ConsumesForNode : public ConsumesNode {
    ConsumesForNode(Assign start, ADTMember end, Variable step, ConsumeMany body,
                    bool parallel = false)
        : start(start), end(end), step(step), body(body),
          parallel(parallel) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    Assign start;
    ADTMember end;
    Variable step;
    ConsumeMany body;
    bool parallel;
};

struct SubsetObjManyNode : public ConsumesNode {
    SubsetObjManyNode(const std::vector<SubsetObj> &subsets)
        : subsets(subsets) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    std::vector<SubsetObj> subsets;
};

struct PatternNode : public StmtNode {
    PatternNode() = default;
    virtual void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
};

struct ComputesForNode : public PatternNode {
    ComputesForNode(Assign start, ADTMember end, Variable step, Pattern body,
                    bool parallel = false)
        : start(start), end(end), step(step), body(body),
          parallel(parallel) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    Assign start;
    ADTMember end;
    Variable step;
    Pattern body;
    bool parallel;
};

struct AllocatesNode : public StmtNode {
    AllocatesNode(Expr reg, Expr smem)
        : reg(reg), smem(smem) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    Expr reg;
    Expr smem;
};

struct ComputesNode : public PatternNode {
    ComputesNode(Produces p, Consumes c, Allocates a)
        : p(p), c(c), a(a) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    Produces p;
    Consumes c;
    Allocates a;
};

struct AnnotationNode : public StmtNode {
    AnnotationNode(Pattern p,
                   std::set<Grid::Unit> occupied,
                   std::vector<Constraint> constraints)
        : p(p),
          occupied(occupied),
          constraints(constraints) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    Pattern p;
    std::set<Grid::Unit> occupied;
    std::vector<Constraint> constraints;
};

template<typename E>
inline bool isa(const ExprNode *e) {
    return e != nullptr && dynamic_cast<const E *>(e) != nullptr;
}

template<typename E>
inline const E *to(const ExprNode *e) {
    assert(isa<E>(e));
    return static_cast<const E *>(e);
}

template<typename E>
inline bool isa(const StmtNode *e) {
    return e != nullptr && dynamic_cast<const E *>(e) != nullptr;
}

template<typename E>
inline const E *to(const StmtNode *e) {
    assert(isa<E>(e));
    return static_cast<const E *>(e);
}

template<typename E>
inline bool isa(const ConstraintNode *e) {
    return e != nullptr && dynamic_cast<const E *>(e) != nullptr;
}

template<typename E>
inline const E *to(const ConstraintNode *e) {
    assert(isa<E>(e));
    return static_cast<const E *>(e);
}

template<typename I>
inline const typename I::Node *getNode(const I &stmt) {
    assert(isa<typename I::Node>(stmt.ptr));
    return static_cast<const typename I::Node *>(stmt.ptr);
}

}  // namespace gern