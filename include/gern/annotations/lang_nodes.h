#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/data_dependency_language.h"
#include "annotations/expr_nodes.h"
#include "annotations/visitor.h"

#include <any>

namespace gern {

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
    ConsumesForNode(Assign start, Expr parameter, Variable step, ConsumeMany body,
                    bool parallel = false)
        : start(start), parameter(parameter), step(step), body(body),
          parallel(parallel) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    Assign start;
    Expr parameter;
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
    ComputesForNode(Assign start, Expr parameter, Variable step, Pattern body,
                    bool parallel = false)
        : start(start), parameter(parameter), step(step), body(body),
          parallel(parallel) {
    }
    void accept(StmtVisitorStrict *v) const override {
        v->visit(this);
    }
    Assign start;
    Expr parameter;
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