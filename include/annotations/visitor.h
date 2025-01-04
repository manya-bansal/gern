#ifndef GERN_VISITOR_H
#define GERN_VISITOR_H

#include "annotations/data_dependency_language.h"
#include <cassert>
#include <functional>

namespace gern {

struct LiteralNode;
struct AddNode;
struct MulNode;
struct SubNode;
struct DivNode;
struct ModNode;
struct VariableNode;
struct EqNode;
struct NeqNode;
struct LeqNode;
struct GeqNode;
struct LessNode;
struct GreaterNode;
struct AndNode;
struct OrNode;

struct AssignNode;
struct SubsetNode;
struct SubsetsNode;
struct ProducesNode;
struct ConsumesNode;
struct ConsumesForNode;
struct AllocatesNode;
struct ComputesForNode;
struct ComputesNode;
struct PatternNode;

class ConstraintNode;

class ExprVisitorStrict {
public:
    virtual ~ExprVisitorStrict() = default;

    virtual void visit(Expr);
    virtual void visit(const LiteralNode *) = 0;
    virtual void visit(const AddNode *) = 0;
    virtual void visit(const SubNode *) = 0;
    virtual void visit(const MulNode *) = 0;
    virtual void visit(const DivNode *) = 0;
    virtual void visit(const ModNode *) = 0;
    virtual void visit(const VariableNode *) = 0;
};

class ConstraintVisitorStrict {
public:
    virtual ~ConstraintVisitorStrict() = default;

    virtual void visit(Constraint);
    virtual void visit(const EqNode *) = 0;
    virtual void visit(const NeqNode *) = 0;
    virtual void visit(const LeqNode *) = 0;
    virtual void visit(const GeqNode *) = 0;
    virtual void visit(const LessNode *) = 0;
    virtual void visit(const GreaterNode *) = 0;
    virtual void visit(const OrNode *) = 0;
    virtual void visit(const AndNode *) = 0;
};

class StmtVisitorStrict {
public:
    virtual ~StmtVisitorStrict() = default;

    virtual void visit(Stmt);
    virtual void visit(const AssignNode *) = 0;
    virtual void visit(const SubsetNode *) = 0;
    virtual void visit(const SubsetsNode *) = 0;
    virtual void visit(const ProducesNode *) = 0;
    virtual void visit(const ConsumesNode *) = 0;
    virtual void visit(const ConsumesForNode *) = 0;
    virtual void visit(const AllocatesNode *) = 0;
    virtual void visit(const ComputesForNode *) = 0;
    virtual void visit(const ComputesNode *) = 0;
    virtual void visit(const PatternNode *) = 0;
};

class AnnotVisitorStrict : public ExprVisitorStrict,
                           public ConstraintVisitorStrict,
                           public StmtVisitorStrict {
public:
    using ConstraintVisitorStrict::visit;
    using ExprVisitorStrict::visit;
    using StmtVisitorStrict::visit;
};

class Printer : public AnnotVisitorStrict {
public:
    using AnnotVisitorStrict::visit;

    Printer(std::ostream &os, int ident = 0)
        : os(os), ident(ident) {
    }

    void visit(Consumes c);
    void visit(ConsumeMany many);
    void visit(Allocates a);

    void visit(const LiteralNode *);
    void visit(const AddNode *);
    void visit(const SubNode *);
    void visit(const MulNode *);
    void visit(const DivNode *);
    void visit(const ModNode *);
    void visit(const VariableNode *);

    void visit(const EqNode *);
    void visit(const NeqNode *);
    void visit(const LeqNode *);
    void visit(const GeqNode *);
    void visit(const LessNode *);
    void visit(const GreaterNode *);
    void visit(const OrNode *);
    void visit(const AndNode *);

    void visit(const AssignNode *);
    void visit(const SubsetNode *);
    void visit(const SubsetsNode *);
    void visit(const ProducesNode *);
    void visit(const ConsumesNode *);
    void visit(const AllocatesNode *);
    void visit(const ComputesForNode *);
    void visit(const ConsumesForNode *);
    void visit(const ComputesNode *);
    void visit(const PatternNode *);

private:
    std::ostream &os;
    int ident;
};

class AnnotVisitor : public AnnotVisitorStrict {

public:
    using AnnotVisitorStrict::visit;

    virtual ~AnnotVisitor() = default;
    void visit(const LiteralNode *);
    void visit(const AddNode *);
    void visit(const SubNode *);
    void visit(const MulNode *);
    void visit(const DivNode *);
    void visit(const ModNode *);
    void visit(const VariableNode *);

    void visit(const EqNode *);
    void visit(const NeqNode *);
    void visit(const LeqNode *);
    void visit(const GeqNode *);
    void visit(const LessNode *);
    void visit(const GreaterNode *);
    void visit(const OrNode *);
    void visit(const AndNode *);

    void visit(const AssignNode *);
    void visit(const SubsetNode *);
    void visit(const SubsetsNode *);
    void visit(const ProducesNode *);
    void visit(const ConsumesNode *);
    void visit(const AllocatesNode *);
    void visit(const ComputesForNode *);
    void visit(const ConsumesForNode *);
    void visit(const ComputesNode *);
    void visit(const PatternNode *);
};

#define RULE(Rule)                                                      \
    std::function<void(const Rule *)> Rule##Func;                       \
    std::function<void(const Rule *, Matcher *)> Rule##CtxFunc;         \
    void unpack(std::function<void(const Rule *)> pattern) {            \
        assert(!Rule##CtxFunc && !Rule##Func);                          \
        Rule##Func = pattern;                                           \
    }                                                                   \
    void unpack(std::function<void(const Rule *, Matcher *)> pattern) { \
        assert(!Rule##CtxFunc && !Rule##Func);                          \
        Rule##CtxFunc = pattern;                                        \
    }                                                                   \
    void visit(const Rule *op) {                                        \
        if (Rule##Func) {                                               \
            Rule##Func(op);                                             \
        } else if (Rule##CtxFunc) {                                     \
            Rule##CtxFunc(op, this);                                    \
            return;                                                     \
        }                                                               \
        AnnotVisitor::visit(op);                                        \
    }

class Matcher : public AnnotVisitor {
public:
    template<class T>
    void match(T stmt) {
        if (!stmt.defined()) {
            return;
        }
        stmt.accept(this);
    }

    template<class IR, class... Patterns>
    void process(IR ir, Patterns... patterns) {
        unpack(patterns...);
        ir.accept(this);
    }

private:
    template<class First, class... Rest>
    void unpack(First first, Rest... rest) {
        unpack(first);
        unpack(rest...);
    }

    using AnnotVisitor::visit;

    RULE(AddNode);
    RULE(SubNode);
    RULE(DivNode);
    RULE(MulNode);
    RULE(ModNode);
    RULE(AndNode);
    RULE(OrNode);
    RULE(EqNode);
    RULE(NeqNode);
    RULE(LeqNode);
    RULE(GeqNode);
    RULE(LessNode);
    RULE(GreaterNode);
    RULE(VariableNode);
    RULE(LiteralNode);
    RULE(AssignNode);
    RULE(SubsetNode);
    RULE(SubsetsNode);
    RULE(ProducesNode);
    RULE(ConsumesNode);
    RULE(AllocatesNode);
    RULE(ComputesForNode);
    RULE(ConsumesForNode);
    RULE(ComputesNode);
    RULE(PatternNode);
};

/**
  Match patterns to the IR.  Use lambda closures to capture environment
  variables (e.g. [&]):

  For example, to print all AddNode and SubNode objects in func:
  ~~~~~~~~~~~~~~~{.cpp}
  match(expr,
    std::function<void(const AddNode*)>([](const AddNode* op) {
      // ...
    })
    ,
    std::function<void(const SubNode*)>([](const SubNode* op) {
      // ...
    })
  );
  ~~~~~~~~~~~~~~~

  Alternatively, mathing rules can also accept a Context to be used to match
  sub-expressions:
  ~~~~~~~~~~~~~~~{.cpp}
  match(expr,
    std::function<void(const SubNode*,Matcher*)>([&](const SubNode* op,
                                                     Matcher* ctx){
      ctx->match(op->a);
    })
  );
  ~~~~~~~~~~~~~~~

  function<void(const Add*, Matcher* ctx)>([&](const Add* op, Matcher* ctx) {
**/
template<class T, class... Patterns>
void match(T stmt, Patterns... patterns) {
    if (!stmt.defined()) {
        return;
    }
    Matcher().process(stmt, patterns...);
}

class Rewriter : public AnnotVisitorStrict {
public:
    virtual ~Rewriter() = default;
    using AnnotVisitorStrict::visit;
    Stmt rewrite(Stmt);
    Expr rewrite(Expr);
    Constraint rewrite(Constraint);

protected:
    Expr expr;
    Stmt stmt;
    Constraint where;

    void visit(const LiteralNode *);
    void visit(const AddNode *);
    void visit(const SubNode *);
    void visit(const MulNode *);
    void visit(const DivNode *);
    void visit(const ModNode *);
    void visit(const VariableNode *);

    void visit(const EqNode *);
    void visit(const NeqNode *);
    void visit(const LeqNode *);
    void visit(const GeqNode *);
    void visit(const LessNode *);
    void visit(const GreaterNode *);
    void visit(const OrNode *);
    void visit(const AndNode *);

    void visit(const AssignNode *);
    void visit(const SubsetNode *);
    void visit(const SubsetsNode *);
    void visit(const ProducesNode *);
    void visit(const ConsumesNode *);
    void visit(const AllocatesNode *);
    void visit(const ComputesForNode *);
    void visit(const ConsumesForNode *);
    void visit(const ComputesNode *);
    void visit(const PatternNode *);
};

template<typename T>
inline std::set<Variable> getVariables(T annot) {
    std::set<Variable> vars;
    match(annot, std::function<void(const VariableNode *)>(
                     [&](const VariableNode *op) { vars.insert(op); }));
    return vars;
}

}  // namespace gern
#endif