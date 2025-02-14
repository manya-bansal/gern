#pragma once

#include "annotations/data_dependency_language.h"
#include "annotations/visitor.h"
#include <functional>

namespace gern {

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
    void visit(const ADTMemberNode *);

    void visit(const EqNode *);
    void visit(const NeqNode *);
    void visit(const LeqNode *);
    void visit(const GeqNode *);
    void visit(const LessNode *);
    void visit(const GreaterNode *);
    void visit(const OrNode *);
    void visit(const AndNode *);
    void visit(const GridDimNode *);

    void visit(const AssignNode *);
    void visit(const SubsetNode *);
    void visit(const SubsetObjManyNode *);
    void visit(const ProducesNode *);
    void visit(const ConsumesNode *);
    void visit(const AllocatesNode *);
    void visit(const ComputesForNode *);
    void visit(const ConsumesForNode *);
    void visit(const ComputesNode *);
    void visit(const PatternNode *);
    void visit(const AnnotationNode *);
};

Consumes mimicConsumes(Pattern p, std::vector<SubsetObj>);
Pattern mimicComputes(Pattern p, Computes);

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

    RULE(VariableNode);
    RULE(ADTMemberNode);
    RULE(LiteralNode);
    RULE(AddNode);
    RULE(SubNode);
    RULE(DivNode);
    RULE(MulNode);
    RULE(ModNode);
    RULE(GridDimNode);

    RULE(AndNode);
    RULE(OrNode);
    RULE(EqNode);
    RULE(NeqNode);
    RULE(LeqNode);
    RULE(GeqNode);
    RULE(LessNode);
    RULE(GreaterNode);

    RULE(AssignNode);
    RULE(SubsetNode);
    RULE(SubsetObjManyNode);
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

template<typename T>
inline std::set<Variable> getVariables(T annot) {
    std::set<Variable> vars;
    match(annot, std::function<void(const VariableNode *)>(
                     [&](const VariableNode *op) { vars.insert(op); }));
    return vars;
}

}  // namespace gern
