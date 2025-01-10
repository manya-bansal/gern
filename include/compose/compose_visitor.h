#pragma once

#include "compose/compose.h"
#include <functional>
#include <iostream>

namespace gern {

class ComputeFunctionCall;
struct PipelineNode;

class CompositionVisitorStrict {
public:
    virtual void visit(Compose);
    virtual void visit(Pipeline);
    virtual void visit(const ComputeFunctionCall *) = 0;
    virtual void visit(const PipelineNode *) = 0;
};

class CompositionVisitor : public CompositionVisitorStrict {
public:
    using CompositionVisitorStrict::visit;
    virtual void visit(const ComputeFunctionCall *);
    virtual void visit(const PipelineNode *);
};

class ComposePrinter : public CompositionVisitorStrict {
public:
    ComposePrinter(std::ostream &os, int ident)
        : os(os), ident(ident) {
    }
    void visit(Compose) override;
    void visit(Pipeline) override;
    virtual void visit(const ComputeFunctionCall *) override;
    virtual void visit(const PipelineNode *) override;

private:
    std::ostream &os;
    int ident = 0;
};

// Utility class to count how many distinct function calls
// a compose object contains.
class ComposeCounter : public CompositionVisitorStrict {
public:
    ComposeCounter() = default;
    int numFuncs(Compose c);
    using CompositionVisitorStrict::visit;
    virtual void visit(const ComputeFunctionCall *) override;
    virtual void visit(const PipelineNode *) override;

private:
    int num = 0;
};

#define PIPELINE_RULE(Rule)                                                     \
    std::function<void(const Rule *)> Rule##Func;                               \
    std::function<void(const Rule *, PipelineMatcher *)> Rule##CtxFunc;         \
    void unpack(std::function<void(const Rule *)> pattern) {                    \
        assert(!Rule##CtxFunc && !Rule##Func);                                  \
        Rule##Func = pattern;                                                   \
    }                                                                           \
    void unpack(std::function<void(const Rule *, PipelineMatcher *)> pattern) { \
        assert(!Rule##CtxFunc && !Rule##Func);                                  \
        Rule##CtxFunc = pattern;                                                \
    }                                                                           \
    void visit(const Rule *op) {                                                \
        if (Rule##Func) {                                                       \
            Rule##Func(op);                                                     \
        } else if (Rule##CtxFunc) {                                             \
            Rule##CtxFunc(op, this);                                            \
            return;                                                             \
        }                                                                       \
        CompositionVisitor::visit(op);                                          \
    }

class PipelineMatcher : public CompositionVisitor {
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

    using CompositionVisitor::visit;

    PIPELINE_RULE(ComputeFunctionCall);
    PIPELINE_RULE(PipelineNode);
};

template<class T, class... Patterns>
void compose_match(T stmt, Patterns... patterns) {
    if (!stmt.defined()) {
        return;
    }
    PipelineMatcher().process(stmt, patterns...);
}

}  // namespace gern