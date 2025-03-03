#pragma once

#include "compose/composable.h"
#include "utils/scoped_set.h"

#include <functional>

namespace gern {

class Computation;
class TiledComputation;
class GlobalNode;

class ComposableVisitorStrict {
public:
    virtual void visit(Composable);
    virtual void visit(const Computation *) = 0;
    virtual void visit(const TiledComputation *) = 0;
    virtual void visit(const ComputeFunctionCall *) = 0;
    virtual void visit(const GlobalNode *) = 0;
    virtual void visit(const StageNode *) = 0;
};

class ComposablePrinter : public ComposableVisitorStrict {
public:
    using ComposableVisitorStrict::visit;

    ComposablePrinter(std::ostream &os, int ident = 0)
        : os(os), ident(ident) {
    }

    void visit(const Computation *);
    void visit(const TiledComputation *);
    void visit(const ComputeFunctionCall *);
    void visit(const GlobalNode *);
    void visit(const StageNode *);

private:
    std::ostream &os;
    int ident;
};

class LegalToCompose : public ComposableVisitorStrict {
public:
    using ComposableVisitorStrict::visit;
    void isLegal(Composable);
    void visit(const Computation *);
    void visit(const TiledComputation *);
    void visit(const ComputeFunctionCall *);
    void visit(const GlobalNode *);
    void visit(const StageNode *);

private:
    void common(std::set<AbstractDataTypePtr> input, AbstractDataTypePtr output);
    std::set<AbstractDataTypePtr> all_reads;
    std::set<AbstractDataTypePtr> all_writes;
    util::ScopedSet<AbstractDataTypePtr> in_scope;
};

class ComposableVisitor : public ComposableVisitorStrict {
public:
    using ComposableVisitorStrict::visit;
    void visit(const Computation *);
    void visit(const TiledComputation *);
    void visit(const ComputeFunctionCall *);
    void visit(const GlobalNode *);
    void visit(const StageNode *);
};

#define COMPOSABLE_RULE(Rule)                                                     \
    std::function<void(const Rule *)> Rule##Func;                                 \
    std::function<void(const Rule *, ComposableMatcher *)> Rule##CtxFunc;         \
    void unpack(std::function<void(const Rule *)> pattern) {                      \
        assert(!Rule##CtxFunc && !Rule##Func);                                    \
        Rule##Func = pattern;                                                     \
    }                                                                             \
    void unpack(std::function<void(const Rule *, ComposableMatcher *)> pattern) { \
        assert(!Rule##CtxFunc && !Rule##Func);                                    \
        Rule##CtxFunc = pattern;                                                  \
    }                                                                             \
    void visit(const Rule *op) {                                                  \
        if (Rule##Func) {                                                         \
            Rule##Func(op);                                                       \
        } else if (Rule##CtxFunc) {                                               \
            Rule##CtxFunc(op, this);                                              \
            return;                                                               \
        }                                                                         \
        ComposableVisitor::visit(op);                                             \
    }

class ComposableMatcher : public ComposableVisitor {
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

    using ComposableVisitorStrict::visit;

    COMPOSABLE_RULE(ComputeFunctionCall);
    COMPOSABLE_RULE(TiledComputation);
    COMPOSABLE_RULE(Computation);
    COMPOSABLE_RULE(GlobalNode);
    COMPOSABLE_RULE(StageNode);
};

template<class T, class... Patterns>
void composable_match(T stmt, Patterns... patterns) {
    if (!stmt.defined()) {
        return;
    }
    ComposableMatcher().process(stmt, patterns...);
}

}  // namespace gern