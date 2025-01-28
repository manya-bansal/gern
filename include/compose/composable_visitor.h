#pragma once

#include "compose/composable.h"
#include "utils/scoped_set.h"

namespace gern {

class Computation;
class TiledComputation;

class ComposableVisitorStrict {
public:
    virtual void visit(Composable);
    virtual void visit(const Computation *) = 0;
    virtual void visit(const TiledComputation *) = 0;
    virtual void visit(const ComputeFunctionCall *) = 0;
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

private:
    void common(std::set<AbstractDataTypePtr> input, AbstractDataTypePtr output);
    std::set<AbstractDataTypePtr> all_reads;
    std::set<AbstractDataTypePtr> all_writes;
    util::ScopedSet<AbstractDataTypePtr> in_scope;
};

}  // namespace gern