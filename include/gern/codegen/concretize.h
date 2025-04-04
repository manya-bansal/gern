#pragma once

#include "annotations/abstract_function.h"
#include "compose/composable.h"
#include "compose/composable_visitor.h"
#include "compose/compose.h"
#include "utils/scoped_map.h"
#include "utils/scoped_set.h"
#include "utils/uncopyable.h"

namespace gern {

class Concretize : public ComposableVisitorStrict {
public:
    Concretize(Composable program);

    Composable concretize();

private:
    Composable program;
    Composable concrete_program;
    util::ScopedMap<AbstractDataTypePtr, std::set<Variable>> adt_in_scope;

    using ComposableVisitorStrict::visit;
    void visit(const Computation *);
    void visit(const TiledComputation *);
    void visit(const ComputeFunctionCall *);
    void visit(const GlobalNode *);
    void visit(const StageNode *);

    util::ScopedMap<Variable, Expr> all_relationships;

    void prepare_for_current_scope(SubsetObj subset);
};

}  // namespace gern