#include "compose/composable.h"
#include "compose/composable_visitor.h"
#include "compose/compose.h"

namespace gern {

Computation::Computation(std::vector<Composable> composed)
    : composed(composed) {
    init_args();
}

void Computation::init_args() {
    for (const auto &c : composed) {
        std::set<Variable> nested_variable_args = c.getVariableArgs();
        variable_args.insert(nested_variable_args.begin(), nested_variable_args.end());
    }
}

void Computation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

std::set<Variable> TiledComputation::getVariableArgs() const {
    return tiled.getVariableArgs();
}

void TiledComputation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

std::set<Variable> Composable::getVariableArgs() const {
    if (!defined()) {
        return {};
    }
    return ptr->getVariableArgs();
}

void Composable::accept(ComposableVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

}  // namespace gern