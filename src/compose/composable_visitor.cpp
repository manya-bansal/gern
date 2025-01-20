#include "compose/composable_visitor.h"
#include "compose/composable.h"

namespace gern {

void ComposableVisitorStrict::visit(Composable c) {
    c.accept(this);
}

}  // namespace gern