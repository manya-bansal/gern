#include "compose/composable.h"
#include "compose/compose.h"

namespace gern {

void Composable::accept(ComposableVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

}  // namespace gern