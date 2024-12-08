#include "compose/compose.h"
#include "compose/compose_visitor.h"

namespace gern {

Compose::Compose(std::vector<Compose> compose)
    : Compose(new const ComposeVec(compose)) {
}
// I am pulling concretize out since I expect
// certain scheduling commands to have loops be
// explicitly instantiated. If this is not the case,
// I can pull concretize into the constructor :)
void Compose::concretize() {
    concrete = true;
}

bool Compose::concretized() const {
    return concrete;
}

void Compose::accept(CompositionVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

void Compose::lower() const {
}

void FunctionCall::accept(CompositionVisitor *v) const {
    v->visit(this);
}

void ComposeVec::accept(CompositionVisitor *v) const {
    v->visit(this);
}

std::ostream &operator<<(std::ostream &os, const Compose &compose) {
    ComposePrinter p{os, 0};
    p.visit(compose);
    return os;
}

}  // namespace gern