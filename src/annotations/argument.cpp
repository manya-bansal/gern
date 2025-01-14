#include "annotations/argument.h"
#include "annotations/argument_visitor.h"

namespace gern {

void Argument::accept(ArgumentVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

std::ostream &operator<<(std::ostream &os, const Argument &a) {
    ArgumentPrinter print(os);
    print.visit(a);
    return os;
}

void DSArg::accept(ArgumentVisitorStrict *v) const {
    v->visit(this);
}

void VarArg::accept(ArgumentVisitorStrict *v) const {
    v->visit(this);
}

void ExprArg::accept(ArgumentVisitorStrict *v) const {
    v->visit(this);
}

}  // namespace gern