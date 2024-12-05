#include "compose/compose_visitor.h"

namespace gern {

void CompositionVisitor::visit(Compose c) {
    if (!c.defined()) {
        return;
    }
    c.accept(this);
}

static void printIdent(std::ostream &os, int ident) {
    for (int i = 0; i < ident; i++) {
        os << "  ";
    }
}

void ComposePrinter::visit(Compose compose) {
    if (!compose.defined()) {
        os << "Compose()";
        return;
    }
    compose.accept(this);
}

void ComposePrinter::visit(const FunctionCall *f) {
    printIdent(os, ident);
    os << *f;
}
void ComposePrinter::visit(const ComposeVec *c) {
    os << "{" << "\n";
    ident++;
    ComposePrinter print(os, ident);
    for (const auto &funcs : c->compose) {
        print.visit(funcs);
        os << "\n";
    }
    ident--;
    os << "}";
}
}  // namespace gern