#include "compose/compose_visitor.h"
#include "compose/pipeline.h"
#include "utils/printer.h"

namespace gern {

void CompositionVisitor::visit(Compose c) {
    if (!c.defined()) {
        return;
    }
    c.accept(this);
}

void ComposePrinter::visit(Compose compose) {
    if (!compose.defined()) {
        os << "Compose()";
        return;
    }
    compose.accept(this);
    // os << " @ ";
    // if (compose.is_at_device()) {
    //     os << "Device";
    // } else {
    //     os << "Host";
    // }
}

void ComposePrinter::visit(const FunctionCall *f) {
    util::printIdent(os, ident);
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

void ComposePrinter::visit(const Pipeline *p) {
    os << "{" << "\n";
    ident++;
    ComposePrinter print(os, ident);
    for (const auto &funcs : p->getFuncs()) {
        print.visit(funcs);
        os << "\n";
    }
    ident--;
    os << "}";
}

int ComposeCounter::numFuncs(Compose c) {
    this->visit(c);
    return num;
}

void ComposeCounter::visit(const FunctionCall *f) {
    (void)f;
    num++;
}

void ComposeCounter::visit(const ComposeVec *v) {
    for (const auto &funcs : v->compose) {
        this->visit(funcs);
    }
}

void ComposeCounter::visit(const Pipeline *v) {
    for (const auto &funcs : v->getFuncs()) {
        this->visit(funcs);
    }
}

}  // namespace gern