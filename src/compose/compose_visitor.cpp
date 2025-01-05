#include "compose/compose_visitor.h"
#include "compose/pipeline.h"
#include "compose/pipeline_visitor.h"
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
}

void ComposePrinter::visit(const FunctionCall *f) {
    util::printIdent(os, ident);
    os << *f;
}

void ComposePrinter::visit(const PipelineNode *p) {
    os << "{" << "\n";
    ident++;

    ComposePrinter print(os, ident);
    auto funcs = p->p.getFuncs();
    int size = funcs.size();
    for (int i = 0; i < size; i++) {
        print.visit(funcs[i]);
        os << ((i != size - 1) ? ",\n" : "");
    }

    ident--;
    os << "}";
}

int ComposeCounter::numFuncs(Compose c) {
    if (!c.defined()) {
        return 0;
    }
    c.accept(this);
    return num;
}

void ComposeCounter::visit(const FunctionCall *f) {
    (void)f;
    num++;
}

void ComposeCounter::visit(const PipelineNode *v) {
    v->p.getFuncs();
    for (const auto &f : v->p.getFuncs()) {
        this->visit(f);
    }
}

}  // namespace gern