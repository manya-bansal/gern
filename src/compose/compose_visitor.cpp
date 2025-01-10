#include "compose/compose_visitor.h"
#include "compose/pipeline.h"
#include "compose/pipeline_visitor.h"
#include "utils/printer.h"

namespace gern {

void CompositionVisitorStrict::visit(Compose c) {
    if (!c.defined()) {
        return;
    }
    c.accept(this);
}

void CompositionVisitorStrict::visit(Pipeline p) {
    for (const auto &f : p.getFuncs()) {
        this->visit(f);
    }
}

void CompositionVisitor::visit(const ComputeFunctionCall *) {
}

void CompositionVisitor::visit(const PipelineNode *op) {
    this->visit(op->p);
}

void ComposePrinter::visit(Compose compose) {
    if (!compose.defined()) {
        os << "Compose()";
        return;
    }
    compose.accept(this);
}

void ComposePrinter::visit(Pipeline p) {
    os << "{" << "\n";
    ident++;

    ComposePrinter print(os, ident);
    auto funcs = p.getFuncs();
    int size = funcs.size();
    for (int i = 0; i < size; i++) {
        print.visit(funcs[i]);
        os << ((i != size - 1) ? ",\n" : "");
    }

    ident--;
    os << "}";
}

void ComposePrinter::visit(const ComputeFunctionCall *f) {
    util::printIdent(os, ident);
    os << *f;
}

void ComposePrinter::visit(const PipelineNode *p) {
    this->visit(p->p);
}

int ComposeCounter::numFuncs(Compose c) {
    if (!c.defined()) {
        return 0;
    }
    c.accept(this);
    return num;
}

void ComposeCounter::visit(const ComputeFunctionCall *f) {
    (void)f;
    num++;
}

void ComposeCounter::visit(const PipelineNode *v) {
    this->visit(v->p);
}

}  // namespace gern