#include "compose/compose_visitor.h"
#include "codegen/lower_visitor.h"
#include "compose/pipeline.h"
#include "utils/printer.h"

namespace gern {

void CompositionVisitorStrict::visit(Compose c) {
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
    os << f->getAnnotation() << std::endl;
    os << *f;
}

void ComposePrinter::visit(const PipelineNode *p) {
    this->visit(p->p);
}

}  // namespace gern