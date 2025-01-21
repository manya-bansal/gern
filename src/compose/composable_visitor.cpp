#include "compose/composable_visitor.h"
#include "compose/composable.h"
#include "compose/compose.h"
#include "utils/printer.h"

namespace gern {

void ComposableVisitorStrict::visit(Composable c) {
    c.accept(this);
}

void ComposablePrinter::visit(const Computation *op) {
    util::iterable_printer(os, op->composed, ident, "\n");
}

void ComposablePrinter::visit(const TiledComputation *op) {
    util::printIdent(os, ident);
    os << "For " << op->field_to_tile << " with " << op->v << "{\n";
    ident++;
    ComposablePrinter printer(os, ident);
    printer.visit(op->tiled);
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void ComposablePrinter::visit(const ComputeFunctionCall *op) {
    os << (*op);
}

}  // namespace gern