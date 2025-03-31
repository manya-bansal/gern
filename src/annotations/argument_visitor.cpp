#include "annotations/argument_visitor.h"

namespace gern {

void ArgumentVisitorStrict::visit(Argument a) {
    a.accept(this);
}
void ArgumentPrinter::visit(const DSArg *node) {
    os << node->getADTPtr();
}

void ArgumentPrinter::visit(const ExprArg *node) {
    os << node->getExpr();
}

}  // namespace gern