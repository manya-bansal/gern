#include "annotations/argument_visitor.h"

namespace gern {

void ArgumentVisitorStrict::visit(Argument a) {
    if (!a.defined()) {
        return;
    }
    a.accept(this);
}

void ArgumentPrinter::visit(const DSArg *node) {
    std::cout << "here" << std::endl;
    os << node->getADTPtr();
    std::cout << "here" << std::endl;
}
void ArgumentPrinter::visit(const VarArg *node) {
    os << node->getVar();
}

}  // namespace gern