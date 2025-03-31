#include "annotations/argument.h"
#include "annotations/argument_visitor.h"

namespace gern {

std::string Argument::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

bool Argument::isSameTypeAs(Argument arg) const {
    if (isa<DSArg>(*this) && isa<DSArg>(arg)) {
        return true;
    }
    if (isa<ExprArg>(*this) && isa<ExprArg>(arg)) {
        return true;
    }
    return false;
}

std::string Argument::getType() const {
    if (!defined()) {
        return "void";
    }
    return ptr->getType();
}

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

std::string DSArg::getType() const {
    return getADTPtr().getType();
}

void ExprArg::accept(ArgumentVisitorStrict *v) const {
    v->visit(this);
}

std::string ExprArg::getType() const {
    return getExpr().getType().str();
}

}  // namespace gern