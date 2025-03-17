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
    if (isa<VarArg>(*this) && isa<VarArg>(arg)) {
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

void VarArg::accept(ArgumentVisitorStrict *v) const {
    v->visit(this);
}

std::string VarArg::getType() const {
    return getVar().getType().str();
}

void ExprArg::accept(ArgumentVisitorStrict *v) const {
    v->visit(this);
}

std::string ExprArg::getType() const {
    return "auto";
}

bool same_parameters(const std::vector<Parameter> &pars1, const std::vector<Parameter> &pars2) {
    std::multiset<std::string> vars1, vars2;
    std::multiset<AbstractDataTypePtr> adt1, adt2;

    for (const auto &u : pars1) {
        if (isa<VarArg>(u)) {
            vars1.insert(to<VarArg>(u)->getVar().getName());
        } else {
            adt1.insert(to<DSArg>(u)->getADTPtr());
        }
    }

    for (const auto &u : pars2) {
        if (isa<VarArg>(u)) {
            vars2.insert(to<VarArg>(u)->getVar().getName());
        } else {
            adt2.insert(to<DSArg>(u)->getADTPtr());
        }
    }

    return adt1 == adt2 && vars1 == vars2;
}

std::vector<std::string> get_parameter_names(const std::vector<Parameter> &pars) {
    std::vector<std::string> names;
    for (const auto &u : pars) {
        if (isa<VarArg>(u)) {
            names.push_back(to<VarArg>(u)->getVar().getName());
        } else {
            names.push_back(to<DSArg>(u)->getADTPtr().getName());
        }
    }
    return names;
}

}  // namespace gern