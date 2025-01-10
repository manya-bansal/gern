#pragma once

#include "annotations/argument.h"
#include <iostream>

namespace gern {

class DSArg;
class VarArg;

class ArgumentVisitorStrict {
public:
    virtual ~ArgumentVisitorStrict() = default;
    virtual void visit(Argument);
    virtual void visit(const DSArg *) = 0;
    virtual void visit(const VarArg *) = 0;
};

class ArgumentPrinter : public ArgumentVisitorStrict {
public:
    ArgumentPrinter(std::ostream &os)
        : os(os) {
    }
    using ArgumentVisitorStrict::visit;

    virtual void visit(const DSArg *);
    virtual void visit(const VarArg *);

private:
    std::ostream &os;
};

}  // namespace gern