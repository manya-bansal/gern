#include "annotations/abstract_function.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include <map>
namespace gern {

std::ostream &operator<<(std::ostream &os, const FunctionCall &f) {

    os << f.getName() << "(";
    auto args = f.getArguments();
    auto args_size = args.size();

    for (size_t i = 0; i < args_size - 1; i++) {
        args[i].print(os);
        os << ", ";
    }
    if (args_size > 0) {
        args[args_size - 1].print(os);
    }
    os << ")";
    return os;
}

Stmt AbstractFunction::rewriteAnnotWithConcreteArgs(std::vector<Argument> concrete_arguments) {

    auto abstract_arguments = getArguments();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> abstract_to_concrete;

    if (concrete_arguments.size() != abstract_arguments.size()) {
        throw error::UserError("Size of both arguments should be the same");
    }

    for (size_t i = 0; i < concrete_arguments.size(); i++) {
        auto conc_arg = concrete_arguments[i];
        auto abstract_arg = abstract_arguments[i];

        if (abstract_arg.getType() == ArgumentType::UNDEFINED) {
            throw error::UserError("Calling with an undefined argument");
        }

        if (abstract_arg.getType() != conc_arg.getType()) {
            throw error::UserError("Calling with mismatched argument types");
        }

        if (abstract_arg.getType() == ArgumentType::DATA_STRUCTURE) {
            const DSArg *abstract_ds = to<DSArg>(abstract_arg.get());
            const DSArg *concrete_ds = to<DSArg>(conc_arg.get());
            abstract_to_concrete[abstract_ds->getADTPtr()] = concrete_ds->getADTPtr();
        }
    }

    return getAnnotation().replaceDSArgs(abstract_to_concrete);
}

}  // namespace gern