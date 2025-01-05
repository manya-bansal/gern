#include "annotations/abstract_function.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "utils/name_generator.h"
#include <map>
namespace gern {

AbstractDataTypePtr FunctionCall::getOutput() const {
    AbstractDataTypePtr ds;
    match(getAnnotation(), std::function<void(const ProducesNode *)>(
                               [&](const ProducesNode *op) { ds = op->output.getDS(); }));
    return ds;
}

std::set<AbstractDataTypePtr> FunctionCall::getInputs() const {
    std::set<AbstractDataTypePtr> inputs;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    match(getAnnotation(), std::function<void(const SubsetsNode *)>(
                               [&](const SubsetsNode *op) {
                                   for (const auto &s : op->subsets) {
                                       inputs.insert(s.getDS());
                                   }
                               }));
    return inputs;
}

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

Pattern AbstractFunction::rewriteAnnotWithConcreteArgs(std::vector<Argument> concrete_arguments) {

    auto abstract_arguments = getArguments();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> abstract_to_concrete_adt;
    std::map<Variable, Variable> fresh_names;

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
            abstract_to_concrete_adt[abstract_ds->getADTPtr()] = concrete_ds->getADTPtr();
        }

        if (abstract_arg.getType() == ArgumentType::GERN_VARIABLE) {
            const VarArg *abstract_ds = to<VarArg>(abstract_arg.get());
            const VarArg *concrete_ds = to<VarArg>(conc_arg.get());
            fresh_names[abstract_ds->getVar()] = concrete_ds->getVar();
        }
    }

    Pattern annotation = getAnnotation();
    std::set<Variable> old_vars = getVariables(annotation);
    // Convert all variables to fresh names for each
    // individual callsite.
    for (const auto &v : old_vars) {
        // If we already have a name (from an argument for example), skip
        if (fresh_names.count(v) > 0) {
            continue;
        }
        if (bindings.count(v.getName())) {
            fresh_names[v] = bindings.at(v.getName());
            continue;
        }
        fresh_names[v] = getUniqueName("_gern_" + v.getName());
    }

    return to<Pattern>(annotation
                           .replaceDSArgs(abstract_to_concrete_adt)
                           .replaceVariables(fresh_names));
}

void AbstractFunction::bindVariables(const std::map<std::string, Variable> &replacements) {

    std::set<Variable> defined_vars = getAnnotation().getDefinedVariables();
    std::set<Variable> interval_vars = getAnnotation().getIntervalVariables();

    std::set<std::string> names_defined_vars;
    std::set<std::string> names_interval_vars;

    for (const auto &v : defined_vars) {
        names_defined_vars.insert(v.getName());
    }
    for (const auto &v : interval_vars) {
        names_interval_vars.insert(v.getName());
    }

    for (const auto &binding : replacements) {
        bool is_interval_var = names_interval_vars.count(binding.first) > 0;
        // If the variable is already bound *in the annotation*,
        // complain, and you are not binding to a grid property,
        // complain.
        if (names_defined_vars.count(binding.first) > 0 && !is_interval_var) {
            throw error::UserError(binding.first + " is already bound in the annotation");
        }

        if (binding.second.isBoundToGrid()) {
            Grid::Property gp = binding.second.getBoundProperty();
            // The the property is not stable over the
            // course of the grid launch, and we have not bound it
            // to an interval var, then we cannot proceed.
            if (!isPropertyStable(gp) && !is_interval_var) {
                throw error::UserError(binding.first + " cannot be bound to an unstable property");
            }

            if (isPropertyStable(gp) && is_interval_var) {
                throw error::UserError(binding.first + " cannot be bound to an stable property, it is an interval var");
            }
        }
    }

    bindings.insert(replacements.begin(), replacements.end());
}

}  // namespace gern