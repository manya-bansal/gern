#include "annotations/abstract_function.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "utils/name_generator.h"
#include <map>

namespace gern {

AbstractDataTypePtr ComputeFunctionCall::getOutput() const {
    AbstractDataTypePtr ds;
    match(getAnnotation(), std::function<void(const ProducesNode *)>(
                               [&](const ProducesNode *op) { ds = op->output.getDS(); }));
    return ds;
}

std::set<AbstractDataTypePtr> ComputeFunctionCall::getInputs() const {
    std::set<AbstractDataTypePtr> inputs;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    match(getAnnotation(), std::function<void(const SubsetObjManyNode *)>(
                               [&](const SubsetObjManyNode *op) {
                                   for (const auto &s : op->subsets) {
                                       inputs.insert(s.getDS());
                                   }
                               }));
    return inputs;
}

FunctionCall FunctionCall::replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacement) const {

    std::vector<Argument> new_args;
    for (const auto &arg : args) {
        if (isa<DSArg>(arg) &&
            replacement.contains(to<DSArg>(arg)->getADTPtr())) {
            new_args.push_back(Argument(replacement.at(to<DSArg>(arg)->getADTPtr())));
        } else {
            new_args.push_back(arg);
        }
    }

    FunctionCall new_call{
        .name = name,
        .args = new_args,
        .template_args = template_args,
        .output = output,
    };
    return new_call;
}

Compose AbstractFunction::generateComputeFunctionCall(std::vector<Argument> concrete_arguments) {

    FunctionSignature f = getFunction();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> abstract_to_concrete_adt;
    std::map<Variable, Variable> fresh_names;

    auto abstract_arguments = f.args;
    if (concrete_arguments.size() != abstract_arguments.size()) {
        throw error::UserError("Size of both arguments should be the same");
    }

    for (size_t i = 0; i < concrete_arguments.size(); i++) {
        auto conc_arg = concrete_arguments[i];
        auto abstract_arg = abstract_arguments[i];

        if (!abstract_arg.isSameTypeAs(conc_arg)) {
            throw error::UserError("Type of " + abstract_arg.str() + " and " + conc_arg.str() + " is not the same");
        }

        if (isa<DSArg>(abstract_arg)) {
            auto abstract_ds = to<DSArg>(abstract_arg.get());
            auto concrete_ds = to<DSArg>(conc_arg.get());
            abstract_to_concrete_adt[abstract_ds->getADTPtr()] = concrete_ds->getADTPtr();
        }

        if (isa<VarArg>(abstract_arg)) {
            auto abstract_ds = to<VarArg>(abstract_arg.get());
            auto concrete_ds = to<VarArg>(conc_arg.get());
            fresh_names[abstract_ds->getVar()] = concrete_ds->getVar();
        }
    }

    for (const auto &template_arg : f.template_args) {
        fresh_names[template_arg] = getUniqueName("_gern_" + template_arg.getName());
    }

    auto annotation = getAnnotation();
    auto old_vars = getVariables(annotation);
    // Convert all variables to fresh names for each
    // individual callsite.
    for (const auto &v : old_vars) {
        // If we have a binding, replace it.
        if (bindings.count(v.getName())) {
            fresh_names[v] = bindings.at(v.getName());
            continue;
        }
        // If we already have a name (from an argument for example), skip
        if (fresh_names.count(v) > 0) {
            continue;
        }
        // Otherwise, generate a new name.
        fresh_names[v] = getUniqueName("_gern_" + v.getName());
    }

    // The binding is only valid for one use, erase it now.
    bindings = {};

    std::vector<Expr> template_args;
    for (const auto &v : f.template_args) {
        template_args.push_back(fresh_names.at(v));
    }

    Pattern rw_annotation = to<Pattern>(annotation
                                            .replaceDSArgs(abstract_to_concrete_adt)
                                            .replaceVariables(fresh_names));
    FunctionCall call{
        .name = f.name,
        .args = concrete_arguments,
        .template_args = template_args,
        .output = f.output,
    };

    return Compose(new const ComputeFunctionCall(call, rw_annotation, getHeader()));
}

Composable AbstractFunction::constructComposableObject(std::vector<Argument> concrete_arguments) {
    FunctionSignature f = getFunction();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> abstract_to_concrete_adt;
    std::map<Variable, Variable> fresh_names;

    auto abstract_arguments = f.args;
    if (concrete_arguments.size() != abstract_arguments.size()) {
        throw error::UserError("Size of both arguments should be the same");
    }

    for (size_t i = 0; i < concrete_arguments.size(); i++) {
        auto conc_arg = concrete_arguments[i];
        auto abstract_arg = abstract_arguments[i];

        if (!abstract_arg.isSameTypeAs(conc_arg)) {
            throw error::UserError("Type of " + abstract_arg.str() + " and " + conc_arg.str() + " is not the same");
        }

        if (isa<DSArg>(abstract_arg)) {
            auto abstract_ds = to<DSArg>(abstract_arg.get());
            auto concrete_ds = to<DSArg>(conc_arg.get());
            abstract_to_concrete_adt[abstract_ds->getADTPtr()] = concrete_ds->getADTPtr();
        }

        if (isa<VarArg>(abstract_arg)) {
            auto abstract_ds = to<VarArg>(abstract_arg.get());
            auto concrete_ds = to<VarArg>(conc_arg.get());
            fresh_names[abstract_ds->getVar()] = concrete_ds->getVar();
        }
    }

    for (const auto &template_arg : f.template_args) {
        fresh_names[template_arg] = getUniqueName("_gern_" + template_arg.getName());
    }

    auto annotation = getAnnotation();
    auto old_vars = getVariables(annotation);
    // Convert all variables to fresh names for each
    // individual callsite.
    for (const auto &v : old_vars) {
        // If we have a binding, replace it.
        if (bindings.count(v.getName())) {
            fresh_names[v] = bindings.at(v.getName());
            continue;
        }
        // If we already have a name (from an argument for example), skip
        if (fresh_names.count(v) > 0) {
            continue;
        }
        // Otherwise, generate a new name.
        fresh_names[v] = getUniqueName("_gern_" + v.getName());
    }

    // The binding is only valid for one use, erase it now.
    bindings = {};

    std::vector<Expr> template_args;
    for (const auto &v : f.template_args) {
        template_args.push_back(fresh_names.at(v));
    }

    Pattern rw_annotation = to<Pattern>(annotation
                                            .replaceDSArgs(abstract_to_concrete_adt)
                                            .replaceVariables(fresh_names));
    FunctionCall call{
        .name = f.name,
        .args = concrete_arguments,
        .template_args = template_args,
        .output = f.output,
    };

    return Composable(new const ComputeFunctionCall(call, rw_annotation, getHeader()));
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