#include "annotations/abstract_function.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "codegen/codegen.h"
#include "utils/name_generator.h"
#include <map>

namespace gern {

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
        .grid = grid,
        .block = block,
        .access = access,
    };
    return new_call;
}

std::vector<Argument> FunctionCall::getAllArguments() const {
    std::vector<Argument> all_args;
    all_args.insert(all_args.end(), args.begin(), args.end());
    all_args.insert(all_args.end(), template_args.begin(), template_args.end());
    return all_args;
}

bool isSameFunctionCall(const FunctionCall &a, const FunctionCall &b) {

    if (a.name != b.name) {
        return false;
    }

    if (a.args.size() != b.args.size()) {
        return false;
    }

    if (a.template_args.size() != b.template_args.size()) {
        return false;
    }

    // Make sure all the arguments are the same.
    for (size_t i = 0; i < a.args.size(); i++) {
        if (!isSameArgument(a.args[i], b.args[i])) {
            return false;
        }
    }

    // Make sure all the template arguments are the same.
    for (size_t i = 0; i < a.template_args.size(); i++) {
        if (!isSameArgument(a.template_args[i], b.template_args[i])) {
            return false;
        }
    }

    return true;
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

        if (isa<ExprArg>(abstract_arg)) {
            auto abstract_ds = to<ExprArg>(abstract_arg.get());
            auto concrete_ds = to<ExprArg>(conc_arg.get());
            fresh_names[abstract_ds->getVar()] = concrete_ds->getVar();
        }
    }

    for (const auto &template_arg : f.template_args) {
        if (bindings.contains(template_arg.getName())) {
            fresh_names[template_arg] = bindings.at(template_arg.getName());
            continue;
        }
        fresh_names[template_arg] = Variable(getUniqueName("_gern_" + template_arg.getName()),
                                             template_arg.getDatatype(),
                                             true);
    }

    Annotation annotation = getAnnotation();
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
        fresh_names[v] = Variable(getUniqueName("_gern_" + v.getName()),
                                  v.getDatatype(),
                                  v.isConstExpr());
    }

    // The binding is only valid for one use, erase it now.
    bindings = {};

    std::vector<Argument> template_args;
    for (const auto &v : f.template_args) {
        template_args.push_back(Argument(fresh_names.at(v)));
    }

    Annotation rw_annotation = replaceVariables(            // Replace all variables with concrete vars.
        replaceADTs(annotation, abstract_to_concrete_adt),  // Replace all abstract ADTs with concrete ADTs.
        fresh_names);

    LaunchArguments grid;
    LaunchArguments block;

    // Replace all the grid and block variables.
    grid.x = replaceVariables(f.grid.x, fresh_names);
    grid.y = replaceVariables(f.grid.y, fresh_names);
    grid.z = replaceVariables(f.grid.z, fresh_names);
    block.x = replaceVariables(f.block.x, fresh_names);
    block.y = replaceVariables(f.block.y, fresh_names);
    block.z = replaceVariables(f.block.z, fresh_names);

    FunctionCall call{
        .name = f.name,
        .args = concrete_arguments,
        .template_args = template_args,
        .output = f.output,
        .grid = grid,
        .block = block,
        .access = f.access,
    };

    return Composable(new const ComputeFunctionCall(call,
                                                    rw_annotation,
                                                    getHeader()));
}

void AbstractFunction::bindVariables(const std::map<std::string, Variable> &replacements) {

    bindings.insert(replacements.begin(), replacements.end());
}

FunctionPtr::FunctionPtr(Composable function, Runner::Options options, std::optional<std::vector<Parameter>> ordered_parameters)
    : function(function), options(options) {
    // Let's lower the function to get the signature.
    Runner runner(function, ordered_parameters);
    runner.compile(options);
    signature = runner.getSignature();
}

Annotation FunctionPtr::getAnnotation() {
    return function.getAnnotation();
}

std::vector<std::string> FunctionPtr::getHeader() {
    return {options.filename};
}

FunctionSignature FunctionPtr::getFunction() {
    return signature;
}

}  // namespace gern
