#include "compose/compose.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "compose/composable_visitor.h"

namespace gern {

LaunchArguments LaunchParameters::constructCall() const {
    LaunchArguments args{
        .x = x,
        .y = y,
        .z = z,
    };
    return args;
}

FunctionCall FunctionSignature::constructCall() const {
    FunctionCall f_call{
        .name = name,
        .args = std::vector<Argument>(args.begin(), args.end()),
        .template_args = std::vector<Expr>(template_args.begin(), template_args.end()),
        .output = output,
        .grid = grid.constructCall(),
        .block = block.constructCall(),
        .access = access,
    };
    return f_call;
}

std::ostream &operator<<(std::ostream &os, const FunctionSignature &f) {
    FunctionCall f_call = f.constructCall();
    os << f_call << std::endl;
    return os;
}

std::ostream &operator<<(std::ostream &os, const FunctionCall &f) {
    os << f.output << " ";
    os << f.name;
    int num_template_args = f.template_args.size();
    if (num_template_args > 0) {
        os << "<";
        for (int i = 0; i < num_template_args; i++) {
            os << f.template_args[i];
            os << ((i != num_template_args - 1) ? ", " : "");
        }
        os << ">";
    }

    int args_size = f.args.size();
    os << "(";
    for (int i = 0; i < args_size; i++) {
        os << f.args[i];
        os << ((i != args_size - 1) ? ", " : "");
    }
    os << ")";
    return os;
}

std::set<Variable> ComputeFunctionCall::getVariableArgs() const {
    std::set<Variable> arg_variables;
    for (const auto &arg : call.args) {
        if (isa<VarArg>(arg)) {
            arg_variables.insert(to<VarArg>(arg)->getVar());
        }
    }
    for (const auto &arg : call.template_args) {
        if (isa<Variable>(arg)) {
            arg_variables.insert(to<Variable>(arg));
        }
    }
    return arg_variables;
}

std::set<Variable> ComputeFunctionCall::getTemplateArgs() const {
    std::set<Variable> arg_variables;
    for (const auto &arg : call.template_args) {
        if (isa<Variable>(arg)) {
            arg_variables.insert(to<Variable>(arg));
        }
    }
    return arg_variables;
}

ComputeFunctionCallPtr ComputeFunctionCall::refreshVariable() const {
    std::set<Variable> arg_variables;
    for (const auto &arg : call.args) {
        if (isa<VarArg>(arg)) {
            arg_variables.insert(to<VarArg>(arg)->getVar());
        }
    }
    for (const auto &arg : call.template_args) {
        if (isa<Variable>(arg)) {
            arg_variables.insert(to<Variable>(arg));
        }
    }
    std::set<Variable> old_vars = getVariables(annotation);
    // Generate fresh names for all old variables, except the
    // variables that are being used as arguments.
    std::map<Variable, Variable> fresh_names;
    for (const auto &v : old_vars) {
        if (arg_variables.contains(v)) {
            continue;
        }
        // Otherwise, generate a new name.
        fresh_names[v] = getUniqueName("_gern_" + v.getName());
    }
    Annotation rw_annotation = replaceVariables(annotation,
                                                fresh_names);
    return new const ComputeFunctionCall(getCall(),
                                         rw_annotation,
                                         header);
}

void ComputeFunctionCall::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

// GCOVR_EXCL_START
std::ostream &operator<<(std::ostream &os, const ComputeFunctionCall &f) {
    os << f.getCall();
    return os;
}
// GCOVR_EXCL_STOP

}  // namespace gern