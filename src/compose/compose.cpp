#include "compose/compose.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "compose/composable_visitor.h"
#include "compose/compose_visitor.h"
#include "compose/pipeline.h"

namespace gern {

FunctionCall FunctionSignature::constructCall() const {
    FunctionCall f_call{
        .name = name,
        .args = std::vector<Argument>(args.begin(), args.end()),
        .template_args = std::vector<Expr>(template_args.begin(), template_args.end()),
        .output = output,
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

std::vector<Expr> ComputeFunctionCall::getMetaDataFields(AbstractDataTypePtr d) const {
    std::vector<Expr> metaFields;
    match(getAnnotation(), std::function<void(const SubsetNode *)>(
                               [&](const SubsetNode *op) {
                                   if (op->data == d) {
                                       metaFields = op->mdFields;
                                   }
                               }));
    return metaFields;
}

std::vector<Variable> ComputeFunctionCall::getProducesFields() const {
    std::vector<Variable> metaFields;
    match(getAnnotation(), std::function<void(const ProducesNode *)>(
                               [&](const ProducesNode *op) {
                                   metaFields = Produces(op).getFieldsAsVars();
                               }));
    return metaFields;
}

bool ComputeFunctionCall::isTemplateArg(Variable v) const {
    for (const auto &arg : getTemplateArguments()) {
        if (arg.ptr == v.ptr) {
            return true;
        }
    }
    return false;
}

ComputeFunctionCall ComputeFunctionCall::replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacement) const {

    // Change the FunctionSignature call.
    auto new_call = getCall().replaceAllDS(replacement);
    // Also change the annotation.
    auto new_annot = to<Pattern>(getAnnotation().replaceDSArgs(replacement));
    return ComputeFunctionCall(new_call,
                               new_annot,
                               getHeader());
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
    Pattern rw_annotation = to<Pattern>(annotation
                                            .replaceVariables(fresh_names));
    return new const ComputeFunctionCall(getCall(), rw_annotation, header);
}

Compose Compose::callAtDevice() {
    is_device_call = true;
    return *this;
}

Compose Compose::callAtHost() {
    is_device_call = false;
    return *this;
}

bool Compose::isDeviceCall() {
    return is_device_call;
}

Compose::Compose(std::vector<Compose> compose, bool fuse)
    : Compose(Pipeline(compose, fuse)) {
}

Compose::Compose(Pipeline p)
    : Compose(new const PipelineNode(p)) {
}

Pattern Compose::getAnnotation() const {
    return ptr->getAnnotation();
}

AbstractDataTypePtr Compose::getOutput() const {
    AbstractDataTypePtr ds;
    match(getAnnotation(), std::function<void(const ProducesNode *)>(
                               [&](const ProducesNode *op) { ds = op->output.getDS(); }));
    return ds;
}

std::set<AbstractDataTypePtr> Compose::getInputs() const {
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

std::vector<Variable> Compose::getProducesFields() const {
    std::vector<Variable> metaFields;
    match(getAnnotation(), std::function<void(const ProducesNode *)>(
                               [&](const ProducesNode *op) {
                                   metaFields = Produces(op).getFieldsAsVars();
                               }));
    return metaFields;
}

std::vector<Expr> Compose::getMetaDataFields(AbstractDataTypePtr d) const {
    std::vector<Expr> metaFields;
    match(getAnnotation(), std::function<void(const SubsetNode *)>(
                               [&](const SubsetNode *op) {
                                   if (op->data == d) {
                                       metaFields = op->mdFields;
                                   }
                               }));
    return metaFields;
}

std::vector<SubsetObj> Compose::getAllConsumesSubsets() const {
    std::vector<SubsetObj> subset;
    match(getAnnotation(), std::function<void(const SubsetObjManyNode *)>(
                               [&](const SubsetObjManyNode *op) {
                                   subset = op->subsets;
                               }));
    return subset;
}

bool Compose::isTemplateArg(Variable v) const {
    if (!defined()) {
        return false;
    }
    return ptr->isTemplateArg(v);
}

Compose Compose::Tile(ADTMember adt, Variable v) {
    if (!defined()) {
        return *this;
    }

    if (adt.getDS() != getOutput()) {
        throw error::UserError("Can only tile the output " + adt.str());
    }

    // Find whether the meta-data field that they are
    // referring is actually possible to tile i.e.
    // it is an interval variable.

    // 1. Find the index at which the meta-data field actually
    //     is set.
    int index = -1;
    std::string field_to_find = adt.getMember();
    std::vector<Variable> mf_fields = adt.getDS().getFields();
    size_t size = mf_fields.size();
    for (int i = 0; i < size; i++) {
        if (mf_fields[i].getName() == field_to_find) {
            index = i;
        }
    }

    if (index < 0) {
        throw error::UserError("Could not find " + field_to_find + " in " + adt.str());
    }

    // Now that we have the index, make sure it is an index variable.
    Variable var_to_bind = getProducesFields()[index];
    std::map<Variable, Variable> interval_vars = getAnnotation().getIntervalAndStepVars();
    if (!interval_vars.contains(var_to_bind)) {
        throw error::UserError("Cannot tile a non-interval var");
    }

    // All is ok, add to bindings now.
    bindings.push_back(interval_vars.at(var_to_bind) = Expr(v));
    return *this;
}

std::vector<Assign> Compose::getBindings() const {
    return bindings;
}

Compose Compose::setBindings(std::vector<Assign> bindings_new) {
    bindings = bindings_new;
    return *this;
}

void Compose::accept(CompositionVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

void ComputeFunctionCall::accept(CompositionVisitorStrict *v) const {
    v->visit(this);
}

void ComputeFunctionCall::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

Compose Compose::replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacements) const {
    Compose c = *this;
    compose_match(Compose(*this),
                  std::function<void(const ComputeFunctionCall *, ComposeMatcher *)>(
                      [&](const ComputeFunctionCall *op, ComposeMatcher *) {
                          auto rw_call = op->replaceAllDS(replacements);
                          c = Compose(new const ComputeFunctionCall(rw_call.getCall(),
                                                                    rw_call.getAnnotation(),
                                                                    rw_call.getHeader()));
                      }),
                  std::function<void(const PipelineNode *, ComposeMatcher *)>(
                      [&](const PipelineNode *op, ComposeMatcher *ctx) {
                          std::vector<Compose> rw_compose;
                          for (const auto &func : op->p.getFuncs()) {
                              ctx->match(func);
                              rw_compose.push_back(c.setBindings(func.getBindings()));
                          }
                          c = Compose(rw_compose);
                      }));
    return c.setBindings(getBindings());
}

Compose For(ADTMember adt, Variable v, Compose c) {
    return c.Tile(adt, v);
}

std::ostream &operator<<(std::ostream &os, const Compose &compose) {
    ComposePrinter p{os, 0};
    p.visit(compose);
    return os;
}

// GCOVR_EXCL_START
std::ostream &operator<<(std::ostream &os, const ComputeFunctionCall &f) {

    os << f.getName() << "(";
    auto args = f.getArguments();
    auto args_size = args.size();

    for (size_t i = 0; i < args_size; i++) {
        os << args[i];
        os << ((i != args_size - 1) ? ", " : "");
    }

    os << ")";
    return os;
}
// GCOVR_EXCL_STOP

}  // namespace gern