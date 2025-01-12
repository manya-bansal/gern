#include "compose/compose.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "compose/compose_visitor.h"
#include "compose/pipeline.h"

namespace gern {

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

const ComputeFunctionCall *ComputeFunctionCall::withSymbolic(const std::map<std::string, Variable> &binding) {
    std::map<Variable, Variable> var_bindings;
    match(annotation, std::function<void(const VariableNode *)>(
                          [&](const VariableNode *op) {
                              if (binding.count(op->name) > 0) {
                                  var_bindings[op] = binding.at(op->name);
                              }
                          }));
    annotation = to<Pattern>(annotation.replaceVariables(var_bindings));
    return this;
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

    // Change the function call.
    auto new_call = getCall().replaceAllDS(replacement);
    // Also change the annotation.
    auto new_annot = to<Pattern>(getAnnotation().replaceDSArgs(replacement));
    return ComputeFunctionCall(new_call,
                               new_annot,
                               getHeader());
}

Compose::Compose(std::vector<Compose> compose)
    : Compose(Pipeline(compose)) {
}

Compose::Compose(Pipeline p)
    : Compose(new const PipelineNode(p)) {
}

void Compose::accept(CompositionVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

int Compose::numFuncs() const {
    ComposeCounter cc;
    return cc.numFuncs(*this);
}

void ComputeFunctionCall::accept(CompositionVisitorStrict *v) const {
    v->visit(this);
}

std::ostream &operator<<(std::ostream &os, const Compose &compose) {
    ComposePrinter p{os, 0};
    p.visit(compose);
    return os;
}

Compose Compose::replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacements) const {
    Compose c = *this;
    compose_match(Compose(*this),
                  std::function<void(const ComputeFunctionCall *, PipelineMatcher *)>(
                      [&](const ComputeFunctionCall *op, PipelineMatcher *) {
                          auto rw_call = op->replaceAllDS(replacements);
                          c = Compose(new const ComputeFunctionCall(rw_call.getCall(),
                                                                    rw_call.getAnnotation(),
                                                                    rw_call.getHeader()));
                      }),
                  std::function<void(const PipelineNode *, PipelineMatcher *)>(
                      [&](const PipelineNode *op, PipelineMatcher *ctx) {
                          std::vector<Compose> rw_compose;
                          for (const auto &func : op->p.getFuncs()) {
                              ctx->match(func);
                              rw_compose.push_back(c);
                          }
                          c = Compose(rw_compose);
                      }));
    return c;
}
}  // namespace gern