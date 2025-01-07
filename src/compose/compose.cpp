#include "compose/compose.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "compose/compose_visitor.h"
#include "compose/pipeline.h"

namespace gern {

std::vector<Expr> FunctionCall::getMetaDataFields(AbstractDataTypePtr d) const {
    std::vector<Expr> metaFields;
    match(getAnnotation(), std::function<void(const SubsetNode *)>(
                               [&](const SubsetNode *op) {
                                   if (op->data == d) {
                                       metaFields = op->mdFields;
                                   }
                               }));
    return metaFields;
}

const FunctionCall *FunctionCall::withSymbolic(const std::map<std::string, Variable> &binding) {
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

Compose::Compose(std::vector<Compose> compose)
    : Compose(Pipeline(compose)) {
}

Compose::Compose(Pipeline p)
    : Compose(new const PipelineNode(p)) {
}

void Compose::accept(CompositionVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

int Compose::numFuncs() const {
    ComposeCounter cc;
    return cc.numFuncs(*this);
}

void FunctionCall::accept(CompositionVisitor *v) const {
    v->visit(this);
}

std::ostream &operator<<(std::ostream &os, const Compose &compose) {
    ComposePrinter p{os, 0};
    p.visit(compose);
    return os;
}

}  // namespace gern