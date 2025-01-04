#include "compose/compose.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "compose/compose_visitor.h"

namespace gern {

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
    : Compose(new const ComposeVec(compose)) {
}
// I am pulling concretize out since I expect
// certain scheduling commands to have loops be
// explicitly instantiated. If this is not the case,
// I can pull concretize into the constructor :)
void Compose::concretize() {
    concrete = true;
}

bool Compose::concretized() const {
    return concrete;
}

void Compose::accept(CompositionVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

void Compose::lower() const {
}

int Compose::numFuncs() const {
    ComposeCounter cc;
    return cc.numFuncs(*this);
}

void FunctionCall::accept(CompositionVisitor *v) const {
    v->visit(this);
}

void ComposeVec::accept(CompositionVisitor *v) const {
    v->visit(this);
}

std::ostream &operator<<(std::ostream &os, const Compose &compose) {
    ComposePrinter p{os, 0};
    p.visit(compose);
    return os;
}

}  // namespace gern