#include "compose/compose.h"
#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "compose/compose_visitor.h"
#include "compose/pipeline.h"

namespace gern {

std::ostream &operator<<(std::ostream &os, const FunctionSignature &f) {
    FunctionCall f_call{
        .name = f.name,
        .args = std::vector<Argument>(f.args.begin(), f.args.end()),
        .template_args = std::vector<Expr>(f.template_args.begin(), f.template_args.end()),
        .output = f.output,
    };
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

void Compose::accept(CompositionVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

void ComputeFunctionCall::accept(CompositionVisitorStrict *v) const {
    v->visit(this);
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