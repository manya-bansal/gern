#include "compose/pipeline.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "codegen/codegen.h"
#include "compose/compose.h"
#include "compose/pipeline_visitor.h"
#include "utils/error.h"

namespace gern {

void LowerIR::accept(PipelineVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

std::ostream &operator<<(std::ostream &os, const LowerIR &n) {
    PipelinePrinter print(os, 0);
    print.visit(n);
    return os;
}

void Pipeline::lower() {

    // Convert it into a ComposeVec
    auto compose_vec = Compose(new const ComposeVec(compose));

    if (compose_vec.numFuncs() != 1) {
        throw error::InternalError("Lowering is only "
                                   "implemented for one function!");
    }

    visit(compose_vec);
}

void Pipeline::generateCode() const {
    codegen::CodeGenerator cg;
    cg.generate_code(*this);
}

std::map<Variable, Expr> Pipeline::getVariableDefinitions() const {
    return variable_definitions;
}

std::vector<LowerIR> Pipeline::getIRNodes() const {
    return lowered;
}

void Pipeline::accept(PipelineVisitor *v) const {
    v->visit(this);
}

std::ostream &operator<<(std::ostream &os, const Pipeline &p) {
    PipelinePrinter print(os, 0);
    print.visit(p);
    return os;
}

void Pipeline::visit(const FunctionCall *c) {

    std::set<AbstractDataTypePtr> inputs = c->getInputs();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;
    std::vector<LowerIR> temp;

    for (const auto &in : inputs) {
        if (!isIntermediate(in)) {
            // Generate the query node.
            AbstractDataTypePtr queried = std::make_shared<const AbstractDataType>("_query_" + in->getName());
            new_ds[in] = queried;
            temp.push_back(new QueryNode(in,
                                         queried,
                                         generateMetaDataFields(in, c)));
        } else {
            // Generate an allocate node!
            throw error::InternalError("Unimplemented");
        }
    }

    // Now generate the outout query.
    AbstractDataTypePtr output = c->getOutput();
    if (!isIntermediate(output)) {
        AbstractDataTypePtr queried = std::make_shared<const AbstractDataType>("_query_" + output->getName());
        new_ds[output] = queried;
        lowered.push_back(new QueryNode(output,
                                        queried,
                                        generateMetaDataFields(output, c)));
    }

    temp.push_back(new ComputeNode(c, new_ds));

    // Now generate all the intervals if any!
    std::vector<LowerIR> intervals = generateConsumesIntervals(c, temp);
    lowered.insert(lowered.end(), intervals.begin(), intervals.end());
}

void Pipeline::visit(const ComposeVec *c) {
    (void)c;
    Pipeline inner_compose(c->compose);
    for (const auto &f : c->compose) {
        inner_compose.visit(f);
    }

    // If its a vec of vec...then, need to do something else.
    // right now, gern will complain if you try.
    // Now generate the outer loops.
    std::vector<LowerIR> intervals = generateOuterIntervals(
        to<FunctionCall>(c->compose[c->compose.size() - 1].ptr), inner_compose.getIRNodes());
    lowered.insert(lowered.end(), intervals.begin(), intervals.end());
}

bool Pipeline::isIntermediate(AbstractDataTypePtr d) const {
    (void)d;
    // This WILL change as I go beyond 1 function.
    return false;
}

std::vector<Expr> Pipeline::generateMetaDataFields(AbstractDataTypePtr d, FunctionCallPtr c) const {

    std::vector<Expr> metaFields;
    match(c->getAnnotation(), std::function<void(const SubsetNode *)>(
                                  [&](const SubsetNode *op) {
                                      if (op->data == d) {
                                          metaFields = op->mdFields;
                                      }
                                  }));
    if (!isIntermediate(d) || d == c->getOutput()) {
        return metaFields;
    } else {
        // Find the function that produces d
        // Solve the constraints for that function,
        // And return the solved constraints.
        throw error::InternalError("Unimplemented");
    }
}

std::vector<LowerIR> Pipeline::generateConsumesIntervals(FunctionCallPtr f, std::vector<LowerIR> body) const {

    std::vector<LowerIR> current = body;
    match(f->getAnnotation(), std::function<void(const ConsumesForNode *)>(
                                  [&](const ConsumesForNode *op) {
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

std::vector<LowerIR> Pipeline::generateOuterIntervals(FunctionCallPtr f, std::vector<LowerIR> body) const {

    std::vector<LowerIR> current = body;
    match(f->getAnnotation(), std::function<void(const ComputesForNode *)>(
                                  [&](const ComputesForNode *op) {
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

void AllocateNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void FreeNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void InsertNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void QueryNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void ComputeNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void IntervalNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void BlankNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

}  // namespace gern