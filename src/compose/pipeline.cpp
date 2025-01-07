#include "compose/pipeline.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "codegen/codegen.h"
#include "compose/compose.h"
#include "compose/pipeline_visitor.h"
#include "utils/debug.h"
#include "utils/error.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>

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

Pipeline::Pipeline(std::vector<Compose> compose)
    : compose(compose) {
    init(compose);
}

Pipeline &Pipeline::at_device() {
    device = true;
    return *this;
}

Pipeline &Pipeline::at_host() {
    device = false;
    return *this;
}

bool Pipeline::is_at_device() const {
    return device;
}

AbstractDataTypePtr Pipeline::getOutput() const {
    return final_output;
}

OutputFunction Pipeline::getOutputFunctions() const {
    return output_function;
}

FunctionCallPtr Pipeline::getProducerFunc(AbstractDataTypePtr ds) const {

    if (output_function.count(ds) > 0) {
        return output_function.at(ds);
    }
    return nullptr;
}

std::set<AbstractDataTypePtr> Pipeline::getInputs() const {
    return all_inputs;
}

Dataflow Pipeline::getDataflow() const {
    return dataflow;
}

int Pipeline::numFuncs() {
    ComposeCounter cc;
    return cc.numFuncs(Compose(*this));
}

void Pipeline::lower() {

    if (has_been_lowered) {
        return;
    }
    visit(*this);
    // Add frees for all the allocated nodes.
    for (const auto &free : to_free) {
        lowered.push_back(new FreeNode(
            free));
    }
    // If its a vec of vec...then, need to do something else.
    // right now, gern will complain if you try.
    // Now generate the outer loops.
    lowered = generateOuterIntervals(
        to<FunctionCall>(compose[compose.size() - 1].ptr), lowered);
    has_been_lowered = true;
}

std::map<Variable, Expr> Pipeline::getVariableDefinitions() const {
    return variable_definitions;
}

std::vector<LowerIR> Pipeline::getIRNodes() const {
    return lowered;
}

std::ostream &operator<<(std::ostream &os, const Pipeline &p) {
    ComposePrinter print(os, 0);
    print.visit(p);
    return os;
}

void Pipeline::visit(const FunctionCall *c) {

    std::set<AbstractDataTypePtr> inputs = c->getInputs();
    std::vector<LowerIR> temp;

    for (const auto &in : inputs) {
        if (!isIntermediate(in)) {
            // Generate the query node.
            AbstractDataTypePtr queried = std::make_shared<const AbstractDataType>("_query_" + in->getName(),
                                                                                   in->getType());
            new_ds[in] = queried;
            temp.push_back(new QueryNode(in,
                                         queried,
                                         generateMetaDataFields(in, c)));
        } else {
        }
    }

    // Now generate the output query.
    AbstractDataTypePtr output = c->getOutput();
    if (!isIntermediate(output)) {
        AbstractDataTypePtr queried = std::make_shared<const AbstractDataType>("_query_" + output->getName(),
                                                                               output->getType());
        new_ds[output] = queried;
        lowered.push_back(new QueryNode(output,
                                        queried,
                                        generateMetaDataFields(output, c)));
    } else {
        // Generate the allocate node.
        AbstractDataTypePtr alloc = std::make_shared<const AbstractDataType>("_alloc_" + output->getName(),
                                                                             output->getType());
        to_free.insert(alloc);
        new_ds[output] = alloc;
        temp.push_back(new AllocateNode(
            alloc,
            generateMetaDataFields(output, c)));
    }

    temp.push_back(new ComputeNode(c, new_ds));

    // Now generate all the intervals if any!
    std::vector<LowerIR> intervals = generateConsumesIntervals(c, temp);
    lowered.insert(lowered.end(), intervals.begin(), intervals.end());
}

void Pipeline::visit(const PipelineNode *) {
    throw error::InternalError("Unimplemented!");
}

bool Pipeline::isIntermediate(AbstractDataTypePtr d) const {
    // We can find a function that produces this output.
    // and it is not the final output.
    return getProducerFunc(d) != nullptr && getOutput() != d;
}

std::vector<Expr> Pipeline::generateMetaDataFields(AbstractDataTypePtr d, FunctionCallPtr c) {

    std::vector<Expr> metaFields = c->getMetaDataFields(d);

    if (!isIntermediate(d)) {
        return metaFields;
    } else {
        // From the producer function, get producer fields.
        FunctionCallPtr producerFunc = getProducerFunc(d);
        std::cout << *producerFunc << std::endl;
        std::cout << *c << std::endl;
        std::vector<Expr> producerFields = producerFunc->getMetaDataFields(d);

        // Insert the definitions.
        for (size_t i = 0; i < metaFields.size(); i++) {
            std::cout << producerFields[i] << " = " << metaFields[i] << std::endl;
            // Make this true by construction!
            if (isa<Variable>(producerFields[i])) {
                lowered.push_back(
                    new DefNode(Assign(to<Variable>(producerFields[i]), metaFields[i])));
            }
        }
        return producerFields;
    }
}

std::vector<LowerIR> Pipeline::generateConsumesIntervals(FunctionCallPtr f, std::vector<LowerIR> body) const {

    std::vector<LowerIR> current = body;
    match(f->getAnnotation(), std::function<void(const ConsumesForNode *, Matcher *)>(
                                  [&](const ConsumesForNode *op, Matcher *ctx) {
                                      ctx->match(op->body);
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

std::vector<LowerIR> Pipeline::generateOuterIntervals(FunctionCallPtr f, std::vector<LowerIR> body) const {

    std::vector<LowerIR> current = body;
    match(f->getAnnotation(), std::function<void(const ComputesForNode *, Matcher *)>(
                                  [&](const ComputesForNode *op, Matcher *ctx) {
                                      ctx->match(op->body);
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

void Pipeline::init(std::vector<Compose> compose) {
    // A pipeline is legal if each output
    // is assigned to only once, an input
    // is never assigned to later, and each
    // intermediate is used at least once as an
    // input.
    struct DataFlowConstructor : public CompositionVisitor {
        DataFlowConstructor(std::vector<Compose> compose)
            : compose(compose) {
        }

        void construct() {
            for (const auto &c : compose) {
                visit(c);
            }

            for (const auto &out : out_flow) {
                std::cout << (in_flow.count(out.first) > 0);
                std::cout << " for " << out.first->getName() << std::endl;
                if (in_flow.count(out.first) <= 0 &&
                    out.first != final_output) {
                    throw error::UserError("Assigning to " + out.first->getName() + " but never using it");
                }
            }
        }

        using CompositionVisitor::visit;
        void visit(const FunctionCall *node) {

            AbstractDataTypePtr output = node->getOutput();
            std::set<AbstractDataTypePtr> inputs = node->getInputs();

            if (out_flow.count(output) > 0) {
                throw error::UserError("Cannot assign to " + output->getName() + " twice!");
            }
            if (in_flow.count(output) > 0) {
                throw error::UserError("Cannot assign to " + output->getName() + "(which is an input)");
            }
            out_flow[output] = inputs;
            output_function[output] = node;
            // Now, construct the in flow.
            for (const auto &in : inputs) {
                std::cout << "Putting in " << in->getName() << std::endl;
                in_flow.insert(in);
            }
            final_output = output;
        }

        void visit(const PipelineNode *node) {
            Dataflow nested_flow = node->p.getDataflow();
            for (const auto &output : nested_flow) {
                if (out_flow.count(output.first) > 0) {
                    throw error::UserError("Cannot assign to " + output.first->getName() + " twice!");
                }
                out_flow[output.first] = output.second;
            }
            final_output = node->p.getOutput();
            OutputFunction nested_out_funs = node->p.getOutputFunctions();
            output_function.insert(nested_out_funs.begin(), nested_out_funs.end());
            std::set<AbstractDataTypePtr> nested_inputs = node->p.getInputs();
            in_flow.insert(nested_inputs.begin(), nested_inputs.end());
        }

        AbstractDataTypePtr final_output;
        Dataflow out_flow;
        std::set<AbstractDataTypePtr> in_flow;
        OutputFunction output_function;
        std::vector<Compose> compose;
    };

    DataFlowConstructor df(compose);
    df.construct();
    dataflow = df.out_flow;
    final_output = df.final_output;
    output_function = df.output_function;
    all_inputs = df.in_flow;
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

void DefNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

bool IntervalNode::isMappedToGrid() const {
    return getIntervalVariable().isBoundToGrid();
}

Variable IntervalNode::getIntervalVariable() const {
    std::set<Variable> v = getVariables(start.getA());
    return *(v.begin());
}

void BlankNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void PipelineNode::accept(CompositionVisitor *v) const {
    v->visit(this);
}

}  // namespace gern