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
    return true_output;
}

std::set<FunctionCallPtr> Pipeline::getConsumerFunctions(AbstractDataTypePtr ds) const {
    std::set<FunctionCallPtr> funcs;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    compose_match(Compose(*this), std::function<void(const FunctionCall *)>(
                                      [&](const FunctionCall *op) {
                                          if (op->getInputs().contains(ds)) {
                                              funcs.insert(op);
                                          }
                                      }));
    return funcs;
}

FunctionCallPtr Pipeline::getProducerFunction(AbstractDataTypePtr ds) const {
    FunctionCallPtr func;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    compose_match(Compose(*this), std::function<void(const FunctionCall *)>(
                                      [&](const FunctionCall *op) {
                                          if (op->getOutput() == ds) {
                                              func = op;
                                          }
                                      }));
    return func;
}

int Pipeline::numFuncs() {
    ComposeCounter cc;
    return cc.numFuncs(Compose(*this));
}

void Pipeline::lower() {

    if (has_been_lowered) {
        return;
    }

    // Generate all the variable definitions
    // that the functions can then directly refer to.
    generateAllDefs();
    // Generate all the allocations.
    // For nested pipelines, this should
    // generated nested allocations for
    // temps as well.
    generateAllAllocs();
    // Generates all the queries and the compute node.
    // by visiting all the functions.
    visit(*this);
    // Generate all the free nodes now.
    generateAllFrees();

    // If its a vec of vec...then, need to do something else.
    // right now, gern will complain if you try.
    // Now generate the outer loops.
    lowered = generateOuterIntervals(
        to<FunctionCall>(compose[compose.size() - 1].ptr), lowered);
    has_been_lowered = true;
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
    // Generate the output query if it not an intermediate.
    AbstractDataTypePtr output = c->getOutput();
    if (!isIntermediate(output)) {
        lowered.push_back(constructQueryNode(output,
                                             c->getMetaDataFields(output)));
    }

    // Generate the queries for the true inputs.
    // since intermediates should have been allocated,
    // Putting these in temp because it may need to wrapped
    // in an interval node.
    std::set<AbstractDataTypePtr> func_inputs = c->getInputs();
    std::vector<LowerIR> temp;
    for (const auto &in : func_inputs) {
        if (!isIntermediate(in)) {
            // Generate the query node.
            temp.push_back(constructQueryNode(in,
                                              c->getMetaDataFields(in)));
        }
    }

    // Actually construct the compute node.
    temp.push_back(constructComputeNode(c));

    // Now generate all the consumer intervals if any!
    std::vector<LowerIR> intervals = generateConsumesIntervals(c, temp);
    lowered.insert(lowered.end(), intervals.begin(), intervals.end());
}

void Pipeline::visit(const PipelineNode *) {
    throw error::InternalError("Unimplemented!");
}

bool Pipeline::isIntermediate(AbstractDataTypePtr d) const {
    // We can find a function that produces this output.
    // and it is not the final output.
    return (intermediates.count(d) > 0);
}

void Pipeline::init(std::vector<Compose> compose) {
    // A pipeline is legal if each output
    // is assigned to only once, an input
    // is never assigned to later, and each
    // intermediate is used at least once as an
    // input.
    struct DataFlowConstructor : public CompositionVisitorStrict {
        DataFlowConstructor(std::vector<Compose> compose)
            : compose(compose) {
        }

        void construct() {
            if (compose.size() == 0) {
                return;
            }
            for (const auto &c : compose) {
                this->visit(c);
            }
            // Remove true_output (the last output produced) from intermediates.
            intermediates.erase(true_output);
            // For all intermediates, ensure that it gets read at least once.
            for (const auto &temp : intermediates) {
                if (all_inputs.count(temp) <= 0) {
                    throw error::UserError(temp->getName() + " is never being read");
                }
            }
        }

        using CompositionVisitorStrict::visit;
        void visit(const FunctionCall *node) {
            // Add output to intermediates.
            AbstractDataTypePtr output = node->getOutput();
            std::set<AbstractDataTypePtr> func_inputs = node->getInputs();
            all_inputs.insert(func_inputs.begin(), func_inputs.end());

            // Check whether the output has already been assigned to.
            if (intermediates.count(output) > 0 ||
                all_nested_temps.count(output) > 0) {
                throw error::UserError("Cannot assign to" + output->getName() + "twice");
            }
            // Check if this output was ever used as an input.
            if (all_inputs.count(output) > 0) {
                throw error::UserError("Assigning to a " + output->getName() + "that has already been read");
            }
            // Check if we are trying to use an intermediate from a nested pipeline.
            for (const auto &in : func_inputs) {
                if (all_nested_temps.count(in)) {
                    throw error::UserError("Trying to read intermediate" + in->getName() + " from nested pipeline");
                }
            }
            intermediates.insert(output);
            true_output = output;  // Update final output.
            outputs_in_order.push_back(output);
        }

        void visit(const PipelineNode *node) {
            // Add output to intermediates.
            AbstractDataTypePtr output = node->p.getOutput();
            if (all_inputs.contains(output)) {
                throw error::UserError("Assigning to a " + output->getName() + "that has already been read");
            }

            std::set<AbstractDataTypePtr> nested_writes = node->p.getAllWriteDataStruct();
            std::set<AbstractDataTypePtr> nested_reads = node->p.getAllReadDataStruct();

            // Check pipeline is writing to one of our outputs.
            for (const auto &in : nested_writes) {
                if (intermediates.contains(in) || all_nested_temps.contains(in)) {
                    throw error::UserError("Trying to write to intermediate" + in->getName() + " from nested pipeline");
                }
            }

            all_nested_temps.insert(nested_writes.begin(), nested_writes.end());
            all_nested_temps.erase(output);  // Remove the final output. We can see this.
            all_inputs.insert(nested_reads.begin(), nested_reads.end());

            intermediates.insert(output);
            true_output = output;  // Update final output.
            outputs_in_order.push_back(output);
        }

        std::vector<Compose> compose;
        std::set<AbstractDataTypePtr> intermediates;
        std::set<AbstractDataTypePtr> all_inputs;
        std::set<AbstractDataTypePtr> all_nested_temps;
        std::vector<AbstractDataTypePtr> outputs_in_order;
        AbstractDataTypePtr true_output;
    };

    DataFlowConstructor df(compose);
    df.construct();
    intermediates = df.intermediates;
    true_output = df.true_output;
    outputs_in_order = df.outputs_in_order;
}

std::set<AbstractDataTypePtr> Pipeline::getAllWriteDataStruct() const {
    std::set<AbstractDataTypePtr> writes;
    compose_match(Compose(*this), std::function<void(const FunctionCall *)>(
                                      [&](const FunctionCall *op) {
                                          writes.insert(op->getOutput());
                                      }));
    return writes;
}

std::set<AbstractDataTypePtr> Pipeline::getAllReadDataStruct() const {
    std::set<AbstractDataTypePtr> reads;
    compose_match(Compose(*this), std::function<void(const FunctionCall *)>(
                                      [&](const FunctionCall *op) {
                                          auto func_inputs = op->getInputs();
                                          reads.insert(func_inputs.begin(), func_inputs.end());
                                      }));
    return reads;
}

void Pipeline::generateAllDefs() {
    if (outputs_in_order.size() <= 1) {
        // Do nothing.
        return;
    }
    auto it = outputs_in_order.rbegin();
    it++;  // Skip the last output.
    // Go in reverse order, and define all the variables.
    for (; it < outputs_in_order.rend(); ++it) {
        // Define the variables assosciated with the producer func.
        // For all functions that consume this data-structure, figure out the relationships for
        // this input.
        AbstractDataTypePtr temp = *it;
        FunctionCallPtr producer_func = getProducerFunction(temp);
        std::set<FunctionCallPtr> consumer_funcs = getConsumerFunctions(temp);
        // No forks allowed, as is. Come back and change!
        if (consumer_funcs.size() != 1) {
            throw error::InternalError("Unimplemented");
        }
        // Writing as a for loop, will eventually change!
        // Only one allowed for right now...
        std::vector<Variable> var_fields = producer_func->getProducesFields();
        for (const auto &cf : consumer_funcs) {
            std::vector<Expr> consumer_fields = cf->getMetaDataFields(temp);
            if (consumer_fields.size() != var_fields.size()) {
                throw error::InternalError("Annotations for " + cf->getName() + " and " + producer_func->getName() +
                                           " do not have the same size for " + temp->getName());
            }
            for (size_t i = 0; i < consumer_fields.size(); i++) {
                lowered.push_back(new DefNode(var_fields[i] = consumer_fields[i],
                                              producer_func->isTemplateArg(var_fields[i])));
            }
        }
    }
}

void Pipeline::generateAllAllocs() {
    // We need to define an allocation for all the
    // intermediates.
    for (const auto &temp : intermediates) {
        // Finally make the allocation.
        lowered.push_back(constructAllocNode(
            temp,
            getProducerFunction(temp)->getMetaDataFields(temp)));
    }
}

void Pipeline::generateAllFrees() {
    for (const auto &ds : to_free) {
        lowered.push_back(constructFreeNode(ds));
    }
}

const QueryNode *Pipeline::constructQueryNode(AbstractDataTypePtr ds, std::vector<Expr> query_args) {
    AbstractDataTypePtr queried = std::make_shared<const AbstractDataType>("_query_" + ds->getName(),
                                                                           ds->getType());
    new_ds[ds] = queried;
    // If any of the queried data-structures need to be free, track that.
    if (ds->freeQuery()) {
        to_free.insert(queried);
    }
    return new QueryNode(ds, queried, query_args);
}

const FreeNode *Pipeline::constructFreeNode(AbstractDataTypePtr ds) {
    return new FreeNode(ds);
}

const AllocateNode *Pipeline::constructAllocNode(AbstractDataTypePtr ds, std::vector<Expr> alloc_args) {
    AbstractDataTypePtr allocated = std::make_shared<const AbstractDataType>("_alloc_" + ds->getName(),
                                                                             ds->getType());
    new_ds[ds] = allocated;
    to_free.insert(allocated);
    return new AllocateNode(allocated, alloc_args);
}

const ComputeNode *Pipeline::constructComputeNode(FunctionCallPtr f) {
    return new ComputeNode(f, new_ds);
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

void PipelineNode::accept(CompositionVisitorStrict *v) const {
    v->visit(this);
}

}  // namespace gern