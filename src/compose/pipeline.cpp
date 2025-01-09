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

    // Generate all the allocations.
    // For nested pipelines, this should
    // generated nested allocations for
    // temps as well.
    generateAllAllocs();
    visit(*this);

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

    std::set<AbstractDataTypePtr> func_inputs = c->getInputs();
    std::vector<LowerIR> temp;

    for (const auto &in : func_inputs) {
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
    return (intermediates.count(d) > 0);
}

std::vector<Expr> Pipeline::generateMetaDataFields(AbstractDataTypePtr d, FunctionCallPtr c) {

    std::vector<Expr> metaFields = c->getMetaDataFields(d);

    if (!isIntermediate(d)) {
        return metaFields;
    } else {
        // From the producer function, get producer fields.
        // FunctionCallPtr producerFunc = getProducerFunc(d);
        // std::cout << *producerFunc << std::endl;
        // std::cout << *c << std::endl;
        // std::vector<Expr> producerFields = producerFunc->getMetaDataFields(d);

        // // Insert the definitions.
        // for (size_t i = 0; i < metaFields.size(); i++) {
        //     std::cout << producerFields[i] << " = " << metaFields[i] << std::endl;
        //     // Make this true by construction!
        //     if (isa<Variable>(producerFields[i])) {
        //         lowered.push_back(
        //             new DefNode(Assign(to<Variable>(producerFields[i]), metaFields[i])));
        //     }
        // }
        // return producerFields;
        throw error::InternalError("Unimpl");
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

void Pipeline::generateAllAllocs() {
    // Generate an allocation for all the intermediates.
    for (const auto &temp : intermediates) {
        std::cout << *temp << std::endl;
        // Get the producer func.
        FunctionCallPtr producer_func = getProducerFunction(temp);

        // Define the variables assosciated with the producer func.
        // For all functions that consume this data-structure, figure out the relationships for
        // this input.
        std::set<FunctionCallPtr> consumer_funcs = getConsumerFunctions(temp);
        // No forks allowed, as is. Come back and change!
        if (consumer_funcs.size() != 1) {
            throw error::InternalError("Unimplemented");
        }

        std::vector<Variable> var_fields = producer_func->getProducesFields();
        std::cout << "out" << std::endl;
        // Writing as a for loop, will eventually change!
        // Only one allowed for right now...
        for (const auto &cf : consumer_funcs) {
            std::vector<Expr> consumer_fields = cf->getMetaDataFields(temp);
            if (consumer_fields.size() != var_fields.size()) {
                throw error::InternalError("Annotations for " + cf->getName() + " and " + producer_func->getName() +
                                           " do not have the same size for " + temp->getName());
            }
            for (size_t i = 0; i < consumer_fields.size(); i++) {
                variable_definitions[var_fields[i]] = consumer_fields[i];
            }
        }

        AbstractDataTypePtr alloc = std::make_shared<const AbstractDataType>("_alloc_" + temp->getName(),
                                                                             temp->getType());
        // Remember the name of the allocated variable.
        new_ds[temp] = alloc;
        // Finally make the allocation.
        lowered.push_back((new AllocateNode(
            alloc,
            producer_func->getMetaDataFields(temp))));
    }
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
        }

        std::vector<Compose> compose;
        std::set<AbstractDataTypePtr> intermediates;
        std::set<AbstractDataTypePtr> all_inputs;
        std::set<AbstractDataTypePtr> all_nested_temps;
        AbstractDataTypePtr true_output;
    };

    DataFlowConstructor df(compose);
    df.construct();
    intermediates = df.intermediates;
    true_output = df.true_output;
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