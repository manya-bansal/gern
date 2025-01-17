#include "compose/pipeline.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "codegen/lower_visitor.h"
#include "compose/compose.h"
#include "utils/debug.h"
#include "utils/error.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>

namespace gern {

Pipeline::Pipeline(std::vector<Compose> compose, bool fuse)
    : compose(compose), fuse(fuse) {
    init(compose);
}

Pipeline::Pipeline(Compose compose)
    : Pipeline(std::vector<Compose>{compose}) {
}

std::vector<Compose> Pipeline::getFuncs() const {
    return compose;
}

AbstractDataTypePtr Pipeline::getOutput() const {
    return true_output;
}

std::set<ComputeFunctionCallPtr> Pipeline::getConsumerFunctions(AbstractDataTypePtr ds) const {
    std::set<ComputeFunctionCallPtr> funcs;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    compose_match(Compose(*this), std::function<void(const ComputeFunctionCall *)>(
                                      [&](const ComputeFunctionCall *op) {
                                          if (op->getInputs().contains(ds)) {
                                              funcs.insert(op);
                                          }
                                      }));
    return funcs;
}

ComputeFunctionCallPtr Pipeline::getProducerFunction(AbstractDataTypePtr ds) const {
    ComputeFunctionCallPtr func;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    compose_match(Compose(*this), std::function<void(const ComputeFunctionCall *)>(
                                      [&](const ComputeFunctionCall *op) {
                                          if (op->getOutput() == ds) {
                                              func = op;
                                          }
                                      }));
    return func;
}

std::ostream &operator<<(std::ostream &os, const Pipeline &p) {
    ComposePrinter print(os, 0);
    print.visit(p);
    return os;
}

std::vector<AbstractDataTypePtr> Pipeline::getAllOutputs() const {
    return all_outputs;
}

std::set<AbstractDataTypePtr> Pipeline::getIntermediates() const {
    return intermediates_set;
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
            intermediates_set.erase(true_output);
            // For all intermediates, ensure that it gets read at least once.
            for (const auto &temp : intermediates_set) {
                if (all_inputs.count(temp) <= 0) {
                    throw error::UserError(temp.getName() + " is never being read");
                }
            }
        }

        using CompositionVisitorStrict::visit;
        void visit(const ComputeFunctionCall *node) {
            // Add output to intermediates.
            AbstractDataTypePtr output = node->getOutput();
            std::set<AbstractDataTypePtr> func_inputs = node->getInputs();
            all_inputs.insert(func_inputs.begin(), func_inputs.end());

            // Check whether the output has already been assigned to.
            if (intermediates_set.count(output) > 0 ||
                all_nested_temps.count(output) > 0) {
                throw error::UserError("Cannot assign to" + output.str() + " twice");
            }
            // Check if this output was ever used as an input.
            if (all_inputs.count(output) > 0) {
                throw error::UserError("Assigning to a " + output.str() + " that has already been read");
            }
            // Check if we are trying to use an intermediate from a nested pipeline.
            for (const auto &in : func_inputs) {
                if (all_nested_temps.count(in)) {
                    throw error::UserError("Trying to read intermediate" + in.str() + " from nested pipeline");
                }
            }
            intermediates_set.insert(output);
            true_output = output;  // Update final output.
            outputs_in_order.push_back(output);
        }

        void visit(const PipelineNode *node) {
            // Add output to intermediates.
            AbstractDataTypePtr output = node->p.getOutput();
            std::set<AbstractDataTypePtr> nested_writes = node->p.getAllWriteDataStruct();
            std::set<AbstractDataTypePtr> nested_reads = node->p.getAllReadDataStruct();

            all_inputs.insert(nested_reads.begin(), nested_reads.end());
            if (all_inputs.contains(output)) {
                throw error::UserError("Assigning to a " + output.getName() + " that has already been read");
            }

            // Check pipeline is writing to one of our outputs.
            for (const auto &in : nested_writes) {
                if (intermediates_set.contains(in) || all_nested_temps.contains(in)) {
                    throw error::UserError("Trying to write to intermediate" + in.getName() + " from nested pipeline");
                }
            }

            all_nested_temps.insert(nested_writes.begin(), nested_writes.end());
            all_nested_temps.erase(output);  // Remove the final output. We can see this.
            intermediates_set.insert(output);
            true_output = output;  // Update final output.
            outputs_in_order.push_back(output);
        }

        std::vector<Compose> compose;
        std::set<AbstractDataTypePtr> intermediates_set;
        std::set<AbstractDataTypePtr> all_inputs;
        std::set<AbstractDataTypePtr> all_nested_temps;
        std::vector<AbstractDataTypePtr> outputs_in_order;
        AbstractDataTypePtr true_output;
    };

    DataFlowConstructor df(compose);
    df.construct();
    intermediates_set = df.intermediates_set;
    all_outputs = df.outputs_in_order;
    true_output = df.true_output;
}

std::set<AbstractDataTypePtr> Pipeline::getAllWriteDataStruct() const {
    std::set<AbstractDataTypePtr> writes;
    compose_match(Compose(*this), std::function<void(const ComputeFunctionCall *)>(
                                      [&](const ComputeFunctionCall *op) {
                                          writes.insert(op->getOutput());
                                      }));
    return writes;
}

std::set<AbstractDataTypePtr> Pipeline::getAllReadDataStruct() const {
    std::set<AbstractDataTypePtr> reads;
    compose_match(Compose(*this), std::function<void(const ComputeFunctionCall *)>(
                                      [&](const ComputeFunctionCall *op) {
                                          auto func_inputs = op->getInputs();
                                          reads.insert(func_inputs.begin(), func_inputs.end());
                                      }));
    return reads;
}

void PipelineNode::accept(CompositionVisitorStrict *v) const {
    v->visit(this);
}

}  // namespace gern