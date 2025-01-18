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
#include <ranges>
#include <vector>

namespace gern {

Pipeline::Pipeline(std::vector<Compose> compose, bool fuse)
    : compose(compose), fuse(fuse) {
    check_if_legal(compose);
    constructAnnotation();
}

Pipeline::Pipeline(Compose compose)
    : Pipeline(std::vector<Compose>{compose}) {
}

std::vector<Compose> Pipeline::getFuncs() const {
    return compose;
}

std::set<AbstractDataTypePtr> Pipeline::getInputs() const {
    return inputs;
}

AbstractDataTypePtr Pipeline::getOutput() const {
    return true_output;
}

std::vector<Assign> Pipeline::getDefinitions() const {
    return definitions;
}

std::set<ComputeFunctionCallPtr> Pipeline::getConsumerFunctions(AbstractDataTypePtr ds) const {
    std::set<ComputeFunctionCallPtr> funcs;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    compose_match(Compose(*this), std::function<void(const ComputeFunctionCall *)>(
                                      [&](const ComputeFunctionCall *op) {
                                          if (op->getInputs().contains(ds)) {
                                              if (fresh_calls.contains(op)) {
                                                  funcs.insert(to<ComputeFunctionCall>(fresh_calls.at(op)));
                                              } else {
                                                  funcs.insert(op);
                                              }
                                          }
                                      }));
    return funcs;
}

std::set<Compose> Pipeline::getConsumers(AbstractDataTypePtr ds) const {
    std::set<Compose> consumers;
    for (const auto &c : compose) {
        std::set<AbstractDataTypePtr> inputs = c.getInputs();
        if (inputs.contains(ds)) {
            consumers.insert(c);
        }
    }
    return consumers;
}

ComputeFunctionCallPtr Pipeline::getProducerFunction(AbstractDataTypePtr ds) const {
    ComputeFunctionCallPtr func;
    // Only the consumes part of the annotation has
    // multiple subsets, so, we will only ever get the inputs.
    compose_match(Compose(*this), std::function<void(const ComputeFunctionCall *)>(
                                      [&](const ComputeFunctionCall *op) {
                                          if (op->getOutput() == ds) {
                                              if (fresh_calls.contains(op)) {
                                                  func = to<ComputeFunctionCall>(fresh_calls.at(op));
                                              } else {
                                                  func = op;
                                              }
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

bool Pipeline::isIntermediate(AbstractDataTypePtr d) const {
    return intermediates_set.contains(d);
}

Pattern Pipeline::getAnnotation() const {
    return annotation;
}

void Pipeline::check_if_legal(std::vector<Compose> compose) {
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
            inputs.insert(func_inputs.begin(), func_inputs.end());

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
            std::set<AbstractDataTypePtr> child_inputs = node->p.getInputs();
            inputs.insert(child_inputs.begin(), child_inputs.end());

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
            ComputeFunctionCallPtr last_func = node->p.getProducerFunction(output);
            fresh_calls[last_func] = last_func->refreshVariable();
        }

        std::vector<Compose> compose;
        std::set<AbstractDataTypePtr> inputs;
        std::set<AbstractDataTypePtr> intermediates_set;
        std::set<AbstractDataTypePtr> all_inputs;
        std::set<AbstractDataTypePtr> all_nested_temps;
        std::vector<AbstractDataTypePtr> outputs_in_order;
        AbstractDataTypePtr true_output;
        std::map<ComputeFunctionCallPtr, Compose> fresh_calls;
    };

    DataFlowConstructor df(compose);
    df.construct();
    intermediates_set = df.intermediates_set;
    all_outputs = df.outputs_in_order;
    true_output = df.true_output;
    fresh_calls = df.fresh_calls;
    std::set_difference(df.inputs.begin(), df.inputs.end(),
                        intermediates_set.begin(), intermediates_set.end(),
                        std::inserter(inputs, inputs.end()));
}

void Pipeline::constructAnnotation() {
    if (compose.size() == 0) {
        return;
    }

    // To construct the annotation, loop through the functions in reverse order.
    auto it = compose.rbegin();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;
    std::vector<Compose> rw_compose;
    std::vector<SubsetObj> input_subsets;
    // The last function produces the output, so this forms the
    // produces part of the pipeline's annotation.
    Compose last_func = *it;
    AbstractDataTypePtr last_output = last_func.getOutput();
    AbstractDataTypePtr queried = PipelineDS::make(getUniqueName("_query_" + last_output.getName()), "auto", last_output);
    Produces produces = Produces::Subset(last_output, last_func.getProducesFields());
    new_ds[last_output] = queried;
    // Put all the input subsets
    it++;  // we are done with the last function.
    //  Go in reverse order, and figure out all the subset relationships.
    for (; it < compose.rend(); ++it) {
        // For the output, figure out which functions consume this output, and then declare.
        Compose func = *it;
        AbstractDataTypePtr output = func.getOutput();
        std::set<Compose> consumer_funcs = getConsumers(output);
        std::vector<Variable> output_annot = func.getProducesFields();

        if (consumer_funcs.size() != 1) {
            std::cout << "WARNING::FORKS ARE NOT IMPL, ASSUMING EQUAL CONSUMPTION!" << std::endl;
        }

        for (const auto &cf : consumer_funcs) {
            std::vector<Expr> bound_fields = cf.getMetaDataFields(output);
            input_subsets.push_back(SubsetObj(output, cf.getMetaDataFields(output)));

            if (output_annot.size() != bound_fields.size()) {
                throw error::UserError("The size of the fields for " + output.str() + " does not match");
            }

            for (size_t i = 0; i < output_annot.size(); i++) {
                definitions.push_back(output_annot[i] = bound_fields[i]);
            }
        }
    }

    // Now handle all the inputs.
    // But, no special case for the final function now.
    it = compose.rbegin();
    for (; it < compose.rend(); ++it) {
        Compose func = *it;
        // Now handle all the inputs.
        for (const auto &in : func.getInputs()) {
            if (!isIntermediate(in)) {
                AbstractDataTypePtr queried = PipelineDS::make("_query_" + in.getName(), "auto", in);
                input_subsets.push_back(SubsetObj(queried, func.getMetaDataFields(in)));
                new_ds[in] = queried;
            }
        }
        rw_compose.push_back(last_func.replaceAllDS(new_ds));
    }

    Consumes consumes = generateConsumesIntervals(last_func, input_subsets);
    annotation = generateProducesIntervals(last_func, Computes(produces, consumes));
}

Consumes Pipeline::generateConsumesIntervals(Compose c, std::vector<SubsetObj> input_subsets) const {
    ConsumeMany consumes = SubsetObjMany(input_subsets);
    match(c.getAnnotation(), std::function<void(const ConsumesForNode *, Matcher *)>(
                                 [&](const ConsumesForNode *op, Matcher *ctx) {
                                     ctx->match(op->body);
                                     consumes = For(op->start, op->end, op->step, consumes);
                                 }));
    return consumes;
}

Pattern Pipeline::generateProducesIntervals(Compose c, Computes computes) const {
    Pattern pattern = computes;
    match(c.getAnnotation(), std::function<void(const ComputesForNode *, Matcher *)>(
                                 [&](const ComputesForNode *op, Matcher *ctx) {
                                     ctx->match(op->body);
                                     pattern = For(op->start, op->end, op->step, computes);
                                 }));
    return pattern;
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

Pattern PipelineNode::getAnnotation() const {
    return p.getAnnotation();
}

}  // namespace gern