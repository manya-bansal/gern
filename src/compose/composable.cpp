#include "compose/composable.h"
#include "annotations/visitor.h"
#include "compose/composable_visitor.h"
#include "compose/compose.h"

namespace gern {

Computation::Computation(std::vector<Composable> composed)
    : composed(composed) {

    if (!composed.empty()) {
        // Set up consumer functions.
        for (const auto &c : composed) {
            std::vector<SubsetObj> inputs = c.getAnnotation().getAllConsumesSubsets();
            for (const auto &input : inputs) {
                consumer_functions[input.getDS()].insert(c);
            }
        }
        check_legal();
        init_args();
        init_template_args();
        init_annotation();
    }
}

std::set<Variable> Computation::getVariableArgs() const {
    return variable_args;
}

std::set<Variable> Computation::getTemplateArgs() const {
    return template_args;
}

Pattern Computation::getAnnotation() const {
    return _annotation;
}

void Computation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

void Computation::check_legal() {
}

void Computation::init_args() {
    for (const auto &c : composed) {
        std::set<Variable> nested_variable_args = c.getVariableArgs();
        variable_args.insert(nested_variable_args.begin(), nested_variable_args.end());
    }
}

void Computation::init_template_args() {
    for (const auto &c : composed) {
        std::set<Variable> nested_variable_args = c.getTemplateArgs();
        template_args.insert(nested_variable_args.begin(), nested_variable_args.end());
    }
}

void Computation::infer_relationships(AbstractDataTypePtr output,
                                      std::vector<Variable> output_fields) {

    // Find all the consumer functions for this output.
    auto consumer_set_it = consumer_functions.find(output);
    if (consumer_set_it != consumer_functions.end()) {
        const std::set<Composable> &consumers = consumer_set_it->second;
        if (consumers.size() != 1) {
            std::cout << "WARNING::FORKS ARE NOT IMPL, ASSUMING EQUAL CONSUMPTION!" << std::endl;
        }
        // Loop through all the consumers, and set up the definitions.
        for (const auto &consumer : consumers) {
            std::vector<Expr> consumer_fields = consumer.getAnnotation().getRequirement(output);
            for (size_t i = 0; i < consumer_fields.size(); i++) {
                declarations.push_back(output_fields[i] = consumer_fields[i]);
            }
            break;  // Can only handle one for right now.
        }
    }
}

void Computation::init_annotation() {
    // First, we generate the output part of the
    // annotation. The output of the last composed
    // object is this composables final output.
    auto it = composed.rbegin();
    auto end = composed.rend();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;
    Pattern last_annotation = it->getAnnotation();
    Produces produces = Produces::Subset(last_annotation.getOutput().getDS(),
                                         last_annotation.getProducesField());
    // Now that we have generated a produces, it's time to generate all the
    // allocations in reverse order, and note the subset relationships.
    std::vector<SubsetObj> input_subsets;
    std::set<AbstractDataTypePtr> intermediates;
    for (; it < end; ++it) {
        Composable c = *it;
        Pattern c_annotation = c.getAnnotation();
        AbstractDataTypePtr intermediate = c_annotation.getOutput().getDS();
        // Declare all the relationships btw inputs and outputs.
        infer_relationships(intermediate, c_annotation.getProducesField());
        intermediates.insert(intermediate);
        // Now add the consumes for the pure inputs.
    }

    for (const auto &c : composed) {
        std::vector<SubsetObj> inputs = c.getAnnotation().getAllConsumesSubsets();
        for (const auto &input : inputs) {
            // The the input is not an intermediate, add.
            if (!intermediates.contains(input.getDS())) {
                input_subsets.push_back(input);
            }
        }
    }

    Consumes consumes = mimicConsumes(last_annotation, input_subsets);
    _annotation = mimicComputes(last_annotation, Computes(produces, consumes));
}

void TiledComputation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

std::set<Variable> TiledComputation::getVariableArgs() const {
    return tiled.getVariableArgs();
}

std::set<Variable> TiledComputation::getTemplateArgs() const {
    return tiled.getTemplateArgs();
}

Pattern TiledComputation::getAnnotation() const {
    return tiled.getAnnotation();
}

std::set<Variable> Composable::getVariableArgs() const {
    if (!defined()) {
        return {};
    }
    return ptr->getVariableArgs();
}

std::set<Variable> Composable::getTemplateArgs() const {
    if (!defined()) {
        return {};
    }
    return ptr->getTemplateArgs();
}

Pattern Composable::getAnnotation() const {
    if (!defined()) {
        return Pattern{};
    }
    return ptr->getAnnotation();
}

void Composable::accept(ComposableVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

}  // namespace gern