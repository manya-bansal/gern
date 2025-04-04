#include "codegen/concretize.h"
#include "annotations/rewriter.h"

namespace gern {

Concretize::Concretize(Composable program)
    : program(program) {
    std::cout << program << std::endl;
}

Composable Concretize::concretize() {
    // The output and all inputs are in scope, this is
    // passed by the user.
    auto pattern = program.getAnnotation().getPattern();
    adt_in_scope.insert(pattern.getOutput().getDS(), getVariables(pattern.getOutput()));
    for (const auto &input : pattern.getInputs()) {
        adt_in_scope.insert(input.getDS(), getVariables(input));
    }
    this->visit(program);
    return program;
}

static void printRelationships(Variable v,
                               util::ScopedMap<Variable, Expr> &all_relationships,
                               std::set<Variable> stop_at) {
    if (stop_at.count(v)) {
        return;
    }

    if (all_relationships.contains(v)) {
        std::cout << v.str() << " -> " << all_relationships.at(v).str() << std::endl;
        auto vars = getVariables(all_relationships.at(v));
        for (const auto &var : vars) {
            printRelationships(var, all_relationships, stop_at);
        }
    }
}

void Concretize::prepare_for_current_scope(SubsetObj subset) {
    auto adt = subset.getDS();
    auto fields = getVariables(subset);

    if (!adt_in_scope.contains_at_current_scope(adt)) {
        if (adt_in_scope.contains(adt)) {
            // If we do have this data structure in scope, then we need to
            // query it.
            auto staged_vars = adt_in_scope.at(adt);
            adt_in_scope.insert(adt, fields);

            for (const auto &field : fields) {
                std::cout << field << std::endl;
                printRelationships(field, all_relationships, staged_vars);
                std::cout << "....." << std::endl;
            }

            std::cout << "Querying " << adt.str() << std::endl;
        } else {
            // The output is not in scope, this is
            // a new data structure, we need to add it to the scope.
            adt_in_scope.insert(adt, fields);

            for (const auto &field : fields) {
                printRelationships(field, all_relationships, {});
            }

            std::cout << "Adding " << adt.str() << std::endl;
        }
    }
}

void Concretize::visit(const Computation *node) {

    for (const auto &c : node->declarations) {
        all_relationships.insert(to<Variable>(c.getA()), c.getB());
    }

    // Stage the output.
    prepare_for_current_scope(node->getAnnotation().getPattern().getOutput());

    // Now visit.
    for (const auto &c : node->composed) {
        prepare_for_current_scope(c.getAnnotation().getPattern().getOutput());
        this->visit(c);
    }
}

void Concretize::visit(const TiledComputation *node) {
    std::cout << node->getAnnotation().getPattern() << std::endl;
    // Visit the body.
    // all_relationships.scope();
    for (const auto &rel : node->old_to_new) {
        all_relationships.insert(rel.first, rel.second);
    }

    if (node->reduce) {
        // If we are reducing, then we stage the output, and then proceed.
        prepare_for_current_scope(node->tiled.getAnnotation().getPattern().getOutput());
    }
    adt_in_scope.scope();

    if (node->reduce) {
        // If we are reducing, then the output must be in scope, since we previously
        // staged it.
        prepare_for_current_scope(node->tiled.getAnnotation().getPattern().getOutput());
    }

    this->visit(node->tiled);

    adt_in_scope.unscope();
    // all_relationships.unscope();
}

void Concretize::visit(const ComputeFunctionCall *node) {
    // For each input and output, first stage it and then proceed.
    auto pattern = node->getAnnotation().getPattern();

    for (const auto &input : pattern.getInputs()) {
        prepare_for_current_scope(input);
    }

    prepare_for_current_scope(pattern.getOutput());
}

void Concretize::visit(const GlobalNode *node) {
    // Stage the output.
    this->visit(node->program);
}

void Concretize::visit(const StageNode *node) {
    for (const auto &rel : node->old_to_new) {
        all_relationships.insert(rel.first, rel.second);
    }
    prepare_for_current_scope(node->staged_subset);
    this->visit(node->body);
}

}  // namespace gern
