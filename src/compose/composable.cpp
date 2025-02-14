#include "compose/composable.h"
#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "compose/composable_visitor.h"
#include "compose/compose.h"
#include "utils/printer.h"

namespace gern {

GlobalNode::GlobalNode(Composable program,
                       std::map<Grid::Dim, Variable> launch_args)
    : program(program), launch_args(launch_args) {

    auto legal_dims = getDims(program.getAnnotation().getOccupiedUnits());
    for (const auto &arg : launch_args) {
        if (!isDimInScope(arg.first, legal_dims)) {
            throw error::UserError("Cannot specify the size of " +
                                   util::str(arg.first) + ".");
        }
    }
}

Annotation GlobalNode::getAnnotation() const {
    return program.getAnnotation();
}

void GlobalNode::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

// Wrap a function in a global interface, mostly for a nicety.
Composable Global(Composable program,
                  std::map<Grid::Dim, Variable> launch_args) {
    return new const GlobalNode(program, launch_args);
}

Computation::Computation(std::vector<Composable> composed)
    : composed(composed) {

    if (!composed.empty()) {
        // Set up consumer functions.
        for (const auto &c : composed) {
            std::vector<SubsetObj> inputs = c.getAnnotation().getPattern().getInputs();
            for (const auto &input : inputs) {
                consumer_functions[input.getDS()].insert(c);
            }
        }
        init_annotation();
    }
}

Annotation Computation::getAnnotation() const {
    return _annotation;
}

void Computation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
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
            std::vector<Expr> consumer_fields = consumer.getAnnotation().getPattern().getRequirement(output);
            for (size_t i = 0; i < consumer_fields.size(); i++) {
                declarations.push_back(output_fields[i] = consumer_fields[i]);
                declarations_map[output_fields[i]] = consumer_fields[i];
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
    Pattern last_pattern = it->getAnnotation().getPattern();
    Produces produces = Produces::Subset(last_pattern.getOutput().getDS(),
                                         last_pattern.getProducesField());
    // Now that we have generated a produces, it's time to generate all the
    // allocations in reverse order, and note the subset relationships.
    std::vector<SubsetObj> input_subsets;
    std::set<AbstractDataTypePtr> intermediates;
    for (; it < end; ++it) {
        Composable c = *it;
        Pattern c_pattern = c.getAnnotation().getPattern();
        AbstractDataTypePtr intermediate = c_pattern.getOutput().getDS();
        // Declare all the relationships btw inputs and outputs.
        infer_relationships(intermediate, c_pattern.getProducesField());
        intermediates.insert(intermediate);
    }

    // Now add the consumes for the pure inputs.
    std::set<Grid::Unit> occupied;
    std::vector<Constraint> constraints;
    for (const auto &c : composed) {
        Annotation annotation = c.getAnnotation();
        std::set<Grid::Unit> c_units = annotation.getOccupiedUnits();
        std::vector<Constraint> c_constraints = annotation.getConstraints();
        occupied.insert(c_units.begin(), c_units.end());
        std::vector<SubsetObj>
            inputs = annotation.getPattern().getInputs();
        constraints.insert(constraints.begin(), c_constraints.begin(), c_constraints.end());
        for (const auto &input : inputs) {
            // The the input is not an intermediate, add.
            if (!intermediates.contains(input.getDS())) {
                auto fields = input.getFields();
                std::vector<Expr> fields_expr;
                for (const auto &field : fields) {
                    fields_expr.push_back(replaceVariables(field, declarations_map));
                }
                input_subsets.push_back(SubsetObj(input.getDS(), fields_expr));
            }
        }
    }

    Consumes consumes = mimicConsumes(last_pattern, input_subsets);
    Pattern p = mimicComputes(last_pattern, Computes(produces, consumes));
    _annotation = Annotation(p, occupied, constraints);
    // _annotation = refreshVariables(Annotation(p, occupied, constraints));
}

void TiledComputation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

TiledComputation::TiledComputation(Expr to_tile,
                                   Variable v,
                                   Composable tiled,
                                   Grid::Unit unit,
                                   bool reduce)
    : to_tile(to_tile),
      v(v),
      tiled(tiled),
      unit(unit),
      reduce(reduce) {
    auto annotation = tiled.getAnnotation();
    auto body_units = annotation.getOccupiedUnits();
    body_units.insert(unit);
    _annotation = resetUnit(annotation,
                            body_units);
    init_binding();
}

Annotation TiledComputation::getAnnotation() const {
    return _annotation;
}

void TiledComputation::init_binding() {
    Annotation annotation = getAnnotation();
    Pattern pattern = annotation.getPattern();
    SubsetObj output_subset = pattern.getOutput();
    AbstractDataTypePtr adt = output_subset.getDS();

    std::map<Expr, std::tuple<Variable, Expr, Variable>> loops;

    if (reduce) {
        loops = pattern.getReducableFields();
    } else {
        loops = pattern.getTileableFields();
    }

    if (!loops.contains(to_tile)) {
        throw error::UserError("Cannot tile " + to_tile.str());
    }

    auto value = loops.at(to_tile);
    captured = std::get<0>(value);
    start = std::get<1>(value);
    parameter = to_tile;
    step = std::get<2>(value);

    // Refresh the variable that just got mapped,
    // and track this relationship in a map.
    _annotation = refreshVariables(_annotation, old_to_new);
    loop_index = old_to_new[captured];  // What's the new loop index?
}

Composable::Composable(const ComposableNode *n)
    : util::IntrusivePtr<const ComposableNode>(n) {
    LegalToCompose check_legal;
    check_legal.isLegal(*this);
}

Composable::Composable(std::vector<Composable> composed)
    : Composable(new const Computation(composed)) {
}

Annotation Composable::getAnnotation() const {
    if (!defined()) {
        return Annotation{};
    }
    return ptr->getAnnotation();
}

void Composable::accept(ComposableVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

bool Composable::isDeviceLaunch() const {
    return isa<GlobalNode>(ptr);
}

TileDummy Tile(Expr tileable, Variable v) {
    return TileDummy(tileable, v, false);
}

TileDummy Reduce(Expr tileable, Variable v) {
    return TileDummy(tileable, v, true);
}

std::ostream &operator<<(std::ostream &os, const Composable &f) {
    ComposablePrinter print(os, 0);
    print.visit(f);
    return os;
}

TileDummy TileDummy::operator||(Grid::Unit p) {
    unit = p;
    return *this;
}

Composable TileDummy::operator()(Composable c) {
    Composable nested = c;
    if (isa<ComputeFunctionCall>(c.ptr)) {
        nested = new const Computation({c});
    }

    if (isLegalUnit(unit) && !reduce) {
        // If unit is set, then make sure that the compose has a reasonable
        // grid unit.
        std::set<Grid::Unit> occupied = c.getAnnotation().getOccupiedUnits();
        if (occupied.empty()) {
            throw error::UserError("The function does not have a legal unit for the current GPU");
        }
        if (!legalToDistribute(occupied, unit)) {
            throw error::UserError("Trying to distribute " + util::str(getLevel(occupied)) + " over unit " +
                                   util::str(unit) + " using " + util::str(unit));
        }
    }

    return new const TiledComputation(to_tile, v, nested, unit, reduce);
}

}  // namespace gern