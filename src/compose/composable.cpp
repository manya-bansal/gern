#include "compose/composable.h"
#include "annotations/visitor.h"
#include "compose/composable_visitor.h"
#include "compose/compose.h"
#include "utils/printer.h"

namespace gern {

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

Annotation Computation::getAnnotation() const {
    return _annotation;
}

void Computation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
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
            std::vector<Expr> consumer_fields = consumer.getAnnotation().getPattern().getRequirement(output);
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
                input_subsets.push_back(input);
            }
        }
    }

    Consumes consumes = mimicConsumes(last_pattern, input_subsets);
    Pattern p = mimicComputes(last_pattern, Computes(produces, consumes));
    _annotation = Annotation(p, occupied, constraints);
}

void TiledComputation::accept(ComposableVisitorStrict *v) const {
    v->visit(this);
}

TiledComputation::TiledComputation(ADTMember adt_member,
                                   Variable v,
                                   Composable tiled,
                                   Grid::Unit unit,
                                   bool reduce)
    : adt_member(adt_member),
      v(v),
      tiled(tiled),
      unit(unit),
      reduce(reduce) {
    auto annotation = tiled.getAnnotation();
    auto body_units = annotation.getOccupiedUnits();
    body_units.insert(unit);
    // Assume the highest ranking unit.
    _annotation = resetUnit(annotation,
                            body_units);
    init_binding();
}

std::set<Variable> TiledComputation::getVariableArgs() const {
    return tiled.getVariableArgs();
}

std::set<Variable> TiledComputation::getTemplateArgs() const {
    return tiled.getTemplateArgs();
}

Annotation TiledComputation::getAnnotation() const {
    return _annotation;
}

void TiledComputation::init_binding() {
    Annotation annotation = getAnnotation();
    Pattern pattern = annotation.getPattern();
    SubsetObj output_subset = pattern.getOutput();
    AbstractDataTypePtr adt = output_subset.getDS();

    std::string field_to_find = adt_member.getMember();
    AbstractDataTypePtr ds = adt_member.getDS();

    std::map<ADTMember, std::tuple<Variable, Expr, Variable>> loops;

    if (reduce) {
        loops = pattern.getReducableFields();
    } else {
        loops = pattern.getTileableFields();
    }

    if (!loops.contains(adt_member)) {
        throw error::UserError("Cannot tile " + adt_member.str());
    }

    auto value = loops.at(adt_member);
    captured = std::get<0>(value);
    start = std::get<1>(value);
    end = adt_member;
    step = std::get<2>(value);
}

std::set<Variable> Composable::getVariableArgs() const {
    if (!defined()) {
        return {};
    }
    return ptr->getVariableArgs();
}

Composable::Composable(const ComposableNode *n)
    : util::IntrusivePtr<const ComposableNode>(n) {
    LegalToCompose check_legal;
    check_legal.isLegal(*this);
}

Composable::Composable(std::vector<Composable> composed)
    : Composable(new const Computation(composed)) {
}

std::set<Variable> Composable::getTemplateArgs() const {
    if (!defined()) {
        return {};
    }
    return ptr->getTemplateArgs();
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

TileDummy Tile(ADTMember member, Variable v) {
    return TileDummy(member, v, false);
}

TileDummy Reduce(ADTMember member, Variable v) {
    return TileDummy(member, v, true);
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

    if (isLegalUnit(unit)) {
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

    return new const TiledComputation(member, v, nested, unit, reduce);
}

}  // namespace gern