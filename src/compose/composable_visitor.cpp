#include "compose/composable_visitor.h"
#include "compose/composable.h"
#include "compose/compose.h"
#include "utils/printer.h"

namespace gern {

void ComposableVisitorStrict::visit(Composable c) {
    c.accept(this);
}

void ComposablePrinter::visit(const Computation *op) {
    util::iterable_printer(os, op->composed, ident, "\n");
}

void ComposablePrinter::visit(const TiledComputation *op) {
    util::printIdent(os, ident);
    os << "For " << op->to_tile << " with " << op->v << "{\n";
    ident++;
    ComposablePrinter printer(os, ident);
    printer.visit(op->tiled);
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

static std::set<AbstractDataTypePtr> getADTSet(std::vector<SubsetObj> subsets) {
    std::set<AbstractDataTypePtr> adt_set;
    for (const auto &subset : subsets) {
        adt_set.insert(subset.getDS());
    }
    return adt_set;
}

void ComposablePrinter::visit(const ComputeFunctionCall *op) {
    os << (*op);
}

void ComposablePrinter::visit(const GlobalNode *node) {
    util::printIdent(os, ident);
    os << "Global {\n";
    ident++;
    ComposablePrinter printer(os, ident);
    printer.visit(node->program);
    os << "\n";
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void ComposablePrinter::visit(const StageNode *node) {
    util::printIdent(os, ident);
    os << "Stage " << node->adt.getName() << " {\n";
    ident++;
    ComposablePrinter printer(os, ident);
    printer.visit(node->body);
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void LegalToCompose::isLegal(Composable c) {
    c.accept(this);
    auto output = c.getAnnotation().getPattern().getOutput().getDS();

    for (const auto &write : all_writes) {
        if (!all_reads.contains(write) && write != output) {
            throw error::UserError(" Wrote to intermediate " +
                                   write.getName() +
                                   ", but never read it");
        }
    }
}

void LegalToCompose::visit(const Computation *node) {
    auto composed = node->composed;
    for (auto const &c : composed) {
        c.accept(this);
        in_scope.insert(c.getAnnotation().getPattern().getOutput().getDS());
    }
}

void LegalToCompose::visit(const TiledComputation *node) {

    in_scope.scope();
    node->tiled.accept(this);

    Annotation annotation = node->getAnnotation();
    Pattern pattern = annotation.getPattern();
    auto inputs = getADTSet(pattern.getInputs());
    auto output = pattern.getOutput().getDS();

    for (const auto &input : inputs) {
        if (all_writes.contains(input) &&
            !in_scope.contains(input)) {
            throw error::UserError("Nested pipeline is writing to input " +
                                   input.getName() +
                                   " from nested pipeline");
        }
    }

    in_scope.unscope();
    in_scope.insert(output);
}

void LegalToCompose::visit(const ComputeFunctionCall *node) {
    Pattern pattern = node->getAnnotation().getPattern();
    auto inputs = getADTSet(pattern.getInputs());
    auto output = pattern.getOutput().getDS();
    common(inputs, output);
}

void LegalToCompose::visit(const GlobalNode *op) {
    this->visit(op->program);
}

void LegalToCompose::visit(const StageNode *node) {
    this->visit(node->body);
}

void LegalToCompose::common(std::set<AbstractDataTypePtr> inputs, AbstractDataTypePtr output) {

    for (auto const &input : inputs) {
        all_reads.insert(input);
    }

    if (all_reads.contains(output)) {
        throw error::UserError("Cannot assign to " + output.str() + " that is being used as input");
    }

    if (all_writes.contains(output)) {
        throw error::UserError("Cannot assign to " + output.str() + " twice ");
    }
    all_writes.insert(output);
}

void ComposableVisitor::visit(const Computation *node) {
    auto composed = node->composed;
    for (auto const &c : composed) {
        this->visit(c);
    }
}

void ComposableVisitor::visit(const TiledComputation *node) {
    this->visit(node->tiled);
}

void ComposableVisitor::visit(const ComputeFunctionCall *) {
}

void ComposableVisitor::visit(const GlobalNode *node) {
    this->visit(node->program);
}

void ComposableVisitor::visit(const StageNode *node) {
    this->visit(node->body);
}

}  // namespace gern