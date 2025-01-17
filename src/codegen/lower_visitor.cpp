#include "codegen/lower_visitor.h"
#include "utils/debug.h"
#include "utils/printer.h"

namespace gern {

void LowerIRVisitor::visit(LowerIR ir) {
    ir.accept(this);
}

template<typename T>
static void vector_printer(std::ostream &os, std::vector<T> v) {
    os << "{";
    int len = v.size();
    for (int i = 0; i < len; i++) {
        os << v[i];
        os << ((i != len - 1) ? "," : "");
    }
    os << "}";
}

void LowerPrinter::visit(const AllocateNode *op) {
    util::printIdent(os, ident);
    os << "Allocate " << op->f.output << " with ";
    vector_printer(os, op->f.args);
}
void LowerPrinter::visit(const FreeNode *op) {
    util::printIdent(os, ident);
    os << "Free " << op->data;
}
void LowerPrinter::visit(const InsertNode *op) {
    util::printIdent(os, ident);
    os << op->f;
}
void LowerPrinter::visit(const QueryNode *op) {
    util::printIdent(os, ident);
    os << "Query " << op->f.output
       << " from " << op->parent
       << " with ";
    vector_printer(os, op->f.args);
}
void LowerPrinter::visit(const ComputeNode *op) {
    util::printIdent(os, ident);
    os << "Compute " << op->f.name
       << " by passing in ";
    vector_printer(os, op->f.args);
}

void LowerPrinter::visit(const IntervalNode *op) {
    util::printIdent(os, ident);
    os << "for ( " << op->start << " ; " << op->end << " ; "
       << op->step << " ) {"
       << "\n";
    ident++;
    this->visit(op->body);
    ident--;
    os << "}";
}

void LowerPrinter::visit(const DefNode *op) {
    util::printIdent(os, ident);
    os << op->assign;
}

void LowerPrinter::visit(const BlankNode *op) {
    (void)op;
    DEBUG("BLANK!");
}

void LowerPrinter::visit(const FunctionBoundary *node) {
    util::printIdent(os, ident);
    os << "Function {" << "\n";
    ident++;
    this->visit(node);
    ident--;
    util::printIdent(os, ident);
    os << "}" << "\n";
}

void LowerPrinter::visit(const BlockNode *node) {
    for (const auto &ir : node->ir_nodes) {
        util::printIdent(os, ident);
        this->visit(ir);
        os << "\n";
    }
}

}  // namespace gern