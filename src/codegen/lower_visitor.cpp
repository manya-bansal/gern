#include "codegen/lower_visitor.h"
#include "annotations/rewriter_helpers.h"
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
    os << "Allocate " << op->f << " with ";
    vector_printer(os, op->f.args);
}
void LowerPrinter::visit(const FreeNode *op) {
    util::printIdent(os, ident);
    os << "Free " << op->call;
}
void LowerPrinter::visit(const InsertNode *op) {
    util::printIdent(os, ident);
    os << op->call.call;
}
void LowerPrinter::visit(const QueryNode *op) {
    util::printIdent(os, ident);
    os << "Query " << op->call.call;
}
void LowerPrinter::visit(const ComputeNode *op) {
    util::printIdent(os, ident);
    os << "Compute " << op->f;
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

void LowerPrinter::visit(const AssertNode *op) {
    bool static_check = isConstExpr(op->constraint);
    os << ((static_check) ? "static_assert" : "assert")
       << op->constraint;
}

// void LowerPrinter::visit(const FunctionBoundary *node) {
//     util::printIdent(os, ident);
//     os << "Function {"
//        << "\n";
//     ident++;
//     this->visit(node->nodes);
//     ident--;
//     util::printIdent(os, ident);
//     os << "}"
//        << "\n";
// }

void LowerPrinter::visit(const BlockNode *node) {
    for (const auto &ir : node->ir_nodes) {
        util::printIdent(os, ident);
        this->visit(ir);
        os << "\n";
    }
}

void LowerPrinter::visit(const GridDeclNode *node) {
    util::printIdent(os, ident);
    os << node->dim << " = " << node->v;
}

void LowerPrinter::visit(const SharedMemoryDeclNode *node) {
    util::printIdent(os, ident);
    os << "shared mem = " << node->size;
}

void LowerPrinter::visit(const OpaqueCall *node) {
    util::printIdent(os, ident);
    os << "Opaque call " << node->f;
}

}  // namespace gern