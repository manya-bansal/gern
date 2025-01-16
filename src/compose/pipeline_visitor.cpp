#include "compose/pipeline_visitor.h"
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

void PipelinePrinter::visit(const AllocateNode *op) {
    util::printIdent(os, ident);
    os << "Allocate " << op->f.output << " with ";
    vector_printer(os, op->f.args);
}
void PipelinePrinter::visit(const FreeNode *op) {
    util::printIdent(os, ident);
    os << "Free " << op->data;
}
void PipelinePrinter::visit(const InsertNode *op) {
    util::printIdent(os, ident);
    os << op->f;
}
void PipelinePrinter::visit(const QueryNode *op) {
    util::printIdent(os, ident);
    os << "Query " << op->f.output
       << " from " << op->parent
       << " with ";
    vector_printer(os, op->f.args);
}
void PipelinePrinter::visit(const ComputeNode *op) {
    util::printIdent(os, ident);
    os << "Compute " << op->f.name
       << " by passing in ";
    vector_printer(os, op->f.args);
}

void PipelinePrinter::visit(const IntervalNode *op) {
    std::vector<LowerIR> nodes = op->body;
    int len_nodes = nodes.size();

    util::printIdent(os, ident);
    os << "for ( " << op->start << " ; " << op->end << " ; "
       << op->step << " ) {"
       << "\n";
    ident++;
    PipelinePrinter print(os, ident);
    for (int i = 0; i < len_nodes; i++) {
        print.visit(nodes[i]);
        os << "\n";
    }
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void PipelinePrinter::visit(const DefNode *op) {
    util::printIdent(os, ident);
    os << op->assign;
}

void PipelinePrinter::visit(const BlankNode *op) {
    (void)op;
    DEBUG("BLANK!");
}

}  // namespace gern