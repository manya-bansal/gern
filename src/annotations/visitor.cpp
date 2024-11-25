#include "annotations/visitor.h"
#include "annotations/lang_nodes.h"

namespace gern {

void ExprVisitorStrict::visit(Expr e) { e.getNode()->accept(this); }

void Printer::visit(const LiteralNode *op) { os << (*op); }
void Printer::visit(const AddNode *op) { os << op->a << " + " << op->b; }
void Printer::visit(const SubNode *op) { os << op->a << " - " << op->b; }
void Printer::visit(const DivNode *op) { os << op->a << " / " << op->b; }
void Printer::visit(const MulNode *op) { os << op->a << " * " << op->b; }
void Printer::visit(const ModNode *op) { os << op->a << " % " << op->b; }

} // namespace gern