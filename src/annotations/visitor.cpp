#include "annotations/visitor.h"
#include "annotations/lang_nodes.h"

namespace gern {

void ExprVisitorStrict::visit(Expr e) { e.getNode()->accept(this); }

void Printer::visit(const LiteralNode *op) { os << (*op); }

} // namespace gern