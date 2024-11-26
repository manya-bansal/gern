#include "annotations/visitor.h"
#include "annotations/lang_nodes.h"

namespace gern {

void ExprVisitorStrict::visit(Expr e) {
  if (e.isDefined()) {
    return;
  }
  e.getNode()->accept(this);
}

void StmtVisitorStrict::visit(Stmt s) {
  if (s.isDefined()) {
    return;
  }
  s.getNode()->accept(this);
}

static void printIdent(std::ostream &os, int ident) {
  for (int i = 0; i < ident; i++) {
    os << "  ";
  }
}

void Printer::visit(const LiteralNode *op) { os << (*op); }
void Printer::visit(const AddNode *op) { os << op->a << " + " << op->b; }
void Printer::visit(const SubNode *op) { os << op->a << " - " << op->b; }
void Printer::visit(const DivNode *op) { os << op->a << " / " << op->b; }
void Printer::visit(const MulNode *op) { os << op->a << " * " << op->b; }
void Printer::visit(const ModNode *op) { os << op->a << " % " << op->b; }
void Printer::visit(const VariableNode *op) { os << op->name; }

#define DEFINE_PRINTER_METHOD(CLASS_NAME, OPERATOR)                            \
  void Printer::visit(const CLASS_NAME *op) {                                  \
    os << "((" << op->a << ") " << #OPERATOR << " (" << op->b << "))"          \
       << std::endl;                                                           \
  }

DEFINE_PRINTER_METHOD(EqNode, ==)
DEFINE_PRINTER_METHOD(NeqNode, !=)
DEFINE_PRINTER_METHOD(LeqNode, <=)
DEFINE_PRINTER_METHOD(GeqNode, >=)
DEFINE_PRINTER_METHOD(LessNode, <)
DEFINE_PRINTER_METHOD(GreaterNode, >)
DEFINE_PRINTER_METHOD(AndNode, &&)
DEFINE_PRINTER_METHOD(OrNode, ||)

void Printer::visit(const ConstraintNode *op) {
  os << op->e << " where " << op->where;
}

void Printer::visit(const SubsetNode *op) {
  printIdent(os, ident);
  os << *(op->data) << " {" << std::endl;
  ident++;
  int size_mf = op->meta_fields.size();
  for (int i = 0; i < size_mf - 1; i++) {
    printIdent(os, ident);
    os << op->meta_fields[i] << "," << std::endl;
  }
  if (size_mf > 0) {
    printIdent(os, ident);
    os << op->meta_fields[size_mf - 1] << std::endl;
  }
  ident--;
  printIdent(os, ident);
  os << "}" << std::endl;
}

} // namespace gern