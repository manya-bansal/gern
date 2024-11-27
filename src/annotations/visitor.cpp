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
  int size_mf = op->mdFields.size();
  for (int i = 0; i < size_mf; i++) {
    printIdent(os, ident);
    os << op->mdFields[i];
    os << ((i != size_mf - 1) ? "," : "") << std::endl;
  }
  ident--;
  printIdent(os, ident);
  os << "}";
}

void Printer::visit(const SubsetsNode *op) {
  Printer p{os, ident};
  int size_mf = op->subsets.size();
  for (int i = 0; i < size_mf; i++) {
    p.visit(op->subsets[i]);
    os << ((i != size_mf - 1) ? ",\n" : "");
  }
}

void Printer::visit(const ProducesNode *op) {
  printIdent(os, ident);
  os << "produces {" << std::endl;
  ident++;
  Printer p{os, ident};
  p.visit(op->output);
  os << std::endl;
  ident--;
  printIdent(os, ident);
  os << "}";
}

void Printer::visit(const ConsumesNode *op) {
  printIdent(os, ident);
  os << "when consumes {" << std::endl;
  ident++;
  Printer p{os, ident};
  p.visit(op->stmt);
  ident--;
  os << std::endl;
  printIdent(os, ident);
  os << "}";
}

void Printer::visit(const AllocatesNode *op) {
  printIdent(os, ident);
  os << "allocates { register : " << op->reg << " , smem : " << op->smem << "}";
}

void Printer::visit(const ForNode *op) {
  printIdent(os, ident);
  os << "for " << op->v << " in [ " << op->start << " : " << op->end << " : "
     << op->step << " ] {" << std::endl;
  ident++;
  Printer p{os, ident};
  p.visit(op->body);
  ident--;
  os << std::endl;
  printIdent(os, ident);
  os << "}";
}

void Printer::visit(const ComputesNode *op) {
  printIdent(os, ident);
  os << "computes {" << std::endl;
  ident++;
  Printer p{os, ident};
  p.visit(op->p);
  os << std::endl;
  p.visit(op->c);
  os << std::endl;
  p.visit(op->a);
  os << std::endl;
  ident--;
  printIdent(os, ident);
  os << "}";
}

#define DEFINE_BINARY_VISITOR_METHOD(CLASS_NAME)                               \
  void AnnotVisitor::visit(const CLASS_NAME *op) {                             \
    op->a.accept(this);                                                        \
    op->b.accept(this);                                                        \
  }

void AnnotVisitor::visit(const LiteralNode *) {}
DEFINE_BINARY_VISITOR_METHOD(AddNode);
DEFINE_BINARY_VISITOR_METHOD(SubNode);
DEFINE_BINARY_VISITOR_METHOD(DivNode);
DEFINE_BINARY_VISITOR_METHOD(MulNode);
DEFINE_BINARY_VISITOR_METHOD(ModNode);

DEFINE_BINARY_VISITOR_METHOD(AndNode);
DEFINE_BINARY_VISITOR_METHOD(OrNode);

DEFINE_BINARY_VISITOR_METHOD(EqNode);
DEFINE_BINARY_VISITOR_METHOD(NeqNode);
DEFINE_BINARY_VISITOR_METHOD(LeqNode);
DEFINE_BINARY_VISITOR_METHOD(GeqNode);
DEFINE_BINARY_VISITOR_METHOD(LessNode);
DEFINE_BINARY_VISITOR_METHOD(GreaterNode);

void AnnotVisitor::visit(const VariableNode *) {}

void AnnotVisitor::visit(const ConstraintNode *op) {
  op->e.accept(this);
  op->where.accept(this);
}

void AnnotVisitor::visit(const SubsetNode *op) {
  for (const auto field : op->mdFields) {
    field.accept(this);
  }
}

void AnnotVisitor::visit(const SubsetsNode *op) {
  for (const auto &subset : op->subsets) {
    subset.accept(this);
  }
}

void AnnotVisitor::visit(const ProducesNode * op) {
  op->output.accept(this);
}

void AnnotVisitor::visit(const ConsumesNode * op) {
  op->stmt.accept(this);
}

void AnnotVisitor::visit(const AllocatesNode * op) {
  op->reg.accept(this);
  op->smem.accept(this);
}

void AnnotVisitor::visit(const ForNode * op) {
  op->v.accept(this);
  op->start.accept(this);
  op->end.accept(this);
  op->step.accept(this);
  op->body.accept(this);
}

void AnnotVisitor::visit(const ComputesNode * op) {
  op->p.accept(this);
  op->c.accept(this);
  op->a.accept(this);
}

} // namespace gern