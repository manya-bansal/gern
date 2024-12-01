#include "annotations/visitor.h"
#include "annotations/lang_nodes.h"

namespace gern {

void ExprVisitorStrict::visit(Expr e) {
  if (!e.defined()) {
    return;
  }
  e.accept(this);
}

void ConstraintVisitorStrict::visit(Constraint c) {
  if (!c.defined()) {
    return;
  }
  c.accept(this);
}

void StmtVisitorStrict::visit(Stmt s) {
  if (!s.defined()) {
    return;
  }
  s.accept(this);
}

static void printIdent(std::ostream &os, int ident) {
  for (int i = 0; i < ident; i++) {
    os << "  ";
  }
}

void Printer::visit(Consumes p) {
  if (!p.defined()) {
    return;
  }
  printIdent(os, ident);
  os << " when consumes {" << std::endl;
  ident++;
  p.accept(this);
  os << std::endl;
  ident--;
  printIdent(os, ident);
  os << "}";
}

void Printer::visit(ConsumeMany many) {
  if (!many.defined()) {
    return;
  }
  many.accept(this);
}

void Printer::visit(const LiteralNode *op) { os << (*op); }
void Printer::visit(const VariableNode *op) { os << op->name; }

#define DEFINE_PRINTER_METHOD(CLASS_NAME, OPERATOR)                            \
  void Printer::visit(const CLASS_NAME *op) {                                  \
    os << "(" << op->a << #OPERATOR << op->b << ")";                           \
  }

DEFINE_PRINTER_METHOD(AddNode, +)
DEFINE_PRINTER_METHOD(SubNode, -)
DEFINE_PRINTER_METHOD(MulNode, *)
DEFINE_PRINTER_METHOD(DivNode, /)
DEFINE_PRINTER_METHOD(ModNode, %)
DEFINE_PRINTER_METHOD(EqNode, ==)
DEFINE_PRINTER_METHOD(NeqNode, !=)
DEFINE_PRINTER_METHOD(LeqNode, <=)
DEFINE_PRINTER_METHOD(GeqNode, >=)
DEFINE_PRINTER_METHOD(LessNode, <)
DEFINE_PRINTER_METHOD(GreaterNode, >)
DEFINE_PRINTER_METHOD(AndNode, &&)
DEFINE_PRINTER_METHOD(OrNode, ||)

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

void Printer::visit(const ConsumesNode *op) {}

void Printer::visit(Allocates a) {
  if (!a.defined()) {
    printIdent(os, ident);
    os << "allocates()";
    return;
  }
  this->visit(a.ptr);
}

void Printer::visit(const AllocatesNode *op) {
  printIdent(os, ident);
  os << "allocates { register : " << op->reg << " , smem : " << op->smem << "}";
}

void Printer::visit(const ConsumesForNode *op) {
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

void Printer::visit(const ComputesForNode *op) {
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

void Printer::visit(const PatternNode *op) {}

#define DEFINE_BINARY_VISITOR_METHOD(CLASS_NAME)                               \
  void AnnotVisitor::visit(const CLASS_NAME *op) {                             \
    this->visit(op->a);                                                        \
    this->visit(op->b);                                                        \
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

void AnnotVisitor::visit(const SubsetNode *op) {
  for (const auto field : op->mdFields) {
    this->visit(field);
  }
}

void AnnotVisitor::visit(const SubsetsNode *op) {
  for (const auto &subset : op->subsets) {
    this->visit(subset);
  }
}

void AnnotVisitor::visit(const ProducesNode *op) { this->visit(op->output); }

void AnnotVisitor::visit(const ConsumesNode *op) {}

void AnnotVisitor::visit(const PatternNode *op) {}

void AnnotVisitor::visit(const AllocatesNode *op) {
  this->visit(op->reg);
  this->visit(op->smem);
}

void AnnotVisitor::visit(const ConsumesForNode *op) {
  this->visit(op->v);
  this->visit(op->start);
  this->visit(op->end);
  this->visit(op->step);
  this->visit(op->body);
}

void AnnotVisitor::visit(const ComputesForNode *op) {
  this->visit(op->v);
  this->visit(op->start);
  this->visit(op->end);
  this->visit(op->step);
  this->visit(op->body);
}

void AnnotVisitor::visit(const ComputesNode *op) {
  this->visit(op->p);
  this->visit(op->c);
  this->visit(op->a);
}

} // namespace gern