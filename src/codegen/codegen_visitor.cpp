#include "codegen/codegen_visitor.h"

#include "codegen/codegen_ir.h"

namespace gern {
namespace codegen {
void CGVisitor::visit(const Literal *op) {
    (void)op;
}

void CGVisitor::visit(const Scope *op) {
    op->stmt.accept(this);
}

void CGVisitor::visit(const Type *op) {
    (void)op;
}

void CGVisitor::visit(const VarDecl *op) {
    (void)op;
}

void CGVisitor::visit(const DeclFunc *op) {
    op->body.accept(this);
}

void CGVisitor::visit(const Block *op) {
    for (auto stmt : op->stmts) {
        stmt.accept(this);
    }
}

void CGVisitor::visit(const EscapeCGExpr *) {
}

void CGVisitor::visit(const EscapeCGStmt *) {
}

void CGVisitor::visit(const VarAssign *op) {
    op->lhs.accept(this);
    op->rhs.accept(this);
}

void CGVisitor::visit(const Var *) {
}

void CGVisitor::visit(const Call *op) {
    for (auto a : op->arg) {
        a.accept(this);
    }
}

void CGVisitor::visit(const VoidCall *op) {
    op->func.accept(this);
}

void CGVisitor::visit(const For *op) {
    op->start.accept(this);
    op->cond.accept(this);
    op->step.accept(this);
    op->body.accept(this);
}

void CGVisitor::visit(const BlankLine *) {
}

void CGVisitor::visit(const Lt *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Gt *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Eq *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Neq *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Gte *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Lte *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Add *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Mul *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Div *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Mod *op) {
    op->a.accept(this);
    op->b.accept(this);
}
void CGVisitor::visit(const Sub *op) {
    op->a.accept(this);
    op->b.accept(this);
}

void CGVisitor::visit(const MetaData *op) {
    op->var.accept(this);
}

// void CGVisitor::visit(const CilkFor* op) {
//     op->stmt.accept(this);
// }

}  // namespace codegen
}  // namespace gern