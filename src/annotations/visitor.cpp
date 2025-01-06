#include "annotations/visitor.h"
#include "annotations/lang_nodes.h"
#include "utils/printer.h"

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

void Printer::visit(Consumes p) {
    if (!p.defined()) {
        return;
    }
    util::printIdent(os, ident);
    os << " when consumes {"
       << "\n";
    ident++;
    p.accept(this);
    os << "\n";
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(ConsumeMany many) {
    if (!many.defined()) {
        return;
    }
    many.accept(this);
}

void Printer::visit(const LiteralNode *op) {
    os << (*op);
}
void Printer::visit(const VariableNode *op) {
    os << op->name;
    if (isGridPropertySet(op->p)) {
        os << op->p;
    }
}

#define DEFINE_PRINTER_METHOD(CLASS_NAME, OPERATOR)      \
    void Printer::visit(const CLASS_NAME *op) {          \
        os << "(" << op->a << #OPERATOR << op->b << ")"; \
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
DEFINE_PRINTER_METHOD(AssignNode, =)

void Printer::visit(const SubsetNode *op) {
    util::printIdent(os, ident);
    os << *(op->data) << " {"
       << "\n";
    ident++;
    int size_mf = op->mdFields.size();
    for (int i = 0; i < size_mf; i++) {
        util::printIdent(os, ident);
        os << op->mdFields[i];
        os << ((i != size_mf - 1) ? "," : "") << "\n";
    }
    ident--;
    util::printIdent(os, ident);
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
    util::printIdent(os, ident);
    os << "produces {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->output);
    os << "\n";
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const ConsumesNode *op) {
    (void)op;
}

void Printer::visit(Allocates a) {
    if (!a.defined()) {
        util::printIdent(os, ident);
        os << "allocates()";
        return;
    }
    this->visit(a.ptr);
}

void Printer::visit(const AllocatesNode *op) {
    util::printIdent(os, ident);
    os << "allocates { register : " << op->reg << " , smem : " << op->smem << "}";
}

void Printer::visit(const ConsumesForNode *op) {
    util::printIdent(os, ident);
    os << "for ( " << op->start << " ; " << op->end << " ; "
       << op->step << " ) {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->body);
    ident--;
    os << "\n";
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const ComputesForNode *op) {
    util::printIdent(os, ident);
    os << "for [ " << op->start << " : " << op->end << " : "
       << op->step << " ] {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->body);
    ident--;
    os << "\n";
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const ComputesNode *op) {
    util::printIdent(os, ident);
    os << "computes {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->p);
    os << "\n";
    p.visit(op->c);
    os << "\n";
    p.visit(op->a);
    os << "\n";
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const PatternNode *op) {
    (void)op;
}

#define DEFINE_BINARY_VISITOR_METHOD(CLASS_NAME)     \
    void AnnotVisitor::visit(const CLASS_NAME *op) { \
        this->visit(op->a);                          \
        this->visit(op->b);                          \
    }

void AnnotVisitor::visit(const LiteralNode *) {
}
DEFINE_BINARY_VISITOR_METHOD(AddNode)
DEFINE_BINARY_VISITOR_METHOD(SubNode)
DEFINE_BINARY_VISITOR_METHOD(DivNode)
DEFINE_BINARY_VISITOR_METHOD(MulNode)
DEFINE_BINARY_VISITOR_METHOD(ModNode)

DEFINE_BINARY_VISITOR_METHOD(AndNode)
DEFINE_BINARY_VISITOR_METHOD(OrNode)

DEFINE_BINARY_VISITOR_METHOD(EqNode)
DEFINE_BINARY_VISITOR_METHOD(NeqNode)
DEFINE_BINARY_VISITOR_METHOD(LeqNode)
DEFINE_BINARY_VISITOR_METHOD(GeqNode)
DEFINE_BINARY_VISITOR_METHOD(LessNode)
DEFINE_BINARY_VISITOR_METHOD(GreaterNode)

DEFINE_BINARY_VISITOR_METHOD(AssignNode)

void AnnotVisitor::visit(const VariableNode *) {
}

void AnnotVisitor::visit(const SubsetNode *op) {
    for (const auto &field : op->mdFields) {
        this->visit(field);
    }
}

void AnnotVisitor::visit(const SubsetsNode *op) {
    for (const auto &subset : op->subsets) {
        this->visit(subset);
    }
}

void AnnotVisitor::visit(const ProducesNode *op) {
    this->visit(op->output);
}

void AnnotVisitor::visit(const ConsumesNode *op) {
    (void)op;
}

void AnnotVisitor::visit(const PatternNode *op) {
    (void)op;
}

void AnnotVisitor::visit(const AllocatesNode *op) {
    this->visit(op->reg);
    this->visit(op->smem);
}

void AnnotVisitor::visit(const ConsumesForNode *op) {
    this->visit(op->start);
    this->visit(op->end);
    this->visit(op->step);
    this->visit(op->body);
}

void AnnotVisitor::visit(const ComputesForNode *op) {
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

// class ExprRewriterStrict
Expr Rewriter::rewrite(Expr e) {
    if (e.defined()) {
        e.accept(this);
    } else {
        expr = Expr();
    }
    return expr;
}

Stmt Rewriter::rewrite(Stmt s) {
    if (s.defined()) {
        s.accept(this);
        Constraint rw_where = this->rewrite(s.getConstraint());
        stmt = stmt.whereStmt(rw_where);
    } else {
        stmt = Stmt();
    }
    return stmt;
}

Constraint Rewriter::rewrite(Constraint c) {
    if (c.defined()) {
        c.accept(this);
    } else {
        where = Constraint();
    }
    return where;
}

void Rewriter::visit(const VariableNode *op) {
    expr = Variable(op);
}

void Rewriter::visit(const LiteralNode *op) {
    expr = op;
}

void Rewriter::visit(const ConsumesNode *op) {
    (void)op;
}

void Rewriter::visit(const PatternNode *op) {
    (void)op;
}

void Rewriter::visit(const SubsetNode *op) {
    std::vector<Expr> rw_expr;
    for (size_t i = 0; i < op->mdFields.size(); i++) {
        rw_expr.push_back(this->rewrite(op->mdFields[i]));
    }
    stmt = Subset(op->data, rw_expr);
}

void Rewriter::visit(const SubsetsNode *op) {
    std::vector<Subset> rw_subsets;
    for (size_t i = 0; i < op->subsets.size(); i++) {
        rw_subsets.push_back(to<Subset>(this->rewrite(op->subsets[i])));
    }
    stmt = Subsets(rw_subsets);
}

void Rewriter::visit(const ProducesNode *op) {
    Subset rw_subset = to<Subset>(this->rewrite(op->output));
    stmt = Produces(rw_subset);
}

void Rewriter::visit(const AllocatesNode *op) {
    Expr rw_reg = op->reg;
    Expr rw_smem = op->smem;
    stmt = Allocates(rw_reg, rw_smem);
}

void Rewriter::visit(const ConsumesForNode *op) {
    Assign rw_start = to<Assign>(this->rewrite(op->start));
    Expr rw_end = this->rewrite(op->end);
    Expr rw_step = this->rewrite(op->step);
    ConsumeMany rw_body = to<ConsumeMany>(this->rewrite(op->body));
    stmt = Consumes(new const ConsumesForNode(rw_start,
                                              rw_end, rw_step,
                                              rw_body, op->parallel));
}

void Rewriter::visit(const ComputesForNode *op) {
    Assign rw_start = to<Assign>(this->rewrite(op->start));
    Expr rw_end = this->rewrite(op->end);
    Expr rw_step = this->rewrite(op->step);
    Pattern rw_body = to<Pattern>(this->rewrite(op->body));
    stmt = Pattern(new const ComputesForNode(rw_start,
                                             rw_end, rw_step,
                                             rw_body, op->parallel));
}

void Rewriter::visit(const ComputesNode *op) {
    Produces rw_produces = to<Produces>(this->rewrite(op->p));
    Consumes rw_consumes = to<Consumes>(this->rewrite(op->c));
    if (op->a.defined()) {
        Allocates rw_allocates = to<Allocates>(this->rewrite(op->a));
        stmt = Computes(rw_produces, rw_consumes, rw_allocates);
    } else {
        stmt = Computes(rw_produces, rw_consumes);
    }
}

#define DEFINE_BINARY_REWRITER_METHOD(CLASS_NAME, PARENT, VAR) \
    void Rewriter::visit(const CLASS_NAME *op) {               \
        Expr rw_a = this->rewrite(op->a);                      \
        Expr rw_b = this->rewrite(op->b);                      \
        VAR = PARENT(new const CLASS_NAME(rw_a, rw_b));        \
    }

DEFINE_BINARY_REWRITER_METHOD(AddNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(SubNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(DivNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(MulNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(ModNode, Expr, expr)

DEFINE_BINARY_REWRITER_METHOD(AndNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(OrNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(EqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(NeqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(LeqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(GeqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(LessNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(GreaterNode, Constraint, where)

DEFINE_BINARY_REWRITER_METHOD(AssignNode, Stmt, stmt)

}  // namespace gern